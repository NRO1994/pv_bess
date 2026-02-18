"""Daily LP-based dispatch optimiser (scipy.optimize.linprog / HiGHS backend).

Solves a 24-hour linear programme for each simulation day to determine optimal
charge/discharge and export decisions under perfect day-ahead price foresight.

Two operating modes are supported:

- **Green Mode**: BESS may only charge from PV surplus; single SoC track.
- **Grey Mode**: BESS may additionally charge from the grid; dual-SoC tracking
  separates green (PV-sourced) and grey (grid-sourced) energy.

Both modes support EEG/PPA floor pricing.  Because both spot prices and the
floor price are known constants at solve time, the effective price
``max(spot[t], fixed)`` is pre-computed and used directly in the objective.
This avoids the need for revenue-helper variables and their associated
linearisation constraints.

Variable indexing (Green Mode)
------------------------------
For *T* hourly timesteps (default 24):

===========  =============  ===================================================
Slice        Length         Variable
===========  =============  ===================================================
0   .. T-1   T              charge_pv[t]           – kWh charged from PV
T   .. 2T-1  T              discharge_green[t]     – kWh discharged (green)
2T  .. 3T-1  T              export_pv[t]           – kWh PV exported to grid
3T  .. 4T-1  T              curtail[t]             – kWh PV curtailed
===========  =============  ===================================================

Total Green Mode variables: 4T

Variable indexing (Grey Mode – extends Green Mode)
---------------------------------------------------
Green Mode variables occupy indices 0 .. 4T-1 as above, followed by:

===========  =============  ===================================================
Slice        Length         Variable
===========  =============  ===================================================
4T  .. 5T-1  T              charge_grid[t]         – kWh charged from grid
5T  .. 6T-1  T              discharge_grey[t]      – kWh discharged (grey)
===========  =============  ===================================================

Total Grey Mode variables: 6T

SoC is tracked implicitly via cumulative charge/discharge constraints
(no SoC decision variables needed).

Public API
----------
DailyDispatchResult – TypedDict with all per-hour arrays + end_soc.
optimize_day        – Solve the daily LP for one day (Green or Grey).
dispatch_offline_day – Produce dispatch results for a BESS-offline day.
"""

from __future__ import annotations

import logging
from typing import TypedDict

import numpy as np
from scipy.optimize import linprog

from pv_bess_model.config.defaults import LP_SOLVER_METHOD

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class DailyDispatchResult(TypedDict):
    """Per-hour dispatch arrays returned by :func:`optimize_day`.

    All arrays have length *T* (number of hourly timesteps, typically 24).
    Energy values are in kWh, revenue in €.
    """

    charge_pv: np.ndarray
    """kWh charged into BESS from PV per hour."""

    discharge_green: np.ndarray
    """kWh removed from BESS SoC (green) per hour."""

    export_pv: np.ndarray
    """kWh PV exported directly to grid per hour."""

    curtail: np.ndarray
    """kWh PV curtailed per hour."""

    charge_grid: np.ndarray
    """kWh charged into BESS from grid per hour (Grey Mode; zeros in Green)."""

    discharge_grey: np.ndarray
    """kWh removed from BESS SoC (grey) per hour (Grey Mode; zeros in Green)."""

    soc: np.ndarray
    """SoC at the *end* of each hour (kWh).  Green Mode: total SoC.
    Grey Mode: sum of green + grey SoC."""

    soc_green: np.ndarray
    """Green SoC at end of each hour (kWh).  In Green Mode equals ``soc``."""

    soc_grey: np.ndarray
    """Grey SoC at end of each hour (kWh).  Zeros in Green Mode."""

    revenue: np.ndarray
    """Hourly revenue in € (export + discharge revenue, net of grid import cost)."""

    end_soc: float
    """Total SoC at the end of the last hour (kWh), for day-to-day coupling."""

    end_soc_green: float
    """Green SoC at end of last hour.  Equals *end_soc* in Green Mode."""

    end_soc_grey: float
    """Grey SoC at end of last hour.  0.0 in Green Mode."""


# ---------------------------------------------------------------------------
# Helper: compute effective prices
# ---------------------------------------------------------------------------


def _effective_green_price(
    spot_prices_eur_per_kwh: np.ndarray,
    price_fixed_eur_per_kwh: float,
) -> np.ndarray:
    """Pre-compute the effective green price per hour.

    ``effective[t] = max(spot[t], fixed)`` when a floor is active,
    otherwise just ``spot[t]``.
    """
    if price_fixed_eur_per_kwh > 0.0:
        return np.maximum(spot_prices_eur_per_kwh, price_fixed_eur_per_kwh)
    return spot_prices_eur_per_kwh.copy()


# ---------------------------------------------------------------------------
# Helper: build Green-Mode LP
# ---------------------------------------------------------------------------


def _build_green_lp(
    pv_production_kwh: np.ndarray,
    spot_prices_eur_per_kwh: np.ndarray,
    price_fixed_eur_per_kwh: float,
    rte: float,
    soc_min_kwh: float,
    soc_max_kwh: float,
    start_soc_kwh: float,
    max_charge_kw: float,
    max_discharge_kw: float,
    grid_max_kw: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct the Green-Mode LP matrices.

    Returns (c, A_ub, b_ub, A_eq, b_eq) suitable for ``scipy.optimize.linprog``.
    All inequality constraints are expressed as A_ub @ x <= b_ub.

    Parameters
    ----------
    pv_production_kwh:
        PV production per hour (kWh), length T.
    spot_prices_eur_per_kwh:
        Spot prices per hour (€/kWh), length T.
    price_fixed_eur_per_kwh:
        Fixed floor price (€/kWh) for EEG/PPA.  0.0 when no floor is active.
    rte:
        Round-trip efficiency as a fraction (e.g. 0.88).
    soc_min_kwh:
        Minimum SoC (kWh).
    soc_max_kwh:
        Maximum SoC (kWh).
    start_soc_kwh:
        SoC at the start of the day (kWh).
    max_charge_kw:
        Maximum charge power (kW = kWh/h).
    max_discharge_kw:
        Maximum discharge power (kW = kWh/h).
    grid_max_kw:
        Maximum grid export power (kW).

    Returns
    -------
    tuple
        (c, A_ub, b_ub, A_eq, b_eq)
    """
    T = len(pv_production_kwh)
    n_vars = 4 * T  # charge_pv, disch_green, export_pv, curtail

    # Effective green price: max(spot, fixed)
    eff = _effective_green_price(spot_prices_eur_per_kwh, price_fixed_eur_per_kwh)

    # --- Objective: max Σ(export[t]*eff[t] + disch_green[t]*RTE*eff[t]) ---
    # linprog minimises → negate
    c = np.zeros(n_vars)
    for t in range(T):
        c[2 * T + t] = -eff[t]            # export_pv[t]
        c[T + t] = -(rte * eff[t])         # discharge_green[t] × RTE × eff
    # charge_pv and curtail have zero objective coefficients

    # --- Equality constraints ---
    # PV energy balance: export[t] + charge_pv[t] + curtail[t] = pv[t]  ∀t
    n_eq = T
    A_eq = np.zeros((n_eq, n_vars))
    b_eq = np.zeros(n_eq)
    for t in range(T):
        A_eq[t, 2 * T + t] = 1.0   # export_pv[t]
        A_eq[t, t] = 1.0            # charge_pv[t]
        A_eq[t, 3 * T + t] = 1.0   # curtail[t]
        b_eq[t] = pv_production_kwh[t]

    # --- Inequality constraints (A_ub @ x <= b_ub) ---
    ub_rows: list[np.ndarray] = []
    ub_rhs: list[float] = []

    for t in range(T):
        # SoC upper: start + Σ_{s=0..t}(charge_pv[s] - disch_green[s]) ≤ soc_max
        row = np.zeros(n_vars)
        row[0: t + 1] = 1.0                    # charge_pv[0..t]
        row[T: T + t + 1] = -1.0               # -discharge_green[0..t]
        ub_rows.append(row)
        ub_rhs.append(soc_max_kwh - start_soc_kwh)

        # SoC lower: start + Σ(charge_pv) - Σ(disch_green) ≥ soc_min
        # → -Σ(charge_pv) + Σ(disch_green) ≤ start - soc_min
        row2 = np.zeros(n_vars)
        row2[0: t + 1] = -1.0
        row2[T: T + t + 1] = 1.0
        ub_rows.append(row2)
        ub_rhs.append(start_soc_kwh - soc_min_kwh)

    for t in range(T):
        # Charge power limit: charge_pv[t] ≤ max_charge_kw
        row = np.zeros(n_vars)
        row[t] = 1.0
        ub_rows.append(row)
        ub_rhs.append(max_charge_kw)

        # Discharge power limit: discharge_green[t] ≤ max_discharge_kw
        row = np.zeros(n_vars)
        row[T + t] = 1.0
        ub_rows.append(row)
        ub_rhs.append(max_discharge_kw)

    for t in range(T):
        # Grid connection limit: export_pv[t] + discharge_green[t] × RTE ≤ grid_max
        row = np.zeros(n_vars)
        row[2 * T + t] = 1.0        # export_pv[t]
        row[T + t] = rte             # discharge_green[t] × RTE
        ub_rows.append(row)
        ub_rhs.append(grid_max_kw)

    A_ub = np.array(ub_rows)
    b_ub = np.array(ub_rhs)

    return c, A_ub, b_ub, A_eq, b_eq


# ---------------------------------------------------------------------------
# Helper: build Grey-Mode LP
# ---------------------------------------------------------------------------


def _build_grey_lp(
    pv_production_kwh: np.ndarray,
    spot_prices_eur_per_kwh: np.ndarray,
    price_fixed_eur_per_kwh: float,
    rte: float,
    soc_min_kwh: float,
    soc_max_kwh: float,
    start_soc_green_kwh: float,
    start_soc_grey_kwh: float,
    max_charge_kw: float,
    max_discharge_kw: float,
    grid_max_kw: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct the Grey-Mode LP matrices.

    Grey Mode extends Green Mode with grid charging (``charge_grid``) and
    grey discharging (``discharge_grey``), plus dual-chamber SoC tracking.

    Parameters
    ----------
    pv_production_kwh:
        PV production per hour (kWh), length T.
    spot_prices_eur_per_kwh:
        Spot prices per hour (€/kWh), length T.
    price_fixed_eur_per_kwh:
        Fixed floor price (€/kWh) for EEG/PPA.
    rte:
        Round-trip efficiency as a fraction.
    soc_min_kwh:
        Minimum total SoC (kWh).
    soc_max_kwh:
        Maximum total SoC (kWh).
    start_soc_green_kwh:
        Green chamber SoC at start of day (kWh).
    start_soc_grey_kwh:
        Grey chamber SoC at start of day (kWh).
    max_charge_kw:
        Maximum charge power (kW).
    max_discharge_kw:
        Maximum discharge power (kW).
    grid_max_kw:
        Maximum grid export power (kW).

    Returns
    -------
    tuple
        (c, A_ub, b_ub, A_eq, b_eq)
    """
    T = len(pv_production_kwh)
    n_vars = 6 * T  # charge_pv, disch_green, export_pv, curtail, charge_grid, disch_grey

    # Effective green price: max(spot, fixed)
    eff = _effective_green_price(spot_prices_eur_per_kwh, price_fixed_eur_per_kwh)

    # --- Objective ---
    # max Σ[ export[t]*eff[t] + disch_green[t]*RTE*eff[t]
    #        + disch_grey[t]*RTE*spot[t] - charge_grid[t]*spot[t] ]
    c = np.zeros(n_vars)
    for t in range(T):
        c[2 * T + t] = -eff[t]                              # export_pv
        c[T + t] = -(rte * eff[t])                           # discharge_green
        c[5 * T + t] = -(rte * spot_prices_eur_per_kwh[t])   # discharge_grey revenue
        c[4 * T + t] = spot_prices_eur_per_kwh[t]            # charge_grid cost

    # --- Equality constraints ---
    # PV energy balance: export[t] + charge_pv[t] + curtail[t] = pv[t]
    n_eq = T
    A_eq = np.zeros((n_eq, n_vars))
    b_eq = np.zeros(n_eq)
    for t in range(T):
        A_eq[t, 2 * T + t] = 1.0   # export_pv
        A_eq[t, t] = 1.0            # charge_pv
        A_eq[t, 3 * T + t] = 1.0   # curtail
        b_eq[t] = pv_production_kwh[t]

    # --- Inequality constraints ---
    ub_rows: list[np.ndarray] = []
    ub_rhs: list[float] = []

    start_total = start_soc_green_kwh + start_soc_grey_kwh

    for t in range(T):
        # soc_green[t] ≥ 0
        # → -(start_green + Σ(cpv[0..t]) - Σ(dg[0..t])) ≤ 0
        # → -Σcpv + Σdg ≤ start_green
        row = np.zeros(n_vars)
        row[0: t + 1] = -1.0         # -charge_pv[0..t]
        row[T: T + t + 1] = 1.0      # +discharge_green[0..t]
        ub_rows.append(row)
        ub_rhs.append(start_soc_green_kwh)

        # soc_grey[t] ≥ 0
        row = np.zeros(n_vars)
        row[4 * T: 4 * T + t + 1] = -1.0   # -charge_grid[0..t]
        row[5 * T: 5 * T + t + 1] = 1.0    # +discharge_grey[0..t]
        ub_rows.append(row)
        ub_rhs.append(start_soc_grey_kwh)

    for t in range(T):
        # Total SoC upper: soc_green + soc_grey ≤ soc_max
        row = np.zeros(n_vars)
        row[0: t + 1] = 1.0                   # charge_pv
        row[T: T + t + 1] = -1.0              # -discharge_green
        row[4 * T: 4 * T + t + 1] = 1.0       # charge_grid
        row[5 * T: 5 * T + t + 1] = -1.0      # -discharge_grey
        ub_rows.append(row)
        ub_rhs.append(soc_max_kwh - start_total)

        # Total SoC lower: soc_green + soc_grey ≥ soc_min
        row = np.zeros(n_vars)
        row[0: t + 1] = -1.0
        row[T: T + t + 1] = 1.0
        row[4 * T: 4 * T + t + 1] = -1.0
        row[5 * T: 5 * T + t + 1] = 1.0
        ub_rows.append(row)
        ub_rhs.append(start_total - soc_min_kwh)

    for t in range(T):
        # Charge power: charge_pv[t] + charge_grid[t] ≤ max_charge_kw
        row = np.zeros(n_vars)
        row[t] = 1.0
        row[4 * T + t] = 1.0
        ub_rows.append(row)
        ub_rhs.append(max_charge_kw)

        # Discharge power: discharge_green[t] + discharge_grey[t] ≤ max_discharge_kw
        row = np.zeros(n_vars)
        row[T + t] = 1.0
        row[5 * T + t] = 1.0
        ub_rows.append(row)
        ub_rhs.append(max_discharge_kw)

    for t in range(T):
        # Grid connection: export_pv[t] + (disch_green[t] + disch_grey[t]) × RTE ≤ grid_max
        row = np.zeros(n_vars)
        row[2 * T + t] = 1.0
        row[T + t] = rte
        row[5 * T + t] = rte
        ub_rows.append(row)
        ub_rhs.append(grid_max_kw)

    A_ub = np.array(ub_rows)
    b_ub = np.array(ub_rhs)

    return c, A_ub, b_ub, A_eq, b_eq


# ---------------------------------------------------------------------------
# Extract results
# ---------------------------------------------------------------------------


def _extract_green_result(
    x: np.ndarray,
    T: int,
    spot_prices_eur_per_kwh: np.ndarray,
    price_fixed_eur_per_kwh: float,
    rte: float,
    start_soc_kwh: float,
) -> DailyDispatchResult:
    """Parse the LP solution vector into a :class:`DailyDispatchResult` (Green)."""
    charge_pv = x[0: T]
    discharge_green = x[T: 2 * T]
    export_pv = x[2 * T: 3 * T]
    curtail = x[3 * T: 4 * T]

    eff = _effective_green_price(spot_prices_eur_per_kwh, price_fixed_eur_per_kwh)

    # Reconstruct SoC trajectory
    soc = np.empty(T)
    cumulative = start_soc_kwh
    for t in range(T):
        cumulative += charge_pv[t] - discharge_green[t]
        soc[t] = cumulative

    # Revenue per hour
    revenue = export_pv * eff + discharge_green * rte * eff

    return DailyDispatchResult(
        charge_pv=charge_pv,
        discharge_green=discharge_green,
        export_pv=export_pv,
        curtail=curtail,
        charge_grid=np.zeros(T),
        discharge_grey=np.zeros(T),
        soc=soc,
        soc_green=soc.copy(),
        soc_grey=np.zeros(T),
        revenue=revenue,
        end_soc=float(soc[-1]),
        end_soc_green=float(soc[-1]),
        end_soc_grey=0.0,
    )


def _extract_grey_result(
    x: np.ndarray,
    T: int,
    spot_prices_eur_per_kwh: np.ndarray,
    price_fixed_eur_per_kwh: float,
    rte: float,
    start_soc_green_kwh: float,
    start_soc_grey_kwh: float,
) -> DailyDispatchResult:
    """Parse the LP solution vector into a :class:`DailyDispatchResult` (Grey)."""
    charge_pv = x[0: T]
    discharge_green = x[T: 2 * T]
    export_pv = x[2 * T: 3 * T]
    curtail = x[3 * T: 4 * T]
    charge_grid = x[4 * T: 5 * T]
    discharge_grey = x[5 * T: 6 * T]

    eff = _effective_green_price(spot_prices_eur_per_kwh, price_fixed_eur_per_kwh)

    # Reconstruct SoC trajectories
    soc_green = np.empty(T)
    soc_grey = np.empty(T)
    cum_green = start_soc_green_kwh
    cum_grey = start_soc_grey_kwh
    for t in range(T):
        cum_green += charge_pv[t] - discharge_green[t]
        cum_grey += charge_grid[t] - discharge_grey[t]
        soc_green[t] = cum_green
        soc_grey[t] = cum_grey

    soc = soc_green + soc_grey

    # Revenue: green at effective price, grey at spot, minus grid import cost
    revenue = (
        export_pv * eff
        + discharge_green * rte * eff
        + discharge_grey * rte * spot_prices_eur_per_kwh
        - charge_grid * spot_prices_eur_per_kwh
    )

    return DailyDispatchResult(
        charge_pv=charge_pv,
        discharge_green=discharge_green,
        export_pv=export_pv,
        curtail=curtail,
        charge_grid=charge_grid,
        discharge_grey=discharge_grey,
        soc=soc,
        soc_green=soc_green,
        soc_grey=soc_grey,
        revenue=revenue,
        end_soc=float(soc[-1]),
        end_soc_green=float(soc_green[-1]),
        end_soc_grey=float(soc_grey[-1]),
    )


# ---------------------------------------------------------------------------
# Public: optimize one day
# ---------------------------------------------------------------------------


def optimize_day(
    pv_production_kwh: np.ndarray,
    spot_prices_eur_per_kwh: np.ndarray,
    price_fixed_eur_per_kwh: float,
    rte: float,
    soc_min_kwh: float,
    soc_max_kwh: float,
    start_soc_kwh: float,
    max_charge_kw: float,
    max_discharge_kw: float,
    grid_max_kw: float,
    mode: str,
    start_soc_green_kwh: float | None = None,
    start_soc_grey_kwh: float | None = None,
) -> DailyDispatchResult:
    """Solve the daily dispatch LP for one day.

    The effective green price is ``max(spot[t], price_fixed)`` when a floor is
    active, else ``spot[t]``.  Since prices are known at solve time, this is
    pre-computed and used directly in the linear objective — no auxiliary
    revenue-helper variables are needed.

    Parameters
    ----------
    pv_production_kwh:
        PV production per hour (kWh), length T (typically 24).
    spot_prices_eur_per_kwh:
        Day-ahead spot prices per hour (€/kWh), length T.
    price_fixed_eur_per_kwh:
        Fixed floor price (€/kWh) for EEG/PPA.
        Set to 0.0 when no floor is active.
    rte:
        Round-trip efficiency as a fraction (e.g. 0.88).
    soc_min_kwh:
        Minimum allowable SoC (kWh).
    soc_max_kwh:
        Maximum allowable SoC (kWh).
    start_soc_kwh:
        Total SoC at the start of the day (kWh).
    max_charge_kw:
        Maximum charge power (kW = kWh/h for 1h timesteps).
    max_discharge_kw:
        Maximum discharge power (kW).
    grid_max_kw:
        Maximum grid export power (kW).
    mode:
        Operating mode: ``"green"`` or ``"grey"``.
    start_soc_green_kwh:
        Green SoC at start (Grey Mode only). Defaults to *start_soc_kwh*.
    start_soc_grey_kwh:
        Grey SoC at start (Grey Mode only). Defaults to 0.0.

    Returns
    -------
    DailyDispatchResult
        All per-hour dispatch arrays and end-of-day SoC values.

    Raises
    ------
    ValueError
        If mode is neither ``"green"`` nor ``"grey"``.
    """
    T = len(pv_production_kwh)

    if mode == "green":
        c, A_ub, b_ub, A_eq, b_eq = _build_green_lp(
            pv_production_kwh=pv_production_kwh,
            spot_prices_eur_per_kwh=spot_prices_eur_per_kwh,
            price_fixed_eur_per_kwh=price_fixed_eur_per_kwh,
            rte=rte,
            soc_min_kwh=soc_min_kwh,
            soc_max_kwh=soc_max_kwh,
            start_soc_kwh=start_soc_kwh,
            max_charge_kw=max_charge_kw,
            max_discharge_kw=max_discharge_kw,
            grid_max_kw=grid_max_kw,
        )
        n_vars = 4 * T
        soc_green_start = start_soc_kwh
        soc_grey_start = 0.0
    elif mode == "grey":
        soc_green_start = (
            start_soc_green_kwh if start_soc_green_kwh is not None else start_soc_kwh
        )
        soc_grey_start = (
            start_soc_grey_kwh if start_soc_grey_kwh is not None else 0.0
        )
        c, A_ub, b_ub, A_eq, b_eq = _build_grey_lp(
            pv_production_kwh=pv_production_kwh,
            spot_prices_eur_per_kwh=spot_prices_eur_per_kwh,
            price_fixed_eur_per_kwh=price_fixed_eur_per_kwh,
            rte=rte,
            soc_min_kwh=soc_min_kwh,
            soc_max_kwh=soc_max_kwh,
            start_soc_green_kwh=soc_green_start,
            start_soc_grey_kwh=soc_grey_start,
            max_charge_kw=max_charge_kw,
            max_discharge_kw=max_discharge_kw,
            grid_max_kw=grid_max_kw,
        )
        n_vars = 6 * T
    else:
        raise ValueError(f"Unknown operating mode: '{mode}'. Use 'green' or 'grey'.")

    # Variable bounds: all ≥ 0
    bounds = [(0.0, None)] * n_vars

    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method=LP_SOLVER_METHOD,
    )

    if not result.success:
        logger.warning(
            "LP solve failed (status=%d: %s). Falling back to zero dispatch.",
            result.status,
            result.message,
        )
        return dispatch_offline_day(
            pv_production_kwh=pv_production_kwh,
            spot_prices_eur_per_kwh=spot_prices_eur_per_kwh,
            price_fixed_eur_per_kwh=price_fixed_eur_per_kwh,
            grid_max_kw=grid_max_kw,
            start_soc_kwh=start_soc_kwh,
            start_soc_green_kwh=soc_green_start,
            start_soc_grey_kwh=soc_grey_start,
        )

    x = result.x

    if mode == "green":
        return _extract_green_result(
            x, T, spot_prices_eur_per_kwh, price_fixed_eur_per_kwh, rte, start_soc_kwh
        )
    else:
        return _extract_grey_result(
            x, T, spot_prices_eur_per_kwh, price_fixed_eur_per_kwh,
            rte, soc_green_start, soc_grey_start,
        )


# ---------------------------------------------------------------------------
# Public: BESS offline day
# ---------------------------------------------------------------------------


def dispatch_offline_day(
    pv_production_kwh: np.ndarray,
    spot_prices_eur_per_kwh: np.ndarray,
    price_fixed_eur_per_kwh: float,
    grid_max_kw: float,
    start_soc_kwh: float,
    start_soc_green_kwh: float | None = None,
    start_soc_grey_kwh: float | None = None,
) -> DailyDispatchResult:
    """Produce dispatch results for a BESS-offline day.

    When the BESS is offline, all BESS decision variables are zero.
    PV is dispatched directly: ``export[t] = min(pv[t], grid_max)``,
    remainder is curtailed.  SoC is frozen at the carry-over value.

    Parameters
    ----------
    pv_production_kwh:
        PV production per hour (kWh), length T.
    spot_prices_eur_per_kwh:
        Spot prices per hour (€/kWh), length T.
    price_fixed_eur_per_kwh:
        Fixed floor price (€/kWh) for revenue calculation.
    grid_max_kw:
        Maximum grid export power (kW).
    start_soc_kwh:
        Total SoC carried over (frozen, kWh).
    start_soc_green_kwh:
        Green SoC carried over. Defaults to *start_soc_kwh*.
    start_soc_grey_kwh:
        Grey SoC carried over. Defaults to 0.0.

    Returns
    -------
    DailyDispatchResult
        Dispatch with all BESS flows at zero; SoC frozen.
    """
    T = len(pv_production_kwh)
    soc_green_val = (
        start_soc_green_kwh if start_soc_green_kwh is not None else start_soc_kwh
    )
    soc_grey_val = (
        start_soc_grey_kwh if start_soc_grey_kwh is not None else 0.0
    )

    export_pv = np.minimum(pv_production_kwh, grid_max_kw)
    curtail = pv_production_kwh - export_pv

    # Effective price per kWh: max(spot, fixed) if fixed > 0
    eff = _effective_green_price(spot_prices_eur_per_kwh, price_fixed_eur_per_kwh)
    revenue = export_pv * eff

    return DailyDispatchResult(
        charge_pv=np.zeros(T),
        discharge_green=np.zeros(T),
        export_pv=export_pv,
        curtail=curtail,
        charge_grid=np.zeros(T),
        discharge_grey=np.zeros(T),
        soc=np.full(T, start_soc_kwh),
        soc_green=np.full(T, soc_green_val),
        soc_grey=np.full(T, soc_grey_val),
        revenue=revenue,
        end_soc=start_soc_kwh,
        end_soc_green=soc_green_val,
        end_soc_grey=soc_grey_val,
    )
