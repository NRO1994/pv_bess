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

Unit conventions
----------------
All inputs and outputs follow a single, consistent unit scheme:

========  ======  ====================================================
Quantity  Unit    Notes
========  ======  ====================================================
Energy    kWh     PV production, charge/discharge amounts, SoC levels
Power     kW      Charge/discharge limits, grid export limit.
                  Equivalent to kWh/h for 1-hour timesteps.
Price     €/kWh   Spot prices AND floor price.  The price loader
                  converts CSV €/MWh → €/kWh before passing to this
                  module.  EEG/PPA modules also return €/kWh.
Revenue   €       Hourly revenue = energy (kWh) × price (€/kWh).
RTE       frac    Round-trip efficiency as a fraction in (0, 1],
                  e.g. 0.88 for 88 %.
========  ======  ====================================================

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
BessParams           – Frozen dataclass bundling BESS physical parameters.
DailyDispatchResult  – TypedDict with all per-hour arrays + end_soc.
OperatingMode        – Literal type alias for ``"green"`` | ``"grey"``.
optimize_day         – Solve the daily LP for one day (Green or Grey).
dispatch_offline_day – Produce dispatch results for a BESS-offline day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np
from scipy.optimize import linprog

from pv_bess_model.config.defaults import LP_SOLVER_METHOD

logger = logging.getLogger(__name__)

#: Accepted operating-mode values.
OperatingMode = Literal["green", "grey"]


# ---------------------------------------------------------------------------
# BESS parameter bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BessParams:
    """Physical BESS parameters that stay constant within a project year.

    All values are already computed for the current degradation state; the
    dispatch engine is responsible for applying annual degradation *before*
    constructing this object.

    Attributes
    ----------
    max_charge_kw : float
        Maximum charging power in **kW** (= kWh/h for 1 h timesteps).
    max_discharge_kw : float
        Maximum discharging power in **kW**.
    round_trip_efficiency : float
        Round-trip efficiency as a **fraction** in (0, 1], e.g. 0.88.
        Losses are applied on discharge only.
    soc_min_kwh : float
        Minimum allowable state-of-charge in **kWh**.
    soc_max_kwh : float
        Maximum allowable state-of-charge in **kWh**.
    """

    max_charge_kw: float
    max_discharge_kw: float
    round_trip_efficiency: float
    soc_min_kwh: float
    soc_max_kwh: float
    timestep_hours: float = 1.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class DailyDispatchResult(TypedDict):
    """Per-hour dispatch arrays returned by :func:`optimize_day`.

    All energy arrays have length *T* (number of hourly timesteps, typically
    24) and are in **kWh**.  Revenue arrays are in **€**.

    Scalar ``end_soc*`` fields carry over the state for day-to-day coupling.
    """

    charge_pv: np.ndarray
    """kWh charged into BESS from PV surplus, per hour. shape (T,)"""

    discharge_green: np.ndarray
    """kWh removed from BESS SoC (green chamber), per hour. shape (T,)"""

    export_pv: np.ndarray
    """kWh PV exported directly to grid, per hour. shape (T,)"""

    curtail: np.ndarray
    """kWh PV curtailed (wasted), per hour. shape (T,)"""

    charge_grid: np.ndarray
    """kWh charged into BESS from grid, per hour. shape (T,)
    Grey Mode only; zeros in Green Mode."""

    discharge_grey: np.ndarray
    """kWh removed from BESS SoC (grey chamber), per hour. shape (T,)
    Grey Mode only; zeros in Green Mode."""

    soc: np.ndarray
    """Total SoC at the *end* of each hour, in kWh. shape (T,)
    Green Mode: equals soc_green.  Grey Mode: soc_green + soc_grey."""

    soc_green: np.ndarray
    """Green-chamber SoC at end of each hour, in kWh. shape (T,)"""

    soc_grey: np.ndarray
    """Grey-chamber SoC at end of each hour, in kWh. shape (T,)
    Zeros in Green Mode."""

    revenue: np.ndarray
    """Hourly revenue in €. shape (T,)
    = export × eff_price + discharge_green × RTE × spot
      + discharge_grey × RTE × spot  − charge_grid × spot"""

    end_soc: float
    """Total SoC at end of last hour (kWh).  For day-to-day coupling."""

    end_soc_green: float
    """Green SoC at end of last hour (kWh).  Equals end_soc in Green Mode."""

    end_soc_grey: float
    """Grey SoC at end of last hour (kWh).  0.0 in Green Mode."""

    effective_price: np.ndarray
    """Effective Price in EUR/kWh. shape (T,)"""


# ---------------------------------------------------------------------------
# Helper: compute effective prices
# ---------------------------------------------------------------------------


def _effective_green_price(
        spot_prices_eur_per_kwh: np.ndarray,
        price_fixed_eur_per_kwh: float,
        goo_premium_eur_per_kwh: float = 0.0,
        price_cap_eur_per_kwh: float = 0.0,
) -> np.ndarray:
    """Pre-compute the effective green price per hour (€/kWh).

    ``effective[t] = max(spot[t], fixed) + goo`` when only a floor is active.
    ``effective[t] = clip(spot[t], fixed, cap) + goo`` when a collar is active.
    Otherwise ``spot[t] + goo`` (or just ``spot[t]`` when goo = 0).

    The GoO premium is added **after** the floor/clip comparison so that the
    seller always receives ``goo`` on top of the effective price.

    Parameters
    ----------
    spot_prices_eur_per_kwh:
        Hourly spot prices (€/kWh).
    price_fixed_eur_per_kwh:
        Floor price (€/kWh).  0.0 when no floor active.
    goo_premium_eur_per_kwh:
        GoO premium added after floor/clip (€/kWh).  Defaults to 0.0.
    price_cap_eur_per_kwh:
        Cap price (€/kWh) for PPA Collar.  0.0 means no cap (unbounded).
    """
    if price_fixed_eur_per_kwh > 0.0:
        eff = np.maximum(spot_prices_eur_per_kwh, price_fixed_eur_per_kwh)
    else:
        eff = spot_prices_eur_per_kwh.copy()
    if price_cap_eur_per_kwh > 0.0:
        eff = np.minimum(eff, price_cap_eur_per_kwh)
    if goo_premium_eur_per_kwh > 0.0:
        eff = eff + goo_premium_eur_per_kwh
    return eff


# ---------------------------------------------------------------------------
# Helper: build Green-Mode LP
# ---------------------------------------------------------------------------


def _build_green_lp(
        pv_production_kwh: np.ndarray,
        eff_prices: np.ndarray,
        spot_prices: np.ndarray,
        rte: float,
        soc_min_kwh: float,
        soc_max_kwh: float,
        start_soc_kwh: float,
        max_charge_kw: float,
        max_discharge_kw: float,
        grid_max_kw: float,
        grid_loss_factor: float = 1.0,
        timestep_hours: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct the Green-Mode LP matrices.

    Returns (c, A_ub, b_ub, A_eq, b_eq) suitable for ``scipy.optimize.linprog``.

    *eff_prices* (€/kWh, floor-adjusted) are used for PV direct export.
    *spot_prices* (€/kWh, raw spot) are used for BESS discharge revenue,
    because BESS discharge is a separate spot-market revenue stream.
    Charging cost (opportunity cost of foregone PV export) remains at
    *eff_prices* implicitly via the energy-balance constraint.
    """
    T = len(pv_production_kwh)
    n_vars = 4 * T  # charge_pv, disch_green, export_pv, curtail

    # --- Objective: max Σ(export[t]*glf*eff[t] + disch_green[t]*RTE*glf*spot[t]) ---
    # linprog minimises → negate
    c = np.zeros(n_vars)
    for t in range(T):
        c[2 * T + t] = -(grid_loss_factor * eff_prices[t])  # export_pv[t] × glf × eff
        c[T + t] = -(rte * grid_loss_factor * spot_prices[t])  # discharge_green[t] × RTE × glf × spot

    # --- Equality constraints ---
    # PV energy balance: export[t] + charge_pv[t] + curtail[t] = pv[t]  ∀t
    A_eq = np.zeros((T, n_vars))
    b_eq = np.zeros(T)
    for t in range(T):
        A_eq[t, 2 * T + t] = 1.0  # export_pv[t]
        A_eq[t, t] = 1.0  # charge_pv[t]
        A_eq[t, 3 * T + t] = 1.0  # curtail[t]
        b_eq[t] = pv_production_kwh[t]

    # --- Inequality constraints (A_ub @ x <= b_ub) ---
    ub_rows: list[np.ndarray] = []
    ub_rhs: list[float] = []

    for t in range(T):
        # SoC upper: start + Σ_{s=0..t}(charge_pv[s] - disch_green[s]) ≤ soc_max
        row = np.zeros(n_vars)
        row[0: t + 1] = 1.0  # charge_pv[0..t]
        row[T: T + t + 1] = -1.0  # -discharge_green[0..t]
        ub_rows.append(row)
        ub_rhs.append(soc_max_kwh - start_soc_kwh)

        # SoC lower: start + Σ(charge_pv) - Σ(disch_green) ≥ soc_min
        # → -Σ(charge_pv) + Σ(disch_green) ≤ start - soc_min
        row2 = np.zeros(n_vars)
        row2[0: t + 1] = -1.0
        row2[T: T + t + 1] = 1.0
        ub_rows.append(row2)
        ub_rhs.append(start_soc_kwh - soc_min_kwh)

    # Energy limit per timestep = power (kW) × timestep_hours (h) → kWh
    max_charge_energy = max_charge_kw * timestep_hours
    max_discharge_energy = max_discharge_kw * timestep_hours
    grid_max_energy = grid_max_kw * timestep_hours

    for t in range(T):
        # Charge power limit: charge_pv[t] ≤ max_charge_energy
        row = np.zeros(n_vars)
        row[t] = 1.0
        ub_rows.append(row)
        ub_rhs.append(max_charge_energy)

        # Discharge power limit: discharge_green[t] ≤ max_discharge_energy
        row = np.zeros(n_vars)
        row[T + t] = 1.0
        ub_rows.append(row)
        ub_rhs.append(max_discharge_energy)

    for t in range(T):
        # Grid connection limit: export_pv[t] × glf + discharge_green[t] × RTE ≤ grid_max_energy
        row = np.zeros(n_vars)
        row[2 * T + t] = grid_loss_factor  # export_pv[t] × glf
        row[T + t] = rte  # discharge_green[t] × RTE
        ub_rows.append(row)
        ub_rhs.append(grid_max_energy)

    A_ub = np.array(ub_rows)
    b_ub = np.array(ub_rhs)

    return c, A_ub, b_ub, A_eq, b_eq


# ---------------------------------------------------------------------------
# Helper: build Grey-Mode LP
# ---------------------------------------------------------------------------


def _build_grey_lp(
        pv_production_kwh: np.ndarray,
        spot_prices_eur_per_kwh: np.ndarray,
        eff_prices: np.ndarray,
        rte: float,
        soc_min_kwh: float,
        soc_max_kwh: float,
        start_soc_green_kwh: float,
        start_soc_grey_kwh: float,
        max_charge_kw: float,
        max_discharge_kw: float,
        grid_max_kw: float,
        grid_loss_factor: float = 1.0,
        timestep_hours: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct the Grey-Mode LP matrices.

    Grey Mode extends Green Mode with grid charging (``charge_grid``) and
    grey discharging (``discharge_grey``), plus dual-chamber SoC tracking.

    *eff_prices* are €/kWh (floor-adjusted, for green energy).
    *spot_prices_eur_per_kwh* are raw spot (€/kWh, for grey energy).
    """
    T = len(pv_production_kwh)
    n_vars = 6 * T  # charge_pv, disch_green, export_pv, curtail, charge_grid, disch_grey

    # --- Objective ---
    # max Σ[ export[t]*glf*eff[t] + disch_green[t]*RTE*glf*spot[t]
    #        + disch_grey[t]*RTE*spot[t] - charge_grid[t]*spot[t] ]
    # BESS discharge (both green and grey) is valued at spot prices,
    # because it is a separate spot-market revenue stream.
    c = np.zeros(n_vars)
    for t in range(T):
        c[2 * T + t] = -(grid_loss_factor * eff_prices[t])  # export_pv × glf × eff
        c[T + t] = -(rte * grid_loss_factor * eff_prices[t])  # discharge_green × RTE × glf × spot
        c[5 * T + t] = -(rte * spot_prices_eur_per_kwh[t])  # discharge_grey revenue (no glf)
        c[4 * T + t] = spot_prices_eur_per_kwh[t]  # charge_grid cost

    # --- Equality constraints ---
    # PV energy balance: export[t] + charge_pv[t] + curtail[t] = pv[t]
    A_eq = np.zeros((T, n_vars))
    b_eq = np.zeros(T)
    for t in range(T):
        A_eq[t, 2 * T + t] = 1.0  # export_pv
        A_eq[t, t] = 1.0  # charge_pv
        A_eq[t, 3 * T + t] = 1.0  # curtail
        b_eq[t] = pv_production_kwh[t]

    # --- Inequality constraints ---
    ub_rows: list[np.ndarray] = []
    ub_rhs: list[float] = []

    start_total = start_soc_green_kwh + start_soc_grey_kwh

    for t in range(T):
        # soc_green[t] ≥ 0
        # → -Σcpv + Σdg ≤ start_green
        row = np.zeros(n_vars)
        row[0: t + 1] = -1.0  # -charge_pv[0..t]
        row[T: T + t + 1] = 1.0  # +discharge_green[0..t]
        ub_rows.append(row)
        ub_rhs.append(start_soc_green_kwh)

        # soc_grey[t] ≥ 0
        row = np.zeros(n_vars)
        row[4 * T: 4 * T + t + 1] = -1.0  # -charge_grid[0..t]
        row[5 * T: 5 * T + t + 1] = 1.0  # +discharge_grey[0..t]
        ub_rows.append(row)
        ub_rhs.append(start_soc_grey_kwh)

    for t in range(T):
        # Total SoC upper: soc_green + soc_grey ≤ soc_max
        row = np.zeros(n_vars)
        row[0: t + 1] = 1.0  # charge_pv
        row[T: T + t + 1] = -1.0  # -discharge_green
        row[4 * T: 4 * T + t + 1] = 1.0  # charge_grid
        row[5 * T: 5 * T + t + 1] = -1.0  # -discharge_grey
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

    # Energy limit per timestep = power (kW) × timestep_hours (h) → kWh
    max_charge_energy = max_charge_kw * timestep_hours
    max_discharge_energy = max_discharge_kw * timestep_hours
    grid_max_energy = grid_max_kw * timestep_hours

    for t in range(T):
        # Charge power: charge_pv[t] + charge_grid[t] ≤ max_charge_energy
        row = np.zeros(n_vars)
        row[t] = 1.0
        row[4 * T + t] = 1.0
        ub_rows.append(row)
        ub_rhs.append(max_charge_energy)

        # Discharge power: discharge_green[t] + discharge_grey[t] ≤ max_discharge_energy
        row = np.zeros(n_vars)
        row[T + t] = 1.0
        row[5 * T + t] = 1.0
        ub_rows.append(row)
        ub_rhs.append(max_discharge_energy)

    for t in range(T):
        # Grid connection: export_pv[t] × glf + (disch_green[t] + disch_grey[t]) × RTE ≤ grid_max_energy
        row = np.zeros(n_vars)
        row[2 * T + t] = grid_loss_factor  # export_pv × glf
        row[T + t] = rte
        row[5 * T + t] = rte
        ub_rows.append(row)
        ub_rhs.append(grid_max_energy)

    A_ub = np.array(ub_rows)
    b_ub = np.array(ub_rhs)

    return c, A_ub, b_ub, A_eq, b_eq


# ---------------------------------------------------------------------------
# Extract results
# ---------------------------------------------------------------------------


def _extract_green_result(x: np.ndarray, T: int, eff_prices: np.ndarray, rte: float, start_soc_kwh: float,
                          grid_loss_factor: float = 1.0) -> DailyDispatchResult:
    """Parse the LP solution vector into a :class:`DailyDispatchResult` (Green).

    PV export revenue uses *eff_prices* (floor/cap-adjusted).
    BESS discharge revenue uses *spot_prices* (raw spot).
    """
    charge_pv = x[0: T]
    discharge_green = x[T: 2 * T]
    export_pv = x[2 * T: 3 * T]
    curtail = x[3 * T: 4 * T]

    # Reconstruct SoC trajectory
    soc = np.empty(T)
    cumulative = start_soc_kwh
    for t in range(T):
        cumulative += charge_pv[t] - discharge_green[t]
        soc[t] = cumulative

    # Add losses to the energy flow
    discharge_green = discharge_green * grid_loss_factor * rte
    export_pv = export_pv * grid_loss_factor
    curtail = curtail * grid_loss_factor

    # Revenue per hour (€):
    # PV export at effective price (floor/cap protected)
    revenue = (export_pv + discharge_green) * eff_prices

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
        effective_price=eff_prices,
    )


def _extract_grey_result(
        x: np.ndarray,
        T: int,
        spot_prices_eur_per_kwh: np.ndarray,
        eff_prices: np.ndarray,
        rte: float,
        start_soc_green_kwh: float,
        start_soc_grey_kwh: float,
        grid_loss_factor: float = 1.0,
) -> DailyDispatchResult:
    """Parse the LP solution vector into a :class:`DailyDispatchResult` (Grey)."""
    charge_pv = x[0: T]
    discharge_green = x[T: 2 * T]
    export_pv = x[2 * T: 3 * T]
    curtail = x[3 * T: 4 * T]
    charge_grid = x[4 * T: 5 * T]
    discharge_grey = x[5 * T: 6 * T]

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

    # Add losses to the energy flow
    discharge_green = discharge_green * grid_loss_factor * rte
    discharge_grey = discharge_grey * rte
    export_pv = export_pv * grid_loss_factor
    curtail = curtail * grid_loss_factor

    # Revenue (€): PV export and green discharge at effective price × glf,
    # BESS discharge (grey) at spot, minus grid import at spot
    revenue = ((export_pv + discharge_green) * eff_prices +
               (discharge_grey - charge_grid) * spot_prices_eur_per_kwh)

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
        effective_price=eff_prices,
    )


# ---------------------------------------------------------------------------
# Public: optimize one day
# ---------------------------------------------------------------------------


def optimize_day(
        pv_production_kwh: np.ndarray,
        spot_prices_eur_per_kwh: np.ndarray,
        price_fixed_eur_per_kwh: float,
        bess: BessParams,
        grid_max_kw: float,
        mode: OperatingMode,
        start_soc_kwh: float,
        start_soc_green_kwh: float | None = None,
        start_soc_grey_kwh: float | None = None,
        goo_premium_eur_per_kwh: float = 0.0,
        price_cap_eur_per_kwh: float = 0.0,
        grid_loss_factor: float = 1.0,
) -> DailyDispatchResult:
    """Solve the daily dispatch LP for one day.

    Parameters
    ----------
    pv_production_kwh : np.ndarray, shape (T,)
        PV production per hour in **kWh**.  *T* is typically 24.
    spot_prices_eur_per_kwh : np.ndarray, shape (T,)
        Day-ahead spot prices per hour in **€/kWh**.
    price_fixed_eur_per_kwh : float
        Fixed floor price in **€/kWh** for EEG/PPA (WITHOUT GoO premium).
        Set to **0.0** when no floor is active.
    bess : BessParams
        Physical BESS parameters (power limits, RTE, SoC bounds).
    grid_max_kw : float
        Maximum grid export power in **kW**.
    mode : ``"green"`` | ``"grey"``
        Operating mode.
    start_soc_kwh : float
        Total SoC at the start of the day in **kWh**.
    start_soc_green_kwh : float | None
        Green-chamber SoC at start in **kWh** (Grey Mode only).
        Defaults to *start_soc_kwh* (entire SoC is green).
    start_soc_grey_kwh : float | None
        Grey-chamber SoC at start in **kWh** (Grey Mode only).
        Defaults to 0.0.
    goo_premium_eur_per_kwh : float
        Guarantee-of-Origin premium in **€/kWh**, added to the effective
        green price after the floor/clip comparison.  Defaults to 0.0.
    price_cap_eur_per_kwh : float
        Cap price in **€/kWh** for PPA Collar.  Set to **0.0** when no cap
        is active (unbounded upside).  Defaults to 0.0.
    grid_loss_factor : float
        Grid loss factor in (0, 1].  Applied to green energy (PV export and
        green BESS discharge) in the objective function, grid constraint, and
        revenue calculation.  Grey energy is **not** affected.  Defaults to
        1.0 (no losses).

    Returns
    -------
    DailyDispatchResult
        All per-hour dispatch arrays (kWh / €) and end-of-day SoC (kWh).

    Raises
    ------
    ValueError
        If *mode* is neither ``"green"`` nor ``"grey"``.
    """
    T = len(pv_production_kwh)
    rte = bess.round_trip_efficiency

    # Pre-compute effective green price: clip(spot, fixed, cap) + goo — €/kWh
    eff = _effective_green_price(
        spot_prices_eur_per_kwh, price_fixed_eur_per_kwh,
        goo_premium_eur_per_kwh, price_cap_eur_per_kwh,
    )

    if mode == "green":
        c, A_ub, b_ub, A_eq, b_eq = _build_green_lp(
            pv_production_kwh=pv_production_kwh,
            eff_prices=eff,
            spot_prices=spot_prices_eur_per_kwh,
            rte=rte,
            soc_min_kwh=bess.soc_min_kwh,
            soc_max_kwh=bess.soc_max_kwh,
            start_soc_kwh=start_soc_kwh,
            max_charge_kw=bess.max_charge_kw,
            max_discharge_kw=bess.max_discharge_kw,
            grid_max_kw=grid_max_kw,
            grid_loss_factor=grid_loss_factor,
            timestep_hours=bess.timestep_hours,
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
            eff_prices=eff,
            rte=rte,
            soc_min_kwh=bess.soc_min_kwh,
            soc_max_kwh=bess.soc_max_kwh,
            start_soc_green_kwh=soc_green_start,
            start_soc_grey_kwh=soc_grey_start,
            max_charge_kw=bess.max_charge_kw,
            max_discharge_kw=bess.max_discharge_kw,
            grid_max_kw=grid_max_kw,
            grid_loss_factor=grid_loss_factor,
            timestep_hours=bess.timestep_hours,
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
            goo_premium_eur_per_kwh=goo_premium_eur_per_kwh,
            price_cap_eur_per_kwh=price_cap_eur_per_kwh,
            grid_loss_factor=grid_loss_factor,
            timestep_hours=bess.timestep_hours,
        )

    x = result.x

    if mode == "green":
        return _extract_green_result(x, T, eff, rte, start_soc_kwh, grid_loss_factor)
    else:
        return _extract_grey_result(
            x, T, spot_prices_eur_per_kwh, eff, rte,
            soc_green_start, soc_grey_start, grid_loss_factor,
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
        goo_premium_eur_per_kwh: float = 0.0,
        price_cap_eur_per_kwh: float = 0.0,
        grid_loss_factor: float = 1.0,
        timestep_hours: float = 1.0,
) -> DailyDispatchResult:
    """Produce dispatch results for a BESS-offline day.

    When the BESS is offline, all BESS decision variables are zero.
    PV is dispatched directly: ``export[t] = min(pv[t], grid_max)``,
    remainder is curtailed.  SoC is frozen at the carry-over value.

    Parameters
    ----------
    pv_production_kwh : np.ndarray, shape (T,)
        PV production per hour in **kWh**.
    spot_prices_eur_per_kwh : np.ndarray, shape (T,)
        Spot prices per hour in **€/kWh**.
    price_fixed_eur_per_kwh : float
        Fixed floor price in **€/kWh** (WITHOUT GoO).  0.0 when no floor.
    grid_max_kw : float
        Maximum grid export power in **kW**.
    start_soc_kwh : float
        Total SoC carried over (frozen) in **kWh**.
    start_soc_green_kwh : float | None
        Green SoC carried over in **kWh**.  Defaults to *start_soc_kwh*.
    start_soc_grey_kwh : float | None
        Grey SoC carried over in **kWh**.  Defaults to 0.0.
    goo_premium_eur_per_kwh : float
        Guarantee-of-Origin premium in **€/kWh**, added after the floor/clip
        comparison.  Defaults to 0.0.
    price_cap_eur_per_kwh : float
        Cap price in **€/kWh** for PPA Collar.  0.0 means no cap.
        Defaults to 0.0.

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

    grid_max_energy = grid_max_kw * timestep_hours
    export_pv = np.minimum(pv_production_kwh * grid_loss_factor, grid_max_energy)
    grid_curtail = np.maximum(pv_production_kwh * grid_loss_factor - grid_max_energy, 0)

    # Effective price per kWh: clip(spot, fixed, cap) + goo
    eff = _effective_green_price(
        spot_prices_eur_per_kwh, price_fixed_eur_per_kwh,
        goo_premium_eur_per_kwh, price_cap_eur_per_kwh,
    )

    # Consider negative prices as curtailment
    price_curtail = np.zeros(len(pv_production_kwh))
    negative_price_mask = eff < 0
    price_curtail[negative_price_mask] = export_pv[negative_price_mask]
    export_pv[negative_price_mask] = 0
    grid_curtail[negative_price_mask] = 0

    curtail = grid_curtail + price_curtail

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
        effective_price=eff
    )
