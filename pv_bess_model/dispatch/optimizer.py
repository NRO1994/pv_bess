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

When a **baseload PPA** is active (``baseload_kw > 0``), the LP uses raw spot
prices for all feed-in variables (no floor/cap/goo in the objective).  The
baseload settlement is a per-hour constant that does not affect LP decisions
and is computed post-hoc by the dispatch engine.

Simultaneous charging and discharging is prevented by fixing discharge
variables to zero whenever the spot price is negative.

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
4T  .. 5T-1  T              soc[t]                 – SoC at end of hour t (kWh)
===========  =============  ===================================================

Total Green Mode variables: 5T

Variable indexing (Grey Mode – extends Green Mode base variables)
-----------------------------------------------------------------
===========  =============  ===================================================
Slice        Length         Variable
===========  =============  ===================================================
0   .. T-1   T              charge_pv[t]           – kWh charged from PV
T   .. 2T-1  T              discharge_green[t]     – kWh discharged (green)
2T  .. 3T-1  T              export_pv[t]           – kWh PV exported to grid
3T  .. 4T-1  T              curtail[t]             – kWh PV curtailed
4T  .. 5T-1  T              charge_grid[t]         – kWh charged from grid
5T  .. 6T-1  T              discharge_grey[t]      – kWh discharged (grey)
6T  .. 7T-1  T              soc_green[t]           – green SoC at end of hour t
7T  .. 8T-1  T              soc_grey[t]            – grey SoC at end of hour t
===========  =============  ===================================================

Total Grey Mode variables: 8T

SoC is tracked via explicit decision variables with linking equality
constraints (staircase structure) and simple variable bounds.  This replaces
the previous cumulative-sum formulation and reduces constraint matrix density
from O(T²) to O(T) nonzeros.

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

    shortfall: np.ndarray
    """kWh shortfall below baseload commitment, per hour. shape (T,)
    Zero when no baseload PPA is active."""


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
# Helper: build Green-Mode LP (explicit SoC variables)
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
        baseload_kwh: float = 0.0,
        shortfall_penalty_prices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Construct the Green-Mode LP matrices with explicit SoC variables.

    Returns ``(c, A_ub, b_ub, A_eq, b_eq, bounds)`` suitable for
    ``scipy.optimize.linprog``.

    Variables without baseload: ``[charge_pv(T), discharge_green(T),
    export_pv(T), curtail(T), soc(T)]`` — total 5T.

    Variables with baseload: ``[charge_pv(T), discharge_green(T),
    export_pv(T), curtail(T), soc(T), shortfall(T)]`` — total 6T.

    When ``baseload_kwh > 0``, the LP adds shortfall variables that track
    how much the total grid export falls below the baseload commitment.
    The shortfall is penalised at ``shortfall_penalty_prices`` (typically
    ``max(spot, ppa_price) + goo``) to incentivise the BESS to discharge
    during shortfall hours.

    SoC is tracked via linking equality constraints (staircase structure)
    with variable bounds ``[soc_min, soc_max]``.

    Discharge is fixed to zero for hours with negative spot prices to prevent
    simultaneous charging and discharging.

    Parameters
    ----------
    baseload_kwh:
        Baseload commitment per interval in kWh.  0.0 when no baseload PPA.
    shortfall_penalty_prices:
        Effective PPA price array (€/kWh) for the shortfall penalty.
        Required when ``baseload_kwh > 0``.  Typically
        ``max(spot, ppa_price) + goo``.
    """
    T = len(pv_production_kwh)
    has_baseload = baseload_kwh > 0.0
    n_vars = 6 * T if has_baseload else 5 * T

    max_charge_energy = max_charge_kw * timestep_hours
    max_discharge_energy = max_discharge_kw * timestep_hours
    grid_max_energy = grid_max_kw * timestep_hours

    # --- Objective: max Σ(export[t]*glf*eff[t] + disch_green[t]*RTE*glf*spot[t]
    #                       - shortfall[t]*shortfall_penalty[t])  ---
    # linprog minimises → negate
    c = np.zeros(n_vars)
    for t in range(T):
        c[2 * T + t] = -(grid_loss_factor * eff_prices[t])  # export_pv[t]
        c[T + t] = -(rte * grid_loss_factor * spot_prices[t])  # discharge_green[t]
    if has_baseload and shortfall_penalty_prices is not None:
        for t in range(T):
            # Shortfall penalty: +shortfall_penalty_prices[t] (positive = cost in minimisation)
            c[5 * T + t] = shortfall_penalty_prices[t]

    # --- Equality constraints (2T rows) ---
    # 1. PV balance: export[t] + charge_pv[t] + curtail[t] = pv[t]     ∀t
    # 2. SoC linking: soc[t] - soc[t-1] - charge_pv[t] + disch[t] = 0  ∀t>0
    #                 soc[0] - charge_pv[0] + disch[0] = start_soc       t=0
    n_eq = 2 * T
    A_eq = np.zeros((n_eq, n_vars))
    b_eq = np.zeros(n_eq)

    for t in range(T):
        # PV energy balance (row t)
        A_eq[t, t] = 1.0            # charge_pv[t]
        A_eq[t, 2 * T + t] = 1.0    # export_pv[t]
        A_eq[t, 3 * T + t] = 1.0    # curtail[t]
        b_eq[t] = pv_production_kwh[t]

        # SoC linking (row T+t)
        row = T + t
        A_eq[row, 4 * T + t] = 1.0  # +soc[t]
        if t > 0:
            A_eq[row, 4 * T + t - 1] = -1.0  # -soc[t-1]
        A_eq[row, t] = -1.0         # -charge_pv[t]
        A_eq[row, T + t] = 1.0      # +discharge_green[t]
        b_eq[row] = start_soc_kwh if t == 0 else 0.0

    # --- Inequality constraints ---
    # Grid limit: export[t]*glf + discharge[t]*RTE ≤ grid_max       (T rows)
    # Baseload shortfall: -export[t]*glf - disch[t]*RTE*glf
    #                     - shortfall[t] ≤ -baseload_kwh              (T rows, if baseload)
    n_ub_rows = 2 * T if has_baseload else T
    A_ub = np.zeros((n_ub_rows, n_vars))
    b_ub = np.zeros(n_ub_rows)
    for t in range(T):
        # Grid limit (row t)
        A_ub[t, 2 * T + t] = grid_loss_factor  # export_pv[t]
        A_ub[t, T + t] = rte                    # discharge_green[t]
        b_ub[t] = grid_max_energy

    if has_baseload:
        for t in range(T):
            # Shortfall constraint (row T+t):
            # export_pv[t]*glf + discharge_green[t]*rte*glf + shortfall[t] >= baseload_kwh
            # In ≤ form: -export_pv[t]*glf - discharge_green[t]*rte*glf - shortfall[t] ≤ -baseload_kwh
            row = T + t
            A_ub[row, 2 * T + t] = -grid_loss_factor       # -export_pv[t]
            A_ub[row, T + t] = -(rte * grid_loss_factor)    # -discharge_green[t]
            A_ub[row, 5 * T + t] = -1.0                     # -shortfall[t]
            b_ub[row] = -baseload_kwh

    # --- Variable bounds ---
    # Charge/discharge power limits and SoC bounds as variable bounds.
    # Discharge fixed to 0 at negative spot prices (prevents simultaneous
    # charge/discharge).
    bounds: list[tuple[float, float | None]] = []
    for t in range(T):
        bounds.append((0.0, max_charge_energy))      # charge_pv[t]
    for t in range(T):
        ub = 0.0 if spot_prices[t] < 0 else max_discharge_energy
        bounds.append((0.0, ub))                      # discharge_green[t]
    for t in range(T):
        bounds.append((0.0, None))                    # export_pv[t]
    for t in range(T):
        bounds.append((0.0, None))                    # curtail[t]
    for t in range(T):
        bounds.append((soc_min_kwh, soc_max_kwh))    # soc[t]
    if has_baseload:
        for t in range(T):
            bounds.append((0.0, None))                # shortfall[t]

    return c, A_ub, b_ub, A_eq, b_eq, bounds


# ---------------------------------------------------------------------------
# Helper: build Grey-Mode LP (explicit SoC variables)
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
        baseload_kwh: float = 0.0,
        shortfall_penalty_prices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Construct the Grey-Mode LP matrices with explicit SoC variables.

    Returns ``(c, A_ub, b_ub, A_eq, b_eq, bounds)`` suitable for
    ``scipy.optimize.linprog``.

    Variables without baseload: ``[charge_pv(T), discharge_green(T),
    export_pv(T), curtail(T), charge_grid(T), discharge_grey(T),
    soc_green(T), soc_grey(T)]`` — total 8T.

    Variables with baseload: same as above plus ``shortfall(T)`` — total 9T.

    Dual-chamber SoC is tracked via linking equality constraints with
    individual variable bounds ``[0, soc_max]`` and total-SoC inequality
    constraints ``soc_min ≤ soc_green + soc_grey ≤ soc_max``.

    Discharge (both green and grey) is fixed to zero for hours with negative
    spot prices to prevent simultaneous charging and discharging.

    Parameters
    ----------
    baseload_kwh:
        Baseload commitment per interval in kWh.  0.0 when no baseload PPA.
    shortfall_penalty_prices:
        Effective PPA price array (€/kWh) for the shortfall penalty.
        Required when ``baseload_kwh > 0``.
    """
    T = len(pv_production_kwh)
    has_baseload = baseload_kwh > 0.0
    n_vars = 9 * T if has_baseload else 8 * T

    max_charge_energy = max_charge_kw * timestep_hours
    max_discharge_energy = max_discharge_kw * timestep_hours
    grid_max_energy = grid_max_kw * timestep_hours

    # --- Objective ---
    c = np.zeros(n_vars)
    for t in range(T):
        c[2 * T + t] = -(grid_loss_factor * eff_prices[t])            # export_pv
        c[T + t] = -(rte * grid_loss_factor * eff_prices[t])          # discharge_green
        c[5 * T + t] = -(rte * spot_prices_eur_per_kwh[t])            # discharge_grey
        c[4 * T + t] = spot_prices_eur_per_kwh[t]                     # charge_grid (cost)
    if has_baseload and shortfall_penalty_prices is not None:
        for t in range(T):
            c[8 * T + t] = shortfall_penalty_prices[t]                # shortfall penalty

    # --- Equality constraints (3T rows) ---
    # 1. PV balance (T rows)
    # 2. SoC green linking (T rows)
    # 3. SoC grey linking (T rows)
    n_eq = 3 * T
    A_eq = np.zeros((n_eq, n_vars))
    b_eq = np.zeros(n_eq)

    for t in range(T):
        # PV balance (row t)
        A_eq[t, t] = 1.0            # charge_pv[t]
        A_eq[t, 2 * T + t] = 1.0    # export_pv[t]
        A_eq[t, 3 * T + t] = 1.0    # curtail[t]
        b_eq[t] = pv_production_kwh[t]

        # SoC green linking (row T+t)
        row_g = T + t
        A_eq[row_g, 6 * T + t] = 1.0  # +soc_green[t]
        if t > 0:
            A_eq[row_g, 6 * T + t - 1] = -1.0  # -soc_green[t-1]
        A_eq[row_g, t] = -1.0          # -charge_pv[t]
        A_eq[row_g, T + t] = 1.0       # +discharge_green[t]
        b_eq[row_g] = start_soc_green_kwh if t == 0 else 0.0

        # SoC grey linking (row 2T+t)
        row_y = 2 * T + t
        A_eq[row_y, 7 * T + t] = 1.0  # +soc_grey[t]
        if t > 0:
            A_eq[row_y, 7 * T + t - 1] = -1.0  # -soc_grey[t-1]
        A_eq[row_y, 4 * T + t] = -1.0  # -charge_grid[t]
        A_eq[row_y, 5 * T + t] = 1.0   # +discharge_grey[t]
        b_eq[row_y] = start_soc_grey_kwh if t == 0 else 0.0

    # --- Inequality constraints ---
    # 1. Total SoC upper: soc_green[t] + soc_grey[t] ≤ soc_max      (T rows)
    # 2. Total SoC lower: -(soc_green[t] + soc_grey[t]) ≤ -soc_min  (T rows)
    # 3. Charge power: charge_pv[t] + charge_grid[t] ≤ max_charge    (T rows)
    # 4. Discharge power: disch_green[t] + disch_grey[t] ≤ max_disch (T rows)
    # 5. Grid limit: export*glf + (dg+dy)*RTE ≤ grid_max             (T rows)
    # 6. Baseload shortfall (if active):
    #    -export*glf - dg*rte*glf - dy*rte - shortfall ≤ -baseload    (T rows)
    n_ub = 6 * T if has_baseload else 5 * T
    A_ub = np.zeros((n_ub, n_vars))
    b_ub = np.zeros(n_ub)

    for t in range(T):
        # Total SoC upper (row t)
        A_ub[t, 6 * T + t] = 1.0       # soc_green[t]
        A_ub[t, 7 * T + t] = 1.0       # soc_grey[t]
        b_ub[t] = soc_max_kwh

        # Total SoC lower (row T+t)
        A_ub[T + t, 6 * T + t] = -1.0  # -soc_green[t]
        A_ub[T + t, 7 * T + t] = -1.0  # -soc_grey[t]
        b_ub[T + t] = -soc_min_kwh

        # Charge power (row 2T+t)
        A_ub[2 * T + t, t] = 1.0            # charge_pv[t]
        A_ub[2 * T + t, 4 * T + t] = 1.0    # charge_grid[t]
        b_ub[2 * T + t] = max_charge_energy

        # Discharge power (row 3T+t)
        A_ub[3 * T + t, T + t] = 1.0        # discharge_green[t]
        A_ub[3 * T + t, 5 * T + t] = 1.0    # discharge_grey[t]
        b_ub[3 * T + t] = max_discharge_energy

        # Grid limit (row 4T+t)
        A_ub[4 * T + t, 2 * T + t] = grid_loss_factor  # export_pv
        A_ub[4 * T + t, T + t] = rte                    # discharge_green
        A_ub[4 * T + t, 5 * T + t] = rte                # discharge_grey
        b_ub[4 * T + t] = grid_max_energy

    if has_baseload:
        for t in range(T):
            # Shortfall constraint (row 5T+t):
            # export*glf + dg*rte*glf + dy*rte + shortfall >= baseload
            row = 5 * T + t
            A_ub[row, 2 * T + t] = -grid_loss_factor       # -export_pv[t]
            A_ub[row, T + t] = -(rte * grid_loss_factor)    # -discharge_green[t]
            A_ub[row, 5 * T + t] = -rte                     # -discharge_grey[t]
            A_ub[row, 8 * T + t] = -1.0                     # -shortfall[t]
            b_ub[row] = -baseload_kwh

    # --- Variable bounds ---
    # Discharge fixed to 0 at negative spot prices (prevents simultaneous charge/discharge).
    bounds: list[tuple[float, float | None]] = []
    for t in range(T):
        bounds.append((0.0, None))  # charge_pv[t]
    for t in range(T):
        ub = 0.0 if spot_prices_eur_per_kwh[t] < 0 else None
        bounds.append((0.0, ub))  # discharge_green[t]
    for t in range(T):
        bounds.append((0.0, None))  # export_pv[t]
    for t in range(T):
        bounds.append((0.0, None))  # curtail[t]
    for t in range(T):
        bounds.append((0.0, None))  # charge_grid[t]
    for t in range(T):
        ub = 0.0 if spot_prices_eur_per_kwh[t] < 0 or (
                    has_baseload and (spot_prices_eur_per_kwh[t] - eff_prices[t]) < 0) else None
        bounds.append((0.0, ub))  # discharge_grey[t]
    for t in range(T):
        bounds.append((0.0, soc_max_kwh))  # soc_green[t]
    for t in range(T):
        bounds.append((0.0, soc_max_kwh))  # soc_grey[t]
    if has_baseload:
        for t in range(T):
            bounds.append((0.0, None))  # shortfall[t]

    return c, A_ub, b_ub, A_eq, b_eq, bounds


# ---------------------------------------------------------------------------
# Extract results
# ---------------------------------------------------------------------------


def _extract_green_result(
        x: np.ndarray,
        T: int,
        spot_prices_eur_per_kwh: np.ndarray,
        eff_prices: np.ndarray,
        fixed_price: float,
        rte: float,
        grid_loss_factor: float = 1.0,
        baseload_kwh: float = 0.0,
) -> DailyDispatchResult:
    """Parse the LP solution vector into a :class:`DailyDispatchResult` (Green).

    SoC is read directly from the explicit SoC decision variables (no
    cumulative reconstruction needed).
    """
    charge_pv = x[0: T]
    discharge_green = x[T: 2 * T]
    export_pv = x[2 * T: 3 * T]
    curtail = x[3 * T: 4 * T]
    soc = x[4 * T: 5 * T]
    shortfall = x[5 * T: 6 * T] if baseload_kwh > 0 else np.zeros(T)

    # Add losses to the energy flow
    discharge_green = discharge_green * grid_loss_factor * rte
    export_pv = export_pv * grid_loss_factor
    curtail = curtail * grid_loss_factor

    # Consider PPA-Baseload
    if baseload_kwh > 0:
        spot_revenue = np.maximum(export_pv + discharge_green - baseload_kwh, 0) * spot_prices_eur_per_kwh
        ppa_revenue = baseload_kwh * fixed_price
        baseload_shortfall_costs = shortfall * spot_prices_eur_per_kwh
        revenue = ppa_revenue + spot_revenue - baseload_shortfall_costs
    else:
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
        shortfall=shortfall,
    )


def _extract_grey_result(
        x: np.ndarray,
        T: int,
        spot_prices_eur_per_kwh: np.ndarray,
        eff_prices: np.ndarray,
        rte: float,
        grid_loss_factor: float = 1.0,
        baseload_kwh: float = 0.0,
        fixed_price: float = 0.0,
) -> DailyDispatchResult:
    """Parse the LP solution vector into a :class:`DailyDispatchResult` (Grey).

    SoC green/grey are read directly from explicit SoC decision variables.
    """
    charge_pv = x[0: T]
    discharge_green = x[T: 2 * T]
    export_pv = x[2 * T: 3 * T]
    curtail = x[3 * T: 4 * T]
    charge_grid = x[4 * T: 5 * T]
    discharge_grey = x[5 * T: 6 * T]
    soc_green = x[6 * T: 7 * T]
    soc_grey = x[7 * T: 8 * T]
    shortfall = x[8 * T: 9 * T] if baseload_kwh > 0 else np.zeros(T)

    soc = soc_green + soc_grey

    # Add losses to the energy flow
    discharge_green = discharge_green * grid_loss_factor * rte
    discharge_grey = discharge_grey * rte
    export_pv = export_pv * grid_loss_factor
    curtail = curtail * grid_loss_factor
    # Consider PPA-Baseload
    if baseload_kwh > 0:
        spot_revenue = np.maximum(export_pv + discharge_green + discharge_grey - baseload_kwh, 0) * spot_prices_eur_per_kwh
        ppa_revenue = baseload_kwh * fixed_price
        baseload_shortfall_costs = shortfall * spot_prices_eur_per_kwh
        revenue = ppa_revenue + spot_revenue - baseload_shortfall_costs
    else:
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
        shortfall=shortfall,
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
        baseload_mw: float = 0.0,
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
        Set to **0.0** when no floor is active.  Ignored when
        ``baseload_kw > 0``.
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
        Ignored when ``baseload_kw > 0`` (GoO is part of the baseload
        settlement computed by the engine).
    price_cap_eur_per_kwh : float
        Cap price in **€/kWh** for PPA Collar.  Set to **0.0** when no cap
        is active (unbounded upside).  Defaults to 0.0.
        Ignored when ``baseload_kw > 0``.
    grid_loss_factor : float
        Grid loss factor in (0, 1].  Applied to green energy (PV export and
        green BESS discharge) in the objective function, grid constraint, and
        revenue calculation.  Grey energy is **not** affected.  Defaults to
        1.0 (no losses).
    baseload_mw : float
        Baseload PPA commitment in **kW**.  When > 0, the LP uses raw spot
        prices (no floor/cap/goo) because the baseload settlement is a
        constant that does not affect LP decisions.  Defaults to 0.0
        (no baseload PPA).

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

    # Convert baseload from MW to kWh per interval
    baseload_kwh = baseload_mw * 1000.0 * bess.timestep_hours if baseload_mw > 0 else 0.0
    has_baseload = baseload_kwh > 0.0

    # Pre-compute effective green price
    if has_baseload:
        # Baseload PPA: export objective uses spot prices.
        # Shortfall penalty uses effective PPA price (max(spot, ppa_price) + goo).
        eff = spot_prices_eur_per_kwh.copy()
        shortfall_penalty = _effective_green_price(
            spot_prices_eur_per_kwh, price_fixed_eur_per_kwh,
            goo_premium_eur_per_kwh, price_cap_eur_per_kwh,
        )
    else:
        # Standard: clip(spot, fixed, cap) + goo — €/kWh
        eff = _effective_green_price(
            spot_prices_eur_per_kwh, price_fixed_eur_per_kwh,
            goo_premium_eur_per_kwh, price_cap_eur_per_kwh,
        )
        shortfall_penalty = None

    if mode == "green":
        c, A_ub, b_ub, A_eq, b_eq, bounds = _build_green_lp(
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
            baseload_kwh=baseload_kwh,
            shortfall_penalty_prices=shortfall_penalty,
        )
        soc_green_start = start_soc_kwh
        soc_grey_start = 0.0
    elif mode == "grey":
        soc_green_start = (
            start_soc_green_kwh if start_soc_green_kwh is not None else start_soc_kwh
        )
        soc_grey_start = (
            start_soc_grey_kwh if start_soc_grey_kwh is not None else 0.0
        )
        c, A_ub, b_ub, A_eq, b_eq, bounds = _build_grey_lp(
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
            baseload_kwh=baseload_kwh,
            shortfall_penalty_prices=shortfall_penalty,
        )
    else:
        raise ValueError(f"Unknown operating mode: '{mode}'. Use 'green' or 'grey'.")

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
            baseload_mw=baseload_mw,
        )

    x = result.x

    if mode == "green":
        return _extract_green_result(x, T, spot_prices_eur_per_kwh, eff, price_fixed_eur_per_kwh, rte, grid_loss_factor,
                                     baseload_mw * 1000.0 * bess.timestep_hours if baseload_mw > 0 else 0.0)
    else:
        return _extract_grey_result(
            x, T, spot_prices_eur_per_kwh, eff, rte, grid_loss_factor, baseload_mw * 1000.0 * bess.timestep_hours if baseload_mw > 0 else 0.0, price_fixed_eur_per_kwh
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
        baseload_mw: float = 0.0,
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
        Ignored when ``baseload_kw > 0``.
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
        comparison.  Defaults to 0.0.  Ignored when ``baseload_kw > 0``.
    price_cap_eur_per_kwh : float
        Cap price in **€/kWh** for PPA Collar.  0.0 means no cap.
        Defaults to 0.0.  Ignored when ``baseload_kw > 0``.
    baseload_mw : float
        Baseload PPA commitment in **kW**.  When > 0, effective price is
        raw spot (no floor/cap/goo).  Defaults to 0.0.

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

    # Effective price per kWh
    if baseload_mw > 0:
        # Baseload PPA: use spot prices (settlement computed by engine)
        eff = spot_prices_eur_per_kwh.copy()
    else:
        # Standard: clip(spot, fixed, cap) + goo
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

    # Compute shortfall for baseload PPA (BESS offline, only PV export)
    baseload_kwh = baseload_mw * 1000.0 * timestep_hours if baseload_mw > 0 else 0.0
    if baseload_kwh > 0:
        shortfall = np.maximum(baseload_kwh - export_pv, 0.0)
        spot_revenue = np.maximum(export_pv - baseload_kwh, 0) * spot_prices_eur_per_kwh
        ppa_revenue = baseload_kwh * price_fixed_eur_per_kwh
        baseload_shortfall_costs = shortfall * spot_prices_eur_per_kwh
        revenue = ppa_revenue + spot_revenue - baseload_shortfall_costs
    else:
        shortfall = np.zeros(T)
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
        effective_price=eff,
        shortfall=shortfall,
    )
