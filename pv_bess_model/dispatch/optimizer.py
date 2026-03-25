"""Daily MILP-based dispatch optimiser (scipy.optimize.milp / HiGHS backend).

Solves a 24-hour mixed-integer linear programme for each simulation day to
determine optimal charge/discharge and export decisions under perfect
day-ahead price foresight.

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

Simultaneous charging and discharging is prevented by a binary indicator
variable δ[t] per timestep: δ[t]=1 allows charging, δ[t]=0 allows
discharging.  This replaces the former heuristic of fixing discharge to zero
at negative spot prices.

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
5T  .. 6T-1  T              delta[t]               – binary: 1=charge, 0=discharge
===========  =============  ===================================================

Total Green Mode variables: 6T (+ shortfall T if baseload)

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
8T  .. 9T-1  T              delta[t]               – binary: 1=charge, 0=discharge
===========  =============  ===================================================

Total Grey Mode variables: 9T (+ shortfall T if baseload)

SoC is tracked via explicit decision variables with linking equality
constraints (staircase structure) and simple variable bounds.  This replaces
the previous cumulative-sum formulation and reduces constraint matrix density
from O(T²) to O(T) nonzeros.

Public API
----------
BessParams              – Frozen dataclass bundling BESS physical parameters.
DailyRevenueBreakdown   – Aggregated daily revenue split by source.
DailyDispatchResult     – TypedDict with all per-hour arrays + end_soc.
OperatingMode           – Literal type alias for ``"green"`` | ``"grey"``.
compute_daily_revenue   – Unified revenue calculation for all marketing types.
optimize_day            – Solve the daily MILP for one day (Green or Grey).
dispatch_offline_day    – Produce dispatch results for a BESS-offline day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np
from scipy.optimize import LinearConstraint, milp
from scipy.sparse import csc_matrix

from pv_bess_model.config.defaults import MILP_TIME_LIMIT

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
# Revenue breakdown
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DailyRevenueBreakdown:
    """Aggregated revenue breakdown for one simulation day.

    All values in EUR.  Positive means income, except *import_cost* and
    *shortfall_cost* which are positive costs (subtracted in
    *total_revenue*).
    """

    revenue_pv: float
    """PV direct feed-in revenue."""

    revenue_green: float
    """BESS green discharge revenue."""

    revenue_grey: float
    """BESS grey discharge revenue.  0.0 in Green Mode."""

    import_cost: float
    """Grid import cost (charge_grid × spot).  0.0 in Green Mode."""

    shortfall_cost: float
    """Baseload shortfall cost (shortfall × spot).  0.0 when no baseload PPA."""

    total_revenue: float
    """Net daily revenue = revenue_pv + revenue_green + revenue_grey
    − import_cost − shortfall_cost."""

    bess_spot_revenue: float
    """BESS revenue at spot price for optimization fee calculation."""


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

    revenue_breakdown: DailyRevenueBreakdown
    """Aggregated daily revenue split by source (PV, BESS green, BESS grey)."""


# ---------------------------------------------------------------------------
# Unified revenue calculation
# ---------------------------------------------------------------------------


def compute_daily_revenue(
        export_pv: np.ndarray,
        discharge_green: np.ndarray,
        discharge_grey: np.ndarray,
        charge_grid: np.ndarray,
        shortfall: np.ndarray,
        spot_prices: np.ndarray,
        eff_prices: np.ndarray,
        fixed_price: float,
        baseload_kwh: float,
) -> tuple[np.ndarray, DailyRevenueBreakdown]:
    """Unified revenue calculation for all marketing types.

    Computes both the per-timestep revenue array (for hourly sample export)
    and the aggregated :class:`DailyRevenueBreakdown` (for annual cashflow).

    All energy arrays must already have losses applied (RTE, grid loss factor).

    Parameters
    ----------
    export_pv:
        PV energy exported to grid per timestep (kWh), post grid-loss.
    discharge_green:
        BESS green discharge per timestep (kWh), post RTE and grid-loss.
    discharge_grey:
        BESS grey discharge per timestep (kWh), post RTE.  Zeros in Green Mode.
    charge_grid:
        BESS grid charging per timestep (kWh).  Zeros in Green Mode.
    shortfall:
        Baseload shortfall per timestep (kWh).  Zeros when no baseload PPA.
    spot_prices:
        Spot prices per timestep (EUR/kWh).
    eff_prices:
        Effective green prices per timestep (EUR/kWh), with floor/cap/goo applied.
        Equals spot_prices when baseload PPA is active.
    fixed_price:
        PPA fixed price (EUR/kWh).  Used only for baseload PPA settlement.
    baseload_kwh:
        Baseload commitment per timestep (kWh).  0.0 when no baseload PPA.

    Returns
    -------
    tuple[np.ndarray, DailyRevenueBreakdown]
        ``(revenue_per_step, breakdown)`` where *revenue_per_step* has shape
        ``(T,)`` and contains the per-timestep revenue in EUR.
    """
    import_cost = charge_grid * spot_prices

    if baseload_kwh > 0:
        # --- Baseload PPA settlement ---
        total_export = export_pv + discharge_green + discharge_grey
        excess = np.maximum(total_export - baseload_kwh, 0.0)
        spot_revenue_arr = excess * spot_prices
        ppa_revenue_scalar = baseload_kwh * fixed_price
        shortfall_cost_arr = shortfall * spot_prices

        revenue_per_step = ppa_revenue_scalar + spot_revenue_arr - shortfall_cost_arr - import_cost
        shortfall_cost = float(np.sum(shortfall_cost_arr))

        # Split gross revenue (PPA + excess at spot) proportionally by
        # each source's energy contribution to total feed-in.
        gross_per_step = ppa_revenue_scalar + spot_revenue_arr
        has_energy = total_export > 0
        total_safe = np.where(has_energy, total_export, 1.0)

        pv_frac = export_pv / total_safe
        green_frac = discharge_green / total_safe
        grey_frac = discharge_grey / total_safe

        revenue_pv = float(np.sum(gross_per_step * pv_frac))
        revenue_green = float(np.sum(gross_per_step * green_frac))
        revenue_grey = float(np.sum(gross_per_step * grey_frac))

        # Attribute unattributed PPA revenue from zero-production timesteps
        # to revenue_pv (contract revenue independent of production).
        unattr_mask = ~has_energy
        if np.any(unattr_mask):
            revenue_pv += float(np.sum(
                np.where(unattr_mask, ppa_revenue_scalar, 0.0),
            ))

        bess_spot_revenue = revenue_grey
    else:
        # --- Non-baseload: EEG, Floor PPA, Collar PPA, Market ---
        revenue_per_step = (
            (export_pv + discharge_green) * eff_prices
            + (discharge_grey - charge_grid) * spot_prices
        )

        revenue_pv = float(np.sum(export_pv * eff_prices))
        revenue_green = float(np.sum(discharge_green * eff_prices))
        revenue_grey = float(np.sum(discharge_grey * spot_prices))
        shortfall_cost = 0.0
        bess_spot_revenue = revenue_grey

    total_revenue = (
        revenue_pv + revenue_green + revenue_grey
        - sum(import_cost) - shortfall_cost
    )

    return revenue_per_step, DailyRevenueBreakdown(
        revenue_pv=revenue_pv,
        revenue_green=revenue_green,
        revenue_grey=revenue_grey,
        import_cost=np.sum(import_cost),
        shortfall_cost=shortfall_cost,
        total_revenue=total_revenue,
        bess_spot_revenue=bess_spot_revenue,
    )


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


def _build_green_milp(
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
) -> tuple[np.ndarray, csc_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct the Green-Mode MILP matrices with binary charge/discharge indicator.

    Returns ``(c, A, constraint_lb, constraint_ub, var_lb, var_ub, integrality)``
    suitable for ``scipy.optimize.milp``.

    Variables without baseload: ``[charge_pv(T), discharge_green(T),
    export_pv(T), curtail(T), soc(T), delta(T)]`` — total 6T.

    Variables with baseload: ``[charge_pv(T), discharge_green(T),
    export_pv(T), curtail(T), soc(T), delta(T), shortfall(T)]`` — total 7T.

    The binary variable ``delta[t]`` prevents simultaneous charging and
    discharging: ``delta[t]=1`` allows charging, ``delta[t]=0`` allows
    discharging.

    Parameters
    ----------
    baseload_kwh:
        Baseload commitment per interval in kWh.  0.0 when no baseload PPA.
    shortfall_penalty_prices:
        Effective PPA price array (EUR/kWh) for the shortfall penalty.
        Required when ``baseload_kwh > 0``.
    """
    T = len(pv_production_kwh)
    has_baseload = baseload_kwh > 0.0
    # Variables: charge_pv(T), discharge_green(T), export_pv(T), curtail(T),
    #            soc(T), delta(T), [shortfall(T)]
    n_vars = 7 * T if has_baseload else 6 * T

    max_charge_energy = max_charge_kw * timestep_hours
    max_discharge_energy = max_discharge_kw * timestep_hours
    grid_max_energy = grid_max_kw * timestep_hours

    # --- Objective (minimise) ---
    c = np.zeros(n_vars)
    for t in range(T):
        c[2 * T + t] = -(grid_loss_factor * eff_prices[t])         # export_pv[t]
        c[T + t] = -(rte * grid_loss_factor * eff_prices[t])       # discharge_green[t]
    if has_baseload and shortfall_penalty_prices is not None:
        for t in range(T):
            c[6 * T + t] = shortfall_penalty_prices[t]             # shortfall penalty

    # --- Build combined constraint matrix ---
    # Rows:
    #   0  ..  T-1   : PV balance (equality)
    #   T  .. 2T-1   : SoC linking (equality)
    #  2T  .. 3T-1   : Grid limit (≤)
    #  3T  .. 4T-1   : Charge indicator: charge_pv[t] ≤ δ[t] × M_charge (≤)
    #  4T  .. 5T-1   : Discharge indicator: discharge_green[t] ≤ (1-δ[t]) × M_discharge (≤)
    #  5T  .. 6T-1   : Baseload shortfall (≤), if has_baseload
    n_rows = 6 * T if has_baseload else 5 * T

    # Pre-allocate in COO format for efficiency
    row_idx = []
    col_idx = []
    data = []

    def _add(r: int, co: int, v: float) -> None:
        row_idx.append(r)
        col_idx.append(co)
        data.append(v)

    constraint_lb = np.empty(n_rows)
    constraint_ub = np.empty(n_rows)

    for t in range(T):
        # --- PV balance (row t): equality ---
        _add(t, t, 1.0)              # charge_pv[t]
        _add(t, 2 * T + t, 1.0)     # export_pv[t]
        _add(t, 3 * T + t, 1.0)     # curtail[t]
        constraint_lb[t] = pv_production_kwh[t]
        constraint_ub[t] = pv_production_kwh[t]

        # --- SoC linking (row T+t): equality ---
        row = T + t
        _add(row, 4 * T + t, 1.0)   # +soc[t]
        if t > 0:
            _add(row, 4 * T + t - 1, -1.0)  # -soc[t-1]
        _add(row, t, -1.0)           # -charge_pv[t]
        _add(row, T + t, 1.0)        # +discharge_green[t]
        rhs = start_soc_kwh if t == 0 else 0.0
        constraint_lb[row] = rhs
        constraint_ub[row] = rhs

        # --- Grid limit (row 2T+t): ≤ ---
        row_g = 2 * T + t
        _add(row_g, 2 * T + t, grid_loss_factor)  # export_pv[t]
        _add(row_g, T + t, rte)                     # discharge_green[t]
        constraint_lb[row_g] = -np.inf
        constraint_ub[row_g] = grid_max_energy

        # --- Charge indicator (row 3T+t): charge_pv[t] - M_charge × δ[t] ≤ 0 ---
        row_ci = 3 * T + t
        _add(row_ci, t, 1.0)                         # charge_pv[t]
        _add(row_ci, 5 * T + t, -max_charge_energy)  # -M_charge × δ[t]
        constraint_lb[row_ci] = -np.inf
        constraint_ub[row_ci] = 0.0

        # --- Discharge indicator (row 4T+t): discharge_green[t] + M_discharge × δ[t] ≤ M_discharge ---
        row_di = 4 * T + t
        _add(row_di, T + t, 1.0)                       # discharge_green[t]
        _add(row_di, 5 * T + t, max_discharge_energy)   # M_discharge × δ[t]
        constraint_lb[row_di] = -np.inf
        constraint_ub[row_di] = max_discharge_energy

    if has_baseload:
        for t in range(T):
            # Shortfall: -export*glf - disch*rte*glf - shortfall ≤ -baseload
            row = 5 * T + t
            _add(row, 2 * T + t, -grid_loss_factor)
            _add(row, T + t, -(rte * grid_loss_factor))
            _add(row, 6 * T + t, -1.0)   # -shortfall[t]
            constraint_lb[row] = -np.inf
            constraint_ub[row] = -baseload_kwh

    A = csc_matrix(
        (data, (row_idx, col_idx)),
        shape=(n_rows, n_vars),
    )

    # --- Variable bounds ---
    var_lb = np.zeros(n_vars)
    var_ub = np.empty(n_vars)

    # charge_pv[t]: [0, max_charge_energy]
    var_ub[0: T] = max_charge_energy
    # discharge_green[t]: [0, max_discharge_energy]
    var_ub[T: 2 * T] = max_discharge_energy
    # export_pv[t]: [0, inf]
    var_ub[2 * T: 3 * T] = np.inf
    # curtail[t]: [0, inf]
    var_ub[3 * T: 4 * T] = np.inf
    # soc[t]: [soc_min, soc_max]
    var_lb[4 * T: 5 * T] = soc_min_kwh
    var_ub[4 * T: 5 * T] = soc_max_kwh
    # delta[t]: [0, 1] binary
    var_ub[5 * T: 6 * T] = 1.0
    if has_baseload:
        # shortfall[t]: [0, inf]
        var_ub[6 * T: 7 * T] = np.inf

    # --- Integrality ---
    integrality = np.zeros(n_vars, dtype=int)
    integrality[5 * T: 6 * T] = 1  # delta[t] is binary

    return c, A, constraint_lb, constraint_ub, var_lb, var_ub, integrality


# ---------------------------------------------------------------------------
# Helper: build Grey-Mode LP (explicit SoC variables)
# ---------------------------------------------------------------------------


def _build_grey_milp(
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
        grid_max_import_kw: float | None = None,
) -> tuple[np.ndarray, csc_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct the Grey-Mode MILP matrices with binary charge/discharge indicator.

    Returns ``(c, A, constraint_lb, constraint_ub, var_lb, var_ub, integrality)``
    suitable for ``scipy.optimize.milp``.

    Variables without baseload: ``[charge_pv(T), discharge_green(T),
    export_pv(T), curtail(T), charge_grid(T), discharge_grey(T),
    soc_green(T), soc_grey(T), delta(T)]`` — total 9T.

    Variables with baseload: same as above plus ``shortfall(T)`` — total 10T.

    The binary variable ``delta[t]`` prevents simultaneous charging and
    discharging across all chambers: total charge ≤ δ[t]×M_charge and
    total discharge ≤ (1-δ[t])×M_discharge.

    Parameters
    ----------
    baseload_kwh:
        Baseload commitment per interval in kWh.  0.0 when no baseload PPA.
    shortfall_penalty_prices:
        Effective PPA price array (EUR/kWh) for the shortfall penalty.
        Required when ``baseload_kwh > 0``.
    """
    T = len(pv_production_kwh)
    has_baseload = baseload_kwh > 0.0
    # Variables: charge_pv(T), discharge_green(T), export_pv(T), curtail(T),
    #            charge_grid(T), discharge_grey(T), soc_green(T), soc_grey(T),
    #            delta(T), [shortfall(T)]
    n_vars = 10 * T if has_baseload else 9 * T

    max_charge_energy = max_charge_kw * timestep_hours
    max_discharge_energy = max_discharge_kw * timestep_hours
    grid_max_energy = grid_max_kw * timestep_hours

    # --- Objective (minimise) ---
    c = np.zeros(n_vars)
    for t in range(T):
        c[2 * T + t] = -(grid_loss_factor * eff_prices[t])            # export_pv
        c[T + t] = -(rte * grid_loss_factor * eff_prices[t])          # discharge_green
        c[5 * T + t] = -(rte * spot_prices_eur_per_kwh[t])            # discharge_grey
        c[4 * T + t] = spot_prices_eur_per_kwh[t]                     # charge_grid (cost)
    if has_baseload and shortfall_penalty_prices is not None:
        for t in range(T):
            c[9 * T + t] = shortfall_penalty_prices[t]                # shortfall penalty

    # --- Build combined constraint matrix ---
    # Equality rows:
    #   0  ..  T-1  : PV balance
    #   T  .. 2T-1  : SoC green linking
    #  2T  .. 3T-1  : SoC grey linking
    # Inequality rows:
    #  3T  .. 4T-1  : Total SoC upper
    #  4T  .. 5T-1  : Total SoC lower
    #  5T  .. 6T-1  : Charge power
    #  6T  .. 7T-1  : Discharge power
    #  7T  .. 8T-1  : Grid limit
    #  8T  .. 9T-1  : Charge indicator: charge_pv[t] + charge_grid[t] - M_charge × δ[t] ≤ 0
    #  9T  .. 10T-1 : Discharge indicator: disch_green[t] + disch_grey[t] + M_discharge × δ[t] ≤ M_discharge
    # 10T  .. 11T-1 : Baseload shortfall (if has_baseload)
    n_rows = 11 * T if has_baseload else 10 * T

    row_idx = []
    col_idx = []
    data = []

    def _add(r: int, co: int, v: float) -> None:
        row_idx.append(r)
        col_idx.append(co)
        data.append(v)

    constraint_lb = np.empty(n_rows)
    constraint_ub = np.empty(n_rows)

    for t in range(T):
        # --- PV balance (row t): equality ---
        _add(t, t, 1.0)              # charge_pv[t]
        _add(t, 2 * T + t, 1.0)     # export_pv[t]
        _add(t, 3 * T + t, 1.0)     # curtail[t]
        constraint_lb[t] = pv_production_kwh[t]
        constraint_ub[t] = pv_production_kwh[t]

        # --- SoC green linking (row T+t): equality ---
        row_g = T + t
        _add(row_g, 6 * T + t, 1.0)  # +soc_green[t]
        if t > 0:
            _add(row_g, 6 * T + t - 1, -1.0)  # -soc_green[t-1]
        _add(row_g, t, -1.0)          # -charge_pv[t]
        _add(row_g, T + t, 1.0)       # +discharge_green[t]
        rhs_g = start_soc_green_kwh if t == 0 else 0.0
        constraint_lb[row_g] = rhs_g
        constraint_ub[row_g] = rhs_g

        # --- SoC grey linking (row 2T+t): equality ---
        row_y = 2 * T + t
        _add(row_y, 7 * T + t, 1.0)  # +soc_grey[t]
        if t > 0:
            _add(row_y, 7 * T + t - 1, -1.0)  # -soc_grey[t-1]
        _add(row_y, 4 * T + t, -1.0)  # -charge_grid[t]
        _add(row_y, 5 * T + t, 1.0)   # +discharge_grey[t]
        rhs_y = start_soc_grey_kwh if t == 0 else 0.0
        constraint_lb[row_y] = rhs_y
        constraint_ub[row_y] = rhs_y

        # --- Total SoC upper (row 3T+t): ≤ ---
        row_su = 3 * T + t
        _add(row_su, 6 * T + t, 1.0)  # soc_green[t]
        _add(row_su, 7 * T + t, 1.0)  # soc_grey[t]
        constraint_lb[row_su] = -np.inf
        constraint_ub[row_su] = soc_max_kwh

        # --- Total SoC lower (row 4T+t): ≥ soc_min → -(sg+sy) ≤ -soc_min ---
        row_sl = 4 * T + t
        _add(row_sl, 6 * T + t, -1.0)
        _add(row_sl, 7 * T + t, -1.0)
        constraint_lb[row_sl] = -np.inf
        constraint_ub[row_sl] = -soc_min_kwh

        # --- Charge power (row 5T+t): ≤ ---
        row_cp = 5 * T + t
        _add(row_cp, t, 1.0)            # charge_pv[t]
        _add(row_cp, 4 * T + t, 1.0)    # charge_grid[t]
        constraint_lb[row_cp] = -np.inf
        constraint_ub[row_cp] = max_charge_energy

        # --- Discharge power (row 6T+t): ≤ ---
        row_dp = 6 * T + t
        _add(row_dp, T + t, 1.0)        # discharge_green[t]
        _add(row_dp, 5 * T + t, 1.0)    # discharge_grey[t]
        constraint_lb[row_dp] = -np.inf
        constraint_ub[row_dp] = max_discharge_energy

        # --- Grid limit (row 7T+t): ≤ ---
        row_gl = 7 * T + t
        _add(row_gl, 2 * T + t, grid_loss_factor)  # export_pv
        _add(row_gl, T + t, rte)                     # discharge_green
        _add(row_gl, 5 * T + t, rte)                 # discharge_grey
        constraint_lb[row_gl] = -np.inf
        constraint_ub[row_gl] = grid_max_energy

        # --- Charge indicator (row 8T+t): charge_pv + charge_grid - M_charge × δ ≤ 0 ---
        row_ci = 8 * T + t
        _add(row_ci, t, 1.0)                         # charge_pv[t]
        _add(row_ci, 4 * T + t, 1.0)                 # charge_grid[t]
        _add(row_ci, 8 * T + t, -max_charge_energy)  # -M_charge × δ[t]
        constraint_lb[row_ci] = -np.inf
        constraint_ub[row_ci] = 0.0

        # --- Discharge indicator (row 9T+t): disch_green + disch_grey + M_discharge × δ ≤ M_discharge ---
        row_di = 9 * T + t
        _add(row_di, T + t, 1.0)                       # discharge_green[t]
        _add(row_di, 5 * T + t, 1.0)                   # discharge_grey[t]
        _add(row_di, 8 * T + t, max_discharge_energy)   # M_discharge × δ[t]
        constraint_lb[row_di] = -np.inf
        constraint_ub[row_di] = max_discharge_energy

    if has_baseload:
        for t in range(T):
            row = 10 * T + t
            _add(row, 2 * T + t, -grid_loss_factor)
            _add(row, T + t, -(rte * grid_loss_factor))
            _add(row, 5 * T + t, -rte)
            _add(row, 9 * T + t, -1.0)   # -shortfall[t]
            constraint_lb[row] = -np.inf
            constraint_ub[row] = -baseload_kwh

    A = csc_matrix(
        (data, (row_idx, col_idx)),
        shape=(n_rows, n_vars),
    )

    # --- Variable bounds ---
    var_lb = np.zeros(n_vars)
    var_ub = np.empty(n_vars)

    # charge_pv[t]: [0, inf] (power limited via constraint)
    var_ub[0: T] = np.inf
    # discharge_green[t]: [0, inf] (power limited via constraint)
    var_ub[T: 2 * T] = np.inf
    # export_pv[t]: [0, inf]
    var_ub[2 * T: 3 * T] = np.inf
    # curtail[t]: [0, inf]
    var_ub[3 * T: 4 * T] = np.inf
    # charge_grid[t]: [0, grid_max_import_energy] (grid import power limit)
    _effective_import_kw = grid_max_import_kw if grid_max_import_kw is not None else grid_max_kw
    var_ub[4 * T: 5 * T] = _effective_import_kw * timestep_hours
    # discharge_grey[t]: [0, inf] (power limited via constraint)
    var_ub[5 * T: 6 * T] = np.inf
    # soc_green[t]: [0, soc_max]
    var_ub[6 * T: 7 * T] = soc_max_kwh
    # soc_grey[t]: [0, soc_max]
    var_ub[7 * T: 8 * T] = soc_max_kwh
    # delta[t]: [0, 1] binary
    var_ub[8 * T: 9 * T] = 1.0
    if has_baseload:
        # shortfall[t]: [0, inf]
        var_ub[9 * T: 10 * T] = np.inf

    # --- Integrality ---
    integrality = np.zeros(n_vars, dtype=int)
    integrality[8 * T: 9 * T] = 1  # delta[t] is binary

    return c, A, constraint_lb, constraint_ub, var_lb, var_ub, integrality


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
    # delta at 5T..6T (binary, discarded)
    shortfall = x[6 * T: 7 * T] if baseload_kwh > 0 else np.zeros(T)

    # Add losses to the energy flow
    discharge_green = discharge_green * grid_loss_factor * rte
    export_pv = export_pv * grid_loss_factor
    curtail = curtail * grid_loss_factor

    charge_grid = np.zeros(T)
    discharge_grey = np.zeros(T)

    revenue, breakdown = compute_daily_revenue(
        export_pv=export_pv,
        discharge_green=discharge_green,
        discharge_grey=discharge_grey,
        charge_grid=charge_grid,
        shortfall=shortfall,
        spot_prices=spot_prices_eur_per_kwh,
        eff_prices=eff_prices,
        fixed_price=fixed_price,
        baseload_kwh=baseload_kwh,
    )

    return DailyDispatchResult(
        charge_pv=charge_pv,
        discharge_green=discharge_green,
        export_pv=export_pv,
        curtail=curtail,
        charge_grid=charge_grid,
        discharge_grey=discharge_grey,
        soc=soc,
        soc_green=soc,
        soc_grey=np.zeros(T),
        revenue=revenue,
        end_soc=float(soc[-1]),
        end_soc_green=float(soc[-1]),
        end_soc_grey=0.0,
        effective_price=eff_prices,
        shortfall=shortfall,
        revenue_breakdown=breakdown,
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
    # delta at 8T..9T (binary, discarded)
    shortfall = x[9 * T: 10 * T] if baseload_kwh > 0 else np.zeros(T)

    soc = soc_green + soc_grey

    # Add losses to the energy flow
    discharge_green = discharge_green * grid_loss_factor * rte
    discharge_grey = discharge_grey * rte
    export_pv = export_pv * grid_loss_factor
    curtail = curtail * grid_loss_factor

    revenue, breakdown = compute_daily_revenue(
        export_pv=export_pv,
        discharge_green=discharge_green,
        discharge_grey=discharge_grey,
        charge_grid=charge_grid,
        shortfall=shortfall,
        spot_prices=spot_prices_eur_per_kwh,
        eff_prices=eff_prices,
        fixed_price=fixed_price,
        baseload_kwh=baseload_kwh,
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
        effective_price=eff_prices,
        shortfall=shortfall,
        revenue_breakdown=breakdown,
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
        grid_max_import_kw: float | None = None,
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
    grid_max_import_kw : float | None
        Maximum grid import power in **kW** (Grey Mode only).  Limits
        ``charge_grid[t]`` per timestep.  When ``None``, falls back to
        ``grid_max_kw``.  Has no effect in Green Mode.

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
        c, A, con_lb, con_ub, var_lb, var_ub, integrality = _build_green_milp(
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
        c, A, con_lb, con_ub, var_lb, var_ub, integrality = _build_grey_milp(
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
            grid_max_import_kw=grid_max_import_kw,
        )
    else:
        raise ValueError(f"Unknown operating mode: '{mode}'. Use 'green' or 'grey'.")

    from scipy.optimize import Bounds as ScipyBounds
    constraints = LinearConstraint(A, con_lb, con_ub)
    bounds = ScipyBounds(var_lb, var_ub)
    options = {"time_limit": MILP_TIME_LIMIT}

    result = milp(
        c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options=options,
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
    shortfall = (
        np.maximum(baseload_kwh - export_pv, 0.0) if baseload_kwh > 0
        else np.zeros(T)
    )

    discharge_green = np.zeros(T)
    discharge_grey = np.zeros(T)
    charge_grid = np.zeros(T)

    revenue, breakdown = compute_daily_revenue(
        export_pv=export_pv,
        discharge_green=discharge_green,
        discharge_grey=discharge_grey,
        charge_grid=charge_grid,
        shortfall=shortfall,
        spot_prices=spot_prices_eur_per_kwh,
        eff_prices=eff,
        fixed_price=price_fixed_eur_per_kwh,
        baseload_kwh=baseload_kwh,
    )

    return DailyDispatchResult(
        charge_pv=np.zeros(T),
        discharge_green=discharge_green,
        export_pv=export_pv,
        curtail=curtail,
        charge_grid=charge_grid,
        discharge_grey=discharge_grey,
        soc=np.full(T, start_soc_kwh),
        soc_green=np.full(T, soc_green_val),
        soc_grey=np.full(T, soc_grey_val),
        revenue=revenue,
        end_soc=start_soc_kwh,
        end_soc_green=soc_green_val,
        end_soc_grey=soc_grey_val,
        effective_price=eff,
        shortfall=shortfall,
        revenue_breakdown=breakdown,
    )
