"""Daily LP-based dispatch optimizer for the portfolio/Systemwert model.

Solves a linear programme for each simulation day to determine optimal
BESS charge/discharge and heat pump dispatch decisions that minimize the
net system cost (grid buy cost minus grid sell revenue) under perfect
day-ahead price foresight.

Unlike the PV+BESS optimizer (``optimizer.py``), this LP:
- Has no grid export limit
- Has no Green/Grey mode distinction
- Has no PPA/EEG floor/cap pricing
- Uses a ``perfect_foresight_discount`` on sell revenues
- Models a bidirectional net grid position (buy AND sell possible)

The LP supports optional BESS and/or heat pump (WP) flexibility.
EV/V2G flex variables will be added in Phase 6.

Variable layout (full flex, T = intervals_per_day)
--------------------------------------------------
===========  =====  ==========================================
Slice        Len    Variable
===========  =====  ==========================================
0   .. T-1   T      grid_sell[t]      – net grid export (kWh)
T   .. 2T-1  T      grid_buy[t]       – net grid import (kWh)
2T  .. 3T-1  T      bess_charge[t]    – BESS charging (kWh)
3T  .. 4T-1  T      bess_discharge[t] – BESS discharging (kWh)
4T  .. 5T    T+1    soc[t]            – SoC (kWh), t=0..T
5T+1..6T     T      wp_load[t]        – WP electrical intake (kWh)
6T+1..7T+1   T+1    thermal_soc[t]    – thermal storage (kWh_th)
===========  =====  ==========================================

BESS and WP blocks are conditionally included.

Typical usage::

    from pv_bess_model.dispatch.optimizer_portfolio import (
        optimize_portfolio_day, PortfolioLPConfig, BessFlexParams,
        HeatPumpFlexParams,
    )

    result = optimize_portfolio_day(
        pv_production=pv_day,
        load_demand=load_day,
        spot_prices=prices_day,
        bess_params=bess,
        config=config,
        hp_params=hp,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from pv_bess_model.config.defaults import (
    DEFAULT_PERFECT_FORESIGHT_DISCOUNT,
    INTERVALS_PER_DAY,
    TIMESTEP_HOURS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration and parameter dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioLPConfig:
    """Static LP configuration for one day.

    Attributes
    ----------
    timestep_hours : float
        Duration of one interval in hours (0.25 for quarter-hourly).
    intervals_per_day : int
        Number of intervals per day (96 for quarter-hourly).
    perfect_foresight_discount : float
        Discount factor on grid-sell revenues (0.0 – 1.0).
        A value < 1.0 reduces the LP's valuation of selling surplus,
        compensating for the over-estimation from perfect price foresight.
    """

    timestep_hours: float = TIMESTEP_HOURS
    intervals_per_day: int = INTERVALS_PER_DAY
    perfect_foresight_discount: float = DEFAULT_PERFECT_FORESIGHT_DISCOUNT


@dataclass(frozen=True)
class BessFlexParams:
    """BESS parameters for the portfolio LP.

    Attributes
    ----------
    capacity_kwh : float
        Usable BESS capacity in kWh (already degraded for this year).
    power_kw : float
        Maximum charge/discharge power in kW.
    rte : float
        Round-trip efficiency as a fraction in (0, 1].
        Losses applied on discharge only.
    min_soc_pct : float
        Minimum SoC as percent of capacity.
    max_soc_pct : float
        Maximum SoC as percent of capacity.
    start_soc_kwh : float
        SoC at the beginning of the day in kWh.
    """

    capacity_kwh: float
    power_kw: float
    rte: float
    min_soc_pct: float
    max_soc_pct: float
    start_soc_kwh: float


@dataclass(frozen=True)
class HeatPumpFlexParams:
    """Heat pump parameters for the portfolio LP.

    Attributes
    ----------
    power_kw : float
        Electrical rated power of the heat pump (kW).
    cop_profile : numpy.ndarray
        COP values per interval (length = intervals_per_day).
    daily_heat_demand_kwh : float
        Total thermal demand for this day in kWh_th.
    thermal_storage_kwh : float
        Thermal storage capacity in kWh_th.
    heat_demand_profile : numpy.ndarray
        Thermal demand per interval (kWh_th, length = intervals_per_day).
    start_thermal_soc_kwh : float
        Thermal SoC at the beginning of the day in kWh_th.
    """

    power_kw: float
    cop_profile: np.ndarray
    daily_heat_demand_kwh: float
    thermal_storage_kwh: float
    heat_demand_profile: np.ndarray
    start_thermal_soc_kwh: float = 0.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PortfolioDailyResult:
    """Result of a daily portfolio LP solve.

    Attributes
    ----------
    grid_sell : numpy.ndarray
        Grid export per interval (kWh), length T.
    grid_buy : numpy.ndarray
        Grid import per interval (kWh), length T.
    bess_charge : numpy.ndarray
        BESS charging per interval (kWh), length T.
    bess_discharge : numpy.ndarray
        BESS discharging per interval (kWh), length T.
    bess_soc : numpy.ndarray
        BESS SoC at each interval boundary (kWh), length T+1.
        soc[0] = start_soc, soc[T] = end_soc.
    system_cost : float
        Daily system cost: buy_cost - sell_revenue (EUR).
        Positive = net cost, negative = net revenue.
    end_soc_kwh : float
        SoC at end of day (kWh) for day-to-day coupling.
    solver_status : str
        LP solver status string.
    """

    grid_sell: np.ndarray
    grid_buy: np.ndarray
    bess_charge: np.ndarray
    bess_discharge: np.ndarray
    bess_soc: np.ndarray
    system_cost: float
    end_soc_kwh: float
    solver_status: str
    wp_load: np.ndarray | None = None
    thermal_soc: np.ndarray | None = None
    end_thermal_soc_kwh: float = 0.0


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def optimize_portfolio_day(
    pv_production: np.ndarray,
    load_demand: np.ndarray,
    spot_prices: np.ndarray,
    bess_params: BessFlexParams | None,
    config: PortfolioLPConfig,
    hp_params: HeatPumpFlexParams | None = None,
) -> PortfolioDailyResult:
    """Solve the daily portfolio dispatch LP.

    Minimizes daily system cost (grid buy minus discounted grid sell)
    subject to energy balance, BESS SoC, power, and optional heat pump
    constraints.

    When neither *bess_params* nor *hp_params* is given, no LP is needed
    and the result equals World A for that day.

    Parameters
    ----------
    pv_production:
        PV production per interval in kWh (length = intervals_per_day).
    load_demand:
        Load demand per interval in kWh (length = intervals_per_day).
    spot_prices:
        Spot price per interval in EUR/kWh (length = intervals_per_day).
    bess_params:
        BESS configuration, or ``None`` for no storage.
    config:
        LP configuration (timestep, discount).
    hp_params:
        Heat pump configuration, or ``None`` for no heat pump.

    Returns
    -------
    PortfolioDailyResult
        Optimal dispatch arrays and daily system cost.
    """
    if bess_params is None and hp_params is None:
        return _solve_no_flex(pv_production, load_demand, spot_prices, config)

    return _solve_lp(
        pv_production, load_demand, spot_prices, bess_params, config, hp_params
    )


# ---------------------------------------------------------------------------
# No-flex case (analytical, no LP needed)
# ---------------------------------------------------------------------------


def _solve_no_flex(
    pv: np.ndarray,
    load: np.ndarray,
    prices: np.ndarray,
    config: PortfolioLPConfig,
) -> PortfolioDailyResult:
    """Compute dispatch without any flexibility (World A for one day)."""
    T = config.intervals_per_day
    netto = pv - load

    grid_sell = np.maximum(netto, 0.0)
    grid_buy = np.maximum(-netto, 0.0)

    sell_revenue = float(np.sum(grid_sell * prices * config.perfect_foresight_discount))
    buy_cost = float(np.sum(grid_buy * prices))
    system_cost = buy_cost - sell_revenue

    return PortfolioDailyResult(
        grid_sell=grid_sell,
        grid_buy=grid_buy,
        bess_charge=np.zeros(T),
        bess_discharge=np.zeros(T),
        bess_soc=np.zeros(T + 1),
        system_cost=system_cost,
        end_soc_kwh=0.0,
        solver_status="no_flex",
    )


# ---------------------------------------------------------------------------
# Unified LP (BESS and/or heat pump)
# ---------------------------------------------------------------------------


def _solve_lp(
    pv: np.ndarray,
    load: np.ndarray,
    prices: np.ndarray,
    bess: BessFlexParams | None,
    config: PortfolioLPConfig,
    hp: HeatPumpFlexParams | None = None,
) -> PortfolioDailyResult:
    """Solve the daily LP with BESS and/or heat pump flexibility.

    Variable layout (conditionally built):
        Base: [grid_sell(T), grid_buy(T)]
        + BESS: [charge(T), discharge(T), soc(T+1)]
        + HP:   [wp_load(T), thermal_soc(T+1)]

    Objective (minimize):
        min Σ [ grid_buy[t] × price[t] − grid_sell[t] × price[t] × discount ]

    Constraints (BESS):
        1. Energy balance: sell[t] - buy[t] + charge[t] - discharge[t]*RTE
                           (+ wp_load[t] if HP) = netto[t]
        2. SoC linking: soc[t+1] = soc[t] + charge[t] - discharge[t]
        3. SoC initial: soc[0] = start_soc

    Constraints (HP):
        4. Daily energy balance: Σ wp_load[t] × COP[t] = daily_heat_demand
        5. Thermal SoC linking:
           thermal_soc[t+1] = thermal_soc[t] + wp_load[t]*COP[t] - heat_demand[t]
        6. Thermal SoC initial: thermal_soc[0] = start_thermal_soc
    """
    T = config.intervals_per_day
    dt = config.timestep_hours
    discount = config.perfect_foresight_discount
    netto = pv - load

    has_bess = bess is not None
    has_hp = hp is not None

    # --- Variable layout ---
    i_sell = 0         # grid_sell: 0..T-1
    i_buy = T          # grid_buy:  T..2T-1
    next_idx = 2 * T

    # BESS variables (conditional)
    i_chg = i_dis = i_soc = -1
    if has_bess:
        i_chg = next_idx             # charge:    next..next+T-1
        i_dis = next_idx + T         # discharge: next+T..next+2T-1
        i_soc = next_idx + 2 * T     # soc:       next+2T..next+3T (T+1 vars)
        next_idx += 3 * T + 1

    # HP variables (conditional)
    i_wp = i_tsoc = -1
    if has_hp:
        i_wp = next_idx              # wp_load:       next..next+T-1
        i_tsoc = next_idx + T        # thermal_soc:   next+T..next+2T (T+1 vars)
        next_idx += 2 * T + 1

    n_vars = next_idx

    # --- BESS derived values ---
    soc_min = soc_max = max_energy = 0.0
    if has_bess:
        soc_min = bess.capacity_kwh * bess.min_soc_pct / 100.0
        soc_max = bess.capacity_kwh * bess.max_soc_pct / 100.0
        max_energy = bess.power_kw * dt

    # --- HP derived values ---
    wp_max_energy = 0.0
    if has_hp:
        wp_max_energy = hp.power_kw * dt

    # --- Objective ---
    c = np.zeros(n_vars)
    for t in range(T):
        c[i_sell + t] = -prices[t] * discount
        c[i_buy + t] = prices[t]

    # --- Bounds ---
    bounds: list[tuple[float, float | None]] = []
    # grid_sell: [0, inf)
    bounds.extend((0.0, None) for _ in range(T))
    # grid_buy: [0, inf)
    bounds.extend((0.0, None) for _ in range(T))

    if has_bess:
        # charge: [0, max_energy]
        bounds.extend((0.0, max_energy) for _ in range(T))
        # discharge: [0, max_energy]
        bounds.extend((0.0, max_energy) for _ in range(T))
        # soc: [soc_min, soc_max]
        bounds.extend((soc_min, soc_max) for _ in range(T + 1))

    if has_hp:
        # wp_load: [0, wp_max_energy]
        bounds.extend((0.0, wp_max_energy) for _ in range(T))
        # thermal_soc: [0, thermal_storage_kwh]
        bounds.extend((0.0, hp.thermal_storage_kwh) for _ in range(T + 1))

    # --- Equality constraints ---
    eq_rows: list[tuple[dict[int, float], float]] = []

    # 1. Energy balance (T rows)
    #    sell[t] - buy[t] + charge[t] - discharge[t]*RTE + wp_load[t] = netto[t]
    for t in range(T):
        row: dict[int, float] = {
            i_sell + t: 1.0,
            i_buy + t: -1.0,
        }
        if has_bess:
            row[i_chg + t] = 1.0
            row[i_dis + t] = -bess.rte
        if has_hp:
            row[i_wp + t] = 1.0
        eq_rows.append((row, netto[t]))

    # 2. BESS SoC linking (T rows)
    if has_bess:
        for t in range(T):
            eq_rows.append((
                {
                    i_soc + t + 1: 1.0,
                    i_soc + t: -1.0,
                    i_chg + t: -1.0,
                    i_dis + t: 1.0,
                },
                0.0,
            ))

    # 3. BESS SoC initial (1 row)
    if has_bess:
        eq_rows.append(({i_soc: 1.0}, bess.start_soc_kwh))

    # 4. HP daily energy balance (1 row)
    #    Σ wp_load[t] × COP[t] = daily_heat_demand
    if has_hp:
        row_hp: dict[int, float] = {}
        for t in range(T):
            row_hp[i_wp + t] = hp.cop_profile[t]
        eq_rows.append((row_hp, hp.daily_heat_demand_kwh))

    # 5. Thermal SoC linking (T rows)
    #    thermal_soc[t+1] = thermal_soc[t] + wp_load[t]*COP[t] - heat_demand[t]
    if has_hp:
        for t in range(T):
            eq_rows.append((
                {
                    i_tsoc + t + 1: 1.0,
                    i_tsoc + t: -1.0,
                    i_wp + t: -hp.cop_profile[t],
                },
                -hp.heat_demand_profile[t],
            ))

    # 6. Thermal SoC initial (1 row)
    if has_hp:
        eq_rows.append(({i_tsoc: 1.0}, hp.start_thermal_soc_kwh))

    # Build dense A_eq, b_eq
    n_eq = len(eq_rows)
    A_eq = np.zeros((n_eq, n_vars))
    b_eq = np.zeros(n_eq)
    for r, (coeffs, rhs) in enumerate(eq_rows):
        for col, val in coeffs.items():
            A_eq[r, col] = val
        b_eq[r] = rhs

    # --- Solve ---
    result = linprog(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        logger.warning(
            "Portfolio LP infeasible (status=%d: %s). "
            "Falling back to no-flex dispatch.",
            result.status,
            result.message,
        )
        fallback = _solve_no_flex(pv, load, prices, config)
        if has_bess:
            fallback.end_soc_kwh = bess.start_soc_kwh
            fallback.bess_soc = np.full(T + 1, bess.start_soc_kwh)
        if has_hp:
            fallback.end_thermal_soc_kwh = hp.start_thermal_soc_kwh
        fallback.solver_status = f"infeasible:{result.message}"
        return fallback

    x = result.x

    grid_sell = x[i_sell: i_sell + T]
    grid_buy = x[i_buy: i_buy + T]

    if has_bess:
        charge = x[i_chg: i_chg + T]
        discharge = x[i_dis: i_dis + T]
        soc = x[i_soc: i_soc + T + 1]
        end_soc = float(soc[-1])
    else:
        charge = np.zeros(T)
        discharge = np.zeros(T)
        soc = np.zeros(T + 1)
        end_soc = 0.0

    wp_load_out: np.ndarray | None = None
    thermal_soc_out: np.ndarray | None = None
    end_thermal_soc = 0.0
    if has_hp:
        wp_load_out = x[i_wp: i_wp + T]
        thermal_soc_out = x[i_tsoc: i_tsoc + T + 1]
        end_thermal_soc = float(thermal_soc_out[-1])

    sell_revenue = float(np.sum(grid_sell * prices * discount))
    buy_cost = float(np.sum(grid_buy * prices))
    system_cost = buy_cost - sell_revenue

    return PortfolioDailyResult(
        grid_sell=grid_sell,
        grid_buy=grid_buy,
        bess_charge=charge,
        bess_discharge=discharge,
        bess_soc=soc,
        system_cost=system_cost,
        end_soc_kwh=end_soc,
        solver_status="optimal",
        wp_load=wp_load_out,
        thermal_soc=thermal_soc_out,
        end_thermal_soc_kwh=end_thermal_soc,
    )
