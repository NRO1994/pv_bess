"""Daily LP-based dispatch optimizer for the portfolio/Systemwert model.

Solves a linear programme for each simulation day to determine optimal
BESS charge/discharge decisions that minimize the net system cost (grid
buy cost minus grid sell revenue) under perfect day-ahead price foresight.

Unlike the PV+BESS optimizer (``optimizer.py``), this LP:
- Has no grid export limit
- Has no Green/Grey mode distinction
- Has no PPA/EEG floor/cap pricing
- Uses a ``perfect_foresight_discount`` on sell revenues
- Models a bidirectional net grid position (buy AND sell possible)

The LP is designed for extensibility: heat pump and EV/V2G flex variables
will be added in Phases 5 and 6 respectively.

Variable layout (BESS-only, T = intervals_per_day)
---------------------------------------------------
===========  =====  ==========================================
Slice        Len    Variable
===========  =====  ==========================================
0   .. T-1   T      grid_sell[t]      – net grid export (kWh)
T   .. 2T-1  T      grid_buy[t]       – net grid import (kWh)
2T  .. 3T-1  T      bess_charge[t]    – BESS charging (kWh)
3T  .. 4T-1  T      bess_discharge[t] – BESS discharging (kWh)
4T  .. 5T    T+1    soc[t]            – SoC (kWh), t=0..T
===========  =====  ==========================================

Total variables: 5T + 1

Typical usage::

    from pv_bess_model.dispatch.optimizer_portfolio import (
        optimize_portfolio_day, PortfolioLPConfig, BessFlexParams,
    )

    result = optimize_portfolio_day(
        pv_production=pv_day,
        load_demand=load_day,
        spot_prices=prices_day,
        bess_params=bess,
        config=config,
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


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def optimize_portfolio_day(
    pv_production: np.ndarray,
    load_demand: np.ndarray,
    spot_prices: np.ndarray,
    bess_params: BessFlexParams | None,
    config: PortfolioLPConfig,
) -> PortfolioDailyResult:
    """Solve the daily portfolio dispatch LP.

    Minimizes daily system cost (grid buy minus discounted grid sell)
    subject to energy balance, BESS SoC, and power constraints.

    When *bess_params* is ``None``, no BESS is available and the result
    equals World A for that day.

    Parameters
    ----------
    pv_production:
        PV production per interval in kWh (length = intervals_per_day).
    load_demand:
        Load demand per interval in kWh (length = intervals_per_day).
    spot_prices:
        Spot price per interval in EUR/kWh (length = intervals_per_day).
    bess_params:
        BESS configuration, or ``None`` for World A (no storage).
    config:
        LP configuration (timestep, discount).

    Returns
    -------
    PortfolioDailyResult
        Optimal dispatch arrays and daily system cost.
    """
    T = config.intervals_per_day

    if bess_params is None:
        return _solve_no_bess(pv_production, load_demand, spot_prices, config)

    return _solve_with_bess(pv_production, load_demand, spot_prices, bess_params, config)


# ---------------------------------------------------------------------------
# No-BESS case (analytical, no LP needed)
# ---------------------------------------------------------------------------


def _solve_no_bess(
    pv: np.ndarray,
    load: np.ndarray,
    prices: np.ndarray,
    config: PortfolioLPConfig,
) -> PortfolioDailyResult:
    """Compute dispatch without BESS (equivalent to World A for one day)."""
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
        solver_status="no_bess",
    )


# ---------------------------------------------------------------------------
# BESS LP
# ---------------------------------------------------------------------------


def _solve_with_bess(
    pv: np.ndarray,
    load: np.ndarray,
    prices: np.ndarray,
    bess: BessFlexParams,
    config: PortfolioLPConfig,
) -> PortfolioDailyResult:
    """Solve the daily LP with BESS flexibility.

    Variable layout:
        x = [grid_sell(T), grid_buy(T), charge(T), discharge(T), soc(T+1)]
        Total: 5T + 1 variables

    Objective (minimize):
        min Σ [ grid_buy[t] × price[t] − grid_sell[t] × price[t] × discount ]

    Constraints:
        1. Energy balance (equality, T constraints):
           grid_sell[t] − grid_buy[t] = netto[t] + discharge[t]×RTE − charge[t]

        2. SoC linking (equality, T constraints):
           soc[t+1] = soc[t] + charge[t] − discharge[t]

        3. SoC initial (equality, 1 constraint):
           soc[0] = start_soc

    Bounds:
        grid_sell[t] ≥ 0
        grid_buy[t] ≥ 0
        0 ≤ charge[t] ≤ P_max × timestep_hours
        0 ≤ discharge[t] ≤ P_max × timestep_hours
        soc_min ≤ soc[t] ≤ soc_max
    """
    T = config.intervals_per_day
    dt = config.timestep_hours
    discount = config.perfect_foresight_discount

    netto = pv - load  # net position without BESS

    soc_min = bess.capacity_kwh * bess.min_soc_pct / 100.0
    soc_max = bess.capacity_kwh * bess.max_soc_pct / 100.0
    max_energy = bess.power_kw * dt  # kWh per interval

    # Variable indices
    i_sell = 0            # grid_sell: 0..T-1
    i_buy = T             # grid_buy:  T..2T-1
    i_chg = 2 * T         # charge:    2T..3T-1
    i_dis = 3 * T         # discharge: 3T..4T-1
    i_soc = 4 * T         # soc:       4T..5T  (T+1 variables)
    n_vars = 5 * T + 1

    # --- Objective: min Σ [ buy[t]*price[t] - sell[t]*price[t]*discount ] ---
    c = np.zeros(n_vars)
    for t in range(T):
        c[i_sell + t] = -prices[t] * discount   # selling = negative cost
        c[i_buy + t] = prices[t]                 # buying = positive cost

    # --- Bounds ---
    bounds: list[tuple[float, float]] = []
    # grid_sell: [0, inf)
    for _ in range(T):
        bounds.append((0.0, None))
    # grid_buy: [0, inf)
    for _ in range(T):
        bounds.append((0.0, None))
    # charge: [0, max_energy]
    for _ in range(T):
        bounds.append((0.0, max_energy))
    # discharge: [0, max_energy]
    for _ in range(T):
        bounds.append((0.0, max_energy))
    # soc: [soc_min, soc_max] for T+1 variables
    for _ in range(T + 1):
        bounds.append((soc_min, soc_max))

    # --- Equality constraints ---
    # We build A_eq and b_eq for:
    #   1. Energy balance (T rows)
    #   2. SoC linking (T rows)
    #   3. SoC initial (1 row)
    n_eq = 2 * T + 1
    A_eq = np.zeros((n_eq, n_vars))
    b_eq = np.zeros(n_eq)

    # 1. Energy balance: sell[t] - buy[t] - discharge[t]*RTE + charge[t] = netto[t]
    #    Rearranged: sell[t] - buy[t] + charge[t] - discharge[t]*RTE = netto[t]
    #    Wait, from spec: grid_sell[t] - grid_buy[t] = pv[t] - load[t] + discharge[t]*RTE - charge[t]
    #    So: sell[t] - buy[t] - discharge[t]*RTE + charge[t] = netto[t]
    #    → sell[t] - buy[t] + charge[t] - discharge[t]*RTE = netto[t]
    #    Nope, let me be precise:
    #    sell[t] - buy[t] = netto[t] + discharge[t]*RTE - charge[t]
    #    → sell[t] - buy[t] - discharge[t]*RTE + charge[t] = netto[t]
    for t in range(T):
        row = t
        A_eq[row, i_sell + t] = 1.0
        A_eq[row, i_buy + t] = -1.0
        A_eq[row, i_chg + t] = 1.0
        A_eq[row, i_dis + t] = -bess.rte
        b_eq[row] = netto[t]

    # 2. SoC linking: soc[t+1] - soc[t] - charge[t] + discharge[t] = 0
    for t in range(T):
        row = T + t
        A_eq[row, i_soc + t + 1] = 1.0
        A_eq[row, i_soc + t] = -1.0
        A_eq[row, i_chg + t] = -1.0
        A_eq[row, i_dis + t] = 1.0
        b_eq[row] = 0.0

    # 3. SoC initial: soc[0] = start_soc
    row = 2 * T
    A_eq[row, i_soc] = 1.0
    b_eq[row] = bess.start_soc_kwh

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
            "Falling back to no-BESS dispatch.",
            result.status,
            result.message,
        )
        fallback = _solve_no_bess(pv, load, prices, config)
        fallback.end_soc_kwh = bess.start_soc_kwh
        fallback.bess_soc = np.full(T + 1, bess.start_soc_kwh)
        fallback.solver_status = f"infeasible:{result.message}"
        return fallback

    x = result.x

    grid_sell = x[i_sell: i_sell + T]
    grid_buy = x[i_buy: i_buy + T]
    charge = x[i_chg: i_chg + T]
    discharge = x[i_dis: i_dis + T]
    soc = x[i_soc: i_soc + T + 1]

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
        end_soc_kwh=float(soc[-1]),
        solver_status="optimal",
    )
