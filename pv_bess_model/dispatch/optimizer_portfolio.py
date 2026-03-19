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

The LP supports optional BESS, heat pump (WP), and/or EV/V2G flexibility.
All flex blocks are conditionally included in the LP.

Variable layout (full flex, T = intervals_per_day)
--------------------------------------------------
Base variables (always present):
  grid_sell[t]      – T values
  grid_buy[t]       – T values

Conditional blocks (appended in order if present):
  BESS:  charge(T), discharge(T), soc(T+1)
  HP:    wp_load(T), thermal_soc(T+1), heat_unmet(T)
  EV:    ev_charge(T), ev_discharge(T), ev_soc(T+1)

Typical usage::

    from pv_bess_model.dispatch.optimizer_portfolio import (
        optimize_portfolio_day, PortfolioLPConfig, BessFlexParams,
        HeatPumpFlexParams, EVFlexParams,
    )

    result = optimize_portfolio_day(
        pv_production=pv_day,
        load_demand=load_day,
        spot_prices=prices_day,
        bess_params=bess,
        config=config,
        hp_params=hp,
        ev_params=ev,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from pv_bess_model.config.defaults import (
    DEFAULT_HEAT_UNMET_PENALTY_EUR_PER_KWH,
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


@dataclass(frozen=True)
class EVFlexParams:
    """EV/Wallbox parameters for the portfolio LP.

    Attributes
    ----------
    power_kw : float
        Charging (and V2G discharging) power of the fleet (kW).
    daily_energy_demand_kwh : float
        Daily energy demand of the EV fleet (kWh).
    usable_battery_kwh : float
        Usable battery capacity of the EV fleet (kWh).
    arrival_interval : int
        Interval index when EVs arrive (0-based, 0..T-1).
    departure_interval : int
        Interval index when EVs depart (0-based, 0..T-1).
        Must be > arrival_interval.
    v2g_enabled : bool
        Whether vehicle-to-grid discharge is allowed.
    v2g_rte : float
        V2G round-trip efficiency as a fraction (0, 1].
        Only relevant when *v2g_enabled* is True.
    min_departure_soc_pct : float
        Minimum SoC at departure as percent of usable capacity.
    start_soc_kwh : float
        EV fleet SoC at arrival in kWh.
    """

    power_kw: float
    daily_energy_demand_kwh: float
    usable_battery_kwh: float
    arrival_interval: int
    departure_interval: int
    v2g_enabled: bool
    v2g_rte: float
    min_departure_soc_pct: float
    start_soc_kwh: float = 0.0


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
    heat_unmet: np.ndarray | None = None
    end_thermal_soc_kwh: float = 0.0
    ev_charge: np.ndarray | None = None
    ev_discharge: np.ndarray | None = None
    ev_soc: np.ndarray | None = None
    end_ev_soc_kwh: float = 0.0


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
    ev_params: EVFlexParams | None = None,
) -> PortfolioDailyResult:
    """Solve the daily portfolio dispatch LP.

    Minimizes daily system cost (grid buy minus discounted grid sell)
    subject to energy balance, BESS SoC, power, and optional heat pump
    and EV/V2G constraints.

    When no flex is given, no LP is needed and the result equals World A
    for that day.

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
    ev_params:
        EV/V2G configuration, or ``None`` for no EV.

    Returns
    -------
    PortfolioDailyResult
        Optimal dispatch arrays and daily system cost.
    """
    if bess_params is None and hp_params is None and ev_params is None:
        return _solve_no_flex(pv_production, load_demand, spot_prices, config)

    return _solve_lp(
        pv_production, load_demand, spot_prices, bess_params, config,
        hp_params, ev_params,
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
    ev: EVFlexParams | None = None,
) -> PortfolioDailyResult:
    """Solve the daily LP with BESS, heat pump, and/or EV flexibility.

    Variable layout (conditionally built):
        Base: [grid_sell(T), grid_buy(T)]
        + BESS: [charge(T), discharge(T), soc(T+1)]
        + HP:   [wp_load(T), thermal_soc(T+1), heat_unmet(T)]
        + EV:   [ev_charge(T), ev_discharge(T), ev_soc(T+1)]

    Objective (minimize):
        min Σ [ grid_buy[t] × price[t] − grid_sell[t] × price[t] × discount ]

    Energy balance (all flex combined):
        sell[t] - buy[t] + charge[t] - discharge[t]*RTE
                         + wp_load[t]
                         + ev_charge[t] - ev_discharge[t]*v2g_rte
                         = netto[t]

    EV constraints:
        - ev_charge[t] = 0, ev_discharge[t] = 0 outside [arrival, departure)
        - ev_soc linking: ev_soc[t+1] = ev_soc[t] + ev_charge[t] - ev_discharge[t]
        - ev_soc initial: ev_soc[arrival] = start_soc
        - ev_soc departure >= min_departure_soc
        - If v2g_enabled=False: ev_discharge[t] = 0 for all t
    """
    T = config.intervals_per_day
    dt = config.timestep_hours
    discount = config.perfect_foresight_discount
    netto = pv - load

    has_bess = bess is not None
    has_hp = hp is not None
    has_ev = ev is not None

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
    i_wp = i_tsoc = i_hunmet = -1
    if has_hp:
        i_wp = next_idx              # wp_load:       next..next+T-1
        i_tsoc = next_idx + T        # thermal_soc:   next+T..next+2T (T+1 vars)
        i_hunmet = next_idx + 2 * T + 1  # heat_unmet: next+2T+1..next+3T
        next_idx += 3 * T + 1

    # EV variables (conditional)
    i_ev_chg = i_ev_dis = i_ev_soc = -1
    if has_ev:
        i_ev_chg = next_idx          # ev_charge:     next..next+T-1
        i_ev_dis = next_idx + T      # ev_discharge:  next+T..next+2T-1
        i_ev_soc = next_idx + 2 * T  # ev_soc:        next+2T..next+3T (T+1 vars)
        next_idx += 3 * T + 1

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

    # --- EV derived values ---
    ev_max_charge = ev_max_discharge = 0.0
    ev_soc_min = ev_soc_max = ev_min_dep_soc = 0.0
    if has_ev:
        ev_max_charge = ev.power_kw * dt
        ev_max_discharge = ev.power_kw * dt
        ev_soc_min = 0.0
        ev_soc_max = ev.usable_battery_kwh
        ev_min_dep_soc = ev.usable_battery_kwh * ev.min_departure_soc_pct / 100.0

    # --- Objective ---
    c = np.zeros(n_vars)
    for t in range(T):
        c[i_sell + t] = -prices[t] * discount
        c[i_buy + t] = prices[t]

    # Penalize unmet heat demand so the LP only uses it when physically necessary
    if has_hp:
        for t in range(T):
            c[i_hunmet + t] = DEFAULT_HEAT_UNMET_PENALTY_EUR_PER_KWH

    # --- Bounds ---
    bounds: list[tuple[float, float]] = []
    # grid_sell: [0, sell_bound_t] per interval
    # Upper bound = max possible local generation (PV + BESS discharge + EV V2G)
    for t in range(T):
        ub_sell = float(pv[t])
        if has_bess:
            ub_sell += max_energy * bess.rte
        if has_ev and ev.v2g_enabled and ev.arrival_interval <= t < ev.departure_interval:
            ub_sell += ev_max_discharge * ev.v2g_rte
        bounds.append((0.0, max(ub_sell, 0.0)))
    # grid_buy: [0, buy_bound_t] per interval
    # Upper bound = max possible local consumption (load + BESS charge + HP + EV charge)
    for t in range(T):
        ub_buy = float(load[t])
        if has_bess:
            ub_buy += max_energy
        if has_hp:
            ub_buy += wp_max_energy
        if has_ev and ev.arrival_interval <= t < ev.departure_interval:
            ub_buy += ev_max_charge
        bounds.append((0.0, max(ub_buy, 0.0)))

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
        # heat_unmet: [0, max demand per interval] – slack for infeasible days
        for t in range(T):
            bounds.append((0.0, float(hp.heat_demand_profile[t])))

    if has_ev:
        # ev_charge: [0, ev_max_charge] within window, fixed to 0 outside
        for t in range(T):
            if ev.arrival_interval <= t < ev.departure_interval:
                bounds.append((0.0, ev_max_charge))
            else:
                bounds.append((0.0, 0.0))
        # ev_discharge: [0, ev_max_discharge] within window if v2g, else 0
        for t in range(T):
            if ev.v2g_enabled and ev.arrival_interval <= t < ev.departure_interval:
                bounds.append((0.0, ev_max_discharge))
            else:
                bounds.append((0.0, 0.0))
        # ev_soc: [0, usable_battery_kwh]
        for t in range(T + 1):
            bounds.append((ev_soc_min, ev_soc_max))

    # --- Equality constraints ---
    eq_rows: list[tuple[dict[int, float], float]] = []

    # 1. Energy balance (T rows)
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
        if has_ev:
            row[i_ev_chg + t] = 1.0
            row[i_ev_dis + t] = -ev.v2g_rte
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

    # 4. HP daily energy balance with slack (1 row)
    #    Σ wp_load[t]*cop[t] + Σ heat_unmet[t] = daily_heat_demand_kwh
    #    heat_unmet[t] absorbs demand the HP cannot cover on peak days,
    #    preventing LP infeasibility while maintaining the daily cycle.
    if has_hp:
        row_hp: dict[int, float] = {}
        for t in range(T):
            row_hp[i_wp + t] = hp.cop_profile[t]
            row_hp[i_hunmet + t] = 1.0
        eq_rows.append((row_hp, hp.daily_heat_demand_kwh))

    # 5. Thermal SoC linking (T rows)
    #    tsoc[t+1] = tsoc[t] + wp_load[t]*cop[t] + heat_unmet[t] - heat_demand[t]
    if has_hp:
        for t in range(T):
            eq_rows.append((
                {
                    i_tsoc + t + 1: 1.0,
                    i_tsoc + t: -1.0,
                    i_wp + t: -hp.cop_profile[t],
                    i_hunmet + t: -1.0,
                },
                -hp.heat_demand_profile[t],
            ))

    # 6. Thermal SoC initial (1 row)
    if has_hp:
        eq_rows.append(({i_tsoc: 1.0}, hp.start_thermal_soc_kwh))

    # 7. EV SoC linking (T rows)
    #    ev_soc[t+1] = ev_soc[t] + ev_charge[t] - ev_discharge[t]
    if has_ev:
        for t in range(T):
            eq_rows.append((
                {
                    i_ev_soc + t + 1: 1.0,
                    i_ev_soc + t: -1.0,
                    i_ev_chg + t: -1.0,
                    i_ev_dis + t: 1.0,
                },
                0.0,
            ))

    # 8. EV SoC initial (1 row): ev_soc[0] = start_soc
    if has_ev:
        eq_rows.append(({i_ev_soc: 1.0}, ev.start_soc_kwh))

    # --- Inequality constraints ---
    # EV departure SoC constraint: ev_soc[departure] >= min_departure_soc
    # Expressed as: -ev_soc[departure] <= -min_departure_soc
    ineq_rows: list[tuple[dict[int, float], float]] = []
    if has_ev:
        ineq_rows.append((
            {i_ev_soc + ev.departure_interval: -1.0},
            -ev_min_dep_soc,
        ))

    # Build dense A_eq, b_eq
    n_eq = len(eq_rows)
    A_eq = np.zeros((n_eq, n_vars))
    b_eq = np.zeros(n_eq)
    for r, (coeffs, rhs) in enumerate(eq_rows):
        for col, val in coeffs.items():
            A_eq[r, col] = val
        b_eq[r] = rhs

    # Build dense A_ub, b_ub (inequality constraints)
    A_ub = None
    b_ub = None
    if ineq_rows:
        n_ineq = len(ineq_rows)
        A_ub = np.zeros((n_ineq, n_vars))
        b_ub = np.zeros(n_ineq)
        for r, (coeffs, rhs) in enumerate(ineq_rows):
            for col, val in coeffs.items():
                A_ub[r, col] = val
            b_ub[r] = rhs

    # --- Solve ---
    result = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
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
        if has_ev:
            fallback.end_ev_soc_kwh = ev.start_soc_kwh
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
    heat_unmet_out: np.ndarray | None = None
    end_thermal_soc = 0.0
    if has_hp:
        wp_load_out = x[i_wp: i_wp + T]
        thermal_soc_out = x[i_tsoc: i_tsoc + T + 1]
        heat_unmet_out = x[i_hunmet: i_hunmet + T]
        end_thermal_soc = float(thermal_soc_out[-1])

    ev_charge_out: np.ndarray | None = None
    ev_discharge_out: np.ndarray | None = None
    ev_soc_out: np.ndarray | None = None
    end_ev_soc = 0.0
    if has_ev:
        ev_charge_out = x[i_ev_chg: i_ev_chg + T]
        ev_discharge_out = x[i_ev_dis: i_ev_dis + T]
        ev_soc_out = x[i_ev_soc: i_ev_soc + T + 1]
        end_ev_soc = float(ev_soc_out[-1])

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
        heat_unmet=heat_unmet_out,
        end_thermal_soc_kwh=end_thermal_soc,
        ev_charge=ev_charge_out,
        ev_discharge=ev_discharge_out,
        ev_soc=ev_soc_out,
        end_ev_soc_kwh=end_ev_soc,
    )
