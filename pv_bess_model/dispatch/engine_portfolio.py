"""Multi-year portfolio dispatch engine with annual flex capacity build-out.

Runs a 25-year simulation loop (configurable lifetime) where flexibility
capacity grows each year via annual additions.  For BESS, a **tranche model**
applies per-tranche degradation so that older tranches degrade independently.

The engine calls the daily portfolio LP optimizer for each of 365 days per year
and aggregates costs into annual system cost values suitable for World-A/B
comparison and system value calculation.

Public API
----------
PortfolioEngineConfig    - Static simulation configuration.
PortfolioAnnualResult    - Per-year aggregate results.
FlexCapacityYear         - Flex capacities for one project year.
compute_bess_tranche_capacity - BESS tranche model calculation.
run_portfolio_simulation - Execute the multi-year portfolio dispatch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from pv_bess_model.config.defaults import (
    DAYS_PER_YEAR,
    DEFAULT_PERFECT_FORESIGHT_DISCOUNT,
    DEFAULT_PORTFOLIO_LIFETIME_YEARS,
    INTERVALS_PER_DAY,
    TIMESTEP_HOURS,
)
from pv_bess_model.dispatch.optimizer_portfolio import (
    BessFlexParams,
    EVFlexParams,
    HeatPumpFlexParams,
    PortfolioDailyResult,
    PortfolioLPConfig,
    optimize_portfolio_day,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioEngineConfig:
    """Configuration for the multi-year portfolio simulation.

    Attributes
    ----------
    lifetime_years : int
        Total project lifetime in years.
    baseline_year : int
        Calendar year corresponding to project year 1.
    timestep_hours : float
        Duration of one interval in hours (0.25 for quarter-hourly).
    intervals_per_day : int
        Number of intervals per day (96 for quarter-hourly).
    intervals_per_year : int
        Number of intervals per year (35,040 for quarter-hourly).
    perfect_foresight_discount : float
        Discount factor on grid-sell revenues (0.0 -- 1.0).
    """

    lifetime_years: int = DEFAULT_PORTFOLIO_LIFETIME_YEARS
    baseline_year: int = 2027
    timestep_hours: float = TIMESTEP_HOURS
    intervals_per_day: int = INTERVALS_PER_DAY
    intervals_per_year: int = DAYS_PER_YEAR * INTERVALS_PER_DAY
    perfect_foresight_discount: float = DEFAULT_PERFECT_FORESIGHT_DISCOUNT


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class PortfolioAnnualResult:
    """Aggregated result of one simulation year.

    Attributes
    ----------
    year : int
        Project year (1-indexed).
    system_cost : float
        Net system cost in EUR (positive = cost, negative = revenue).
    total_grid_sell_kwh : float
        Total energy sold to grid in kWh.
    total_grid_buy_kwh : float
        Total energy bought from grid in kWh.
    total_grid_sell_eur : float
        Total revenue from grid sales in EUR.
    total_grid_buy_eur : float
        Total cost of grid purchases in EUR.
    total_bess_throughput_kwh : float
        Total BESS discharge in kWh.
    bess_capacity_kwh : float
        Effective BESS capacity this year (after tranche degradation).
    bess_power_kw : float
        Effective BESS power this year.
    """

    year: int
    system_cost: float
    total_grid_sell_kwh: float
    total_grid_buy_kwh: float
    total_grid_sell_eur: float
    total_grid_buy_eur: float
    total_bess_throughput_kwh: float
    bess_capacity_kwh: float
    bess_power_kw: float
    wp_power_kw: float = 0.0
    total_wp_electrical_kwh: float = 0.0
    ev_power_kw: float = 0.0
    total_ev_charge_kwh: float = 0.0
    total_ev_discharge_kwh: float = 0.0


@dataclass
class FlexCapacityYear:
    """Flex capacities for one project year (after build-out + degradation).

    Attributes
    ----------
    bess_capacity_kwh : float
        Total BESS energy capacity in kWh (tranche-degraded).
    bess_power_kw : float
        Total BESS power in kW (sum of annual additions, no power degradation).
    """

    bess_capacity_kwh: float
    bess_power_kw: float
    wp_power_kw: float = 0.0
    ev_power_kw: float = 0.0


# ---------------------------------------------------------------------------
# BESS tranche model
# ---------------------------------------------------------------------------


def compute_bess_tranche_capacity(
    annual_addition_kw: float,
    e_to_p_ratio: float,
    project_year: int,
    degradation_rate: float,
    start_year: int = 1,
) -> tuple[float, float]:
    """Compute aggregated BESS capacity using the tranche degradation model.

    Each annual addition is treated as a separate tranche that degrades
    independently from its installation year.  All tranches are aggregated
    into a single BESS for the LP.

    For project year *n* (1-indexed), the total capacity is::

        Σ_{i=start_year}^{n} [ addition_kw × e_to_p × (1 - deg_rate)^(n - i) ]

    Power does not degrade -- only energy capacity.

    Parameters
    ----------
    annual_addition_kw:
        kW of BESS power added each year.
    e_to_p_ratio:
        Energy-to-power ratio in hours.
    project_year:
        Current project year (1-indexed).
    degradation_rate:
        Annual capacity degradation as a fraction (e.g. 0.02 for 2 %).
    start_year:
        First project year in which additions begin (1-indexed, default 1).

    Returns
    -------
    tuple[float, float]
        ``(total_power_kw, total_capacity_kwh)``
    """
    if project_year < 1:
        raise ValueError(f"project_year must be >= 1, got {project_year}.")
    if annual_addition_kw < 0:
        raise ValueError(
            f"annual_addition_kw must be >= 0, got {annual_addition_kw}."
        )

    if annual_addition_kw == 0.0 or project_year < start_year:
        return 0.0, 0.0

    total_power_kw = 0.0
    total_capacity_kwh = 0.0

    for install_year in range(start_year, project_year + 1):
        age = project_year - install_year  # years since installation
        tranche_power = annual_addition_kw
        tranche_capacity = annual_addition_kw * e_to_p_ratio * (
            (1.0 - degradation_rate) ** age
        )
        total_power_kw += tranche_power
        total_capacity_kwh += tranche_capacity

    return total_power_kw, total_capacity_kwh


# ---------------------------------------------------------------------------
# WP capacity model
# ---------------------------------------------------------------------------


def compute_wp_capacity(
    annual_addition_kw: float,
    project_year: int,
    start_year: int = 1,
) -> float:
    """Compute cumulative heat pump power for a project year.

    Heat pumps have no degradation — capacity grows linearly.

    Parameters
    ----------
    annual_addition_kw:
        kW of WP power added each year.
    project_year:
        Current project year (1-indexed).
    start_year:
        First project year in which additions begin (1-indexed, default 1).

    Returns
    -------
    float
        Total installed WP power in kW.
    """
    if project_year < start_year or annual_addition_kw <= 0.0:
        return 0.0
    n_years = project_year - start_year + 1
    return annual_addition_kw * n_years


# ---------------------------------------------------------------------------
# EV capacity model
# ---------------------------------------------------------------------------


def compute_ev_capacity(
    annual_addition_kw: float,
    project_year: int,
    start_year: int = 1,
) -> float:
    """Compute cumulative EV fleet charging power for a project year.

    EV capacity has no degradation — power grows linearly.

    Parameters
    ----------
    annual_addition_kw:
        kW of EV charging power added each year.
    project_year:
        Current project year (1-indexed).
    start_year:
        First project year in which additions begin (1-indexed, default 1).

    Returns
    -------
    float
        Total installed EV charging power in kW.
    """
    if project_year < start_year or annual_addition_kw <= 0.0:
        return 0.0
    n_years = project_year - start_year + 1
    return annual_addition_kw * n_years


# ---------------------------------------------------------------------------
# Multi-year simulation
# ---------------------------------------------------------------------------


def run_portfolio_simulation(
    config: PortfolioEngineConfig,
    pv_profile_base: np.ndarray,
    load_profile_base: np.ndarray,
    spot_prices_base: np.ndarray,
    annual_addition_kw: float,
    e_to_p_ratio: float,
    bess_rte: float,
    bess_min_soc_pct: float,
    bess_max_soc_pct: float,
    bess_degradation_rate: float,
    pv_degradation_rate: float,
    load_growth_factor: float = 1.0,
    start_year: int = 1,
    wp_annual_addition_kw: float = 0.0,
    wp_cop_profile_base: np.ndarray | None = None,
    wp_heat_demand_profile_base: np.ndarray | None = None,
    wp_daily_heat_demand_base: np.ndarray | None = None,
    wp_thermal_storage_kwh: float = 0.0,
    wp_start_year: int = 1,
    wp_base_power_kw: float = 0.0,
    ev_annual_addition_kw: float = 0.0,
    ev_daily_energy_demand_kwh_base: float = 0.0,
    ev_usable_battery_kwh_per_kw: float = 0.0,
    ev_arrival_interval: int = 0,
    ev_departure_interval: int = 0,
    ev_v2g_enabled: bool = False,
    ev_v2g_rte: float = 0.9,
    ev_min_departure_soc_pct: float = 80.0,
    ev_start_year: int = 1,
    ev_base_power_kw: float = 0.0,
    pv_profiles_by_year: list[np.ndarray] | None = None,
) -> list[PortfolioAnnualResult]:
    """Run a multi-year portfolio simulation with annual flex build-out.

    For each project year:
      1. Apply PV degradation to the base PV profile.
      2. Apply load growth to the base load profile.
      3. Compute BESS capacity via tranche model (with degradation).
      4. Compute WP capacity (linear, no degradation).
      5. Compute EV capacity (linear, no degradation).
      6. Run 365 daily LP optimizations.
      7. Aggregate daily results into annual totals.

    Parameters
    ----------
    config:
        Engine configuration (lifetime, timestep, etc.).
    pv_profile_base:
        Base PV production profile (kWh values for one year).
    load_profile_base:
        Base load profile (kWh values for one year).
    spot_prices_base:
        Spot prices (EUR/kWh values for one year).
    annual_addition_kw:
        BESS power added each year (kW). 0 for World A.
    e_to_p_ratio:
        Energy-to-power ratio in hours.
    bess_rte:
        BESS round-trip efficiency as a fraction (0, 1].
    bess_min_soc_pct:
        Minimum SoC as percent of capacity.
    bess_max_soc_pct:
        Maximum SoC as percent of capacity.
    bess_degradation_rate:
        Annual capacity degradation fraction (e.g. 0.02).
    pv_degradation_rate:
        Annual PV degradation fraction (e.g. 0.004).
    load_growth_factor:
        Multiplicative annual load growth factor (e.g. 1.01 = +1%/year).
    start_year:
        First project year when BESS additions begin (1-indexed).
    wp_annual_addition_kw:
        WP electrical power added each year (kW). 0 for no WP.
    wp_cop_profile_base:
        Base COP profile per interval (length = intervals_per_year).
    wp_heat_demand_profile_base:
        Base heat demand profile per interval (kWh_th).
    wp_daily_heat_demand_base:
        Daily heat demand array (kWh_th, length = 365).
    wp_thermal_storage_kwh:
        Base thermal storage capacity in kWh_th (scales with WP capacity).
    wp_start_year:
        First project year when WP additions begin (1-indexed).
    wp_base_power_kw:
        Reference WP power for scaling heat demand and thermal storage.
    ev_annual_addition_kw:
        EV charging power added each year (kW). 0 for no EV.
    ev_daily_energy_demand_kwh_base:
        Daily energy demand at base power level (kWh).
    ev_usable_battery_kwh_per_kw:
        Usable battery capacity per kW of charging power (kWh/kW).
    ev_arrival_interval:
        Interval index when EVs arrive (0-based).
    ev_departure_interval:
        Interval index when EVs depart (0-based).
    ev_v2g_enabled:
        Whether V2G discharge is allowed.
    ev_v2g_rte:
        V2G round-trip efficiency as a fraction.
    ev_min_departure_soc_pct:
        Minimum SoC at departure as percent of usable capacity.
    ev_start_year:
        First project year when EV additions begin (1-indexed).
    ev_base_power_kw:
        Reference EV power for scaling demand and battery capacity.
    pv_profiles_by_year:
        Pre-computed aggregated PV profiles per project year (index 0 = year 1).
        When provided, overrides ``pv_profile_base * (1-rate)^year`` logic,
        supporting per-asset commissioning years and lifetimes.

    Returns
    -------
    list[PortfolioAnnualResult]
        One result per project year (index 0 = year 1).
    """
    T = config.intervals_per_day
    lp_config = PortfolioLPConfig(
        timestep_hours=config.timestep_hours,
        intervals_per_day=config.intervals_per_day,
        perfect_foresight_discount=config.perfect_foresight_discount,
    )

    # Check if WP is active
    has_wp = (
        wp_annual_addition_kw > 0.0
        and wp_cop_profile_base is not None
        and wp_heat_demand_profile_base is not None
        and wp_daily_heat_demand_base is not None
        and wp_base_power_kw > 0.0
    )

    # Check if EV is active
    has_ev = (
        ev_annual_addition_kw > 0.0
        and ev_base_power_kw > 0.0
        and ev_departure_interval > ev_arrival_interval
    )

    annual_results: list[PortfolioAnnualResult] = []

    # SoC carried across years
    carry_soc_kwh = 0.0
    carry_thermal_soc_kwh = 0.0
    carry_ev_soc_kwh = 0.0

    for year in range(1, config.lifetime_years + 1):
        # 1. PV degradation (per-asset profiles or simple scalar)
        if pv_profiles_by_year is not None:
            pv_profile = pv_profiles_by_year[year - 1]
        else:
            pv_factor = (1.0 - pv_degradation_rate) ** year
            pv_profile = pv_profile_base * pv_factor

        # 2. Load growth
        load_factor = load_growth_factor ** year
        load_profile = load_profile_base * load_factor

        # 3. BESS capacity (tranche model)
        bess_power, bess_capacity = compute_bess_tranche_capacity(
            annual_addition_kw=annual_addition_kw,
            e_to_p_ratio=e_to_p_ratio,
            project_year=year,
            degradation_rate=bess_degradation_rate,
            start_year=start_year,
        )

        # Build BESS params (or None for no-BESS)
        bess_params: BessFlexParams | None = None
        if bess_power > 0.0 and bess_capacity > 0.0:
            soc_min = bess_capacity * bess_min_soc_pct / 100.0
            soc_max = bess_capacity * bess_max_soc_pct / 100.0

            # Clip carry-over SoC to valid range for new capacity
            start_soc = max(soc_min, min(carry_soc_kwh, soc_max))

            bess_params = BessFlexParams(
                capacity_kwh=bess_capacity,
                power_kw=bess_power,
                rte=bess_rte,
                min_soc_pct=bess_min_soc_pct,
                max_soc_pct=bess_max_soc_pct,
                start_soc_kwh=start_soc,
            )

        # 4. WP capacity (linear, no degradation)
        wp_power = 0.0
        wp_scale = 0.0
        if has_wp:
            wp_power = compute_wp_capacity(
                annual_addition_kw=wp_annual_addition_kw,
                project_year=year,
                start_year=wp_start_year,
            )
            if wp_power > 0.0:
                wp_scale = wp_power / wp_base_power_kw

        # 5. EV capacity (linear, no degradation)
        ev_power = 0.0
        ev_scale = 0.0
        if has_ev:
            ev_power = compute_ev_capacity(
                annual_addition_kw=ev_annual_addition_kw,
                project_year=year,
                start_year=ev_start_year,
            )
            if ev_power > 0.0:
                ev_scale = ev_power / ev_base_power_kw

        # 6. Run 365 daily LP optimizations
        year_system_cost = 0.0
        year_grid_sell_kwh = 0.0
        year_grid_buy_kwh = 0.0
        year_grid_sell_eur = 0.0
        year_grid_buy_eur = 0.0
        year_bess_throughput = 0.0
        year_wp_electrical_kwh = 0.0
        year_ev_charge_kwh = 0.0
        year_ev_discharge_kwh = 0.0

        for day in range(DAYS_PER_YEAR):
            day_start = day * T
            day_end = (day + 1) * T

            pv_day = pv_profile[day_start:day_end]
            load_day = load_profile[day_start:day_end]
            prices_day = spot_prices_base[day_start:day_end]

            # Update start SoC for BESS from carry-over
            if bess_params is not None and day > 0:
                soc_min = bess_capacity * bess_min_soc_pct / 100.0
                soc_max = bess_capacity * bess_max_soc_pct / 100.0
                bess_params = BessFlexParams(
                    capacity_kwh=bess_params.capacity_kwh,
                    power_kw=bess_params.power_kw,
                    rte=bess_params.rte,
                    min_soc_pct=bess_params.min_soc_pct,
                    max_soc_pct=bess_params.max_soc_pct,
                    start_soc_kwh=carry_soc_kwh,
                )

            # Build HP params for this day (or None)
            hp_params: HeatPumpFlexParams | None = None
            if has_wp and wp_power > 0.0:
                cop_day = wp_cop_profile_base[day_start:day_end]
                heat_demand_day = wp_heat_demand_profile_base[day_start:day_end] * wp_scale
                daily_heat = wp_daily_heat_demand_base[day] * wp_scale
                thermal_cap = wp_thermal_storage_kwh * wp_scale

                # Clip thermal SoC carry-over
                start_tsoc = max(0.0, min(carry_thermal_soc_kwh, thermal_cap))

                hp_params = HeatPumpFlexParams(
                    power_kw=wp_power,
                    cop_profile=cop_day,
                    daily_heat_demand_kwh=daily_heat,
                    thermal_storage_kwh=thermal_cap,
                    heat_demand_profile=heat_demand_day,
                    start_thermal_soc_kwh=start_tsoc,
                )

            # Build EV params for this day (or None)
            ev_day_params: EVFlexParams | None = None
            if has_ev and ev_power > 0.0:
                ev_demand = ev_daily_energy_demand_kwh_base * ev_scale
                ev_battery = ev_usable_battery_kwh_per_kw * ev_power

                # EV arrival SoC: carry-over minus driving consumption
                # (EVs departed, drove, consumed energy, arrived back)
                arrival_soc = max(0.0, carry_ev_soc_kwh - ev_demand)
                arrival_soc = min(arrival_soc, ev_battery)

                ev_day_params = EVFlexParams(
                    power_kw=ev_power,
                    daily_energy_demand_kwh=ev_demand,
                    usable_battery_kwh=ev_battery,
                    arrival_interval=ev_arrival_interval,
                    departure_interval=ev_departure_interval,
                    v2g_enabled=ev_v2g_enabled,
                    v2g_rte=ev_v2g_rte,
                    min_departure_soc_pct=ev_min_departure_soc_pct,
                    start_soc_kwh=arrival_soc,
                )

            result: PortfolioDailyResult = optimize_portfolio_day(
                pv_production=pv_day,
                load_demand=load_day,
                spot_prices=prices_day,
                bess_params=bess_params,
                config=lp_config,
                hp_params=hp_params,
                ev_params=ev_day_params,
            )

            year_system_cost += result.system_cost

            sell_kwh = float(np.sum(result.grid_sell))
            buy_kwh = float(np.sum(result.grid_buy))
            year_grid_sell_kwh += sell_kwh
            year_grid_buy_kwh += buy_kwh
            year_grid_sell_eur += float(
                np.sum(result.grid_sell * prices_day * config.perfect_foresight_discount)
            )
            year_grid_buy_eur += float(np.sum(result.grid_buy * prices_day))
            year_bess_throughput += float(np.sum(result.bess_discharge))

            if result.wp_load is not None:
                year_wp_electrical_kwh += float(np.sum(result.wp_load))

            if result.ev_charge is not None:
                year_ev_charge_kwh += float(np.sum(result.ev_charge))
            if result.ev_discharge is not None:
                year_ev_discharge_kwh += float(np.sum(result.ev_discharge))

            carry_soc_kwh = result.end_soc_kwh
            carry_thermal_soc_kwh = result.end_thermal_soc_kwh
            carry_ev_soc_kwh = result.end_ev_soc_kwh

        # 7. Record annual result
        annual_results.append(
            PortfolioAnnualResult(
                year=year,
                system_cost=year_system_cost,
                total_grid_sell_kwh=year_grid_sell_kwh,
                total_grid_buy_kwh=year_grid_buy_kwh,
                total_grid_sell_eur=year_grid_sell_eur,
                total_grid_buy_eur=year_grid_buy_eur,
                total_bess_throughput_kwh=year_bess_throughput,
                bess_capacity_kwh=bess_capacity,
                bess_power_kw=bess_power,
                wp_power_kw=wp_power,
                total_wp_electrical_kwh=year_wp_electrical_kwh,
                ev_power_kw=ev_power,
                total_ev_charge_kwh=year_ev_charge_kwh,
                total_ev_discharge_kwh=year_ev_discharge_kwh,
            )
        )

        logger.debug(
            "Year %d: system_cost=%.0f EUR, bess=%.0f kW / %.0f kWh, "
            "wp=%.0f kW, ev=%.0f kW, sell=%.0f MWh, buy=%.0f MWh",
            year,
            year_system_cost,
            bess_power,
            bess_capacity,
            wp_power,
            ev_power,
            year_grid_sell_kwh / 1000.0,
            year_grid_buy_kwh / 1000.0,
        )

    return annual_results
