"""Report data aggregator for the portfolio/Systemwert HTML dashboard.

Collects all portfolio simulation results into a single
``PortfolioReportData`` dataclass that serves as the data source for
the ``dashboard_portfolio.html`` template.

Public API
----------
PortfolioReportData     -- Dataclass holding all report data.
collect_portfolio_report_data -- Factory function to aggregate results.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

import numpy as np

from pv_bess_model.config.defaults import (
    INTERVALS_PER_DAY,
    REPORT_MODEL_VERSION,
)
from pv_bess_model.config.loader_portfolio import (
    BessFlexConfig,
    EVFlexConfig,
    GenerationConfig,
    HeatPumpFlexConfig,
    LoadGroupConfig,
    MetaModelConfig,
    PortfolioConfig,
)
from pv_bess_model.dispatch.engine_portfolio import PortfolioAnnualResult
from pv_bess_model.portfolio.marginal_value import MarginalValuePoint
from pv_bess_model.portfolio.system_value import SystemValueResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON sanitisation
# ---------------------------------------------------------------------------


def _sanitize_for_json(obj: Any) -> Any:
    """Replace NaN/Infinity with ``None`` for JSON serialisation."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.floating):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    return obj


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class PortfolioReportData:
    """All data required to render the portfolio HTML report.

    Fields are grouped by the tab they are displayed in.
    """

    # -- Meta --
    scenario_name: str
    creation_date: str  # DD.MM.YYYY
    baseline_year: int
    lifetime_years: int
    model_version: str
    perfect_foresight_discount: float

    # -- Tab 1: Portfolio overview --
    generation: list[dict]  # [{name, type, peak_kwp, lat, lon}]
    load_groups: list[dict]  # [{name, slp_type, customers, annual_kwh}]
    flexibilities: list[dict]  # [{name, type, rates, ...}]

    # -- Tab 2: World A --
    world_a_annual_costs: list[float]  # EUR per year
    world_a_total_cost: float  # EUR cumulative

    # -- Tab 3: System value curves --
    system_value_points: list[dict]
    # [{flex_name, flex_type, annual_addition_kw, e_to_p_ratio,
    #   cumulative_system_value_eur}]

    # -- Tab 4: Marginal value curves --
    marginal_value_points: list[dict]
    # [{flex_name, flex_type, annual_addition_kw, e_to_p_ratio,
    #   marginal_value_eur_per_kw_a, cumulative_system_value_eur}]

    # -- Tab 2: World A – average week profiles (optional) --
    world_a_avg_week_loads: list[dict] = field(default_factory=list)
    # [{name, values: [672 floats]}]  – per load group, Mo-So × 96 intervals
    world_a_avg_week_generation: list[dict] = field(default_factory=list)
    # [{name, values: [672 floats]}]  – per generation unit, Mo-So × 96 intervals
    world_a_avg_week_prices: list[dict] = field(default_factory=list)
    # [{name, values: [672 floats]}]  – per price scenario, Mo-So × 96 intervals

    # -- Tab 5: Dispatch sample (optional) --
    dispatch_sample_year: int | None = None
    dispatch_sample_summary: dict | None = None
    # {total_sell_kwh, total_buy_kwh, total_bess_kwh, ...}

    def to_json(self) -> str:
        """Serialise to a compact JSON string (NaN/Inf → null)."""
        raw = asdict(self)
        clean = _sanitize_for_json(raw)
        return json.dumps(clean, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Helper extractors
# ---------------------------------------------------------------------------


def _extract_generation(gen_configs: list[GenerationConfig]) -> list[dict]:
    """Convert generation configs to report dicts."""
    return [
        {
            "name": g.name,
            "type": g.type,
            "peak_kwp": g.peak_power_kwp,
            "latitude": g.latitude,
            "longitude": g.longitude,
            "degradation_pct": g.degradation_rate_pct_per_year,
            "commissioning_year": g.commissioning_year,
            "lifetime_years": g.lifetime_years,
        }
        for g in gen_configs
    ]


def _extract_load_groups(load_configs: list[LoadGroupConfig]) -> list[dict]:
    """Convert load group configs to report dicts."""
    return [
        {
            "name": lg.name,
            "slp_type": lg.slp_type,
            "customer_count": lg.customer_count,
            "annual_kwh_per_customer": lg.annual_consumption_kwh_per_customer,
            "total_mwh": (
                lg.customer_count
                * lg.annual_consumption_kwh_per_customer
                / 1000.0
            ),
            "growth_factor": lg.annual_growth_factor,
        }
        for lg in load_configs
    ]


def _extract_flexibilities(
    flex_configs: list[BessFlexConfig | HeatPumpFlexConfig | EVFlexConfig],
) -> list[dict]:
    """Convert flex configs to report dicts."""
    result: list[dict] = []
    for f in flex_configs:
        entry: dict[str, Any] = {
            "name": f.name,
            "type": f.type,
            "start_year": f.start_year,
        }
        if isinstance(f, BessFlexConfig):
            entry["annual_addition_kw"] = f.annual_addition_kw
            entry["e_to_p_ratio_hours"] = f.e_to_p_ratio_hours
            entry["rte_pct"] = f.round_trip_efficiency_pct
            entry["degradation_pct"] = f.degradation_rate_pct_per_year
        elif isinstance(f, HeatPumpFlexConfig):
            entry["annual_addition_kw"] = f.annual_addition_kw
            entry["cop_nominal"] = f.cop_nominal
            entry["thermal_demand_mwh"] = f.annual_thermal_demand_mwh
            entry["thermal_storage_kwh"] = f.thermal_storage_kwh
        elif isinstance(f, EVFlexConfig):
            entry["annual_additional_units"] = f.annual_additional_units
            entry["kw_per_unit"] = f.mean_kw_per_unit
            entry["v2g_enabled"] = f.v2g_enabled
            entry["battery_kwh_per_unit"] = f.usable_battery_kwh_per_unit
        result.append(entry)
    return result


def _extract_system_value_points(
    sv_result: SystemValueResult,
) -> list[dict]:
    """Convert system value points to report dicts."""
    return [
        {
            "flex_name": pt.flex_name,
            "flex_type": pt.flex_type,
            "annual_addition_kw": pt.annual_addition_kw,
            "e_to_p_ratio": pt.e_to_p_ratio,
            "cumulative_system_value_eur": pt.cumulative_system_value_eur,
            "annual_system_values": pt.annual_system_values,
        }
        for pt in sv_result.points
    ]


def _extract_marginal_value_points(
    marginals: list[MarginalValuePoint],
) -> list[dict]:
    """Convert marginal value points to report dicts."""
    return [
        {
            "flex_name": mv.flex_name,
            "flex_type": mv.flex_type,
            "annual_addition_kw": mv.annual_addition_kw,
            "e_to_p_ratio": mv.e_to_p_ratio,
            "cumulative_system_value_eur": mv.cumulative_system_value_eur,
            "marginal_value_eur_per_kw_a": mv.marginal_value_eur_per_kw_a,
            "cumulative_cost_eur": mv.cumulative_cost_eur,
            "marginal_cost_eur_per_kw_a": mv.marginal_cost_eur_per_kw_a,
            "is_optimal": mv.is_optimal,
            "delta_kw": mv.delta_kw,
            "delta_value_eur": mv.delta_value_eur,
        }
        for mv in marginals
    ]


def _extract_dispatch_summary(
    annual_results: list[PortfolioAnnualResult],
    year_index: int = 0,
) -> dict | None:
    """Extract summary stats for a dispatch sample year."""
    if not annual_results or year_index >= len(annual_results):
        return None
    ar = annual_results[year_index]
    return {
        "year": ar.year,
        "system_cost_eur": ar.system_cost,
        "total_sell_kwh": ar.total_grid_sell_kwh,
        "total_buy_kwh": ar.total_grid_buy_kwh,
        "total_sell_eur": ar.total_grid_sell_eur,
        "total_buy_eur": ar.total_grid_buy_eur,
        "bess_throughput_kwh": ar.total_bess_throughput_kwh,
        "bess_capacity_kwh": ar.bess_capacity_kwh,
        "bess_power_kw": ar.bess_power_kw,
        "wp_power_kw": ar.wp_power_kw,
        "wp_electrical_kwh": ar.total_wp_electrical_kwh,
        "ev_power_kw": ar.ev_power_kw,
        "ev_charge_kwh": ar.total_ev_charge_kwh,
        "ev_discharge_kwh": ar.total_ev_discharge_kwh,
    }


# ---------------------------------------------------------------------------
# Average week computation
# ---------------------------------------------------------------------------


def compute_average_week(
    profile_qh: np.ndarray,
    year: int,
) -> list[float]:
    """Compute the average week profile from a quarter-hourly annual profile.

    Groups 365 days by weekday (Monday=0 … Sunday=6), averages all
    occurrences of each weekday, and concatenates to produce 672 values
    (7 days × 96 quarter-hour intervals).

    Parameters
    ----------
    profile_qh:
        Quarter-hourly profile (35,040 values for 365 days).
    year:
        Calendar year used to determine weekday of January 1st.

    Returns
    -------
    list[float]
        672 values representing the average week (Mon–Sun).
    """
    import datetime

    n_days = 365
    ipd = INTERVALS_PER_DAY  # 96
    expected = n_days * ipd

    if len(profile_qh) < expected:
        logger.warning(
            "Profile has %d values, expected %d. Padding with zeros.",
            len(profile_qh),
            expected,
        )
        padded = np.zeros(expected, dtype=float)
        padded[: len(profile_qh)] = profile_qh
        profile_qh = padded

    # Reshape to (365, 96)
    daily = profile_qh[:expected].reshape(n_days, ipd)

    # Determine weekday for each day (0=Monday, 6=Sunday)
    jan1 = datetime.date(year, 1, 1)
    weekdays = np.array([(jan1 + datetime.timedelta(days=d)).weekday() for d in range(n_days)])

    # Average per weekday
    avg_week = np.zeros((7, ipd), dtype=float)
    for wd in range(7):
        mask = weekdays == wd
        if np.any(mask):
            avg_week[wd] = daily[mask].mean(axis=0)

    return avg_week.flatten().tolist()


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def collect_portfolio_report_data(
    config: PortfolioConfig,
    system_value_result: SystemValueResult,
    marginal_values: list[MarginalValuePoint],
    annual_results: list[PortfolioAnnualResult] | None = None,
    avg_week_loads: list[dict] | None = None,
    avg_week_generation: list[dict] | None = None,
    avg_week_prices: list[dict] | None = None,
) -> PortfolioReportData:
    """Aggregate all portfolio simulation results into report data.

    Parameters
    ----------
    config:
        Validated portfolio configuration.
    system_value_result:
        Complete enumeration result with World A costs and all points.
    marginal_values:
        Marginal value points.
    annual_results:
        Annual results from a representative simulation (for dispatch summary).
    avg_week_loads:
        Average week profiles per load group (from ``compute_average_week``).
    avg_week_generation:
        Average week profiles per generation unit.
    avg_week_prices:
        Average week profiles per price scenario.

    Returns
    -------
    PortfolioReportData
        Fully populated report data.
    """
    meta = config.meta

    dispatch_summary = None
    dispatch_year = None
    if annual_results:
        dispatch_summary = _extract_dispatch_summary(annual_results, year_index=0)
        dispatch_year = meta.baseline_year

    return PortfolioReportData(
        # Meta
        scenario_name=meta.name,
        creation_date=date.today().strftime("%d.%m.%Y"),
        baseline_year=meta.baseline_year,
        lifetime_years=meta.project_lifetime_years,
        model_version=REPORT_MODEL_VERSION,
        perfect_foresight_discount=meta.perfect_foresight_discount,
        # Portfolio overview
        generation=_extract_generation(config.generation),
        load_groups=_extract_load_groups(config.load),
        flexibilities=_extract_flexibilities(config.flexibilities),
        # World A
        world_a_annual_costs=system_value_result.world_a_annual_costs,
        world_a_total_cost=sum(system_value_result.world_a_annual_costs),
        # World A – average week profiles
        world_a_avg_week_loads=avg_week_loads or [],
        world_a_avg_week_generation=avg_week_generation or [],
        world_a_avg_week_prices=avg_week_prices or [],
        # System value
        system_value_points=_extract_system_value_points(system_value_result),
        # Marginal value
        marginal_value_points=_extract_marginal_value_points(marginal_values),
        # Dispatch sample
        dispatch_sample_year=dispatch_year,
        dispatch_sample_summary=dispatch_summary,
    )
