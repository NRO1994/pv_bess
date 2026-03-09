"""Report data aggregator for the interactive HTML dashboard.

Collects all simulation results into a single ``HtmlReportData`` dataclass
that serves as the data source for the HTML report template.

Public API
----------
HtmlReportData          -- Dataclass holding all report data.
collect_report_data     -- Factory function to aggregate results.
"""

from __future__ import annotations

import base64
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from pv_bess_model.config.defaults import (
    INTERVALS_PER_HOUR,
    INTERVALS_PER_YEAR,
    MWH_TO_KWH,
    PRICE_DATA_ORIGIN,
    REPORT_MODEL_VERSION,
)
from pv_bess_model.config.loader import PriceWeatherScenario, ScenarioConfig
from pv_bess_model.finance.cashflow import CashflowProjection
from pv_bess_model.finance.metrics import FinancialMetrics
from pv_bess_model.optimization.grid_search import GridPointResult, GridSearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logo helpers
# ---------------------------------------------------------------------------

_TOOL_LOGO_PATH = Path(".data/tool_logo.png")
_COMPANY_LOGO_PATH = Path(".data/logo_stadtwerke_luebeck.png")


def _encode_logo_b64(path: Path) -> str | None:
    """Read a PNG file and return its Base64-encoded data URI, or ``None``."""
    try:
        abs_path = path if path.is_absolute() else Path.cwd() / path
        if not abs_path.exists():
            logger.debug("Logo file not found: %s", abs_path)
            return None
        raw = abs_path.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except OSError:
        logger.warning("Failed to read logo file: %s", path, exc_info=True)
        return None


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
class HtmlReportData:
    """All data required to render the interactive HTML report.

    Fields are grouped by the tab they are displayed in.
    """

    # -- Meta --
    scenario_name: str
    scenario_json_filename: str
    creation_date: str  # DD.MM.YYYY
    commissioning_year: int
    model_version: str

    # -- Input parameters (Tab 1) --
    pv_peak_kwp: float
    pv_azimuth: float
    pv_tilt: float
    pv_degradation_pct: float
    bess_scale_range: list[float]
    bess_ep_ratios: list[float]
    bess_rte_pct: float
    grid_max_export_kw: float
    operating_mode: str
    marketing_type: str
    marketing_params: dict
    latitude: float
    longitude: float
    lifetime_years: int
    leverage_pct: float
    interest_rate_pct: float
    loan_tenor_years: int
    inflation_rate: float

    # -- Timeseries data (Tab 2) --
    pv_monthly_by_year: dict[int, list[float]]  # {weather_year: [jan..dec in GWh]}
    pv_production_model: str
    price_scenario_annual_means: list[dict]  # [{name, weather_year, means: [y1..yn]}]
    price_origin: str

    # -- Sensitivity results (Tab 4-6, optional) --
    eeg_sensitivity: list[dict] | None
    ppa_collar: list[dict] | None
    ppa_collar_duration: int
    ppa_baseload: list[dict] | None
    ppa_baseload_duration: int

    # -- Grid search (Tab 3) --
    grid_search_points: list[dict]
    optimal_scale_pct: float
    optimal_ep_ratio: float
    optimal_bess_power_kw: float
    optimal_bess_capacity_kwh: float

    # -- Cashflow (Tab 7) --
    cashflow_years: list[dict]

    # -- KPIs (Tab 7) --
    metrics: dict

    # -- Logos (Base64 data URIs) --
    tool_logo_b64: str | None
    company_logo_b64: str | None

    # -- LLM texts (populated after manual Copilot step) --
    llm_texts: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialise to a compact JSON string (NaN/Inf → null)."""
        raw = asdict(self)
        clean = _sanitize_for_json(raw)
        return json.dumps(clean, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# PV monthly aggregation
# ---------------------------------------------------------------------------


def _compute_pv_monthly_gwh(
    weather_timeseries: dict[int, np.ndarray],
) -> dict[int, list[float]]:
    """Aggregate per-weather-year timeseries into monthly GWh values.

    Parameters
    ----------
    weather_timeseries:
        ``{weather_year: array}`` where arrays have 8 760 (hourly) or
        35 040 (15-min) elements in kWh (per interval).

    Returns
    -------
    dict[int, list[float]]
        ``{year: [jan_gwh, feb_gwh, ..., dec_gwh]}``.
    """
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    result: dict[int, list[float]] = {}

    for year, ts in sorted(weather_timeseries.items()):
        n = len(ts)
        intervals_per_hour = 4 if n >= INTERVALS_PER_YEAR else 1

        monthly: list[float] = []
        offset = 0
        for days in days_per_month:
            intervals = days * 24 * intervals_per_hour
            chunk = ts[offset : offset + intervals]
            kwh = float(np.sum(chunk)) / intervals_per_hour
            monthly.append(kwh / 1e6)  # GWh
            offset += intervals
        result[year] = monthly

    return result


# ---------------------------------------------------------------------------
# Price scenario annual means
# ---------------------------------------------------------------------------


def _compute_price_annual_means(
    scenario_prices: list[PriceWeatherScenario],
) -> list[dict]:
    """Compute annual mean prices (EUR/MWh) per scenario over the project lifetime.

    Parameters
    ----------
    scenario_prices:
        List of ``PriceWeatherScenario`` with ``price_per_year`` populated.

    Returns
    -------
    list[dict]
        ``[{name, weather_year, means: [mean_y1, mean_y2, ...]}]``
        where means are in EUR/MWh.
    """
    result: list[dict] = []
    for sc in scenario_prices:
        if sc.price_per_year is None:
            continue
        yearly_means: list[float] = []
        for year_arr in sc.price_per_year:
            mean_kwh = float(np.mean(year_arr))
            yearly_means.append(mean_kwh * MWH_TO_KWH)  # EUR/MWh
        result.append({
            "name": sc.label,
            "weather_year": sc.weather_year,
            "means": yearly_means,
        })
    return result


# ---------------------------------------------------------------------------
# Grid search data
# ---------------------------------------------------------------------------


def _extract_grid_search_points(grid_result: GridSearchResult) -> list[dict]:
    """Convert grid search results into a list of dicts for the report."""
    points: list[dict] = []
    for pt in grid_result.points:
        irr = None
        if pt.metrics is not None and pt.metrics.equity_irr is not None:
            irr = pt.metrics.equity_irr * 100.0
        points.append({
            "scale_pct": pt.scale_pct,
            "ep_ratio": pt.e_to_p_ratio,
            "bess_power_kw": pt.bess_power_kw,
            "bess_capacity_kwh": pt.bess_capacity_kwh,
            "capex_total": pt.capex_total,
            "opex_base": pt.opex_base,
            "revenue_year1": pt.revenue_year1,
            "equity_irr": irr,
            "is_optimal": pt.is_optimal,
        })
    return points


# ---------------------------------------------------------------------------
# Cashflow data
# ---------------------------------------------------------------------------


def _extract_cashflow_years(
    opt: GridPointResult,
    commissioning_year: int,
) -> list[dict]:
    """Build per-year cashflow dicts for the stacked bar chart.

    Revenue items are positive, cost items are negative.
    """
    if opt.cashflow is None:
        return []

    # Get revenue breakdown from annual dispatch results
    annual_results = []
    if opt.run_result is not None:
        annual_results = opt.run_result.annual_results

    years: list[dict] = []
    for i, cf in enumerate(opt.cashflow.years):
        year_data: dict[str, Any] = {
            "year": commissioning_year + i,
            # Revenue (positive)
            "revenue_total": cf.revenue,
            # Costs (negative)
            "capex": -abs(cf.capex) if cf.capex != 0 else 0,
            "opex": -abs(cf.opex),
            "debt_service": -abs(cf.debt_service),
            "tax_total": -abs(cf.total_tax),
            "grid_import_cost": -abs(cf.grid_import_costs) if cf.grid_import_costs else 0,
            "baseload_matching_cost": -abs(cf.baseload_matching_costs) if cf.baseload_matching_costs else 0,
            # Derived
            "equity_cf": cf.equity_cf,
            "depreciation": -abs(cf.depreciation),
        }

        # Revenue breakdown from dispatch results
        if i < len(annual_results):
            ar = annual_results[i]
            year_data["revenue_pv"] = ar.revenue_pv_export
            year_data["revenue_bess_green"] = ar.revenue_bess_green
            year_data["revenue_bess_grey"] = ar.revenue_bess_grey
        else:
            year_data["revenue_pv"] = cf.revenue
            year_data["revenue_bess_green"] = 0.0
            year_data["revenue_bess_grey"] = 0.0

        years.append(year_data)

    # Add cumulative equity CF
    cumulative = 0.0
    for y in years:
        cumulative += y["equity_cf"]
        y["cumulative_equity_cf"] = cumulative

    return years


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _extract_metrics(metrics: FinancialMetrics) -> dict:
    """Convert ``FinancialMetrics`` to a plain dict."""
    return {
        "equity_irr": (metrics.equity_irr * 100.0) if metrics.equity_irr is not None else None,
        "project_irr": (metrics.project_irr * 100.0) if metrics.project_irr is not None else None,
        "npv": metrics.npv,
        "dscr_min": metrics.dscr_min,
        "dscr_avg": metrics.dscr_avg,
        "lcoe": (metrics.lcoe * 100.0) if metrics.lcoe is not None else None,  # ct/kWh
        "payback_year": metrics.payback_year,
    }


# ---------------------------------------------------------------------------
# Sensitivity helpers
# ---------------------------------------------------------------------------


def _extract_sensitivity(result: Any) -> list[dict] | None:
    """Extract sensitivity analysis points into a list of dicts.

    Parameters
    ----------
    result:
        A ``SensitivityResult`` or ``None``.

    Returns
    -------
    list[dict] | None
        Each dict contains ``params`` and summary statistics, or ``None``
        if no result available.
    """
    if result is None:
        return None

    points: list[dict] = []
    for ap in result.points:
        entry: dict[str, Any] = {**ap.params}

        # Extract equity IRR stats from MC result
        eq_stats = ap.mc_result.overall_stats.get("equity_irr")
        if eq_stats is not None:
            entry["irr_mean"] = eq_stats.mean * 100.0
            entry["irr_median"] = eq_stats.median * 100.0
            entry["irr_std"] = eq_stats.std * 100.0
            entry["irr_p10"] = eq_stats.p10 * 100.0
            entry["irr_p90"] = eq_stats.p90 * 100.0
        points.append(entry)

    return points if points else None


# ---------------------------------------------------------------------------
# Marketing params
# ---------------------------------------------------------------------------


def _extract_marketing_params(scenario: ScenarioConfig) -> dict:
    """Extract marketing/revenue-stream parameters from the scenario."""
    marketing = scenario.finance.get("revenue_streams", {}).get("marketing", {})
    ppa = scenario.finance.get("revenue_streams", {}).get("ppa", {})

    params: dict[str, Any] = {}

    mtype = marketing.get("type", "none")
    if mtype == "eeg":
        params["floor_price_ct_kwh"] = marketing.get("floor_price_eur_per_kwh", 0) * 100
        params["fixed_price_years"] = marketing.get("fixed_price_years", 0)
        params["eeg_inflation"] = marketing.get("eeg_inflation", False)

    ppa_type = ppa.get("type", "none")
    if ppa_type != "none":
        params["ppa_type"] = ppa_type
        params["ppa_duration_years"] = ppa.get("duration_years", 0)
        params["inflation_on_ppa"] = ppa.get("inflation_on_ppa", False)
        params["goo_premium_ct_kwh"] = ppa.get("guarantee_of_origin_eur_per_kwh", 0) * 100

        if ppa_type == "ppa_floor":
            params["floor_price_ct_kwh"] = ppa.get("floor_price_eur_per_kwh", 0) * 100
        elif ppa_type == "ppa_collar":
            params["floor_price_ct_kwh"] = ppa.get("floor_price_eur_per_kwh", 0) * 100
            params["cap_price_ct_kwh"] = ppa.get("cap_price_eur_per_kwh", 0) * 100
        elif ppa_type == "ppa_baseload":
            params["baseload_mw"] = ppa.get("baseload_mw", 0)
            params["ppa_price_ct_kwh"] = ppa.get("pay_as_produced_price_eur_per_kwh", 0) * 100
        elif ppa_type == "ppa_pay_as_produced":
            params["ppa_price_ct_kwh"] = ppa.get("pay_as_produced_price_eur_per_kwh", 0) * 100

    return params


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def collect_report_data(
    scenario: ScenarioConfig,
    grid_result: GridSearchResult,
    opt: GridPointResult,
    metrics: FinancialMetrics,
    weather_data_for_report: dict[int, np.ndarray] | None,
    scenario_prices: list[PriceWeatherScenario],
    commissioning_year: int,
    eeg_sens_result: Any | None = None,
    collar_result: Any | None = None,
    baseload_result: Any | None = None,
    analyses:dict[str, Any] | None = None,
) -> HtmlReportData:
    """Aggregate all simulation results into an ``HtmlReportData`` instance.

    Parameters
    ----------
    scenario:
        Validated scenario configuration.
    grid_result:
        Complete grid search result.
    opt:
        Optimal grid point.
    metrics:
        Financial metrics for the optimal point.
    weather_data_for_report:
        ``{weather_year: pv_timeseries_array}`` or ``None``.
    scenario_prices:
        List of ``PriceWeatherScenario`` instances with prices loaded.
    commissioning_year:
        Project commissioning year.
    eeg_sens_result:
        EEG sensitivity result or ``None``.
    collar_result:
        PPA Collar sensitivity result or ``None``.
    baseload_result:
        PPA Baseload sensitivity result or ``None``.

    Returns
    -------
    HtmlReportData
        Fully populated report data (except ``llm_texts``).
    """
    pv = scenario.pv
    bess = scenario.bess
    finance = scenario.finance
    location = scenario.project_settings.get("location", {})

    # Marketing type
    marketing = finance.get("revenue_streams", {}).get("marketing", {})
    ppa = finance.get("revenue_streams", {}).get("ppa", {})
    marketing_type = marketing.get("type", "none")
    if ppa.get("type", "none") != "none":
        marketing_type = ppa["type"]

    # PV monthly data
    pv_monthly = {}
    if weather_data_for_report:
        pv_monthly = _compute_pv_monthly_gwh(weather_data_for_report)

    # Scenario JSON filename
    json_filename = ""
    if scenario.path is not None:
        json_filename = scenario.path.name

    return HtmlReportData(
        # Meta
        scenario_name=scenario.name,
        scenario_json_filename=json_filename,
        creation_date=date.today().strftime("%d.%m.%Y"),
        commissioning_year=commissioning_year,
        model_version=REPORT_MODEL_VERSION,
        # Input parameters
        pv_peak_kwp=scenario.pv_peak_kwp,
        pv_azimuth=float(pv["design"].get("azimuth_deg", 0)),
        pv_tilt=float(pv["design"].get("tilt_deg", 0)),
        pv_degradation_pct=float(pv["performance"].get("degradation_rate_pct_per_year", 0)),
        bess_scale_range=scenario.bess_scale_pct_list,
        bess_ep_ratios=scenario.e_to_p_ratio_hours_list,
        bess_rte_pct=float(bess["performance"].get("round_trip_efficiency_pct", 0)),
        grid_max_export_kw=float(scenario.grid_connection.get("max_export_kw", 0)),
        operating_mode=scenario.operating_mode,
        marketing_type=marketing_type,
        marketing_params=_extract_marketing_params(scenario),
        latitude=float(location.get("latitude", 0)),
        longitude=float(location.get("longitude", 0)),
        lifetime_years=scenario.lifetime_years,
        leverage_pct=float(finance.get("leverage_pct", 0)),
        interest_rate_pct=float(finance.get("interest_rate_pct", 0)),
        loan_tenor_years=int(finance.get("loan_tenor_years", 0)),
        inflation_rate=float(finance.get("inflation_rate", 0)),
        # Timeseries
        pv_monthly_by_year=pv_monthly,
        pv_production_model=location.get("pvgis_database", ""),
        price_scenario_annual_means=_compute_price_annual_means(scenario_prices),
        price_origin=PRICE_DATA_ORIGIN,
        # Sensitivity
        eeg_sensitivity=_extract_sensitivity(eeg_sens_result),
        ppa_collar=_extract_sensitivity(collar_result),
        ppa_collar_duration=analyses.get("ppa_collar", {}).get("duration_years", 0),
        ppa_baseload=_extract_sensitivity(baseload_result),
        ppa_baseload_duration=analyses.get("ppa_baseload", {}).get("duration_years", 0),
        # Grid search
        grid_search_points=_extract_grid_search_points(grid_result),
        optimal_scale_pct=opt.scale_pct,
        optimal_ep_ratio=opt.e_to_p_ratio,
        optimal_bess_power_kw=opt.bess_power_kw,
        optimal_bess_capacity_kwh=opt.bess_capacity_kwh,
        # Cashflow
        cashflow_years=_extract_cashflow_years(opt, commissioning_year),
        # KPIs
        metrics=_extract_metrics(metrics),
        # Logos
        tool_logo_b64=_encode_logo_b64(_TOOL_LOGO_PATH),
        company_logo_b64=_encode_logo_b64(_COMPANY_LOGO_PATH),
    )
