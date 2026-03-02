"""CLI entrypoint and orchestrator for the PV + BESS co-location financial model.

Execution flow
--------------
1.  Load & validate scenario JSON.
2.  Fetch PVGIS data (or load from cache).
3.  Compute P50 and P90 hourly PV timeseries.
4.  Load price CSV timeseries, extend to project lifetime.
5.  Grid search: ratio-based BESS sizing sweep.
6.  Monte Carlo (on optimum, if enabled).
7.  Write output CSVs.
8.  Print summary to stdout.

Usage
-----
    python -m pv_bess_model.main --scenario scenarios/my_scenario.json
    python -m pv_bess_model.main --scenario my.json --no-mc
    python -m pv_bess_model.main --scenario my.json --dry-run
    python -m pv_bess_model.main --scenario my.json -v
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from pv_bess_model.config.defaults import (
    CSV_DECIMAL_SEPARATOR,
    CSV_DELIMITER,
    CSV_INPUT_DECIMAL_SEPARATOR,
    CSV_TIMESTAMP_COLUMN,
    CSV_TIMESTAMP_FORMAT,
    DEFAULT_AFA_YEARS_BESS,
    DEFAULT_AFA_YEARS_PV,
    DEFAULT_BESS_AVAILABILITY_PCT,
    DEFAULT_BESS_DEGRADATION_RATE_PCT,
    DEFAULT_BESS_REPLACEMENT_CAPACITY_FACTOR_PCT,
    DEFAULT_BESS_MAX_SOC_PCT,
    DEFAULT_BESS_MIN_SOC_PCT,
    DEFAULT_BESS_RTE_PCT,
    DEFAULT_DEBT_SIZING_DOWNSIDE_PCT,
    DEFAULT_DISCOUNT_RATE,
    DEFAULT_GEWERBESTEUER_HEBESATZ,
    DEFAULT_GEWERBESTEUER_MESSZAHL,
    DEFAULT_KOERPERSCHAFTSTEUER_PCT,
    DEFAULT_INFLATION_RATE,
    DEFAULT_INTEREST_RATE_PCT,
    DEFAULT_LEVERAGE_PCT,
    DEFAULT_LIFETIME_YEARS,
    DEFAULT_LOAN_TENOR_YEARS,
    DEFAULT_MC_ITERATIONS,
    DEFAULT_MC_SIGMA_BESS_AVAILABILITY_PCT,
    DEFAULT_MC_SIGMA_CAPEX_BESS_PCT,
    DEFAULT_MC_SIGMA_CAPEX_PV_PCT,
    DEFAULT_MC_SIGMA_OPEX_BESS_PCT,
    DEFAULT_MC_SIGMA_OPEX_PV_PCT,
    DEFAULT_MC_SIGMA_PV_AVAILABILITY_PCT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PV_DEGRADATION_RATE_PCT,
    DEFAULT_PV_AVAILABILITY_PCT,
    DEFAULT_SKIP_BASELINE,
    DEFAULT_SOLIDARITAETSZUSCHLAG_PCT,
    HOURS_PER_DAY,
    HOURS_PER_YEAR,
    INTERVALS_PER_DAY,
    INTERVALS_PER_HOUR,
    INTERVALS_PER_YEAR,
    MARKETING_TYPE_EEG,
    PPA_TYPE_COLLAR,
    PPA_TYPE_FLOOR,
    PPA_TYPE_NONE,
    PPA_TYPE_PAY_AS_PRODUCED,
    TIMESTEP_HOURS,
)
from pv_bess_model.config.loader import (
    PriceData,
    PriceWeatherScenario,
    ScenarioConfig,
    extend_price_timeseries,
    load_price_csv,
    load_scenario,
    parse_scenarios,
)
from pv_bess_model.dispatch.engine import run_simulation
from pv_bess_model.finance.cashflow import build_cashflow_projection
from pv_bess_model.finance.costs import calculate_total_costs
from pv_bess_model.finance.debt import build_annuity_schedule
from pv_bess_model.finance.inflation import inflate_value
from pv_bess_model.finance.metrics import compute_all_metrics
from pv_bess_model.market.eeg import EegConfig, eeg_config_from_dict, effective_eeg_price
from pv_bess_model.market.ppa import (
    PpaConfig,
    pay_as_produced_price,
    ppa_config_from_dict,
)
from pv_bess_model.optimization.analyses import (
    run_eeg_sensitivity,
    run_ppa_baseload_analysis,
    run_ppa_collar_analysis,
)
from pv_bess_model.optimization.grid_search import GridSearchConfig, run_grid_search
from pv_bess_model.optimization.monte_carlo import MCParams, run_monte_carlo
from pv_bess_model.output.csv_writer import (
    CsvConfig,
    write_cashflows_csv,
    write_dispatch_sample_csv,
    write_eeg_sensitivity_csv,
    write_grid_search_csv,
    write_monte_carlo_csv,
    write_ppa_baseload_csv,
    write_ppa_collar_csv,
    write_summary_csv,
)
from pv_bess_model.pv.pvgis_client import PVGISClient
from pv_bess_model.pv.timeseries import (
    align_weather_to_forecast_year,
    compute_p50_p90,
    hourly_to_quarter_hourly,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="python -m pv_bess_model.main",
        description="PV + BESS Co-Location Financial Model",
    )
    p.add_argument(
        "--scenario",
        required=True,
        metavar="PATH",
        help="Path to scenario JSON file.",
    )
    p.add_argument(
        "--output",
        metavar="DIR",
        default=None,
        help="Output directory (overrides scenario JSON setting).",
    )
    p.add_argument(
        "--no-mc",
        action="store_true",
        default=False,
        help="Skip Monte Carlo simulation even if enabled in JSON.",
    )
    p.add_argument(
        "--bess-power",
        type=float,
        default=None,
        metavar="KW",
        help="Fixed BESS power in kW (bypasses grid search).",
    )
    p.add_argument(
        "--bess-capacity",
        type=float,
        default=None,
        metavar="KWH",
        help="Fixed BESS capacity in kWh (bypasses grid search).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG logging.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate JSON and inputs, then exit without running simulation.",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        default=False,
        help="Generate report without LLM-generated texts (placeholders only).",
    )
    p.add_argument(
        "--no-report",
        action="store_true",
        default=False,
        help="Skip PDF report generation entirely.",
    )
    return p


# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------


def _build_fixed_prices_yearly(
    scenario: ScenarioConfig,
    inflation_rate: float,
) -> list[float]:
    """Build the per-year floor/fixed price list for the dispatch engine.

    Returns 0.0 for each year when no floor is active (pure market).

    Parameters
    ----------
    scenario:
        Validated scenario configuration.
    inflation_rate:
        Annual inflation rate as a fraction.

    Returns
    -------
    list[float]
        Floor price in €/kWh for each project year (length = lifetime_years).
        Index 0 = year 1.
    """
    lifetime = scenario.lifetime_years
    marketing = scenario.finance.get("revenue_streams", {}).get("marketing", {})
    ppa_dict = scenario.finance.get("revenue_streams", {}).get("ppa", {})

    marketing_type = marketing.get("type", "none")
    ppa_type = ppa_dict.get("type", PPA_TYPE_NONE)

    fixed_prices: list[float] = []

    for year in range(1, lifetime + 1):
        price = 0.0

        if marketing_type == "eeg":
            eeg_cfg = eeg_config_from_dict(marketing)
            price = effective_eeg_price(eeg_cfg, year, inflation_rate)

        elif ppa_type == PPA_TYPE_FLOOR:
            ppa_cfg = ppa_config_from_dict(ppa_dict)
            if year <= ppa_cfg.duration_years:
                base = ppa_cfg.floor_price_eur_per_kwh or 0.0
                if ppa_cfg.inflation_enabled:
                    price = inflate_value(base, inflation_rate, year)
                else:
                    price = base
                # GoO premium is NOT added here; it is passed separately via
                # _build_goo_prices_yearly() and added after the floor comparison.

        elif ppa_type == PPA_TYPE_COLLAR:
            ppa_cfg = ppa_config_from_dict(ppa_dict)
            if year <= ppa_cfg.duration_years:
                base = ppa_cfg.floor_price_eur_per_kwh or 0.0
                if ppa_cfg.inflation_enabled:
                    price = inflate_value(base, inflation_rate, year)
                else:
                    price = base
                # GoO premium is NOT added here; it is passed separately via
                # _build_goo_prices_yearly() and added after the clip operation.
                # Cap price is handled via _build_cap_prices_yearly().

        elif ppa_type == PPA_TYPE_PAY_AS_PRODUCED:
            ppa_cfg = ppa_config_from_dict(ppa_dict)
            price = pay_as_produced_price(ppa_cfg, year, inflation_rate)
            if year <= ppa_cfg.duration_years:
                price += ppa_cfg.goo_premium_eur_per_kwh

        fixed_prices.append(price)

    return fixed_prices


def _build_goo_prices_yearly(
    scenario: ScenarioConfig,
) -> list[float]:
    """Build the per-year GoO premium list for the dispatch engine.

    Returns the Guarantee-of-Origin (GoO) premium in €/kWh for each project
    year.  The premium is non-zero only during the active PPA period for floor
    and collar PPA types.  Pay-as-produced GoO is already baked into the
    ``fixed_prices_yearly`` value and should not be doubled here.

    Parameters
    ----------
    scenario:
        Validated scenario configuration.

    Returns
    -------
    list[float]
        GoO premium in €/kWh for each project year (length = lifetime_years).
        Index 0 = year 1.  0.0 for years outside the PPA period.
    """
    lifetime = scenario.lifetime_years
    ppa_dict = scenario.finance.get("revenue_streams", {}).get("ppa", {})
    ppa_type = ppa_dict.get("type", PPA_TYPE_NONE)

    if ppa_type in (PPA_TYPE_FLOOR, PPA_TYPE_COLLAR):
        ppa_cfg = ppa_config_from_dict(ppa_dict)
        goo = ppa_cfg.goo_premium_eur_per_kwh
        duration = ppa_cfg.duration_years
        return [goo if year <= duration else 0.0 for year in range(1, lifetime + 1)]

    return [0.0] * lifetime


def _build_cap_prices_yearly(
    scenario: ScenarioConfig,
    inflation_rate: float,
) -> list[float]:
    """Build the per-year cap price list for the dispatch engine.

    Only relevant for PPA Collar.  Returns 0.0 for each year when no cap is
    active (0.0 = no cap, i.e. unbounded upside).

    Parameters
    ----------
    scenario:
        Validated scenario configuration.
    inflation_rate:
        Annual inflation rate as a fraction.

    Returns
    -------
    list[float]
        Cap price in €/kWh for each project year (length = lifetime_years).
        Index 0 = year 1.  0.0 means no cap for that year.
    """
    lifetime = scenario.lifetime_years
    ppa_dict = scenario.finance.get("revenue_streams", {}).get("ppa", {})
    ppa_type = ppa_dict.get("type", PPA_TYPE_NONE)

    if ppa_type != PPA_TYPE_COLLAR:
        return [0.0] * lifetime

    ppa_cfg = ppa_config_from_dict(ppa_dict)
    cap_prices: list[float] = []
    for year in range(1, lifetime + 1):
        if year <= ppa_cfg.duration_years:
            base = ppa_cfg.cap_price_eur_per_kwh or 0.0
            if ppa_cfg.inflation_enabled:
                price = inflate_value(base, inflation_rate, year)
            else:
                price = base
        else:
            price = 0.0
        cap_prices.append(price)
    return cap_prices


def _build_spot_prices_yearly(
    price_array: np.ndarray,
    lifetime_years: int,
    inflation_rate: float,
    apply_inflation: bool,
    intervals_per_year: int = HOURS_PER_YEAR,
) -> list[np.ndarray]:
    """Split an extended price array into per-year slices with optional inflation.

    Parameters
    ----------
    price_array:
        Extended price array of length >= ``lifetime_years × intervals_per_year``
        (€/kWh).
    lifetime_years:
        Number of project years.
    inflation_rate:
        Annual inflation rate as a fraction.
    apply_inflation:
        Whether to scale each year's prices by the annual inflation factor.
    intervals_per_year:
        Number of intervals per year (8 760 for hourly, 35 040 for 15-min).

    Returns
    -------
    list[np.ndarray]
        One array of length ``intervals_per_year`` per project year.
    """
    yearly: list[np.ndarray] = []
    for y in range(1, lifetime_years + 1):
        start = (y - 1) * intervals_per_year
        end = y * intervals_per_year
        year_prices = price_array[start:end].copy()
        if apply_inflation:
            factor = inflate_value(1.0, inflation_rate, y)
            year_prices = year_prices * factor
        yearly.append(year_prices)
    return yearly


# ---------------------------------------------------------------------------
# Price extension helper
# ---------------------------------------------------------------------------


def _extend_all_price_columns(
    price_data: PriceData,
    required_columns: list[str],
    target_years: int,
    intervals_per_year: int = HOURS_PER_YEAR,
) -> dict[str, np.ndarray]:
    """Extend all required price columns to cover the full project lifetime.

    Each column in *required_columns* is extended using the repeat-last-year
    logic from :func:`extend_price_timeseries`.

    Parameters
    ----------
    price_data:
        Loaded price data with column arrays.
    required_columns:
        List of column names to extend.
    target_years:
        Project lifetime in years.
    intervals_per_year:
        Number of intervals per year for extension (8 760 for hourly,
        35 040 for 15-min).

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of column name → extended price array (length =
        ``target_years × intervals_per_year``).
    """
    extended: dict[str, np.ndarray] = {}
    for col in required_columns:
        extended[col] = extend_price_timeseries(
            price_data.get_column(col),
            target_years=target_years,
            hours_per_year=intervals_per_year,
        )
    return extended


# ---------------------------------------------------------------------------
# Cost helpers (extract from scenario JSON)
# ---------------------------------------------------------------------------


def _extract_cost_dicts(scenario: ScenarioConfig) -> tuple[dict, dict, dict, dict]:
    """Extract CAPEX and OPEX config dicts from the scenario JSON.

    Returns
    -------
    tuple
        ``(pv_capex, pv_opex, bess_capex, bess_opex)`` – raw cost config dicts.
    """
    pv = scenario.pv
    bess = scenario.bess
    pv_capex = pv.get("costs", {}).get("capex", {})
    pv_opex = pv.get("costs", {}).get("opex", {})
    bess_capex = bess.get("costs", {}).get("capex", {})
    bess_opex = bess.get("costs", {}).get("opex", {})
    return pv_capex, pv_opex, bess_capex, bess_opex


def _extract_grid_cost_dicts(scenario: ScenarioConfig) -> tuple[dict, dict]:
    """Extract grid connection CAPEX and OPEX config dicts."""
    grid = scenario.grid_connection
    return grid.get("costs", {}).get("capex", {}), grid.get("costs", {}).get("opex", {})


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    """Execute the full scenario run.

    Parameters
    ----------
    args:
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """
    # ------------------------------------------------------------------
    # Step 1: Load & validate scenario JSON
    # ------------------------------------------------------------------
    logger.info("Loading scenario: %s", args.scenario)
    try:
        scenario = load_scenario(args.scenario)
    except (FileNotFoundError, ValueError, Exception) as exc:
        logger.error("Failed to load scenario: %s", exc)
        return 1

    if args.dry_run:
        print(f"Dry run: scenario '{scenario.name}' validated successfully.")
        return 0

    # Determine output directory (CLI > JSON > default)
    scenario_output_dir = scenario.raw.get("scenario", {}).get("output", {}).get("directory")
    if args.output:
        output_base = Path(args.output)
    elif scenario_output_dir:
        output_base = Path(scenario_output_dir)
    else:
        output_base = Path(DEFAULT_OUTPUT_DIR)
    output_dir = output_base / scenario.name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # CSV formatting settings (CLI > JSON > defaults)
    _out_block = scenario.raw.get("scenario", {}).get("output", {})
    csv_config = CsvConfig(
        delimiter=_out_block.get("csv_separator", CSV_DELIMITER),
        decimal=_out_block.get("csv_decimal", CSV_DECIMAL_SEPARATOR),
        timestamp_column=_out_block.get("csv_timestamp_column", CSV_TIMESTAMP_COLUMN),
        timestamp_format=_out_block.get("csv_timestamp_format", CSV_TIMESTAMP_FORMAT),
    )

    # Finance parameters
    finance = scenario.finance
    inflation_rate = float(finance.get("inflation_rate", DEFAULT_INFLATION_RATE))
    leverage_pct = float(finance.get("leverage_pct", 0.0))
    interest_rate_pct = float(finance.get("interest_rate_pct", DEFAULT_INTEREST_RATE_PCT))
    loan_tenor_years = int(finance.get("loan_tenor_years", DEFAULT_LOAN_TENOR_YEARS))
    discount_rate = float(scenario.project_settings.get("discount_rate", DEFAULT_DISCOUNT_RATE))
    debt_sizing_downside_pct = float(
        finance.get("debt_sizing_downside_pct", DEFAULT_DEBT_SIZING_DOWNSIDE_PCT)
    )

    tax = finance.get("tax", {})
    afa_years_pv = int(tax.get("afa_years_pv", DEFAULT_AFA_YEARS_PV))
    afa_years_bess = int(tax.get("afa_years_bess", DEFAULT_AFA_YEARS_BESS))
    gewerbesteuer_hebesatz = int(tax.get("gewerbesteuer_hebesatz", DEFAULT_GEWERBESTEUER_HEBESATZ))
    gewerbesteuer_messzahl = float(tax.get("gewerbesteuer_messzahl", DEFAULT_GEWERBESTEUER_MESSZAHL))
    koerperschaftsteuer_pct = float(tax.get("koerperschaftsteuer_pct", DEFAULT_KOERPERSCHAFTSTEUER_PCT))
    solidaritaetszuschlag_pct = float(tax.get("solidaritaetszuschlag_pct", DEFAULT_SOLIDARITAETSZUSCHLAG_PCT))

    # PV parameters
    pv = scenario.pv
    pv_design = pv["design"]
    pv_perf = pv.get("performance", {})
    pv_peak_kwp = float(pv_design["peak_power_kwp"])
    pv_degradation_rate = float(pv_perf.get("degradation_rate_pct_per_year", DEFAULT_PV_DEGRADATION_RATE_PCT)) / 100.0
    pv_availability_pct = float(pv_perf.get("pv_availability_pct", DEFAULT_PV_AVAILABILITY_PCT))

    # BESS parameters
    bess = scenario.bess
    bess_perf = bess.get("performance", {})
    bess_rte = float(bess_perf.get("round_trip_efficiency_pct", DEFAULT_BESS_RTE_PCT)) / 100.0
    bess_min_soc_pct = float(bess_perf.get("min_soc_pct", DEFAULT_BESS_MIN_SOC_PCT))
    bess_max_soc_pct = float(bess_perf.get("max_soc_pct", DEFAULT_BESS_MAX_SOC_PCT))
    bess_degradation_rate = float(bess_perf.get("degradation_rate_pct_per_year", DEFAULT_BESS_DEGRADATION_RATE_PCT)) / 100.0
    bess_availability_pct = float(bess_perf.get("bess_availability_pct", DEFAULT_BESS_AVAILABILITY_PCT))

    bess_costs = bess.get("costs", {})
    optimization_fee_pct = float(bess_costs.get("optimization_fee_pct", 0.0))
    replacement_cfg = bess_costs.get("replacement", {})
    replacement_enabled = bool(replacement_cfg.get("enabled", False))
    replacement_year = int(replacement_cfg.get("year", 0))
    replacement_fixed_eur = float(replacement_cfg.get("fixed_eur", 0.0))
    replacement_eur_per_kw = float(replacement_cfg.get("eur_per_kw", 0.0))
    replacement_eur_per_kwh = float(replacement_cfg.get("eur_per_kwh", 0.0))
    replacement_pct_of_capex = float(replacement_cfg.get("pct_of_capex", 0.0))
    replacement_capacity_factor_pct = float(
        replacement_cfg.get(
            "capacity_factor_pct", DEFAULT_BESS_REPLACEMENT_CAPACITY_FACTOR_PCT
        )
    )

    # Grid connection
    grid_connection = scenario.grid_connection
    grid_max_kw = float(grid_connection.get("max_export_kw", pv_peak_kwp))
    system_loss_pct = float(grid_connection.get("system_loss_pct", 0.0))
    grid_loss_factor = 1.0 - system_loss_pct / 100.0

    # BESS design space (for grid search)
    bess_design_space = bess.get("design_space", {})
    scale_pct_list = [float(v) for v in bess_design_space.get("scale_pct_of_pv", [0.0])]
    e_to_p_list = [float(v) for v in bess_design_space.get("e_to_p_ratio_hours", [2.0])]
    skip_baseline = bool(
        scenario.raw.get("scenario", {}).get("skip_baseline", DEFAULT_SKIP_BASELINE)
    )

    # Absolute BESS sizing for BESS-Only scenarios (pv_peak_kwp == 0)
    bess_absolute_power_kw: float | None = (
        float(bess_design_space["absolute_power_kw"])
        if "absolute_power_kw" in bess_design_space
        else None
    )
    bess_absolute_capacity_kwh: float | None = (
        float(bess_design_space["absolute_capacity_kwh"])
        if "absolute_capacity_kwh" in bess_design_space
        else None
    )

    if pv_peak_kwp == 0:
        logger.info(
            "pv_peak_kwp = 0: BESS-Only scenario detected. "
            "PVGIS fetch will be skipped; PV timeseries set to zero."
        )
        if scenario.operating_mode == "green":
            logger.warning(
                "BESS-Only with operating_mode='green': the BESS cannot be "
                "charged (no PV surplus). Revenue will be zero. "
                "Consider switching to operating_mode='grey'."
            )

    # Handle fixed BESS override from CLI
    if args.bess_power is not None and args.bess_capacity is not None:
        if pv_peak_kwp > 0:
            scale_pct_list = [args.bess_power / pv_peak_kwp * 100.0]
            e_to_p_list = [args.bess_capacity / args.bess_power]
            logger.info(
                "CLI override: BESS power=%.1f kW, capacity=%.1f kWh → "
                "scale=%.2f %%, E/P=%.2f h",
                args.bess_power,
                args.bess_capacity,
                scale_pct_list[0],
                e_to_p_list[0],
            )
        else:
            # BESS-Only: CLI override sets absolute sizing directly
            bess_absolute_power_kw = float(args.bess_power)
            bess_absolute_capacity_kwh = float(args.bess_capacity)
            scale_pct_list = [100.0]  # non-zero scale triggers absolute sizing
            e_to_p_list = [1.0]       # dummy (capacity is absolute)
            logger.info(
                "CLI override (BESS-Only): BESS power=%.1f kW, capacity=%.1f kWh",
                bess_absolute_power_kw,
                bess_absolute_capacity_kwh,
            )

    # ------------------------------------------------------------------
    # Step 2: Parse scenarios and fetch weather-year PV data
    # ------------------------------------------------------------------
    location = scenario.project_settings.get("location", {})
    latitude = float(location.get("latitude", 51.0))
    longitude = float(location.get("longitude", 10.0))
    pvgis_database = location.get("pvgis_database", "PVGIS-SARAH3")
    mounting_type = pv_design.get("mounting_type", "free")
    azimuth_deg = float(pv_design.get("azimuth_deg", 0))
    tilt_deg = float(pv_design.get("tilt_deg", 30))

    # Parse price-weather scenarios from the new schema
    price_inputs = finance.get("price_inputs", {})
    try:
        scenarios_list: list[PriceWeatherScenario] = parse_scenarios(scenario)
    except ValueError:
        scenarios_list = []

    # Determine resolution: use 15min if scenarios are defined
    ts_intervals_per_year = INTERVALS_PER_YEAR
    ts_intervals_per_day = INTERVALS_PER_DAY
    ts_timestep_hours = TIMESTEP_HOURS


    commissioning_year = scenario.commissioning_year

    # Weather data for report charts (populated in PV flows below)
    weather_data_for_report: dict[int, np.ndarray] | None = None

    if pv_peak_kwp > 0 and scenarios_list:
        # New flow: fetch unique weather years, align, convert to 15min
        unique_weather_years = sorted({s.weather_year for s in scenarios_list})
        logger.info(
            "Fetching PVGIS data for %d unique weather year(s): %s",
            len(unique_weather_years),
            unique_weather_years,
        )
        client = PVGISClient()
        weather_year_hourly: dict[int, np.ndarray] = {}
        for wy in unique_weather_years:
            try:
                hourly_ts = client.fetch_single_year(
                    year=wy,
                    latitude=latitude,
                    longitude=longitude,
                    peak_power_kwp=pv_peak_kwp,
                    mounting_type=mounting_type,
                    azimuth_deg=azimuth_deg,
                    tilt_deg=tilt_deg,
                    pvgis_database=pvgis_database,
                )
            except Exception as exc:
                logger.error("PVGIS fetch for year %d failed: %s", wy, exc)
                return 1
            weather_year_hourly[wy] = hourly_ts

        weather_data_for_report = weather_year_hourly

        # Align and convert to 15min per weather year
        weather_year_15min: dict[int, np.ndarray] = {}
        for wy, hourly_ts in weather_year_hourly.items():
            weather_year_15min[wy] = hourly_to_quarter_hourly(hourly_ts)

        # Assign PV timeseries to each scenario
        for sc in scenarios_list:
            sc.pv_timeseries_15min = weather_year_15min[sc.weather_year]

        # Central scenario PV timeseries for grid search
        central_scenarios = [s for s in scenarios_list if s.is_central]
        if central_scenarios:
            central_scenario = central_scenarios[0]
            central_pv_timeseries = central_scenario.pv_timeseries_15min
            central_pv_timeseries_year = central_scenario.weather_year
        else:
            central_pv_timeseries = scenarios_list[0].pv_timeseries_15min
            central_pv_timeseries_year = scenarios_list[0].weather_year

        logger.info(
            "PV timeseries (15min, central scenario): annual=%.0f kWh",
            float(np.sum(central_pv_timeseries)),
        )

    else:
        # BESS-Only: zero PV production
        central_pv_timeseries = np.zeros(ts_intervals_per_year, dtype=float)
        logger.info(
            "BESS-Only: PV timeseries set to zero (%d × 0 kWh).",
            ts_intervals_per_year,
        )

    # ------------------------------------------------------------------
    # Step 4: Load price CSV(s), extend to lifetime
    # ------------------------------------------------------------------
    mc_cfg = scenario.monte_carlo
    mc_enabled = scenario.mc_enabled and not args.no_mc

    # Resolve relative CSV path against scenario file directory
    scenario_dir = scenario.path.parent if scenario.path else Path(".")

    lifetime = scenario.lifetime_years

    for sc in scenarios_list:
        try:
            price_data = load_price_csv(
                path=sc.price_csv,
                required_columns=[sc.csv_column],
                price_unit=sc.price_unit,
                commissioning_year=commissioning_year,
                delimiter=sc.csv_separator,
                decimal=sc.csv_decimal,
                timestamp_column=sc.csv_timestamp_column,
                timestamp_format=sc.csv_timestamp_format,
            )
        except Exception as exc:
            logger.error("Price CSV load failed: %s", exc)
            return 1

        # Extend prices to lifetime (hourly first, then replicate to 15min)
        extended_prices_hourly = _extend_all_price_columns(
            price_data, [sc.csv_column], lifetime, HOURS_PER_YEAR
        )

        # Replicate hourly prices to 15min: each hour repeats 4x
        # NOTE: prices are NOT divided by 4 (price per kWh stays the same)
        extended_prices_15min: dict[str, np.ndarray] = {}
        for col, arr in extended_prices_hourly.items():
            extended_prices_15min[col] = np.repeat(arr, INTERVALS_PER_HOUR)

        spot_prices_yearly = _build_spot_prices_yearly(
            extended_prices_15min[sc.csv_column],
            lifetime,
            inflation_rate,
            sc.inflation_on_input_data,
            intervals_per_year=ts_intervals_per_year,
        )

        sc.price_per_year = spot_prices_yearly

    # Fixed prices per year (EEG / PPA floor, WITHOUT GoO)
    fixed_prices_yearly = _build_fixed_prices_yearly(scenario, inflation_rate)
    # GoO premium per year (for floor and collar PPA types)
    goo_prices_yearly = _build_goo_prices_yearly(scenario)
    # Cap prices per year (PPA Collar only; 0.0 = no cap)
    cap_prices_yearly = _build_cap_prices_yearly(scenario, inflation_rate)

    # ------------------------------------------------------------------
    # Step 5: Grid Search
    # ------------------------------------------------------------------
    pv_capex_cfg, pv_opex_cfg, bess_capex_cfg, bess_opex_cfg = _extract_cost_dicts(scenario)
    grid_capex_cfg, grid_opex_cfg = _extract_grid_cost_dicts(scenario)

    grid_search_config = GridSearchConfig(
        scale_pct_of_pv=scale_pct_list,
        e_to_p_ratio_hours=e_to_p_list,
        pv_peak_kwp=pv_peak_kwp,
        pv_base_timeseries=central_pv_timeseries,
        pv_base_timeseries_year=central_pv_timeseries_year,
        pv_degradation_rate=pv_degradation_rate,
        pv_costs_capex=pv_capex_cfg,
        pv_costs_opex=pv_opex_cfg,
        pv_availability_pct=pv_availability_pct,
        bess_rte=bess_rte,
        bess_min_soc_pct=bess_min_soc_pct,
        bess_max_soc_pct=bess_max_soc_pct,
        bess_degradation_rate=bess_degradation_rate,
        bess_availability_pct=bess_availability_pct,
        bess_costs_capex=bess_capex_cfg,
        bess_costs_opex=bess_opex_cfg,
        replacement_enabled=replacement_enabled,
        replacement_year=replacement_year,
        replacement_fixed_eur=replacement_fixed_eur,
        replacement_eur_per_kw=replacement_eur_per_kw,
        replacement_eur_per_kwh=replacement_eur_per_kwh,
        replacement_pct_of_capex=replacement_pct_of_capex,
        replacement_capacity_factor_pct=replacement_capacity_factor_pct,
        optimization_fee_pct=optimization_fee_pct,
        grid_max_kw=grid_max_kw,
        grid_loss_factor=grid_loss_factor,
        grid_costs_capex=grid_capex_cfg,
        grid_costs_opex=grid_opex_cfg,
        operating_mode=scenario.operating_mode,
        spot_prices_yearly=[sc.price_per_year for sc in scenarios_list if sc.is_central is True][0],
        fixed_prices_yearly=fixed_prices_yearly,
        goo_prices_yearly=goo_prices_yearly,
        cap_prices_yearly=cap_prices_yearly,
        lifetime_years=lifetime,
        commissioning_year=commissioning_year,
        leverage_pct=leverage_pct,
        interest_rate_pct=interest_rate_pct,
        loan_tenor_years=loan_tenor_years,
        inflation_rate=inflation_rate,
        discount_rate=discount_rate,
        afa_years_pv=afa_years_pv,
        afa_years_bess=afa_years_bess,
        gewerbesteuer_messzahl=gewerbesteuer_messzahl,
        gewerbesteuer_hebesatz=gewerbesteuer_hebesatz,
        koerperschaftsteuer_pct=koerperschaftsteuer_pct,
        solidaritaetszuschlag_pct=solidaritaetszuschlag_pct,
        debt_sizing_downside_pct=debt_sizing_downside_pct,
        timestep_hours=ts_timestep_hours,
        intervals_per_day=ts_intervals_per_day,
        intervals_per_year=ts_intervals_per_year,
        max_workers=1 if args.verbose else None,
        skip_baseline=skip_baseline,
        bess_absolute_power_kw=bess_absolute_power_kw,
        bess_absolute_capacity_kwh=bess_absolute_capacity_kwh,
    )

    logger.info("Starting grid search…")
    grid_result = run_grid_search(grid_search_config)

    if grid_result.optimal is None:
        logger.error("Grid search found no valid optimum (all IRRs are None).")
        return 1

    opt = grid_result.optimal
    logger.info(
        "Optimal: scale=%.0f %%, E/P=%.1f h, power=%.0f kW, capacity=%.0f kWh, "
        "Equity IRR=%.2f %%",
        opt.scale_pct,
        opt.e_to_p_ratio,
        opt.bess_power_kw,
        opt.bess_capacity_kwh,
        (opt.equity_irr or 0.0) * 100.0,
    )

    # ------------------------------------------------------------------
    # Re-run P50 simulation for optimal configuration (needed for CSVs)
    # ------------------------------------------------------------------
    from pv_bess_model.bess.replacement import ReplacementConfig
    from pv_bess_model.dispatch.engine import (
        DispatchEngineConfig,
        compute_deterministic_offline_days,
    )

    replacement = ReplacementConfig(
        enabled=replacement_enabled,
        year=replacement_year,
        fixed_eur=replacement_fixed_eur,
        eur_per_kw=replacement_eur_per_kw,
        eur_per_kwh=replacement_eur_per_kwh,
        capacity_factor_pct=replacement_capacity_factor_pct,
    )
    engine_config = DispatchEngineConfig(
        mode=scenario.operating_mode,
        grid_max_kw=grid_max_kw,
        bess_nameplate_kwh=opt.bess_capacity_kwh,
        bess_max_charge_kw=opt.bess_power_kw,
        bess_max_discharge_kw=opt.bess_power_kw,
        bess_rte=bess_rte,
        grid_loss_factor=grid_loss_factor,
        bess_min_soc_pct=bess_min_soc_pct,
        bess_max_soc_pct=bess_max_soc_pct,
        bess_degradation_rate=bess_degradation_rate,
        pv_degradation_rate=pv_degradation_rate,
        replacement=replacement,
        lifetime_years=lifetime,
        commissioning_year=commissioning_year,
        bess_power_kw=opt.bess_power_kw,
        timestep_hours=ts_timestep_hours,
        intervals_per_day=ts_intervals_per_day,
        intervals_per_year=ts_intervals_per_year,
    )
    offline_days = compute_deterministic_offline_days(bess_availability_pct)
    offline_days_yearly = [offline_days] * lifetime

    sim = run_simulation(
        config=engine_config,
        pv_base_timeseries=central_pv_timeseries,
        pv_base_timeseries_year=central_pv_timeseries_year,
        spot_prices_yearly=spot_prices_yearly,
        fixed_prices_yearly=fixed_prices_yearly,
        offline_days_yearly=offline_days_yearly,
        pv_offline_days_yearly=[{val + 28 for val in s} for s in offline_days_yearly],  # Shift BESS offline days by 4 weeks
        goo_prices_yearly=goo_prices_yearly,
        cap_prices_yearly=cap_prices_yearly,
    )

    annual_revenues = [r.total_revenue for r in sim.annual_results]
    annual_pv_kwh = [r.pv_export for r in sim.annual_results]
    annual_bess_throughput = [r.bess_throughput for r in sim.annual_results]
    annual_bess_spot_revenues = [r.bess_spot_revenue for r in sim.annual_results]
    total_production_kwh = sum(annual_pv_kwh)

    # Build cashflow
    debt_schedule = build_annuity_schedule(
        total_capex=opt.capex_total,
        leverage_pct=leverage_pct,
        annual_interest_rate=interest_rate_pct / 100.0,
        tenor_years=loan_tenor_years,
    )
    _upgrade = replacement_capacity_factor_pct / 100.0
    replacement_cost = (
        replacement_fixed_eur
        + replacement_eur_per_kw * opt.bess_power_kw
        + replacement_eur_per_kwh * opt.bess_capacity_kwh * _upgrade
        + replacement_pct_of_capex * opt.capex_bess
    )
    cashflow = build_cashflow_projection(
        lifetime_years=lifetime,
        annual_revenues=annual_revenues,
        base_opex=opt.opex_base,
        inflation_rate=inflation_rate,
        capex_total=opt.capex_total,
        capex_pv=opt.capex_pv,
        capex_bess=opt.capex_bess,
        debt_schedule=debt_schedule,
        afa_years_pv=afa_years_pv,
        afa_years_bess=afa_years_bess,
        gewerbesteuer_messzahl=gewerbesteuer_messzahl,
        gewerbesteuer_hebesatz=gewerbesteuer_hebesatz,
        koerperschaftsteuer_pct=koerperschaftsteuer_pct,
        solidaritaetszuschlag_pct=solidaritaetszuschlag_pct,
        replacement_cost=replacement_cost if replacement_enabled else 0.0,
        replacement_year=replacement_year if replacement_enabled else None,
        replacement_leverage_pct=leverage_pct,
        replacement_interest_rate=interest_rate_pct / 100.0,
        replacement_loan_tenor_years=loan_tenor_years,
        optimization_fee_pct=optimization_fee_pct,
        annual_bess_spot_revenues=annual_bess_spot_revenues,
    )

    annual_opex = []
    for y in range(1, lifetime + 1):
        opex_y = inflate_value(opt.opex_base, inflation_rate, y)
        if optimization_fee_pct > 0.0:
            opex_y += annual_bess_spot_revenues[y - 1] * optimization_fee_pct / 100.0
        annual_opex.append(opex_y)
    annual_debt_service = [cashflow.years[y - 1].debt_service for y in range(1, lifetime + 1)]
    annual_dscr: list[float | None] = []
    for y in range(lifetime):
        ds = annual_debt_service[y]
        ebitda = annual_revenues[y] - annual_opex[y]
        if ds > 0.0:
            annual_dscr.append(ebitda / ds)
        else:
            annual_dscr.append(None)

    total_opex_lifetime = sum(annual_opex)
    metrics = compute_all_metrics(
        equity_cashflows=cashflow.equity_cashflows,
        project_cashflows=cashflow.project_cashflows,
        annual_revenues=annual_revenues,
        annual_opex=annual_opex,
        annual_debt_service=annual_debt_service,
        total_capex=opt.capex_total,
        total_opex_lifetime=total_opex_lifetime,
        total_production_kwh=total_production_kwh,
        discount_rate=discount_rate,
    )

    # ------------------------------------------------------------------
    # Step 6: Build MC parameters (needed for MC and/or analyses)
    # ------------------------------------------------------------------
    analyses_cfg = scenario.raw.get("scenario", {}).get("analyses", {})
    any_analysis_enabled = (
        analyses_cfg.get("eeg_sensitivity", {}).get("enabled", False)
        or analyses_cfg.get("ppa_collar", {}).get("enabled", False)
        or analyses_cfg.get("ppa_baseload", {}).get("enabled", False)
    )

    need_mc_params = mc_enabled or any_analysis_enabled
    mc_result = None
    mc_params: MCParams | None = None
    scenario_prices: dict[str, list[np.ndarray]] = {}

    if need_mc_params:
        mc_iterations = int(mc_cfg.get("iterations", 1000))
        sigma_capex_pv = float(mc_cfg.get("sigma_capex_pv_pct", DEFAULT_MC_SIGMA_CAPEX_PV_PCT)) / 100.0
        sigma_capex_bess = float(mc_cfg.get("sigma_capex_bess_pct", DEFAULT_MC_SIGMA_CAPEX_BESS_PCT)) / 100.0
        sigma_opex_pv = float(mc_cfg.get("sigma_opex_pv_pct", DEFAULT_MC_SIGMA_OPEX_PV_PCT)) / 100.0
        sigma_opex_bess = float(mc_cfg.get("sigma_opex_bess_pct", DEFAULT_MC_SIGMA_OPEX_BESS_PCT)) / 100.0
        sigma_pv_avail = float(mc_cfg.get("sigma_pv_availability_pct", DEFAULT_MC_SIGMA_PV_AVAILABILITY_PCT)) / 100.0
        sigma_bess_avail = float(mc_cfg.get("sigma_bess_availability_pct", DEFAULT_MC_SIGMA_BESS_AVAILABILITY_PCT)) / 100.0

        mc_price_scenarios: dict[str, dict] = {}

        if scenarios_list and scenario_prices_map:
            scenario_prices = scenario_prices_map
            mc_price_scenarios = mc_price_scenarios_from_list
        else:
            price_scenarios_cfg = mc_cfg.get("price_scenarios", {})
            if price_scenarios_cfg:
                for name, cfg_item in price_scenarios_cfg.items():
                    col = cfg_item["csv_column"]
                    scenario_prices[name] = _build_spot_prices_yearly(
                        extended_prices[col], lifetime, inflation_rate, inflation_on_prices
                    )
                mc_price_scenarios = {
                    k: {"csv_column": v["csv_column"], "weight": float(v["weight"])}
                    for k, v in price_scenarios_cfg.items()
                }
            else:
                scenario_prices = {"mid": spot_prices_yearly}
                mc_price_scenarios = {"mid": {"csv_column": "MID", "weight": 1.0}}

        mc_params = MCParams(
            iterations=mc_iterations,
            sigma_capex_pv=sigma_capex_pv,
            sigma_capex_bess=sigma_capex_bess,
            sigma_opex_pv=sigma_opex_pv,
            sigma_opex_bess=sigma_opex_bess,
            sigma_pv_availability=sigma_pv_avail,
            mu_bess_availability=bess_availability_pct / 100.0,
            sigma_bess_availability=sigma_bess_avail,
            price_scenarios=mc_price_scenarios,
            max_workers=1 if args.verbose else None,
        )

    # ------------------------------------------------------------------
    # Step 6a: Monte Carlo (only if enabled)
    # ------------------------------------------------------------------
    if mc_enabled and mc_params is not None:
        logger.info("Starting Monte Carlo (%d iterations)…", mc_params.iterations)
        mc_result = run_monte_carlo(
            base_config=grid_search_config,
            optimal=opt,
            mc_params=mc_params,
            scenario_prices=scenario_prices,
            scenario_pv_timeseries=scenario_pv_map if scenario_pv_map else None,
        )

    # ------------------------------------------------------------------
    # Step 6b: Post-Grid-Search Analyses
    # ------------------------------------------------------------------
    eeg_sens_result = None
    collar_result = None
    baseload_result = None

    if any_analysis_enabled and mc_params is not None:
        marketing = scenario.finance.get("revenue_streams", {}).get("marketing", {})
        eeg_inflation_flag = bool(marketing.get("eeg_inflation", False))
        eeg_fixed_price_years = int(marketing.get("fixed_price_years", 20))
        _sc_pv = scenario_pv_map if scenario_pv_map else None

        if analyses_cfg.get("eeg_sensitivity", {}).get("enabled", False):
            eeg_cfg = analyses_cfg["eeg_sensitivity"]
            floor_prices = eeg_cfg["floor_prices_eur_per_kwh"]
            logger.info(
                "Running EEG sensitivity analysis (%d price points)…",
                len(floor_prices),
            )
            eeg_sens_result = run_eeg_sensitivity(
                base_config=grid_search_config,
                optimal=opt,
                mc_params=mc_params,
                scenario_prices=scenario_prices,
                floor_prices=floor_prices,
                inflation_rate=inflation_rate,
                eeg_inflation=eeg_inflation_flag,
                fixed_price_years=eeg_fixed_price_years,
                scenario_pv_timeseries=_sc_pv,
            )

        if analyses_cfg.get("ppa_collar", {}).get("enabled", False):
            collar_cfg = analyses_cfg["ppa_collar"]
            logger.info(
                "Running PPA Collar analysis (%d × %d = %d combinations)…",
                len(collar_cfg["floor_prices_eur_per_mwh"]),
                len(collar_cfg["cap_spreads_eur_per_mwh"]),
                len(collar_cfg["floor_prices_eur_per_mwh"])
                * len(collar_cfg["cap_spreads_eur_per_mwh"]),
            )
            collar_result = run_ppa_collar_analysis(
                base_config=grid_search_config,
                optimal=opt,
                mc_params=mc_params,
                scenario_prices=scenario_prices,
                floor_prices_eur_per_mwh=collar_cfg["floor_prices_eur_per_mwh"],
                cap_spreads_eur_per_mwh=collar_cfg["cap_spreads_eur_per_mwh"],
                duration_years=collar_cfg["duration_years"],
                inflation_on_ppa=collar_cfg.get("inflation_on_ppa", False),
                goo_premium_eur_per_kwh=collar_cfg["goo_premium_eur_per_kwh"],
                inflation_rate=inflation_rate,
                scenario_pv_timeseries=_sc_pv,
            )

        if analyses_cfg.get("ppa_baseload", {}).get("enabled", False):
            bl_cfg = analyses_cfg["ppa_baseload"]
            logger.info(
                "Running PPA Baseload analysis (%d × %d = %d combinations)…",
                len(bl_cfg["ppa_prices_eur_per_mwh"]),
                len(bl_cfg["baseload_levels_mw"]),
                len(bl_cfg["ppa_prices_eur_per_mwh"])
                * len(bl_cfg["baseload_levels_mw"]),
            )
            baseload_result = run_ppa_baseload_analysis(
                base_config=grid_search_config,
                optimal=opt,
                mc_params=mc_params,
                scenario_prices=scenario_prices,
                ppa_prices_eur_per_mwh=bl_cfg["ppa_prices_eur_per_mwh"],
                baseload_levels_mw=bl_cfg["baseload_levels_mw"],
                duration_years=bl_cfg["duration_years"],
                inflation_on_ppa=bl_cfg.get("inflation_on_ppa", False),
                goo_premium_eur_per_kwh=bl_cfg["goo_premium_eur_per_kwh"],
                inflation_rate=inflation_rate,
                scenario_pv_timeseries=_sc_pv,
            )

    # ------------------------------------------------------------------
    # Step 7: Write output CSVs
    # ------------------------------------------------------------------
    marketing_type = (
        scenario.finance.get("revenue_streams", {})
        .get("marketing", {})
        .get("type", "market")
    )

    write_summary_csv(
        path=output_dir / f"{scenario.name}_summary.csv",
        scenario_name=scenario.name,
        pv_peak_kwp=pv_peak_kwp,
        operating_mode=scenario.operating_mode,
        marketing_type=marketing_type,
        lifetime_years=lifetime,
        grid_result=grid_result,
        cashflow=cashflow,
        equity_irr=metrics.equity_irr,
        project_irr=metrics.project_irr,
        npv=metrics.npv,
        dscr_min=metrics.dscr_min,
        dscr_avg=metrics.dscr_avg,
        lcoe=metrics.lcoe,
        payback_year=metrics.payback_year,
        total_production_kwh=total_production_kwh,
        config=csv_config,
    )

    write_cashflows_csv(
        path=output_dir / f"{scenario.name}_cashflows.csv",
        cashflow=cashflow,
        annual_pv_production_kwh=annual_pv_kwh,
        annual_bess_throughput_kwh=annual_bess_throughput,
        annual_dscr=annual_dscr,
        commissioning_year=scenario.commissioning_year,
        config=csv_config,
    )

    write_grid_search_csv(
        path=output_dir / f"{scenario.name}_grid_search.csv",
        grid_result=grid_result,
        config=csv_config,
    )

    if mc_result is not None:
        write_monte_carlo_csv(
            path=output_dir / f"{scenario.name}_monte_carlo.csv",
            mc_result=mc_result,
            config=csv_config,
        )

    if eeg_sens_result is not None:
        write_eeg_sensitivity_csv(
            path=output_dir / f"{scenario.name}_eeg_sensitivity.csv",
            result=eeg_sens_result,
            config=csv_config,
        )

    if collar_result is not None:
        collar_cfg = analyses_cfg["ppa_collar"]
        write_ppa_collar_csv(
            path=output_dir / f"{scenario.name}_ppa_collar.csv",
            result=collar_result,
            duration_years=collar_cfg["duration_years"],
            config=csv_config,
        )

    if baseload_result is not None:
        bl_cfg = analyses_cfg["ppa_baseload"]
        write_ppa_baseload_csv(
            path=output_dir / f"{scenario.name}_ppa_baseload.csv",
            result=baseload_result,
            duration_years=bl_cfg["duration_years"],
            config=csv_config,
        )

    # Dispatch sample (if requested)
    export_dispatch = _out_block.get("export_dispatch_sample", True)
    if export_dispatch:
        write_dispatch_sample_csv(
            path=output_dir / f"{scenario.name}_dispatch_sample.csv",
            hourly_sample=sim.hourly_sample,
            start_year=scenario.commissioning_year,
            config=csv_config,
        )

    # ------------------------------------------------------------------
    # Step 7b: PDF Report generation
    # ------------------------------------------------------------------
    _generate_report(
        scenario=scenario,
        output_dir=output_dir,
        _out_block=_out_block,
        args=args,
        grid_result=grid_result,
        opt=opt,
        metrics=metrics,
        mc_result=mc_result,
        weather_data_for_report=weather_data_for_report,
        scenario_prices=scenario_prices,
        scenarios_list=scenarios_list,
        commissioning_year=commissioning_year,
        eeg_sens_result=eeg_sens_result,
        collar_result=collar_result,
        baseload_result=baseload_result,
    )

    # ------------------------------------------------------------------
    # Step 8: Print summary
    # ------------------------------------------------------------------
    _print_summary(scenario.name, opt, metrics, mc_result)

    return 0


def _summarize_sensitivity(result) -> str:
    """Build a short text summary of a sensitivity analysis result for LLM input.

    Parameters
    ----------
    result:
        ``SensitivityResult`` instance.

    Returns
    -------
    str
        Formatted key findings string.
    """
    lines: list[str] = []
    for pt in result.points:
        params_str = ", ".join(f"{k}={v}" for k, v in pt.params.items())
        eq_stats = pt.mc_result.overall_stats.get("equity_irr")
        if eq_stats is not None:
            lines.append(f"- {params_str}: IRR mean={eq_stats.mean * 100:.2f}%, std={eq_stats.std * 100:.2f}%")
    return "\n".join(lines) if lines else "Keine Ergebnisse verfügbar."


def _build_results_summary(opt, metrics, mc_result) -> str:
    """Build a comprehensive results summary for the LLM conclusion.

    Parameters
    ----------
    opt:
        ``GridPointResult`` optimal configuration.
    metrics:
        Computed financial metrics.
    mc_result:
        Monte Carlo result or None.

    Returns
    -------
    str
        Formatted summary string.
    """
    parts: list[str] = [
        f"Optimale BESS-Skalierung: {opt.scale_pct:.0f}% der PV-Leistung",
        f"E/P-Verhältnis: {opt.e_to_p_ratio:.1f}h",
        f"BESS: {opt.bess_power_kw:.0f} kW / {opt.bess_capacity_kwh:.0f} kWh",
        f"CAPEX: {opt.capex_total:,.0f} EUR",
        f"Equity IRR: {(opt.equity_irr or 0.0) * 100:.2f}%",
        f"Project IRR: {(metrics.project_irr or 0.0) * 100:.2f}%",
        f"NPV: {metrics.npv:,.0f} EUR",
    ]
    if metrics.dscr_min is not None:
        parts.append(f"Min DSCR: {metrics.dscr_min:.2f}")
    if metrics.lcoe is not None:
        parts.append(f"LCOE: {metrics.lcoe * 100:.3f} ct/kWh")

    if mc_result is not None:
        eq_stats = mc_result.overall_stats.get("equity_irr")
        if eq_stats is not None:
            import math
            if not math.isnan(eq_stats.median):
                parts.append(f"MC Equity IRR: median={eq_stats.median * 100:.2f}%, "
                             f"P10={eq_stats.p10 * 100:.2f}%, P90={eq_stats.p90 * 100:.2f}%")

    return "\n".join(f"- {p}" for p in parts)


def _generate_report(
    scenario,
    output_dir: Path,
    _out_block: dict,
    args: argparse.Namespace,
    grid_result,
    opt,
    metrics,
    mc_result,
    weather_data_for_report: dict[int, np.ndarray] | None,
    scenario_prices: dict,
    scenarios_list: list,
    commissioning_year: int,
    eeg_sens_result,
    collar_result,
    baseload_result,
) -> None:
    """Generate the PDF report (Step 7b).

    This function handles chart creation, optional LLM text generation,
    and PDF rendering. All errors are caught and logged without
    interrupting the main flow.

    Parameters
    ----------
    scenario:
        Validated scenario configuration.
    output_dir:
        Output directory for the report.
    _out_block:
        Raw output block from scenario JSON.
    args:
        Parsed CLI arguments.
    grid_result:
        Grid search result.
    opt:
        Optimal grid point.
    metrics:
        Financial metrics.
    mc_result:
        Monte Carlo result or None.
    weather_data_for_report:
        Weather year PV data or None.
    scenario_prices:
        Price scenario data.
    scenarios_list:
        List of PriceWeatherScenario instances.
    commissioning_year:
        Project commissioning year.
    eeg_sens_result:
        EEG sensitivity result or None.
    collar_result:
        PPA Collar result or None.
    baseload_result:
        PPA Baseload result or None.
    """
    import os

    report_cfg = _out_block.get("report", {})
    report_enabled = report_cfg.get("enabled", False)

    if not report_enabled or args.no_report:
        return

    # Import report modules with graceful degradation
    try:
        from pv_bess_model.output.report.charts import create_all_charts
    except ImportError:
        logger.warning("matplotlib not available. Skipping report generation.")
        return

    from pv_bess_model.output.report.pdf_builder import ReportConfig, build_report

    config = ReportConfig(
        enabled=True,
        company_name=report_cfg.get("company_name", ""),
        logo_path=report_cfg.get("logo_path"),
    )

    # Build scenario labels from scenarios_list
    scenario_labels: dict[str, str] | None = None
    if scenarios_list:
        scenario_labels = {}
        for sc in scenarios_list:
            label = getattr(sc, "label", None) or sc.name
            scenario_labels[sc.name] = label

    # Create charts (always, even without LLM)
    logger.info("Generating report charts…")
    try:
        chart_paths = create_all_charts(
            output_dir=output_dir,
            grid_result=grid_result,
            weather_timeseries=weather_data_for_report,
            scenario_prices=scenario_prices if scenario_prices else None,
            scenario_labels=scenario_labels,
            commissioning_year=commissioning_year,
            eeg_result=eeg_sens_result,
            collar_result=collar_result,
            baseload_result=baseload_result,
        )
    except Exception:
        logger.error("Chart generation failed. Skipping report.", exc_info=True)
        return

    # Generate LLM texts
    texts: dict[str, str] = {}

    if not args.no_llm:
        api_key_env = report_cfg.get("llm_api_key_env", "ANTHROPIC_API_KEY")
        api_key = os.environ.get(api_key_env, "")

        if api_key:
            try:
                from pv_bess_model.output.report.llm_client import (
                    LLMClient,
                    generate_conclusion,
                    generate_grid_search_text,
                    generate_input_summary,
                    generate_model_description,
                    generate_price_scenario_text,
                    generate_pv_yield_text,
                    generate_sensitivity_text,
                )

                llm_model = report_cfg.get("llm_model", None)
                client_kwargs: dict = {
                    "api_key": api_key,
                    "cache_dir": output_dir,
                }
                if llm_model:
                    client_kwargs["model"] = llm_model

                client = LLMClient(**client_kwargs)

                # Page 0: Model description
                claude_md_path = Path(__file__).parent.parent / "CLAUDE.md"
                claude_md_excerpt = ""
                if claude_md_path.exists():
                    claude_md_excerpt = claude_md_path.read_text(encoding="utf-8")[:2000]
                texts["text_model_description"] = generate_model_description(client, claude_md_excerpt)

                # Page 1: Input summary
                input_params = {
                    "PV-Leistung": f"{scenario.pv['design']['peak_power_kwp']:,.0f} kWp",
                    "Projektlaufzeit": f"{scenario.lifetime_years} Jahre",
                    "Betriebsmodus": scenario.operating_mode,
                    "Fremdkapitalquote": f"{scenario.finance.get('leverage_pct', 0)}%",
                    "Inflationsrate": f"{scenario.finance.get('inflation_rate', 0) * 100:.1f}%",
                }
                texts["text_input_summary"] = generate_input_summary(client, input_params)

                # Page 2: PV yield
                if weather_data_for_report:
                    annual_kwh = {y: float(np.sum(ts)) for y, ts in weather_data_for_report.items()}
                    texts["text_pv_yield"] = generate_pv_yield_text(client, annual_kwh)

                # Page 3: Price scenarios
                if scenario_prices and len(scenario_prices) > 1:
                    scenario_means = {}
                    for name, yearly in scenario_prices.items():
                        all_vals = np.concatenate(yearly) if yearly else np.array([0.0])
                        scenario_means[name] = float(np.mean(all_vals)) * 1000.0  # EUR/MWh
                    texts["text_price_scenarios"] = generate_price_scenario_text(client, scenario_means)

                # Page 4: Grid search
                pv_only_irr = None
                for pt in grid_result.points:
                    if pt.scale_pct == 0.0 and pt.equity_irr is not None:
                        pv_only_irr = pt.equity_irr * 100.0
                        break
                texts["text_grid_search"] = generate_grid_search_text(
                    client,
                    optimal_scale=opt.scale_pct,
                    optimal_ep=opt.e_to_p_ratio,
                    optimal_irr=(opt.equity_irr or 0.0) * 100.0,
                    pv_only_irr=pv_only_irr,
                )

                # Pages 5-7: Sensitivity analyses
                if eeg_sens_result is not None:
                    texts["text_eeg_sensitivity"] = generate_sensitivity_text(
                        client, "EEG-Sensitivität", _summarize_sensitivity(eeg_sens_result)
                    )
                if collar_result is not None:
                    texts["text_ppa_collar"] = generate_sensitivity_text(
                        client, "PPA Collar-Analyse", _summarize_sensitivity(collar_result)
                    )
                if baseload_result is not None:
                    texts["text_ppa_baseload"] = generate_sensitivity_text(
                        client, "PPA Baseload-Analyse", _summarize_sensitivity(baseload_result)
                    )

                # Page 8: Conclusion
                texts["text_conclusion"] = generate_conclusion(
                    client, _build_results_summary(opt, metrics, mc_result)
                )

                logger.info("LLM text generation complete.")
            except ImportError:
                logger.warning("anthropic package not available. Report will use placeholder texts.")
            except Exception:
                logger.warning("LLM text generation failed.", exc_info=True)
        else:
            logger.warning(
                "No API key found in env var '%s'. Report will use placeholder texts.",
                api_key_env,
            )

    # Build and render PDF
    logger.info("Assembling PDF report…")
    pdf_path = build_report(
        scenario_name=scenario.name,
        output_dir=output_dir,
        chart_paths=chart_paths,
        texts=texts,
        config=config,
        scenario=scenario,
    )
    if pdf_path is not None:
        print(f"  Report: {pdf_path}")


def _print_summary(
    scenario_name: str,
    opt,
    metrics,
    mc_result,
) -> None:
    """Print a concise result summary to stdout."""
    irr_str = f"{(opt.equity_irr or 0.0) * 100:.2f} %" if opt.equity_irr else "n/a"
    npv_str = f"{opt.npv:,.0f} €"
    print()
    print("=" * 60)
    print(f"  Scenario: {scenario_name}")
    print("=" * 60)
    print(f"  Optimal BESS scale:    {opt.scale_pct:.0f} % of PV")
    print(f"  Optimal E/P ratio:     {opt.e_to_p_ratio:.1f} h")
    print(f"  BESS power:            {opt.bess_power_kw:.0f} kW")
    print(f"  BESS capacity:         {opt.bess_capacity_kwh:.0f} kWh")
    print(f"  Total CAPEX:           {opt.capex_total:,.0f} €")
    print()
    print(f"  Equity IRR:            {irr_str}")
    print(f"  Project IRR:           {(metrics.project_irr or 0.0) * 100:.2f} %")
    print(f"  NPV (@discount rate):  {metrics.npv:,.0f} €")
    if metrics.dscr_min is not None:
        print(f"  Min DSCR:              {metrics.dscr_min:.2f}")
    if metrics.lcoe is not None:
        print(f"  LCOE:                  {metrics.lcoe * 100:.3f} €ct/kWh")
    if metrics.payback_year is not None:
        print(f"  Payback year:          {metrics.payback_year}")

    if mc_result is not None:
        eq_stats = mc_result.overall_stats.get("equity_irr")
        if eq_stats is not None:
            import math
            if not math.isnan(eq_stats.median):
                print()
                print(f"  MC Equity IRR median:  {eq_stats.median * 100:.2f} %")
                print(f"  MC Equity IRR P10:     {eq_stats.p10 * 100:.2f} %")
                print(f"  MC Equity IRR P90:     {eq_stats.p90 * 100:.2f} %")

    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run the scenario."""
    parser = _build_parser()
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    sys.exit(run(args))


if __name__ == "__main__":
    main()
