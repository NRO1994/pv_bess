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
    DEFAULT_LOAN_TENOR_YEARS,
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
    HOURS_PER_YEAR,
    INTERVALS_PER_DAY,
    INTERVALS_PER_HOUR,
    INTERVALS_PER_YEAR,
    MARKETING_TYPE_EEG,
    PPA_TYPE_COLLAR,
    PPA_TYPE_FLOOR,
    PPA_TYPE_NONE,
    PPA_TYPE_PAY_AS_PRODUCED,
    TIMESTEP_HOURS, PPA_TYPE_BASELOAD, MWH_TO_KWH,
)
from pv_bess_model.config.loader import (
    PriceData,
    PriceWeatherScenario,
    ScenarioConfig,
    extend_price_timeseries,
    load_inflation_csv,
    load_price_csv,
    load_scenario,
    parse_inflation_timeseries_config,
    parse_scenarios,
)
from pv_bess_model.finance.inflation import (
    build_opex_inflation_factors,
    build_price_inflation_factors,
)
from pv_bess_model.market.eeg import eeg_config_from_dict
from pv_bess_model.market.ppa import ppa_config_from_dict
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
    write_combined_monte_carlo_csv,
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
        "--no-report",
        action="store_true",
        default=False,
        help="Skip HTML report generation entirely.",
    )
    p.add_argument(
        "--skip-llm-prompt",
        action="store_true",
        default=False,
        help="Skip the interactive LLM prompt pause; generate report with placeholder texts.",
    )
    p.add_argument(
        "--llm-response",
        metavar="PATH",
        default=None,
        help="Path to a pre-prepared LLM response JSON file (skips interactive pause).",
    )
    return p


# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------


def _build_fixed_prices_yearly(
    scenario: ScenarioConfig,
    opex_inflation_factors: list[float],
) -> list[float]:
    """Build the per-year floor/fixed price list for the dispatch engine.

    Returns 0.0 for each year when no floor is active (pure market).
    Uses *opex_inflation_factors* (base = commissioning year) for contract
    price inflation (EEG, PPA floor/cap/pay-as-produced).

    Parameters
    ----------
    scenario:
        Validated scenario configuration.
    opex_inflation_factors:
        Per-year cumulative inflation factors (base = commissioning year).

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
        factor = opex_inflation_factors[year - 1]

        if marketing_type == "eeg":
            eeg_cfg = eeg_config_from_dict(marketing)
            if year <= eeg_cfg.fixed_price_years:
                base = eeg_cfg.floor_price_eur_per_kwh
                price = base * factor if eeg_cfg.inflation_enabled else base

        elif ppa_type == PPA_TYPE_FLOOR:
            ppa_cfg = ppa_config_from_dict(ppa_dict)
            if year <= ppa_cfg.duration_years:
                base = ppa_cfg.floor_price_eur_per_kwh or 0.0
                price = base * factor if ppa_cfg.inflation_enabled else base
                # GoO premium is NOT added here; it is passed separately via
                # _build_goo_prices_yearly() and added after the floor comparison.

        elif ppa_type == PPA_TYPE_COLLAR:
            ppa_cfg = ppa_config_from_dict(ppa_dict)
            if year <= ppa_cfg.duration_years:
                base = ppa_cfg.floor_price_eur_per_kwh or 0.0
                price = base * factor if ppa_cfg.inflation_enabled else base
                # GoO premium is NOT added here; it is passed separately via
                # _build_goo_prices_yearly() and added after the clip operation.
                # Cap price is handled via _build_cap_prices_yearly().

        elif ppa_type in (PPA_TYPE_PAY_AS_PRODUCED, PPA_TYPE_BASELOAD):
            ppa_cfg = ppa_config_from_dict(ppa_dict)
            if year <= ppa_cfg.duration_years:
                base = ppa_cfg.pay_as_produced_price_eur_per_kwh or 0.0
                price = base * factor if ppa_cfg.inflation_enabled else base
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
    # If there is no PV, there is no chance for GoO
    if scenario.pv.get('design', {}).get('peak_power', 0) == 0:
        return [0.0] * lifetime

    ppa_cfg = ppa_config_from_dict(scenario.finance.get("revenue_streams", {}).get("ppa", {}))
    goo_prices = [ppa_cfg.goo_premium_eur_per_kwh] * lifetime
    eeg_dict = scenario.finance.get("revenue_streams", {}).get("marketing", {})
    eeg_type = eeg_dict.get("type", PPA_TYPE_NONE)

    if eeg_type == MARKETING_TYPE_EEG: # Years under EEG does not have any GoO - only during direct marketing
        duration = eeg_dict.get("fixed_price_years", lifetime)
        goo_prices = [0.0] * duration + goo_prices[duration:]

    return goo_prices


def _build_cap_prices_yearly(
    scenario: ScenarioConfig,
    opex_inflation_factors: list[float],
) -> list[float]:
    """Build the per-year cap price list for the dispatch engine.

    Only relevant for PPA Collar.  Returns 0.0 for each year when no cap is
    active (0.0 = no cap, i.e. unbounded upside).

    Parameters
    ----------
    scenario:
        Validated scenario configuration.
    opex_inflation_factors:
        Per-year cumulative inflation factors (base = commissioning year).

    Returns
    -------
    list[float]
        Cap price in €/kWh for each project year (length = lifetime_years).
        Index 0 = year 1.  0.0 means no cap for that year.
    """
    lifetime = scenario.lifetime_years
    ppa_dict = scenario.finance.get("revenue_streams", {}).get("ppa", {})
    ppa_type = ppa_dict.get("type", PPA_TYPE_NONE)

    if ppa_type not in (PPA_TYPE_COLLAR, PPA_TYPE_PAY_AS_PRODUCED, PPA_TYPE_BASELOAD):
        return [0.0] * lifetime

    ppa_cfg = ppa_config_from_dict(ppa_dict)
    if ppa_type == PPA_TYPE_COLLAR:
        base = ppa_cfg.cap_price_eur_per_kwh or 0.0
    elif ppa_type == PPA_TYPE_PAY_AS_PRODUCED:
        base = ppa_cfg.pay_as_produced_price_eur_per_kwh
    elif ppa_type == PPA_TYPE_BASELOAD:
        base = ppa_cfg.pay_as_produced_price_eur_per_kwh
    else:
        base = 0.0

    cap_prices: list[float] = []
    for year in range(1, lifetime + 1):
        if year <= ppa_cfg.duration_years:
            if ppa_cfg.inflation_enabled:
                price = base * opex_inflation_factors[year - 1]
            else:
                price = base
        else:
            price = 0.0
        cap_prices.append(price)
    return cap_prices


def _build_spot_prices_yearly(
    price_array: np.ndarray,
    lifetime_years: int,
    price_inflation_factors: list[float],
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
    price_inflation_factors:
        Per-year cumulative inflation factors for spot prices.
        Base = year before first forecast year (= commissioning_year − 1).
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
            year_prices = year_prices * price_inflation_factors[y - 1]
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

    # Load inflation timeseries (if configured) and build factor arrays
    inflation_ts_config = parse_inflation_timeseries_config(scenario)
    yearly_inflation_rates: dict[int, float] | None = None
    if inflation_ts_config is not None:
        yearly_inflation_rates = load_inflation_csv(
            inflation_ts_config,
            scenario_path=scenario.path,
        )
        logger.info(
            "Using inflation timeseries (%d years) instead of fixed rate %.4f",
            len(yearly_inflation_rates),
            inflation_rate,
        )

    opex_inflation_factors = build_opex_inflation_factors(
        inflation_rate=inflation_rate,
        lifetime_years=scenario.lifetime_years,
        yearly_rates=yearly_inflation_rates,
        commissioning_year=scenario.commissioning_year,
    )
    price_inflation_factors = build_price_inflation_factors(
        inflation_rate=inflation_rate,
        lifetime_years=scenario.lifetime_years,
        yearly_rates=yearly_inflation_rates,
        commissioning_year=scenario.commissioning_year,
    )

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
    bess_opex = bess_costs.get("opex", {})
    optimization_fee_pct = float(bess_opex.get("optimization_fee_pct", 0.0))
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
    grid_max_import_kw: float | None = (
        float(grid_connection["max_import_kw"])
        if "max_import_kw" in grid_connection
        else None
    )
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
    pv_sub_arrays = scenario.pv_sub_arrays

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
                if pv_sub_arrays is not None:
                    # Dual-azimuth mode: fetch each sub-array and sum profiles
                    logger.info(
                        "Dual-azimuth mode: fetching %d sub-arrays for year %d",
                        len(pv_sub_arrays), wy,
                    )
                    combined = None
                    for i, sa in enumerate(pv_sub_arrays):
                        sa_ts = client.fetch_single_year(
                            year=wy,
                            latitude=latitude,
                            longitude=longitude,
                            peak_power_kwp=float(sa["power_kwp"]),
                            mounting_type=mounting_type,
                            azimuth_deg=float(sa["azimuth_deg"]),
                            tilt_deg=float(sa["tilt_deg"]),
                            pvgis_database=pvgis_database,
                        )
                        logger.info(
                            "  Sub-array %d: %.0f kWp, azi=%.1f°, tilt=%.1f° → %.0f kWh/a",
                            i + 1, sa["power_kwp"], sa["azimuth_deg"],
                            sa["tilt_deg"], float(np.sum(sa_ts)),
                        )
                        combined = sa_ts if combined is None else combined + sa_ts
                    hourly_ts = combined
                else:
                    # Single-azimuth mode (default)
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

        # Convert to 15min per weather year
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
        central_pv_timeseries_year = scenarios_list[0].weather_year
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
            price_data, [sc.csv_column], lifetime, INTERVALS_PER_YEAR
        )

        # Replicate hourly prices to 15min: each hour repeats 4x
        # NOTE: prices are NOT divided by 4 (price per kWh stays the same)
        extended_prices_15min: dict[str, np.ndarray] = {}
        for col, arr in extended_prices_hourly.items():
            if len(arr) < INTERVALS_PER_YEAR * lifetime:
                extended_prices_15min[col] = np.repeat(arr, INTERVALS_PER_HOUR)
            else:
                extended_prices_15min[col] = arr

        spot_prices_yearly = _build_spot_prices_yearly(
            extended_prices_15min[sc.csv_column],
            lifetime,
            price_inflation_factors,
            sc.inflation_on_input_data,
            intervals_per_year=ts_intervals_per_year,
        )

        sc.price_per_year = spot_prices_yearly

    # Fixed prices per year (EEG / PPA floor, WITHOUT GoO)
    fixed_prices_yearly = _build_fixed_prices_yearly(scenario, opex_inflation_factors)
    # GoO premium per year (for floor and collar PPA types)
    goo_prices_yearly = _build_goo_prices_yearly(scenario)
    # Cap prices per year (PPA Collar only; 0.0 = no cap)
    cap_prices_yearly = _build_cap_prices_yearly(scenario, opex_inflation_factors)

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
        grid_max_import_kw=grid_max_import_kw,
        grid_loss_factor=grid_loss_factor,
        grid_costs_capex=grid_capex_cfg,
        grid_costs_opex=grid_opex_cfg,
        operating_mode=scenario.operating_mode,
        spot_prices_yearly=[sc.price_per_year for sc in scenarios_list if sc.is_central is True][0],
        fixed_prices_yearly=fixed_prices_yearly,
        baseload_mw=scenario.finance.get("revenue_streams", {}).get("ppa", {}).get("baseload_mw", 0),
        goo_prices_yearly=goo_prices_yearly,
        cap_prices_yearly=cap_prices_yearly,
        lifetime_years=lifetime,
        commissioning_year=commissioning_year,
        leverage_pct=leverage_pct,
        interest_rate_pct=interest_rate_pct,
        loan_tenor_years=loan_tenor_years,
        opex_inflation_factors=opex_inflation_factors,
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

    optimal_setup = grid_result.optimal
    logger.info(
        "Optimal: scale=%.0f %%, E/P=%.1f h, power=%.0f kW, capacity=%.0f kWh, "
        "Equity IRR=%.2f %%",
        optimal_setup.scale_pct,
        optimal_setup.e_to_p_ratio,
        optimal_setup.bess_power_kw,
        optimal_setup.bess_capacity_kwh,
        (optimal_setup.metrics.equity_irr or 0.0) * 100.0,
    )

    # Extract equity_irr_target from scenario JSON (may be None)
    equity_irr_target: float | None = finance.get("equity_irr_target", None)

    # ------------------------------------------------------------------
    # Step 6: Build MC parameters
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

    if need_mc_params:
        mc_iterations = int(mc_cfg.get("iterations", 1000))
        sigma_capex_pv = float(mc_cfg.get("sigma_capex_pv_pct", DEFAULT_MC_SIGMA_CAPEX_PV_PCT)) / 100.0
        sigma_capex_bess = float(mc_cfg.get("sigma_capex_bess_pct", DEFAULT_MC_SIGMA_CAPEX_BESS_PCT)) / 100.0
        sigma_opex_pv = float(mc_cfg.get("sigma_opex_pv_pct", DEFAULT_MC_SIGMA_OPEX_PV_PCT)) / 100.0
        sigma_opex_bess = float(mc_cfg.get("sigma_opex_bess_pct", DEFAULT_MC_SIGMA_OPEX_BESS_PCT)) / 100.0
        sigma_pv_avail = float(mc_cfg.get("sigma_pv_availability_pct", DEFAULT_MC_SIGMA_PV_AVAILABILITY_PCT)) / 100.0
        sigma_bess_avail = float(mc_cfg.get("sigma_bess_availability_pct", DEFAULT_MC_SIGMA_BESS_AVAILABILITY_PCT)) / 100.0

        mc_params = MCParams(
            iterations=mc_iterations,
            sigma_capex_pv=sigma_capex_pv,
            sigma_capex_bess=sigma_capex_bess,
            sigma_opex_pv=sigma_opex_pv,
            sigma_opex_bess=sigma_opex_bess,
            sigma_pv_availability=sigma_pv_avail,
            mu_bess_availability=bess_availability_pct / 100.0,
            sigma_bess_availability=sigma_bess_avail,
            price_scenarios=scenarios_list,
            max_workers=1 if args.verbose else None,
        )

    # ------------------------------------------------------------------
    # Step 5b: Baseline "Direktvermarktung" MC run (pure spot market)
    # ------------------------------------------------------------------
    import dataclasses as _dc

    baseline_market_irr: float | None = None
    baseline_mc_result = None

    if need_mc_params and mc_params is not None:
        baseline_market_config = _dc.replace(
            grid_search_config,
            scale_pct_of_pv=[optimal_setup.scale_pct],
            e_to_p_ratio_hours=[optimal_setup.e_to_p_ratio],
            fixed_prices_yearly=[0.0] * lifetime,
            goo_prices_yearly=[0.0] * lifetime,
            cap_prices_yearly=[0.0] * lifetime,
            baseload_mw=0,
            skip_baseline=True,
        )
        logger.info("Computing baseline Direktvermarktung IRR via Monte Carlo (pure spot market)...")
        baseline_mc_result = run_monte_carlo(
            base_config=baseline_market_config,
            optimal=optimal_setup,
            mc_params=mc_params,
            scenario_prices=scenarios_list,
            fixed_price_years=0,
            analysis_label="Direktvermarktungs-Baseline",
        )
        eq_stats = baseline_mc_result.overall_stats.get("equity_irr")
        if eq_stats is not None and not np.isnan(eq_stats.p50):
            baseline_market_irr = eq_stats.p50
            logger.info(
                "Baseline Direktvermarktung Equity IRR P50: %.2f %%",
                baseline_market_irr * 100.0,
            )
    else:
        # Fallback: deterministic grid search when MC params are not available
        baseline_market_config = _dc.replace(
            grid_search_config,
            scale_pct_of_pv=[optimal_setup.scale_pct],
            e_to_p_ratio_hours=[optimal_setup.e_to_p_ratio],
            fixed_prices_yearly=[0.0] * lifetime,
            goo_prices_yearly=[0.0] * lifetime,
            cap_prices_yearly=[0.0] * lifetime,
            baseload_mw=0,
            skip_baseline=True,
        )
        logger.info("Computing baseline Direktvermarktung IRR (pure spot market, deterministic)…")
        baseline_result = run_grid_search(baseline_market_config)
        if baseline_result.optimal is not None and baseline_result.optimal.metrics is not None:
            baseline_market_irr = baseline_result.optimal.metrics.equity_irr
            logger.info(
                "Baseline Direktvermarktung Equity IRR: %.2f %%",
                (baseline_market_irr or 0.0) * 100.0,
            )

    # ------------------------------------------------------------------
    # Step 6a: Post-Grid-Search Analyses
    # ------------------------------------------------------------------
    eeg_sens_result = None
    collar_result = None
    baseload_result = None

    if any_analysis_enabled and mc_params is not None:
        marketing = scenario.finance.get("revenue_streams", {}).get("marketing", {})
        eeg_inflation_flag = bool(marketing.get("eeg_inflation", False))
        eeg_fixed_price_years = int(marketing.get("fixed_price_years", 20))

        if analyses_cfg.get("eeg_sensitivity", {}).get("enabled", False):
            eeg_cfg = analyses_cfg["eeg_sensitivity"]
            floor_prices = eeg_cfg["floor_prices_eur_per_kwh"]
            logger.info(
                "Running EEG sensitivity analysis (%d price points)",
                len(floor_prices),
            )
            eeg_sens_result = run_eeg_sensitivity(
                base_config=grid_search_config,
                optimal=optimal_setup,
                mc_params=mc_params,
                scenario_prices=scenarios_list,
                floor_prices=floor_prices,
                opex_inflation_factors=opex_inflation_factors,
                eeg_inflation=eeg_inflation_flag,
                fixed_price_years=eeg_fixed_price_years,
            )

        if analyses_cfg.get("ppa_collar", {}).get("enabled", False):
            collar_cfg = analyses_cfg["ppa_collar"]
            logger.info(
                "Running PPA Collar analysis (%d × %d = %d combinations)",
                len(collar_cfg["floor_prices_eur_per_kwh"]),
                len(collar_cfg["cap_spreads_eur_per_kwh"]),
                len(collar_cfg["floor_prices_eur_per_kwh"])
                * len(collar_cfg["cap_spreads_eur_per_kwh"]),
            )
            collar_result = run_ppa_collar_analysis(
                base_config=grid_search_config,
                optimal=optimal_setup,
                mc_params=mc_params,
                scenario_prices=scenarios_list,
                floor_prices_eur_per_kwh=collar_cfg["floor_prices_eur_per_kwh"],
                cap_spreads_eur_per_kwh=collar_cfg["cap_spreads_eur_per_kwh"],
                duration_years=collar_cfg["duration_years"],
                inflation_on_ppa=collar_cfg.get("inflation_on_ppa", False),
                goo_premium_eur_per_kwh=collar_cfg["goo_premium_eur_per_kwh"],
                opex_inflation_factors=opex_inflation_factors,
            )

        if analyses_cfg.get("ppa_baseload", {}).get("enabled", False):
            bl_cfg = analyses_cfg["ppa_baseload"]
            logger.info(
                "Running PPA Baseload analysis (%d × %d = %d combinations)",
                len(bl_cfg["ppa_prices_eur_per_kwh"]),
                len(bl_cfg["baseload_levels_mw"]),
                len(bl_cfg["ppa_prices_eur_per_kwh"])
                * len(bl_cfg["baseload_levels_mw"]),
            )
            baseload_result = run_ppa_baseload_analysis(
                base_config=grid_search_config,
                optimal=optimal_setup,
                mc_params=mc_params,
                scenario_prices=scenarios_list,
                ppa_prices_eur_per_kwh=bl_cfg["ppa_prices_eur_per_kwh"],
                baseload_levels_mw=bl_cfg["baseload_levels_mw"],
                duration_years=bl_cfg["duration_years"],
                inflation_on_ppa=bl_cfg.get("inflation_on_ppa", False),
                goo_premium_eur_per_kwh=bl_cfg["goo_premium_eur_per_kwh"],
                opex_inflation_factors=opex_inflation_factors,
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
        cashflow=optimal_setup.cashflow,
        equity_irr=optimal_setup.metrics.equity_irr,
        project_irr=optimal_setup.metrics.project_irr,
        npv=optimal_setup.metrics.npv,
        dscr_min=optimal_setup.metrics.dscr_min,
        dscr_avg=optimal_setup.metrics.dscr_avg,
        lcoe=optimal_setup.metrics.lcoe,
        payback_year=optimal_setup.metrics.payback_year,
        total_production_kwh=np.sum([r.pv_export for r in optimal_setup.run_result.annual_results]),
        config=csv_config,
    )

    write_cashflows_csv(
        path=output_dir / f"{scenario.name}_cashflows.csv",
        cashflow=optimal_setup.cashflow,
        annual_pv_production_kwh=[r.pv_production for r in optimal_setup.run_result.annual_results],
        annual_bess_throughput_kwh=[r.bess_throughput for r in optimal_setup.run_result.annual_results],
        annual_dscr=optimal_setup.metrics.annual_dscr,
        commissioning_year=scenario.commissioning_year,
        config=csv_config,
        annual_revenue_pv_eur=[r.revenue_pv_export for r in optimal_setup.run_result.annual_results],
        annual_revenue_bess_green_eur=[r.revenue_bess_green for r in optimal_setup.run_result.annual_results],
        annual_revenue_bess_grey_eur=[r.revenue_bess_grey for r in optimal_setup.run_result.annual_results],
        annual_pv_grid_export_kwh=[r.pv_export for r in optimal_setup.run_result.annual_results],
    )

    if len(grid_result.points) > 1:
        write_grid_search_csv(
            path=output_dir / f"{scenario.name}_grid_search.csv",
            grid_result=grid_result,
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
            hourly_sample=optimal_setup.run_result.hourly_sample,
            start_year=scenario.commissioning_year,
            config=csv_config,
        )

    # Combined Monte Carlo CSV (all analyses in one file)
    all_mc_results = []
    if baseline_mc_result is not None:
        all_mc_results.append(baseline_mc_result)
    for sens_result in (eeg_sens_result, collar_result, baseload_result):
        if sens_result is not None:
            for point in sens_result.points:
                all_mc_results.append(point.mc_result)
    if all_mc_results:
        write_combined_monte_carlo_csv(
            path=output_dir / f"{scenario.name}_monte_carlo.csv",
            mc_results=all_mc_results,
            config=csv_config,
        )

    # ------------------------------------------------------------------
    # Step 7b: HTML Report generation
    # ------------------------------------------------------------------
    _generate_report(
        scenario=scenario,
        output_dir=output_dir,
        _out_block=_out_block,
        args=args,
        grid_result=grid_result,
        opt=optimal_setup,
        metrics=optimal_setup.metrics,
        mc_result=mc_result,
        weather_data_for_report=weather_data_for_report,
        scenario_prices=scenarios_list,
        commissioning_year=commissioning_year,
        eeg_sens_result=eeg_sens_result,
        collar_result=collar_result,
        baseload_result=baseload_result,
        analyses=analyses_cfg,
        baseline_market_irr=baseline_market_irr,
        equity_irr_target=equity_irr_target,
        price_inflation_factors=list(yearly_inflation_rates.values()) if yearly_inflation_rates is not None else None,
        all_mc_results=all_mc_results,
    )

    # ------------------------------------------------------------------
    # Step 8: Print summary
    # ------------------------------------------------------------------
    _print_summary(scenario.name, optimal_setup.metrics, mc_result)

    return 0


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
    scenario_prices: list[PriceWeatherScenario],
    commissioning_year: int,
    eeg_sens_result,
    collar_result,
    baseload_result,
    analyses: dict,
    baseline_market_irr: float | None = None,
    equity_irr_target: float | None = None,
    price_inflation_factors: list[float] | None = None,
    all_mc_results: list | None = None,
) -> None:
    """Generate the interactive HTML report (Step 7b).

    New flow:
    1. Collect all data into ``HtmlReportData``.
    2. Save rendered LLM prompt to output directory.
    3. Interactive pause for LLM response (unless ``--skip-llm-prompt``
       or ``--llm-response`` is used).
    4. Build and write the HTML report.

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
    report_cfg = _out_block.get("report", {})
    report_enabled = report_cfg.get("enabled", False)

    if not report_enabled or args.no_report:
        return

    from pv_bess_model.output.report.data_collector import collect_report_data
    from pv_bess_model.output.report.html_builder import build_html_report

    # Step 1: Collect all report data
    logger.info("Collecting report data…")
    try:
        report_data = collect_report_data(
            scenario=scenario,
            grid_result=grid_result,
            opt=opt,
            metrics=metrics,
            weather_data_for_report=weather_data_for_report,
            scenario_prices=scenario_prices,
            commissioning_year=commissioning_year,
            eeg_sens_result=eeg_sens_result,
            collar_result=collar_result,
            baseload_result=baseload_result,
            analyses=analyses,
            baseline_market_irr=baseline_market_irr,
            equity_irr_target=equity_irr_target,
            price_inflation_factors=price_inflation_factors,
            all_mc_results=all_mc_results,
        )
    except Exception:
        logger.error("Report data collection failed.", exc_info=True)
        return

    # Step 2 + 3: LLM prompt workflow
    llm_texts = _resolve_llm_texts(report_data, output_dir, args)
    report_data.llm_texts = llm_texts

    # Step 4: Build HTML report
    logger.info("Assembling HTML report…")
    try:
        html_path = build_html_report(report_data, output_dir)
        print(f"  Report: {html_path}")
    except Exception:
        logger.error("HTML report generation failed.", exc_info=True)


_MAX_LLM_INPUT_RETRIES: int = 3


def _resolve_llm_texts(
    report_data,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, str]:
    """Determine LLM texts via CLI flags or interactive prompt.

    Parameters
    ----------
    report_data:
        ``HtmlReportData`` instance for prompt rendering.
    output_dir:
        Output directory (for saving the prompt file).
    args:
        Parsed CLI arguments.

    Returns
    -------
    dict[str, str]
        Tab key → text mapping.
    """
    from pv_bess_model.output.report.llm_prompt import (
        get_fallback_texts,
        load_llm_response,
        save_rendered_prompt,
    )

    # Save rendered prompt (always, for reference)
    try:
        prompt_path = save_rendered_prompt(report_data, output_dir)
    except Exception:
        logger.warning("Failed to save LLM prompt.", exc_info=True)
        prompt_path = None

    # Case 1: --llm-response <path> provided via CLI
    if args.llm_response:
        response_path = Path(args.llm_response)
        if not response_path.exists():
            logger.warning("LLM response file not found: %s. Using placeholder texts.", response_path)
            return get_fallback_texts()
        try:
            return load_llm_response(response_path)
        except (ValueError, OSError) as exc:
            logger.warning("Failed to load LLM response: %s. Using placeholder texts.", exc)
            return get_fallback_texts()

    # Case 2: --skip-llm-prompt → no pause, use placeholders
    if args.skip_llm_prompt:
        logger.info("--skip-llm-prompt: Skipping LLM text integration.")
        return get_fallback_texts()

    # Case 3: Interactive pause
    prompt_display = str(prompt_path) if prompt_path else "(Speicherung fehlgeschlagen)"
    separator = "\u2550" * 60

    print()
    print(f"  {separator}")
    print(f"    LLM-Prompt gespeichert: {prompt_display}")
    print()
    print("    Bitte kopieren Sie den Prompt in Copilot und speichern")
    print("    Sie die Antwort als JSON-Datei.")
    print()

    for attempt in range(_MAX_LLM_INPUT_RETRIES):
        try:
            user_input = input("    Pfad zur LLM-Antwort-Datei (Enter zum Überspringen): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            logger.info("Input interrupted. Using placeholder texts.")
            return get_fallback_texts()

        # User pressed Enter without input → skip
        if not user_input:
            print(f"  {separator}")
            print()
            return get_fallback_texts()

        response_path = Path(user_input)
        if not response_path.exists():
            print(f"    Datei nicht gefunden: {response_path}")
            if attempt < _MAX_LLM_INPUT_RETRIES - 1:
                print("    Bitte erneut versuchen.")
            continue

        try:
            texts = load_llm_response(response_path)
            print(f"    LLM-Texte erfolgreich geladen.")
            print(f"  {separator}")
            print()
            return texts
        except (ValueError, OSError) as exc:
            print(f"    Fehler: {exc}")
            if attempt < _MAX_LLM_INPUT_RETRIES - 1:
                print("    Bitte erneut versuchen.")

    print("    Maximale Versuche erreicht. Verwende Platzhalter-Texte.")
    print(f"  {separator}")
    print()
    return get_fallback_texts()


def _print_summary(
    scenario_name: str,
    metrics,
    mc_result,
) -> None:
    """Print a concise result summary to stdout."""
    irr_str = f"{(metrics.equity_irr or 0.0) * 100:.2f} %" if metrics.equity_irr else "n/a"
    npv_str = f"{metrics.npv:,.0f} €"
    print()
    print("=" * 60)
    print(f"  Scenario: {scenario_name}")
    print("=" * 60)
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
