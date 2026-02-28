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
from pv_bess_model.optimization.grid_search import GridSearchConfig, run_grid_search
from pv_bess_model.optimization.monte_carlo import MCParams, run_monte_carlo
from pv_bess_model.output.csv_writer import (
    CsvConfig,
    write_cashflows_csv,
    write_dispatch_sample_csv,
    write_grid_search_csv,
    write_monte_carlo_csv,
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
    afa_years_pv = int(tax.get("afa_years_pv", 20))
    afa_years_bess = int(tax.get("afa_years_bess", 10))
    gewerbesteuer_hebesatz = int(tax.get("gewerbesteuer_hebesatz", 400))
    gewerbesteuer_messzahl = float(tax.get("gewerbesteuer_messzahl", 0.035))
    koerperschaftsteuer_pct = float(tax.get("koerperschaftsteuer_pct", DEFAULT_KOERPERSCHAFTSTEUER_PCT))
    solidaritaetszuschlag_pct = float(tax.get("solidaritaetszuschlag_pct", DEFAULT_SOLIDARITAETSZUSCHLAG_PCT))

    # PV parameters
    pv = scenario.pv
    pv_design = pv["design"]
    pv_perf = pv.get("performance", {})
    pv_peak_kwp = float(pv_design["peak_power_kwp"])
    pv_degradation_rate = float(pv_perf.get("degradation_rate_pct_per_year", DEFAULT_PV_DEGRADATION_RATE_PCT)) / 100.0

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
    pvgis_database = location.get("pvgis_database", "PVGIS-SARAH2")
    mounting_type = pv_design.get("mounting_type", "free")
    azimuth_deg = float(pv_design.get("azimuth_deg", 0))
    tilt_deg = float(pv_design.get("tilt_deg", 30))

    # Parse price-weather scenarios from the new schema
    price_inputs = finance.get("price_inputs", {})
    scenarios_list: list[PriceWeatherScenario] = parse_scenarios(scenario.raw)

    # Determine resolution: use 15min if scenarios are defined
    use_15min = len(scenarios_list) > 0
    if use_15min:
        ts_intervals_per_year = INTERVALS_PER_YEAR
        ts_intervals_per_day = INTERVALS_PER_DAY
        ts_timestep_hours = TIMESTEP_HOURS
    else:
        ts_intervals_per_year = HOURS_PER_YEAR
        ts_intervals_per_day = HOURS_PER_DAY
        ts_timestep_hours = 1.0

    commissioning_year = scenario.commissioning_year

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

        # Align and convert to 15min per weather year
        weather_year_15min: dict[int, np.ndarray] = {}
        for wy, hourly_ts in weather_year_hourly.items():
            aligned = align_weather_to_forecast_year(
                hourly_ts, wy, commissioning_year
            )
            weather_year_15min[wy] = hourly_to_quarter_hourly(aligned)

        # Assign PV timeseries to each scenario
        for sc in scenarios_list:
            sc.pv_timeseries_15min = weather_year_15min[sc.weather_year]

        # Central scenario PV timeseries for grid search
        central_scenarios = [s for s in scenarios_list if s.is_central]
        if central_scenarios:
            central_scenario = central_scenarios[0]
            central_pv_timeseries = central_scenario.pv_timeseries_15min
        else:
            central_pv_timeseries = scenarios_list[0].pv_timeseries_15min

        logger.info(
            "PV timeseries (15min, central scenario): annual=%.0f kWh",
            float(np.sum(central_pv_timeseries)),
        )

    elif pv_peak_kwp > 0:
        # Legacy flow: fetch all years, compute P50/P90 (hourly)
        logger.info(
            "Fetching PVGIS data (lat=%.4f, lon=%.4f, %s)…",
            latitude,
            longitude,
            pvgis_database,
        )
        client = PVGISClient()
        try:
            yearly_pvgis = client.fetch_hourly_production(
                latitude=latitude,
                longitude=longitude,
                peak_power_kwp=pv_peak_kwp,
                system_loss_pct=0.0,
                mounting_type=mounting_type,
                azimuth_deg=azimuth_deg,
                tilt_deg=tilt_deg,
                pvgis_database=pvgis_database,
            )
        except Exception as exc:
            logger.error("PVGIS fetch failed: %s", exc)
            return 1

        p50_timeseries, p90_timeseries = compute_p50_p90(yearly_pvgis)
        central_pv_timeseries = p50_timeseries
        logger.info(
            "PV timeseries: P50 annual=%.0f kWh, P90 annual=%.0f kWh",
            float(np.sum(p50_timeseries)),
            float(np.sum(p90_timeseries)),
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
    price_csv_path = price_inputs.get("day_ahead_csv", "")
    price_unit = price_inputs.get("price_unit", "eur_per_mwh")
    inflation_on_prices = bool(price_inputs.get("inflation_on_input_data", False))
    price_csv_delimiter = price_inputs.get("csv_separator", CSV_DELIMITER)
    price_csv_decimal = price_inputs.get("csv_decimal", CSV_INPUT_DECIMAL_SEPARATOR)
    price_csv_timestamp_column = price_inputs.get("csv_timestamp_column", CSV_TIMESTAMP_COLUMN)
    price_csv_timestamp_format = price_inputs.get("csv_timestamp_format", None)

    mc_cfg = scenario.monte_carlo
    mc_enabled = scenario.mc_enabled and not args.no_mc

    # Resolve relative CSV path against scenario file directory
    scenario_dir = scenario.path.parent if scenario.path else Path(".")

    lifetime = scenario.lifetime_years

    if scenarios_list:
        # New flow: per-scenario price loading
        # Collect unique CSV columns needed
        required_columns = sorted({s.csv_column for s in scenarios_list})

        # Use the first scenario's CSV path as default
        default_csv = price_csv_path
        default_csv_path = Path(default_csv)
        if not default_csv_path.is_absolute():
            default_csv_path = scenario_dir / default_csv_path

        logger.info("Loading price CSV: %s (columns: %s)", default_csv_path, required_columns)
        try:
            price_data = load_price_csv(
                path=default_csv_path,
                required_columns=required_columns,
                price_unit=price_unit,
                commissioning_year=commissioning_year,
                delimiter=price_csv_delimiter,
                decimal=price_csv_decimal,
                timestamp_column=price_csv_timestamp_column,
                timestamp_format=price_csv_timestamp_format,
            )
        except Exception as exc:
            logger.error("Price CSV load failed: %s", exc)
            return 1

        # Extend prices to lifetime (hourly first, then replicate to 15min)
        extended_prices_hourly = _extend_all_price_columns(
            price_data, required_columns, lifetime, HOURS_PER_YEAR
        )

        # Replicate hourly prices to 15min: each hour repeats 4x
        # NOTE: prices are NOT divided by 4 (price per kWh stays the same)
        extended_prices_15min: dict[str, np.ndarray] = {}
        for col, arr in extended_prices_hourly.items():
            extended_prices_15min[col] = np.repeat(arr, INTERVALS_PER_HOUR)

        # Central scenario column for grid search
        central_column = (
            central_scenario.csv_column
            if central_scenarios
            else scenarios_list[0].csv_column
        )
        spot_prices_yearly = _build_spot_prices_yearly(
            extended_prices_15min[central_column],
            lifetime,
            inflation_rate,
            inflation_on_prices,
            intervals_per_year=ts_intervals_per_year,
        )

        # Build per-scenario price timeseries for MC
        scenario_prices_map: dict[str, list[np.ndarray]] = {}
        scenario_pv_map: dict[str, np.ndarray] = {}
        mc_price_scenarios_from_list: dict[str, dict] = {}
        for sc in scenarios_list:
            scenario_prices_map[sc.name] = _build_spot_prices_yearly(
                extended_prices_15min[sc.csv_column],
                lifetime,
                inflation_rate,
                inflation_on_prices,
                intervals_per_year=ts_intervals_per_year,
            )
            if sc.pv_timeseries_15min is not None:
                scenario_pv_map[sc.name] = sc.pv_timeseries_15min
            mc_price_scenarios_from_list[sc.name] = {
                "csv_column": sc.csv_column,
                "weight": sc.weight,
            }

    else:
        # Legacy flow: single CSV with optional MC price_scenarios
        price_scenarios_cfg = mc_cfg.get("price_scenarios", {}) if mc_enabled else {}

        if price_scenarios_cfg:
            required_columns = [v["csv_column"] for v in price_scenarios_cfg.values()]
        else:
            required_columns = ["MID"]

        csv_path = Path(price_csv_path)
        if not csv_path.is_absolute():
            csv_path = scenario_dir / csv_path

        logger.info("Loading price CSV: %s", csv_path)
        try:
            price_data = load_price_csv(
                path=csv_path,
                required_columns=required_columns,
                price_unit=price_unit,
                commissioning_year=commissioning_year,
                delimiter=price_csv_delimiter,
                decimal=price_csv_decimal,
                timestamp_column=price_csv_timestamp_column,
                timestamp_format=price_csv_timestamp_format,
            )
        except Exception as exc:
            logger.error("Price CSV load failed: %s", exc)
            return 1

        mid_column = required_columns[0]
        extended_prices = _extend_all_price_columns(price_data, required_columns, lifetime)

        spot_prices_yearly = _build_spot_prices_yearly(
            extended_prices[mid_column], lifetime, inflation_rate, inflation_on_prices
        )

        # For MC: build scenario prices
        scenario_prices_map = {}
        scenario_pv_map = {}
        mc_price_scenarios_from_list = {}

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
        pv_degradation_rate=pv_degradation_rate,
        pv_costs_capex=pv_capex_cfg,
        pv_costs_opex=pv_opex_cfg,
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
        spot_prices_yearly=spot_prices_yearly,
        fixed_prices_yearly=fixed_prices_yearly,
        goo_prices_yearly=goo_prices_yearly,
        cap_prices_yearly=cap_prices_yearly,
        lifetime_years=lifetime,
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
        spot_prices_yearly=spot_prices_yearly,
        fixed_prices_yearly=fixed_prices_yearly,
        offline_days_yearly=offline_days_yearly,
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
    # Step 6: Monte Carlo
    # ------------------------------------------------------------------
    mc_result = None
    if mc_enabled:
        mc_iterations = int(mc_cfg.get("iterations", 1000))
        sigma_capex_pv = float(mc_cfg.get("sigma_capex_pv_pct", DEFAULT_MC_SIGMA_CAPEX_PV_PCT)) / 100.0
        sigma_capex_bess = float(mc_cfg.get("sigma_capex_bess_pct", DEFAULT_MC_SIGMA_CAPEX_BESS_PCT)) / 100.0
        sigma_opex_pv = float(mc_cfg.get("sigma_opex_pv_pct", DEFAULT_MC_SIGMA_OPEX_PV_PCT)) / 100.0
        sigma_opex_bess = float(mc_cfg.get("sigma_opex_bess_pct", DEFAULT_MC_SIGMA_OPEX_BESS_PCT)) / 100.0
        sigma_pv_avail = float(mc_cfg.get("sigma_pv_availability_pct", DEFAULT_MC_SIGMA_PV_AVAILABILITY_PCT)) / 100.0
        sigma_bess_avail = float(mc_cfg.get("sigma_bess_availability_pct", DEFAULT_MC_SIGMA_BESS_AVAILABILITY_PCT)) / 100.0

        # Build scenario price mapping
        scenario_prices: dict[str, list[np.ndarray]] = {}
        mc_price_scenarios: dict[str, dict] = {}

        if scenarios_list and scenario_prices_map:
            # New flow: use per-scenario prices from parsed scenarios
            scenario_prices = scenario_prices_map
            mc_price_scenarios = mc_price_scenarios_from_list
        else:
            # Legacy flow
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

        logger.info("Starting Monte Carlo (%d iterations)…", mc_iterations)
        mc_result = run_monte_carlo(
            base_config=grid_search_config,
            optimal=opt,
            mc_params=mc_params,
            scenario_prices=scenario_prices,
            scenario_pv_timeseries=scenario_pv_map if scenario_pv_map else None,
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
    # Step 8: Print summary
    # ------------------------------------------------------------------
    _print_summary(scenario.name, opt, metrics, mc_result)

    return 0


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
