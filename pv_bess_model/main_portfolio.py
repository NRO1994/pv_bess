"""CLI entrypoint for the Portfolio / Systemwert meta-model.

Evaluates the economic system value of additional flexibility assets
(BESS, heat pumps, EV/V2G) within a utility portfolio by comparing
World A (no flexibility) against World B (with flexibility).

Usage
-----
    python -m pv_bess_model.main_portfolio --config portfolio/systemwert_2027.json
    python -m pv_bess_model.main_portfolio --config portfolio/systemwert_2027.json --dry-run
    python -m pv_bess_model.main_portfolio --config portfolio/systemwert_2027.json -v

Pipeline
--------
1.  Load & validate portfolio JSON
2.  Fetch PVGIS data (PV + temperature) per weather_year
3.  Generate SLP profiles per weather_year
4.  Compute heat demand profiles (if heat pump flex present)
5.  Load price timeseries
6.  Compute World A (central scenario, no flexibility)
7.  Run enumeration: all flex × addition_rate × E/P (parallelised)
8.  Compute marginal value curves
9.  Write CSV files
10. Generate HTML dashboard
11. Print summary to stdout
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from pv_bess_model.config.defaults import INTERVALS_PER_HOUR
from pv_bess_model.config.loader_portfolio import (
    BessFlexConfig,
    EVFlexConfig,
    HeatPumpFlexConfig,
    PortfolioConfig,
    load_portfolio,
)
from pv_bess_model.dispatch.engine_portfolio import PortfolioEngineConfig
from pv_bess_model.market.price_loader import load_market_prices
from pv_bess_model.output.csv_writer_portfolio import (
    write_baseline_csv,
    write_marginal_value_csv,
    write_system_value_csv,
)
from pv_bess_model.output.report.data_collector_portfolio import (
    collect_portfolio_report_data,
)
from pv_bess_model.output.report.html_builder import build_portfolio_html_report
from pv_bess_model.portfolio.generation import build_aggregated_pv_profile
from pv_bess_model.portfolio.heat_demand import (
    compute_cop,
    compute_daily_heat_demand,
    compute_heat_demand,
)
from pv_bess_model.portfolio.load_profiles import generate_slp, scale_slp
from pv_bess_model.portfolio.marginal_value import (
    MarginalValuePoint,
    compute_marginal_values,
)
from pv_bess_model.portfolio.system_value import (
    SystemValueResult,
    run_enumeration,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="python -m pv_bess_model.main_portfolio",
        description="Portfolio / Systemwert – Flexibility Value Assessment",
    )
    p.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to portfolio configuration JSON file.",
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
        help="Skip HTML report generation.",
    )
    return p


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------


def _print_config_summary(config: PortfolioConfig) -> None:
    """Print a human-readable summary of the loaded configuration."""
    meta = config.meta
    print("\n" + "=" * 70)
    print(f"  Portfolio / Systemwert: {meta.name}")
    print("=" * 70)
    print(f"  Baseline year:          {meta.baseline_year}")
    print(f"  Project lifetime:       {meta.project_lifetime_years} years")
    print(f"  Foresight discount:     {meta.perfect_foresight_discount:.0%}")
    print(f"  Bundesland:             {meta.bundesland}")

    print(f"\n  Generation ({len(config.generation)} asset(s)):")
    for g in config.generation:
        print(f"    - {g.name}: {g.peak_power_kwp:.0f} kWp "
              f"({g.latitude:.2f}°N, {g.longitude:.2f}°E)")

    print(f"\n  Load ({len(config.load)} group(s)):")
    for lg in config.load:
        total_mwh = (
            lg.customer_count
            * lg.annual_consumption_kwh_per_customer
            / 1000.0
        )
        print(f"    - {lg.name}: {lg.customer_count} customers × "
              f"{lg.annual_consumption_kwh_per_customer:.0f} kWh/a "
              f"= {total_mwh:,.0f} MWh/a (SLP {lg.slp_type})")

    print(f"\n  Flexibilities ({len(config.flexibilities)} instance(s)):")
    for flex in config.flexibilities:
        if isinstance(flex, BessFlexConfig):
            n_points = len(flex.annual_addition_kw) * len(flex.e_to_p_ratio_hours)
            print(f"    - [{flex.type}] {flex.name}: "
                  f"{len(flex.annual_addition_kw)} rates × "
                  f"{len(flex.e_to_p_ratio_hours)} E/P = {n_points} points")
        elif isinstance(flex, HeatPumpFlexConfig):
            print(f"    - [{flex.type}] {flex.name}: "
                  f"{len(flex.annual_addition_kw)} rates, "
                  f"COP={flex.cop_nominal:.1f}")
        elif isinstance(flex, EVFlexConfig):
            print(f"    - [{flex.type}] {flex.name}: "
                  f"{len(flex.annual_additional_units)} rates, "
                  f"{flex.mean_kw_per_unit:.0f} kW/unit"
                  f"{' (V2G)' if flex.v2g_enabled else ''}")

    print(f"\n  Price scenarios ({len(config.price_scenarios)}):")
    for ps in config.price_scenarios:
        central = " [CENTRAL]" if ps.is_central else ""
        print(f"    - {ps.name} (weather={ps.weather_year}, "
              f"w={ps.weight:.2f}){central}")

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the portfolio/Systemwert simulation pipeline.

    Parameters
    ----------
    argv:
        Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # --- Logging -----------------------------------------------------------
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Load and validate config ------------------------------------------
    try:
        config = load_portfolio(args.config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Configuration error: %s", exc)
        return 1
    except Exception as exc:
        logger.error("Failed to load portfolio config: %s", exc)
        return 1

    _print_config_summary(config)

    if args.dry_run:
        print("Dry run: JSON validated successfully. Exiting.")
        return 0

    t_start = time.time()
    meta = config.meta

    # --- Step 2: Fetch PVGIS data (PV + temperature) -----------------------
    # Use the central price-weather scenario's weather year
    central_scenario = _get_central_scenario(config)
    if central_scenario is None:
        logger.error("No central price-weather scenario found (is_central=true).")
        return 1

    weather_year = central_scenario.weather_year
    logger.info("Fetching PVGIS data for weather year %d ...", weather_year)

    pv_profile_qh, temperature_hourly = build_aggregated_pv_profile(
        generation_configs=config.generation,
        weather_year=weather_year,
    )
    logger.info(
        "PV profile: %.0f MWh/a (35,040 intervals), temperature: %d hours",
        float(np.sum(pv_profile_qh)) / 1000.0,
        len(temperature_hourly),
    )

    # --- Step 3: Generate SLP profiles -------------------------------------
    logger.info("Generating SLP load profiles ...")
    load_profile_qh = np.zeros_like(pv_profile_qh)
    for lg in config.load:
        slp_norm = generate_slp(
            slp_type=lg.slp_type,
            year=weather_year,
            bundesland=meta.bundesland,
        )
        scaled = scale_slp(
            slp_norm,
            annual_consumption_kwh=lg.annual_consumption_kwh_per_customer,
            customer_count=lg.customer_count,
        )
        load_profile_qh += scaled

    logger.info(
        "Load profile: %.0f MWh/a (aggregated %d groups)",
        float(np.sum(load_profile_qh)) / 1000.0,
        len(config.load),
    )

    # --- Step 4: Heat demand profiles (if WP flex present) -----------------
    # Heat demand and COP profiles are computed here for logging; they are
    # also computed inside the engine on a per-year basis with scaling.
    has_hp = any(isinstance(f, HeatPumpFlexConfig) for f in config.flexibilities)
    if has_hp:
        logger.info("Computing heat demand / COP profiles ...")
        for flex in config.flexibilities:
            if isinstance(flex, HeatPumpFlexConfig):
                heat_demand_qh = compute_heat_demand(
                    temperature_hourly,
                    annual_thermal_demand_mwh=flex.annual_thermal_demand_mwh,
                )
                cop_profile = compute_cop(
                    temperature_hourly,
                    cop_nominal=flex.cop_nominal,
                    cop_reference_temp_c=flex.cop_reference_temp_c,
                )
                daily_heat = compute_daily_heat_demand(heat_demand_qh)
                logger.info(
                    "  WP '%s': thermal demand %.0f MWh/a, mean COP %.1f",
                    flex.name,
                    float(np.sum(heat_demand_qh)) / 1000.0,
                    float(np.mean(cop_profile)),
                )

    # --- Step 5: Load price timeseries ------------------------------------
    logger.info("Loading price timeseries ...")
    csv_path = central_scenario.price_csv
    if csv_path is None:
        logger.error("Central scenario has no price_csv defined.")
        return 1

    # Resolve relative path from config file location
    if config.path is not None and not Path(csv_path).is_absolute():
        csv_path = str(config.path.parent / csv_path)

    market_prices = load_market_prices(
        csv_path=csv_path,
        required_columns=[central_scenario.csv_column],
        lifetime_years=meta.project_lifetime_years,
        commissioning_year=meta.baseline_year,
        delimiter=central_scenario.csv_separator or ";",
        decimal=central_scenario.csv_decimal or ",",
        timestamp_column=central_scenario.csv_timestamp_column or "timestamp",
        timestamp_format=central_scenario.csv_timestamp_format,
    )

    # Get year-1 prices as base for the engine (EUR/kWh, 8760 hourly values)
    spot_prices_hourly = market_prices.get_year_prices(
        central_scenario.csv_column, year=1
    )
    # Convert hourly prices to quarter-hourly by repeating each value 4x
    spot_prices_qh = np.repeat(spot_prices_hourly, INTERVALS_PER_HOUR)

    logger.info(
        "Prices loaded: column='%s', mean=%.1f EUR/MWh",
        central_scenario.csv_column,
        float(np.mean(spot_prices_hourly)) * 1000.0,
    )

    # --- Step 6+7: World A + Enumeration -----------------------------------
    logger.info("Starting enumeration (World A + all flex combinations) ...")

    # Compute aggregate PV degradation rate (weighted average across assets)
    pv_degradation_rate = _compute_avg_pv_degradation(config)

    # Compute aggregate load growth factor (weighted average across groups)
    load_growth_factor = _compute_avg_load_growth(config)

    engine_config = PortfolioEngineConfig(
        lifetime_years=meta.project_lifetime_years,
        baseline_year=meta.baseline_year,
        perfect_foresight_discount=meta.perfect_foresight_discount,
    )

    system_value_result = run_enumeration(
        config=engine_config,
        pv_profile_base=pv_profile_qh,
        load_profile_base=load_profile_qh,
        spot_prices_base=spot_prices_qh,
        flexibilities=config.flexibilities,
        pv_degradation_rate=pv_degradation_rate,
        load_growth_factor=load_growth_factor,
    )

    # --- Step 8: Marginal value curves ------------------------------------
    logger.info("Computing marginal value curves ...")
    marginal_values = compute_marginal_values(system_value_result.points)
    logger.info("Computed %d marginal value points.", len(marginal_values))

    # --- Steps 9-11: Output -----------------------------------------------
    write_output(
        config=config,
        system_value_result=system_value_result,
        marginal_values=marginal_values,
        generate_report=not args.no_report,
    )

    # --- Summary -----------------------------------------------------------
    elapsed = time.time() - t_start
    _print_results_summary(system_value_result, marginal_values, elapsed)

    return 0


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _get_central_scenario(config: PortfolioConfig):
    """Return the central price-weather scenario, or None."""
    for ps in config.price_scenarios:
        if ps.is_central:
            return ps
    return None


def _compute_avg_pv_degradation(config: PortfolioConfig) -> float:
    """Compute weighted average PV degradation rate across generation assets."""
    if not config.generation:
        return 0.0
    total_kwp = sum(g.peak_power_kwp for g in config.generation)
    if total_kwp == 0:
        return 0.0
    weighted = sum(
        g.degradation_rate_pct_per_year / 100.0 * g.peak_power_kwp
        for g in config.generation
    )
    return weighted / total_kwp


def _compute_avg_load_growth(config: PortfolioConfig) -> float:
    """Compute weighted average load growth factor across load groups."""
    if not config.load:
        return 1.0
    total_kwh = sum(
        lg.customer_count * lg.annual_consumption_kwh_per_customer
        for lg in config.load
    )
    if total_kwh == 0:
        return 1.0
    weighted = sum(
        lg.annual_growth_factor
        * lg.customer_count
        * lg.annual_consumption_kwh_per_customer
        for lg in config.load
    )
    return weighted / total_kwh


def _print_results_summary(
    sv_result: SystemValueResult,
    marginals: list[MarginalValuePoint],
    elapsed: float,
) -> None:
    """Print a human-readable results summary to stdout."""
    print("\n" + "=" * 70)
    print("  ERGEBNISSE")
    print("=" * 70)

    wa_total = sum(sv_result.world_a_annual_costs)
    print(f"  Welt A Gesamtkosten:     {wa_total:>14,.0f} EUR")
    print(f"  Enumerations-Punkte:     {len(sv_result.points):>14d}")

    if sv_result.points:
        best = max(sv_result.points, key=lambda p: p.cumulative_system_value_eur)
        print(f"\n  Bester Punkt:")
        print(f"    Flex:                  {best.flex_name}")
        print(f"    Zubau-Rate:            {best.annual_addition_kw:>10.0f} kW/a")
        if best.e_to_p_ratio is not None:
            print(f"    E/P-Verhältnis:        {best.e_to_p_ratio:>10.1f} h")
        print(f"    Systemwert:            {best.cumulative_system_value_eur:>14,.0f} EUR")

    if marginals:
        max_mv = max(marginals, key=lambda m: m.marginal_value_eur_per_kw_a)
        print(f"\n  Höchster Grenznutzen:")
        print(f"    Flex:                  {max_mv.flex_name}")
        print(f"    Zubau-Rate:            {max_mv.annual_addition_kw:>10.0f} kW/a")
        print(f"    Grenznutzen:           {max_mv.marginal_value_eur_per_kw_a:>14,.0f} EUR/kW/a")

    print(f"\n  Laufzeit:                {elapsed:>14.1f} s")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Output generation (Phase 7)
# ---------------------------------------------------------------------------


def write_output(
    config: PortfolioConfig,
    system_value_result: SystemValueResult,
    marginal_values: list[MarginalValuePoint],
    generate_report: bool = True,
    annual_results: list | None = None,
) -> None:
    """Write all portfolio output files (CSV + HTML dashboard).

    Parameters
    ----------
    config:
        Validated portfolio configuration.
    system_value_result:
        Complete enumeration result from ``run_enumeration()``.
    marginal_values:
        Marginal value points from ``compute_marginal_values()``.
    generate_report:
        Whether to generate the HTML dashboard report.
    annual_results:
        Annual results from a representative simulation (for dispatch summary).
    """
    meta = config.meta
    output_dir = Path(meta.output_directory or f".data/output/{meta.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    delimiter = meta.csv_separator
    decimal = meta.csv_decimal

    # 1. Baseline CSV (World A)
    baseline_path = output_dir / f"{meta.name}_baseline.csv"
    write_baseline_csv(
        baseline_path,
        system_value_result.world_a_annual_costs,
        baseline_year=meta.baseline_year,
        delimiter=delimiter,
        decimal=decimal,
    )

    # 2. System value CSV
    sv_path = output_dir / f"{meta.name}_system_value.csv"
    write_system_value_csv(
        sv_path,
        system_value_result,
        delimiter=delimiter,
        decimal=decimal,
    )

    # 3. Marginal value CSV
    mv_path = output_dir / f"{meta.name}_marginal_value.csv"
    write_marginal_value_csv(
        mv_path,
        marginal_values,
        delimiter=delimiter,
        decimal=decimal,
    )

    # 4. Dispatch sample CSV (if enabled and data available)
    # Dispatch CSV requires daily results from the engine, which are
    # produced by the simulation pipeline (Phases 3-6). The dispatch
    # sample writer is called from the pipeline, not from here.

    print(f"\n  CSV output written to: {output_dir}")
    print(f"    - {baseline_path.name}")
    print(f"    - {sv_path.name}")
    print(f"    - {mv_path.name}")

    # 5. HTML dashboard
    if generate_report:
        report_data = collect_portfolio_report_data(
            config=config,
            system_value_result=system_value_result,
            marginal_values=marginal_values,
            annual_results=annual_results,
        )
        report_path = build_portfolio_html_report(report_data, output_dir)
        print(f"    - {report_path.name}")

    print()


if __name__ == "__main__":
    sys.exit(main())
