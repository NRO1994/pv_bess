"""CLI entrypoint for the Portfolio / Systemwert meta-model.

Evaluates the economic system value of additional flexibility assets
(BESS, heat pumps, EV/V2G) within a utility portfolio by comparing
World A (no flexibility) against World B (with flexibility).

Usage
-----
    python -m pv_bess_model.main_portfolio --config portfolio/systemwert_2027.json
    python -m pv_bess_model.main_portfolio --config portfolio/systemwert_2027.json --dry-run
    python -m pv_bess_model.main_portfolio --config portfolio/systemwert_2027.json -v
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pv_bess_model.config.loader_portfolio import (
    BessFlexConfig,
    EVFlexConfig,
    HeatPumpFlexConfig,
    PortfolioConfig,
    load_portfolio,
)
from pv_bess_model.output.csv_writer_portfolio import (
    write_baseline_csv,
    write_marginal_value_csv,
    write_system_value_csv,
)
from pv_bess_model.output.report.data_collector_portfolio import (
    collect_portfolio_report_data,
)
from pv_bess_model.output.report.html_builder import build_portfolio_html_report
from pv_bess_model.portfolio.marginal_value import MarginalValuePoint, compute_marginal_values
from pv_bess_model.portfolio.system_value import SystemValueResult

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

    # --- Simulation pipeline ------------------------------------------------
    # Phase 2: PVGIS fetch, SLP generation, price loading
    # Phase 3: World A calculation, Portfolio LP
    # Phase 4: Multi-year engine, enumeration, system value
    # Phase 5: Heat pump flex in LP
    # Phase 6: EV/V2G flex in LP
    #
    # Phases 2-6 produce:
    #   - system_value_result: SystemValueResult
    #   - annual_results: list[PortfolioAnnualResult] (optional, for dispatch)
    #
    # These are passed to Phase 7 for output generation.

    print("Simulation pipeline (Phases 2-6) not yet implemented.")
    print("Output generation (Phase 7) is ready and will run once upstream")
    print("phases provide simulation results.")
    return 0


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
