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
    DEFAULT_AFA_YEARS_BESS,
    DEFAULT_AFA_YEARS_PV,
    DEFAULT_BESS_AVAILABILITY_PCT,
    DEFAULT_BESS_DEGRADATION_RATE_PCT,
    DEFAULT_BESS_MAX_SOC_PCT,
    DEFAULT_BESS_MIN_SOC_PCT,
    DEFAULT_BESS_RTE_PCT,
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
    DEFAULT_MC_SIGMA_CAPEX_PCT,
    DEFAULT_MC_SIGMA_OPEX_PCT,
    DEFAULT_MC_SIGMA_PV_YIELD_PCT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PV_DEGRADATION_RATE_PCT,
    DEFAULT_SOLIDARITAETSZUSCHLAG_PCT,
    DEFAULT_SYSTEM_LOSS_PCT,
    HOURS_PER_YEAR,
    MARKETING_TYPE_EEG,
    PPA_TYPE_FLOOR,
    PPA_TYPE_NONE,
    PPA_TYPE_PAY_AS_PRODUCED,
)
from pv_bess_model.config.loader import (
    PriceData,
    ScenarioConfig,
    extend_price_timeseries,
    load_price_csv,
    load_scenario,
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
    write_cashflows_csv,
    write_dispatch_sample_csv,
    write_grid_search_csv,
    write_monte_carlo_csv,
    write_summary_csv,
)
from pv_bess_model.pv.pvgis_client import PVGISClient
from pv_bess_model.pv.timeseries import compute_p50_p90

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
                price += ppa_cfg.goo_premium_eur_per_kwh

        elif ppa_type == PPA_TYPE_PAY_AS_PRODUCED:
            ppa_cfg = ppa_config_from_dict(ppa_dict)
            price = pay_as_produced_price(ppa_cfg, year, inflation_rate)
            if year <= ppa_cfg.duration_years:
                price += ppa_cfg.goo_premium_eur_per_kwh

        fixed_prices.append(price)

    return fixed_prices


def _build_spot_prices_yearly(
    price_array: np.ndarray,
    lifetime_years: int,
    inflation_rate: float,
    apply_inflation: bool,
) -> list[np.ndarray]:
    """Split an extended price array into per-year slices with optional inflation.

    Parameters
    ----------
    price_array:
        Extended price array of length >= ``lifetime_years × HOURS_PER_YEAR`` (€/kWh).
    lifetime_years:
        Number of project years.
    inflation_rate:
        Annual inflation rate as a fraction.
    apply_inflation:
        Whether to scale each year's prices by the annual inflation factor.

    Returns
    -------
    list[np.ndarray]
        One (8760,) array per project year (index 0 = year 1).
    """
    yearly: list[np.ndarray] = []
    for y in range(1, lifetime_years + 1):
        start = (y - 1) * HOURS_PER_YEAR
        end = y * HOURS_PER_YEAR
        year_prices = price_array[start:end].copy()
        if apply_inflation:
            factor = inflate_value(1.0, inflation_rate, y)
            year_prices = year_prices * factor
        yearly.append(year_prices)
    return yearly


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

    # Determine output directory
    output_base = Path(args.output) if args.output else Path(DEFAULT_OUTPUT_DIR)
    output_dir = output_base / scenario.name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # Finance parameters
    finance = scenario.finance
    inflation_rate = float(finance.get("inflation_rate", 0.02))
    leverage_pct = float(finance.get("leverage_pct", 0.0))
    interest_rate_pct = float(finance.get("interest_rate_pct", 4.5))
    loan_tenor_years = int(finance.get("loan_tenor_years", 18))
    discount_rate = float(scenario.project_settings.get("discount_rate", 0.06))
    debt_uses_p90 = bool(finance.get("debt_uses_p90", False))

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
    pv_degradation_rate = float(pv_perf.get("degradation_rate_pct_per_year", 0.4)) / 100.0
    system_loss_pct = float(pv_perf.get("system_loss_pct", 14.0))

    # BESS parameters
    bess = scenario.bess
    bess_perf = bess.get("performance", {})
    bess_rte = float(bess_perf.get("round_trip_efficiency_pct", 88.0)) / 100.0
    bess_min_soc_pct = float(bess_perf.get("min_soc_pct", 10.0))
    bess_max_soc_pct = float(bess_perf.get("max_soc_pct", 90.0))
    bess_degradation_rate = float(bess_perf.get("degradation_rate_pct_per_year", 2.0)) / 100.0
    bess_availability_pct = float(bess_perf.get("bess_availability_pct", 100.0))

    bess_costs = bess.get("costs", {})
    replacement_cfg = bess_costs.get("replacement", {})
    replacement_enabled = bool(replacement_cfg.get("enabled", False))
    replacement_year = int(replacement_cfg.get("year", 0))
    replacement_fixed_eur = float(replacement_cfg.get("fixed_eur", 0.0))
    replacement_eur_per_kw = float(replacement_cfg.get("eur_per_kw", 0.0))
    replacement_eur_per_kwh = float(replacement_cfg.get("eur_per_kwh", 0.0))
    replacement_pct_of_capex = float(replacement_cfg.get("pct_of_capex", 0.0))

    # Grid connection
    grid_connection = scenario.grid_connection
    grid_max_kw = float(grid_connection.get("max_export_kw", pv_peak_kwp))

    # BESS design space (for grid search)
    bess_design_space = bess.get("design_space", {})
    scale_pct_list = [float(v) for v in bess_design_space.get("scale_pct_of_pv", [0.0])]
    e_to_p_list = [float(v) for v in bess_design_space.get("e_to_p_ratio_hours", [2.0])]

    # Handle fixed BESS override from CLI
    if args.bess_power is not None and args.bess_capacity is not None:
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

    # ------------------------------------------------------------------
    # Step 2: Fetch PVGIS data
    # ------------------------------------------------------------------
    location = scenario.project_settings.get("location", {})
    latitude = float(location.get("latitude", 51.0))
    longitude = float(location.get("longitude", 10.0))
    pvgis_database = location.get("pvgis_database", "PVGIS-SARAH2")
    mounting_type = pv_design.get("mounting_type", "free")
    azimuth_deg = float(pv_design.get("azimuth_deg", 0))
    tilt_deg = float(pv_design.get("tilt_deg", 30))

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
            system_loss_pct=system_loss_pct,
            mounting_type=mounting_type,
            azimuth_deg=azimuth_deg,
            tilt_deg=tilt_deg,
            pvgis_database=pvgis_database,
        )
    except Exception as exc:
        logger.error("PVGIS fetch failed: %s", exc)
        return 1

    # ------------------------------------------------------------------
    # Step 3: Compute P50 and P90 timeseries
    # ------------------------------------------------------------------
    p50_timeseries, p90_timeseries = compute_p50_p90(yearly_pvgis)
    logger.info(
        "PV timeseries: P50 annual=%.0f kWh, P90 annual=%.0f kWh",
        float(np.sum(p50_timeseries)),
        float(np.sum(p90_timeseries)),
    )

    # ------------------------------------------------------------------
    # Step 4: Load price CSV, extend to lifetime
    # ------------------------------------------------------------------
    price_inputs = finance.get("price_inputs", {})
    price_csv_path = price_inputs.get("day_ahead_csv", "")
    price_unit = price_inputs.get("price_unit", "eur_per_mwh")
    inflation_on_prices = bool(price_inputs.get("inflation_on_input_data", False))

    # Collect required price columns
    mc_cfg = scenario.monte_carlo
    mc_enabled = scenario.mc_enabled and not args.no_mc

    price_scenarios_cfg = mc_cfg.get("price_scenarios", {}) if mc_enabled else {}

    if price_scenarios_cfg:
        required_columns = [v["csv_column"] for v in price_scenarios_cfg.values()]
    else:
        # Default: use "MID" column, fall back to first available numeric column
        required_columns = ["MID"]

    # Resolve relative CSV path against scenario file directory
    scenario_dir = scenario.path.parent if scenario.path else Path(".")
    csv_path = Path(price_csv_path)
    if not csv_path.is_absolute():
        csv_path = scenario_dir / csv_path

    logger.info("Loading price CSV: %s", csv_path)
    try:
        price_data = load_price_csv(
            path=csv_path,
            required_columns=required_columns,
            price_unit=price_unit,
        )
    except Exception as exc:
        logger.error("Price CSV load failed: %s", exc)
        return 1

    lifetime = scenario.lifetime_years

    # Extend the "mid" price column to full project lifetime
    mid_column = required_columns[0]
    mid_prices_extended = extend_price_timeseries(
        price_data.get_column(mid_column),
        target_years=lifetime,
        hours_per_year=HOURS_PER_YEAR,
    )
    spot_prices_yearly = _build_spot_prices_yearly(
        mid_prices_extended, lifetime, inflation_rate, inflation_on_prices
    )

    # P90 prices (same column unless specifically configured)
    spot_prices_yearly_p90 = spot_prices_yearly  # conservative: same as P50

    # Fixed prices per year (EEG / PPA floor)
    fixed_prices_yearly = _build_fixed_prices_yearly(scenario, inflation_rate)

    # ------------------------------------------------------------------
    # Step 5: Grid Search
    # ------------------------------------------------------------------
    pv_capex_cfg, pv_opex_cfg, bess_capex_cfg, bess_opex_cfg = _extract_cost_dicts(scenario)
    grid_capex_cfg, grid_opex_cfg = _extract_grid_cost_dicts(scenario)

    grid_search_config = GridSearchConfig(
        scale_pct_of_pv=scale_pct_list,
        e_to_p_ratio_hours=e_to_p_list,
        pv_peak_kwp=pv_peak_kwp,
        pv_base_timeseries_p50=p50_timeseries,
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
        grid_max_kw=grid_max_kw,
        grid_costs_capex=grid_capex_cfg,
        grid_costs_opex=grid_opex_cfg,
        operating_mode=scenario.operating_mode,
        spot_prices_yearly=spot_prices_yearly,
        fixed_prices_yearly=fixed_prices_yearly,
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
        debt_uses_p90=debt_uses_p90,
        pv_base_timeseries_p90=p90_timeseries if debt_uses_p90 else None,
        spot_prices_yearly_p90=spot_prices_yearly_p90 if debt_uses_p90 else None,
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
    )
    engine_config = DispatchEngineConfig(
        mode=scenario.operating_mode,
        grid_max_kw=grid_max_kw,
        bess_nameplate_kwh=opt.bess_capacity_kwh,
        bess_max_charge_kw=opt.bess_power_kw,
        bess_max_discharge_kw=opt.bess_power_kw,
        bess_rte=bess_rte,
        bess_min_soc_pct=bess_min_soc_pct,
        bess_max_soc_pct=bess_max_soc_pct,
        bess_degradation_rate=bess_degradation_rate,
        pv_degradation_rate=pv_degradation_rate,
        replacement=replacement,
        lifetime_years=lifetime,
        bess_power_kw=opt.bess_power_kw,
    )
    offline_days = compute_deterministic_offline_days(bess_availability_pct)
    offline_days_yearly = [offline_days] * lifetime

    sim = run_simulation(
        config=engine_config,
        pv_base_timeseries=p50_timeseries,
        spot_prices_yearly=spot_prices_yearly,
        fixed_prices_yearly=fixed_prices_yearly,
        offline_days_yearly=offline_days_yearly,
    )

    annual_revenues = [r.total_revenue for r in sim.annual_results]
    annual_pv_kwh = [r.pv_production for r in sim.annual_results]
    annual_bess_throughput = [r.bess_throughput for r in sim.annual_results]
    total_production_kwh = sum(annual_pv_kwh)

    # Build cashflow
    debt_schedule = build_annuity_schedule(
        total_capex=opt.capex_total,
        leverage_pct=leverage_pct,
        annual_interest_rate=interest_rate_pct / 100.0,
        tenor_years=loan_tenor_years,
    )
    replacement_cost = (
        replacement_fixed_eur
        + replacement_eur_per_kw * opt.bess_power_kw
        + replacement_eur_per_kwh * opt.bess_capacity_kwh
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
    )

    annual_opex = [inflate_value(opt.opex_base, inflation_rate, y) for y in range(1, lifetime + 1)]
    annual_debt_service = [cashflow.years[y].debt_service for y in range(1, lifetime + 1)]
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
        sigma_pv = float(mc_cfg.get("sigma_pv_yield_pct", 5.0)) / 100.0
        sigma_capex = float(mc_cfg.get("sigma_capex_pct", 8.0)) / 100.0
        sigma_opex = float(mc_cfg.get("sigma_opex_pct", 5.0)) / 100.0
        sigma_avail = float(mc_cfg.get("sigma_bess_availability_pct", 2.0)) / 100.0

        # Build scenario price mapping
        scenario_prices: dict[str, list[np.ndarray]] = {}
        if price_scenarios_cfg:
            for name, cfg in price_scenarios_cfg.items():
                col = cfg["csv_column"]
                extended = extend_price_timeseries(
                    price_data.get_column(col),
                    target_years=lifetime,
                    hours_per_year=HOURS_PER_YEAR,
                )
                scenario_prices[name] = _build_spot_prices_yearly(
                    extended, lifetime, inflation_rate, inflation_on_prices
                )
            mc_price_scenarios = {
                k: {"csv_column": v["csv_column"], "weight": float(v["weight"])}
                for k, v in price_scenarios_cfg.items()
            }
        else:
            # Single scenario
            scenario_prices = {"mid": spot_prices_yearly}
            mc_price_scenarios = {"mid": {"csv_column": mid_column, "weight": 1.0}}

        mc_params = MCParams(
            iterations=mc_iterations,
            sigma_pv_yield=sigma_pv,
            sigma_capex=sigma_capex,
            sigma_opex=sigma_opex,
            mu_bess_availability=bess_availability_pct / 100.0,
            sigma_bess_availability=sigma_avail,
            price_scenarios=mc_price_scenarios,
        )

        logger.info("Starting Monte Carlo (%d iterations)…", mc_iterations)
        mc_result = run_monte_carlo(
            base_config=grid_search_config,
            optimal=opt,
            mc_params=mc_params,
            scenario_prices=scenario_prices,
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
    )

    write_cashflows_csv(
        path=output_dir / f"{scenario.name}_cashflows.csv",
        cashflow=cashflow,
        annual_pv_production_kwh=annual_pv_kwh,
        annual_bess_throughput_kwh=annual_bess_throughput,
        annual_dscr=annual_dscr,
    )

    write_grid_search_csv(
        path=output_dir / f"{scenario.name}_grid_search.csv",
        grid_result=grid_result,
    )

    if mc_result is not None:
        write_monte_carlo_csv(
            path=output_dir / f"{scenario.name}_monte_carlo.csv",
            mc_result=mc_result,
        )

    # Dispatch sample (if requested)
    export_dispatch = scenario.raw.get("scenario", {}).get(
        "output", {}
    ).get("export_dispatch_sample", True)
    if export_dispatch:
        write_dispatch_sample_csv(
            path=output_dir / f"{scenario.name}_dispatch_sample.csv",
            hourly_sample=sim.hourly_sample,
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
