"""Ratio-based grid search over BESS sizing to maximise Equity IRR.

The search space is defined by two dimensions:

1. **BESS scale** (% of PV peak power): How large is the BESS relative to PV?
2. **Energy-to-power ratio** (hours): What is the storage duration?

All combinations are evaluated in parallel. For each combination a full
multi-year dispatch simulation is run (P50 PV timeseries, mid price data),
followed by a complete cashflow and IRR calculation.

Public API
----------
GridSearchConfig    – All inputs required by the grid search.
GridPointResult     – Financial result for one (scale, E/P) combination.
GridSearchResult    – Complete grid search output (all points + optimum).
run_grid_search     – Main entry point.
"""

from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass, field

import numpy as np

from pv_bess_model.bess.replacement import ReplacementConfig
from pv_bess_model.config.defaults import (
    DEFAULT_DISCOUNT_RATE,
    DAYS_PER_YEAR,
    GRID_SEARCH_SCALE_ZERO_PCT,
)
from pv_bess_model.dispatch.engine import (
    DispatchEngineConfig,
    compute_deterministic_offline_days,
    run_simulation,
)
from pv_bess_model.finance.cashflow import build_cashflow_projection
from pv_bess_model.finance.costs import calculate_total_costs
from pv_bess_model.finance.debt import build_annuity_schedule
from pv_bess_model.finance.inflation import inflate_value
from pv_bess_model.finance.metrics import calculate_dscr, compute_all_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GridSearchConfig:
    """Complete configuration for the ratio-based BESS sizing grid search.

    All fields must be primitive types or numpy arrays (serializable for
    multiprocessing). The caller is responsible for pre-loading and extending
    price timeseries to the full project lifetime before constructing this
    object.

    Parameters
    ----------
    scale_pct_of_pv:
        BESS scale percentages relative to PV peak power. Must include 0.0
        (PV-only baseline). Example: [0, 20, 40, 60, 80, 100].
    e_to_p_ratio_hours:
        Energy-to-power ratios in hours. Example: [1, 2, 4].
    pv_peak_kwp:
        PV installed peak power in kWp.
    pv_base_timeseries_p50:
        Undegraded P50 hourly PV production timeseries, shape (8760,) in kWh.
    pv_degradation_rate:
        Annual PV production degradation rate as a fraction (e.g. 0.004).
    pv_costs_capex:
        CAPEX cost config dict for PV (keys: ``fixed_eur``, ``eur_per_kw``).
    pv_costs_opex:
        OPEX cost config dict for PV.
    bess_rte:
        BESS round-trip efficiency as a fraction in (0, 1].
    bess_min_soc_pct:
        Minimum BESS SoC as % of current capacity.
    bess_max_soc_pct:
        Maximum BESS SoC as % of current capacity.
    bess_degradation_rate:
        Annual BESS capacity degradation as a fraction (e.g. 0.02).
    bess_availability_pct:
        BESS availability in % (0–100). Determines deterministic offline days.
    bess_costs_capex:
        CAPEX cost config dict for BESS.
    bess_costs_opex:
        OPEX cost config dict for BESS.
    replacement_enabled:
        Whether mid-life BESS replacement is active.
    replacement_year:
        Project year (1-indexed) of replacement. Ignored if disabled.
    replacement_fixed_eur:
        Fixed replacement cost component in €.
    replacement_eur_per_kw:
        Replacement cost per kW of BESS power (€/kW).
    replacement_eur_per_kwh:
        Replacement cost per kWh of BESS capacity (€/kWh).
    replacement_pct_of_capex:
        Replacement cost as fraction of BESS CAPEX.
    grid_max_kw:
        Maximum grid export power in kW.
    grid_costs_capex:
        CAPEX cost config dict for grid connection.
    grid_costs_opex:
        OPEX cost config dict for grid connection.
    operating_mode:
        Dispatch operating mode: ``"green"`` or ``"grey"``.
    spot_prices_yearly:
        Per-year spot price arrays (each shape (8760,), in €/kWh).
        Length must equal ``lifetime_years``.
    fixed_prices_yearly:
        Per-year EEG/PPA floor prices (€/kWh). 0.0 after the fixed-price
        period. Length must equal ``lifetime_years``.
    lifetime_years:
        Project lifetime in years.
    leverage_pct:
        Debt leverage as % of total CAPEX (e.g. 75.0).
    interest_rate_pct:
        Annual debt interest rate in % (e.g. 4.5).
    loan_tenor_years:
        Loan tenor in years.
    inflation_rate:
        Annual inflation rate as a fraction (e.g. 0.02).
    discount_rate:
        Discount rate for NPV calculation as a fraction.
    afa_years_pv:
        PV depreciation period in years (AfA).
    afa_years_bess:
        BESS depreciation period in years (AfA).
    gewerbesteuer_messzahl:
        German trade tax Messzahl (e.g. 0.035).
    gewerbesteuer_hebesatz:
        German trade tax Hebesatz (e.g. 400).
    debt_uses_p90:
        If True and P90 timeseries is provided, compute DSCR from P90
        production for conservative debt coverage analysis.
    pv_base_timeseries_p90:
        Undegraded P90 hourly PV production timeseries (kWh), or None.
    spot_prices_yearly_p90:
        Per-year spot prices for the P90 simulation. If None, the P50 prices
        are reused.
    max_workers:
        Number of parallel worker processes. None = ``os.cpu_count()``.
    """

    # Design space
    scale_pct_of_pv: list[float]
    e_to_p_ratio_hours: list[float]

    # PV
    pv_peak_kwp: float
    pv_base_timeseries_p50: np.ndarray
    pv_degradation_rate: float
    pv_costs_capex: dict
    pv_costs_opex: dict

    # BESS performance
    bess_rte: float
    bess_min_soc_pct: float
    bess_max_soc_pct: float
    bess_degradation_rate: float
    bess_availability_pct: float
    bess_costs_capex: dict
    bess_costs_opex: dict

    # BESS replacement
    replacement_enabled: bool
    replacement_year: int
    replacement_fixed_eur: float
    replacement_eur_per_kw: float
    replacement_eur_per_kwh: float
    replacement_pct_of_capex: float

    # Grid
    grid_max_kw: float
    grid_costs_capex: dict
    grid_costs_opex: dict

    # Operating mode
    operating_mode: str

    # Pre-computed per-year prices (€/kWh)
    spot_prices_yearly: list[np.ndarray]
    fixed_prices_yearly: list[float]

    # Finance
    lifetime_years: int
    leverage_pct: float
    interest_rate_pct: float
    loan_tenor_years: int
    inflation_rate: float
    discount_rate: float
    afa_years_pv: int
    afa_years_bess: int
    gewerbesteuer_messzahl: float
    gewerbesteuer_hebesatz: float

    # P90 for conservative debt analysis (optional)
    debt_uses_p90: bool = False
    pv_base_timeseries_p90: np.ndarray | None = None
    spot_prices_yearly_p90: list[np.ndarray] | None = None

    # Parallelism
    max_workers: int | None = None


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class GridPointResult:
    """Financial result for one (scale_pct, e_to_p_ratio) combination.

    Attributes
    ----------
    scale_pct:
        BESS scale as % of PV peak power.
    e_to_p_ratio:
        Energy-to-power ratio in hours.
    bess_power_kw:
        Derived BESS power rating in kW.
    bess_capacity_kwh:
        Derived BESS energy capacity in kWh.
    capex_total:
        Total project CAPEX in €.
    capex_pv:
        PV CAPEX in €.
    capex_bess:
        BESS CAPEX in €.
    opex_base:
        Base-year total OPEX (before inflation) in €/year.
    revenue_year1:
        Total revenue in the first project year in €.
    equity_irr:
        Post-leverage, post-tax Equity IRR, or None if not computable.
    project_irr:
        Pre-leverage Project IRR, or None if not computable.
    npv:
        NPV at the configured discount rate in €.
    dscr_min:
        Minimum DSCR over the loan tenor (P90 if ``debt_uses_p90``).
    dscr_avg:
        Average DSCR over the loan tenor (P90 if ``debt_uses_p90``).
    is_optimal:
        True for the combination with the highest Equity IRR.
    """

    scale_pct: float
    e_to_p_ratio: float
    bess_power_kw: float
    bess_capacity_kwh: float
    capex_total: float
    capex_pv: float
    capex_bess: float
    opex_base: float
    revenue_year1: float
    equity_irr: float | None
    project_irr: float | None
    npv: float
    dscr_min: float | None = None
    dscr_avg: float | None = None
    is_optimal: bool = False


@dataclass
class GridSearchResult:
    """Complete result of the BESS sizing grid search.

    Attributes
    ----------
    points:
        All evaluated (scale, E/P) combinations including the PV-only baseline.
    optimal:
        The combination with the highest Equity IRR, or None if all IRRs are
        None.
    """

    points: list[GridPointResult]
    optimal: GridPointResult | None


# ---------------------------------------------------------------------------
# Internal worker helpers
# ---------------------------------------------------------------------------


@dataclass
class _GridPointArgs:
    """All parameters needed to evaluate one grid point (pickle-safe).

    Every field must be serialisable so instances can be sent to worker
    processes via ``concurrent.futures.ProcessPoolExecutor``.
    """

    scale_pct: float
    e_to_p_ratio: float
    bess_power_kw: float
    bess_capacity_kwh: float

    # Engine config
    operating_mode: str
    grid_max_kw: float
    bess_rte: float
    bess_min_soc_pct: float
    bess_max_soc_pct: float
    bess_degradation_rate: float
    pv_degradation_rate: float
    replacement_enabled: bool
    replacement_year: int
    replacement_fixed_eur: float
    replacement_eur_per_kw: float
    replacement_eur_per_kwh: float
    lifetime_years: int

    # PV
    pv_base_timeseries: np.ndarray  # shape (8760,)

    # Prices per year
    spot_prices_yearly: list  # list[np.ndarray]
    fixed_prices_yearly: list  # list[float]
    offline_days_yearly: list  # list[set[int]]

    # Pre-computed costs
    capex_pv: float
    capex_bess: float
    capex_grid: float
    capex_other: float
    capex_total: float
    opex_base: float
    replacement_cost: float

    # Finance
    leverage_pct: float
    interest_rate_pct: float
    loan_tenor_years: int
    inflation_rate: float
    discount_rate: float
    afa_years_pv: int
    afa_years_bess: int
    gewerbesteuer_messzahl: float
    gewerbesteuer_hebesatz: float

    # P90 (optional)
    pv_base_timeseries_p90: np.ndarray | None = None
    spot_prices_yearly_p90: list | None = None  # list[np.ndarray] | None


def _evaluate_grid_point(args: _GridPointArgs) -> GridPointResult:
    """Evaluate a single (scale, E/P ratio) grid point.

    This is a module-level function so it can be pickled and sent to worker
    processes. All necessary data is contained in *args*.

    Parameters
    ----------
    args:
        Packed arguments for this grid point.

    Returns
    -------
    GridPointResult
        Financial metrics for this BESS sizing combination.
    """
    replacement = ReplacementConfig(
        enabled=args.replacement_enabled,
        year=args.replacement_year,
        fixed_eur=args.replacement_fixed_eur,
        eur_per_kw=args.replacement_eur_per_kw,
        eur_per_kwh=args.replacement_eur_per_kwh,
    )

    engine_config = DispatchEngineConfig(
        mode=args.operating_mode,
        grid_max_kw=args.grid_max_kw,
        bess_nameplate_kwh=args.bess_capacity_kwh,
        bess_max_charge_kw=args.bess_power_kw,
        bess_max_discharge_kw=args.bess_power_kw,
        bess_rte=args.bess_rte,
        bess_min_soc_pct=args.bess_min_soc_pct,
        bess_max_soc_pct=args.bess_max_soc_pct,
        bess_degradation_rate=args.bess_degradation_rate,
        pv_degradation_rate=args.pv_degradation_rate,
        replacement=replacement,
        lifetime_years=args.lifetime_years,
        bess_power_kw=args.bess_power_kw,
    )

    # P50 simulation – used for equity cashflows
    sim_p50 = run_simulation(
        config=engine_config,
        pv_base_timeseries=args.pv_base_timeseries,
        spot_prices_yearly=args.spot_prices_yearly,
        fixed_prices_yearly=args.fixed_prices_yearly,
        offline_days_yearly=args.offline_days_yearly,
    )

    annual_revenues_p50 = [r.total_revenue for r in sim_p50.annual_results]
    total_production_kwh = sum(r.pv_production for r in sim_p50.annual_results)

    # Optional P90 simulation – used for conservative DSCR calculation
    annual_revenues_p90: list[float] | None = None
    if args.pv_base_timeseries_p90 is not None:
        p90_prices = args.spot_prices_yearly_p90 or args.spot_prices_yearly
        sim_p90 = run_simulation(
            config=engine_config,
            pv_base_timeseries=args.pv_base_timeseries_p90,
            spot_prices_yearly=p90_prices,
            fixed_prices_yearly=args.fixed_prices_yearly,
            offline_days_yearly=args.offline_days_yearly,
        )
        annual_revenues_p90 = [r.total_revenue for r in sim_p90.annual_results]

    # Debt schedule (always based on CAPEX × leverage)
    debt_schedule = build_annuity_schedule(
        total_capex=args.capex_total,
        leverage_pct=args.leverage_pct,
        annual_interest_rate=args.interest_rate_pct / 100.0,
        tenor_years=args.loan_tenor_years,
    )

    # Cashflow projection (P50 revenues → equity IRR / NPV)
    replacement_cost = args.replacement_cost if args.replacement_enabled else 0.0
    replacement_year_cf: int | None = (
        args.replacement_year if args.replacement_enabled else None
    )
    cf = build_cashflow_projection(
        lifetime_years=args.lifetime_years,
        annual_revenues=annual_revenues_p50,
        base_opex=args.opex_base,
        inflation_rate=args.inflation_rate,
        capex_total=args.capex_total,
        capex_pv=args.capex_pv,
        capex_bess=args.capex_bess,
        debt_schedule=debt_schedule,
        afa_years_pv=args.afa_years_pv,
        afa_years_bess=args.afa_years_bess,
        gewerbesteuer_messzahl=args.gewerbesteuer_messzahl,
        gewerbesteuer_hebesatz=args.gewerbesteuer_hebesatz,
        replacement_cost=replacement_cost,
        replacement_year=replacement_year_cf,
    )

    # Per-year OPEX list (inflation-adjusted) for DSCR computation
    annual_opex = [
        inflate_value(args.opex_base, args.inflation_rate, y)
        for y in range(1, args.lifetime_years + 1)
    ]
    annual_debt_service = [
        cf.years[y].debt_service for y in range(1, args.lifetime_years + 1)
    ]
    total_opex_lifetime = sum(annual_opex)

    # Primary metrics (P50 revenues)
    metrics = compute_all_metrics(
        equity_cashflows=cf.equity_cashflows,
        project_cashflows=cf.project_cashflows,
        annual_revenues=annual_revenues_p50,
        annual_opex=annual_opex,
        annual_debt_service=annual_debt_service,
        total_capex=args.capex_total,
        total_opex_lifetime=total_opex_lifetime,
        total_production_kwh=total_production_kwh,
        discount_rate=args.discount_rate,
    )

    dscr_min = metrics.dscr_min
    dscr_avg = metrics.dscr_avg

    # Override DSCR with P90 revenues for conservative debt coverage
    if annual_revenues_p90 is not None:
        dscr_min, dscr_avg = calculate_dscr(
            annual_revenues=annual_revenues_p90,
            annual_opex=annual_opex,
            annual_debt_service=annual_debt_service,
        )

    revenue_year1 = annual_revenues_p50[0] if annual_revenues_p50 else 0.0

    return GridPointResult(
        scale_pct=args.scale_pct,
        e_to_p_ratio=args.e_to_p_ratio,
        bess_power_kw=args.bess_power_kw,
        bess_capacity_kwh=args.bess_capacity_kwh,
        capex_total=args.capex_total,
        capex_pv=args.capex_pv,
        capex_bess=args.capex_bess,
        opex_base=args.opex_base,
        revenue_year1=revenue_year1,
        equity_irr=metrics.equity_irr,
        project_irr=metrics.project_irr,
        npv=metrics.npv,
        dscr_min=dscr_min,
        dscr_avg=dscr_avg,
        is_optimal=False,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_grid_search(config: GridSearchConfig) -> GridSearchResult:
    """Run the ratio-based BESS sizing grid search.

    Evaluates all ``(scale_pct, e_to_p_ratio)`` combinations in parallel,
    runs a full multi-year dispatch simulation for each, builds a complete
    cashflow projection, and identifies the combination with the highest
    Equity IRR.

    The PV-only baseline (scale = 0 %) is always included even if not
    explicitly listed in ``config.scale_pct_of_pv``.

    Parameters
    ----------
    config:
        Complete grid search configuration.

    Returns
    -------
    GridSearchResult
        All evaluated combinations and the identified optimum.
    """
    # Ensure PV-only baseline is included
    scales = list(config.scale_pct_of_pv)
    if GRID_SEARCH_SCALE_ZERO_PCT not in scales:
        scales = [GRID_SEARCH_SCALE_ZERO_PCT] + scales
        logger.info("Added scale=0 %% (PV-only baseline) to grid search.")

    # Deterministic offline days – same for every grid point
    offline_days: set[int] = compute_deterministic_offline_days(
        config.bess_availability_pct
    )
    offline_days_yearly: list[set[int]] = [
        offline_days for _ in range(config.lifetime_years)
    ]

    # Build worker args for every (scale, E/P) combination
    worker_args: list[_GridPointArgs] = []
    for scale_pct in scales:
        for e_to_p in config.e_to_p_ratio_hours:
            bess_power_kw = config.pv_peak_kwp * scale_pct / 100.0
            bess_capacity_kwh = bess_power_kw * e_to_p

            # Cost configs aggregated for calculate_total_costs
            capex_cfg = {
                "pv": config.pv_costs_capex,
                "bess": config.bess_costs_capex,
                "grid": config.grid_costs_capex,
            }
            opex_cfg = {
                "pv": config.pv_costs_opex,
                "bess": config.bess_costs_opex,
                "grid": config.grid_costs_opex,
            }
            costs = calculate_total_costs(
                capex_config=capex_cfg,
                opex_config=opex_cfg,
                pv_peak_kwp=config.pv_peak_kwp,
                bess_power_kw=bess_power_kw,
                bess_capacity_kwh=bess_capacity_kwh,
                grid_max_export_kw=config.grid_max_kw,
            )

            # Replacement cost (constant per kW/kWh regardless of scale)
            replacement_cost = (
                config.replacement_fixed_eur
                + config.replacement_eur_per_kw * bess_power_kw
                + config.replacement_eur_per_kwh * bess_capacity_kwh
                + config.replacement_pct_of_capex * costs.capex_bess
            )

            worker_args.append(
                _GridPointArgs(
                    scale_pct=scale_pct,
                    e_to_p_ratio=e_to_p,
                    bess_power_kw=bess_power_kw,
                    bess_capacity_kwh=bess_capacity_kwh,
                    operating_mode=config.operating_mode,
                    grid_max_kw=config.grid_max_kw,
                    bess_rte=config.bess_rte,
                    bess_min_soc_pct=config.bess_min_soc_pct,
                    bess_max_soc_pct=config.bess_max_soc_pct,
                    bess_degradation_rate=config.bess_degradation_rate,
                    pv_degradation_rate=config.pv_degradation_rate,
                    replacement_enabled=config.replacement_enabled,
                    replacement_year=config.replacement_year,
                    replacement_fixed_eur=config.replacement_fixed_eur,
                    replacement_eur_per_kw=config.replacement_eur_per_kw,
                    replacement_eur_per_kwh=config.replacement_eur_per_kwh,
                    lifetime_years=config.lifetime_years,
                    pv_base_timeseries=config.pv_base_timeseries_p50,
                    spot_prices_yearly=config.spot_prices_yearly,
                    fixed_prices_yearly=config.fixed_prices_yearly,
                    offline_days_yearly=offline_days_yearly,
                    capex_pv=costs.capex_pv,
                    capex_bess=costs.capex_bess,
                    capex_grid=costs.capex_grid,
                    capex_other=costs.capex_other,
                    capex_total=costs.capex_total,
                    opex_base=costs.opex_total,
                    replacement_cost=replacement_cost,
                    leverage_pct=config.leverage_pct,
                    interest_rate_pct=config.interest_rate_pct,
                    loan_tenor_years=config.loan_tenor_years,
                    inflation_rate=config.inflation_rate,
                    discount_rate=config.discount_rate,
                    afa_years_pv=config.afa_years_pv,
                    afa_years_bess=config.afa_years_bess,
                    gewerbesteuer_messzahl=config.gewerbesteuer_messzahl,
                    gewerbesteuer_hebesatz=config.gewerbesteuer_hebesatz,
                    pv_base_timeseries_p90=(
                        config.pv_base_timeseries_p90
                        if config.debt_uses_p90
                        else None
                    ),
                    spot_prices_yearly_p90=(
                        config.spot_prices_yearly_p90
                        if config.debt_uses_p90
                        else None
                    ),
                )
            )

    n_combinations = len(worker_args)
    logger.info(
        "Grid search: %d combinations (%d scales × %d E/P ratios).",
        n_combinations,
        len(scales),
        len(config.e_to_p_ratio_hours),
    )

    # Parallel evaluation
    results: list[GridPointResult] = []
    if config.max_workers == 1:
        # Single-process execution (simpler for debugging / unit tests)
        results = [_evaluate_grid_point(a) for a in worker_args]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=config.max_workers
        ) as executor:
            futures = {
                executor.submit(_evaluate_grid_point, a): a for a in worker_args
            }
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

    # Sort results by (scale_pct, e_to_p_ratio) for deterministic output order
    results.sort(key=lambda r: (r.scale_pct, r.e_to_p_ratio))

    # Identify optimum: highest Equity IRR (None treated as -inf)
    optimal: GridPointResult | None = None
    best_irr: float = float("-inf")
    for r in results:
        irr = r.equity_irr if r.equity_irr is not None else float("-inf")
        if irr > best_irr:
            best_irr = irr
            optimal = r

    if optimal is not None:
        optimal.is_optimal = True
        logger.info(
            "Grid search optimum: scale=%.0f %%, E/P=%.1f h, Equity IRR=%.2f %%.",
            optimal.scale_pct,
            optimal.e_to_p_ratio,
            (optimal.equity_irr or 0.0) * 100,
        )
    else:
        logger.warning("Grid search: no valid Equity IRR found in any combination.")

    return GridSearchResult(points=results, optimal=optimal)
