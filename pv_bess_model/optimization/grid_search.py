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
    GRID_SEARCH_SCALE_ZERO_PCT,
    HOURS_PER_DAY,
    HOURS_PER_YEAR,
)
from pv_bess_model.dispatch.engine import (
    DispatchEngineConfig,
    compute_deterministic_offline_days,
    run_simulation, SimulationResult,
)
from pv_bess_model.finance.cashflow import build_cashflow_projection, CashflowProjection
from pv_bess_model.finance.costs import calculate_total_costs
from pv_bess_model.finance.debt import build_annuity_schedule

from pv_bess_model.finance.metrics import calculate_dscr, compute_all_metrics, FinancialMetrics

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
    pv_base_timeseries:
        Undegraded PV production timeseries (central scenario), shape
        ``(intervals_per_year,)`` in kWh.  Typically 35 040 values at 15-min
        resolution or 8 760 at hourly resolution.
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
        Replacement cost per kWh of the **new** (post-upgrade) BESS capacity (€/kWh).
    replacement_pct_of_capex:
        Replacement cost as fraction of BESS CAPEX.
    replacement_capacity_factor_pct:
        Capacity upgrade multiplier in percent (100 = no upgrade, 120 = 20 % larger
        replacement unit). Applied to the nameplate capacity after replacement.
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
    baseload_mw:
        in PPA agreed baseload level
    lifetime_years:
        Project lifetime in years.
    leverage_pct:
        Debt leverage as % of total CAPEX (e.g. 75.0).
    interest_rate_pct:
        Annual debt interest rate in % (e.g. 4.5).
    loan_tenor_years:
        Loan tenor in years.
    opex_inflation_factors:
        Per-year cumulative inflation factors for OPEX (length = lifetime_years).
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
    debt_sizing_downside_pct:
        Percentage by which PV production is reduced for conservative DSCR
        calculation (e.g. 10.0 means 10 % downside).  0.0 disables.
    timestep_hours:
        Duration of one dispatch interval in hours (0.25 for 15-min).
    intervals_per_day:
        Number of dispatch intervals per day (96 for 15-min, 24 for hourly).
    intervals_per_year:
        Number of dispatch intervals per year (35 040 for 15-min, 8 760 for hourly).
    max_workers:
        Number of parallel worker processes. None = ``os.cpu_count()``.
    bess_absolute_power_kw:
        Absolute BESS power in kW for BESS-Only scenarios (``pv_peak_kwp``
        = 0).  When set, all non-zero scale entries in ``scale_pct_of_pv``
        use this value instead of the ratio-derived power.  Must be
        provided together with ``bess_absolute_capacity_kwh``.
    bess_absolute_capacity_kwh:
        Absolute BESS energy capacity in kWh for BESS-Only scenarios.
        Paired with ``bess_absolute_power_kw`` (both or neither).
    """

    # Design space
    scale_pct_of_pv: list[float]
    e_to_p_ratio_hours: list[float]

    # PV
    pv_peak_kwp: float
    pv_base_timeseries: np.ndarray
    pv_base_timeseries_year: int
    pv_degradation_rate: float
    pv_costs_capex: dict
    pv_costs_opex: dict
    pv_availability_pct: float

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
    replacement_capacity_factor_pct: float

    # Grid
    grid_max_kw: float
    grid_loss_factor: float
    grid_costs_capex: dict
    grid_costs_opex: dict

    # Operating mode
    operating_mode: str

    # Pre-computed per-year prices (€/kWh)
    spot_prices_yearly: list[np.ndarray]
    fixed_prices_yearly: list[float]
    baseload_mw: float

    # Finance
    lifetime_years: int
    commissioning_year: int
    leverage_pct: float
    interest_rate_pct: float
    loan_tenor_years: int
    opex_inflation_factors: list[float]
    discount_rate: float
    afa_years_pv: int
    afa_years_bess: int
    gewerbesteuer_messzahl: float
    gewerbesteuer_hebesatz: float
    koerperschaftsteuer_pct: float
    solidaritaetszuschlag_pct: float

    # BESS optimization fee (% of BESS spot revenue)
    optimization_fee_pct: float = 0.0

    # GoO premiums per year (optional – 0.0 for years without active PPA GoO clause)
    goo_prices_yearly: list[float] = field(default_factory=list)

    # Cap prices per year (PPA Collar; 0.0 = no cap / unbounded upside)
    cap_prices_yearly: list[float] = field(default_factory=list)

    # Conservative debt sizing (downside PV production)
    debt_sizing_downside_pct: float = 0.0

    # Timestep configuration
    timestep_hours: float = 1.0
    intervals_per_day: int = HOURS_PER_DAY
    intervals_per_year: int = HOURS_PER_YEAR

    # Parallelism
    max_workers: int | None = None

    # Baseline control
    skip_baseline: bool = False

    # BESS-Only absolute sizing (used when pv_peak_kwp == 0)
    bess_absolute_power_kw: float | None = None
    bess_absolute_capacity_kwh: float | None = None


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
    is_optimal:
        True for the combination with the highest Equity IRR.
    run_result:
        All relevant information of the P50 simulation run.
    metrics:
        All metrics from the financial perspective
    cashflow:
        Cashflow calculation based on all data above
    """

    scale_pct: float
    e_to_p_ratio: float
    bess_power_kw: float
    bess_capacity_kwh: float
    capex_total: float
    capex_pv: float
    capex_bess: float
    capex_grid: float
    capex_other: float
    opex_base: float
    opex_pv: float
    opex_bess: float
    opex_grid: float
    opex_other: float
    revenue_year1: float
    is_optimal: bool = False
    cashflow: CashflowProjection | None = None
    metrics: FinancialMetrics | None = None
    run_result: SimulationResult | None = None


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
    grid_loss_factor: float
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
    replacement_capacity_factor_pct: float
    lifetime_years: int
    commissioning_year: int

    # PV
    pv_base_timeseries: np.ndarray  # shape (8760,)
    pv_base_timeseries_year: int

    # Prices per year
    spot_prices_yearly: list  # list[np.ndarray]
    fixed_prices_yearly: list  # list[float]
    goo_prices_yearly: list  # list[float]
    cap_prices_yearly: list  # list[float]
    offline_days_bess_yearly: list  # list[set[int]]
    offline_days_pv_yearly: list  # list[set[int]]
    baseload_mw: float  # Baseload level defined for ppa

    # Pre-computed costs
    capex_pv: float
    capex_bess: float
    capex_grid: float
    capex_other: float
    capex_total: float
    opex_base: float
    opex_pv: float
    opex_bess: float
    opex_grid: float
    opex_other: float
    replacement_cost: float

    # Finance
    leverage_pct: float
    interest_rate_pct: float
    loan_tenor_years: int
    opex_inflation_factors: list[float]
    discount_rate: float
    afa_years_pv: int
    afa_years_bess: int
    gewerbesteuer_messzahl: float
    gewerbesteuer_hebesatz: float
    koerperschaftsteuer_pct: float
    solidaritaetszuschlag_pct: float

    # BESS optimization fee
    optimization_fee_pct: float = 0.0

    # Conservative debt sizing (downside PV production)
    debt_sizing_downside_pct: float = 0.0

    # Timestep configuration
    timestep_hours: float = 1.0
    intervals_per_day: int = 24
    intervals_per_year: int = 8760


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
        capacity_factor_pct=args.replacement_capacity_factor_pct,
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
        commissioning_year=args.commissioning_year,
        bess_power_kw=args.bess_power_kw,
        grid_loss_factor=args.grid_loss_factor,
        timestep_hours=args.timestep_hours,
        intervals_per_day=args.intervals_per_day,
        intervals_per_year=args.intervals_per_year,
    )

    # Central scenario simulation – used for equity cashflows
    sim_p50 = run_simulation(
        config=engine_config,
        pv_base_timeseries=args.pv_base_timeseries,
        pv_base_timeseries_year=args.pv_base_timeseries_year,
        spot_prices_yearly=args.spot_prices_yearly,
        fixed_prices_yearly=args.fixed_prices_yearly,
        baseload_mw=args.baseload_mw,
        offline_days_yearly=args.offline_days_bess_yearly,
        pv_offline_days_yearly=args.offline_days_pv_yearly,
        goo_prices_yearly=args.goo_prices_yearly,
        cap_prices_yearly=args.cap_prices_yearly,
    )

    annual_revenues_p50 = [r.total_revenue for r in sim_p50.annual_results]
    annual_bess_spot_revenues_p50 = [r.bess_spot_revenue for r in sim_p50.annual_results]
    total_production_kwh = sum(r.pv_production + r.bess_discharge_grey + r.bess_discharge_green for r in sim_p50.annual_results)

    # Optional downside simulation – used for conservative DSCR calculation, P90 only on PV revenue
    annual_revenues_downside = [
        (r.revenue_pv_export * (1 - args.debt_sizing_downside_pct / 100)) +
        r.bess_spot_revenue - r.grid_import_cost for r in sim_p50.annual_results]

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
        opex_inflation_factors=args.opex_inflation_factors,
        capex_total=args.capex_total,
        capex_pv=args.capex_pv,
        capex_bess=args.capex_bess,
        debt_schedule=debt_schedule,
        afa_years_pv=args.afa_years_pv,
        afa_years_bess=args.afa_years_bess,
        gewerbesteuer_messzahl=args.gewerbesteuer_messzahl,
        gewerbesteuer_hebesatz=args.gewerbesteuer_hebesatz,
        koerperschaftsteuer_pct=args.koerperschaftsteuer_pct,
        solidaritaetszuschlag_pct=args.solidaritaetszuschlag_pct,
        replacement_cost=replacement_cost,
        replacement_year=replacement_year_cf,
        replacement_leverage_pct=args.leverage_pct,
        replacement_interest_rate=args.interest_rate_pct / 100.0,
        replacement_loan_tenor_years=args.loan_tenor_years,
        optimization_fee_pct=args.optimization_fee_pct,
        annual_bess_spot_revenues=annual_bess_spot_revenues_p50,
    )

    # Primary metrics (P50 revenues)
    annual_pv_production_kwh = [r.pv_production for r in sim_p50.annual_results]
    annual_bess_discharge_kwh = [r.bess_throughput for r in sim_p50.annual_results]
    metrics = compute_all_metrics(
        equity_cashflows=cf.equity_cashflows,
        project_cashflows=cf.project_cashflows,
        annual_revenues=annual_revenues_p50,
        annual_opex=[y.opex for y in cf.years],
        annual_debt_service=[cf.years[y - 1].debt_service for y in range(1, args.lifetime_years + 1)],
        total_capex=args.capex_total,
        total_opex_lifetime=sum([y.opex for y in cf.years]),
        total_production_kwh=total_production_kwh,
        discount_rate=args.discount_rate,
        annual_pv_production_kwh=annual_pv_production_kwh,
        annual_bess_discharge_kwh=annual_bess_discharge_kwh,
    )

    # Override DSCR with downside revenues for conservative debt coverage
    if annual_revenues_downside is not None:
        metrics.dscr_min, metrics.dscr_avg, _ = calculate_dscr(
            annual_revenues=annual_revenues_downside,
            annual_opex=[y.opex for y in cf.years],
            annual_debt_service=[cf.years[y - 1].debt_service for y in range(1, args.lifetime_years + 1)],
        )

    revenue_year1 = annual_revenues_p50[0] if annual_revenues_p50 else 0.0

    for y in range(args.lifetime_years):
        cf.years[y].grid_import_costs = sim_p50.annual_results[y].grid_import_cost
        cf.years[y].baseload_matching_costs = sim_p50.annual_results[y].missing_baseload_cost

    return GridPointResult(
        scale_pct=args.scale_pct,
        e_to_p_ratio=args.e_to_p_ratio,
        bess_power_kw=args.bess_power_kw,
        bess_capacity_kwh=args.bess_capacity_kwh,
        capex_total=args.capex_total,
        capex_pv=args.capex_pv,
        capex_bess=args.capex_bess,
        capex_grid=args.capex_grid,
        capex_other=args.capex_other,
        opex_base=args.opex_base,
        opex_pv=args.opex_pv,
        opex_bess=args.opex_bess,
        opex_grid=args.opex_grid,
        opex_other=args.opex_other,
        revenue_year1=revenue_year1,
        is_optimal=False,
        metrics=metrics,
        run_result=sim_p50,
        cashflow=cf,
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
    # Ensure PV-only baseline is included (unless explicitly skipped)
    scales = list(config.scale_pct_of_pv)
    if config.skip_baseline:
        logger.info(
            "skip_baseline=True: PV-only baseline (scale=0 %%) will NOT be added."
        )
    elif GRID_SEARCH_SCALE_ZERO_PCT not in scales:
        scales = [GRID_SEARCH_SCALE_ZERO_PCT] + scales
        logger.info("Added scale=0 %% (PV-only baseline) to grid search.")

    # Deterministic offline days – same for every grid point
    offline_days_bess: set[int] = compute_deterministic_offline_days(
        config.bess_availability_pct
    )
    offline_days_bess_yearly: list[set[int]] = [
        offline_days_bess for _ in range(config.lifetime_years)
    ]
    offline_days_pv: set[int] = compute_deterministic_offline_days(
        config.pv_availability_pct, offset=28
    )
    offline_days_pv_yearly: list[set[int]] = [
        offline_days_pv for _ in range(config.lifetime_years)
    ]

    # Build worker args for every (scale, E/P) combination
    worker_args: list[_GridPointArgs] = []
    for scale_pct in scales:
        for e_to_p in config.e_to_p_ratio_hours:
            if config.pv_peak_kwp > 0:
                # Standard ratio-based sizing
                bess_power_kw = config.pv_peak_kwp * scale_pct / 100.0
                bess_capacity_kwh = bess_power_kw * e_to_p
            elif scale_pct > 0 and config.bess_absolute_power_kw is not None:
                # BESS-Only: absolute sizing overrides ratio computation
                bess_power_kw = config.bess_absolute_power_kw
                bess_capacity_kwh = (
                    config.bess_absolute_capacity_kwh
                    if config.bess_absolute_capacity_kwh is not None
                    else bess_power_kw * e_to_p
                )
            else:
                # Baseline (scale = 0) or no absolute values → no BESS
                bess_power_kw = 0.0
                bess_capacity_kwh = 0.0

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

            # Replacement cost: eur_per_kwh applies to the NEW (upgraded) capacity
            _upgrade = config.replacement_capacity_factor_pct / 100.0
            replacement_cost = (
                    config.replacement_fixed_eur
                    + config.replacement_eur_per_kw * bess_power_kw
                    + config.replacement_eur_per_kwh * bess_capacity_kwh * _upgrade
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
                    grid_loss_factor=config.grid_loss_factor,
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
                    replacement_capacity_factor_pct=config.replacement_capacity_factor_pct,
                    lifetime_years=config.lifetime_years,
                    commissioning_year=config.commissioning_year,
                    pv_base_timeseries=config.pv_base_timeseries,
                    pv_base_timeseries_year=config.pv_base_timeseries_year,
                    spot_prices_yearly=config.spot_prices_yearly,
                    fixed_prices_yearly=config.fixed_prices_yearly,
                    baseload_mw=config.baseload_mw,
                    goo_prices_yearly=config.goo_prices_yearly if config.goo_prices_yearly else [0.0] * config.lifetime_years,
                    cap_prices_yearly=config.cap_prices_yearly if config.cap_prices_yearly else [0.0] * config.lifetime_years,
                    offline_days_bess_yearly=offline_days_bess_yearly,
                    offline_days_pv_yearly=offline_days_pv_yearly,
                    capex_pv=costs.capex_pv,
                    capex_bess=costs.capex_bess,
                    capex_grid=costs.capex_grid,
                    capex_other=costs.capex_other,
                    capex_total=costs.capex_total,
                    opex_base=costs.opex_total,
                    opex_pv=costs.opex_pv,
                    opex_bess=costs.opex_bess,
                    opex_grid=costs.opex_grid,
                    opex_other=costs.opex_other,
                    replacement_cost=replacement_cost,
                    leverage_pct=config.leverage_pct,
                    interest_rate_pct=config.interest_rate_pct,
                    loan_tenor_years=config.loan_tenor_years,
                    opex_inflation_factors=config.opex_inflation_factors,
                    discount_rate=config.discount_rate,
                    afa_years_pv=config.afa_years_pv,
                    afa_years_bess=config.afa_years_bess,
                    gewerbesteuer_messzahl=config.gewerbesteuer_messzahl,
                    gewerbesteuer_hebesatz=config.gewerbesteuer_hebesatz,
                    koerperschaftsteuer_pct=config.koerperschaftsteuer_pct,
                    solidaritaetszuschlag_pct=config.solidaritaetszuschlag_pct,
                    optimization_fee_pct=config.optimization_fee_pct,
                    debt_sizing_downside_pct=config.debt_sizing_downside_pct,
                    timestep_hours=config.timestep_hours,
                    intervals_per_day=config.intervals_per_day,
                    intervals_per_year=config.intervals_per_year,
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
    if config.max_workers == 1 or n_combinations == 1:
        # Single-process execution: either forced via max_workers=1 (debug /
        # unit tests) or only one combination to evaluate (multiprocessing
        # overhead is not warranted – can only happen when skip_baseline=True
        # and a single scale + E/P value is specified).
        if n_combinations == 1 and config.max_workers != 1:
            logger.info(
                "Grid search: single configuration – skipping multiprocessing."
            )
        for idx, a in enumerate(worker_args, start=1):
            result = _evaluate_grid_point(a)
            results.append(result)
            logger.debug(
                "Grid search [%d/%d]: scale=%.0f %%, E/P=%.1f h, Equity IRR=%s.",
                idx,
                n_combinations,
                a.scale_pct,
                a.e_to_p_ratio,
                f"{(result.metrics.equity_irr or 0.0) * 100:.2f} %" if result.metrics.equity_irr is not None else "N/A",
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=config.max_workers
        ) as executor:
            futures = {
                executor.submit(_evaluate_grid_point, a): a for a in worker_args
            }
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                logger.debug(
                    "Grid search [%d/%d] done: scale=%.0f %%, E/P=%.1f h.",
                    completed,
                    n_combinations,
                    result.scale_pct,
                    result.e_to_p_ratio,
                )

    # Sort results by (scale_pct, e_to_p_ratio) for deterministic output order
    results.sort(key=lambda r: (r.scale_pct, r.e_to_p_ratio))

    # Identify optimum: highest Equity IRR (None treated as -inf)
    optimal: GridPointResult | None = None
    best_irr: float = float("-inf")
    if len(results) == 1:
        optimal = results[0]
    else:
        for r in results:
            irr = r.metrics.equity_irr if r.metrics.equity_irr is not None else float("-inf")
            if irr > best_irr:
                best_irr = irr
                optimal = r

    if optimal is not None:
        optimal.is_optimal = True
        logger.info(
            "Grid search optimum: scale=%.0f %%, E/P=%.1f h, Equity IRR=%.2f %%.",
            optimal.scale_pct,
            optimal.e_to_p_ratio,
            (optimal.metrics.equity_irr or 0.0) * 100,
        )
    else:
        logger.warning("Grid search: no valid Equity IRR found in any combination.")

    return GridSearchResult(points=results, optimal=optimal)
