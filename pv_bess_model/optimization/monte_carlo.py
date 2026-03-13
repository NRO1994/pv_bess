"""Monte Carlo simulation on the optimal BESS configuration from grid search.

**Refactored approach (Session 5, Phase 1, Step 4):**

Instead of running a full dispatch simulation per MC iteration (N x S dispatch
runs), dispatch is executed only **once per price scenario** with 100 %
PV and BESS availability.  The dispatch simulations are parallelised across
scenarios.  MC noise factors are then applied sequentially to the pre-computed
financial results in the main thread:

- CAPEX / OPEX factors: scale asset-level costs.
- PV availability factor: scales PV-related revenues (PV export and BESS
  green discharge, since green charging comes from PV surplus).
- BESS availability factor: scales BESS-related revenues (green + grey
  discharge) and grid import costs.

This reduces the compute cost from ``N_iterations x S_scenarios x 365 x
lifetime`` LP solves to just ``S_scenarios x 365 x lifetime`` LP solves,
giving a typical speed-up of ~1000x.

Public API
----------
MCParams            - Monte Carlo hyper-parameters (iterations, sigma values, etc.).
MCIterationResult   - Metrics from a single MC iteration.
MCStatistics        - Descriptive statistics over a set of values.
MCResult            - Complete MC output with all iterations and summary stats.
run_monte_carlo     - Main entry point.
"""

from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass, field

import numpy as np

from pv_bess_model.bess.replacement import ReplacementConfig
from pv_bess_model.config.defaults import (
    BESS_NOISE_CLIP_MAX,
    DEFAULT_MC_ITERATIONS,
    MC_WEIGHT_TOLERANCE,
)
from pv_bess_model.config.loader import PriceWeatherScenario
from pv_bess_model.dispatch.engine import (
    AnnualResult,
    DispatchEngineConfig,
    run_simulation,
)
from pv_bess_model.finance.cashflow import build_cashflow_projection
from pv_bess_model.finance.debt import build_annuity_schedule
from pv_bess_model.finance.inflation import inflate_value
from pv_bess_model.finance.metrics import compute_all_metrics
from pv_bess_model.optimization.grid_search import GridPointResult, GridSearchConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MC configuration
# ---------------------------------------------------------------------------


@dataclass
class MCParams:
    """Monte Carlo hyper-parameters.

    Parameters
    ----------
    iterations:
        Number of MC iterations.
    sigma_capex_pv:
        Standard deviation for the PV CAPEX noise factor N(1, sigma).
    sigma_capex_bess:
        Standard deviation for the BESS CAPEX noise factor N(1, sigma).
    sigma_opex_pv:
        Standard deviation for the PV OPEX noise factor N(1, sigma).
    sigma_opex_bess:
        Standard deviation for the BESS OPEX noise factor N(1, sigma).
    sigma_pv_availability:
        Standard deviation for the PV availability noise factor.
        PV availability is sampled as N(1.0, sigma), clipped to [0, 1].
    mu_bess_availability:
        Mean of the BESS availability noise factor (fraction, 0-1).
        e.g. 0.97 for 97 %.
    sigma_bess_availability:
        Standard deviation of the BESS availability noise factor.
    price_scenarios:
        List of PriceWeatherScenario objects.
        Weights must sum to 1.0 (within ``MC_WEIGHT_TOLERANCE``).
    seed:
        Base random seed for reproducibility. Each iteration uses
        ``seed + iteration`` as its own seed.
    max_workers:
        Number of parallel worker processes for the dispatch phase.
        None = os.cpu_count().
    """

    iterations: int = DEFAULT_MC_ITERATIONS
    sigma_capex_pv: float = 0.05
    sigma_capex_bess: float = 0.10
    sigma_opex_pv: float = 0.03
    sigma_opex_bess: float = 0.08
    sigma_pv_availability: float = 0.02
    mu_bess_availability: float = 0.97
    sigma_bess_availability: float = 0.02
    price_scenarios: list[PriceWeatherScenario] = field(default_factory=dict)
    seed: int = 0
    max_workers: int | None = None

    def __post_init__(self) -> None:
        """Set default price scenarios and validate weights."""
        weights = sum(version.weight for version in self.price_scenarios)
        if abs(weights - 1.0) > MC_WEIGHT_TOLERANCE:
            raise ValueError(
                f"MC price scenario weights must sum to 1.0, got {weights:.6f}."
            )


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class MCIterationResult:
    """Metrics from a single Monte Carlo iteration.

    Attributes
    ----------
    iteration:
        1-indexed iteration number.
    price_scenario:
        Name of the sampled price scenario (e.g. ``"mid"``).
    capex_factor_pv:
        Sampled PV CAPEX noise factor.
    capex_factor_bess:
        Sampled BESS CAPEX noise factor.
    opex_factor_pv:
        Sampled PV OPEX noise factor.
    opex_factor_bess:
        Sampled BESS OPEX noise factor.
    pv_availability_factor:
        Sampled PV availability factor (fraction, clipped to [0, 1]).
    bess_availability_factor:
        Sampled BESS availability factor (fraction, clipped to [mu, 1]).
    equity_irr:
        Post-leverage, post-tax Equity IRR (or None).
    project_irr:
        Pre-leverage Project IRR (or None).
    npv:
        NPV at the configured discount rate in EUR.
    dscr_min:
        Minimum DSCR over the loan tenor (or None).
    capture_rate:
        Average revenue per kWh fed into the grid (EUR/kWh), or None.
    fixed_price_years:
        Number of years with a fixed (EEG/PPA) price guarantee.
    analysis_label:
        Label identifying the analysis context (e.g. "Direktvermarktungs-baseline").
    """

    iteration: int
    price_scenario: str
    capex_factor_pv: float
    capex_factor_bess: float
    opex_factor_pv: float
    opex_factor_bess: float
    pv_availability_factor: float
    bess_availability_factor: float
    equity_irr: float | None
    project_irr: float | None
    npv: float
    dscr_min: float | None
    capture_rate: float | None
    fixed_price_years: int
    analysis_label: str


@dataclass
class MCStatistics:
    """Descriptive statistics over a scalar metric across MC iterations.

    Attributes
    ----------
    mean, median, std:
        Standard moments.
    p10, p25, p50, p75, p90:
        Percentiles (10th through 90th).
    """

    mean: float
    median: float
    std: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float


@dataclass
class MCResult:
    """Complete Monte Carlo simulation output.

    Attributes
    ----------
    iterations:
        All per-iteration results.
    overall_stats:
        Descriptive statistics across all iterations, keyed by metric name
        (``"equity_irr"``, ``"project_irr"``, ``"npv"``, ``"dscr_min"``).
    per_scenario_stats:
        Same statistics broken down by price scenario name.
    """

    iterations: list[MCIterationResult]
    overall_stats: dict[str, MCStatistics]
    per_scenario_stats: dict[str, dict[str, MCStatistics]]


# ---------------------------------------------------------------------------
# Pre-computed dispatch results per scenario
# ---------------------------------------------------------------------------


@dataclass
class _ScenarioDispatch:
    """Cached dispatch results for one price scenario (100 % availability).

    All lists have length ``lifetime_years``.
    """

    scenario_name: str
    annual_revenue_pv_export: list[float]
    annual_revenue_bess_green: list[float]
    annual_revenue_bess_grey: list[float]
    annual_grid_import_cost: list[float]
    annual_missing_baseload_cost: list[float]
    annual_total_revenue: list[float]
    annual_bess_spot_revenue: list[float]
    total_production_kwh: float
    annual_pv_production_kwh: list[float]
    annual_bess_discharge_kwh: list[float]


# ---------------------------------------------------------------------------
# Dispatch worker for parallel scenario simulation
# ---------------------------------------------------------------------------

# Module-level state set by the process pool initialiser so that each
# worker has access to the heavy shared data without re-serialising it
# per task.
_DISPATCH_WORKER_STATE: dict | None = None


def _dispatch_worker_init(state: dict) -> None:
    """Initialise a dispatch worker process with shared read-only data."""
    global _DISPATCH_WORKER_STATE
    _DISPATCH_WORKER_STATE = state


def _run_scenario_dispatch(scenario_name: str) -> _ScenarioDispatch:
    """Run dispatch for one price scenario at 100 % availability.

    Reads shared state from module-level ``_DISPATCH_WORKER_STATE``.

    Parameters
    ----------
    scenario_name:
        Name of the scenario to simulate (key into the shared scenario map).

    Returns
    -------
    _ScenarioDispatch
        Pre-computed per-year revenue breakdown.
    """
    assert _DISPATCH_WORKER_STATE is not None, "Dispatch worker state not initialised."

    base: GridSearchConfig = _DISPATCH_WORKER_STATE["base_config"]
    optimal: GridPointResult = _DISPATCH_WORKER_STATE["optimal"]
    scenario_map: dict[str, PriceWeatherScenario] = _DISPATCH_WORKER_STATE["scenario_map"]

    scenario = scenario_map[scenario_name]

    replacement = ReplacementConfig(
        enabled=base.replacement_enabled,
        year=base.replacement_year,
        fixed_eur=base.replacement_fixed_eur,
        eur_per_kw=base.replacement_eur_per_kw,
        eur_per_kwh=base.replacement_eur_per_kwh,
    )
    engine_config = DispatchEngineConfig(
        mode=base.operating_mode,
        grid_max_kw=base.grid_max_kw,
        bess_nameplate_kwh=optimal.bess_capacity_kwh,
        bess_max_charge_kw=optimal.bess_power_kw,
        bess_max_discharge_kw=optimal.bess_power_kw,
        bess_rte=base.bess_rte,
        bess_min_soc_pct=base.bess_min_soc_pct,
        bess_max_soc_pct=base.bess_max_soc_pct,
        bess_degradation_rate=base.bess_degradation_rate,
        pv_degradation_rate=base.pv_degradation_rate,
        replacement=replacement,
        lifetime_years=base.lifetime_years,
        bess_power_kw=optimal.bess_power_kw,
        grid_loss_factor=base.grid_loss_factor,
        timestep_hours=base.timestep_hours,
        intervals_per_day=base.intervals_per_day,
        intervals_per_year=base.intervals_per_year,
        commissioning_year=base.commissioning_year,
    )

    # 100 % availability: no offline days
    no_offline = [set() for _ in range(base.lifetime_years)]

    sim = run_simulation(
        config=engine_config,
        pv_base_timeseries=scenario.pv_timeseries_15min,
        spot_prices_yearly=scenario.price_per_year,
        fixed_prices_yearly=base.fixed_prices_yearly,
        offline_days_yearly=no_offline,
        goo_prices_yearly=base.goo_prices_yearly if base.goo_prices_yearly else None,
        cap_prices_yearly=base.cap_prices_yearly if base.cap_prices_yearly else None,
        pv_offline_days_yearly=None,
        pv_base_timeseries_year=scenario.weather_year,
        baseload_kw=base.baseload_mw,
    )

    return _ScenarioDispatch(
        scenario_name=scenario.name,
        annual_revenue_pv_export=[r.revenue_pv_export for r in sim.annual_results],
        annual_revenue_bess_green=[r.revenue_bess_green for r in sim.annual_results],
        annual_revenue_bess_grey=[r.revenue_bess_grey for r in sim.annual_results],
        annual_grid_import_cost=[r.grid_import_cost for r in sim.annual_results],
        annual_missing_baseload_cost=[r.missing_baseload_cost for r in sim.annual_results],
        annual_total_revenue=[r.total_revenue for r in sim.annual_results],
        annual_bess_spot_revenue=[r.bess_spot_revenue for r in sim.annual_results],
        total_production_kwh=sum(r.pv_production for r in sim.annual_results),
        annual_pv_production_kwh=[r.pv_production for r in sim.annual_results],
        annual_bess_discharge_kwh=[r.bess_throughput for r in sim.annual_results],
    )


# ---------------------------------------------------------------------------
# Internal: single MC iteration (no dispatch, only financial noise)
# ---------------------------------------------------------------------------


def _run_mc_iteration_fast(
    iteration: int,
    base: GridSearchConfig,
    optimal: GridPointResult,
    mc: MCParams,
    scenario_dispatches: dict[str, _ScenarioDispatch],
    scenario_prices: list[PriceWeatherScenario],
    fixed_price_years: int = 0,
    analysis_label: str = "",
) -> MCIterationResult:
    """Execute one Monte Carlo iteration using pre-computed dispatch results.

    No dispatch simulation is run.  Instead, the pre-computed revenues are
    scaled by the sampled PV/BESS availability factors, and CAPEX/OPEX are
    scaled by their respective noise factors.

    Parameters
    ----------
    iteration:
        1-indexed iteration number (also used as random seed offset).
    base:
        Grid search configuration.
    optimal:
        Optimal grid point.
    mc:
        MC hyper-parameters.
    scenario_dispatches:
        Pre-computed dispatch results keyed by scenario name.
    scenario_prices:
        List of price-weather scenarios (for weighted sampling).

    Returns
    -------
    MCIterationResult
        Sampled inputs and resulting financial metrics.
    """
    rng = np.random.default_rng(seed=mc.seed + iteration)

    # --- Sample price scenario ---
    weights = [s.weight for s in scenario_prices]
    scenario_idx = rng.choice(len(scenario_prices), p=weights)
    selected: PriceWeatherScenario = scenario_prices[scenario_idx]
    dispatch = scenario_dispatches[selected.name]

    # --- Sample noise factors ---
    capex_factor_pv = float(rng.normal(1.0, mc.sigma_capex_pv))
    capex_factor_bess = float(rng.normal(1.0, mc.sigma_capex_bess))
    opex_factor_pv = float(rng.normal(1.0, mc.sigma_opex_pv))
    opex_factor_bess = float(rng.normal(1.0, mc.sigma_opex_bess))

    # PV availability: N(1.0, sigma), clipped to [0, 1]
    pv_availability_factor = float(
        np.clip(rng.normal(1.0, mc.sigma_pv_availability), 0.0, BESS_NOISE_CLIP_MAX)
    )

    # BESS availability: N(mu_sample, sigma), clipped to [mu_bess, 1.0]
    mu_sample = (mc.mu_bess_availability + 1.0) / 2.0
    raw_avail = float(rng.normal(mu_sample, mc.sigma_bess_availability))
    bess_availability_factor = float(
        np.clip(raw_avail, mc.mu_bess_availability, BESS_NOISE_CLIP_MAX)
    )

    # --- Scale CAPEX / OPEX per asset ---
    capex_pv = optimal.capex_pv * capex_factor_pv
    capex_bess = optimal.capex_bess * capex_factor_bess
    capex_grid = optimal.capex_grid
    capex_other = optimal.capex_other
    capex_total = capex_pv + capex_bess + capex_grid + capex_other

    opex_pv = optimal.opex_pv * opex_factor_pv
    opex_bess = optimal.opex_bess * opex_factor_bess
    opex_grid = optimal.opex_grid
    opex_other = optimal.opex_other
    opex_base = opex_pv + opex_bess + opex_grid + opex_other

    # --- Replacement cost (scales with BESS CAPEX factor) ---
    replacement_cost = (
        base.replacement_fixed_eur
        + base.replacement_eur_per_kw * optimal.bess_power_kw
        + base.replacement_eur_per_kwh * optimal.bess_capacity_kwh
        + base.replacement_pct_of_capex * capex_bess
    )

    # --- Apply availability factors to pre-computed revenues ---
    #
    # PV availability affects:
    #   - PV export revenue (direct PV-to-grid)
    #   - BESS green revenue (green charging comes from PV surplus)
    #
    # BESS availability affects:
    #   - BESS green discharge revenue
    #   - BESS grey discharge revenue
    #   - Grid import costs (grey mode charging)
    #
    # Combined: BESS green revenue is scaled by both factors (PV supplies
    # the energy, BESS must be online to discharge it).
    annual_revenues: list[float] = []
    annual_bess_spot_revenues: list[float] = []
    for y in range(base.lifetime_years):
        rev_pv = dispatch.annual_revenue_pv_export[y] * pv_availability_factor
        rev_bess_green = (
            dispatch.annual_revenue_bess_green[y]
            * pv_availability_factor
            * bess_availability_factor
        )
        rev_bess_grey = (
            dispatch.annual_revenue_bess_grey[y] * bess_availability_factor
        )
        grid_cost = (
            dispatch.annual_grid_import_cost[y] * bess_availability_factor
        )
        baseload_cost = dispatch.annual_missing_baseload_cost[y]

        total_rev = rev_pv + rev_bess_green + rev_bess_grey - grid_cost - baseload_cost
        annual_revenues.append(total_rev)

        bess_spot = rev_bess_green + rev_bess_grey
        annual_bess_spot_revenues.append(bess_spot)

    total_production_kwh = dispatch.total_production_kwh * pv_availability_factor
    annual_pv_production_kwh = [
        p * pv_availability_factor for p in dispatch.annual_pv_production_kwh
    ]
    annual_bess_discharge_kwh = [
        d * bess_availability_factor for d in dispatch.annual_bess_discharge_kwh
    ]

    # --- Build cashflow projection ---
    debt_schedule = build_annuity_schedule(
        total_capex=capex_total,
        leverage_pct=base.leverage_pct,
        annual_interest_rate=base.interest_rate_pct / 100.0,
        tenor_years=base.loan_tenor_years,
    )
    replacement_year_cf: int | None = (
        base.replacement_year if base.replacement_enabled else None
    )
    cf = build_cashflow_projection(
        lifetime_years=base.lifetime_years,
        annual_revenues=annual_revenues,
        base_opex=opex_base,
        inflation_rate=base.inflation_rate,
        capex_total=capex_total,
        capex_pv=capex_pv,
        capex_bess=capex_bess,
        debt_schedule=debt_schedule,
        afa_years_pv=base.afa_years_pv,
        afa_years_bess=base.afa_years_bess,
        gewerbesteuer_messzahl=base.gewerbesteuer_messzahl,
        gewerbesteuer_hebesatz=base.gewerbesteuer_hebesatz,
        koerperschaftsteuer_pct=base.koerperschaftsteuer_pct,
        solidaritaetszuschlag_pct=base.solidaritaetszuschlag_pct,
        replacement_cost=replacement_cost,
        replacement_year=replacement_year_cf,
        replacement_leverage_pct=base.leverage_pct,
        replacement_interest_rate=base.interest_rate_pct / 100.0,
        replacement_loan_tenor_years=base.loan_tenor_years,
        optimization_fee_pct=base.optimization_fee_pct,
        annual_bess_spot_revenues=annual_bess_spot_revenues,
    )

    annual_opex = []
    for y in range(1, base.lifetime_years + 1):
        opex_y = inflate_value(opex_base, base.inflation_rate, y)
        if base.optimization_fee_pct > 0.0:
            opex_y += annual_bess_spot_revenues[y - 1] * base.optimization_fee_pct / 100.0
        annual_opex.append(opex_y)
    annual_debt_service = [
        cf.years[y - 1].debt_service for y in range(1, base.lifetime_years + 1)
    ]
    total_opex_lifetime = sum(annual_opex)

    metrics = compute_all_metrics(
        equity_cashflows=cf.equity_cashflows,
        project_cashflows=cf.project_cashflows,
        annual_revenues=annual_revenues,
        annual_opex=annual_opex,
        annual_debt_service=annual_debt_service,
        total_capex=capex_total,
        total_opex_lifetime=total_opex_lifetime,
        total_production_kwh=total_production_kwh,
        discount_rate=base.discount_rate,
        annual_pv_production_kwh=annual_pv_production_kwh,
        annual_bess_discharge_kwh=annual_bess_discharge_kwh,
    )

    return MCIterationResult(
        iteration=iteration,
        price_scenario=selected.name,
        capex_factor_pv=capex_factor_pv,
        capex_factor_bess=capex_factor_bess,
        opex_factor_pv=opex_factor_pv,
        opex_factor_bess=opex_factor_bess,
        pv_availability_factor=pv_availability_factor,
        bess_availability_factor=bess_availability_factor,
        equity_irr=metrics.equity_irr,
        project_irr=metrics.project_irr,
        npv=metrics.npv,
        dscr_min=metrics.dscr_min,
        capture_rate=metrics.capture_rate,
        fixed_price_years=fixed_price_years,
        analysis_label=analysis_label,
    )


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _compute_statistics(values: list[float | None]) -> MCStatistics:
    """Compute descriptive statistics over a list of metric values.

    None values (e.g. failed IRR convergence) are excluded from the
    calculation.

    Parameters
    ----------
    values:
        Metric values, possibly containing None entries.

    Returns
    -------
    MCStatistics
        Statistics from valid (non-None) values, or all-NaN if no valid
        values exist.
    """
    valid = np.array([v for v in values if v is not None], dtype=float)
    if len(valid) == 0:
        nan = float("nan")
        return MCStatistics(
            mean=nan, median=nan, std=nan,
            p10=nan, p25=nan, p50=nan, p75=nan, p90=nan,
        )
    return MCStatistics(
        mean=float(np.mean(valid)),
        median=float(np.median(valid)),
        std=float(np.std(valid)),
        p10=float(np.percentile(valid, 10)),
        p25=float(np.percentile(valid, 25)),
        p50=float(np.percentile(valid, 50)),
        p75=float(np.percentile(valid, 75)),
        p90=float(np.percentile(valid, 90)),
    )


def _build_stats(
    results: list[MCIterationResult],
) -> tuple[dict[str, MCStatistics], dict[str, dict[str, MCStatistics]]]:
    """Build overall and per-scenario statistics from all iteration results.

    Parameters
    ----------
    results:
        All ``MCIterationResult`` objects from the simulation.

    Returns
    -------
    tuple[dict, dict]
        ``(overall_stats, per_scenario_stats)`` where each inner dict maps
        metric names to :class:`MCStatistics`.
    """
    metric_names = ("equity_irr", "project_irr", "npv", "dscr_min")

    overall_stats: dict[str, MCStatistics] = {
        m: _compute_statistics([getattr(r, m) for r in results])
        for m in metric_names
    }

    scenarios = sorted({r.price_scenario for r in results})
    per_scenario_stats: dict[str, dict[str, MCStatistics]] = {
        scenario: {
            m: _compute_statistics(
                [getattr(r, m) for r in results if r.price_scenario == scenario]
            )
            for m in metric_names
        }
        for scenario in scenarios
    }

    return overall_stats, per_scenario_stats


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_monte_carlo(
    base_config: GridSearchConfig,
    optimal: GridPointResult,
    mc_params: MCParams,
    scenario_prices: list[PriceWeatherScenario],
    fixed_price_years: int = 0,
    analysis_label: str = "",
) -> MCResult:
    """Run the Monte Carlo simulation on the optimal BESS configuration.

    **Two-phase approach:**

    1. **Dispatch phase** (parallelised): Run one full multi-year dispatch
       simulation per price scenario with 100 % PV and BESS availability.
       Scenarios are dispatched in parallel via ``ProcessPoolExecutor``.
    2. **MC noise phase** (sequential, main thread): For each iteration,
       sample noise factors and apply them to the pre-computed revenues
       and costs -- no dispatch is re-run.

    Parameters
    ----------
    base_config:
        The ``GridSearchConfig`` used for the grid search.
    optimal:
        The optimal grid point from the grid search (highest Equity IRR).
    mc_params:
        Monte Carlo hyper-parameters (iterations, sigma values, price scenarios).
    scenario_prices:
        List of ``PriceWeatherScenario`` objects with ``price_per_year``
        and ``pv_timeseries_15min`` populated.

    Returns
    -------
    MCResult
        All iteration results plus overall and per-scenario statistics.
    """
    n_scenarios = len(scenario_prices)
    logger.info(
        "Monte Carlo: %d iterations, %d price scenario(s), max_workers=%s.",
        mc_params.iterations,
        n_scenarios,
        mc_params.max_workers,
    )

    # Validate: mc_params.price_scenarios and scenario_prices must match
    mc_names = {s.name for s in mc_params.price_scenarios}
    sp_names = {s.name for s in scenario_prices}
    if mc_names != sp_names:
        missing = mc_names - sp_names
        extra = sp_names - mc_names
        raise ValueError(
            f"MC price scenario mismatch: mc_params has {mc_names}, "
            f"scenario_prices has {sp_names}. "
            f"Missing: {missing}, extra: {extra}."
        )

    # ------------------------------------------------------------------
    # Phase 1: Run dispatch once per price scenario (parallelised)
    # ------------------------------------------------------------------
    logger.info(
        "MC Phase 1: Running dispatch for %d price scenario(s) "
        "with 100 %% availability.",
        n_scenarios,
    )

    scenario_map: dict[str, PriceWeatherScenario] = {
        s.name: s for s in scenario_prices
    }
    shared_state: dict = {
        "base_config": base_config,
        "optimal": optimal,
        "scenario_map": scenario_map,
    }
    scenario_names = [s.name for s in scenario_prices]

    scenario_dispatches: dict[str, _ScenarioDispatch] = {}

    if mc_params.max_workers == 1 or n_scenarios <= 1:
        # Single-process: debug-friendly and used by unit tests
        _dispatch_worker_init(shared_state)
        for name in scenario_names:
            logger.debug("MC dispatch: scenario '%s'.", name)
            scenario_dispatches[name] = _run_scenario_dispatch(name)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=mc_params.max_workers,
            initializer=_dispatch_worker_init,
            initargs=(shared_state,),
        ) as executor:
            futures = {
                executor.submit(_run_scenario_dispatch, name): name
                for name in scenario_names
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                scenario_dispatches[result.scenario_name] = result
                logger.debug(
                    "MC dispatch complete: scenario '%s'.",
                    result.scenario_name,
                )

    logger.info(
        "MC Phase 1 complete: %d dispatch simulation(s) cached.", n_scenarios
    )

    # ------------------------------------------------------------------
    # Phase 2: MC iterations (sequential, main thread -- no dispatch)
    # ------------------------------------------------------------------
    logger.info(
        "MC Phase 2: Running %d iterations (financial noise only).",
        mc_params.iterations,
    )
    n_iterations = mc_params.iterations
    log_interval = max(1, n_iterations // 10)

    results: list[MCIterationResult] = []
    for i in range(1, n_iterations + 1):
        result = _run_mc_iteration_fast(
            iteration=i,
            base=base_config,
            optimal=optimal,
            mc=mc_params,
            scenario_dispatches=scenario_dispatches,
            scenario_prices=scenario_prices,
            fixed_price_years=fixed_price_years,
            analysis_label=analysis_label,
        )
        results.append(result)
        if i % log_interval == 0 or i == n_iterations:
            logger.debug(
                "Monte Carlo: %d/%d iterations complete.", i, n_iterations
            )

    results.sort(key=lambda r: r.iteration)

    overall_stats, per_scenario_stats = _build_stats(results)

    eq_stats = overall_stats.get("equity_irr")
    if eq_stats is not None and not np.isnan(eq_stats.median):
        logger.info(
            "MC complete: Equity IRR median=%.2f %%, P10=%.2f %%, P90=%.2f %%.",
            eq_stats.median * 100,
            eq_stats.p10 * 100,
            eq_stats.p90 * 100,
        )

    return MCResult(
        iterations=results,
        overall_stats=overall_stats,
        per_scenario_stats=per_scenario_stats,
    )