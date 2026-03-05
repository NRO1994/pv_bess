"""Monte Carlo simulation on the optimal BESS configuration from grid search.

Runs stochastic multi-year dispatch simulations on top of the grid search
optimum.  Each iteration samples noise factors (PV yield, CAPEX, OPEX, BESS
availability) and a price scenario (low / mid / high), then runs a full
multi-year simulation to produce financial metrics.

Public API
----------
MCParams            – Monte Carlo hyper-parameters (iterations, σ values, etc.).
MCIterationResult   – Metrics from a single MC iteration.
MCStatistics        – Descriptive statistics over a set of values.
MCResult            – Complete MC output with all iterations and summary stats.
run_monte_carlo     – Main entry point.
"""

from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass, field

import numpy as np

from pv_bess_model.bess.replacement import ReplacementConfig
from pv_bess_model.config.defaults import (
    BESS_NOISE_CLIP_MAX,
    DAYS_PER_YEAR,
    DEFAULT_MC_ITERATIONS,
    MC_WEIGHT_TOLERANCE,
)
from pv_bess_model.config.loader import PriceWeatherScenario
from pv_bess_model.dispatch.engine import DispatchEngineConfig, run_simulation
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
        Standard deviation for the PV CAPEX noise factor N(1, σ).
    sigma_capex_bess:
        Standard deviation for the BESS CAPEX noise factor N(1, σ).
    sigma_opex_pv:
        Standard deviation for the PV OPEX noise factor N(1, σ).
    sigma_opex_bess:
        Standard deviation for the BESS OPEX noise factor N(1, σ).
    sigma_pv_availability:
        Standard deviation for the PV availability noise factor.
        PV availability is sampled as N(1.0, σ), clipped to [0, 1].
    mu_bess_availability:
        Mean of the BESS availability noise factor (fraction, 0–1).
        e.g. 0.97 for 97 %.
    sigma_bess_availability:
        Standard deviation of the BESS availability noise factor.
    price_scenarios:
        Mapping from scenario name to ``{"csv_column": str, "weight": float}``.
        Weights must sum to 1.0 (within ``MC_WEIGHT_TOLERANCE``).
    seed:
        Base random seed for reproducibility. Each iteration uses
        ``seed + iteration`` as its own seed.
    max_workers:
        Number of parallel worker processes. None = os.cpu_count().
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
        NPV at the configured discount rate in €.
    dscr_min:
        Minimum DSCR over the loan tenor (or None).
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
# Internal: worker shared state (initializer pattern)
# ---------------------------------------------------------------------------

# Module-level global set once per worker process via the initializer.
_MC_WORKER_STATE: dict | None = None


def _mc_worker_init(state: dict) -> None:
    """Initialise the worker process with shared read-only data.

    Parameters
    ----------
    state:
        Dict containing all shared data for MC workers.  Keys:
        ``grid_config``, ``optimal``, ``scenario_prices``, ``mc_params``.
    """
    global _MC_WORKER_STATE
    _MC_WORKER_STATE = state


# ---------------------------------------------------------------------------
# Internal: single iteration worker
# ---------------------------------------------------------------------------


def _run_mc_iteration(iteration: int) -> MCIterationResult:
    """Execute one Monte Carlo iteration.

    The shared configuration is read from the module-level
    ``_MC_WORKER_STATE`` set by the worker initialiser.

    Parameters
    ----------
    iteration:
        1-indexed iteration number (also used as random seed offset).

    Returns
    -------
    MCIterationResult
        Sampled inputs and resulting financial metrics.
    """
    assert _MC_WORKER_STATE is not None, "Worker state not initialised."

    base: GridSearchConfig = _MC_WORKER_STATE["grid_config"]
    optimal: GridPointResult = _MC_WORKER_STATE["optimal"]
    scenario_prices: dict[str, list] = _MC_WORKER_STATE["scenario_prices"]
    mc: MCParams = _MC_WORKER_STATE["mc_params"]

    rng = np.random.default_rng(seed=mc.seed + iteration)

    # --- Sample price scenario ---
    weights = [mc.price_scenarios[n].weight for n in range(len(scenario_prices))]
    scenario_idx = rng.choice(list(range(len(scenario_prices))), p=weights)
    selected_price_scenario: PriceWeatherScenario = scenario_prices[scenario_idx]

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
    raw_avail = float(
        rng.normal(mu_sample, mc.sigma_bess_availability)
    )
    bess_availability_factor = float(
        np.clip(raw_avail, mc.mu_bess_availability, BESS_NOISE_CLIP_MAX)
    )

    # --- Stochastic BESS offline days: redrawn randomly per year ---
    n_bess_offline_days = round((1.0 - bess_availability_factor) * DAYS_PER_YEAR)
    n_bess_offline_days = max(0, min(n_bess_offline_days, DAYS_PER_YEAR))
    offline_days_yearly: list[set[int]] = []
    for _ in range(base.lifetime_years):
        if n_bess_offline_days > 0:
            day_indices = rng.choice(DAYS_PER_YEAR, size=n_bess_offline_days, replace=False)
            offline_days_yearly.append({int(d) for d in day_indices})
        else:
            offline_days_yearly.append(set())

    # --- Stochastic PV offline days: redrawn randomly per year ---
    n_pv_offline_days = round((1.0 - pv_availability_factor) * DAYS_PER_YEAR)
    n_pv_offline_days = max(0, min(n_pv_offline_days, DAYS_PER_YEAR))
    pv_offline_days_yearly: list[set[int]] = []
    for _ in range(base.lifetime_years):
        if n_pv_offline_days > 0:
            day_indices = rng.choice(DAYS_PER_YEAR, size=n_pv_offline_days, replace=False)
            pv_offline_days_yearly.append({int(d) for d in day_indices})
        else:
            pv_offline_days_yearly.append(set())

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

    # --- Replacement cost (scales with BESS CAPEX) ---
    replacement_cost = (
        base.replacement_fixed_eur
        + base.replacement_eur_per_kw * optimal.bess_power_kw
        + base.replacement_eur_per_kwh * optimal.bess_capacity_kwh
        + base.replacement_pct_of_capex * capex_bess
    )

    # --- Engine config ---
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

    # --- Run dispatch simulation ---
    sim = run_simulation(
        config=engine_config,
        pv_base_timeseries=selected_price_scenario.pv_timeseries_15min,
        spot_prices_yearly=selected_price_scenario.price_per_year,
        fixed_prices_yearly=base.fixed_prices_yearly,
        offline_days_yearly=offline_days_yearly,
        goo_prices_yearly=base.goo_prices_yearly if base.goo_prices_yearly else None,
        cap_prices_yearly=base.cap_prices_yearly if base.cap_prices_yearly else None,
        pv_offline_days_yearly=pv_offline_days_yearly if n_pv_offline_days > 0 else None,
        pv_base_timeseries_year=selected_price_scenario.weather_year,
        baseload_kw=base.baseload_mw
    )

    annual_revenues = [r.total_revenue for r in sim.annual_results]
    annual_bess_spot_revenues = [r.bess_spot_revenue for r in sim.annual_results]
    total_production_kwh = sum(r.pv_production for r in sim.annual_results)

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
    )

    return MCIterationResult(
        iteration=iteration,
        price_scenario=selected_price_scenario.name,
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
) -> MCResult:
    """Run the Monte Carlo simulation on the optimal BESS configuration.

    Each iteration samples stochastic noise factors (PV yield, CAPEX, OPEX,
    BESS availability) and a price scenario, then runs a full multi-year
    dispatch + cashflow simulation.

    Worker processes share the large read-only state (base config + all price
    scenarios) via the ``ProcessPoolExecutor`` initialiser, so price arrays
    are serialised only once per worker process rather than once per iteration.

    Parameters
    ----------
    base_config:
        The ``GridSearchConfig`` used for the grid search.  Provides all base
        parameters (BESS specs, finance, degradation rates, etc.).
    optimal:
        The optimal grid point from the grid search (highest Equity IRR).
        Determines BESS sizing, base CAPEX and OPEX.
    mc_params:
        Monte Carlo hyper-parameters (iterations, σ values, price scenarios).
    scenario_prices:
        Mapping from scenario name (e.g. ``"mid"``) to a list of per-year
        spot price arrays (each shape ``(intervals_per_year,)``, in €/kWh).
        Length of each list must equal ``base_config.lifetime_years``.
    scenario_pv_timeseries:
        Optional mapping from scenario name to the undegraded PV production
        timeseries for that scenario.  Each array has shape
        ``(intervals_per_year,)`` in kWh.  When provided, each MC iteration
        uses the PV timeseries from the sampled scenario.  When ``None``,
        all iterations use ``base_config.pv_base_timeseries``.

    Returns
    -------
    MCResult
        All iteration results plus overall and per-scenario statistics.

    Raises
    ------
    ValueError
        If a scenario name in ``mc_params.price_scenarios`` is not present in
        ``scenario_prices``.
    """
    logger.info(
        "Monte Carlo: %d iterations, %d price scenario(s), max_workers=%s.",
        mc_params.iterations,
        len(mc_params.price_scenarios),
        mc_params.max_workers,
    )

    shared_state: dict = {
        "grid_config": base_config,
        "optimal": optimal,
        "scenario_prices": scenario_prices,
        "mc_params": mc_params,
    }

    iteration_indices = list(range(1, mc_params.iterations + 1))
    results: list[MCIterationResult] = []

    n_iterations = mc_params.iterations
    log_interval = max(1, n_iterations // 10)

    if mc_params.max_workers == 1:
        # Single-process path: easier to debug and use in unit tests
        _mc_worker_init(shared_state)
        for i in iteration_indices:
            result = _run_mc_iteration(i)
            results.append(result)
            if i % log_interval == 0 or i == n_iterations:
                logger.debug("Monte Carlo: %d/%d iterations complete.", i, n_iterations)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=mc_params.max_workers,
            initializer=_mc_worker_init,
            initargs=(shared_state,),
        ) as executor:
            futures = {
                executor.submit(_run_mc_iteration, i): i
                for i in iteration_indices
            }
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                completed += 1
                if completed % log_interval == 0 or completed == n_iterations:
                    logger.debug(
                        "Monte Carlo: %d/%d iterations complete.", completed, n_iterations
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
