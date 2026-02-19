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
    BESS_NOISE_CLIP_MIN,
    DAYS_PER_YEAR,
    DEFAULT_MC_ITERATIONS,
    MC_WEIGHT_TOLERANCE,
)
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
    sigma_pv_yield:
        Standard deviation for the PV yield noise factor N(1, σ).
        e.g. 0.05 for 5 %.
    sigma_capex:
        Standard deviation for the CAPEX noise factor N(1, σ).
    sigma_opex:
        Standard deviation for the OPEX noise factor N(1, σ).
    mu_bess_availability:
        Mean of the BESS availability noise factor (fraction, 0–1).
        e.g. 0.97 for 97 %.
    sigma_bess_availability:
        Standard deviation of the BESS availability noise factor.
    price_scenarios:
        Mapping from scenario name to ``{"csv_column": str, "weight": float}``.
        Weights must sum to 1.0 (within ``MC_WEIGHT_TOLERANCE``).
        Example::

            {
                "low":  {"csv_column": "LOW",  "weight": 0.25},
                "mid":  {"csv_column": "MID",  "weight": 0.50},
                "high": {"csv_column": "HIGH", "weight": 0.25},
            }

    seed:
        Base random seed for reproducibility. Each iteration uses
        ``seed + iteration`` as its own seed.
    max_workers:
        Number of parallel worker processes. None = os.cpu_count().
    """

    iterations: int = DEFAULT_MC_ITERATIONS
    sigma_pv_yield: float = 0.05
    sigma_capex: float = 0.08
    sigma_opex: float = 0.05
    mu_bess_availability: float = 0.97
    sigma_bess_availability: float = 0.02
    price_scenarios: dict[str, dict] = field(default_factory=dict)
    seed: int = 0
    max_workers: int | None = None

    def __post_init__(self) -> None:
        """Set default price scenarios and validate weights."""
        if not self.price_scenarios:
            self.price_scenarios = {"mid": {"csv_column": "MID", "weight": 1.0}}
        weights = sum(v["weight"] for v in self.price_scenarios.values())
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
    pv_yield_factor:
        Sampled PV yield noise factor.
    capex_factor:
        Sampled CAPEX noise factor.
    opex_factor:
        Sampled OPEX noise factor.
    bess_availability_factor:
        Sampled BESS availability factor (fraction, clipped to [0, 1]).
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
    pv_yield_factor: float
    capex_factor: float
    opex_factor: float
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
    scenario_names = list(mc.price_scenarios.keys())
    weights = [mc.price_scenarios[n]["weight"] for n in scenario_names]
    scenario_name = str(rng.choice(scenario_names, p=weights))
    spot_prices_yearly: list[np.ndarray] = scenario_prices[scenario_name]

    # --- Sample noise factors ---
    pv_yield_factor = float(rng.normal(1.0, mc.sigma_pv_yield))
    capex_factor = float(rng.normal(1.0, mc.sigma_capex))
    opex_factor = float(rng.normal(1.0, mc.sigma_opex))
    raw_avail = float(
        rng.normal(mc.mu_bess_availability, mc.sigma_bess_availability)
    )
    bess_availability_factor = float(
        np.clip(raw_avail, BESS_NOISE_CLIP_MIN, BESS_NOISE_CLIP_MAX)
    )

    # --- Stochastic offline days: redrawn randomly per year ---
    n_offline_days = round((1.0 - bess_availability_factor) * DAYS_PER_YEAR)
    n_offline_days = max(0, min(n_offline_days, DAYS_PER_YEAR))
    offline_days_yearly: list[set[int]] = []
    for _ in range(base.lifetime_years):
        if n_offline_days > 0:
            day_indices = rng.choice(DAYS_PER_YEAR, size=n_offline_days, replace=False)
            offline_days_yearly.append({int(d) for d in day_indices})
        else:
            offline_days_yearly.append(set())

    # --- Apply PV yield factor ---
    pv_timeseries = base.pv_base_timeseries_p50 * pv_yield_factor

    # --- Scale CAPEX / OPEX ---
    capex_total = optimal.capex_total * capex_factor
    capex_pv = optimal.capex_pv * capex_factor
    capex_bess = optimal.capex_bess * capex_factor
    opex_base = optimal.opex_base * opex_factor

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
    )

    # --- Run dispatch simulation ---
    sim = run_simulation(
        config=engine_config,
        pv_base_timeseries=pv_timeseries,
        spot_prices_yearly=spot_prices_yearly,
        fixed_prices_yearly=base.fixed_prices_yearly,
        offline_days_yearly=offline_days_yearly,
    )

    annual_revenues = [r.total_revenue for r in sim.annual_results]
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
    )

    annual_opex = [
        inflate_value(opex_base, base.inflation_rate, y)
        for y in range(1, base.lifetime_years + 1)
    ]
    annual_debt_service = [
        cf.years[y].debt_service for y in range(1, base.lifetime_years + 1)
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
        price_scenario=scenario_name,
        pv_yield_factor=pv_yield_factor,
        capex_factor=capex_factor,
        opex_factor=opex_factor,
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
    scenario_prices: dict[str, list[np.ndarray]],
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
        spot price arrays (each shape (8760,), in €/kWh).  Length of each
        list must equal ``base_config.lifetime_years``.

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
    for name in mc_params.price_scenarios:
        if name not in scenario_prices:
            raise ValueError(
                f"MC price scenario '{name}' not found in scenario_prices. "
                f"Available keys: {sorted(scenario_prices)}."
            )

    logger.info(
        "Monte Carlo: %d iterations, %d price scenario(s), max_workers=%s.",
        mc_params.iterations,
        len(mc_params.price_scenarios),
        mc_params.max_workers,
    )

    shared_state = {
        "grid_config": base_config,
        "optimal": optimal,
        "scenario_prices": scenario_prices,
        "mc_params": mc_params,
    }

    iteration_indices = list(range(1, mc_params.iterations + 1))
    results: list[MCIterationResult] = []

    if mc_params.max_workers == 1:
        # Single-process path: easier to debug and use in unit tests
        _mc_worker_init(shared_state)
        results = [_run_mc_iteration(i) for i in iteration_indices]
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
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

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
