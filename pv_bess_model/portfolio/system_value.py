"""World A/B comparison, system value calculation, and enumeration.

World A is the baseline: PV generation vs. aggregated load, no flexibility.
The net position per quarter-hour determines grid buy/sell at spot price.

World B adds flexibility (BESS, heat pump, EV/V2G) which can shift energy
in time, reducing the net system cost.

System value = cost(World A) - cost(World B)

The enumeration runs all defined (flex_instance x addition_rate x e_to_p_ratio)
combinations using full multi-year dispatch simulations, parallelised across
independent points with ``concurrent.futures``.

Typical usage::

    from pv_bess_model.portfolio.system_value import (
        compute_world_a,
        run_enumeration,
    )

    result = compute_world_a(pv_profile, load_profile, spot_prices)
    enum_result = run_enumeration(config, pv, load, prices, flexibilities)
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np

from pv_bess_model.config.loader_portfolio import BessFlexConfig, FlexConfig
from pv_bess_model.dispatch.engine_portfolio import (
    PortfolioEngineConfig,
    run_portfolio_simulation,
)

logger = logging.getLogger(__name__)


@dataclass
class WorldAResult:
    """Result of the World A (no flexibility) calculation.

    Attributes
    ----------
    system_cost : float
        Net system cost in EUR.  Positive = net cost (buy > sell),
        negative = net revenue (sell > buy).
    total_sell_eur : float
        Total revenue from selling surplus to the grid (EUR).
    total_buy_eur : float
        Total cost of buying deficit from the grid (EUR).
    total_sell_kwh : float
        Total energy sold to the grid (kWh).
    total_buy_kwh : float
        Total energy bought from the grid (kWh).
    netto : numpy.ndarray
        Net position per interval (kWh).  Positive = surplus, negative = deficit.
    """

    system_cost: float
    total_sell_eur: float
    total_buy_eur: float
    total_sell_kwh: float
    total_buy_kwh: float
    netto: np.ndarray


def compute_world_a(
    pv_profile: np.ndarray,
    load_profile: np.ndarray,
    spot_prices: np.ndarray,
) -> WorldAResult:
    """Compute World A system cost (no flexibility).

    For each quarter-hour interval::

        netto[t] = pv[t] - load[t]
        netto > 0  →  surplus sold at spot[t]
        netto < 0  →  deficit bought at spot[t]

    System cost = total_buy - total_sell  (positive = net cost).

    Parameters
    ----------
    pv_profile:
        Aggregated PV production per interval in kWh.
    load_profile:
        Aggregated load per interval in kWh.
    spot_prices:
        Spot price per interval in EUR/kWh.

    Returns
    -------
    WorldAResult
        System cost breakdown and net position array.

    Raises
    ------
    ValueError
        When input arrays have different lengths.
    """
    if not (len(pv_profile) == len(load_profile) == len(spot_prices)):
        raise ValueError(
            f"All input arrays must have the same length. "
            f"Got pv={len(pv_profile)}, load={len(load_profile)}, "
            f"prices={len(spot_prices)}."
        )

    netto = pv_profile - load_profile

    # Surplus (netto > 0) → sell to grid
    sell_kwh = np.maximum(netto, 0.0)
    sell_eur = np.sum(sell_kwh * spot_prices)

    # Deficit (netto < 0) → buy from grid
    buy_kwh = np.maximum(-netto, 0.0)
    buy_eur = np.sum(buy_kwh * spot_prices)

    system_cost = buy_eur - sell_eur

    logger.info(
        "World A: sell=%.0f MWh (%.0f EUR), buy=%.0f MWh (%.0f EUR), "
        "system_cost=%.0f EUR.",
        float(np.sum(sell_kwh)) / 1000.0,
        sell_eur,
        float(np.sum(buy_kwh)) / 1000.0,
        buy_eur,
        system_cost,
    )

    return WorldAResult(
        system_cost=float(system_cost),
        total_sell_eur=float(sell_eur),
        total_buy_eur=float(buy_eur),
        total_sell_kwh=float(np.sum(sell_kwh)),
        total_buy_kwh=float(np.sum(buy_kwh)),
        netto=netto,
    )


# ---------------------------------------------------------------------------
# System value result containers
# ---------------------------------------------------------------------------


@dataclass
class SystemValuePoint:
    """Result for one (flex_instance, addition_rate, e_to_p_ratio) point.

    Attributes
    ----------
    flex_name : str
        Name of the flex instance (from config).
    flex_type : str
        Flex type identifier (``"bess"``, ``"heat_pump"``, ``"ev_charging"``).
    annual_addition_kw : float
        Annual BESS power addition rate in kW (or equivalent for other flex).
    e_to_p_ratio : float | None
        Energy-to-power ratio in hours (BESS only, ``None`` for others).
    cumulative_system_value_eur : float
        Sum over all project years of ``cost_A[y] - cost_B[y]``.
    annual_system_values : list[float]
        Per-year system value (``cost_A[y] - cost_B[y]``), length = lifetime.
    marginal_value_eur_per_kw_a : float
        Marginal value relative to the previous smaller addition rate.
        Set to 0.0 for the first (or zero) rate.
    """

    flex_name: str
    flex_type: str
    annual_addition_kw: float
    e_to_p_ratio: float | None
    cumulative_system_value_eur: float
    annual_system_values: list[float]
    marginal_value_eur_per_kw_a: float = 0.0


@dataclass
class SystemValueResult:
    """Aggregated result of the full enumeration.

    Attributes
    ----------
    world_a_annual_costs : list[float]
        Annual system costs for World A (no flexibility), length = lifetime.
    points : list[SystemValuePoint]
        All evaluated enumeration points.
    """

    world_a_annual_costs: list[float] = field(default_factory=list)
    points: list[SystemValuePoint] = field(default_factory=list)


# ---------------------------------------------------------------------------
# World A multi-year (no flexibility)
# ---------------------------------------------------------------------------


def compute_world_a_multiyear(
    config: PortfolioEngineConfig,
    pv_profile_base: np.ndarray,
    load_profile_base: np.ndarray,
    spot_prices_base: np.ndarray,
    pv_degradation_rate: float,
    load_growth_factor: float = 1.0,
) -> list[float]:
    """Compute annual World A system costs over the project lifetime.

    Uses ``run_portfolio_simulation`` with ``annual_addition_kw=0`` so no
    BESS is present, giving a pure World A result.

    Parameters
    ----------
    config:
        Engine configuration.
    pv_profile_base:
        Base PV production profile (one year, 35,040 values).
    load_profile_base:
        Base load profile (one year, 35,040 values).
    spot_prices_base:
        Base spot prices (one year, 35,040 values).
    pv_degradation_rate:
        Annual PV degradation fraction.
    load_growth_factor:
        Multiplicative annual load growth factor.

    Returns
    -------
    list[float]
        Annual system costs, length = ``config.lifetime_years``.
    """
    results = run_portfolio_simulation(
        config=config,
        pv_profile_base=pv_profile_base,
        load_profile_base=load_profile_base,
        spot_prices_base=spot_prices_base,
        annual_addition_kw=0.0,
        e_to_p_ratio=1.0,
        bess_rte=1.0,
        bess_min_soc_pct=0.0,
        bess_max_soc_pct=100.0,
        bess_degradation_rate=0.0,
        pv_degradation_rate=pv_degradation_rate,
        load_growth_factor=load_growth_factor,
    )
    return [r.system_cost for r in results]


# ---------------------------------------------------------------------------
# Single enumeration point
# ---------------------------------------------------------------------------


def _evaluate_bess_point(
    config: PortfolioEngineConfig,
    pv_profile_base: np.ndarray,
    load_profile_base: np.ndarray,
    spot_prices_base: np.ndarray,
    world_a_costs: list[float],
    flex_name: str,
    annual_addition_kw: float,
    e_to_p_ratio: float,
    bess_rte: float,
    bess_min_soc_pct: float,
    bess_max_soc_pct: float,
    bess_degradation_rate: float,
    pv_degradation_rate: float,
    load_growth_factor: float,
    start_year: int,
) -> SystemValuePoint:
    """Evaluate a single BESS enumeration point.

    Runs the full multi-year simulation and computes system value as
    delta to World A.
    """
    results = run_portfolio_simulation(
        config=config,
        pv_profile_base=pv_profile_base,
        load_profile_base=load_profile_base,
        spot_prices_base=spot_prices_base,
        annual_addition_kw=annual_addition_kw,
        e_to_p_ratio=e_to_p_ratio,
        bess_rte=bess_rte,
        bess_min_soc_pct=bess_min_soc_pct,
        bess_max_soc_pct=bess_max_soc_pct,
        bess_degradation_rate=bess_degradation_rate,
        pv_degradation_rate=pv_degradation_rate,
        load_growth_factor=load_growth_factor,
        start_year=start_year,
    )

    annual_system_values = [
        world_a_costs[i] - results[i].system_cost
        for i in range(len(results))
    ]
    cumulative = sum(annual_system_values)

    return SystemValuePoint(
        flex_name=flex_name,
        flex_type="bess",
        annual_addition_kw=annual_addition_kw,
        e_to_p_ratio=e_to_p_ratio,
        cumulative_system_value_eur=cumulative,
        annual_system_values=annual_system_values,
    )


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def run_enumeration(
    config: PortfolioEngineConfig,
    pv_profile_base: np.ndarray,
    load_profile_base: np.ndarray,
    spot_prices_base: np.ndarray,
    flexibilities: list[FlexConfig],
    pv_degradation_rate: float,
    load_growth_factor: float = 1.0,
    max_workers: int | None = None,
) -> SystemValueResult:
    """Run enumeration over all flex x addition_rate x e_to_p_ratio combinations.

    First computes World A (no flex), then evaluates each enumeration point.
    BESS points are parallelised with ``concurrent.futures``.

    Parameters
    ----------
    config:
        Engine configuration (lifetime, timestep, etc.).
    pv_profile_base:
        Base PV production profile (35,040 values).
    load_profile_base:
        Base load profile (35,040 values).
    spot_prices_base:
        Base spot prices (35,040 values).
    flexibilities:
        List of flex configurations to enumerate.
    pv_degradation_rate:
        Annual PV degradation fraction.
    load_growth_factor:
        Multiplicative annual load growth factor.
    max_workers:
        Maximum parallel workers.  ``None`` = CPU count.

    Returns
    -------
    SystemValueResult
        World A costs and all enumeration points with system values.
    """
    # 1. Compute World A
    logger.info("Computing World A (no flexibility) ...")
    world_a_costs = compute_world_a_multiyear(
        config=config,
        pv_profile_base=pv_profile_base,
        load_profile_base=load_profile_base,
        spot_prices_base=spot_prices_base,
        pv_degradation_rate=pv_degradation_rate,
        load_growth_factor=load_growth_factor,
    )
    logger.info(
        "World A total cost: %.0f EUR (over %d years)",
        sum(world_a_costs),
        len(world_a_costs),
    )

    # 2. Build enumeration tasks
    points: list[SystemValuePoint] = []

    for flex in flexibilities:
        if isinstance(flex, BessFlexConfig):
            bess_points = _enumerate_bess(
                config=config,
                pv_profile_base=pv_profile_base,
                load_profile_base=load_profile_base,
                spot_prices_base=spot_prices_base,
                world_a_costs=world_a_costs,
                flex=flex,
                pv_degradation_rate=pv_degradation_rate,
                load_growth_factor=load_growth_factor,
                max_workers=max_workers,
            )
            points.extend(bess_points)
        else:
            logger.warning(
                "Flex type '%s' (%s) not yet supported in enumeration, skipping.",
                flex.type,
                flex.name,
            )

    logger.info(
        "Enumeration complete: %d points evaluated.",
        len(points),
    )

    return SystemValueResult(
        world_a_annual_costs=world_a_costs,
        points=points,
    )


def _enumerate_bess(
    config: PortfolioEngineConfig,
    pv_profile_base: np.ndarray,
    load_profile_base: np.ndarray,
    spot_prices_base: np.ndarray,
    world_a_costs: list[float],
    flex: BessFlexConfig,
    pv_degradation_rate: float,
    load_growth_factor: float,
    max_workers: int | None,
) -> list[SystemValuePoint]:
    """Enumerate all (addition_rate x e_to_p_ratio) combinations for one BESS.

    Runs evaluations in parallel using ``ProcessPoolExecutor``.
    """
    rte = flex.round_trip_efficiency_pct / 100.0
    deg = flex.degradation_rate_pct_per_year / 100.0

    tasks: list[tuple[float, float]] = []
    for rate in flex.annual_addition_kw:
        for etp in flex.e_to_p_ratio_hours:
            tasks.append((rate, etp))

    n_tasks = len(tasks)
    logger.info(
        "BESS '%s': enumerating %d points (%d rates x %d E/P ratios) ...",
        flex.name,
        n_tasks,
        len(flex.annual_addition_kw),
        len(flex.e_to_p_ratio_hours),
    )

    results: list[SystemValuePoint] = []

    # For small task counts, run sequentially to avoid process overhead
    if n_tasks <= 2:
        for rate, etp in tasks:
            point = _evaluate_bess_point(
                config=config,
                pv_profile_base=pv_profile_base,
                load_profile_base=load_profile_base,
                spot_prices_base=spot_prices_base,
                world_a_costs=world_a_costs,
                flex_name=flex.name,
                annual_addition_kw=rate,
                e_to_p_ratio=etp,
                bess_rte=rte,
                bess_min_soc_pct=flex.min_soc_pct,
                bess_max_soc_pct=flex.max_soc_pct,
                bess_degradation_rate=deg,
                pv_degradation_rate=pv_degradation_rate,
                load_growth_factor=load_growth_factor,
                start_year=flex.start_year,
            )
            results.append(point)
            logger.info(
                "  BESS '%s' rate=%.0f kW/a E/P=%.1fh: "
                "system_value=%.0f EUR",
                flex.name,
                rate,
                etp,
                point.cumulative_system_value_eur,
            )
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {}
            for rate, etp in tasks:
                future = executor.submit(
                    _evaluate_bess_point,
                    config=config,
                    pv_profile_base=pv_profile_base,
                    load_profile_base=load_profile_base,
                    spot_prices_base=spot_prices_base,
                    world_a_costs=world_a_costs,
                    flex_name=flex.name,
                    annual_addition_kw=rate,
                    e_to_p_ratio=etp,
                    bess_rte=rte,
                    bess_min_soc_pct=flex.min_soc_pct,
                    bess_max_soc_pct=flex.max_soc_pct,
                    bess_degradation_rate=deg,
                    pv_degradation_rate=pv_degradation_rate,
                    load_growth_factor=load_growth_factor,
                    start_year=flex.start_year,
                )
                future_map[future] = (rate, etp)

            for future in as_completed(future_map):
                rate, etp = future_map[future]
                point = future.result()
                results.append(point)
                logger.info(
                    "  BESS '%s' rate=%.0f kW/a E/P=%.1fh: "
                    "system_value=%.0f EUR",
                    flex.name,
                    rate,
                    etp,
                    point.cumulative_system_value_eur,
                )

    # Sort by (e_to_p_ratio, annual_addition_kw) for consistent output
    results.sort(key=lambda p: (p.e_to_p_ratio or 0, p.annual_addition_kw))

    return results
