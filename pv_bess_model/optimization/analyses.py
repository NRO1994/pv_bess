"""Post-grid-search sensitivity analyses.

Three analysis types are supported, each sweeping over marketing parameters
and running a full Monte Carlo simulation for every parameter combination:

1. **EEG sensitivity** – Sweep over EEG floor prices.
2. **PPA Collar optimisation** – 2D sweep over floor price × cap spread.
3. **PPA Baseload optimisation** – 2D sweep over PPA price × baseload MW.

Public API
----------
AnalysisResult      – Result of a single analysis point (params + MC result).
SensitivityResult   – Complete result of a sensitivity analysis (all points).
run_eeg_sensitivity – EEG floor price sweep.
run_ppa_collar_analysis – PPA Collar 2D sweep.
run_ppa_baseload_analysis – PPA Baseload 2D sweep.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

import numpy as np

from pv_bess_model.config.loader import PriceWeatherScenario
from pv_bess_model.finance.inflation import inflate_value
from pv_bess_model.market.eeg import EegConfig, effective_eeg_price
from pv_bess_model.optimization.grid_search import GridPointResult, GridSearchConfig
from pv_bess_model.optimization.monte_carlo import MCParams, MCResult, run_monte_carlo

logger = logging.getLogger(__name__)




# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class AnalysisResult:
    """Result of a single analysis point (one parameter combination + MC)."""

    params: dict[str, float]
    mc_result: MCResult


@dataclass
class SensitivityResult:
    """Complete result of a sensitivity analysis."""

    analysis_type: str
    points: list[AnalysisResult]


# ---------------------------------------------------------------------------
# Config modification helpers
# ---------------------------------------------------------------------------


def _build_eeg_fixed_prices(
    floor_price_eur_per_kwh: float,
    lifetime: int,
    inflation_rate: float,
    eeg_inflation: bool,
    fixed_price_years: int,
) -> list[float]:
    """Build per-year fixed prices for an EEG floor price scenario.

    Parameters
    ----------
    floor_price_eur_per_kwh:
        EEG floor price in EUR/kWh.
    lifetime:
        Project lifetime in years.
    inflation_rate:
        Annual inflation rate as fraction.
    eeg_inflation:
        Whether to apply inflation to the EEG floor price.
    fixed_price_years:
        Number of years the EEG floor applies.

    Returns
    -------
    list[float]
        Floor price per year (length = lifetime). 0.0 after fixed_price_years.
    """
    cfg = EegConfig(
        floor_price_eur_per_kwh=floor_price_eur_per_kwh,
        fixed_price_years=fixed_price_years,
        inflation_enabled=eeg_inflation,
    )
    return [
        effective_eeg_price(cfg, year, inflation_rate)
        for year in range(1, lifetime + 1)
    ]


def _build_collar_prices(
    floor_eur_per_kwh: float,
    cap_eur_per_kwh: float,
    duration_years: int,
    inflation_on_ppa: bool,
    goo_premium_eur_per_kwh: float,
    inflation_rate: float,
    lifetime: int,
) -> tuple[list[float], list[float], list[float]]:
    """Build per-year fixed, cap, and GoO prices for a PPA Collar scenario.

    Returns
    -------
    tuple[list[float], list[float], list[float]]
        (fixed_prices_yearly, cap_prices_yearly, goo_prices_yearly)
    """
    fixed_prices: list[float] = []
    cap_prices: list[float] = []
    goo_prices: list[float] = []

    for year in range(1, lifetime + 1):
        if year <= duration_years:
            if inflation_on_ppa:
                floor = inflate_value(floor_eur_per_kwh, inflation_rate, year)
                cap = inflate_value(cap_eur_per_kwh, inflation_rate, year)
            else:
                floor = floor_eur_per_kwh
                cap = cap_eur_per_kwh
            fixed_prices.append(floor)
            cap_prices.append(cap)
            goo_prices.append(goo_premium_eur_per_kwh)
        else:
            fixed_prices.append(0.0)
            cap_prices.append(0.0)
            goo_prices.append(0.0)

    return fixed_prices, cap_prices, goo_prices


def _build_baseload_prices(
    ppa_price_eur_per_kwh: float,
    duration_years: int,
    inflation_on_ppa: bool,
    goo_premium_eur_per_kwh: float,
    inflation_rate: float,
    lifetime: int,
) -> tuple[list[float], list[float]]:
    """Build per-year fixed and GoO prices for a PPA Baseload scenario.

    Baseload is modelled as a fixed price per kWh (like pay-as-produced),
    since the dispatch decisions do not change. GoO is added on top.

    Returns
    -------
    tuple[list[float], list[float]]
        (fixed_prices_yearly, goo_prices_yearly)
    """
    fixed_prices: list[float] = []
    goo_prices: list[float] = []

    for year in range(1, lifetime + 1):
        if year <= duration_years:
            if inflation_on_ppa:
                price = inflate_value(ppa_price_eur_per_kwh, inflation_rate, year)
            else:
                price = ppa_price_eur_per_kwh
            # For baseload, GoO is baked into fixed price (like pay-as-produced)
            fixed_prices.append(price + goo_premium_eur_per_kwh)
            goo_prices.append(0.0)
        else:
            fixed_prices.append(0.0)
            goo_prices.append(0.0)

    return fixed_prices, goo_prices


# ---------------------------------------------------------------------------
# Public analysis functions
# ---------------------------------------------------------------------------


def run_eeg_sensitivity(
    base_config: GridSearchConfig,
    optimal: GridPointResult,
    mc_params: MCParams,
    scenario_prices: list[PriceWeatherScenario],
    floor_prices: list[float],
    inflation_rate: float,
    eeg_inflation: bool,
    fixed_price_years: int,
    scenario_pv_timeseries: dict[str, np.ndarray] | None = None,
) -> SensitivityResult:
    """Run EEG floor price sensitivity analysis.

    For each floor price, a modified config is created and a full MC
    simulation is executed.

    Parameters
    ----------
    base_config:
        GridSearchConfig from the grid search (used as template).
    optimal:
        Optimal grid point from grid search.
    mc_params:
        Monte Carlo parameters (iterations, sigma values, etc.).
    scenario_prices:
        Per-scenario spot price arrays for MC.
    floor_prices:
        List of EEG floor prices to sweep (EUR/kWh).
    inflation_rate:
        Annual inflation rate as fraction.
    eeg_inflation:
        Whether EEG floor price is inflation-adjusted.
    fixed_price_years:
        Number of years the EEG floor applies.
    scenario_pv_timeseries:
        Optional per-scenario PV timeseries for MC.

    Returns
    -------
    SensitivityResult
        Results for all floor price points.
    """
    points: list[AnalysisResult] = []
    n_total = len(floor_prices)

    for idx, floor_price in enumerate(floor_prices, start=1):
        logger.info(
            "EEG sensitivity %d/%d: floor=%.4f EUR/kWh",
            idx, n_total, floor_price,
        )

        new_fixed = _build_eeg_fixed_prices(
            floor_price_eur_per_kwh=floor_price,
            lifetime=base_config.lifetime_years,
            inflation_rate=inflation_rate,
            eeg_inflation=eeg_inflation,
            fixed_price_years=fixed_price_years,
        )

        modified_config = dataclasses.replace(
            base_config,
            fixed_prices_yearly=new_fixed,
            # EEG has no GoO or cap
            goo_prices_yearly=[0.0] * base_config.lifetime_years,
            cap_prices_yearly=[0.0] * base_config.lifetime_years,
        )

        mc_result = run_monte_carlo(
            base_config=modified_config,
            optimal=optimal,
            mc_params=mc_params,
            scenario_prices=scenario_prices,
        )

        points.append(AnalysisResult(
            params={"floor_price_eur_per_kwh": floor_price},
            mc_result=mc_result,
        ))

    return SensitivityResult(analysis_type="eeg_sensitivity", points=points)


def run_ppa_collar_analysis(
    base_config: GridSearchConfig,
    optimal: GridPointResult,
    mc_params: MCParams,
    scenario_prices: list[PriceWeatherScenario],
    floor_prices_eur_per_kwh: list[float],
    cap_spreads_eur_per_kwh: list[float],
    duration_years: int,
    inflation_on_ppa: bool,
    goo_premium_eur_per_kwh: float,
    inflation_rate: float,
) -> SensitivityResult:
    """Run PPA Collar 2D sensitivity analysis.

    Sweeps over all (floor_price, cap_spread) combinations. Cap price is
    computed as floor_price + cap_spread.

    Parameters
    ----------
    base_config:
        GridSearchConfig from the grid search (used as template).
    optimal:
        Optimal grid point from grid search.
    mc_params:
        Monte Carlo parameters.
    scenario_prices:
        Per-scenario spot price arrays for MC.
    floor_prices_eur_per_kwh:
        Floor prices to sweep (EUR/kWh).
    cap_spreads_eur_per_kwh:
        Cap spreads to sweep (EUR/kWh). Cap = floor + spread.
    duration_years:
        PPA contract duration in years.
    inflation_on_ppa:
        Whether PPA prices are inflation-adjusted.
    goo_premium_eur_per_kwh:
        GoO premium in EUR/kWh.
    inflation_rate:
        Annual inflation rate as fraction.
    scenario_pv_timeseries:
        Optional per-scenario PV timeseries for MC.

    Returns
    -------
    SensitivityResult
        Results for all (floor, cap_spread) combinations.
    """
    points: list[AnalysisResult] = []
    combinations = [
        (floor, spread)
        for floor in floor_prices_eur_per_kwh
        for spread in cap_spreads_eur_per_kwh
    ]
    n_total = len(combinations)

    for idx, (floor_kwh, spread_kwh) in enumerate(combinations, start=1):
        cap_kwh = floor_kwh + spread_kwh

        logger.info(
            "PPA Collar %d/%d: floor=%.4f EUR/kWh, cap_spread=%.4f EUR/kWh, "
            "cap=%.4f EUR/kWh",
            idx, n_total, floor_kwh, spread_kwh, cap_kwh,
        )

        new_fixed, new_cap, new_goo = _build_collar_prices(
            floor_eur_per_kwh=floor_kwh,
            cap_eur_per_kwh=cap_kwh,
            duration_years=duration_years,
            inflation_on_ppa=inflation_on_ppa,
            goo_premium_eur_per_kwh=goo_premium_eur_per_kwh,
            inflation_rate=inflation_rate,
            lifetime=base_config.lifetime_years,
        )

        modified_config = dataclasses.replace(
            base_config,
            fixed_prices_yearly=new_fixed,
            cap_prices_yearly=new_cap,
            goo_prices_yearly=new_goo,
        )

        mc_result = run_monte_carlo(
            base_config=modified_config,
            optimal=optimal,
            mc_params=mc_params,
            scenario_prices=scenario_prices,
        )

        points.append(AnalysisResult(
            params={
                "floor_price_eur_per_kwh": floor_kwh,
                "cap_spread_eur_per_kwh": spread_kwh,
                "cap_price_eur_per_kwh": cap_kwh,
            },
            mc_result=mc_result,
        ))

    return SensitivityResult(analysis_type="ppa_collar", points=points)


def run_ppa_baseload_analysis(
    base_config: GridSearchConfig,
    optimal: GridPointResult,
    mc_params: MCParams,
    scenario_prices: list[PriceWeatherScenario],
    ppa_prices_eur_per_kwh: list[float],
    baseload_levels_mw: list[float],
    duration_years: int,
    inflation_on_ppa: bool,
    goo_premium_eur_per_kwh: float,
    inflation_rate: float,
) -> SensitivityResult:
    """Run PPA Baseload 2D sensitivity analysis.

    Sweeps over all (ppa_price, baseload_mw) combinations. Baseload is
    modelled as a fixed price (like pay-as-produced) since dispatch
    decisions do not change.

    Parameters
    ----------
    base_config:
        GridSearchConfig from the grid search (used as template).
    optimal:
        Optimal grid point from grid search.
    mc_params:
        Monte Carlo parameters.
    scenario_prices:
        Per-scenario spot price arrays for MC.
    ppa_prices_eur_per_kwh:
        PPA prices to sweep (EUR/kWh).
    baseload_levels_mw:
        Baseload levels to sweep (MW). Used for labelling only (dispatch
        is not altered).
    duration_years:
        PPA contract duration in years.
    inflation_on_ppa:
        Whether PPA prices are inflation-adjusted.
    goo_premium_eur_per_kwh:
        GoO premium in EUR/kWh.
    inflation_rate:
        Annual inflation rate as fraction.
    scenario_pv_timeseries:
        Optional per-scenario PV timeseries for MC.

    Returns
    -------
    SensitivityResult
        Results for all (ppa_price, baseload_mw) combinations.
    """
    points: list[AnalysisResult] = []
    combinations = [
        (price, bl)
        for price in ppa_prices_eur_per_kwh
        for bl in baseload_levels_mw
    ]
    n_total = len(combinations)

    for idx, (price_kwh, baseload_mw) in enumerate(combinations, start=1):
        logger.info(
            "PPA Baseload %d/%d: price=%.4f EUR/kWh, baseload=%.2f MW",
            idx, n_total, price_kwh, baseload_mw,
        )

        new_fixed, new_goo = _build_baseload_prices(
            ppa_price_eur_per_kwh=price_kwh,
            duration_years=duration_years,
            inflation_on_ppa=inflation_on_ppa,
            goo_premium_eur_per_kwh=goo_premium_eur_per_kwh,
            inflation_rate=inflation_rate,
            lifetime=base_config.lifetime_years,
        )

        modified_config = dataclasses.replace(
            base_config,
            fixed_prices_yearly=new_fixed,
            goo_prices_yearly=new_goo,
            # Baseload has no cap
            cap_prices_yearly=[0.0] * base_config.lifetime_years,
            baseload_mw=baseload_mw,
        )

        mc_result = run_monte_carlo(
            base_config=modified_config,
            optimal=optimal,
            mc_params=mc_params,
            scenario_prices=scenario_prices,
        )

        points.append(AnalysisResult(
            params={
                "ppa_price_eur_per_kwh": price_kwh,
                "baseload_mw": baseload_mw,
            },
            mc_result=mc_result,
        ))

    return SensitivityResult(analysis_type="ppa_baseload", points=points)
