"""Tests for optimization/monte_carlo.py.

Uses tiny synthetic data (2 project years, 100 kWp PV, 10-20 MC iterations)
so that every test completes in a few seconds.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pv_bess_model.config.defaults import (
    INTERVALS_PER_HOUR,
    INTERVALS_PER_YEAR,
    TIMESTEP_HOURS,
)
from pv_bess_model.config.loader import PriceWeatherScenario
from pv_bess_model.optimization.grid_search import (
    GridSearchConfig,
    GridPointResult,
    GridSearchResult,
    run_grid_search,
)
from pv_bess_model.optimization.monte_carlo import (
    MCParams,
    MCResult,
    MCStatistics,
    run_monte_carlo,
)


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

LIFETIME_YEARS = 2
PV_PEAK_KWP = 100.0
MC_ITERATIONS = 15   # fast yet enough for variance checks


def _make_price_array(price_eur_per_kwh: float = 0.06) -> np.ndarray:
    return np.full(INTERVALS_PER_YEAR, price_eur_per_kwh, dtype=float)


def _make_pv_array(peak_kwh: float = 25.0) -> np.ndarray:
    hour_of_day = np.arange(INTERVALS_PER_YEAR) % (24 * INTERVALS_PER_HOUR) // INTERVALS_PER_HOUR
    daylight = np.where(
        (hour_of_day >= 6) & (hour_of_day <= 18),
        np.sin(np.pi * (hour_of_day - 6) / 12),
        0.0,
    )
    return (peak_kwh / INTERVALS_PER_HOUR * daylight).astype(float)


def _make_price_weather_scenario(
    name: str,
    csv_column: str,
    weight: float,
    pv_timeseries: np.ndarray,
    price_per_year: np.ndarray,
    weather_year: int = 2020,
) -> PriceWeatherScenario:
    """Build a PriceWeatherScenario for testing."""
    return PriceWeatherScenario(
        name=name,
        label=name,
        csv_column=csv_column,
        weather_year=weather_year,
        weight=weight,
        is_central=(name == "mid"),
        pv_timeseries_15min=pv_timeseries,
        price_per_year=price_per_year,
    )


def _make_single_mid_scenario_list() -> list[PriceWeatherScenario]:
    """Build a single-element list with a 'mid' PriceWeatherScenario (weight=1.0)."""
    pv = _make_pv_array(25.0)
    return [PriceWeatherScenario(
        name="mid", label="mid", csv_column="MID",
        weather_year=2020, weight=1.0, is_central=True,
        pv_timeseries_15min=pv,
        price_per_year=None,  # will be set from base_config in fixtures
    )]


def _zero_sigma_params(
    iterations: int = MC_ITERATIONS,
    seed: int = 0,
    mu_bess_availability: float = 1.0,
    **overrides,
) -> MCParams:
    """Build MCParams with all sigmas at zero (deterministic) unless overridden."""
    # Build default price_scenarios as list[PriceWeatherScenario]
    if "price_scenarios" not in overrides:
        pv = _make_pv_array(25.0)
        default_scenarios = [PriceWeatherScenario(
            name="mid", label="mid", csv_column="MID",
            weather_year=2020, weight=1.0, is_central=True,
            pv_timeseries_15min=pv,
            price_per_year=None,
        )]
    else:
        default_scenarios = overrides.pop("price_scenarios")

    defaults = dict(
        iterations=iterations,
        sigma_capex_pv=0.0,
        sigma_capex_bess=0.0,
        sigma_opex_pv=0.0,
        sigma_opex_bess=0.0,
        sigma_pv_availability=0.0,
        mu_bess_availability=mu_bess_availability,
        sigma_bess_availability=0.0,
        price_scenarios=default_scenarios,
        seed=seed,
        max_workers=1,
    )
    defaults.update(overrides)
    return MCParams(**defaults)


@pytest.fixture(scope="module")
def base_config() -> GridSearchConfig:
    """Minimal GridSearchConfig shared by all MC tests (module scope for speed).

    CAPEX is set so that Year 1 CF is negative (investment year) while
    subsequent years are positive, giving a valid IRR sign change.
    Zero leverage keeps things simple.
    """
    spot = _make_price_array(0.06)
    pv = _make_pv_array(25.0)
    return GridSearchConfig(
        scale_pct_of_pv=[0.0, 30.0],
        e_to_p_ratio_hours=[2.0],
        pv_peak_kwp=PV_PEAK_KWP,
        pv_base_timeseries=pv,
        pv_degradation_rate=0.004,
        # CAPEX large enough that Year 1 CF is negative (CAPEX > Year 1 revenue)
        pv_costs_capex={"eur_per_kw": 50.0},
        pv_costs_opex={"pct_of_capex": 0.01},
        bess_rte=0.90,
        bess_min_soc_pct=10.0,
        bess_max_soc_pct=90.0,
        bess_degradation_rate=0.02,
        bess_availability_pct=100.0,
        bess_costs_capex={"eur_per_kw": 10.0, "eur_per_kwh": 10.0},
        bess_costs_opex={"pct_of_capex": 0.02},
        replacement_enabled=False,
        replacement_year=0,
        replacement_fixed_eur=0.0,
        replacement_eur_per_kw=0.0,
        replacement_eur_per_kwh=0.0,
        replacement_pct_of_capex=0.0,
        replacement_capacity_factor_pct=100.0,
        grid_max_kw=80.0,
        grid_loss_factor=1.0,
        grid_costs_capex={},
        grid_costs_opex={},
        operating_mode="green",
        spot_prices_yearly=[spot.copy() for _ in range(LIFETIME_YEARS)],
        fixed_prices_yearly=[0.0] * LIFETIME_YEARS,
        lifetime_years=LIFETIME_YEARS,
        leverage_pct=0.0,      # no debt → no debt-service drag
        interest_rate_pct=4.5,
        loan_tenor_years=2,
        inflation_rate=0.02,
        discount_rate=0.06,
        afa_years_pv=5,
        afa_years_bess=5,
        gewerbesteuer_messzahl=0.035,
        gewerbesteuer_hebesatz=400,
        koerperschaftsteuer_pct=15.0,
        solidaritaetszuschlag_pct=5.5,

        pv_base_timeseries_year=2020,
        pv_availability_pct=100.0,
        baseload_mw=0.0,
        commissioning_year=2027,
        timestep_hours=TIMESTEP_HOURS,
        intervals_per_day=INTERVALS_PER_HOUR * 24,
        intervals_per_year=INTERVALS_PER_YEAR,
        max_workers=1,
    )


@pytest.fixture(scope="module")
def deterministic_optimal(base_config: GridSearchConfig) -> GridPointResult:
    """Run the deterministic grid search and return the optimal point."""
    result = run_grid_search(base_config)
    assert result.optimal is not None
    return result.optimal


@pytest.fixture(scope="module")
def mid_scenario_prices(base_config: GridSearchConfig) -> list[PriceWeatherScenario]:
    """Single 'mid' price scenario matching the base_config spot prices."""
    pv = base_config.pv_base_timeseries
    return [PriceWeatherScenario(
        name="mid", label="mid", csv_column="MID",
        weather_year=2020, weight=1.0, is_central=True,
        pv_timeseries_15min=pv,
        price_per_year=base_config.spot_prices_yearly,
    )]


# ---------------------------------------------------------------------------
# MCParams validation
# ---------------------------------------------------------------------------


class TestMCParamsValidation:
    def test_single_scenario_weight_one(self) -> None:
        """Single scenario with weight=1.0 is valid."""
        pv = _make_pv_array()
        params = MCParams(
            iterations=5,
            price_scenarios=[PriceWeatherScenario(
                name="mid", label="mid", csv_column="MID",
                weather_year=2020, weight=1.0, is_central=True,
                pv_timeseries_15min=pv,
            )],
        )
        assert params.price_scenarios[0].weight == pytest.approx(1.0)

    def test_weights_not_summing_to_one_raises(self) -> None:
        """Weights that do not sum to 1.0 raise ValueError."""
        pv = _make_pv_array()
        with pytest.raises(ValueError, match="weights must sum to 1.0"):
            MCParams(
                price_scenarios=[
                    PriceWeatherScenario(name="low", label="low", csv_column="LOW", weather_year=2020, weight=0.3, pv_timeseries_15min=pv),
                    PriceWeatherScenario(name="high", label="high", csv_column="HIGH", weather_year=2020, weight=0.3, pv_timeseries_15min=pv),
                ]
            )

    def test_two_scenarios_valid(self) -> None:
        """Two scenarios summing to 1.0 do not raise."""
        pv = _make_pv_array()
        params = MCParams(
            price_scenarios=[
                PriceWeatherScenario(name="low", label="low", csv_column="LOW", weather_year=2020, weight=0.4, pv_timeseries_15min=pv),
                PriceWeatherScenario(name="high", label="high", csv_column="HIGH", weather_year=2020, weight=0.6, pv_timeseries_15min=pv),
            ]
        )
        total = sum(s.weight for s in params.price_scenarios)
        assert total == pytest.approx(1.0)

    def test_three_scenarios_valid(self) -> None:
        pv = _make_pv_array()
        params = MCParams(
            price_scenarios=[
                PriceWeatherScenario(name="low", label="low", csv_column="LOW", weather_year=2020, weight=0.25, pv_timeseries_15min=pv),
                PriceWeatherScenario(name="mid", label="mid", csv_column="MID", weather_year=2020, weight=0.50, pv_timeseries_15min=pv),
                PriceWeatherScenario(name="high", label="high", csv_column="HIGH", weather_year=2020, weight=0.25, pv_timeseries_15min=pv),
            ]
        )
        total = sum(s.weight for s in params.price_scenarios)
        assert total == pytest.approx(1.0)

    def test_mismatched_scenario_count_raises(
        self,
        base_config: GridSearchConfig,
        deterministic_optimal: GridPointResult,
    ) -> None:
        """MCParams with 2 scenarios but scenario_prices with 1 entry raises an error."""
        pv = _make_pv_array()
        two_param_scenarios = [
            PriceWeatherScenario(name="low", label="low", csv_column="LOW", weather_year=2020, weight=0.5, pv_timeseries_15min=pv),
            PriceWeatherScenario(name="high", label="high", csv_column="HIGH", weather_year=2020, weight=0.5, pv_timeseries_15min=pv),
        ]
        params = MCParams(
            iterations=2,
            price_scenarios=two_param_scenarios,
        )
        one_scenario = [PriceWeatherScenario(
            name="mid", label="mid", csv_column="MID",
            weather_year=2020, weight=1.0, is_central=True,
            pv_timeseries_15min=base_config.pv_base_timeseries,
            price_per_year=base_config.spot_prices_yearly,
        )]
        with pytest.raises((ValueError, IndexError)):
            run_monte_carlo(
                base_config=base_config,
                optimal=deterministic_optimal,
                mc_params=params,
                scenario_prices=one_scenario,
            )


# ---------------------------------------------------------------------------
# Deterministic result: σ = 0 for ALL noise factors
# ---------------------------------------------------------------------------


class TestDeterministicWithZeroSigma:
    """Most important test: when all σ = 0, every MC iteration reproduces the
    deterministic grid search result exactly."""

    def test_all_iterations_match_deterministic_irr(
        self,
        base_config: GridSearchConfig,
        deterministic_optimal: GridPointResult,
        mid_scenario_prices: dict,
    ) -> None:
        """With σ=0 and μ_avail=1.0, all MC equity IRRs equal the grid search IRR."""
        params = _zero_sigma_params(iterations=MC_ITERATIONS, seed=42)
        result = run_monte_carlo(
            base_config=base_config,
            optimal=deterministic_optimal,
            mc_params=params,
            scenario_prices=mid_scenario_prices,
        )
        det_irr = deterministic_optimal.metrics.equity_irr
        assert det_irr is not None, "Deterministic IRR must be computable"
        for it in result.iterations:
            assert it.equity_irr is not None
            assert it.equity_irr == pytest.approx(det_irr, rel=1e-6), (
                f"Iteration {it.iteration}: MC IRR {it.equity_irr:.6f} "
                f"!= deterministic {det_irr:.6f}"
            )

    def test_all_noise_factors_are_one_with_zero_sigma(
        self,
        base_config: GridSearchConfig,
        deterministic_optimal: GridPointResult,
        mid_scenario_prices: dict,
    ) -> None:
        """With σ=0, all sampled noise factors equal their mean (1.0)."""
        params = _zero_sigma_params(iterations=5, seed=0)
        result = run_monte_carlo(
            base_config=base_config,
            optimal=deterministic_optimal,
            mc_params=params,
            scenario_prices=mid_scenario_prices,
        )
        for it in result.iterations:
            assert it.capex_factor_pv == pytest.approx(1.0, abs=1e-12)
            assert it.capex_factor_bess == pytest.approx(1.0, abs=1e-12)
            assert it.opex_factor_pv == pytest.approx(1.0, abs=1e-12)
            assert it.opex_factor_bess == pytest.approx(1.0, abs=1e-12)
            assert it.pv_availability_factor == pytest.approx(1.0, abs=1e-12)
            assert it.bess_availability_factor == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Single price scenario → no scenario sampling
# ---------------------------------------------------------------------------


class TestSinglePriceScenario:
    def test_all_iterations_use_single_scenario(
        self,
        base_config: GridSearchConfig,
        deterministic_optimal: GridPointResult,
        mid_scenario_prices: dict,
    ) -> None:
        """With one scenario at weight=1, every iteration selects that scenario."""
        params = _zero_sigma_params(iterations=MC_ITERATIONS, seed=7)
        result = run_monte_carlo(
            base_config=base_config,
            optimal=deterministic_optimal,
            mc_params=params,
            scenario_prices=mid_scenario_prices,
        )
        for it in result.iterations:
            assert it.price_scenario == "mid"

    def test_per_scenario_stats_has_only_one_entry(
        self,
        base_config: GridSearchConfig,
        deterministic_optimal: GridPointResult,
        mid_scenario_prices: dict,
    ) -> None:
        params = _zero_sigma_params(iterations=5)
        result = run_monte_carlo(
            base_config=base_config,
            optimal=deterministic_optimal,
            mc_params=params,
            scenario_prices=mid_scenario_prices,
        )
        assert set(result.per_scenario_stats.keys()) == {"mid"}


# ---------------------------------------------------------------------------
# σ > 0 produces variance
# ---------------------------------------------------------------------------


class TestVarianceWithNonzeroSigma:
    def test_irr_variance_nonzero(
        self,
        base_config: GridSearchConfig,
        deterministic_optimal: GridPointResult,
        mid_scenario_prices: dict,
    ) -> None:
        """With σ_capex_pv > 0 there must be variance in equity IRR across iterations."""
        params = _zero_sigma_params(
            iterations=MC_ITERATIONS, seed=123,
            sigma_capex_pv=0.10,
        )
        result = run_monte_carlo(
            base_config=base_config,
            optimal=deterministic_optimal,
            mc_params=params,
            scenario_prices=mid_scenario_prices,
        )
        irrs = [it.equity_irr for it in result.iterations if it.equity_irr is not None]
        assert len(irrs) > 1
        assert max(irrs) > min(irrs), "Expected IRR to vary across iterations"

    def test_capex_factors_vary(
        self,
        base_config: GridSearchConfig,
        deterministic_optimal: GridPointResult,
        mid_scenario_prices: dict,
    ) -> None:
        """With σ > 0, sampled capex/opex factors are not all identical."""
        params = _zero_sigma_params(
            iterations=MC_ITERATIONS, seed=99,
            sigma_capex_pv=0.05,
            sigma_capex_bess=0.05,
            sigma_opex_pv=0.05,
            sigma_opex_bess=0.05,
        )
        result = run_monte_carlo(
            base_config=base_config,
            optimal=deterministic_optimal,
            mc_params=params,
            scenario_prices=mid_scenario_prices,
        )
        factors = [it.capex_factor_pv for it in result.iterations]
        assert len(set(factors)) > 1, "capex_factor_pv must vary when σ > 0"


# ---------------------------------------------------------------------------
# Statistics correctness
# ---------------------------------------------------------------------------


class TestStatisticsCorrectness:
    """MC statistics (mean, P10, P50, P90, std) must match numpy reference values."""

    def _run_mc(
        self,
        base_config: GridSearchConfig,
        optimal: GridPointResult,
        scenario_prices: dict,
        sigma_capex_pv: float = 0.08,
        iterations: int = MC_ITERATIONS,
        seed: int = 77,
    ) -> MCResult:
        params = _zero_sigma_params(
            iterations=iterations, seed=seed,
            sigma_capex_pv=sigma_capex_pv,
        )
        return run_monte_carlo(
            base_config=base_config,
            optimal=optimal,
            mc_params=params,
            scenario_prices=scenario_prices,
        )

    def test_overall_stats_keys_present(
        self,
        base_config: GridSearchConfig,
        deterministic_optimal: GridPointResult,
        mid_scenario_prices: dict,
    ) -> None:
        result = self._run_mc(base_config, deterministic_optimal, mid_scenario_prices)
        expected_keys = {"equity_irr", "project_irr", "npv", "dscr_min"}
        assert expected_keys.issubset(result.overall_stats.keys())

    def test_statistics_mean_matches_numpy(
        self,
        base_config: GridSearchConfig,
        deterministic_optimal: GridPointResult,
        mid_scenario_prices: dict,
    ) -> None:
        result = self._run_mc(base_config, deterministic_optimal, mid_scenario_prices)
        irrs = np.array([
            it.equity_irr for it in result.iterations if it.equity_irr is not None
        ])
        expected_mean = float(np.mean(irrs))
        reported_mean = result.overall_stats["equity_irr"].mean
        assert reported_mean == pytest.approx(expected_mean, rel=1e-9)

    def test_statistics_p10_p50_p90_order(
        self,
        base_config: GridSearchConfig,
        deterministic_optimal: GridPointResult,
        mid_scenario_prices: dict,
    ) -> None:
        """P10 ≤ P50 ≤ P90 for equity IRR statistics."""
        result = self._run_mc(base_config, deterministic_optimal, mid_scenario_prices)
        stats = result.overall_stats["equity_irr"]
        assert stats.p10 <= stats.p50
        assert stats.p50 <= stats.p90

    def test_statistics_p50_matches_numpy_median(
        self,
        base_config: GridSearchConfig,
        deterministic_optimal: GridPointResult,
        mid_scenario_prices: dict,
    ) -> None:
        result = self._run_mc(base_config, deterministic_optimal, mid_scenario_prices)
        irrs = np.array([
            it.equity_irr for it in result.iterations if it.equity_irr is not None
        ])
        expected_p50 = float(np.percentile(irrs, 50))
        assert result.overall_stats["equity_irr"].p50 == pytest.approx(expected_p50, rel=1e-9)

    def test_statistics_std_matches_numpy(
        self,
        base_config: GridSearchConfig,
        deterministic_optimal: GridPointResult,
        mid_scenario_prices: dict,
    ) -> None:
        result = self._run_mc(base_config, deterministic_optimal, mid_scenario_prices)
        irrs = np.array([
            it.equity_irr for it in result.iterations if it.equity_irr is not None
        ])
        expected_std = float(np.std(irrs))
        assert result.overall_stats["equity_irr"].std == pytest.approx(expected_std, rel=1e-9)

    def test_zero_sigma_std_is_zero(
        self,
        base_config: GridSearchConfig,
        deterministic_optimal: GridPointResult,
        mid_scenario_prices: dict,
    ) -> None:
        """With σ=0, all IRRs are equal → std ≈ 0."""
        params = _zero_sigma_params(iterations=10)
        result = run_monte_carlo(
            base_config=base_config,
            optimal=deterministic_optimal,
            mc_params=params,
            scenario_prices=mid_scenario_prices,
        )
        assert result.overall_stats["equity_irr"].std == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Per-scenario stats breakdown
# ---------------------------------------------------------------------------


class TestPerScenarioBreakdown:
    def test_two_scenarios_both_appear_in_breakdown(
        self,
        base_config: GridSearchConfig,
        deterministic_optimal: GridPointResult,
    ) -> None:
        """With two price scenarios, per_scenario_stats has exactly two keys."""
        pv = base_config.pv_base_timeseries
        spot_lo = _make_price_array(0.04)
        spot_hi = _make_price_array(0.08)
        two_scenarios = [
            PriceWeatherScenario(
                name="low", label="low", csv_column="LOW", weather_year=2020, weight=0.5,
                pv_timeseries_15min=pv,
                price_per_year=[spot_lo.copy() for _ in range(LIFETIME_YEARS)],
            ),
            PriceWeatherScenario(
                name="high", label="high", csv_column="HIGH", weather_year=2020, weight=0.5,
                pv_timeseries_15min=pv,
                price_per_year=[spot_hi.copy() for _ in range(LIFETIME_YEARS)],
            ),
        ]
        params = _zero_sigma_params(
            iterations=20, seed=11,
            price_scenarios=two_scenarios,
        )
        result = run_monte_carlo(
            base_config=base_config,
            optimal=deterministic_optimal,
            mc_params=params,
            scenario_prices=two_scenarios,
        )
        assert set(result.per_scenario_stats.keys()) == {"low", "high"}

    def test_per_scenario_irr_differs_by_price(
        self,
        base_config: GridSearchConfig,
        deterministic_optimal: GridPointResult,
    ) -> None:
        """Higher prices → higher median IRR in per_scenario_stats."""
        pv = base_config.pv_base_timeseries
        spot_lo = _make_price_array(0.03)
        spot_hi = _make_price_array(0.10)
        two_scenarios = [
            PriceWeatherScenario(
                name="low", label="low", csv_column="LOW", weather_year=2020, weight=0.5,
                pv_timeseries_15min=pv,
                price_per_year=[spot_lo.copy() for _ in range(LIFETIME_YEARS)],
            ),
            PriceWeatherScenario(
                name="high", label="high", csv_column="HIGH", weather_year=2020, weight=0.5,
                pv_timeseries_15min=pv,
                price_per_year=[spot_hi.copy() for _ in range(LIFETIME_YEARS)],
            ),
        ]
        params = _zero_sigma_params(
            iterations=40, seed=22,
            price_scenarios=two_scenarios,
        )
        result = run_monte_carlo(
            base_config=base_config,
            optimal=deterministic_optimal,
            mc_params=params,
            scenario_prices=two_scenarios,
        )
        lo_stats = result.per_scenario_stats.get("low", {}).get("equity_irr")
        hi_stats = result.per_scenario_stats.get("high", {}).get("equity_irr")
        if lo_stats and hi_stats and not math.isnan(lo_stats.median) and not math.isnan(hi_stats.median):
            assert hi_stats.median > lo_stats.median

    def test_per_scenario_stats_have_all_metric_keys(
        self,
        base_config: GridSearchConfig,
        deterministic_optimal: GridPointResult,
        mid_scenario_prices: dict,
    ) -> None:
        params = _zero_sigma_params(iterations=5)
        result = run_monte_carlo(
            base_config=base_config,
            optimal=deterministic_optimal,
            mc_params=params,
            scenario_prices=mid_scenario_prices,
        )
        for scenario_stats in result.per_scenario_stats.values():
            assert "equity_irr" in scenario_stats
            assert "project_irr" in scenario_stats
            assert "npv" in scenario_stats
            assert "dscr_min" in scenario_stats


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


class TestOutputStructure:
    def test_iteration_count_matches_params(
        self,
        base_config: GridSearchConfig,
        deterministic_optimal: GridPointResult,
        mid_scenario_prices: dict,
    ) -> None:
        n = 12
        params = _zero_sigma_params(iterations=n)
        result = run_monte_carlo(
            base_config=base_config,
            optimal=deterministic_optimal,
            mc_params=params,
            scenario_prices=mid_scenario_prices,
        )
        assert len(result.iterations) == n

    def test_iterations_sorted_by_index(
        self,
        base_config: GridSearchConfig,
        deterministic_optimal: GridPointResult,
        mid_scenario_prices: dict,
    ) -> None:
        params = _zero_sigma_params(
            iterations=MC_ITERATIONS, seed=55,
            sigma_capex_pv=0.05,
        )
        result = run_monte_carlo(
            base_config=base_config,
            optimal=deterministic_optimal,
            mc_params=params,
            scenario_prices=mid_scenario_prices,
        )
        indices = [it.iteration for it in result.iterations]
        assert indices == sorted(indices)
        assert indices[0] == 1
        assert indices[-1] == MC_ITERATIONS

    def test_iteration_fields_populated(
        self,
        base_config: GridSearchConfig,
        deterministic_optimal: GridPointResult,
        mid_scenario_prices: dict,
    ) -> None:
        """All MCIterationResult fields are set to valid types."""
        params = _zero_sigma_params(iterations=3)
        result = run_monte_carlo(
            base_config=base_config,
            optimal=deterministic_optimal,
            mc_params=params,
            scenario_prices=mid_scenario_prices,
        )
        for it in result.iterations:
            assert isinstance(it.iteration, int)
            assert isinstance(it.price_scenario, str)
            assert isinstance(it.capex_factor_pv, float)
            assert isinstance(it.capex_factor_bess, float)
            assert isinstance(it.opex_factor_pv, float)
            assert isinstance(it.opex_factor_bess, float)
            assert isinstance(it.pv_availability_factor, float)
            assert isinstance(it.bess_availability_factor, float)
            assert it.npv is not None and isinstance(it.npv, float)

    def test_bess_availability_factor_clipped_to_mu(
        self,
        base_config: GridSearchConfig,
        deterministic_optimal: GridPointResult,
        mid_scenario_prices: dict,
    ) -> None:
        """bess_availability_factor is always in [mu, 1] even with large σ."""
        mu = 0.5
        params = _zero_sigma_params(
            iterations=MC_ITERATIONS, seed=33,
            mu_bess_availability=mu,
            sigma_bess_availability=5.0,   # huge σ
        )
        result = run_monte_carlo(
            base_config=base_config,
            optimal=deterministic_optimal,
            mc_params=params,
            scenario_prices=mid_scenario_prices,
        )
        for it in result.iterations:
            assert mu <= it.bess_availability_factor <= 1.0


# ---------------------------------------------------------------------------
# Separate CAPEX/OPEX sigma tests
# ---------------------------------------------------------------------------


class TestSeparateSigmas:
    """Verify that per-asset sigmas independently affect only their target."""

    def test_separate_capex_sigma_pv_only(
        self,
        base_config: GridSearchConfig,
        deterministic_optimal: GridPointResult,
        mid_scenario_prices: dict,
    ) -> None:
        """sigma_capex_pv > 0, sigma_capex_bess = 0 → only PV CAPEX varies."""
        params = _zero_sigma_params(
            iterations=MC_ITERATIONS, seed=100,
            sigma_capex_pv=0.10,
            sigma_capex_bess=0.0,
        )
        result = run_monte_carlo(
            base_config=base_config,
            optimal=deterministic_optimal,
            mc_params=params,
            scenario_prices=mid_scenario_prices,
        )
        pv_factors = [it.capex_factor_pv for it in result.iterations]
        bess_factors = [it.capex_factor_bess for it in result.iterations]
        assert len(set(pv_factors)) > 1, "PV CAPEX factor should vary"
        assert all(f == pytest.approx(1.0, abs=1e-12) for f in bess_factors), (
            "BESS CAPEX factor should be 1.0 when sigma=0"
        )

    def test_separate_opex_sigma_bess_only(
        self,
        base_config: GridSearchConfig,
        deterministic_optimal: GridPointResult,
        mid_scenario_prices: dict,
    ) -> None:
        """sigma_opex_pv = 0, sigma_opex_bess > 0 → only BESS OPEX varies."""
        params = _zero_sigma_params(
            iterations=MC_ITERATIONS, seed=101,
            sigma_opex_pv=0.0,
            sigma_opex_bess=0.10,
        )
        result = run_monte_carlo(
            base_config=base_config,
            optimal=deterministic_optimal,
            mc_params=params,
            scenario_prices=mid_scenario_prices,
        )
        pv_factors = [it.opex_factor_pv for it in result.iterations]
        bess_factors = [it.opex_factor_bess for it in result.iterations]
        assert all(f == pytest.approx(1.0, abs=1e-12) for f in pv_factors), (
            "PV OPEX factor should be 1.0 when sigma=0"
        )
        assert len(set(bess_factors)) > 1, "BESS OPEX factor should vary"


# ---------------------------------------------------------------------------
# PV availability / offline days
# ---------------------------------------------------------------------------


class TestPvAvailability:
    def test_pv_availability_factor_clipped_to_zero_one(
        self,
        base_config: GridSearchConfig,
        deterministic_optimal: GridPointResult,
        mid_scenario_prices: dict,
    ) -> None:
        """pv_availability_factor is always in [0, 1] even with large σ."""
        params = _zero_sigma_params(
            iterations=MC_ITERATIONS, seed=200,
            sigma_pv_availability=5.0,
        )
        result = run_monte_carlo(
            base_config=base_config,
            optimal=deterministic_optimal,
            mc_params=params,
            scenario_prices=mid_scenario_prices,
        )
        for it in result.iterations:
            assert 0.0 <= it.pv_availability_factor <= 1.0

    def test_pv_availability_reduces_revenue(
        self,
        base_config: GridSearchConfig,
        deterministic_optimal: GridPointResult,
        mid_scenario_prices: dict,
    ) -> None:
        """With σ_pv_availability large enough, some iterations should have lower revenue."""
        params_no_avail = _zero_sigma_params(iterations=1, seed=42)
        params_with_avail = _zero_sigma_params(
            iterations=1, seed=42,
            sigma_pv_availability=0.30,  # very large → PV offline days
        )
        result_no = run_monte_carlo(
            base_config=base_config,
            optimal=deterministic_optimal,
            mc_params=params_no_avail,
            scenario_prices=mid_scenario_prices,
        )
        result_with = run_monte_carlo(
            base_config=base_config,
            optimal=deterministic_optimal,
            mc_params=params_with_avail,
            scenario_prices=mid_scenario_prices,
        )
        # If PV availability factor < 1.0, NPV should be lower or equal
        it_no = result_no.iterations[0]
        it_with = result_with.iterations[0]
        if it_with.pv_availability_factor < 1.0:
            assert it_with.npv <= it_no.npv
