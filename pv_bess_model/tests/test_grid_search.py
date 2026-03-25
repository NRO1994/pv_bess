"""Tests for optimization/grid_search.py.

Uses small synthetic data (3 project years, 100 kWp PV, 3 scales, 2 E/P
ratios, single-process execution) so each test completes in a few seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from pv_bess_model.config.defaults import (
    INTERVALS_PER_HOUR,
    INTERVALS_PER_YEAR,
    TIMESTEP_HOURS,
)
from pv_bess_model.optimization.grid_search import (
    GridSearchConfig,
    GridSearchResult,
    run_grid_search,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LIFETIME_YEARS = 3
PV_PEAK_KWP = 100.0
SCALES = [0.0, 20.0, 50.0]   # 3 scale levels
E_TO_P_RATIOS = [1.0, 2.0]   # 2 E/P ratios → 6 combinations total
GRID_MAX_KW = 80.0


def _make_price_array(price_eur_per_kwh: float = 0.05) -> np.ndarray:
    """Return a flat price array in €/kWh at quarter-hourly resolution."""
    return np.full(INTERVALS_PER_YEAR, price_eur_per_kwh, dtype=float)


def _make_pv_array(peak_kwh: float = 20.0) -> np.ndarray:
    """Return a PV profile at quarter-hourly resolution: half-sine during hours 6-18."""
    hour_of_day = np.arange(INTERVALS_PER_YEAR) % (24 * INTERVALS_PER_HOUR) // INTERVALS_PER_HOUR
    daylight = np.where(
        (hour_of_day >= 6) & (hour_of_day <= 18),
        np.sin(np.pi * (hour_of_day - 6) / 12),
        0.0,
    )
    return (peak_kwh / INTERVALS_PER_HOUR * daylight).astype(float)


def _make_config(
    scales: list[float] | None = None,
    e_to_p: list[float] | None = None,
    lifetime: int = LIFETIME_YEARS,
) -> GridSearchConfig:
    """Build a minimal GridSearchConfig for testing.

    CAPEX is set so that Year 1 CF is negative (investment year) while
    subsequent years are positive, giving a valid IRR sign change.
    Zero leverage keeps things simple.
    """
    if scales is None:
        scales = SCALES
    if e_to_p is None:
        e_to_p = E_TO_P_RATIOS
    spot = _make_price_array(0.05)
    pv = _make_pv_array(20.0)
    return GridSearchConfig(
        scale_pct_of_pv=scales,
        e_to_p_ratio_hours=e_to_p,
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
        grid_max_kw=GRID_MAX_KW,
        grid_max_import_kw=None,
        grid_loss_factor=1.0,
        grid_costs_capex={},   # no grid costs to keep CAPEX minimal
        grid_costs_opex={},
        operating_mode="green",
        spot_prices_yearly=[spot.copy() for _ in range(lifetime)],
        fixed_prices_yearly=[0.0] * lifetime,
        lifetime_years=lifetime,
        leverage_pct=0.0,      # no debt → no debt-service drag
        interest_rate_pct=4.5,
        loan_tenor_years=3,
        opex_inflation_factors=[(1.0 + 0.02) ** i for i in range(lifetime)],
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
def grid_result() -> GridSearchResult:
    """Run the grid search once and share across all tests in this module."""
    return run_grid_search(_make_config())


# ---------------------------------------------------------------------------
# BESS sizing derivation
# ---------------------------------------------------------------------------


class TestBessSizingDerivation:
    """BESS power = pv_peak × scale / 100; capacity = power × e_to_p."""

    def test_bess_power_from_scale(self, grid_result: GridSearchResult) -> None:
        for pt in grid_result.points:
            expected = PV_PEAK_KWP * pt.scale_pct / 100.0
            assert pt.bess_power_kw == pytest.approx(expected, rel=1e-9)

    def test_bess_capacity_from_e_to_p(self, grid_result: GridSearchResult) -> None:
        for pt in grid_result.points:
            expected = pt.bess_power_kw * pt.e_to_p_ratio
            assert pt.bess_capacity_kwh == pytest.approx(expected, rel=1e-9)

    def test_specific_sizing_20pct_2h(self, grid_result: GridSearchResult) -> None:
        """scale=20%, E/P=2h → power=20 kW, capacity=40 kWh."""
        matches = [
            pt for pt in grid_result.points
            if pt.scale_pct == pytest.approx(20.0) and pt.e_to_p_ratio == pytest.approx(2.0)
        ]
        assert len(matches) == 1
        assert matches[0].bess_power_kw == pytest.approx(20.0)
        assert matches[0].bess_capacity_kwh == pytest.approx(40.0)

    def test_specific_sizing_50pct_1h(self, grid_result: GridSearchResult) -> None:
        """scale=50%, E/P=1h → power=50 kW, capacity=50 kWh."""
        matches = [
            pt for pt in grid_result.points
            if pt.scale_pct == pytest.approx(50.0) and pt.e_to_p_ratio == pytest.approx(1.0)
        ]
        assert len(matches) == 1
        assert matches[0].bess_power_kw == pytest.approx(50.0)
        assert matches[0].bess_capacity_kwh == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# PV-only baseline (scale = 0 %)
# ---------------------------------------------------------------------------


class TestPvOnlyBaseline:
    def test_scale_zero_always_present(self, grid_result: GridSearchResult) -> None:
        """Results always contain at least one entry with scale_pct == 0."""
        assert any(pt.scale_pct == pytest.approx(0.0) for pt in grid_result.points)

    def test_scale_zero_has_zero_bess_power(self, grid_result: GridSearchResult) -> None:
        for pt in grid_result.points:
            if pt.scale_pct == pytest.approx(0.0):
                assert pt.bess_power_kw == pytest.approx(0.0)

    def test_scale_zero_has_zero_bess_capacity(self, grid_result: GridSearchResult) -> None:
        for pt in grid_result.points:
            if pt.scale_pct == pytest.approx(0.0):
                assert pt.bess_capacity_kwh == pytest.approx(0.0)

    def test_scale_zero_auto_inserted_when_missing(self) -> None:
        """If the caller omits scale=0, run_grid_search inserts it automatically."""
        result = run_grid_search(_make_config(scales=[20.0, 50.0], e_to_p=[1.0], lifetime=1))
        assert any(pt.scale_pct == pytest.approx(0.0) for pt in result.points)

    def test_scale_zero_has_zero_bess_capex(self, grid_result: GridSearchResult) -> None:
        """PV-only baseline: no per-kW or per-kWh BESS CAPEX terms → zero BESS CAPEX."""
        for pt in grid_result.points:
            if pt.scale_pct == pytest.approx(0.0):
                assert pt.capex_bess == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Matrix dimensions
# ---------------------------------------------------------------------------


class TestMatrixDimensions:
    def test_total_number_of_results(self, grid_result: GridSearchResult) -> None:
        """len(results) == len(scales) × len(e_to_p_ratios)."""
        assert len(grid_result.points) == len(SCALES) * len(E_TO_P_RATIOS)

    def test_all_scale_e_to_p_combinations_present(
        self, grid_result: GridSearchResult
    ) -> None:
        found = {(pt.scale_pct, pt.e_to_p_ratio) for pt in grid_result.points}
        for s in SCALES:
            for e in E_TO_P_RATIOS:
                assert (s, e) in found, f"Missing (scale={s}%, E/P={e}h)"

    def test_results_sorted_by_scale_then_e_to_p(
        self, grid_result: GridSearchResult
    ) -> None:
        pairs = [(pt.scale_pct, pt.e_to_p_ratio) for pt in grid_result.points]
        assert pairs == sorted(pairs)

    def test_single_scale_single_e_to_p(self) -> None:
        """Config with scales=[0] and e_to_p=[1] → exactly 1 result."""
        result = run_grid_search(_make_config(scales=[0.0], e_to_p=[1.0]))
        assert len(result.points) == 1


# ---------------------------------------------------------------------------
# Optimum identification
# ---------------------------------------------------------------------------


class TestOptimumIdentification:
    def test_optimal_has_max_irr(self, grid_result: GridSearchResult) -> None:
        """optimal.equity_irr == max of all non-None equity IRRs."""
        assert grid_result.optimal is not None
        valid_irrs = [
            pt.metrics.equity_irr for pt in grid_result.points
            if pt.metrics is not None and pt.metrics.equity_irr is not None
        ]
        assert grid_result.optimal.metrics.equity_irr == pytest.approx(max(valid_irrs), rel=1e-9)

    def test_exactly_one_optimal_flag(self, grid_result: GridSearchResult) -> None:
        flagged = [pt for pt in grid_result.points if pt.is_optimal]
        assert len(flagged) == 1

    def test_optimal_flag_matches_returned_optimal(
        self, grid_result: GridSearchResult
    ) -> None:
        flagged = [pt for pt in grid_result.points if pt.is_optimal]
        assert flagged[0] is grid_result.optimal

    def test_non_optimal_points_not_flagged(self, grid_result: GridSearchResult) -> None:
        for pt in grid_result.points:
            if pt is not grid_result.optimal:
                assert not pt.is_optimal

    def test_optimal_not_none_with_positive_prices(
        self, grid_result: GridSearchResult
    ) -> None:
        """With positive spot prices there must be a valid optimum."""
        assert grid_result.optimal is not None


# ---------------------------------------------------------------------------
# CAPEX / revenue sanity checks
# ---------------------------------------------------------------------------


class TestCapexSanity:
    def test_all_capex_total_positive(self, grid_result: GridSearchResult) -> None:
        for pt in grid_result.points:
            assert pt.capex_total > 0.0

    def test_bess_capex_increases_with_scale(self, grid_result: GridSearchResult) -> None:
        """For a fixed E/P ratio, BESS CAPEX is non-decreasing as scale increases."""
        e = E_TO_P_RATIOS[0]
        pts = sorted(
            [pt for pt in grid_result.points if pt.e_to_p_ratio == pytest.approx(e)],
            key=lambda p: p.scale_pct,
        )
        for i in range(1, len(pts)):
            assert pts[i].capex_bess >= pts[i - 1].capex_bess

    def test_revenue_year1_positive(self, grid_result: GridSearchResult) -> None:
        """All points with positive PV production earn positive first-year revenue."""
        for pt in grid_result.points:
            assert pt.revenue_year1 > 0.0

    def test_opex_base_positive(self, grid_result: GridSearchResult) -> None:
        for pt in grid_result.points:
            assert pt.opex_base > 0.0


# ---------------------------------------------------------------------------
# Optimization fee in grid search (Feature 03)
# ---------------------------------------------------------------------------


class TestOptimizationFeeGridSearch:
    """Verify that optimization_fee_pct affects Equity IRR in the grid search."""

    @staticmethod
    def _make_grey_config_with_variable_prices(
        scales: list[float], lifetime: int = 3, fee_pct: float = 0.0,
    ) -> GridSearchConfig:
        """Build grey-mode config with variable spot prices for arbitrage."""
        # Variable prices: cheap at night, expensive during day → arbitrage profit
        base_price = np.zeros(INTERVALS_PER_YEAR, dtype=float)
        hour_of_day = np.arange(INTERVALS_PER_YEAR) % (24 * INTERVALS_PER_HOUR) // INTERVALS_PER_HOUR
        base_price = np.where(hour_of_day < 8, 0.02, 0.10)  # cheap night, expensive day
        cfg = _make_config(scales=scales, e_to_p=[2.0], lifetime=lifetime)
        cfg.operating_mode = "grey"
        cfg.optimization_fee_pct = fee_pct
        cfg.spot_prices_yearly = [base_price.copy() for _ in range(lifetime)]
        return cfg

    def test_fee_reduces_equity_irr(self) -> None:
        """A positive optimization fee should reduce equity IRR vs. fee=0."""
        result_no_fee = run_grid_search(
            self._make_grey_config_with_variable_prices([0.0, 50.0], fee_pct=0.0)
        )
        result_with_fee = run_grid_search(
            self._make_grey_config_with_variable_prices([0.0, 50.0], fee_pct=20.0)
        )

        # Compare the BESS point (scale=50%), not PV-only (scale=0% has no BESS revenue)
        bess_no_fee = [p for p in result_no_fee.points if p.scale_pct == pytest.approx(50.0)][0]
        bess_with_fee = [p for p in result_with_fee.points if p.scale_pct == pytest.approx(50.0)][0]

        assert bess_no_fee.metrics is not None and bess_no_fee.metrics.equity_irr is not None
        assert bess_with_fee.metrics is not None and bess_with_fee.metrics.equity_irr is not None
        assert bess_with_fee.metrics.equity_irr < bess_no_fee.metrics.equity_irr

    def test_fee_zero_no_effect_on_pv_only(self) -> None:
        """PV-only (scale=0%) should have same IRR regardless of fee setting."""
        config_no_fee = _make_config(scales=[0.0], e_to_p=[1.0], lifetime=3)
        config_no_fee.optimization_fee_pct = 0.0
        result_no_fee = run_grid_search(config_no_fee)

        config_with_fee = _make_config(scales=[0.0], e_to_p=[1.0], lifetime=3)
        config_with_fee.optimization_fee_pct = 20.0
        result_with_fee = run_grid_search(config_with_fee)

        irr_no_fee = result_no_fee.points[0].metrics.equity_irr if result_no_fee.points[0].metrics else None
        irr_with_fee = result_with_fee.points[0].metrics.equity_irr if result_with_fee.points[0].metrics else None
        assert irr_no_fee is not None
        assert irr_with_fee is not None
        assert irr_no_fee == pytest.approx(irr_with_fee, rel=1e-9)
