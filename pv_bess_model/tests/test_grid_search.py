"""Tests for optimization/grid_search.py.

Uses small synthetic data (3 project years, 100 kWp PV, 3 scales, 2 E/P
ratios, single-process execution) so each test completes in a few seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

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
    """Return a flat 8760-length price array in €/kWh."""
    return np.full(8760, price_eur_per_kwh, dtype=float)


def _make_pv_array(peak_kwh: float = 20.0) -> np.ndarray:
    """Return an 8760 PV profile: half-sine during hours 6-18, zero otherwise."""
    hour_of_day = np.arange(8760) % 24
    daylight = np.where(
        (hour_of_day >= 6) & (hour_of_day <= 18),
        np.sin(np.pi * (hour_of_day - 6) / 12),
        0.0,
    )
    return (peak_kwh * daylight).astype(float)


def _make_config(
    scales: list[float] | None = None,
    e_to_p: list[float] | None = None,
    lifetime: int = LIFETIME_YEARS,
) -> GridSearchConfig:
    """Build a minimal GridSearchConfig for testing.

    Uses artificially low CAPEX (€1/kW PV, €1/kW + €1/kWh BESS) and zero
    leverage so that all combinations yield positive equity cashflows over
    the short 3-year lifetime, making IRR computable.
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
        pv_base_timeseries_p50=pv,
        pv_degradation_rate=0.004,
        # Very low CAPEX so the project is profitable over 3 years
        pv_costs_capex={"eur_per_kw": 1.0},
        pv_costs_opex={"pct_of_capex": 0.01},
        bess_rte=0.90,
        bess_min_soc_pct=10.0,
        bess_max_soc_pct=90.0,
        bess_degradation_rate=0.02,
        bess_availability_pct=100.0,
        bess_costs_capex={"eur_per_kw": 1.0, "eur_per_kwh": 1.0},
        bess_costs_opex={"pct_of_capex": 0.02},
        replacement_enabled=False,
        replacement_year=0,
        replacement_fixed_eur=0.0,
        replacement_eur_per_kw=0.0,
        replacement_eur_per_kwh=0.0,
        replacement_pct_of_capex=0.0,
        grid_max_kw=GRID_MAX_KW,
        grid_costs_capex={},   # no grid costs to keep CAPEX minimal
        grid_costs_opex={},
        operating_mode="green",
        spot_prices_yearly=[spot.copy() for _ in range(lifetime)],
        fixed_prices_yearly=[0.0] * lifetime,
        lifetime_years=lifetime,
        leverage_pct=0.0,      # no debt → no debt-service drag
        interest_rate_pct=4.5,
        loan_tenor_years=3,
        inflation_rate=0.02,
        discount_rate=0.06,
        afa_years_pv=5,
        afa_years_bess=5,
        gewerbesteuer_messzahl=0.035,
        gewerbesteuer_hebesatz=400,
        debt_uses_p90=False,
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
            pt.equity_irr for pt in grid_result.points if pt.equity_irr is not None
        ]
        assert grid_result.optimal.equity_irr == pytest.approx(max(valid_irrs), rel=1e-9)

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
