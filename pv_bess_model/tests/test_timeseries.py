"""Unit tests for pv_bess_model.pv.timeseries and pv_bess_model.pv.degradation.

timeseries.py
-------------
- compute_p50_p90: correct P50/P90 statistics from synthetic multi-year data
- compute_p50_p90: empty input raises ValueError
- compute_p50_p90: wrong-length array raises ValueError
- compute_p50_p90: single year → P50 == P90 == values
- compute_p50_p90: negative values are clipped to 0
- percentile_timeseries: arbitrary percentile, out-of-range raises

degradation.py
--------------
- apply_degradation: year-1 factor = (1-rate)^1, year-N = (1-rate)^N
- apply_degradation: zero degradation → all years identical to base
- apply_degradation: output length equals lifetime_years
- apply_degradation: production is always <= base (non-negative rate)
- apply_degradation: invalid rate (negative, ≥ 1) raises ValueError
- apply_degradation: lifetime_years < 1 raises ValueError
- degradation_factor: correct scalar for spot checks
- degradation_factor: invalid inputs raise ValueError
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pv_bess_model.config.defaults import HOURS_PER_YEAR
from pv_bess_model.pv.degradation import apply_degradation, degradation_factor
from pv_bess_model.pv.timeseries import compute_p50_p90, percentile_timeseries

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_yearly(years_values: dict[int, float]) -> dict[int, np.ndarray]:
    """Create synthetic yearly data with constant hourly production."""
    return {y: np.full(HOURS_PER_YEAR, v, dtype=float) for y, v in years_values.items()}


def _make_yearly_rng(n_years: int = 10, seed: int = 42) -> dict[int, np.ndarray]:
    """Create n_years of random 8760-h production with known seed."""
    rng = np.random.default_rng(seed)
    return {2005 + i: rng.uniform(0, 1000, HOURS_PER_YEAR) for i in range(n_years)}


# ---------------------------------------------------------------------------
# compute_p50_p90 – happy path
# ---------------------------------------------------------------------------


class TestComputeP50P90:
    def test_single_year_p50_equals_values(self):
        """With one year, P50 must equal the input values exactly."""
        data = _make_yearly({2010: 500.0})
        p50, p90 = compute_p50_p90(data)
        assert len(p50) == HOURS_PER_YEAR
        assert np.all(p50 == pytest.approx(500.0))

    def test_single_year_p90_equals_values(self):
        """With one year, P90 (10th percentile of one value) equals that value."""
        data = _make_yearly({2010: 300.0})
        p50, p90 = compute_p50_p90(data)
        assert np.all(p90 == pytest.approx(300.0))

    def test_two_years_p50_is_median(self):
        """P50 of [100, 200] per hour = 150."""
        data = _make_yearly({2005: 100.0, 2006: 200.0})
        p50, _ = compute_p50_p90(data)
        assert np.all(p50 == pytest.approx(150.0))

    def test_two_years_p90_is_10th_percentile(self):
        """P90 of [100, 200] per hour = 110 (10th percentile of [100, 200])."""
        data = _make_yearly({2005: 100.0, 2006: 200.0})
        _, p90 = compute_p50_p90(data)
        expected = np.percentile([100.0, 200.0], 10)
        assert np.all(p90 == pytest.approx(expected))

    def test_five_years_known_percentiles(self):
        """Use five constant-value years and verify P50 and P90."""
        # Values per hour: [10, 20, 30, 40, 50]
        data = {
            2001 + i: np.full(HOURS_PER_YEAR, float((i + 1) * 10)) for i in range(5)
        }
        p50, p90 = compute_p50_p90(data)
        expected_p50 = np.percentile([10, 20, 30, 40, 50], 50)
        expected_p90 = np.percentile([10, 20, 30, 40, 50], 10)
        assert np.all(p50 == pytest.approx(expected_p50))
        assert np.all(p90 == pytest.approx(expected_p90))

    def test_output_shape(self):
        data = _make_yearly_rng(n_years=5)
        p50, p90 = compute_p50_p90(data)
        assert p50.shape == (HOURS_PER_YEAR,)
        assert p90.shape == (HOURS_PER_YEAR,)

    def test_p90_le_p50(self):
        """P90 (10th percentile) must always be ≤ P50 (50th percentile)."""
        data = _make_yearly_rng(n_years=10)
        p50, p90 = compute_p50_p90(data)
        assert np.all(p90 <= p50 + 1e-9)

    def test_negative_values_clipped_to_zero(self):
        """Both P50 and P90 must be clipped to ≥ 0."""
        data = {2005: np.full(HOURS_PER_YEAR, -100.0)}
        p50, p90 = compute_p50_p90(data)
        assert np.all(p50 == 0.0)
        assert np.all(p90 == 0.0)

    def test_mixed_positive_negative_p50_clipped(self):
        """If the median is negative, P50 must be 0."""
        data = {
            2005: np.full(HOURS_PER_YEAR, -50.0),
            2006: np.full(HOURS_PER_YEAR, -10.0),
            2007: np.full(HOURS_PER_YEAR, 100.0),
        }
        p50, p90 = compute_p50_p90(data)
        expected_median = np.percentile([-50, -10, 100], 50)
        if expected_median < 0:
            assert np.all(p50 == 0.0)
        else:
            assert np.all(p50 == pytest.approx(expected_median))

    def test_per_hour_statistics_independent(self):
        """Different hours can have different cross-year distributions."""
        rng = np.random.default_rng(7)
        n_years = 8
        data = {2005 + i: rng.uniform(0, 500, HOURS_PER_YEAR) for i in range(n_years)}
        p50, p90 = compute_p50_p90(data)
        # At hour 0, manually verify
        values_h0 = np.array([data[y][0] for y in sorted(data)])
        assert p50[0] == pytest.approx(np.percentile(values_h0, 50))
        assert p90[0] == pytest.approx(np.percentile(values_h0, 10))

    def test_ten_years_matches_numpy_percentile(self):
        data = _make_yearly_rng(n_years=10, seed=123)
        p50, p90 = compute_p50_p90(data)
        matrix = np.stack([data[y] for y in sorted(data)], axis=0)
        expected_p50 = np.maximum(np.percentile(matrix, 50, axis=0), 0)
        expected_p90 = np.maximum(np.percentile(matrix, 10, axis=0), 0)
        np.testing.assert_allclose(p50, expected_p50)
        np.testing.assert_allclose(p90, expected_p90)


# ---------------------------------------------------------------------------
# compute_p50_p90 – error cases
# ---------------------------------------------------------------------------


class TestComputeP50P90Errors:
    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_p50_p90({})

    def test_wrong_length_array_raises(self):
        bad = {2010: np.ones(100)}
        with pytest.raises(ValueError, match="8760"):
            compute_p50_p90(bad)

    def test_error_message_names_bad_year(self):
        data = {2005: np.ones(HOURS_PER_YEAR), 2006: np.ones(500)}
        with pytest.raises(ValueError, match="2006"):
            compute_p50_p90(data)

    def test_multiple_bad_years_all_named(self):
        data = {2005: np.ones(100), 2006: np.ones(200), 2007: np.ones(HOURS_PER_YEAR)}
        with pytest.raises(ValueError) as exc_info:
            compute_p50_p90(data)
        msg = str(exc_info.value)
        assert "2005" in msg
        assert "2006" in msg


# ---------------------------------------------------------------------------
# percentile_timeseries
# ---------------------------------------------------------------------------


class TestPercentileTimeseries:
    def test_p50_matches_compute_p50_p90(self):
        data = _make_yearly_rng(n_years=6)
        p50_direct, _ = compute_p50_p90(data)
        p50_generic = percentile_timeseries(data, 50.0)
        np.testing.assert_allclose(p50_direct, p50_generic)

    def test_p90_matches_compute_p50_p90(self):
        data = _make_yearly_rng(n_years=6)
        _, p90_direct = compute_p50_p90(data)
        p90_generic = percentile_timeseries(data, 10.0)
        np.testing.assert_allclose(p90_direct, p90_generic)

    def test_p100_is_max(self):
        data = _make_yearly({2005: 100.0, 2006: 200.0, 2007: 300.0})
        p100 = percentile_timeseries(data, 100.0)
        assert np.all(p100 == pytest.approx(300.0))

    def test_p0_is_min(self):
        data = _make_yearly({2005: 10.0, 2006: 50.0, 2007: 90.0})
        p0 = percentile_timeseries(data, 0.0)
        assert np.all(p0 == pytest.approx(10.0))

    def test_out_of_range_percentile_raises(self):
        data = _make_yearly({2010: 100.0})
        with pytest.raises(ValueError, match="percentile"):
            percentile_timeseries(data, -1.0)
        with pytest.raises(ValueError, match="percentile"):
            percentile_timeseries(data, 101.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            percentile_timeseries({}, 50.0)

    def test_result_non_negative(self):
        data = _make_yearly({2005: -200.0, 2006: -100.0})
        out = percentile_timeseries(data, 50.0)
        assert np.all(out >= 0.0)


# ---------------------------------------------------------------------------
# apply_degradation
# ---------------------------------------------------------------------------


class TestApplyDegradation:
    def test_output_length_equals_lifetime(self):
        base = np.ones(HOURS_PER_YEAR)
        result = apply_degradation(base, degradation_rate=0.004, lifetime_years=25)
        assert len(result) == 25

    def test_year_1_factor(self):
        base = np.ones(HOURS_PER_YEAR) * 1000.0
        result = apply_degradation(base, degradation_rate=0.004, lifetime_years=1)
        expected = 1000.0 * (1 - 0.004) ** 1
        assert np.all(result[0] == pytest.approx(expected))

    def test_year_25_factor(self):
        base = np.ones(HOURS_PER_YEAR) * 1000.0
        result = apply_degradation(base, degradation_rate=0.004, lifetime_years=25)
        expected = 1000.0 * (1 - 0.004) ** 25
        assert np.all(result[24] == pytest.approx(expected, rel=1e-9))

    def test_zero_degradation_all_years_identical(self):
        base = np.arange(HOURS_PER_YEAR, dtype=float)
        result = apply_degradation(base, degradation_rate=0.0, lifetime_years=10)
        for year_arr in result:
            np.testing.assert_array_equal(year_arr, base)

    def test_production_decreases_monotonically(self):
        """Each successive year must have lower or equal production."""
        base = np.full(HOURS_PER_YEAR, 500.0)
        result = apply_degradation(base, degradation_rate=0.005, lifetime_years=20)
        totals = [arr.sum() for arr in result]
        assert all(totals[i] >= totals[i + 1] for i in range(len(totals) - 1))

    def test_production_never_exceeds_base(self):
        base = np.full(HOURS_PER_YEAR, 800.0)
        result = apply_degradation(base, degradation_rate=0.004, lifetime_years=25)
        for arr in result:
            assert np.all(arr <= base + 1e-9)

    def test_array_shape_preserved(self):
        base = np.ones(HOURS_PER_YEAR)
        result = apply_degradation(base, degradation_rate=0.01, lifetime_years=5)
        for arr in result:
            assert arr.shape == (HOURS_PER_YEAR,)

    def test_each_year_is_independent_array(self):
        """Modifying one year's array must not affect others."""
        base = np.ones(HOURS_PER_YEAR)
        result = apply_degradation(base, degradation_rate=0.01, lifetime_years=3)
        original_year2_value = result[1][0].copy()
        result[0][:] = 9999.0  # mutate year 1
        assert result[1][0] == pytest.approx(original_year2_value)

    def test_small_rate_accumulated_over_many_years(self):
        rate = 0.004
        years = 25
        base = np.ones(HOURS_PER_YEAR) * 100.0
        result = apply_degradation(base, rate, years)
        for i, arr in enumerate(result):
            expected = 100.0 * (1 - rate) ** (i + 1)
            assert arr[0] == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# apply_degradation – error cases
# ---------------------------------------------------------------------------


class TestApplyDegradationErrors:
    def test_negative_rate_raises(self):
        with pytest.raises(ValueError, match="degradation_rate"):
            apply_degradation(
                np.ones(HOURS_PER_YEAR), degradation_rate=-0.01, lifetime_years=5
            )

    def test_rate_one_raises(self):
        with pytest.raises(ValueError, match="degradation_rate"):
            apply_degradation(
                np.ones(HOURS_PER_YEAR), degradation_rate=1.0, lifetime_years=5
            )

    def test_rate_above_one_raises(self):
        with pytest.raises(ValueError, match="degradation_rate"):
            apply_degradation(
                np.ones(HOURS_PER_YEAR), degradation_rate=1.5, lifetime_years=5
            )

    def test_rate_in_percent_raises(self):
        """A rate of 0.4 (should be 0.004 for 0.4%) is still valid (<1),
        but 4.0 must raise."""
        with pytest.raises(ValueError):
            apply_degradation(
                np.ones(HOURS_PER_YEAR), degradation_rate=4.0, lifetime_years=5
            )

    def test_zero_lifetime_years_raises(self):
        with pytest.raises(ValueError, match="lifetime_years"):
            apply_degradation(
                np.ones(HOURS_PER_YEAR), degradation_rate=0.004, lifetime_years=0
            )

    def test_negative_lifetime_years_raises(self):
        with pytest.raises(ValueError, match="lifetime_years"):
            apply_degradation(
                np.ones(HOURS_PER_YEAR), degradation_rate=0.004, lifetime_years=-1
            )


# ---------------------------------------------------------------------------
# degradation_factor
# ---------------------------------------------------------------------------


class TestDegradationFactor:
    def test_year_1_value(self):
        assert degradation_factor(0.004, 1) == pytest.approx((1 - 0.004) ** 1)

    def test_year_25_value(self):
        assert degradation_factor(0.004, 25) == pytest.approx(
            (1 - 0.004) ** 25, rel=1e-9
        )

    def test_zero_degradation_factor_is_1(self):
        for year in (1, 10, 25):
            assert degradation_factor(0.0, year) == pytest.approx(1.0)

    def test_factor_always_in_zero_one(self):
        for rate in (0.001, 0.004, 0.01, 0.05):
            for year in (1, 5, 10, 25):
                f = degradation_factor(rate, year)
                assert 0.0 < f <= 1.0

    def test_invalid_rate_raises(self):
        with pytest.raises(ValueError):
            degradation_factor(-0.01, 1)
        with pytest.raises(ValueError):
            degradation_factor(1.0, 1)

    def test_invalid_year_raises(self):
        with pytest.raises(ValueError):
            degradation_factor(0.004, 0)
        with pytest.raises(ValueError):
            degradation_factor(0.004, -5)

    def test_consistent_with_apply_degradation(self):
        """factor(rate, y) must match the scalar that apply_degradation uses."""
        base = np.ones(HOURS_PER_YEAR)
        rate = 0.006
        years = 10
        result = apply_degradation(base, rate, years)
        for i, arr in enumerate(result):
            expected = degradation_factor(rate, i + 1)
            assert arr[0] == pytest.approx(expected, rel=1e-9)
