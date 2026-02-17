"""Unit tests for pv_bess_model.pv.timeseries.

Covers:
- compute_p50_p90: correct P50/P90 statistics from synthetic multi-year data
- compute_p50_p90: 3-year dataset with manually verifiable P50/P90 values
- compute_p50_p90: all years identical → P50 = P90 = input value
- compute_p50_p90: single year → P50 = P90 = values
- compute_p50_p90: leap-year-sized input (8784h) raises ValueError
- compute_p50_p90: negative values are clipped to 0
- compute_p50_p90: empty input raises ValueError
- compute_p50_p90: wrong-length array raises ValueError with year name
- percentile_timeseries: arbitrary percentile, out-of-range raises

Degradation tests live in test_degradation.py (mirrors pv/degradation.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from pv_bess_model.config.defaults import HOURS_PER_YEAR
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

    def test_single_year_p50_equals_p90(self):
        """With only one year both statistics must be identical."""
        data = _make_yearly({2015: 123.4})
        p50, p90 = compute_p50_p90(data)
        np.testing.assert_allclose(p50, p90)

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

    def test_three_years_known_values(self):
        """3-year dataset, manually verifiable P50 and P90.

        Hours in each year are constant so the cross-year distribution at
        every hour is the same vector: [3.0, 9.0, 15.0] (sorted).

        Manual calculation (numpy linear interpolation):
          P50 = percentile([3, 9, 15], 50):
                index = 0.50 × (3-1) = 1.0  →  value = 9.0
          P90 = percentile([3, 9, 15], 10):
                index = 0.10 × (3-1) = 0.2  →  3.0 + 0.2×(9.0-3.0) = 4.2
        """
        data = _make_yearly({2005: 3.0, 2006: 9.0, 2007: 15.0})
        p50, p90 = compute_p50_p90(data)
        assert np.all(p50 == pytest.approx(9.0))
        assert np.all(p90 == pytest.approx(4.2))

    def test_four_years_known_values(self):
        """4-year dataset, manually verifiable P50 and P90.

        Cross-year distribution at each hour: [10, 20, 30, 40] (sorted).

        Manual calculation (numpy linear interpolation):
          P50 = percentile([10, 20, 30, 40], 50):
                index = 0.50 × (4-1) = 1.5  →  20 + 0.5×(30-20) = 25.0
          P90 = percentile([10, 20, 30, 40], 10):
                index = 0.10 × (4-1) = 0.3  →  10 + 0.3×(20-10) = 13.0
        """
        data = _make_yearly({2010: 10.0, 2011: 20.0, 2012: 30.0, 2013: 40.0})
        p50, p90 = compute_p50_p90(data)
        assert np.all(p50 == pytest.approx(25.0))
        assert np.all(p90 == pytest.approx(13.0))

    def test_all_identical_years_p50_equals_p90(self):
        """When every historical year has the same production, P50 = P90 = that value."""
        value = 42.5
        data = {year: np.full(HOURS_PER_YEAR, value) for year in range(2005, 2015)}
        p50, p90 = compute_p50_p90(data)
        np.testing.assert_allclose(p50, p90)
        np.testing.assert_allclose(p50, value)

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

    def test_8784_hour_input_raises(self):
        """Leap-year-sized arrays (8784 h) must raise ValueError.

        compute_p50_p90 requires every array to be exactly 8760 h.
        Stripping of leap-day hours is the responsibility of pvgis_client.
        """
        data = {2020: np.ones(8784)}
        with pytest.raises(ValueError, match="8760"):
            compute_p50_p90(data)

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
