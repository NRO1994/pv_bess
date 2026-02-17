"""Unit tests for pv_bess_model.pv.degradation.

Degradation formula (CLAUDE.md):
  production[Y] = base_production × (1 − degradation_rate) ^ Y
  Y is 1-indexed (Y=1 is the first operating year; Y=0 does not exist).

Covers:
- apply_degradation: year 0 is not part of the output (first entry = year 1)
- apply_degradation: year-10 factor at 0.4 %/year ≈ 0.9608
- apply_degradation: 0 % rate → every year equals the base timeseries
- apply_degradation: year-1 factor = (1-rate)^1, year-N = (1-rate)^N
- apply_degradation: output length equals lifetime_years
- apply_degradation: production decreases monotonically
- apply_degradation: production never exceeds base
- apply_degradation: array shape preserved
- apply_degradation: each year is an independent array (mutation safety)
- apply_degradation: small rate accumulated over many years
- apply_degradation: invalid rate (negative, ≥ 1) raises ValueError
- apply_degradation: lifetime_years < 1 raises ValueError
- degradation_factor: year 0 raises ValueError
- degradation_factor: spot checks at known values
- degradation_factor: zero rate → factor = 1 for all years
- degradation_factor: factor always in (0, 1] for valid inputs
- degradation_factor: consistent with apply_degradation scalars
"""

from __future__ import annotations

import numpy as np
import pytest

from pv_bess_model.config.defaults import HOURS_PER_YEAR
from pv_bess_model.pv.degradation import apply_degradation, degradation_factor

# ---------------------------------------------------------------------------
# apply_degradation – core behaviour
# ---------------------------------------------------------------------------


class TestApplyDegradation:
    def test_year_0_not_in_output(self):
        """Year 0 (construction) must not appear in the output list.

        The first element (index 0) corresponds to project year 1 and already
        has the degradation factor (1-rate)^1 applied.  There is no element
        representing an undegraded year 0.
        """
        rate = 0.004
        base = np.ones(HOURS_PER_YEAR) * 100.0
        result = apply_degradation(base, degradation_rate=rate, lifetime_years=5)
        # result[0] must be year-1 production (degraded), not year-0 (base)
        assert result[0][0] == pytest.approx(100.0 * (1 - rate) ** 1)
        assert result[0][0] < 100.0  # strictly less than undegraded base

    def test_year_10_factor_at_0_4pct(self):
        """At 0.4 %/year, year-10 factor must be ≈ 0.9608.

        Analytical:  (1 − 0.004)^10 = 0.996^10 ≈ 0.96075… ≈ 0.9608.
        """
        rate = 0.004
        expected_factor = (1.0 - rate) ** 10  # ≈ 0.96075...
        assert expected_factor == pytest.approx(0.9608, abs=5e-4)

        base = np.ones(HOURS_PER_YEAR) * 500.0
        result = apply_degradation(base, degradation_rate=rate, lifetime_years=10)
        assert result[9][0] == pytest.approx(500.0 * expected_factor, rel=1e-9)

    def test_zero_rate_output_equals_input(self):
        """0 % degradation rate → every year's timeseries equals the base."""
        base = np.arange(HOURS_PER_YEAR, dtype=float)
        result = apply_degradation(base, degradation_rate=0.0, lifetime_years=10)
        for year_arr in result:
            np.testing.assert_array_equal(year_arr, base)

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
        """A rate of 4.0 (should be 0.04 for 4%) must raise."""
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
    def test_year_0_raises(self):
        """Year 0 (construction) is not a simulation year and must raise ValueError."""
        with pytest.raises(ValueError):
            degradation_factor(0.004, 0)

    def test_year_1_value(self):
        assert degradation_factor(0.004, 1) == pytest.approx((1 - 0.004) ** 1)

    def test_year_10_at_0_4pct(self):
        """At 0.4 %/year, year-10 factor must be ≈ 0.9608."""
        factor = degradation_factor(0.004, 10)
        assert factor == pytest.approx(0.9608, abs=5e-4)

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
