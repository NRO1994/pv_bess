"""Tests for finance/inflation.py – inflate_series and build_inflation_factors."""

from __future__ import annotations

import numpy as np
import pytest

from pv_bess_model.finance.inflation import (
    build_inflation_factors,
    inflate_series,
    inflate_value,
)


class TestInflateSeries:
    """inflate_series applies compound inflation to an array."""

    def test_year_1_no_inflation(self) -> None:
        base = np.array([100.0, 200.0, 300.0])
        result = inflate_series(base, 0.02, year=1)
        np.testing.assert_allclose(result, base)

    def test_year_2_single_inflation(self) -> None:
        base = np.array([100.0, 200.0])
        result = inflate_series(base, 0.05, year=2)
        np.testing.assert_allclose(result, base * 1.05)

    def test_year_5_compound_inflation(self) -> None:
        base = np.array([1000.0])
        result = inflate_series(base, 0.03, year=5)
        expected = 1000.0 * (1.03 ** 4)
        np.testing.assert_allclose(result, [expected])

    def test_zero_rate_no_change(self) -> None:
        base = np.array([42.0, 99.0])
        result = inflate_series(base, 0.0, year=10)
        np.testing.assert_allclose(result, base)

    def test_preserves_shape(self) -> None:
        base = np.zeros(8760)
        result = inflate_series(base, 0.02, year=3)
        assert result.shape == base.shape


class TestBuildInflationFactors:
    """build_inflation_factors returns cumulative factors for each year."""

    def test_first_factor_is_one(self) -> None:
        factors = build_inflation_factors(0.02, 5)
        assert factors[0] == pytest.approx(1.0)

    def test_second_factor(self) -> None:
        factors = build_inflation_factors(0.03, 3)
        assert factors[1] == pytest.approx(1.03)

    def test_length_matches_n_years(self) -> None:
        factors = build_inflation_factors(0.02, 25)
        assert len(factors) == 25

    def test_zero_rate_all_ones(self) -> None:
        factors = build_inflation_factors(0.0, 10)
        np.testing.assert_allclose(factors, np.ones(10))

    def test_monotonically_increasing(self) -> None:
        factors = build_inflation_factors(0.02, 10)
        assert all(factors[i] < factors[i + 1] for i in range(9))

    def test_compound_formula(self) -> None:
        rate = 0.05
        factors = build_inflation_factors(rate, 4)
        expected = np.array([1.0, 1.05, 1.05**2, 1.05**3])
        np.testing.assert_allclose(factors, expected)

    def test_single_year(self) -> None:
        factors = build_inflation_factors(0.10, 1)
        assert len(factors) == 1
        assert factors[0] == pytest.approx(1.0)
