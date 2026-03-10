"""Tests for optimization/analyses.py – config modification helper functions.

Only tests the pure helper functions (_build_eeg_fixed_prices,
_build_collar_prices, _build_baseload_prices). The public run_* functions
depend on run_monte_carlo and are tested via integration tests.
"""

from __future__ import annotations

import pytest

from pv_bess_model.optimization.analyses import (
    _build_baseload_prices,
    _build_collar_prices,
    _build_eeg_fixed_prices,
)


class TestBuildEegFixedPrices:
    """_build_eeg_fixed_prices generates per-year floor prices."""

    def test_basic_no_inflation(self) -> None:
        result = _build_eeg_fixed_prices(
            floor_price_eur_per_kwh=0.07,
            lifetime=5,
            inflation_rate=0.02,
            eeg_inflation=False,
            fixed_price_years=3,
        )
        assert len(result) == 5
        assert result[0] == pytest.approx(0.07)
        assert result[1] == pytest.approx(0.07)
        assert result[2] == pytest.approx(0.07)
        assert result[3] == pytest.approx(0.0)
        assert result[4] == pytest.approx(0.0)

    def test_with_inflation(self) -> None:
        result = _build_eeg_fixed_prices(
            floor_price_eur_per_kwh=0.07,
            lifetime=3,
            inflation_rate=0.05,
            eeg_inflation=True,
            fixed_price_years=3,
        )
        assert len(result) == 3
        # Year 1: no inflation
        assert result[0] == pytest.approx(0.07)
        # Year 2: inflated
        assert result[1] == pytest.approx(0.07 * 1.05)
        # Year 3: compound
        assert result[2] == pytest.approx(0.07 * 1.05**2)

    def test_fixed_years_exceeds_lifetime(self) -> None:
        result = _build_eeg_fixed_prices(
            floor_price_eur_per_kwh=0.07,
            lifetime=2,
            inflation_rate=0.0,
            eeg_inflation=False,
            fixed_price_years=10,
        )
        assert len(result) == 2
        assert all(p == pytest.approx(0.07) for p in result)

    def test_zero_fixed_years(self) -> None:
        result = _build_eeg_fixed_prices(
            floor_price_eur_per_kwh=0.07,
            lifetime=3,
            inflation_rate=0.0,
            eeg_inflation=False,
            fixed_price_years=0,
        )
        assert all(p == pytest.approx(0.0) for p in result)


class TestBuildCollarPrices:
    """_build_collar_prices generates floor, cap, and GoO per-year arrays."""

    def test_basic_no_inflation(self) -> None:
        fixed, cap, goo = _build_collar_prices(
            floor_eur_per_kwh=0.05,
            cap_eur_per_kwh=0.10,
            duration_years=2,
            inflation_on_ppa=False,
            goo_premium_eur_per_kwh=0.005,
            inflation_rate=0.02,
            lifetime=4,
        )
        assert len(fixed) == len(cap) == len(goo) == 4
        # Within duration
        assert fixed[0] == pytest.approx(0.05)
        assert fixed[1] == pytest.approx(0.05)
        assert cap[0] == pytest.approx(0.10)
        assert cap[1] == pytest.approx(0.10)
        assert goo[0] == pytest.approx(0.005)
        assert goo[1] == pytest.approx(0.005)
        # After duration
        assert fixed[2] == pytest.approx(0.0)
        assert cap[2] == pytest.approx(0.0)
        assert goo[2] == pytest.approx(0.0)

    def test_with_inflation(self) -> None:
        fixed, cap, goo = _build_collar_prices(
            floor_eur_per_kwh=0.05,
            cap_eur_per_kwh=0.10,
            duration_years=2,
            inflation_on_ppa=True,
            goo_premium_eur_per_kwh=0.005,
            inflation_rate=0.03,
            lifetime=3,
        )
        # Year 1: no inflation
        assert fixed[0] == pytest.approx(0.05)
        assert cap[0] == pytest.approx(0.10)
        # Year 2: inflated
        assert fixed[1] == pytest.approx(0.05 * 1.03)
        assert cap[1] == pytest.approx(0.10 * 1.03)
        # Year 3: after duration
        assert fixed[2] == pytest.approx(0.0)
        assert cap[2] == pytest.approx(0.0)


class TestBuildBaseloadPrices:
    """_build_baseload_prices generates fixed and GoO per-year arrays."""

    def test_basic_no_inflation(self) -> None:
        fixed, goo = _build_baseload_prices(
            ppa_price_eur_per_kwh=0.06,
            duration_years=2,
            inflation_on_ppa=False,
            goo_premium_eur_per_kwh=0.005,
            inflation_rate=0.02,
            lifetime=4,
        )
        assert len(fixed) == len(goo) == 4
        # Within duration: price + goo baked in, goo array = 0
        assert fixed[0] == pytest.approx(0.065)
        assert fixed[1] == pytest.approx(0.065)
        assert goo[0] == pytest.approx(0.0)
        # After duration
        assert fixed[2] == pytest.approx(0.0)
        assert goo[2] == pytest.approx(0.0)

    def test_with_inflation(self) -> None:
        fixed, goo = _build_baseload_prices(
            ppa_price_eur_per_kwh=0.06,
            duration_years=2,
            inflation_on_ppa=True,
            goo_premium_eur_per_kwh=0.005,
            inflation_rate=0.04,
            lifetime=3,
        )
        # Year 1: no inflation on price, goo still added
        assert fixed[0] == pytest.approx(0.06 + 0.005)
        # Year 2: inflated price + goo
        assert fixed[1] == pytest.approx(0.06 * 1.04 + 0.005)
        # Year 3: after duration
        assert fixed[2] == pytest.approx(0.0)

    def test_zero_goo(self) -> None:
        fixed, goo = _build_baseload_prices(
            ppa_price_eur_per_kwh=0.08,
            duration_years=1,
            inflation_on_ppa=False,
            goo_premium_eur_per_kwh=0.0,
            inflation_rate=0.0,
            lifetime=2,
        )
        assert fixed[0] == pytest.approx(0.08)
        assert fixed[1] == pytest.approx(0.0)
