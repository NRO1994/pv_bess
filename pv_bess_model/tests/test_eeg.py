"""Unit tests for pv_bess_model.market.eeg."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pv_bess_model.market.eeg import (
    EegConfig,
    apply_eeg_floor,
    eeg_config_from_dict,
    effective_eeg_price,
)


# ---------------------------------------------------------------------------
# EegConfig construction
# ---------------------------------------------------------------------------


class TestEegConfigFromDict:
    """Tests for eeg_config_from_dict()."""

    def test_all_fields_parsed(self) -> None:
        d = {
            "type": "eeg",
            "floor_price_eur_per_kwh": 0.0735,
            "fixed_price_years": 20,
            "eeg_inflation": True,
        }
        cfg = eeg_config_from_dict(d)
        assert math.isclose(cfg.floor_price_eur_per_kwh, 0.0735)
        assert cfg.fixed_price_years == 20
        assert cfg.inflation_enabled is True

    def test_inflation_defaults_to_false(self) -> None:
        d = {
            "type": "eeg",
            "floor_price_eur_per_kwh": 0.0735,
            "fixed_price_years": 20,
        }
        cfg = eeg_config_from_dict(d)
        assert cfg.inflation_enabled is False


# ---------------------------------------------------------------------------
# effective_eeg_price()
# ---------------------------------------------------------------------------


class TestEffectiveEegPrice:
    """Tests for effective_eeg_price()."""

    @pytest.fixture
    def eeg_no_inflation(self) -> EegConfig:
        return EegConfig(
            floor_price_eur_per_kwh=0.0735,
            fixed_price_years=20,
            inflation_enabled=False,
        )

    @pytest.fixture
    def eeg_with_inflation(self) -> EegConfig:
        return EegConfig(
            floor_price_eur_per_kwh=0.0735,
            fixed_price_years=20,
            inflation_enabled=True,
        )

    def test_within_period_no_inflation(self, eeg_no_inflation: EegConfig) -> None:
        """During fixed-price period without inflation, price equals base."""
        price = effective_eeg_price(eeg_no_inflation, year=1, inflation_rate=0.02)
        assert math.isclose(price, 0.0735)

    def test_within_period_year_10_no_inflation(self, eeg_no_inflation: EegConfig) -> None:
        price = effective_eeg_price(eeg_no_inflation, year=10, inflation_rate=0.02)
        assert math.isclose(price, 0.0735)

    def test_last_year_of_period(self, eeg_no_inflation: EegConfig) -> None:
        """Year 20 (last year of 20-year period) still has the floor."""
        price = effective_eeg_price(eeg_no_inflation, year=20, inflation_rate=0.02)
        assert math.isclose(price, 0.0735)

    def test_after_period_returns_zero(self, eeg_no_inflation: EegConfig) -> None:
        """Year 21 is past the 20-year period → price = 0.0."""
        price = effective_eeg_price(eeg_no_inflation, year=21, inflation_rate=0.02)
        assert price == 0.0

    def test_year_25_returns_zero(self, eeg_no_inflation: EegConfig) -> None:
        price = effective_eeg_price(eeg_no_inflation, year=25, inflation_rate=0.02)
        assert price == 0.0

    def test_with_inflation_year_1(self, eeg_with_inflation: EegConfig) -> None:
        """Year 1 is the base year – no inflation applied."""
        price = effective_eeg_price(eeg_with_inflation, year=1, inflation_rate=0.02)
        assert math.isclose(price, 0.0735, rel_tol=1e-9)

    def test_with_inflation_year_10(self, eeg_with_inflation: EegConfig) -> None:
        price = effective_eeg_price(eeg_with_inflation, year=10, inflation_rate=0.02)
        expected = 0.0735 * (1.02 ** 9)
        assert math.isclose(price, expected, rel_tol=1e-9)

    def test_with_inflation_after_period_still_zero(self, eeg_with_inflation: EegConfig) -> None:
        """Inflation does not extend the floor beyond the fixed-price period."""
        price = effective_eeg_price(eeg_with_inflation, year=21, inflation_rate=0.02)
        assert price == 0.0

    def test_zero_inflation_rate(self, eeg_with_inflation: EegConfig) -> None:
        """Inflation enabled but rate = 0 → no escalation."""
        price = effective_eeg_price(eeg_with_inflation, year=5, inflation_rate=0.0)
        assert math.isclose(price, 0.0735)


# ---------------------------------------------------------------------------
# apply_eeg_floor()
# ---------------------------------------------------------------------------


class TestApplyEegFloor:
    """Tests for apply_eeg_floor()."""

    def test_spot_above_floor_unchanged(self) -> None:
        """When all spot prices exceed the floor, output equals spot."""
        spot = np.array([0.10, 0.12, 0.09, 0.15])
        floor = 0.05
        result = apply_eeg_floor(spot, floor)
        np.testing.assert_array_almost_equal(result, spot)

    def test_spot_below_floor_lifted(self) -> None:
        """When spot prices are below the floor, they are lifted to the floor."""
        spot = np.array([0.02, 0.03, 0.01, 0.04])
        floor = 0.05
        result = apply_eeg_floor(spot, floor)
        np.testing.assert_array_almost_equal(result, np.full(4, 0.05))

    def test_mixed_prices(self) -> None:
        """Mixed: some above, some below the floor."""
        spot = np.array([0.02, 0.08, 0.05, 0.10])
        floor = 0.05
        expected = np.array([0.05, 0.08, 0.05, 0.10])
        result = apply_eeg_floor(spot, floor)
        np.testing.assert_array_almost_equal(result, expected)

    def test_spot_equal_to_floor(self) -> None:
        """Spot exactly at floor → stays at floor."""
        spot = np.array([0.0735, 0.0735])
        result = apply_eeg_floor(spot, 0.0735)
        np.testing.assert_array_almost_equal(result, spot)

    def test_negative_spot_prices_lifted(self) -> None:
        """Negative spot prices are lifted to the floor."""
        spot = np.array([-0.01, -0.05, 0.08])
        floor = 0.0735
        expected = np.array([0.0735, 0.0735, 0.08])
        result = apply_eeg_floor(spot, floor)
        np.testing.assert_array_almost_equal(result, expected)

    def test_zero_floor_returns_spot_copy(self) -> None:
        """Floor = 0.0 means no floor active → spot prices unchanged."""
        spot = np.array([-0.01, 0.05, 0.10])
        result = apply_eeg_floor(spot, 0.0)
        np.testing.assert_array_almost_equal(result, spot)
        # Must be a copy, not the same object
        assert result is not spot

    def test_output_shape_matches_input(self) -> None:
        spot = np.ones(8760) * 0.04
        result = apply_eeg_floor(spot, 0.05)
        assert result.shape == spot.shape

    def test_does_not_modify_input(self) -> None:
        """Input array must not be mutated."""
        spot = np.array([0.02, 0.08])
        original = spot.copy()
        apply_eeg_floor(spot, 0.05)
        np.testing.assert_array_equal(spot, original)
