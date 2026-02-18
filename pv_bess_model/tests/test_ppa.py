"""Unit tests for pv_bess_model.market.ppa."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pv_bess_model.market.ppa import (
    PPA_TYPE_BASELOAD,
    PPA_TYPE_COLLAR,
    PPA_TYPE_FLOOR,
    PPA_TYPE_NONE,
    PPA_TYPE_PAY_AS_PRODUCED,
    PpaConfig,
    apply_collar_ppa,
    apply_floor_ppa,
    apply_pay_as_produced,
    baseload_level_kwh,
    baseload_revenue,
    effective_collar_prices,
    effective_floor_price,
    effective_ppa_price_for_year,
    pay_as_produced_price,
    ppa_config_from_dict,
)


# ---------------------------------------------------------------------------
# ppa_config_from_dict
# ---------------------------------------------------------------------------


class TestPpaConfigFromDict:
    """Tests for ppa_config_from_dict()."""

    def test_floor_ppa_parsed(self) -> None:
        d = {
            "type": "ppa_floor",
            "pay_as_produced_price_eur_per_kwh": None,
            "baseload_mw": None,
            "floor_price_eur_per_kwh": 0.055,
            "cap_price_eur_per_kwh": None,
            "duration_years": 15,
            "inflation_on_ppa": False,
            "guarantee_of_origin_eur_per_kwh": 0.005,
        }
        cfg = ppa_config_from_dict(d)
        assert cfg.ppa_type == PPA_TYPE_FLOOR
        assert math.isclose(cfg.floor_price_eur_per_kwh, 0.055)
        assert cfg.cap_price_eur_per_kwh is None
        assert cfg.duration_years == 15
        assert cfg.inflation_enabled is False
        assert math.isclose(cfg.goo_premium_eur_per_kwh, 0.005)

    def test_collar_ppa_parsed(self) -> None:
        d = {
            "type": "ppa_collar",
            "pay_as_produced_price_eur_per_kwh": None,
            "baseload_mw": None,
            "floor_price_eur_per_kwh": 0.04,
            "cap_price_eur_per_kwh": 0.10,
            "duration_years": 10,
            "inflation_on_ppa": True,
            "guarantee_of_origin_eur_per_kwh": 0.003,
        }
        cfg = ppa_config_from_dict(d)
        assert cfg.ppa_type == PPA_TYPE_COLLAR
        assert math.isclose(cfg.floor_price_eur_per_kwh, 0.04)
        assert math.isclose(cfg.cap_price_eur_per_kwh, 0.10)
        assert cfg.inflation_enabled is True

    def test_pay_as_produced_parsed(self) -> None:
        d = {
            "type": "ppa_pay_as_produced",
            "pay_as_produced_price_eur_per_kwh": 0.065,
            "baseload_mw": None,
            "floor_price_eur_per_kwh": None,
            "cap_price_eur_per_kwh": None,
            "duration_years": 12,
            "inflation_on_ppa": False,
            "guarantee_of_origin_eur_per_kwh": 0.005,
        }
        cfg = ppa_config_from_dict(d)
        assert cfg.ppa_type == PPA_TYPE_PAY_AS_PRODUCED
        assert math.isclose(cfg.pay_as_produced_price_eur_per_kwh, 0.065)

    def test_baseload_parsed(self) -> None:
        d = {
            "type": "ppa_baseload",
            "pay_as_produced_price_eur_per_kwh": 0.06,
            "baseload_mw": 2.5,
            "floor_price_eur_per_kwh": None,
            "cap_price_eur_per_kwh": None,
            "duration_years": 10,
            "inflation_on_ppa": False,
            "guarantee_of_origin_eur_per_kwh": 0.005,
        }
        cfg = ppa_config_from_dict(d)
        assert cfg.ppa_type == PPA_TYPE_BASELOAD
        assert math.isclose(cfg.baseload_mw, 2.5)

    def test_none_type_parsed(self) -> None:
        d = {
            "type": "none",
            "pay_as_produced_price_eur_per_kwh": None,
            "baseload_mw": None,
            "floor_price_eur_per_kwh": None,
            "cap_price_eur_per_kwh": None,
            "duration_years": 10,
            "inflation_on_ppa": False,
            "guarantee_of_origin_eur_per_kwh": 0.005,
        }
        cfg = ppa_config_from_dict(d)
        assert cfg.ppa_type == PPA_TYPE_NONE

    def test_invalid_type_raises(self) -> None:
        d = {"type": "invalid_ppa"}
        with pytest.raises(ValueError, match="Unknown PPA type"):
            ppa_config_from_dict(d)


# ---------------------------------------------------------------------------
# Pay-as-produced
# ---------------------------------------------------------------------------


class TestPayAsProduced:
    """Tests for pay_as_produced_price() and apply_pay_as_produced()."""

    @pytest.fixture
    def pap_config(self) -> PpaConfig:
        return PpaConfig(
            ppa_type=PPA_TYPE_PAY_AS_PRODUCED,
            pay_as_produced_price_eur_per_kwh=0.065,
            baseload_mw=None,
            floor_price_eur_per_kwh=None,
            cap_price_eur_per_kwh=None,
            duration_years=12,
            inflation_enabled=False,
            goo_premium_eur_per_kwh=0.005,
        )

    @pytest.fixture
    def pap_config_inflation(self) -> PpaConfig:
        return PpaConfig(
            ppa_type=PPA_TYPE_PAY_AS_PRODUCED,
            pay_as_produced_price_eur_per_kwh=0.065,
            baseload_mw=None,
            floor_price_eur_per_kwh=None,
            cap_price_eur_per_kwh=None,
            duration_years=12,
            inflation_enabled=True,
            goo_premium_eur_per_kwh=0.005,
        )

    def test_price_within_period(self, pap_config: PpaConfig) -> None:
        price = pay_as_produced_price(pap_config, year=5, inflation_rate=0.02)
        assert math.isclose(price, 0.065)

    def test_price_after_period(self, pap_config: PpaConfig) -> None:
        price = pay_as_produced_price(pap_config, year=13, inflation_rate=0.02)
        assert price == 0.0

    def test_price_with_inflation(self, pap_config_inflation: PpaConfig) -> None:
        price = pay_as_produced_price(pap_config_inflation, year=5, inflation_rate=0.02)
        expected = 0.065 * (1.02 ** 5)
        assert math.isclose(price, expected, rel_tol=1e-9)

    def test_apply_revenue(self) -> None:
        production = np.array([100.0, 200.0, 50.0, 0.0])
        price = 0.065
        goo = 0.005
        result = apply_pay_as_produced(production, price, goo)
        expected = production * (price + goo)
        np.testing.assert_array_almost_equal(result, expected)

    def test_apply_zero_production(self) -> None:
        production = np.zeros(24)
        result = apply_pay_as_produced(production, 0.065, 0.005)
        np.testing.assert_array_almost_equal(result, np.zeros(24))


# ---------------------------------------------------------------------------
# Baseload PPA
# ---------------------------------------------------------------------------


class TestBaseloadPPA:
    """Tests for baseload_level_kwh() and baseload_revenue()."""

    def test_baseload_from_mw(self) -> None:
        """2.5 MW → 2500 kWh/h."""
        bl = baseload_level_kwh(baseload_mw=2.5, annual_production_kwh=0.0)
        assert math.isclose(bl, 2500.0)

    def test_baseload_auto_from_production(self) -> None:
        """Auto-derive: 8760000 kWh / 8760 h = 1000 kWh/h."""
        bl = baseload_level_kwh(baseload_mw=None, annual_production_kwh=8_760_000.0)
        assert math.isclose(bl, 1000.0)

    def test_revenue_export_exceeds_baseload(self) -> None:
        """When export > baseload → excess sold at spot, ppa revenue on baseload."""
        export = np.array([200.0, 300.0])
        spot = np.array([0.05, 0.06])
        baseload = 100.0
        ppa_price = 0.07
        goo = 0.005
        result = baseload_revenue(export, spot, baseload, ppa_price, goo)
        expected = np.array([
            100.0 * 0.075 + 100.0 * 0.05,  # baseload rev + 100 excess × spot
            100.0 * 0.075 + 200.0 * 0.06,  # baseload rev + 200 excess × spot
        ])
        np.testing.assert_array_almost_equal(result, expected)

    def test_revenue_export_below_baseload(self) -> None:
        """When export < baseload → shortfall bought at spot (negative net)."""
        export = np.array([50.0, 80.0])
        spot = np.array([0.10, 0.08])
        baseload = 100.0
        ppa_price = 0.07
        goo = 0.005
        result = baseload_revenue(export, spot, baseload, ppa_price, goo)
        expected = np.array([
            100.0 * 0.075 + (50.0 - 100.0) * 0.10,   # baseload rev - shortfall cost
            100.0 * 0.075 + (80.0 - 100.0) * 0.08,
        ])
        np.testing.assert_array_almost_equal(result, expected)

    def test_revenue_exact_baseload(self) -> None:
        """When export == baseload → no imbalance."""
        export = np.array([100.0])
        spot = np.array([0.05])
        baseload = 100.0
        ppa_price = 0.07
        goo = 0.005
        result = baseload_revenue(export, spot, baseload, ppa_price, goo)
        expected = np.array([100.0 * 0.075])
        np.testing.assert_array_almost_equal(result, expected)

    def test_revenue_zero_export(self) -> None:
        """Zero export → full shortfall cost."""
        export = np.array([0.0])
        spot = np.array([0.10])
        baseload = 100.0
        ppa_price = 0.07
        goo = 0.005
        result = baseload_revenue(export, spot, baseload, ppa_price, goo)
        expected = np.array([100.0 * 0.075 + (-100.0) * 0.10])
        np.testing.assert_array_almost_equal(result, expected)


# ---------------------------------------------------------------------------
# Floor PPA
# ---------------------------------------------------------------------------


class TestFloorPPA:
    """Tests for effective_floor_price() and apply_floor_ppa()."""

    @pytest.fixture
    def floor_config(self) -> PpaConfig:
        return PpaConfig(
            ppa_type=PPA_TYPE_FLOOR,
            pay_as_produced_price_eur_per_kwh=None,
            baseload_mw=None,
            floor_price_eur_per_kwh=0.055,
            cap_price_eur_per_kwh=None,
            duration_years=15,
            inflation_enabled=False,
            goo_premium_eur_per_kwh=0.005,
        )

    @pytest.fixture
    def floor_config_inflation(self) -> PpaConfig:
        return PpaConfig(
            ppa_type=PPA_TYPE_FLOOR,
            pay_as_produced_price_eur_per_kwh=None,
            baseload_mw=None,
            floor_price_eur_per_kwh=0.055,
            cap_price_eur_per_kwh=None,
            duration_years=15,
            inflation_enabled=True,
            goo_premium_eur_per_kwh=0.005,
        )

    def test_floor_price_within_period(self, floor_config: PpaConfig) -> None:
        """Floor + GoO = 0.055 + 0.005 = 0.060."""
        price = effective_floor_price(floor_config, year=5, inflation_rate=0.02)
        assert math.isclose(price, 0.060)

    def test_floor_price_after_period(self, floor_config: PpaConfig) -> None:
        price = effective_floor_price(floor_config, year=16, inflation_rate=0.02)
        assert price == 0.0

    def test_floor_price_with_inflation(self, floor_config_inflation: PpaConfig) -> None:
        price = effective_floor_price(floor_config_inflation, year=5, inflation_rate=0.02)
        expected = 0.055 * (1.02 ** 5) + 0.005
        assert math.isclose(price, expected, rel_tol=1e-9)

    def test_apply_floor_spot_above(self) -> None:
        spot = np.array([0.08, 0.10])
        result = apply_floor_ppa(spot, 0.06)
        np.testing.assert_array_almost_equal(result, spot)

    def test_apply_floor_spot_below(self) -> None:
        spot = np.array([0.02, 0.04])
        result = apply_floor_ppa(spot, 0.06)
        np.testing.assert_array_almost_equal(result, np.array([0.06, 0.06]))

    def test_apply_floor_mixed(self) -> None:
        spot = np.array([0.02, 0.08, 0.06, 0.10])
        expected = np.array([0.06, 0.08, 0.06, 0.10])
        result = apply_floor_ppa(spot, 0.06)
        np.testing.assert_array_almost_equal(result, expected)

    def test_apply_floor_negative_spot(self) -> None:
        """Negative spot prices lifted to floor."""
        spot = np.array([-0.01, 0.08])
        result = apply_floor_ppa(spot, 0.06)
        np.testing.assert_array_almost_equal(result, np.array([0.06, 0.08]))

    def test_apply_floor_zero_returns_copy(self) -> None:
        """Floor = 0.0 means no floor → spot copy."""
        spot = np.array([0.02, 0.08])
        result = apply_floor_ppa(spot, 0.0)
        np.testing.assert_array_almost_equal(result, spot)
        assert result is not spot


# ---------------------------------------------------------------------------
# Collar PPA
# ---------------------------------------------------------------------------


class TestCollarPPA:
    """Tests for effective_collar_prices() and apply_collar_ppa()."""

    @pytest.fixture
    def collar_config(self) -> PpaConfig:
        return PpaConfig(
            ppa_type=PPA_TYPE_COLLAR,
            pay_as_produced_price_eur_per_kwh=None,
            baseload_mw=None,
            floor_price_eur_per_kwh=0.04,
            cap_price_eur_per_kwh=0.10,
            duration_years=10,
            inflation_enabled=False,
            goo_premium_eur_per_kwh=0.003,
        )

    @pytest.fixture
    def collar_config_inflation(self) -> PpaConfig:
        return PpaConfig(
            ppa_type=PPA_TYPE_COLLAR,
            pay_as_produced_price_eur_per_kwh=None,
            baseload_mw=None,
            floor_price_eur_per_kwh=0.04,
            cap_price_eur_per_kwh=0.10,
            duration_years=10,
            inflation_enabled=True,
            goo_premium_eur_per_kwh=0.003,
        )

    def test_collar_prices_within_period(self, collar_config: PpaConfig) -> None:
        floor, cap = effective_collar_prices(collar_config, year=5, inflation_rate=0.02)
        assert math.isclose(floor, 0.04 + 0.003)  # floor + GoO
        assert math.isclose(cap, 0.10)

    def test_collar_prices_after_period(self, collar_config: PpaConfig) -> None:
        floor, cap = effective_collar_prices(collar_config, year=11, inflation_rate=0.02)
        assert floor == 0.0
        assert cap == 0.0

    def test_collar_prices_with_inflation(self, collar_config_inflation: PpaConfig) -> None:
        floor, cap = effective_collar_prices(collar_config_inflation, year=5, inflation_rate=0.02)
        expected_floor = 0.04 * (1.02 ** 5) + 0.003
        expected_cap = 0.10 * (1.02 ** 5)
        assert math.isclose(floor, expected_floor, rel_tol=1e-9)
        assert math.isclose(cap, expected_cap, rel_tol=1e-9)

    def test_apply_collar_within_bounds(self) -> None:
        """Spot between floor and cap → unchanged."""
        spot = np.array([0.05, 0.06, 0.08])
        result = apply_collar_ppa(spot, 0.04, 0.10)
        np.testing.assert_array_almost_equal(result, spot)

    def test_apply_collar_below_floor(self) -> None:
        """Spot below floor → lifted to floor."""
        spot = np.array([0.01, 0.02, 0.03])
        result = apply_collar_ppa(spot, 0.04, 0.10)
        np.testing.assert_array_almost_equal(result, np.full(3, 0.04))

    def test_apply_collar_above_cap(self) -> None:
        """Spot above cap → capped."""
        spot = np.array([0.12, 0.15, 0.20])
        result = apply_collar_ppa(spot, 0.04, 0.10)
        np.testing.assert_array_almost_equal(result, np.full(3, 0.10))

    def test_apply_collar_mixed(self) -> None:
        """Mixed: below floor, within, and above cap."""
        spot = np.array([0.01, 0.06, 0.15])
        expected = np.array([0.04, 0.06, 0.10])
        result = apply_collar_ppa(spot, 0.04, 0.10)
        np.testing.assert_array_almost_equal(result, expected)

    def test_apply_collar_negative_spot(self) -> None:
        """Negative spot lifted to floor."""
        spot = np.array([-0.05, 0.06])
        result = apply_collar_ppa(spot, 0.04, 0.10)
        np.testing.assert_array_almost_equal(result, np.array([0.04, 0.06]))

    def test_apply_collar_both_zero_returns_copy(self) -> None:
        """Floor=0, cap=0 (expired) → spot unchanged."""
        spot = np.array([0.05, 0.12])
        result = apply_collar_ppa(spot, 0.0, 0.0)
        np.testing.assert_array_almost_equal(result, spot)
        assert result is not spot

    def test_apply_collar_floor_equals_cap(self) -> None:
        """Floor == cap → all values clamped to that single value."""
        spot = np.array([0.01, 0.05, 0.10])
        result = apply_collar_ppa(spot, 0.05, 0.05)
        np.testing.assert_array_almost_equal(result, np.full(3, 0.05))


# ---------------------------------------------------------------------------
# Unified dispatcher: effective_ppa_price_for_year
# ---------------------------------------------------------------------------


class TestEffectivePpaPriceForYear:
    """Tests for effective_ppa_price_for_year()."""

    @pytest.fixture
    def spot(self) -> np.ndarray:
        return np.array([0.02, 0.06, 0.08, 0.12])

    def test_none_type_returns_spot(self, spot: np.ndarray) -> None:
        cfg = PpaConfig(
            ppa_type=PPA_TYPE_NONE,
            pay_as_produced_price_eur_per_kwh=None,
            baseload_mw=None,
            floor_price_eur_per_kwh=None,
            cap_price_eur_per_kwh=None,
            duration_years=10,
            inflation_enabled=False,
            goo_premium_eur_per_kwh=0.0,
        )
        result = effective_ppa_price_for_year(cfg, year=1, inflation_rate=0.02,
                                               spot_prices_eur_per_kwh=spot)
        np.testing.assert_array_almost_equal(result, spot)

    def test_after_expiry_returns_spot(self, spot: np.ndarray) -> None:
        cfg = PpaConfig(
            ppa_type=PPA_TYPE_FLOOR,
            pay_as_produced_price_eur_per_kwh=None,
            baseload_mw=None,
            floor_price_eur_per_kwh=0.05,
            cap_price_eur_per_kwh=None,
            duration_years=10,
            inflation_enabled=False,
            goo_premium_eur_per_kwh=0.005,
        )
        result = effective_ppa_price_for_year(cfg, year=11, inflation_rate=0.02,
                                               spot_prices_eur_per_kwh=spot)
        np.testing.assert_array_almost_equal(result, spot)

    def test_pay_as_produced_returns_constant_array(self, spot: np.ndarray) -> None:
        cfg = PpaConfig(
            ppa_type=PPA_TYPE_PAY_AS_PRODUCED,
            pay_as_produced_price_eur_per_kwh=0.065,
            baseload_mw=None,
            floor_price_eur_per_kwh=None,
            cap_price_eur_per_kwh=None,
            duration_years=10,
            inflation_enabled=False,
            goo_premium_eur_per_kwh=0.005,
        )
        result = effective_ppa_price_for_year(cfg, year=5, inflation_rate=0.02,
                                               spot_prices_eur_per_kwh=spot)
        np.testing.assert_array_almost_equal(result, np.full(4, 0.065))

    def test_floor_applies_max(self, spot: np.ndarray) -> None:
        cfg = PpaConfig(
            ppa_type=PPA_TYPE_FLOOR,
            pay_as_produced_price_eur_per_kwh=None,
            baseload_mw=None,
            floor_price_eur_per_kwh=0.055,
            cap_price_eur_per_kwh=None,
            duration_years=15,
            inflation_enabled=False,
            goo_premium_eur_per_kwh=0.005,
        )
        result = effective_ppa_price_for_year(cfg, year=5, inflation_rate=0.02,
                                               spot_prices_eur_per_kwh=spot)
        floor = 0.055 + 0.005  # = 0.06
        expected = np.maximum(spot, floor)
        np.testing.assert_array_almost_equal(result, expected)

    def test_collar_applies_clip(self, spot: np.ndarray) -> None:
        cfg = PpaConfig(
            ppa_type=PPA_TYPE_COLLAR,
            pay_as_produced_price_eur_per_kwh=None,
            baseload_mw=None,
            floor_price_eur_per_kwh=0.04,
            cap_price_eur_per_kwh=0.10,
            duration_years=10,
            inflation_enabled=False,
            goo_premium_eur_per_kwh=0.003,
        )
        result = effective_ppa_price_for_year(cfg, year=5, inflation_rate=0.02,
                                               spot_prices_eur_per_kwh=spot)
        floor = 0.04 + 0.003  # = 0.043
        cap = 0.10
        expected = np.clip(spot, floor, cap)
        np.testing.assert_array_almost_equal(result, expected)

    def test_baseload_requires_export(self, spot: np.ndarray) -> None:
        cfg = PpaConfig(
            ppa_type=PPA_TYPE_BASELOAD,
            pay_as_produced_price_eur_per_kwh=0.06,
            baseload_mw=1.0,
            floor_price_eur_per_kwh=None,
            cap_price_eur_per_kwh=None,
            duration_years=10,
            inflation_enabled=False,
            goo_premium_eur_per_kwh=0.005,
        )
        with pytest.raises(ValueError, match="grid_export_kwh"):
            effective_ppa_price_for_year(cfg, year=5, inflation_rate=0.02,
                                         spot_prices_eur_per_kwh=spot)

    def test_baseload_returns_revenue(self, spot: np.ndarray) -> None:
        cfg = PpaConfig(
            ppa_type=PPA_TYPE_BASELOAD,
            pay_as_produced_price_eur_per_kwh=0.06,
            baseload_mw=0.1,  # 100 kWh/h
            floor_price_eur_per_kwh=None,
            cap_price_eur_per_kwh=None,
            duration_years=10,
            inflation_enabled=False,
            goo_premium_eur_per_kwh=0.005,
        )
        export = np.array([50.0, 150.0, 100.0, 200.0])
        result = effective_ppa_price_for_year(
            cfg, year=1, inflation_rate=0.02,
            spot_prices_eur_per_kwh=spot,
            grid_export_kwh=export,
        )
        bl = 100.0  # 0.1 MW × 1000
        effective_ppa = 0.06 + 0.005
        expected = bl * effective_ppa + (export - bl) * spot
        np.testing.assert_array_almost_equal(result, expected)


# ---------------------------------------------------------------------------
# PPA expiry mid-project (tests across boundary)
# ---------------------------------------------------------------------------


class TestPpaExpiryMidProject:
    """Tests that PPA parameters correctly switch off at expiry."""

    @pytest.fixture
    def floor_config_10y(self) -> PpaConfig:
        return PpaConfig(
            ppa_type=PPA_TYPE_FLOOR,
            pay_as_produced_price_eur_per_kwh=None,
            baseload_mw=None,
            floor_price_eur_per_kwh=0.06,
            cap_price_eur_per_kwh=None,
            duration_years=10,
            inflation_enabled=False,
            goo_premium_eur_per_kwh=0.005,
        )

    def test_year_10_floor_active(self, floor_config_10y: PpaConfig) -> None:
        price = effective_floor_price(floor_config_10y, year=10, inflation_rate=0.0)
        assert math.isclose(price, 0.065)  # 0.06 + 0.005

    def test_year_11_floor_zero(self, floor_config_10y: PpaConfig) -> None:
        price = effective_floor_price(floor_config_10y, year=11, inflation_rate=0.0)
        assert price == 0.0

    def test_dispatcher_year_10_has_floor(self, floor_config_10y: PpaConfig) -> None:
        spot = np.array([0.02, 0.08])
        result = effective_ppa_price_for_year(
            floor_config_10y, year=10, inflation_rate=0.0,
            spot_prices_eur_per_kwh=spot,
        )
        expected = np.array([0.065, 0.08])
        np.testing.assert_array_almost_equal(result, expected)

    def test_dispatcher_year_11_pure_spot(self, floor_config_10y: PpaConfig) -> None:
        spot = np.array([0.02, 0.08])
        result = effective_ppa_price_for_year(
            floor_config_10y, year=11, inflation_rate=0.0,
            spot_prices_eur_per_kwh=spot,
        )
        np.testing.assert_array_almost_equal(result, spot)


# ---------------------------------------------------------------------------
# Inflation on PPA
# ---------------------------------------------------------------------------


class TestPpaInflation:
    """Tests for inflation escalation on PPA parameters."""

    def test_pay_as_produced_inflation(self) -> None:
        cfg = PpaConfig(
            ppa_type=PPA_TYPE_PAY_AS_PRODUCED,
            pay_as_produced_price_eur_per_kwh=0.065,
            baseload_mw=None,
            floor_price_eur_per_kwh=None,
            cap_price_eur_per_kwh=None,
            duration_years=10,
            inflation_enabled=True,
            goo_premium_eur_per_kwh=0.005,
        )
        price = pay_as_produced_price(cfg, year=5, inflation_rate=0.03)
        expected = 0.065 * (1.03 ** 5)
        assert math.isclose(price, expected, rel_tol=1e-9)

    def test_collar_inflation_floor_and_cap(self) -> None:
        cfg = PpaConfig(
            ppa_type=PPA_TYPE_COLLAR,
            pay_as_produced_price_eur_per_kwh=None,
            baseload_mw=None,
            floor_price_eur_per_kwh=0.04,
            cap_price_eur_per_kwh=0.10,
            duration_years=10,
            inflation_enabled=True,
            goo_premium_eur_per_kwh=0.003,
        )
        floor, cap = effective_collar_prices(cfg, year=3, inflation_rate=0.02)
        expected_floor = 0.04 * (1.02 ** 3) + 0.003
        expected_cap = 0.10 * (1.02 ** 3)
        assert math.isclose(floor, expected_floor, rel_tol=1e-9)
        assert math.isclose(cap, expected_cap, rel_tol=1e-9)

    def test_no_inflation_price_constant(self) -> None:
        cfg = PpaConfig(
            ppa_type=PPA_TYPE_FLOOR,
            pay_as_produced_price_eur_per_kwh=None,
            baseload_mw=None,
            floor_price_eur_per_kwh=0.05,
            cap_price_eur_per_kwh=None,
            duration_years=20,
            inflation_enabled=False,
            goo_premium_eur_per_kwh=0.005,
        )
        price_y1 = effective_floor_price(cfg, year=1, inflation_rate=0.02)
        price_y10 = effective_floor_price(cfg, year=10, inflation_rate=0.02)
        assert math.isclose(price_y1, price_y10)
