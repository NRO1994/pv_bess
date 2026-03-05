"""Unit tests for bess/replacement.py – FIX-S2-15: capacity upgrade factor.

Tests cover:
- Default capacity_factor_pct is 100.0 (no upgrade)
- replacement_cost() scales eur_per_kwh by the new (upgraded) capacity
- replacement_cost() leaves eur_per_kw and fixed_eur unaffected by the factor
- replacement_config_from_dict() reads capacity_factor_pct from JSON
- replacement_config_from_dict() defaults to 100.0 when field is absent
- Dispatch engine applies upgrade factor to bess_cap in replacement year
- Dispatch engine resets degradation from the upgraded capacity baseline
- Schema accepts capacity_factor_pct in _BESS_REPLACEMENT
- Schema rejects negative capacity_factor_pct
"""

from __future__ import annotations

import numpy as np
import pytest

from pv_bess_model.bess.replacement import (
    ReplacementConfig,
    replacement_config_from_dict,
)
from pv_bess_model.config.defaults import DEFAULT_BESS_REPLACEMENT_CAPACITY_FACTOR_PCT
from pv_bess_model.config.schema import validate_scenario
from pv_bess_model.dispatch.engine import DispatchEngineConfig, run_simulation

ATOL = 1e-6


# ---------------------------------------------------------------------------
# ReplacementConfig – capacity_factor_pct default
# ---------------------------------------------------------------------------


class TestReplacementConfigDefault:
    """capacity_factor_pct defaults to 100 (no upgrade)."""

    def test_default_is_100(self) -> None:
        cfg = ReplacementConfig(enabled=False, year=0)
        assert cfg.capacity_factor_pct == DEFAULT_BESS_REPLACEMENT_CAPACITY_FACTOR_PCT
        assert cfg.capacity_factor_pct == 100.0

    def test_explicit_value_stored(self) -> None:
        cfg = ReplacementConfig(enabled=True, year=5, capacity_factor_pct=120.0)
        assert cfg.capacity_factor_pct == 120.0


# ---------------------------------------------------------------------------
# ReplacementConfig.replacement_cost() – upgrade factor on eur_per_kwh
# ---------------------------------------------------------------------------


class TestReplacementCost:
    """replacement_cost() applies upgrade factor only to the eur_per_kwh component."""

    def test_no_upgrade_equals_original_formula(self) -> None:
        """With factor=100, cost = fixed + kw*power + kwh*capacity (unchanged)."""
        cfg = ReplacementConfig(
            enabled=True,
            year=10,
            fixed_eur=5_000.0,
            eur_per_kw=100.0,
            eur_per_kwh=150.0,
            capacity_factor_pct=100.0,
        )
        expected = 5_000.0 + 100.0 * 2_000.0 + 150.0 * 4_000.0
        assert abs(cfg.replacement_cost(2_000.0, 4_000.0) - expected) < ATOL

    def test_upgrade_120_pct_scales_kwh_component(self) -> None:
        """With factor=120, eur_per_kwh is multiplied by 1.2."""
        cfg = ReplacementConfig(
            enabled=True,
            year=10,
            fixed_eur=0.0,
            eur_per_kw=0.0,
            eur_per_kwh=200.0,
            capacity_factor_pct=120.0,
        )
        # new capacity = 4_000 × 1.2 = 4_800 kWh
        expected = 200.0 * 4_000.0 * 1.2
        assert abs(cfg.replacement_cost(1_000.0, 4_000.0) - expected) < ATOL

    def test_upgrade_50_pct_scales_kwh_down(self) -> None:
        """Factor of 50 halves the eur_per_kwh component (smaller replacement)."""
        cfg = ReplacementConfig(
            enabled=True,
            year=10,
            fixed_eur=0.0,
            eur_per_kw=0.0,
            eur_per_kwh=100.0,
            capacity_factor_pct=50.0,
        )
        expected = 100.0 * 2_000.0 * 0.5
        assert abs(cfg.replacement_cost(500.0, 2_000.0) - expected) < ATOL

    def test_kw_component_unaffected_by_upgrade(self) -> None:
        """eur_per_kw uses the original power rating regardless of upgrade factor."""
        cfg_100 = ReplacementConfig(
            enabled=True, year=5, eur_per_kw=150.0, capacity_factor_pct=100.0
        )
        cfg_150 = ReplacementConfig(
            enabled=True, year=5, eur_per_kw=150.0, capacity_factor_pct=150.0
        )
        power = 1_000.0
        capacity = 2_000.0
        # kw component must be identical
        assert abs(
            cfg_100.replacement_cost(power, capacity)
            - cfg_150.replacement_cost(power, capacity)
            - cfg_150.eur_per_kwh * capacity * (1.5 - 1.0)  # only kwh part differs
        ) < ATOL
        assert abs(
            cfg_100.eur_per_kw * power - cfg_150.eur_per_kw * power
        ) < ATOL  # kW cost identical

    def test_fixed_component_unaffected_by_upgrade(self) -> None:
        """fixed_eur is independent of the upgrade factor."""
        cfg = ReplacementConfig(
            enabled=True,
            year=5,
            fixed_eur=10_000.0,
            eur_per_kw=0.0,
            eur_per_kwh=0.0,
            capacity_factor_pct=200.0,
        )
        # Only fixed_eur → result = 10_000 regardless of factor
        assert abs(cfg.replacement_cost(1_000.0, 2_000.0) - 10_000.0) < ATOL

    def test_all_components_combined(self) -> None:
        """Verify additive formula: fixed + kw*power + kwh*(capacity*factor)."""
        factor = 1.3
        fixed = 3_000.0
        per_kw = 80.0
        per_kwh = 120.0
        power = 2_500.0
        capacity = 5_000.0
        cfg = ReplacementConfig(
            enabled=True,
            year=12,
            fixed_eur=fixed,
            eur_per_kw=per_kw,
            eur_per_kwh=per_kwh,
            capacity_factor_pct=factor * 100.0,
        )
        expected = fixed + per_kw * power + per_kwh * (capacity * factor)
        assert abs(cfg.replacement_cost(power, capacity) - expected) < ATOL


# ---------------------------------------------------------------------------
# replacement_config_from_dict() – reading and defaulting
# ---------------------------------------------------------------------------


class TestReplacementConfigFromDict:
    """Factory function reads capacity_factor_pct correctly."""

    def test_reads_capacity_factor_pct(self) -> None:
        d = {"enabled": True, "year": 8, "capacity_factor_pct": 125.0}
        cfg = replacement_config_from_dict(d)
        assert cfg.capacity_factor_pct == 125.0

    def test_defaults_to_100_when_absent(self) -> None:
        d = {"enabled": True, "year": 8}
        cfg = replacement_config_from_dict(d)
        assert cfg.capacity_factor_pct == 100.0

    def test_zero_factor_accepted(self) -> None:
        """Edge case: factor=0 produces a zero-capacity replacement."""
        d = {"enabled": True, "year": 5, "capacity_factor_pct": 0.0}
        cfg = replacement_config_from_dict(d)
        assert cfg.capacity_factor_pct == 0.0

    def test_all_fields_round_trip(self) -> None:
        d = {
            "enabled": True,
            "year": 10,
            "fixed_eur": 5_000.0,
            "eur_per_kw": 80.0,
            "eur_per_kwh": 150.0,
            "capacity_factor_pct": 110.0,
        }
        cfg = replacement_config_from_dict(d)
        assert cfg.enabled is True
        assert cfg.year == 10
        assert cfg.fixed_eur == 5_000.0
        assert cfg.eur_per_kw == 80.0
        assert cfg.eur_per_kwh == 150.0
        assert cfg.capacity_factor_pct == 110.0


# ---------------------------------------------------------------------------
# JSON schema – capacity_factor_pct validation
# ---------------------------------------------------------------------------


def _minimal_scenario(replacement_extra: dict) -> dict:
    """Return a minimal but valid scenario dict with replacement settings."""
    base = {
        "scenario": {"name": "test", "output": {"directory": "./out"}},
        "project_settings": {
            "lifetime_years": 5,
            "commissioning_year": 2027,
            "discount_rate": 0.06,
            "operating_mode": "green",
            "location": {
                "latitude": 50.0,
                "longitude": 10.0,
                "pvgis_database": "PVGIS-SARAH2",
            },
            "technology": {
                "pv": {
                    "design": {
                        "peak_power_kwp": 1000.0,
                        "mounting_type": "free",
                        "azimuth_deg": 0,
                        "tilt_deg": 30,
                    },
                    "performance": {"degradation_rate_pct_per_year": 0.4},
                    "costs": {
                        "capex": {"eur_per_kw": 800.0},
                        "opex": {"pct_of_capex": 0.01},
                    },
                },
                "bess": {
                    "design_space": {
                        "scale_pct_of_pv": [0.0, 20.0],
                        "e_to_p_ratio_hours": [2.0],
                    },
                    "performance": {
                        "round_trip_efficiency_pct": 88.0,
                        "min_soc_pct": 10.0,
                        "max_soc_pct": 90.0,
                        "degradation_rate_pct_per_year": 2.0,
                        "bess_availability_pct": 97.0,
                    },
                    "costs": {
                        "capex": {"eur_per_kwh": 250.0},
                        "opex": {"pct_of_capex": 0.015},
                        "replacement": {"enabled": False, **replacement_extra},
                    },
                },
                "grid_connection": {
                    "max_export_kw": 900.0,
                    "costs": {
                        "capex": {"eur_per_kw": 100.0},
                        "opex": {"pct_of_capex": 0.015},
                    },
                },
            },
            "finance": {
                "leverage_pct": 70.0,
                "interest_rate_pct": 4.5,
                "loan_tenor_years": 18,
                "debt_uses_p90": False,
                "inflation_rate": 0.02,
                "revenue_streams": {"marketing": {"type": "market"}},
                "price_inputs": {
                    "day_ahead_csv": "prices.csv",
                    "price_unit": "eur_per_mwh",
                },
                "tax": {
                    "afa_years_pv": 20,
                    "afa_years_bess": 10,
                    "gewerbesteuer_hebesatz": 400,
                    "gewerbesteuer_messzahl": 0.035,
                },
            },
        },
    }
    return base


class TestSchemaCapacityFactor:
    """Schema validation for capacity_factor_pct."""

    def test_schema_accepts_valid_factor(self) -> None:
        scenario = _minimal_scenario({"capacity_factor_pct": 120.0})
        validate_scenario(scenario)  # must not raise

    def test_schema_accepts_factor_100(self) -> None:
        scenario = _minimal_scenario({"capacity_factor_pct": 100.0})
        validate_scenario(scenario)

    def test_schema_accepts_zero_factor(self) -> None:
        scenario = _minimal_scenario({"capacity_factor_pct": 0.0})
        validate_scenario(scenario)

    def test_schema_rejects_negative_factor(self) -> None:
        import jsonschema

        scenario = _minimal_scenario({"capacity_factor_pct": -10.0})
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(scenario)

    def test_schema_accepts_absent_factor(self) -> None:
        """capacity_factor_pct is optional; schema must not require it."""
        scenario = _minimal_scenario({})  # no capacity_factor_pct
        validate_scenario(scenario)


# ---------------------------------------------------------------------------
# Dispatch engine – upgrade factor applied in replacement year
# ---------------------------------------------------------------------------


def _make_engine_config(
    nameplate_kwh: float,
    replacement: ReplacementConfig,
    lifetime: int = 3,
    deg_rate: float = 0.05,
) -> DispatchEngineConfig:
    """Build a minimal DispatchEngineConfig for replacement tests."""
    return DispatchEngineConfig(
        mode="green",
        grid_max_kw=500.0,
        bess_nameplate_kwh=nameplate_kwh,
        bess_max_charge_kw=nameplate_kwh / 2.0,
        bess_max_discharge_kw=nameplate_kwh / 2.0,
        bess_rte=0.90,
        bess_min_soc_pct=10.0,
        bess_max_soc_pct=90.0,
        bess_degradation_rate=deg_rate,
        pv_degradation_rate=0.0,
        replacement=replacement,
        lifetime_years=lifetime,
        bess_power_kw=nameplate_kwh / 2.0,
    )


def _flat_timeseries() -> np.ndarray:
    """Return a constant 8760-h PV timeseries and price array."""
    return np.full(8760, 10.0, dtype=float)


def _flat_spot() -> np.ndarray:
    return np.full(8760, 0.05, dtype=float)


class TestEngineCapacityUpgrade:
    """Engine correctly applies capacity_factor_pct on replacement year."""

    def test_upgrade_120_sets_larger_bess_cap(self) -> None:
        """After replacement with factor=120, bess_cap = nameplate × 1.2."""
        nameplate = 200.0
        deg = 0.05
        replacement = ReplacementConfig(
            enabled=True, year=2, capacity_factor_pct=120.0
        )
        config = _make_engine_config(nameplate, replacement, lifetime=3, deg_rate=deg)
        pv = _flat_timeseries()
        spot = _flat_spot()
        result = run_simulation(
            config=config,
            pv_base_timeseries=pv,
            spot_prices_yearly=[spot] * 3,
            fixed_prices_yearly=[0.0] * 3,
            offline_days_yearly=[set()] * 3,
            goo_prices_yearly=[0.0] * 3,
            cap_prices_yearly=[0.0] * 3,
        )
        # Year 2 = replacement year → capacity = 200 × 1.2 = 240
        cap_y2 = result.annual_results[1].bess_capacity_kwh
        assert abs(cap_y2 - nameplate * 1.2) < 1e-4

    def test_no_upgrade_default_100_same_as_original_behavior(self) -> None:
        """Factor=100 (default) must produce nameplate capacity in replacement year."""
        nameplate = 200.0
        deg = 0.05
        replacement = ReplacementConfig(
            enabled=True, year=2, capacity_factor_pct=100.0
        )
        config = _make_engine_config(nameplate, replacement, lifetime=3, deg_rate=deg)
        pv = _flat_timeseries()
        spot = _flat_spot()
        result = run_simulation(
            config=config,
            pv_base_timeseries=pv,
            spot_prices_yearly=[spot] * 3,
            fixed_prices_yearly=[0.0] * 3,
            offline_days_yearly=[set()] * 3,
            goo_prices_yearly=[0.0] * 3,
            cap_prices_yearly=[0.0] * 3,
        )
        cap_y2 = result.annual_results[1].bess_capacity_kwh
        assert abs(cap_y2 - nameplate) < 1e-4

    def test_degradation_after_upgrade_restarts_from_upgraded_nameplate(self) -> None:
        """Year 3 capacity = upgraded_nameplate × (1-deg)^1 (age=1 after reset)."""
        nameplate = 200.0
        deg = 0.10
        upgrade_pct = 150.0
        replacement = ReplacementConfig(
            enabled=True, year=2, capacity_factor_pct=upgrade_pct
        )
        config = _make_engine_config(nameplate, replacement, lifetime=3, deg_rate=deg)
        pv = _flat_timeseries()
        spot = _flat_spot()
        result = run_simulation(
            config=config,
            pv_base_timeseries=pv,
            spot_prices_yearly=[spot] * 3,
            fixed_prices_yearly=[0.0] * 3,
            offline_days_yearly=[set()] * 3,
            goo_prices_yearly=[0.0] * 3,
            cap_prices_yearly=[0.0] * 3,
        )
        upgraded_nameplate = nameplate * upgrade_pct / 100.0
        # Year 3: age=1 after reset → cap = upgraded_nameplate × (1-deg)^1
        expected_y3 = upgraded_nameplate * (1.0 - deg) ** 1
        cap_y3 = result.annual_results[2].bess_capacity_kwh
        assert abs(cap_y3 - expected_y3) < 1e-4

    def test_upgrade_cost_scales_kwh_component(self) -> None:
        """Replacement cost reported in annual result uses upgraded capacity for kwh."""
        nameplate = 200.0
        power = 100.0
        per_kwh = 150.0
        factor_pct = 120.0
        replacement = ReplacementConfig(
            enabled=True,
            year=2,
            fixed_eur=0.0,
            eur_per_kw=0.0,
            eur_per_kwh=per_kwh,
            capacity_factor_pct=factor_pct,
        )
        config = _make_engine_config(nameplate, replacement, lifetime=3, deg_rate=0.0)
        config = DispatchEngineConfig(
            mode="green",
            grid_max_kw=500.0,
            bess_nameplate_kwh=nameplate,
            bess_max_charge_kw=power,
            bess_max_discharge_kw=power,
            bess_rte=0.90,
            bess_min_soc_pct=10.0,
            bess_max_soc_pct=90.0,
            bess_degradation_rate=0.0,
            pv_degradation_rate=0.0,
            replacement=replacement,
            lifetime_years=3,
            bess_power_kw=power,
        )
        pv = _flat_timeseries()
        spot = _flat_spot()
        result = run_simulation(
            config=config,
            pv_base_timeseries=pv,
            spot_prices_yearly=[spot] * 3,
            fixed_prices_yearly=[0.0] * 3,
            offline_days_yearly=[set()] * 3,
            goo_prices_yearly=[0.0] * 3,
            cap_prices_yearly=[0.0] * 3,
        )
        expected_cost = per_kwh * nameplate * (factor_pct / 100.0)
        cost_y2 = result.annual_results[1].replacement_cost
        assert abs(cost_y2 - expected_cost) < 1e-4
        # Other years: no replacement cost
        assert abs(result.annual_results[0].replacement_cost) < 1e-4
        assert abs(result.annual_results[2].replacement_cost) < 1e-4
