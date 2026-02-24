"""Unit tests for pv_bess_model.bess.replacement (ReplacementConfig)."""

from __future__ import annotations

import math

import pytest

from pv_bess_model.bess.replacement import (
    ReplacementConfig,
    replacement_config_from_dict,
)


# ---------------------------------------------------------------------------
# ReplacementConfig – cost calculation
# ---------------------------------------------------------------------------


class TestReplacementConfig:
    """Tests for ReplacementConfig.replacement_cost()."""

    def test_cost_all_components(self) -> None:
        cfg = ReplacementConfig(
            enabled=True,
            year=12,
            fixed_eur=10_000.0,
            eur_per_kw=120.0,
            eur_per_kwh=141.0,
        )
        # 10 000 + 120×2000 + 141×4000 = 10 000 + 240 000 + 564 000 = 814 000
        cost = cfg.replacement_cost(bess_power_kw=2_000.0, bess_capacity_kwh=4_000.0)
        assert math.isclose(cost, 814_000.0)

    def test_cost_fixed_only(self) -> None:
        cfg = ReplacementConfig(enabled=True, year=10, fixed_eur=50_000.0)
        cost = cfg.replacement_cost(1_000.0, 2_000.0)
        assert math.isclose(cost, 50_000.0)

    def test_cost_zero_when_all_components_zero(self) -> None:
        cfg = ReplacementConfig(enabled=True, year=5)
        cost = cfg.replacement_cost(1_000.0, 2_000.0)
        assert cost == 0.0


# ---------------------------------------------------------------------------
# replacement_config_from_dict()
# ---------------------------------------------------------------------------


class TestReplacementConfigFromDict:
    """Tests for parsing ReplacementConfig from a scenario JSON dict."""

    def test_full_dict_parsed_correctly(self) -> None:
        d = {
            "enabled": True,
            "year": 12,
            "fixed_eur": 0.0,
            "eur_per_kw": 120.0,
            "eur_per_kwh": 141.0,
        }
        cfg = replacement_config_from_dict(d)
        assert cfg.enabled is True
        assert cfg.year == 12
        assert math.isclose(cfg.eur_per_kw, 120.0)
        assert math.isclose(cfg.eur_per_kwh, 141.0)

    def test_missing_cost_keys_default_to_zero(self) -> None:
        d = {"enabled": True, "year": 5}
        cfg = replacement_config_from_dict(d)
        assert cfg.fixed_eur == 0.0
        assert cfg.eur_per_kw == 0.0
        assert cfg.eur_per_kwh == 0.0

    def test_disabled_replacement_from_dict(self) -> None:
        d = {"enabled": False, "year": 12, "eur_per_kwh": 141.0}
        cfg = replacement_config_from_dict(d)
        assert cfg.enabled is False

    def test_matches_scenario_json_example(self) -> None:
        """Ensure parsing matches the example in the scenario JSON schema."""
        d = {
            "enabled": False,
            "year": 12,
            "fixed_eur": 0.0,
            "eur_per_kw": 120.0,
            "eur_per_kwh": 141.0,
        }
        cfg = replacement_config_from_dict(d)
        assert cfg.enabled is False
        assert cfg.year == 12
        assert math.isclose(cfg.eur_per_kw, 120.0)
        assert math.isclose(cfg.eur_per_kwh, 141.0)
        # Disabled – cost calculation still works (just not triggered)
        cost = cfg.replacement_cost(100.0, 200.0)
        expected = 0.0 + 120.0 * 100.0 + 141.0 * 200.0
        assert math.isclose(cost, expected)
