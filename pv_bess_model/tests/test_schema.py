"""Unit tests for pv_bess_model.config.schema.

Covers:
- Valid complete scenario passes without error
- Required top-level and nested fields are enforced
- Enum constraints (operating_mode, price_unit, ppa type, …)
- Numeric range constraints (tilt_deg, pct_of_capex, …)
- Type constraints (boolean, integer, number, null)
- Cross-field: MC weight sum validation
- Cross-field: SoC min < max validation
- get_schema() returns the schema dict
"""

from __future__ import annotations

import copy

import jsonschema
import pytest

from pv_bess_model.config.schema import get_schema, validate_scenario

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deep_del(d: dict, *keys: str) -> dict:
    """Return a deep copy of *d* with the nested key path removed."""
    out = copy.deepcopy(d)
    node = out
    for k in keys[:-1]:
        node = node[k]
    del node[keys[-1]]
    return out


def _deep_set(d: dict, value, *keys: str) -> dict:
    """Return a deep copy of *d* with the nested key path set to *value*."""
    out = copy.deepcopy(d)
    node = out
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value
    return out


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestValidScenario:
    """A fully valid scenario must pass without raising."""

    def test_green_eeg_scenario_is_valid(self, sample_scenario_config_green):
        validate_scenario(sample_scenario_config_green)

    def test_grey_ppa_scenario_is_valid(self, sample_scenario_config_grey):
        validate_scenario(sample_scenario_config_grey)

    def test_get_schema_returns_dict(self):
        schema = get_schema()
        assert isinstance(schema, dict)
        assert "$schema" in schema
        assert "properties" in schema

    def test_get_schema_is_copy(self):
        s1 = get_schema()
        s2 = get_schema()
        s1["injected"] = True
        assert "injected" not in s2

    def test_equity_irr_target_null_allowed(self, sample_scenario_config_green):
        cfg = _deep_set(
            sample_scenario_config_green,
            None,
            "project_settings",
            "finance",
            "equity_irr_target",
        )
        validate_scenario(cfg)  # must not raise

    def test_mc_disabled_weights_not_checked(self, sample_scenario_config_green):
        """MC weight validation is skipped when MC is disabled."""
        cfg = copy.deepcopy(sample_scenario_config_green)
        mc = cfg["scenario"].setdefault("monte_carlo", {})
        mc["enabled"] = False
        mc["price_scenarios"] = {
            "a": {"csv_column": "A", "weight": 0.3},
            "b": {"csv_column": "B", "weight": 0.3},
        }
        validate_scenario(cfg)  # sum=0.6, but MC disabled → no error


# ---------------------------------------------------------------------------
# Missing required top-level fields
# ---------------------------------------------------------------------------


class TestMissingTopLevelFields:
    def test_missing_scenario_block(self, sample_scenario_config_green):
        bad = _deep_del(sample_scenario_config_green, "scenario")
        with pytest.raises(jsonschema.ValidationError, match="'scenario'"):
            validate_scenario(bad)

    def test_missing_project_settings(self, sample_scenario_config_green):
        bad = _deep_del(sample_scenario_config_green, "project_settings")
        with pytest.raises(jsonschema.ValidationError, match="'project_settings'"):
            validate_scenario(bad)

    def test_extra_top_level_key_rejected(self, sample_scenario_config_green):
        bad = copy.deepcopy(sample_scenario_config_green)
        bad["unknown_key"] = "oops"
        with pytest.raises(jsonschema.ValidationError, match="additional"):
            validate_scenario(bad)


# ---------------------------------------------------------------------------
# scenario block
# ---------------------------------------------------------------------------


class TestScenarioBlock:
    def test_missing_name(self, sample_scenario_config_green):
        bad = _deep_del(sample_scenario_config_green, "scenario", "name")
        with pytest.raises(jsonschema.ValidationError, match="name"):
            validate_scenario(bad)

    def test_empty_name_rejected(self, sample_scenario_config_green):
        bad = _deep_set(sample_scenario_config_green, "", "scenario", "name")
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(bad)

    def test_output_missing_directory_rejected(self, sample_scenario_config_green):
        bad = copy.deepcopy(sample_scenario_config_green)
        bad["scenario"]["output"] = {"export_dispatch_sample": True}
        with pytest.raises(jsonschema.ValidationError, match="directory"):
            validate_scenario(bad)


# ---------------------------------------------------------------------------
# project_settings required fields
# ---------------------------------------------------------------------------


class TestProjectSettingsRequired:
    @pytest.mark.parametrize(
        "field_path",
        [
            ("project_settings", "lifetime_years"),
            ("project_settings", "discount_rate"),
            ("project_settings", "operating_mode"),
            ("project_settings", "location"),
            ("project_settings", "technology"),
            ("project_settings", "finance"),
        ],
    )
    def test_missing_required_field(self, sample_scenario_config_green, field_path):
        bad = _deep_del(sample_scenario_config_green, *field_path)
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(bad)


# ---------------------------------------------------------------------------
# operating_mode enum
# ---------------------------------------------------------------------------


class TestOperatingMode:
    def test_green_accepted(self, sample_scenario_config_green):
        validate_scenario(sample_scenario_config_green)

    def test_grey_accepted(self, sample_scenario_config_grey):
        validate_scenario(sample_scenario_config_grey)

    def test_invalid_mode_rejected(self, sample_scenario_config_green):
        bad = _deep_set(
            sample_scenario_config_green, "blue", "project_settings", "operating_mode"
        )
        with pytest.raises(jsonschema.ValidationError, match="'blue'"):
            validate_scenario(bad)

    def test_empty_string_mode_rejected(self, sample_scenario_config_green):
        bad = _deep_set(
            sample_scenario_config_green, "", "project_settings", "operating_mode"
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(bad)


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------


class TestLocation:
    def test_latitude_out_of_range(self, sample_scenario_config_green):
        bad = _deep_set(
            sample_scenario_config_green,
            95.0,
            "project_settings",
            "location",
            "latitude",
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(bad)

    def test_longitude_out_of_range(self, sample_scenario_config_green):
        bad = _deep_set(
            sample_scenario_config_green,
            -200.0,
            "project_settings",
            "location",
            "longitude",
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(bad)

    def test_missing_pvgis_database(self, sample_scenario_config_green):
        bad = _deep_del(
            sample_scenario_config_green,
            "project_settings",
            "location",
            "pvgis_database",
        )
        with pytest.raises(jsonschema.ValidationError, match="pvgis_database"):
            validate_scenario(bad)


# ---------------------------------------------------------------------------
# PV design
# ---------------------------------------------------------------------------


class TestPVDesign:
    def test_zero_peak_power_rejected(self, sample_scenario_config_green):
        bad = _deep_set(
            sample_scenario_config_green,
            0,
            "project_settings",
            "technology",
            "pv",
            "design",
            "peak_power_kwp",
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(bad)

    def test_negative_peak_power_rejected(self, sample_scenario_config_green):
        bad = _deep_set(
            sample_scenario_config_green,
            -100,
            "project_settings",
            "technology",
            "pv",
            "design",
            "peak_power_kwp",
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(bad)

    def test_invalid_mounting_type(self, sample_scenario_config_green):
        bad = _deep_set(
            sample_scenario_config_green,
            "rooftop",
            "project_settings",
            "technology",
            "pv",
            "design",
            "mounting_type",
        )
        with pytest.raises(jsonschema.ValidationError, match="'rooftop'"):
            validate_scenario(bad)

    def test_tilt_over_90_rejected(self, sample_scenario_config_green):
        bad = _deep_set(
            sample_scenario_config_green,
            91,
            "project_settings",
            "technology",
            "pv",
            "design",
            "tilt_deg",
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(bad)

    def test_missing_pv_design_field(self, sample_scenario_config_green):
        bad = _deep_del(
            sample_scenario_config_green,
            "project_settings",
            "technology",
            "pv",
            "design",
            "peak_power_kwp",
        )
        with pytest.raises(jsonschema.ValidationError, match="peak_power_kwp"):
            validate_scenario(bad)


# ---------------------------------------------------------------------------
# BESS
# ---------------------------------------------------------------------------


class TestBESS:
    def test_empty_scale_list_rejected(self, sample_scenario_config_green):
        bad = _deep_set(
            sample_scenario_config_green,
            [],
            "project_settings",
            "technology",
            "bess",
            "design_space",
            "scale_pct_of_pv",
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(bad)

    def test_zero_e_to_p_ratio_rejected(self, sample_scenario_config_green):
        bad = _deep_set(
            sample_scenario_config_green,
            [0],
            "project_settings",
            "technology",
            "bess",
            "design_space",
            "e_to_p_ratio_hours",
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(bad)

    def test_rte_over_100_rejected(self, sample_scenario_config_green):
        bad = _deep_set(
            sample_scenario_config_green,
            101,
            "project_settings",
            "technology",
            "bess",
            "performance",
            "round_trip_efficiency_pct",
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(bad)

    def test_missing_bess_performance_field(self, sample_scenario_config_green):
        bad = _deep_del(
            sample_scenario_config_green,
            "project_settings",
            "technology",
            "bess",
            "performance",
            "bess_availability_pct",
        )
        with pytest.raises(jsonschema.ValidationError, match="bess_availability_pct"):
            validate_scenario(bad)

    def test_soc_min_equals_max_rejected(self, sample_scenario_config_green):
        """min_soc_pct == max_soc_pct must be caught by cross-field check."""
        bad = copy.deepcopy(sample_scenario_config_green)
        bad["project_settings"]["technology"]["bess"]["performance"][
            "min_soc_pct"
        ] = 50.0
        bad["project_settings"]["technology"]["bess"]["performance"][
            "max_soc_pct"
        ] = 50.0
        with pytest.raises(ValueError, match="min_soc_pct"):
            validate_scenario(bad)

    def test_soc_min_greater_than_max_rejected(self, sample_scenario_config_green):
        bad = copy.deepcopy(sample_scenario_config_green)
        bad["project_settings"]["technology"]["bess"]["performance"][
            "min_soc_pct"
        ] = 80.0
        bad["project_settings"]["technology"]["bess"]["performance"][
            "max_soc_pct"
        ] = 20.0
        with pytest.raises(ValueError, match="min_soc_pct"):
            validate_scenario(bad)

    def test_soc_min_less_than_max_passes(self, sample_scenario_config_green):
        cfg = copy.deepcopy(sample_scenario_config_green)
        cfg["project_settings"]["technology"]["bess"]["performance"][
            "min_soc_pct"
        ] = 5.0
        cfg["project_settings"]["technology"]["bess"]["performance"][
            "max_soc_pct"
        ] = 95.0
        validate_scenario(cfg)


# ---------------------------------------------------------------------------
# Cost components
# ---------------------------------------------------------------------------


class TestCostComponents:
    def test_negative_capex_fixed_eur_rejected(self, sample_scenario_config_green):
        bad = _deep_set(
            sample_scenario_config_green,
            -1,
            "project_settings",
            "technology",
            "pv",
            "costs",
            "capex",
            "fixed_eur",
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(bad)

    def test_pct_of_capex_over_1_rejected(self, sample_scenario_config_green):
        bad = _deep_set(
            sample_scenario_config_green,
            1.5,
            "project_settings",
            "technology",
            "pv",
            "costs",
            "opex",
            "pct_of_capex",
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(bad)

    def test_unknown_cost_key_rejected(self, sample_scenario_config_green):
        bad = copy.deepcopy(sample_scenario_config_green)
        bad["project_settings"]["technology"]["pv"]["costs"]["capex"]["typo_key"] = 99
        with pytest.raises(jsonschema.ValidationError, match="additional"):
            validate_scenario(bad)

    def test_all_cost_fields_optional(self, sample_scenario_config_green):
        """An empty cost block (no fields) is valid – all default to 0."""
        cfg = copy.deepcopy(sample_scenario_config_green)
        cfg["project_settings"]["technology"]["pv"]["costs"]["capex"] = {}
        validate_scenario(cfg)


# ---------------------------------------------------------------------------
# Finance – tax
# ---------------------------------------------------------------------------


class TestFinanceTax:
    def test_missing_afa_years_pv(self, sample_scenario_config_green):
        bad = _deep_del(
            sample_scenario_config_green,
            "project_settings",
            "finance",
            "tax",
            "afa_years_pv",
        )
        with pytest.raises(jsonschema.ValidationError, match="afa_years_pv"):
            validate_scenario(bad)

    def test_afa_years_zero_rejected(self, sample_scenario_config_green):
        bad = _deep_set(
            sample_scenario_config_green,
            0,
            "project_settings",
            "finance",
            "tax",
            "afa_years_pv",
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(bad)

    def test_messzahl_over_1_rejected(self, sample_scenario_config_green):
        bad = _deep_set(
            sample_scenario_config_green,
            1.1,
            "project_settings",
            "finance",
            "tax",
            "gewerbesteuer_messzahl",
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(bad)


# ---------------------------------------------------------------------------
# Finance – price_inputs
# ---------------------------------------------------------------------------


class TestPriceInputs:
    def test_invalid_price_unit_rejected(self, sample_scenario_config_green):
        bad = _deep_set(
            sample_scenario_config_green,
            "eur_per_twh",
            "project_settings",
            "finance",
            "price_inputs",
            "price_unit",
        )
        with pytest.raises(jsonschema.ValidationError, match="'eur_per_twh'"):
            validate_scenario(bad)

    def test_missing_day_ahead_csv(self, sample_scenario_config_green):
        bad = _deep_del(
            sample_scenario_config_green,
            "project_settings",
            "finance",
            "price_inputs",
            "day_ahead_csv",
        )
        with pytest.raises(jsonschema.ValidationError, match="day_ahead_csv"):
            validate_scenario(bad)


# ---------------------------------------------------------------------------
# Finance – marketing / PPA enums
# ---------------------------------------------------------------------------


class TestMarketingAndPPA:
    def test_invalid_marketing_type(self, sample_scenario_config_green):
        bad = _deep_set(
            sample_scenario_config_green,
            "unknown",
            "project_settings",
            "finance",
            "revenue_streams",
            "marketing",
            "type",
        )
        with pytest.raises(jsonschema.ValidationError, match="'unknown'"):
            validate_scenario(bad)

    def test_invalid_ppa_type(self, sample_scenario_config_green):
        bad = copy.deepcopy(sample_scenario_config_green)
        bad["project_settings"]["finance"]["revenue_streams"]["ppa"] = {
            "type": "ppa_magic",
        }
        with pytest.raises(jsonschema.ValidationError, match="'ppa_magic'"):
            validate_scenario(bad)

    def test_all_ppa_types_accepted(self, sample_scenario_config_green):
        for ppa_type in (
            "none",
            "ppa_pay_as_produced",
            "ppa_baseload",
            "ppa_floor",
            "ppa_collar",
        ):
            cfg = copy.deepcopy(sample_scenario_config_green)
            cfg["project_settings"]["finance"]["revenue_streams"]["ppa"] = {
                "type": ppa_type
            }
            validate_scenario(cfg)  # must not raise


# ---------------------------------------------------------------------------
# Monte Carlo cross-field weight validation
# ---------------------------------------------------------------------------


class TestMCWeights:
    def _mc_scenario(self, base, weights: dict[str, float]) -> dict:
        cfg = copy.deepcopy(base)
        mc = cfg["scenario"].setdefault("monte_carlo", {})
        mc["enabled"] = True
        mc["price_scenarios"] = {
            k: {"csv_column": k.upper(), "weight": w} for k, w in weights.items()
        }
        return cfg

    def test_weights_sum_to_1_passes(self, sample_scenario_config_green):
        cfg = self._mc_scenario(
            sample_scenario_config_green,
            {"low": 0.25, "mid": 0.50, "high": 0.25},
        )
        validate_scenario(cfg)

    def test_weights_sum_below_1_raises(self, sample_scenario_config_green):
        cfg = self._mc_scenario(
            sample_scenario_config_green,
            {"low": 0.25, "mid": 0.25},
        )
        with pytest.raises(ValueError, match="sum to 1"):
            validate_scenario(cfg)

    def test_weights_sum_above_1_raises(self, sample_scenario_config_green):
        cfg = self._mc_scenario(
            sample_scenario_config_green,
            {"low": 0.5, "mid": 0.5, "high": 0.1},
        )
        with pytest.raises(ValueError, match="sum to 1"):
            validate_scenario(cfg)

    def test_single_weight_1_passes(self, sample_scenario_config_green):
        cfg = self._mc_scenario(sample_scenario_config_green, {"mid": 1.0})
        validate_scenario(cfg)

    def test_weight_negative_rejected_by_schema(self, sample_scenario_config_green):
        cfg = self._mc_scenario(sample_scenario_config_green, {"a": -0.5, "b": 1.5})
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(cfg)


# ---------------------------------------------------------------------------
# Grid connection
# ---------------------------------------------------------------------------


class TestGridConnection:
    def test_missing_max_export_kw(self, sample_scenario_config_green):
        bad = _deep_del(
            sample_scenario_config_green,
            "project_settings",
            "technology",
            "grid_connection",
            "max_export_kw",
        )
        with pytest.raises(jsonschema.ValidationError, match="max_export_kw"):
            validate_scenario(bad)

    def test_zero_max_export_kw_rejected(self, sample_scenario_config_green):
        bad = _deep_set(
            sample_scenario_config_green,
            0,
            "project_settings",
            "technology",
            "grid_connection",
            "max_export_kw",
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(bad)
