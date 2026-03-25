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

    def test_mc_disabled_weight_sum_not_checked_in_mc_block(
        self, sample_scenario_config_green
    ):
        """MC can be disabled without affecting price_inputs.scenarios weight check."""
        cfg = copy.deepcopy(sample_scenario_config_green)
        mc = cfg["scenario"].setdefault("monte_carlo", {})
        mc["enabled"] = False
        validate_scenario(cfg)  # must not raise


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
    def test_zero_peak_power_requires_absolute_bess_values(
        self, sample_scenario_config_green
    ):
        """peak_power_kwp = 0 is allowed by schema but requires absolute BESS sizing
        via the cross-field validator."""
        bad = _deep_set(
            sample_scenario_config_green,
            0,
            "project_settings",
            "technology",
            "pv",
            "design",
            "peak_power_kwp",
        )
        with pytest.raises(ValueError, match="absolute_power_kw"):
            validate_scenario(bad)

    def test_zero_peak_power_with_absolute_values_is_valid(
        self, sample_scenario_config_green
    ):
        """peak_power_kwp = 0 + absolute_power_kw + absolute_capacity_kwh must pass."""
        cfg = _deep_set(
            sample_scenario_config_green,
            0,
            "project_settings",
            "technology",
            "pv",
            "design",
            "peak_power_kwp",
        )
        cfg["project_settings"]["technology"]["bess"]["design_space"][
            "absolute_power_kw"
        ] = 1000.0
        cfg["project_settings"]["technology"]["bess"]["design_space"][
            "absolute_capacity_kwh"
        ] = 2000.0
        validate_scenario(cfg)  # must not raise

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
    def test_missing_scenarios_key_rejected(self, sample_scenario_config_green):
        bad = _deep_del(
            sample_scenario_config_green,
            "project_settings",
            "finance",
            "price_inputs",
            "scenarios",
        )
        with pytest.raises(jsonschema.ValidationError, match="scenarios"):
            validate_scenario(bad)

    def test_empty_scenarios_list_rejected(self, sample_scenario_config_green):
        bad = _deep_set(
            sample_scenario_config_green,
            [],
            "project_settings",
            "finance",
            "price_inputs",
            "scenarios",
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(bad)

    def test_scenario_missing_price_csv_rejected(self, sample_scenario_config_green):
        bad = copy.deepcopy(sample_scenario_config_green)
        scenarios = bad["project_settings"]["finance"]["price_inputs"]["scenarios"]
        del scenarios[0]["price_csv"]
        with pytest.raises(jsonschema.ValidationError, match="price_csv"):
            validate_scenario(bad)

    def test_scenario_missing_csv_column_rejected(self, sample_scenario_config_green):
        bad = copy.deepcopy(sample_scenario_config_green)
        scenarios = bad["project_settings"]["finance"]["price_inputs"]["scenarios"]
        del scenarios[0]["csv_column"]
        with pytest.raises(jsonschema.ValidationError, match="csv_column"):
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
            "ppa_floor",
            "ppa_collar",
        ):
            cfg = copy.deepcopy(sample_scenario_config_green)
            cfg["project_settings"]["finance"]["revenue_streams"]["ppa"] = {
                "type": ppa_type
            }
            validate_scenario(cfg)  # must not raise

    def test_baseload_ppa_with_valid_baseload_mw_accepted(
        self, sample_scenario_config_green
    ):
        cfg = copy.deepcopy(sample_scenario_config_green)
        cfg["project_settings"]["finance"]["revenue_streams"]["ppa"] = {
            "type": "ppa_baseload",
            "baseload_mw": 2.0,
        }
        validate_scenario(cfg)  # must not raise


# ---------------------------------------------------------------------------
# Baseload PPA cross-field validation
# ---------------------------------------------------------------------------


class TestBaseloadPpaValidation:
    """Tests for _validate_baseload_ppa() cross-field checks."""

    def test_baseload_ppa_missing_baseload_mw_raises(
        self, sample_scenario_config_green
    ):
        """baseload_mw=None when type=ppa_baseload must raise."""
        cfg = copy.deepcopy(sample_scenario_config_green)
        cfg["project_settings"]["finance"]["revenue_streams"]["ppa"] = {
            "type": "ppa_baseload",
            "baseload_mw": None,
            "duration_years": 10,
            "inflation_on_ppa": False,
            "guarantee_of_origin_eur_per_kwh": 0.005,
        }
        with pytest.raises(ValueError, match="baseload_mw is required"):
            validate_scenario(cfg)

    def test_baseload_ppa_without_baseload_mw_key_raises(
        self, sample_scenario_config_green
    ):
        """Missing baseload_mw key entirely when type=ppa_baseload must raise."""
        cfg = copy.deepcopy(sample_scenario_config_green)
        cfg["project_settings"]["finance"]["revenue_streams"]["ppa"] = {
            "type": "ppa_baseload",
        }
        with pytest.raises(ValueError, match="baseload_mw is required"):
            validate_scenario(cfg)

    def test_baseload_mw_large_value_accepted(
        self, sample_scenario_config_green
    ):
        """baseload_mw > pv_peak is now accepted by the schema validator."""
        cfg = copy.deepcopy(sample_scenario_config_green)
        cfg["project_settings"]["finance"]["revenue_streams"]["ppa"] = {
            "type": "ppa_baseload",
            "baseload_mw": 6.0,
            "duration_years": 10,
            "inflation_on_ppa": False,
            "guarantee_of_origin_eur_per_kwh": 0.005,
        }
        validate_scenario(cfg)  # must not raise

    def test_baseload_mw_equals_pv_peak_passes(
        self, sample_scenario_config_green
    ):
        """baseload_mw * 1000 == pv_peak_kwp is allowed."""
        cfg = copy.deepcopy(sample_scenario_config_green)
        # PV peak = 5000 kWp, so 5.0 MW = 5000 kW == 5000 kWp
        cfg["project_settings"]["finance"]["revenue_streams"]["ppa"] = {
            "type": "ppa_baseload",
            "baseload_mw": 5.0,
            "duration_years": 10,
            "inflation_on_ppa": False,
            "guarantee_of_origin_eur_per_kwh": 0.005,
        }
        validate_scenario(cfg)  # must not raise

    def test_baseload_mw_below_pv_peak_passes(
        self, sample_scenario_config_green
    ):
        """baseload_mw * 1000 < pv_peak_kwp is allowed."""
        cfg = copy.deepcopy(sample_scenario_config_green)
        cfg["project_settings"]["finance"]["revenue_streams"]["ppa"] = {
            "type": "ppa_baseload",
            "baseload_mw": 2.0,
            "duration_years": 10,
            "inflation_on_ppa": False,
            "guarantee_of_origin_eur_per_kwh": 0.005,
        }
        validate_scenario(cfg)  # must not raise

    def test_non_baseload_ppa_type_skips_validation(
        self, sample_scenario_config_green
    ):
        """Non-baseload PPA types should not trigger baseload validation."""
        for ppa_type in ("none", "ppa_floor", "ppa_collar", "ppa_pay_as_produced"):
            cfg = copy.deepcopy(sample_scenario_config_green)
            cfg["project_settings"]["finance"]["revenue_streams"]["ppa"] = {
                "type": ppa_type,
                "baseload_mw": None,
            }
            validate_scenario(cfg)  # must not raise


# ---------------------------------------------------------------------------
# Monte Carlo cross-field weight validation
# ---------------------------------------------------------------------------


class TestMCWeights:
    """Weights are now on price_inputs.scenarios[*].weight, not MC price_scenarios."""

    def _base_scenario_template(self) -> dict:
        """Return a minimal scenario dict for building weight tests."""
        return {
            "name": "mid",
            "csv_column": "MID",
            "weather_year": 2017,
            "weight": 1.0,
            "is_central": True,
            "price_csv": "data/day_ahead_prices.csv",
            "inflation_on_input_data": True,
            "csv_separator": ";",
            "csv_decimal": ".",
            "csv_timestamp_column": "timestamp",
            "csv_timestamp_format": "ISO8601",
        }

    def _set_scenarios(self, base, scenario_list: list[dict]) -> dict:
        cfg = copy.deepcopy(base)
        cfg["project_settings"]["finance"]["price_inputs"]["scenarios"] = scenario_list
        return cfg

    def test_weights_sum_to_1_passes(self, sample_scenario_config_green):
        tpl = self._base_scenario_template()
        scenarios = [
            {**tpl, "name": "low", "csv_column": "LOW", "weight": 0.25, "is_central": False},
            {**tpl, "name": "mid", "csv_column": "MID", "weight": 0.50, "is_central": True},
            {**tpl, "name": "high", "csv_column": "HIGH", "weight": 0.25, "is_central": False},
        ]
        cfg = self._set_scenarios(sample_scenario_config_green, scenarios)
        validate_scenario(cfg)

    def test_weights_sum_below_1_raises(self, sample_scenario_config_green):
        tpl = self._base_scenario_template()
        scenarios = [
            {**tpl, "name": "low", "csv_column": "LOW", "weight": 0.25, "is_central": False},
            {**tpl, "name": "mid", "csv_column": "MID", "weight": 0.25, "is_central": True},
        ]
        cfg = self._set_scenarios(sample_scenario_config_green, scenarios)
        with pytest.raises(ValueError, match="sum to 1"):
            validate_scenario(cfg)

    def test_weights_sum_above_1_raises(self, sample_scenario_config_green):
        tpl = self._base_scenario_template()
        scenarios = [
            {**tpl, "name": "low", "csv_column": "LOW", "weight": 0.5, "is_central": False},
            {**tpl, "name": "mid", "csv_column": "MID", "weight": 0.5, "is_central": True},
            {**tpl, "name": "high", "csv_column": "HIGH", "weight": 0.1, "is_central": False},
        ]
        cfg = self._set_scenarios(sample_scenario_config_green, scenarios)
        with pytest.raises(ValueError, match="sum to 1"):
            validate_scenario(cfg)

    def test_single_weight_1_passes(self, sample_scenario_config_green):
        tpl = self._base_scenario_template()
        cfg = self._set_scenarios(sample_scenario_config_green, [tpl])
        validate_scenario(cfg)

    def test_weight_negative_rejected_by_schema(self, sample_scenario_config_green):
        tpl = self._base_scenario_template()
        scenarios = [{**tpl, "weight": -0.5}]
        cfg = self._set_scenarios(sample_scenario_config_green, scenarios)
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


# ---------------------------------------------------------------------------
# Grid max_import_kw schema tests
# ---------------------------------------------------------------------------


class TestGridMaxImportKwSchema:
    """Schema validation for the optional ``max_import_kw`` field."""

    def test_grid_connection_with_max_import_kw(self, sample_scenario_config_green):
        """Schema accepts max_import_kw as an optional property."""
        good = _deep_set(
            sample_scenario_config_green,
            2000.0,
            "project_settings",
            "technology",
            "grid_connection",
            "max_import_kw",
        )
        validate_scenario(good)  # should not raise

    def test_grid_connection_without_max_import_kw(self, sample_scenario_config_green):
        """Schema still accepts grid_connection without max_import_kw (backward compat)."""
        validate_scenario(sample_scenario_config_green)  # should not raise

    def test_grid_connection_max_import_kw_zero_rejected(self, sample_scenario_config_green):
        """max_import_kw: 0 must be rejected (exclusiveMinimum)."""
        bad = _deep_set(
            sample_scenario_config_green,
            0,
            "project_settings",
            "technology",
            "grid_connection",
            "max_import_kw",
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(bad)
