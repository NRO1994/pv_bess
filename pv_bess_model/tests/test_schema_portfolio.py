"""Unit tests for pv_bess_model.config.schema_portfolio.

Covers:
- Valid complete portfolio config passes without error
- Required top-level and nested fields are enforced
- Enum constraints (flex type, slp_type, mounting_type)
- Numeric range constraints (percentages, coordinates)
- Cross-field: MC weight sum validation
- Cross-field: BESS SoC min < max validation
- get_schema() returns the schema dict
"""

from __future__ import annotations

import copy

import jsonschema
import pytest

from pv_bess_model.config.schema_portfolio import get_schema, validate_portfolio


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
# Minimal valid config fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_portfolio_config() -> dict:
    """Return a minimal valid portfolio configuration dictionary."""
    return {
        "meta_model": {
            "name": "Test_Systemwert",
            "baseline_year": 2027,
            "project_lifetime_years": 25,
            "perfect_foresight_discount": 0.8,
            "bundesland": "SH",
            "output": {
                "directory": "./outputs/test",
                "export_dispatch_sample": True,
                "csv_separator": ";",
                "csv_decimal": ",",
            },
        },
        "portfolio": {
            "generation": [
                {
                    "type": "pv",
                    "name": "PV Test",
                    "peak_power_kwp": 19500,
                    "location": {
                        "latitude": 53.55,
                        "longitude": 9.99,
                    },
                    "degradation_rate_pct_per_year": 0.4,
                    "system_loss_pct": 14.0,
                    "mounting_type": "free",
                    "azimuth_deg": 0,
                    "tilt_deg": 30,
                    "start_year": 1,
                },
            ],
            "load": [
                {
                    "type": "slp",
                    "name": "Haushaltskunden",
                    "slp_type": "H25",
                    "customer_count": 8500,
                    "annual_consumption_kwh_per_customer": 3200,
                    "annual_growth_factor": 1.01,
                },
            ],
        },
        "flexibilities": [
            {
                "type": "bess",
                "name": "Grossspeicher",
                "annual_addition_kw": [0, 50, 100],
                "e_to_p_ratio_hours": [2, 4],
                "round_trip_efficiency_pct": 88.0,
                "min_soc_pct": 10.0,
                "max_soc_pct": 90.0,
                "degradation_rate_pct_per_year": 2.0,
                "start_year": 1,
            },
        ],
        "price_inputs": {
            "scenarios": [
                {
                    "name": "Central",
                    "label": "Zentralszenario",
                    "csv_column": "price_central",
                    "weather_year": 2018,
                    "weight": 0.6,
                    "is_central": True,
                    "price_csv": "./inputs/day_ahead_prices.csv",
                    "inflation_on_input_data": False,
                    "csv_separator": ";",
                    "csv_decimal": ",",
                    "csv_timestamp_column": "timestamp",
                    "csv_timestamp_format": "%Y-%m-%dT%H:%M:%S",
                },
                {
                    "name": "High",
                    "label": "Hohes Preisniveau",
                    "csv_column": "price_high",
                    "weather_year": 2015,
                    "weight": 0.4,
                    "is_central": False,
                    "price_csv": "./inputs/day_ahead_prices.csv",
                    "inflation_on_input_data": False,
                    "csv_separator": ";",
                    "csv_decimal": ",",
                    "csv_timestamp_column": "timestamp",
                    "csv_timestamp_format": "%Y-%m-%dT%H:%M:%S",
                },
            ],
        },
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestValidPortfolio:
    """A fully valid portfolio config must pass without raising."""

    def test_valid_config_passes(self, valid_portfolio_config):
        validate_portfolio(valid_portfolio_config)

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

    def test_valid_with_heat_pump(self, valid_portfolio_config):
        cfg = copy.deepcopy(valid_portfolio_config)
        cfg["flexibilities"].append({
            "type": "heat_pump",
            "name": "WP Test",
            "annual_addition_kw": [0, 50, 100],
            "cop_nominal": 3.5,
            "cop_reference_temp_c": 7.0,
            "annual_thermal_demand_mwh": 15000,
            "thermal_storage_kwh": 10000,
            "start_year": 1,
        })
        validate_portfolio(cfg)

    def test_valid_with_ev_charging(self, valid_portfolio_config):
        cfg = copy.deepcopy(valid_portfolio_config)
        cfg["flexibilities"].append({
            "type": "ev_charging",
            "name": "EV Test",
            "mean_kw_per_unit": 11.0,
            "annual_additional_units": [0, 5, 10],
            "daily_energy_demand_kwh_per_unit": 30.0,
            "time_window": {
                "arrival_hour": 17,
                "departure_hour": 7,
            },
            "v2g_enabled": True,
            "v2g_rte_pct": 90.0,
            "min_departure_soc_pct": 80.0,
            "usable_battery_kwh_per_unit": 33.0,
            "start_year": 1,
        })
        validate_portfolio(cfg)

    def test_valid_with_all_flex_types(self, valid_portfolio_config):
        """Config with BESS, heat pump, and EV together passes."""
        cfg = copy.deepcopy(valid_portfolio_config)
        cfg["flexibilities"].extend([
            {
                "type": "heat_pump",
                "name": "WP",
                "annual_addition_kw": [100],
                "cop_nominal": 4.0,
                "annual_thermal_demand_mwh": 5000,
                "thermal_storage_kwh": 2000,
            },
            {
                "type": "ev_charging",
                "name": "EV",
                "mean_kw_per_unit": 11.0,
                "annual_additional_units": [5],
                "daily_energy_demand_kwh_per_unit": 30.0,
                "time_window": {"arrival_hour": 17, "departure_hour": 7},
                "usable_battery_kwh_per_unit": 33.0,
            },
        ])
        validate_portfolio(cfg)

    def test_optional_fields_omitted(self, valid_portfolio_config):
        """Optional fields can be omitted from meta_model."""
        cfg = copy.deepcopy(valid_portfolio_config)
        del cfg["meta_model"]["perfect_foresight_discount"]
        del cfg["meta_model"]["bundesland"]
        del cfg["meta_model"]["output"]
        validate_portfolio(cfg)


# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------


class TestRequiredFields:
    """Missing required fields must cause validation errors."""

    @pytest.mark.parametrize("key", [
        "meta_model", "portfolio", "flexibilities", "price_inputs",
    ])
    def test_missing_top_level_key(self, valid_portfolio_config, key):
        cfg = _deep_del(valid_portfolio_config, key)
        with pytest.raises(jsonschema.ValidationError):
            validate_portfolio(cfg)

    def test_missing_meta_name(self, valid_portfolio_config):
        cfg = _deep_del(valid_portfolio_config, "meta_model", "name")
        with pytest.raises(jsonschema.ValidationError):
            validate_portfolio(cfg)

    def test_missing_baseline_year(self, valid_portfolio_config):
        cfg = _deep_del(valid_portfolio_config, "meta_model", "baseline_year")
        with pytest.raises(jsonschema.ValidationError):
            validate_portfolio(cfg)

    def test_missing_generation(self, valid_portfolio_config):
        cfg = _deep_del(valid_portfolio_config, "portfolio", "generation")
        with pytest.raises(jsonschema.ValidationError):
            validate_portfolio(cfg)

    def test_missing_load(self, valid_portfolio_config):
        cfg = _deep_del(valid_portfolio_config, "portfolio", "load")
        with pytest.raises(jsonschema.ValidationError):
            validate_portfolio(cfg)

    def test_missing_pv_peak_power(self, valid_portfolio_config):
        cfg = _deep_del(
            valid_portfolio_config,
            "portfolio", "generation", 0, "peak_power_kwp",
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_portfolio(cfg)

    def test_missing_slp_type(self, valid_portfolio_config):
        cfg = _deep_del(
            valid_portfolio_config,
            "portfolio", "load", 0, "slp_type",
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_portfolio(cfg)

    def test_missing_bess_e_to_p(self, valid_portfolio_config):
        cfg = _deep_del(
            valid_portfolio_config,
            "flexibilities", 0, "e_to_p_ratio_hours",
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_portfolio(cfg)

    def test_missing_price_scenarios(self, valid_portfolio_config):
        cfg = _deep_del(valid_portfolio_config, "price_inputs", "scenarios")
        with pytest.raises(jsonschema.ValidationError):
            validate_portfolio(cfg)


# ---------------------------------------------------------------------------
# Enum constraints
# ---------------------------------------------------------------------------


class TestEnumConstraints:
    """Enum fields must only accept valid values."""

    def test_invalid_generation_type(self, valid_portfolio_config):
        cfg = _deep_set(
            valid_portfolio_config, "wind",
            "portfolio", "generation", 0, "type",
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_portfolio(cfg)

    def test_invalid_slp_type(self, valid_portfolio_config):
        cfg = _deep_set(
            valid_portfolio_config, "X99",
            "portfolio", "load", 0, "slp_type",
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_portfolio(cfg)

    def test_invalid_flex_type(self, valid_portfolio_config):
        cfg = copy.deepcopy(valid_portfolio_config)
        cfg["flexibilities"][0]["type"] = "unknown_flex"
        with pytest.raises(jsonschema.ValidationError):
            validate_portfolio(cfg)

    def test_invalid_mounting_type(self, valid_portfolio_config):
        cfg = _deep_set(
            valid_portfolio_config, "floating",
            "portfolio", "generation", 0, "mounting_type",
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_portfolio(cfg)


# ---------------------------------------------------------------------------
# Numeric range constraints
# ---------------------------------------------------------------------------


class TestRangeConstraints:
    """Numeric fields must stay within valid ranges."""

    def test_negative_peak_power(self, valid_portfolio_config):
        cfg = _deep_set(
            valid_portfolio_config, -100,
            "portfolio", "generation", 0, "peak_power_kwp",
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_portfolio(cfg)

    def test_latitude_out_of_range(self, valid_portfolio_config):
        cfg = _deep_set(
            valid_portfolio_config, 100.0,
            "portfolio", "generation", 0, "location", "latitude",
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_portfolio(cfg)

    def test_foresight_discount_over_1(self, valid_portfolio_config):
        cfg = _deep_set(
            valid_portfolio_config, 1.5,
            "meta_model", "perfect_foresight_discount",
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_portfolio(cfg)

    def test_negative_customer_count(self, valid_portfolio_config):
        cfg = _deep_set(
            valid_portfolio_config, 0,
            "portfolio", "load", 0, "customer_count",
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_portfolio(cfg)

    def test_bess_rte_over_100(self, valid_portfolio_config):
        cfg = _deep_set(
            valid_portfolio_config, 105.0,
            "flexibilities", 0, "round_trip_efficiency_pct",
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_portfolio(cfg)

    def test_baseline_year_too_old(self, valid_portfolio_config):
        cfg = _deep_set(valid_portfolio_config, 2019, "meta_model", "baseline_year")
        with pytest.raises(jsonschema.ValidationError):
            validate_portfolio(cfg)


# ---------------------------------------------------------------------------
# Cross-field: scenario weights
# ---------------------------------------------------------------------------


class TestScenarioValidation:
    """Cross-field scenario validations."""

    def test_weights_not_summing_to_1(self, valid_portfolio_config):
        cfg = _deep_set(
            valid_portfolio_config, 0.3,
            "price_inputs", "scenarios", 1, "weight",
        )
        with pytest.raises(ValueError, match="weights must sum to 1.0"):
            validate_portfolio(cfg)

    def test_no_central_scenario(self, valid_portfolio_config):
        cfg = _deep_set(
            valid_portfolio_config, False,
            "price_inputs", "scenarios", 0, "is_central",
        )
        with pytest.raises(ValueError, match="is_central"):
            validate_portfolio(cfg)

    def test_two_central_scenarios(self, valid_portfolio_config):
        cfg = _deep_set(
            valid_portfolio_config, True,
            "price_inputs", "scenarios", 1, "is_central",
        )
        with pytest.raises(ValueError, match="is_central"):
            validate_portfolio(cfg)


# ---------------------------------------------------------------------------
# Cross-field: BESS SoC limits
# ---------------------------------------------------------------------------


class TestBessSocLimits:
    """BESS min_soc_pct must be strictly less than max_soc_pct."""

    def test_min_soc_equals_max_soc(self, valid_portfolio_config):
        cfg = _deep_set(valid_portfolio_config, 50.0, "flexibilities", 0, "min_soc_pct")
        cfg = _deep_set(cfg, 50.0, "flexibilities", 0, "max_soc_pct")
        with pytest.raises(ValueError, match="min_soc_pct"):
            validate_portfolio(cfg)

    def test_min_soc_greater_than_max_soc(self, valid_portfolio_config):
        cfg = _deep_set(valid_portfolio_config, 90.0, "flexibilities", 0, "min_soc_pct")
        cfg = _deep_set(cfg, 10.0, "flexibilities", 0, "max_soc_pct")
        with pytest.raises(ValueError, match="min_soc_pct"):
            validate_portfolio(cfg)


# ---------------------------------------------------------------------------
# Additional properties rejected
# ---------------------------------------------------------------------------


class TestAdditionalProperties:
    """Unknown keys must cause validation errors (additionalProperties: false)."""

    def test_extra_top_level_key(self, valid_portfolio_config):
        cfg = copy.deepcopy(valid_portfolio_config)
        cfg["unknown_key"] = "value"
        with pytest.raises(jsonschema.ValidationError):
            validate_portfolio(cfg)

    def test_extra_meta_key(self, valid_portfolio_config):
        cfg = copy.deepcopy(valid_portfolio_config)
        cfg["meta_model"]["extra"] = 42
        with pytest.raises(jsonschema.ValidationError):
            validate_portfolio(cfg)


# ---------------------------------------------------------------------------
# Empty arrays
# ---------------------------------------------------------------------------


class TestEmptyArrays:
    """Arrays with minItems=1 must reject empty lists."""

    def test_empty_generation(self, valid_portfolio_config):
        cfg = _deep_set(valid_portfolio_config, [], "portfolio", "generation")
        with pytest.raises(jsonschema.ValidationError):
            validate_portfolio(cfg)

    def test_empty_load(self, valid_portfolio_config):
        cfg = _deep_set(valid_portfolio_config, [], "portfolio", "load")
        with pytest.raises(jsonschema.ValidationError):
            validate_portfolio(cfg)

    def test_empty_flexibilities(self, valid_portfolio_config):
        cfg = _deep_set(valid_portfolio_config, [], "flexibilities")
        with pytest.raises(jsonschema.ValidationError):
            validate_portfolio(cfg)

    def test_empty_annual_addition(self, valid_portfolio_config):
        cfg = _deep_set(valid_portfolio_config, [], "flexibilities", 0, "annual_addition_kw")
        with pytest.raises(jsonschema.ValidationError):
            validate_portfolio(cfg)

    def test_empty_scenarios(self, valid_portfolio_config):
        cfg = _deep_set(valid_portfolio_config, [], "price_inputs", "scenarios")
        with pytest.raises(jsonschema.ValidationError):
            validate_portfolio(cfg)
