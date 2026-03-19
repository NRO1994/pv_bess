"""Unit tests for pv_bess_model.config.loader_portfolio.

Covers:
- Loading from file and from dict
- Parsing of all dataclass fields
- Correct dispatch to typed flex configs (BESS, heat pump, EV)
- Default values applied when optional fields are missing
- Price scenario reuse from PriceWeatherScenario
- Error handling (file not found, invalid JSON, validation errors)
"""

from __future__ import annotations

import copy
import json
import textwrap

import pytest

from pv_bess_model.config.loader import PriceWeatherScenario
from pv_bess_model.config.loader_portfolio import (
    BessFlexConfig,
    EVFlexConfig,
    GenerationConfig,
    HeatPumpFlexConfig,
    LoadGroupConfig,
    MetaModelConfig,
    PortfolioConfig,
    load_portfolio,
    load_portfolio_dict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_portfolio_dict() -> dict:
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
                    "name": "PV Aggregiert",
                    "peak_power_kwp": 19500,
                    "location": {"latitude": 53.55, "longitude": 9.99},
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
                    "weight": 1.0,
                    "is_central": True,
                    "price_csv": "./inputs/prices.csv",
                    "inflation_on_input_data": False,
                    "csv_separator": ";",
                    "csv_decimal": ",",
                    "csv_timestamp_column": "timestamp",
                    "csv_timestamp_format": "%Y-%m-%dT%H:%M:%S",
                },
            ],
        },
    }


@pytest.fixture
def valid_portfolio_file(tmp_path, valid_portfolio_dict):
    """Write a valid portfolio JSON to a temp file and return its path."""
    path = tmp_path / "test_portfolio.json"
    path.write_text(json.dumps(valid_portfolio_dict, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Loading from file
# ---------------------------------------------------------------------------


class TestLoadPortfolioFile:
    """Test load_portfolio() from file path."""

    def test_load_returns_portfolio_config(self, valid_portfolio_file):
        cfg = load_portfolio(valid_portfolio_file)
        assert isinstance(cfg, PortfolioConfig)
        assert cfg.path is not None

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_portfolio(tmp_path / "nonexistent.json")

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{ invalid json }", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_portfolio(path)


# ---------------------------------------------------------------------------
# Loading from dict
# ---------------------------------------------------------------------------


class TestLoadPortfolioDict:
    """Test load_portfolio_dict() from pre-loaded dict."""

    def test_returns_portfolio_config(self, valid_portfolio_dict):
        cfg = load_portfolio_dict(valid_portfolio_dict)
        assert isinstance(cfg, PortfolioConfig)
        assert cfg.path is None

    def test_raw_preserved(self, valid_portfolio_dict):
        cfg = load_portfolio_dict(valid_portfolio_dict)
        assert cfg.raw == valid_portfolio_dict


# ---------------------------------------------------------------------------
# Meta model parsing
# ---------------------------------------------------------------------------


class TestMetaParsing:
    """MetaModelConfig fields are correctly parsed."""

    def test_meta_fields(self, valid_portfolio_dict):
        cfg = load_portfolio_dict(valid_portfolio_dict)
        meta = cfg.meta
        assert isinstance(meta, MetaModelConfig)
        assert meta.name == "Test_Systemwert"
        assert meta.baseline_year == 2027
        assert meta.project_lifetime_years == 25
        assert meta.perfect_foresight_discount == 0.8
        assert meta.bundesland == "SH"
        assert meta.output_directory == "./outputs/test"
        assert meta.csv_separator == ";"
        assert meta.csv_decimal == ","

    def test_meta_defaults(self, valid_portfolio_dict):
        """Optional fields get default values when not provided."""
        cfg = copy.deepcopy(valid_portfolio_dict)
        del cfg["meta_model"]["perfect_foresight_discount"]
        del cfg["meta_model"]["bundesland"]
        del cfg["meta_model"]["output"]
        result = load_portfolio_dict(cfg)
        assert result.meta.perfect_foresight_discount == 0.8
        assert result.meta.bundesland == "SH"
        assert result.meta.output_directory is None


# ---------------------------------------------------------------------------
# Generation parsing
# ---------------------------------------------------------------------------


class TestGenerationParsing:
    """GenerationConfig fields are correctly parsed."""

    def test_pv_fields(self, valid_portfolio_dict):
        cfg = load_portfolio_dict(valid_portfolio_dict)
        assert len(cfg.generation) == 1
        gen = cfg.generation[0]
        assert isinstance(gen, GenerationConfig)
        assert gen.type == "pv"
        assert gen.name == "PV Aggregiert"
        assert gen.peak_power_kwp == 19500.0
        assert gen.latitude == 53.55
        assert gen.longitude == 9.99
        assert gen.degradation_rate_pct_per_year == 0.4
        assert gen.system_loss_pct == 14.0
        assert gen.mounting_type == "free"
        assert gen.azimuth_deg == 0.0
        assert gen.tilt_deg == 30.0
        assert gen.start_year == 1


# ---------------------------------------------------------------------------
# Load group parsing
# ---------------------------------------------------------------------------


class TestLoadGroupParsing:
    """LoadGroupConfig fields are correctly parsed."""

    def test_slp_fields(self, valid_portfolio_dict):
        cfg = load_portfolio_dict(valid_portfolio_dict)
        assert len(cfg.load) == 1
        lg = cfg.load[0]
        assert isinstance(lg, LoadGroupConfig)
        assert lg.type == "slp"
        assert lg.slp_type == "H25"
        assert lg.customer_count == 8500
        assert lg.annual_consumption_kwh_per_customer == 3200.0
        assert lg.annual_growth_factor == 1.01

    def test_load_growth_default(self, valid_portfolio_dict):
        """annual_growth_factor defaults to 1.0."""
        cfg = copy.deepcopy(valid_portfolio_dict)
        del cfg["portfolio"]["load"][0]["annual_growth_factor"]
        result = load_portfolio_dict(cfg)
        assert result.load[0].annual_growth_factor == 1.0


# ---------------------------------------------------------------------------
# BESS flex parsing
# ---------------------------------------------------------------------------


class TestBessFlexParsing:
    """BessFlexConfig fields are correctly parsed."""

    def test_bess_fields(self, valid_portfolio_dict):
        cfg = load_portfolio_dict(valid_portfolio_dict)
        assert len(cfg.flexibilities) == 1
        flex = cfg.flexibilities[0]
        assert isinstance(flex, BessFlexConfig)
        assert flex.type == "bess"
        assert flex.name == "Grossspeicher"
        assert flex.annual_addition_kw == [0.0, 50.0, 100.0]
        assert flex.e_to_p_ratio_hours == [2.0, 4.0]
        assert flex.round_trip_efficiency_pct == 88.0
        assert flex.min_soc_pct == 10.0
        assert flex.max_soc_pct == 90.0
        assert flex.degradation_rate_pct_per_year == 2.0
        assert flex.start_year == 1

    def test_bess_defaults(self, valid_portfolio_dict):
        """BESS optional fields get defaults from config/defaults.py."""
        cfg = copy.deepcopy(valid_portfolio_dict)
        flex = cfg["flexibilities"][0]
        del flex["round_trip_efficiency_pct"]
        del flex["min_soc_pct"]
        del flex["max_soc_pct"]
        del flex["degradation_rate_pct_per_year"]
        del flex["start_year"]
        result = load_portfolio_dict(cfg)
        bess = result.flexibilities[0]
        assert isinstance(bess, BessFlexConfig)
        assert bess.round_trip_efficiency_pct == 88.0
        assert bess.min_soc_pct == 10.0
        assert bess.max_soc_pct == 90.0  # DEFAULT_BESS_MAX_SOC_PCT
        assert bess.degradation_rate_pct_per_year == 2.0
        assert bess.start_year == 1


# ---------------------------------------------------------------------------
# Heat pump flex parsing
# ---------------------------------------------------------------------------


class TestHeatPumpFlexParsing:
    """HeatPumpFlexConfig fields are correctly parsed."""

    def test_heat_pump_fields(self, valid_portfolio_dict):
        cfg = copy.deepcopy(valid_portfolio_dict)
        cfg["flexibilities"].append({
            "type": "heat_pump",
            "name": "WP Luft-Wasser",
            "annual_addition_kw": [0, 50, 100],
            "cop_nominal": 3.5,
            "cop_reference_temp_c": 7.0,
            "annual_thermal_demand_mwh": 15000,
            "thermal_storage_kwh": 10000,
            "start_year": 2,
        })
        result = load_portfolio_dict(cfg)
        hp = result.flexibilities[1]
        assert isinstance(hp, HeatPumpFlexConfig)
        assert hp.type == "heat_pump"
        assert hp.name == "WP Luft-Wasser"
        assert hp.annual_addition_kw == [0.0, 50.0, 100.0]
        assert hp.cop_nominal == 3.5
        assert hp.cop_reference_temp_c == 7.0
        assert hp.annual_thermal_demand_mwh == 15000.0
        assert hp.thermal_storage_kwh == 10000.0
        assert hp.start_year == 2


# ---------------------------------------------------------------------------
# EV flex parsing
# ---------------------------------------------------------------------------


class TestEVFlexParsing:
    """EVFlexConfig fields are correctly parsed."""

    def test_ev_fields(self, valid_portfolio_dict):
        cfg = copy.deepcopy(valid_portfolio_dict)
        cfg["flexibilities"].append({
            "type": "ev_charging",
            "name": "Wallbox V2G",
            "mean_kw_per_unit": 11.0,
            "annual_additional_units": [0, 5, 10],
            "daily_energy_demand_kwh_per_unit": 30.0,
            "time_window": {"arrival_hour": 17, "departure_hour": 7},
            "v2g_enabled": True,
            "v2g_rte_pct": 90.0,
            "min_departure_soc_pct": 80.0,
            "usable_battery_kwh_per_unit": 33.0,
            "start_year": 1,
        })
        result = load_portfolio_dict(cfg)
        ev = result.flexibilities[1]
        assert isinstance(ev, EVFlexConfig)
        assert ev.type == "ev_charging"
        assert ev.name == "Wallbox V2G"
        assert ev.mean_kw_per_unit == 11.0
        assert ev.annual_additional_units == [0, 5, 10]
        assert ev.daily_energy_demand_kwh_per_unit == 30.0
        assert ev.arrival_hour == 17
        assert ev.departure_hour == 7
        assert ev.v2g_enabled is True
        assert ev.v2g_rte_pct == 90.0
        assert ev.min_departure_soc_pct == 80.0
        assert ev.usable_battery_kwh_per_unit == 33.0

    def test_ev_defaults(self, valid_portfolio_dict):
        """EV optional fields get defaults."""
        cfg = copy.deepcopy(valid_portfolio_dict)
        cfg["flexibilities"].append({
            "type": "ev_charging",
            "name": "EV no V2G",
            "mean_kw_per_unit": 11.0,
            "annual_additional_units": [5],
            "daily_energy_demand_kwh_per_unit": 30.0,
            "time_window": {"arrival_hour": 17, "departure_hour": 7},
            "usable_battery_kwh_per_unit": 33.0,
        })
        result = load_portfolio_dict(cfg)
        ev = result.flexibilities[1]
        assert isinstance(ev, EVFlexConfig)
        assert ev.v2g_enabled is False
        assert ev.v2g_rte_pct == 90.0
        assert ev.min_departure_soc_pct == 80.0
        assert ev.start_year == 1


# ---------------------------------------------------------------------------
# Price scenario parsing
# ---------------------------------------------------------------------------


class TestPriceScenarioParsing:
    """Price scenarios are parsed into PriceWeatherScenario dataclass."""

    def test_scenario_type(self, valid_portfolio_dict):
        cfg = load_portfolio_dict(valid_portfolio_dict)
        assert len(cfg.price_scenarios) == 1
        ps = cfg.price_scenarios[0]
        assert isinstance(ps, PriceWeatherScenario)
        assert ps.name == "Central"
        assert ps.csv_column == "price_central"
        assert ps.weather_year == 2018
        assert ps.weight == 1.0
        assert ps.is_central is True

    def test_multiple_scenarios(self, valid_portfolio_dict):
        cfg = copy.deepcopy(valid_portfolio_dict)
        cfg["price_inputs"]["scenarios"].append({
            "name": "High",
            "csv_column": "price_high",
            "weather_year": 2015,
            "weight": 0.0,
            "is_central": False,
            "price_csv": "./inputs/prices.csv",
            "csv_separator": ";",
            "csv_decimal": ",",
            "csv_timestamp_column": "timestamp",
            "csv_timestamp_format": "%Y-%m-%dT%H:%M:%S",
        })
        # Fix weights to sum to 1.0
        cfg["price_inputs"]["scenarios"][0]["weight"] = 0.6
        cfg["price_inputs"]["scenarios"][1]["weight"] = 0.4
        result = load_portfolio_dict(cfg)
        assert len(result.price_scenarios) == 2
        assert result.price_scenarios[1].name == "High"


# ---------------------------------------------------------------------------
# Mixed flex types
# ---------------------------------------------------------------------------


class TestMixedFlexTypes:
    """Multiple flex types in one config are dispatched correctly."""

    def test_three_flex_types(self, valid_portfolio_dict):
        cfg = copy.deepcopy(valid_portfolio_dict)
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
        result = load_portfolio_dict(cfg)
        assert len(result.flexibilities) == 3
        assert isinstance(result.flexibilities[0], BessFlexConfig)
        assert isinstance(result.flexibilities[1], HeatPumpFlexConfig)
        assert isinstance(result.flexibilities[2], EVFlexConfig)
