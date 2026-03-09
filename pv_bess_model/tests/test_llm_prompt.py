"""Tests for pv_bess_model.output.report.llm_prompt."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pv_bess_model.output.report.llm_prompt import (
    _EXPECTED_KEYS,
    _FALLBACK_TEXT,
    get_fallback_texts,
    load_llm_response,
    render_prompt,
    save_rendered_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_data() -> MagicMock:
    """Create a mock HtmlReportData for prompt rendering."""
    data = MagicMock()
    data.scenario_name = "TestScenario"
    data.creation_date = "01.01.2027"
    data.commissioning_year = 2027
    data.scenario_json_filename = "test.json"
    data.pv_peak_kwp = 5000
    data.pv_azimuth = 0.0
    data.pv_tilt = 30.0
    data.pv_degradation_pct = 0.4
    data.bess_rte_pct = 88.0
    data.grid_max_export_kw = 4000
    data.operating_mode = "green"
    data.latitude = 53.55
    data.longitude = 9.99
    data.lifetime_years = 25
    data.leverage_pct = 75.0
    data.interest_rate_pct = 4.5
    data.loan_tenor_years = 18
    data.inflation_rate = 0.02
    data.marketing_type = "eeg"
    data.marketing_params = {
        "floor_price_ct_kwh": 7.35,
        "fixed_price_years": 20,
    }
    data.optimal_scale_pct = 40.0
    data.optimal_ep_ratio = 2.0
    data.optimal_bess_power_kw = 2000
    data.optimal_bess_capacity_kwh = 4000
    data.grid_search_points = [{"scale_pct": 0}, {"scale_pct": 40}]
    data.metrics = {
        "equity_irr": 8.5,
        "project_irr": 6.5,
        "npv": 150_000.0,
        "dscr_min": 1.25,
        "dscr_avg": 1.45,
        "lcoe": 4.5,
        "payback_year": 12,
    }
    data.pv_monthly_by_year = {2019: [0.5] * 12, 2020: [0.6] * 12}
    data.pv_production_model = "PVGIS-SARAH3"
    data.price_scenario_annual_means = [
        {"name": "Mid", "means": [50.0] * 25},
    ]
    data.price_origin = "Prognos 2026"
    data.eeg_sensitivity = None
    data.ppa_collar = None
    data.ppa_baseload = None
    return data


# ---------------------------------------------------------------------------
# render_prompt
# ---------------------------------------------------------------------------

class TestRenderPrompt:
    def test_placeholders_replaced(self):
        data = _make_mock_data()
        result = render_prompt(data)
        assert "{{scenario_name}}" not in result
        assert "TestScenario" in result
        assert "{{pv_peak_kwp}}" not in result
        assert "5,000" in result  # formatted with comma

    def test_metrics_in_prompt(self):
        data = _make_mock_data()
        result = render_prompt(data)
        assert "8.50" in result  # equity IRR
        assert "150,000" in result  # NPV

    def test_marketing_details_included(self):
        data = _make_mock_data()
        result = render_prompt(data)
        assert "7.35" in result  # floor price
        assert "20 Jahre" in result  # fixed price years

    def test_none_metrics_show_na(self):
        data = _make_mock_data()
        data.metrics = {
            "equity_irr": None,
            "project_irr": None,
            "npv": None,
            "dscr_min": None,
            "dscr_avg": None,
            "lcoe": None,
            "payback_year": None,
        }
        result = render_prompt(data)
        assert "n/a" in result

    def test_sensitivity_section_empty_when_none(self):
        data = _make_mock_data()
        data.eeg_sensitivity = None
        data.ppa_collar = None
        data.ppa_baseload = None
        result = render_prompt(data)
        # The sensitivity section placeholder should be replaced with empty string;
        # check that no actual sensitivity data lines appear (the heading may exist
        # in the template itself, so we check for the data format instead)
        assert "-> IRR" not in result

    def test_sensitivity_section_with_eeg(self):
        data = _make_mock_data()
        data.eeg_sensitivity = [
            {"floor_price_eur_per_kwh": 0.07, "irr_mean": 8.0},
        ]
        result = render_prompt(data)
        assert "EEG-Sensitivitaet" in result

    def test_weather_years(self):
        data = _make_mock_data()
        result = render_prompt(data)
        assert "2019" in result
        assert "2020" in result


# ---------------------------------------------------------------------------
# save_rendered_prompt
# ---------------------------------------------------------------------------

class TestSaveRenderedPrompt:
    def test_saves_file(self, tmp_path):
        data = _make_mock_data()
        result = save_rendered_prompt(data, tmp_path)
        assert result.exists()
        assert "TestScenario" in result.name
        content = result.read_text(encoding="utf-8")
        assert "TestScenario" in content

    def test_creates_directory(self, tmp_path):
        nested = tmp_path / "sub" / "dir"
        data = _make_mock_data()
        result = save_rendered_prompt(data, nested)
        assert nested.exists()
        assert result.exists()


# ---------------------------------------------------------------------------
# load_llm_response
# ---------------------------------------------------------------------------

class TestLoadLlmResponse:
    def test_valid_full_response(self, tmp_path):
        response = {key: f"Text for {key}" for key in _EXPECTED_KEYS}
        path = tmp_path / "response.json"
        path.write_text(json.dumps(response), encoding="utf-8")
        result = load_llm_response(path)
        for key in _EXPECTED_KEYS:
            assert result[key] == f"Text for {key}"

    def test_missing_keys_get_fallback(self, tmp_path):
        response = {"tab_1_overview": "Some text"}
        path = tmp_path / "response.json"
        path.write_text(json.dumps(response), encoding="utf-8")
        result = load_llm_response(path)
        assert result["tab_1_overview"] == "Some text"
        assert result["tab_2_timeseries"] == _FALLBACK_TEXT

    def test_null_value_preserved(self, tmp_path):
        response = {key: f"Text" for key in _EXPECTED_KEYS}
        response["tab_4_eeg"] = None  # Explicitly null
        path = tmp_path / "response.json"
        path.write_text(json.dumps(response), encoding="utf-8")
        result = load_llm_response(path)
        assert result["tab_4_eeg"] is None  # Not fallback text

    def test_non_string_value_gets_fallback(self, tmp_path):
        response = {key: f"Text" for key in _EXPECTED_KEYS}
        response["tab_1_overview"] = 42  # Invalid type
        path = tmp_path / "response.json"
        path.write_text(json.dumps(response), encoding="utf-8")
        result = load_llm_response(path)
        assert result["tab_1_overview"] == _FALLBACK_TEXT

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "response.json"
        path.write_text("not valid json {{{", encoding="utf-8")
        with pytest.raises(ValueError, match="kein gueltiges JSON"):
            load_llm_response(path)

    def test_non_dict_json_raises(self, tmp_path):
        path = tmp_path / "response.json"
        path.write_text(json.dumps(["a", "b"]), encoding="utf-8")
        with pytest.raises(ValueError, match="JSON-Objekt"):
            load_llm_response(path)

    def test_empty_dict_all_fallback(self, tmp_path):
        path = tmp_path / "response.json"
        path.write_text("{}", encoding="utf-8")
        result = load_llm_response(path)
        for key in _EXPECTED_KEYS:
            assert result[key] == _FALLBACK_TEXT


# ---------------------------------------------------------------------------
# get_fallback_texts
# ---------------------------------------------------------------------------

class TestGetFallbackTexts:
    def test_returns_all_keys(self):
        result = get_fallback_texts()
        for key in _EXPECTED_KEYS:
            assert key in result
            assert result[key] == _FALLBACK_TEXT

    def test_returns_new_dict(self):
        a = get_fallback_texts()
        b = get_fallback_texts()
        assert a is not b
