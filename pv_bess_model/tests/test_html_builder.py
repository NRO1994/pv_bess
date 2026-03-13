"""Tests for pv_bess_model.output.report.html_builder."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pv_bess_model.output.report.html_builder import build_html_report


def _make_mock_data(scenario_name: str = "TestScenario") -> MagicMock:
    """Create a mock HtmlReportData with a valid to_json method."""
    data = MagicMock()
    data.scenario_name = scenario_name
    data.to_json.return_value = json.dumps(
        {"scenario_name": scenario_name, "pv_peak_kwp": 5000},
        separators=(",", ":"),
    )
    return data


class TestBuildHtmlReport:
    def test_creates_html_file(self, tmp_path):
        data = _make_mock_data()
        result = build_html_report(data, tmp_path)
        assert result.exists()
        assert result.suffix == ".html"

    def test_filename_contains_scenario_name(self, tmp_path):
        data = _make_mock_data("MyScenario")
        result = build_html_report(data, tmp_path)
        assert "MyScenario" in result.name

    def test_html_contains_doctype(self, tmp_path):
        data = _make_mock_data()
        result = build_html_report(data, tmp_path)
        content = result.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content

    def test_html_closes_properly(self, tmp_path):
        data = _make_mock_data()
        result = build_html_report(data, tmp_path)
        content = result.read_text(encoding="utf-8")
        assert "</html>" in content

    def test_scenario_name_in_title(self, tmp_path):
        data = _make_mock_data("EEG_Green_Example")
        result = build_html_report(data, tmp_path)
        content = result.read_text(encoding="utf-8")
        assert "EEG_Green_Example" in content
        # Specifically check the <title> tag
        assert "<title>" in content

    def test_json_data_injected(self, tmp_path):
        data = _make_mock_data()
        result = build_html_report(data, tmp_path)
        content = result.read_text(encoding="utf-8")
        # The placeholder should be replaced
        assert "{{REPORT_DATA_JSON}}" not in content
        # The actual JSON should be present
        assert '"scenario_name":"TestScenario"' in content

    def test_creates_output_dir(self, tmp_path):
        nested = tmp_path / "sub" / "dir"
        data = _make_mock_data()
        result = build_html_report(data, nested)
        assert nested.exists()
        assert result.exists()

    def test_overwrites_existing(self, tmp_path):
        data = _make_mock_data()
        path1 = build_html_report(data, tmp_path)
        path1.write_text("old content", encoding="utf-8")
        path2 = build_html_report(data, tmp_path)
        assert path1 == path2
        content = path2.read_text(encoding="utf-8")
        assert "old content" not in content
        assert "<!DOCTYPE html>" in content
