"""Tests for the portfolio HTML report builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from pv_bess_model.output.report.data_collector_portfolio import PortfolioReportData
from pv_bess_model.output.report.html_builder import build_portfolio_html_report


@pytest.fixture
def sample_report_data() -> PortfolioReportData:
    """Minimal report data for testing."""
    return PortfolioReportData(
        scenario_name="Test_Portfolio",
        creation_date="19.03.2026",
        baseline_year=2027,
        lifetime_years=3,
        model_version="1.0",
        perfect_foresight_discount=0.2,
        generation=[{"name": "PV_1", "type": "pv", "peak_kwp": 5000}],
        load_groups=[{"name": "HH", "slp_type": "H25", "customer_count": 100,
                      "annual_kwh_per_customer": 3500, "total_mwh": 350, "growth_factor": 1.0}],
        flexibilities=[{"name": "BESS_1", "type": "bess", "start_year": 1,
                        "annual_addition_kw": [100], "e_to_p_ratio_hours": [2],
                        "rte_pct": 90, "degradation_pct": 2}],
        world_a_annual_costs=[10000, 11000, 12000],
        world_a_total_cost=33000,
        system_value_points=[],
        marginal_value_points=[],
    )


class TestBuildPortfolioHtmlReport:
    """Tests for build_portfolio_html_report."""

    def test_creates_html_file(
        self, tmp_path: Path, sample_report_data: PortfolioReportData
    ) -> None:
        """HTML file is created in the output directory."""
        result_path = build_portfolio_html_report(sample_report_data, tmp_path)
        assert result_path.exists()
        assert result_path.suffix == ".html"

    def test_filename_contains_scenario(
        self, tmp_path: Path, sample_report_data: PortfolioReportData
    ) -> None:
        """Filename includes the scenario name."""
        result_path = build_portfolio_html_report(sample_report_data, tmp_path)
        assert "Test_Portfolio" in result_path.name

    def test_html_contains_data_json(
        self, tmp_path: Path, sample_report_data: PortfolioReportData
    ) -> None:
        """HTML contains the injected JSON data (not the placeholder)."""
        result_path = build_portfolio_html_report(sample_report_data, tmp_path)
        html = result_path.read_text(encoding="utf-8")
        assert "{{REPORT_DATA_JSON}}" not in html
        assert "Test_Portfolio" in html

    def test_title_replaced(
        self, tmp_path: Path, sample_report_data: PortfolioReportData
    ) -> None:
        """<title> tag contains the scenario name."""
        result_path = build_portfolio_html_report(sample_report_data, tmp_path)
        html = result_path.read_text(encoding="utf-8")
        assert "{{scenario_name}}" not in html
        assert "<title>Test_Portfolio" in html

    def test_creates_output_dir(
        self, tmp_path: Path, sample_report_data: PortfolioReportData
    ) -> None:
        """Output directory is created if it doesn't exist."""
        nested = tmp_path / "sub" / "dir"
        result_path = build_portfolio_html_report(sample_report_data, nested)
        assert nested.exists()
        assert result_path.exists()
