"""Tests for portfolio output orchestration in main_portfolio.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from pv_bess_model.config.loader_portfolio import (
    BessFlexConfig,
    GenerationConfig,
    LoadGroupConfig,
    MetaModelConfig,
    PortfolioConfig,
)
from pv_bess_model.main_portfolio import write_output
from pv_bess_model.portfolio.marginal_value import MarginalValuePoint
from pv_bess_model.portfolio.system_value import SystemValuePoint, SystemValueResult


@pytest.fixture
def output_config(tmp_path: Path) -> PortfolioConfig:
    """Portfolio config with output directory pointing to tmp_path."""
    return PortfolioConfig(
        raw={},
        meta=MetaModelConfig(
            name="Test_Out",
            baseline_year=2027,
            project_lifetime_years=3,
            output_directory=str(tmp_path / "output"),
        ),
        generation=[
            GenerationConfig(
                type="pv", name="PV_1", peak_power_kwp=5000,
                latitude=53.87, longitude=10.69,
            ),
        ],
        load=[
            LoadGroupConfig(
                type="slp", name="HH", slp_type="H25",
                customer_count=100,
                annual_consumption_kwh_per_customer=3500,
            ),
        ],
        flexibilities=[
            BessFlexConfig(
                type="bess", name="BESS_1",
                annual_addition_kw=[100.0],
                e_to_p_ratio_hours=[2.0],
            ),
        ],
        price_scenarios=[],
    )


@pytest.fixture
def sv_result() -> SystemValueResult:
    """Sample system value result."""
    return SystemValueResult(
        world_a_annual_costs=[10000.0, 11000.0, 12000.0],
        points=[
            SystemValuePoint(
                flex_name="BESS_1", flex_type="bess",
                annual_addition_kw=100.0, e_to_p_ratio=2.0,
                cumulative_system_value_eur=5000.0,
                annual_system_values=[1500.0, 1700.0, 1800.0],
            ),
        ],
    )


@pytest.fixture
def marginals() -> list[MarginalValuePoint]:
    """Sample marginal values."""
    return [
        MarginalValuePoint(
            flex_name="BESS_1", flex_type="bess",
            annual_addition_kw=100.0, e_to_p_ratio=2.0,
            cumulative_system_value_eur=5000.0,
            marginal_value_eur_per_kw_a=50.0,
            delta_kw=100.0, delta_value_eur=5000.0,
        ),
    ]


class TestWriteOutput:
    """Tests for the write_output orchestration function."""

    def test_creates_csv_files(
        self,
        tmp_path: Path,
        output_config: PortfolioConfig,
        sv_result: SystemValueResult,
        marginals: list[MarginalValuePoint],
    ) -> None:
        """All three CSV files are created."""
        write_output(output_config, sv_result, marginals, generate_report=False)

        out_dir = tmp_path / "output"
        assert (out_dir / "Test_Out_baseline.csv").exists()
        assert (out_dir / "Test_Out_system_value.csv").exists()
        assert (out_dir / "Test_Out_marginal_value.csv").exists()

    def test_creates_html_report(
        self,
        tmp_path: Path,
        output_config: PortfolioConfig,
        sv_result: SystemValueResult,
        marginals: list[MarginalValuePoint],
    ) -> None:
        """HTML report is created when generate_report=True."""
        write_output(output_config, sv_result, marginals, generate_report=True)

        out_dir = tmp_path / "output"
        html_files = list(out_dir.glob("*.html"))
        assert len(html_files) == 1
        assert "Test_Out" in html_files[0].name

    def test_no_report_when_disabled(
        self,
        tmp_path: Path,
        output_config: PortfolioConfig,
        sv_result: SystemValueResult,
        marginals: list[MarginalValuePoint],
    ) -> None:
        """No HTML report when generate_report=False."""
        write_output(output_config, sv_result, marginals, generate_report=False)

        out_dir = tmp_path / "output"
        html_files = list(out_dir.glob("*.html"))
        assert len(html_files) == 0

    def test_output_dir_created(
        self,
        tmp_path: Path,
        output_config: PortfolioConfig,
        sv_result: SystemValueResult,
        marginals: list[MarginalValuePoint],
    ) -> None:
        """Output directory is created automatically."""
        out_dir = tmp_path / "output"
        assert not out_dir.exists()

        write_output(output_config, sv_result, marginals, generate_report=False)
        assert out_dir.exists()
