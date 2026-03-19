"""Tests for the portfolio report data collector."""

from __future__ import annotations

import json

import pytest

from pv_bess_model.config.loader_portfolio import (
    BessFlexConfig,
    EVFlexConfig,
    GenerationConfig,
    HeatPumpFlexConfig,
    LoadGroupConfig,
    MetaModelConfig,
    PortfolioConfig,
)
from pv_bess_model.dispatch.engine_portfolio import PortfolioAnnualResult
from pv_bess_model.output.report.data_collector_portfolio import (
    PortfolioReportData,
    collect_portfolio_report_data,
)
from pv_bess_model.portfolio.marginal_value import MarginalValuePoint
from pv_bess_model.portfolio.system_value import SystemValuePoint, SystemValueResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config() -> PortfolioConfig:
    """Minimal portfolio config for testing."""
    return PortfolioConfig(
        raw={},
        meta=MetaModelConfig(
            name="Test_Portfolio",
            baseline_year=2027,
            project_lifetime_years=3,
            perfect_foresight_discount=0.2,
        ),
        generation=[
            GenerationConfig(
                type="pv",
                name="PV_Nord",
                peak_power_kwp=5000.0,
                latitude=53.87,
                longitude=10.69,
            ),
        ],
        load=[
            LoadGroupConfig(
                type="slp",
                name="Haushalte",
                slp_type="H25",
                customer_count=100,
                annual_consumption_kwh_per_customer=3500.0,
            ),
        ],
        flexibilities=[
            BessFlexConfig(
                type="bess",
                name="BESS_1",
                annual_addition_kw=[100.0, 200.0],
                e_to_p_ratio_hours=[2.0],
            ),
        ],
        price_scenarios=[],
    )


@pytest.fixture
def sample_sv_result() -> SystemValueResult:
    """Sample system value result."""
    return SystemValueResult(
        world_a_annual_costs=[10000.0, 11000.0, 12000.0],
        points=[
            SystemValuePoint(
                flex_name="BESS_1",
                flex_type="bess",
                annual_addition_kw=100.0,
                e_to_p_ratio=2.0,
                cumulative_system_value_eur=5000.0,
                annual_system_values=[1500.0, 1700.0, 1800.0],
            ),
        ],
    )


@pytest.fixture
def sample_marginals() -> list[MarginalValuePoint]:
    """Sample marginal values."""
    return [
        MarginalValuePoint(
            flex_name="BESS_1",
            flex_type="bess",
            annual_addition_kw=100.0,
            e_to_p_ratio=2.0,
            cumulative_system_value_eur=5000.0,
            marginal_value_eur_per_kw_a=50.0,
            delta_kw=100.0,
            delta_value_eur=5000.0,
        ),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPortfolioReportData:
    """Tests for PortfolioReportData."""

    def test_to_json_produces_valid_json(
        self,
        sample_config: PortfolioConfig,
        sample_sv_result: SystemValueResult,
        sample_marginals: list[MarginalValuePoint],
    ) -> None:
        """to_json() produces valid JSON."""
        data = collect_portfolio_report_data(
            sample_config, sample_sv_result, sample_marginals
        )
        json_str = data.to_json()
        parsed = json.loads(json_str)
        assert parsed["scenario_name"] == "Test_Portfolio"

    def test_to_json_handles_nan(self) -> None:
        """NaN/Inf values are replaced with null in JSON output."""
        data = PortfolioReportData(
            scenario_name="test",
            creation_date="01.01.2027",
            baseline_year=2027,
            lifetime_years=3,
            model_version="1.0",
            perfect_foresight_discount=float("nan"),
            generation=[],
            load_groups=[],
            flexibilities=[],
            world_a_annual_costs=[float("inf")],
            world_a_total_cost=float("nan"),
            system_value_points=[],
            marginal_value_points=[],
        )
        json_str = data.to_json()
        parsed = json.loads(json_str)
        assert parsed["perfect_foresight_discount"] is None
        assert parsed["world_a_total_cost"] is None
        assert parsed["world_a_annual_costs"][0] is None


class TestCollectPortfolioReportData:
    """Tests for collect_portfolio_report_data."""

    def test_basic_fields(
        self,
        sample_config: PortfolioConfig,
        sample_sv_result: SystemValueResult,
        sample_marginals: list[MarginalValuePoint],
    ) -> None:
        """Basic fields are populated correctly."""
        data = collect_portfolio_report_data(
            sample_config, sample_sv_result, sample_marginals
        )
        assert data.scenario_name == "Test_Portfolio"
        assert data.baseline_year == 2027
        assert data.lifetime_years == 3
        assert data.perfect_foresight_discount == 0.2

    def test_generation_extracted(
        self,
        sample_config: PortfolioConfig,
        sample_sv_result: SystemValueResult,
        sample_marginals: list[MarginalValuePoint],
    ) -> None:
        """Generation configs are extracted as dicts."""
        data = collect_portfolio_report_data(
            sample_config, sample_sv_result, sample_marginals
        )
        assert len(data.generation) == 1
        assert data.generation[0]["name"] == "PV_Nord"
        assert data.generation[0]["peak_kwp"] == 5000.0

    def test_load_groups_extracted(
        self,
        sample_config: PortfolioConfig,
        sample_sv_result: SystemValueResult,
        sample_marginals: list[MarginalValuePoint],
    ) -> None:
        """Load groups are extracted with computed total_mwh."""
        data = collect_portfolio_report_data(
            sample_config, sample_sv_result, sample_marginals
        )
        assert len(data.load_groups) == 1
        assert data.load_groups[0]["name"] == "Haushalte"
        assert data.load_groups[0]["total_mwh"] == pytest.approx(350.0)

    def test_flexibilities_extracted(
        self,
        sample_config: PortfolioConfig,
        sample_sv_result: SystemValueResult,
        sample_marginals: list[MarginalValuePoint],
    ) -> None:
        """Flex configs are extracted as dicts."""
        data = collect_portfolio_report_data(
            sample_config, sample_sv_result, sample_marginals
        )
        assert len(data.flexibilities) == 1
        assert data.flexibilities[0]["type"] == "bess"
        assert data.flexibilities[0]["annual_addition_kw"] == [100.0, 200.0]

    def test_world_a_costs(
        self,
        sample_config: PortfolioConfig,
        sample_sv_result: SystemValueResult,
        sample_marginals: list[MarginalValuePoint],
    ) -> None:
        """World A costs are copied correctly."""
        data = collect_portfolio_report_data(
            sample_config, sample_sv_result, sample_marginals
        )
        assert data.world_a_annual_costs == [10000.0, 11000.0, 12000.0]
        assert data.world_a_total_cost == 33000.0

    def test_system_value_points(
        self,
        sample_config: PortfolioConfig,
        sample_sv_result: SystemValueResult,
        sample_marginals: list[MarginalValuePoint],
    ) -> None:
        """System value points are extracted."""
        data = collect_portfolio_report_data(
            sample_config, sample_sv_result, sample_marginals
        )
        assert len(data.system_value_points) == 1
        assert data.system_value_points[0]["cumulative_system_value_eur"] == 5000.0

    def test_marginal_value_points(
        self,
        sample_config: PortfolioConfig,
        sample_sv_result: SystemValueResult,
        sample_marginals: list[MarginalValuePoint],
    ) -> None:
        """Marginal value points are extracted."""
        data = collect_portfolio_report_data(
            sample_config, sample_sv_result, sample_marginals
        )
        assert len(data.marginal_value_points) == 1
        assert data.marginal_value_points[0]["marginal_value_eur_per_kw_a"] == 50.0

    def test_dispatch_summary_without_results(
        self,
        sample_config: PortfolioConfig,
        sample_sv_result: SystemValueResult,
        sample_marginals: list[MarginalValuePoint],
    ) -> None:
        """Without annual results, dispatch summary is None."""
        data = collect_portfolio_report_data(
            sample_config, sample_sv_result, sample_marginals
        )
        assert data.dispatch_sample_summary is None
        assert data.dispatch_sample_year is None

    def test_dispatch_summary_with_results(
        self,
        sample_config: PortfolioConfig,
        sample_sv_result: SystemValueResult,
        sample_marginals: list[MarginalValuePoint],
    ) -> None:
        """With annual results, dispatch summary is populated."""
        annual = [
            PortfolioAnnualResult(
                year=1,
                system_cost=5000.0,
                total_grid_sell_kwh=10000.0,
                total_grid_buy_kwh=8000.0,
                total_grid_sell_eur=500.0,
                total_grid_buy_eur=400.0,
                total_bess_throughput_kwh=3000.0,
                bess_capacity_kwh=200.0,
                bess_power_kw=100.0,
                wp_power_kw=50.0,
                total_wp_electrical_kwh=1000.0,
                ev_power_kw=22.0,
                total_ev_charge_kwh=500.0,
                total_ev_discharge_kwh=100.0,
            ),
        ]
        data = collect_portfolio_report_data(
            sample_config, sample_sv_result, sample_marginals,
            annual_results=annual,
        )
        assert data.dispatch_sample_summary is not None
        assert data.dispatch_sample_summary["system_cost_eur"] == 5000.0
        assert data.dispatch_sample_summary["wp_power_kw"] == 50.0
        assert data.dispatch_sample_summary["ev_power_kw"] == 22.0
        assert data.dispatch_sample_year == 2027

    def test_ev_flex_extraction(self) -> None:
        """EV flex configs are extracted correctly."""
        config = PortfolioConfig(
            raw={},
            meta=MetaModelConfig(name="EV_Test", baseline_year=2027),
            generation=[],
            load=[],
            flexibilities=[
                EVFlexConfig(
                    type="ev_charging",
                    name="EV_Fleet",
                    mean_kw_per_unit=11.0,
                    annual_additional_units=[10, 20],
                    daily_energy_demand_kwh_per_unit=8.0,
                    arrival_hour=17,
                    departure_hour=7,
                    v2g_enabled=True,
                    usable_battery_kwh_per_unit=45.0,
                ),
            ],
            price_scenarios=[],
        )
        sv = SystemValueResult(world_a_annual_costs=[], points=[])
        data = collect_portfolio_report_data(config, sv, [])

        assert len(data.flexibilities) == 1
        f = data.flexibilities[0]
        assert f["type"] == "ev_charging"
        assert f["v2g_enabled"] is True
        assert f["annual_additional_units"] == [10, 20]

    def test_heat_pump_flex_extraction(self) -> None:
        """Heat pump flex configs are extracted correctly."""
        config = PortfolioConfig(
            raw={},
            meta=MetaModelConfig(name="HP_Test", baseline_year=2027),
            generation=[],
            load=[],
            flexibilities=[
                HeatPumpFlexConfig(
                    type="heat_pump",
                    name="WP_1",
                    annual_addition_kw=[50.0, 100.0],
                    cop_nominal=3.5,
                    annual_thermal_demand_mwh=1200.0,
                    thermal_storage_kwh=500.0,
                ),
            ],
            price_scenarios=[],
        )
        sv = SystemValueResult(world_a_annual_costs=[], points=[])
        data = collect_portfolio_report_data(config, sv, [])

        assert len(data.flexibilities) == 1
        f = data.flexibilities[0]
        assert f["type"] == "heat_pump"
        assert f["cop_nominal"] == 3.5
        assert f["thermal_demand_mwh"] == 1200.0
