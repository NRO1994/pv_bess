"""Tests for pv_bess_model.output.report.data_collector."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from pv_bess_model.config.loader import PriceWeatherScenario, ScenarioConfig
from pv_bess_model.dispatch.engine import AnnualResult, SimulationResult, HourlySample
from pv_bess_model.finance.cashflow import AnnualCashflow, CashflowProjection
from pv_bess_model.finance.metrics import FinancialMetrics
from pv_bess_model.optimization.grid_search import GridPointResult, GridSearchResult
from pv_bess_model.output.report.data_collector import (
    HtmlReportData,
    _compute_price_annual_means,
    _compute_pv_monthly_gwh,
    _encode_logo_b64,
    _extract_cashflow_years,
    _extract_grid_search_points,
    _extract_marketing_params,
    _extract_metrics,
    _extract_sensitivity,
    _sanitize_for_json,
    collect_report_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_scenario_raw(
    *,
    marketing_type: str = "eeg",
    ppa_type: str = "none",
) -> dict:
    """Return a minimal raw scenario dict."""
    return {
        "scenario": {"name": "test_scenario"},
        "project_settings": {
            "lifetime_years": 25,
            "commissioning_year": 2027,
            "operating_mode": "green",
            "location": {
                "latitude": 53.55,
                "longitude": 9.99,
                "pvgis_database": "PVGIS-SARAH2",
            },
            "technology": {
                "pv": {
                    "design": {
                        "peak_power_kwp": 5000,
                        "azimuth_deg": 0,
                        "tilt_deg": 30,
                    },
                    "performance": {
                        "degradation_rate_pct_per_year": 0.4,
                    },
                    "costs": {"capex": {}, "opex": {}},
                },
                "bess": {
                    "design_space": {
                        "scale_pct_of_pv": [0, 20, 40],
                        "e_to_p_ratio_hours": [1, 2],
                    },
                    "performance": {
                        "round_trip_efficiency_pct": 88.0,
                    },
                    "costs": {"capex": {}, "opex": {}},
                },
                "grid_connection": {
                    "max_export_kw": 4000,
                    "costs": {"capex": {}, "opex": {}},
                },
            },
            "finance": {
                "leverage_pct": 75,
                "interest_rate_pct": 4.5,
                "loan_tenor_years": 18,
                "inflation_rate": 0.02,
                "revenue_streams": {
                    "marketing": {
                        "type": marketing_type,
                        "floor_price_eur_per_kwh": 0.0735,
                        "fixed_price_years": 20,
                        "eeg_inflation": False,
                    },
                    "ppa": {
                        "type": ppa_type,
                        "floor_price_eur_per_kwh": 0.06,
                        "cap_price_eur_per_kwh": 0.12,
                        "duration_years": 10,
                        "inflation_on_ppa": False,
                        "guarantee_of_origin_eur_per_kwh": 0.005,
                        "pay_as_produced_price_eur_per_kwh": 0.08,
                        "baseload_mw": 2.0,
                    },
                },
                "price_inputs": {
                    "day_ahead_csv": "prices.csv",
                    "price_unit": "eur_per_mwh",
                },
            },
        },
    }


def _make_scenario(*, marketing_type: str = "eeg", ppa_type: str = "none") -> ScenarioConfig:
    raw = _make_scenario_raw(marketing_type=marketing_type, ppa_type=ppa_type)
    return ScenarioConfig(
        raw=raw,
        name="test_scenario",
        operating_mode="green",
        lifetime_years=25,
        commissioning_year=2027,
        path=Path("/fake/test_scenario.json"),
    )


def _make_metrics() -> FinancialMetrics:
    return FinancialMetrics(
        equity_irr=0.085,
        project_irr=0.065,
        npv=150_000.0,
        dscr_min=1.25,
        dscr_avg=1.45,
        lcoe=0.045,
        payback_year=12,
        annual_dscr=[1.25, 1.30, 1.45],
        capture_rate=0.065,
    )


def _make_annual_cashflow(year: int = 2027, capex: float = 0.0) -> AnnualCashflow:
    return AnnualCashflow(
        year=year,
        revenue=200_000.0,
        grid_import_costs=0.0,
        baseload_matching_costs=0.0,
        opex=30_000.0,
        capex=capex,
        debt_service=50_000.0,
        debt_interest=20_000.0,
        debt_repayment=30_000.0,
        depreciation=25_000.0,
        gewerbesteuer=5_000.0,
        koerperschaftsteuer=8_000.0,
        solidaritaetszuschlag=440.0,
        total_tax=13_440.0,
        project_cf=120_000.0,
        equity_cf=60_000.0,
    )


def _make_annual_result(year: int = 1) -> AnnualResult:
    return AnnualResult(
        year=year,
        revenue_pv_export=150_000.0,
        revenue_bess_green=40_000.0,
        revenue_bess_grey=10_000.0,
        grid_import_cost=0.0,
        missing_baseload_cost=0.0,
        total_revenue=200_000.0,
        pv_production=5_000_000.0,
        pv_export=4_500_000.0,
        pv_curtailed=100_000.0,
        bess_charge_pv=400_000.0,
        bess_charge_grid=0.0,
        bess_discharge_green=350_000.0,
        bess_discharge_grey=50_000.0,
        bess_throughput=800_000.0,
        bess_capacity_kwh=4000.0,
        replacement_cost=0.0,
        bess_spot_revenue=0.0,
    )


def _make_grid_point(*, scale: float = 40.0, optimal: bool = False) -> GridPointResult:
    cf = CashflowProjection(
        years=[_make_annual_cashflow(2027, capex=500_000 if True else 0)],
        equity_cashflows=np.array([-500_000, 60_000]),
        project_cashflows=np.array([-500_000, 120_000]),
    )
    sim = SimulationResult(
        annual_results=[_make_annual_result()],
        hourly_sample=HourlySample(
            pv_production=np.zeros(8760),
            spot_prices=np.zeros(8760),
            effective_prices=np.zeros(8760),
            soc=np.zeros(8760),
            soc_green=np.zeros(8760),
            soc_grey=np.zeros(8760),
            charge_pv=np.zeros(8760),
            charge_grid=np.zeros(8760),
            discharge_green=np.zeros(8760),
            discharge_grey=np.zeros(8760),
            export_pv=np.zeros(8760),
            curtail=np.zeros(8760),
            revenue=np.zeros(8760),
            baseload_shortfall=np.zeros(8760),
        ),
    )
    return GridPointResult(
        scale_pct=scale,
        e_to_p_ratio=2.0,
        bess_power_kw=2000.0,
        bess_capacity_kwh=4000.0,
        capex_total=500_000.0,
        capex_pv=200_000.0,
        capex_bess=200_000.0,
        capex_grid=50_000.0,
        capex_other=50_000.0,
        opex_base=30_000.0,
        opex_pv=10_000.0,
        opex_bess=10_000.0,
        opex_grid=5_000.0,
        opex_other=5_000.0,
        revenue_year1=200_000.0,
        is_optimal=optimal,
        cashflow=cf,
        metrics=_make_metrics(),
        run_result=sim,
    )


# ---------------------------------------------------------------------------
# _sanitize_for_json
# ---------------------------------------------------------------------------

class TestSanitizeForJson:
    def test_nan_replaced_with_none(self):
        assert _sanitize_for_json(float("nan")) is None

    def test_inf_replaced_with_none(self):
        assert _sanitize_for_json(float("inf")) is None
        assert _sanitize_for_json(float("-inf")) is None

    def test_normal_float_preserved(self):
        assert _sanitize_for_json(3.14) == 3.14

    def test_numpy_nan(self):
        assert _sanitize_for_json(np.float64("nan")) is None

    def test_numpy_integer(self):
        assert _sanitize_for_json(np.int64(42)) == 42
        assert isinstance(_sanitize_for_json(np.int64(42)), int)

    def test_numpy_float(self):
        result = _sanitize_for_json(np.float64(2.5))
        assert result == 2.5
        assert isinstance(result, float)

    def test_numpy_array(self):
        arr = np.array([1.0, float("nan"), 3.0])
        result = _sanitize_for_json(arr)
        assert result == [1.0, None, 3.0]

    def test_nested_dict(self):
        data = {"a": float("nan"), "b": {"c": float("inf"), "d": 1.0}}
        result = _sanitize_for_json(data)
        assert result == {"a": None, "b": {"c": None, "d": 1.0}}

    def test_nested_list(self):
        data = [1.0, [float("nan"), 2.0]]
        result = _sanitize_for_json(data)
        assert result == [1.0, [None, 2.0]]

    def test_string_passthrough(self):
        assert _sanitize_for_json("hello") == "hello"

    def test_none_passthrough(self):
        assert _sanitize_for_json(None) is None


# ---------------------------------------------------------------------------
# _encode_logo_b64
# ---------------------------------------------------------------------------

class TestEncodeLogoB64:
    def test_existing_png(self, tmp_path):
        logo = tmp_path / "logo.png"
        logo.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10)
        result = _encode_logo_b64(logo)
        assert result is not None
        assert result.startswith("data:image/png;base64,")

    def test_missing_file_returns_none(self, tmp_path):
        result = _encode_logo_b64(tmp_path / "nonexistent.png")
        assert result is None

    def test_relative_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        logo = tmp_path / "relative_logo.png"
        logo.write_bytes(b"\x89PNG")
        result = _encode_logo_b64(Path("relative_logo.png"))
        assert result is not None


# ---------------------------------------------------------------------------
# _compute_pv_monthly_gwh
# ---------------------------------------------------------------------------

class TestComputePvMonthlyGwh:
    def test_hourly_timeseries(self):
        # 8760 hours, constant 1000 kWh per hour
        ts = {2020: np.ones(8760) * 1000.0}
        result = _compute_pv_monthly_gwh(ts)
        assert 2020 in result
        assert len(result[2020]) == 12
        # January: 31 * 24 * 1000 kWh = 744_000 kWh = 0.000744 GWh
        assert result[2020][0] == pytest.approx(31 * 24 * 1000 / 1e6, rel=1e-6)

    def test_quarter_hourly_timeseries(self):
        # 35040 intervals (= 8760 * 4), constant 250 kWh per interval
        # Energy per hour = 250 * 4 / 4 = 250 kWh (divided by intervals_per_hour)
        ts = {2020: np.ones(35040) * 250.0}
        result = _compute_pv_monthly_gwh(ts)
        assert len(result[2020]) == 12
        # January: 31 * 24 * 250 kWh/h (after /4) = same as 250 kWh/h
        expected_jan = 31 * 24 * 250.0 / 1e6
        assert result[2020][0] == pytest.approx(expected_jan, rel=1e-6)

    def test_multiple_years(self):
        ts = {
            2019: np.ones(8760) * 500.0,
            2020: np.ones(8760) * 1000.0,
        }
        result = _compute_pv_monthly_gwh(ts)
        assert 2019 in result
        assert 2020 in result
        assert result[2019][0] < result[2020][0]


# ---------------------------------------------------------------------------
# _compute_price_annual_means
# ---------------------------------------------------------------------------

class TestComputePriceAnnualMeans:
    def test_basic(self):
        # Price per year: list of arrays, each 8760 values in EUR/kWh
        scenario = PriceWeatherScenario(
            name="mid",
            label="Mid",
            csv_column="MID",
            weather_year=2020,
            weight=1.0,
            price_per_year=[
                np.ones(8760) * 0.05,  # 50 EUR/MWh
                np.ones(8760) * 0.06,  # 60 EUR/MWh
            ],
        )
        result = _compute_price_annual_means([scenario])
        assert len(result) == 1
        assert result[0]["name"] == "Mid"
        assert result[0]["weather_year"] == 2020
        assert len(result[0]["means"]) == 2
        assert result[0]["means"][0] == pytest.approx(50.0, rel=1e-6)
        assert result[0]["means"][1] == pytest.approx(60.0, rel=1e-6)

    def test_no_prices(self):
        scenario = PriceWeatherScenario(
            name="empty",
            label="Empty",
            csv_column="X",
            weather_year=2020,
            weight=1.0,
            price_per_year=None,
        )
        result = _compute_price_annual_means([scenario])
        assert result == []


# ---------------------------------------------------------------------------
# _extract_grid_search_points
# ---------------------------------------------------------------------------

class TestExtractGridSearchPoints:
    def test_basic_extraction(self):
        pt1 = _make_grid_point(scale=0.0)
        pt2 = _make_grid_point(scale=40.0, optimal=True)
        grid_result = GridSearchResult(points=[pt1, pt2], optimal=pt2)
        result = _extract_grid_search_points(grid_result)
        assert len(result) == 2
        assert result[0]["scale_pct"] == 0.0
        assert result[1]["scale_pct"] == 40.0
        assert result[1]["is_optimal"] is True
        assert result[1]["equity_irr"] == pytest.approx(8.5, rel=1e-6)

    def test_none_metrics(self):
        pt = _make_grid_point(scale=0.0)
        pt.metrics = None
        grid_result = GridSearchResult(points=[pt], optimal=None)
        result = _extract_grid_search_points(grid_result)
        assert result[0]["equity_irr"] is None


# ---------------------------------------------------------------------------
# _extract_cashflow_years
# ---------------------------------------------------------------------------

class TestExtractCashflowYears:
    def test_basic(self):
        opt = _make_grid_point(optimal=True)
        result = _extract_cashflow_years(opt, 2027)
        assert len(result) == 1
        y = result[0]
        assert y["year"] == 2027
        assert y["revenue_total"] == 200_000.0
        assert y["capex"] == -500_000.0
        assert y["opex"] == -30_000.0
        assert y["revenue_pv"] == 150_000.0
        assert y["revenue_bess_green"] == 40_000.0
        assert y["revenue_bess_grey"] == 10_000.0
        assert "cumulative_equity_cf" in y

    def test_no_cashflow(self):
        opt = _make_grid_point()
        opt.cashflow = None
        assert _extract_cashflow_years(opt, 2027) == []

    def test_cumulative(self):
        opt = _make_grid_point(optimal=True)
        # Add a second year
        opt.cashflow.years.append(_make_annual_cashflow(2028))
        opt.run_result.annual_results.append(_make_annual_result(year=2))
        result = _extract_cashflow_years(opt, 2027)
        assert len(result) == 2
        assert result[0]["cumulative_equity_cf"] == result[0]["equity_cf"]
        assert result[1]["cumulative_equity_cf"] == pytest.approx(
            result[0]["equity_cf"] + result[1]["equity_cf"]
        )


# ---------------------------------------------------------------------------
# _extract_metrics
# ---------------------------------------------------------------------------

class TestExtractMetrics:
    def test_basic(self):
        m = _make_metrics()
        result = _extract_metrics(m)
        assert result["equity_irr"] == pytest.approx(8.5, rel=1e-6)
        assert result["project_irr"] == pytest.approx(6.5, rel=1e-6)
        assert result["npv"] == 150_000.0
        assert result["dscr_min"] == 1.25
        assert result["lcoe"] == pytest.approx(4.5, rel=1e-6)  # ct/kWh
        assert result["payback_year"] == 12

    def test_none_irr(self):
        m = _make_metrics()
        m.equity_irr = None
        m.project_irr = None
        m.lcoe = None
        m.payback_year = None
        result = _extract_metrics(m)
        assert result["equity_irr"] is None
        assert result["project_irr"] is None
        assert result["lcoe"] is None
        assert result["payback_year"] is None


# ---------------------------------------------------------------------------
# _extract_marketing_params
# ---------------------------------------------------------------------------

class TestExtractMarketingParams:
    def test_eeg(self):
        sc = _make_scenario(marketing_type="eeg")
        result = _extract_marketing_params(sc)
        assert result["floor_price_ct_kwh"] == pytest.approx(7.35, rel=1e-4)
        assert result["fixed_price_years"] == 20

    def test_ppa_floor(self):
        sc = _make_scenario(marketing_type="eeg", ppa_type="ppa_floor")
        result = _extract_marketing_params(sc)
        assert result["ppa_type"] == "ppa_floor"
        assert "floor_price_ct_kwh" in result
        assert result["goo_premium_ct_kwh"] == pytest.approx(0.5, rel=1e-4)

    def test_ppa_collar(self):
        sc = _make_scenario(ppa_type="ppa_collar")
        result = _extract_marketing_params(sc)
        assert "floor_price_ct_kwh" in result
        assert "cap_price_ct_kwh" in result

    def test_ppa_baseload(self):
        sc = _make_scenario(ppa_type="ppa_baseload")
        result = _extract_marketing_params(sc)
        assert result["baseload_mw"] == 2.0

    def test_ppa_pay_as_produced(self):
        sc = _make_scenario(ppa_type="ppa_pay_as_produced")
        result = _extract_marketing_params(sc)
        assert "ppa_price_ct_kwh" in result


# ---------------------------------------------------------------------------
# _extract_sensitivity
# ---------------------------------------------------------------------------

class TestExtractSensitivity:
    def test_none_returns_none(self):
        assert _extract_sensitivity(None) is None

    def test_basic(self):
        from pv_bess_model.optimization.analyses import AnalysisResult, SensitivityResult
        from pv_bess_model.optimization.monte_carlo import MCResult, MCStatistics

        stats = MCStatistics(
            mean=0.08, median=0.082, std=0.01,
            p10=0.065, p25=0.072, p50=0.082, p75=0.09, p90=0.095,
        )
        mc = MCResult(
            iterations=[],
            overall_stats={"equity_irr": stats},
            per_scenario_stats={},
        )
        point = AnalysisResult(
            params={"floor_price_eur_per_kwh": 0.07},
            mc_result=mc,
        )
        sens = SensitivityResult(analysis_type="eeg", points=[point])
        result = _extract_sensitivity(sens)
        assert result is not None
        assert len(result) == 1
        assert result[0]["irr_mean"] == pytest.approx(8.0, rel=1e-6)
        assert result[0]["floor_price_eur_per_kwh"] == 0.07


# ---------------------------------------------------------------------------
# HtmlReportData.to_json
# ---------------------------------------------------------------------------

class TestHtmlReportDataToJson:
    def test_serialisable(self):
        opt = _make_grid_point(optimal=True)
        data = HtmlReportData(
            scenario_name="test",
            scenario_json_filename="test.json",
            creation_date="01.01.2027",
            commissioning_year=2027,
            model_version="0.1.0",
            pv_peak_kwp=5000,
            pv_azimuth=0,
            pv_tilt=30,
            pv_degradation_pct=0.4,
            bess_scale_range=[0, 20, 40],
            bess_ep_ratios=[1, 2],
            bess_rte_pct=88.0,
            grid_max_export_kw=4000,
            operating_mode="green",
            marketing_type="eeg",
            marketing_params={},
            latitude=53.55,
            longitude=9.99,
            lifetime_years=25,
            leverage_pct=75,
            interest_rate_pct=4.5,
            loan_tenor_years=18,
            inflation_rate=0.02,
            pv_monthly_by_year={2020: [0.5] * 12},
            pv_production_model="PVGIS-SARAH2",
            price_scenario_annual_means=[],
            price_origin="Prognos 2026",
            eeg_sensitivity=None,
            ppa_collar=None,
            ppa_baseload=None,
            grid_search_points=[],
            optimal_scale_pct=40,
            optimal_ep_ratio=2.0,
            optimal_bess_power_kw=2000,
            optimal_bess_capacity_kwh=4000,
            cashflow_years=[],
            metrics={"equity_irr": 8.5},
            tool_logo_b64=None,
            company_logo_b64=None,
            ppa_collar_duration=2,
            ppa_baseload_duration=2
        )
        raw = data.to_json()
        parsed = json.loads(raw)
        assert parsed["scenario_name"] == "test"
        assert parsed["pv_peak_kwp"] == 5000

    def test_nan_in_metrics(self):
        data = HtmlReportData(
            scenario_name="nan_test",
            scenario_json_filename="t.json",
            creation_date="01.01.2027",
            commissioning_year=2027,
            model_version="0.1.0",
            pv_peak_kwp=5000,
            pv_azimuth=0,
            pv_tilt=30,
            pv_degradation_pct=0.4,
            bess_scale_range=[],
            bess_ep_ratios=[],
            bess_rte_pct=88.0,
            grid_max_export_kw=4000,
            operating_mode="green",
            marketing_type="eeg",
            marketing_params={},
            latitude=53.55,
            longitude=9.99,
            lifetime_years=25,
            leverage_pct=75,
            interest_rate_pct=4.5,
            loan_tenor_years=18,
            inflation_rate=0.02,
            pv_monthly_by_year={},
            pv_production_model="",
            price_scenario_annual_means=[],
            price_origin="",
            eeg_sensitivity=None,
            ppa_collar=None,
            ppa_baseload=None,
            ppa_collar_duration=2,
            ppa_baseload_duration=2,
            grid_search_points=[],
            optimal_scale_pct=0,
            optimal_ep_ratio=0,
            optimal_bess_power_kw=0,
            optimal_bess_capacity_kwh=0,
            cashflow_years=[],
            metrics={"equity_irr": float("nan"), "npv": float("inf")},
            tool_logo_b64=None,
            company_logo_b64=None,
        )
        raw = data.to_json()
        parsed = json.loads(raw)  # Must not raise
        assert parsed["metrics"]["equity_irr"] is None
        assert parsed["metrics"]["npv"] is None


# ---------------------------------------------------------------------------
# collect_report_data (integration)
# ---------------------------------------------------------------------------

class TestCollectReportData:
    def test_collects_all_fields(self):
        scenario = _make_scenario()
        opt = _make_grid_point(optimal=True)
        metrics = _make_metrics()
        grid_result = GridSearchResult(points=[opt], optimal=opt)

        price_sc = PriceWeatherScenario(
            name="mid", label="Mid", csv_column="MID",
            weather_year=2020, weight=1.0,
            price_per_year=[np.ones(8760) * 0.05],
        )

        with patch(
            "pv_bess_model.output.report.data_collector._encode_logo_b64",
            return_value=None,
        ):
            data = collect_report_data(
                scenario=scenario,
                grid_result=grid_result,
                opt=opt,
                metrics=metrics,
                weather_data_for_report={2020: np.ones(8760) * 1000},
                scenario_prices=[price_sc],
                commissioning_year=2027,
                analyses={"ppa_collar": {"duration_years":1}, "ppa_baseload": {"duration_years":1}},
            )

        assert data.scenario_name == "test_scenario"
        assert data.pv_peak_kwp == 5000
        assert data.latitude == pytest.approx(53.55)
        assert data.operating_mode == "green"
        assert len(data.grid_search_points) == 1
        assert len(data.cashflow_years) == 1
        assert data.metrics["equity_irr"] == pytest.approx(8.5)
        assert data.pv_monthly_by_year is not None
        assert 2020 in data.pv_monthly_by_year

        # Verify JSON serialisability
        json_str = data.to_json()
        parsed = json.loads(json_str)
        assert parsed["scenario_name"] == "test_scenario"
