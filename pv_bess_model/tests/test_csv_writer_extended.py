"""Tests for output/csv_writer.py – summary, grid search, MC, dispatch sample CSVs.

Builds minimal mock data objects and verifies that the CSV writers produce
correct files with the expected columns and row counts.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from pv_bess_model.config.defaults import CSV_DELIMITER
from pv_bess_model.dispatch.engine import HourlySample
from pv_bess_model.finance.cashflow import AnnualCashflow, CashflowProjection
from pv_bess_model.finance.metrics import FinancialMetrics
from pv_bess_model.optimization.grid_search import GridPointResult, GridSearchResult
from pv_bess_model.optimization.monte_carlo import MCIterationResult, MCResult, MCStatistics
from pv_bess_model.optimization.analyses import AnalysisResult, SensitivityResult
from pv_bess_model.output.csv_writer import (
    CsvConfig,
    _write_dicts,
    write_dispatch_sample_csv,
    write_eeg_sensitivity_csv,
    write_grid_search_csv,
    write_monte_carlo_csv,
    write_ppa_baseload_csv,
    write_ppa_collar_csv,
    write_summary_csv,
)


# ---------------------------------------------------------------------------
# Mock data builders
# ---------------------------------------------------------------------------


def _make_cashflow(n_years: int = 3) -> CashflowProjection:
    years = []
    for y in range(1, n_years + 1):
        capex = 100_000.0 if y == 1 else 0.0
        revenue = 30_000.0
        opex = 5_000.0
        debt_interest = 2_000.0
        debt_repayment = 3_000.0
        debt_service = debt_interest + debt_repayment
        depreciation = 5_000.0
        gwst = 500.0
        kst = 300.0
        soli = 16.5
        total_tax = gwst + kst + soli
        project_cf = revenue - opex - capex
        equity_cf = project_cf - debt_service - total_tax
        years.append(AnnualCashflow(
            year=y,
            capex=capex,
            revenue=revenue,
            opex=opex,
            grid_import_costs=0.0,
            baseload_matching_costs=0.0,
            debt_service=debt_service,
            debt_interest=debt_interest,
            debt_repayment=debt_repayment,
            depreciation=depreciation,
            gewerbesteuer=gwst,
            koerperschaftsteuer=kst,
            solidaritaetszuschlag=soli,
            total_tax=total_tax,
            project_cf=project_cf,
            equity_cf=equity_cf,
        ))
    equity_cfs = np.array([y.equity_cf for y in years])
    project_cfs = np.array([y.project_cf for y in years])
    return CashflowProjection(
        years=years,
        equity_cashflows=equity_cfs,
        project_cashflows=project_cfs,
    )


def _make_metrics() -> FinancialMetrics:
    return FinancialMetrics(
        equity_irr=0.08,
        project_irr=0.06,
        npv=50_000.0,
        dscr_min=1.2,
        dscr_avg=1.5,
        lcoe=0.04,
        payback_year=5,
        annual_dscr=np.array([1.2, 1.3, 1.4]),
        capture_rate=0.065,
    )


def _make_grid_point(scale: float = 50.0, optimal: bool = True) -> GridPointResult:
    return GridPointResult(
        scale_pct=scale,
        e_to_p_ratio=2.0,
        bess_power_kw=500.0,
        bess_capacity_kwh=1000.0,
        capex_total=200_000.0,
        capex_pv=100_000.0,
        capex_bess=80_000.0,
        capex_grid=20_000.0,
        capex_other=0.0,
        opex_base=10_000.0,
        opex_pv=4_000.0,
        opex_bess=3_000.0,
        opex_grid=3_000.0,
        opex_other=0.0,
        revenue_year1=30_000.0,
        is_optimal=optimal,
        metrics=_make_metrics(),
    )


def _make_grid_result() -> GridSearchResult:
    pv_only = _make_grid_point(scale=0.0, optimal=False)
    bess = _make_grid_point(scale=50.0, optimal=True)
    return GridSearchResult(points=[pv_only, bess], optimal=bess)


def _make_mc_result(n_iterations: int = 5) -> MCResult:
    iterations = []
    for i in range(1, n_iterations + 1):
        iterations.append(MCIterationResult(
            iteration=i,
            price_scenario="mid",
            capex_factor_pv=1.0,
            capex_factor_bess=1.0,
            opex_factor_pv=1.0,
            opex_factor_bess=1.0,
            pv_availability_factor=1.0,
            bess_availability_factor=0.97,
            equity_irr=0.08 + i * 0.001,
            project_irr=0.06,
            npv=50_000.0,
            dscr_min=1.2,
            capture_rate=0.065,
            fixed_price_years=20,
            analysis_label="EEG-Sensitivität",
        ))

    irrs = np.array([it.equity_irr for it in iterations])
    overall_stats = {
        "equity_irr": MCStatistics(
            mean=float(np.mean(irrs)),
            median=float(np.median(irrs)),
            std=float(np.std(irrs)),
            p10=float(np.percentile(irrs, 10)),
            p25=float(np.percentile(irrs, 25)),
            p50=float(np.percentile(irrs, 50)),
            p75=float(np.percentile(irrs, 75)),
            p90=float(np.percentile(irrs, 90)),
        ),
        "project_irr": MCStatistics(
            mean=0.06, median=0.06, std=0.0,
            p10=0.06, p25=0.06, p50=0.06, p75=0.06, p90=0.06,
        ),
        "npv": MCStatistics(
            mean=50_000.0, median=50_000.0, std=0.0,
            p10=50_000.0, p25=50_000.0, p50=50_000.0, p75=50_000.0, p90=50_000.0,
        ),
        "dscr_min": MCStatistics(
            mean=1.2, median=1.2, std=0.0,
            p10=1.2, p25=1.2, p50=1.2, p75=1.2, p90=1.2,
        ),
    }

    return MCResult(
        iterations=iterations,
        overall_stats=overall_stats,
        per_scenario_stats={"mid": overall_stats},
    )


def _make_hourly_sample(n: int = 24) -> HourlySample:
    return HourlySample(
        pv_production=np.ones(n) * 10.0,
        spot_prices=np.ones(n) * 0.05,
        effective_prices=np.ones(n) * 0.06,
        soc=np.ones(n) * 50.0,
        soc_green=np.ones(n) * 50.0,
        soc_grey=np.zeros(n),
        charge_pv=np.ones(n) * 2.0,
        charge_grid=np.zeros(n),
        discharge_green=np.ones(n) * 1.0,
        discharge_grey=np.zeros(n),
        export_pv=np.ones(n) * 7.0,
        curtail=np.zeros(n),
        revenue=np.ones(n) * 0.5,
        baseload_shortfall=np.zeros(n),
    )


def _read_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=CSV_DELIMITER)
        return list(reader)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWriteSummaryCsv:
    def test_produces_file_with_correct_columns(self, tmp_path: Path) -> None:
        p = tmp_path / "summary.csv"
        write_summary_csv(
            path=p,
            scenario_name="test",
            pv_peak_kwp=5000.0,
            operating_mode="green",
            marketing_type="eeg",
            lifetime_years=25,
            grid_result=_make_grid_result(),
            cashflow=_make_cashflow(),
            equity_irr=0.08,
            project_irr=0.06,
            npv=50_000.0,
            dscr_min=1.2,
            dscr_avg=1.5,
            lcoe=0.04,
            payback_year=5,
            total_production_kwh=1_000_000.0,
        )
        rows = _read_csv(p)
        assert len(rows) == 1
        assert "scenario_name" in rows[0]
        assert "equity_irr_pct" in rows[0]
        assert "total_capex_eur" in rows[0]
        assert rows[0]["scenario_name"] == "test"
        assert rows[0]["operating_mode"] == "green"

    def test_none_values_produce_empty_strings(self, tmp_path: Path) -> None:
        p = tmp_path / "summary_none.csv"
        write_summary_csv(
            path=p,
            scenario_name="test",
            pv_peak_kwp=5000.0,
            operating_mode="green",
            marketing_type="eeg",
            lifetime_years=25,
            grid_result=_make_grid_result(),
            cashflow=_make_cashflow(),
            equity_irr=None,
            project_irr=None,
            npv=0.0,
            dscr_min=None,
            dscr_avg=None,
            lcoe=None,
            payback_year=None,
            total_production_kwh=0.0,
        )
        rows = _read_csv(p)
        assert rows[0]["equity_irr_pct"] == ""
        assert rows[0]["payback_year"] == ""
        assert rows[0]["lcoe_eur_per_kwh"] == ""


class TestWriteGridSearchCsv:
    def test_produces_correct_row_count(self, tmp_path: Path) -> None:
        p = tmp_path / "grid.csv"
        grid_result = _make_grid_result()
        write_grid_search_csv(p, grid_result)
        rows = _read_csv(p)
        assert len(rows) == 2  # PV-only + BESS point

    def test_columns_present(self, tmp_path: Path) -> None:
        p = tmp_path / "grid.csv"
        write_grid_search_csv(p, _make_grid_result())
        rows = _read_csv(p)
        expected = {
            "scale_pct_of_pv", "e_to_p_ratio_h", "bess_power_kw",
            "bess_capacity_kwh", "capex_total_eur", "equity_irr_pct",
            "is_optimal",
        }
        assert expected.issubset(set(rows[0].keys()))

    def test_optimal_flag(self, tmp_path: Path) -> None:
        p = tmp_path / "grid.csv"
        write_grid_search_csv(p, _make_grid_result())
        rows = _read_csv(p)
        optimal_flags = [r["is_optimal"] for r in rows]
        assert optimal_flags.count("True") == 1


class TestWriteMonteCarloCsv:
    def test_produces_correct_row_count(self, tmp_path: Path) -> None:
        p = tmp_path / "mc.csv"
        mc = _make_mc_result(n_iterations=10)
        write_monte_carlo_csv(p, mc)
        rows = _read_csv(p)
        assert len(rows) == 10

    def test_columns_present(self, tmp_path: Path) -> None:
        p = tmp_path / "mc.csv"
        write_monte_carlo_csv(p, _make_mc_result(3))
        rows = _read_csv(p)
        expected = {
            "iteration", "price_scenario", "capex_factor_pv",
            "equity_irr_pct", "npv_eur", "dscr_min",
        }
        assert expected.issubset(set(rows[0].keys()))

    def test_iteration_numbers(self, tmp_path: Path) -> None:
        p = tmp_path / "mc.csv"
        write_monte_carlo_csv(p, _make_mc_result(5))
        rows = _read_csv(p)
        iterations = [r["iteration"] for r in rows]
        assert iterations == ["1", "2", "3", "4", "5"]


class TestWriteDispatchSampleCsv:
    def test_hourly_resolution(self, tmp_path: Path) -> None:
        from pv_bess_model.config.defaults import HOURS_PER_YEAR
        p = tmp_path / "dispatch.csv"
        sample = _make_hourly_sample(HOURS_PER_YEAR)
        write_dispatch_sample_csv(p, sample, start_year=2027)
        rows = _read_csv(p)
        assert len(rows) == HOURS_PER_YEAR

    def test_quarter_hourly_resolution(self, tmp_path: Path) -> None:
        from pv_bess_model.config.defaults import INTERVALS_PER_YEAR
        p = tmp_path / "dispatch_15min.csv"
        sample = _make_hourly_sample(INTERVALS_PER_YEAR)
        write_dispatch_sample_csv(p, sample, start_year=2027)
        rows = _read_csv(p)
        assert len(rows) == INTERVALS_PER_YEAR

    def test_columns_present(self, tmp_path: Path) -> None:
        p = tmp_path / "dispatch.csv"
        sample = _make_hourly_sample(24)
        write_dispatch_sample_csv(p, sample, start_year=2027)
        rows = _read_csv(p)
        expected = {
            "pv_production_kwh", "price_spot_eur_per_kwh",
            "bess_soc_kwh", "bess_charge_pv_kwh", "pv_grid_export_kwh",
            "curtailed_kwh", "revenue_eur",
        }
        assert expected.issubset(set(rows[0].keys()))

    def test_timestamp_column_present(self, tmp_path: Path) -> None:
        p = tmp_path / "dispatch.csv"
        sample = _make_hourly_sample(24)
        write_dispatch_sample_csv(p, sample, start_year=2027)
        rows = _read_csv(p)
        # First key should be the timestamp column
        assert "timestamp" in rows[0] or any("timestamp" in k.lower() for k in rows[0].keys())


class TestCsvConfig:
    def test_defaults(self) -> None:
        cfg = CsvConfig()
        assert cfg.delimiter == CSV_DELIMITER
        assert isinstance(cfg.decimal, str)
        assert isinstance(cfg.timestamp_format, str)

    def test_custom_delimiter(self, tmp_path: Path) -> None:
        p = tmp_path / "mc.csv"
        cfg = CsvConfig(delimiter=",", decimal=".")
        write_monte_carlo_csv(p, _make_mc_result(2), config=cfg)
        content = p.read_text(encoding="utf-8")
        # Should use comma delimiter
        assert "," in content.split("\n")[0]


# ---------------------------------------------------------------------------
# Sensitivity CSV writers
# ---------------------------------------------------------------------------


def _make_sensitivity_result(
    analysis_type: str, n_points: int = 2, param_key: str = "floor_price_eur_per_kwh",
) -> SensitivityResult:
    points = []
    for i in range(n_points):
        mc = _make_mc_result(3)
        points.append(AnalysisResult(
            params={param_key: 0.05 + i * 0.01},
            mc_result=mc,
        ))
    return SensitivityResult(analysis_type=analysis_type, points=points)


class TestWriteEegSensitivityCsv:
    def test_produces_correct_row_count(self, tmp_path: Path) -> None:
        p = tmp_path / "eeg.csv"
        result = _make_sensitivity_result("eeg_sensitivity", n_points=3)
        write_eeg_sensitivity_csv(p, result)
        rows = _read_csv(p)
        assert len(rows) == 3

    def test_columns_present(self, tmp_path: Path) -> None:
        p = tmp_path / "eeg.csv"
        write_eeg_sensitivity_csv(p, _make_sensitivity_result("eeg_sensitivity"))
        rows = _read_csv(p)
        expected = {
            "floor_price_eur_per_kwh", "mc_iterations",
            "equity_irr_mean", "equity_irr_p50", "equity_irr_p90",
        }
        assert expected.issubset(set(rows[0].keys()))


class TestWritePpaCollarCsv:
    def test_produces_correct_row_count(self, tmp_path: Path) -> None:
        p = tmp_path / "collar.csv"
        result = SensitivityResult(
            analysis_type="ppa_collar",
            points=[
                AnalysisResult(
                    params={
                        "floor_price_eur_per_kwh": 0.05,
                        "cap_spread_eur_per_kwh": 0.03,
                        "cap_price_eur_per_kwh": 0.08,
                    },
                    mc_result=_make_mc_result(3),
                ),
                AnalysisResult(
                    params={
                        "floor_price_eur_per_kwh": 0.06,
                        "cap_spread_eur_per_kwh": 0.04,
                        "cap_price_eur_per_kwh": 0.10,
                    },
                    mc_result=_make_mc_result(3),
                ),
            ],
        )
        write_ppa_collar_csv(p, result, duration_years=10)
        rows = _read_csv(p)
        assert len(rows) == 2

    def test_columns_present(self, tmp_path: Path) -> None:
        p = tmp_path / "collar.csv"
        result = SensitivityResult(
            analysis_type="ppa_collar",
            points=[AnalysisResult(
                params={
                    "floor_price_eur_per_kwh": 0.05,
                    "cap_spread_eur_per_kwh": 0.03,
                    "cap_price_eur_per_kwh": 0.08,
                },
                mc_result=_make_mc_result(3),
            )],
        )
        write_ppa_collar_csv(p, result, duration_years=10)
        rows = _read_csv(p)
        expected = {
            "floor_price_eur_per_kwh", "cap_spread_eur_per_kwh",
            "cap_price_eur_per_kwh", "duration_years",
            "equity_irr_mean",
        }
        assert expected.issubset(set(rows[0].keys()))


class TestWritePpaBaseloadCsv:
    def test_produces_correct_row_count(self, tmp_path: Path) -> None:
        p = tmp_path / "baseload.csv"
        result = SensitivityResult(
            analysis_type="ppa_baseload",
            points=[
                AnalysisResult(
                    params={"ppa_price_eur_per_kwh": 0.06, "baseload_mw": 1.0},
                    mc_result=_make_mc_result(3),
                ),
            ],
        )
        write_ppa_baseload_csv(p, result, duration_years=15)
        rows = _read_csv(p)
        assert len(rows) == 1

    def test_columns_present(self, tmp_path: Path) -> None:
        p = tmp_path / "baseload.csv"
        result = SensitivityResult(
            analysis_type="ppa_baseload",
            points=[AnalysisResult(
                params={"ppa_price_eur_per_kwh": 0.06, "baseload_mw": 1.0},
                mc_result=_make_mc_result(3),
            )],
        )
        write_ppa_baseload_csv(p, result, duration_years=15)
        rows = _read_csv(p)
        expected = {
            "ppa_price_eur_per_kwh", "baseload_mw", "duration_years",
            "equity_irr_mean",
        }
        assert expected.issubset(set(rows[0].keys()))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestWriteDicts:
    def test_empty_rows_creates_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.csv"
        _write_dicts(p, [])
        assert p.exists()
        assert p.read_text(encoding="utf-8") == ""

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        p = tmp_path / "sub" / "dir" / "file.csv"
        _write_dicts(p, [{"a": "1", "b": "2"}])
        assert p.exists()
        rows = _read_csv(p)
        assert len(rows) == 1
