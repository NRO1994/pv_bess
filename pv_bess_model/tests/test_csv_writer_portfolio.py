"""Tests for the portfolio CSV writer module."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from pv_bess_model.dispatch.optimizer_portfolio import PortfolioDailyResult
from pv_bess_model.output.csv_writer_portfolio import (
    write_baseline_csv,
    write_marginal_value_csv,
    write_portfolio_dispatch_sample_csv,
    write_system_value_csv,
)
from pv_bess_model.portfolio.marginal_value import MarginalValuePoint
from pv_bess_model.portfolio.system_value import SystemValuePoint, SystemValueResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_csv(tmp_path: Path) -> Path:
    """Return a temporary CSV file path."""
    return tmp_path / "test.csv"


@pytest.fixture
def sample_world_a_costs() -> list[float]:
    """Sample World A annual costs (3 years)."""
    return [10000.0, 11000.0, 12000.0]


@pytest.fixture
def sample_system_value_result() -> SystemValueResult:
    """Sample system value result with 2 points."""
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
                marginal_value_eur_per_kw_a=50.0,
            ),
            SystemValuePoint(
                flex_name="BESS_1",
                flex_type="bess",
                annual_addition_kw=200.0,
                e_to_p_ratio=2.0,
                cumulative_system_value_eur=8000.0,
                annual_system_values=[2500.0, 2700.0, 2800.0],
                marginal_value_eur_per_kw_a=30.0,
            ),
        ],
    )


@pytest.fixture
def sample_marginal_values() -> list[MarginalValuePoint]:
    """Sample marginal value points."""
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
        MarginalValuePoint(
            flex_name="BESS_1",
            flex_type="bess",
            annual_addition_kw=200.0,
            e_to_p_ratio=2.0,
            cumulative_system_value_eur=8000.0,
            marginal_value_eur_per_kw_a=30.0,
            delta_kw=100.0,
            delta_value_eur=3000.0,
        ),
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV file and return list of dicts."""
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        return list(reader)


# ---------------------------------------------------------------------------
# Tests: write_baseline_csv
# ---------------------------------------------------------------------------


class TestWriteBaselineCsv:
    """Tests for write_baseline_csv."""

    def test_writes_correct_rows(
        self, tmp_csv: Path, sample_world_a_costs: list[float]
    ) -> None:
        """Baseline CSV has one row per year with cost and cumulative cost."""
        write_baseline_csv(tmp_csv, sample_world_a_costs, baseline_year=2027)

        rows = _read_csv(tmp_csv)
        assert len(rows) == 3

        assert rows[0]["year"] == "2027"
        assert rows[0]["project_year"] == "1"
        assert rows[1]["year"] == "2028"
        assert rows[2]["year"] == "2029"

    def test_cumulative_cost(
        self, tmp_csv: Path, sample_world_a_costs: list[float]
    ) -> None:
        """Cumulative costs should accumulate across years."""
        write_baseline_csv(tmp_csv, sample_world_a_costs, baseline_year=2027)

        rows = _read_csv(tmp_csv)
        # cumulative: 10000, 21000, 33000
        assert "21000" in rows[1]["cumulative_system_cost_eur"]
        assert "33000" in rows[2]["cumulative_system_cost_eur"]

    def test_empty_costs(self, tmp_csv: Path) -> None:
        """Empty cost list produces an empty file."""
        write_baseline_csv(tmp_csv, [])
        assert tmp_csv.read_text().strip() == ""


# ---------------------------------------------------------------------------
# Tests: write_system_value_csv
# ---------------------------------------------------------------------------


class TestWriteSystemValueCsv:
    """Tests for write_system_value_csv."""

    def test_writes_correct_rows(
        self, tmp_csv: Path, sample_system_value_result: SystemValueResult
    ) -> None:
        """System value CSV has one row per enumeration point."""
        write_system_value_csv(tmp_csv, sample_system_value_result)

        rows = _read_csv(tmp_csv)
        assert len(rows) == 2

        assert rows[0]["flex_name"] == "BESS_1"
        assert rows[0]["flex_type"] == "bess"

    def test_e_to_p_ratio_none(self, tmp_csv: Path) -> None:
        """Points with e_to_p_ratio=None write empty string."""
        result = SystemValueResult(
            world_a_annual_costs=[1000.0],
            points=[
                SystemValuePoint(
                    flex_name="WP_1",
                    flex_type="heat_pump",
                    annual_addition_kw=50.0,
                    e_to_p_ratio=None,
                    cumulative_system_value_eur=2000.0,
                    annual_system_values=[2000.0],
                ),
            ],
        )
        write_system_value_csv(tmp_csv, result)

        rows = _read_csv(tmp_csv)
        assert rows[0]["e_to_p_ratio_h"] == ""

    def test_empty_result(self, tmp_csv: Path) -> None:
        """Empty result produces an empty file."""
        result = SystemValueResult()
        write_system_value_csv(tmp_csv, result)
        assert tmp_csv.read_text().strip() == ""


# ---------------------------------------------------------------------------
# Tests: write_marginal_value_csv
# ---------------------------------------------------------------------------


class TestWriteMarginalValueCsv:
    """Tests for write_marginal_value_csv."""

    def test_writes_correct_rows(
        self, tmp_csv: Path, sample_marginal_values: list[MarginalValuePoint]
    ) -> None:
        """Marginal value CSV has one row per step."""
        write_marginal_value_csv(tmp_csv, sample_marginal_values)

        rows = _read_csv(tmp_csv)
        assert len(rows) == 2

        assert rows[0]["flex_name"] == "BESS_1"
        assert "50" in rows[0]["marginal_value_eur_per_kw_a"]
        assert "30" in rows[1]["marginal_value_eur_per_kw_a"]

    def test_delta_fields(
        self, tmp_csv: Path, sample_marginal_values: list[MarginalValuePoint]
    ) -> None:
        """Delta fields are written correctly."""
        write_marginal_value_csv(tmp_csv, sample_marginal_values)

        rows = _read_csv(tmp_csv)
        assert "100" in rows[0]["delta_kw"]
        assert "5000" in rows[0]["delta_value_eur"]
        assert "3000" in rows[1]["delta_value_eur"]

    def test_empty_list(self, tmp_csv: Path) -> None:
        """Empty marginal values list produces an empty file."""
        write_marginal_value_csv(tmp_csv, [])
        assert tmp_csv.read_text().strip() == ""


# ---------------------------------------------------------------------------
# Tests: write_portfolio_dispatch_sample_csv
# ---------------------------------------------------------------------------


class TestWriteDispatchSampleCsv:
    """Tests for write_portfolio_dispatch_sample_csv."""

    @pytest.fixture
    def sample_daily_results(self) -> list[PortfolioDailyResult]:
        """Create 2 days of sample daily results (T=24 for speed)."""
        T = 24
        results = []
        for _ in range(2):
            results.append(
                PortfolioDailyResult(
                    grid_sell=np.ones(T) * 10.0,
                    grid_buy=np.ones(T) * 5.0,
                    bess_charge=np.ones(T) * 2.0,
                    bess_discharge=np.ones(T) * 1.5,
                    bess_soc=np.ones(T + 1) * 50.0,
                    system_cost=100.0,
                    end_soc_kwh=50.0,
                    solver_status="optimal",
                )
            )
        return results

    def test_writes_correct_row_count(
        self,
        tmp_csv: Path,
        sample_daily_results: list[PortfolioDailyResult],
    ) -> None:
        """Dispatch CSV has T * n_days rows."""
        T = 24
        pv = np.ones(2 * T) * 20.0
        load = np.ones(2 * T) * 15.0
        prices = np.ones(2 * T) * 0.05

        write_portfolio_dispatch_sample_csv(
            tmp_csv,
            sample_daily_results,
            pv_profile=pv,
            load_profile=load,
            spot_prices=prices,
            year=2027,
            intervals_per_day=T,
        )

        rows = _read_csv(tmp_csv)
        assert len(rows) == 48  # 2 days * 24 intervals

    def test_base_columns_present(
        self,
        tmp_csv: Path,
        sample_daily_results: list[PortfolioDailyResult],
    ) -> None:
        """CSV has all base columns."""
        T = 24
        pv = np.ones(2 * T) * 20.0
        load = np.ones(2 * T) * 15.0
        prices = np.ones(2 * T) * 0.05

        write_portfolio_dispatch_sample_csv(
            tmp_csv,
            sample_daily_results,
            pv_profile=pv,
            load_profile=load,
            spot_prices=prices,
            year=2027,
            intervals_per_day=T,
        )

        rows = _read_csv(tmp_csv)
        expected_columns = {
            "timestamp",
            "pv_production_kwh",
            "load_demand_kwh",
            "spot_price_eur_per_kwh",
            "grid_sell_kwh",
            "grid_buy_kwh",
            "bess_charge_kwh",
            "bess_discharge_kwh",
            "bess_soc_kwh",
        }
        assert expected_columns.issubset(set(rows[0].keys()))

    def test_wp_columns_when_present(
        self,
        tmp_csv: Path,
    ) -> None:
        """WP columns appear when wp_load and thermal_soc are set."""
        T = 24
        dr = PortfolioDailyResult(
            grid_sell=np.ones(T),
            grid_buy=np.ones(T),
            bess_charge=np.zeros(T),
            bess_discharge=np.zeros(T),
            bess_soc=np.zeros(T + 1),
            system_cost=0.0,
            end_soc_kwh=0.0,
            solver_status="optimal",
            wp_load=np.ones(T) * 3.0,
            thermal_soc=np.ones(T + 1) * 10.0,
        )

        pv = np.ones(T) * 20.0
        load = np.ones(T) * 15.0
        prices = np.ones(T) * 0.05

        write_portfolio_dispatch_sample_csv(
            tmp_csv,
            [dr],
            pv_profile=pv,
            load_profile=load,
            spot_prices=prices,
            year=2027,
            intervals_per_day=T,
        )

        rows = _read_csv(tmp_csv)
        assert "wp_load_kwh" in rows[0]
        assert "thermal_soc_kwh" in rows[0]

    def test_ev_columns_when_present(
        self,
        tmp_csv: Path,
    ) -> None:
        """EV columns appear when ev_charge, ev_discharge, ev_soc are set."""
        T = 24
        dr = PortfolioDailyResult(
            grid_sell=np.ones(T),
            grid_buy=np.ones(T),
            bess_charge=np.zeros(T),
            bess_discharge=np.zeros(T),
            bess_soc=np.zeros(T + 1),
            system_cost=0.0,
            end_soc_kwh=0.0,
            solver_status="optimal",
            ev_charge=np.ones(T) * 5.0,
            ev_discharge=np.ones(T) * 2.0,
            ev_soc=np.ones(T + 1) * 25.0,
        )

        pv = np.ones(T) * 20.0
        load = np.ones(T) * 15.0
        prices = np.ones(T) * 0.05

        write_portfolio_dispatch_sample_csv(
            tmp_csv,
            [dr],
            pv_profile=pv,
            load_profile=load,
            spot_prices=prices,
            year=2027,
            intervals_per_day=T,
        )

        rows = _read_csv(tmp_csv)
        assert "ev_charge_kwh" in rows[0]
        assert "ev_discharge_kwh" in rows[0]
        assert "ev_soc_kwh" in rows[0]

    def test_no_optional_columns_by_default(
        self,
        tmp_csv: Path,
        sample_daily_results: list[PortfolioDailyResult],
    ) -> None:
        """Without WP or EV data, optional columns are absent."""
        T = 24
        pv = np.ones(2 * T) * 20.0
        load = np.ones(2 * T) * 15.0
        prices = np.ones(2 * T) * 0.05

        write_portfolio_dispatch_sample_csv(
            tmp_csv,
            sample_daily_results,
            pv_profile=pv,
            load_profile=load,
            spot_prices=prices,
            year=2027,
            intervals_per_day=T,
        )

        rows = _read_csv(tmp_csv)
        assert "wp_load_kwh" not in rows[0]
        assert "ev_charge_kwh" not in rows[0]
