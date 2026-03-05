"""Tests for write_cashflows_csv – year column based on commissioning year (FIX-22).

Verifies that:
- When commissioning_year is provided, the year column shows calendar years.
- When commissioning_year is None, the year column shows project year indices.
- Calendar years are correctly computed as commissioning_year + project_year - 1.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from pv_bess_model.config.defaults import CSV_DELIMITER
from pv_bess_model.finance.cashflow import AnnualCashflow, CashflowProjection
from pv_bess_model.output.csv_writer import write_cashflows_csv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LIFETIME = 3


def _make_annual_cashflow(year: int) -> AnnualCashflow:
    """Create a minimal AnnualCashflow for testing."""
    return AnnualCashflow(
        year=year,
        revenue=1000.0 * year,
        opex=100.0 * year,
        capex=5000.0 if year == 1 else 0.0,
        debt_service=200.0,
        debt_interest=120.0,
        debt_repayment=80.0,
        depreciation=250.0,
        gewerbesteuer=10.0,
        koerperschaftsteuer=5.0,
        solidaritaetszuschlag=0.275,
        total_tax=15.275,
        project_cf=900.0 * year - (5000.0 if year == 1 else 0.0),
        equity_cf=700.0 * year - (5000.0 if year == 1 else 0.0),
    )


def _make_projection(lifetime: int = LIFETIME) -> CashflowProjection:
    """Build a CashflowProjection with *lifetime* years."""
    years = [_make_annual_cashflow(y) for y in range(1, lifetime + 1)]
    equity_cfs = np.array([y.equity_cf for y in years])
    project_cfs = np.array([y.project_cf for y in years])
    return CashflowProjection(
        years=years,
        equity_cashflows=equity_cfs,
        project_cashflows=project_cfs,
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read back a CSV written by write_cashflows_csv."""
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=CSV_DELIMITER)
        return list(reader)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCashflowCsvYearColumn:
    """Tests for the year column in the cashflows CSV (FIX-22)."""

    @pytest.fixture
    def projection(self) -> CashflowProjection:
        return _make_projection()

    @pytest.fixture
    def pv_kwh(self) -> list[float]:
        return [1000.0, 990.0, 980.0]

    @pytest.fixture
    def bess_kwh(self) -> list[float]:
        return [500.0, 490.0, 480.0]

    @pytest.fixture
    def dscr(self) -> list[float | None]:
        return [1.5, 1.6, 1.7]

    def test_calendar_years_with_commissioning_year(
        self, tmp_path, projection, pv_kwh, bess_kwh, dscr,
    ) -> None:
        """With commissioning_year=2027, years should be 2027, 2028, 2029."""
        path = tmp_path / "cashflows.csv"
        write_cashflows_csv(
            path=path,
            cashflow=projection,
            annual_pv_production_kwh=pv_kwh,
            annual_bess_throughput_kwh=bess_kwh,
            annual_dscr=dscr,
            commissioning_year=2027,
        )
        rows = _read_csv_rows(path)
        assert len(rows) == LIFETIME
        assert rows[0]["year"] == "2027"
        assert rows[1]["year"] == "2028"
        assert rows[2]["year"] == "2029"

    def test_project_year_indices_without_commissioning_year(
        self, tmp_path, projection, pv_kwh, bess_kwh, dscr,
    ) -> None:
        """Without commissioning_year (None), years should be 1, 2, 3."""
        path = tmp_path / "cashflows.csv"
        write_cashflows_csv(
            path=path,
            cashflow=projection,
            annual_pv_production_kwh=pv_kwh,
            annual_bess_throughput_kwh=bess_kwh,
            annual_dscr=dscr,
            commissioning_year=None,
        )
        rows = _read_csv_rows(path)
        assert rows[0]["year"] == "1"
        assert rows[1]["year"] == "2"
        assert rows[2]["year"] == "3"

    def test_calendar_year_formula(
        self, tmp_path, projection, pv_kwh, bess_kwh, dscr,
    ) -> None:
        """Calendar year = commissioning_year + project_year - 1."""
        commissioning = 2030
        path = tmp_path / "cashflows.csv"
        write_cashflows_csv(
            path=path,
            cashflow=projection,
            annual_pv_production_kwh=pv_kwh,
            annual_bess_throughput_kwh=bess_kwh,
            annual_dscr=dscr,
            commissioning_year=commissioning,
        )
        rows = _read_csv_rows(path)
        for i, row in enumerate(rows):
            expected = commissioning + i
            assert row["year"] == str(expected)

    def test_25_year_lifetime(self, tmp_path) -> None:
        """Calendar years are correct for a 25-year project."""
        lifetime = 25
        commissioning = 2027
        proj = _make_projection(lifetime)
        pv_kwh = [1000.0] * lifetime
        bess_kwh = [500.0] * lifetime
        dscr = [1.5] * lifetime

        path = tmp_path / "cashflows.csv"
        write_cashflows_csv(
            path=path,
            cashflow=proj,
            annual_pv_production_kwh=pv_kwh,
            annual_bess_throughput_kwh=bess_kwh,
            annual_dscr=dscr,
            commissioning_year=commissioning,
        )
        rows = _read_csv_rows(path)
        assert len(rows) == lifetime
        assert rows[0]["year"] == "2027"
        assert rows[-1]["year"] == str(2027 + lifetime - 1)

    def test_row_count_matches_lifetime(
        self, tmp_path, projection, pv_kwh, bess_kwh, dscr,
    ) -> None:
        """Number of CSV rows equals the number of project years."""
        path = tmp_path / "cashflows.csv"
        write_cashflows_csv(
            path=path,
            cashflow=projection,
            annual_pv_production_kwh=pv_kwh,
            annual_bess_throughput_kwh=bess_kwh,
            annual_dscr=dscr,
            commissioning_year=2027,
        )
        rows = _read_csv_rows(path)
        assert len(rows) == len(projection.years)

    def test_other_columns_still_present(
        self, tmp_path, projection, pv_kwh, bess_kwh, dscr,
    ) -> None:
        """All expected columns are present regardless of commissioning_year."""
        path = tmp_path / "cashflows.csv"
        write_cashflows_csv(
            path=path,
            cashflow=projection,
            annual_pv_production_kwh=pv_kwh,
            annual_bess_throughput_kwh=bess_kwh,
            annual_dscr=dscr,
            commissioning_year=2027,
        )
        rows = _read_csv_rows(path)
        expected_cols = {
            "year", "capex_eur", "pv_production_mwh", "bess_throughput_mwh",
            "revenue_eur", "opex_eur", "debt_interest_eur", "debt_repayment_eur",
            "depreciation_eur", "gewerbesteuer_eur", "koerperschaftsteuer_eur",
            "solidaritaetszuschlag_eur", "total_tax_eur",
            "project_cf_eur", "equity_cf_eur", "cumulative_equity_cf_eur", "dscr",
        }
        assert set(rows[0].keys()) == expected_cols
