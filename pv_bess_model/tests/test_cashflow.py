"""Tests for finance/cashflow.py – Annual cashflow projection.

Tests verify the new cashflow structure (FIX-08):
- No Year 0: arrays have length = lifetime_years, index 0 = Year 1.
- CAPEX is booked in Year 1 (commissioning year).
- Equity CF Year 1 = Revenue - OPEX - Equity Investment - Debt Service - Tax.
- Equity CF Year 2+ = Revenue - OPEX - Debt Service - Tax.
- Project CF Year 1 = Revenue - OPEX - Total CAPEX - Tax.
- Project CF Year 2+ = Revenue - OPEX - Tax.
- Verlustvortrag 3-year scenario: (-100k, +60k, +80k).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pv_bess_model.finance.cashflow import build_cashflow_projection
from pv_bess_model.finance.debt import build_annuity_schedule
from pv_bess_model.finance.inflation import inflate_value
from pv_bess_model.finance.tax import calculate_gewerbesteuer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_schedule(
    capex: float = 1_000_000.0,
    leverage: float = 75.0,
    rate: float = 0.045,
    tenor: int = 18,
):
    """Build a standard annuity schedule for tests."""
    return build_annuity_schedule(capex, leverage, rate, tenor)


def _build_simple_projection(
    lifetime: int = 5,
    revenues: list[float] | None = None,
    base_opex: float = 50_000.0,
    inflation: float = 0.0,
    capex_total: float = 1_000_000.0,
    capex_pv: float = 700_000.0,
    capex_bess: float = 300_000.0,
    leverage: float = 0.0,
    rate: float = 0.045,
    tenor: int = 18,
    afa_pv: int = 20,
    afa_bess: int = 10,
    messzahl: float = 0.035,
    hebesatz: float = 400.0,
    replacement_cost: float = 0.0,
    replacement_year: int | None = None,
):
    """Build a cashflow projection with sensible defaults for testing."""
    if revenues is None:
        revenues = [200_000.0] * lifetime
    sched = build_annuity_schedule(capex_total, leverage, rate, tenor)
    return build_cashflow_projection(
        lifetime_years=lifetime,
        annual_revenues=revenues,
        base_opex=base_opex,
        inflation_rate=inflation,
        capex_total=capex_total,
        capex_pv=capex_pv,
        capex_bess=capex_bess,
        debt_schedule=sched,
        afa_years_pv=afa_pv,
        afa_years_bess=afa_bess,
        gewerbesteuer_messzahl=messzahl,
        gewerbesteuer_hebesatz=hebesatz,
        replacement_cost=replacement_cost,
        replacement_year=replacement_year,
    )


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------


class TestCashflowStructure:
    """Tests for basic cashflow projection structure (FIX-08: no Year 0)."""

    def test_array_length(self) -> None:
        """Arrays and year list must have length = lifetime (no Year 0)."""
        proj = _build_simple_projection(lifetime=25)
        assert len(proj.equity_cashflows) == 25
        assert len(proj.project_cashflows) == 25
        assert len(proj.years) == 25

    def test_year_indices_start_at_one(self) -> None:
        """Year objects must be indexed 1 through lifetime."""
        proj = _build_simple_projection(lifetime=5)
        for i, annual in enumerate(proj.years):
            assert annual.year == i + 1

    def test_capex_only_in_year_one(self) -> None:
        """Year 1 carries CAPEX, years 2+ have capex = 0."""
        capex = 1_000_000.0
        proj = _build_simple_projection(capex_total=capex, leverage=0.0)
        assert proj.years[0].capex == capex
        for y in proj.years[1:]:
            assert y.capex == 0.0

    def test_capex_field_present(self) -> None:
        """AnnualCashflow has a capex field."""
        proj = _build_simple_projection(lifetime=3)
        for y in proj.years:
            assert hasattr(y, "capex")


# ---------------------------------------------------------------------------
# Year 1 CAPEX tests (replaces old Year 0 tests)
# ---------------------------------------------------------------------------


class TestYear1Capex:
    """Tests for Year 1 which now includes CAPEX (FIX-08)."""

    def test_equity_cf_year1_no_debt(self) -> None:
        """No leverage → equity CF year 1 = Revenue - OPEX - CAPEX - Tax."""
        capex = 2_000_000.0
        proj = _build_simple_projection(
            lifetime=3,
            revenues=[200_000.0] * 3,
            capex_total=capex,
            leverage=0.0,
        )
        y1 = proj.years[0]
        # Equity investment = CAPEX (no debt)
        expected = y1.revenue - y1.opex - capex - y1.total_tax
        assert math.isclose(y1.equity_cf, expected, rel_tol=1e-9)

    def test_equity_cf_year1_with_debt(self) -> None:
        """75 % leverage → equity CF year 1 includes only equity portion of CAPEX."""
        capex = 1_000_000.0
        proj = _build_simple_projection(
            lifetime=3,
            revenues=[200_000.0] * 3,
            capex_total=capex,
            leverage=75.0,
        )
        y1 = proj.years[0]
        equity_investment = capex * 0.25  # 25 % equity
        expected = y1.revenue - y1.opex - y1.debt_service - y1.total_tax - equity_investment
        assert math.isclose(y1.equity_cf, expected, rel_tol=1e-9)

    def test_project_cf_year1(self) -> None:
        """Project CF year 1 = Revenue - OPEX - full CAPEX - Tax."""
        capex = 1_000_000.0
        proj = _build_simple_projection(
            lifetime=3,
            revenues=[200_000.0] * 3,
            capex_total=capex,
            leverage=75.0,
        )
        y1 = proj.years[0]
        expected = y1.revenue - y1.opex - capex - y1.total_tax
        assert math.isclose(y1.project_cf, expected, rel_tol=1e-9)

    def test_year1_has_revenue_and_opex(self) -> None:
        """Year 1 now includes revenue and OPEX (unlike old Year 0)."""
        proj = _build_simple_projection(
            lifetime=3,
            revenues=[200_000.0, 150_000.0, 100_000.0],
        )
        y1 = proj.years[0]
        assert y1.revenue == 200_000.0
        assert y1.opex > 0.0

    def test_year1_has_debt_service(self) -> None:
        """Year 1 has debt service when leverage > 0."""
        proj = _build_simple_projection(
            lifetime=3,
            leverage=75.0,
        )
        y1 = proj.years[0]
        assert y1.debt_service > 0.0


# ---------------------------------------------------------------------------
# Equity CF identity tests (years 2+)
# ---------------------------------------------------------------------------


class TestEquityCfIdentity:
    """Verify Equity CF = Revenue - OPEX - Debt Service - Tax (for years 2+)."""

    def test_equity_cf_no_tax_no_debt(self) -> None:
        """Without debt and with zero tax, equity CF years 2+ = revenue - opex."""
        proj = _build_simple_projection(
            lifetime=3,
            revenues=[100_000.0, 100_000.0, 100_000.0],
            base_opex=30_000.0,
            inflation=0.0,
            leverage=0.0,
            capex_pv=0.0,
            capex_bess=0.0,
            messzahl=0.0,  # disable GewSt
        )
        # Years 2+ (index 1+): no CAPEX subtracted
        for y in proj.years[1:]:
            expected = y.revenue - y.opex - y.debt_service - y.total_tax
            assert math.isclose(y.equity_cf, expected, rel_tol=1e-9)

    def test_equity_cf_with_debt(self) -> None:
        """With debt, equity CF = revenue - opex - debt_service - tax (years 2+)."""
        proj = _build_simple_projection(
            lifetime=3,
            revenues=[300_000.0] * 3,
            base_opex=50_000.0,
            inflation=0.0,
            capex_total=1_000_000.0,
            leverage=75.0,
        )
        for y in proj.years[1:]:
            expected = y.revenue - y.opex - y.debt_service - y.total_tax
            assert math.isclose(y.equity_cf, expected, rel_tol=1e-9)

    def test_equity_cf_array_matches_year_objects(self) -> None:
        """equity_cashflows array must match AnnualCashflow.equity_cf."""
        proj = _build_simple_projection(lifetime=5)
        for i, annual in enumerate(proj.years):
            assert math.isclose(proj.equity_cashflows[i], annual.equity_cf, rel_tol=1e-9)

    def test_project_cf_array_matches_year_objects(self) -> None:
        """project_cashflows array must match AnnualCashflow.project_cf."""
        proj = _build_simple_projection(lifetime=5)
        for i, annual in enumerate(proj.years):
            assert math.isclose(proj.project_cashflows[i], annual.project_cf, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Inflation tests
# ---------------------------------------------------------------------------


class TestInflation:
    """Verify OPEX inflation is applied correctly."""

    def test_opex_inflated(self) -> None:
        """OPEX in year y = base_opex × (1 + rate)^y."""
        base = 100_000.0
        rate = 0.03
        proj = _build_simple_projection(
            lifetime=5, base_opex=base, inflation=rate,
            capex_pv=0.0, capex_bess=0.0, leverage=0.0, messzahl=0.0,
        )
        for y in proj.years:
            expected_opex = inflate_value(base, rate, y.year)
            assert math.isclose(y.opex, expected_opex, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# BESS replacement tests
# ---------------------------------------------------------------------------


class TestBessReplacement:
    """Verify BESS replacement cost is added in the correct year."""

    def test_replacement_adds_opex(self) -> None:
        """Replacement cost is added to OPEX in the specified year only."""
        repl_cost = 500_000.0
        repl_year = 3
        proj = _build_simple_projection(
            lifetime=5, inflation=0.0,
            replacement_cost=repl_cost, replacement_year=repl_year,
            capex_pv=0.0, capex_bess=0.0, leverage=0.0, messzahl=0.0,
        )
        base_opex = 50_000.0
        for y in proj.years:
            if y.year == repl_year:
                assert math.isclose(y.opex, base_opex + repl_cost)
            else:
                assert math.isclose(y.opex, base_opex)


# ---------------------------------------------------------------------------
# Verlustvortrag 3-year scenario
# ---------------------------------------------------------------------------


class TestVerlustvortrag:
    """3-year scenario testing loss carry-forward through cashflow projection.

    Setup: no depreciation (capex=0), no inflation, no debt.
    Year 1: revenue=100k, opex=200k → taxable = -100k → Vortrag = -100k, GewSt=0
    Year 2: revenue=260k, opex=200k → taxable = +60k, after Vortrag: -40k → GewSt=0
    Year 3: revenue=280k, opex=200k → taxable = +80k, after Vortrag: +40k → GewSt on 40k
    """

    def test_verlustvortrag_three_years(self) -> None:
        """Verify GewSt is 0 for years 1-2 and correct for year 3."""
        messzahl = 0.035
        hebesatz = 400.0
        proj = _build_simple_projection(
            lifetime=3,
            revenues=[100_000.0, 260_000.0, 280_000.0],
            base_opex=200_000.0,
            inflation=0.0,
            capex_total=0.0,
            capex_pv=0.0,
            capex_bess=0.0,
            leverage=0.0,
            messzahl=messzahl,
            hebesatz=hebesatz,
        )
        # Year 1 (index 0): loss, no GewSt
        assert proj.years[0].gewerbesteuer == 0.0

        # Year 2 (index 1): positive taxable but absorbed by carry-forward
        assert proj.years[1].gewerbesteuer == 0.0

        # Year 3 (index 2): 40k adjusted taxable income → GewSt = 40 000 × 0.035 × 400/100 = 5 600
        expected_gewst = calculate_gewerbesteuer(40_000.0, messzahl, hebesatz)
        assert math.isclose(expected_gewst, 5_600.0)
        assert math.isclose(proj.years[2].gewerbesteuer, 5_600.0)

    def test_verlustvortrag_equity_cf(self) -> None:
        """Verify equity CF reflects GewSt correctly in the carry-forward scenario."""
        proj = _build_simple_projection(
            lifetime=3,
            revenues=[100_000.0, 260_000.0, 280_000.0],
            base_opex=200_000.0,
            inflation=0.0,
            capex_total=0.0,
            capex_pv=0.0,
            capex_bess=0.0,
            leverage=0.0,
            messzahl=0.035,
            hebesatz=400.0,
        )
        # Year 1 (index 0): equity CF = 100k - 200k - 0 (no capex) - 0 = -100k
        assert math.isclose(proj.years[0].equity_cf, -100_000.0)

        # Year 2 (index 1): equity CF = 260k - 200k - 0 - 0 = 60k
        assert math.isclose(proj.years[1].equity_cf, 60_000.0)

        # Year 3 (index 2): equity CF = 280k - 200k - 0 - total_tax
        # total_tax = GewSt(5600) + KSt(40000*0.15=6000) + Soli(6000*0.055=330) = 11930
        expected_total_tax = 5_600.0 + 6_000.0 + 330.0
        assert math.isclose(
            proj.years[2].equity_cf, 280_000.0 - 200_000.0 - expected_total_tax
        )


# ---------------------------------------------------------------------------
# Depreciation tests
# ---------------------------------------------------------------------------


class TestDepreciation:
    """Verify depreciation is tracked correctly (for reporting, not CF)."""

    def test_depreciation_within_period(self) -> None:
        """Years within AfA period have positive depreciation."""
        proj = _build_simple_projection(
            lifetime=5,
            capex_pv=1_000_000.0,
            capex_bess=500_000.0,
            afa_pv=20,
            afa_bess=10,
            leverage=0.0,
            messzahl=0.0,
        )
        # Year 1 (index 0)
        y1 = proj.years[0]
        expected = 1_000_000.0 / 20 + 500_000.0 / 10
        assert math.isclose(y1.depreciation, expected)

    def test_depreciation_beyond_bess_afa(self) -> None:
        """After BESS AfA period, only PV depreciation remains."""
        proj = _build_simple_projection(
            lifetime=15,
            revenues=[200_000.0] * 15,
            capex_pv=1_000_000.0,
            capex_bess=500_000.0,
            afa_pv=20,
            afa_bess=10,
            leverage=0.0,
            messzahl=0.0,
        )
        # Year 11 (index 10): BESS AfA ended, only PV remains
        y11 = proj.years[10]
        expected = 1_000_000.0 / 20  # only PV
        assert math.isclose(y11.depreciation, expected)


# ---------------------------------------------------------------------------
# Project CF tests
# ---------------------------------------------------------------------------


class TestProjectCf:
    """Verify Project CF = Revenue - OPEX - Tax - CAPEX(year 1 only)."""

    def test_project_cf_year1_includes_capex(self) -> None:
        """Project CF in Year 1 subtracts full CAPEX."""
        capex = 500_000.0
        proj = _build_simple_projection(
            lifetime=3,
            revenues=[200_000.0] * 3,
            base_opex=50_000.0,
            inflation=0.0,
            capex_total=capex,
            capex_pv=0.0,
            capex_bess=0.0,
            leverage=0.0,
            messzahl=0.0,
        )
        y1 = proj.years[0]
        # project_cf = revenue - opex - tax - capex
        expected = y1.revenue - y1.opex - y1.total_tax - capex
        assert math.isclose(y1.project_cf, expected, rel_tol=1e-9)

    def test_project_cf_year2_no_capex(self) -> None:
        """Project CF in Year 2+ does not subtract CAPEX."""
        capex = 500_000.0
        proj = _build_simple_projection(
            lifetime=3,
            revenues=[200_000.0] * 3,
            base_opex=50_000.0,
            inflation=0.0,
            capex_total=capex,
            capex_pv=0.0,
            capex_bess=0.0,
            leverage=0.0,
            messzahl=0.0,
        )
        y2 = proj.years[1]
        # project_cf = revenue - opex - tax (no capex)
        expected = y2.revenue - y2.opex - y2.total_tax
        assert math.isclose(y2.project_cf, expected, rel_tol=1e-9)

    def test_project_cf_independent_of_leverage(self) -> None:
        """Project CF is the same regardless of leverage (pre-leverage metric)."""
        capex = 1_000_000.0
        proj_no_debt = _build_simple_projection(
            lifetime=3, capex_total=capex, leverage=0.0,
            capex_pv=0.0, capex_bess=0.0, messzahl=0.0,
        )
        proj_with_debt = _build_simple_projection(
            lifetime=3, capex_total=capex, leverage=75.0,
            capex_pv=0.0, capex_bess=0.0, messzahl=0.0,
        )
        for i in range(3):
            assert math.isclose(
                proj_no_debt.project_cashflows[i],
                proj_with_debt.project_cashflows[i],
                rel_tol=1e-9,
            )


# ---------------------------------------------------------------------------
# Edge case: zero CAPEX
# ---------------------------------------------------------------------------


class TestZeroCapex:
    """Edge case: project with zero total CAPEX."""

    def test_zero_capex_no_crash(self) -> None:
        """Should produce valid results with zero CAPEX."""
        proj = _build_simple_projection(
            lifetime=3,
            capex_total=0.0,
            capex_pv=0.0,
            capex_bess=0.0,
            leverage=0.0,
        )
        assert len(proj.years) == 3
        assert proj.years[0].capex == 0.0

    def test_zero_capex_equity_cf_equals_project_cf(self) -> None:
        """With zero CAPEX and no debt, equity CF equals project CF."""
        proj = _build_simple_projection(
            lifetime=3,
            capex_total=0.0,
            capex_pv=0.0,
            capex_bess=0.0,
            leverage=0.0,
            messzahl=0.0,
        )
        for i in range(3):
            assert math.isclose(
                proj.equity_cashflows[i],
                proj.project_cashflows[i],
                rel_tol=1e-9,
            )
