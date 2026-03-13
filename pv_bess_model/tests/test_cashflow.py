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
    replacement_leverage: float = 0.0,
    optimization_fee_pct: float = 0.0,
    annual_bess_spot_revenues: list[float] | None = None,
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
        replacement_leverage_pct=replacement_leverage,
        replacement_interest_rate=rate,
        replacement_loan_tenor_years=tenor,
        optimization_fee_pct=optimization_fee_pct,
        annual_bess_spot_revenues=annual_bess_spot_revenues,
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
    """Verify Equity CF = Revenue - OPEX - Debt interest - Tax (for years 2+)."""

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
            expected = y.revenue - y.opex - y.debt_interest - y.total_tax
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
    """Verify BESS replacement cost is treated as CAPEX (Feature 04)."""

    def test_replacement_as_capex_not_opex(self) -> None:
        """Replacement cost appears as CAPEX, not OPEX, in the replacement year."""
        repl_cost = 500_000.0
        repl_year = 3
        proj = _build_simple_projection(
            lifetime=5, inflation=0.0,
            replacement_cost=repl_cost, replacement_year=repl_year,
            capex_pv=0.0, capex_bess=0.0, capex_total=0.0,
            leverage=0.0, messzahl=0.0,
        )
        base_opex = 50_000.0
        for y in proj.years:
            # OPEX is never affected by replacement
            assert math.isclose(y.opex, base_opex)
            # CAPEX only in replacement year (capex_total=0 → no year-1 CAPEX)
            if y.year == repl_year:
                assert math.isclose(y.capex, repl_cost)
            else:
                assert math.isclose(y.capex, 0.0)

    def test_replacement_capex_reduces_equity_cf(self) -> None:
        """Replacement CAPEX is subtracted from equity CF in replacement year."""
        repl_cost = 500_000.0
        repl_year = 3
        proj_no_repl = _build_simple_projection(
            lifetime=5, inflation=0.0,
            capex_total=0.0, capex_pv=0.0, capex_bess=0.0,
            leverage=0.0, messzahl=0.0,
        )
        proj_with_repl = _build_simple_projection(
            lifetime=5, inflation=0.0,
            replacement_cost=repl_cost, replacement_year=repl_year,
            capex_total=0.0, capex_pv=0.0, capex_bess=0.0,
            leverage=0.0, messzahl=0.0,
        )
        # In the replacement year, equity CF drops by the replacement cost
        # (plus any tax effect from the new depreciation)
        repl_idx = repl_year - 1
        assert proj_with_repl.years[repl_idx].equity_cf < proj_no_repl.years[repl_idx].equity_cf

    def test_replacement_capex_reduces_project_cf(self) -> None:
        """Replacement CAPEX is subtracted from project CF in replacement year."""
        repl_cost = 500_000.0
        repl_year = 3
        proj_no_repl = _build_simple_projection(
            lifetime=5, inflation=0.0,
            capex_total=0.0, capex_pv=0.0, capex_bess=0.0,
            leverage=0.0, messzahl=0.0,
        )
        proj_with_repl = _build_simple_projection(
            lifetime=5, inflation=0.0,
            replacement_cost=repl_cost, replacement_year=repl_year,
            capex_total=0.0, capex_pv=0.0, capex_bess=0.0,
            leverage=0.0, messzahl=0.0,
        )
        repl_idx = repl_year - 1
        assert proj_with_repl.years[repl_idx].project_cf < proj_no_repl.years[repl_idx].project_cf

    def test_replacement_starts_new_afa(self) -> None:
        """Replacement starts a second AfA line from the replacement year.

        Example: BESS CAPEX = 1M, Replacement = 500k, AfA = 10 years,
        Replacement in year 12.
        Years 1-10: AfA = 100k/yr (original)
        Year 11: AfA = 0 (original finished, replacement not yet)
        Years 12-21: AfA = 50k/yr (replacement)
        """
        proj = _build_simple_projection(
            lifetime=25, inflation=0.0,
            revenues=[500_000.0] * 25,
            capex_total=1_000_000.0,
            capex_pv=0.0,
            capex_bess=1_000_000.0,
            replacement_cost=500_000.0,
            replacement_year=12,
            leverage=0.0, messzahl=0.0,
            afa_pv=20, afa_bess=10,
        )
        # Years 1-10: AfA = 1M / 10 = 100k
        for y in proj.years[:10]:
            assert math.isclose(y.depreciation, 100_000.0), f"Year {y.year}: {y.depreciation}"
        # Year 11: original AfA ended, replacement not yet started
        assert math.isclose(proj.years[10].depreciation, 0.0)
        # Years 12-21: replacement AfA = 500k / 10 = 50k
        for y in proj.years[11:21]:
            assert math.isclose(y.depreciation, 50_000.0), f"Year {y.year}: {y.depreciation}"
        # Year 22+: all AfA finished
        for y in proj.years[21:]:
            assert math.isclose(y.depreciation, 0.0), f"Year {y.year}: {y.depreciation}"

    def test_overlapping_afa_lines(self) -> None:
        """If replacement is within original AfA period, both lines overlap.

        Example: BESS CAPEX = 1M, Replacement = 500k, AfA = 10 years,
        Replacement in year 5.
        Years 1-4: AfA = 100k (original only)
        Years 5-10: AfA = 100k + 50k = 150k (overlap)
        Years 11-14: AfA = 50k (replacement only)
        Year 15+: AfA = 0
        """
        proj = _build_simple_projection(
            lifetime=20, inflation=0.0,
            revenues=[500_000.0] * 20,
            capex_total=1_000_000.0,
            capex_pv=0.0,
            capex_bess=1_000_000.0,
            replacement_cost=500_000.0,
            replacement_year=5,
            leverage=0.0, messzahl=0.0,
            afa_pv=20, afa_bess=10,
        )
        # Years 1-4: original only = 100k
        for y in proj.years[:4]:
            assert math.isclose(y.depreciation, 100_000.0), f"Year {y.year}: {y.depreciation}"
        # Years 5-10: overlap = 100k + 50k = 150k
        for y in proj.years[4:10]:
            assert math.isclose(y.depreciation, 150_000.0), f"Year {y.year}: {y.depreciation}"
        # Years 11-14: replacement only = 50k
        for y in proj.years[10:14]:
            assert math.isclose(y.depreciation, 50_000.0), f"Year {y.year}: {y.depreciation}"
        # Year 15+: all finished
        for y in proj.years[14:]:
            assert math.isclose(y.depreciation, 0.0), f"Year {y.year}: {y.depreciation}"

    def test_no_replacement_unchanged(self) -> None:
        """Without replacement, behavior is identical (no CAPEX outflow, no second AfA)."""
        proj = _build_simple_projection(
            lifetime=5, inflation=0.0,
            capex_total=1_000_000.0,
            capex_pv=500_000.0,
            capex_bess=500_000.0,
            leverage=0.0, messzahl=0.0,
            afa_pv=20, afa_bess=10,
        )
        # No replacement CAPEX in any year (only initial CAPEX in year 1)
        assert proj.years[0].capex == 1_000_000.0
        for y in proj.years[1:]:
            assert y.capex == 0.0
        # Depreciation = PV(500k/20) + BESS(500k/10) = 25k + 50k = 75k
        for y in proj.years:
            assert math.isclose(y.depreciation, 75_000.0)

    def test_replacement_afa_reduces_tax(self) -> None:
        """Replacement AfA reduces taxable income and thus tax in post-replacement years."""
        common = dict(
            lifetime=15, inflation=0.0,
            revenues=[300_000.0] * 15,
            base_opex=50_000.0,
            capex_total=0.0, capex_pv=0.0, capex_bess=0.0,
            leverage=0.0, afa_pv=20, afa_bess=10,
        )
        proj_no_repl = _build_simple_projection(**common)
        proj_with_repl = _build_simple_projection(
            **common, replacement_cost=500_000.0, replacement_year=5,
        )
        # After replacement year, tax should be lower due to extra depreciation
        for y_idx in range(5, 14):  # years 6-14 (replacement AfA active)
            assert proj_with_repl.years[y_idx].total_tax < proj_no_repl.years[y_idx].total_tax

    def test_replacement_verlustvortrag(self) -> None:
        """Large replacement CAPEX can create negative CF, testing Verlustvortrag."""
        proj = _build_simple_projection(
            lifetime=5, inflation=0.0,
            revenues=[100_000.0] * 5,
            base_opex=50_000.0,
            capex_total=0.0, capex_pv=0.0, capex_bess=0.0,
            replacement_cost=1_000_000.0, replacement_year=2,
            leverage=0.0, messzahl=0.035, hebesatz=400.0,
            afa_pv=20, afa_bess=10,
        )
        # Year 2 has a massive CAPEX outflow
        assert proj.years[1].capex == 1_000_000.0
        # With 1M/10=100k depreciation reducing 50k income, taxable is negative
        # → Verlustvortrag should propagate


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


# ---------------------------------------------------------------------------
# BESS Optimization Fee tests (Feature 03)
# ---------------------------------------------------------------------------


class TestOptimizationFee:
    """Verify BESS optimization fee as revenue-dependent OPEX."""

    def test_fee_zero_no_effect(self) -> None:
        """Fee = 0%: OPEX is unchanged regardless of BESS revenue."""
        base_opex = 50_000.0
        proj = _build_simple_projection(
            lifetime=3,
            base_opex=base_opex,
            inflation=0.0,
            capex_pv=0.0,
            capex_bess=0.0,
            leverage=0.0,
            messzahl=0.0,
            optimization_fee_pct=0.0,
            annual_bess_spot_revenues=[100_000.0, 100_000.0, 100_000.0],
        )
        for y in proj.years:
            assert math.isclose(y.opex, base_opex)

    def test_fee_5_pct_with_100k_revenue(self) -> None:
        """Fee = 5%, BESS spot revenue = 100k EUR → optimization OPEX = 5k EUR."""
        base_opex = 50_000.0
        fee_pct = 5.0
        bess_rev = 100_000.0
        proj = _build_simple_projection(
            lifetime=3,
            base_opex=base_opex,
            inflation=0.0,
            capex_pv=0.0,
            capex_bess=0.0,
            leverage=0.0,
            messzahl=0.0,
            optimization_fee_pct=fee_pct,
            annual_bess_spot_revenues=[bess_rev] * 3,
        )
        expected_opex = base_opex + bess_rev * fee_pct / 100.0  # 50k + 5k = 55k
        for y in proj.years:
            assert math.isclose(y.opex, expected_opex)

    def test_fee_not_inflated(self) -> None:
        """The optimization fee is NOT subject to inflation (revenue already current-year).

        Base OPEX is inflated, but the fee portion stays proportional to
        current-year BESS revenue.
        """
        base_opex = 50_000.0
        fee_pct = 10.0
        inflation = 0.03
        bess_rev = 80_000.0  # Same BESS revenue each year (for simplicity)

        proj = _build_simple_projection(
            lifetime=3,
            base_opex=base_opex,
            inflation=inflation,
            capex_pv=0.0,
            capex_bess=0.0,
            leverage=0.0,
            messzahl=0.0,
            optimization_fee_pct=fee_pct,
            annual_bess_spot_revenues=[bess_rev] * 3,
        )

        fee_amount = bess_rev * fee_pct / 100.0  # 8_000 EUR, same each year
        for y in proj.years:
            inflated_base = inflate_value(base_opex, inflation, y.year)
            expected = inflated_base + fee_amount
            assert math.isclose(y.opex, expected, rel_tol=1e-9)

    def test_fee_pv_only_no_bess_revenue(self) -> None:
        """PV-only scenario: BESS spot revenue = 0 → fee adds nothing to OPEX."""
        base_opex = 50_000.0
        proj = _build_simple_projection(
            lifetime=3,
            base_opex=base_opex,
            inflation=0.0,
            capex_pv=0.0,
            capex_bess=0.0,
            leverage=0.0,
            messzahl=0.0,
            optimization_fee_pct=5.0,
            annual_bess_spot_revenues=[0.0, 0.0, 0.0],
        )
        for y in proj.years:
            assert math.isclose(y.opex, base_opex)

    def test_fee_none_bess_revenues(self) -> None:
        """If annual_bess_spot_revenues is None, fee has no effect even with fee_pct > 0."""
        base_opex = 50_000.0
        proj = _build_simple_projection(
            lifetime=3,
            base_opex=base_opex,
            inflation=0.0,
            capex_pv=0.0,
            capex_bess=0.0,
            leverage=0.0,
            messzahl=0.0,
            optimization_fee_pct=5.0,
            annual_bess_spot_revenues=None,
        )
        for y in proj.years:
            assert math.isclose(y.opex, base_opex)

    def test_fee_reduces_equity_cf(self) -> None:
        """Higher optimization fee should reduce equity CF."""
        common = dict(
            lifetime=3,
            revenues=[200_000.0] * 3,
            base_opex=50_000.0,
            inflation=0.0,
            capex_total=0.0,
            capex_pv=0.0,
            capex_bess=0.0,
            leverage=0.0,
            messzahl=0.0,
            annual_bess_spot_revenues=[100_000.0] * 3,
        )
        proj_no_fee = _build_simple_projection(**common, optimization_fee_pct=0.0)
        proj_with_fee = _build_simple_projection(**common, optimization_fee_pct=10.0)

        for i in range(3):
            assert proj_with_fee.equity_cashflows[i] < proj_no_fee.equity_cashflows[i]

    def test_fee_varies_with_yearly_bess_revenue(self) -> None:
        """Fee adapts to varying BESS revenue per year."""
        base_opex = 50_000.0
        fee_pct = 5.0
        bess_revenues = [100_000.0, 50_000.0, 200_000.0]
        proj = _build_simple_projection(
            lifetime=3,
            base_opex=base_opex,
            inflation=0.0,
            capex_pv=0.0,
            capex_bess=0.0,
            leverage=0.0,
            messzahl=0.0,
            optimization_fee_pct=fee_pct,
            annual_bess_spot_revenues=bess_revenues,
        )

        for i, y in enumerate(proj.years):
            expected_fee = bess_revenues[i] * fee_pct / 100.0
            expected_opex = base_opex + expected_fee
            assert math.isclose(y.opex, expected_opex)


# ---------------------------------------------------------------------------
# Debt-financed BESS replacement (FIX-S2-13)
# ---------------------------------------------------------------------------


class TestReplacementDebtFinancing:
    """Tests for BESS replacement financed with debt (FIX-S2-13).

    When replacement_leverage_pct > 0, only the equity portion of the
    replacement cost reduces the equity cashflow.  The debt portion is
    added to the debt schedule and serviced via increased debt payments.
    """

    def test_equity_cf_only_reduced_by_equity_share(self) -> None:
        """Equity CF in replacement year drops by equity share, not full cost."""
        repl_cost = 500_000.0
        repl_year = 3
        leverage = 75.0  # 75% debt → equity share = 125 000 €
        equity_share = repl_cost * (1.0 - leverage / 100.0)

        proj_no_repl = _build_simple_projection(
            lifetime=5, inflation=0.0,
            capex_total=0.0, capex_pv=0.0, capex_bess=0.0,
            leverage=0.0, messzahl=0.0,
        )
        proj_with_repl = _build_simple_projection(
            lifetime=5, inflation=0.0,
            replacement_cost=repl_cost, replacement_year=repl_year,
            replacement_leverage=leverage,
            capex_total=0.0, capex_pv=0.0, capex_bess=0.0,
            leverage=0.0, messzahl=0.0,
        )

        repl_idx = repl_year - 1
        # The equity CF reduction in the replacement year should be close to
        # the equity share (plus debt service for replacement loan in that year,
        # minus any tax effects from the new AfA)
        diff = proj_no_repl.years[repl_idx].equity_cf - proj_with_repl.years[repl_idx].equity_cf
        # Diff must be > equity share (because debt service starts too)
        # but << full replacement cost
        assert diff < repl_cost, "Equity CF should not drop by full replacement cost"
        assert diff >= equity_share - 1.0, "Equity CF should drop by at least the equity share"

    def test_full_equity_vs_debt_financed_replacement_year(self) -> None:
        """With debt financing, equity CF in replacement year is higher than 100% equity."""
        repl_cost = 500_000.0
        repl_year = 3

        proj_100pct_equity = _build_simple_projection(
            lifetime=5, inflation=0.0,
            replacement_cost=repl_cost, replacement_year=repl_year,
            replacement_leverage=0.0,
            capex_total=0.0, capex_pv=0.0, capex_bess=0.0,
            leverage=0.0, messzahl=0.0,
        )
        proj_75pct_debt = _build_simple_projection(
            lifetime=5, inflation=0.0,
            replacement_cost=repl_cost, replacement_year=repl_year,
            replacement_leverage=75.0,
            capex_total=0.0, capex_pv=0.0, capex_bess=0.0,
            leverage=0.0, messzahl=0.0,
        )

        repl_idx = repl_year - 1
        # With debt financing, year-of-replacement equity CF is less negative
        assert proj_75pct_debt.years[repl_idx].equity_cf > proj_100pct_equity.years[repl_idx].equity_cf

    def test_debt_service_increases_after_replacement(self) -> None:
        """Debt service increases from the replacement year onward."""
        repl_cost = 500_000.0
        repl_year = 3
        proj = _build_simple_projection(
            lifetime=5, inflation=0.0,
            replacement_cost=repl_cost, replacement_year=repl_year,
            replacement_leverage=75.0,
            capex_total=0.0, capex_pv=0.0, capex_bess=0.0,
            leverage=0.0, messzahl=0.0,
            tenor=5,
        )
        # Before replacement: no debt (leverage=0 on original CAPEX=0)
        for y in proj.years[:repl_year - 1]:
            assert y.debt_service == 0.0

        # From replacement year onward: debt service > 0
        for y in proj.years[repl_year - 1:]:
            assert y.debt_service > 0.0

    def test_debt_interest_and_repayment_sum_to_service(self) -> None:
        """Interest + repayment = debt_service in every year after replacement."""
        repl_cost = 500_000.0
        repl_year = 3
        proj = _build_simple_projection(
            lifetime=10, inflation=0.0,
            replacement_cost=repl_cost, replacement_year=repl_year,
            replacement_leverage=75.0,
            capex_total=1_000_000.0, capex_pv=500_000.0, capex_bess=500_000.0,
            leverage=75.0, messzahl=0.0,
            tenor=10,
        )
        for y in proj.years:
            assert math.isclose(
                y.debt_interest + y.debt_repayment, y.debt_service, rel_tol=1e-9
            ), f"Year {y.year}: {y.debt_interest} + {y.debt_repayment} != {y.debt_service}"

    def test_zero_replacement_leverage_is_full_equity(self) -> None:
        """With replacement_leverage=0, behavior is identical to full equity financing."""
        repl_cost = 500_000.0
        repl_year = 3
        proj_default = _build_simple_projection(
            lifetime=5, inflation=0.0,
            replacement_cost=repl_cost, replacement_year=repl_year,
            capex_total=0.0, capex_pv=0.0, capex_bess=0.0,
            leverage=0.0, messzahl=0.0,
        )
        proj_explicit_zero = _build_simple_projection(
            lifetime=5, inflation=0.0,
            replacement_cost=repl_cost, replacement_year=repl_year,
            replacement_leverage=0.0,
            capex_total=0.0, capex_pv=0.0, capex_bess=0.0,
            leverage=0.0, messzahl=0.0,
        )
        for y1, y2 in zip(proj_default.years, proj_explicit_zero.years):
            assert math.isclose(y1.equity_cf, y2.equity_cf)
            assert math.isclose(y1.debt_service, y2.debt_service)

    def test_project_cf_unchanged_by_leverage(self) -> None:
        """Project CF (pre-leverage) is unaffected by how replacement is financed."""
        repl_cost = 500_000.0
        repl_year = 3
        proj_equity = _build_simple_projection(
            lifetime=5, inflation=0.0,
            replacement_cost=repl_cost, replacement_year=repl_year,
            replacement_leverage=0.0,
            capex_total=0.0, capex_pv=0.0, capex_bess=0.0,
            leverage=0.0, messzahl=0.0,
        )
        proj_debt = _build_simple_projection(
            lifetime=5, inflation=0.0,
            replacement_cost=repl_cost, replacement_year=repl_year,
            replacement_leverage=75.0,
            capex_total=0.0, capex_pv=0.0, capex_bess=0.0,
            leverage=0.0, messzahl=0.0,
        )
        for y1, y2 in zip(proj_equity.years, proj_debt.years):
            assert math.isclose(y1.project_cf, y2.project_cf), (
                f"Year {y1.year}: project CF {y1.project_cf} != {y2.project_cf}"
            )

    def test_capex_column_shows_full_replacement_cost(self) -> None:
        """The capex field always shows the full replacement cost, regardless of leverage."""
        repl_cost = 500_000.0
        repl_year = 3
        proj = _build_simple_projection(
            lifetime=5, inflation=0.0,
            replacement_cost=repl_cost, replacement_year=repl_year,
            replacement_leverage=75.0,
            capex_total=0.0, capex_pv=0.0, capex_bess=0.0,
            leverage=0.0, messzahl=0.0,
        )
        assert math.isclose(proj.years[repl_year - 1].capex, repl_cost)

    def test_replacement_with_original_debt(self) -> None:
        """Both original project debt and replacement debt combine correctly."""
        capex = 1_000_000.0
        leverage = 75.0
        tenor = 18
        repl_cost = 500_000.0
        repl_year = 10

        proj = _build_simple_projection(
            lifetime=25, inflation=0.0,
            revenues=[300_000.0] * 25,
            capex_total=capex, capex_pv=500_000.0, capex_bess=500_000.0,
            leverage=leverage, tenor=tenor,
            replacement_cost=repl_cost, replacement_year=repl_year,
            replacement_leverage=leverage,
            messzahl=0.0,
        )

        # Before replacement: debt service = original annuity (constant)
        ds_before = proj.years[0].debt_service
        for y in proj.years[1:repl_year - 1]:
            assert math.isclose(y.debt_service, ds_before, rel_tol=1e-9)

        # From replacement year: debt service increases
        for y in proj.years[repl_year - 1:tenor]:
            assert y.debt_service > ds_before + 1.0, (
                f"Year {y.year}: ds {y.debt_service} not > original {ds_before}"
            )


# ---------------------------------------------------------------------------
# PV-only depreciation (AfA) tests – no BESS
# ---------------------------------------------------------------------------


class TestPvOnlyDepreciation:
    """Verify AfA is constant across all years when BESS CAPEX is zero.

    In a PV-only case (no BESS), the depreciation should equal
    capex_pv / afa_years_pv for years within the PV AfA period and 0
    afterwards.  It must never change at the BESS AfA boundary (e.g.
    year 10) because there is no BESS to depreciate.
    """

    def test_pv_only_afa_constant_within_pv_period(self) -> None:
        """With capex_bess=0, depreciation must be constant for years 1..afa_pv."""
        capex_pv = 1_000_000.0
        afa_pv = 20
        afa_bess = 10
        lifetime = 25
        expected_depr = capex_pv / afa_pv  # 50 000 €/year

        proj = _build_simple_projection(
            lifetime=lifetime,
            revenues=[200_000.0] * lifetime,
            capex_total=capex_pv,
            capex_pv=capex_pv,
            capex_bess=0.0,
            afa_pv=afa_pv,
            afa_bess=afa_bess,
            leverage=0.0,
            messzahl=0.0,
        )

        # Years 1 through afa_pv: constant PV depreciation
        for y in proj.years[:afa_pv]:
            assert math.isclose(y.depreciation, expected_depr), (
                f"Year {y.year}: depreciation {y.depreciation} != {expected_depr}"
            )

        # Years after afa_pv: depreciation must be 0
        for y in proj.years[afa_pv:]:
            assert math.isclose(y.depreciation, 0.0), (
                f"Year {y.year}: depreciation {y.depreciation} != 0.0"
            )

    def test_pv_only_no_afa_jump_at_bess_afa_boundary(self) -> None:
        """Depreciation must not change at the BESS AfA boundary (year 10/11)."""
        capex_pv = 1_000_000.0
        afa_bess = 10
        lifetime = 15

        proj = _build_simple_projection(
            lifetime=lifetime,
            revenues=[200_000.0] * lifetime,
            capex_total=capex_pv,
            capex_pv=capex_pv,
            capex_bess=0.0,
            afa_pv=20,
            afa_bess=afa_bess,
            leverage=0.0,
            messzahl=0.0,
        )

        # Depreciation at the BESS AfA boundary should be identical
        depr_before = proj.years[afa_bess - 1].depreciation  # year 10
        depr_after = proj.years[afa_bess].depreciation        # year 11
        assert math.isclose(depr_before, depr_after), (
            f"Depreciation jump at BESS AfA boundary: "
            f"year {afa_bess}={depr_before}, year {afa_bess + 1}={depr_after}"
        )
