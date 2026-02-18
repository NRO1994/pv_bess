"""Tests for finance/cashflow.py – Annual cashflow projection.

Tests verify:
- Array length = lifetime + 1 (year 0 through N)
- CAPEX only in year 0
- Equity CF = Revenue - OPEX - Debt Service - Tax
- Verlustvortrag 3-year scenario: (-100k, +60k, +80k)
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
    """Tests for basic cashflow projection structure."""

    def test_array_length(self) -> None:
        """Arrays and year list must have length = lifetime + 1."""
        proj = _build_simple_projection(lifetime=25)
        assert len(proj.equity_cashflows) == 26
        assert len(proj.project_cashflows) == 26
        assert len(proj.years) == 26

    def test_year_indices(self) -> None:
        """Year objects must be indexed 0 through lifetime."""
        proj = _build_simple_projection(lifetime=5)
        for i, annual in enumerate(proj.years):
            assert annual.year == i

    def test_capex_only_in_year_zero(self) -> None:
        """Year 0 carries negative project CF (= -CAPEX), years 1+ are operating."""
        capex = 1_000_000.0
        proj = _build_simple_projection(capex_total=capex, leverage=0.0)
        assert proj.years[0].project_cf == -capex
        for y in proj.years[1:]:
            assert y.revenue > 0.0  # operating years have revenue


# ---------------------------------------------------------------------------
# Year 0 tests
# ---------------------------------------------------------------------------


class TestYearZero:
    """Tests for year 0 (investment year)."""

    def test_equity_cf_year0_no_debt(self) -> None:
        """No leverage → equity CF = -CAPEX."""
        capex = 2_000_000.0
        proj = _build_simple_projection(capex_total=capex, leverage=0.0)
        assert math.isclose(proj.equity_cashflows[0], -capex)

    def test_equity_cf_year0_with_debt(self) -> None:
        """75 % leverage → equity CF = -(CAPEX × 25 %) = -250 000."""
        capex = 1_000_000.0
        proj = _build_simple_projection(capex_total=capex, leverage=75.0)
        expected_equity = -(capex * 0.25)
        assert math.isclose(proj.equity_cashflows[0], expected_equity)

    def test_project_cf_year0(self) -> None:
        """Project CF year 0 = -CAPEX (always full CAPEX regardless of leverage)."""
        capex = 1_000_000.0
        proj = _build_simple_projection(capex_total=capex, leverage=75.0)
        assert math.isclose(proj.project_cashflows[0], -capex)

    def test_year0_no_revenue_no_opex(self) -> None:
        """Year 0 has zero revenue, OPEX, debt service, and tax."""
        proj = _build_simple_projection()
        y0 = proj.years[0]
        assert y0.revenue == 0.0
        assert y0.opex == 0.0
        assert y0.debt_service == 0.0
        assert y0.gewerbesteuer == 0.0


# ---------------------------------------------------------------------------
# Equity CF identity tests
# ---------------------------------------------------------------------------


class TestEquityCfIdentity:
    """Verify Equity CF = Revenue - OPEX - Debt Service - GewSt."""

    def test_equity_cf_no_tax_no_debt(self) -> None:
        """Without debt and with zero tax, equity CF = revenue - opex."""
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
        for y in proj.years[1:]:
            expected = y.revenue - y.opex - y.debt_service - y.gewerbesteuer
            assert math.isclose(y.equity_cf, expected, rel_tol=1e-9)

    def test_equity_cf_with_debt(self) -> None:
        """With debt, equity CF = revenue - opex - debt_service - tax."""
        proj = _build_simple_projection(
            lifetime=3,
            revenues=[300_000.0] * 3,
            base_opex=50_000.0,
            inflation=0.0,
            capex_total=1_000_000.0,
            leverage=75.0,
        )
        for y in proj.years[1:]:
            expected = y.revenue - y.opex - y.debt_service - y.gewerbesteuer
            assert math.isclose(y.equity_cf, expected, rel_tol=1e-9)

    def test_equity_cf_array_matches_year_objects(self) -> None:
        """equity_cashflows array must match AnnualCashflow.equity_cf."""
        proj = _build_simple_projection(lifetime=5)
        for i, annual in enumerate(proj.years):
            assert math.isclose(proj.equity_cashflows[i], annual.equity_cf, rel_tol=1e-9)


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
        for y in proj.years[1:]:
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
        for y in proj.years[1:]:
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
        # Year 1: loss, no GewSt
        assert proj.years[1].gewerbesteuer == 0.0

        # Year 2: positive taxable but absorbed by carry-forward
        assert proj.years[2].gewerbesteuer == 0.0

        # Year 3: 40k adjusted taxable income → GewSt = 40 000 × 0.035 × 400/100 = 5 600
        expected_gewst = calculate_gewerbesteuer(40_000.0, messzahl, hebesatz)
        assert math.isclose(expected_gewst, 5_600.0)
        assert math.isclose(proj.years[3].gewerbesteuer, 5_600.0)

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
        # Year 1: equity CF = 100k - 200k - 0 - 0 = -100k
        assert math.isclose(proj.years[1].equity_cf, -100_000.0)

        # Year 2: equity CF = 260k - 200k - 0 - 0 = 60k
        assert math.isclose(proj.years[2].equity_cf, 60_000.0)

        # Year 3: equity CF = 280k - 200k - 0 - 5 600 = 74 400
        assert math.isclose(proj.years[3].equity_cf, 280_000.0 - 200_000.0 - 5_600.0)


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
        y1 = proj.years[1]
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
        # Year 11: BESS AfA ended, only PV remains
        y11 = proj.years[11]
        expected = 1_000_000.0 / 20  # only PV
        assert math.isclose(y11.depreciation, expected)
