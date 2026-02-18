"""Tests for finance/debt.py – Annuity loan model.

Reference: 75 % leverage on 5 750 000 € = 4 312 500 € loan,
  4.5 % interest, 18 years.  Annuity ≈ 354 646.62 €/a.
"""

from __future__ import annotations

import math

import numpy_financial as npf
import pytest

from pv_bess_model.finance.debt import (
    build_annuity_schedule,
    calculate_annuity,
    get_debt_service,
)

CAPEX_TOTAL = 5_750_000.0
LEVERAGE_PCT = 75.0
RATE = 0.045
TENOR = 18
LOAN = CAPEX_TOTAL * LEVERAGE_PCT / 100.0  # 4 312 500 €
ANNUITY = abs(float(npf.pmt(RATE, TENOR, LOAN)))


# ---------------------------------------------------------------------------
# calculate_annuity
# ---------------------------------------------------------------------------


class TestCalculateAnnuity:
    """Tests for the standalone annuity formula."""

    def test_reference_annuity(self) -> None:
        """Annuity must match numpy_financial.pmt for the reference case."""
        result = calculate_annuity(LOAN, RATE, TENOR)
        assert math.isclose(result, ANNUITY, rel_tol=1e-9)

    def test_zero_loan(self) -> None:
        """Zero loan amount returns 0."""
        assert calculate_annuity(0.0, RATE, TENOR) == 0.0

    def test_zero_tenor(self) -> None:
        """Zero tenor returns 0."""
        assert calculate_annuity(LOAN, RATE, 0) == 0.0

    def test_negative_loan(self) -> None:
        """Negative loan amount returns 0."""
        assert calculate_annuity(-100_000.0, RATE, TENOR) == 0.0


# ---------------------------------------------------------------------------
# build_annuity_schedule
# ---------------------------------------------------------------------------


class TestBuildAnnuitySchedule:
    """Tests for the full annuity schedule builder."""

    def test_loan_amount(self) -> None:
        """Loan = 5 750 000 × 0.75 = 4 312 500 €."""
        sched = build_annuity_schedule(CAPEX_TOTAL, LEVERAGE_PCT, RATE, TENOR)
        assert math.isclose(sched.loan_amount, 4_312_500.0)

    def test_annual_payment_constant(self) -> None:
        """Annual payment must equal the calculated annuity."""
        sched = build_annuity_schedule(CAPEX_TOTAL, LEVERAGE_PCT, RATE, TENOR)
        assert math.isclose(sched.annual_payment, ANNUITY, rel_tol=1e-9)

    def test_schedule_length(self) -> None:
        """All schedule lists must have length = tenor."""
        sched = build_annuity_schedule(CAPEX_TOTAL, LEVERAGE_PCT, RATE, TENOR)
        assert len(sched.interest_payments) == TENOR
        assert len(sched.principal_payments) == TENOR
        assert len(sched.remaining_balance) == TENOR

    def test_principal_sum_equals_loan(self) -> None:
        """Sum of all principal repayments must equal the loan amount."""
        sched = build_annuity_schedule(CAPEX_TOTAL, LEVERAGE_PCT, RATE, TENOR)
        total_principal = sum(sched.principal_payments)
        assert math.isclose(total_principal, sched.loan_amount, rel_tol=1e-6)

    def test_interest_sum_positive(self) -> None:
        """Total interest paid must be positive (cost of borrowing)."""
        sched = build_annuity_schedule(CAPEX_TOTAL, LEVERAGE_PCT, RATE, TENOR)
        total_interest = sum(sched.interest_payments)
        assert total_interest > 0.0

    def test_interest_plus_principal_equals_annuity(self) -> None:
        """Each year: interest + principal = annuity."""
        sched = build_annuity_schedule(CAPEX_TOTAL, LEVERAGE_PCT, RATE, TENOR)
        for i in range(TENOR):
            total = sched.interest_payments[i] + sched.principal_payments[i]
            assert math.isclose(total, sched.annual_payment, rel_tol=1e-9)

    def test_final_balance_near_zero(self) -> None:
        """Remaining balance after last year must be approximately 0."""
        sched = build_annuity_schedule(CAPEX_TOTAL, LEVERAGE_PCT, RATE, TENOR)
        assert sched.remaining_balance[-1] < 0.01

    def test_interest_decreases_over_time(self) -> None:
        """Interest portion must decrease year-over-year (standard annuity)."""
        sched = build_annuity_schedule(CAPEX_TOTAL, LEVERAGE_PCT, RATE, TENOR)
        for i in range(1, TENOR):
            assert sched.interest_payments[i] < sched.interest_payments[i - 1]

    def test_principal_increases_over_time(self) -> None:
        """Principal portion must increase year-over-year (standard annuity)."""
        sched = build_annuity_schedule(CAPEX_TOTAL, LEVERAGE_PCT, RATE, TENOR)
        for i in range(1, TENOR):
            assert sched.principal_payments[i] > sched.principal_payments[i - 1]

    def test_leverage_zero_percent(self) -> None:
        """0 % leverage → no loan, empty schedule."""
        sched = build_annuity_schedule(CAPEX_TOTAL, 0.0, RATE, TENOR)
        assert sched.loan_amount == 0.0
        assert sched.annual_payment == 0.0
        assert sched.interest_payments == []
        assert sched.principal_payments == []

    def test_leverage_hundred_percent(self) -> None:
        """100 % leverage → loan = CAPEX."""
        sched = build_annuity_schedule(CAPEX_TOTAL, 100.0, RATE, TENOR)
        assert math.isclose(sched.loan_amount, CAPEX_TOTAL)
        assert sched.annual_payment > 0.0
        total_principal = sum(sched.principal_payments)
        assert math.isclose(total_principal, CAPEX_TOTAL, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# get_debt_service
# ---------------------------------------------------------------------------


class TestGetDebtService:
    """Tests for get_debt_service."""

    def test_year_zero_no_debt_service(self) -> None:
        """Year 0 is CAPEX year – no debt service."""
        sched = build_annuity_schedule(CAPEX_TOTAL, LEVERAGE_PCT, RATE, TENOR)
        assert get_debt_service(sched, 0) == 0.0

    def test_year_within_tenor(self) -> None:
        """Years 1..18 should return the annuity."""
        sched = build_annuity_schedule(CAPEX_TOTAL, LEVERAGE_PCT, RATE, TENOR)
        for y in range(1, TENOR + 1):
            assert math.isclose(get_debt_service(sched, y), ANNUITY, rel_tol=1e-9)

    def test_year_after_tenor(self) -> None:
        """Year 19+ (beyond tenor) should return 0."""
        sched = build_annuity_schedule(CAPEX_TOTAL, LEVERAGE_PCT, RATE, TENOR)
        assert get_debt_service(sched, TENOR + 1) == 0.0
        assert get_debt_service(sched, 25) == 0.0

    def test_negative_year(self) -> None:
        """Negative year should return 0."""
        sched = build_annuity_schedule(CAPEX_TOTAL, LEVERAGE_PCT, RATE, TENOR)
        assert get_debt_service(sched, -1) == 0.0
