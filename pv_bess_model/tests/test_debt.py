"""Tests for finance/debt.py – Annuity loan model.

Reference: 75 % leverage on 5 750 000 € = 4 312 500 € loan,
  4.5 % interest, 18 years.  Annuity ≈ 354 646.62 €/a.
"""

from __future__ import annotations

import math

import numpy_financial as npf
import pytest

from pv_bess_model.finance.debt import (
    add_replacement_debt,
    build_annuity_schedule,
    calculate_annuity,
    get_debt_components,
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


# ---------------------------------------------------------------------------
# add_replacement_debt (FIX-S2-13)
# ---------------------------------------------------------------------------

# Replacement parameters
REPL_COST = 500_000.0
REPL_LEVERAGE = 75.0
REPL_LOAN = REPL_COST * REPL_LEVERAGE / 100.0  # 375 000 €
REPL_YEAR = 10  # replacement in project year 10
LIFETIME = 25  # project lifetime


class TestAddReplacementDebt:
    """Tests for add_replacement_debt (FIX-S2-13)."""

    def _base_schedule(self) -> AnnuitySchedule:
        return build_annuity_schedule(CAPEX_TOTAL, LEVERAGE_PCT, RATE, TENOR)

    def test_combined_loan_amount(self) -> None:
        """Combined loan = original loan + replacement loan."""
        base = self._base_schedule()
        combined = add_replacement_debt(
            base, REPL_COST, REPL_LEVERAGE, RATE, REPL_YEAR, TENOR, LIFETIME,
        )
        assert math.isclose(combined.loan_amount, LOAN + REPL_LOAN)

    def test_schedule_length_at_least_original(self) -> None:
        """Combined schedule is at least as long as the original."""
        base = self._base_schedule()
        combined = add_replacement_debt(
            base, REPL_COST, REPL_LEVERAGE, RATE, REPL_YEAR, TENOR, LIFETIME,
        )
        assert len(combined.interest_payments) >= TENOR

    def test_payments_unchanged_before_replacement(self) -> None:
        """Before replacement year, debt service equals original schedule."""
        base = self._base_schedule()
        combined = add_replacement_debt(
            base, REPL_COST, REPL_LEVERAGE, RATE, REPL_YEAR, TENOR, LIFETIME,
        )
        for y in range(1, REPL_YEAR):
            orig_svc = get_debt_service(base, y)
            comb_svc = get_debt_service(combined, y)
            assert math.isclose(orig_svc, comb_svc, rel_tol=1e-9), (
                f"Year {y}: {orig_svc} vs {comb_svc}"
            )

    def test_payments_increase_from_replacement_year(self) -> None:
        """From replacement year onward, debt service increases."""
        base = self._base_schedule()
        combined = add_replacement_debt(
            base, REPL_COST, REPL_LEVERAGE, RATE, REPL_YEAR, TENOR, LIFETIME,
        )
        for y in range(REPL_YEAR, TENOR + 1):
            orig_svc = get_debt_service(base, y)
            comb_svc = get_debt_service(combined, y)
            assert comb_svc > orig_svc, f"Year {y}: combined {comb_svc} not > original {orig_svc}"

    def test_replacement_principal_sums_to_loan(self) -> None:
        """Sum of replacement principal = replacement loan amount."""
        base = self._base_schedule()
        combined = add_replacement_debt(
            base, REPL_COST, REPL_LEVERAGE, RATE, REPL_YEAR, TENOR, LIFETIME,
        )
        total_principal_combined = sum(combined.principal_payments)
        total_principal_orig = sum(base.principal_payments)
        repl_principal = total_principal_combined - total_principal_orig
        assert math.isclose(repl_principal, REPL_LOAN, rel_tol=1e-6)

    def test_remaining_tenor_calculation(self) -> None:
        """Replacement tenor = min(remaining_project_life, loan_tenor).

        With LIFETIME=25, REPL_YEAR=10, TENOR=18:
        remaining_project = 25 - 10 + 1 = 16, min(16, 18) = 16.
        Replacement runs years 10..25, schedule length = max(18, 25) = 25.
        """
        base = self._base_schedule()
        remaining_project = LIFETIME - REPL_YEAR + 1  # 16
        expected_repl_tenor = min(remaining_project, TENOR)  # min(16, 18) = 16
        combined = add_replacement_debt(
            base, REPL_COST, REPL_LEVERAGE, RATE, REPL_YEAR, TENOR, LIFETIME,
        )
        # Replacement runs from year 10 for 16 years → ends at year 25
        assert len(combined.interest_payments) == LIFETIME
        # After project lifetime ends, no more debt service
        assert get_debt_service(combined, LIFETIME + 1) == 0.0

    def test_replacement_after_original_tenor_expires(self) -> None:
        """If replacement year > original tenor, schedule extends.

        late_repl_year=20, LIFETIME=25, TENOR=18:
        remaining_project = 25 - 20 + 1 = 6, min(6, 18) = 6.
        Replacement runs years 20..25.
        """
        base = self._base_schedule()
        late_repl_year = TENOR + 2  # year 20
        repl_tenor = min(LIFETIME - late_repl_year + 1, TENOR)  # min(6, 18) = 6
        combined = add_replacement_debt(
            base, REPL_COST, REPL_LEVERAGE, RATE, late_repl_year, TENOR, LIFETIME,
        )
        # schedule length = max(18, 20-1+6) = 25
        assert len(combined.interest_payments) == LIFETIME
        # Before replacement: years 1-18 = original, year 19 = 0
        assert get_debt_service(combined, TENOR + 1) == 0.0
        # Year 20 = replacement loan starts
        assert get_debt_service(combined, late_repl_year) > 0.0
        # Year 25 = last replacement payment
        assert get_debt_service(combined, LIFETIME) > 0.0
        # Year 26 = nothing
        assert get_debt_service(combined, LIFETIME + 1) == 0.0

    def test_zero_leverage_returns_original(self) -> None:
        """Zero replacement leverage returns the original schedule unchanged."""
        base = self._base_schedule()
        combined = add_replacement_debt(
            base, REPL_COST, 0.0, RATE, REPL_YEAR, TENOR, LIFETIME,
        )
        assert combined is base

    def test_zero_replacement_cost_returns_original(self) -> None:
        """Zero replacement cost returns the original schedule unchanged."""
        base = self._base_schedule()
        combined = add_replacement_debt(
            base, 0.0, REPL_LEVERAGE, RATE, REPL_YEAR, TENOR, LIFETIME,
        )
        assert combined is base

    def test_get_debt_components_combined_schedule(self) -> None:
        """get_debt_components works correctly on the combined schedule."""
        base = self._base_schedule()
        combined = add_replacement_debt(
            base, REPL_COST, REPL_LEVERAGE, RATE, REPL_YEAR, TENOR, LIFETIME,
        )
        for y in range(1, len(combined.interest_payments) + 1):
            interest, repayment, total = get_debt_components(combined, y)
            assert math.isclose(interest + repayment, total, rel_tol=1e-9)
            assert interest >= 0.0
            assert repayment >= 0.0

    def test_final_balance_near_zero(self) -> None:
        """Remaining balance at end of combined schedule is approximately 0."""
        base = self._base_schedule()
        combined = add_replacement_debt(
            base, REPL_COST, REPL_LEVERAGE, RATE, REPL_YEAR, TENOR, LIFETIME,
        )
        assert combined.remaining_balance[-1] < 0.01

    def test_tenor_capped_by_project_lifetime(self) -> None:
        """Replacement tenor is capped so loan ends at project end.

        Replacement in year 22 of a 25-year project with loan_tenor=18:
        remaining_project = 25 - 22 + 1 = 4, min(4, 18) = 4.
        """
        base = self._base_schedule()
        late_year = 22
        short_lifetime = 25
        combined = add_replacement_debt(
            base, REPL_COST, REPL_LEVERAGE, RATE, late_year, TENOR, short_lifetime,
        )
        # Replacement loan should run for 4 years (years 22-25)
        # No debt service after project end
        assert get_debt_service(combined, short_lifetime) > 0.0
        assert get_debt_service(combined, short_lifetime + 1) == 0.0
        # Principal of replacement loan fully repaid
        repl_principal = sum(combined.principal_payments) - sum(base.principal_payments)
        assert math.isclose(repl_principal, REPL_LOAN, rel_tol=1e-6)

    def test_tenor_uses_loan_tenor_when_shorter(self) -> None:
        """When loan tenor < remaining project life, loan tenor is used.

        Replacement in year 2 of a 25-year project with loan_tenor=5:
        remaining_project = 25 - 2 + 1 = 24, min(24, 5) = 5.
        """
        base = build_annuity_schedule(CAPEX_TOTAL, LEVERAGE_PCT, RATE, 5)
        short_tenor = 5
        combined = add_replacement_debt(
            base, REPL_COST, REPL_LEVERAGE, RATE, 2, short_tenor, LIFETIME,
        )
        # Replacement runs years 2..6, original runs years 1..5
        # Combined schedule length = max(5, 2-1+5) = 6
        assert len(combined.interest_payments) == 6
        # Year 7 = nothing
        assert get_debt_service(combined, 7) == 0.0
