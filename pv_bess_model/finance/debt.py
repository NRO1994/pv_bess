"""Annuity loan model: principal, interest, and annual debt service.

Implements a simple annuity loan where the annual payment (debt service) is constant
over the loan tenor. Each annual payment is split into interest and principal
repayment components.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy_financial as npf


@dataclass(frozen=True)
class AnnuitySchedule:
    """Full annuity loan schedule over the loan tenor.

    Attributes:
        loan_amount: Initial loan principal in euros.
        annual_payment: Constant annual annuity payment (positive = outflow).
        interest_payments: Interest portion per year (list, length = tenor).
        principal_payments: Principal portion per year (list, length = tenor).
        remaining_balance: Outstanding balance at end of each year (list, length = tenor).
    """

    loan_amount: float
    annual_payment: float
    interest_payments: list[float]
    principal_payments: list[float]
    remaining_balance: list[float]


def calculate_annuity(
    loan_amount: float,
    annual_interest_rate: float,
    tenor_years: int,
) -> float:
    """Calculate the constant annual annuity payment for a loan.

    Args:
        loan_amount: Loan principal in euros.
        annual_interest_rate: Annual interest rate as a decimal (e.g. 0.045 for 4.5 %).
        tenor_years: Loan tenor in years.

    Returns:
        Annual annuity payment as a positive value (cash outflow).
    """
    if loan_amount <= 0.0 or tenor_years <= 0:
        return 0.0
    return abs(float(npf.pmt(annual_interest_rate, tenor_years, loan_amount)))


def build_annuity_schedule(
    total_capex: float,
    leverage_pct: float,
    annual_interest_rate: float,
    tenor_years: int,
    depreciation_period:int
) -> AnnuitySchedule:
    """Build the full year-by-year annuity schedule.

    Args:
        total_capex: Total project CAPEX in euros.
        leverage_pct: Debt leverage as percentage of total CAPEX (e.g. 75.0).
        annual_interest_rate: Annual interest rate as decimal (e.g. 0.045).
        tenor_years: Loan tenor in years.

    Returns:
        :class:`AnnuitySchedule` with per-year interest/principal split.
    """
    loan_amount = total_capex * leverage_pct / 100.0

    if loan_amount <= 0.0 or tenor_years <= 0:
        return AnnuitySchedule(
            loan_amount=0.0,
            annual_payment=0.0,
            interest_payments=[],
            principal_payments=[],
            remaining_balance=[],
        )

    annual_payment = calculate_annuity(loan_amount, annual_interest_rate, tenor_years)

    interest_payments: list[float] = []
    principal_payments: list[float] = []
    remaining_balance: list[float] = []
    balance = loan_amount

    for _ in range(tenor_years):
        interest = balance * annual_interest_rate
        principal = annual_payment - interest
        balance = balance - principal

        interest_payments.append(interest)
        principal_payments.append(principal)
        remaining_balance.append(max(balance, 0.0))

    return AnnuitySchedule(
        loan_amount=loan_amount,
        annual_payment=annual_payment,
        interest_payments=interest_payments,
        principal_payments=principal_payments,
        remaining_balance=remaining_balance,
    )


def get_debt_service(schedule: AnnuitySchedule, year: int) -> float:
    """Return the debt service payment for a given project year (1-indexed).

    Args:
        schedule: The annuity schedule.
        year: Project year (1-indexed). Year 0 is the CAPEX year with no debt service.

    Returns:
        Annual debt service payment (0.0 if *year* is outside the schedule).
    """
    if year < 1 or year > len(schedule.interest_payments):
        return 0.0
    idx = year - 1
    return schedule.interest_payments[idx] + schedule.principal_payments[idx]


def add_replacement_debt(
    existing_schedule: AnnuitySchedule,
    replacement_cost: float,
    leverage_pct: float,
    annual_interest_rate: float,
    replacement_year: int,
    loan_tenor_years: int,
    lifetime_years: int,
) -> AnnuitySchedule:
    """Add replacement debt to an existing annuity schedule.

    At ``replacement_year`` (1-indexed), a new loan is taken out for the
    debt-financed portion of the replacement cost.  The replacement loan
    tenor is ``min(remaining_project_lifetime, loan_tenor_years)`` so that
    the loan is fully repaid by the end of the project.

    The combined schedule extends to whichever loan ends later.

    Args:
        existing_schedule: Original project debt schedule.
        replacement_cost: Total BESS replacement cost in euros.
        leverage_pct: Debt share of the replacement cost (e.g. 75.0).
        annual_interest_rate: Annual interest rate for the replacement loan.
        replacement_year: Project year (1-indexed) of the replacement event.
        loan_tenor_years: Maximum loan tenor in years for the replacement loan.
        lifetime_years: Total project lifetime in years.  The replacement
            loan must be repaid by the end of the project.

    Returns:
        A new :class:`AnnuitySchedule` that combines original and replacement
        debt payments.  ``loan_amount`` reflects the sum of both loans.
        ``annual_payment`` is set to 0.0 since the combined payment is no
        longer constant — use per-year lists instead.
    """
    replacement_loan = replacement_cost * leverage_pct / 100.0

    if replacement_loan <= 0.0 or replacement_year < 1:
        return existing_schedule

    # Replacement tenor: the shorter of the configured loan tenor and the
    # remaining project lifetime, so the loan is fully amortised by project end.
    remaining_project_years = max(lifetime_years - replacement_year + 1, 1)
    remaining_tenor = min(remaining_project_years, loan_tenor_years)

    repl_annuity = calculate_annuity(replacement_loan, annual_interest_rate, remaining_tenor)

    # Build per-year arrays for the replacement loan
    repl_interest: list[float] = []
    repl_principal: list[float] = []
    repl_balance_list: list[float] = []
    balance = replacement_loan
    for _ in range(remaining_tenor):
        interest = balance * annual_interest_rate
        principal = repl_annuity - interest
        balance = balance - principal
        repl_interest.append(interest)
        repl_principal.append(principal)
        repl_balance_list.append(max(balance, 0.0))

    # Determine total schedule length: max of original end and replacement end
    repl_end_year = replacement_year - 1 + remaining_tenor  # 0-indexed end
    orig_len = len(existing_schedule.interest_payments)
    total_len = max(orig_len, repl_end_year)

    # Merge arrays
    combined_interest: list[float] = []
    combined_principal: list[float] = []
    combined_balance: list[float] = []

    for i in range(total_len):
        # Original component
        orig_i = existing_schedule.interest_payments[i] if i < orig_len else 0.0
        orig_p = existing_schedule.principal_payments[i] if i < orig_len else 0.0
        orig_b = existing_schedule.remaining_balance[i] if i < orig_len else 0.0

        # Replacement component (starts at replacement_year - 1, 0-indexed)
        repl_offset = i - (replacement_year - 1)
        if 0 <= repl_offset < remaining_tenor:
            r_i = repl_interest[repl_offset]
            r_p = repl_principal[repl_offset]
            r_b = repl_balance_list[repl_offset]
        else:
            r_i = 0.0
            r_p = 0.0
            r_b = 0.0

        combined_interest.append(orig_i + r_i)
        combined_principal.append(orig_p + r_p)
        combined_balance.append(orig_b + r_b)

    return AnnuitySchedule(
        loan_amount=existing_schedule.loan_amount + replacement_loan,
        annual_payment=0.0,  # no longer constant
        interest_payments=combined_interest,
        principal_payments=combined_principal,
        remaining_balance=combined_balance,
    )


def get_debt_components(
    schedule: AnnuitySchedule,
    year: int,
) -> tuple[float, float, float]:
    """Return the interest, principal repayment, and total debt service for a year.

    Args:
        schedule: The annuity schedule.
        year: Project year (1-indexed).

    Returns:
        A three-tuple ``(interest, repayment, total)`` where all values are 0.0
        when *year* falls outside the loan tenor.
    """
    if year < 1 or year > len(schedule.interest_payments):
        return (0.0, 0.0, 0.0)
    interest = schedule.interest_payments[year - 1]
    repayment = schedule.principal_payments[year - 1]
    total = interest + repayment
    return (interest, repayment, total)
