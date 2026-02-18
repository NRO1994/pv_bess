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
        Annual debt service payment (0.0 if *year* is outside the loan tenor).
    """
    if year < 1 or year > len(schedule.interest_payments):
        return 0.0
    return schedule.annual_payment
