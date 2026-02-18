"""Annual cashflow projection: revenue, OPEX, debt service, tax, equity CF.

Builds a year-by-year cashflow table for the full project lifetime.
Year 0 carries the equity portion of CAPEX (negative). Years 1..N carry
operating cashflows: Revenue - OPEX - Debt Service - Tax = Equity CF.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pv_bess_model.finance.debt import AnnuitySchedule, get_debt_service
from pv_bess_model.finance.inflation import inflate_value
from pv_bess_model.finance.tax import calculate_tax_for_year


@dataclass
class AnnualCashflow:
    """Cashflow breakdown for a single project year."""

    year: int
    revenue: float
    opex: float
    debt_service: float
    depreciation: float
    gewerbesteuer: float
    project_cf: float
    equity_cf: float


@dataclass
class CashflowProjection:
    """Complete multi-year cashflow projection.

    Attributes:
        years: List of :class:`AnnualCashflow` (year 0 through lifetime).
        equity_cashflows: Array of equity CFs for IRR/NPV calculation.
        project_cashflows: Array of project CFs (pre-leverage) for Project IRR.
    """

    years: list[AnnualCashflow]
    equity_cashflows: np.ndarray
    project_cashflows: np.ndarray


def build_cashflow_projection(
    lifetime_years: int,
    annual_revenues: list[float],
    base_opex: float,
    inflation_rate: float,
    capex_total: float,
    capex_pv: float,
    capex_bess: float,
    debt_schedule: AnnuitySchedule,
    afa_years_pv: int,
    afa_years_bess: int,
    gewerbesteuer_messzahl: float,
    gewerbesteuer_hebesatz: float,
    replacement_cost: float = 0.0,
    replacement_year: int | None = None,
) -> CashflowProjection:
    """Build the complete annual cashflow projection.

    Args:
        lifetime_years: Project lifetime (number of operating years, 1-indexed).
        annual_revenues: Revenue per operating year (list of length ``lifetime_years``).
            Index 0 = year 1, index 1 = year 2, etc.
        base_opex: Base-year annual OPEX (before inflation).
        inflation_rate: Annual inflation rate as decimal.
        capex_total: Total project CAPEX.
        capex_pv: PV CAPEX (for depreciation).
        capex_bess: BESS CAPEX (for depreciation).
        debt_schedule: Annuity schedule from :func:`~pv_bess_model.finance.debt.build_annuity_schedule`.
        afa_years_pv: PV depreciation period in years.
        afa_years_bess: BESS depreciation period in years.
        gewerbesteuer_messzahl: GewSt Messzahl.
        gewerbesteuer_hebesatz: GewSt Hebesatz.
        replacement_cost: BESS replacement cost (added as OPEX in replacement year).
        replacement_year: Year of BESS replacement (1-indexed), or None.

    Returns:
        :class:`CashflowProjection` with per-year detail and summary arrays.
    """
    equity_cf_array = np.zeros(lifetime_years + 1)
    project_cf_array = np.zeros(lifetime_years + 1)
    yearly_results: list[AnnualCashflow] = []

    # Year 0: CAPEX outflow (equity portion)
    equity_investment = capex_total - debt_schedule.loan_amount
    equity_cf_array[0] = -equity_investment
    project_cf_array[0] = -capex_total

    yearly_results.append(
        AnnualCashflow(
            year=0,
            revenue=0.0,
            opex=0.0,
            debt_service=0.0,
            depreciation=0.0,
            gewerbesteuer=0.0,
            project_cf=-capex_total,
            equity_cf=-equity_investment,
        )
    )

    loss_carryforward = 0.0

    for y in range(1, lifetime_years + 1):
        revenue = annual_revenues[y - 1]

        # OPEX with inflation (year index for inflation = y, since year 0 = base)
        opex = inflate_value(base_opex, inflation_rate, y)

        # Add BESS replacement cost in the replacement year
        if replacement_year is not None and y == replacement_year:
            opex += replacement_cost

        debt_svc = get_debt_service(debt_schedule, y)

        # Tax calculation with Verlustvortrag
        tax_result = calculate_tax_for_year(
            revenue=revenue,
            opex=opex,
            capex_pv=capex_pv,
            capex_bess=capex_bess,
            afa_years_pv=afa_years_pv,
            afa_years_bess=afa_years_bess,
            project_year=y,
            loss_carryforward_in=loss_carryforward,
            messzahl=gewerbesteuer_messzahl,
            hebesatz=gewerbesteuer_hebesatz,
        )
        loss_carryforward = tax_result.loss_carryforward_remaining

        # Project CF (pre-leverage): Revenue - OPEX - Tax
        proj_cf = revenue - opex - tax_result.gewerbesteuer

        # Equity CF (post-leverage): Revenue - OPEX - Debt Service - Tax
        eq_cf = revenue - opex - debt_svc - tax_result.gewerbesteuer

        equity_cf_array[y] = eq_cf
        project_cf_array[y] = proj_cf

        yearly_results.append(
            AnnualCashflow(
                year=y,
                revenue=revenue,
                opex=opex,
                debt_service=debt_svc,
                depreciation=tax_result.depreciation_total,
                gewerbesteuer=tax_result.gewerbesteuer,
                project_cf=proj_cf,
                equity_cf=eq_cf,
            )
        )

    return CashflowProjection(
        years=yearly_results,
        equity_cashflows=equity_cf_array,
        project_cashflows=project_cf_array,
    )
