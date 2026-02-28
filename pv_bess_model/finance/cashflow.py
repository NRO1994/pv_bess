"""Annual cashflow projection: revenue, OPEX, debt service, tax, equity CF.

Builds a year-by-year cashflow table for the full project lifetime.
CAPEX and first-year OPEX both fall in Year 1 (the commissioning year).
There is no separate Year 0.

Equity CF Year 1 = Revenue - OPEX - Equity Investment - Debt Service - Tax.
Equity CF Year 2+ = Revenue - OPEX - Debt Service - Tax.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pv_bess_model.config.defaults import (
    DEFAULT_KOERPERSCHAFTSTEUER_PCT,
    DEFAULT_SOLIDARITAETSZUSCHLAG_PCT,
)
from pv_bess_model.finance.debt import (
    AnnuitySchedule,
    add_replacement_debt,
    get_debt_components,
    get_debt_service,
)
from pv_bess_model.finance.inflation import inflate_value
from pv_bess_model.finance.tax import calculate_tax_for_year


@dataclass
class AnnualCashflow:
    """Cashflow breakdown for a single project year."""

    year: int
    revenue: float
    opex: float
    capex: float
    debt_service: float
    debt_interest: float
    debt_repayment: float
    depreciation: float
    gewerbesteuer: float
    koerperschaftsteuer: float
    solidaritaetszuschlag: float
    total_tax: float
    project_cf: float
    equity_cf: float


@dataclass
class CashflowProjection:
    """Complete multi-year cashflow projection.

    Attributes:
        years: List of :class:`AnnualCashflow` for years 1 through lifetime.
            Index 0 = Year 1 (commissioning year).
        equity_cashflows: Array of equity CFs for IRR/NPV calculation.
            Length = lifetime_years. Index 0 = Year 1.
        project_cashflows: Array of project CFs (pre-leverage) for Project IRR.
            Length = lifetime_years. Index 0 = Year 1.
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
    koerperschaftsteuer_pct: float = DEFAULT_KOERPERSCHAFTSTEUER_PCT,
    solidaritaetszuschlag_pct: float = DEFAULT_SOLIDARITAETSZUSCHLAG_PCT,
    replacement_cost: float = 0.0,
    replacement_year: int | None = None,
    replacement_leverage_pct: float = 0.0,
    replacement_interest_rate: float = 0.0,
    replacement_loan_tenor_years: int = 0,
    optimization_fee_pct: float = 0.0,
    annual_bess_spot_revenues: list[float] | None = None,
) -> CashflowProjection:
    """Build the complete annual cashflow projection.

    CAPEX is booked in Year 1 (the commissioning year) together with the
    first year of operations. There is no separate Year 0.

    Args:
        lifetime_years: Project lifetime (number of operating years).
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
        koerperschaftsteuer_pct: KSt rate in percent (default from defaults.py).
        solidaritaetszuschlag_pct: Soli rate in percent (default from defaults.py).
        replacement_cost: BESS replacement cost (CAPEX outflow in replacement year).
        replacement_year: Year of BESS replacement (1-indexed), or None.
        replacement_leverage_pct: Debt share of replacement cost (e.g. 75.0).
            When > 0, the debt-financed portion is added to the debt schedule
            and only the equity portion reduces the equity cashflow.
        replacement_interest_rate: Annual interest rate for the replacement loan.
        replacement_loan_tenor_years: Original loan tenor, used to derive the
            remaining tenor for the replacement loan.
        optimization_fee_pct: BESS optimization service fee as percentage of BESS spot
            revenue. Not inflation-adjusted (already based on current-year revenue).
        annual_bess_spot_revenues: BESS spot revenue per year for optimization fee
            calculation. Length must equal ``lifetime_years``. None if no BESS.

    Returns:
        :class:`CashflowProjection` with per-year detail and summary arrays.
    """
    equity_cf_array = np.zeros(lifetime_years)
    project_cf_array = np.zeros(lifetime_years)
    yearly_results: list[AnnualCashflow] = []

    equity_investment = capex_total - debt_schedule.loan_amount

    # Incorporate replacement debt into the schedule if applicable
    active_debt_schedule = debt_schedule
    replacement_equity_share = replacement_cost  # default: 100% equity-financed
    if (
        replacement_year is not None
        and replacement_cost > 0.0
        and replacement_leverage_pct > 0.0
    ):
        active_debt_schedule = add_replacement_debt(
            existing_schedule=debt_schedule,
            replacement_cost=replacement_cost,
            leverage_pct=replacement_leverage_pct,
            annual_interest_rate=replacement_interest_rate,
            replacement_year=replacement_year,
            loan_tenor_years=replacement_loan_tenor_years,
            lifetime_years=lifetime_years,
        )
        replacement_equity_share = replacement_cost * (1.0 - replacement_leverage_pct / 100.0)

    loss_carryforward = 0.0

    for y in range(1, lifetime_years + 1):
        idx = y - 1
        revenue = annual_revenues[idx]

        # OPEX with inflation
        opex = inflate_value(base_opex, inflation_rate, y)

        # Optimization fee (not inflated - already based on current-year revenue)
        if optimization_fee_pct > 0.0 and annual_bess_spot_revenues:
            opex += annual_bess_spot_revenues[idx] * optimization_fee_pct / 100.0

        # Replacement CAPEX outflow
        replacement_capex_this_year = 0.0
        replacement_equity_this_year = 0.0
        if replacement_year is not None and y == replacement_year:
            replacement_capex_this_year = replacement_cost
            replacement_equity_this_year = replacement_equity_share

        debt_interest, debt_repayment, debt_svc = get_debt_components(active_debt_schedule, y)

        # Tax calculation with Verlustvortrag and replacement AfA
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
            kst_rate_pct=koerperschaftsteuer_pct,
            soli_rate_pct=solidaritaetszuschlag_pct,
            capex_bess_replacement=replacement_cost if replacement_year else 0.0,
            replacement_year=replacement_year,
        )
        loss_carryforward = tax_result.loss_carryforward_remaining

        # CAPEX is booked in Year 1; replacement CAPEX in replacement year
        capex_this_year = (capex_total if y == 1 else 0.0) + replacement_capex_this_year
        equity_capex_this_year = (equity_investment if y == 1 else 0.0) + replacement_equity_this_year

        # Project CF (pre-leverage): Revenue - OPEX - Tax - CAPEX
        proj_cf = revenue - opex - tax_result.total_tax - capex_this_year

        # Equity CF (post-leverage): Revenue - OPEX - Debt Service - Tax - Equity CAPEX
        eq_cf = revenue - opex - debt_svc - tax_result.total_tax - equity_capex_this_year

        equity_cf_array[idx] = eq_cf
        project_cf_array[idx] = proj_cf

        yearly_results.append(
            AnnualCashflow(
                year=y,
                revenue=revenue,
                opex=opex,
                capex=capex_this_year,
                debt_service=debt_svc,
                debt_interest=debt_interest,
                debt_repayment=debt_repayment,
                depreciation=tax_result.depreciation_total,
                gewerbesteuer=tax_result.gewerbesteuer,
                koerperschaftsteuer=tax_result.koerperschaftsteuer,
                solidaritaetszuschlag=tax_result.solidaritaetszuschlag,
                total_tax=tax_result.total_tax,
                project_cf=proj_cf,
                equity_cf=eq_cf,
            )
        )

    return CashflowProjection(
        years=yearly_results,
        equity_cashflows=equity_cf_array,
        project_cashflows=project_cf_array,
    )
