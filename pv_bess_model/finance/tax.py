"""Simplified German tax treatment: linear AfA depreciation and Gewerbesteuer.

Two tax components:
1. Linear depreciation (AfA) with separate periods for PV and BESS.
2. Gewerbesteuer (GewSt) with Verlustvortrag (loss carry-forward).
"""

from __future__ import annotations

from dataclasses import dataclass

from pv_bess_model.config.defaults import (
    DEFAULT_AFA_YEARS_BESS,
    DEFAULT_AFA_YEARS_PV,
    DEFAULT_GEWERBESTEUER_HEBESATZ,
    DEFAULT_GEWERBESTEUER_MESSZAHL,
)


@dataclass(frozen=True)
class TaxResult:
    """Tax computation result for a single project year.

    Attributes:
        depreciation_pv: PV AfA amount for this year.
        depreciation_bess: BESS AfA amount for this year.
        depreciation_total: Total depreciation (PV + BESS).
        taxable_income: Income after OPEX and depreciation (before Verlustvortrag).
        loss_carryforward_applied: Amount of prior losses offset against this year's income.
        adjusted_taxable_income: Taxable income after applying loss carry-forward.
        gewerbesteuer: Gewerbesteuer payable for this year.
        loss_carryforward_remaining: Cumulative loss carry-forward after this year.
    """

    depreciation_pv: float
    depreciation_bess: float
    depreciation_total: float
    taxable_income: float
    loss_carryforward_applied: float
    adjusted_taxable_income: float
    gewerbesteuer: float
    loss_carryforward_remaining: float


def calculate_annual_depreciation(
    capex: float,
    afa_years: int,
    project_year: int,
) -> float:
    """Calculate linear depreciation (AfA) for a single asset in a given year.

    Args:
        capex: CAPEX of the asset in euros.
        afa_years: Depreciation period in years.
        project_year: Current project year (1-indexed). Year 0 is CAPEX year.

    Returns:
        Depreciation amount for this year (0 if outside depreciation period).
    """
    if project_year < 1 or project_year > afa_years or afa_years <= 0:
        return 0.0
    return capex / afa_years


def calculate_gewerbesteuer(
    taxable_income: float,
    messzahl: float = DEFAULT_GEWERBESTEUER_MESSZAHL,
    hebesatz: float = DEFAULT_GEWERBESTEUER_HEBESATZ,
) -> float:
    """Calculate Gewerbesteuer for a given taxable income.

    Formula: GewSt = max(0, taxable_income) × Messzahl × Hebesatz / 100

    Args:
        taxable_income: Taxable income in euros (after Verlustvortrag adjustment).
        messzahl: Statutory base rate (default 3.5 %).
        hebesatz: Municipal multiplier in percent (e.g. 400).

    Returns:
        Gewerbesteuer amount in euros (always >= 0).
    """
    if taxable_income <= 0.0:
        return 0.0
    return taxable_income * messzahl * hebesatz / 100.0


def calculate_tax_for_year(
    revenue: float,
    opex: float,
    capex_pv: float,
    capex_bess: float,
    afa_years_pv: int,
    afa_years_bess: int,
    project_year: int,
    loss_carryforward_in: float,
    messzahl: float = DEFAULT_GEWERBESTEUER_MESSZAHL,
    hebesatz: float = DEFAULT_GEWERBESTEUER_HEBESATZ,
) -> TaxResult:
    """Calculate tax for a single project year with Verlustvortrag.

    The Verlustvortrag logic follows these steps:
    1. Compute taxable_income = revenue - opex - depreciation
    2. Apply loss carry-forward: taxable_income += loss_carryforward_in (negative)
    3. If taxable_income < 0 after carry-forward: update carry-forward, GewSt = 0
    4. If taxable_income >= 0: compute GewSt, carry-forward = 0

    Args:
        revenue: Total revenue for this year.
        opex: Total OPEX for this year (positive value).
        capex_pv: PV CAPEX (depreciation base).
        capex_bess: BESS CAPEX (depreciation base).
        afa_years_pv: PV depreciation period in years.
        afa_years_bess: BESS depreciation period in years.
        project_year: Current project year (1-indexed).
        loss_carryforward_in: Cumulative loss carry-forward entering this year
            (negative value or 0).
        messzahl: GewSt Messzahl.
        hebesatz: GewSt Hebesatz.

    Returns:
        :class:`TaxResult` with full breakdown.
    """
    # 1. Calculate depreciation
    depr_pv = calculate_annual_depreciation(capex_pv, afa_years_pv, project_year)
    depr_bess = calculate_annual_depreciation(capex_bess, afa_years_bess, project_year)
    depr_total = depr_pv + depr_bess

    # 1. taxable_income = Revenue - OPEX - AfA
    taxable_income = revenue - opex - depr_total

    # 2. Apply Verlustvortrag: add prior losses (negative value)
    adjusted = taxable_income + loss_carryforward_in

    # 3. If adjusted taxable income < 0: no GewSt, accumulate loss
    if adjusted < 0.0:
        gewst = 0.0
        loss_carryforward_out = adjusted  # carry the full negative amount forward
        loss_applied = -loss_carryforward_in  # all prior losses were "used" but not enough
        # Actually: we applied all prior losses plus this year's loss is added
        loss_applied = 0.0  # no taxable income was offset (still negative)
        # Clarification: loss_carryforward_applied = amount that actually reduced
        # positive income to zero. Since adjusted < 0, either:
        # - taxable_income was already negative (no carry-forward offset needed)
        # - taxable_income was positive but carry-forward exceeded it
        if taxable_income > 0.0:
            # Carry-forward fully absorbed the positive income
            loss_applied = taxable_income
        adjusted_for_result = 0.0
    else:
        # 4. Positive adjusted income: compute GewSt, reset carry-forward
        gewst = calculate_gewerbesteuer(adjusted, messzahl, hebesatz)
        # The carry-forward that was actually used to offset income
        loss_applied = -loss_carryforward_in  # full amount was consumed
        loss_carryforward_out = 0.0
        adjusted_for_result = adjusted

    return TaxResult(
        depreciation_pv=depr_pv,
        depreciation_bess=depr_bess,
        depreciation_total=depr_total,
        taxable_income=taxable_income,
        loss_carryforward_applied=loss_applied,
        adjusted_taxable_income=adjusted_for_result,
        gewerbesteuer=gewst,
        loss_carryforward_remaining=loss_carryforward_out,
    )
