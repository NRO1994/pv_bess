"""Financial metrics: Equity IRR, Project IRR, NPV, DSCR, LCOE, payback period.

All IRR/NPV computations use ``numpy_financial``. IRR convergence failures
return ``None`` instead of raising exceptions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy_financial as npf

from pv_bess_model.config.defaults import (
    DEFAULT_DISCOUNT_RATE,
    DSCR_MINIMUM_THRESHOLD,
)

logger = logging.getLogger(__name__)

@dataclass()
class FinancialMetrics:
    """Container for all computed financial metrics.

    Attributes:
        equity_irr: Equity IRR (post-leverage, post-tax), or None if non-convergent.
        project_irr: Project IRR (pre-leverage), or None if non-convergent.
        npv: Net present value at the given discount rate.
        dscr_min: Minimum DSCR across the loan tenor.
        dscr_avg: Average DSCR across the loan tenor.
        annual_dscr: list of floats, for each year of the lifetime the dscr value
        lcoe: Levelized cost of energy in €/kWh.
        payback_year: First year where cumulative equity CF turns positive, or None.
    """

    equity_irr: float | None
    project_irr: float | None
    npv: float
    dscr_min: float | None
    dscr_avg: float | None
    lcoe: float | None
    payback_year: int | None
    annual_dscr: list[float]
    capture_rate: float | None
    """Average revenue per kWh of energy fed into the grid (EUR/kWh)."""


def safe_irr(cashflows: np.ndarray) -> float | None:
    """Compute IRR, returning None on convergence failure.

    Args:
        cashflows: Array of cashflows (year 0 through N).

    Returns:
        IRR as a decimal, or None if the solver does not converge.
    """
    try:
        result = float(npf.irr(cashflows))
        if np.isnan(result) or np.isinf(result):
            return None
        return result
    except (ValueError, FloatingPointError):
        return None


def calculate_npv(
    cashflows: np.ndarray,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
) -> float:
    """Calculate Net Present Value.

    Args:
        cashflows: Array of cashflows (year 0 through N).
        discount_rate: Annual discount rate as decimal.

    Returns:
        NPV in euros.
    """
    return float(npf.npv(discount_rate, cashflows))


def calculate_dscr(
    annual_revenues: list[float],
    annual_opex: list[float],
    annual_debt_service: list[float],
) -> tuple[None, None, np.ndarray[tuple[int], np.dtype[float]]] | tuple[float, float, list[float]]:
    """Calculate minimum and average DSCR over the loan tenor.

    DSCR = (Revenue - OPEX) / Debt Service for each year.
    Only years with positive debt service are included.

    Args:
        annual_revenues: Revenue per year during loan tenor.
        annual_opex: OPEX per year during loan tenor.
        annual_debt_service: Debt service per year during loan tenor.

    Returns:
        Tuple of (min_dscr, avg_dscr, annual_dsct), or (None, None, np.zeros(lifetime)) if no debt service years.
    """
    dscr_values: list[float] = []
    for rev, opex, ds in zip(annual_revenues, annual_opex, annual_debt_service):
        if ds > 0.0:
            dscr = (rev - opex) / ds
            dscr_values.append(dscr)

    if not dscr_values:
        return None, None, np.zeros(len(annual_revenues))

    min_dscr = min(dscr_values)
    avg_dscr = sum(dscr_values) / len(dscr_values)

    if min_dscr < DSCR_MINIMUM_THRESHOLD:
        logger.debug(
            "Minimum DSCR %.2f is below threshold %.2f – debt may not be serviceable.",
            min_dscr,
            DSCR_MINIMUM_THRESHOLD,
        )

    return min_dscr, avg_dscr, dscr_values


def calculate_lcoe(
    total_costs: float,
    total_production_kwh: float,
) -> float | None:
    """Calculate Levelized Cost of Energy.

    Args:
        total_costs: Lifetime costs (CAPEX + sum of discounted OPEX) in euros.
        total_production_kwh: Lifetime energy production in kWh.

    Returns:
        LCOE in €/kWh, or None if production is zero.
    """
    if total_production_kwh <= 0.0:
        return None
    return total_costs / total_production_kwh


def calculate_payback_year(equity_cashflows: np.ndarray) -> int | None:
    """Find the first year where cumulative equity cashflow turns positive.

    Args:
        equity_cashflows: Array of equity CFs (year 0 through N).

    Returns:
        Year index (0-based) where cumulative CF first becomes positive,
        or None if payback is never reached.
    """
    cumulative = np.cumsum(equity_cashflows)
    positive_years = np.where(cumulative > 0.0)[0]
    if len(positive_years) == 0:
        return None
    return int(positive_years[0])


def calculate_capture_rate(
    annual_revenues: list[float],
    annual_pv_production_kwh: list[float],
    annual_bess_discharge_kwh: list[float],
) -> float | None:
    """Calculate the average revenue per kWh of energy fed into the grid.

    For each project year the ratio ``revenue / (pv_production + bess_discharge)``
    is computed.  The capture rate is the mean of these annual ratios.

    Args:
        annual_revenues: Total revenue per year (EUR).
        annual_pv_production_kwh: PV production per year (kWh).
        annual_bess_discharge_kwh: BESS discharge (green + grey) per year (kWh).

    Returns:
        Capture rate in EUR/kWh, or None if no energy was produced.
    """
    ratios: list[float] = []
    for rev, pv, bess in zip(
        annual_revenues, annual_pv_production_kwh, annual_bess_discharge_kwh
    ):
        energy = pv + bess
        if energy > 0.0:
            ratios.append(rev / energy)
    if not ratios:
        return None
    return sum(ratios) / len(ratios)


def compute_all_metrics(
    equity_cashflows: np.ndarray,
    project_cashflows: np.ndarray,
    annual_revenues: list[float],
    annual_opex: list[float],
    annual_debt_service: list[float],
    total_capex: float,
    total_opex_lifetime: float,
    total_production_kwh: float,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
    annual_pv_production_kwh: list[float] | None = None,
    annual_bess_discharge_kwh: list[float] | None = None,
) -> FinancialMetrics:
    """Compute all financial metrics from cashflow data.

    Args:
        equity_cashflows: Equity CF array (year 0 through N).
        project_cashflows: Project CF array (year 0 through N).
        annual_revenues: Revenue per operating year (for DSCR).
        annual_opex: OPEX per operating year (for DSCR).
        annual_debt_service: Debt service per operating year (for DSCR).
        total_capex: Total project CAPEX.
        total_opex_lifetime: Sum of all OPEX over lifetime (for LCOE).
        total_production_kwh: Total energy production over lifetime (for LCOE).
        discount_rate: Discount rate for NPV.
        annual_pv_production_kwh: PV production per year (kWh), for capture rate.
        annual_bess_discharge_kwh: BESS discharge per year (kWh), for capture rate.

    Returns:
        :class:`FinancialMetrics` with all computed values.
    """
    equity_irr = safe_irr(equity_cashflows)
    project_irr = safe_irr(project_cashflows)
    npv = calculate_npv(equity_cashflows, discount_rate)
    dscr_min, dscr_avg, annual_dscr = calculate_dscr(
        annual_revenues, annual_opex, annual_debt_service,
    )
    lcoe = calculate_lcoe(total_capex + total_opex_lifetime, total_production_kwh)
    payback = calculate_payback_year(equity_cashflows)

    capture_rate: float | None = None
    if annual_pv_production_kwh is not None and annual_bess_discharge_kwh is not None:
        capture_rate = calculate_capture_rate(
            annual_revenues, annual_pv_production_kwh, annual_bess_discharge_kwh,
        )

    return FinancialMetrics(
        equity_irr=equity_irr,
        project_irr=project_irr,
        npv=npv,
        dscr_min=dscr_min,
        dscr_avg=dscr_avg,
        lcoe=lcoe,
        payback_year=payback,
        annual_dscr=annual_dscr,
        capture_rate=capture_rate,
    )
