"""Lifecycle cost calculation for flexibility assets.

Computes total CAPEX, OPEX, and personnel costs over the project lifetime
for a given annual addition rate.  Costs are undiscounted and not inflated,
consistent with the system value calculation.

The cost model handles three components per year:

1. **CAPEX** — incurred each year when new capacity is installed.
   Optionally reduced by a learning curve (annual cost degression).
2. **OPEX** — annual cost based on cumulative installed capacity.
   For BESS, cumulative kWh follows the tranche degradation model.
3. **Personnel** — step-function cost based on cumulative installed kW.

Typical usage::

    from pv_bess_model.portfolio.flex_costs import compute_flex_lifecycle_cost

    result = compute_flex_lifecycle_cost(
        costs=flex_config.costs,
        annual_addition_kw=500.0,
        e_to_p_ratio=2.0,
        degradation_rate=0.02,
        lifetime_years=25,
        start_year=1,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pv_bess_model.config.loader_portfolio import FlexCostConfig

logger = logging.getLogger(__name__)


@dataclass
class FlexLifecycleCost:
    """Lifecycle cost result for one (flex_instance, addition_rate) point.

    Attributes
    ----------
    annual_capex : list[float]
        CAPEX per project year (EUR), length = lifetime_years.
    annual_opex : list[float]
        OPEX per project year (EUR), length = lifetime_years.
    annual_personnel : list[float]
        Personnel cost per project year (EUR), length = lifetime_years.
    annual_total : list[float]
        Total cost per project year (EUR), length = lifetime_years.
    cumulative_cost : float
        Sum of annual_total over all years (EUR).
    """

    annual_capex: list[float] = field(default_factory=list)
    annual_opex: list[float] = field(default_factory=list)
    annual_personnel: list[float] = field(default_factory=list)
    annual_total: list[float] = field(default_factory=list)
    cumulative_cost: float = 0.0


def compute_flex_lifecycle_cost(
    costs: FlexCostConfig,
    annual_addition_kw: float,
    lifetime_years: int,
    e_to_p_ratio: float = 0.0,
    degradation_rate: float = 0.0,
    start_year: int = 1,
) -> FlexLifecycleCost:
    """Compute lifecycle cost for a flex asset at a given annual addition rate.

    Parameters
    ----------
    costs:
        Cost configuration (CAPEX, OPEX, learning curve, personnel steps).
    annual_addition_kw:
        Annual power addition rate in kW/year.
    lifetime_years:
        Project lifetime in years.
    e_to_p_ratio:
        Energy-to-power ratio in hours (BESS only, 0.0 for other flex types).
    degradation_rate:
        Annual capacity degradation as a fraction (e.g. 0.02 for 2 %).
        Only affects cumulative kWh for OPEX calculation.
    start_year:
        First project year (1-indexed) in which annual additions begin.

    Returns
    -------
    FlexLifecycleCost
        Annual and cumulative cost breakdown.
    """
    annual_capex: list[float] = []
    annual_opex: list[float] = []
    annual_personnel: list[float] = []

    learning_rate = costs.capex_learning_rate_pct / 100.0

    for year in range(1, lifetime_years + 1):
        # --- CAPEX ---
        if year >= start_year and annual_addition_kw > 0.0:
            years_since_start = year - start_year
            learning_factor = (1.0 - learning_rate) ** years_since_start

            addition_kwh = annual_addition_kw * e_to_p_ratio

            capex_year = (
                costs.capex_fixed_eur
                + costs.capex_eur_per_kw * annual_addition_kw
                + costs.capex_eur_per_kwh * addition_kwh
            ) * learning_factor
        else:
            capex_year = 0.0

        # --- Cumulative capacity for OPEX and personnel ---
        if year >= start_year and annual_addition_kw > 0.0:
            cumulative_kw = annual_addition_kw * (year - start_year + 1)

            # Cumulative kWh with tranche degradation (consistent with
            # engine_portfolio.compute_bess_tranche_capacity)
            cumulative_kwh = 0.0
            for install_year in range(start_year, year + 1):
                age = year - install_year
                cumulative_kwh += (
                    annual_addition_kw
                    * e_to_p_ratio
                    * (1.0 - degradation_rate) ** age
                )
        else:
            cumulative_kw = 0.0
            cumulative_kwh = 0.0

        # --- OPEX ---
        if cumulative_kw > 0.0:
            opex_year = (
                costs.opex_fixed_eur
                + costs.opex_eur_per_kw * cumulative_kw
                + costs.opex_eur_per_kwh * cumulative_kwh
            )
        else:
            opex_year = 0.0

        # --- Personnel (step function on cumulative kW) ---
        personnel_year = _compute_personnel_cost(
            costs.personnel_steps, cumulative_kw
        )

        annual_capex.append(capex_year)
        annual_opex.append(opex_year)
        annual_personnel.append(personnel_year)

    annual_total = [
        c + o + p
        for c, o, p in zip(annual_capex, annual_opex, annual_personnel)
    ]
    cumulative = sum(annual_total)

    return FlexLifecycleCost(
        annual_capex=annual_capex,
        annual_opex=annual_opex,
        annual_personnel=annual_personnel,
        annual_total=annual_total,
        cumulative_cost=cumulative,
    )


def _compute_personnel_cost(
    steps: list,
    cumulative_kw: float,
) -> float:
    """Look up personnel cost for a given cumulative capacity.

    Steps are sorted by ``threshold_kw`` ascending.  The cost of the highest
    step whose threshold is <= ``cumulative_kw`` is returned.

    Parameters
    ----------
    steps:
        Sorted list of ``PersonnelStep`` instances.
    cumulative_kw:
        Current cumulative installed power in kW.

    Returns
    -------
    float
        Annual personnel cost in EUR.
    """
    if not steps:
        return 0.0

    result = 0.0
    for step in steps:
        if cumulative_kw >= step.threshold_kw:
            result = step.annual_cost_eur
        else:
            break

    return result
