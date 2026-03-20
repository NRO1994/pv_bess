"""Marginal value and marginal cost curves for flexibility assets.

Computes discrete marginal value (EUR/kW per year of addition) as the
incremental system value per additional unit of flexibility capacity,
and marginal cost as the incremental lifecycle cost per additional unit.

Both are computed as discrete derivatives::

    marginal[i] = (value[i] - value[i-1]) / (kw[i] - kw[i-1])

The optimal addition rate is the last point where marginal value exceeds
marginal cost (i.e. the net marginal value is positive).

Typical usage::

    from pv_bess_model.portfolio.marginal_value import compute_marginal_values

    points = enum_result.points  # from run_enumeration()
    marginals = compute_marginal_values(points)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pv_bess_model.portfolio.system_value import SystemValuePoint

logger = logging.getLogger(__name__)


@dataclass
class MarginalValuePoint:
    """Marginal value and cost for one step in the addition-rate curve.

    Attributes
    ----------
    flex_name : str
        Name of the flex instance.
    flex_type : str
        Flex type identifier.
    annual_addition_kw : float
        Annual addition rate at this point (kW/a).
    e_to_p_ratio : float | None
        Energy-to-power ratio (BESS only).
    cumulative_system_value_eur : float
        Total system value at this addition rate (EUR).
    marginal_value_eur_per_kw_a : float
        Incremental system value per additional kW/a (EUR / kW/a).
    delta_kw : float
        Step size: ``kw[i] - kw[i-1]``.
    delta_value_eur : float
        Value increment: ``value[i] - value[i-1]``.
    cumulative_cost_eur : float
        Total lifecycle cost at this addition rate (EUR).
    marginal_cost_eur_per_kw_a : float
        Incremental lifecycle cost per additional kW/a (EUR / kW/a).
    is_optimal : bool
        True if this is the last point where marginal value > marginal cost.
    """

    flex_name: str
    flex_type: str
    annual_addition_kw: float
    e_to_p_ratio: float | None
    cumulative_system_value_eur: float
    marginal_value_eur_per_kw_a: float
    delta_kw: float
    delta_value_eur: float
    cumulative_cost_eur: float = 0.0
    marginal_cost_eur_per_kw_a: float = 0.0
    is_optimal: bool = False


def compute_marginal_values(
    points: list[SystemValuePoint],
) -> list[MarginalValuePoint]:
    """Compute marginal value and cost curves from enumeration points.

    Groups points by ``(flex_name, flex_type, e_to_p_ratio)``, sorts each
    group by ``annual_addition_kw``, and computes discrete derivatives for
    both system value and lifecycle cost.

    For the first point in each group (or when ``delta_kw == 0``), the
    marginal value and cost are set to 0.0.

    The ``is_optimal`` flag is set on the last point in each group where
    ``marginal_value > marginal_cost``.  If costs are all zero (no cost
    config), no optimal point is marked.

    Parameters
    ----------
    points:
        Enumeration points from ``run_enumeration()``.

    Returns
    -------
    list[MarginalValuePoint]
        Marginal value points, sorted by group key then addition rate.
    """
    if not points:
        return []

    # Group by (flex_name, flex_type, e_to_p_ratio)
    groups: dict[tuple[str, str, float | None], list[SystemValuePoint]] = {}
    for p in points:
        key = (p.flex_name, p.flex_type, p.e_to_p_ratio)
        groups.setdefault(key, []).append(p)

    result: list[MarginalValuePoint] = []

    for key, group in sorted(groups.items()):
        # Sort by addition rate
        group.sort(key=lambda p: p.annual_addition_kw)

        prev_kw = 0.0
        prev_value = 0.0
        prev_cost = 0.0

        group_start_idx = len(result)

        for p in group:
            delta_kw = p.annual_addition_kw - prev_kw
            delta_value = p.cumulative_system_value_eur - prev_value
            delta_cost = p.cumulative_cost_eur - prev_cost

            if delta_kw > 0:
                marginal_value = delta_value / delta_kw
                marginal_cost = delta_cost / delta_kw
            else:
                marginal_value = 0.0
                marginal_cost = 0.0

            result.append(
                MarginalValuePoint(
                    flex_name=p.flex_name,
                    flex_type=p.flex_type,
                    annual_addition_kw=p.annual_addition_kw,
                    e_to_p_ratio=p.e_to_p_ratio,
                    cumulative_system_value_eur=p.cumulative_system_value_eur,
                    marginal_value_eur_per_kw_a=marginal_value,
                    delta_kw=delta_kw,
                    delta_value_eur=delta_value,
                    cumulative_cost_eur=p.cumulative_cost_eur,
                    marginal_cost_eur_per_kw_a=marginal_cost,
                )
            )

            prev_kw = p.annual_addition_kw
            prev_value = p.cumulative_system_value_eur
            prev_cost = p.cumulative_cost_eur

        # Mark optimal point: last point where marginal_value > marginal_cost
        group_points = result[group_start_idx:]
        has_costs = any(m.cumulative_cost_eur > 0 for m in group_points)
        if has_costs:
            optimal_idx = None
            for i, m in enumerate(group_points):
                if m.delta_kw > 0 and m.marginal_value_eur_per_kw_a > m.marginal_cost_eur_per_kw_a:
                    optimal_idx = i
            if optimal_idx is not None:
                group_points[optimal_idx].is_optimal = True

        logger.debug(
            "Marginal values for '%s' (E/P=%s): %d steps, "
            "max marginal=%.0f EUR/kW/a",
            key[0],
            key[2],
            len(group),
            max((m.marginal_value_eur_per_kw_a for m in group_points), default=0),
        )

    return result
