"""Marginal value curves for flexibility assets.

Computes discrete marginal value (EUR/kW per year of addition) as the
incremental system value per additional unit of flexibility capacity.
This enables comparison across flexibility types and identification of
diminishing returns.

The marginal value is computed as a discrete derivative::

    marginal[i] = (value[i] - value[i-1]) / (kw[i] - kw[i-1])

where ``value`` is the cumulative system value and ``kw`` is the annual
addition rate.  The result is in EUR per kW/a of additional capacity.

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
    """Marginal value for one step in the addition-rate curve.

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
    """

    flex_name: str
    flex_type: str
    annual_addition_kw: float
    e_to_p_ratio: float | None
    cumulative_system_value_eur: float
    marginal_value_eur_per_kw_a: float
    delta_kw: float
    delta_value_eur: float


def compute_marginal_values(
    points: list[SystemValuePoint],
) -> list[MarginalValuePoint]:
    """Compute marginal value curves from enumeration points.

    Groups points by ``(flex_name, flex_type, e_to_p_ratio)``, sorts each
    group by ``annual_addition_kw``, and computes the discrete derivative.

    For the first point in each group (or when ``delta_kw == 0``), the
    marginal value is set to 0.0.

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

        for p in group:
            delta_kw = p.annual_addition_kw - prev_kw
            delta_value = p.cumulative_system_value_eur - prev_value

            if delta_kw > 0:
                marginal = delta_value / delta_kw
            else:
                marginal = 0.0

            result.append(
                MarginalValuePoint(
                    flex_name=p.flex_name,
                    flex_type=p.flex_type,
                    annual_addition_kw=p.annual_addition_kw,
                    e_to_p_ratio=p.e_to_p_ratio,
                    cumulative_system_value_eur=p.cumulative_system_value_eur,
                    marginal_value_eur_per_kw_a=marginal,
                    delta_kw=delta_kw,
                    delta_value_eur=delta_value,
                )
            )

            prev_kw = p.annual_addition_kw
            prev_value = p.cumulative_system_value_eur

        logger.debug(
            "Marginal values for '%s' (E/P=%s): %d steps, "
            "max marginal=%.0f EUR/kW/a",
            key[0],
            key[2],
            len(group),
            max((m.marginal_value_eur_per_kw_a for m in result[-len(group):]), default=0),
        )

    return result
