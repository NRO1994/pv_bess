"""Optional mid-life BESS replacement logic.

At a user-specified project year the battery's capacity is reset to the
original nameplate value multiplied by *capacity_factor_pct / 100*, degradation
restarts from zero, and the replacement cost is added as additional OPEX for
that year.

Replacement cost follows the unified cost schema:
    cost = fixed_eur
         + eur_per_kw  × bess_power_kw
         + eur_per_kwh × (bess_capacity_kwh × capacity_factor_pct / 100)

The ``eur_per_kwh`` component is thus scaled by the **new** (post-upgrade)
capacity.  All other cost components refer to the original power rating since
the power electronics and grid connection remain unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pv_bess_model.config.defaults import DEFAULT_BESS_REPLACEMENT_CAPACITY_FACTOR_PCT

logger = logging.getLogger(__name__)


@dataclass
class ReplacementConfig:
    """Configuration for a single mid-life BESS replacement event.

    Attributes:
        enabled: Whether the replacement is active.  If *False* the
            replacement is ignored entirely.
        year: Project year (1-indexed) at which the replacement occurs.
        fixed_eur: Fixed replacement cost component in euros.
        eur_per_kw: Cost per kW of rated BESS power (€/kW).
        eur_per_kwh: Cost per kWh of the **new** (post-upgrade) BESS
            capacity (€/kWh).
        capacity_factor_pct: Capacity upgrade multiplier in percent.
            100 (default) means the replacement unit has the same nameplate
            capacity as the original.  120 means the replacement has 20 % more
            energy capacity (technology upgrade scenario).
    """

    enabled: bool
    year: int
    fixed_eur: float = 0.0
    eur_per_kw: float = 0.0
    eur_per_kwh: float = 0.0
    capacity_factor_pct: float = field(
        default_factory=lambda: DEFAULT_BESS_REPLACEMENT_CAPACITY_FACTOR_PCT
    )

    def replacement_cost(
        self, bess_power_kw: float, bess_capacity_kwh: float
    ) -> float:
        """Compute the replacement cost using the unified cost schema.

        The ``eur_per_kwh`` component is applied to the **new** (upgraded)
        capacity, which equals ``bess_capacity_kwh × capacity_factor_pct / 100``.
        The ``eur_per_kw`` component uses the original power rating, since the
        power electronics and grid connection are assumed to remain unchanged.

        Args:
            bess_power_kw: Original rated BESS power in kW.
            bess_capacity_kwh: Original rated BESS capacity in kWh (nameplate
                before any upgrade scaling).

        Returns:
            Total replacement cost in euros.
        """
        upgrade = self.capacity_factor_pct / 100.0
        new_capacity_kwh = bess_capacity_kwh * upgrade
        return (
            self.fixed_eur
            + self.eur_per_kw * bess_power_kw
            + self.eur_per_kwh * new_capacity_kwh
        )


def replacement_config_from_dict(config_dict: dict) -> ReplacementConfig:
    """Build a :class:`ReplacementConfig` from a scenario JSON sub-dictionary.

    Reads the ``bess.costs.replacement`` block of the scenario JSON.  Missing
    optional cost keys default to 0.0.  ``capacity_factor_pct`` defaults to
    :data:`~pv_bess_model.config.defaults.DEFAULT_BESS_REPLACEMENT_CAPACITY_FACTOR_PCT`
    (100.0) when absent.

    Args:
        config_dict: Dictionary corresponding to the ``replacement`` block in
            the scenario JSON.

    Returns:
        A populated :class:`ReplacementConfig` instance.
    """
    return ReplacementConfig(
        enabled=bool(config_dict.get("enabled", False)),
        year=int(config_dict.get("year", 0)),
        fixed_eur=float(config_dict.get("fixed_eur", 0.0)),
        eur_per_kw=float(config_dict.get("eur_per_kw", 0.0)),
        eur_per_kwh=float(config_dict.get("eur_per_kwh", 0.0)),
        capacity_factor_pct=float(
            config_dict.get(
                "capacity_factor_pct", DEFAULT_BESS_REPLACEMENT_CAPACITY_FACTOR_PCT
            )
        ),
    )
