"""Optional mid-life BESS replacement logic.

At a user-specified project year the battery's capacity is reset to the
original nameplate value, degradation restarts from zero, and the replacement
cost is added as additional OPEX for that year.

Replacement cost follows the unified cost schema:
    cost = fixed_eur + eur_per_kw × bess_power_kw + eur_per_kwh × bess_capacity_kwh
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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
        eur_per_kwh: Cost per kWh of rated BESS capacity (€/kWh).
    """

    enabled: bool
    year: int
    fixed_eur: float = 0.0
    eur_per_kw: float = 0.0
    eur_per_kwh: float = 0.0

    def replacement_cost(
        self, bess_power_kw: float, bess_capacity_kwh: float
    ) -> float:
        """Compute the replacement cost using the unified cost schema.

        Args:
            bess_power_kw: Rated BESS power in kW.
            bess_capacity_kwh: Rated BESS capacity in kWh.

        Returns:
            Total replacement cost in euros.
        """
        return (
            self.fixed_eur
            + self.eur_per_kw * bess_power_kw
            + self.eur_per_kwh * bess_capacity_kwh
        )


def replacement_config_from_dict(config_dict: dict) -> ReplacementConfig:
    """Build a :class:`ReplacementConfig` from a scenario JSON sub-dictionary.

    Reads the ``bess.costs.replacement`` block of the scenario JSON.  Missing
    optional cost keys default to 0.0.

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
    )
