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

from pv_bess_model.bess.battery import BatteryState

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


def apply_replacement(
    battery: BatteryState,
    config: ReplacementConfig,
    current_year: int,
    nameplate_capacity_kwh: float,
    bess_power_kw: float,
) -> float:
    """Apply a BESS replacement event if the current year matches.

    When triggered:
    - ``battery.max_capacity_kwh`` is reset to *nameplate_capacity_kwh*
      (restarting the degradation clock from a full-capacity battery).
    - ``battery.current_soc_kwh`` is clipped to the new upper limit if needed.
    - The replacement cost is returned to be booked as additional OPEX.

    If the replacement is disabled or the year does not match, the battery is
    left unchanged and 0.0 is returned.

    Args:
        battery: The :class:`BatteryState` instance to update in-place.
        config: Replacement configuration.
        current_year: Current project year (1-indexed).
        nameplate_capacity_kwh: Original BESS capacity in kWh to restore.
        bess_power_kw: Rated BESS power (kW) used for the cost calculation.

    Returns:
        Additional OPEX (replacement cost) for this year in euros, or 0.0 if
        no replacement takes place.
    """
    if not config.enabled or current_year != config.year:
        return 0.0

    logger.info(
        "BESS replacement at project year %d: capacity %.1f kWh → %.1f kWh.",
        current_year,
        battery.max_capacity_kwh,
        nameplate_capacity_kwh,
    )

    battery.max_capacity_kwh = nameplate_capacity_kwh
    # Clip SoC defensively after capacity increase
    battery.current_soc_kwh = min(battery.current_soc_kwh, battery.max_soc_kwh)

    return config.replacement_cost(bess_power_kw, nameplate_capacity_kwh)


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
