"""EEG floor tariff logic: effective price = max(spot, eeg_floor).

The EEG tariff acts as a *Mindestpreis* (floor price), not a fixed price.
For each hour the seller receives the higher of the day-ahead spot price and
the EEG tariff.  The floor applies for the first N years of the project;
after that, revenue is at pure market price.

Public API
----------
EegConfig            – Parsed EEG parameters.
eeg_config_from_dict – Build an EegConfig from the scenario JSON marketing block.
effective_eeg_price  – Compute the EEG floor price for a given project year.
apply_eeg_floor      – Apply the floor to an hourly spot-price array.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EegConfig:
    """Parsed EEG tariff configuration.

    Attributes
    ----------
    floor_price_eur_per_kwh:
        Base EEG floor price in €/kWh (before inflation).
    fixed_price_years:
        Number of project years during which the floor applies (1-indexed).
    inflation_enabled:
        Whether the floor price is escalated annually with the inflation rate.
    """

    floor_price_eur_per_kwh: float
    fixed_price_years: int
    inflation_enabled: bool


def eeg_config_from_dict(marketing: dict) -> EegConfig:
    """Build an :class:`EegConfig` from the ``revenue_streams.marketing`` block.

    Parameters
    ----------
    marketing:
        The ``finance.revenue_streams.marketing`` dict from the scenario JSON.

    Returns
    -------
    EegConfig
    """
    return EegConfig(
        floor_price_eur_per_kwh=float(marketing["floor_price_eur_per_kwh"]),
        fixed_price_years=int(marketing["fixed_price_years"]),
        inflation_enabled=bool(marketing.get("eeg_inflation", False)),
    )


def effective_eeg_price(
    config: EegConfig,
    year: int,
    inflation_rate: float,
) -> float:
    """Return the EEG floor price for a given project year (€/kWh).

    If the year is beyond the fixed-price period the floor is 0.0 (pure market).
    If inflation is enabled the base floor is escalated:
        ``base × (1 + inflation_rate) ^ year``

    Parameters
    ----------
    config:
        EEG configuration.
    year:
        Project year (1-indexed).
    inflation_rate:
        Annual inflation rate as a fraction (e.g. 0.02 for 2 %).

    Returns
    -------
    float
        Effective floor price in €/kWh, or 0.0 after the fixed-price period.
    """
    if year > config.fixed_price_years:
        return 0.0

    base = config.floor_price_eur_per_kwh
    if config.inflation_enabled:
        return base * (1.0 + inflation_rate) ** year
    return base


def apply_eeg_floor(
    spot_prices_eur_per_kwh: np.ndarray,
    floor_price_eur_per_kwh: float,
) -> np.ndarray:
    """Apply the EEG floor to an hourly spot-price array.

    For each hour: ``effective_price = max(spot_price, floor_price)``.

    Parameters
    ----------
    spot_prices_eur_per_kwh:
        Hourly spot prices in €/kWh (any length).
    floor_price_eur_per_kwh:
        Floor price in €/kWh.  When 0.0 the spot prices are returned unchanged.

    Returns
    -------
    np.ndarray
        Effective prices in €/kWh (same shape as input).
    """
    if floor_price_eur_per_kwh <= 0.0:
        return spot_prices_eur_per_kwh.copy()
    return np.maximum(spot_prices_eur_per_kwh, floor_price_eur_per_kwh)
