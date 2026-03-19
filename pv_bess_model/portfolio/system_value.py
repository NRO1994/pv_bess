"""World A/B comparison and system value calculation.

World A is the baseline: PV generation vs. aggregated load, no flexibility.
The net position per quarter-hour determines grid buy/sell at spot price.

World B adds flexibility (BESS, heat pump, EV/V2G) which can shift energy
in time, reducing the net system cost.

System value = cost(World A) − cost(World B)

Typical usage::

    from pv_bess_model.portfolio.system_value import compute_world_a

    result = compute_world_a(pv_profile, load_profile, spot_prices)
    print(f"Annual system cost: {result.system_cost:.0f} EUR")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WorldAResult:
    """Result of the World A (no flexibility) calculation.

    Attributes
    ----------
    system_cost : float
        Net system cost in EUR.  Positive = net cost (buy > sell),
        negative = net revenue (sell > buy).
    total_sell_eur : float
        Total revenue from selling surplus to the grid (EUR).
    total_buy_eur : float
        Total cost of buying deficit from the grid (EUR).
    total_sell_kwh : float
        Total energy sold to the grid (kWh).
    total_buy_kwh : float
        Total energy bought from the grid (kWh).
    netto : numpy.ndarray
        Net position per interval (kWh).  Positive = surplus, negative = deficit.
    """

    system_cost: float
    total_sell_eur: float
    total_buy_eur: float
    total_sell_kwh: float
    total_buy_kwh: float
    netto: np.ndarray


def compute_world_a(
    pv_profile: np.ndarray,
    load_profile: np.ndarray,
    spot_prices: np.ndarray,
) -> WorldAResult:
    """Compute World A system cost (no flexibility).

    For each quarter-hour interval::

        netto[t] = pv[t] - load[t]
        netto > 0  →  surplus sold at spot[t]
        netto < 0  →  deficit bought at spot[t]

    System cost = total_buy - total_sell  (positive = net cost).

    Parameters
    ----------
    pv_profile:
        Aggregated PV production per interval in kWh.
    load_profile:
        Aggregated load per interval in kWh.
    spot_prices:
        Spot price per interval in EUR/kWh.

    Returns
    -------
    WorldAResult
        System cost breakdown and net position array.

    Raises
    ------
    ValueError
        When input arrays have different lengths.
    """
    if not (len(pv_profile) == len(load_profile) == len(spot_prices)):
        raise ValueError(
            f"All input arrays must have the same length. "
            f"Got pv={len(pv_profile)}, load={len(load_profile)}, "
            f"prices={len(spot_prices)}."
        )

    netto = pv_profile - load_profile

    # Surplus (netto > 0) → sell to grid
    sell_kwh = np.maximum(netto, 0.0)
    sell_eur = np.sum(sell_kwh * spot_prices)

    # Deficit (netto < 0) → buy from grid
    buy_kwh = np.maximum(-netto, 0.0)
    buy_eur = np.sum(buy_kwh * spot_prices)

    system_cost = buy_eur - sell_eur

    logger.info(
        "World A: sell=%.0f MWh (%.0f EUR), buy=%.0f MWh (%.0f EUR), "
        "system_cost=%.0f EUR.",
        float(np.sum(sell_kwh)) / 1000.0,
        sell_eur,
        float(np.sum(buy_kwh)) / 1000.0,
        buy_eur,
        system_cost,
    )

    return WorldAResult(
        system_cost=float(system_cost),
        total_sell_eur=float(sell_eur),
        total_buy_eur=float(buy_eur),
        total_sell_kwh=float(np.sum(sell_kwh)),
        total_buy_kwh=float(np.sum(buy_kwh)),
        netto=netto,
    )
