"""PPA models: pay-as-produced, baseload, floor, collar.

Four PPA structures are supported, selectable per scenario:

1. **pay_as_produced** – Buyer pays a fixed price per kWh produced.
2. **baseload** – Seller commits to a flat power profile; shortfall/excess
   settled at market price.
3. **floor** – Minimum price guaranteed; seller keeps upside above floor.
   Same mechanics as EEG: ``effective = max(spot, floor)``.
4. **collar** – Floor and cap price boundaries:
   ``effective = clip(spot, floor, cap)``.

All models:
- Have a limited duration in years; after expiry → pure market price.
- Support optional inflation escalation on all price-related parameters.
- Add a Guarantee-of-Origin (GoO) premium to the effective price during the
  PPA period.

Public API
----------
PpaConfig            – Parsed PPA parameters.
ppa_config_from_dict – Build a PpaConfig from the scenario JSON ppa block.
effective_ppa_price  – Compute effective price array for a given year.
baseload_revenue     – Compute net revenue for a baseload PPA settlement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from pv_bess_model.config.defaults import (
    HOURS_PER_YEAR,
    PPA_TYPE_BASELOAD,
    PPA_TYPE_COLLAR,
    PPA_TYPE_FLOOR,
    PPA_TYPE_NONE,
    PPA_TYPE_PAY_AS_PRODUCED,
)
from pv_bess_model.finance.inflation import inflate_value

logger = logging.getLogger(__name__)

_VALID_PPA_TYPES: frozenset[str] = frozenset(
    {
        PPA_TYPE_NONE,
        PPA_TYPE_PAY_AS_PRODUCED,
        PPA_TYPE_BASELOAD,
        PPA_TYPE_FLOOR,
        PPA_TYPE_COLLAR,
    }
)


@dataclass(frozen=True)
class PpaConfig:
    """Parsed PPA configuration.

    Attributes
    ----------
    ppa_type:
        One of ``"none"``, ``"ppa_pay_as_produced"``, ``"ppa_baseload"``,
        ``"ppa_floor"``, ``"ppa_collar"``.
    pay_as_produced_price_eur_per_kwh:
        Fixed price for pay-as-produced PPA (€/kWh). ``None`` if unused.
    baseload_mw:
        Committed baseload level in MW. ``None`` for auto-calculation or if
        unused.
    floor_price_eur_per_kwh:
        Floor price for floor/collar PPA (€/kWh). ``None`` if unused.
    cap_price_eur_per_kwh:
        Cap price for collar PPA (€/kWh). ``None`` if unused.
    duration_years:
        Number of project years the PPA is active (1-indexed). After this
        period, revenue switches to pure market price.
    inflation_enabled:
        Whether price parameters are escalated annually.
    goo_premium_eur_per_kwh:
        Guarantee-of-Origin premium added to effective price during the PPA
        period (€/kWh).
    """

    ppa_type: str
    pay_as_produced_price_eur_per_kwh: float | None
    baseload_mw: float | None
    floor_price_eur_per_kwh: float | None
    cap_price_eur_per_kwh: float | None
    duration_years: int
    inflation_enabled: bool
    goo_premium_eur_per_kwh: float


def ppa_config_from_dict(ppa_dict: dict) -> PpaConfig:
    """Build a :class:`PpaConfig` from the ``revenue_streams.ppa`` block.

    Parameters
    ----------
    ppa_dict:
        The ``finance.revenue_streams.ppa`` dict from the scenario JSON.

    Returns
    -------
    PpaConfig

    Raises
    ------
    ValueError
        If the PPA type is unrecognised.
    """
    ppa_type = ppa_dict["type"]
    if ppa_type not in _VALID_PPA_TYPES:
        raise ValueError(
            f"Unknown PPA type '{ppa_type}'. "
            f"Valid types: {sorted(_VALID_PPA_TYPES)}"
        )

    def _opt_float(key: str) -> float | None:
        v = ppa_dict.get(key)
        return float(v) if v is not None else None

    return PpaConfig(
        ppa_type=ppa_type,
        pay_as_produced_price_eur_per_kwh=_opt_float(
            "pay_as_produced_price_eur_per_kwh"
        ),
        baseload_mw=_opt_float("baseload_mw"),
        floor_price_eur_per_kwh=_opt_float("floor_price_eur_per_kwh"),
        cap_price_eur_per_kwh=_opt_float("cap_price_eur_per_kwh"),
        duration_years=int(ppa_dict.get("duration_years", 0)),
        inflation_enabled=bool(ppa_dict.get("inflation_on_ppa", False)),
        goo_premium_eur_per_kwh=float(
            ppa_dict.get("guarantee_of_origin_eur_per_kwh", 0.0)
        ),
    )


# ---------------------------------------------------------------------------
# Pay-as-produced
# ---------------------------------------------------------------------------


def pay_as_produced_price(
    config: PpaConfig,
    year: int,
    inflation_rate: float,
) -> float:
    """Return the fixed pay-as-produced price for *year* (€/kWh).

    After the PPA duration the price is 0.0 (revenue comes from spot market).

    Parameters
    ----------
    config:
        PPA configuration.
    year:
        Project year (1-indexed).
    inflation_rate:
        Annual inflation rate as a fraction.

    Returns
    -------
    float
        Fixed price in €/kWh, or 0.0 if the PPA has expired.
    """
    if year > config.duration_years:
        return 0.0
    base = config.pay_as_produced_price_eur_per_kwh or 0.0
    if config.inflation_enabled:
        return inflate_value(base, inflation_rate, year)
    return base


def apply_pay_as_produced(
    production_kwh: np.ndarray,
    ppa_price_eur_per_kwh: float,
    goo_premium_eur_per_kwh: float,
) -> np.ndarray:
    """Compute hourly revenue for pay-as-produced PPA.

    ``revenue[t] = production[t] × (ppa_price + goo_premium)``

    Parameters
    ----------
    production_kwh:
        Hourly production in kWh (grid-delivered, after losses).
    ppa_price_eur_per_kwh:
        Fixed PPA price in €/kWh (already inflation-adjusted).
    goo_premium_eur_per_kwh:
        GoO premium added to the price (€/kWh).

    Returns
    -------
    np.ndarray
        Hourly revenue in euros.
    """
    effective = ppa_price_eur_per_kwh + goo_premium_eur_per_kwh
    return production_kwh * effective


# ---------------------------------------------------------------------------
# Baseload PPA
# ---------------------------------------------------------------------------


def baseload_level_kwh(
    baseload_mw: float | None,
    annual_production_kwh: float,
) -> float:
    """Determine the hourly baseload commitment in kWh.

    If *baseload_mw* is provided, convert to kWh per hour (``MW × 1000``).
    Otherwise, derive from annual production: ``annual / HOURS_PER_YEAR``.

    Parameters
    ----------
    baseload_mw:
        Explicit baseload level in MW, or ``None`` for auto-calculation.
    annual_production_kwh:
        Total annual production in kWh (used for auto-calculation).

    Returns
    -------
    float
        Hourly baseload commitment in kWh.
    """
    if baseload_mw is not None:
        return baseload_mw * 1000.0  # MW → kW = kWh/h
    return annual_production_kwh / HOURS_PER_YEAR


def baseload_revenue(
    grid_export_kwh: np.ndarray,
    spot_prices_eur_per_kwh: np.ndarray,
    baseload_kwh: float,
    ppa_price_eur_per_kwh: float,
    goo_premium_eur_per_kwh: float,
) -> np.ndarray:
    """Compute hourly net revenue for a baseload PPA.

    Settlement per hour::

        If export ≥ baseload:
            revenue = baseload × (ppa_price + goo) + excess × spot
        If export < baseload:
            revenue = baseload × (ppa_price + goo) - shortfall × spot

    Equivalently::

        revenue = baseload × (ppa_price + goo) + (export - baseload) × spot

    Parameters
    ----------
    grid_export_kwh:
        Hourly grid export in kWh (PV direct + BESS discharge × RTE).
    spot_prices_eur_per_kwh:
        Hourly spot prices in €/kWh.
    baseload_kwh:
        Hourly baseload commitment in kWh.
    ppa_price_eur_per_kwh:
        PPA price in €/kWh (already inflation-adjusted).
    goo_premium_eur_per_kwh:
        GoO premium in €/kWh.

    Returns
    -------
    np.ndarray
        Hourly net revenue in euros (can be negative if shortfall costs
        exceed baseload revenue).
    """
    effective_ppa = ppa_price_eur_per_kwh + goo_premium_eur_per_kwh
    ppa_revenue = baseload_kwh * effective_ppa
    imbalance = (grid_export_kwh - baseload_kwh) * spot_prices_eur_per_kwh
    return ppa_revenue + imbalance


# ---------------------------------------------------------------------------
# Floor PPA
# ---------------------------------------------------------------------------


def effective_floor_price(
    config: PpaConfig,
    year: int,
    inflation_rate: float,
) -> float:
    """Return the effective floor price for *year* (€/kWh), including GoO premium.

    After the PPA duration the floor is 0.0 and GoO premium no longer applies.

    Parameters
    ----------
    config:
        PPA configuration (must be ``ppa_floor`` type).
    year:
        Project year (1-indexed).
    inflation_rate:
        Annual inflation rate as a fraction.

    Returns
    -------
    float
        Floor price + GoO premium in €/kWh, or 0.0 after PPA expiry.
    """
    if year > config.duration_years:
        return 0.0
    base = config.floor_price_eur_per_kwh or 0.0
    if config.inflation_enabled:
        inflated = inflate_value(base, inflation_rate, year)
    else:
        inflated = base
    return inflated + config.goo_premium_eur_per_kwh


def apply_floor_ppa(
    spot_prices_eur_per_kwh: np.ndarray,
    floor_price_eur_per_kwh: float,
) -> np.ndarray:
    """Apply a floor price to an hourly spot-price array.

    ``effective[t] = max(spot[t], floor)``

    Parameters
    ----------
    spot_prices_eur_per_kwh:
        Hourly spot prices (€/kWh).
    floor_price_eur_per_kwh:
        Floor price (€/kWh, already includes GoO and inflation).

    Returns
    -------
    np.ndarray
        Effective prices (€/kWh, same shape as input).
    """
    if floor_price_eur_per_kwh <= 0.0:
        return spot_prices_eur_per_kwh.copy()
    return np.maximum(spot_prices_eur_per_kwh, floor_price_eur_per_kwh)


# ---------------------------------------------------------------------------
# Collar PPA
# ---------------------------------------------------------------------------


def effective_collar_prices(
    config: PpaConfig,
    year: int,
    inflation_rate: float,
) -> tuple[float, float]:
    """Return the (floor, cap) prices for *year* (€/kWh), including GoO on floor.

    After the PPA duration both return 0.0 (pure market).

    Parameters
    ----------
    config:
        PPA configuration (must be ``ppa_collar`` type).
    year:
        Project year (1-indexed).
    inflation_rate:
        Annual inflation rate as a fraction.

    Returns
    -------
    tuple[float, float]
        (floor_eur_per_kwh, cap_eur_per_kwh)
    """
    if year > config.duration_years:
        return 0.0, 0.0
    base_floor = config.floor_price_eur_per_kwh or 0.0
    base_cap = config.cap_price_eur_per_kwh or 0.0
    if config.inflation_enabled:
        floor = inflate_value(base_floor, inflation_rate, year)
        cap = inflate_value(base_cap, inflation_rate, year)
    else:
        floor = base_floor
        cap = base_cap
    floor += config.goo_premium_eur_per_kwh
    return floor, cap


def apply_collar_ppa(
    spot_prices_eur_per_kwh: np.ndarray,
    floor_price_eur_per_kwh: float,
    cap_price_eur_per_kwh: float,
) -> np.ndarray:
    """Apply a collar (floor + cap) to an hourly spot-price array.

    ``effective[t] = clip(spot[t], floor, cap)``

    When the collar has expired (both prices 0.0), the spot prices are
    returned unchanged.

    Parameters
    ----------
    spot_prices_eur_per_kwh:
        Hourly spot prices (€/kWh).
    floor_price_eur_per_kwh:
        Floor price (€/kWh, already includes GoO and inflation).
    cap_price_eur_per_kwh:
        Cap price (€/kWh, already inflation-adjusted).

    Returns
    -------
    np.ndarray
        Effective prices (€/kWh, same shape as input).
    """
    if floor_price_eur_per_kwh <= 0.0 and cap_price_eur_per_kwh <= 0.0:
        return spot_prices_eur_per_kwh.copy()
    return np.clip(
        spot_prices_eur_per_kwh,
        floor_price_eur_per_kwh,
        cap_price_eur_per_kwh,
    )


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------


def effective_ppa_price_for_year(
    config: PpaConfig,
    year: int,
    inflation_rate: float,
    spot_prices_eur_per_kwh: np.ndarray,
    grid_export_kwh: np.ndarray | None = None,
    annual_production_kwh: float | None = None,
) -> np.ndarray:
    """Compute the effective hourly price or revenue array for any PPA type.

    This is the primary dispatch-integration entry point.  Depending on the
    PPA type, it returns either an effective price array (floor, collar,
    pay-as-produced) that the caller multiplies by production, or a revenue
    array (baseload) that can be used directly.

    For ``ppa_pay_as_produced``, ``ppa_floor``, and ``ppa_collar``:
        Returns an effective **price** array (€/kWh).

    For ``ppa_baseload``:
        Returns an hourly **revenue** array (€).  The caller must use this
        directly rather than multiplying by production.

    For ``none``:
        Returns the spot prices unchanged (with GoO premium added during PPA
        period — but since type is ``none``, no GoO applies and spot is passed
        through).

    Parameters
    ----------
    config:
        PPA configuration.
    year:
        Project year (1-indexed).
    inflation_rate:
        Annual inflation rate as a fraction.
    spot_prices_eur_per_kwh:
        Hourly spot prices (€/kWh) for this year.
    grid_export_kwh:
        Hourly grid export (kWh); required for ``ppa_baseload``.
    annual_production_kwh:
        Total annual production (kWh); used for auto-deriving baseload level.

    Returns
    -------
    np.ndarray
        Effective prices (€/kWh) or revenue (€) depending on PPA type.
    """
    if config.ppa_type == PPA_TYPE_NONE:
        return spot_prices_eur_per_kwh.copy()

    # After PPA expires → pure market
    if year > config.duration_years:
        return spot_prices_eur_per_kwh.copy()

    if config.ppa_type == PPA_TYPE_PAY_AS_PRODUCED:
        price = pay_as_produced_price(config, year, inflation_rate)
        # Return a constant price array (caller multiplies by production)
        return np.full_like(spot_prices_eur_per_kwh, price)

    if config.ppa_type == PPA_TYPE_FLOOR:
        floor = effective_floor_price(config, year, inflation_rate)
        return apply_floor_ppa(spot_prices_eur_per_kwh, floor)

    if config.ppa_type == PPA_TYPE_COLLAR:
        floor, cap = effective_collar_prices(config, year, inflation_rate)
        return apply_collar_ppa(spot_prices_eur_per_kwh, floor, cap)

    if config.ppa_type == PPA_TYPE_BASELOAD:
        if grid_export_kwh is None:
            raise ValueError(
                "grid_export_kwh is required for baseload PPA revenue calculation."
            )
        base_price = config.pay_as_produced_price_eur_per_kwh or 0.0
        if config.inflation_enabled:
            inflated_price = inflate_value(base_price, inflation_rate, year)
        else:
            inflated_price = base_price
        bl = baseload_level_kwh(
            config.baseload_mw,
            annual_production_kwh or 0.0,
        )
        return baseload_revenue(
            grid_export_kwh=grid_export_kwh,
            spot_prices_eur_per_kwh=spot_prices_eur_per_kwh,
            baseload_kwh=bl,
            ppa_price_eur_per_kwh=inflated_price,
            goo_premium_eur_per_kwh=config.goo_premium_eur_per_kwh,
        )

    # Should not reach here (validated at construction)
    raise ValueError(f"Unsupported PPA type: {config.ppa_type}")  # pragma: no cover
