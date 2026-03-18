"""Inflation escalation logic for OPEX, PPA price, EEG price, and spot prices.

Supports two modes:

1. **Fixed rate** (backward compatible): A single annual inflation rate applied
   via compound formula ``base × (1 + rate) ^ (year - 1)``.  Year 1 is the
   base year (factor = 1.0).

2. **Timeseries**: Per-year inflation rates loaded from a CSV, yielding
   cumulative factor arrays.  Two base-year conventions:

   - **OPEX / contract prices** (EEG, PPA): base = commissioning year.
     Year 1 factor = 1.0.
   - **Spot prices**: base = year before first forecast year
     (= commissioning_year − 1).  Year 1 factor ≠ 1.0.
"""

from __future__ import annotations

import logging

import numpy as np

from pv_bess_model.config.defaults import DEFAULT_INFLATION_RATE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Original scalar helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------


def inflate_value(
    base_value: float,
    inflation_rate: float,
    year: int,
) -> float:
    """Apply compound inflation to a base value for a given project year.

    Year 1 is the base year (factor = 1.0). Inflation starts from year 2.

    Args:
        base_value: The value in the base year (year 1).
        inflation_rate: Annual inflation rate as decimal (e.g. 0.02 for 2 %).
        year: Project year (1-indexed). Year 1 = no inflation.

    Returns:
        Inflated value.
    """
    return base_value * (1.0 + inflation_rate) ** max(0, year - 1)


def inflate_series(
    base_values: np.ndarray,
    inflation_rate: float,
    year: int,
) -> np.ndarray:
    """Apply compound inflation to an array of base values for a given project year.

    Year 1 is the base year (factor = 1.0). Inflation starts from year 2.

    Args:
        base_values: Array of values in the base year.
        inflation_rate: Annual inflation rate as decimal.
        year: Project year (1-indexed). Year 1 = no inflation.

    Returns:
        Inflated array (same shape as input).
    """
    factor = (1.0 + inflation_rate) ** max(0, year - 1)
    return base_values * factor


def build_inflation_factors(
    inflation_rate: float,
    n_years: int,
) -> np.ndarray:
    """Build an array of cumulative inflation factors for each project year.

    Index 0 corresponds to year 1 (factor = 1.0, no inflation).
    Index 1 corresponds to year 2 (factor = (1 + rate)^1), etc.

    Args:
        inflation_rate: Annual inflation rate as decimal.
        n_years: Number of project years (length of output array).

    Returns:
        Array of shape ``(n_years,)`` where element ``i`` equals
        ``(1 + inflation_rate) ** i``.
    """
    years = np.arange(n_years)
    return (1.0 + inflation_rate) ** years


# ---------------------------------------------------------------------------
# Timeseries-based cumulative factor builders
# ---------------------------------------------------------------------------


def build_opex_inflation_factors(
    inflation_rate: float,
    lifetime_years: int,
    yearly_rates: dict[int, float] | None = None,
    commissioning_year: int | None = None,
) -> list[float]:
    """Build cumulative inflation factors for OPEX and contract prices.

    Base year = commissioning year → factor[0] (year 1) = 1.0.

    When *yearly_rates* is ``None``, falls back to the fixed-rate formula
    ``(1 + rate) ^ (y - 1)`` (backward compatible).

    When *yearly_rates* is provided, cumulative factors are computed from the
    per-year rates starting at commissioning_year + 1:

    - factor[0] = 1.0  (year 1 = base year, no inflation)
    - factor[1] = (1 + rate_{cy+1})
    - factor[2] = (1 + rate_{cy+1}) × (1 + rate_{cy+2})
    - ...

    Args:
        inflation_rate: Fixed annual inflation rate (fallback).
        lifetime_years: Number of project years.
        yearly_rates: Optional dict mapping calendar year → annual rate as
            fraction (e.g. {2027: 0.025, 2028: 0.02}).
        commissioning_year: Calendar year of commissioning.  Required when
            *yearly_rates* is provided.

    Returns:
        List of length *lifetime_years* with cumulative inflation factors.
    """
    if yearly_rates is None:
        return [(1.0 + inflation_rate) ** i for i in range(lifetime_years)]

    if commissioning_year is None:
        raise ValueError(
            "commissioning_year is required when yearly_rates is provided."
        )

    factors: list[float] = []
    cumulative = 1.0
    for y in range(lifetime_years):
        if y == 0:
            # Year 1 = commissioning year = base → no inflation
            factors.append(1.0)
        else:
            calendar_year = commissioning_year + y
            rate = yearly_rates.get(calendar_year, inflation_rate)
            if calendar_year not in yearly_rates:
                logger.warning(
                    "Inflation timeseries has no entry for year %d, "
                    "using fixed rate %.4f as fallback.",
                    calendar_year,
                    inflation_rate,
                )
            cumulative *= (1.0 + rate)
            factors.append(cumulative)
    return factors


def build_price_inflation_factors(
    inflation_rate: float,
    lifetime_years: int,
    yearly_rates: dict[int, float] | None = None,
    commissioning_year: int | None = None,
) -> list[float]:
    """Build cumulative inflation factors for spot electricity prices.

    Base year = year before first forecast year (= commissioning_year − 1).
    This means year 1 already carries one year of inflation.

    When *yearly_rates* is ``None``, falls back to the fixed-rate formula
    ``(1 + rate) ^ (y - 1)`` (backward compatible – year 1 = 1.0).

    When *yearly_rates* is provided:

    - factor[0] = (1 + rate_{cy})
    - factor[1] = (1 + rate_{cy}) × (1 + rate_{cy+1})
    - ...

    Args:
        inflation_rate: Fixed annual inflation rate (fallback).
        lifetime_years: Number of project years.
        yearly_rates: Optional dict mapping calendar year → annual rate.
        commissioning_year: Calendar year of commissioning.

    Returns:
        List of length *lifetime_years* with cumulative inflation factors.
    """
    if yearly_rates is None:
        # Backward compatible: year 1 factor = 1.0 (same as opex)
        return [(1.0 + inflation_rate) ** i for i in range(lifetime_years)]

    if commissioning_year is None:
        raise ValueError(
            "commissioning_year is required when yearly_rates is provided."
        )

    factors: list[float] = []
    cumulative = 1.0
    n = 0 # Start index for inflation rate post commissioning year
    for calendar_year in sorted(yearly_rates.keys()):
        if calendar_year < commissioning_year:
            n += 1
        rate = yearly_rates.get(calendar_year, inflation_rate)
        cumulative *= (1.0 + rate)
        factors.append(cumulative)

    # Check if inflation timeline is as long as lifetime
    relevant_factors = factors[n:lifetime_years+n]
    last_val = cumulative

    while len(relevant_factors) < lifetime_years:
        logger.warning(
            "Inflation timeseries has %d too less entries, "
            "using fixed rate %.4f as fallback.",
            lifetime_years - len(relevant_factors),
            inflation_rate,
        )
        last_val *= inflation_rate
        relevant_factors.append(last_val)

    return relevant_factors
