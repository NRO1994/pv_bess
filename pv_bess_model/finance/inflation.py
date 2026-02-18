"""Inflation escalation logic for OPEX, PPA price, EEG price, and spot prices.

Formula: inflated_value[year] = base_value × (1 + inflation_rate) ^ year
"""

from __future__ import annotations

import numpy as np

from pv_bess_model.config.defaults import DEFAULT_INFLATION_RATE


def inflate_value(
    base_value: float,
    inflation_rate: float,
    year: int,
) -> float:
    """Apply compound inflation to a base value for a given year.

    Args:
        base_value: The value in year 0 (base year).
        inflation_rate: Annual inflation rate as decimal (e.g. 0.02 for 2 %).
        year: Number of years of inflation to apply (0-indexed: year 0 = no inflation).

    Returns:
        Inflated value.
    """
    return base_value * (1.0 + inflation_rate) ** year


def inflate_series(
    base_values: np.ndarray,
    inflation_rate: float,
    year: int,
) -> np.ndarray:
    """Apply compound inflation to an array of base values for a given year.

    Args:
        base_values: Array of values in the base year.
        inflation_rate: Annual inflation rate as decimal.
        year: Number of years of inflation to apply.

    Returns:
        Inflated array (same shape as input).
    """
    factor = (1.0 + inflation_rate) ** year
    return base_values * factor


def build_inflation_factors(
    inflation_rate: float,
    n_years: int,
) -> np.ndarray:
    """Build an array of cumulative inflation factors for each project year.

    Args:
        inflation_rate: Annual inflation rate as decimal.
        n_years: Number of years (length of output array).

    Returns:
        Array of shape ``(n_years,)`` where element ``i`` equals
        ``(1 + inflation_rate) ** i``.
    """
    years = np.arange(n_years)
    return (1.0 + inflation_rate) ** years
