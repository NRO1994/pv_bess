"""Inflation escalation logic for OPEX, PPA price, EEG price, and spot prices.

Year 1 is the base year (no inflation applied). Inflation begins in year 2:

    inflated_value[year] = base_value × (1 + inflation_rate) ^ max(0, year - 1)

This means year 1 returns the base value unchanged, year 2 returns
``base × (1 + rate)``, year 3 returns ``base × (1 + rate)²``, etc.
"""

from __future__ import annotations

import numpy as np

from pv_bess_model.config.defaults import DEFAULT_INFLATION_RATE


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
