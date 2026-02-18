"""Load CSV price timeseries and extend to full project lifetime.

This module provides the high-level interface for loading day-ahead electricity
prices from CSV files and preparing them for use in the dispatch engine.  It
delegates low-level CSV parsing and validation to :mod:`pv_bess_model.config.loader`.

Public API
----------
load_market_prices  – Load and extend price data for one or more MC scenarios.
get_year_prices     – Slice one year from an extended price array.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from pv_bess_model.config.defaults import HOURS_PER_YEAR
from pv_bess_model.config.loader import (
    PriceData,
    extend_price_timeseries,
    load_price_csv,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed result container
# ---------------------------------------------------------------------------


class MarketPrices:
    """Extended price timeseries ready for multi-year simulation.

    Each column is an extended array of length ``lifetime_years × HOURS_PER_YEAR``
    in **€/kWh**.

    Attributes
    ----------
    columns : dict[str, np.ndarray]
        Mapping of column name → extended price array (€/kWh).
    lifetime_years : int
        Number of project years covered.
    """

    def __init__(
        self,
        columns: dict[str, np.ndarray],
        lifetime_years: int,
    ) -> None:
        self.columns = columns
        self.lifetime_years = lifetime_years

    def get_column(self, name: str) -> np.ndarray:
        """Return the full extended price array for *name* (€/kWh).

        Raises
        ------
        KeyError
            If *name* is not a known column.
        """
        if name not in self.columns:
            available = ", ".join(sorted(self.columns))
            raise KeyError(
                f"Price column '{name}' not found. "
                f"Available columns: {available}"
            )
        return self.columns[name]

    def get_year_prices(self, name: str, year: int) -> np.ndarray:
        """Return the hourly prices for a single project year.

        Parameters
        ----------
        name:
            Column name (e.g. ``"MID"``).
        year:
            1-indexed project year (1 … lifetime_years).

        Returns
        -------
        np.ndarray
            Array of length :data:`HOURS_PER_YEAR` (€/kWh).

        Raises
        ------
        ValueError
            If *year* is out of range.
        KeyError
            If *name* is not a known column.
        """
        if year < 1 or year > self.lifetime_years:
            raise ValueError(
                f"year must be in [1, {self.lifetime_years}], got {year}"
            )
        full = self.get_column(name)
        start = (year - 1) * HOURS_PER_YEAR
        return full[start : start + HOURS_PER_YEAR]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def load_market_prices(
    csv_path: str | Path,
    required_columns: list[str],
    price_unit: str,
    lifetime_years: int,
) -> MarketPrices:
    """Load a price CSV file and extend all columns to the full project lifetime.

    This is the main entry-point used by the dispatch engine and grid search.
    It delegates CSV parsing/validation to :func:`config.loader.load_price_csv`
    and extends every required column to ``lifetime_years`` full years using
    :func:`config.loader.extend_price_timeseries`.

    Parameters
    ----------
    csv_path:
        Path to the day-ahead price CSV file.
    required_columns:
        Column names to load (e.g. ``["MID"]`` or ``["LOW", "MID", "HIGH"]``).
    price_unit:
        Unit of the price values in the CSV (``"eur_per_mwh"`` or ``"eur_per_kwh"``).
    lifetime_years:
        Number of project years to cover.

    Returns
    -------
    MarketPrices
        Extended price data with all columns covering the full project lifetime.
    """
    price_data: PriceData = load_price_csv(
        path=csv_path,
        required_columns=required_columns,
        price_unit=price_unit,
    )

    extended: dict[str, np.ndarray] = {}
    for col_name, col_array in price_data.columns.items():
        extended[col_name] = extend_price_timeseries(
            prices=col_array,
            target_years=lifetime_years,
            hours_per_year=HOURS_PER_YEAR,
        )

    logger.info(
        "Extended price timeseries to %d years (%d hours per column).",
        lifetime_years,
        lifetime_years * HOURS_PER_YEAR,
    )

    return MarketPrices(columns=extended, lifetime_years=lifetime_years)


def get_year_prices(
    prices: np.ndarray,
    year: int,
) -> np.ndarray:
    """Slice a single project year from a full-lifetime price array.

    This is a convenience function for code that already holds the raw array
    rather than a :class:`MarketPrices` instance.

    Parameters
    ----------
    prices:
        Full price array of length ``lifetime_years × HOURS_PER_YEAR``.
    year:
        1-indexed project year.

    Returns
    -------
    np.ndarray
        Array of length :data:`HOURS_PER_YEAR`.
    """
    start = (year - 1) * HOURS_PER_YEAR
    end = start + HOURS_PER_YEAR
    if end > len(prices):
        raise ValueError(
            f"Price array too short for year {year}: need {end} values, "
            f"have {len(prices)}."
        )
    return prices[start:end]


def collect_scenario_columns(
    price_scenarios: dict[str, dict[str, Any]],
) -> list[str]:
    """Extract the list of CSV column names from the MC price_scenarios block.

    Parameters
    ----------
    price_scenarios:
        The ``scenario.monte_carlo.price_scenarios`` dict, e.g.::

            {"low": {"csv_column": "LOW", "weight": 0.25}, ...}

    Returns
    -------
    list[str]
        Sorted, deduplicated list of CSV column names to load.
    """
    return sorted({v["csv_column"] for v in price_scenarios.values()})
