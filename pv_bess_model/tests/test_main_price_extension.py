"""Unit tests for price extension logic in pv_bess_model.main.

Covers FIX-10: All price columns (not just MID) must be extended to the
full project lifetime immediately after loading.  Grid search uses "mid",
Monte Carlo uses the pre-extended dict for all scenarios.
"""

from __future__ import annotations

import numpy as np
import pytest

from pv_bess_model.config.defaults import HOURS_PER_YEAR
from pv_bess_model.config.loader import PriceData
from pv_bess_model.main import _build_spot_prices_yearly, _extend_all_price_columns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_price_data(
    columns: dict[str, float],
    n_years: int = 1,
) -> PriceData:
    """Create a PriceData with constant values per column.

    Parameters
    ----------
    columns:
        Mapping of column name → constant price value (€/kWh).
    n_years:
        Number of full years of data.
    """
    n_hours = n_years * HOURS_PER_YEAR
    cols = {name: np.full(n_hours, val) for name, val in columns.items()}
    return PriceData(columns=cols, n_hours=n_hours, price_unit_input="eur_per_kwh")


def _make_price_data_per_year(
    columns: dict[str, list[float]],
) -> PriceData:
    """Create PriceData with different constant values per year per column.

    Parameters
    ----------
    columns:
        Mapping of column name → list of values, one per year.
        E.g. {"MID": [0.05, 0.06]} creates 2 years: year1=0.05, year2=0.06.
    """
    n_years = len(next(iter(columns.values())))
    n_hours = n_years * HOURS_PER_YEAR
    cols: dict[str, np.ndarray] = {}
    for name, year_vals in columns.items():
        arr = np.empty(n_hours)
        for y, val in enumerate(year_vals):
            arr[y * HOURS_PER_YEAR : (y + 1) * HOURS_PER_YEAR] = val
        cols[name] = arr
    return PriceData(columns=cols, n_hours=n_hours, price_unit_input="eur_per_kwh")


# ---------------------------------------------------------------------------
# _extend_all_price_columns
# ---------------------------------------------------------------------------


class TestExtendAllPriceColumns:
    """Tests for _extend_all_price_columns()."""

    def test_single_column_extended(self) -> None:
        """A single column is extended to the target years."""
        pd = _make_price_data({"MID": 0.05}, n_years=1)
        result = _extend_all_price_columns(pd, ["MID"], target_years=3)

        assert "MID" in result
        assert len(result["MID"]) == 3 * HOURS_PER_YEAR
        np.testing.assert_allclose(result["MID"], 0.05)

    def test_all_columns_extended(self) -> None:
        """All required columns (LOW, MID, HIGH) are extended, not just the first."""
        pd = _make_price_data({"LOW": 0.03, "MID": 0.05, "HIGH": 0.08}, n_years=1)
        result = _extend_all_price_columns(pd, ["LOW", "MID", "HIGH"], target_years=5)

        assert set(result.keys()) == {"LOW", "MID", "HIGH"}
        for col in ["LOW", "MID", "HIGH"]:
            assert len(result[col]) == 5 * HOURS_PER_YEAR

    def test_column_values_preserved(self) -> None:
        """Each column keeps its own values after extension."""
        pd = _make_price_data({"LOW": 0.03, "MID": 0.05, "HIGH": 0.08}, n_years=1)
        result = _extend_all_price_columns(pd, ["LOW", "MID", "HIGH"], target_years=3)

        np.testing.assert_allclose(result["LOW"], 0.03)
        np.testing.assert_allclose(result["MID"], 0.05)
        np.testing.assert_allclose(result["HIGH"], 0.08)

    def test_last_year_repeated_for_all_columns(self) -> None:
        """When CSV has 2 years, each column repeats its own last year."""
        pd = _make_price_data_per_year({
            "LOW": [0.03, 0.04],
            "HIGH": [0.08, 0.10],
        })
        result = _extend_all_price_columns(pd, ["LOW", "HIGH"], target_years=4)

        # Year 1 = original, Year 2 = original, Year 3+4 = repeat of year 2
        low_y3 = result["LOW"][2 * HOURS_PER_YEAR : 3 * HOURS_PER_YEAR]
        low_y4 = result["LOW"][3 * HOURS_PER_YEAR : 4 * HOURS_PER_YEAR]
        np.testing.assert_allclose(low_y3, 0.04)
        np.testing.assert_allclose(low_y4, 0.04)

        high_y3 = result["HIGH"][2 * HOURS_PER_YEAR : 3 * HOURS_PER_YEAR]
        high_y4 = result["HIGH"][3 * HOURS_PER_YEAR : 4 * HOURS_PER_YEAR]
        np.testing.assert_allclose(high_y3, 0.10)
        np.testing.assert_allclose(high_y4, 0.10)

    def test_exact_years_no_extension(self) -> None:
        """When CSV covers exactly the target years, no extension happens."""
        pd = _make_price_data({"MID": 0.05}, n_years=3)
        result = _extend_all_price_columns(pd, ["MID"], target_years=3)
        assert len(result["MID"]) == 3 * HOURS_PER_YEAR

    def test_longer_csv_truncated(self) -> None:
        """When CSV has more years than target, it is truncated."""
        pd = _make_price_data({"MID": 0.05}, n_years=5)
        result = _extend_all_price_columns(pd, ["MID"], target_years=3)
        assert len(result["MID"]) == 3 * HOURS_PER_YEAR

    def test_subset_of_columns_extended(self) -> None:
        """Only the requested columns are in the result."""
        pd = _make_price_data({"LOW": 0.03, "MID": 0.05, "HIGH": 0.08}, n_years=1)
        result = _extend_all_price_columns(pd, ["MID", "HIGH"], target_years=2)

        assert set(result.keys()) == {"MID", "HIGH"}
        assert "LOW" not in result

    def test_mc_and_grid_search_use_same_extended_data(self) -> None:
        """Verify that grid search (mid) and MC (all columns) can use the same dict.

        This is the core regression test for FIX-10: previously, only MID was
        extended for grid search, and LOW/HIGH were extended separately (and only
        when MC was enabled).  Now all columns are extended upfront.
        """
        pd = _make_price_data({"LOW": 0.03, "MID": 0.05, "HIGH": 0.08}, n_years=1)
        lifetime = 25
        extended = _extend_all_price_columns(
            pd, ["LOW", "MID", "HIGH"], target_years=lifetime
        )

        # Grid search uses MID
        assert len(extended["MID"]) == lifetime * HOURS_PER_YEAR

        # MC uses all three — they must all be available and correctly sized
        for col in ["LOW", "MID", "HIGH"]:
            assert col in extended
            assert len(extended[col]) == lifetime * HOURS_PER_YEAR

    def test_empty_column_list(self) -> None:
        """Edge case: no required columns → empty dict."""
        pd = _make_price_data({"MID": 0.05}, n_years=1)
        result = _extend_all_price_columns(pd, [], target_years=3)
        assert result == {}


# ---------------------------------------------------------------------------
# Integration: _build_spot_prices_yearly with extended prices
# ---------------------------------------------------------------------------


class TestBuildSpotPricesYearlyWithExtendedPrices:
    """Test that _build_spot_prices_yearly works correctly with multi-column
    extended prices (verifying the FIX-10 pipeline)."""

    def test_each_column_produces_correct_yearly_slices(self) -> None:
        """Extend all columns, then build yearly slices for each."""
        pd = _make_price_data({"LOW": 0.03, "MID": 0.05, "HIGH": 0.08}, n_years=1)
        lifetime = 3
        extended = _extend_all_price_columns(
            pd, ["LOW", "MID", "HIGH"], target_years=lifetime
        )

        for col, expected_val in [("LOW", 0.03), ("MID", 0.05), ("HIGH", 0.08)]:
            yearly = _build_spot_prices_yearly(
                extended[col],
                lifetime_years=lifetime,
                inflation_rate=0.0,
                apply_inflation=False,
            )
            assert len(yearly) == lifetime
            for year_arr in yearly:
                assert len(year_arr) == HOURS_PER_YEAR
                np.testing.assert_allclose(year_arr, expected_val)

    def test_inflation_applied_per_year(self) -> None:
        """Inflation is applied correctly to each year slice."""
        pd = _make_price_data({"MID": 0.05}, n_years=1)
        extended = _extend_all_price_columns(pd, ["MID"], target_years=3)

        yearly = _build_spot_prices_yearly(
            extended["MID"],
            lifetime_years=3,
            inflation_rate=0.02,
            apply_inflation=True,
        )

        # Year 1: factor = (1.02)^0 = 1.0 (inflation starts year 2)
        # inflate_value(1.0, 0.02, 1) = 1.0  (exponent = max(0, 1-1) = 0)
        np.testing.assert_allclose(yearly[0], 0.05, rtol=1e-9)
        # Year 2: factor = (1.02)^1
        np.testing.assert_allclose(yearly[1], 0.05 * 1.02, rtol=1e-9)
        # Year 3: factor = (1.02)^2
        np.testing.assert_allclose(yearly[2], 0.05 * 1.02**2, rtol=1e-9)

    def test_different_columns_independent(self) -> None:
        """Extending and slicing LOW/HIGH gives different values."""
        pd = _make_price_data({"LOW": 0.02, "HIGH": 0.10}, n_years=1)
        lifetime = 2
        extended = _extend_all_price_columns(
            pd, ["LOW", "HIGH"], target_years=lifetime
        )

        yearly_low = _build_spot_prices_yearly(
            extended["LOW"], lifetime, 0.0, False
        )
        yearly_high = _build_spot_prices_yearly(
            extended["HIGH"], lifetime, 0.0, False
        )

        np.testing.assert_allclose(yearly_low[0], 0.02)
        np.testing.assert_allclose(yearly_high[0], 0.10)
        # LOW and HIGH must not be mixed up
        assert not np.allclose(yearly_low[0], yearly_high[0])
