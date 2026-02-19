"""Unit tests for pv_bess_model.market.price_loader."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pv_bess_model.config.defaults import HOURS_PER_YEAR
from pv_bess_model.market.price_loader import (
    MarketPrices,
    collect_scenario_columns,
    get_year_prices,
    load_market_prices,
)


# ---------------------------------------------------------------------------
# CSV writing helpers
# ---------------------------------------------------------------------------


def _write_price_csv(
    path: Path,
    n_years: int,
    columns: dict[str, float],
    year2_multiplier: float = 2.0,
) -> None:
    """Write a synthetic price CSV to *path*.

    Year 1 values are the base values in *columns*.
    Year 2+ values are base × year2_multiplier (to enable extension tests).

    Parameters
    ----------
    path:
        Destination file path.
    n_years:
        Number of full years to write (n_years × 8760 rows).
    columns:
        Mapping of column name → year-1 base price value (€/MWh or €/kWh).
    year2_multiplier:
        Factor applied to year 2+ rows.
    """
    n_rows = n_years * HOURS_PER_YEAR
    timestamps = [f"2020-01-01T{h % 24:02d}:00:00" for h in range(n_rows)]
    data: dict[str, list] = {"timestamp": timestamps}
    for col, base_val in columns.items():
        row_vals = []
        for i in range(n_rows):
            year_idx = i // HOURS_PER_YEAR  # 0-indexed
            val = base_val if year_idx == 0 else base_val * year2_multiplier
            row_vals.append(val)
        data[col] = row_vals
    df = pd.DataFrame(data)
    df.to_csv(path, index=False, sep=";")


def _write_price_csv_constant(
    path: Path,
    n_rows: int,
    columns: dict[str, float],
) -> None:
    """Write a CSV with constant values in each column."""
    timestamps = [f"2020-01-01T{h % 24:02d}:00:00" for h in range(n_rows)]
    data: dict[str, list] = {"timestamp": timestamps}
    for col, val in columns.items():
        data[col] = [val] * n_rows
    pd.DataFrame(data).to_csv(path, index=False, sep=";")


# ---------------------------------------------------------------------------
# MarketPrices – unit tests (no I/O)
# ---------------------------------------------------------------------------


class TestMarketPrices:
    """Tests for the MarketPrices container class."""

    @pytest.fixture
    def market_prices_2y(self) -> MarketPrices:
        """Two columns, 2 years each."""
        mid = np.linspace(0.05, 0.07, 2 * HOURS_PER_YEAR)
        high = mid * 1.2
        return MarketPrices(columns={"MID": mid, "HIGH": high}, lifetime_years=2)

    def test_get_column_returns_correct_array(self, market_prices_2y: MarketPrices) -> None:
        col = market_prices_2y.get_column("MID")
        assert len(col) == 2 * HOURS_PER_YEAR

    def test_get_column_missing_raises_key_error(self, market_prices_2y: MarketPrices) -> None:
        with pytest.raises(KeyError, match="LOW"):
            market_prices_2y.get_column("LOW")

    def test_get_year_prices_year_1_correct_slice(self, market_prices_2y: MarketPrices) -> None:
        full = market_prices_2y.get_column("MID")
        year1 = market_prices_2y.get_year_prices("MID", year=1)
        assert len(year1) == HOURS_PER_YEAR
        np.testing.assert_array_equal(year1, full[:HOURS_PER_YEAR])

    def test_get_year_prices_year_2_correct_slice(self, market_prices_2y: MarketPrices) -> None:
        full = market_prices_2y.get_column("MID")
        year2 = market_prices_2y.get_year_prices("MID", year=2)
        np.testing.assert_array_equal(year2, full[HOURS_PER_YEAR:])

    def test_get_year_prices_out_of_range_raises(self, market_prices_2y: MarketPrices) -> None:
        with pytest.raises(ValueError, match="year must be in"):
            market_prices_2y.get_year_prices("MID", year=3)

    def test_get_year_prices_year_zero_raises(self, market_prices_2y: MarketPrices) -> None:
        with pytest.raises(ValueError):
            market_prices_2y.get_year_prices("MID", year=0)

    def test_get_year_prices_missing_column_raises(self, market_prices_2y: MarketPrices) -> None:
        with pytest.raises(KeyError):
            market_prices_2y.get_year_prices("LOW", year=1)

    def test_column_length_matches_lifetime(self, market_prices_2y: MarketPrices) -> None:
        for name, arr in market_prices_2y.columns.items():
            assert len(arr) == market_prices_2y.lifetime_years * HOURS_PER_YEAR


# ---------------------------------------------------------------------------
# load_market_prices() – loading and unit conversion
# ---------------------------------------------------------------------------


class TestLoadMarketPricesConversion:
    """Tests for price_unit conversion in load_market_prices()."""

    def test_eur_per_mwh_converted_to_eur_per_kwh(self, tmp_path: Path) -> None:
        """€/MWh values must be divided by 1000 when price_unit='eur_per_mwh'."""
        csv = tmp_path / "prices.csv"
        _write_price_csv_constant(csv, n_rows=HOURS_PER_YEAR, columns={"MID": 50.0})

        mp = load_market_prices(
            csv_path=csv,
            required_columns=["MID"],
            price_unit="eur_per_mwh",
            lifetime_years=1,
        )
        # 50 €/MWh → 0.05 €/kWh
        assert math.isclose(float(mp.get_column("MID")[0]), 0.05, rel_tol=1e-9)

    def test_eur_per_kwh_no_conversion(self, tmp_path: Path) -> None:
        """€/kWh values are used as-is (no division)."""
        csv = tmp_path / "prices.csv"
        _write_price_csv_constant(csv, n_rows=HOURS_PER_YEAR, columns={"MID": 0.08})

        mp = load_market_prices(
            csv_path=csv,
            required_columns=["MID"],
            price_unit="eur_per_kwh",
            lifetime_years=1,
        )
        assert math.isclose(float(mp.get_column("MID")[0]), 0.08, rel_tol=1e-9)

    def test_all_values_converted(self, tmp_path: Path) -> None:
        """Every value in the column is converted, not just the first."""
        csv = tmp_path / "prices.csv"
        # Write alternating 40 / 60 values
        timestamps = [f"2020-01-01T{h % 24:02d}:00:00" for h in range(HOURS_PER_YEAR)]
        vals = [40.0 if i % 2 == 0 else 60.0 for i in range(HOURS_PER_YEAR)]
        pd.DataFrame({"timestamp": timestamps, "MID": vals}).to_csv(csv, index=False, sep=";")

        mp = load_market_prices(
            csv_path=csv,
            required_columns=["MID"],
            price_unit="eur_per_mwh",
            lifetime_years=1,
        )
        col = mp.get_column("MID")
        # Odd indices should be 0.060, even 0.040
        np.testing.assert_allclose(col[0::2], 0.04, rtol=1e-9)
        np.testing.assert_allclose(col[1::2], 0.06, rtol=1e-9)

    def test_multiple_columns_loaded(self, tmp_path: Path) -> None:
        """All requested columns are loaded and converted."""
        csv = tmp_path / "prices.csv"
        _write_price_csv_constant(
            csv,
            n_rows=HOURS_PER_YEAR,
            columns={"LOW": 30.0, "MID": 50.0, "HIGH": 80.0},
        )

        mp = load_market_prices(
            csv_path=csv,
            required_columns=["LOW", "MID", "HIGH"],
            price_unit="eur_per_mwh",
            lifetime_years=1,
        )
        assert math.isclose(float(mp.get_column("LOW")[0]), 0.03)
        assert math.isclose(float(mp.get_column("MID")[0]), 0.05)
        assert math.isclose(float(mp.get_column("HIGH")[0]), 0.08)

    def test_negative_prices_preserved(self, tmp_path: Path) -> None:
        """Negative spot prices are valid and must survive conversion."""
        csv = tmp_path / "prices.csv"
        timestamps = [f"2020-01-01T{h % 24:02d}:00:00" for h in range(HOURS_PER_YEAR)]
        vals = [-20.0] * HOURS_PER_YEAR
        pd.DataFrame({"timestamp": timestamps, "MID": vals}).to_csv(csv, index=False, sep=";")

        mp = load_market_prices(
            csv_path=csv,
            required_columns=["MID"],
            price_unit="eur_per_mwh",
            lifetime_years=1,
        )
        assert math.isclose(float(mp.get_column("MID")[0]), -0.02)


# ---------------------------------------------------------------------------
# load_market_prices() – timeseries extension
# ---------------------------------------------------------------------------


class TestLoadMarketPricesExtension:
    """Tests for year-repetition logic when lifetime > CSV data."""

    def test_exact_match_no_extension(self, tmp_path: Path) -> None:
        """When CSV covers exactly lifetime_years, no extension needed."""
        csv = tmp_path / "prices.csv"
        _write_price_csv(csv, n_years=2, columns={"MID": 50.0}, year2_multiplier=3.0)

        mp = load_market_prices(
            csv_path=csv,
            required_columns=["MID"],
            price_unit="eur_per_mwh",
            lifetime_years=2,
        )
        assert len(mp.get_column("MID")) == 2 * HOURS_PER_YEAR

    def test_extension_repeats_last_year(self, tmp_path: Path) -> None:
        """When lifetime > CSV years, the last full year in CSV is repeated."""
        csv = tmp_path / "prices.csv"
        # Year 1: 50 €/MWh → 0.05 €/kWh; Year 2: 100 €/MWh → 0.10 €/kWh
        _write_price_csv(csv, n_years=2, columns={"MID": 50.0}, year2_multiplier=2.0)

        mp = load_market_prices(
            csv_path=csv,
            required_columns=["MID"],
            price_unit="eur_per_mwh",
            lifetime_years=5,
        )
        # Years 3, 4, 5 should all equal year 2 values (0.10 €/kWh)
        year2 = mp.get_year_prices("MID", year=2)
        for yr in [3, 4, 5]:
            np.testing.assert_array_equal(mp.get_year_prices("MID", year=yr), year2)

    def test_extension_single_year_csv(self, tmp_path: Path) -> None:
        """A 1-year CSV is repeated for all project years."""
        csv = tmp_path / "prices.csv"
        _write_price_csv_constant(csv, n_rows=HOURS_PER_YEAR, columns={"MID": 60.0})

        mp = load_market_prices(
            csv_path=csv,
            required_columns=["MID"],
            price_unit="eur_per_mwh",
            lifetime_years=25,
        )
        assert len(mp.get_column("MID")) == 25 * HOURS_PER_YEAR
        # All values should be 0.06 €/kWh
        np.testing.assert_allclose(mp.get_column("MID"), 0.06, rtol=1e-9)

    def test_output_length_correct(self, tmp_path: Path) -> None:
        """Output array length equals lifetime_years × HOURS_PER_YEAR."""
        csv = tmp_path / "prices.csv"
        _write_price_csv_constant(csv, n_rows=HOURS_PER_YEAR, columns={"MID": 50.0})

        for lifetime in [1, 5, 10, 25]:
            mp = load_market_prices(
                csv_path=csv,
                required_columns=["MID"],
                price_unit="eur_per_mwh",
                lifetime_years=lifetime,
            )
            assert len(mp.get_column("MID")) == lifetime * HOURS_PER_YEAR

    def test_get_year_prices_year_1_values(self, tmp_path: Path) -> None:
        """Year 1 prices match the first year of the CSV."""
        csv = tmp_path / "prices.csv"
        _write_price_csv_constant(csv, n_rows=HOURS_PER_YEAR, columns={"MID": 40.0})

        mp = load_market_prices(
            csv_path=csv,
            required_columns=["MID"],
            price_unit="eur_per_mwh",
            lifetime_years=3,
        )
        year1 = mp.get_year_prices("MID", year=1)
        assert len(year1) == HOURS_PER_YEAR
        np.testing.assert_allclose(year1, 0.04, rtol=1e-9)


# ---------------------------------------------------------------------------
# load_market_prices() – validation errors
# ---------------------------------------------------------------------------


class TestLoadMarketPricesValidation:
    """Tests for validation errors propagated from load_price_csv."""

    def test_missing_column_raises_value_error(self, tmp_path: Path) -> None:
        """Requesting a column not present in the CSV raises ValueError."""
        csv = tmp_path / "prices.csv"
        _write_price_csv_constant(csv, n_rows=HOURS_PER_YEAR, columns={"MID": 50.0})

        with pytest.raises(ValueError, match="missing"):
            load_market_prices(
                csv_path=csv,
                required_columns=["MISSING_COL"],
                price_unit="eur_per_mwh",
                lifetime_years=1,
            )

    def test_too_few_rows_raises_value_error(self, tmp_path: Path) -> None:
        """A CSV with fewer than 8760 rows raises ValueError."""
        csv = tmp_path / "prices.csv"
        # Write only 100 rows
        timestamps = [f"2020-01-01T{h % 24:02d}:00:00" for h in range(100)]
        pd.DataFrame({"timestamp": timestamps, "MID": [50.0] * 100}).to_csv(csv, index=False, sep=";")

        with pytest.raises(ValueError, match="8760"):
            load_market_prices(
                csv_path=csv,
                required_columns=["MID"],
                price_unit="eur_per_mwh",
                lifetime_years=1,
            )

    def test_nan_in_column_raises_value_error(self, tmp_path: Path) -> None:
        """NaN values in required columns raise ValueError."""
        csv = tmp_path / "prices.csv"
        vals = [50.0] * HOURS_PER_YEAR
        vals[100] = float("nan")
        timestamps = [f"2020-01-01T{h % 24:02d}:00:00" for h in range(HOURS_PER_YEAR)]
        pd.DataFrame({"timestamp": timestamps, "MID": vals}).to_csv(csv, index=False, sep=";")

        with pytest.raises(ValueError, match="NaN"):
            load_market_prices(
                csv_path=csv,
                required_columns=["MID"],
                price_unit="eur_per_mwh",
                lifetime_years=1,
            )

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """A non-existent file raises FileNotFoundError."""
        csv = tmp_path / "nonexistent.csv"
        with pytest.raises(FileNotFoundError):
            load_market_prices(
                csv_path=csv,
                required_columns=["MID"],
                price_unit="eur_per_mwh",
                lifetime_years=1,
            )

    def test_invalid_price_unit_raises(self, tmp_path: Path) -> None:
        """An unknown price_unit raises ValueError."""
        csv = tmp_path / "prices.csv"
        _write_price_csv_constant(csv, n_rows=HOURS_PER_YEAR, columns={"MID": 50.0})

        with pytest.raises(ValueError, match="price_unit"):
            load_market_prices(
                csv_path=csv,
                required_columns=["MID"],
                price_unit="eur_per_ton",
                lifetime_years=1,
            )


# ---------------------------------------------------------------------------
# get_year_prices() standalone function
# ---------------------------------------------------------------------------


class TestGetYearPricesStandalone:
    """Tests for the module-level get_year_prices() helper."""

    @pytest.fixture
    def prices_3y(self) -> np.ndarray:
        """3-year price array with year-index encoded as value (1-indexed)."""
        arr = np.empty(3 * HOURS_PER_YEAR)
        for y in range(3):
            arr[y * HOURS_PER_YEAR : (y + 1) * HOURS_PER_YEAR] = float(y + 1)
        return arr

    def test_year_1_correct_values(self, prices_3y: np.ndarray) -> None:
        year1 = get_year_prices(prices_3y, year=1)
        assert len(year1) == HOURS_PER_YEAR
        np.testing.assert_array_equal(year1, np.ones(HOURS_PER_YEAR))

    def test_year_2_correct_values(self, prices_3y: np.ndarray) -> None:
        year2 = get_year_prices(prices_3y, year=2)
        np.testing.assert_array_equal(year2, np.full(HOURS_PER_YEAR, 2.0))

    def test_year_3_correct_values(self, prices_3y: np.ndarray) -> None:
        year3 = get_year_prices(prices_3y, year=3)
        np.testing.assert_array_equal(year3, np.full(HOURS_PER_YEAR, 3.0))

    def test_year_beyond_array_raises(self, prices_3y: np.ndarray) -> None:
        with pytest.raises(ValueError, match="too short"):
            get_year_prices(prices_3y, year=4)

    def test_output_length_is_hours_per_year(self, prices_3y: np.ndarray) -> None:
        for yr in [1, 2, 3]:
            result = get_year_prices(prices_3y, year=yr)
            assert len(result) == HOURS_PER_YEAR


# ---------------------------------------------------------------------------
# collect_scenario_columns()
# ---------------------------------------------------------------------------


class TestCollectScenarioColumns:
    """Tests for collect_scenario_columns()."""

    def test_single_scenario(self) -> None:
        scenarios = {"mid": {"csv_column": "MID", "weight": 1.0}}
        result = collect_scenario_columns(scenarios)
        assert result == ["MID"]

    def test_three_scenarios_sorted(self) -> None:
        scenarios = {
            "low": {"csv_column": "LOW", "weight": 0.25},
            "mid": {"csv_column": "MID", "weight": 0.50},
            "high": {"csv_column": "HIGH", "weight": 0.25},
        }
        result = collect_scenario_columns(scenarios)
        assert result == ["HIGH", "LOW", "MID"]

    def test_deduplication(self) -> None:
        """Two scenario names mapping to same CSV column → one entry."""
        scenarios = {
            "base": {"csv_column": "MID", "weight": 0.60},
            "alt": {"csv_column": "MID", "weight": 0.40},
        }
        result = collect_scenario_columns(scenarios)
        assert result == ["MID"]

    def test_empty_scenarios(self) -> None:
        result = collect_scenario_columns({})
        assert result == []

    def test_returns_sorted_list(self) -> None:
        scenarios = {
            "z": {"csv_column": "ZZZ", "weight": 0.5},
            "a": {"csv_column": "AAA", "weight": 0.5},
        }
        result = collect_scenario_columns(scenarios)
        assert result == sorted(result)
