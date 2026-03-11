"""Integration tests for the PVGIS API client.

These tests make **real HTTP requests** to the public PVGIS API and are
excluded from the default test run.  Execute them explicitly with::

    pytest -m integration

Requirements
------------
- Network access to https://re.jrc.ec.europa.eu/api/v5_3/
- The PVGIS API must be reachable and respond within the configured timeout.

Test location
-------------
Hamburg, Germany (53.55 °N, 9.99 °E) – a well-known reference site covered
by the PVGIS-SARAH3 database.  Using a small 1 kWp system minimises response
payload while still producing meaningful production values.

Expected PVGIS-SARAH3 coverage: 2005 – 2020 (16 years, ≥ 10 required).
Expected annual yield: 700 – 1 200 kWh for 1 kWp in Hamburg.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pv_bess_model.config.defaults import HOURS_PER_YEAR, INTERVALS_PER_YEAR
from pv_bess_model.pv.pvgis_client import PVGISClient

# ---------------------------------------------------------------------------
# Test site parameters
# ---------------------------------------------------------------------------

_LAT: float = 53.55
_LON: float = 9.99
_PEAK_POWER_KWP: float = 1.0  # small system to keep response compact
_SYSTEM_LOSS_PCT: float = 14.0
_MOUNTING_TYPE: str = "free"
_AZIMUTH_DEG: float = 0.0
_TILT_DEG: float = 30.0
_DATABASE: str = "PVGIS-SARAH3"

# PVGIS-SARAH3 currently covers 2005–2020 → at least 10 years expected
_MIN_YEARS: int = 10

# Plausible annual yield range for 1 kWp in Hamburg (kWh)
_ANNUAL_KWH_MIN: float = 700.0
_ANNUAL_KWH_MAX: float = 1_200.0


# ---------------------------------------------------------------------------
# Class-scoped fixture – one API call shared across all tests in the class
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def pvgis_yearly_data() -> dict[int, np.ndarray]:
    """Fetch PVGIS data once per class; no on-disk cache to ensure freshness."""
    client = PVGISClient(cache_dir=None)
    return client.fetch_hourly_production(
        latitude=_LAT,
        longitude=_LON,
        peak_power_kwp=_PEAK_POWER_KWP,
        system_loss_pct=_SYSTEM_LOSS_PCT,
        mounting_type=_MOUNTING_TYPE,
        azimuth_deg=_AZIMUTH_DEG,
        tilt_deg=_TILT_DEG,
        pvgis_database=_DATABASE,
    )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPVGISClientIntegration:
    """Real PVGIS API calls with known coordinates (Hamburg, PVGIS-SARAH3)."""

    def test_returns_at_least_min_years(
        self, pvgis_yearly_data: dict[int, np.ndarray]
    ) -> None:
        """API must return at least _MIN_YEARS of historical data."""
        assert len(pvgis_yearly_data) >= _MIN_YEARS, (
            f"Expected ≥ {_MIN_YEARS} years, got {len(pvgis_yearly_data)}"
        )

    def test_all_years_have_8760_values(
        self, pvgis_yearly_data: dict[int, np.ndarray]
    ) -> None:
        """Each year's array must have exactly 8 760 hourly values."""
        wrong = {
            year: len(arr)
            for year, arr in pvgis_yearly_data.items()
            if len(arr) != HOURS_PER_YEAR
        }
        assert not wrong, (
            f"Years with wrong length: "
            + ", ".join(f"{y}: {n}" for y, n in sorted(wrong.items()))
        )

    def test_years_form_consecutive_sequence(
        self, pvgis_yearly_data: dict[int, np.ndarray]
    ) -> None:
        """Returned years must form an unbroken consecutive sequence."""
        years = sorted(pvgis_yearly_data.keys())
        expected = list(range(years[0], years[-1] + 1))
        assert years == expected, f"Non-consecutive years: {years}"

    def test_result_keys_are_int_values_are_ndarray(
        self, pvgis_yearly_data: dict[int, np.ndarray]
    ) -> None:
        """Return type must be dict[int, numpy.ndarray]."""
        for year, arr in pvgis_yearly_data.items():
            assert isinstance(year, int), f"Key {year!r} is not int"
            assert isinstance(arr, np.ndarray), f"Value for {year} is not ndarray"
            assert np.issubdtype(arr.dtype, np.floating), (
                f"Year {year} dtype {arr.dtype} is not floating"
            )

    def test_all_values_non_negative(
        self, pvgis_yearly_data: dict[int, np.ndarray]
    ) -> None:
        """PV production values must never be negative."""
        for year, arr in pvgis_yearly_data.items():
            assert np.all(arr >= 0.0), (
                f"Year {year}: {int(np.sum(arr < 0))} negative values found"
            )

    def test_annual_production_in_plausible_range(
        self, pvgis_yearly_data: dict[int, np.ndarray]
    ) -> None:
        """Annual yield for 1 kWp in Hamburg must be within 700–1 200 kWh."""
        for year, arr in pvgis_yearly_data.items():
            annual_kwh = float(np.sum(arr))
            assert _ANNUAL_KWH_MIN <= annual_kwh <= _ANNUAL_KWH_MAX, (
                f"Year {year}: annual yield {annual_kwh:.0f} kWh outside"
                f" expected range [{_ANNUAL_KWH_MIN:.0f}, {_ANNUAL_KWH_MAX:.0f}] kWh"
            )

    def test_summer_production_exceeds_winter(
        self, pvgis_yearly_data: dict[int, np.ndarray]
    ) -> None:
        """July production should exceed January production (seasonal effect)."""
        # Use the first available year
        first_year = min(pvgis_yearly_data.keys())
        arr = pvgis_yearly_data[first_year]

        # January: hours 0–743 (31 days × 24), July: hours 4344–5087 (31 days × 24)
        jan_kwh = float(np.sum(arr[0:744]))
        jul_kwh = float(np.sum(arr[4344:5088]))

        assert jul_kwh > jan_kwh, (
            f"Year {first_year}: July ({jul_kwh:.1f} kWh) not greater than"
            f" January ({jan_kwh:.1f} kWh)"
        )

    def test_daytime_hours_have_positive_production(
        self, pvgis_yearly_data: dict[int, np.ndarray]
    ) -> None:
        """On a summer day, at least some midday hours must produce power."""
        # Use the first available year, mid-July day (day 196 of year, 0-indexed)
        first_year = min(pvgis_yearly_data.keys())
        arr = pvgis_yearly_data[first_year]

        # Extract a week in mid-July (days 196–202) and check total > 0
        mid_july_start = 196 * 24
        mid_july_end = 203 * 24
        mid_july_kwh = float(np.sum(arr[mid_july_start:mid_july_end]))

        assert mid_july_kwh > 0.0, (
            f"Year {first_year}: mid-July production is zero – unexpected"
        )

    def test_reference_csv_readable_by_load_price_csv(
        self, data_dir: Path
    ) -> None:
        """The reference_prices.csv in .data/ can be loaded by load_price_csv."""
        from pv_bess_model.config.loader import load_price_csv

        csv_path = data_dir / "integration_test_inputs" / "suite" / "integration_suite_prices.csv"
        assert csv_path.exists(), f"Reference CSV not found: {csv_path}"

        price_data = load_price_csv(
            csv_path,
            required_columns=["LOW", "MID", "HIGH"],
            decimal=","
        )
        assert price_data.n_hours == HOURS_PER_YEAR * 20
        assert "LOW" in price_data.columns
        assert "MID" in price_data.columns
        assert "HIGH" in price_data.columns
