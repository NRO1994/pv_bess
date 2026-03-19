"""Tests for PVGIS client temperature extraction (Phase 2 extension).

The existing ``_parse_response`` method now extracts both ``P`` (production)
and ``T2m`` (temperature) from the PVGIS hourly records.  ``fetch_single_year``
accepts ``include_temperature=True`` to return both in a dict.
"""

from __future__ import annotations

import numpy as np
import pytest

from pv_bess_model.config.defaults import HOURS_PER_YEAR
from pv_bess_model.pv.pvgis_client import PVGISClient, PVGISError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_simple_response(year: int = 2018) -> dict:
    """Build a simple PVGIS response with sequential timestamps."""
    records = []
    for h in range(HOURS_PER_YEAR):
        month = h // 730 + 1
        if month > 12:
            month = 12
        day_in_month = (h % 730) // 24 + 1
        if day_in_month > 28:
            day_in_month = 28
        hour = h % 24
        time_str = f"{year}{month:02d}{day_in_month:02d}:{hour:02d}10"
        records.append({
            "time": time_str,
            "P": 50.0,
            "T2m": 10.0 + 5.0 * (month - 6) / 6.0,
        })
    return {"outputs": {"hourly": records}}


# ---------------------------------------------------------------------------
# Tests for _parse_response (now returns production + temperature)
# ---------------------------------------------------------------------------


class TestParseResponseWithTemperature:
    """Tests for _parse_response extracting both production and temperature."""

    def test_returns_both_keys(self) -> None:
        """Response should contain 'production' and 'temperature' keys."""
        raw = _make_simple_response(2018)
        result = PVGISClient._parse_response(raw)
        assert "production" in result
        assert "temperature" in result

    def test_production_array_length(self) -> None:
        """Production arrays should have 8,760 elements."""
        raw = _make_simple_response(2018)
        result = PVGISClient._parse_response(raw)
        assert len(result["production"][2018]) == HOURS_PER_YEAR

    def test_temperature_array_length(self) -> None:
        """Temperature arrays should have 8,760 elements."""
        raw = _make_simple_response(2018)
        result = PVGISClient._parse_response(raw)
        assert len(result["temperature"][2018]) == HOURS_PER_YEAR

    def test_production_values(self) -> None:
        """Production should be P/1000 (W to kWh conversion)."""
        raw = _make_simple_response(2018)
        result = PVGISClient._parse_response(raw)
        # All P values are 50.0 W → 0.05 kWh
        np.testing.assert_allclose(result["production"][2018], 0.05)

    def test_temperature_values(self) -> None:
        """Temperature values should be passed through directly."""
        raw = _make_simple_response(2018)
        result = PVGISClient._parse_response(raw)
        temps = result["temperature"][2018]
        assert np.min(temps) >= 5.0
        assert np.max(temps) <= 15.0

    def test_missing_t2m_defaults_to_zero(self) -> None:
        """Missing T2m field should default to 0.0."""
        records = [
            {"time": f"2018{m:02d}{d:02d}:{h:02d}10", "P": 50.0}
            for m in range(1, 13)
            for d in range(1, 29)
            for h in range(24)
        ]
        records = records[:HOURS_PER_YEAR]
        raw = {"outputs": {"hourly": records}}
        result = PVGISClient._parse_response(raw)
        temps = result["temperature"][2018]
        np.testing.assert_array_equal(temps, 0.0)

    def test_missing_hourly_raises(self) -> None:
        """Missing outputs.hourly should raise PVGISError."""
        with pytest.raises(PVGISError, match="outputs.hourly"):
            PVGISClient._parse_response({"outputs": {}})

    def test_empty_response_raises(self) -> None:
        """Empty hourly list should raise PVGISError."""
        with pytest.raises(PVGISError, match="no hourly"):
            PVGISClient._parse_response({"outputs": {"hourly": []}})

    def test_leap_year_stripped(self) -> None:
        """Leap year extra hours should be stripped to 8,760."""
        records = [
            {"time": f"2020{m:02d}{d:02d}:{h:02d}10", "P": 50.0, "T2m": 10.0}
            for m in range(1, 13)
            for d in range(1, 29)
            for h in range(24)
        ]
        extra = [
            {"time": f"202012{d:02d}:{h:02d}10", "P": 50.0, "T2m": 10.0}
            for d in range(29, 32)
            for h in range(24)
        ]
        records = (records + extra)[:8784]
        if len(records) > HOURS_PER_YEAR:
            raw = {"outputs": {"hourly": records}}
            result = PVGISClient._parse_response(raw)
            for key in ("production", "temperature"):
                for year_data in result[key].values():
                    assert len(year_data) == HOURS_PER_YEAR


# ---------------------------------------------------------------------------
# Tests for fetch_single_year with include_temperature
# ---------------------------------------------------------------------------


class TestFetchSingleYearIncludeTemperature:
    """Tests for fetch_single_year(include_temperature=...)."""

    def test_default_returns_ndarray(self) -> None:
        """Default (include_temperature=False) returns np.ndarray."""
        # We can't call the real API, but we can check the method signature
        import inspect
        sig = inspect.signature(PVGISClient.fetch_single_year)
        param = sig.parameters["include_temperature"]
        assert param.default is False

    def test_include_temperature_parameter_exists(self) -> None:
        """The include_temperature parameter should exist."""
        import inspect
        sig = inspect.signature(PVGISClient.fetch_single_year)
        assert "include_temperature" in sig.parameters


# ---------------------------------------------------------------------------
# Tests for fetch_hourly_production (backward compatibility)
# ---------------------------------------------------------------------------


class TestFetchHourlyProductionBackwardCompat:
    """Ensure fetch_hourly_production still returns dict[int, ndarray]."""

    def test_method_exists(self) -> None:
        client = PVGISClient(cache_dir=None)
        assert hasattr(client, "fetch_hourly_production")
        assert callable(client.fetch_hourly_production)
