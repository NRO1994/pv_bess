"""Unit tests for pv_bess_model.pv.pvgis_client.

All tests are fully offline – no real HTTP calls are made.
The PVGIS API is mocked with ``unittest.mock`` / ``pytest-monkeypatch``.

Covers:
- PVGISClient._build_params: correct mapping of scenario fields
- PVGISClient._cache_key: deterministic, content-addressed
- PVGISClient._cache_path: None when cache disabled, Path when enabled
- Cache round-trip: miss → fetch → write; hit → read (no fetch)
- _strip_leap_day: non-leap, leap, short (zero-pad), exact
- _parse_response: correct year grouping, W→kWh conversion, leap stripping
- _parse_response: missing 'outputs' key raises PVGISError
- _parse_response: empty hourly list raises PVGISError
- Retry logic: retries on 429/500/503, raises after max retries
- Retry logic: immediate raise on non-retryable 4xx (e.g. 400, 404)
- fetch_hourly_production: unknown mounting_type raises ValueError
- fetch_hourly_production: end-to-end with mocked requests.get
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pv_bess_model.config.defaults import HOURS_PER_YEAR
from pv_bess_model.pv.pvgis_client import PVGISClient, PVGISError, _strip_leap_day

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hourly_records(
    year: int, n_hours: int, power_w: float = 1000.0
) -> list[dict]:
    """Synthetic PVGIS hourly records for *year* with constant power."""
    records = []
    day, month, h = 1, 1, 0
    for _ in range(n_hours):
        hh = h % 24
        records.append({"time": f"{year}{month:02d}{day:02d}:{hh:02d}10", "P": power_w})
        h += 1
        if h % 24 == 0:
            day += 1
            if day > 28:
                day = 1
                if month < 12:
                    month += 1
    return records


def _pvgis_response(years_hours: dict[int, int], power_w: float = 1000.0) -> dict:
    """Build a minimal PVGIS-like JSON response."""
    all_records: list[dict] = []
    for year, n_hours in sorted(years_hours.items()):
        all_records.extend(_make_hourly_records(year, n_hours, power_w))
    return {"outputs": {"hourly": all_records}}


@pytest.fixture
def client_no_cache() -> PVGISClient:
    return PVGISClient(cache_dir=None, max_retries=1, backoff_factor=0.0)


@pytest.fixture
def client_with_cache(tmp_path) -> PVGISClient:
    return PVGISClient(cache_dir=tmp_path / "cache", max_retries=1, backoff_factor=0.0)


# ---------------------------------------------------------------------------
# _strip_leap_day
# ---------------------------------------------------------------------------


class TestStripLeapDay:
    def test_non_leap_year_unchanged(self):
        arr = np.ones(HOURS_PER_YEAR)
        out = _strip_leap_day(arr, 2019)
        assert len(out) == HOURS_PER_YEAR
        np.testing.assert_array_equal(out, arr)

    def test_leap_year_8784_stripped_to_8760(self):
        arr = np.arange(8784, dtype=float)
        out = _strip_leap_day(arr, 2020)
        assert len(out) == HOURS_PER_YEAR
        np.testing.assert_array_equal(out, arr[:HOURS_PER_YEAR])

    def test_strip_removes_tail_not_head(self):
        arr = np.zeros(8784)
        arr[:HOURS_PER_YEAR] = 1.0
        arr[HOURS_PER_YEAR:] = 99.0
        out = _strip_leap_day(arr, 2016)
        assert np.all(out == 1.0), "Leap-day hours must be discarded from the tail"

    def test_short_array_zero_padded(self):
        arr = np.ones(100)
        out = _strip_leap_day(arr, 2019)
        assert len(out) == HOURS_PER_YEAR
        np.testing.assert_array_equal(out[:100], 1.0)
        np.testing.assert_array_equal(out[100:], 0.0)

    def test_exact_8760_unchanged(self):
        arr = np.arange(HOURS_PER_YEAR, dtype=float)
        out = _strip_leap_day(arr, 2021)
        np.testing.assert_array_equal(out, arr)


# ---------------------------------------------------------------------------
# _build_params
# ---------------------------------------------------------------------------


class TestBuildParams:
    def test_all_keys_present(self, client_no_cache):
        p = client_no_cache._build_params(
            latitude=53.55,
            longitude=9.99,
            peak_power_kwp=5000,
            system_loss_pct=14.0,
            mounting_type="free",
            azimuth_deg=0,
            tilt_deg=30,
            pvgis_database="PVGIS-SARAH2",
        )
        assert p["lat"] == 53.55
        assert p["lon"] == 9.99
        assert p["peakpower"] == 5000
        assert p["loss"] == 14.0
        assert p["mountingplace"] == "free"
        assert p["aspect"] == 0
        assert p["angle"] == 30
        assert p["raddatabase"] == "PVGIS-SARAH2"
        assert p["outputformat"] == "json"
        assert p["pvcalculation"] == 1

    def test_building_mounting_mapped(self, client_no_cache):
        p = client_no_cache._build_params(
            latitude=0,
            longitude=0,
            peak_power_kwp=100,
            system_loss_pct=10,
            mounting_type="building",
            azimuth_deg=0,
            tilt_deg=15,
            pvgis_database="PVGIS-ERA5",
        )
        assert p["mountingplace"] == "building"

    def test_invalid_mounting_raises(self, client_no_cache):
        with pytest.raises(ValueError, match="rooftop"):
            client_no_cache._build_params(
                latitude=0,
                longitude=0,
                peak_power_kwp=100,
                system_loss_pct=10,
                mounting_type="rooftop",
                azimuth_deg=0,
                tilt_deg=15,
                pvgis_database="PVGIS-SARAH2",
            )


# ---------------------------------------------------------------------------
# Cache key / path
# ---------------------------------------------------------------------------


class TestCacheKey:
    def test_deterministic(self, client_no_cache):
        params = {"lat": 53.55, "lon": 9.99}
        assert client_no_cache._cache_key(params) == client_no_cache._cache_key(params)

    def test_different_params_different_key(self, client_no_cache):
        k1 = client_no_cache._cache_key({"lat": 53.55})
        k2 = client_no_cache._cache_key({"lat": 53.56})
        assert k1 != k2

    def test_key_is_32_chars(self, client_no_cache):
        assert len(client_no_cache._cache_key({"a": 1})) == 32

    def test_key_order_independent(self, client_no_cache):
        k1 = client_no_cache._cache_key({"a": 1, "b": 2})
        k2 = client_no_cache._cache_key({"b": 2, "a": 1})
        assert k1 == k2

    def test_cache_path_none_when_disabled(self, client_no_cache):
        assert client_no_cache._cache_path({"lat": 0}) is None

    def test_cache_path_is_json_file(self, client_with_cache):
        p = client_with_cache._cache_path({"lat": 0})
        assert isinstance(p, Path)
        assert p.suffix == ".json"
        assert "pvgis_" in p.name


# ---------------------------------------------------------------------------
# Cache round-trip
# ---------------------------------------------------------------------------


class TestCacheRoundTrip:
    def _ok_response(self, data: dict) -> MagicMock:
        r = MagicMock()
        r.status_code = 200
        r.json.return_value = data
        return r

    def test_miss_writes_cache(self, client_with_cache):
        data = _pvgis_response({2005: HOURS_PER_YEAR})
        params = {"lat": 1.0, "lon": 2.0}
        cache_file = client_with_cache._cache_path(params)

        with patch("requests.get", return_value=self._ok_response(data)):
            client_with_cache._get_with_cache(params)

        assert cache_file.exists()
        assert "outputs" in json.loads(cache_file.read_text())

    def test_hit_skips_network(self, client_with_cache):
        data = _pvgis_response({2005: HOURS_PER_YEAR})
        params = {"lat": 10.0, "lon": 20.0}
        cache_file = client_with_cache._cache_path(params)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data))

        with patch("requests.get") as mock_get:
            client_with_cache._get_with_cache(params)
            mock_get.assert_not_called()

    def test_no_cache_never_writes_files(self, client_no_cache, tmp_path):
        data = _pvgis_response({2005: HOURS_PER_YEAR})
        with patch("requests.get", return_value=self._ok_response(data)):
            client_no_cache._get_with_cache({"lat": 5.0})
        assert list(tmp_path.rglob("*.json")) == []


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_single_non_leap_year(self, client_no_cache):
        data = _pvgis_response({2019: HOURS_PER_YEAR}, power_w=2000.0)
        result = client_no_cache._parse_response(data)
        assert 2019 in result
        assert len(result[2019]) == HOURS_PER_YEAR
        assert np.all(result[2019] == pytest.approx(2.0))

    def test_multiple_years(self, client_no_cache):
        data = _pvgis_response({2005: HOURS_PER_YEAR, 2006: HOURS_PER_YEAR})
        result = client_no_cache._parse_response(data)
        assert set(result.keys()) == {2005, 2006}
        for arr in result.values():
            assert len(arr) == HOURS_PER_YEAR

    def test_leap_year_stripped(self, client_no_cache):
        data = _pvgis_response({2020: 8784})
        result = client_no_cache._parse_response(data)
        assert len(result[2020]) == HOURS_PER_YEAR

    def test_w_to_kwh_conversion(self, client_no_cache):
        data = _pvgis_response({2010: HOURS_PER_YEAR}, power_w=1000.0)
        result = client_no_cache._parse_response(data)
        assert result[2010][0] == pytest.approx(1.0)

    def test_zero_power(self, client_no_cache):
        data = _pvgis_response({2010: HOURS_PER_YEAR}, power_w=0.0)
        result = client_no_cache._parse_response(data)
        assert np.all(result[2010] == 0.0)

    def test_missing_outputs_raises(self, client_no_cache):
        with pytest.raises(PVGISError, match="outputs"):
            client_no_cache._parse_response({"something_else": {}})

    def test_empty_hourly_raises(self, client_no_cache):
        with pytest.raises(PVGISError, match="no hourly records"):
            client_no_cache._parse_response({"outputs": {"hourly": []}})

    def test_missing_p_key_defaults_zero(self, client_no_cache):
        records = [{"time": f"20100101:{i:02d}10"} for i in range(HOURS_PER_YEAR)]
        result = client_no_cache._parse_response({"outputs": {"hourly": records}})
        assert np.all(result[2010] == 0.0)

    def test_years_sorted(self, client_no_cache):
        data = _pvgis_response(
            {2010: HOURS_PER_YEAR, 2005: HOURS_PER_YEAR, 2008: HOURS_PER_YEAR}
        )
        result = client_no_cache._parse_response(data)
        assert list(result.keys()) == [2005, 2008, 2010]


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    def _bad_resp(self, status: int) -> MagicMock:
        r = MagicMock()
        r.status_code = status
        r.json.return_value = {}
        r.text = f"error {status}"
        return r

    def _ok_resp(self) -> MagicMock:
        r = MagicMock()
        r.status_code = 200
        r.json.return_value = _pvgis_response({2005: HOURS_PER_YEAR})
        return r

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_retryable_status_exhausts_retries(self, status):
        client = PVGISClient(cache_dir=None, max_retries=3, backoff_factor=0.0)
        with patch("requests.get", side_effect=[self._bad_resp(status)] * 3):
            with patch("time.sleep"):
                with pytest.raises(PVGISError):
                    client._fetch({"lat": 0})

    def test_success_on_second_attempt(self):
        client = PVGISClient(cache_dir=None, max_retries=3, backoff_factor=0.0)
        with patch("requests.get", side_effect=[self._bad_resp(429), self._ok_resp()]):
            with patch("time.sleep"):
                result = client._fetch({"lat": 0})
        assert "outputs" in result

    @pytest.mark.parametrize("status", [400, 404, 422])
    def test_non_retryable_4xx_raises_immediately(self, status):
        client = PVGISClient(cache_dir=None, max_retries=5, backoff_factor=0.0)
        with patch("requests.get", return_value=self._bad_resp(status)) as mock_get:
            with pytest.raises(PVGISError, match=str(status)):
                client._fetch({"lat": 0})
        assert mock_get.call_count == 1

    def test_timeout_triggers_retry_then_success(self):
        import requests as req_lib

        client = PVGISClient(cache_dir=None, max_retries=2, backoff_factor=0.0)
        with patch("requests.get", side_effect=[req_lib.Timeout(), self._ok_resp()]):
            with patch("time.sleep"):
                result = client._fetch({"lat": 0})
        assert "outputs" in result

    def test_all_timeouts_raises(self):
        import requests as req_lib

        client = PVGISClient(cache_dir=None, max_retries=2, backoff_factor=0.0)
        with patch("requests.get", side_effect=req_lib.Timeout()):
            with patch("time.sleep"):
                with pytest.raises(PVGISError):
                    client._fetch({"lat": 0})

    def test_exponential_backoff_wait_times(self):
        client = PVGISClient(cache_dir=None, max_retries=3, backoff_factor=2.0)
        with patch("requests.get", side_effect=[self._bad_resp(429)] * 3):
            with patch("time.sleep") as mock_sleep:
                with pytest.raises(PVGISError):
                    client._fetch({"lat": 0})
        waits = [c[0][0] for c in mock_sleep.call_args_list]
        # backoff_factor × 2^(attempt-1): 2.0, 4.0, 8.0
        assert waits == pytest.approx([2.0, 4.0, 8.0])


# ---------------------------------------------------------------------------
# fetch_hourly_production end-to-end
# ---------------------------------------------------------------------------


class TestFetchHourlyProduction:
    def test_end_to_end_mocked(self, client_no_cache):
        data = _pvgis_response(
            {2005: HOURS_PER_YEAR, 2006: HOURS_PER_YEAR, 2007: HOURS_PER_YEAR},
            power_w=500.0,
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = data

        with patch("requests.get", return_value=mock_resp):
            result = client_no_cache.fetch_hourly_production(
                latitude=53.55,
                longitude=9.99,
                peak_power_kwp=5000,
                system_loss_pct=14.0,
                mounting_type="free",
                azimuth_deg=0,
                tilt_deg=30,
                pvgis_database="PVGIS-SARAH2",
            )

        assert set(result.keys()) == {2005, 2006, 2007}
        for arr in result.values():
            assert len(arr) == HOURS_PER_YEAR
            assert np.all(arr == pytest.approx(0.5))

    def test_invalid_mounting_no_network_call(self, client_no_cache):
        with patch("requests.get") as mock_get:
            with pytest.raises(ValueError, match="rooftop"):
                client_no_cache.fetch_hourly_production(
                    latitude=0,
                    longitude=0,
                    peak_power_kwp=100,
                    system_loss_pct=10,
                    mounting_type="rooftop",
                    azimuth_deg=0,
                    tilt_deg=20,
                )
        mock_get.assert_not_called()

    def test_cache_used_on_second_call(self, tmp_path):
        client = PVGISClient(
            cache_dir=tmp_path / "cache", max_retries=1, backoff_factor=0.0
        )
        data = _pvgis_response({2010: HOURS_PER_YEAR})
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = data

        kwargs = dict(
            latitude=10.0,
            longitude=20.0,
            peak_power_kwp=1000,
            system_loss_pct=12,
            mounting_type="free",
            azimuth_deg=5,
            tilt_deg=25,
        )
        with patch("requests.get", return_value=mock_resp) as mock_get:
            client.fetch_hourly_production(**kwargs)
            client.fetch_hourly_production(**kwargs)
        assert mock_get.call_count == 1, "Second call should be served from cache"
