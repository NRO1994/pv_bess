"""PVGIS API client – fetch hourly historical PV production data.

Uses the EU PVGIS REST API v5.3, ``seriescalc`` endpoint, to download
year-by-year hourly production timeseries for a given PV system configuration.

Key behaviour
-------------
- Fetches **all available historical years** in a single API call.
- Caches the raw JSON response on disk (``~/.pv_bess_cache/``) keyed by a
  SHA-256 hash of the query parameters, so repeated runs with the same inputs
  never hit the network.
- Retries up to :data:`~pv_bess_model.config.defaults.PVGIS_RETRY_MAX` times
  with exponential backoff on HTTP 429 (rate limit) or 5xx errors.

Typical usage::

    from pv_bess_model.pv.pvgis_client import PVGISClient
    client = PVGISClient()
    yearly = client.fetch_hourly_production(
        latitude=53.55, longitude=9.99,
        peak_power_kwp=5000, system_loss_pct=14.0,
        mounting_type="free", azimuth_deg=0, tilt_deg=30,
        pvgis_database="PVGIS-SARAH2",
    )
    # yearly: {2005: np.ndarray(8760,), 2006: np.ndarray(8760,), ...}
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

from pv_bess_model.config.defaults import (
    HOURS_PER_YEAR,
    PVGIS_API_BASE_URL,
    PVGIS_CACHE_DIR,
    PVGIS_OUTPUT_FORMAT,
    PVGIS_REQUEST_TIMEOUT_S,
    PVGIS_RETRY_BACKOFF_FACTOR,
    PVGIS_RETRY_MAX,
    PVGIS_SERIESCALC_ENDPOINT,
)

logger = logging.getLogger(__name__)

_MOUNTING_MAP: dict[str, str] = {
    "free": "free",
    "building": "building",
}

# HTTP status codes that warrant a retry (rate-limit and server errors)
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


class PVGISError(RuntimeError):
    """Raised when the PVGIS API returns an error that cannot be retried."""


class PVGISClient:
    """Thin client for the PVGIS ``seriescalc`` endpoint.

    Parameters
    ----------
    cache_dir:
        Directory for persistent JSON response cache.  Defaults to
        ``~/.pv_bess_cache``.  Pass ``None`` to disable caching entirely
        (useful in tests).
    base_url:
        PVGIS API base URL.  Override for testing or alternative deployments.
    timeout:
        HTTP request timeout in seconds.
    max_retries:
        Maximum number of retry attempts on transient errors.
    backoff_factor:
        Initial wait time (seconds) for exponential backoff.
        Actual wait on attempt *k* = ``backoff_factor × 2^(k-1)``.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = PVGIS_CACHE_DIR,
        base_url: str = PVGIS_API_BASE_URL,
        timeout: int = PVGIS_REQUEST_TIMEOUT_S,
        max_retries: int = PVGIS_RETRY_MAX,
        backoff_factor: float = PVGIS_RETRY_BACKOFF_FACTOR,
    ) -> None:
        self._base_url = base_url.rstrip("/") + "/"
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor

        if cache_dir is None:
            self._cache_dir: Path | None = None
        else:
            self._cache_dir = Path(cache_dir).expanduser()
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_hourly_production(
        self,
        latitude: float,
        longitude: float,
        peak_power_kwp: float,
        system_loss_pct: float,
        mounting_type: str,
        azimuth_deg: float,
        tilt_deg: float,
        pvgis_database: str = "PVGIS-SARAH2",
    ) -> dict[int, np.ndarray]:
        """Fetch hourly PV production for all available historical years.

        Parameters
        ----------
        latitude:
            Site latitude in decimal degrees (−90 … +90).
        longitude:
            Site longitude in decimal degrees (−180 … +180).
        peak_power_kwp:
            PV system peak power in kWp.
        system_loss_pct:
            Total system loss in percent (e.g. ``14.0``).
        mounting_type:
            ``"free"`` (free-standing) or ``"building"`` (BIPV).
        azimuth_deg:
            Panel azimuth in degrees (0 = south, −90 = east, +90 = west).
        tilt_deg:
            Panel tilt angle from horizontal in degrees.
        pvgis_database:
            PVGIS radiation database (e.g. ``"PVGIS-SARAH2"``).

        Returns
        -------
        dict[int, numpy.ndarray]
            ``{calendar_year: hourly_production_array}``
            Each array has exactly 8 760 elements (leap-day hours stripped).
            Values are in **kWh** per hour.

        Raises
        ------
        PVGISError
            When the API returns a non-retryable error or all retries are
            exhausted.
        ValueError
            When *mounting_type* is not recognised.
        """
        params = self._build_params(
            latitude=latitude,
            longitude=longitude,
            peak_power_kwp=peak_power_kwp,
            system_loss_pct=system_loss_pct,
            mounting_type=mounting_type,
            azimuth_deg=azimuth_deg,
            tilt_deg=tilt_deg,
            pvgis_database=pvgis_database,
        )
        raw = self._get_with_cache(params)
        return self._parse_response(raw)

    # ------------------------------------------------------------------
    # Parameter construction
    # ------------------------------------------------------------------

    def _build_params(
        self,
        latitude: float,
        longitude: float,
        peak_power_kwp: float,
        system_loss_pct: float,
        mounting_type: str,
        azimuth_deg: float,
        tilt_deg: float,
        pvgis_database: str,
    ) -> dict[str, Any]:
        """Return the query-parameter dict for the seriescalc endpoint."""
        if mounting_type not in _MOUNTING_MAP:
            raise ValueError(
                f"Unknown mounting_type '{mounting_type}'. "
                f"Must be one of {list(_MOUNTING_MAP)}."
            )
        return {
            "lat": latitude,
            "lon": longitude,
            "peakpower": peak_power_kwp,
            "loss": system_loss_pct,
            "mountingplace": _MOUNTING_MAP[mounting_type],
            "aspect": azimuth_deg,
            "angle": tilt_deg,
            "raddatabase": pvgis_database,
            "outputformat": PVGIS_OUTPUT_FORMAT,
            "pvcalculation": 1,
            "components": 1,
        }

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(params: dict[str, Any]) -> str:
        """Return a 32-char SHA-256 hex digest for *params*."""
        canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:32]

    def _cache_path(self, params: dict[str, Any]) -> Path | None:
        """Return the cache file ``Path``, or ``None`` if caching is disabled."""
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"pvgis_{self._cache_key(params)}.json"

    def _get_with_cache(self, params: dict[str, Any]) -> dict:
        """Return parsed JSON, served from disk cache when available."""
        cache_file = self._cache_path(params)

        if cache_file is not None and cache_file.exists():
            logger.info("PVGIS cache hit: %s", cache_file)
            with cache_file.open("r", encoding="utf-8") as fh:
                return json.load(fh)

        logger.info("PVGIS cache miss – fetching from API")
        raw = self._fetch(params)

        if cache_file is not None:
            logger.debug("Writing PVGIS cache: %s", cache_file)
            with cache_file.open("w", encoding="utf-8") as fh:
                json.dump(raw, fh)

        return raw

    # ------------------------------------------------------------------
    # HTTP with retry/backoff
    # ------------------------------------------------------------------

    def _fetch(self, params: dict[str, Any]) -> dict:
        """Execute the HTTP GET with exponential backoff retry."""
        url = self._base_url + PVGIS_SERIESCALC_ENDPOINT
        last_exc: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                logger.debug(
                    "PVGIS request attempt %d/%d: %s",
                    attempt,
                    self._max_retries,
                    url,
                )
                resp = requests.get(url, params=params, timeout=self._timeout)

                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code in _RETRYABLE_STATUS_CODES:
                    wait = self._backoff_factor * (2 ** (attempt - 1))
                    logger.warning(
                        "PVGIS HTTP %d on attempt %d/%d – retrying in %.1fs",
                        resp.status_code,
                        attempt,
                        self._max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    last_exc = PVGISError(
                        f"HTTP {resp.status_code} from PVGIS "
                        f"after {attempt} attempt(s)"
                    )
                    continue

                # Non-retryable client error (4xx except 429)
                try:
                    err_body = resp.json()
                    detail = (
                        err_body.get("message")
                        or err_body.get("description")
                        or str(err_body)
                    )
                except Exception:
                    detail = resp.text[:300]
                raise PVGISError(f"PVGIS API error (HTTP {resp.status_code}): {detail}")

            except requests.Timeout as exc:
                wait = self._backoff_factor * (2 ** (attempt - 1))
                logger.warning(
                    "PVGIS timeout on attempt %d/%d – retrying in %.1fs",
                    attempt,
                    self._max_retries,
                    wait,
                )
                time.sleep(wait)
                last_exc = exc

            except requests.ConnectionError as exc:
                wait = self._backoff_factor * (2 ** (attempt - 1))
                logger.warning(
                    "PVGIS connection error on attempt %d/%d – retrying in %.1fs: %s",
                    attempt,
                    self._max_retries,
                    wait,
                    exc,
                )
                time.sleep(wait)
                last_exc = exc

        raise PVGISError(
            f"PVGIS request failed after {self._max_retries} attempt(s)."
        ) from last_exc

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw: dict) -> dict[int, np.ndarray]:
        """Parse the PVGIS JSON response into per-year hourly arrays.

        The ``seriescalc`` response contains ``outputs.hourly``, a list of
        records of the form::

            {"time": "20050101:0010", "P": 0.0, "Gb(i)": ..., ...}

        ``"P"`` is AC power in **W**; dividing by 1 000 gives **kWh** for a
        1-hour timestep.

        Parameters
        ----------
        raw:
            Parsed JSON dict from the PVGIS API.

        Returns
        -------
        dict[int, numpy.ndarray]
            ``{year: array_of_8760_kwh}``

        Raises
        ------
        PVGISError
            When the response is missing expected keys.
        """
        try:
            hourly_records: list[dict] = raw["outputs"]["hourly"]
        except (KeyError, TypeError) as exc:
            raise PVGISError(
                "Unexpected PVGIS response structure: " "missing 'outputs.hourly' key."
            ) from exc

        # Group records by year
        yearly_lists: dict[int, list[float]] = {}
        for record in hourly_records:
            time_str: str = record["time"]  # e.g. "20050101:0010"
            year = int(time_str[:4])
            power_w: float = float(record.get("P", 0.0))
            yearly_lists.setdefault(year, []).append(power_w / 1000.0)

        if not yearly_lists:
            raise PVGISError("PVGIS response contained no hourly records.")

        result: dict[int, np.ndarray] = {}
        for year, values in sorted(yearly_lists.items()):
            arr = _strip_leap_day(np.array(values, dtype=float), year)
            result[year] = arr
            logger.debug("Parsed PVGIS year %d → %d hourly values", year, len(arr))

        return result


# ---------------------------------------------------------------------------
# Module-level utilities (also used by timeseries.py)
# ---------------------------------------------------------------------------


def _strip_leap_day(arr: np.ndarray, year: int) -> np.ndarray:
    """Truncate *arr* to exactly :data:`HOURS_PER_YEAR` (8 760) elements.

    For leap years PVGIS appends 24 extra hours for December 31st at positions
    8 760 – 8 783.  We discard them so every year has the same length.

    If *arr* is shorter than 8 760 elements (should not happen with a healthy
    PVGIS response) the array is zero-padded and a warning is logged.

    Parameters
    ----------
    arr:
        Raw hourly production array (length 8 760 or 8 784).
    year:
        Calendar year (used for log messages only).

    Returns
    -------
    numpy.ndarray
        Array of exactly 8 760 ``float64`` values.
    """
    if len(arr) > HOURS_PER_YEAR:
        logger.debug(
            "Year %d: stripping %d extra hours (leap year)",
            year,
            len(arr) - HOURS_PER_YEAR,
        )
        return arr[:HOURS_PER_YEAR]
    if len(arr) < HOURS_PER_YEAR:
        logger.warning(
            "Year %d has only %d values (expected %d) – zero-padding",
            year,
            len(arr),
            HOURS_PER_YEAR,
        )
        padded = np.zeros(HOURS_PER_YEAR, dtype=float)
        padded[: len(arr)] = arr
        return padded
    return arr
