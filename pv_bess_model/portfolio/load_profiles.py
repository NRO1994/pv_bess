"""BDEW standard load profile (SLP) generation and caching.

Generates 35,040 quarter-hourly load profile values from BDEW coefficients
stored in ``.data/bdew_profile_2025.json``. Each profile is assembled by:

1. Reading BDEW coefficients per profile type, month, and day type (96 values).
2. Generating a calendar for the target year (Werktag/Samstag/Sonn- u. Feiertag).
3. For H25, P25, S25: applying the BDEW dynamization polynomial.
4. Normalizing so the annual sum equals :data:`SLP_NORMIERUNG_KWH` (1,000,000 kWh).
5. Caching the result as a ``.npy`` file for fast reload.

The generated profile is then scaled by customer count and annual consumption
per customer to produce the actual load timeseries.

Typical usage::

    from pv_bess_model.portfolio.load_profiles import generate_slp, scale_slp

    slp = generate_slp("H25", year=2027, bundesland="SH")
    load = scale_slp(slp, annual_consumption_kwh=3200.0, customer_count=8500)
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

import holidays
import numpy as np

from pv_bess_model.config.defaults import (
    DEFAULT_BUNDESLAND,
    DEFAULT_SLP_CACHE_DIR,
    INTERVALS_PER_DAY,
    SLP_NORMIERUNG_KWH,
)

logger = logging.getLogger(__name__)

# Path to the BDEW profile JSON (relative to project root)
_BDEW_JSON_PATH = Path(".data/bdew_profile_2025.json")

# Mapping from calendar month (1-indexed) to BDEW JSON month keys
_MONTH_KEYS: list[str] = [
    "Jan", "Feb", "Mrz", "Apr", "Mai", "Jun",
    "Jul", "Aug", "Sep", "Okt", "Nov", "Dez",
]

# Profile types that receive the BDEW dynamization function
_DYNAMIZED_PROFILES: frozenset[str] = frozenset({"H25", "P25", "S25"})

# BDEW dynamization polynomial coefficients for H25/P25/S25
# x = x₀ × (-3.92E-10 × t⁴ + 3.20E-7 × t³ - 7.02E-5 × t² + 2.10E-3 × t + 1.24)
_DYN_COEFF_A4: float = -3.92e-10
_DYN_COEFF_A3: float = 3.20e-7
_DYN_COEFF_A2: float = -7.02e-5
_DYN_COEFF_A1: float = 2.10e-3
_DYN_COEFF_A0: float = 1.24


def generate_slp(
    slp_type: str,
    year: int,
    bundesland: str = DEFAULT_BUNDESLAND,
    bdew_json_path: str | Path | None = None,
    cache_dir: str | Path | None = DEFAULT_SLP_CACHE_DIR,
) -> np.ndarray:
    """Generate a normalized BDEW standard load profile for a given year.

    The profile is built from the BDEW coefficient tables by assigning each
    day of the target year its correct day type (WT = weekday, SA = Saturday,
    FT = Sunday/holiday) and month, then reading the corresponding 96
    quarter-hourly values.

    For H25, P25, and S25 profiles the BDEW dynamization polynomial is
    applied (day-of-year based).

    The result is normalized so that the annual sum equals
    :data:`SLP_NORMIERUNG_KWH` (1,000,000 kWh).

    Parameters
    ----------
    slp_type:
        BDEW profile type (``"H25"``, ``"G25"``, ``"L25"``, ``"P25"``,
        ``"S25"``).
    year:
        Calendar year for day-type assignment (e.g. 2027).
    bundesland:
        German federal state code for holiday calendar (e.g. ``"SH"``).
    bdew_json_path:
        Path to the BDEW coefficient JSON file.  Defaults to
        ``.data/bdew_profile_2025.json``.
    cache_dir:
        Directory for caching generated profiles.  ``None`` disables caching.

    Returns
    -------
    numpy.ndarray
        Array of 35,040 quarter-hourly values normalized to
        :data:`SLP_NORMIERUNG_KWH` annual sum.

    Raises
    ------
    FileNotFoundError
        When the BDEW JSON file does not exist.
    KeyError
        When the requested *slp_type* is not found in the JSON data.
    """
    # Check cache first
    cached = _load_from_cache(slp_type, year, cache_dir, bdew_json_path)
    if cached is not None:
        return cached

    bdew_path = Path(bdew_json_path) if bdew_json_path else _BDEW_JSON_PATH
    bdew_data = _load_bdew_json(bdew_path)

    if slp_type not in bdew_data:
        raise KeyError(
            f"SLP type '{slp_type}' not found in BDEW data. "
            f"Available types: {list(bdew_data.keys())}"
        )

    profile_data = bdew_data[slp_type]
    calendar = generate_calendar(year, bundesland)
    apply_dyn = slp_type in _DYNAMIZED_PROFILES

    # Build 35,040 values
    values: list[float] = []
    for day_index, (month_idx, day_type) in enumerate(calendar):
        month_key = _MONTH_KEYS[month_idx - 1]
        day_values_96 = profile_data[month_key][day_type]

        if apply_dyn:
            day_of_year = day_index + 1  # 1-indexed
            dyn_factor = _dynamization_factor(day_of_year)
            day_values_96 = [round(v * dyn_factor, 3) for v in day_values_96]

        values.extend(day_values_96)

    raw_array = np.array(values, dtype=float)

    # Normalize to SLP_NORMIERUNG_KWH
    raw_sum = np.sum(raw_array)
    if raw_sum > 0:
        normalized = raw_array * (SLP_NORMIERUNG_KWH / raw_sum)
    else:
        logger.warning(
            "SLP %s for year %d has zero sum – returning zeros.", slp_type, year
        )
        normalized = raw_array

    logger.info(
        "Generated SLP %s for year %d: %d intervals, "
        "raw sum=%.1f, normalized sum=%.1f kWh.",
        slp_type,
        year,
        len(normalized),
        raw_sum,
        np.sum(normalized),
    )

    _save_to_cache(normalized, slp_type, year, cache_dir)
    return normalized


def scale_slp(
    slp_normalized: np.ndarray,
    annual_consumption_kwh: float,
    customer_count: int,
) -> np.ndarray:
    """Scale a normalized SLP to actual consumption.

    Parameters
    ----------
    slp_normalized:
        Normalized SLP array (35,040 values, summing to
        :data:`SLP_NORMIERUNG_KWH`).
    annual_consumption_kwh:
        Annual consumption per customer in kWh.
    customer_count:
        Number of customers.

    Returns
    -------
    numpy.ndarray
        Scaled load profile in kWh per quarter-hour interval.
    """
    total_annual_kwh = annual_consumption_kwh * customer_count
    return slp_normalized * (total_annual_kwh / SLP_NORMIERUNG_KWH)


def generate_calendar(
    year: int,
    bundesland: str = DEFAULT_BUNDESLAND,
) -> list[tuple[int, str]]:
    """Generate a day-type calendar for 365 days (no leap day).

    Each entry is ``(month_number_1indexed, day_type)`` where day_type is
    one of ``"WT"`` (weekday), ``"SA"`` (Saturday), ``"FT"`` (Sunday or
    public holiday).

    Leap years: December 31 is excluded (consistent with PVGIS handling),
    giving exactly 365 entries.

    Parameters
    ----------
    year:
        Calendar year.
    bundesland:
        German federal state code for the ``holidays`` library.

    Returns
    -------
    list[tuple[int, str]]
        365 entries of ``(month, day_type)``.
    """
    de_holidays = holidays.Germany(years=year, subdiv=bundesland)

    result: list[tuple[int, str]] = []
    date = datetime.date(year, 1, 1)
    end_date = datetime.date(year, 12, 31)

    while date <= end_date:
        # Skip Dec 31 for leap years to keep 365 days
        if _is_leap_year(year) and date.month == 12 and date.day == 31:
            date += datetime.timedelta(days=1)
            continue

        month = date.month
        weekday = date.weekday()  # 0=Monday, 6=Sunday

        if date in de_holidays or weekday == 6:  # Sunday or holiday
            day_type = "FT"
        elif weekday == 5:  # Saturday
            day_type = "SA"
        else:
            day_type = "WT"

        result.append((month, day_type))
        date += datetime.timedelta(days=1)

    if len(result) != 365:
        logger.warning(
            "Calendar for year %d has %d days (expected 365).",
            year,
            len(result),
        )

    return result


# ---------------------------------------------------------------------------
# Dynamization
# ---------------------------------------------------------------------------


def _dynamization_factor(day_of_year: int) -> float:
    """Compute the BDEW dynamization factor for a given day of the year.

    The polynomial is::

        f(t) = -3.92E-10 × t⁴ + 3.20E-7 × t³ - 7.02E-5 × t² + 2.10E-3 × t + 1.24

    where ``t`` is the day of the year (1 = January 1st).

    The result is rounded to four decimal places per BDEW specification.

    Parameters
    ----------
    day_of_year:
        1-indexed day of year (1 = Jan 1, 365 = Dec 31).

    Returns
    -------
    float
        Dynamization factor (dimensionless multiplier).
    """
    t = float(day_of_year)
    factor = (
        _DYN_COEFF_A4 * t**4
        + _DYN_COEFF_A3 * t**3
        + _DYN_COEFF_A2 * t**2
        + _DYN_COEFF_A1 * t
        + _DYN_COEFF_A0
    )
    return round(factor, 4)


# ---------------------------------------------------------------------------
# BDEW JSON loading
# ---------------------------------------------------------------------------


def _load_bdew_json(path: Path) -> dict:
    """Load and parse the BDEW profile JSON file.

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON with structure ``{profile_type: {month: {day_type: [96 values]}}}``.

    Raises
    ------
    FileNotFoundError
        When the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"BDEW profile JSON not found: {path}. "
            "Expected at .data/bdew_profile_2025.json"
        )
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_path(
    slp_type: str,
    year: int,
    cache_dir: str | Path | None,
) -> Path | None:
    """Return the cache file path, or ``None`` if caching is disabled."""
    if cache_dir is None:
        return None
    cache = Path(cache_dir)
    return cache / f"{slp_type.lower()}_{year}.npy"


def _load_from_cache(
    slp_type: str,
    year: int,
    cache_dir: str | Path | None,
    bdew_json_path: str | Path | None,
) -> np.ndarray | None:
    """Load a cached SLP array if it exists and is newer than the source JSON."""
    cp = _cache_path(slp_type, year, cache_dir)
    if cp is None or not cp.exists():
        return None

    # Invalidate if BDEW JSON is newer than cache
    bdew_path = Path(bdew_json_path) if bdew_json_path else _BDEW_JSON_PATH
    if bdew_path.exists() and bdew_path.stat().st_mtime > cp.stat().st_mtime:
        logger.debug("SLP cache invalidated (source JSON newer): %s", cp)
        return None

    logger.info("SLP cache hit: %s", cp)
    return np.load(cp)


def _save_to_cache(
    data: np.ndarray,
    slp_type: str,
    year: int,
    cache_dir: str | Path | None,
) -> None:
    """Save a generated SLP array to the cache directory."""
    cp = _cache_path(slp_type, year, cache_dir)
    if cp is None:
        return

    cp.parent.mkdir(parents=True, exist_ok=True)
    np.save(cp, data)
    logger.debug("SLP cached: %s", cp)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_leap_year(year: int) -> bool:
    """Return ``True`` if *year* is a leap year."""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
