"""Tests for portfolio.load_profiles – BDEW SLP generation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pv_bess_model.config.defaults import INTERVALS_PER_DAY, SLP_NORMIERUNG_KWH
from pv_bess_model.portfolio.load_profiles import (
    _dynamization_factor,
    _is_leap_year,
    generate_calendar,
    generate_slp,
    scale_slp,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_bdew_json(tmp_path: Path) -> Path:
    """Create a minimal BDEW JSON with constant profile values."""
    # Each profile type, each month, each day type = 96 values of 1.0
    profile_types = ["H25", "G25", "L25", "P25", "S25"]
    months = [
        "Jan", "Feb", "Mrz", "Apr", "Mai", "Jun",
        "Jul", "Aug", "Sep", "Okt", "Nov", "Dez",
    ]
    day_types = ["SA", "FT", "WT"]

    data: dict = {}
    for pt in profile_types:
        data[pt] = {}
        for month in months:
            data[pt][month] = {}
            for dt in day_types:
                data[pt][month][dt] = [1.0] * 96

    json_path = tmp_path / "bdew_test.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")
    return json_path


@pytest.fixture
def varied_bdew_json(tmp_path: Path) -> Path:
    """Create a BDEW JSON with varied profile values for testing dynamization."""
    months = [
        "Jan", "Feb", "Mrz", "Apr", "Mai", "Jun",
        "Jul", "Aug", "Sep", "Okt", "Nov", "Dez",
    ]
    day_types = ["SA", "FT", "WT"]

    data: dict = {}
    # Only H25 needed for dynamization tests
    data["H25"] = {}
    data["G25"] = {}
    for month_idx, month in enumerate(months):
        # H25: month-varying values (winter higher, summer lower)
        base = 30.0 - abs(month_idx - 5) * 2
        h25_values = [base + i * 0.1 for i in range(96)]

        data["H25"][month] = {dt: h25_values.copy() for dt in day_types}

        # G25: different values per day type to enable year-comparison tests
        data["G25"][month] = {
            "WT": [10.0 + i * 0.1 for i in range(96)],
            "SA": [8.0 + i * 0.05 for i in range(96)],
            "FT": [6.0 + i * 0.02 for i in range(96)],
        }

    json_path = tmp_path / "bdew_varied.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")
    return json_path


# ---------------------------------------------------------------------------
# Calendar generation
# ---------------------------------------------------------------------------


class TestGenerateCalendar:
    """Tests for generate_calendar()."""

    def test_always_365_days(self) -> None:
        """Calendar must have exactly 365 entries for any year."""
        for year in (2024, 2025, 2027, 2028):  # 2024/2028 are leap years
            cal = generate_calendar(year, bundesland="SH")
            assert len(cal) == 365, f"Year {year} has {len(cal)} days"

    def test_day_types_only_valid(self) -> None:
        """All day types must be WT, SA, or FT."""
        cal = generate_calendar(2027, bundesland="SH")
        valid = {"WT", "SA", "FT"}
        for _, dt in cal:
            assert dt in valid, f"Invalid day type: {dt}"

    def test_months_range(self) -> None:
        """Month values must be 1–12."""
        cal = generate_calendar(2027, bundesland="SH")
        for m, _ in cal:
            assert 1 <= m <= 12

    def test_saturdays_are_sa(self) -> None:
        """Saturdays that are not holidays should be SA."""
        import datetime
        import holidays as hol

        year = 2027
        cal = generate_calendar(year, bundesland="SH")
        de_holidays = hol.Germany(years=year, subdiv="SH")
        date = datetime.date(year, 1, 1)
        for i, (_, dt) in enumerate(cal):
            if date.weekday() == 5 and date not in de_holidays:
                assert dt == "SA", f"Day {date} is Saturday but got {dt}"
            date += datetime.timedelta(days=1)
            # Skip leap day
            if _is_leap_year(year) and date.month == 12 and date.day == 31:
                date += datetime.timedelta(days=1)

    def test_sundays_are_ft(self) -> None:
        """Sundays should always be FT."""
        import datetime

        year = 2027
        cal = generate_calendar(year, bundesland="SH")
        date = datetime.date(year, 1, 1)
        for _, dt in cal:
            if date.weekday() == 6:
                assert dt == "FT", f"Day {date} is Sunday but got {dt}"
            date += datetime.timedelta(days=1)
            if _is_leap_year(year) and date.month == 12 and date.day == 31:
                date += datetime.timedelta(days=1)

    def test_holidays_are_ft(self) -> None:
        """Public holidays should be FT regardless of weekday."""
        import datetime
        import holidays as hol

        year = 2027
        de_holidays = hol.Germany(years=year, subdiv="SH")
        cal = generate_calendar(year, bundesland="SH")
        date = datetime.date(year, 1, 1)
        for _, dt in cal:
            if date in de_holidays:
                assert dt == "FT", f"Holiday {date} got {dt}"
            date += datetime.timedelta(days=1)
            if _is_leap_year(year) and date.month == 12 and date.day == 31:
                date += datetime.timedelta(days=1)

    def test_different_bundesland(self) -> None:
        """Different Bundesland should affect holiday count."""
        cal_sh = generate_calendar(2027, bundesland="SH")
        cal_by = generate_calendar(2027, bundesland="BY")
        ft_sh = sum(1 for _, dt in cal_sh if dt == "FT")
        ft_by = sum(1 for _, dt in cal_by if dt == "FT")
        # Bayern has more holidays than Schleswig-Holstein
        assert ft_by >= ft_sh


# ---------------------------------------------------------------------------
# Dynamization
# ---------------------------------------------------------------------------


class TestDynamization:
    """Tests for _dynamization_factor()."""

    def test_january_1st(self) -> None:
        """Day 1 should have a factor close to the polynomial value."""
        f = _dynamization_factor(1)
        expected = -3.92e-10 * 1 + 3.20e-7 * 1 - 7.02e-5 * 1 + 2.10e-3 * 1 + 1.24
        assert abs(f - round(expected, 4)) < 1e-6

    def test_midsummer_lower(self) -> None:
        """Factor in summer (day ~180) should be lower than in winter."""
        f_winter = _dynamization_factor(1)
        f_summer = _dynamization_factor(180)
        assert f_summer < f_winter

    def test_symmetric_roughly(self) -> None:
        """Factor should be roughly symmetric around midsummer."""
        f_jan = _dynamization_factor(15)
        f_dec = _dynamization_factor(350)
        # Not exact symmetry, but both should be > 1
        assert f_jan > 1.0
        assert f_dec > 1.0

    def test_rounded_to_4_decimals(self) -> None:
        """Factor must be rounded to 4 decimal places per BDEW spec."""
        for day in (1, 90, 180, 270, 365):
            f = _dynamization_factor(day)
            assert f == round(f, 4)


# ---------------------------------------------------------------------------
# SLP generation
# ---------------------------------------------------------------------------


class TestGenerateSlp:
    """Tests for generate_slp()."""

    def test_output_length(self, minimal_bdew_json: Path) -> None:
        """Output must have exactly 35,040 values."""
        slp = generate_slp("H25", 2027, bdew_json_path=minimal_bdew_json, cache_dir=None)
        assert len(slp) == 365 * INTERVALS_PER_DAY

    def test_normalized_sum(self, minimal_bdew_json: Path) -> None:
        """Output must sum to SLP_NORMIERUNG_KWH."""
        slp = generate_slp("G25", 2027, bdew_json_path=minimal_bdew_json, cache_dir=None)
        assert abs(np.sum(slp) - SLP_NORMIERUNG_KWH) < 0.01

    def test_all_positive(self, minimal_bdew_json: Path) -> None:
        """All SLP values must be non-negative."""
        slp = generate_slp("H25", 2027, bdew_json_path=minimal_bdew_json, cache_dir=None)
        assert np.all(slp >= 0)

    def test_unknown_slp_type_raises(self, minimal_bdew_json: Path) -> None:
        """Requesting an unknown profile type should raise KeyError."""
        with pytest.raises(KeyError, match="X99"):
            generate_slp("X99", 2027, bdew_json_path=minimal_bdew_json, cache_dir=None)

    def test_missing_json_raises(self) -> None:
        """Missing BDEW JSON should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            generate_slp("H25", 2027, bdew_json_path="/nonexistent.json", cache_dir=None)

    def test_h25_dynamized(self, minimal_bdew_json: Path) -> None:
        """H25 should have dynamization applied (values differ by day)."""
        slp = generate_slp("H25", 2027, bdew_json_path=minimal_bdew_json, cache_dir=None)
        # Day 1 (winter) values should differ from day 180 (summer) values
        day1_sum = np.sum(slp[:INTERVALS_PER_DAY])
        day180_sum = np.sum(slp[179 * INTERVALS_PER_DAY : 180 * INTERVALS_PER_DAY])
        # Winter day should have higher normalized consumption than summer day
        assert day1_sum != day180_sum

    def test_g25_not_dynamized(self, minimal_bdew_json: Path) -> None:
        """G25 should NOT have dynamization applied.

        With all-1.0 input values and no dynamization, all days within
        the same month and day-type should be identical.
        """
        slp = generate_slp("G25", 2027, bdew_json_path=minimal_bdew_json, cache_dir=None)
        # Find two weekdays in January
        cal = generate_calendar(2027, bundesland="SH")
        jan_wt_indices = [
            i for i, (m, dt) in enumerate(cal) if m == 1 and dt == "WT"
        ]
        if len(jan_wt_indices) >= 2:
            d1 = jan_wt_indices[0]
            d2 = jan_wt_indices[1]
            vals1 = slp[d1 * INTERVALS_PER_DAY : (d1 + 1) * INTERVALS_PER_DAY]
            vals2 = slp[d2 * INTERVALS_PER_DAY : (d2 + 1) * INTERVALS_PER_DAY]
            np.testing.assert_array_almost_equal(vals1, vals2)

    def test_different_years_differ(self, varied_bdew_json: Path) -> None:
        """Different calendar years should produce different profiles (different
        weekday distributions lead to different day-type assignments)."""
        slp_2027 = generate_slp(
            "G25", 2027, bdew_json_path=varied_bdew_json, cache_dir=None
        )
        slp_2028 = generate_slp(
            "G25", 2028, bdew_json_path=varied_bdew_json, cache_dir=None
        )
        # G25 has no dynamization, so day-type assignment is the only
        # difference – with varied SA/FT/WT values this should produce
        # different profiles for years with different calendar layouts.
        # 2027 starts on Friday, 2028 on Saturday.
        assert not np.allclose(slp_2027, slp_2028)


# ---------------------------------------------------------------------------
# SLP caching
# ---------------------------------------------------------------------------


class TestSlpCache:
    """Tests for SLP caching logic."""

    def test_cache_produces_file(self, minimal_bdew_json: Path, tmp_path: Path) -> None:
        """Generating an SLP should create a .npy cache file."""
        cache_dir = tmp_path / "slp_cache"
        generate_slp(
            "H25", 2027, bdew_json_path=minimal_bdew_json, cache_dir=cache_dir
        )
        expected = cache_dir / "h25_2027.npy"
        assert expected.exists()

    def test_cache_hit_returns_same(
        self, minimal_bdew_json: Path, tmp_path: Path
    ) -> None:
        """Cache hit should return identical array."""
        cache_dir = tmp_path / "slp_cache"
        slp1 = generate_slp(
            "H25", 2027, bdew_json_path=minimal_bdew_json, cache_dir=cache_dir
        )
        slp2 = generate_slp(
            "H25", 2027, bdew_json_path=minimal_bdew_json, cache_dir=cache_dir
        )
        np.testing.assert_array_equal(slp1, slp2)

    def test_no_cache_when_disabled(
        self, minimal_bdew_json: Path, tmp_path: Path
    ) -> None:
        """cache_dir=None should not create any files."""
        generate_slp("H25", 2027, bdew_json_path=minimal_bdew_json, cache_dir=None)
        # No .npy files should exist in tmp_path
        npy_files = list(tmp_path.rglob("*.npy"))
        assert len(npy_files) == 0


# ---------------------------------------------------------------------------
# SLP scaling
# ---------------------------------------------------------------------------


class TestScaleSlp:
    """Tests for scale_slp()."""

    def test_scaling_factor(self) -> None:
        """Scaling should multiply by (consumption × count / normierung)."""
        slp = np.ones(35_040) * (SLP_NORMIERUNG_KWH / 35_040)
        scaled = scale_slp(slp, annual_consumption_kwh=3200.0, customer_count=100)
        expected_sum = 3200.0 * 100
        assert abs(np.sum(scaled) - expected_sum) < 0.01

    def test_single_customer(self) -> None:
        """One customer with 1M kWh should return the normalized SLP unchanged."""
        slp = np.ones(35_040) * (SLP_NORMIERUNG_KWH / 35_040)
        scaled = scale_slp(slp, annual_consumption_kwh=SLP_NORMIERUNG_KWH, customer_count=1)
        np.testing.assert_array_almost_equal(scaled, slp)

    def test_zero_customers(self) -> None:
        """Zero customers should produce all-zero profile."""
        slp = np.ones(35_040) * (SLP_NORMIERUNG_KWH / 35_040)
        scaled = scale_slp(slp, annual_consumption_kwh=3200.0, customer_count=0)
        np.testing.assert_array_equal(scaled, np.zeros(35_040))


# ---------------------------------------------------------------------------
# Integration test with real BDEW data
# ---------------------------------------------------------------------------


class TestSlpWithRealData:
    """Tests using the real BDEW JSON file."""

    REAL_JSON = Path(".data/bdew_profile_2025.json")

    @pytest.mark.skipif(
        not Path(".data/bdew_profile_2025.json").exists(),
        reason="BDEW JSON not available",
    )
    def test_real_h25_profile(self) -> None:
        """Generate H25 profile from real BDEW data."""
        slp = generate_slp("H25", 2027, bdew_json_path=self.REAL_JSON, cache_dir=None)
        assert len(slp) == 35_040
        assert abs(np.sum(slp) - SLP_NORMIERUNG_KWH) < 1.0  # tolerance for rounding
        assert np.all(slp >= 0)

    @pytest.mark.skipif(
        not Path(".data/bdew_profile_2025.json").exists(),
        reason="BDEW JSON not available",
    )
    def test_real_g25_profile(self) -> None:
        """Generate G25 profile from real BDEW data."""
        slp = generate_slp("G25", 2027, bdew_json_path=self.REAL_JSON, cache_dir=None)
        assert len(slp) == 35_040
        assert abs(np.sum(slp) - SLP_NORMIERUNG_KWH) < 1.0

    @pytest.mark.skipif(
        not Path(".data/bdew_profile_2025.json").exists(),
        reason="BDEW JSON not available",
    )
    def test_all_profile_types(self) -> None:
        """All five profile types should be generatable."""
        for slp_type in ("H25", "G25", "L25", "P25", "S25"):
            slp = generate_slp(
                slp_type, 2027, bdew_json_path=self.REAL_JSON, cache_dir=None
            )
            assert len(slp) == 35_040
            assert np.sum(slp) > 0, f"{slp_type} has zero sum"


# ---------------------------------------------------------------------------
# Leap year handling
# ---------------------------------------------------------------------------


class TestLeapYear:
    """Tests for _is_leap_year helper."""

    def test_leap_years(self) -> None:
        assert _is_leap_year(2024) is True
        assert _is_leap_year(2028) is True
        assert _is_leap_year(2000) is True

    def test_non_leap_years(self) -> None:
        assert _is_leap_year(2027) is False
        assert _is_leap_year(2025) is False
        assert _is_leap_year(1900) is False
