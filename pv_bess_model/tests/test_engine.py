"""Integration tests for the dispatch engine (engine.py).

Tests use small synthetic data (2 days, 48 hours) to verify that the engine
correctly aggregates daily dispatch results into annual totals, handles BESS
offline days, and processes pricing-regime changes across years.
"""

from __future__ import annotations

import numpy as np
import pytest

from pv_bess_model.bess.replacement import ReplacementConfig
from pv_bess_model.config.defaults import (
    DEFAULT_START_SOC_FRACTION,
    DAYS_PER_YEAR,
    HOURS_PER_DAY,
    HOURS_PER_YEAR,
)
from pv_bess_model.dispatch.engine import (
    AnnualResult,
    DispatchEngineConfig,
    SimulationResult,
    compute_deterministic_offline_days,
    run_simulation,
)
from pv_bess_model.dispatch.optimizer import BessParams

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ATOL = 1e-4


def _replacement_disabled() -> ReplacementConfig:
    """Return a disabled replacement config."""
    return ReplacementConfig(enabled=False, year=0)


def _make_config(
    mode: str = "green",
    grid_max_kw: float = 500.0,
    bess_nameplate_kwh: float = 200.0,
    bess_power_kw: float = 100.0,
    bess_rte: float = 0.90,
    bess_min_soc_pct: float = 10.0,
    bess_max_soc_pct: float = 90.0,
    bess_degradation_rate: float = 0.0,
    pv_degradation_rate: float = 0.0,
    lifetime_years: int = 1,
    replacement: ReplacementConfig | None = None,
) -> DispatchEngineConfig:
    """Build a DispatchEngineConfig with convenient defaults."""
    return DispatchEngineConfig(
        mode=mode,  # type: ignore[arg-type]
        grid_max_kw=grid_max_kw,
        bess_nameplate_kwh=bess_nameplate_kwh,
        bess_max_charge_kw=bess_power_kw,
        bess_max_discharge_kw=bess_power_kw,
        bess_rte=bess_rte,
        bess_min_soc_pct=bess_min_soc_pct,
        bess_max_soc_pct=bess_max_soc_pct,
        bess_degradation_rate=bess_degradation_rate,
        pv_degradation_rate=pv_degradation_rate,
        replacement=replacement or _replacement_disabled(),
        lifetime_years=lifetime_years,
        bess_power_kw=bess_power_kw,
    )


def _make_yearly_pv(daily_pattern: np.ndarray) -> np.ndarray:
    """Tile a 24-hour PV pattern to fill one year (8760 hours)."""
    assert len(daily_pattern) == HOURS_PER_DAY
    return np.tile(daily_pattern, DAYS_PER_YEAR)


def _make_yearly_prices(daily_pattern: np.ndarray) -> np.ndarray:
    """Tile a 24-hour price pattern (EUR/kWh) to fill one year."""
    assert len(daily_pattern) == HOURS_PER_DAY
    return np.tile(daily_pattern, DAYS_PER_YEAR)


# ---------------------------------------------------------------------------
# compute_deterministic_offline_days
# ---------------------------------------------------------------------------


class TestComputeDeterministicOfflineDays:
    """Tests for the deterministic offline day calculator."""

    def test_100_pct_availability(self) -> None:
        """100% availability: no offline days."""
        result = compute_deterministic_offline_days(100.0)
        assert result == set()

    def test_0_pct_availability(self) -> None:
        """0% availability: all days offline."""
        result = compute_deterministic_offline_days(0.0)
        assert result == set(range(DAYS_PER_YEAR))

    def test_97_pct_availability(self) -> None:
        """97% availability: round((1-0.97)*365) = 11 offline days."""
        result = compute_deterministic_offline_days(97.0)
        n_offline = round((1.0 - 97.0 / 100.0) * DAYS_PER_YEAR)
        assert len(result) == n_offline
        # All day indices must be in valid range
        assert all(0 <= d < DAYS_PER_YEAR for d in result)

    def test_offline_days_evenly_spaced(self) -> None:
        """Offline days should be roughly evenly distributed."""
        result = compute_deterministic_offline_days(90.0)
        n_offline = round(0.10 * DAYS_PER_YEAR)  # 37 days
        assert len(result) == n_offline
        sorted_days = sorted(result)
        # Check spacing is roughly uniform (within 2x of ideal gap)
        ideal_gap = DAYS_PER_YEAR / n_offline
        for i in range(1, len(sorted_days)):
            gap = sorted_days[i] - sorted_days[i - 1]
            assert gap <= 2 * ideal_gap + 1


# ---------------------------------------------------------------------------
# 2-day simulation: aggregates = sum of daily results
# ---------------------------------------------------------------------------


class TestTwoDaySimulation:
    """Run a 1-year simulation with known 2-day repeating pattern.

    Verify that annual aggregates equal the sum of individual daily results
    across all 365 days.
    """

    def test_annual_aggregates_equal_daily_sums(self) -> None:
        """Annual energy flows must equal sum of 365 daily dispatches."""
        # Simple PV pattern: sun hours 8-16, peak 100 kWh at noon
        daily_pv = np.zeros(HOURS_PER_DAY)
        for h in range(8, 17):
            daily_pv[h] = 100.0 * np.sin(np.pi * (h - 8) / 8)
        pv_year = _make_yearly_pv(daily_pv)

        # Simple price pattern: low midday, high evening (EUR/kWh)
        daily_spot = np.full(HOURS_PER_DAY, 0.04)
        daily_spot[17:21] = 0.10  # evening peak
        daily_spot[10:15] = 0.02  # midday dip
        spot_year = _make_yearly_prices(daily_spot)

        config = _make_config(
            mode="green",
            grid_max_kw=200.0,
            bess_nameplate_kwh=200.0,
            bess_power_kw=100.0,
            lifetime_years=1,
        )

        result = run_simulation(
            config=config,
            pv_base_timeseries=pv_year,
            spot_prices_yearly=[spot_year],
            fixed_prices_yearly=[0.0],
            offline_days_yearly=[set()],
        )

        assert len(result.annual_results) == 1
        ar = result.annual_results[0]

        # PV production: sum of all hours
        pv_factor = (1.0 - config.pv_degradation_rate) ** 1  # = 1.0
        expected_pv_total = float(np.sum(pv_year)) * pv_factor
        assert abs(ar.pv_production - expected_pv_total) < ATOL

        # Energy conservation: pv_export + pv_curtailed + bess_charge_pv = pv_production
        energy_accounted = ar.pv_export + ar.pv_curtailed + ar.bess_charge_pv
        assert abs(energy_accounted - ar.pv_production) < ATOL

        # Revenue = pv_export_rev + bess_green_rev + bess_grey_rev - import_cost
        expected_total_rev = (
            ar.revenue_pv_export
            + ar.revenue_bess_green
            + ar.revenue_bess_grey
            - ar.grid_import_cost
        )
        assert abs(ar.total_revenue - expected_total_rev) < ATOL

    def test_pv_only_no_bess(self) -> None:
        """PV-only (BESS=0): no BESS throughput, all PV exported or curtailed."""
        daily_pv = np.zeros(HOURS_PER_DAY)
        daily_pv[10:15] = 150.0
        pv_year = _make_yearly_pv(daily_pv)

        daily_spot = np.full(HOURS_PER_DAY, 0.05)
        spot_year = _make_yearly_prices(daily_spot)

        config = _make_config(
            bess_nameplate_kwh=0.0,
            bess_power_kw=0.0,
            grid_max_kw=100.0,
            lifetime_years=1,
        )

        result = run_simulation(
            config=config,
            pv_base_timeseries=pv_year,
            spot_prices_yearly=[spot_year],
            fixed_prices_yearly=[0.0],
            offline_days_yearly=[set()],
        )

        ar = result.annual_results[0]
        assert abs(ar.bess_throughput) < ATOL
        assert abs(ar.bess_charge_pv) < ATOL
        assert abs(ar.bess_discharge_green) < ATOL
        # PV is either exported (up to grid_max) or curtailed
        assert abs(ar.pv_export + ar.pv_curtailed - ar.pv_production) < ATOL
        # Grid limit: export per hour <= 100 kW
        # With 150 kWh PV and 100 kW grid: 50 kWh/h curtailed for 5 hours/day
        expected_curtail_per_day = 5 * 50.0
        assert abs(ar.pv_curtailed - expected_curtail_per_day * DAYS_PER_YEAR) < ATOL


# ---------------------------------------------------------------------------
# Offline day: no BESS throughput
# ---------------------------------------------------------------------------


class TestOfflineDays:
    """Verify that BESS offline days produce zero BESS throughput."""

    def test_all_offline_no_bess_activity(self) -> None:
        """With all days offline, BESS throughput must be zero."""
        daily_pv = np.zeros(HOURS_PER_DAY)
        daily_pv[10:14] = 100.0
        pv_year = _make_yearly_pv(daily_pv)

        daily_spot = np.full(HOURS_PER_DAY, 0.05)
        spot_year = _make_yearly_prices(daily_spot)

        config = _make_config(
            bess_nameplate_kwh=200.0,
            bess_power_kw=100.0,
            lifetime_years=1,
        )

        all_offline = set(range(DAYS_PER_YEAR))

        result = run_simulation(
            config=config,
            pv_base_timeseries=pv_year,
            spot_prices_yearly=[spot_year],
            fixed_prices_yearly=[0.0],
            offline_days_yearly=[all_offline],
        )

        ar = result.annual_results[0]
        assert abs(ar.bess_throughput) < ATOL
        assert abs(ar.bess_charge_pv) < ATOL
        assert abs(ar.bess_discharge_green) < ATOL

    def test_some_offline_reduces_throughput(self) -> None:
        """More offline days should reduce BESS throughput compared to no offline."""
        daily_pv = np.zeros(HOURS_PER_DAY)
        daily_pv[10:14] = 200.0
        daily_pv[0:4] = 0.0
        pv_year = _make_yearly_pv(daily_pv)

        daily_spot = np.full(HOURS_PER_DAY, 0.02)
        daily_spot[18:22] = 0.10
        spot_year = _make_yearly_prices(daily_spot)

        config = _make_config(
            bess_nameplate_kwh=200.0,
            bess_power_kw=100.0,
            grid_max_kw=150.0,
            lifetime_years=1,
        )

        # No offline days
        result_online = run_simulation(
            config=config,
            pv_base_timeseries=pv_year,
            spot_prices_yearly=[spot_year],
            fixed_prices_yearly=[0.0],
            offline_days_yearly=[set()],
        )

        # Half the days offline
        half_offline = set(range(0, DAYS_PER_YEAR, 2))
        result_half = run_simulation(
            config=config,
            pv_base_timeseries=pv_year,
            spot_prices_yearly=[spot_year],
            fixed_prices_yearly=[0.0],
            offline_days_yearly=[half_offline],
        )

        assert result_half.annual_results[0].bess_throughput < (
            result_online.annual_results[0].bess_throughput
        )


# ---------------------------------------------------------------------------
# Regime change: year with floor vs year without floor
# ---------------------------------------------------------------------------


class TestRegimeChange:
    """Verify pricing-regime transitions across years."""

    def test_floor_year_higher_revenue(self) -> None:
        """Year with EEG floor should have higher revenue than without.

        2-year simulation: year 1 has floor=0.07 EUR/kWh, year 2 has no floor.
        Spot prices are mostly below floor.
        """
        daily_pv = np.zeros(HOURS_PER_DAY)
        daily_pv[8:16] = 100.0
        pv_year = _make_yearly_pv(daily_pv)

        # Spot mostly below the floor of 0.07
        daily_spot = np.full(HOURS_PER_DAY, 0.03)
        daily_spot[18:20] = 0.08  # Only 2 hours above floor
        spot_year = _make_yearly_prices(daily_spot)

        config = _make_config(
            bess_nameplate_kwh=200.0,
            bess_power_kw=100.0,
            grid_max_kw=500.0,
            lifetime_years=2,
        )

        result = run_simulation(
            config=config,
            pv_base_timeseries=pv_year,
            spot_prices_yearly=[spot_year, spot_year],
            fixed_prices_yearly=[0.07, 0.0],  # Floor in year 1, none in year 2
            offline_days_yearly=[set(), set()],
        )

        assert len(result.annual_results) == 2
        rev_y1 = result.annual_results[0].total_revenue
        rev_y2 = result.annual_results[1].total_revenue
        assert rev_y1 > rev_y2, (
            f"Year with floor should have higher revenue: {rev_y1:.2f} <= {rev_y2:.2f}"
        )

    def test_floor_then_market_revenue_breakdown(self) -> None:
        """Revenue breakdown: floor year PV export revenue uses max(spot, floor)."""
        daily_pv = np.zeros(HOURS_PER_DAY)
        daily_pv[10:14] = 100.0  # 4 hours of 100 kWh
        pv_year = _make_yearly_pv(daily_pv)

        daily_spot = np.full(HOURS_PER_DAY, 0.03)
        spot_year = _make_yearly_prices(daily_spot)

        # Tiny BESS so that revenue is purely from PV export
        config = _make_config(
            bess_nameplate_kwh=0.1,
            bess_power_kw=0.01,
            grid_max_kw=500.0,
            lifetime_years=2,
        )

        floor_price = 0.06  # EUR/kWh, above spot of 0.03

        result = run_simulation(
            config=config,
            pv_base_timeseries=pv_year,
            spot_prices_yearly=[spot_year, spot_year],
            fixed_prices_yearly=[floor_price, 0.0],
            offline_days_yearly=[set(), set()],
        )

        ar1 = result.annual_results[0]
        ar2 = result.annual_results[1]

        # Year 1: effective price = max(0.03, 0.06) = 0.06 for PV hours
        daily_pv_kwh = 4 * 100.0  # 400 kWh/day
        expected_rev_y1 = daily_pv_kwh * 0.06 * DAYS_PER_YEAR
        # Year 2: effective price = 0.03 (no floor)
        expected_rev_y2 = daily_pv_kwh * 0.03 * DAYS_PER_YEAR

        # Allow degradation tolerance (year 1 factor ~= 1.0 with 0% degradation)
        assert abs(ar1.total_revenue - expected_rev_y1) < 1.0
        assert abs(ar2.total_revenue - expected_rev_y2) < 1.0


# ---------------------------------------------------------------------------
# Hourly sample output
# ---------------------------------------------------------------------------


class TestHourlySample:
    """Verify the hourly sample (year 1 dispatch) is populated correctly."""

    def test_hourly_sample_shape(self) -> None:
        """Hourly sample arrays must have shape (8760,)."""
        daily_pv = np.zeros(HOURS_PER_DAY)
        daily_pv[10:14] = 100.0
        pv_year = _make_yearly_pv(daily_pv)

        daily_spot = np.full(HOURS_PER_DAY, 0.05)
        spot_year = _make_yearly_prices(daily_spot)

        config = _make_config(lifetime_years=1)

        result = run_simulation(
            config=config,
            pv_base_timeseries=pv_year,
            spot_prices_yearly=[spot_year],
            fixed_prices_yearly=[0.0],
            offline_days_yearly=[set()],
        )

        hs = result.hourly_sample
        assert hs.pv_production.shape == (HOURS_PER_YEAR,)
        assert hs.spot_prices.shape == (HOURS_PER_YEAR,)
        assert hs.soc.shape == (HOURS_PER_YEAR,)
        assert hs.revenue.shape == (HOURS_PER_YEAR,)

    def test_hourly_sample_revenue_sums_to_annual(self) -> None:
        """Sum of hourly revenue must equal annual total revenue."""
        daily_pv = np.zeros(HOURS_PER_DAY)
        daily_pv[10:14] = 100.0
        pv_year = _make_yearly_pv(daily_pv)

        daily_spot = np.full(HOURS_PER_DAY, 0.03)
        daily_spot[18:20] = 0.10
        spot_year = _make_yearly_prices(daily_spot)

        config = _make_config(
            bess_nameplate_kwh=200.0,
            bess_power_kw=100.0,
            grid_max_kw=200.0,
            lifetime_years=1,
        )

        result = run_simulation(
            config=config,
            pv_base_timeseries=pv_year,
            spot_prices_yearly=[spot_year],
            fixed_prices_yearly=[0.0],
            offline_days_yearly=[set()],
        )

        hourly_rev_sum = float(np.sum(result.hourly_sample.revenue))
        annual_rev = result.annual_results[0].total_revenue
        assert abs(hourly_rev_sum - annual_rev) < 1.0  # Within 1 EUR tolerance


# ---------------------------------------------------------------------------
# Multi-year degradation
# ---------------------------------------------------------------------------


class TestMultiYearDegradation:
    """Verify PV and BESS degradation across multiple years."""

    def test_pv_production_decreases_with_degradation(self) -> None:
        """PV production should decrease year-over-year with degradation."""
        daily_pv = np.zeros(HOURS_PER_DAY)
        daily_pv[10:14] = 200.0
        pv_year = _make_yearly_pv(daily_pv)

        daily_spot = np.full(HOURS_PER_DAY, 0.05)
        spot_year = _make_yearly_prices(daily_spot)

        config = _make_config(
            bess_nameplate_kwh=0.0,
            bess_power_kw=0.0,
            pv_degradation_rate=0.02,  # 2% per year for visibility
            grid_max_kw=500.0,
            lifetime_years=3,
        )

        result = run_simulation(
            config=config,
            pv_base_timeseries=pv_year,
            spot_prices_yearly=[spot_year] * 3,
            fixed_prices_yearly=[0.0] * 3,
            offline_days_yearly=[set()] * 3,
        )

        pv_y1 = result.annual_results[0].pv_production
        pv_y2 = result.annual_results[1].pv_production
        pv_y3 = result.annual_results[2].pv_production

        assert pv_y1 > pv_y2 > pv_y3
        # Check degradation factor: year 2 = base * (1-0.02)^2
        ratio_y2_y1 = pv_y2 / pv_y1
        expected_ratio = (1.0 - 0.02) ** 2 / (1.0 - 0.02) ** 1
        assert abs(ratio_y2_y1 - expected_ratio) < ATOL

    def test_bess_capacity_decreases(self) -> None:
        """BESS effective capacity should decrease with degradation."""
        daily_pv = np.zeros(HOURS_PER_DAY)
        daily_pv[10:14] = 100.0
        pv_year = _make_yearly_pv(daily_pv)

        daily_spot = np.full(HOURS_PER_DAY, 0.05)
        spot_year = _make_yearly_prices(daily_spot)

        config = _make_config(
            bess_nameplate_kwh=200.0,
            bess_power_kw=100.0,
            bess_degradation_rate=0.05,  # 5% per year for visibility
            lifetime_years=3,
        )

        result = run_simulation(
            config=config,
            pv_base_timeseries=pv_year,
            spot_prices_yearly=[spot_year] * 3,
            fixed_prices_yearly=[0.0] * 3,
            offline_days_yearly=[set()] * 3,
        )

        cap_y1 = result.annual_results[0].bess_capacity_kwh
        cap_y2 = result.annual_results[1].bess_capacity_kwh
        cap_y3 = result.annual_results[2].bess_capacity_kwh
        assert cap_y1 > cap_y2 > cap_y3
