"""Unit tests for the daily LP dispatch optimizer.

Tests cover Green Mode, Grey Mode, and edge cases with numerical verification
of the LP solution against hand-computed reference values.

All prices passed to the optimizer are in EUR/kWh (the price_loader converts
EUR/MWh -> EUR/kWh before this module).
"""

from __future__ import annotations

import numpy as np
import pytest

from pv_bess_model.dispatch.optimizer import (
    BessParams,
    DailyDispatchResult,
    dispatch_offline_day,
    optimize_day,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ATOL = 1e-4
"""Absolute tolerance for floating-point comparisons (kWh / EUR)."""


def _make_bess(
    power_kw: float = 100.0,
    capacity_kwh: float = 200.0,
    rte: float = 0.90,
    min_soc_pct: float = 10.0,
    max_soc_pct: float = 90.0,
) -> BessParams:
    """Build a BessParams from convenient shorthand."""
    return BessParams(
        max_charge_kw=power_kw,
        max_discharge_kw=power_kw,
        round_trip_efficiency=rte,
        soc_min_kwh=capacity_kwh * min_soc_pct / 100.0,
        soc_max_kwh=capacity_kwh * max_soc_pct / 100.0,
    )


def _assert_energy_balance(result: DailyDispatchResult, pv: np.ndarray) -> None:
    """Assert PV energy balance: export + charge_pv + curtail = production."""
    lhs = result["export_pv"] + result["charge_pv"] + result["curtail"]
    np.testing.assert_allclose(lhs, pv, atol=ATOL)


def _assert_soc_within_bounds(
    result: DailyDispatchResult, soc_min: float, soc_max: float
) -> None:
    """Assert SoC stays within [min, max] at every timestep."""
    assert np.all(result["soc"] >= soc_min - ATOL), (
        f"SoC below min: {result['soc'].min():.4f} < {soc_min}"
    )
    assert np.all(result["soc"] <= soc_max + ATOL), (
        f"SoC above max: {result['soc'].max():.4f} > {soc_max}"
    )


def _assert_grid_limit(
    result: DailyDispatchResult, rte: float, grid_max: float
) -> None:
    """Assert grid limit: export_pv + (disch_green + disch_grey) x RTE <= grid_max."""
    grid_out = (
        result["export_pv"]
        + (result["discharge_green"] + result["discharge_grey"]) * rte
    )
    assert np.all(grid_out <= grid_max + ATOL), (
        f"Grid limit exceeded: max = {grid_out.max():.4f} > {grid_max}"
    )


def _assert_soc_tracking_green(
    result: DailyDispatchResult, start_soc: float
) -> None:
    """Assert SoC[t] = SoC[t-1] + charge_pv[t] - discharge_green[t] (Green Mode)."""
    T = len(result["soc"])
    expected_soc = np.empty(T)
    cumulative = start_soc
    for t in range(T):
        cumulative += result["charge_pv"][t] - result["discharge_green"][t]
        expected_soc[t] = cumulative
    np.testing.assert_allclose(result["soc"], expected_soc, atol=ATOL)


# ============================================================================
# GREEN MODE TESTS
# ============================================================================


class TestGreenModeReferenceOptimizer4h:
    """Reference 4-hour LP test with exact hand-computed results."""

    def test_optimal_dispatch_matches_reference(
        self, reference_optimizer_4h: dict
    ) -> None:
        """The LP solution must match the documented optimal dispatch."""
        ref = reference_optimizer_4h
        bess = _make_bess(
            power_kw=ref["bess_power_kw"],
            capacity_kwh=ref["bess_capacity_kwh"],
            rte=ref["rte"],
        )
        # Spot prices in fixture are EUR/MWh, optimizer expects EUR/kWh
        spot_eur_kwh = ref["spot_prices_eur_per_mwh"] / 1000.0

        result = optimize_day(
            pv_production_kwh=ref["pv_production_kwh"],
            spot_prices_eur_per_kwh=spot_eur_kwh,
            price_fixed_eur_per_kwh=ref["price_fixed_eur_per_kwh"],
            bess=bess,
            grid_max_kw=ref["grid_max_kw"],
            mode=ref["mode"],
            start_soc_kwh=ref["start_soc_kwh"],
        )

        np.testing.assert_allclose(
            result["charge_pv"], ref["expected_charge_pv_kwh"], atol=ATOL
        )
        np.testing.assert_allclose(
            result["export_pv"], ref["expected_export_pv_kwh"], atol=ATOL
        )
        np.testing.assert_allclose(
            result["curtail"], ref["expected_curtail_kwh"], atol=ATOL
        )
        np.testing.assert_allclose(
            result["discharge_green"],
            ref["expected_discharge_green_kwh"],
            atol=ATOL,
        )
        np.testing.assert_allclose(
            result["soc"], ref["expected_soc_kwh"], atol=ATOL
        )

    def test_total_revenue_matches_reference(
        self, reference_optimizer_4h: dict
    ) -> None:
        """Total revenue must equal 121/9 ~ 13.4444 EUR."""
        ref = reference_optimizer_4h
        bess = _make_bess(
            power_kw=ref["bess_power_kw"],
            capacity_kwh=ref["bess_capacity_kwh"],
            rte=ref["rte"],
        )
        spot_eur_kwh = ref["spot_prices_eur_per_mwh"] / 1000.0

        result = optimize_day(
            pv_production_kwh=ref["pv_production_kwh"],
            spot_prices_eur_per_kwh=spot_eur_kwh,
            price_fixed_eur_per_kwh=ref["price_fixed_eur_per_kwh"],
            bess=bess,
            grid_max_kw=ref["grid_max_kw"],
            mode=ref["mode"],
            start_soc_kwh=ref["start_soc_kwh"],
        )

        total_rev = float(np.sum(result["revenue"]))
        assert abs(total_rev - ref["expected_total_revenue_eur"]) < ATOL

    def test_grid_export_matches_reference(
        self, reference_optimizer_4h: dict
    ) -> None:
        """Grid export per hour (PV + BESS x RTE) matches reference."""
        ref = reference_optimizer_4h
        bess = _make_bess(
            power_kw=ref["bess_power_kw"],
            capacity_kwh=ref["bess_capacity_kwh"],
            rte=ref["rte"],
        )
        spot_eur_kwh = ref["spot_prices_eur_per_mwh"] / 1000.0

        result = optimize_day(
            pv_production_kwh=ref["pv_production_kwh"],
            spot_prices_eur_per_kwh=spot_eur_kwh,
            price_fixed_eur_per_kwh=ref["price_fixed_eur_per_kwh"],
            bess=bess,
            grid_max_kw=ref["grid_max_kw"],
            mode=ref["mode"],
            start_soc_kwh=ref["start_soc_kwh"],
        )

        grid_export = result["export_pv"] + result["discharge_green"] * ref["rte"]
        np.testing.assert_allclose(
            grid_export, ref["expected_grid_export_kwh"], atol=ATOL
        )


class TestGreenModePvEnergyBalance:
    """PV energy balance: export + charge_pv + curtail = production."""

    def test_energy_balance_normal(self) -> None:
        """Standard case with moderate PV and BESS."""
        pv = np.array([0.0, 100.0, 200.0, 50.0])
        spot = np.array([0.05, 0.03, 0.02, 0.08])
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=300.0,
            mode="green",
            start_soc_kwh=100.0,
        )
        _assert_energy_balance(result, pv)

    def test_energy_balance_24h(
        self, sample_pv_timeseries_24h: np.ndarray
    ) -> None:
        """24-hour profile: energy balance must hold for every hour."""
        pv = sample_pv_timeseries_24h
        spot = np.linspace(0.02, 0.10, 24)
        bess = _make_bess(power_kw=200.0, capacity_kwh=400.0)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=600.0,
            mode="green",
            start_soc_kwh=200.0,
        )
        _assert_energy_balance(result, pv)


class TestGreenModeSocTracking:
    """SoC tracking: SoC[t] = SoC[t-1] + charge_pv[t] - discharge_green[t]."""

    def test_soc_tracking_4h(self, reference_optimizer_4h: dict) -> None:
        """SoC trajectory matches cumulative charge/discharge from reference case."""
        ref = reference_optimizer_4h
        bess = _make_bess(
            power_kw=ref["bess_power_kw"],
            capacity_kwh=ref["bess_capacity_kwh"],
            rte=ref["rte"],
        )
        spot_eur_kwh = ref["spot_prices_eur_per_mwh"] / 1000.0

        result = optimize_day(
            pv_production_kwh=ref["pv_production_kwh"],
            spot_prices_eur_per_kwh=spot_eur_kwh,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=ref["grid_max_kw"],
            mode="green",
            start_soc_kwh=ref["start_soc_kwh"],
        )
        _assert_soc_tracking_green(result, ref["start_soc_kwh"])


class TestGreenModeSocBounds:
    """SoC must stay within [soc_min, soc_max] at every timestep."""

    def test_soc_within_bounds(self, reference_optimizer_4h: dict) -> None:
        """Reference case: SoC must stay within 20-180 kWh."""
        ref = reference_optimizer_4h
        bess = _make_bess(
            power_kw=ref["bess_power_kw"],
            capacity_kwh=ref["bess_capacity_kwh"],
            rte=ref["rte"],
        )
        spot_eur_kwh = ref["spot_prices_eur_per_mwh"] / 1000.0

        result = optimize_day(
            pv_production_kwh=ref["pv_production_kwh"],
            spot_prices_eur_per_kwh=spot_eur_kwh,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=ref["grid_max_kw"],
            mode="green",
            start_soc_kwh=ref["start_soc_kwh"],
        )
        _assert_soc_within_bounds(result, bess.soc_min_kwh, bess.soc_max_kwh)

    def test_soc_bounds_tight_capacity(self) -> None:
        """BESS with very tight usable range must still respect SoC bounds."""
        bess = _make_bess(
            power_kw=50.0,
            capacity_kwh=100.0,
            min_soc_pct=40.0,
            max_soc_pct=60.0,
        )
        pv = np.array([0.0, 80.0, 80.0, 0.0])
        spot = np.array([0.01, 0.01, 0.01, 0.10])

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=200.0,
            mode="green",
            start_soc_kwh=50.0,
        )
        _assert_soc_within_bounds(result, bess.soc_min_kwh, bess.soc_max_kwh)


class TestGreenModeGridLimit:
    """Grid connection limit: export_pv + discharge_green x RTE <= P_grid_max."""

    def test_grid_limit_binding(self) -> None:
        """When PV exceeds grid limit, surplus must be charged or curtailed."""
        pv = np.array([300.0, 300.0, 0.0, 0.0])
        spot = np.array([0.05, 0.05, 0.05, 0.10])
        bess = _make_bess(power_kw=200.0, capacity_kwh=400.0)
        grid_max = 150.0

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=grid_max,
            mode="green",
            start_soc_kwh=40.0,
        )
        _assert_grid_limit(result, bess.round_trip_efficiency, grid_max)
        _assert_energy_balance(result, pv)


class TestGreenModePriceIncentives:
    """Optimizer should shift energy from low-price to high-price hours."""

    def test_flat_prices_all_export(self) -> None:
        """All PV should be exported when prices are flat and no constraint binds."""
        pv = np.array([100.0, 100.0, 100.0, 100.0])
        spot = np.array([0.10, 0.10, 0.10, 0.10])
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=100.0,
        )
        # With flat prices, no incentive to shift -> all PV exported
        np.testing.assert_allclose(result["export_pv"], pv, atol=ATOL)
        np.testing.assert_allclose(result["charge_pv"], np.zeros(4), atol=ATOL)

    def test_shift_low_to_high(self) -> None:
        """Cheap-hour PV should charge BESS, expensive hour should discharge.

        Setup: 4 hours, huge PV at t=0 (low price), zero PV at t=3 (high price).
        The optimizer should charge at t=0 and discharge at t=3.
        """
        pv = np.array([200.0, 0.0, 0.0, 0.0])
        spot = np.array([0.01, 0.01, 0.01, 0.10])
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=80.0,
        )
        # Should charge at t=0 (cheap) and discharge at t=3 (expensive)
        assert result["charge_pv"][0] > 0.0, "Should charge at low-price hour"
        assert result["discharge_green"][3] > 0.0, "Should discharge at high-price hour"
        _assert_energy_balance(result, pv)


class TestGreenModeEegFloor:
    """EEG floor price: effective price = max(spot, floor)."""

    def test_floor_raises_revenue(self) -> None:
        """With floor > spot, revenue should increase compared to no floor."""
        pv = np.array([100.0, 100.0, 100.0, 100.0])
        spot = np.array([0.02, 0.02, 0.02, 0.02])
        floor = 0.05  # EUR/kWh -- above spot
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0)

        result_no_floor = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=100.0,
        )
        result_with_floor = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=100.0,
        )
        rev_no = float(np.sum(result_no_floor["revenue"]))
        rev_yes = float(np.sum(result_with_floor["revenue"]))
        assert rev_yes > rev_no, (
            f"Floor should increase revenue: {rev_yes:.4f} <= {rev_no:.4f}"
        )

    def test_floor_price_numerical(self) -> None:
        """With flat PV, no BESS activity, revenue = PV x floor when spot < floor."""
        pv = np.array([100.0, 100.0, 100.0, 100.0])
        spot = np.array([0.02, 0.02, 0.02, 0.02])
        floor = 0.05  # EUR/kWh
        # Use tiny BESS so it effectively does not participate
        bess = _make_bess(power_kw=0.01, capacity_kwh=0.1)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=0.05,
        )
        # Revenue should approximately equal PV x floor (since floor > spot).
        # Tiny BESS may contribute a negligible amount (~0.002 EUR).
        expected_revenue = float(np.sum(pv)) * floor
        actual_revenue = float(np.sum(result["revenue"]))
        assert abs(actual_revenue - expected_revenue) < 0.01

    def test_floor_mixed_prices(self) -> None:
        """When some spot > floor and some spot < floor, effective = max(spot, floor)."""
        pv = np.array([100.0, 100.0, 100.0, 100.0])
        spot = np.array([0.02, 0.08, 0.03, 0.10])
        floor = 0.05  # EUR/kWh
        # Tiny BESS -> no shifting
        bess = _make_bess(power_kw=0.01, capacity_kwh=0.1)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=0.05,
        )
        eff = np.maximum(spot, floor)
        expected_revenue = float(np.sum(pv * eff))
        actual_revenue = float(np.sum(result["revenue"]))
        # Tiny BESS may contribute a negligible amount (~0.003 EUR).
        assert abs(actual_revenue - expected_revenue) < 0.01


class TestGreenModeBessOffline:
    """BESS offline day: all BESS variables = 0, SoC frozen."""

    def test_offline_all_bess_zero(self) -> None:
        """Offline day: charge, discharge all zero."""
        pv = np.array([100.0, 200.0, 50.0, 0.0])
        spot = np.array([0.05, 0.03, 0.02, 0.08])
        start_soc = 75.0

        result = dispatch_offline_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            grid_max_kw=150.0,
            start_soc_kwh=start_soc,
        )

        np.testing.assert_allclose(result["charge_pv"], np.zeros(4), atol=ATOL)
        np.testing.assert_allclose(result["discharge_green"], np.zeros(4), atol=ATOL)
        np.testing.assert_allclose(result["charge_grid"], np.zeros(4), atol=ATOL)
        np.testing.assert_allclose(result["discharge_grey"], np.zeros(4), atol=ATOL)

    def test_offline_soc_frozen(self) -> None:
        """Offline day: SoC remains at start value for all hours."""
        pv = np.array([100.0, 200.0, 50.0, 0.0])
        spot = np.array([0.05, 0.03, 0.02, 0.08])
        start_soc = 75.0

        result = dispatch_offline_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            grid_max_kw=150.0,
            start_soc_kwh=start_soc,
        )

        np.testing.assert_allclose(result["soc"], np.full(4, start_soc), atol=ATOL)
        assert abs(result["end_soc"] - start_soc) < ATOL

    def test_offline_pv_dispatch(self) -> None:
        """Offline: export = min(pv, grid_max), curtail = pv - export."""
        pv = np.array([100.0, 200.0, 50.0, 0.0])
        spot = np.array([0.05, 0.03, 0.02, 0.08])
        grid_max = 150.0

        result = dispatch_offline_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            grid_max_kw=grid_max,
            start_soc_kwh=75.0,
        )

        expected_export = np.minimum(pv, grid_max)
        expected_curtail = pv - expected_export
        np.testing.assert_allclose(result["export_pv"], expected_export, atol=ATOL)
        np.testing.assert_allclose(result["curtail"], expected_curtail, atol=ATOL)

    def test_offline_revenue_with_floor(self) -> None:
        """Offline day with EEG floor: revenue = export x max(spot, floor)."""
        pv = np.array([100.0, 100.0])
        spot = np.array([0.02, 0.08])
        floor = 0.05
        grid_max = 200.0

        result = dispatch_offline_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            grid_max_kw=grid_max,
            start_soc_kwh=50.0,
        )

        eff = np.maximum(spot, floor)
        expected_rev = pv * eff  # export = pv since grid_max > pv
        np.testing.assert_allclose(result["revenue"], expected_rev, atol=ATOL)


class TestGreenModeSocDayToDay:
    """SoC day-to-day coupling: end_soc of day 1 = start_soc of day 2."""

    def test_soc_coupling(self) -> None:
        """Run two consecutive days and verify SoC coupling."""
        pv = np.array([200.0, 0.0, 0.0, 0.0])
        spot = np.array([0.01, 0.01, 0.01, 0.10])
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0)
        start_soc = 50.0

        result_day1 = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=start_soc,
        )

        # Day 2 uses end_soc from day 1
        result_day2 = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=result_day1["end_soc"],
        )

        # Verify SoC tracking in day 2 starts from day 1 end SoC
        _assert_soc_tracking_green(result_day2, result_day1["end_soc"])
        _assert_soc_within_bounds(result_day2, bess.soc_min_kwh, bess.soc_max_kwh)


# ============================================================================
# GREY MODE TESTS
# ============================================================================


class TestGreyModeGridCharging:
    """Grey Mode: charge_grid > 0 is possible (grid arbitrage)."""

    def test_grey_allows_grid_charging(self) -> None:
        """In Grey Mode with cheap grid + expensive later, grid charging occurs."""
        pv = np.array([0.0, 0.0, 0.0, 0.0])  # No PV
        spot = np.array([0.01, 0.01, 0.10, 0.10])
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=500.0,
            mode="grey",
            start_soc_kwh=50.0,
            start_soc_green_kwh=50.0,
            start_soc_grey_kwh=0.0,
        )

        # Grey mode should charge from grid at cheap hours
        assert float(np.sum(result["charge_grid"])) > 0.0, (
            "Grey mode should charge from grid at low prices"
        )
        # And discharge at expensive hours
        assert float(np.sum(result["discharge_grey"])) > 0.0, (
            "Grey mode should discharge at high prices"
        )


class TestGreyModeDualChamber:
    """Grey Mode: soc_green + soc_grey <= soc_max."""

    def test_dual_chamber_bounds(self) -> None:
        """Both chambers together must not exceed soc_max."""
        pv = np.array([200.0, 200.0, 0.0, 0.0])
        spot = np.array([0.01, 0.01, 0.10, 0.10])
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=500.0,
            mode="grey",
            start_soc_kwh=40.0,
            start_soc_green_kwh=40.0,
            start_soc_grey_kwh=0.0,
        )

        total_soc = result["soc_green"] + result["soc_grey"]
        assert np.all(total_soc <= bess.soc_max_kwh + ATOL)
        assert np.all(total_soc >= bess.soc_min_kwh - ATOL)
        assert np.all(result["soc_green"] >= -ATOL)
        assert np.all(result["soc_grey"] >= -ATOL)


class TestGreyModeArbitrage:
    """Grey Mode arbitrage: charge cheap, discharge expensive."""

    def test_arbitrage_net_revenue_positive(self) -> None:
        """Grid arbitrage should produce positive net revenue when spread is large."""
        pv = np.array([0.0, 0.0, 0.0, 0.0])  # Pure arbitrage, no PV
        spot = np.array([0.01, 0.01, 0.20, 0.20])
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0, rte=0.90)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=500.0,
            mode="grey",
            start_soc_kwh=20.0,
            start_soc_green_kwh=20.0,
            start_soc_grey_kwh=0.0,
        )

        total_revenue = float(np.sum(result["revenue"]))
        assert total_revenue > 0.0, (
            f"Grey arbitrage should be profitable: revenue = {total_revenue:.4f}"
        )

    def test_arbitrage_numerical(self) -> None:
        """Verify numerical arbitrage result for a simple 2-hour case.

        Setup: 2 hours, no PV, spot = [0.01, 0.10], BESS 100 kW / 200 kWh,
        RTE = 0.90, SoC limits 20-180 kWh, start SoC = 20 (at min).
        Optimal: charge 100 kWh at t=0 (cost 1.00 EUR), discharge 100 kWh at t=1
        (grid output = 90 kWh, revenue = 9.00 EUR). Net = 8.00 EUR.
        """
        pv = np.array([0.0, 0.0])
        spot = np.array([0.01, 0.10])
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0, rte=0.90)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=500.0,
            mode="grey",
            start_soc_kwh=20.0,
            start_soc_green_kwh=20.0,
            start_soc_grey_kwh=0.0,
        )

        # Charge at t=0: 100 kWh from grid
        assert abs(result["charge_grid"][0] - 100.0) < ATOL
        # Discharge at t=1: 100 kWh
        assert abs(result["discharge_grey"][1] - 100.0) < ATOL
        # Net revenue: -100 x 0.01 + 100 x 0.9 x 0.10 = -1.00 + 9.00 = 8.00
        total_rev = float(np.sum(result["revenue"]))
        assert abs(total_rev - 8.00) < ATOL


class TestGreenBlocksGridCharging:
    """Green Mode must produce zero grid charging."""

    def test_no_grid_charging_in_green(self) -> None:
        """Even with huge price spread, Green Mode cannot charge from grid."""
        pv = np.array([0.0, 0.0, 0.0, 0.0])
        spot = np.array([0.001, 0.001, 0.50, 0.50])
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=100.0,
        )

        np.testing.assert_allclose(result["charge_grid"], np.zeros(4), atol=ATOL)
        np.testing.assert_allclose(result["discharge_grey"], np.zeros(4), atol=ATOL)


class TestGreyModeSocDayToDay:
    """Grey Mode: both soc_green and soc_grey carry over between days."""

    def test_grey_soc_coupling(self) -> None:
        """Run two days in Grey Mode; verify both SoC chambers carry over."""
        pv = np.array([100.0, 0.0, 0.0, 0.0])
        spot = np.array([0.01, 0.01, 0.10, 0.10])
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0)

        result_day1 = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=500.0,
            mode="grey",
            start_soc_kwh=50.0,
            start_soc_green_kwh=30.0,
            start_soc_grey_kwh=20.0,
        )

        # Day 2 starts from day 1's end SoC values
        result_day2 = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=500.0,
            mode="grey",
            start_soc_kwh=result_day1["end_soc"],
            start_soc_green_kwh=result_day1["end_soc_green"],
            start_soc_grey_kwh=result_day1["end_soc_grey"],
        )

        # Verify both chambers non-negative
        assert np.all(result_day2["soc_green"] >= -ATOL)
        assert np.all(result_day2["soc_grey"] >= -ATOL)
        # Total SoC within bounds
        total = result_day2["soc_green"] + result_day2["soc_grey"]
        assert np.all(total <= bess.soc_max_kwh + ATOL)
        assert np.all(total >= bess.soc_min_kwh - ATOL)


# ============================================================================
# EDGE CASES
# ============================================================================


class TestEdgeCaseNegativePrices:
    """Negative prices: optimizer should curtail rather than export."""

    def test_negative_prices_curtail(self) -> None:
        """All-negative prices: PV should be curtailed, not exported."""
        pv = np.array([100.0, 100.0, 100.0, 100.0])
        spot = np.array([-0.05, -0.05, -0.05, -0.05])
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=100.0,
        )

        # With negative prices and no floor, exporting loses money -> curtail all
        total_export = float(np.sum(result["export_pv"]))
        total_curtail = float(np.sum(result["curtail"]))
        assert total_curtail > 300.0, (
            f"Should curtail most PV under negative prices, got curtail={total_curtail:.1f}"
        )
        # Revenue should be non-negative (no export at negative prices)
        total_rev = float(np.sum(result["revenue"]))
        assert total_rev >= -ATOL, (
            f"Revenue should be >= 0 with optimal dispatch, got {total_rev:.4f}"
        )

    def test_negative_spot_with_floor_still_exports(self) -> None:
        """With floor > 0 and negative spot, PV should still export at floor price."""
        pv = np.array([100.0, 100.0, 100.0, 100.0])
        spot = np.array([-0.05, -0.05, -0.05, -0.05])
        floor = 0.05  # Positive floor above spot
        bess = _make_bess(power_kw=0.01, capacity_kwh=0.1)  # Minimal BESS

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=0.05,
        )

        # Floor is positive -> effective price = max(-0.05, 0.05) = 0.05 -> export
        total_export = float(np.sum(result["export_pv"]))
        assert abs(total_export - 400.0) < ATOL, (
            f"Should export all PV at floor price, got export={total_export:.1f}"
        )


class TestEdgeCaseZeroPv:
    """Zero PV production (e.g. night): Green stays still, Grey can discharge."""

    def test_green_zero_pv_no_activity(self) -> None:
        """Green Mode with zero PV: no charge, no export, no curtail."""
        pv = np.zeros(4)
        spot = np.array([0.01, 0.01, 0.10, 0.10])
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=100.0,
        )

        np.testing.assert_allclose(result["charge_pv"], np.zeros(4), atol=ATOL)
        np.testing.assert_allclose(result["export_pv"], np.zeros(4), atol=ATOL)
        np.testing.assert_allclose(result["curtail"], np.zeros(4), atol=ATOL)

    def test_grey_zero_pv_can_arbitrage(self) -> None:
        """Grey Mode with zero PV: grid arbitrage is still possible."""
        pv = np.zeros(4)
        spot = np.array([0.01, 0.01, 0.20, 0.20])
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0, rte=0.90)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=500.0,
            mode="grey",
            start_soc_kwh=20.0,
            start_soc_green_kwh=20.0,
            start_soc_grey_kwh=0.0,
        )

        # Grey should charge from grid at cheap hours and discharge at expensive
        assert float(np.sum(result["charge_grid"])) > 0.0
        assert float(np.sum(result["discharge_grey"])) > 0.0


class TestEdgeCaseZeroGridMax:
    """P_grid_max = 0: everything must be curtailed (no export possible)."""

    def test_zero_grid_all_curtailed(self) -> None:
        """With grid_max = 0, all PV is curtailed, no revenue."""
        pv = np.array([100.0, 200.0, 50.0, 0.0])
        spot = np.array([0.05, 0.10, 0.05, 0.10])
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=0.0,
            mode="green",
            start_soc_kwh=100.0,
        )

        # No export possible -> all curtailed
        np.testing.assert_allclose(result["export_pv"], np.zeros(4), atol=ATOL)
        # Revenue should be zero
        total_rev = float(np.sum(result["revenue"]))
        assert abs(total_rev) < ATOL
        # Energy balance must still hold
        _assert_energy_balance(result, pv)


class TestEdgeCaseInvalidMode:
    """Invalid operating mode should raise ValueError."""

    def test_invalid_mode_raises(self) -> None:
        """Passing an invalid mode string raises ValueError."""
        pv = np.array([100.0])
        spot = np.array([0.05])
        bess = _make_bess()

        with pytest.raises(ValueError, match="Unknown operating mode"):
            optimize_day(
                pv_production_kwh=pv,
                spot_prices_eur_per_kwh=spot,
                price_fixed_eur_per_kwh=0.0,
                bess=bess,
                grid_max_kw=100.0,
                mode="invalid",  # type: ignore[arg-type]
                start_soc_kwh=50.0,
            )
