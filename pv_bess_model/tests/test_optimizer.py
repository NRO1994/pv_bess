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
    """Assert grid limit: export_pv + disch_green + disch_grey <= grid_max.

    Note: discharge_green/grey are returned post-RTE by the optimizer,
    so no additional RTE multiplication is needed.
    """
    grid_out = (
        result["export_pv"]
        + result["discharge_green"] + result["discharge_grey"]
    )
    assert np.all(grid_out <= grid_max + ATOL), (
        f"Grid limit exceeded: max = {grid_out.max():.4f} > {grid_max}"
    )


def _assert_soc_tracking_green(
    result: DailyDispatchResult, start_soc: float, rte: float = 0.9
) -> None:
    """Assert SoC tracking from returned (post-RTE) discharge values.

    The optimizer returns discharge_green post-RTE (raw × rte × glf).
    SoC is computed from raw values, so we divide back by rte to recover raw.
    """
    T = len(result["soc"])
    expected_soc = np.empty(T)
    cumulative = start_soc
    for t in range(T):
        # discharge_green is post-RTE; raw = discharge_green / rte
        raw_discharge = result["discharge_green"][t] / rte if rte > 0 else 0.0
        cumulative += result["charge_pv"][t] - raw_discharge
        expected_soc[t] = cumulative
    np.testing.assert_allclose(result["soc"], expected_soc, atol=ATOL)


# ============================================================================
# GREEN MODE TESTS
# ============================================================================


class TestGreenModeReferenceOptimizer4h:
    """Reference 4-hour MILP test with hand-computed results.

    Note: t=1 and t=2 have the same price (10 EUR/MWh), so the MILP may
    distribute charging between them differently than the LP.  Tests check
    aggregate totals, key SoC states, and constraint satisfaction rather
    than exact per-timestep dispatch patterns.
    """

    def test_optimal_dispatch_matches_reference(
        self, reference_optimizer_4h: dict
    ) -> None:
        """Key dispatch invariants must hold (aggregate totals, SoC endpoints)."""
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

        # Total charge must equal sum of reference (680/9 kWh)
        assert abs(np.sum(result["charge_pv"]) - np.sum(ref["expected_charge_pv_kwh"])) < ATOL
        # No curtailment
        np.testing.assert_allclose(
            result["curtail"], ref["expected_curtail_kwh"], atol=ATOL
        )
        # t=0 discharge and t=3 discharge are uniquely determined
        assert abs(result["discharge_green"][0] - ref["expected_discharge_green_kwh"][0]) < ATOL
        assert abs(result["discharge_green"][3] - ref["expected_discharge_green_kwh"][3]) < ATOL
        # SoC at end of t=0 (after discharge) and end of t=3 (min SoC)
        assert abs(result["soc"][0] - ref["expected_soc_kwh"][0]) < ATOL
        assert abs(result["soc"][3] - ref["expected_soc_kwh"][3]) < ATOL
        # SoC before discharge at t=3 must be 120 kWh
        assert abs(result["soc"][2] - 120.0) < ATOL
        # Energy balance
        _assert_energy_balance(result, ref["pv_production_kwh"])

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
        """Total grid export and key timesteps match reference."""
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

        grid_export = result["export_pv"] + result["discharge_green"]
        # t=0 grid limit binding (150 kWh) and t=3 discharge (90 kWh)
        assert abs(grid_export[0] - ref["expected_grid_export_kwh"][0]) < ATOL
        assert abs(grid_export[3] - ref["expected_grid_export_kwh"][3]) < ATOL
        # Total grid export must match
        assert abs(np.sum(grid_export) - np.sum(ref["expected_grid_export_kwh"])) < ATOL


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
        _assert_soc_tracking_green(result, ref["start_soc_kwh"], rte=ref["rte"])


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
        _assert_soc_tracking_green(result_day2, result_day1["end_soc"], rte=0.9)
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
            start_soc_green_kwh=0.0,
            start_soc_grey_kwh=20.0,
        )

        # Charge at t=0: 100 kWh from grid
        assert abs(result["charge_grid"][0] - 100.0) < ATOL
        # Discharge at t=1: 100 kWh raw, returned as post-RTE = 90 kWh
        assert abs(result["discharge_grey"][1] - 90.0) < ATOL
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


class TestNoSimultaneousChargeDischarge:
    """MILP binary variable must prevent charge and discharge in same timestep."""

    def _assert_no_simultaneous(self, result: DailyDispatchResult) -> None:
        """Assert no timestep has both charge > 0 and discharge > 0."""
        T = len(result["charge_pv"])
        for t in range(T):
            total_charge = result["charge_pv"][t] + result["charge_grid"][t]
            total_discharge = result["discharge_green"][t] + result["discharge_grey"][t]
            assert not (total_charge > ATOL and total_discharge > ATOL), (
                f"t={t}: simultaneous charge={total_charge:.4f} and "
                f"discharge={total_discharge:.4f}"
            )

    def test_green_positive_prices(self) -> None:
        """Green mode with positive prices: no simultaneous charge/discharge."""
        pv = np.array([200.0, 200.0, 50.0, 0.0])
        spot = np.array([0.03, 0.01, 0.01, 0.08])
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=150.0,
            mode="green",
            start_soc_kwh=100.0,
        )
        self._assert_no_simultaneous(result)

    def test_green_floor_price(self) -> None:
        """Green mode with floor price (degenerate case): no simultaneous."""
        pv = np.array([100.0, 100.0, 100.0, 0.0])
        spot = np.array([0.02, 0.02, 0.02, 0.08])
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.05,  # floor above spot
            bess=bess,
            grid_max_kw=150.0,
            mode="green",
            start_soc_kwh=100.0,
        )
        self._assert_no_simultaneous(result)

    def test_green_negative_prices(self) -> None:
        """Green mode with negative prices: no simultaneous."""
        pv = np.array([100.0, 100.0, 50.0, 0.0])
        spot = np.array([-0.02, 0.01, -0.01, 0.10])
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=150.0,
            mode="green",
            start_soc_kwh=100.0,
        )
        self._assert_no_simultaneous(result)

    def test_grey_arbitrage(self) -> None:
        """Grey mode arbitrage: no simultaneous charge/discharge."""
        pv = np.array([50.0, 50.0, 0.0, 0.0])
        spot = np.array([0.01, 0.01, 0.08, 0.10])
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=500.0,
            mode="grey",
            start_soc_kwh=100.0,
        )
        self._assert_no_simultaneous(result)

    def test_grey_negative_prices(self) -> None:
        """Grey mode with negative prices: no simultaneous."""
        pv = np.array([100.0, 100.0, 0.0, 0.0])
        spot = np.array([-0.03, 0.01, -0.02, 0.10])
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=500.0,
            mode="grey",
            start_soc_kwh=100.0,
        )
        self._assert_no_simultaneous(result)

    def test_collar_mixed_prices(self) -> None:
        """Collar PPA with mixed prices: no simultaneous."""
        pv = np.array([150.0, 150.0, 50.0, 0.0])
        spot = np.array([0.02, 0.04, 0.07, 0.10])
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.03,  # floor
            price_cap_eur_per_kwh=0.08,    # cap
            goo_premium_eur_per_kwh=0.005,
            bess=bess,
            grid_max_kw=200.0,
            mode="green",
            start_soc_kwh=100.0,
        )
        self._assert_no_simultaneous(result)


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


# ============================================================================
# PPA COLLAR (CAP PRICE) TESTS
# ============================================================================


class TestCollarPriceInOptimizer:
    """Tests for PPA Collar cap price integration in the LP optimizer."""

    def test_collar_spot_below_floor_revenue_at_floor(self) -> None:
        """When spot < floor < cap, effective = floor + goo."""
        pv = np.array([100.0, 100.0, 100.0, 100.0])
        spot = np.array([0.02, 0.02, 0.02, 0.02])
        floor = 0.05
        cap = 0.10
        goo = 0.003
        bess = _make_bess(power_kw=0.01, capacity_kwh=0.1)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=0.05,
            goo_premium_eur_per_kwh=goo,
            price_cap_eur_per_kwh=cap,
        )

        # effective = max(0.02, 0.05) = 0.05, cap not binding, + goo = 0.053
        expected_revenue = float(np.sum(pv)) * (floor + goo)
        actual_revenue = float(np.sum(result["revenue"]))
        assert abs(actual_revenue - expected_revenue) < 0.01

    def test_collar_spot_between_floor_and_cap_revenue_at_spot(self) -> None:
        """When floor < spot < cap, effective = spot + goo."""
        pv = np.array([100.0, 100.0, 100.0, 100.0])
        spot = np.array([0.07, 0.07, 0.07, 0.07])
        floor = 0.05
        cap = 0.10
        goo = 0.003
        bess = _make_bess(power_kw=0.01, capacity_kwh=0.1)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=0.05,
            goo_premium_eur_per_kwh=goo,
            price_cap_eur_per_kwh=cap,
        )

        # effective = max(0.07, 0.05) = 0.07, min(0.07, 0.10) = 0.07, + goo = 0.073
        expected_revenue = float(np.sum(pv)) * (0.07 + goo)
        actual_revenue = float(np.sum(result["revenue"]))
        assert abs(actual_revenue - expected_revenue) < 0.01

    def test_collar_spot_above_cap_revenue_at_cap(self) -> None:
        """When spot > cap > floor, effective = cap + goo (capped)."""
        pv = np.array([100.0, 100.0, 100.0, 100.0])
        spot = np.array([0.15, 0.15, 0.15, 0.15])
        floor = 0.05
        cap = 0.10
        goo = 0.003
        bess = _make_bess(power_kw=0.01, capacity_kwh=0.1)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=0.05,
            goo_premium_eur_per_kwh=goo,
            price_cap_eur_per_kwh=cap,
        )

        # effective = max(0.15, 0.05) = 0.15, min(0.15, 0.10) = 0.10, + goo = 0.103
        expected_revenue = float(np.sum(pv)) * (cap + goo)
        actual_revenue = float(np.sum(result["revenue"]))
        assert abs(actual_revenue - expected_revenue) < 0.01

    def test_collar_mixed_prices_three_regions(self) -> None:
        """Mixed spot prices spanning all three regions (below floor, within, above cap)."""
        pv = np.array([100.0, 100.0, 100.0])
        spot = np.array([0.02, 0.07, 0.15])
        floor = 0.05
        cap = 0.10
        goo = 0.003
        bess = _make_bess(power_kw=0.01, capacity_kwh=0.1)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=0.05,
            goo_premium_eur_per_kwh=goo,
            price_cap_eur_per_kwh=cap,
        )

        # clip(spot, 0.05, 0.10) = [0.05, 0.07, 0.10], + goo = [0.053, 0.073, 0.103]
        eff = np.clip(spot, floor, cap) + goo
        expected_revenue = float(np.sum(pv * eff))
        actual_revenue = float(np.sum(result["revenue"]))
        assert abs(actual_revenue - expected_revenue) < 0.01

    def test_collar_after_expiry_no_cap_no_floor(self) -> None:
        """After PPA expiry (cap=0, floor=0), effective = spot (no goo)."""
        pv = np.array([100.0, 100.0])
        spot = np.array([0.08, 0.12])
        bess = _make_bess(power_kw=0.01, capacity_kwh=0.1)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=0.05,
            goo_premium_eur_per_kwh=0.0,
            price_cap_eur_per_kwh=0.0,
        )

        expected_revenue = float(np.sum(pv * spot))
        actual_revenue = float(np.sum(result["revenue"]))
        assert abs(actual_revenue - expected_revenue) < 0.01

    def test_collar_cap_limits_revenue_compared_to_floor_only(self) -> None:
        """With cap active and spot > cap, revenue is less than floor-only case."""
        pv = np.array([100.0, 100.0, 100.0, 100.0])
        spot = np.array([0.15, 0.15, 0.15, 0.15])
        floor = 0.05
        cap = 0.10
        bess = _make_bess(power_kw=0.01, capacity_kwh=0.1)

        result_floor_only = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=0.05,
            price_cap_eur_per_kwh=0.0,
        )
        result_collar = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=0.05,
            price_cap_eur_per_kwh=cap,
        )

        rev_floor = float(np.sum(result_floor_only["revenue"]))
        rev_collar = float(np.sum(result_collar["revenue"]))
        assert rev_collar < rev_floor, (
            f"Collar should reduce revenue when cap binds: "
            f"collar={rev_collar:.4f} >= floor_only={rev_floor:.4f}"
        )

    def test_collar_offline_day_applies_cap(self) -> None:
        """dispatch_offline_day also applies cap price correctly."""
        pv = np.array([100.0, 100.0])
        spot = np.array([0.15, 0.03])
        floor = 0.05
        cap = 0.10
        goo = 0.003

        result = dispatch_offline_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            grid_max_kw=500.0,
            start_soc_kwh=50.0,
            goo_premium_eur_per_kwh=goo,
            price_cap_eur_per_kwh=cap,
        )

        # t=0: clip(0.15, 0.05, 0.10) = 0.10, + goo = 0.103
        # t=1: clip(0.03, 0.05, 0.10) = 0.05, + goo = 0.053
        expected_eff = np.array([0.103, 0.053])
        expected_rev = pv * expected_eff
        np.testing.assert_allclose(result["revenue"], expected_rev, atol=ATOL)


# ============================================================================
# EEG FLOOR WITHOUT GOO IN OPTIMIZER
# ============================================================================


class TestEegFloorNoGooInOptimizer:
    """Verify that EEG floor price is applied WITHOUT GoO premium.

    The EEG marketing type does not use GoO.  When goo_premium=0.0 and
    cap_price=0.0, the effective price should be max(spot, floor).
    """

    def test_eeg_floor_no_goo_numerical(self) -> None:
        """With EEG floor and no GoO, effective = max(spot, floor)."""
        pv = np.array([100.0, 100.0, 100.0, 100.0])
        spot = np.array([0.02, 0.08, 0.0735, 0.10])
        floor = 0.0735
        bess = _make_bess(power_kw=0.01, capacity_kwh=0.1)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=0.05,
            goo_premium_eur_per_kwh=0.0,
            price_cap_eur_per_kwh=0.0,
        )

        eff = np.maximum(spot, floor)
        expected_revenue = float(np.sum(pv * eff))
        actual_revenue = float(np.sum(result["revenue"]))
        assert abs(actual_revenue - expected_revenue) < 0.01

    def test_eeg_floor_spot_all_above(self) -> None:
        """When all spot prices exceed EEG floor, floor has no effect."""
        pv = np.array([100.0, 100.0])
        spot = np.array([0.10, 0.12])
        floor = 0.0735
        bess = _make_bess(power_kw=0.01, capacity_kwh=0.1)

        result_with_floor = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=0.05,
        )
        result_no_floor = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=0.05,
        )

        rev_with = float(np.sum(result_with_floor["revenue"]))
        rev_without = float(np.sum(result_no_floor["revenue"]))
        assert abs(rev_with - rev_without) < 0.01


# ---------------------------------------------------------------------------
# Grid loss factor tests
# ---------------------------------------------------------------------------


class TestGridLossFactor:
    """Tests for the grid_loss_factor parameter."""

    def test_grid_loss_factor_one_matches_default(self):
        """grid_loss_factor=1.0 produces identical results to default (no loss)."""
        pv = np.array([100.0, 200.0, 50.0, 0.0])
        spot = np.array([0.05, 0.10, 0.08, 0.06])
        bess = _make_bess(power_kw=50.0, capacity_kwh=100.0)

        result_default = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=300.0,
            mode="green",
            start_soc_kwh=50.0,
        )
        result_one = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=300.0,
            mode="green",
            start_soc_kwh=50.0,
            grid_loss_factor=1.0,
        )

        np.testing.assert_allclose(
            result_default["revenue"], result_one["revenue"], atol=ATOL,
        )
        np.testing.assert_allclose(
            result_default["export_pv"], result_one["export_pv"], atol=ATOL,
        )

    def test_grid_loss_factor_reduces_green_revenue(self):
        """Green revenue (PV export + green BESS discharge) is reduced by glf."""
        pv = np.array([200.0, 0.0])
        spot = np.array([0.10, 0.10])
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0)
        grid_max = 300.0

        result_no_loss = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=grid_max,
            mode="green",
            start_soc_kwh=100.0,
            grid_loss_factor=1.0,
        )
        glf = 0.86
        result_with_loss = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=grid_max,
            mode="green",
            start_soc_kwh=100.0,
            grid_loss_factor=glf,
        )

        rev_no_loss = float(np.sum(result_no_loss["revenue"]))
        rev_with_loss = float(np.sum(result_with_loss["revenue"]))
        # Revenue should be approximately reduced by glf
        assert rev_with_loss < rev_no_loss
        assert rev_with_loss > 0.0

    def test_grid_loss_factor_no_effect_on_grey_revenue(self):
        """Grey BESS discharge revenue is NOT affected by grid_loss_factor."""
        pv = np.array([0.0, 0.0])
        spot = np.array([0.02, 0.10])
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0, rte=1.0)

        # Grey mode: charge from grid at low price, discharge at high price
        result_no_loss = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=300.0,
            mode="grey",
            start_soc_kwh=20.0,
            start_soc_green_kwh=0.0,
            start_soc_grey_kwh=20.0,
            grid_loss_factor=1.0,
        )
        result_with_loss = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=300.0,
            mode="grey",
            start_soc_kwh=20.0,
            start_soc_green_kwh=0.0,
            start_soc_grey_kwh=20.0,
            grid_loss_factor=0.86,
        )

        # Grey revenue should be identical (no PV, no green energy)
        grey_rev_no_loss = float(np.sum(
            result_no_loss["discharge_grey"] * 1.0 * spot
            - result_no_loss["charge_grid"] * spot
        ))
        grey_rev_with_loss = float(np.sum(
            result_with_loss["discharge_grey"] * 1.0 * spot
            - result_with_loss["charge_grid"] * spot
        ))
        assert abs(grey_rev_no_loss - grey_rev_with_loss) < ATOL

    def test_grid_loss_factor_in_grid_constraint(self):
        """Grid constraint: export_pv × glf + discharge × RTE ≤ grid_max."""
        glf = 0.80
        rte = 0.90
        grid_max = 100.0
        pv = np.array([200.0, 200.0, 200.0, 200.0])
        spot = np.array([0.10, 0.10, 0.10, 0.10])
        bess = _make_bess(power_kw=50.0, capacity_kwh=100.0, rte=rte)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=grid_max,
            mode="green",
            start_soc_kwh=50.0,
            grid_loss_factor=glf,
        )

        # Check grid constraint: export_pv × glf + discharge_green × RTE ≤ grid_max
        grid_out = result["export_pv"] * glf + result["discharge_green"] * rte
        assert np.all(grid_out <= grid_max + ATOL)

    def test_offline_day_grid_loss_factor(self):
        """dispatch_offline_day applies grid_loss_factor to revenue."""
        pv = np.array([100.0, 200.0])
        spot = np.array([0.10, 0.10])
        glf = 0.86

        result = dispatch_offline_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            grid_max_kw=300.0,
            start_soc_kwh=50.0,
            grid_loss_factor=glf,
        )

        expected_revenue = pv * glf * spot
        np.testing.assert_allclose(result["revenue"], expected_revenue, atol=ATOL)
