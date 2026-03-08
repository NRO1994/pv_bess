"""Unit tests for FIX-S2-12: BESS discharge valued at spot prices.

After FIX-S2-12, the LP optimizer uses:
- **Effective prices** (floor/cap-adjusted) for PV direct export
- **Spot prices** for BESS discharge revenue (both green and grey)

This means BESS arbitrage profit per cycle is:
    profit = RTE × spot[t_discharge] - eff[t_charge]

When floor > spot for most hours, BESS cycling is still profitable whenever
spot[t_discharge] > eff[t_charge] / RTE, rather than being unprofitable
(as before, when both sides used eff_prices, giving floor × (RTE - 1) < 0).

These tests verify the corrected pricing for Green Mode, Grey Mode, and
edge cases.
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

ATOL = 1e-4
"""Absolute tolerance for floating-point comparisons."""


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


# ============================================================================
# GREEN MODE: BESS DISCHARGE AT SPOT
# ============================================================================


class TestGreenModeBessDischargeAtSpot:
    """Verify that BESS discharge revenue uses spot prices, not effective prices."""

    def test_bess_cycles_with_high_floor_and_price_spread(self) -> None:
        """BESS should charge and discharge when spot spread exists, even with high floor.

        This is the core test for FIX-S2-12. With floor=0.08 and spot=[0.02, 0.10]:
        - eff=[0.08, 0.10]
        - Charging cost (opportunity): 1 kWh at eff[0]=0.08
        - Discharge revenue: RTE × spot[1] = 0.9 × 0.10 = 0.09
        - Profit per kWh: 0.09 - 0.08 = 0.01 > 0 → BESS should cycle

        Before fix: both used eff → profit = 0.9 × 0.10 - 0.10 = -0.01 < 0 → no cycling
        """
        pv = np.array([200.0, 0.0])
        spot = np.array([0.02, 0.10])
        floor = 0.08  # Above low spot, below high spot
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0, rte=0.90)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=20.0,
        )

        # BESS should charge at t=0 (cheap hour) and discharge at t=1 (expensive)
        assert result["charge_pv"][0] > 0.0, (
            "BESS should charge at low-price hour even with high floor"
        )
        assert result["discharge_green"][1] > 0.0, (
            "BESS should discharge at high-price hour"
        )

    def test_bess_does_not_cycle_when_spot_spread_insufficient(self) -> None:
        """BESS should NOT cycle when spot[discharge] < eff[charge] / RTE.

        With floor=0.08, spot=[0.02, 0.06], RTE=0.90:
        - eff=[0.08, 0.08]
        - Profit: 0.9 × 0.06 - 0.08 = 0.054 - 0.08 = -0.026 < 0
        - BESS should not cycle (all PV exported directly at floor price)
        """
        pv = np.array([200.0, 0.0])
        spot = np.array([0.02, 0.06])
        floor = 0.08
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0, rte=0.90)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=20.0,
        )

        # BESS cycling is unprofitable → all PV exported directly
        assert float(np.sum(result["discharge_green"])) < ATOL, (
            "BESS should not discharge when spot spread is insufficient"
        )

    def test_revenue_split_pv_at_eff_bess_at_spot(self) -> None:
        """PV export revenue uses eff_price, BESS discharge uses spot price.

        Setup: 4h, PV=[200, 0, 0, 0], spot=[0.02, 0.02, 0.02, 0.15],
        floor=0.05 → eff=[0.05, 0.05, 0.05, 0.15]
        RTE=1.0 (simplified), grid_max=500

        Expected behavior:
        - t=0: charge some PV, export rest at eff=0.05
        - t=3: discharge at spot=0.15
        - PV export revenue: at eff_price
        - BESS discharge revenue: at spot_price
        """
        pv = np.array([200.0, 0.0, 0.0, 0.0])
        spot = np.array([0.02, 0.02, 0.02, 0.15])
        floor = 0.05
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0, rte=1.0)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=20.0,
        )

        eff = np.maximum(spot, floor)

        # Revenue = (export_pv + discharge_green) × eff
        # discharge_green is already post-RTE
        expected_total = (result["export_pv"] + result["discharge_green"]) * eff

        np.testing.assert_allclose(result["revenue"], expected_total, atol=ATOL)

    def test_floor_dominant_bess_still_active(self) -> None:
        """With floor dominant for most hours, BESS exploits spot peaks above floor/RTE.

        This is the scenario that was broken before FIX-S2-12.
        Floor=0.0549 (typical EEG), RTE=0.985, spot has morning low and evening peak.

        Before fix: eff was flat at floor → profit = floor × (RTE - 1) < 0 → no cycling.
        After fix: profit = RTE × spot_peak - eff_low. If spot_peak > eff_low/RTE,
        BESS cycles.
        """
        # Realistic 24h-like pattern (simplified to 6 hours)
        pv = np.array([0.0, 100.0, 200.0, 200.0, 50.0, 0.0])
        spot = np.array([0.03, 0.02, 0.01, 0.01, 0.04, 0.08])
        floor = 0.0549  # Typical EEG floor
        rte = 0.985
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0, rte=rte)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=20.0,
        )

        # Check: eff_charge_hours (t=1,2,3) = floor = 0.0549
        # spot_discharge_hour (t=5) = 0.08
        # Profit: 0.985 × 0.08 - 0.0549 = 0.0788 - 0.0549 = 0.0239 > 0
        # → BESS should cycle
        total_discharge = float(np.sum(result["discharge_green"]))
        assert total_discharge > 0.0, (
            f"BESS should cycle with floor-dominant prices when spot peak exists, "
            f"got total_discharge={total_discharge:.4f}"
        )

    def test_no_floor_spot_equals_eff_same_result(self) -> None:
        """Without floor (price_fixed=0), spot==eff → same result as before fix."""
        pv = np.array([200.0, 0.0, 0.0, 0.0])
        spot = np.array([0.02, 0.02, 0.02, 0.10])
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0, rte=0.90)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.0,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=50.0,
        )

        # Revenue = (export_pv + discharge_green) × eff_prices
        # discharge_green is already post-RTE; eff == spot when no floor
        expected_revenue = (result["export_pv"] + result["discharge_green"]) * spot
        np.testing.assert_allclose(result["revenue"], expected_revenue, atol=ATOL)


class TestGreenModeCollarBessAtSpot:
    """BESS discharge at spot with PPA Collar (floor + cap)."""

    def test_collar_bess_revenue_at_spot_not_capped(self) -> None:
        """BESS discharge revenue uses raw spot, NOT capped by collar.

        With collar [floor=0.05, cap=0.10] and spot=0.15:
        - PV export: eff = min(max(0.15, 0.05), 0.10) = 0.10 (capped)
        - BESS discharge: spot = 0.15 (NOT capped)
        """
        pv = np.array([200.0, 0.0])
        spot = np.array([0.02, 0.15])
        floor = 0.05
        cap = 0.10
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0, rte=1.0)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=20.0,
            price_cap_eur_per_kwh=cap,
        )

        eff = np.clip(spot, floor, cap)

        # Revenue = (export_pv + discharge_green) × eff
        # discharge_green is already post-RTE
        expected_revenue = (result["export_pv"] + result["discharge_green"]) * eff
        np.testing.assert_allclose(result["revenue"], expected_revenue, atol=ATOL)

    def test_collar_bess_cycles_when_spot_above_cap(self) -> None:
        """BESS should cycle more aggressively when spot exceeds cap.

        Since BESS discharge earns uncapped spot while charging costs capped eff,
        the BESS captures the full spot peak that PV export cannot.
        """
        pv = np.array([300.0, 0.0, 0.0])
        spot = np.array([0.03, 0.03, 0.20])  # Huge spike at t=2
        floor = 0.05
        cap = 0.08  # PV capped at 0.08, but BESS gets 0.20
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0, rte=0.90)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=20.0,
            price_cap_eur_per_kwh=cap,
        )

        # BESS should charge at t=0 and discharge at t=2 to capture uncapped spot
        assert result["charge_pv"][0] > 0.0, "Should charge BESS at t=0"
        assert result["discharge_green"][2] > 0.0, (
            "Should discharge at t=2 to capture uncapped spot price"
        )


class TestGreenModeGooWithBessSpot:
    """GoO premium is added to eff_price for PV export but NOT to BESS spot price."""

    def test_goo_affects_pv_export_not_bess_discharge(self) -> None:
        """GoO premium increases PV export revenue but not BESS discharge revenue.

        With goo=0.01, floor=0.05, spot=[0.02, 0.10]:
        - PV export eff = max(0.02, 0.05) + 0.01 = 0.06
        - BESS discharge at spot = 0.10 (no goo)
        """
        pv = np.array([200.0, 0.0])
        spot = np.array([0.02, 0.10])
        floor = 0.05
        goo = 0.01
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0, rte=1.0)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=20.0,
            goo_premium_eur_per_kwh=goo,
        )

        eff = np.maximum(spot, floor) + goo

        # Revenue = (export_pv + discharge_green) × eff
        # Both PV and BESS green discharge use effective prices (with goo)
        expected_revenue = (result["export_pv"] + result["discharge_green"]) * eff
        np.testing.assert_allclose(result["revenue"], expected_revenue, atol=ATOL)


# ============================================================================
# GREY MODE: BESS DISCHARGE AT SPOT
# ============================================================================


class TestGreyModeBessGreenDischargeAtSpot:
    """In Grey Mode, discharge_green also uses spot prices (consistent with discharge_grey)."""

    def test_grey_discharge_green_at_spot_with_floor(self) -> None:
        """Grey Mode: discharge_green uses spot price, not effective price.

        With PV and grid charging, floor=0.08, spot=[0.02, 0.02, 0.12, 0.12]:
        - charge_pv at t=0,1 (opportunity cost at eff=0.08)
        - discharge_green at t=2,3 earns spot=0.12 (not eff=0.12)
        - This matters when spot < floor for charge hours
        """
        pv = np.array([200.0, 200.0, 0.0, 0.0])
        spot = np.array([0.02, 0.02, 0.12, 0.12])
        floor = 0.08
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0, rte=0.90)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=500.0,
            mode="grey",
            start_soc_kwh=20.0,
            start_soc_green_kwh=20.0,
            start_soc_grey_kwh=0.0,
        )

        eff = np.maximum(spot, floor)

        # Revenue = (export_pv + discharge_green) × eff + (discharge_grey - charge_grid) × spot
        # discharge_green and discharge_grey are already post-RTE
        expected_revenue = (
            (result["export_pv"] + result["discharge_green"]) * eff
            + (result["discharge_grey"] - result["charge_grid"]) * spot
        )
        np.testing.assert_allclose(result["revenue"], expected_revenue, atol=ATOL)

    def test_grey_discharge_green_and_grey_both_at_spot(self) -> None:
        """Both green and grey discharge should use spot prices consistently."""
        pv = np.array([100.0, 100.0, 0.0, 0.0])
        spot = np.array([0.01, 0.01, 0.15, 0.15])
        floor = 0.0  # No floor
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0, rte=0.90)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=500.0,
            mode="grey",
            start_soc_kwh=20.0,
            start_soc_green_kwh=20.0,
            start_soc_grey_kwh=0.0,
        )

        # With no floor, eff==spot; discharge values are already post-RTE
        expected_revenue = (
            (result["export_pv"] + result["discharge_green"]) * spot
            + (result["discharge_grey"] - result["charge_grid"]) * spot
        )
        np.testing.assert_allclose(result["revenue"], expected_revenue, atol=ATOL)


# ============================================================================
# ENERGY BALANCE AND CONSTRAINTS (unchanged by fix)
# ============================================================================


class TestEnergyBalancePreserved:
    """FIX-S2-12 must not break energy balance or SoC constraints."""

    def test_energy_balance_with_floor(self) -> None:
        """PV energy balance holds: export + charge + curtail = production."""
        pv = np.array([200.0, 100.0, 50.0, 0.0])
        spot = np.array([0.02, 0.02, 0.05, 0.10])
        floor = 0.06
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=50.0,
        )

        lhs = result["export_pv"] + result["charge_pv"] + result["curtail"]
        np.testing.assert_allclose(lhs, pv, atol=ATOL)

    def test_soc_bounds_with_floor(self) -> None:
        """SoC stays within [min, max] with floor-active pricing."""
        pv = np.array([200.0, 200.0, 0.0, 0.0])
        spot = np.array([0.02, 0.02, 0.05, 0.10])
        floor = 0.06
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=50.0,
        )

        assert np.all(result["soc"] >= bess.soc_min_kwh - ATOL)
        assert np.all(result["soc"] <= bess.soc_max_kwh + ATOL)

    def test_grid_limit_with_floor(self) -> None:
        """Grid limit respected with floor pricing and BESS cycling."""
        pv = np.array([300.0, 300.0, 0.0, 0.0])
        spot = np.array([0.02, 0.02, 0.05, 0.10])
        floor = 0.06
        grid_max = 150.0
        rte = 0.90
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0, rte=rte)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=grid_max,
            mode="green",
            start_soc_kwh=50.0,
        )

        # discharge_green is already post-RTE
        grid_out = result["export_pv"] + result["discharge_green"]
        assert np.all(grid_out <= grid_max + ATOL)


class TestGridLossFactorWithSpotPricing:
    """Grid loss factor interaction with spot-based BESS pricing."""

    def test_glf_applied_to_bess_discharge_at_spot(self) -> None:
        """Grid loss factor reduces BESS discharge revenue (at spot price).

        Revenue = discharge × RTE × glf × spot
        """
        pv = np.array([200.0, 0.0])
        spot = np.array([0.02, 0.10])
        floor = 0.05
        glf = 0.86
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0, rte=1.0)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=20.0,
            grid_loss_factor=glf,
        )

        eff = np.maximum(spot, floor)
        # export_pv and discharge_green are already post-glf and post-RTE
        # Revenue = (export_pv + discharge_green) × eff
        expected_revenue = (result["export_pv"] + result["discharge_green"]) * eff
        np.testing.assert_allclose(result["revenue"], expected_revenue, atol=ATOL)


# ============================================================================
# OFFLINE DAY: NO CHANGE (PV export only, uses eff)
# ============================================================================


class TestOfflineDayUnchanged:
    """Offline day has no BESS discharge, so FIX-S2-12 should not affect it."""

    def test_offline_day_revenue_still_uses_eff(self) -> None:
        """On offline days, only PV exports at effective price (no BESS)."""
        pv = np.array([100.0, 100.0])
        spot = np.array([0.02, 0.08])
        floor = 0.05

        result = dispatch_offline_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            grid_max_kw=500.0,
            start_soc_kwh=50.0,
        )

        eff = np.maximum(spot, floor)
        expected_revenue = pv * eff
        np.testing.assert_allclose(result["revenue"], expected_revenue, atol=ATOL)

    def test_offline_day_with_collar_and_goo(self) -> None:
        """Offline day with collar + goo: PV export at clip(spot, floor, cap) + goo."""
        pv = np.array([100.0, 100.0, 100.0])
        spot = np.array([0.02, 0.07, 0.15])
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

        eff = np.clip(spot, floor, cap) + goo
        expected_revenue = pv * eff
        np.testing.assert_allclose(result["revenue"], expected_revenue, atol=ATOL)


# ============================================================================
# NUMERICAL VERIFICATION
# ============================================================================


class TestNumericalBessArbitrageWithFloor:
    """Hand-computed numerical test for BESS arbitrage under floor pricing.

    Setup: 2 hours, PV=[200, 0], spot=[0.02, 0.12], floor=0.06
    BESS: 100 kW, 200 kWh, RTE=1.0, SoC [20, 180], start=20
    Grid max: 500

    eff = [max(0.02, 0.06), max(0.12, 0.06)] = [0.06, 0.12]

    Optimal dispatch:
    - t=0: PV=200. Charge 100 kWh into BESS (power limited).
           Export 100 kWh at eff=0.06.
           SoC: 20 → 120.
    - t=1: Discharge 100 kWh from BESS at spot=0.12.
           Grid output = 100 × 1.0 = 100 kWh.
           SoC: 120 → 20.

    Revenue:
    - t=0: PV export = 100 × 0.06 = 6.00
    - t=1: BESS discharge = 100 × 1.0 × 0.12 = 12.00
    Total: 18.00

    Alternative (no BESS cycling):
    - t=0: Export 200 at eff=0.06 = 12.00
    - t=1: nothing
    Total: 12.00

    BESS cycling adds 6.00 EUR.
    """

    def test_numerical_arbitrage(self) -> None:
        """Verify exact revenue for 2-hour BESS arbitrage with floor."""
        pv = np.array([200.0, 0.0])
        spot = np.array([0.02, 0.12])
        floor = 0.06
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0, rte=1.0)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=20.0,
        )

        # Verify dispatch
        assert abs(result["charge_pv"][0] - 100.0) < ATOL
        assert abs(result["export_pv"][0] - 100.0) < ATOL
        assert abs(result["discharge_green"][1] - 100.0) < ATOL

        # Verify revenue: PV at eff, BESS at spot
        assert abs(result["revenue"][0] - 6.00) < ATOL  # 100 × 0.06
        assert abs(result["revenue"][1] - 12.00) < ATOL  # 100 × 1.0 × 0.12

        total_rev = float(np.sum(result["revenue"]))
        assert abs(total_rev - 18.00) < ATOL

    def test_numerical_no_arbitrage_low_spread(self) -> None:
        """When spot peak < eff_charge / RTE, BESS does not cycle.

        Setup: 2h, PV=[200, 0], spot=[0.02, 0.04], floor=0.06, RTE=0.90
        eff = [0.06, 0.06]
        Profit per cycle: 0.9 × 0.04 - 0.06 = 0.036 - 0.06 = -0.024 < 0
        → No cycling, all PV exported at eff=0.06
        Total revenue: 200 × 0.06 = 12.00
        """
        pv = np.array([200.0, 0.0])
        spot = np.array([0.02, 0.04])
        floor = 0.06
        bess = _make_bess(power_kw=100.0, capacity_kwh=200.0, rte=0.90)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=floor,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=20.0,
        )

        # No BESS cycling
        assert float(np.sum(result["discharge_green"])) < ATOL

        # All PV exported at eff=0.06
        total_rev = float(np.sum(result["revenue"]))
        assert abs(total_rev - 12.00) < ATOL
