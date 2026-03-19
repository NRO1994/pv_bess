"""Unit tests for the baseload PPA integration in the daily LP optimizer.

Tests verify that the LP correctly:
- Adds shortfall variables when baseload_kw > 0
- Penalises shortfall at the effective PPA price (max(spot, ppa_price) + goo)
- Dispatches the BESS to reduce shortfall during low-PV hours
- Returns correct shortfall arrays in the result
- Has zero shortfall when no baseload PPA is active
- Works for both Green and Grey modes

All prices passed to the optimizer are in EUR/kWh.
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


# ---------------------------------------------------------------------------
# Tests: Baseload shortfall tracking
# ---------------------------------------------------------------------------


class TestBaseloadShortfallGreenMode:
    """Test shortfall tracking in Green Mode with baseload PPA."""

    def test_no_baseload_shortfall_is_zero(self) -> None:
        """Without baseload PPA, shortfall array must be all zeros."""
        T = 4
        pv = np.array([50.0, 100.0, 100.0, 50.0])
        spot = np.array([0.05, 0.04, 0.06, 0.08])
        bess = _make_bess()

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.07,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=100.0,
            baseload_mw=0.0,
        )

        np.testing.assert_array_equal(result["shortfall"], np.zeros(T))

    def test_shortfall_when_pv_below_baseload(self) -> None:
        """When PV alone cannot meet baseload and BESS is empty, shortfall > 0."""
        T = 4
        # PV only produces in hours 1-2, baseload is 80 kWh/h
        pv = np.array([0.0, 150.0, 150.0, 0.0])
        spot = np.array([0.05, 0.04, 0.04, 0.05])
        bess = _make_bess(power_kw=50.0, capacity_kwh=100.0)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.07,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=10.0,  # min SoC
            baseload_mw=0.08,  # 80 kWh/h (0.08 MW)
        )

        # In hours 0 and 3, PV=0 and BESS may not have enough → shortfall expected
        # Shortfall should be >= 0 everywhere
        assert np.all(result["shortfall"] >= -ATOL)
        # There should be some shortfall in hour 0 (PV=0, BESS starts near empty)
        assert result["shortfall"][0] > 0, "Expected shortfall in hour 0 (no PV, near-empty BESS)"

    def test_no_shortfall_when_pv_exceeds_baseload(self) -> None:
        """When PV production >= baseload every hour, shortfall must be zero."""
        T = 4
        pv = np.array([200.0, 200.0, 200.0, 200.0])
        spot = np.array([0.05, 0.04, 0.06, 0.08])
        bess = _make_bess()

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.07,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=100.0,
            baseload_mw=0.1,  # 100 kWh/h (0.1 MW)
        )

        np.testing.assert_allclose(result["shortfall"], 0.0, atol=ATOL)

    def test_bess_reduces_shortfall(self) -> None:
        """BESS should discharge to reduce shortfall in low-PV hours."""
        T = 4
        # PV high in hours 0-1, zero in hours 2-3
        pv = np.array([200.0, 200.0, 0.0, 0.0])
        spot = np.array([0.04, 0.04, 0.06, 0.06])
        bess = _make_bess(power_kw=80.0, capacity_kwh=200.0, rte=1.0)
        baseload_kwh = 50.0  # 50 kWh/h → 0.05 MW

        # With BESS: charge in hours 0-1, discharge in hours 2-3
        result_with_bess = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.07,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=20.0,  # near min SoC
            baseload_mw=0.05,  # 50 kWh/h
        )

        # Without BESS (offline day): shortfall in hours 2-3 = 50 kWh each
        result_no_bess = dispatch_offline_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.07,
            grid_max_kw=500.0,
            start_soc_kwh=20.0,
            baseload_mw=0.05,
        )

        total_shortfall_with = np.sum(result_with_bess["shortfall"])
        total_shortfall_without = np.sum(result_no_bess["shortfall"])

        assert total_shortfall_with < total_shortfall_without - ATOL, (
            f"BESS should reduce total shortfall: "
            f"with={total_shortfall_with:.2f}, without={total_shortfall_without:.2f}"
        )

    def test_shortfall_constraint_exact_baseload(self) -> None:
        """When export exactly equals baseload, shortfall should be zero."""
        T = 4
        baseload_kwh = 100.0  # kWh per hour
        pv = np.full(T, baseload_kwh)  # PV = baseload exactly
        spot = np.array([0.05, 0.05, 0.05, 0.05])
        bess = _make_bess()

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.07,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=100.0,
            baseload_mw=0.1,  # 100 kWh/h
        )

        np.testing.assert_allclose(result["shortfall"], 0.0, atol=ATOL)

    def test_energy_balance_preserved_with_baseload(self) -> None:
        """PV energy balance must hold regardless of baseload."""
        T = 4
        pv = np.array([50.0, 150.0, 200.0, 30.0])
        spot = np.array([0.05, 0.04, 0.06, 0.08])
        bess = _make_bess()

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.07,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=100.0,
            baseload_mw=0.08,  # 80 kWh/h
        )

        _assert_energy_balance(result, pv)

    def test_shortfall_penalty_incentivises_discharge(self) -> None:
        """With floor price > spot, BESS should prefer discharging in shortfall
        hours even if spot is lower.

        Setup: 2 hours, PV=0 in both, BESS has charge for 1 hour discharge.
        Hour 0: spot=0.03, Hour 1: spot=0.04. Baseload=50 kWh/h.
        Floor price = 0.08 (>> spot).

        Without baseload: BESS discharges in hour 1 (higher spot).
        With baseload: BESS should still discharge, but shortfall penalty is
        high in both hours (eff = 0.08), so order doesn't change.  What
        matters is that BOTH hours have shortfall and the BESS reduces it.
        """
        T = 2
        pv = np.array([0.0, 0.0])
        spot = np.array([0.03, 0.04])
        bess = _make_bess(power_kw=50.0, capacity_kwh=200.0, rte=1.0,
                          min_soc_pct=0.0, max_soc_pct=100.0)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.08,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=100.0,  # Enough for 2 hours of discharge
            baseload_mw=0.05,  # 50 kWh/h
        )

        # BESS should discharge in BOTH hours to reduce shortfall
        assert result["discharge_green"][0] > ATOL, "BESS should discharge in hour 0"
        assert result["discharge_green"][1] > ATOL, "BESS should discharge in hour 1"
        # Shortfall should be reduced (not 50 in each hour)
        assert np.all(result["shortfall"] < 50.0 - ATOL)


class TestBaseloadShortfallGreyMode:
    """Test shortfall tracking in Grey Mode with baseload PPA."""

    def test_no_baseload_shortfall_is_zero_grey(self) -> None:
        """Without baseload PPA, shortfall array must be all zeros in grey mode."""
        T = 4
        pv = np.array([50.0, 100.0, 100.0, 50.0])
        spot = np.array([0.05, 0.04, 0.06, 0.08])
        bess = _make_bess()

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.07,
            bess=bess,
            grid_max_kw=500.0,
            mode="grey",
            start_soc_kwh=100.0,
            baseload_mw=0.0,
        )

        np.testing.assert_array_equal(result["shortfall"], np.zeros(T))

    def test_shortfall_with_baseload_grey(self) -> None:
        """Grey mode should also track shortfall when baseload active."""
        T = 4
        pv = np.array([0.0, 200.0, 200.0, 0.0])
        spot = np.array([0.05, 0.04, 0.04, 0.05])
        bess = _make_bess(power_kw=50.0, capacity_kwh=100.0)

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.07,
            bess=bess,
            grid_max_kw=500.0,
            mode="grey",
            start_soc_kwh=10.0,
            baseload_mw=0.08,  # 80 kWh/h
        )

        assert np.all(result["shortfall"] >= -ATOL)
        # Hour 0: PV=0, near-empty BESS → shortfall expected
        assert result["shortfall"][0] > 0

    def test_grey_energy_balance_with_baseload(self) -> None:
        """PV energy balance must hold in grey mode with baseload."""
        T = 4
        pv = np.array([50.0, 150.0, 200.0, 30.0])
        spot = np.array([0.05, 0.04, 0.06, 0.08])
        bess = _make_bess()

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.07,
            bess=bess,
            grid_max_kw=500.0,
            mode="grey",
            start_soc_kwh=100.0,
            baseload_mw=0.08,
        )

        _assert_energy_balance(result, pv)


class TestBaseloadOfflineDay:
    """Test shortfall computation for BESS offline days."""

    def test_offline_day_shortfall_computed(self) -> None:
        """On offline days, shortfall = max(baseload - export, 0)."""
        T = 4
        pv = np.array([0.0, 150.0, 200.0, 0.0])
        spot = np.array([0.05, 0.04, 0.06, 0.08])
        baseload_kwh = 100.0  # kWh per hour

        result = dispatch_offline_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.07,
            grid_max_kw=500.0,
            start_soc_kwh=100.0,
            baseload_mw=0.1,  # 100 kWh/h (0.1 MW)
        )

        # Hour 0: PV=0, export=0 → shortfall=100
        assert result["shortfall"][0] == pytest.approx(baseload_kwh, abs=ATOL)
        # Hour 1: PV=150, export=150 → shortfall=0
        assert result["shortfall"][1] == pytest.approx(0.0, abs=ATOL)
        # Hour 2: PV=200, export=200 → shortfall=0
        assert result["shortfall"][2] == pytest.approx(0.0, abs=ATOL)
        # Hour 3: PV=0, export=0 → shortfall=100
        assert result["shortfall"][3] == pytest.approx(baseload_kwh, abs=ATOL)

    def test_offline_day_no_baseload_zero_shortfall(self) -> None:
        """Without baseload, offline day shortfall must be zero."""
        T = 4
        pv = np.array([0.0, 150.0, 200.0, 0.0])
        spot = np.array([0.05, 0.04, 0.06, 0.08])

        result = dispatch_offline_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.07,
            grid_max_kw=500.0,
            start_soc_kwh=100.0,
            baseload_mw=0.0,
        )

        np.testing.assert_array_equal(result["shortfall"], np.zeros(T))

    def test_offline_day_shortfall_grid_limited(self) -> None:
        """Shortfall should account for grid export limit."""
        T = 4
        pv = np.array([200.0, 200.0, 200.0, 200.0])
        spot = np.array([0.05, 0.05, 0.05, 0.05])
        grid_max_kw = 80.0  # Grid limits export to 80 kWh/h
        baseload_kwh = 100.0  # kWh per hour

        result = dispatch_offline_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.07,
            grid_max_kw=grid_max_kw,
            start_soc_kwh=100.0,
            baseload_mw=0.1,  # 100 kWh/h
        )

        # Export limited to 80 kWh/h, baseload=100 → shortfall=20 each hour
        np.testing.assert_allclose(result["shortfall"], 20.0, atol=ATOL)


class TestBaseloadDoesNotAffectOtherStructures:
    """Verify that baseload_kw=0 doesn't change behavior for other marketing
    structures (EEG, Floor PPA, Collar PPA)."""

    def test_eeg_floor_unchanged_without_baseload(self) -> None:
        """EEG floor pricing must work the same when baseload_kw=0."""
        T = 4
        pv = np.array([100.0, 100.0, 100.0, 100.0])
        spot = np.array([0.03, 0.05, 0.08, 0.10])
        bess = _make_bess()

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.07,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=100.0,
            baseload_mw=0.0,
        )

        # Effective price should be max(spot, 0.07)
        expected_eff = np.maximum(spot, 0.07)
        np.testing.assert_allclose(result["effective_price"], expected_eff, atol=ATOL)
        np.testing.assert_array_equal(result["shortfall"], np.zeros(T))

    def test_collar_ppa_unchanged_without_baseload(self) -> None:
        """Collar PPA (floor + cap) must work the same when baseload_kw=0."""
        T = 4
        pv = np.array([100.0, 100.0, 100.0, 100.0])
        spot = np.array([0.03, 0.05, 0.08, 0.12])
        bess = _make_bess()

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.06,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=100.0,
            baseload_mw=0.0,
            goo_premium_eur_per_kwh=0.005,
            price_cap_eur_per_kwh=0.10,
        )

        # Effective price should be clip(spot, 0.06, 0.10) + 0.005
        expected_eff = np.clip(spot, 0.06, 0.10) + 0.005
        np.testing.assert_allclose(result["effective_price"], expected_eff, atol=ATOL)
        np.testing.assert_array_equal(result["shortfall"], np.zeros(T))


class TestBaseloadWithGooPremium:
    """Test that GoO premium is correctly applied to shortfall penalty."""

    def test_goo_included_in_shortfall_penalty(self) -> None:
        """The shortfall penalty should include the GoO premium.

        With GoO, shortfall is more expensive → LP should try harder to avoid it.
        """
        T = 4
        pv = np.array([0.0, 200.0, 200.0, 0.0])
        spot = np.array([0.05, 0.04, 0.04, 0.05])
        bess = _make_bess(power_kw=80.0, capacity_kwh=200.0, rte=1.0)

        # With GoO
        result_with_goo = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.07,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=20.0,
            baseload_mw=0.06,  # 60 kWh/h
            goo_premium_eur_per_kwh=0.01,
        )

        # Without GoO
        result_no_goo = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.07,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=20.0,
            baseload_mw=0.06,
            goo_premium_eur_per_kwh=0.0,
        )

        # Both should have valid shortfall
        assert np.all(result_with_goo["shortfall"] >= -ATOL)
        assert np.all(result_no_goo["shortfall"] >= -ATOL)

        # With higher GoO penalty, total shortfall should be <= without GoO
        # (LP tries harder to avoid shortfall when penalty is higher)
        total_with = np.sum(result_with_goo["shortfall"])
        total_without = np.sum(result_no_goo["shortfall"])
        assert total_with <= total_without + ATOL


class TestBaseloadShortfallConsistency:
    """Test consistency between LP shortfall and actual grid export."""

    def test_shortfall_equals_baseload_minus_export(self) -> None:
        """shortfall[t] = max(baseload - grid_export[t], 0) must hold."""
        T = 4
        pv = np.array([0.0, 200.0, 100.0, 0.0])
        spot = np.array([0.05, 0.04, 0.06, 0.05])
        bess = _make_bess(power_kw=50.0, capacity_kwh=200.0, rte=0.9)
        baseload_kwh = 80.0  # kWh per hour

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.07,
            bess=bess,
            grid_max_kw=500.0,
            mode="green",
            start_soc_kwh=20.0,
            baseload_mw=0.08,  # 80 kWh/h
        )

        # Grid export = export_pv + discharge_green (both post-RTE/GLF)
        grid_export = result["export_pv"] + result["discharge_green"]
        expected_shortfall = np.maximum(baseload_kwh - grid_export, 0.0)

        np.testing.assert_allclose(
            result["shortfall"], expected_shortfall, atol=ATOL,
            err_msg="LP shortfall must equal max(baseload - grid_export, 0)",
        )

    def test_shortfall_consistency_grey_mode(self) -> None:
        """shortfall consistency in grey mode: includes grey discharge."""
        T = 4
        pv = np.array([0.0, 200.0, 100.0, 0.0])
        spot = np.array([0.05, 0.04, 0.06, 0.05])
        bess = _make_bess(power_kw=50.0, capacity_kwh=200.0, rte=0.9)
        baseload_kwh = 80.0

        result = optimize_day(
            pv_production_kwh=pv,
            spot_prices_eur_per_kwh=spot,
            price_fixed_eur_per_kwh=0.07,
            bess=bess,
            grid_max_kw=500.0,
            mode="grey",
            start_soc_kwh=20.0,
            baseload_mw=0.08,
        )

        # Grid export = export_pv + discharge_green + discharge_grey
        grid_export = (result["export_pv"] + result["discharge_green"]
                       + result["discharge_grey"])
        expected_shortfall = np.maximum(baseload_kwh - grid_export, 0.0)

        np.testing.assert_allclose(
            result["shortfall"], expected_shortfall, atol=ATOL,
            err_msg="LP shortfall must equal max(baseload - grid_export, 0)",
        )
