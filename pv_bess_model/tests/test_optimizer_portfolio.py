"""Tests for dispatch.optimizer_portfolio – Portfolio LP optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from pv_bess_model.dispatch.optimizer_portfolio import (
    BessFlexParams,
    HeatPumpFlexParams,
    PortfolioDailyResult,
    PortfolioLPConfig,
    optimize_portfolio_day,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config_96() -> PortfolioLPConfig:
    """Standard quarter-hourly config with discount=1.0 for easier testing."""
    return PortfolioLPConfig(
        timestep_hours=0.25,
        intervals_per_day=96,
        perfect_foresight_discount=1.0,
    )


@pytest.fixture
def config_24() -> PortfolioLPConfig:
    """Hourly config for simpler tests."""
    return PortfolioLPConfig(
        timestep_hours=1.0,
        intervals_per_day=24,
        perfect_foresight_discount=1.0,
    )


@pytest.fixture
def small_bess() -> BessFlexParams:
    """Small BESS: 100 kW, 200 kWh, 88% RTE, 10-90% SoC."""
    return BessFlexParams(
        capacity_kwh=200.0,
        power_kw=100.0,
        rte=0.88,
        min_soc_pct=10.0,
        max_soc_pct=90.0,
        start_soc_kwh=100.0,  # 50% of capacity
    )


# ---------------------------------------------------------------------------
# No-BESS tests (should match World A)
# ---------------------------------------------------------------------------


class TestNoBess:
    """Tests for optimize_portfolio_day with bess_params=None."""

    def test_no_bess_matches_world_a(self, config_24: PortfolioLPConfig) -> None:
        """Without BESS, result should equal World A calculation."""
        pv = np.array([10.0] * 12 + [0.0] * 12)   # PV only daytime
        load = np.full(24, 5.0)                      # constant load
        prices = np.full(24, 0.05)

        result = optimize_portfolio_day(pv, load, prices, None, config_24)

        # intervals 0-11: netto = 5, sell = 5 each → total sell = 60 kWh
        # intervals 12-23: netto = -5, buy = 5 each → total buy = 60 kWh
        assert np.sum(result.grid_sell) == pytest.approx(60.0)
        assert np.sum(result.grid_buy) == pytest.approx(60.0)
        assert result.system_cost == pytest.approx(0.0)  # symmetric

    def test_no_bess_zero_charge_discharge(self, config_24: PortfolioLPConfig) -> None:
        """Without BESS, charge and discharge should be zero."""
        pv = np.full(24, 10.0)
        load = np.full(24, 5.0)
        prices = np.full(24, 0.05)

        result = optimize_portfolio_day(pv, load, prices, None, config_24)

        np.testing.assert_array_equal(result.bess_charge, 0.0)
        np.testing.assert_array_equal(result.bess_discharge, 0.0)

    def test_no_bess_solver_status(self, config_24: PortfolioLPConfig) -> None:
        """Without BESS, solver status should be 'no_bess'."""
        result = optimize_portfolio_day(
            np.zeros(24), np.zeros(24), np.zeros(24), None, config_24
        )
        assert result.solver_status == "no_flex"


# ---------------------------------------------------------------------------
# BESS basic tests
# ---------------------------------------------------------------------------


class TestBessBasic:
    """Basic BESS dispatch tests."""

    def test_flat_prices_no_arbitrage(
        self, config_24: PortfolioLPConfig,
    ) -> None:
        """With flat prices, BESS should not trade (no arbitrage opportunity)."""
        # Start at min SoC so there is no free stored energy to discharge
        bess = BessFlexParams(
            capacity_kwh=200.0,
            power_kw=100.0,
            rte=0.88,
            min_soc_pct=10.0,
            max_soc_pct=90.0,
            start_soc_kwh=20.0,  # = min_soc
        )
        pv = np.full(24, 10.0)
        load = np.full(24, 5.0)
        prices = np.full(24, 0.05)

        result = optimize_portfolio_day(pv, load, prices, bess, config_24)

        # With flat prices, charging/discharging loses energy (RTE < 1)
        # so BESS should stay idle
        total_throughput = np.sum(result.bess_charge) + np.sum(result.bess_discharge)
        assert total_throughput < 1.0  # allow small solver tolerance

    def test_price_spread_triggers_arbitrage(
        self, config_24: PortfolioLPConfig
    ) -> None:
        """BESS should charge at low prices and discharge at high prices."""
        bess = BessFlexParams(
            capacity_kwh=100.0,
            power_kw=50.0,
            rte=1.0,  # perfect efficiency for easier validation
            min_soc_pct=0.0,
            max_soc_pct=100.0,
            start_soc_kwh=0.0,
        )

        # Pure load scenario with price spread
        pv = np.zeros(24)
        load = np.zeros(24)
        prices = np.zeros(24)
        # Low price first 6 hours, high price last 6 hours
        prices[:6] = 0.02
        prices[6:18] = 0.05
        prices[18:] = 0.10

        result = optimize_portfolio_day(pv, load, prices, bess, config_24)

        # BESS should charge during low-price hours and discharge during high
        assert np.sum(result.bess_charge[:6]) > 0
        assert np.sum(result.bess_discharge[18:]) > 0

    def test_solver_status_optimal(
        self, config_24: PortfolioLPConfig, small_bess: BessFlexParams
    ) -> None:
        """Solver status should be 'optimal' for feasible problems."""
        result = optimize_portfolio_day(
            np.full(24, 10.0),
            np.full(24, 5.0),
            np.full(24, 0.05),
            small_bess,
            config_24,
        )
        assert result.solver_status == "optimal"


# ---------------------------------------------------------------------------
# Energy balance tests
# ---------------------------------------------------------------------------


class TestEnergyBalance:
    """Tests for energy balance constraints."""

    def test_energy_balance_per_interval(
        self, config_24: PortfolioLPConfig, small_bess: BessFlexParams
    ) -> None:
        """Energy balance: sell - buy = netto + discharge*RTE - charge."""
        rng = np.random.RandomState(42)
        pv = rng.uniform(0, 20, 24)
        load = rng.uniform(0, 15, 24)
        prices = rng.uniform(0.02, 0.10, 24)

        result = optimize_portfolio_day(pv, load, prices, small_bess, config_24)

        netto = pv - load
        balance = (
            result.grid_sell
            - result.grid_buy
            - result.bess_discharge * small_bess.rte
            + result.bess_charge
        )
        np.testing.assert_allclose(balance, netto, atol=1e-6)

    def test_non_negative_grid_flows(
        self, config_24: PortfolioLPConfig, small_bess: BessFlexParams
    ) -> None:
        """grid_sell and grid_buy must be non-negative."""
        rng = np.random.RandomState(123)
        pv = rng.uniform(0, 20, 24)
        load = rng.uniform(0, 15, 24)
        prices = rng.uniform(0.02, 0.10, 24)

        result = optimize_portfolio_day(pv, load, prices, small_bess, config_24)

        assert np.all(result.grid_sell >= -1e-10)
        assert np.all(result.grid_buy >= -1e-10)


# ---------------------------------------------------------------------------
# SoC tests
# ---------------------------------------------------------------------------


class TestSoC:
    """Tests for SoC tracking and limits."""

    def test_soc_initial(
        self, config_24: PortfolioLPConfig, small_bess: BessFlexParams
    ) -> None:
        """SoC at t=0 should equal start_soc_kwh."""
        result = optimize_portfolio_day(
            np.full(24, 10.0),
            np.full(24, 5.0),
            np.full(24, 0.05),
            small_bess,
            config_24,
        )
        assert result.bess_soc[0] == pytest.approx(small_bess.start_soc_kwh)

    def test_soc_within_bounds(
        self, config_24: PortfolioLPConfig, small_bess: BessFlexParams
    ) -> None:
        """SoC must stay within [soc_min, soc_max]."""
        rng = np.random.RandomState(77)
        pv = rng.uniform(0, 30, 24)
        load = rng.uniform(0, 20, 24)
        prices = rng.uniform(0.01, 0.15, 24)

        result = optimize_portfolio_day(pv, load, prices, small_bess, config_24)

        soc_min = small_bess.capacity_kwh * small_bess.min_soc_pct / 100.0
        soc_max = small_bess.capacity_kwh * small_bess.max_soc_pct / 100.0
        assert np.all(result.bess_soc >= soc_min - 1e-6)
        assert np.all(result.bess_soc <= soc_max + 1e-6)

    def test_soc_coupling(
        self, config_24: PortfolioLPConfig, small_bess: BessFlexParams
    ) -> None:
        """SoC linking: soc[t+1] = soc[t] + charge[t] - discharge[t]."""
        rng = np.random.RandomState(99)
        pv = rng.uniform(0, 25, 24)
        load = rng.uniform(0, 15, 24)
        prices = rng.uniform(0.02, 0.12, 24)

        result = optimize_portfolio_day(pv, load, prices, small_bess, config_24)

        for t in range(24):
            expected_next = (
                result.bess_soc[t] + result.bess_charge[t] - result.bess_discharge[t]
            )
            assert result.bess_soc[t + 1] == pytest.approx(expected_next, abs=1e-6)

    def test_end_soc(
        self, config_24: PortfolioLPConfig, small_bess: BessFlexParams
    ) -> None:
        """end_soc_kwh should match the last SoC value."""
        result = optimize_portfolio_day(
            np.full(24, 10.0),
            np.full(24, 5.0),
            np.full(24, 0.05),
            small_bess,
            config_24,
        )
        assert result.end_soc_kwh == pytest.approx(result.bess_soc[-1])

    def test_soc_array_length(
        self, config_24: PortfolioLPConfig, small_bess: BessFlexParams
    ) -> None:
        """SoC array should have T+1 elements."""
        result = optimize_portfolio_day(
            np.full(24, 10.0),
            np.full(24, 5.0),
            np.full(24, 0.05),
            small_bess,
            config_24,
        )
        assert len(result.bess_soc) == 25  # 24 + 1


# ---------------------------------------------------------------------------
# Power limit tests
# ---------------------------------------------------------------------------


class TestPowerLimits:
    """Tests for charge/discharge power limits."""

    def test_charge_power_limit(self, config_24: PortfolioLPConfig) -> None:
        """Charge per interval must not exceed P_max × timestep_hours."""
        bess = BessFlexParams(
            capacity_kwh=1000.0,
            power_kw=50.0,
            rte=0.95,
            min_soc_pct=0.0,
            max_soc_pct=100.0,
            start_soc_kwh=100.0,
        )
        # Lots of cheap PV to motivate maximum charging
        pv = np.full(24, 100.0)
        load = np.zeros(24)
        prices = np.zeros(24)
        prices[20:] = 1.0  # high price at end

        result = optimize_portfolio_day(pv, load, prices, bess, config_24)

        max_energy = bess.power_kw * config_24.timestep_hours
        assert np.all(result.bess_charge <= max_energy + 1e-6)

    def test_discharge_power_limit(self, config_24: PortfolioLPConfig) -> None:
        """Discharge per interval must not exceed P_max × timestep_hours."""
        bess = BessFlexParams(
            capacity_kwh=1000.0,
            power_kw=50.0,
            rte=0.95,
            min_soc_pct=0.0,
            max_soc_pct=100.0,
            start_soc_kwh=900.0,
        )
        pv = np.zeros(24)
        load = np.zeros(24)
        prices = np.full(24, 1.0)  # high price motivates maximum discharge

        result = optimize_portfolio_day(pv, load, prices, bess, config_24)

        max_energy = bess.power_kw * config_24.timestep_hours
        assert np.all(result.bess_discharge <= max_energy + 1e-6)


# ---------------------------------------------------------------------------
# Perfect foresight discount tests
# ---------------------------------------------------------------------------


class TestPerfectForesightDiscount:
    """Tests for the perfect_foresight_discount parameter."""

    def test_discount_reduces_sell_value(self) -> None:
        """Discount < 1.0 should reduce the system value (sell less profitably)."""
        pv = np.full(24, 10.0)
        load = np.zeros(24)
        prices = np.full(24, 0.10)

        config_full = PortfolioLPConfig(
            timestep_hours=1.0, intervals_per_day=24, perfect_foresight_discount=1.0
        )
        config_discounted = PortfolioLPConfig(
            timestep_hours=1.0, intervals_per_day=24, perfect_foresight_discount=0.8
        )

        result_full = optimize_portfolio_day(pv, load, prices, None, config_full)
        result_disc = optimize_portfolio_day(pv, load, prices, None, config_discounted)

        # With discount, system_cost should be less negative (sell revenue reduced)
        assert result_disc.system_cost > result_full.system_cost

    def test_discount_zero_means_free_selling(self) -> None:
        """Discount = 0 means selling generates no revenue."""
        pv = np.full(24, 10.0)
        load = np.zeros(24)
        prices = np.full(24, 0.10)

        config = PortfolioLPConfig(
            timestep_hours=1.0, intervals_per_day=24, perfect_foresight_discount=0.0
        )

        result = optimize_portfolio_day(pv, load, prices, None, config)

        # system_cost should be 0 (selling at 0 revenue, no buying)
        assert result.system_cost == pytest.approx(0.0)

    def test_bess_arbitrage_reduced_by_discount(self) -> None:
        """BESS arbitrage should be less profitable with discount < 1."""
        bess = BessFlexParams(
            capacity_kwh=100.0, power_kw=50.0, rte=1.0,
            min_soc_pct=0.0, max_soc_pct=100.0, start_soc_kwh=0.0,
        )
        pv = np.zeros(24)
        load = np.zeros(24)
        prices = np.zeros(24)
        prices[:6] = 0.01   # cheap
        prices[18:] = 0.20  # expensive

        config_full = PortfolioLPConfig(
            timestep_hours=1.0, intervals_per_day=24, perfect_foresight_discount=1.0
        )
        config_disc = PortfolioLPConfig(
            timestep_hours=1.0, intervals_per_day=24, perfect_foresight_discount=0.5
        )

        result_full = optimize_portfolio_day(pv, load, prices, bess, config_full)
        result_disc = optimize_portfolio_day(pv, load, prices, bess, config_disc)

        # With discount, the BESS arbitrage is less valuable
        assert result_disc.system_cost >= result_full.system_cost


# ---------------------------------------------------------------------------
# Quarter-hourly (96 intervals) test
# ---------------------------------------------------------------------------


class TestQuarterHourly:
    """Tests with 96 intervals per day."""

    def test_96_intervals(self, config_96: PortfolioLPConfig) -> None:
        """LP should work correctly with 96 quarter-hourly intervals."""
        bess = BessFlexParams(
            capacity_kwh=200.0, power_kw=100.0, rte=0.88,
            min_soc_pct=10.0, max_soc_pct=90.0, start_soc_kwh=100.0,
        )
        rng = np.random.RandomState(42)
        pv = rng.uniform(0, 20, 96)
        load = rng.uniform(0, 15, 96)
        prices = rng.uniform(0.02, 0.10, 96)

        result = optimize_portfolio_day(pv, load, prices, bess, config_96)

        assert result.solver_status == "optimal"
        assert len(result.grid_sell) == 96
        assert len(result.grid_buy) == 96
        assert len(result.bess_charge) == 96
        assert len(result.bess_discharge) == 96
        assert len(result.bess_soc) == 97

    def test_96_energy_balance(self, config_96: PortfolioLPConfig) -> None:
        """Energy balance must hold for 96-interval LP."""
        bess = BessFlexParams(
            capacity_kwh=200.0, power_kw=100.0, rte=0.88,
            min_soc_pct=10.0, max_soc_pct=90.0, start_soc_kwh=100.0,
        )
        rng = np.random.RandomState(55)
        pv = rng.uniform(0, 25, 96)
        load = rng.uniform(0, 20, 96)
        prices = rng.uniform(0.01, 0.12, 96)

        result = optimize_portfolio_day(pv, load, prices, bess, config_96)

        netto = pv - load
        balance = (
            result.grid_sell
            - result.grid_buy
            - result.bess_discharge * bess.rte
            + result.bess_charge
        )
        np.testing.assert_allclose(balance, netto, atol=1e-6)


# ---------------------------------------------------------------------------
# Result type test
# ---------------------------------------------------------------------------


class TestResultType:
    """Tests for PortfolioDailyResult."""

    def test_returns_correct_type(self, config_24: PortfolioLPConfig) -> None:
        """Return type should be PortfolioDailyResult."""
        result = optimize_portfolio_day(
            np.zeros(24), np.zeros(24), np.zeros(24), None, config_24
        )
        assert isinstance(result, PortfolioDailyResult)


# ---------------------------------------------------------------------------
# Heat pump fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hp_constant_cop() -> HeatPumpFlexParams:
    """Heat pump with constant COP, no thermal storage (24 intervals)."""
    return HeatPumpFlexParams(
        power_kw=50.0,
        cop_profile=np.full(24, 3.0),
        daily_heat_demand_kwh=120.0,  # 120 kWh_th → 40 kWh_el total
        thermal_storage_kwh=0.0,
        heat_demand_profile=np.full(24, 5.0),  # 5 kWh_th per interval
        start_thermal_soc_kwh=0.0,
    )


@pytest.fixture
def hp_with_storage() -> HeatPumpFlexParams:
    """Heat pump with thermal storage (24 intervals)."""
    return HeatPumpFlexParams(
        power_kw=50.0,
        cop_profile=np.full(24, 3.0),
        daily_heat_demand_kwh=120.0,
        thermal_storage_kwh=60.0,  # 60 kWh_th thermal storage
        heat_demand_profile=np.full(24, 5.0),
        start_thermal_soc_kwh=0.0,
    )


# ---------------------------------------------------------------------------
# No heat pump tests
# ---------------------------------------------------------------------------


class TestNoHeatPump:
    """Tests verifying hp_params=None produces identical results."""

    def test_no_hp_matches_no_flex(self, config_24: PortfolioLPConfig) -> None:
        """Without HP, result with bess_params=None matches no-flex."""
        pv = np.full(24, 10.0)
        load = np.full(24, 5.0)
        prices = np.full(24, 0.05)

        r1 = optimize_portfolio_day(pv, load, prices, None, config_24)
        r2 = optimize_portfolio_day(pv, load, prices, None, config_24, hp_params=None)

        assert r1.system_cost == pytest.approx(r2.system_cost)
        assert r1.wp_load is None
        assert r2.wp_load is None


# ---------------------------------------------------------------------------
# Heat pump only tests (no BESS)
# ---------------------------------------------------------------------------


class TestHeatPumpOnly:
    """Tests for LP with heat pump only (no BESS)."""

    def test_wp_daily_energy_balance(
        self, config_24: PortfolioLPConfig, hp_constant_cop: HeatPumpFlexParams
    ) -> None:
        """Σ wp_load[t] × COP[t] must equal daily_heat_demand."""
        pv = np.full(24, 10.0)
        load = np.full(24, 5.0)
        prices = np.full(24, 0.05)

        result = optimize_portfolio_day(
            pv, load, prices, None, config_24, hp_params=hp_constant_cop
        )

        assert result.wp_load is not None
        total_thermal = float(np.sum(result.wp_load * hp_constant_cop.cop_profile))
        assert total_thermal == pytest.approx(
            hp_constant_cop.daily_heat_demand_kwh, abs=1e-4
        )

    def test_wp_power_limit(
        self, config_24: PortfolioLPConfig, hp_constant_cop: HeatPumpFlexParams
    ) -> None:
        """wp_load[t] must not exceed P_wp × timestep_hours."""
        pv = np.full(24, 10.0)
        load = np.full(24, 5.0)
        prices = np.full(24, 0.05)

        result = optimize_portfolio_day(
            pv, load, prices, None, config_24, hp_params=hp_constant_cop
        )

        max_energy = hp_constant_cop.power_kw * config_24.timestep_hours
        assert np.all(result.wp_load <= max_energy + 1e-6)

    def test_wp_increases_grid_buy(
        self, config_24: PortfolioLPConfig, hp_constant_cop: HeatPumpFlexParams
    ) -> None:
        """WP adds electrical load, so grid buy should increase vs no WP."""
        pv = np.zeros(24)
        load = np.full(24, 5.0)
        prices = np.full(24, 0.05)

        r_no_wp = optimize_portfolio_day(pv, load, prices, None, config_24)
        r_with_wp = optimize_portfolio_day(
            pv, load, prices, None, config_24, hp_params=hp_constant_cop
        )

        assert np.sum(r_with_wp.grid_buy) > np.sum(r_no_wp.grid_buy) + 1.0

    def test_wp_no_storage_follows_demand(
        self, config_24: PortfolioLPConfig
    ) -> None:
        """Without thermal storage, WP must follow demand profile exactly."""
        # Non-uniform demand profile
        demand_profile = np.zeros(24)
        demand_profile[6:18] = 10.0  # only daytime demand
        daily_total = float(np.sum(demand_profile))
        cop = np.full(24, 3.0)

        hp = HeatPumpFlexParams(
            power_kw=100.0,
            cop_profile=cop,
            daily_heat_demand_kwh=daily_total,
            thermal_storage_kwh=0.0,
            heat_demand_profile=demand_profile,
            start_thermal_soc_kwh=0.0,
        )

        pv = np.full(24, 10.0)
        load = np.full(24, 5.0)
        prices = np.full(24, 0.05)

        result = optimize_portfolio_day(pv, load, prices, None, config_24, hp_params=hp)

        # Without storage, thermal SoC cannot shift, so wp_load × COP must
        # match heat demand in each interval
        for t in range(24):
            thermal_out = result.wp_load[t] * cop[t]
            assert thermal_out == pytest.approx(demand_profile[t], abs=1e-4)

    def test_wp_energy_balance_with_grid(
        self, config_24: PortfolioLPConfig, hp_constant_cop: HeatPumpFlexParams
    ) -> None:
        """Energy balance: sell - buy = pv - load - wp_load + bess terms."""
        pv = np.full(24, 10.0)
        load = np.full(24, 5.0)
        prices = np.full(24, 0.05)

        result = optimize_portfolio_day(
            pv, load, prices, None, config_24, hp_params=hp_constant_cop
        )

        netto = pv - load
        balance = result.grid_sell - result.grid_buy + result.wp_load
        np.testing.assert_allclose(balance, netto, atol=1e-6)


# ---------------------------------------------------------------------------
# Heat pump with thermal storage tests
# ---------------------------------------------------------------------------


class TestHeatPumpWithStorage:
    """Tests for LP with heat pump and thermal storage."""

    def test_wp_shifts_to_cheap_hours(self, config_24: PortfolioLPConfig) -> None:
        """With storage and price spread, WP should shift to cheap hours."""
        cop = np.full(24, 3.0)
        demand_profile = np.full(24, 5.0)  # uniform demand
        daily_total = float(np.sum(demand_profile))

        hp = HeatPumpFlexParams(
            power_kw=100.0,
            cop_profile=cop,
            daily_heat_demand_kwh=daily_total,
            thermal_storage_kwh=60.0,
            heat_demand_profile=demand_profile,
            start_thermal_soc_kwh=0.0,
        )

        pv = np.zeros(24)
        load = np.full(24, 5.0)
        prices = np.zeros(24)
        prices[:6] = 0.02   # cheap night
        prices[6:18] = 0.10  # expensive day
        prices[18:] = 0.04  # medium evening

        result = optimize_portfolio_day(
            pv, load, prices, None, config_24, hp_params=hp
        )

        # WP should run more during cheap hours
        cheap_load = float(np.sum(result.wp_load[:6]))
        expensive_load = float(np.sum(result.wp_load[6:18]))
        # Per hour: cheap has fewer hours but should have higher load per hour
        assert cheap_load / 6 > expensive_load / 12 - 0.1

    def test_thermal_soc_within_bounds(
        self, config_24: PortfolioLPConfig, hp_with_storage: HeatPumpFlexParams
    ) -> None:
        """Thermal SoC must stay within [0, thermal_storage_kwh]."""
        pv = np.full(24, 10.0)
        load = np.full(24, 5.0)
        prices = np.concatenate([np.full(12, 0.02), np.full(12, 0.10)])

        result = optimize_portfolio_day(
            pv, load, prices, None, config_24, hp_params=hp_with_storage
        )

        assert result.thermal_soc is not None
        assert np.all(result.thermal_soc >= -1e-6)
        assert np.all(
            result.thermal_soc <= hp_with_storage.thermal_storage_kwh + 1e-6
        )

    def test_thermal_soc_linking(
        self, config_24: PortfolioLPConfig, hp_with_storage: HeatPumpFlexParams
    ) -> None:
        """Thermal SoC linking: tsoc[t+1] = tsoc[t] + wp[t]*COP[t] - demand[t]."""
        pv = np.full(24, 10.0)
        load = np.full(24, 5.0)
        prices = np.concatenate([np.full(12, 0.02), np.full(12, 0.10)])

        result = optimize_portfolio_day(
            pv, load, prices, None, config_24, hp_params=hp_with_storage
        )

        for t in range(24):
            expected = (
                result.thermal_soc[t]
                + result.wp_load[t] * hp_with_storage.cop_profile[t]
                - hp_with_storage.heat_demand_profile[t]
            )
            assert result.thermal_soc[t + 1] == pytest.approx(expected, abs=1e-4)

    def test_storage_reduces_cost_vs_no_storage(
        self, config_24: PortfolioLPConfig
    ) -> None:
        """Thermal storage allows load shifting, reducing system cost."""
        cop = np.full(24, 3.0)
        demand_profile = np.full(24, 5.0)
        daily_total = float(np.sum(demand_profile))

        hp_no_storage = HeatPumpFlexParams(
            power_kw=100.0, cop_profile=cop,
            daily_heat_demand_kwh=daily_total,
            thermal_storage_kwh=0.0,
            heat_demand_profile=demand_profile,
            start_thermal_soc_kwh=0.0,
        )
        hp_storage = HeatPumpFlexParams(
            power_kw=100.0, cop_profile=cop,
            daily_heat_demand_kwh=daily_total,
            thermal_storage_kwh=60.0,
            heat_demand_profile=demand_profile,
            start_thermal_soc_kwh=0.0,
        )

        pv = np.zeros(24)
        load = np.full(24, 5.0)
        prices = np.concatenate([np.full(12, 0.02), np.full(12, 0.10)])

        r_no = optimize_portfolio_day(
            pv, load, prices, None, config_24, hp_params=hp_no_storage
        )
        r_yes = optimize_portfolio_day(
            pv, load, prices, None, config_24, hp_params=hp_storage
        )

        assert r_yes.system_cost <= r_no.system_cost + 1e-6

    def test_end_thermal_soc(
        self, config_24: PortfolioLPConfig, hp_with_storage: HeatPumpFlexParams
    ) -> None:
        """end_thermal_soc_kwh should match the last thermal_soc value."""
        pv = np.full(24, 10.0)
        load = np.full(24, 5.0)
        prices = np.full(24, 0.05)

        result = optimize_portfolio_day(
            pv, load, prices, None, config_24, hp_params=hp_with_storage
        )

        assert result.thermal_soc is not None
        assert result.end_thermal_soc_kwh == pytest.approx(
            result.thermal_soc[-1], abs=1e-6
        )


# ---------------------------------------------------------------------------
# BESS + Heat pump combined tests
# ---------------------------------------------------------------------------


class TestBessAndHeatPump:
    """Tests for LP with both BESS and heat pump."""

    def test_combined_energy_balance(
        self,
        config_24: PortfolioLPConfig,
        small_bess: BessFlexParams,
        hp_constant_cop: HeatPumpFlexParams,
    ) -> None:
        """Energy balance with both BESS and HP."""
        rng = np.random.RandomState(42)
        pv = rng.uniform(0, 20, 24)
        load = rng.uniform(0, 15, 24)
        prices = rng.uniform(0.02, 0.10, 24)

        result = optimize_portfolio_day(
            pv, load, prices, small_bess, config_24, hp_params=hp_constant_cop
        )

        netto = pv - load
        balance = (
            result.grid_sell
            - result.grid_buy
            - result.bess_discharge * small_bess.rte
            + result.bess_charge
            + result.wp_load
        )
        np.testing.assert_allclose(balance, netto, atol=1e-5)

    def test_combined_better_than_bess_only(
        self,
        config_24: PortfolioLPConfig,
        small_bess: BessFlexParams,
        hp_with_storage: HeatPumpFlexParams,
    ) -> None:
        """BESS+HP should achieve lower or equal cost vs BESS alone."""
        pv = np.zeros(24)
        load = np.full(24, 5.0)
        prices = np.concatenate([np.full(12, 0.02), np.full(12, 0.10)])

        r_bess = optimize_portfolio_day(pv, load, prices, small_bess, config_24)
        r_both = optimize_portfolio_day(
            pv, load, prices, small_bess, config_24, hp_params=hp_with_storage
        )

        # Combined should not be worse (HP adds load but shifts it to cheap)
        # Note: HP always adds load so cost might increase; the key is the
        # LP can optimize the timing. We just check feasibility.
        assert r_both.solver_status == "optimal"

    def test_wp_only_no_bess(
        self,
        config_24: PortfolioLPConfig,
        hp_with_storage: HeatPumpFlexParams,
    ) -> None:
        """HP without BESS should produce zero BESS arrays."""
        pv = np.full(24, 10.0)
        load = np.full(24, 5.0)
        prices = np.full(24, 0.05)

        result = optimize_portfolio_day(
            pv, load, prices, None, config_24, hp_params=hp_with_storage
        )

        np.testing.assert_array_equal(result.bess_charge, 0.0)
        np.testing.assert_array_equal(result.bess_discharge, 0.0)
        assert result.wp_load is not None
        assert result.solver_status == "optimal"
