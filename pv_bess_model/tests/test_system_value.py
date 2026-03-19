"""Tests for portfolio.system_value – World A, enumeration, system value."""

from __future__ import annotations

import numpy as np
import pytest

from pv_bess_model.config.loader_portfolio import (
    BessFlexConfig,
    EVFlexConfig,
    HeatPumpFlexConfig,
)
from pv_bess_model.dispatch.engine_portfolio import PortfolioEngineConfig
from pv_bess_model.portfolio.system_value import (
    SystemValuePoint,
    SystemValueResult,
    WorldAResult,
    compute_world_a,
    compute_world_a_multiyear,
    run_enumeration,
)


class TestComputeWorldA:
    """Tests for compute_world_a()."""

    def test_pure_pv_no_load(self) -> None:
        """Pure PV (no load) → all surplus sold, system_cost < 0 (net revenue)."""
        pv = np.full(96, 10.0)   # 10 kWh per interval
        load = np.zeros(96)
        prices = np.full(96, 0.05)  # 5 ct/kWh

        result = compute_world_a(pv, load, prices)

        assert result.total_sell_kwh == pytest.approx(960.0)
        assert result.total_buy_kwh == pytest.approx(0.0)
        assert result.total_sell_eur == pytest.approx(48.0)
        assert result.total_buy_eur == pytest.approx(0.0)
        assert result.system_cost == pytest.approx(-48.0)

    def test_pure_load_no_pv(self) -> None:
        """Pure load (no PV) → all deficit bought, system_cost > 0."""
        pv = np.zeros(96)
        load = np.full(96, 10.0)
        prices = np.full(96, 0.05)

        result = compute_world_a(pv, load, prices)

        assert result.total_sell_kwh == pytest.approx(0.0)
        assert result.total_buy_kwh == pytest.approx(960.0)
        assert result.system_cost == pytest.approx(48.0)

    def test_pv_equals_load(self) -> None:
        """PV = load → no grid interaction, system_cost = 0."""
        pv = np.full(96, 10.0)
        load = np.full(96, 10.0)
        prices = np.full(96, 0.05)

        result = compute_world_a(pv, load, prices)

        assert result.system_cost == pytest.approx(0.0)
        assert result.total_sell_kwh == pytest.approx(0.0)
        assert result.total_buy_kwh == pytest.approx(0.0)

    def test_energy_conservation(self) -> None:
        """Total sell + total buy should equal total absolute net position."""
        rng = np.random.RandomState(42)
        pv = rng.uniform(0, 20, 96)
        load = rng.uniform(0, 20, 96)
        prices = rng.uniform(0.02, 0.10, 96)

        result = compute_world_a(pv, load, prices)

        netto_abs = np.sum(np.abs(pv - load))
        assert result.total_sell_kwh + result.total_buy_kwh == pytest.approx(netto_abs)

    def test_netto_array(self) -> None:
        """Netto array should equal pv - load."""
        pv = np.array([10.0, 5.0, 0.0, 15.0])
        load = np.array([5.0, 10.0, 0.0, 5.0])
        prices = np.array([0.05, 0.05, 0.05, 0.05])

        result = compute_world_a(pv, load, prices)

        np.testing.assert_array_almost_equal(result.netto, [5.0, -5.0, 0.0, 10.0])

    def test_different_length_raises(self) -> None:
        """Mismatched array lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            compute_world_a(np.zeros(96), np.zeros(48), np.zeros(96))

    def test_varying_prices(self) -> None:
        """Revenue/cost should weight by price per interval."""
        pv = np.array([10.0, 0.0])       # sell 10 in interval 0
        load = np.array([0.0, 10.0])      # buy 10 in interval 1
        prices = np.array([0.10, 0.02])   # sell@10ct, buy@2ct

        result = compute_world_a(pv, load, prices)

        assert result.total_sell_eur == pytest.approx(1.0)   # 10 × 0.10
        assert result.total_buy_eur == pytest.approx(0.2)    # 10 × 0.02
        assert result.system_cost == pytest.approx(-0.8)     # 0.2 - 1.0

    def test_negative_prices(self) -> None:
        """Negative spot prices: selling at negative price = cost."""
        pv = np.array([10.0])
        load = np.array([0.0])
        prices = np.array([-0.05])

        result = compute_world_a(pv, load, prices)

        # Selling 10 kWh at -0.05 EUR/kWh = -0.5 EUR revenue
        assert result.total_sell_eur == pytest.approx(-0.5)
        # system_cost = 0 - (-0.5) = 0.5 (net cost from negative price)
        assert result.system_cost == pytest.approx(0.5)

    def test_returns_world_a_result(self) -> None:
        """Return type should be WorldAResult."""
        result = compute_world_a(np.zeros(4), np.zeros(4), np.zeros(4))
        assert isinstance(result, WorldAResult)


# ---------------------------------------------------------------------------
# Fixtures for multi-year / enumeration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def short_engine_config() -> PortfolioEngineConfig:
    """Short-lifetime engine config (2 years, hourly)."""
    return PortfolioEngineConfig(
        lifetime_years=2,
        baseline_year=2027,
        timestep_hours=1.0,
        intervals_per_day=24,
        intervals_per_year=365 * 24,
        perfect_foresight_discount=1.0,
    )


@pytest.fixture
def flat_pv() -> np.ndarray:
    """Constant PV: 10 kWh/h over 8760 hours."""
    return np.full(365 * 24, 10.0)


@pytest.fixture
def flat_load() -> np.ndarray:
    """Constant load: 8 kWh/h over 8760 hours."""
    return np.full(365 * 24, 8.0)


@pytest.fixture
def flat_prices() -> np.ndarray:
    """Constant price: 0.05 EUR/kWh over 8760 hours."""
    return np.full(365 * 24, 0.05)


@pytest.fixture
def varying_daily_prices() -> np.ndarray:
    """Daily pattern with price spread for BESS arbitrage."""
    daily = np.concatenate([
        np.full(12, 0.02),  # night: cheap
        np.full(12, 0.10),  # day: expensive
    ])
    return np.tile(daily, 365)


# ---------------------------------------------------------------------------
# World A multi-year tests
# ---------------------------------------------------------------------------


class TestComputeWorldAMultiyear:
    """Tests for compute_world_a_multiyear()."""

    def test_result_length(
        self,
        short_engine_config: PortfolioEngineConfig,
        flat_pv: np.ndarray,
        flat_load: np.ndarray,
        flat_prices: np.ndarray,
    ) -> None:
        """Returns one cost per year."""
        costs = compute_world_a_multiyear(
            config=short_engine_config,
            pv_profile_base=flat_pv,
            load_profile_base=flat_load,
            spot_prices_base=flat_prices,
            pv_degradation_rate=0.0,
        )
        assert len(costs) == 2

    def test_constant_inputs_constant_cost(
        self,
        short_engine_config: PortfolioEngineConfig,
        flat_pv: np.ndarray,
        flat_load: np.ndarray,
        flat_prices: np.ndarray,
    ) -> None:
        """Without degradation/growth, annual cost is constant."""
        costs = compute_world_a_multiyear(
            config=short_engine_config,
            pv_profile_base=flat_pv,
            load_profile_base=flat_load,
            spot_prices_base=flat_prices,
            pv_degradation_rate=0.0,
        )
        assert costs[0] == pytest.approx(costs[1], abs=1e-6)


# ---------------------------------------------------------------------------
# Enumeration tests
# ---------------------------------------------------------------------------


class TestRunEnumeration:
    """Tests for the enumeration / system value calculation."""

    def test_zero_addition_system_value_is_zero(
        self,
        short_engine_config: PortfolioEngineConfig,
        flat_pv: np.ndarray,
        flat_load: np.ndarray,
        flat_prices: np.ndarray,
    ) -> None:
        """System value at addition_kw=0 is 0 (no flex = World A)."""
        flex = BessFlexConfig(
            type="bess",
            name="test_bess",
            annual_addition_kw=[0.0],
            e_to_p_ratio_hours=[2.0],
            round_trip_efficiency_pct=88.0,
            min_soc_pct=10.0,
            max_soc_pct=90.0,
            degradation_rate_pct_per_year=0.0,
        )
        result = run_enumeration(
            config=short_engine_config,
            pv_profile_base=flat_pv,
            load_profile_base=flat_load,
            spot_prices_base=flat_prices,
            flexibilities=[flex],
            pv_degradation_rate=0.0,
            max_workers=1,
        )
        assert len(result.points) == 1
        assert result.points[0].cumulative_system_value_eur == pytest.approx(0.0, abs=1e-6)

    def test_system_value_monotonic_with_arbitrage(
        self,
        short_engine_config: PortfolioEngineConfig,
        flat_pv: np.ndarray,
        flat_load: np.ndarray,
        varying_daily_prices: np.ndarray,
    ) -> None:
        """System value increases monotonically with BESS addition rate."""
        flex = BessFlexConfig(
            type="bess",
            name="test_bess",
            annual_addition_kw=[0.0, 50.0, 100.0],
            e_to_p_ratio_hours=[2.0],
            round_trip_efficiency_pct=100.0,
            min_soc_pct=0.0,
            max_soc_pct=100.0,
            degradation_rate_pct_per_year=0.0,
        )
        result = run_enumeration(
            config=short_engine_config,
            pv_profile_base=flat_pv,
            load_profile_base=flat_load,
            spot_prices_base=varying_daily_prices,
            flexibilities=[flex],
            pv_degradation_rate=0.0,
            max_workers=1,
        )
        points = sorted(result.points, key=lambda p: p.annual_addition_kw)
        assert len(points) == 3

        values = [p.cumulative_system_value_eur for p in points]
        # Monotonically non-decreasing
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1] - 1e-6

    def test_enumeration_point_count(
        self,
        short_engine_config: PortfolioEngineConfig,
        flat_pv: np.ndarray,
        flat_load: np.ndarray,
        flat_prices: np.ndarray,
    ) -> None:
        """Number of points = len(rates) × len(e_to_p_ratios)."""
        flex = BessFlexConfig(
            type="bess",
            name="test_bess",
            annual_addition_kw=[0.0, 50.0, 100.0],
            e_to_p_ratio_hours=[1.0, 2.0],
            round_trip_efficiency_pct=88.0,
            min_soc_pct=10.0,
            max_soc_pct=90.0,
            degradation_rate_pct_per_year=0.0,
        )
        result = run_enumeration(
            config=short_engine_config,
            pv_profile_base=flat_pv,
            load_profile_base=flat_load,
            spot_prices_base=flat_prices,
            flexibilities=[flex],
            pv_degradation_rate=0.0,
            max_workers=1,
        )
        assert len(result.points) == 6  # 3 rates × 2 E/P

    def test_world_a_costs_populated(
        self,
        short_engine_config: PortfolioEngineConfig,
        flat_pv: np.ndarray,
        flat_load: np.ndarray,
        flat_prices: np.ndarray,
    ) -> None:
        """World A costs should be populated in the result."""
        flex = BessFlexConfig(
            type="bess",
            name="test_bess",
            annual_addition_kw=[0.0],
            e_to_p_ratio_hours=[2.0],
            round_trip_efficiency_pct=88.0,
            min_soc_pct=10.0,
            max_soc_pct=90.0,
            degradation_rate_pct_per_year=0.0,
        )
        result = run_enumeration(
            config=short_engine_config,
            pv_profile_base=flat_pv,
            load_profile_base=flat_load,
            spot_prices_base=flat_prices,
            flexibilities=[flex],
            pv_degradation_rate=0.0,
            max_workers=1,
        )
        assert len(result.world_a_annual_costs) == 2
        # With constant surplus (pv=10 > load=8), cost should be negative
        for cost in result.world_a_annual_costs:
            assert cost < 0  # net revenue

    def test_annual_system_values_length(
        self,
        short_engine_config: PortfolioEngineConfig,
        flat_pv: np.ndarray,
        flat_load: np.ndarray,
        flat_prices: np.ndarray,
    ) -> None:
        """Each point has annual_system_values with length = lifetime."""
        flex = BessFlexConfig(
            type="bess",
            name="test_bess",
            annual_addition_kw=[50.0],
            e_to_p_ratio_hours=[2.0],
            round_trip_efficiency_pct=88.0,
            min_soc_pct=10.0,
            max_soc_pct=90.0,
            degradation_rate_pct_per_year=0.0,
        )
        result = run_enumeration(
            config=short_engine_config,
            pv_profile_base=flat_pv,
            load_profile_base=flat_load,
            spot_prices_base=flat_prices,
            flexibilities=[flex],
            pv_degradation_rate=0.0,
            max_workers=1,
        )
        assert len(result.points) == 1
        assert len(result.points[0].annual_system_values) == 2


# ---------------------------------------------------------------------------
# Fixtures for HP / EV enumeration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def constant_temperature() -> np.ndarray:
    """Constant 5°C hourly temperature for 8760 hours."""
    return np.full(8760, 5.0)


# ---------------------------------------------------------------------------
# Heat Pump enumeration tests
# ---------------------------------------------------------------------------


class TestHeatPumpEnumeration:
    """Tests for heat pump flex enumeration in run_enumeration()."""

    def test_hp_zero_addition_system_value_is_zero(
        self,
        short_engine_config: PortfolioEngineConfig,
        flat_pv: np.ndarray,
        flat_load: np.ndarray,
        flat_prices: np.ndarray,
        constant_temperature: np.ndarray,
    ) -> None:
        """System value at addition_kw=0 is 0 (no flex = World A)."""
        flex = HeatPumpFlexConfig(
            type="heat_pump",
            name="test_wp",
            annual_addition_kw=[0.0],
            cop_nominal=3.0,
            cop_reference_temp_c=7.0,
            annual_thermal_demand_mwh=100.0,
            thermal_storage_kwh=0.0,
        )
        result = run_enumeration(
            config=short_engine_config,
            pv_profile_base=flat_pv,
            load_profile_base=flat_load,
            spot_prices_base=flat_prices,
            flexibilities=[flex],
            pv_degradation_rate=0.0,
            max_workers=1,
            temperature_hourly=constant_temperature,
        )
        assert len(result.points) == 1
        assert result.points[0].flex_type == "heat_pump"
        assert result.points[0].e_to_p_ratio is None
        assert result.points[0].cumulative_system_value_eur == pytest.approx(
            0.0, abs=1e-6
        )

    def test_hp_point_count(
        self,
        short_engine_config: PortfolioEngineConfig,
        flat_pv: np.ndarray,
        flat_load: np.ndarray,
        flat_prices: np.ndarray,
        constant_temperature: np.ndarray,
    ) -> None:
        """Number of points = len(annual_addition_kw)."""
        flex = HeatPumpFlexConfig(
            type="heat_pump",
            name="test_wp",
            annual_addition_kw=[0.0, 50.0, 100.0],
            cop_nominal=3.0,
            cop_reference_temp_c=7.0,
            annual_thermal_demand_mwh=100.0,
            thermal_storage_kwh=0.0,
        )
        result = run_enumeration(
            config=short_engine_config,
            pv_profile_base=flat_pv,
            load_profile_base=flat_load,
            spot_prices_base=flat_prices,
            flexibilities=[flex],
            pv_degradation_rate=0.0,
            max_workers=1,
            temperature_hourly=constant_temperature,
        )
        assert len(result.points) == 3
        for p in result.points:
            assert p.flex_type == "heat_pump"

    def test_hp_requires_temperature(
        self,
        short_engine_config: PortfolioEngineConfig,
        flat_pv: np.ndarray,
        flat_load: np.ndarray,
        flat_prices: np.ndarray,
    ) -> None:
        """Raises ValueError if temperature_hourly is None for HP."""
        flex = HeatPumpFlexConfig(
            type="heat_pump",
            name="test_wp",
            annual_addition_kw=[50.0],
            cop_nominal=3.0,
            annual_thermal_demand_mwh=100.0,
            thermal_storage_kwh=0.0,
        )
        with pytest.raises(ValueError, match="temperature_hourly"):
            run_enumeration(
                config=short_engine_config,
                pv_profile_base=flat_pv,
                load_profile_base=flat_load,
                spot_prices_base=flat_prices,
                flexibilities=[flex],
                pv_degradation_rate=0.0,
                max_workers=1,
            )

    def test_hp_with_price_spread_has_positive_value(
        self,
        short_engine_config: PortfolioEngineConfig,
        flat_pv: np.ndarray,
        flat_load: np.ndarray,
        varying_daily_prices: np.ndarray,
        constant_temperature: np.ndarray,
    ) -> None:
        """HP with thermal storage and price spread should have positive value."""
        flex = HeatPumpFlexConfig(
            type="heat_pump",
            name="test_wp",
            annual_addition_kw=[50.0],
            cop_nominal=3.0,
            cop_reference_temp_c=7.0,
            annual_thermal_demand_mwh=100.0,
            thermal_storage_kwh=200.0,
        )
        result = run_enumeration(
            config=short_engine_config,
            pv_profile_base=flat_pv,
            load_profile_base=flat_load,
            spot_prices_base=varying_daily_prices,
            flexibilities=[flex],
            pv_degradation_rate=0.0,
            max_workers=1,
            temperature_hourly=constant_temperature,
        )
        assert len(result.points) == 1
        # Thermal storage enables shifting HP load to cheap hours
        assert result.points[0].cumulative_system_value_eur >= 0.0


# ---------------------------------------------------------------------------
# EV enumeration tests
# ---------------------------------------------------------------------------


class TestEVEnumeration:
    """Tests for EV flex enumeration in run_enumeration()."""

    def test_ev_zero_units_system_value_is_zero(
        self,
        short_engine_config: PortfolioEngineConfig,
        flat_pv: np.ndarray,
        flat_load: np.ndarray,
        flat_prices: np.ndarray,
    ) -> None:
        """System value at 0 units is 0 (no flex = World A)."""
        flex = EVFlexConfig(
            type="ev_charging",
            name="test_ev",
            mean_kw_per_unit=11.0,
            annual_additional_units=[0],
            daily_energy_demand_kwh_per_unit=30.0,
            arrival_hour=8,
            departure_hour=18,
            v2g_enabled=False,
            usable_battery_kwh_per_unit=60.0,
        )
        result = run_enumeration(
            config=short_engine_config,
            pv_profile_base=flat_pv,
            load_profile_base=flat_load,
            spot_prices_base=flat_prices,
            flexibilities=[flex],
            pv_degradation_rate=0.0,
            max_workers=1,
        )
        assert len(result.points) == 1
        assert result.points[0].flex_type == "ev_charging"
        assert result.points[0].e_to_p_ratio is None
        assert result.points[0].cumulative_system_value_eur == pytest.approx(
            0.0, abs=1e-6
        )

    def test_ev_point_count(
        self,
        short_engine_config: PortfolioEngineConfig,
        flat_pv: np.ndarray,
        flat_load: np.ndarray,
        flat_prices: np.ndarray,
    ) -> None:
        """Number of points = len(annual_additional_units)."""
        flex = EVFlexConfig(
            type="ev_charging",
            name="test_ev",
            mean_kw_per_unit=11.0,
            annual_additional_units=[0, 5, 10],
            daily_energy_demand_kwh_per_unit=30.0,
            arrival_hour=8,
            departure_hour=18,
            v2g_enabled=False,
            usable_battery_kwh_per_unit=60.0,
        )
        result = run_enumeration(
            config=short_engine_config,
            pv_profile_base=flat_pv,
            load_profile_base=flat_load,
            spot_prices_base=flat_prices,
            flexibilities=[flex],
            pv_degradation_rate=0.0,
            max_workers=1,
        )
        assert len(result.points) == 3
        for p in result.points:
            assert p.flex_type == "ev_charging"

    def test_ev_v2g_with_price_spread(
        self,
        short_engine_config: PortfolioEngineConfig,
        flat_pv: np.ndarray,
        flat_load: np.ndarray,
        varying_daily_prices: np.ndarray,
    ) -> None:
        """EV with V2G and price spread should have positive value."""
        flex = EVFlexConfig(
            type="ev_charging",
            name="test_ev",
            mean_kw_per_unit=11.0,
            annual_additional_units=[10],
            daily_energy_demand_kwh_per_unit=20.0,
            arrival_hour=8,
            departure_hour=18,
            v2g_enabled=True,
            v2g_rte_pct=90.0,
            min_departure_soc_pct=50.0,
            usable_battery_kwh_per_unit=60.0,
        )
        result = run_enumeration(
            config=short_engine_config,
            pv_profile_base=flat_pv,
            load_profile_base=flat_load,
            spot_prices_base=varying_daily_prices,
            flexibilities=[flex],
            pv_degradation_rate=0.0,
            max_workers=1,
        )
        assert len(result.points) == 1
        # V2G enables arbitrage with price spread
        assert result.points[0].cumulative_system_value_eur >= 0.0


# ---------------------------------------------------------------------------
# Mixed flex enumeration tests
# ---------------------------------------------------------------------------


class TestMixedFlexEnumeration:
    """Tests for enumeration with multiple flex types."""

    def test_bess_and_hp_combined(
        self,
        short_engine_config: PortfolioEngineConfig,
        flat_pv: np.ndarray,
        flat_load: np.ndarray,
        flat_prices: np.ndarray,
        constant_temperature: np.ndarray,
    ) -> None:
        """BESS + HP flexibilities are both enumerated."""
        bess = BessFlexConfig(
            type="bess",
            name="test_bess",
            annual_addition_kw=[50.0],
            e_to_p_ratio_hours=[2.0],
            round_trip_efficiency_pct=88.0,
            min_soc_pct=10.0,
            max_soc_pct=90.0,
            degradation_rate_pct_per_year=0.0,
        )
        hp = HeatPumpFlexConfig(
            type="heat_pump",
            name="test_wp",
            annual_addition_kw=[50.0],
            cop_nominal=3.0,
            annual_thermal_demand_mwh=100.0,
            thermal_storage_kwh=0.0,
        )
        result = run_enumeration(
            config=short_engine_config,
            pv_profile_base=flat_pv,
            load_profile_base=flat_load,
            spot_prices_base=flat_prices,
            flexibilities=[bess, hp],
            pv_degradation_rate=0.0,
            max_workers=1,
            temperature_hourly=constant_temperature,
        )
        assert len(result.points) == 2
        types = {p.flex_type for p in result.points}
        assert types == {"bess", "heat_pump"}

    def test_all_three_flex_types(
        self,
        short_engine_config: PortfolioEngineConfig,
        flat_pv: np.ndarray,
        flat_load: np.ndarray,
        flat_prices: np.ndarray,
        constant_temperature: np.ndarray,
    ) -> None:
        """BESS + HP + EV flexibilities are all enumerated."""
        bess = BessFlexConfig(
            type="bess",
            name="test_bess",
            annual_addition_kw=[50.0],
            e_to_p_ratio_hours=[2.0],
            round_trip_efficiency_pct=88.0,
            min_soc_pct=10.0,
            max_soc_pct=90.0,
            degradation_rate_pct_per_year=0.0,
        )
        hp = HeatPumpFlexConfig(
            type="heat_pump",
            name="test_wp",
            annual_addition_kw=[50.0],
            cop_nominal=3.0,
            annual_thermal_demand_mwh=100.0,
            thermal_storage_kwh=0.0,
        )
        ev = EVFlexConfig(
            type="ev_charging",
            name="test_ev",
            mean_kw_per_unit=11.0,
            annual_additional_units=[5],
            daily_energy_demand_kwh_per_unit=30.0,
            arrival_hour=8,
            departure_hour=18,
            v2g_enabled=False,
            usable_battery_kwh_per_unit=60.0,
        )
        result = run_enumeration(
            config=short_engine_config,
            pv_profile_base=flat_pv,
            load_profile_base=flat_load,
            spot_prices_base=flat_prices,
            flexibilities=[bess, hp, ev],
            pv_degradation_rate=0.0,
            max_workers=1,
            temperature_hourly=constant_temperature,
        )
        assert len(result.points) == 3
        types = {p.flex_type for p in result.points}
        assert types == {"bess", "heat_pump", "ev_charging"}
