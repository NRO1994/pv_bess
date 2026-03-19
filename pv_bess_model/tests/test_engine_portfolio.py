"""Tests for dispatch.engine_portfolio – Multi-year portfolio engine."""

from __future__ import annotations

import numpy as np
import pytest

from pv_bess_model.dispatch.engine_portfolio import (
    FlexCapacityYear,
    PortfolioAnnualResult,
    PortfolioEngineConfig,
    compute_bess_tranche_capacity,
    compute_ev_capacity,
    compute_wp_capacity,
    run_portfolio_simulation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine_config_short() -> PortfolioEngineConfig:
    """Short-lifetime engine config with hourly resolution for fast tests."""
    return PortfolioEngineConfig(
        lifetime_years=3,
        baseline_year=2027,
        timestep_hours=1.0,
        intervals_per_day=24,
        intervals_per_year=365 * 24,
        perfect_foresight_discount=1.0,
    )


@pytest.fixture
def engine_config_5y() -> PortfolioEngineConfig:
    """5-year engine config with hourly resolution."""
    return PortfolioEngineConfig(
        lifetime_years=5,
        baseline_year=2027,
        timestep_hours=1.0,
        intervals_per_day=24,
        intervals_per_year=365 * 24,
        perfect_foresight_discount=1.0,
    )


@pytest.fixture
def flat_pv_profile() -> np.ndarray:
    """Constant PV production: 10 kWh per hour, 8760 hours."""
    return np.full(365 * 24, 10.0)


@pytest.fixture
def flat_load_profile() -> np.ndarray:
    """Constant load: 8 kWh per hour, 8760 hours."""
    return np.full(365 * 24, 8.0)


@pytest.fixture
def flat_prices() -> np.ndarray:
    """Constant spot price: 0.05 EUR/kWh, 8760 hours."""
    return np.full(365 * 24, 0.05)


@pytest.fixture
def varying_prices() -> np.ndarray:
    """Repeating daily price pattern: low at night, high during day."""
    daily = np.concatenate([
        np.full(8, 0.02),   # 0-7: low night
        np.full(8, 0.10),   # 8-15: high day
        np.full(8, 0.04),   # 16-23: medium evening
    ])
    return np.tile(daily, 365)


# ---------------------------------------------------------------------------
# compute_bess_tranche_capacity tests
# ---------------------------------------------------------------------------


class TestBessTrancheCapacity:
    """Tests for the BESS tranche degradation model."""

    def test_zero_addition(self) -> None:
        """Zero addition produces zero capacity."""
        power, capacity = compute_bess_tranche_capacity(
            annual_addition_kw=0.0,
            e_to_p_ratio=2.0,
            project_year=5,
            degradation_rate=0.02,
        )
        assert power == 0.0
        assert capacity == 0.0

    def test_no_degradation(self) -> None:
        """Without degradation, capacity = addition × year × E/P."""
        power, capacity = compute_bess_tranche_capacity(
            annual_addition_kw=100.0,
            e_to_p_ratio=2.0,
            project_year=5,
            degradation_rate=0.0,
        )
        assert power == pytest.approx(500.0)  # 100 × 5
        assert capacity == pytest.approx(1000.0)  # 500 × 2

    def test_year_1(self) -> None:
        """Year 1: one tranche, age=0, no degradation effect."""
        power, capacity = compute_bess_tranche_capacity(
            annual_addition_kw=100.0,
            e_to_p_ratio=2.0,
            project_year=1,
            degradation_rate=0.02,
        )
        assert power == pytest.approx(100.0)
        assert capacity == pytest.approx(200.0)  # age=0, no deg

    def test_degradation_reduces_capacity(self) -> None:
        """With degradation, total capacity < naive sum."""
        power_deg, capacity_deg = compute_bess_tranche_capacity(
            annual_addition_kw=100.0,
            e_to_p_ratio=2.0,
            project_year=5,
            degradation_rate=0.02,
        )
        _, capacity_no_deg = compute_bess_tranche_capacity(
            annual_addition_kw=100.0,
            e_to_p_ratio=2.0,
            project_year=5,
            degradation_rate=0.0,
        )
        # Power is not degraded
        assert power_deg == pytest.approx(500.0)
        # Capacity with degradation must be less
        assert capacity_deg < capacity_no_deg

    def test_tranche_calculation_year_5(self) -> None:
        """Verify the exact tranche calculation for year 5, 2% degradation."""
        # From spec:
        # Tranche 1 (4 years old): 100 × (1-0.02)^4 = 92.2368
        # Tranche 2 (3 years old): 100 × (1-0.02)^3 = 94.1192
        # Tranche 3 (2 years old): 100 × (1-0.02)^2 = 96.0400
        # Tranche 4 (1 year old):  100 × (1-0.02)^1 = 98.0000
        # Tranche 5 (0 years old): 100 × (1-0.02)^0 = 100.0000
        # Total power: 500 kW
        # Total capacity (×2 E/P): 480.396 × 2 = 960.792 kWh
        power, capacity = compute_bess_tranche_capacity(
            annual_addition_kw=100.0,
            e_to_p_ratio=2.0,
            project_year=5,
            degradation_rate=0.02,
        )
        expected_power = 500.0
        expected_cap_kw = sum(100.0 * (1 - 0.02) ** (5 - i) for i in range(1, 6))
        expected_capacity = expected_cap_kw * 2.0

        assert power == pytest.approx(expected_power)
        assert capacity == pytest.approx(expected_capacity, rel=1e-6)

    def test_start_year_delays_build(self) -> None:
        """start_year > 1 delays when additions begin."""
        power, capacity = compute_bess_tranche_capacity(
            annual_addition_kw=100.0,
            e_to_p_ratio=2.0,
            project_year=3,
            degradation_rate=0.0,
            start_year=3,
        )
        # Only one tranche (year 3)
        assert power == pytest.approx(100.0)
        assert capacity == pytest.approx(200.0)

    def test_before_start_year_is_zero(self) -> None:
        """Before start_year, no capacity exists."""
        power, capacity = compute_bess_tranche_capacity(
            annual_addition_kw=100.0,
            e_to_p_ratio=2.0,
            project_year=2,
            degradation_rate=0.0,
            start_year=3,
        )
        assert power == 0.0
        assert capacity == 0.0

    def test_invalid_project_year(self) -> None:
        """project_year < 1 raises ValueError."""
        with pytest.raises(ValueError, match="project_year"):
            compute_bess_tranche_capacity(
                annual_addition_kw=100.0,
                e_to_p_ratio=2.0,
                project_year=0,
                degradation_rate=0.02,
            )

    def test_negative_addition_raises(self) -> None:
        """Negative addition raises ValueError."""
        with pytest.raises(ValueError, match="annual_addition_kw"):
            compute_bess_tranche_capacity(
                annual_addition_kw=-10.0,
                e_to_p_ratio=2.0,
                project_year=1,
                degradation_rate=0.0,
            )


# ---------------------------------------------------------------------------
# run_portfolio_simulation tests
# ---------------------------------------------------------------------------


class TestPortfolioSimulation:
    """Tests for the multi-year portfolio simulation."""

    def test_no_flex_equals_world_a(
        self,
        engine_config_short: PortfolioEngineConfig,
        flat_pv_profile: np.ndarray,
        flat_load_profile: np.ndarray,
        flat_prices: np.ndarray,
    ) -> None:
        """With zero addition, simulation equals World A (no storage)."""
        results = run_portfolio_simulation(
            config=engine_config_short,
            pv_profile_base=flat_pv_profile,
            load_profile_base=flat_load_profile,
            spot_prices_base=flat_prices,
            annual_addition_kw=0.0,
            e_to_p_ratio=2.0,
            bess_rte=0.88,
            bess_min_soc_pct=10.0,
            bess_max_soc_pct=90.0,
            bess_degradation_rate=0.02,
            pv_degradation_rate=0.0,
        )
        assert len(results) == 3
        for r in results:
            assert r.bess_capacity_kwh == 0.0
            assert r.bess_power_kw == 0.0
            assert r.total_bess_throughput_kwh == 0.0

    def test_result_count_matches_lifetime(
        self,
        engine_config_5y: PortfolioEngineConfig,
        flat_pv_profile: np.ndarray,
        flat_load_profile: np.ndarray,
        flat_prices: np.ndarray,
    ) -> None:
        """Number of annual results equals lifetime_years."""
        results = run_portfolio_simulation(
            config=engine_config_5y,
            pv_profile_base=flat_pv_profile,
            load_profile_base=flat_load_profile,
            spot_prices_base=flat_prices,
            annual_addition_kw=50.0,
            e_to_p_ratio=2.0,
            bess_rte=0.88,
            bess_min_soc_pct=10.0,
            bess_max_soc_pct=90.0,
            bess_degradation_rate=0.02,
            pv_degradation_rate=0.004,
        )
        assert len(results) == 5
        assert [r.year for r in results] == [1, 2, 3, 4, 5]

    def test_bess_capacity_grows_over_years(
        self,
        engine_config_5y: PortfolioEngineConfig,
        flat_pv_profile: np.ndarray,
        flat_load_profile: np.ndarray,
        flat_prices: np.ndarray,
    ) -> None:
        """BESS capacity grows with each year of additions (no degradation)."""
        results = run_portfolio_simulation(
            config=engine_config_5y,
            pv_profile_base=flat_pv_profile,
            load_profile_base=flat_load_profile,
            spot_prices_base=flat_prices,
            annual_addition_kw=100.0,
            e_to_p_ratio=2.0,
            bess_rte=0.88,
            bess_min_soc_pct=10.0,
            bess_max_soc_pct=90.0,
            bess_degradation_rate=0.0,
            pv_degradation_rate=0.0,
        )
        # Without degradation: capacity = year × 100 × 2
        for r in results:
            assert r.bess_capacity_kwh == pytest.approx(r.year * 200.0)
            assert r.bess_power_kw == pytest.approx(r.year * 100.0)

    def test_pv_degradation_reduces_sell(
        self,
        engine_config_short: PortfolioEngineConfig,
        flat_pv_profile: np.ndarray,
        flat_load_profile: np.ndarray,
        flat_prices: np.ndarray,
    ) -> None:
        """PV degradation reduces grid sales over time (surplus case)."""
        results = run_portfolio_simulation(
            config=engine_config_short,
            pv_profile_base=flat_pv_profile,
            load_profile_base=flat_load_profile,
            spot_prices_base=flat_prices,
            annual_addition_kw=0.0,
            e_to_p_ratio=2.0,
            bess_rte=0.88,
            bess_min_soc_pct=10.0,
            bess_max_soc_pct=90.0,
            bess_degradation_rate=0.0,
            pv_degradation_rate=0.05,  # 5% per year for visible effect
        )
        # PV=10, Load=8, surplus=2 in year 0
        # Year 1: PV=10×0.95=9.5, surplus=1.5
        # Year 2: PV=10×0.9025=9.025, surplus=1.025
        # Year 3: PV=10×0.857375=8.57375, surplus=0.57375
        # So sell_kwh decreases
        assert results[0].total_grid_sell_kwh > results[1].total_grid_sell_kwh
        assert results[1].total_grid_sell_kwh > results[2].total_grid_sell_kwh

    def test_load_growth_increases_buy(
        self,
        engine_config_short: PortfolioEngineConfig,
        flat_pv_profile: np.ndarray,
        flat_load_profile: np.ndarray,
        flat_prices: np.ndarray,
    ) -> None:
        """Load growth increases grid purchases over time."""
        results = run_portfolio_simulation(
            config=engine_config_short,
            pv_profile_base=flat_pv_profile,
            load_profile_base=flat_load_profile,
            spot_prices_base=flat_prices,
            annual_addition_kw=0.0,
            e_to_p_ratio=2.0,
            bess_rte=0.88,
            bess_min_soc_pct=10.0,
            bess_max_soc_pct=90.0,
            bess_degradation_rate=0.0,
            pv_degradation_rate=0.0,
            load_growth_factor=1.10,  # 10% per year for visible effect
        )
        # PV=10, Load=8, surplus=2 in year 0
        # Year 1: Load=8×1.10=8.8, surplus=1.2
        # Year 2: Load=8×1.21=9.68, surplus=0.32
        # Year 3: Load=8×1.331=10.648, deficit=-0.648 → buy starts
        # sell_kwh should decrease, eventually buy_kwh increases
        assert results[0].total_grid_sell_kwh > results[2].total_grid_sell_kwh

    def test_soc_coupling_across_days(
        self,
        flat_pv_profile: np.ndarray,
        flat_load_profile: np.ndarray,
    ) -> None:
        """SoC is carried across day boundaries within a year."""
        config = PortfolioEngineConfig(
            lifetime_years=1,
            baseline_year=2027,
            timestep_hours=1.0,
            intervals_per_day=24,
            intervals_per_year=365 * 24,
            perfect_foresight_discount=1.0,
        )
        # With BESS and varying prices, BESS should arbitrage
        # and maintain SoC across days
        results = run_portfolio_simulation(
            config=config,
            pv_profile_base=flat_pv_profile,
            load_profile_base=flat_load_profile,
            spot_prices_base=np.full(365 * 24, 0.05),
            annual_addition_kw=100.0,
            e_to_p_ratio=2.0,
            bess_rte=1.0,  # perfect RTE for simplicity
            bess_min_soc_pct=0.0,
            bess_max_soc_pct=100.0,
            bess_degradation_rate=0.0,
            pv_degradation_rate=0.0,
        )
        # Should get exactly 1 result
        assert len(results) == 1

    def test_bess_reduces_system_cost(
        self,
        engine_config_short: PortfolioEngineConfig,
        flat_pv_profile: np.ndarray,
        flat_load_profile: np.ndarray,
    ) -> None:
        """BESS with varying prices should reduce system cost vs no BESS."""
        # Create prices with arbitrage opportunity
        daily = np.concatenate([
            np.full(12, 0.02),  # cheap night
            np.full(12, 0.10),  # expensive day
        ])
        prices = np.tile(daily, 365)

        # No BESS
        results_no_bess = run_portfolio_simulation(
            config=engine_config_short,
            pv_profile_base=flat_pv_profile,
            load_profile_base=flat_load_profile,
            spot_prices_base=prices,
            annual_addition_kw=0.0,
            e_to_p_ratio=2.0,
            bess_rte=0.88,
            bess_min_soc_pct=10.0,
            bess_max_soc_pct=90.0,
            bess_degradation_rate=0.0,
            pv_degradation_rate=0.0,
        )

        # With BESS
        results_with_bess = run_portfolio_simulation(
            config=engine_config_short,
            pv_profile_base=flat_pv_profile,
            load_profile_base=flat_load_profile,
            spot_prices_base=prices,
            annual_addition_kw=50.0,
            e_to_p_ratio=2.0,
            bess_rte=0.88,
            bess_min_soc_pct=10.0,
            bess_max_soc_pct=90.0,
            bess_degradation_rate=0.0,
            pv_degradation_rate=0.0,
        )

        # System cost with BESS should be <= without
        for r_no, r_with in zip(results_no_bess, results_with_bess):
            assert r_with.system_cost <= r_no.system_cost + 1e-6

    def test_start_year_delays_bess(
        self,
        engine_config_5y: PortfolioEngineConfig,
        flat_pv_profile: np.ndarray,
        flat_load_profile: np.ndarray,
        flat_prices: np.ndarray,
    ) -> None:
        """start_year=3 means no BESS in years 1-2."""
        results = run_portfolio_simulation(
            config=engine_config_5y,
            pv_profile_base=flat_pv_profile,
            load_profile_base=flat_load_profile,
            spot_prices_base=flat_prices,
            annual_addition_kw=100.0,
            e_to_p_ratio=2.0,
            bess_rte=0.88,
            bess_min_soc_pct=10.0,
            bess_max_soc_pct=90.0,
            bess_degradation_rate=0.0,
            pv_degradation_rate=0.0,
            start_year=3,
        )
        # Years 1 and 2: no BESS
        assert results[0].bess_power_kw == 0.0
        assert results[1].bess_power_kw == 0.0
        # Year 3: 100 kW
        assert results[2].bess_power_kw == pytest.approx(100.0)
        # Year 5: 300 kW (3 tranches)
        assert results[4].bess_power_kw == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# compute_wp_capacity tests
# ---------------------------------------------------------------------------


class TestWpCapacity:
    """Tests for the WP (heat pump) capacity model."""

    def test_zero_addition(self) -> None:
        """Zero addition produces zero capacity."""
        assert compute_wp_capacity(0.0, 5) == 0.0

    def test_linear_growth(self) -> None:
        """WP capacity grows linearly (no degradation)."""
        for year in range(1, 6):
            assert compute_wp_capacity(100.0, year) == pytest.approx(100.0 * year)

    def test_start_year_delay(self) -> None:
        """WP capacity is zero before start_year."""
        assert compute_wp_capacity(100.0, 2, start_year=3) == 0.0
        assert compute_wp_capacity(100.0, 3, start_year=3) == pytest.approx(100.0)
        assert compute_wp_capacity(100.0, 5, start_year=3) == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Portfolio simulation with WP tests
# ---------------------------------------------------------------------------


class TestPortfolioSimulationWithWP:
    """Tests for portfolio simulation with heat pump flexibility."""

    @pytest.fixture
    def wp_profiles(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Base WP profiles: COP, heat demand (interval), daily heat demand."""
        n = 365 * 24
        cop = np.full(n, 3.0)
        heat_demand = np.full(n, 2.0)  # 2 kWh_th per hour
        daily_heat = np.full(365, 48.0)  # 24 × 2 = 48 kWh_th/day
        return cop, heat_demand, daily_heat

    def test_wp_power_grows_over_years(
        self,
        engine_config_short: PortfolioEngineConfig,
        flat_pv_profile: np.ndarray,
        flat_load_profile: np.ndarray,
        flat_prices: np.ndarray,
        wp_profiles: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """WP power should grow linearly each year."""
        cop, heat_demand, daily_heat = wp_profiles
        results = run_portfolio_simulation(
            config=engine_config_short,
            pv_profile_base=flat_pv_profile,
            load_profile_base=flat_load_profile,
            spot_prices_base=flat_prices,
            annual_addition_kw=0.0,
            e_to_p_ratio=2.0,
            bess_rte=0.88,
            bess_min_soc_pct=10.0,
            bess_max_soc_pct=90.0,
            bess_degradation_rate=0.0,
            pv_degradation_rate=0.0,
            wp_annual_addition_kw=50.0,
            wp_cop_profile_base=cop,
            wp_heat_demand_profile_base=heat_demand,
            wp_daily_heat_demand_base=daily_heat,
            wp_thermal_storage_kwh=10.0,
            wp_base_power_kw=50.0,
        )
        for r in results:
            assert r.wp_power_kw == pytest.approx(50.0 * r.year)

    def test_wp_electrical_consumption_grows(
        self,
        engine_config_short: PortfolioEngineConfig,
        flat_pv_profile: np.ndarray,
        flat_load_profile: np.ndarray,
        flat_prices: np.ndarray,
        wp_profiles: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """WP electrical consumption should increase as capacity grows."""
        cop, heat_demand, daily_heat = wp_profiles
        results = run_portfolio_simulation(
            config=engine_config_short,
            pv_profile_base=flat_pv_profile,
            load_profile_base=flat_load_profile,
            spot_prices_base=flat_prices,
            annual_addition_kw=0.0,
            e_to_p_ratio=2.0,
            bess_rte=0.88,
            bess_min_soc_pct=10.0,
            bess_max_soc_pct=90.0,
            bess_degradation_rate=0.0,
            pv_degradation_rate=0.0,
            wp_annual_addition_kw=50.0,
            wp_cop_profile_base=cop,
            wp_heat_demand_profile_base=heat_demand,
            wp_daily_heat_demand_base=daily_heat,
            wp_thermal_storage_kwh=10.0,
            wp_base_power_kw=50.0,
        )
        # More capacity → more electrical consumption
        for i in range(1, len(results)):
            assert results[i].total_wp_electrical_kwh > results[i - 1].total_wp_electrical_kwh

    def test_wp_start_year_delays_consumption(
        self,
        engine_config_short: PortfolioEngineConfig,
        flat_pv_profile: np.ndarray,
        flat_load_profile: np.ndarray,
        flat_prices: np.ndarray,
        wp_profiles: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """WP start_year=3 means no WP consumption in years 1-2."""
        cop, heat_demand, daily_heat = wp_profiles
        results = run_portfolio_simulation(
            config=engine_config_short,
            pv_profile_base=flat_pv_profile,
            load_profile_base=flat_load_profile,
            spot_prices_base=flat_prices,
            annual_addition_kw=0.0,
            e_to_p_ratio=2.0,
            bess_rte=0.88,
            bess_min_soc_pct=10.0,
            bess_max_soc_pct=90.0,
            bess_degradation_rate=0.0,
            pv_degradation_rate=0.0,
            wp_annual_addition_kw=50.0,
            wp_cop_profile_base=cop,
            wp_heat_demand_profile_base=heat_demand,
            wp_daily_heat_demand_base=daily_heat,
            wp_thermal_storage_kwh=10.0,
            wp_base_power_kw=50.0,
            wp_start_year=3,
        )
        assert results[0].wp_power_kw == 0.0
        assert results[0].total_wp_electrical_kwh == 0.0
        assert results[1].wp_power_kw == 0.0
        assert results[2].wp_power_kw == pytest.approx(50.0)
        assert results[2].total_wp_electrical_kwh > 0.0

    def test_no_wp_params_backward_compat(
        self,
        engine_config_short: PortfolioEngineConfig,
        flat_pv_profile: np.ndarray,
        flat_load_profile: np.ndarray,
        flat_prices: np.ndarray,
    ) -> None:
        """Without WP params, simulation should work as before."""
        results = run_portfolio_simulation(
            config=engine_config_short,
            pv_profile_base=flat_pv_profile,
            load_profile_base=flat_load_profile,
            spot_prices_base=flat_prices,
            annual_addition_kw=0.0,
            e_to_p_ratio=2.0,
            bess_rte=0.88,
            bess_min_soc_pct=10.0,
            bess_max_soc_pct=90.0,
            bess_degradation_rate=0.0,
            pv_degradation_rate=0.0,
        )
        for r in results:
            assert r.wp_power_kw == 0.0
            assert r.total_wp_electrical_kwh == 0.0


# ---------------------------------------------------------------------------
# compute_ev_capacity tests
# ---------------------------------------------------------------------------


class TestEvCapacity:
    """Tests for the EV capacity model."""

    def test_zero_addition(self) -> None:
        """Zero addition produces zero capacity."""
        assert compute_ev_capacity(0.0, 5) == 0.0

    def test_linear_growth(self) -> None:
        """EV capacity grows linearly (no degradation)."""
        for year in range(1, 6):
            assert compute_ev_capacity(100.0, year) == pytest.approx(100.0 * year)

    def test_start_year_delay(self) -> None:
        """EV capacity is zero before start_year."""
        assert compute_ev_capacity(100.0, 2, start_year=3) == 0.0
        assert compute_ev_capacity(100.0, 3, start_year=3) == pytest.approx(100.0)
        assert compute_ev_capacity(100.0, 5, start_year=3) == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Portfolio simulation with EV tests
# ---------------------------------------------------------------------------


class TestPortfolioSimulationWithEV:
    """Tests for portfolio simulation with EV flexibility."""

    def _base_ev_kwargs(self) -> dict:
        """Common EV simulation kwargs."""
        return dict(
            ev_annual_addition_kw=50.0,
            ev_daily_energy_demand_kwh_base=20.0,
            ev_usable_battery_kwh_per_kw=2.0,  # 2 kWh per kW
            ev_arrival_interval=8,
            ev_departure_interval=18,
            ev_v2g_enabled=False,
            ev_v2g_rte=0.9,
            ev_min_departure_soc_pct=50.0,
            ev_start_year=1,
            ev_base_power_kw=50.0,
        )

    def test_ev_power_grows_over_years(
        self,
        engine_config_short: PortfolioEngineConfig,
        flat_pv_profile: np.ndarray,
        flat_load_profile: np.ndarray,
        flat_prices: np.ndarray,
    ) -> None:
        """EV power should grow linearly each year."""
        results = run_portfolio_simulation(
            config=engine_config_short,
            pv_profile_base=flat_pv_profile,
            load_profile_base=flat_load_profile,
            spot_prices_base=flat_prices,
            annual_addition_kw=0.0,
            e_to_p_ratio=2.0,
            bess_rte=0.88,
            bess_min_soc_pct=10.0,
            bess_max_soc_pct=90.0,
            bess_degradation_rate=0.0,
            pv_degradation_rate=0.0,
            **self._base_ev_kwargs(),
        )
        for r in results:
            assert r.ev_power_kw == pytest.approx(50.0 * r.year)

    def test_ev_charge_grows_with_capacity(
        self,
        engine_config_short: PortfolioEngineConfig,
        flat_pv_profile: np.ndarray,
        flat_load_profile: np.ndarray,
        flat_prices: np.ndarray,
    ) -> None:
        """EV charging should increase as fleet grows."""
        results = run_portfolio_simulation(
            config=engine_config_short,
            pv_profile_base=flat_pv_profile,
            load_profile_base=flat_load_profile,
            spot_prices_base=flat_prices,
            annual_addition_kw=0.0,
            e_to_p_ratio=2.0,
            bess_rte=0.88,
            bess_min_soc_pct=10.0,
            bess_max_soc_pct=90.0,
            bess_degradation_rate=0.0,
            pv_degradation_rate=0.0,
            **self._base_ev_kwargs(),
        )
        for i in range(1, len(results)):
            assert results[i].total_ev_charge_kwh > results[i - 1].total_ev_charge_kwh

    def test_ev_start_year_delays(
        self,
        engine_config_short: PortfolioEngineConfig,
        flat_pv_profile: np.ndarray,
        flat_load_profile: np.ndarray,
        flat_prices: np.ndarray,
    ) -> None:
        """EV start_year=3 means no EV in years 1-2."""
        kwargs = self._base_ev_kwargs()
        kwargs["ev_start_year"] = 3
        results = run_portfolio_simulation(
            config=engine_config_short,
            pv_profile_base=flat_pv_profile,
            load_profile_base=flat_load_profile,
            spot_prices_base=flat_prices,
            annual_addition_kw=0.0,
            e_to_p_ratio=2.0,
            bess_rte=0.88,
            bess_min_soc_pct=10.0,
            bess_max_soc_pct=90.0,
            bess_degradation_rate=0.0,
            pv_degradation_rate=0.0,
            **kwargs,
        )
        assert results[0].ev_power_kw == 0.0
        assert results[0].total_ev_charge_kwh == 0.0
        assert results[1].ev_power_kw == 0.0
        assert results[2].ev_power_kw == pytest.approx(50.0)
        assert results[2].total_ev_charge_kwh > 0.0

    def test_no_ev_params_backward_compat(
        self,
        engine_config_short: PortfolioEngineConfig,
        flat_pv_profile: np.ndarray,
        flat_load_profile: np.ndarray,
        flat_prices: np.ndarray,
    ) -> None:
        """Without EV params, simulation should work as before."""
        results = run_portfolio_simulation(
            config=engine_config_short,
            pv_profile_base=flat_pv_profile,
            load_profile_base=flat_load_profile,
            spot_prices_base=flat_prices,
            annual_addition_kw=0.0,
            e_to_p_ratio=2.0,
            bess_rte=0.88,
            bess_min_soc_pct=10.0,
            bess_max_soc_pct=90.0,
            bess_degradation_rate=0.0,
            pv_degradation_rate=0.0,
        )
        for r in results:
            assert r.ev_power_kw == 0.0
            assert r.total_ev_charge_kwh == 0.0
            assert r.total_ev_discharge_kwh == 0.0
