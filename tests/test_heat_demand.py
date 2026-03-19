"""Tests for portfolio.heat_demand – degree-day heat demand and COP curves."""

from __future__ import annotations

import numpy as np
import pytest

from pv_bess_model.config.defaults import (
    DEFAULT_COP_TEMP_COEFFICIENT,
    DEFAULT_HEAT_DEMAND_HEIZGRENZE_C,
    HOURS_PER_YEAR,
    INTERVALS_PER_HOUR,
)
from pv_bess_model.portfolio.heat_demand import (
    compute_cop,
    compute_daily_heat_demand,
    compute_heat_demand,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def constant_temp() -> np.ndarray:
    """Constant 5°C temperature for 8,760 hours."""
    return np.full(HOURS_PER_YEAR, 5.0)


@pytest.fixture
def warm_temp() -> np.ndarray:
    """Constant 20°C – above heating threshold."""
    return np.full(HOURS_PER_YEAR, 20.0)


@pytest.fixture
def cold_temp() -> np.ndarray:
    """Constant -5°C – well below heating threshold."""
    return np.full(HOURS_PER_YEAR, -5.0)


@pytest.fixture
def mixed_temp() -> np.ndarray:
    """Realistic sinusoidal annual temperature profile."""
    hours = np.arange(HOURS_PER_YEAR, dtype=float)
    # Ranges roughly from -5°C (winter) to 25°C (summer)
    return 10.0 + 15.0 * np.sin(2 * np.pi * (hours - 2190) / HOURS_PER_YEAR)


# ---------------------------------------------------------------------------
# compute_heat_demand tests
# ---------------------------------------------------------------------------


class TestComputeHeatDemand:
    """Tests for compute_heat_demand()."""

    def test_output_length(self, constant_temp: np.ndarray) -> None:
        """Output must have 35,040 quarter-hourly values."""
        result = compute_heat_demand(constant_temp, annual_thermal_demand_mwh=15000)
        assert len(result) == HOURS_PER_YEAR * INTERVALS_PER_HOUR

    def test_energy_conservation(self, constant_temp: np.ndarray) -> None:
        """Annual sum must equal the input demand."""
        demand_mwh = 15000.0
        result = compute_heat_demand(constant_temp, annual_thermal_demand_mwh=demand_mwh)
        total_kwh = np.sum(result)
        expected_kwh = demand_mwh * 1000.0
        assert abs(total_kwh - expected_kwh) < 1.0  # < 1 kWh tolerance

    def test_all_above_threshold_zero_demand(self, warm_temp: np.ndarray) -> None:
        """When all temperatures exceed the heating threshold, demand is zero."""
        result = compute_heat_demand(warm_temp, annual_thermal_demand_mwh=15000)
        np.testing.assert_array_equal(result, 0.0)

    def test_all_below_threshold_uniform(self, cold_temp: np.ndarray) -> None:
        """When all temperatures are equally below threshold, demand is uniform."""
        demand_mwh = 15000.0
        result = compute_heat_demand(cold_temp, annual_thermal_demand_mwh=demand_mwh)
        # All values should be equal (uniform distribution)
        expected_per_qh = demand_mwh * 1000.0 / (HOURS_PER_YEAR * INTERVALS_PER_HOUR)
        np.testing.assert_allclose(result, expected_per_qh, rtol=1e-10)

    def test_winter_higher_than_summer(self, mixed_temp: np.ndarray) -> None:
        """Heat demand should be higher in winter than in summer."""
        result = compute_heat_demand(mixed_temp, annual_thermal_demand_mwh=15000)
        # January (hours 0-744) vs July (hours 4344-5088)
        jan_qh = result[:744 * INTERVALS_PER_HOUR]
        jul_qh = result[4344 * INTERVALS_PER_HOUR : 5088 * INTERVALS_PER_HOUR]
        assert np.sum(jan_qh) > np.sum(jul_qh)

    def test_zero_demand(self, constant_temp: np.ndarray) -> None:
        """Zero annual demand should produce all-zero output."""
        result = compute_heat_demand(constant_temp, annual_thermal_demand_mwh=0.0)
        np.testing.assert_array_equal(result, 0.0)

    def test_negative_demand_raises(self, constant_temp: np.ndarray) -> None:
        """Negative annual demand should raise ValueError."""
        with pytest.raises(ValueError, match="annual_thermal_demand_mwh"):
            compute_heat_demand(constant_temp, annual_thermal_demand_mwh=-100.0)

    def test_wrong_length_raises(self) -> None:
        """Wrong temperature array length should raise ValueError."""
        with pytest.raises(ValueError, match="8.?760"):
            compute_heat_demand(np.zeros(100), annual_thermal_demand_mwh=15000)

    def test_custom_heizgrenze(self, constant_temp: np.ndarray) -> None:
        """Custom heating threshold should affect demand distribution."""
        # 5°C temp with 10°C threshold → demand exists
        result_10 = compute_heat_demand(
            constant_temp, annual_thermal_demand_mwh=15000, heizgrenze_c=10.0
        )
        # 5°C temp with 4°C threshold → no demand (temp above threshold)
        result_4 = compute_heat_demand(
            constant_temp, annual_thermal_demand_mwh=15000, heizgrenze_c=4.0
        )
        assert np.sum(result_10) > 0
        np.testing.assert_array_equal(result_4, 0.0)

    def test_non_negative(self, mixed_temp: np.ndarray) -> None:
        """All heat demand values must be non-negative."""
        result = compute_heat_demand(mixed_temp, annual_thermal_demand_mwh=15000)
        assert np.all(result >= 0)


# ---------------------------------------------------------------------------
# compute_cop tests
# ---------------------------------------------------------------------------


class TestComputeCop:
    """Tests for compute_cop()."""

    def test_output_length(self, constant_temp: np.ndarray) -> None:
        """Output must have 35,040 quarter-hourly values."""
        result = compute_cop(constant_temp, cop_nominal=3.5, cop_reference_temp_c=7.0)
        assert len(result) == HOURS_PER_YEAR * INTERVALS_PER_HOUR

    def test_cop_at_reference_temp(self) -> None:
        """COP at reference temperature should equal the nominal COP."""
        temp = np.full(HOURS_PER_YEAR, 7.0)
        result = compute_cop(temp, cop_nominal=3.5, cop_reference_temp_c=7.0)
        np.testing.assert_allclose(result, 3.5, rtol=1e-10)

    def test_cop_increases_with_temperature(self) -> None:
        """COP should increase when temperature is above reference."""
        temp = np.full(HOURS_PER_YEAR, 15.0)
        result = compute_cop(temp, cop_nominal=3.5, cop_reference_temp_c=7.0)
        assert np.all(result > 3.5)

    def test_cop_decreases_below_reference(self) -> None:
        """COP should decrease when temperature is below reference."""
        temp = np.full(HOURS_PER_YEAR, -5.0)
        result = compute_cop(temp, cop_nominal=3.5, cop_reference_temp_c=7.0)
        assert np.all(result < 3.5)

    def test_cop_minimum_clamp(self) -> None:
        """COP should never go below 1.0 even at extreme cold."""
        temp = np.full(HOURS_PER_YEAR, -100.0)
        result = compute_cop(temp, cop_nominal=3.5, cop_reference_temp_c=7.0)
        assert np.all(result >= 1.0)

    def test_cop_formula_exact(self) -> None:
        """Verify exact COP formula: COP_nom × (1 + coeff × (T - T_ref))."""
        temp = np.full(HOURS_PER_YEAR, 12.0)
        cop_nom = 4.0
        t_ref = 7.0
        coeff = DEFAULT_COP_TEMP_COEFFICIENT
        expected = cop_nom * (1.0 + coeff * (12.0 - t_ref))
        result = compute_cop(temp, cop_nominal=cop_nom, cop_reference_temp_c=t_ref)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_wrong_length_raises(self) -> None:
        """Wrong temperature length should raise ValueError."""
        with pytest.raises(ValueError, match="8.?760"):
            compute_cop(np.zeros(100), cop_nominal=3.5, cop_reference_temp_c=7.0)

    def test_zero_cop_raises(self) -> None:
        """COP <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="cop_nominal"):
            compute_cop(np.zeros(HOURS_PER_YEAR), cop_nominal=0.0, cop_reference_temp_c=7.0)

    def test_custom_coefficient(self) -> None:
        """Custom temperature coefficient should affect COP."""
        temp = np.full(HOURS_PER_YEAR, 12.0)
        result_low = compute_cop(
            temp, cop_nominal=3.5, cop_reference_temp_c=7.0, temp_coefficient=0.01
        )
        result_high = compute_cop(
            temp, cop_nominal=3.5, cop_reference_temp_c=7.0, temp_coefficient=0.05
        )
        # Higher coefficient → larger deviation from nominal at same temp
        assert float(np.mean(result_high)) > float(np.mean(result_low))


# ---------------------------------------------------------------------------
# compute_daily_heat_demand tests
# ---------------------------------------------------------------------------


class TestComputeDailyHeatDemand:
    """Tests for compute_daily_heat_demand()."""

    def test_output_length(self, constant_temp: np.ndarray) -> None:
        """Output must have 365 daily values."""
        heat_qh = compute_heat_demand(constant_temp, annual_thermal_demand_mwh=15000)
        daily = compute_daily_heat_demand(heat_qh)
        assert len(daily) == 365

    def test_daily_sum_equals_total(self, constant_temp: np.ndarray) -> None:
        """Sum of daily values should equal total annual demand."""
        demand_mwh = 15000.0
        heat_qh = compute_heat_demand(constant_temp, annual_thermal_demand_mwh=demand_mwh)
        daily = compute_daily_heat_demand(heat_qh)
        assert abs(np.sum(daily) - demand_mwh * 1000.0) < 1.0

    def test_wrong_length_raises(self) -> None:
        """Wrong array length should raise ValueError."""
        with pytest.raises(ValueError, match="35.?040"):
            compute_daily_heat_demand(np.zeros(100))

    def test_uniform_days(self, cold_temp: np.ndarray) -> None:
        """With uniform temperature, all daily demands should be equal."""
        heat_qh = compute_heat_demand(cold_temp, annual_thermal_demand_mwh=15000)
        daily = compute_daily_heat_demand(heat_qh)
        np.testing.assert_allclose(daily, daily[0], rtol=1e-10)
