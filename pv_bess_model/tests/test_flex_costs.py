"""Tests for portfolio.flex_costs – Lifecycle cost calculation."""

from __future__ import annotations

import pytest

from pv_bess_model.config.loader_portfolio import FlexCostConfig, PersonnelStep
from pv_bess_model.portfolio.flex_costs import (
    FlexLifecycleCost,
    compute_flex_lifecycle_cost,
    _compute_personnel_cost,
)


# ---------------------------------------------------------------------------
# Personnel cost step function
# ---------------------------------------------------------------------------


class TestPersonnelCost:
    """Tests for the personnel step function."""

    def test_no_steps(self) -> None:
        """No personnel steps returns 0."""
        assert _compute_personnel_cost([], 1000.0) == 0.0

    def test_below_first_threshold(self) -> None:
        """Below the first threshold returns 0."""
        steps = [PersonnelStep(500.0, 65000.0)]
        assert _compute_personnel_cost(steps, 200.0) == 0.0

    def test_at_threshold(self) -> None:
        """Exactly at threshold triggers the step."""
        steps = [PersonnelStep(500.0, 65000.0)]
        assert _compute_personnel_cost(steps, 500.0) == 65000.0

    def test_between_steps(self) -> None:
        """Between two thresholds uses the lower step."""
        steps = [
            PersonnelStep(500.0, 65000.0),
            PersonnelStep(1500.0, 130000.0),
        ]
        assert _compute_personnel_cost(steps, 1000.0) == 65000.0

    def test_above_all_steps(self) -> None:
        """Above all thresholds uses the highest step."""
        steps = [
            PersonnelStep(500.0, 65000.0),
            PersonnelStep(1500.0, 130000.0),
        ]
        assert _compute_personnel_cost(steps, 5000.0) == 130000.0


# ---------------------------------------------------------------------------
# Lifecycle cost
# ---------------------------------------------------------------------------


class TestComputeFlexLifecycleCost:
    """Tests for the lifecycle cost computation."""

    def test_zero_cost_config(self) -> None:
        """Default (all-zero) cost config returns zero costs."""
        costs = FlexCostConfig()
        result = compute_flex_lifecycle_cost(
            costs=costs,
            annual_addition_kw=500.0,
            lifetime_years=25,
        )
        assert result.cumulative_cost == 0.0
        assert all(c == 0.0 for c in result.annual_total)

    def test_capex_only(self) -> None:
        """CAPEX without OPEX, no learning curve."""
        costs = FlexCostConfig(capex_eur_per_kw=100.0)
        result = compute_flex_lifecycle_cost(
            costs=costs,
            annual_addition_kw=500.0,
            lifetime_years=25,
            start_year=1,
        )
        # 500 kW × 100 EUR/kW = 50,000 EUR per year × 25 years
        assert result.cumulative_cost == pytest.approx(50000.0 * 25)
        assert result.annual_capex[0] == pytest.approx(50000.0)
        assert result.annual_capex[24] == pytest.approx(50000.0)

    def test_capex_with_kwh(self) -> None:
        """CAPEX includes eur_per_kwh for BESS."""
        costs = FlexCostConfig(
            capex_eur_per_kw=250.0,
            capex_eur_per_kwh=200.0,
        )
        result = compute_flex_lifecycle_cost(
            costs=costs,
            annual_addition_kw=500.0,
            lifetime_years=1,
            e_to_p_ratio=2.0,
        )
        # 500 × 250 + 500 × 2 × 200 = 125,000 + 200,000 = 325,000
        assert result.annual_capex[0] == pytest.approx(325000.0)

    def test_capex_fixed(self) -> None:
        """Fixed CAPEX is included per year."""
        costs = FlexCostConfig(capex_fixed_eur=5000.0)
        result = compute_flex_lifecycle_cost(
            costs=costs,
            annual_addition_kw=100.0,
            lifetime_years=3,
        )
        assert result.annual_capex[0] == pytest.approx(5000.0)
        assert result.annual_capex[2] == pytest.approx(5000.0)

    def test_learning_curve(self) -> None:
        """CAPEX decreases with learning curve."""
        costs = FlexCostConfig(
            capex_eur_per_kw=100.0,
            capex_learning_rate_pct=10.0,
        )
        result = compute_flex_lifecycle_cost(
            costs=costs,
            annual_addition_kw=500.0,
            lifetime_years=3,
        )
        base = 500 * 100  # 50,000
        assert result.annual_capex[0] == pytest.approx(base)
        assert result.annual_capex[1] == pytest.approx(base * 0.9)
        assert result.annual_capex[2] == pytest.approx(base * 0.9**2)

    def test_opex_grows_with_cumulative_kw(self) -> None:
        """OPEX grows linearly with cumulative installed power."""
        costs = FlexCostConfig(opex_eur_per_kw=10.0)
        result = compute_flex_lifecycle_cost(
            costs=costs,
            annual_addition_kw=500.0,
            lifetime_years=3,
        )
        # Year 1: 500 kW × 10 = 5,000
        # Year 2: 1000 kW × 10 = 10,000
        # Year 3: 1500 kW × 10 = 15,000
        assert result.annual_opex[0] == pytest.approx(5000.0)
        assert result.annual_opex[1] == pytest.approx(10000.0)
        assert result.annual_opex[2] == pytest.approx(15000.0)

    def test_opex_kwh_with_bess_degradation(self) -> None:
        """OPEX based on kWh accounts for tranche degradation."""
        costs = FlexCostConfig(opex_eur_per_kwh=1.0)
        result = compute_flex_lifecycle_cost(
            costs=costs,
            annual_addition_kw=100.0,
            lifetime_years=2,
            e_to_p_ratio=2.0,
            degradation_rate=0.1,  # 10% per year for easy math
        )
        # Year 1: 100 × 2 × (1-0.1)^0 = 200 kWh → 200 EUR
        assert result.annual_opex[0] == pytest.approx(200.0)
        # Year 2: tranche1 = 100×2×0.9 = 180, tranche2 = 100×2×1.0 = 200
        # total = 380 kWh → 380 EUR
        assert result.annual_opex[1] == pytest.approx(380.0)

    def test_start_year_delays_costs(self) -> None:
        """No costs before start_year."""
        costs = FlexCostConfig(capex_eur_per_kw=100.0, opex_eur_per_kw=10.0)
        result = compute_flex_lifecycle_cost(
            costs=costs,
            annual_addition_kw=500.0,
            lifetime_years=5,
            start_year=3,
        )
        # Years 1-2: no costs
        assert result.annual_capex[0] == 0.0
        assert result.annual_capex[1] == 0.0
        assert result.annual_opex[0] == 0.0
        assert result.annual_opex[1] == 0.0
        # Year 3: first addition
        assert result.annual_capex[2] == pytest.approx(50000.0)
        assert result.annual_opex[2] == pytest.approx(5000.0)
        # Year 4: cumulative 1000 kW
        assert result.annual_opex[3] == pytest.approx(10000.0)

    def test_personnel_steps_over_time(self) -> None:
        """Personnel costs increase as cumulative kW crosses thresholds."""
        costs = FlexCostConfig(
            personnel_steps=[
                PersonnelStep(threshold_kw=0, annual_cost_eur=0),
                PersonnelStep(threshold_kw=200, annual_cost_eur=65000),
                PersonnelStep(threshold_kw=600, annual_cost_eur=130000),
            ],
        )
        result = compute_flex_lifecycle_cost(
            costs=costs,
            annual_addition_kw=100.0,
            lifetime_years=8,
        )
        # Year 1: 100 kW → 0 EUR (below 200)
        assert result.annual_personnel[0] == 0.0
        # Year 2: 200 kW → 65,000 EUR
        assert result.annual_personnel[1] == 65000.0
        # Year 5: 500 kW → 65,000 EUR (still below 600)
        assert result.annual_personnel[4] == 65000.0
        # Year 6: 600 kW → 130,000 EUR
        assert result.annual_personnel[5] == 130000.0

    def test_annual_total_is_sum(self) -> None:
        """Annual total equals CAPEX + OPEX + Personnel."""
        costs = FlexCostConfig(
            capex_eur_per_kw=100.0,
            opex_eur_per_kw=10.0,
            personnel_steps=[PersonnelStep(0, 5000)],
        )
        result = compute_flex_lifecycle_cost(
            costs=costs,
            annual_addition_kw=500.0,
            lifetime_years=2,
        )
        for i in range(2):
            expected = (
                result.annual_capex[i]
                + result.annual_opex[i]
                + result.annual_personnel[i]
            )
            assert result.annual_total[i] == pytest.approx(expected)

    def test_cumulative_equals_sum_of_total(self) -> None:
        """Cumulative cost equals sum of annual totals."""
        costs = FlexCostConfig(
            capex_eur_per_kw=100.0,
            opex_eur_per_kw=10.0,
        )
        result = compute_flex_lifecycle_cost(
            costs=costs,
            annual_addition_kw=500.0,
            lifetime_years=25,
        )
        assert result.cumulative_cost == pytest.approx(sum(result.annual_total))

    def test_zero_addition_rate(self) -> None:
        """Zero addition rate produces zero costs everywhere."""
        costs = FlexCostConfig(
            capex_eur_per_kw=100.0,
            capex_fixed_eur=5000.0,
            opex_eur_per_kw=10.0,
        )
        result = compute_flex_lifecycle_cost(
            costs=costs,
            annual_addition_kw=0.0,
            lifetime_years=10,
        )
        assert result.cumulative_cost == 0.0

    def test_no_kwh_dimension_for_non_bess(self) -> None:
        """Non-BESS flex types: eur_per_kwh has no effect when e_to_p=0."""
        costs = FlexCostConfig(
            capex_eur_per_kw=100.0,
            capex_eur_per_kwh=200.0,  # should have no effect
        )
        result = compute_flex_lifecycle_cost(
            costs=costs,
            annual_addition_kw=500.0,
            lifetime_years=1,
            e_to_p_ratio=0.0,  # non-BESS
        )
        # Only kW component: 500 × 100 = 50,000
        assert result.annual_capex[0] == pytest.approx(50000.0)
