"""Tests for portfolio.flex_costs – Lifecycle cost calculation."""

from __future__ import annotations

import pytest

from pv_bess_model.config.loader_portfolio import (
    FlexCostConfig,
    PersonnelStep,
    ReplacementConfig,
)
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
        """Annual total equals CAPEX + OPEX + Personnel + Replacement."""
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
                + result.annual_replacement[i]
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


# ---------------------------------------------------------------------------
# Replacement CAPEX
# ---------------------------------------------------------------------------


class TestReplacementCost:
    """Tests for BESS tranche replacement cost calculation."""

    def test_no_replacement_config(self) -> None:
        """No replacement config → zero replacement costs."""
        costs = FlexCostConfig(capex_eur_per_kw=100.0)
        result = compute_flex_lifecycle_cost(
            costs=costs,
            annual_addition_kw=500.0,
            lifetime_years=15,
            e_to_p_ratio=2.0,
        )
        assert all(r == 0.0 for r in result.annual_replacement)

    def test_replacement_timing_single_tranche(self) -> None:
        """Tranche installed in year 1 is replaced in year 1+after_years."""
        repl = ReplacementConfig(
            after_years=5, eur_per_kw=100.0, apply_learning_rate=False,
        )
        costs = FlexCostConfig(replacement=repl)
        result = compute_flex_lifecycle_cost(
            costs=costs,
            annual_addition_kw=500.0,
            lifetime_years=6,
            start_year=1,
        )
        # Years 1-5: no replacement
        for y in range(5):
            assert result.annual_replacement[y] == 0.0
        # Year 6: tranche from year 1 (age=5) → 500 × 100 = 50,000
        assert result.annual_replacement[5] == pytest.approx(50000.0)

    def test_replacement_rolling_multiple_tranches(self) -> None:
        """Multiple tranches get replaced in successive years."""
        repl = ReplacementConfig(
            after_years=3, eur_per_kw=100.0, apply_learning_rate=False,
        )
        costs = FlexCostConfig(replacement=repl)
        result = compute_flex_lifecycle_cost(
            costs=costs,
            annual_addition_kw=200.0,
            lifetime_years=7,
            start_year=1,
        )
        # base cost per tranche = 200 × 100 = 20,000
        # Year 4: tranche 1 (age=3) replaced → 20,000
        assert result.annual_replacement[3] == pytest.approx(20000.0)
        # Year 5: tranche 2 (age=3) replaced → 20,000
        assert result.annual_replacement[4] == pytest.approx(20000.0)
        # Year 7: tranche 1 (age=6, 2nd repl) + tranche 4 (age=3, 1st repl)
        assert result.annual_replacement[6] == pytest.approx(40000.0)

    def test_replacement_with_kwh(self) -> None:
        """Replacement cost includes eur_per_kwh component."""
        repl = ReplacementConfig(
            after_years=5,
            eur_per_kw=100.0,
            eur_per_kwh=50.0,
            apply_learning_rate=False,
        )
        costs = FlexCostConfig(replacement=repl)
        result = compute_flex_lifecycle_cost(
            costs=costs,
            annual_addition_kw=500.0,
            lifetime_years=6,
            e_to_p_ratio=2.0,
            start_year=1,
        )
        # Year 6: 500 × 100 + 500 × 2 × 50 = 50,000 + 50,000 = 100,000
        assert result.annual_replacement[5] == pytest.approx(100000.0)

    def test_replacement_with_fixed_eur(self) -> None:
        """Replacement cost includes fixed component."""
        repl = ReplacementConfig(
            after_years=3, fixed_eur=10000.0, apply_learning_rate=False,
        )
        costs = FlexCostConfig(replacement=repl)
        result = compute_flex_lifecycle_cost(
            costs=costs,
            annual_addition_kw=500.0,
            lifetime_years=4,
            start_year=1,
        )
        # Year 4: one tranche replaced → fixed 10,000
        assert result.annual_replacement[3] == pytest.approx(10000.0)

    def test_replacement_with_learning_rate(self) -> None:
        """Learning rate reduces replacement cost over time."""
        repl = ReplacementConfig(
            after_years=3, eur_per_kw=100.0, apply_learning_rate=True,
        )
        costs = FlexCostConfig(
            capex_learning_rate_pct=10.0,
            replacement=repl,
        )
        result = compute_flex_lifecycle_cost(
            costs=costs,
            annual_addition_kw=500.0,
            lifetime_years=7,
            start_year=1,
        )
        base = 500 * 100  # 50,000
        # Year 4 (idx 3): tranche 1 replaced, years_since_start=3
        assert result.annual_replacement[3] == pytest.approx(base * 0.9**3)
        # Year 5 (idx 4): tranche 2 replaced, years_since_start=4
        assert result.annual_replacement[4] == pytest.approx(base * 0.9**4)

    def test_replacement_without_learning_rate(self) -> None:
        """apply_learning_rate=False keeps replacement cost constant."""
        repl = ReplacementConfig(
            after_years=3, eur_per_kw=100.0, apply_learning_rate=False,
        )
        costs = FlexCostConfig(
            capex_learning_rate_pct=10.0,
            replacement=repl,
        )
        result = compute_flex_lifecycle_cost(
            costs=costs,
            annual_addition_kw=500.0,
            lifetime_years=7,
            start_year=1,
        )
        base = 500 * 100  # 50,000
        # All replacements cost the same (no learning)
        assert result.annual_replacement[3] == pytest.approx(base)
        assert result.annual_replacement[4] == pytest.approx(base)

    def test_replacement_included_in_total(self) -> None:
        """Replacement costs are included in annual_total and cumulative."""
        repl = ReplacementConfig(
            after_years=2, eur_per_kw=100.0, apply_learning_rate=False,
        )
        costs = FlexCostConfig(
            capex_eur_per_kw=50.0,
            opex_eur_per_kw=5.0,
            replacement=repl,
        )
        result = compute_flex_lifecycle_cost(
            costs=costs,
            annual_addition_kw=500.0,
            lifetime_years=4,
            start_year=1,
        )
        for i in range(4):
            expected = (
                result.annual_capex[i]
                + result.annual_opex[i]
                + result.annual_personnel[i]
                + result.annual_replacement[i]
            )
            assert result.annual_total[i] == pytest.approx(expected)
        assert result.cumulative_cost == pytest.approx(sum(result.annual_total))

    def test_replacement_with_start_year(self) -> None:
        """Replacement timing respects start_year offset."""
        repl = ReplacementConfig(
            after_years=3, eur_per_kw=100.0, apply_learning_rate=False,
        )
        costs = FlexCostConfig(replacement=repl)
        result = compute_flex_lifecycle_cost(
            costs=costs,
            annual_addition_kw=500.0,
            lifetime_years=8,
            start_year=3,
        )
        # First tranche installed in year 3, replaced in year 6
        for y in range(5):
            assert result.annual_replacement[y] == 0.0
        assert result.annual_replacement[5] == pytest.approx(50000.0)  # year 6

    def test_replacement_opex_uses_reset_degradation(self) -> None:
        """Cumulative kWh for OPEX uses replacement-reset degradation."""
        repl = ReplacementConfig(
            after_years=2, capacity_factor_pct=100.0,
        )
        costs = FlexCostConfig(
            opex_eur_per_kwh=1.0,
            replacement=repl,
        )
        result = compute_flex_lifecycle_cost(
            costs=costs,
            annual_addition_kw=100.0,
            lifetime_years=3,
            e_to_p_ratio=2.0,
            degradation_rate=0.1,
            start_year=1,
        )
        # Year 3: tranche1 age=2, repl after 2 → effective_age=0, cap_factor=1.0
        # tranche1: 100 × 2 × 1.0 × 0.9^0 = 200
        # tranche2 age=1 → effective_age=1, cap_factor=1.0
        # tranche2: 100 × 2 × 1.0 × 0.9^1 = 180
        # tranche3 age=0 → effective_age=0, cap_factor=1.0
        # tranche3: 100 × 2 × 1.0 × 0.9^0 = 200
        # total = 580
        assert result.annual_opex[2] == pytest.approx(580.0)

    def test_replacement_capacity_factor_affects_opex(self) -> None:
        """capacity_factor_pct < 100 reduces kWh after replacement."""
        repl = ReplacementConfig(
            after_years=2, capacity_factor_pct=80.0,
        )
        costs = FlexCostConfig(
            opex_eur_per_kwh=1.0,
            replacement=repl,
        )
        result = compute_flex_lifecycle_cost(
            costs=costs,
            annual_addition_kw=100.0,
            lifetime_years=3,
            e_to_p_ratio=2.0,
            degradation_rate=0.0,
            start_year=1,
        )
        # Year 3: tranche1 age=2, after_years=2 → n_repl=1, effective_age=0
        # cap_factor=0.8^1=0.8 → 100 × 2 × 0.8 × 1.0 = 160
        # tranche2 age=1 → n_repl=0, effective_age=1 → 100 × 2 × 1.0 = 200
        # tranche3 age=0 → 100 × 2 × 1.0 = 200
        # total = 560
        assert result.annual_opex[2] == pytest.approx(560.0)

    def test_backwards_compatibility_no_replacement(self) -> None:
        """Results without replacement match pre-replacement behavior."""
        costs_no_repl = FlexCostConfig(
            capex_eur_per_kw=100.0,
            opex_eur_per_kwh=1.0,
        )
        costs_none_repl = FlexCostConfig(
            capex_eur_per_kw=100.0,
            opex_eur_per_kwh=1.0,
            replacement=None,
        )
        kwargs = dict(
            annual_addition_kw=500.0,
            lifetime_years=10,
            e_to_p_ratio=2.0,
            degradation_rate=0.02,
        )
        r1 = compute_flex_lifecycle_cost(costs=costs_no_repl, **kwargs)
        r2 = compute_flex_lifecycle_cost(costs=costs_none_repl, **kwargs)
        assert r1.cumulative_cost == pytest.approx(r2.cumulative_cost)
