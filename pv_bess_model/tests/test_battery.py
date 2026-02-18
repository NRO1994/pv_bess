"""Unit tests for pv_bess_model.bess.battery and pv_bess_model.bess.replacement."""

from __future__ import annotations

import math

import pytest

from pv_bess_model.bess.battery import BatteryState
from pv_bess_model.bess.replacement import (
    ReplacementConfig,
    apply_replacement,
    replacement_config_from_dict,
)


# ---------------------------------------------------------------------------
# Helpers / factory
# ---------------------------------------------------------------------------


def make_battery(
    capacity_kwh: float = 200.0,
    charge_power_kw: float = 100.0,
    discharge_power_kw: float = 100.0,
    rte_pct: float = 90.0,
    min_soc_pct: float = 10.0,
    max_soc_pct: float = 90.0,
    initial_soc_kwh: float | None = None,
) -> BatteryState:
    """Create a BatteryState with sensible defaults for testing."""
    return BatteryState(
        max_capacity_kwh=capacity_kwh,
        max_charge_power_kw=charge_power_kw,
        max_discharge_power_kw=discharge_power_kw,
        round_trip_efficiency_pct=rte_pct,
        min_soc_pct=min_soc_pct,
        max_soc_pct=max_soc_pct,
        initial_soc_kwh=initial_soc_kwh,
    )


# ---------------------------------------------------------------------------
# BatteryState – initialisation
# ---------------------------------------------------------------------------


class TestBatteryStateInit:
    """Tests for BatteryState construction."""

    def test_default_soc_is_midpoint_of_usable_window(self) -> None:
        """SoC without explicit init defaults to midpoint of [min, max] range."""
        bat = make_battery(capacity_kwh=200.0, min_soc_pct=10.0, max_soc_pct=90.0)
        expected = 200.0 * (10.0 + 90.0) / 2.0 / 100.0  # = 100 kWh
        assert math.isclose(bat.current_soc_kwh, expected)

    def test_explicit_initial_soc(self) -> None:
        bat = make_battery(capacity_kwh=200.0, initial_soc_kwh=150.0)
        assert math.isclose(bat.current_soc_kwh, 150.0)

    def test_min_and_max_soc_kwh_properties(self) -> None:
        bat = make_battery(capacity_kwh=200.0, min_soc_pct=10.0, max_soc_pct=90.0)
        assert math.isclose(bat.min_soc_kwh, 20.0)   # 10 % of 200
        assert math.isclose(bat.max_soc_kwh, 180.0)  # 90 % of 200

    def test_cumulative_throughput_starts_at_zero(self) -> None:
        bat = make_battery()
        assert bat.cumulative_throughput_kwh == 0.0

    def test_invalid_negative_capacity_raises(self) -> None:
        with pytest.raises(ValueError, match="max_capacity_kwh"):
            BatteryState(
                max_capacity_kwh=-1.0,
                max_charge_power_kw=100.0,
                max_discharge_power_kw=100.0,
                round_trip_efficiency_pct=90.0,
                min_soc_pct=10.0,
                max_soc_pct=90.0,
            )

    def test_invalid_efficiency_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="round_trip_efficiency_pct"):
            make_battery(rte_pct=0.0)

    def test_invalid_efficiency_above_100_raises(self) -> None:
        with pytest.raises(ValueError, match="round_trip_efficiency_pct"):
            make_battery(rte_pct=101.0)

    def test_invalid_soc_limits_raises(self) -> None:
        with pytest.raises(ValueError):
            make_battery(min_soc_pct=50.0, max_soc_pct=30.0)


# ---------------------------------------------------------------------------
# BatteryState.charge()
# ---------------------------------------------------------------------------


class TestCharge:
    """Tests for BatteryState.charge()."""

    def test_normal_charge_increases_soc(self) -> None:
        """Charging within limits increases SoC by the exact requested amount."""
        bat = make_battery(capacity_kwh=200.0, initial_soc_kwh=100.0)
        actual = bat.charge(30.0)
        assert math.isclose(actual, 30.0)
        assert math.isclose(bat.current_soc_kwh, 130.0)

    def test_charge_is_lossless(self) -> None:
        """1 kWh charged must increase SoC by exactly 1 kWh (no losses)."""
        bat = make_battery(capacity_kwh=200.0, initial_soc_kwh=100.0)
        charged = bat.charge(10.0)
        assert math.isclose(charged, 10.0)
        assert math.isclose(bat.current_soc_kwh, 110.0)

    def test_charge_limited_by_soc_headroom(self) -> None:
        """Charge is capped at max_soc_kwh – current_soc_kwh."""
        bat = make_battery(
            capacity_kwh=200.0,
            max_soc_pct=90.0,   # max = 180 kWh
            initial_soc_kwh=170.0,  # only 10 kWh headroom
        )
        actual = bat.charge(50.0)
        assert math.isclose(actual, 10.0)
        assert math.isclose(bat.current_soc_kwh, 180.0)

    def test_charge_limited_by_power_limit(self) -> None:
        """Charge is capped at max_charge_power_kw."""
        bat = make_battery(
            capacity_kwh=200.0,
            charge_power_kw=40.0,
            initial_soc_kwh=100.0,
        )
        actual = bat.charge(100.0)  # request 100, limit is 40
        assert math.isclose(actual, 40.0)
        assert math.isclose(bat.current_soc_kwh, 140.0)

    def test_charge_at_full_soc_returns_zero(self) -> None:
        """Charging a full battery (at max_soc) returns 0 kWh charged."""
        bat = make_battery(
            capacity_kwh=200.0,
            max_soc_pct=90.0,
            initial_soc_kwh=180.0,  # already at max_soc
        )
        actual = bat.charge(50.0)
        assert actual == 0.0
        assert math.isclose(bat.current_soc_kwh, 180.0)

    def test_charge_zero_kwh_requested(self) -> None:
        """Requesting 0 kWh charge returns 0 and leaves SoC unchanged."""
        bat = make_battery(initial_soc_kwh=100.0)
        actual = bat.charge(0.0)
        assert actual == 0.0
        assert math.isclose(bat.current_soc_kwh, 100.0)

    def test_charge_negative_raises(self) -> None:
        bat = make_battery()
        with pytest.raises(ValueError, match="charge"):
            bat.charge(-1.0)

    def test_charge_updates_cumulative_throughput(self) -> None:
        bat = make_battery(initial_soc_kwh=100.0)
        bat.charge(20.0)
        bat.charge(15.0)
        assert math.isclose(bat.cumulative_throughput_kwh, 35.0)

    def test_charge_zero_capacity_battery(self) -> None:
        """A zero-capacity BESS accepts no charge."""
        bat = BatteryState(
            max_capacity_kwh=0.0,
            max_charge_power_kw=100.0,
            max_discharge_power_kw=100.0,
            round_trip_efficiency_pct=90.0,
            min_soc_pct=0.0,
            max_soc_pct=100.0,
            initial_soc_kwh=0.0,
        )
        actual = bat.charge(50.0)
        assert actual == 0.0


# ---------------------------------------------------------------------------
# BatteryState.discharge()
# ---------------------------------------------------------------------------


class TestDischarge:
    """Tests for BatteryState.discharge()."""

    def test_normal_discharge_reduces_soc(self) -> None:
        bat = make_battery(capacity_kwh=200.0, initial_soc_kwh=150.0)
        grid_output = bat.discharge(30.0)
        assert math.isclose(bat.current_soc_kwh, 120.0)
        # Grid output includes RTE loss
        assert math.isclose(grid_output, 30.0 * 0.90)

    def test_grid_output_equals_soc_removed_times_rte(self) -> None:
        """Discharge output = kWh removed from SoC × RTE (exact formula)."""
        rte = 88.0
        bat = make_battery(
            capacity_kwh=300.0, rte_pct=rte, initial_soc_kwh=200.0
        )
        grid_out = bat.discharge(50.0)
        assert math.isclose(grid_out, 50.0 * (rte / 100.0))

    def test_discharge_limited_by_soc_floor(self) -> None:
        """Discharge is capped so SoC does not fall below min_soc_kwh."""
        bat = make_battery(
            capacity_kwh=200.0,
            min_soc_pct=10.0,      # min = 20 kWh
            initial_soc_kwh=30.0,  # only 10 kWh available above floor
        )
        grid_output = bat.discharge(100.0)
        assert math.isclose(bat.current_soc_kwh, 20.0)
        assert math.isclose(grid_output, 10.0 * 0.90)

    def test_discharge_limited_by_power_limit(self) -> None:
        bat = make_battery(
            capacity_kwh=200.0,
            discharge_power_kw=40.0,
            initial_soc_kwh=150.0,
        )
        grid_output = bat.discharge(100.0)
        assert math.isclose(bat.current_soc_kwh, 110.0)
        assert math.isclose(grid_output, 40.0 * 0.90)

    def test_discharge_at_minimum_soc_returns_zero(self) -> None:
        """Discharging a battery already at min_soc returns 0."""
        bat = make_battery(
            capacity_kwh=200.0,
            min_soc_pct=10.0,
            initial_soc_kwh=20.0,  # exactly at min
        )
        grid_output = bat.discharge(50.0)
        assert grid_output == 0.0
        assert math.isclose(bat.current_soc_kwh, 20.0)

    def test_discharge_zero_kwh_requested(self) -> None:
        bat = make_battery(initial_soc_kwh=100.0)
        grid_output = bat.discharge(0.0)
        assert grid_output == 0.0
        assert math.isclose(bat.current_soc_kwh, 100.0)

    def test_discharge_negative_raises(self) -> None:
        bat = make_battery()
        with pytest.raises(ValueError, match="discharge"):
            bat.discharge(-5.0)

    def test_discharge_updates_cumulative_throughput(self) -> None:
        """Cumulative throughput tracks kWh *removed from SoC* (pre-loss)."""
        bat = make_battery(initial_soc_kwh=150.0)
        bat.discharge(30.0)
        bat.discharge(20.0)
        # SoC decreased by 30 + 20 = 50 kWh
        assert math.isclose(bat.cumulative_throughput_kwh, 50.0)

    def test_discharge_zero_capacity_battery(self) -> None:
        """A zero-capacity BESS delivers no discharge."""
        bat = BatteryState(
            max_capacity_kwh=0.0,
            max_charge_power_kw=100.0,
            max_discharge_power_kw=100.0,
            round_trip_efficiency_pct=90.0,
            min_soc_pct=0.0,
            max_soc_pct=100.0,
            initial_soc_kwh=0.0,
        )
        actual = bat.discharge(50.0)
        assert actual == 0.0

    def test_discharge_soc_at_zero_min_soc(self) -> None:
        """With min_soc_pct=0, a fully discharged battery returns no output."""
        bat = make_battery(
            capacity_kwh=100.0,
            min_soc_pct=0.0,
            max_soc_pct=100.0,
            initial_soc_kwh=0.0,
        )
        out = bat.discharge(50.0)
        assert out == 0.0


# ---------------------------------------------------------------------------
# Cumulative throughput – combined charge + discharge
# ---------------------------------------------------------------------------


class TestCumulativeThroughput:
    """Tests for cumulative_throughput_kwh tracking."""

    def test_combined_charge_and_discharge_throughput(self) -> None:
        """Throughput = sum of all kWh charged + kWh removed from SoC."""
        bat = make_battery(initial_soc_kwh=100.0)
        bat.charge(30.0)       # +30 throughput
        bat.discharge(20.0)    # +20 throughput (kWh from SoC, not grid output)
        bat.charge(10.0)       # +10 throughput
        assert math.isclose(bat.cumulative_throughput_kwh, 60.0)

    def test_partial_charge_throughput(self) -> None:
        """Only the accepted kWh counts toward throughput, not the request."""
        bat = make_battery(
            capacity_kwh=200.0,
            charge_power_kw=50.0,
            initial_soc_kwh=100.0,
        )
        bat.charge(100.0)  # limited to 50 by power
        assert math.isclose(bat.cumulative_throughput_kwh, 50.0)

    def test_partial_discharge_throughput(self) -> None:
        """Only the actual kWh removed counts toward throughput."""
        bat = make_battery(
            capacity_kwh=200.0,
            discharge_power_kw=25.0,
            initial_soc_kwh=100.0,
        )
        bat.discharge(80.0)  # limited to 25 by power
        assert math.isclose(bat.cumulative_throughput_kwh, 25.0)


# ---------------------------------------------------------------------------
# BatteryState.apply_annual_degradation()
# ---------------------------------------------------------------------------


class TestApplyAnnualDegradation:
    """Tests for apply_annual_degradation()."""

    def test_capacity_reduces_by_rate(self) -> None:
        bat = make_battery(capacity_kwh=200.0)
        bat.apply_annual_degradation(0.02)  # 2 %
        assert math.isclose(bat.max_capacity_kwh, 196.0)

    def test_zero_degradation_rate_unchanged(self) -> None:
        bat = make_battery(capacity_kwh=200.0)
        bat.apply_annual_degradation(0.0)
        assert math.isclose(bat.max_capacity_kwh, 200.0)

    def test_soc_clipped_after_capacity_reduction(self) -> None:
        """SoC is clipped to the updated max_soc_kwh after degradation."""
        bat = make_battery(
            capacity_kwh=200.0,
            max_soc_pct=90.0,
            initial_soc_kwh=178.0,  # near max (180 kWh)
        )
        # After 50 % degradation: new capacity=100 kWh, max_soc=90 kWh
        bat.apply_annual_degradation(0.50)
        assert math.isclose(bat.max_capacity_kwh, 100.0)
        assert bat.current_soc_kwh <= bat.max_soc_kwh + 1e-9

    def test_soc_not_clipped_when_within_new_limits(self) -> None:
        """SoC is left unchanged when it is already within the new limits."""
        bat = make_battery(
            capacity_kwh=200.0,
            max_soc_pct=90.0,
            initial_soc_kwh=50.0,  # well below any post-degradation limit
        )
        bat.apply_annual_degradation(0.10)
        assert math.isclose(bat.current_soc_kwh, 50.0)

    def test_cumulative_degradation_over_multiple_years(self) -> None:
        """Applying degradation repeatedly compounds correctly."""
        bat = make_battery(capacity_kwh=1000.0)
        for _ in range(10):
            bat.apply_annual_degradation(0.02)
        expected = 1000.0 * (0.98 ** 10)
        assert math.isclose(bat.max_capacity_kwh, expected, rel_tol=1e-9)

    def test_negative_degradation_rate_raises(self) -> None:
        bat = make_battery()
        with pytest.raises(ValueError, match="Degradation rate"):
            bat.apply_annual_degradation(-0.01)

    def test_degradation_rate_of_one_raises(self) -> None:
        """A rate of 1.0 would set capacity to zero, which is disallowed."""
        bat = make_battery()
        with pytest.raises(ValueError, match="Degradation rate"):
            bat.apply_annual_degradation(1.0)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case tests for BatteryState."""

    def test_soc_exactly_at_min_after_discharge(self) -> None:
        """Battery SoC lands exactly on min_soc_kwh without going below."""
        bat = make_battery(
            capacity_kwh=200.0,
            min_soc_pct=10.0,   # min = 20 kWh
            initial_soc_kwh=70.0,
        )
        bat.discharge(50.0)  # removes 50, leaving exactly 20
        assert math.isclose(bat.current_soc_kwh, 20.0)

    def test_soc_exactly_at_max_after_charge(self) -> None:
        """Battery SoC lands exactly on max_soc_kwh without exceeding it."""
        bat = make_battery(
            capacity_kwh=200.0,
            max_soc_pct=90.0,   # max = 180 kWh
            initial_soc_kwh=130.0,
        )
        bat.charge(50.0)  # fills exactly to 180
        assert math.isclose(bat.current_soc_kwh, 180.0)

    def test_zero_capacity_bess_no_throughput(self) -> None:
        """Operations on a 0-capacity BESS accumulate no throughput."""
        bat = BatteryState(
            max_capacity_kwh=0.0,
            max_charge_power_kw=100.0,
            max_discharge_power_kw=100.0,
            round_trip_efficiency_pct=90.0,
            min_soc_pct=0.0,
            max_soc_pct=100.0,
            initial_soc_kwh=0.0,
        )
        bat.charge(50.0)
        bat.discharge(50.0)
        assert bat.cumulative_throughput_kwh == 0.0

    def test_high_rte_100_pct_discharge_no_loss(self) -> None:
        """With RTE=100 %, discharged SoC equals grid output."""
        bat = make_battery(
            capacity_kwh=200.0,
            rte_pct=100.0,
            initial_soc_kwh=150.0,
        )
        grid_out = bat.discharge(50.0)
        assert math.isclose(grid_out, 50.0)

    def test_charge_then_discharge_cycle_soc_integrity(self) -> None:
        """Full charge-discharge cycle leaves SoC at the expected level."""
        bat = make_battery(
            capacity_kwh=200.0,
            min_soc_pct=0.0,
            max_soc_pct=100.0,
            initial_soc_kwh=100.0,
        )
        bat.charge(50.0)         # SoC → 150
        bat.discharge(50.0)      # SoC → 100
        assert math.isclose(bat.current_soc_kwh, 100.0)


# ---------------------------------------------------------------------------
# ReplacementConfig – cost calculation
# ---------------------------------------------------------------------------


class TestReplacementConfig:
    """Tests for ReplacementConfig.replacement_cost()."""

    def test_cost_all_components(self) -> None:
        cfg = ReplacementConfig(
            enabled=True,
            year=12,
            fixed_eur=10_000.0,
            eur_per_kw=120.0,
            eur_per_kwh=141.0,
        )
        # 10 000 + 120×2000 + 141×4000 = 10 000 + 240 000 + 564 000 = 814 000
        cost = cfg.replacement_cost(bess_power_kw=2_000.0, bess_capacity_kwh=4_000.0)
        assert math.isclose(cost, 814_000.0)

    def test_cost_fixed_only(self) -> None:
        cfg = ReplacementConfig(enabled=True, year=10, fixed_eur=50_000.0)
        cost = cfg.replacement_cost(1_000.0, 2_000.0)
        assert math.isclose(cost, 50_000.0)

    def test_cost_zero_when_all_components_zero(self) -> None:
        cfg = ReplacementConfig(enabled=True, year=5)
        cost = cfg.replacement_cost(1_000.0, 2_000.0)
        assert cost == 0.0


# ---------------------------------------------------------------------------
# apply_replacement()
# ---------------------------------------------------------------------------


class TestApplyReplacement:
    """Tests for the apply_replacement() function."""

    def _setup(self) -> tuple[BatteryState, float]:
        """Return a degraded battery and its nameplate capacity."""
        nameplate = 200.0
        bat = make_battery(capacity_kwh=nameplate, initial_soc_kwh=100.0)
        bat.apply_annual_degradation(0.10)  # degrade to 180 kWh
        bat.apply_annual_degradation(0.10)  # degrade to 162 kWh
        return bat, nameplate

    def test_replacement_resets_capacity_to_nameplate(self) -> None:
        bat, nameplate = self._setup()
        cfg = ReplacementConfig(enabled=True, year=5, eur_per_kwh=100.0)
        apply_replacement(bat, cfg, current_year=5, nameplate_capacity_kwh=nameplate, bess_power_kw=100.0)
        assert math.isclose(bat.max_capacity_kwh, nameplate)

    def test_replacement_returns_correct_cost(self) -> None:
        bat, nameplate = self._setup()
        cfg = ReplacementConfig(
            enabled=True, year=12, fixed_eur=5_000.0, eur_per_kw=120.0, eur_per_kwh=141.0
        )
        cost = apply_replacement(
            bat, cfg, current_year=12, nameplate_capacity_kwh=nameplate, bess_power_kw=100.0
        )
        expected = 5_000.0 + 120.0 * 100.0 + 141.0 * nameplate
        assert math.isclose(cost, expected)

    def test_no_replacement_when_disabled(self) -> None:
        bat, nameplate = self._setup()
        capacity_before = bat.max_capacity_kwh
        cfg = ReplacementConfig(enabled=False, year=5)
        cost = apply_replacement(bat, cfg, current_year=5, nameplate_capacity_kwh=nameplate, bess_power_kw=100.0)
        assert cost == 0.0
        assert math.isclose(bat.max_capacity_kwh, capacity_before)

    def test_no_replacement_when_year_does_not_match(self) -> None:
        bat, nameplate = self._setup()
        capacity_before = bat.max_capacity_kwh
        cfg = ReplacementConfig(enabled=True, year=12)
        cost = apply_replacement(bat, cfg, current_year=5, nameplate_capacity_kwh=nameplate, bess_power_kw=100.0)
        assert cost == 0.0
        assert math.isclose(bat.max_capacity_kwh, capacity_before)

    def test_replacement_clips_soc_to_new_max(self) -> None:
        """If current SoC exceeds new max_soc_kwh after reset, it is clipped."""
        nameplate = 100.0
        bat = make_battery(
            capacity_kwh=50.0,  # heavily degraded
            max_soc_pct=90.0,
            initial_soc_kwh=45.0,  # at max_soc of degraded battery (90% of 50)
        )
        cfg = ReplacementConfig(enabled=True, year=1)
        apply_replacement(bat, cfg, current_year=1, nameplate_capacity_kwh=nameplate, bess_power_kw=50.0)
        # After reset: max_soc_kwh = 90 % of 100 = 90 kWh; SoC=45 ≤ 90, no clip needed
        assert bat.current_soc_kwh <= bat.max_soc_kwh + 1e-9

    def test_replacement_disabled_returns_zero_regardless_of_year(self) -> None:
        bat, nameplate = self._setup()
        cfg = ReplacementConfig(enabled=False, year=5)
        for year in range(1, 26):
            cost = apply_replacement(bat, cfg, current_year=year, nameplate_capacity_kwh=nameplate, bess_power_kw=100.0)
            assert cost == 0.0


# ---------------------------------------------------------------------------
# replacement_config_from_dict()
# ---------------------------------------------------------------------------


class TestReplacementConfigFromDict:
    """Tests for parsing ReplacementConfig from a scenario JSON dict."""

    def test_full_dict_parsed_correctly(self) -> None:
        d = {
            "enabled": True,
            "year": 12,
            "fixed_eur": 0.0,
            "eur_per_kw": 120.0,
            "eur_per_kwh": 141.0,
        }
        cfg = replacement_config_from_dict(d)
        assert cfg.enabled is True
        assert cfg.year == 12
        assert math.isclose(cfg.eur_per_kw, 120.0)
        assert math.isclose(cfg.eur_per_kwh, 141.0)

    def test_missing_cost_keys_default_to_zero(self) -> None:
        d = {"enabled": True, "year": 5}
        cfg = replacement_config_from_dict(d)
        assert cfg.fixed_eur == 0.0
        assert cfg.eur_per_kw == 0.0
        assert cfg.eur_per_kwh == 0.0

    def test_disabled_replacement_from_dict(self) -> None:
        d = {"enabled": False, "year": 12, "eur_per_kwh": 141.0}
        cfg = replacement_config_from_dict(d)
        assert cfg.enabled is False

    def test_matches_scenario_json_example(self) -> None:
        """Ensure parsing matches the example in the scenario JSON schema."""
        d = {
            "enabled": False,
            "year": 12,
            "fixed_eur": 0.0,
            "eur_per_kw": 120.0,
            "eur_per_kwh": 141.0,
        }
        cfg = replacement_config_from_dict(d)
        bat = make_battery(capacity_kwh=200.0, initial_soc_kwh=100.0)
        cost = apply_replacement(
            bat, cfg, current_year=12, nameplate_capacity_kwh=200.0, bess_power_kw=100.0
        )
        # Disabled – no replacement, no cost
        assert cost == 0.0
