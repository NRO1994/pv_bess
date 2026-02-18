"""Tests for finance/costs.py – Unified CAPEX/OPEX cost schema.

Reference values from CLAUDE.md example JSON:
  PV   5 000 kWp          : 50 000 + 800 × 5 000           = 4 050 000 €
  BESS 2 000 kW / 4 000 kWh: 50 000 + 100×2 000 + 250×4 000 = 1 250 000 €
  Grid 4 000 kW            : 50 000 + 100 × 4 000           =   450 000 €
  Total                                                       = 5 750 000 €

  PV OPEX: 5 000 + 12×5 000 + 0.003×4 050 000               =    77 150 €/a
  BESS OPEX: 25 000 + 0.015×1 250 000                        =    43 750 €/a
  Grid OPEX: 25 000 + 0.015×450 000                          =    31 750 €/a
"""

from __future__ import annotations

import math

import pytest

from pv_bess_model.finance.costs import (
    calculate_asset_capex,
    calculate_asset_opex,
    calculate_replacement_cost,
    calculate_total_costs,
)


# ---------------------------------------------------------------------------
# CAPEX tests
# ---------------------------------------------------------------------------


class TestAssetCapex:
    """Tests for calculate_asset_capex."""

    def test_pv_capex_reference(self) -> None:
        """PV CAPEX = 50 000 + 800 × 5 000 = 4 050 000 €."""
        result = calculate_asset_capex(
            {"fixed_eur": 50_000.0, "eur_per_kw": 800.0},
            reference_kw=5_000.0,
        )
        assert math.isclose(result, 4_050_000.0)

    def test_bess_capex_reference(self) -> None:
        """BESS CAPEX = 50 000 + 100×2 000 + 250×4 000 = 1 250 000 €."""
        result = calculate_asset_capex(
            {"fixed_eur": 50_000.0, "eur_per_kw": 100.0, "eur_per_kwh": 250.0},
            reference_kw=2_000.0,
            reference_kwh=4_000.0,
        )
        assert math.isclose(result, 1_250_000.0)

    def test_grid_capex_reference(self) -> None:
        """Grid CAPEX = 50 000 + 100 × 4 000 = 450 000 €."""
        result = calculate_asset_capex(
            {"fixed_eur": 50_000.0, "eur_per_kw": 100.0},
            reference_kw=4_000.0,
        )
        assert math.isclose(result, 450_000.0)

    def test_missing_fields_treated_as_zero(self) -> None:
        """Empty config dict should yield 0 CAPEX."""
        assert calculate_asset_capex({}, reference_kw=5_000.0) == 0.0

    def test_partial_fields_missing(self) -> None:
        """Only fixed_eur set; eur_per_kw and eur_per_kwh default to 0."""
        result = calculate_asset_capex(
            {"fixed_eur": 10_000.0},
            reference_kw=999.0,
            reference_kwh=999.0,
        )
        assert math.isclose(result, 10_000.0)

    def test_zero_reference_kw(self) -> None:
        """Zero reference kW should only return fixed component."""
        result = calculate_asset_capex(
            {"fixed_eur": 50_000.0, "eur_per_kw": 800.0},
            reference_kw=0.0,
        )
        assert math.isclose(result, 50_000.0)


# ---------------------------------------------------------------------------
# OPEX tests
# ---------------------------------------------------------------------------


class TestAssetOpex:
    """Tests for calculate_asset_opex."""

    def test_pv_opex_reference(self) -> None:
        """PV OPEX = 5 000 + 12×5 000 + 0.003×4 050 000 = 77 150 €/a."""
        capex_pv = 4_050_000.0
        result = calculate_asset_opex(
            {"fixed_eur": 5_000.0, "eur_per_kw": 12.0, "pct_of_capex": 0.003},
            reference_kw=5_000.0,
            reference_kwh=0.0,
            asset_capex=capex_pv,
        )
        expected = 5_000.0 + 12.0 * 5_000.0 + 0.003 * 4_050_000.0
        assert math.isclose(expected, 77_150.0)
        assert math.isclose(result, 77_150.0)

    def test_bess_opex_reference(self) -> None:
        """BESS OPEX = 25 000 + 0.015×1 250 000 = 43 750 €/a."""
        capex_bess = 1_250_000.0
        result = calculate_asset_opex(
            {"fixed_eur": 25_000.0, "pct_of_capex": 0.015},
            reference_kw=2_000.0,
            reference_kwh=4_000.0,
            asset_capex=capex_bess,
        )
        expected = 25_000.0 + 0.015 * 1_250_000.0
        assert math.isclose(expected, 43_750.0)
        assert math.isclose(result, 43_750.0)

    def test_grid_opex_reference(self) -> None:
        """Grid OPEX = 25 000 + 0.015×450 000 = 31 750 €/a."""
        capex_grid = 450_000.0
        result = calculate_asset_opex(
            {"fixed_eur": 25_000.0, "pct_of_capex": 0.015},
            reference_kw=4_000.0,
            reference_kwh=0.0,
            asset_capex=capex_grid,
        )
        expected = 25_000.0 + 0.015 * 450_000.0
        assert math.isclose(expected, 31_750.0)
        assert math.isclose(result, 31_750.0)

    def test_pct_of_capex_uses_asset_capex_not_total(self) -> None:
        """pct_of_capex must reference this asset's CAPEX, not the project total."""
        asset_capex = 1_000_000.0
        result = calculate_asset_opex(
            {"pct_of_capex": 0.01},
            reference_kw=0.0,
            reference_kwh=0.0,
            asset_capex=asset_capex,
        )
        assert math.isclose(result, 10_000.0)

    def test_missing_fields_treated_as_zero(self) -> None:
        """Empty OPEX config should yield 0."""
        result = calculate_asset_opex(
            {}, reference_kw=5_000.0, reference_kwh=4_000.0, asset_capex=1_000_000.0,
        )
        assert result == 0.0


# ---------------------------------------------------------------------------
# Total costs tests
# ---------------------------------------------------------------------------


class TestTotalCosts:
    """Tests for calculate_total_costs using conftest fixtures."""

    def test_total_capex_reference(
        self, sample_capex_config: dict, sample_opex_config: dict,
    ) -> None:
        """Total CAPEX = 4 050 000 + 1 250 000 + 450 000 = 5 750 000 €."""
        tc = calculate_total_costs(
            sample_capex_config, sample_opex_config,
            pv_peak_kwp=5_000.0, bess_power_kw=2_000.0,
            bess_capacity_kwh=4_000.0, grid_max_export_kw=4_000.0,
        )
        assert math.isclose(tc.capex_pv, 4_050_000.0)
        assert math.isclose(tc.capex_bess, 1_250_000.0)
        assert math.isclose(tc.capex_grid, 450_000.0)
        assert math.isclose(tc.capex_other, 0.0)
        assert math.isclose(tc.capex_total, 5_750_000.0)

    def test_total_opex_reference(
        self, sample_capex_config: dict, sample_opex_config: dict,
    ) -> None:
        """Total base OPEX = 77 150 + 43 750 + 31 750 = 152 650 €/a."""
        tc = calculate_total_costs(
            sample_capex_config, sample_opex_config,
            pv_peak_kwp=5_000.0, bess_power_kw=2_000.0,
            bess_capacity_kwh=4_000.0, grid_max_export_kw=4_000.0,
        )
        assert math.isclose(tc.opex_pv, 77_150.0)
        assert math.isclose(tc.opex_bess, 43_750.0)
        assert math.isclose(tc.opex_grid, 31_750.0)
        expected_total = 77_150.0 + 43_750.0 + 31_750.0
        assert math.isclose(tc.opex_total, expected_total)

    def test_zero_bess_produces_pv_only_baseline(
        self, sample_capex_config: dict, sample_opex_config: dict,
    ) -> None:
        """Scale 0 % → BESS power/capacity = 0 → BESS CAPEX = fixed only."""
        tc = calculate_total_costs(
            sample_capex_config, sample_opex_config,
            pv_peak_kwp=5_000.0, bess_power_kw=0.0,
            bess_capacity_kwh=0.0, grid_max_export_kw=4_000.0,
        )
        assert math.isclose(tc.capex_bess, 50_000.0)
        assert tc.capex_total < 5_750_000.0

    def test_other_asset_empty_config(
        self, sample_capex_config: dict, sample_opex_config: dict,
    ) -> None:
        """Empty 'other' asset config contributes 0 to totals."""
        tc = calculate_total_costs(
            sample_capex_config, sample_opex_config,
            pv_peak_kwp=5_000.0, bess_power_kw=2_000.0,
            bess_capacity_kwh=4_000.0, grid_max_export_kw=4_000.0,
        )
        assert tc.capex_other == 0.0
        assert tc.opex_other == 0.0


# ---------------------------------------------------------------------------
# BESS replacement cost tests
# ---------------------------------------------------------------------------


class TestReplacementCost:
    """Tests for calculate_replacement_cost."""

    def test_replacement_cost_reference(self) -> None:
        """Replacement: 0 (fixed) + 120×2 000 + 141×4 000 = 804 000 €."""
        result = calculate_replacement_cost(
            {"fixed_eur": 0.0, "eur_per_kw": 120.0, "eur_per_kwh": 141.0},
            bess_power_kw=2_000.0,
            bess_capacity_kwh=4_000.0,
            bess_capex=1_250_000.0,
        )
        expected = 0.0 + 120.0 * 2_000.0 + 141.0 * 4_000.0
        assert math.isclose(expected, 804_000.0)
        assert math.isclose(result, 804_000.0)

    def test_replacement_with_pct_of_capex(self) -> None:
        """pct_of_capex in replacement references original BESS CAPEX."""
        bess_capex = 1_250_000.0
        result = calculate_replacement_cost(
            {"pct_of_capex": 0.10},
            bess_power_kw=2_000.0,
            bess_capacity_kwh=4_000.0,
            bess_capex=bess_capex,
        )
        assert math.isclose(result, 0.10 * bess_capex)

    def test_replacement_empty_config(self) -> None:
        """Empty replacement config yields 0 cost."""
        result = calculate_replacement_cost(
            {}, bess_power_kw=2_000.0, bess_capacity_kwh=4_000.0, bess_capex=1_250_000.0,
        )
        assert result == 0.0
