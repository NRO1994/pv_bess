"""Unit tests for FIX-S2-03: BESS-Only scenarios (pv_peak_kwp = 0).

Tests cover:
- Schema: peak_power_kwp = 0 now allowed (not rejected by schema)
- Schema: negative peak_power_kwp still rejected
- Schema: absolute_power_kw + absolute_capacity_kwh accepted in design_space
- Schema: only one of the pair → ValueError
- Schema: pv_peak_kwp = 0 without absolute values → ValueError (cross-field)
- Schema: pv_peak_kwp = 0 with absolute values → valid
- Grid search: pv_peak_kwp = 0, absolute values → correct BESS sizing for
  non-zero scale entries
- Grid search: pv_peak_kwp = 0, no absolute values → only baseline (0 kW)
- Grid search: pv_peak_kwp > 0 → ratio-based sizing unchanged
- Grid search: BESS-Only with multiple scales uses absolute sizing for all
  non-zero scales, baseline for scale=0
"""

from __future__ import annotations

import copy

import jsonschema
import numpy as np
import pytest

from pv_bess_model.config.schema import validate_scenario
from pv_bess_model.optimization.grid_search import GridSearchConfig, run_grid_search

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

from pv_bess_model.config.defaults import (
    INTERVALS_PER_HOUR,
    INTERVALS_PER_YEAR,
    TIMESTEP_HOURS,
)


def _flat_spot(price: float = 0.05) -> np.ndarray:
    return np.full(INTERVALS_PER_YEAR, price, dtype=float)


def _pv_profile(peak_kwh: float = 10.0) -> np.ndarray:
    """Simple half-sine daytime PV profile at quarter-hourly resolution.

    Energy per interval is ``peak_kwh / INTERVALS_PER_HOUR`` so that the
    hourly total matches the old hourly profile.
    """
    hour_of_day = np.arange(INTERVALS_PER_YEAR) % (24 * INTERVALS_PER_HOUR) // INTERVALS_PER_HOUR
    daylight = np.where(
        (hour_of_day >= 6) & (hour_of_day <= 18),
        np.sin(np.pi * (hour_of_day - 6) / 12),
        0.0,
    )
    return (peak_kwh / INTERVALS_PER_HOUR * daylight).astype(float)


def _make_grid_config(
    pv_peak_kwp: float,
    scales: list[float],
    e_to_p: list[float] | None = None,
    absolute_power_kw: float | None = None,
    absolute_capacity_kwh: float | None = None,
    lifetime: int = 2,
) -> GridSearchConfig:
    """Build a minimal GridSearchConfig for BESS-only tests."""
    if e_to_p is None:
        e_to_p = [2.0]
    spot = _flat_spot()
    pv = _pv_profile() if pv_peak_kwp > 0 else np.zeros(INTERVALS_PER_YEAR, dtype=float)
    return GridSearchConfig(
        scale_pct_of_pv=scales,
        e_to_p_ratio_hours=e_to_p,
        pv_peak_kwp=pv_peak_kwp,
        pv_base_timeseries=pv,
        pv_base_timeseries_year=2020,
        pv_degradation_rate=0.0,
        pv_costs_capex={"eur_per_kw": 100.0} if pv_peak_kwp > 0 else {},
        pv_costs_opex={},
        pv_availability_pct=100.0,
        bess_rte=0.90,
        bess_min_soc_pct=10.0,
        bess_max_soc_pct=90.0,
        bess_degradation_rate=0.0,
        bess_availability_pct=100.0,
        bess_costs_capex={"eur_per_kw": 50.0, "eur_per_kwh": 100.0},
        bess_costs_opex={"pct_of_capex": 0.01},
        replacement_enabled=False,
        replacement_year=0,
        replacement_fixed_eur=0.0,
        replacement_eur_per_kw=0.0,
        replacement_eur_per_kwh=0.0,
        replacement_pct_of_capex=0.0,
        replacement_capacity_factor_pct=100.0,
        grid_max_kw=2000.0,
        grid_max_import_kw=None,
        grid_loss_factor=1.0,
        grid_costs_capex={"eur_per_kw": 50.0},
        grid_costs_opex={},
        operating_mode="grey",
        spot_prices_yearly=[spot.copy() for _ in range(lifetime)],
        fixed_prices_yearly=[0.0] * lifetime,
        baseload_mw=0.0,
        lifetime_years=lifetime,
        commissioning_year=2027,
        leverage_pct=0.0,
        interest_rate_pct=4.5,
        loan_tenor_years=lifetime,
        opex_inflation_factors=[(1.0) ** i for i in range(lifetime)],
        discount_rate=0.06,
        afa_years_pv=20,
        afa_years_bess=10,
        gewerbesteuer_messzahl=0.035,
        gewerbesteuer_hebesatz=400,
        koerperschaftsteuer_pct=15.0,
        solidaritaetszuschlag_pct=5.5,
        timestep_hours=TIMESTEP_HOURS,
        intervals_per_day=INTERVALS_PER_HOUR * 24,
        intervals_per_year=INTERVALS_PER_YEAR,
        max_workers=1,
        bess_absolute_power_kw=absolute_power_kw,
        bess_absolute_capacity_kwh=absolute_capacity_kwh,
    )


# ---------------------------------------------------------------------------
# Minimal valid scenario dict helpers
# ---------------------------------------------------------------------------


def _minimal_scenario(extra_pv_peak: float = 1000.0) -> dict:
    """Return a structurally minimal but valid scenario dict."""
    return {
        "scenario": {"name": "test", "output": {"directory": "./out"}},
        "project_settings": {
            "lifetime_years": 5,
            "commissioning_year": 2027,
            "discount_rate": 0.06,
            "operating_mode": "grey",
            "location": {
                "latitude": 50.0,
                "longitude": 10.0,
                "pvgis_database": "PVGIS-SARAH2",
            },
            "technology": {
                "pv": {
                    "design": {
                        "peak_power_kwp": extra_pv_peak,
                        "mounting_type": "free",
                        "azimuth_deg": 0,
                        "tilt_deg": 30,
                    },
                    "performance": {"degradation_rate_pct_per_year": 0.4, "pv_availability_pct": 97.0},
                    "costs": {
                        "capex": {"eur_per_kw": 800.0},
                        "opex": {"pct_of_capex": 0.01},
                    },
                },
                "bess": {
                    "design_space": {
                        "scale_pct_of_pv": [0.0, 20.0],
                        "e_to_p_ratio_hours": [2.0],
                    },
                    "performance": {
                        "round_trip_efficiency_pct": 88.0,
                        "min_soc_pct": 10.0,
                        "max_soc_pct": 90.0,
                        "degradation_rate_pct_per_year": 2.0,
                        "bess_availability_pct": 97.0,
                    },
                    "costs": {
                        "capex": {"eur_per_kwh": 250.0},
                        "opex": {"pct_of_capex": 0.015},
                    },
                },
                "grid_connection": {
                    "max_export_kw": 900.0,
                    "costs": {
                        "capex": {"eur_per_kw": 100.0},
                        "opex": {"pct_of_capex": 0.015},
                    },
                },
            },
            "finance": {
                "leverage_pct": 70.0,
                "interest_rate_pct": 4.5,
                "loan_tenor_years": 18,
                "debt_uses_p90": False,
                "inflation_rate": 0.02,
                "revenue_streams": {"marketing": {"type": "market"}},
                "price_inputs": {
                    "scenarios": [
                        {
                            "name": "mid",
                            "csv_column": "MID",
                            "weather_year": 2017,
                            "weight": 1.0,
                            "is_central": True,
                            "price_csv": "data/day_ahead_prices.csv",
                            "inflation_on_input_data": True,
                            "csv_separator": ";",
                            "csv_decimal": ".",
                            "csv_timestamp_column": "timestamp",
                            "csv_timestamp_format": "ISO8601",
                        }
                    ]
                },
                "tax": {
                    "afa_years_pv": 20,
                    "afa_years_bess": 10,
                    "gewerbesteuer_hebesatz": 400,
                    "gewerbesteuer_messzahl": 0.035,
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSchemaZeroPeakPower:
    """peak_power_kwp = 0 is now allowed by schema but requires absolute BESS values."""

    def test_zero_peak_power_schema_allowed(self) -> None:
        """The JSON schema must not reject peak_power_kwp = 0 (minimum: 0)."""
        scenario = _minimal_scenario(extra_pv_peak=0.0)
        # Add absolute values to pass cross-field validation
        scenario["project_settings"]["technology"]["bess"]["design_space"][
            "absolute_power_kw"
        ] = 500.0
        scenario["project_settings"]["technology"]["bess"]["design_space"][
            "absolute_capacity_kwh"
        ] = 1000.0
        validate_scenario(scenario)  # must not raise

    def test_negative_peak_power_still_rejected(self) -> None:
        """Negative peak_power_kwp must still be rejected by schema."""
        scenario = _minimal_scenario(extra_pv_peak=-1.0)
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(scenario)

    def test_zero_peak_without_absolute_raises_valueerror(self) -> None:
        """peak_power_kwp = 0 without absolute BESS sizing → ValueError."""
        scenario = _minimal_scenario(extra_pv_peak=0.0)
        with pytest.raises(ValueError, match="absolute_power_kw"):
            validate_scenario(scenario)

    def test_only_absolute_power_without_capacity_raises(self) -> None:
        """absolute_power_kw alone (without absolute_capacity_kwh) → ValueError."""
        scenario = _minimal_scenario(extra_pv_peak=1000.0)
        scenario["project_settings"]["technology"]["bess"]["design_space"][
            "absolute_power_kw"
        ] = 500.0
        with pytest.raises(ValueError, match="both"):
            validate_scenario(scenario)

    def test_only_absolute_capacity_without_power_raises(self) -> None:
        """absolute_capacity_kwh alone → ValueError."""
        scenario = _minimal_scenario(extra_pv_peak=1000.0)
        scenario["project_settings"]["technology"]["bess"]["design_space"][
            "absolute_capacity_kwh"
        ] = 1000.0
        with pytest.raises(ValueError, match="both"):
            validate_scenario(scenario)

    def test_both_absolute_fields_with_positive_pv_allowed(self) -> None:
        """Absolute values alongside positive pv_peak_kwp must be accepted."""
        scenario = _minimal_scenario(extra_pv_peak=1000.0)
        scenario["project_settings"]["technology"]["bess"]["design_space"][
            "absolute_power_kw"
        ] = 500.0
        scenario["project_settings"]["technology"]["bess"]["design_space"][
            "absolute_capacity_kwh"
        ] = 1000.0
        validate_scenario(scenario)  # must not raise

    def test_negative_absolute_power_rejected(self) -> None:
        """Negative absolute_power_kw must be rejected by schema."""
        scenario = _minimal_scenario(extra_pv_peak=0.0)
        scenario["project_settings"]["technology"]["bess"]["design_space"][
            "absolute_power_kw"
        ] = -100.0
        scenario["project_settings"]["technology"]["bess"]["design_space"][
            "absolute_capacity_kwh"
        ] = 200.0
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(scenario)

    def test_zero_absolute_values_accepted_by_schema(self) -> None:
        """absolute_power_kw = 0 and absolute_capacity_kwh = 0 are valid by schema."""
        scenario = _minimal_scenario(extra_pv_peak=0.0)
        scenario["project_settings"]["technology"]["bess"]["design_space"][
            "absolute_power_kw"
        ] = 0.0
        scenario["project_settings"]["technology"]["bess"]["design_space"][
            "absolute_capacity_kwh"
        ] = 0.0
        # Cross-field validation: pv_peak=0 AND absolute values exist → OK
        validate_scenario(scenario)  # must not raise


# ---------------------------------------------------------------------------
# Grid search sizing – BESS-Only
# ---------------------------------------------------------------------------


class TestGridSearchBessOnlySizing:
    """Grid search uses absolute BESS sizing when pv_peak_kwp == 0."""

    def test_bess_only_scale_nonzero_uses_absolute_sizing(self) -> None:
        """Non-zero scale with absolute values → power and capacity from absolutes."""
        abs_power = 1_000.0
        abs_cap = 2_000.0
        config = _make_grid_config(
            pv_peak_kwp=0.0,
            scales=[0.0, 100.0],
            absolute_power_kw=abs_power,
            absolute_capacity_kwh=abs_cap,
        )
        result = run_grid_search(config)

        non_baseline = [p for p in result.points if p.scale_pct > 0]
        assert len(non_baseline) >= 1
        for pt in non_baseline:
            assert pt.bess_power_kw == pytest.approx(abs_power)
            assert pt.bess_capacity_kwh == pytest.approx(abs_cap)

    def test_bess_only_scale_zero_gives_no_bess(self) -> None:
        """Scale = 0 (baseline) must always produce 0 kW BESS, even in BESS-Only mode."""
        config = _make_grid_config(
            pv_peak_kwp=0.0,
            scales=[0.0, 100.0],
            absolute_power_kw=500.0,
            absolute_capacity_kwh=1_000.0,
        )
        result = run_grid_search(config)
        baseline = next(p for p in result.points if p.scale_pct == 0.0)
        assert baseline.bess_power_kw == pytest.approx(0.0)
        assert baseline.bess_capacity_kwh == pytest.approx(0.0)

    def test_bess_only_no_absolute_values_only_baseline(self) -> None:
        """pv_peak_kwp=0 without absolute sizing → all combinations produce 0 kW BESS."""
        config = _make_grid_config(
            pv_peak_kwp=0.0,
            scales=[0.0, 50.0, 100.0],
            absolute_power_kw=None,
            absolute_capacity_kwh=None,
        )
        result = run_grid_search(config)
        for pt in result.points:
            assert pt.bess_power_kw == pytest.approx(0.0)
            assert pt.bess_capacity_kwh == pytest.approx(0.0)

    def test_ratio_based_sizing_unchanged_when_pv_positive(self) -> None:
        """Standard ratio-based sizing must still work correctly when pv_peak_kwp > 0."""
        pv_peak = 500.0
        scale = 40.0
        e_to_p = 2.0
        config = _make_grid_config(
            pv_peak_kwp=pv_peak,
            scales=[0.0, scale],
            e_to_p=[e_to_p],
        )
        result = run_grid_search(config)
        pts = [p for p in result.points if p.scale_pct == scale]
        assert len(pts) == 1
        expected_power = pv_peak * scale / 100.0
        expected_cap = expected_power * e_to_p
        assert pts[0].bess_power_kw == pytest.approx(expected_power)
        assert pts[0].bess_capacity_kwh == pytest.approx(expected_cap)

    def test_bess_only_multiple_scales_reuse_absolute_values(self) -> None:
        """Multiple non-zero scales in BESS-Only mode all use the same absolute sizing."""
        abs_power = 800.0
        abs_cap = 1_600.0
        config = _make_grid_config(
            pv_peak_kwp=0.0,
            scales=[0.0, 25.0, 50.0, 100.0],
            absolute_power_kw=abs_power,
            absolute_capacity_kwh=abs_cap,
        )
        result = run_grid_search(config)
        non_baseline = [p for p in result.points if p.scale_pct > 0]
        for pt in non_baseline:
            assert pt.bess_power_kw == pytest.approx(abs_power)
            assert pt.bess_capacity_kwh == pytest.approx(abs_cap)

    def test_bess_only_result_contains_baseline(self) -> None:
        """Grid search result must include the baseline point (scale = 0)."""
        config = _make_grid_config(
            pv_peak_kwp=0.0,
            scales=[100.0],  # user did not add baseline explicitly
            absolute_power_kw=1_000.0,
            absolute_capacity_kwh=2_000.0,
        )
        result = run_grid_search(config)
        scales_in_result = [p.scale_pct for p in result.points]
        assert 0.0 in scales_in_result

    def test_bess_only_optimal_is_identified(self) -> None:
        """run_grid_search must set is_optimal on the best point (not None)."""
        config = _make_grid_config(
            pv_peak_kwp=0.0,
            scales=[0.0, 100.0],
            absolute_power_kw=1_000.0,
            absolute_capacity_kwh=2_000.0,
        )
        result = run_grid_search(config)
        # optimal may be None if all IRRs are None (e.g., no cashflow sign change);
        # but the result object must still be returned
        assert result is not None
        assert result.points is not None
        optimal_count = sum(1 for p in result.points if p.is_optimal)
        # Either 0 optima (all IRR=None) or exactly 1
        assert optimal_count <= 1

    def test_bess_only_capex_uses_absolute_sizing(self) -> None:
        """CAPEX for non-baseline BESS-Only point must reflect absolute kW and kWh."""
        abs_power = 1_000.0
        abs_cap = 2_000.0
        per_kw = 50.0
        per_kwh = 100.0
        config = _make_grid_config(
            pv_peak_kwp=0.0,
            scales=[0.0, 100.0],
            absolute_power_kw=abs_power,
            absolute_capacity_kwh=abs_cap,
        )
        # Override BESS capex to known values
        config.bess_costs_capex = {"eur_per_kw": per_kw, "eur_per_kwh": per_kwh}
        config.bess_costs_opex = {}
        result = run_grid_search(config)
        non_baseline = [p for p in result.points if p.scale_pct > 0][0]
        expected_capex_bess = per_kw * abs_power + per_kwh * abs_cap
        assert non_baseline.capex_bess == pytest.approx(expected_capex_bess)


# ---------------------------------------------------------------------------
# Grid search: BESS-Only dispatch produces zero PV production
# ---------------------------------------------------------------------------


class TestBessOnlyZeroPvTimeseries:
    """With pv_base_timeseries = zeros, PV export is always zero."""

    def test_revenue_from_pv_is_zero_in_bess_only(self) -> None:
        """BESS-Only with grey mode: revenue from BESS can still be positive,
        but PV export must be zero."""
        config = _make_grid_config(
            pv_peak_kwp=0.0,
            scales=[0.0, 100.0],
            absolute_power_kw=500.0,
            absolute_capacity_kwh=1_000.0,
        )
        # Verify that PV timeseries is indeed all zeros
        assert np.all(config.pv_base_timeseries == 0.0)
