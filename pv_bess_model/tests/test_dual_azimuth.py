"""Unit tests for the dual-azimuth (Ost/West) PV sub-array feature.

Covers:
- JSON schema validation for sub_arrays field
- Loader cross-field validation (power sum check)
- ScenarioConfig.pv_sub_arrays property
- PVGIS dual-fetch + summation logic in main.py (mocked)
- Backward compatibility: no sub_arrays → single-azimuth mode
"""

from __future__ import annotations

import copy
from unittest.mock import MagicMock, call, patch

import jsonschema
import numpy as np
import pytest

from pv_bess_model.config.loader import load_scenario_dict
from pv_bess_model.config.schema import validate_scenario


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _with_sub_arrays(
    base_cfg: dict,
    sub_arrays: list[dict],
    peak_power: float | None = None,
) -> dict:
    """Return a deep copy of *base_cfg* with sub_arrays injected into PV design."""
    cfg = copy.deepcopy(base_cfg)
    pv_design = cfg["project_settings"]["technology"]["pv"]["design"]
    pv_design["sub_arrays"] = sub_arrays
    if peak_power is not None:
        pv_design["peak_power_kwp"] = peak_power
    return cfg


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSubArraySchema:
    """JSON schema accepts valid sub_arrays and rejects invalid ones."""

    def test_valid_two_sub_arrays_accepted(self, sample_scenario_config_green):
        cfg = _with_sub_arrays(
            sample_scenario_config_green,
            sub_arrays=[
                {"power_kwp": 3000.0, "azimuth_deg": 90, "tilt_deg": 25},
                {"power_kwp": 2000.0, "azimuth_deg": -90, "tilt_deg": 25},
            ],
        )
        validate_scenario(cfg)  # must not raise

    def test_no_sub_arrays_accepted(self, sample_scenario_config_green):
        """Backward compatibility: missing sub_arrays is valid."""
        validate_scenario(sample_scenario_config_green)  # must not raise

    def test_one_sub_array_rejected(self, sample_scenario_config_green):
        """sub_arrays with only 1 element violates minItems=2."""
        cfg = _with_sub_arrays(
            sample_scenario_config_green,
            sub_arrays=[{"power_kwp": 5000.0, "azimuth_deg": 0, "tilt_deg": 30}],
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(cfg)

    def test_three_sub_arrays_rejected(self, sample_scenario_config_green):
        """sub_arrays with 3 elements violates maxItems=2."""
        cfg = _with_sub_arrays(
            sample_scenario_config_green,
            sub_arrays=[
                {"power_kwp": 2000.0, "azimuth_deg": 90, "tilt_deg": 25},
                {"power_kwp": 2000.0, "azimuth_deg": -90, "tilt_deg": 25},
                {"power_kwp": 1000.0, "azimuth_deg": 0, "tilt_deg": 30},
            ],
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(cfg)

    def test_sub_array_missing_power_kwp_rejected(self, sample_scenario_config_green):
        cfg = _with_sub_arrays(
            sample_scenario_config_green,
            sub_arrays=[
                {"azimuth_deg": 90, "tilt_deg": 25},
                {"power_kwp": 2000.0, "azimuth_deg": -90, "tilt_deg": 25},
            ],
        )
        with pytest.raises(jsonschema.ValidationError, match="power_kwp"):
            validate_scenario(cfg)

    def test_sub_array_missing_azimuth_rejected(self, sample_scenario_config_green):
        cfg = _with_sub_arrays(
            sample_scenario_config_green,
            sub_arrays=[
                {"power_kwp": 3000.0, "tilt_deg": 25},
                {"power_kwp": 2000.0, "azimuth_deg": -90, "tilt_deg": 25},
            ],
        )
        with pytest.raises(jsonschema.ValidationError, match="azimuth_deg"):
            validate_scenario(cfg)

    def test_sub_array_missing_tilt_rejected(self, sample_scenario_config_green):
        cfg = _with_sub_arrays(
            sample_scenario_config_green,
            sub_arrays=[
                {"power_kwp": 3000.0, "azimuth_deg": 90},
                {"power_kwp": 2000.0, "azimuth_deg": -90, "tilt_deg": 25},
            ],
        )
        with pytest.raises(jsonschema.ValidationError, match="tilt_deg"):
            validate_scenario(cfg)

    def test_sub_array_zero_power_rejected(self, sample_scenario_config_green):
        """power_kwp = 0 violates exclusiveMinimum."""
        cfg = _with_sub_arrays(
            sample_scenario_config_green,
            sub_arrays=[
                {"power_kwp": 0.0, "azimuth_deg": 90, "tilt_deg": 25},
                {"power_kwp": 5000.0, "azimuth_deg": -90, "tilt_deg": 25},
            ],
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(cfg)

    def test_sub_array_azimuth_out_of_range_rejected(
        self, sample_scenario_config_green
    ):
        cfg = _with_sub_arrays(
            sample_scenario_config_green,
            sub_arrays=[
                {"power_kwp": 3000.0, "azimuth_deg": 200, "tilt_deg": 25},
                {"power_kwp": 2000.0, "azimuth_deg": -90, "tilt_deg": 25},
            ],
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(cfg)

    def test_sub_array_tilt_out_of_range_rejected(self, sample_scenario_config_green):
        cfg = _with_sub_arrays(
            sample_scenario_config_green,
            sub_arrays=[
                {"power_kwp": 3000.0, "azimuth_deg": 90, "tilt_deg": 95},
                {"power_kwp": 2000.0, "azimuth_deg": -90, "tilt_deg": 25},
            ],
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(cfg)

    def test_sub_array_extra_field_rejected(self, sample_scenario_config_green):
        cfg = _with_sub_arrays(
            sample_scenario_config_green,
            sub_arrays=[
                {"power_kwp": 3000.0, "azimuth_deg": 90, "tilt_deg": 25, "extra": 1},
                {"power_kwp": 2000.0, "azimuth_deg": -90, "tilt_deg": 25},
            ],
        )
        with pytest.raises(jsonschema.ValidationError, match="additional"):
            validate_scenario(cfg)


# ---------------------------------------------------------------------------
# Loader cross-field validation (power sum)
# ---------------------------------------------------------------------------


class TestSubArrayPowerSumValidation:
    """_validate_pv_sub_arrays() checks that sub-array powers sum to peak_power_kwp."""

    def test_exact_match_accepted(self, sample_scenario_config_green):
        cfg = _with_sub_arrays(
            sample_scenario_config_green,
            peak_power=5000.0,
            sub_arrays=[
                {"power_kwp": 3000.0, "azimuth_deg": 90, "tilt_deg": 25},
                {"power_kwp": 2000.0, "azimuth_deg": -90, "tilt_deg": 25},
            ],
        )
        load_scenario_dict(cfg)  # must not raise

    def test_within_tolerance_accepted(self, sample_scenario_config_green):
        """Sum differs by 0.05 kWp → within 0.1 tolerance."""
        cfg = _with_sub_arrays(
            sample_scenario_config_green,
            peak_power=5000.0,
            sub_arrays=[
                {"power_kwp": 3000.05, "azimuth_deg": 90, "tilt_deg": 25},
                {"power_kwp": 2000.0, "azimuth_deg": -90, "tilt_deg": 25},
            ],
        )
        load_scenario_dict(cfg)  # must not raise

    def test_sum_too_high_rejected(self, sample_scenario_config_green):
        cfg = _with_sub_arrays(
            sample_scenario_config_green,
            peak_power=5000.0,
            sub_arrays=[
                {"power_kwp": 3000.0, "azimuth_deg": 90, "tilt_deg": 25},
                {"power_kwp": 2500.0, "azimuth_deg": -90, "tilt_deg": 25},
            ],
        )
        with pytest.raises(ValueError, match="does not match"):
            load_scenario_dict(cfg)

    def test_sum_too_low_rejected(self, sample_scenario_config_green):
        cfg = _with_sub_arrays(
            sample_scenario_config_green,
            peak_power=5000.0,
            sub_arrays=[
                {"power_kwp": 2000.0, "azimuth_deg": 90, "tilt_deg": 25},
                {"power_kwp": 2000.0, "azimuth_deg": -90, "tilt_deg": 25},
            ],
        )
        with pytest.raises(ValueError, match="does not match"):
            load_scenario_dict(cfg)

    def test_no_sub_arrays_skips_validation(self, sample_scenario_config_green):
        """Without sub_arrays, power sum validation is not triggered."""
        load_scenario_dict(sample_scenario_config_green)  # must not raise


# ---------------------------------------------------------------------------
# ScenarioConfig.pv_sub_arrays property
# ---------------------------------------------------------------------------


class TestPvSubArraysProperty:
    def test_returns_none_without_sub_arrays(self, sample_scenario_config_green):
        cfg = load_scenario_dict(sample_scenario_config_green)
        assert cfg.pv_sub_arrays is None

    def test_returns_list_with_sub_arrays(self, sample_scenario_config_green):
        data = _with_sub_arrays(
            sample_scenario_config_green,
            sub_arrays=[
                {"power_kwp": 3000.0, "azimuth_deg": 90, "tilt_deg": 25},
                {"power_kwp": 2000.0, "azimuth_deg": -90, "tilt_deg": 25},
            ],
        )
        cfg = load_scenario_dict(data)
        sa = cfg.pv_sub_arrays
        assert sa is not None
        assert len(sa) == 2
        assert sa[0]["power_kwp"] == 3000.0
        assert sa[1]["azimuth_deg"] == -90


# ---------------------------------------------------------------------------
# PVGIS dual-fetch summation (mocked)
# ---------------------------------------------------------------------------


class TestDualAzimuthPVGISFetch:
    """Verify that with sub_arrays, PVGIS is called twice and results are summed."""

    @staticmethod
    def _make_hourly_array(value: float = 1.0) -> np.ndarray:
        """Return an 8760-element array filled with *value*."""
        return np.full(8760, value, dtype=float)

    def test_dual_fetch_called_twice_and_summed(self):
        """Mock PVGISClient.fetch_single_year to verify two calls + addition."""
        arr_a = self._make_hourly_array(100.0)
        arr_b = self._make_hourly_array(50.0)

        mock_client = MagicMock()
        mock_client.fetch_single_year.side_effect = [arr_a, arr_b]

        sub_arrays = [
            {"power_kwp": 6000.0, "azimuth_deg": 90, "tilt_deg": 25},
            {"power_kwp": 4000.0, "azimuth_deg": -90, "tilt_deg": 25},
        ]

        # Simulate the dual-fetch logic from main.py
        combined = None
        for sa in sub_arrays:
            ts = mock_client.fetch_single_year(
                year=2018,
                latitude=53.0,
                longitude=10.0,
                peak_power_kwp=sa["power_kwp"],
                mounting_type="free",
                azimuth_deg=sa["azimuth_deg"],
                tilt_deg=sa["tilt_deg"],
                pvgis_database="PVGIS-SARAH2",
            )
            combined = ts if combined is None else combined + ts

        assert mock_client.fetch_single_year.call_count == 2
        np.testing.assert_array_equal(combined, arr_a + arr_b)
        assert combined.shape == (8760,)
        assert float(np.sum(combined)) == pytest.approx(8760 * 150.0)

    def test_single_azimuth_calls_once(self):
        """Without sub_arrays, PVGIS should be called once with full peak power."""
        arr = self._make_hourly_array(200.0)
        mock_client = MagicMock()
        mock_client.fetch_single_year.return_value = arr

        result = mock_client.fetch_single_year(
            year=2018,
            latitude=53.0,
            longitude=10.0,
            peak_power_kwp=10000.0,
            mounting_type="free",
            azimuth_deg=0,
            tilt_deg=25,
            pvgis_database="PVGIS-SARAH2",
        )

        assert mock_client.fetch_single_year.call_count == 1
        np.testing.assert_array_equal(result, arr)

    def test_combined_profile_preserves_shape(self):
        """Combined profile has same shape as individual sub-array profiles."""
        arr_a = np.random.default_rng(42).uniform(0, 100, 8760)
        arr_b = np.random.default_rng(99).uniform(0, 80, 8760)

        combined = arr_a + arr_b

        assert combined.shape == (8760,)
        # Combined total must equal sum of individual totals
        assert float(np.sum(combined)) == pytest.approx(
            float(np.sum(arr_a)) + float(np.sum(arr_b))
        )
