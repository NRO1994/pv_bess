"""Unit tests for pv_bess_model.config.loader.

Covers:
- load_scenario_dict(): valid config, missing file, invalid JSON
- load_price_csv(): correct loading + unit conversion, NaN columns,
  missing columns, too few rows, unknown price unit
- extend_price_timeseries(): repeat-last-year logic, exact sizing
- ScenarioConfig accessor properties
"""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

from pv_bess_model.config.defaults import HOURS_PER_YEAR, MIN_PRICE_TIMESERIES_HOURS
from pv_bess_model.config.loader import (
    PriceData,
    ScenarioConfig,
    extend_price_timeseries,
    load_price_csv,
    load_scenario,
    load_scenario_dict,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_scenario_json(path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_price_csv(
    tmp_path,
    n_rows: int = HOURS_PER_YEAR,
    columns: list[str] | None = None,
    nan_col: str | None = None,
    nan_row: int = 0,
    filename: str = "prices.csv",
) -> object:
    """Write a synthetic price CSV and return its Path."""
    if columns is None:
        columns = ["MID"]
    rng = np.random.default_rng(0)
    lines = ["timestamp," + ",".join(columns)]
    for i in range(n_rows):
        ts = f"2023-01-01T{i:05d}"  # dummy timestamp, not parsed
        values = []
        for col in columns:
            if col == nan_col and i == nan_row:
                values.append("")
            else:
                values.append(f"{rng.uniform(10, 90):.4f}")
        lines.append(ts + "," + ",".join(values))
    p = tmp_path / filename
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# load_scenario_dict
# ---------------------------------------------------------------------------


class TestLoadScenarioDict:
    def test_valid_green_config(self, sample_scenario_config_green):
        cfg = load_scenario_dict(sample_scenario_config_green)
        assert isinstance(cfg, ScenarioConfig)
        assert cfg.name == "EEG_Green_Test"
        assert cfg.operating_mode == "green"
        assert cfg.lifetime_years == 25
        assert cfg.path is None

    def test_valid_grey_config(self, sample_scenario_config_grey):
        cfg = load_scenario_dict(sample_scenario_config_grey)
        assert cfg.operating_mode == "grey"

    def test_raw_dict_preserved(self, sample_scenario_config_green):
        cfg = load_scenario_dict(sample_scenario_config_green)
        # raw is the same dict object (no copying in load_scenario_dict)
        assert (
            cfg.raw["scenario"]["name"]
            == sample_scenario_config_green["scenario"]["name"]
        )

    def test_invalid_config_raises_validation_error(self, sample_scenario_config_green):
        import jsonschema

        bad = copy.deepcopy(sample_scenario_config_green)
        del bad["scenario"]["name"]
        with pytest.raises(jsonschema.ValidationError):
            load_scenario_dict(bad)


# ---------------------------------------------------------------------------
# load_scenario (file-based)
# ---------------------------------------------------------------------------


class TestLoadScenarioFile:
    def test_load_from_valid_file(self, tmp_path, sample_scenario_config_green):
        p = tmp_path / "scenario.json"
        _write_scenario_json(p, sample_scenario_config_green)
        cfg = load_scenario(p)
        assert cfg.name == "EEG_Green_Test"
        assert cfg.path == p.resolve()

    def test_load_from_string_path(self, tmp_path, sample_scenario_config_green):
        p = tmp_path / "scenario.json"
        _write_scenario_json(p, sample_scenario_config_green)
        cfg = load_scenario(str(p))
        assert cfg.name == "EEG_Green_Test"

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_scenario(tmp_path / "does_not_exist.json")

    def test_invalid_json_raises_decode_error(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not: valid json}", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_scenario(p)

    def test_schema_violation_raises_validation_error(self, tmp_path):
        import jsonschema

        p = tmp_path / "bad_scenario.json"
        p.write_text(json.dumps({"scenario": {"name": "x"}}), encoding="utf-8")
        with pytest.raises(jsonschema.ValidationError):
            load_scenario(p)


# ---------------------------------------------------------------------------
# ScenarioConfig accessor properties
# ---------------------------------------------------------------------------


class TestScenarioConfigAccessors:
    @pytest.fixture
    def cfg(self, sample_scenario_config_green) -> ScenarioConfig:
        return load_scenario_dict(sample_scenario_config_green)

    def test_project_settings(self, cfg):
        assert cfg.project_settings["lifetime_years"] == 25

    def test_technology(self, cfg):
        assert "pv" in cfg.technology

    def test_finance(self, cfg):
        assert "leverage_pct" in cfg.finance

    def test_pv(self, cfg):
        assert cfg.pv["design"]["peak_power_kwp"] == 5_000

    def test_bess(self, cfg):
        assert "design_space" in cfg.bess

    def test_grid_connection(self, cfg):
        assert cfg.grid_connection["max_export_kw"] == 4_000

    def test_pv_peak_kwp(self, cfg):
        assert cfg.pv_peak_kwp == 5_000.0

    def test_bess_scale_list(self, cfg):
        assert 0.0 in cfg.bess_scale_pct_list
        assert 40.0 in cfg.bess_scale_pct_list

    def test_e_to_p_ratio_list(self, cfg):
        assert 2.0 in cfg.e_to_p_ratio_hours_list

    def test_mc_enabled_false(self, cfg):
        assert cfg.mc_enabled is False

    def test_mc_enabled_true(self, sample_scenario_config_green):
        data = copy.deepcopy(sample_scenario_config_green)
        data["scenario"].setdefault("monte_carlo", {})["enabled"] = True
        # weight sum must be 1 → reuse single scenario with weight 1
        data["scenario"]["monte_carlo"]["price_scenarios"] = {
            "mid": {"csv_column": "MID", "weight": 1.0}
        }
        cfg = load_scenario_dict(data)
        assert cfg.mc_enabled is True

    def test_price_unit(self, cfg):
        assert cfg.price_unit == "eur_per_mwh"

    def test_price_csv_path(self, cfg):
        assert "day_ahead_prices.csv" in cfg.price_csv_path


# ---------------------------------------------------------------------------
# load_price_csv – happy path
# ---------------------------------------------------------------------------


class TestLoadPriceCSVHappyPath:
    def test_basic_load_single_column(self, tmp_path):
        p = _make_price_csv(tmp_path, n_rows=HOURS_PER_YEAR, columns=["MID"])
        data = load_price_csv(p, required_columns=["MID"], price_unit="eur_per_mwh")
        assert isinstance(data, PriceData)
        assert "MID" in data.columns
        assert len(data.columns["MID"]) == HOURS_PER_YEAR
        assert data.n_hours == HOURS_PER_YEAR

    def test_multiple_columns_loaded(self, tmp_path):
        p = _make_price_csv(
            tmp_path, n_rows=HOURS_PER_YEAR, columns=["LOW", "MID", "HIGH"]
        )
        data = load_price_csv(
            p, required_columns=["LOW", "MID", "HIGH"], price_unit="eur_per_mwh"
        )
        assert set(data.columns.keys()) == {"LOW", "MID", "HIGH"}

    def test_unit_mwh_to_kwh_conversion(self, tmp_path):
        """Prices in €/MWh must be divided by 1000 to yield €/kWh."""
        p = tmp_path / "prices.csv"
        rows = ["timestamp,MID"] + [
            f"2023-01-01T{i:05d},1000.0" for i in range(HOURS_PER_YEAR)
        ]
        p.write_text("\n".join(rows))
        data = load_price_csv(p, required_columns=["MID"], price_unit="eur_per_mwh")
        assert math.isclose(data.columns["MID"][0], 1.0, rel_tol=1e-9)

    def test_unit_kwh_no_conversion(self, tmp_path):
        """Prices already in €/kWh must not be modified."""
        p = tmp_path / "prices.csv"
        rows = ["timestamp,MID"] + [
            f"2023-01-01T{i:05d},0.05" for i in range(HOURS_PER_YEAR)
        ]
        p.write_text("\n".join(rows))
        data = load_price_csv(p, required_columns=["MID"], price_unit="eur_per_kwh")
        assert math.isclose(data.columns["MID"][0], 0.05, rel_tol=1e-9)

    def test_more_than_one_year_accepted(self, tmp_path):
        p = _make_price_csv(tmp_path, n_rows=2 * HOURS_PER_YEAR, columns=["MID"])
        data = load_price_csv(p, required_columns=["MID"], price_unit="eur_per_mwh")
        assert data.n_hours == 2 * HOURS_PER_YEAR

    def test_extra_columns_in_csv_are_ignored(self, tmp_path):
        """Only required columns are extracted; extras are silently ignored."""
        p = _make_price_csv(
            tmp_path, n_rows=HOURS_PER_YEAR, columns=["LOW", "MID", "HIGH"]
        )
        data = load_price_csv(p, required_columns=["MID"], price_unit="eur_per_mwh")
        assert list(data.columns.keys()) == ["MID"]

    def test_negative_prices_preserved(self, tmp_path):
        """Negative spot prices are valid and must not be clipped."""
        p = tmp_path / "prices.csv"
        rows = "timestamp,MID\n"
        rows += "2023-01-01T00:00:00,-50.0\n" * HOURS_PER_YEAR
        p.write_text(rows)
        data = load_price_csv(p, required_columns=["MID"], price_unit="eur_per_mwh")
        assert data.columns["MID"][0] < 0

    def test_get_column(self, tmp_path):
        p = _make_price_csv(tmp_path, n_rows=HOURS_PER_YEAR, columns=["MID"])
        data = load_price_csv(p, required_columns=["MID"], price_unit="eur_per_mwh")
        arr = data.get_column("MID")
        assert isinstance(arr, np.ndarray)

    def test_get_column_missing_raises_key_error(self, tmp_path):
        p = _make_price_csv(tmp_path, n_rows=HOURS_PER_YEAR, columns=["MID"])
        data = load_price_csv(p, required_columns=["MID"], price_unit="eur_per_mwh")
        with pytest.raises(KeyError, match="HIGH"):
            data.get_column("HIGH")


# ---------------------------------------------------------------------------
# load_price_csv – error cases
# ---------------------------------------------------------------------------


class TestLoadPriceCSVErrors:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_price_csv(
                tmp_path / "missing.csv",
                required_columns=["MID"],
                price_unit="eur_per_mwh",
            )

    def test_missing_required_column(self, tmp_path):
        p = _make_price_csv(tmp_path, n_rows=HOURS_PER_YEAR, columns=["MID"])
        with pytest.raises(ValueError, match="HIGH"):
            load_price_csv(
                p, required_columns=["MID", "HIGH"], price_unit="eur_per_mwh"
            )

    def test_error_message_lists_available_columns(self, tmp_path):
        p = _make_price_csv(tmp_path, n_rows=HOURS_PER_YEAR, columns=["MID"])
        with pytest.raises(ValueError, match="MID"):
            load_price_csv(p, required_columns=["MISSING"], price_unit="eur_per_mwh")

    def test_too_few_rows_raises(self, tmp_path):
        p = _make_price_csv(tmp_path, n_rows=100, columns=["MID"])
        with pytest.raises(ValueError, match=str(MIN_PRICE_TIMESERIES_HOURS)):
            load_price_csv(p, required_columns=["MID"], price_unit="eur_per_mwh")

    def test_exactly_8760_rows_accepted(self, tmp_path):
        p = _make_price_csv(tmp_path, n_rows=8760, columns=["MID"])
        data = load_price_csv(p, required_columns=["MID"], price_unit="eur_per_mwh")
        assert data.n_hours == 8760

    def test_nan_in_required_column_raises(self, tmp_path):
        p = _make_price_csv(
            tmp_path,
            n_rows=HOURS_PER_YEAR,
            columns=["MID"],
            nan_col="MID",
            nan_row=42,
        )
        with pytest.raises(ValueError, match="NaN"):
            load_price_csv(p, required_columns=["MID"], price_unit="eur_per_mwh")

    def test_nan_error_names_affected_column(self, tmp_path):
        p = _make_price_csv(
            tmp_path,
            n_rows=HOURS_PER_YEAR,
            columns=["LOW", "MID"],
            nan_col="MID",
            nan_row=0,
        )
        with pytest.raises(ValueError, match="MID"):
            load_price_csv(p, required_columns=["LOW", "MID"], price_unit="eur_per_mwh")

    def test_nan_error_mentions_row_index(self, tmp_path):
        p = _make_price_csv(
            tmp_path,
            n_rows=HOURS_PER_YEAR,
            columns=["MID"],
            nan_col="MID",
            nan_row=100,
        )
        with pytest.raises(ValueError, match="100"):
            load_price_csv(p, required_columns=["MID"], price_unit="eur_per_mwh")

    def test_unknown_price_unit_raises(self, tmp_path):
        p = _make_price_csv(tmp_path, n_rows=HOURS_PER_YEAR, columns=["MID"])
        with pytest.raises(ValueError, match="eur_per_twh"):
            load_price_csv(p, required_columns=["MID"], price_unit="eur_per_twh")

    def test_multiple_nan_columns_all_listed(self, tmp_path):
        """Both affected columns must appear in the error message."""
        p = tmp_path / "prices.csv"
        rows = ["timestamp,LOW,HIGH"]
        for i in range(HOURS_PER_YEAR):
            low = "" if i == 10 else "20.0"
            high = "" if i == 20 else "80.0"
            rows.append(f"ts,{low},{high}")
        p.write_text("\n".join(rows))
        with pytest.raises(ValueError) as exc_info:
            load_price_csv(
                p, required_columns=["LOW", "HIGH"], price_unit="eur_per_mwh"
            )
        msg = str(exc_info.value)
        assert "LOW" in msg
        assert "HIGH" in msg


# ---------------------------------------------------------------------------
# extend_price_timeseries
# ---------------------------------------------------------------------------


class TestExtendPriceTimeseries:
    def test_exact_years_no_extension(self):
        arr = np.ones(3 * HOURS_PER_YEAR)
        out = extend_price_timeseries(
            arr, target_years=3, hours_per_year=HOURS_PER_YEAR
        )
        assert len(out) == 3 * HOURS_PER_YEAR
        np.testing.assert_array_equal(out, arr)

    def test_shorter_input_extends_with_last_year(self):
        year1 = np.full(HOURS_PER_YEAR, 10.0)
        year2 = np.full(HOURS_PER_YEAR, 20.0)
        arr = np.concatenate([year1, year2])
        out = extend_price_timeseries(
            arr, target_years=4, hours_per_year=HOURS_PER_YEAR
        )
        assert len(out) == 4 * HOURS_PER_YEAR
        # First two years are kept intact
        np.testing.assert_array_equal(out[:HOURS_PER_YEAR], year1)
        np.testing.assert_array_equal(out[HOURS_PER_YEAR : 2 * HOURS_PER_YEAR], year2)
        # Years 3 and 4 are copies of year 2 (the last year)
        np.testing.assert_array_equal(
            out[2 * HOURS_PER_YEAR : 3 * HOURS_PER_YEAR], year2
        )
        np.testing.assert_array_equal(out[3 * HOURS_PER_YEAR :], year2)

    def test_single_year_input_repeated(self):
        arr = np.arange(HOURS_PER_YEAR, dtype=float)
        out = extend_price_timeseries(
            arr, target_years=3, hours_per_year=HOURS_PER_YEAR
        )
        assert len(out) == 3 * HOURS_PER_YEAR
        np.testing.assert_array_equal(out[:HOURS_PER_YEAR], arr)
        np.testing.assert_array_equal(out[HOURS_PER_YEAR : 2 * HOURS_PER_YEAR], arr)
        np.testing.assert_array_equal(out[2 * HOURS_PER_YEAR :], arr)

    def test_longer_input_truncated(self):
        arr = np.ones(5 * HOURS_PER_YEAR)
        out = extend_price_timeseries(
            arr, target_years=3, hours_per_year=HOURS_PER_YEAR
        )
        assert len(out) == 3 * HOURS_PER_YEAR

    def test_target_1_year(self):
        arr = np.ones(HOURS_PER_YEAR)
        out = extend_price_timeseries(
            arr, target_years=1, hours_per_year=HOURS_PER_YEAR
        )
        assert len(out) == HOURS_PER_YEAR

    def test_too_short_input_raises(self):
        arr = np.ones(100)
        with pytest.raises(ValueError, match="8760"):
            extend_price_timeseries(arr, target_years=1, hours_per_year=HOURS_PER_YEAR)

    def test_output_length_formula(self):
        """Output must always be exactly target_years * hours_per_year."""
        for n_years in (1, 5, 10, 25):
            arr = np.ones(HOURS_PER_YEAR)
            out = extend_price_timeseries(
                arr, target_years=n_years, hours_per_year=HOURS_PER_YEAR
            )
            assert len(out) == n_years * HOURS_PER_YEAR

    def test_values_preserved_exactly(self):
        """The extension must use the last year's values unchanged."""
        rng = np.random.default_rng(99)
        arr = rng.uniform(-20, 100, HOURS_PER_YEAR)
        out = extend_price_timeseries(
            arr, target_years=3, hours_per_year=HOURS_PER_YEAR
        )
        np.testing.assert_array_equal(out[HOURS_PER_YEAR : 2 * HOURS_PER_YEAR], arr)
        np.testing.assert_array_equal(out[2 * HOURS_PER_YEAR :], arr)
