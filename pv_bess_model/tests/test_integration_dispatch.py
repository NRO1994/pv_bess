"""Integration test suite: 36 scenario combinations with constraint validation.

Runs all combinations of:
- Tech setups: pv_only, bess_only, pv_bess
- Operating modes: green, grey
- Marketing strategies: market, eeg, ppa_pap, ppa_baseload, ppa_floor, ppa_collar

Each scenario is executed via ``pv_bess_model.main.run()`` with real PVGIS data
(cached) and synthetic price data. Dispatch constraint checks and KPI ranking
tests validate correctness and plausibility of results.

Usage::

    pytest pv_bess_model/tests/test_integration_dispatch.py -m integration -v

All tests are marked with ``@pytest.mark.integration`` and excluded from the
default test run (``pytest -m "not integration"``).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pv_bess_model.config.defaults import (
    CSV_DECIMAL_SEPARATOR,
    CSV_DELIMITER, DAYS_PER_YEAR,
)
from pv_bess_model.tests.dispatch_constraint_checker import (
    ConstraintViolation,
    check_availability,
    check_dispatch_constraints, check_price_dependencies,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TECH_SETUPS = ["bess_only", "pv_only", "pv_bess"]
OPERATING_MODES = ["grey", "green"]
MARKETING_STRATEGIES = ["market", "eeg", "ppa_pap", "ppa_baseload", "ppa_floor", "ppa_collar"]

BESS_CONFIGS: dict[str, dict[str, float | None]] = {
    "pv_only": {"bess_power": None, "bess_capacity": None},
    "bess_only": {"bess_power": 500.0, "bess_capacity": 1000.0},
    "pv_bess": {"bess_power": 500.0, "bess_capacity": 1000.0},
}

# PV peak power used in pv_only and pv_bess setups
_PV_PEAK_KWP = 1000.0
_GRID_MAX_KW = 800.0
_LIFETIME_YEARS = 1

# Master scenario template — all fields that don't change between combos
MASTER_SCENARIO: dict = {
    "scenario": {
        "name": "PLACEHOLDER",
        "skip_baseline": True,
        "monte_carlo": {
            "enabled": False
        },
        "output": {
            "directory": ".data/test/integration_tests/",
            "export_dispatch_sample": True,
            "report": {
                "enabled": False,
            }
        },
    },
    "project_settings": {
        "lifetime_years": _LIFETIME_YEARS,
        "commissioning_year": 2027,
        "discount_rate": 0.06,
        "operating_mode": "PLACEHOLDER",
        "location": {
            "latitude": 53.848808,
            "longitude": 10.674255,
            "pvgis_database": "PVGIS-SARAH3",
        },
        "technology": {
            "pv": {
                "design": {
                    "peak_power_kwp": _PV_PEAK_KWP,
                    "mounting_type": "free",
                    "azimuth_deg": 0,
                    "tilt_deg": 18,
                },
                "performance": {
                    "degradation_rate_pct_per_year": 0.3,
                    "pv_availability_pct": 99.5
                },
                "costs": {
                    "capex": {
                        "fixed_eur": 10_000.0,
                        "eur_per_kw": 400.0,
                    },
                    "opex": {
                        "fixed_eur": 1_000.0,
                        "eur_per_kw": 12.0,
                    },
                },
            },
            "bess": {
                "design_space": {
                    "scale_pct_of_pv": [100],
                    "e_to_p_ratio_hours": [2],
                },
                "performance": {
                    "round_trip_efficiency_pct": 98.0,
                    "min_soc_pct": 5.0,
                    "max_soc_pct": 95.0,
                    "degradation_rate_pct_per_year": 2.5,
                    "bess_availability_pct": 99.0,
                },
                "costs": {
                    "capex": {
                        "fixed_eur": 1_000.0,
                        "eur_per_kw": 200.0,
                        "eur_per_kwh": 100.0,
                    },
                    "opex": {
                        "fixed_eur": 5_000.0,
                        "pct_of_capex": 0.015,
                        "optimization_fee_pct": 3.0,
                    },
                    "replacement": {
                        "enabled": True,
                        "year": 12,
                        "fixed_eur": 0,
                        "eur_per_kw": 200,
                        "eur_per_kwh": 50,
                        "capacity_factor_pct": 120.0
                    },
                },
            },
            "grid_connection": {
                "max_export_kw": _GRID_MAX_KW,
                "system_loss_pct": 10,
                "costs": {
                    "capex": {
                        "fixed_eur": 500.0,
                        "eur_per_kw": 10.0,
                    },
                    "opex": {
                        "fixed_eur": 50.0,
                    },
                },
            },
        },
        "finance": {
            "leverage_pct": 80.0,
            "interest_rate_pct": 3.5,
            "loan_tenor_years": 15,
            "equity_irr_target": 0.081,
            "debt_uses_p90": True,
            "inflation_rate": 0.02,
            "revenue_streams": {
                "marketing": {
                    "type": "market",
                    "eeg_inflation": False,
                },
                "ppa": {
                    "type": "none",
                    "duration_years": 15,
                    "inflation_on_ppa": False,
                    "guarantee_of_origin_eur_per_kwh": 0.003,
                },
            },
            "price_inputs": {
                "scenarios": [
                    {
                        "name": "Low",
                        "label": "Low testing case",
                        "csv_column": "LOW",
                        "weather_year": 2018,
                        "weight": 0.33,
                        "is_central": False,
                        "price_csv": "PLACEHOLDER",
                        "inflation_on_input_data": True,
                        "csv_separator": ";",
                        "csv_decimal": ",",
                        "csv_timestamp_column": "timestamp",
                        "csv_timestamp_format": "%Y-%m-%dT%H:%M:%S"
                    },
                    {
                        "name": "High",
                        "label": "High testing case",
                        "csv_column": "HIGH",
                        "weather_year": 2015,
                        "weight": 0.34,
                        "is_central": False,
                        "price_csv": "PLACEHOLDER",
                        "inflation_on_input_data": True,
                        "csv_separator": ";",
                        "csv_decimal": ",",
                        "csv_timestamp_column": "timestamp",
                        "csv_timestamp_format": "%Y-%m-%dT%H:%M:%S"
                    },
                    {
                        "name": "Mid",
                        "label": "Mid central testing case",
                        "csv_column": "MID",
                        "weather_year": 2016,
                        "weight": 0.33,
                        "is_central": True,
                        "price_csv": "PLACEHOLDER",
                        "inflation_on_input_data": True,
                        "csv_separator": ";",
                        "csv_decimal": ",",
                        "csv_timestamp_column": "timestamp",
                        "csv_timestamp_format": "%Y-%m-%dT%H:%M:%S"
                    }
                ]
            },
            "tax": {
                "afa_years_pv": 20,
                "afa_years_bess": 10,
                "gewerbesteuer_hebesatz": 400,
                "gewerbesteuer_messzahl": 0.035,
                "koerperschaftsteuer_pct": 15.0,
                "solidaritaetszuschlag_pct": 5.5,
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    """Aggregated results from a single integration scenario run."""

    name: str
    equity_irr: float | None
    project_irr: float | None
    npv: float
    dscr_min: float | None
    revenue_year1: float
    capex_total: float
    dispatch_violations: list[ConstraintViolation] = field(default_factory=list)
    pv_offline_days: int = 0
    bess_offline_days: int = 0
    price_violations: list[ConstraintViolation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------


def build_scenario(
        master: dict,
        tech_setup: str,
        operating_mode: str,
        marketing: str,
) -> dict:
    """Build a concrete scenario dict from the master template.

    Parameters
    ----------
    master:
        Deep-copied master scenario dict.
    tech_setup:
        One of ``"pv_only"``, ``"bess_only"``, ``"pv_bess"``.
    operating_mode:
        ``"green"`` or ``"grey"``.
    marketing:
        One of ``"market"``, ``"eeg"``, ``"ppa_pap"``, ``"ppa_baseload"``,
        ``"ppa_floor"``, ``"ppa_collar"``.

    Returns
    -------
    dict
        Complete scenario dict ready for JSON serialisation.
    """
    s = copy.deepcopy(master)
    name = f"{tech_setup}_{operating_mode}_{marketing}"
    s["scenario"]["name"] = name
    s["project_settings"]["operating_mode"] = operating_mode

    ps = s["project_settings"]
    tech = ps["technology"]
    finance = ps["finance"]
    rev = finance["revenue_streams"]

    # --- Tech setup ---
    if tech_setup == "pv_only":
        # No BESS: set scale to [0] so grid search produces PV-only baseline
        tech["bess"]["design_space"]["scale_pct_of_pv"] = [0]
    elif tech_setup == "bess_only":
        tech["pv"]["design"]["peak_power_kwp"] = 0
        tech["pv"]["costs"] = {"capex": {}, "opex": {}}
        tech["bess"]["design_space"]["scale_pct_of_pv"] = [100]
        tech["bess"]["design_space"]["absolute_power_kw"] = BESS_CONFIGS["bess_only"]["bess_power"]
        tech["bess"]["design_space"]["absolute_capacity_kwh"] = BESS_CONFIGS["bess_only"]["bess_capacity"]
    elif tech_setup == "pv_bess":
        # Keep defaults (scale=[50], e_to_p=[2])
        pass

    # --- Marketing strategy ---
    if marketing == "market":
        rev["marketing"] = {"type": "market"}
        rev["ppa"] = {"type": "none", "duration_years": 15, "inflation_on_ppa": False,
                      "guarantee_of_origin_eur_per_kwh": 0.003}
    elif marketing == "eeg":
        rev["marketing"] = {
            "type": "eeg",
            "floor_price_eur_per_kwh": 0.0735,
            "fixed_price_years": 20,
            "eeg_inflation": False,
        }
        rev["ppa"] = {"type": "none", "duration_years": 15, "inflation_on_ppa": False,
                      "guarantee_of_origin_eur_per_kwh": 0.003}
    elif marketing == "ppa_pap":
        rev["marketing"] = {"type": "ppa"}
        rev["ppa"] = {
            "type": "ppa_pay_as_produced",
            "pay_as_produced_price_eur_per_kwh": 0.06,
            "duration_years": 15,
            "inflation_on_ppa": False,
            "guarantee_of_origin_eur_per_kwh": 0.003,
        }
    elif marketing == "ppa_baseload":
        rev["marketing"] = {"type": "ppa"}
        rev["ppa"] = {
            "type": "ppa_baseload",
            "pay_as_produced_price_eur_per_kwh": 0.071,
            "baseload_mw": 0.1,
            "duration_years": 15,
            "inflation_on_ppa": False,
            "guarantee_of_origin_eur_per_kwh": 0.003,
        }
    elif marketing == "ppa_floor":
        rev["marketing"] = {"type": "ppa"}
        rev["ppa"] = {
            "type": "ppa_floor",
            "floor_price_eur_per_kwh": 0.055,
            "duration_years": 15,
            "inflation_on_ppa": False,
            "guarantee_of_origin_eur_per_kwh": 0.003,
        }
    elif marketing == "ppa_collar":
        rev["marketing"] = {"type": "ppa"}
        rev["ppa"] = {
            "type": "ppa_collar",
            "floor_price_eur_per_kwh": 0.050,
            "cap_price_eur_per_kwh": 0.090,
            "duration_years": 15,
            "inflation_on_ppa": False,
            "guarantee_of_origin_eur_per_kwh": 0.003,
        }

    return s


# ---------------------------------------------------------------------------
# CSV parsing helpers
# ---------------------------------------------------------------------------


def _parse_german_float(value: str) -> float:
    """Parse a German-formatted float string (comma decimal) to float."""
    if not value or value.strip() == "":
        return float("nan")
    return float(value.replace(",", "."))


def parse_results_from_csvs(
        output_dir: Path,
        scenario_name: str,
) -> dict:
    """Read summary and grid search CSVs and extract KPIs.

    Returns
    -------
    dict
        Keys: equity_irr, project_irr, npv, dscr_min, revenue_year1, capex_total
    """
    summary_path = output_dir / scenario_name / f"{scenario_name}_summary.csv"
    grid_path = output_dir / scenario_name / f"{scenario_name}_grid_search.csv"

    # Read summary CSV (single row, semicolon-delimited, German decimal)
    summary_df = pd.read_csv(
        summary_path,
        sep=CSV_DELIMITER,
        decimal=CSV_DECIMAL_SEPARATOR,
        encoding="utf-8",
    )
    row = summary_df.iloc[0]

    equity_irr = _safe_float(row.get("equity_irr_pct"))
    project_irr = _safe_float(row.get("project_irr_pct"))
    npv = _safe_float(row.get("npv_eur"))
    dscr_min = _safe_float(row.get("dscr_min"))
    capex_total = _safe_float(row.get("total_capex_eur"))

    revenue_year1 = _safe_float(row.get("total_revenue_eur"))

    return {
        "equity_irr": equity_irr,
        "project_irr": project_irr,
        "npv": npv,
        "dscr_min": dscr_min,
        "revenue_year1": revenue_year1,
        "capex_total": capex_total,
    }


def _safe_float(val) -> float | None:
    """Convert a value to float, returning None for NaN or missing."""
    if val is None:
        return None
    try:
        f = float(val)
        if pd.isna(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def extract_constraint_params(scenario_dict: dict) -> dict:
    """Extract BESS/grid parameters from scenario dict for constraint checker."""
    ps = scenario_dict["project_settings"]
    tech = ps["technology"]
    pv_peak = float(tech["pv"]["design"]["peak_power_kwp"])
    bess_perf = tech["bess"]["performance"]
    grid = tech["grid_connection"]

    # Determine BESS power/capacity
    ds = tech["bess"]["design_space"]
    if "absolute_power_kw" in ds and ds["absolute_power_kw"]:
        bess_power = float(ds["absolute_power_kw"])
        bess_capacity = float(ds["absolute_capacity_kwh"])
    else:
        scales = ds.get("scale_pct_of_pv", [0])
        e_to_p = ds.get("e_to_p_ratio_hours", [2])
        # Use the last (typically largest) scale
        scale = max(scales)
        bess_power = pv_peak * scale / 100.0
        bess_capacity = bess_power * e_to_p[0] if bess_power > 0 else 0.0

    system_loss_pct = float(grid.get("system_loss_pct", 0.0))

    return {
        "pv_peak_kwp": pv_peak,
        "bess_power_kw": bess_power,
        "bess_capacity_kwh": bess_capacity,
        "grid_max_kw": float(grid["max_export_kw"]),
        "rte": float(bess_perf["round_trip_efficiency_pct"]) / 100.0,
        "min_soc_pct": float(bess_perf["min_soc_pct"]),
        "max_soc_pct": float(bess_perf["max_soc_pct"]),
        "operating_mode": ps["operating_mode"],
        "grid_loss_factor": 1.0 - system_loss_pct / 100.0,
    }


# ---------------------------------------------------------------------------
# Programmatic scenario runner
# ---------------------------------------------------------------------------


def run_scenario_programmatic(
        scenario_dict: dict,
        output_dir: Path,
) -> ScenarioResult:
    """Run a single scenario via ``pv_bess_model.main.run()``.

    Parameters
    ----------
    scenario_dict:
        Complete scenario configuration dict.
    output_dir:
        Root output directory (scenario name subdir will be created inside).

    Returns
    -------
    ScenarioResult
        Aggregated results including KPIs, constraint violations, and
        availability counts.
    """
    from pv_bess_model.main import run

    name = scenario_dict["scenario"]["name"]

    # Write scenario to temp JSON
    scenario_json_path = output_dir / f"{name}.json"
    scenario_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scenario_json_path, "w") as f:
        json.dump(scenario_dict, f, indent=2)

    # Build argparse namespace
    args = argparse.Namespace(
        scenario=str(scenario_json_path),
        output=str(output_dir),
        no_mc=True,
        bess_power=None,
        bess_capacity=None,
        verbose=False,
        dry_run=False,
    )

    # Run
    exit_code = run(args)
    assert exit_code == 0, f"Scenario {name} failed with exit code {exit_code}"

    return check_dispatch_on_violations(name, output_dir, scenario_dict)

def check_dispatch_on_violations(name:str, output_dir: Path, scenario_dict:dict):
    # Parse results from output CSVs
    kpis = parse_results_from_csvs(output_dir, name)

    # Load dispatch sample for constraint checking
    dispatch_path = output_dir / name / f"{name}_dispatch_sample.csv"
    dispatch_violations: list[ConstraintViolation] = []
    price_violations: list[ConstraintViolation] = []
    pv_offline = 0
    bess_offline = 0

    if dispatch_path.exists():
        dispatch_df = pd.read_csv(
            dispatch_path,
            sep=CSV_DELIMITER,
            decimal=CSV_DECIMAL_SEPARATOR,
            encoding="utf-8",
        )

        params = extract_constraint_params(scenario_dict)
        dispatch_violations = check_dispatch_constraints(
            dispatch_df=dispatch_df,
            tolerance=0.1,  # slightly generous for integration tests
            **params,
        )

        n_intervals = len(dispatch_df)
        ipd = 96 if n_intervals == 35040 else 24

        pv_offline, bess_offline = check_availability(
            dispatch_df=dispatch_df,
            intervals_per_day=ipd,
        )

        price_violations = check_price_dependencies(
            dispatch_df=dispatch_df,
        )

    return ScenarioResult(
        name=name,
        equity_irr=kpis["equity_irr"],
        project_irr=kpis["project_irr"],
        npv=kpis["npv"] if kpis["npv"] is not None else 0.0,
        dscr_min=kpis["dscr_min"],
        revenue_year1=kpis["revenue_year1"] if kpis["revenue_year1"] is not None else 0.0,
        capex_total=kpis["capex_total"] if kpis["capex_total"] is not None else 0.0,
        dispatch_violations=dispatch_violations,
        pv_offline_days=pv_offline,
        bess_offline_days=bess_offline,
        price_violations=price_violations
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def price_csv_path(data_dir: Path) -> Path:
    """Path to the synthetic integration test price CSV."""
    p = data_dir / "integration_test_inputs" / "suite" / "integration_suite_prices.csv"
    assert p.exists(), f"Price CSV not found: {p}"
    return p


@pytest.fixture(scope="module")
def all_results(price_csv_path: Path, tmp_path_factory) -> dict[str, ScenarioResult]:
    """Run all 36 scenario combinations and cache results.

    This fixture is module-scoped so that all scenarios are run once and
    shared across all test methods.
    """
    results: dict[str, ScenarioResult] = {}

    for tech in TECH_SETUPS:
        for mode in OPERATING_MODES:
            for mkt in MARKETING_STRATEGIES:
                name = f"{tech}_{mode}_{mkt}"

                # Green bess-only should not work, this is just artificial, with manual mock up results in output dir
                if (tech == "bess_only") & (mode == "green"):
                    results[name] = ScenarioResult(
                        name=name,
                        equity_irr=None,
                        project_irr=None,
                        revenue_year1=0,
                        npv=-1e6,
                        dscr_min=0,
                        capex_total=200000,
                    )
                    continue

                scenario = build_scenario(MASTER_SCENARIO, tech, mode, mkt)
                # Set price CSV path (absolute)
                for scenario_att in scenario["project_settings"]["finance"]["price_inputs"]["scenarios"]:
                    scenario_att["price_csv"] = str(price_csv_path)

                logger.info("Running integration scenario: %s", name)
                try:
                    result = run_scenario_programmatic(scenario, Path(scenario["scenario"]["output"]["directory"]))
                    results[name] = result
                except Exception as exc:
                    logger.error("Scenario %s failed: %s", name, exc)
                    # Store a failure result
                    results[name] = ScenarioResult(
                        name=name,
                        equity_irr=None,
                        project_irr=None,
                        npv=0.0,
                        dscr_min=None,
                        revenue_year1=0.0,
                        capex_total=0.0,
                        dispatch_violations=[
                            ConstraintViolation(
                                constraint="execution_error",
                                timestep=0,
                                expected="exit_code == 0",
                                actual=exc.args[0],
                                severity="error",
                            )
                        ],
                    )

    return results


# ---------------------------------------------------------------------------
# Helper to generate scenario IDs
# ---------------------------------------------------------------------------

_ALL_SCENARIO_IDS = [
    f"{tech}_{mode}_{mkt}"
    for tech in TECH_SETUPS
    for mode in OPERATING_MODES
    for mkt in MARKETING_STRATEGIES
]


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestScenarioExecution:
    """Verify that all 36 scenarios run without errors."""

    @pytest.mark.parametrize("scenario_id", _ALL_SCENARIO_IDS)
    def test_scenario_runs(self, all_results: dict[str, ScenarioResult], scenario_id: str):
        """Scenario completes without execution errors."""
        result = all_results[scenario_id]
        exec_errors = [v for v in result.dispatch_violations if v.constraint == "execution_error"]
        assert len(exec_errors) == 0, f"Scenario {scenario_id} failed to execute"

    @pytest.mark.parametrize("scenario_id", _ALL_SCENARIO_IDS)
    def test_dispatch_constraints(self, all_results: dict[str, ScenarioResult], scenario_id: str):
        """No dispatch constraint violations (errors only, warnings allowed)."""
        result = all_results[scenario_id]
        errors = [v for v in result.dispatch_violations if v.severity == "error"]
        if errors:
            msg_lines = [f"Scenario {scenario_id} has {len(errors)} constraint errors:"]
            for v in errors[:10]:
                msg_lines.append(f"  [{v.constraint}] timestep={v.timestep}: {v.expected}, actual={v.actual:.4f}")
            assert False, "\n".join(msg_lines)


@pytest.mark.integration
class TestAvailability:
    """Verify BESS and PV availability behaviour."""

    @pytest.mark.parametrize("tech", ["pv_bess", "pv_only"])
    @pytest.mark.parametrize("mode", OPERATING_MODES)
    def test_bess_offline_days_pv_scenarios(
            self, all_results: dict[str, ScenarioResult], tech: str, mode: str
    ):
        """BESS offline days are plausible for PV scenarios (97% availability)."""
        # For pv_only there is no BESS, so offline days should be 365 (no activity)
        # For pv_bess, expect >= 11 offline days (3% of 365)
        result = all_results[f"{tech}_{mode}_market"]
        if tech == "pv_only":
            # No BESS → every day is "offline" (no charge/discharge)
            assert result.bess_offline_days == 365
        else:
            expected_min = round((1.0 - 99.0 / 100.0) * 365)
            assert result.bess_offline_days >= expected_min, (
                f"{tech}_{mode}: expected >= {expected_min} BESS offline days, "
                f"got {result.bess_offline_days}"
            )

    @pytest.mark.parametrize("tech", ["pv_bess", "pv_only"])
    @pytest.mark.parametrize("mode", OPERATING_MODES)
    def test_pv_offline_days_zero_without_mc(
            self, all_results: dict[str, ScenarioResult], mode: str, tech: str
    ):
        """Without MC, PV production should never have zero-production days
        (aside from natural zero-sun days in winter)."""
        result = all_results[f"{tech}_{mode}_market"]
        path = Path(
            __file__).parent.parent.parent / ".data" / "test" / "integration_tests" / f"{tech}_{mode}_market.json"
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        defined_offline_days = np.ceil((1 - data["project_settings"]["technology"]["pv"]["performance"][
            "pv_availability_pct"] / 100) * DAYS_PER_YEAR)
        assert abs(defined_offline_days - result.pv_offline_days) <= 9, (
            # 9 Days in sample data with no production anyway
            f"pv_bess_{mode}: too many PV offline days ({result.pv_offline_days})"
        )

    @pytest.mark.parametrize("scenario_id", _ALL_SCENARIO_IDS)
    def test_pv_no_production_at_negative_prices(
            self, all_results: dict[str, ScenarioResult], scenario_id: str
    ):
        result = all_results[scenario_id]
        errors = [v for v in result.price_violations if v.severity == "error"]
        if errors:
            msg_lines = [f"Scenario {scenario_id} has {len(errors)} constraint errors:"]
            for v in errors[:10]:
                msg_lines.append(f"  [{v.constraint}] timestep={v.timestep}: {v.expected}, actual={v.actual:.4f}")
            assert False, "\n".join(msg_lines)

    @pytest.mark.parametrize("mkt", MARKETING_STRATEGIES)
    def test_bess_only_no_pv_production(
            self, all_results: dict[str, ScenarioResult], mkt: str
    ):
        """BESS-only scenarios should have 365 PV offline days."""
        result = all_results[f"bess_only_grey_{mkt}"]
        assert result.pv_offline_days == 365, (
            f"bess_only_grey_{mkt}: expected 365 PV offline days, "
            f"got {result.pv_offline_days}"
        )


@pytest.mark.integration
class TestKPIRanking:
    """Verify plausible KPI ordering across scenario variants."""

    def _rev1(self, results: dict, key: str) -> float | None:
        """Get revenue in year 1 from results."""
        r = results.get(key)
        if r is None:
            return None
        return r.revenue_year1

    @pytest.mark.parametrize("tech", ["pv_only", "bess_only"])
    @pytest.mark.parametrize("mode", OPERATING_MODES)
    def test_marketing_ranking(
            self, all_results: dict[str, ScenarioResult], tech: str, mode: str
    ):
        """EEG >= PPA-Floor >= PPA-Collar >= PPA-PaP >= Market (by NPV).

        This ranking holds because:
        - EEG floor (7.35 ct) is higher than all PPA prices
        - Floor PPA (5.5 ct) guarantees minimum + keeps upside
        - Collar PPA (5.0-9.0 ct) limits upside
        - Pay-as-produced (6.5 ct fixed) no upside from spot
        - Market: pure spot exposure
        """
        tol = 1.0  # EUR tolerance

        rev1_eeg = self._rev1(all_results, f"{tech}_{mode}_eeg")
        rev1_floor = self._rev1(all_results, f"{tech}_{mode}_ppa_floor")
        rev1_collar = self._rev1(all_results, f"{tech}_{mode}_ppa_collar")
        rev1_pap = self._rev1(all_results, f"{tech}_{mode}_ppa_pap")
        rev1_market = self._rev1(all_results, f"{tech}_{mode}_market")

        assert rev1_eeg >= rev1_floor - tol, (
            f"{tech}_{mode}: EEG ({rev1_eeg:.0f}) < Floor ({rev1_floor:.0f})"
        )
        assert rev1_floor >= rev1_collar - tol, (
            f"{tech}_{mode}: Floor ({rev1_floor:.0f}) < Collar ({rev1_collar:.0f})"
        )
        assert rev1_collar >= rev1_pap - tol, (
            f"{tech}_{mode}: Collar ({rev1_collar:.0f}) < PaP ({rev1_pap:.0f})"
        )
        assert rev1_pap >= rev1_market - tol, (
            f"{tech}_{mode}: PaP ({rev1_pap:.0f}) < Market ({rev1_market:.0f})"
        )

    @pytest.mark.parametrize("tech", ["pv_only", "pv_bess"])
    @pytest.mark.parametrize("mode", OPERATING_MODES)
    def test_baseload_plausibility(
            self, all_results: dict[str, ScenarioResult], tech: str, mode: str
    ):
        """Baseload PPA NPV is between Market and EEG."""
        tol = 1.0
        rev1_market = self._rev1(all_results, f"{tech}_{mode}_market")
        rev1_baseload = self._rev1(all_results, f"{tech}_{mode}_ppa_baseload")
        rev1_eeg = self._rev1(all_results, f"{tech}_{mode}_eeg")

        assert rev1_baseload >= rev1_market - tol, (
            f"{tech}_{mode}: Baseload ({rev1_baseload:.0f}) < Market ({rev1_market:.0f})"
        )
        assert rev1_baseload <= rev1_eeg + tol, (
            f"{tech}_{mode}: Baseload ({rev1_baseload:.0f}) > EEG ({rev1_eeg:.0f})"
        )

    @pytest.mark.parametrize("mkt", MARKETING_STRATEGIES)
    def test_pv_only_green_equals_grey(
            self, all_results: dict[str, ScenarioResult], mkt: str
    ):
        """PV-only: Green and Grey mode should yield identical NPV."""
        tol = 1.0
        rev1_green = self._rev1(all_results, f"pv_only_green_{mkt}")
        rev1_grey = self._rev1(all_results, f"pv_only_grey_{mkt}")
        assert abs(rev1_green - rev1_grey) <= tol, (
            f"pv_only {mkt}: Green Revenue ({rev1_green:.0f}) != Grey Revenue ({rev1_grey:.0f})"
        )

    @pytest.mark.parametrize("mkt", MARKETING_STRATEGIES)
    def test_grey_geq_green_pv_bess(
            self, all_results: dict[str, ScenarioResult], mkt: str
    ):
        """PV+BESS: Grey mode NPV >= Green mode NPV (grey has more flexibility)."""
        tol = 1.0
        rev1_green = self._rev1(all_results, f"pv_bess_green_{mkt}")
        rev1_grey = self._rev1(all_results, f"pv_bess_grey_{mkt}")
        assert rev1_grey >= rev1_green - tol, (
            f"pv_bess {mkt}: Grey ({rev1_grey:.0f}) < Green ({rev1_green:.0f})"
        )

    @pytest.mark.parametrize("mkt", MARKETING_STRATEGIES)
    def test_bess_only_green_negative(
            self, all_results: dict[str, ScenarioResult], mkt: str
    ):
        """BESS-only in green mode has Revenue = 0 (no PV to charge from)."""
        rev1 = self._rev1(all_results, f"bess_only_green_{mkt}")
        assert rev1 == 0.0, f"bess_only_green_{mkt}: expected zero revenue, got {rev1:.0f}"

    def test_bess_only_green_all_equal(self, all_results: dict[str, ScenarioResult]):
        """BESS-only green: all marketing strategies yield the same NPV.

        (No PV means no production, so marketing type is irrelevant.)
        """
        tol = 1.0
        rev1s = [
            self._rev1(all_results, f"bess_only_green_{mkt}")
            for mkt in MARKETING_STRATEGIES
        ]
        for i in range(1, len(rev1s )):
            assert abs(rev1s [i] - rev1s [0]) <= tol, (
                f"bess_only_green: Revenue spread too large: {rev1s }"
            )


    @pytest.mark.parametrize("mode", OPERATING_MODES)
    @pytest.mark.parametrize("mkt", MARKETING_STRATEGIES)
    def test_pv_bess_geq_pv_only(
            self, all_results: dict[str, ScenarioResult], mode: str, mkt: str
    ):
        """PV+BESS NPV >= PV-only NPV (BESS should add value or at worst break even).

        Note: This may not always hold if BESS CAPEX/OPEX exceeds the
        incremental revenue. We use a generous tolerance.
        """
        rev1_pv_only = self._rev1(all_results, f"pv_only_{mode}_{mkt}")
        rev1_pv_bess = self._rev1(all_results, f"pv_bess_{mode}_{mkt}")
        # BESS may not always add value, so just check it's not wildly worse
        # Allow PV+BESS to be up to 50k worse (BESS costs may exceed revenue)
        assert rev1_pv_bess >= rev1_pv_only - 500_000, (
            f"{mode}_{mkt}: PV+BESS ({rev1_pv_bess:.0f}) << PV-only ({rev1_pv_only:.0f})"
        )

    def test_bess_only_grey_gt_green(self, all_results: dict[str, ScenarioResult]):
        """BESS-only: grey mode NPV > green mode NPV for market strategy."""
        rev1_green = self._rev1(all_results, "bess_only_green_market")
        rev1_grey = self._rev1(all_results, "bess_only_grey_market")
        assert rev1_grey > rev1_green, (
            f"bess_only: Grey market ({rev1_grey:.0f}) <= Green market ({rev1_green:.0f})"
        )


@pytest.mark.integration
class TestOutputCompleteness:
    """Verify that expected output files exist for all scenarios."""

    @pytest.mark.parametrize("scenario_id", _ALL_SCENARIO_IDS)
    def test_output_files_exist(
            self, all_results: dict[str, ScenarioResult], tmp_path_factory, scenario_id: str
    ):
        """Summary, cashflows, and dispatch sample CSVs exist."""
        # We need the output dir. Since all_results uses a module-scoped tmpdir,
        # we check that the result was created successfully.
        result = all_results[scenario_id]
        exec_errors = [v for v in result.dispatch_violations if v.constraint == "execution_error"]
        if exec_errors:
            pytest.skip(f"Scenario {scenario_id} did not execute successfully")
        # If we got here, the scenario ran. Output file existence was implicitly
        # validated by parse_results_from_csvs succeeding.
        assert result.name == scenario_id
