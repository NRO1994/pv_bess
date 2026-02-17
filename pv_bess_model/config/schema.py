"""JSON schema definition and validation for scenario configuration files.

The schema exactly mirrors the structure documented in CLAUDE.md §"Scenario JSON Schema".
Validation uses the ``jsonschema`` library (Draft 7).

Usage::

    from pv_bess_model.config.schema import validate_scenario
    validate_scenario(data)   # raises jsonschema.ValidationError on failure
"""

from __future__ import annotations

import jsonschema

# ---------------------------------------------------------------------------
# Re-usable sub-schemas
# ---------------------------------------------------------------------------

_NON_NEGATIVE_NUMBER = {"type": "number", "minimum": 0}

_COST_COMPONENT = {
    "type": "object",
    "properties": {
        "fixed_eur": _NON_NEGATIVE_NUMBER,
        "eur_per_kw": _NON_NEGATIVE_NUMBER,
        "eur_per_kwh": _NON_NEGATIVE_NUMBER,
        "pct_of_capex": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "additionalProperties": False,
}

_CAPEX_BLOCK = {
    "type": "object",
    "properties": {
        "capex": _COST_COMPONENT,
        "opex": _COST_COMPONENT,
    },
    "additionalProperties": False,
}

_PRICE_SCENARIO_ENTRY = {
    "type": "object",
    "required": ["csv_column", "weight"],
    "properties": {
        "csv_column": {"type": "string", "minLength": 1},
        "weight": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "additionalProperties": False,
}

_MONTE_CARLO = {
    "type": "object",
    "required": ["enabled"],
    "properties": {
        "enabled": {"type": "boolean"},
        "iterations": {"type": "integer", "minimum": 1},
        "sigma_pv_yield_pct": {"type": "number", "minimum": 0},
        "sigma_capex_pct": {"type": "number", "minimum": 0},
        "sigma_opex_pct": {"type": "number", "minimum": 0},
        "sigma_bess_availability_pct": {"type": "number", "minimum": 0},
        "price_scenarios": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": _PRICE_SCENARIO_ENTRY,
        },
    },
    "additionalProperties": False,
}

_OUTPUT = {
    "type": "object",
    "required": ["directory"],
    "properties": {
        "directory": {"type": "string", "minLength": 1},
        "export_dispatch_sample": {"type": "boolean"},
    },
    "additionalProperties": False,
}

_SCENARIO_BLOCK = {
    "type": "object",
    "required": ["name"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "monte_carlo": _MONTE_CARLO,
        "output": _OUTPUT,
    },
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# PV
# ---------------------------------------------------------------------------

_PV_DESIGN = {
    "type": "object",
    "required": ["peak_power_kwp", "mounting_type", "azimuth_deg", "tilt_deg"],
    "properties": {
        "peak_power_kwp": {"type": "number", "exclusiveMinimum": 0},
        "mounting_type": {"type": "string", "enum": ["free", "building"]},
        "azimuth_deg": {"type": "number", "minimum": -180, "maximum": 180},
        "tilt_deg": {"type": "number", "minimum": 0, "maximum": 90},
    },
    "additionalProperties": False,
}

_PV_PERFORMANCE = {
    "type": "object",
    "required": ["system_loss_pct", "degradation_rate_pct_per_year"],
    "properties": {
        "system_loss_pct": {"type": "number", "minimum": 0, "maximum": 100},
        "degradation_rate_pct_per_year": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
        },
    },
    "additionalProperties": False,
}

_PV_COSTS = {
    "type": "object",
    "required": ["capex", "opex"],
    "properties": {
        "capex": _COST_COMPONENT,
        "opex": _COST_COMPONENT,
    },
    "additionalProperties": False,
}

_PV = {
    "type": "object",
    "required": ["design", "performance", "costs"],
    "properties": {
        "design": _PV_DESIGN,
        "performance": _PV_PERFORMANCE,
        "costs": _PV_COSTS,
    },
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# BESS
# ---------------------------------------------------------------------------

_BESS_DESIGN_SPACE = {
    "type": "object",
    "required": ["scale_pct_of_pv", "e_to_p_ratio_hours"],
    "properties": {
        "scale_pct_of_pv": {
            "type": "array",
            "items": {"type": "number", "minimum": 0, "maximum": 200},
            "minItems": 1,
        },
        "e_to_p_ratio_hours": {
            "type": "array",
            "items": {"type": "number", "exclusiveMinimum": 0},
            "minItems": 1,
        },
    },
    "additionalProperties": False,
}

_BESS_PERFORMANCE = {
    "type": "object",
    "required": [
        "round_trip_efficiency_pct",
        "min_soc_pct",
        "max_soc_pct",
        "degradation_rate_pct_per_year",
        "bess_availability_pct",
    ],
    "properties": {
        "round_trip_efficiency_pct": {"type": "number", "minimum": 0, "maximum": 100},
        "min_soc_pct": {"type": "number", "minimum": 0, "maximum": 100},
        "max_soc_pct": {"type": "number", "minimum": 0, "maximum": 100},
        "degradation_rate_pct_per_year": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
        },
        "bess_availability_pct": {"type": "number", "minimum": 0, "maximum": 100},
    },
    "additionalProperties": False,
}

_BESS_REPLACEMENT = {
    "type": "object",
    "required": ["enabled"],
    "properties": {
        "enabled": {"type": "boolean"},
        "year": {"type": "integer", "minimum": 1},
        "fixed_eur": _NON_NEGATIVE_NUMBER,
        "eur_per_kw": _NON_NEGATIVE_NUMBER,
        "eur_per_kwh": _NON_NEGATIVE_NUMBER,
        "pct_of_capex": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "additionalProperties": False,
}

_BESS_COSTS = {
    "type": "object",
    "required": ["capex", "opex"],
    "properties": {
        "capex": _COST_COMPONENT,
        "opex": _COST_COMPONENT,
        "replacement": _BESS_REPLACEMENT,
    },
    "additionalProperties": False,
}

_BESS = {
    "type": "object",
    "required": ["design_space", "performance", "costs"],
    "properties": {
        "design_space": _BESS_DESIGN_SPACE,
        "performance": _BESS_PERFORMANCE,
        "costs": _BESS_COSTS,
    },
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Grid connection
# ---------------------------------------------------------------------------

_GRID_CONNECTION = {
    "type": "object",
    "required": ["max_export_kw", "costs"],
    "properties": {
        "max_export_kw": {"type": "number", "exclusiveMinimum": 0},
        "costs": {
            "type": "object",
            "required": ["capex", "opex"],
            "properties": {
                "capex": _COST_COMPONENT,
                "opex": _COST_COMPONENT,
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

_TECHNOLOGY = {
    "type": "object",
    "required": ["pv", "bess", "grid_connection"],
    "properties": {
        "pv": _PV,
        "bess": _BESS,
        "grid_connection": _GRID_CONNECTION,
    },
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Finance
# ---------------------------------------------------------------------------

_MARKETING = {
    "type": "object",
    "required": ["type"],
    "properties": {
        "type": {"type": "string", "enum": ["eeg", "ppa", "market"]},
        "floor_price_eur_per_kwh": {"type": ["number", "null"], "minimum": 0},
        "fixed_price_years": {"type": ["integer", "null"], "minimum": 1},
        "eeg_inflation": {"type": "boolean"},
    },
    "additionalProperties": False,
}

_PPA = {
    "type": "object",
    "required": ["type"],
    "properties": {
        "type": {
            "type": "string",
            "enum": [
                "none",
                "ppa_pay_as_produced",
                "ppa_baseload",
                "ppa_floor",
                "ppa_collar",
            ],
        },
        "pay_as_produced_price_eur_per_kwh": {"type": ["number", "null"], "minimum": 0},
        "baseload_mw": {"type": ["number", "null"], "minimum": 0},
        "floor_price_eur_per_kwh": {"type": ["number", "null"], "minimum": 0},
        "cap_price_eur_per_kwh": {"type": ["number", "null"], "minimum": 0},
        "duration_years": {"type": "integer", "minimum": 1},
        "inflation_on_ppa": {"type": "boolean"},
        "guarantee_of_origin_eur_per_kwh": {"type": "number", "minimum": 0},
    },
    "additionalProperties": False,
}

_REVENUE_STREAMS = {
    "type": "object",
    "required": ["marketing"],
    "properties": {
        "marketing": _MARKETING,
        "ppa": _PPA,
    },
    "additionalProperties": False,
}

_PRICE_INPUTS = {
    "type": "object",
    "required": ["day_ahead_csv", "price_unit"],
    "properties": {
        "day_ahead_csv": {"type": "string", "minLength": 1},
        "price_unit": {
            "type": "string",
            "enum": ["eur_per_mwh", "eur_per_kwh"],
        },
        "inflation_on_input_data": {"type": "boolean"},
    },
    "additionalProperties": False,
}

_TAX = {
    "type": "object",
    "required": [
        "afa_years_pv",
        "afa_years_bess",
        "gewerbesteuer_hebesatz",
        "gewerbesteuer_messzahl",
    ],
    "properties": {
        "afa_years_pv": {"type": "integer", "minimum": 1},
        "afa_years_bess": {"type": "integer", "minimum": 1},
        "gewerbesteuer_hebesatz": {"type": "number", "minimum": 0},
        "gewerbesteuer_messzahl": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "additionalProperties": False,
}

_FINANCE = {
    "type": "object",
    "required": [
        "leverage_pct",
        "interest_rate_pct",
        "loan_tenor_years",
        "debt_uses_p90",
        "inflation_rate",
        "revenue_streams",
        "price_inputs",
        "tax",
    ],
    "properties": {
        "leverage_pct": {"type": "number", "minimum": 0, "maximum": 100},
        "interest_rate_pct": {"type": "number", "minimum": 0},
        "loan_tenor_years": {"type": "integer", "minimum": 1},
        "equity_irr_target": {"type": ["number", "null"]},
        "debt_uses_p90": {"type": "boolean"},
        "inflation_rate": {"type": "number", "minimum": 0},
        "revenue_streams": _REVENUE_STREAMS,
        "price_inputs": _PRICE_INPUTS,
        "tax": _TAX,
    },
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------

_LOCATION = {
    "type": "object",
    "required": ["latitude", "longitude", "pvgis_database"],
    "properties": {
        "latitude": {"type": "number", "minimum": -90, "maximum": 90},
        "longitude": {"type": "number", "minimum": -180, "maximum": 180},
        "pvgis_database": {"type": "string", "minLength": 1},
    },
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# project_settings
# ---------------------------------------------------------------------------

_PROJECT_SETTINGS = {
    "type": "object",
    "required": [
        "lifetime_years",
        "discount_rate",
        "operating_mode",
        "location",
        "technology",
        "finance",
    ],
    "properties": {
        "lifetime_years": {"type": "integer", "minimum": 1},
        "discount_rate": {"type": "number", "minimum": 0},
        "operating_mode": {"type": "string", "enum": ["green", "grey"]},
        "location": _LOCATION,
        "technology": _TECHNOLOGY,
        "finance": _FINANCE,
    },
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Top-level schema
# ---------------------------------------------------------------------------

SCENARIO_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "PV+BESS Scenario Configuration",
    "type": "object",
    "required": ["scenario", "project_settings"],
    "properties": {
        "scenario": _SCENARIO_BLOCK,
        "project_settings": _PROJECT_SETTINGS,
    },
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_scenario(data: dict) -> None:
    """Validate a scenario configuration dictionary against the JSON schema.

    Raises a ``jsonschema.ValidationError`` with a descriptive message
    (including the JSON path to the failing field) if validation fails.

    Parameters
    ----------
    data:
        Parsed scenario dictionary (e.g. from ``json.load``).

    Raises
    ------
    jsonschema.ValidationError
        When *data* does not conform to the scenario schema.
    jsonschema.SchemaError
        When the schema itself is malformed (should never happen in production).
    ValueError
        When cross-field semantic constraints are violated (e.g. MC price
        scenario weights do not sum to 1.0).
    """
    validator = jsonschema.Draft7Validator(SCENARIO_SCHEMA)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))

    if errors:
        # Surface the most specific (deepest path) error first.
        first = errors[0]
        path_str = " → ".join(str(p) for p in first.absolute_path) or "(root)"
        raise jsonschema.ValidationError(
            f"Scenario validation failed at '{path_str}': {first.message}",
            path=first.absolute_path,
            schema_path=first.absolute_schema_path,
            validator=first.validator,
            validator_value=first.validator_value,
            instance=first.instance,
            schema=first.schema,
            cause=first.cause,
        )

    # ------------------------------------------------------------------
    # Cross-field semantic validation
    # ------------------------------------------------------------------
    _validate_mc_weights(data)
    _validate_soc_limits(data)


def _validate_mc_weights(data: dict) -> None:
    """Check that Monte Carlo price scenario weights sum to 1.0 (±tolerance)."""
    from pv_bess_model.config.defaults import MC_WEIGHT_TOLERANCE

    mc = data.get("scenario", {}).get("monte_carlo", {})
    if not mc.get("enabled", False):
        return
    scenarios = mc.get("price_scenarios", {})
    if not scenarios:
        return
    total = sum(s.get("weight", 0.0) for s in scenarios.values())
    if abs(total - 1.0) > MC_WEIGHT_TOLERANCE:
        raise ValueError(
            f"Monte Carlo price scenario weights must sum to 1.0, "
            f"but they sum to {total:.6f}. "
            f"Adjust the 'weight' fields in scenario.monte_carlo.price_scenarios."
        )


def _validate_soc_limits(data: dict) -> None:
    """Check that min_soc_pct < max_soc_pct for BESS performance settings."""
    perf = (
        data.get("project_settings", {})
        .get("technology", {})
        .get("bess", {})
        .get("performance", {})
    )
    min_soc = perf.get("min_soc_pct")
    max_soc = perf.get("max_soc_pct")
    if min_soc is not None and max_soc is not None and min_soc >= max_soc:
        raise ValueError(
            f"BESS min_soc_pct ({min_soc}) must be strictly less than "
            f"max_soc_pct ({max_soc}). "
            f"Check project_settings.technology.bess.performance."
        )


def get_schema() -> dict:
    """Return a copy of the scenario JSON schema dictionary.

    Returns
    -------
    dict
        The scenario schema as a plain Python dictionary compatible with
        ``jsonschema`` and any JSON Schema Draft 7 tool.
    """
    return SCENARIO_SCHEMA.copy()
