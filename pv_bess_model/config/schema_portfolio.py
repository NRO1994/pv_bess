"""JSON schema definition and validation for portfolio/Systemwert configuration files.

The schema validates the structure documented in the feature specification
(``12_systemwert_flexibilitaet.md`` §10 "JSON-Input").
Validation uses the ``jsonschema`` library (Draft 7).

Usage::

    from pv_bess_model.config.schema_portfolio import validate_portfolio
    validate_portfolio(data)   # raises jsonschema.ValidationError on failure
"""

from __future__ import annotations

import jsonschema

# ---------------------------------------------------------------------------
# Re-usable sub-schemas
# ---------------------------------------------------------------------------

_NON_NEGATIVE_NUMBER = {"type": "number", "minimum": 0}
_POSITIVE_NUMBER = {"type": "number", "exclusiveMinimum": 0}
_POSITIVE_INTEGER = {"type": "integer", "minimum": 1}
_PERCENTAGE = {"type": "number", "minimum": 0, "maximum": 100}

# ---------------------------------------------------------------------------
# Reusable flex cost sub-schemas
# ---------------------------------------------------------------------------

_FLEX_COST_CAPEX = {
    "type": "object",
    "properties": {
        "fixed_eur": _NON_NEGATIVE_NUMBER,
        "eur_per_kw": _NON_NEGATIVE_NUMBER,
        "eur_per_kwh": _NON_NEGATIVE_NUMBER,
    },
    "additionalProperties": False,
}

_FLEX_COST_OPEX = {
    "type": "object",
    "properties": {
        "fixed_eur": _NON_NEGATIVE_NUMBER,
        "eur_per_kw": _NON_NEGATIVE_NUMBER,
        "eur_per_kwh": _NON_NEGATIVE_NUMBER,
    },
    "additionalProperties": False,
}

_PERSONNEL_STEP = {
    "type": "object",
    "required": ["threshold_kw", "annual_cost_eur"],
    "properties": {
        "threshold_kw": _NON_NEGATIVE_NUMBER,
        "annual_cost_eur": _NON_NEGATIVE_NUMBER,
    },
    "additionalProperties": False,
}

_FLEX_COSTS = {
    "type": "object",
    "properties": {
        "capex": _FLEX_COST_CAPEX,
        "opex": _FLEX_COST_OPEX,
        "capex_learning_rate_pct": _PERCENTAGE,
        "personnel_steps": {
            "type": "array",
            "items": _PERSONNEL_STEP,
        },
    },
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# meta_model
# ---------------------------------------------------------------------------

_OUTPUT = {
    "type": "object",
    "required": ["directory"],
    "properties": {
        "directory": {"type": "string", "minLength": 1},
        "export_dispatch_sample": {"type": "boolean"},
        "csv_separator": {"type": "string", "minLength": 1},
        "csv_decimal": {"type": "string", "minLength": 1},
    },
    "additionalProperties": False,
}

_META_MODEL = {
    "type": "object",
    "required": ["name", "baseline_year", "project_lifetime_years"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "baseline_year": {"type": "integer", "minimum": 2020},
        "project_lifetime_years": _POSITIVE_INTEGER,
        "perfect_foresight_discount": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "bundesland": {"type": "string", "minLength": 2, "maxLength": 2},
        "output": _OUTPUT,
    },
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# portfolio.generation[]
# ---------------------------------------------------------------------------

_LOCATION = {
    "type": "object",
    "required": ["latitude", "longitude"],
    "properties": {
        "latitude": {"type": "number", "minimum": -90, "maximum": 90},
        "longitude": {"type": "number", "minimum": -180, "maximum": 180},
        "pvgis_database": {"type": "string", "minLength": 1},
    },
    "additionalProperties": False,
}

_GENERATION_PV = {
    "type": "object",
    "required": ["type", "name", "peak_power_kwp", "location"],
    "properties": {
        "type": {"type": "string", "enum": ["pv"]},
        "name": {"type": "string", "minLength": 1},
        "peak_power_kwp": _POSITIVE_NUMBER,
        "location": _LOCATION,
        "degradation_rate_pct_per_year": _PERCENTAGE,
        "system_loss_pct": _PERCENTAGE,
        "mounting_type": {"type": "string", "enum": ["free", "building"]},
        "azimuth_deg": {"type": "number", "minimum": -180, "maximum": 180},
        "tilt_deg": {"type": "number", "minimum": 0, "maximum": 90},
        "start_year": _POSITIVE_INTEGER,
        "commissioning_year": {"type": "integer", "minimum": 1900, "maximum": 2200},
        "lifetime_years": _POSITIVE_INTEGER,
    },
    "additionalProperties": False,
}

_GENERATION = {
    "type": "array",
    "items": _GENERATION_PV,
    "minItems": 1,
}

# ---------------------------------------------------------------------------
# portfolio.load[]
# ---------------------------------------------------------------------------

_LOAD_GROUP = {
    "type": "object",
    "required": ["type", "name", "slp_type", "customer_count",
                 "annual_consumption_kwh_per_customer"],
    "properties": {
        "type": {"type": "string", "enum": ["slp"]},
        "name": {"type": "string", "minLength": 1},
        "slp_type": {
            "type": "string",
            "enum": ["H25", "G25", "L25", "P25", "S25"],
        },
        "customer_count": {"type": "integer", "minimum": 1},
        "annual_consumption_kwh_per_customer": _POSITIVE_NUMBER,
        "annual_growth_factor": _POSITIVE_NUMBER,
    },
    "additionalProperties": False,
}

_LOAD = {
    "type": "array",
    "items": _LOAD_GROUP,
    "minItems": 1,
}

# ---------------------------------------------------------------------------
# portfolio block
# ---------------------------------------------------------------------------

_PORTFOLIO = {
    "type": "object",
    "required": ["generation", "load"],
    "properties": {
        "generation": _GENERATION,
        "load": _LOAD,
    },
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# flexibilities[]
# ---------------------------------------------------------------------------

_FLEX_BESS = {
    "type": "object",
    "required": ["type", "name", "annual_addition_kw", "e_to_p_ratio_hours"],
    "properties": {
        "type": {"type": "string", "enum": ["bess"]},
        "name": {"type": "string", "minLength": 1},
        "annual_addition_kw": {
            "type": "array",
            "items": _NON_NEGATIVE_NUMBER,
            "minItems": 1,
        },
        "e_to_p_ratio_hours": {
            "type": "array",
            "items": _POSITIVE_NUMBER,
            "minItems": 1,
        },
        "round_trip_efficiency_pct": _PERCENTAGE,
        "min_soc_pct": _PERCENTAGE,
        "max_soc_pct": _PERCENTAGE,
        "degradation_rate_pct_per_year": _PERCENTAGE,
        "start_year": _POSITIVE_INTEGER,
        "costs": _FLEX_COSTS,
    },
    "additionalProperties": False,
}

_FLEX_HEAT_PUMP = {
    "type": "object",
    "required": ["type", "name", "annual_addition_kw",
                 "cop_nominal", "annual_thermal_demand_mwh",
                 "thermal_storage_kwh"],
    "properties": {
        "type": {"type": "string", "enum": ["heat_pump"]},
        "name": {"type": "string", "minLength": 1},
        "annual_addition_kw": {
            "type": "array",
            "items": _NON_NEGATIVE_NUMBER,
            "minItems": 1,
        },
        "cop_nominal": _POSITIVE_NUMBER,
        "cop_reference_temp_c": {"type": "number"},
        "annual_thermal_demand_mwh": _POSITIVE_NUMBER,
        "thermal_storage_kwh": _NON_NEGATIVE_NUMBER,
        "start_year": _POSITIVE_INTEGER,
        "costs": _FLEX_COSTS,
    },
    "additionalProperties": False,
}

_FLEX_EV_TIME_WINDOW = {
    "type": "object",
    "required": ["arrival_hour", "departure_hour"],
    "properties": {
        "arrival_hour": {"type": "integer", "minimum": 0, "maximum": 23},
        "departure_hour": {"type": "integer", "minimum": 0, "maximum": 23},
    },
    "additionalProperties": False,
}

_FLEX_EV = {
    "type": "object",
    "required": ["type", "name", "mean_kw_per_unit", "annual_additional_units",
                 "daily_energy_demand_kwh_per_unit", "time_window",
                 "usable_battery_kwh_per_unit"],
    "properties": {
        "type": {"type": "string", "enum": ["ev_charging"]},
        "name": {"type": "string", "minLength": 1},
        "mean_kw_per_unit": _POSITIVE_NUMBER,
        "annual_additional_units": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
            "minItems": 1,
        },
        "daily_energy_demand_kwh_per_unit": _POSITIVE_NUMBER,
        "time_window": _FLEX_EV_TIME_WINDOW,
        "v2g_enabled": {"type": "boolean"},
        "v2g_rte_pct": _PERCENTAGE,
        "min_departure_soc_pct": _PERCENTAGE,
        "usable_battery_kwh_per_unit": _POSITIVE_NUMBER,
        "start_year": _POSITIVE_INTEGER,
        "costs": _FLEX_COSTS,
    },
    "additionalProperties": False,
}

_FLEXIBILITY = {
    "oneOf": [_FLEX_BESS, _FLEX_HEAT_PUMP, _FLEX_EV],
}

_FLEXIBILITIES = {
    "type": "array",
    "items": _FLEXIBILITY,
    "minItems": 1,
}

# ---------------------------------------------------------------------------
# price_inputs (reuses same structure as existing PV+BESS model)
# ---------------------------------------------------------------------------

_PRICE_WEATHER_SCENARIO = {
    "type": "object",
    "required": [
        "name",
        "csv_column",
        "weather_year",
        "weight",
        "price_csv",
        "csv_separator",
        "csv_decimal",
        "csv_timestamp_column",
        "csv_timestamp_format",
    ],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "label": {"type": "string"},
        "csv_column": {"type": "string", "minLength": 1},
        "weather_year": {"type": "integer", "minimum": 1900},
        "weight": {"type": "number", "minimum": 0, "maximum": 1},
        "is_central": {"type": "boolean"},
        "price_csv": {"type": "string", "minLength": 1},
        "inflation_on_input_data": {"type": "boolean"},
        "csv_separator": {"type": "string", "minLength": 1},
        "csv_decimal": {"type": "string", "minLength": 1},
        "csv_timestamp_column": {"type": "string", "minLength": 1},
        "csv_timestamp_format": {"type": "string", "minLength": 1},
    },
    "additionalProperties": False,
}

_PRICE_INPUTS = {
    "type": "object",
    "required": ["scenarios"],
    "properties": {
        "scenarios": {
            "type": "array",
            "items": _PRICE_WEATHER_SCENARIO,
            "minItems": 1,
        },
    },
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Top-level schema
# ---------------------------------------------------------------------------

PORTFOLIO_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Portfolio / Systemwert Configuration",
    "type": "object",
    "required": ["meta_model", "portfolio", "flexibilities", "price_inputs"],
    "properties": {
        "meta_model": _META_MODEL,
        "portfolio": _PORTFOLIO,
        "flexibilities": _FLEXIBILITIES,
        "price_inputs": _PRICE_INPUTS,
    },
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_portfolio(data: dict) -> None:
    """Validate a portfolio configuration dictionary against the JSON schema.

    Raises a ``jsonschema.ValidationError`` with a descriptive message
    (including the JSON path to the failing field) if validation fails.

    Parameters
    ----------
    data:
        Parsed portfolio configuration dictionary (e.g. from ``json.load``).

    Raises
    ------
    jsonschema.ValidationError
        When *data* does not conform to the portfolio schema.
    ValueError
        When cross-field semantic constraints are violated.
    """
    validator = jsonschema.Draft7Validator(PORTFOLIO_SCHEMA)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))

    if errors:
        first = errors[0]
        path_str = " → ".join(str(p) for p in first.absolute_path) or "(root)"
        raise jsonschema.ValidationError(
            f"Portfolio validation failed at '{path_str}': {first.message}",
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
    _validate_scenarios(data)
    _validate_bess_soc_limits(data)


def _validate_scenarios(data: dict) -> None:
    """Validate price-weather scenarios.

    Checks:
    - Exactly one scenario has ``is_central: true``.
    - Scenario weights sum to 1.0 (±tolerance).
    """
    from pv_bess_model.config.defaults import MC_WEIGHT_TOLERANCE

    scenarios = data.get("price_inputs", {}).get("scenarios", [])
    if not scenarios:
        return

    central_count = sum(1 for s in scenarios if s.get("is_central", False))
    if central_count != 1:
        raise ValueError(
            f"Exactly one scenario must have 'is_central: true', "
            f"but {central_count} scenario(s) are marked as central. "
            f"Check price_inputs.scenarios."
        )

    total_weight = sum(s.get("weight", 0.0) for s in scenarios)
    if abs(total_weight - 1.0) > MC_WEIGHT_TOLERANCE:
        raise ValueError(
            f"Scenario weights must sum to 1.0, but they sum to "
            f"{total_weight:.6f}. "
            f"Adjust the 'weight' fields in price_inputs.scenarios."
        )


def _validate_bess_soc_limits(data: dict) -> None:
    """Check that min_soc_pct < max_soc_pct for all BESS flex entries."""
    for flex in data.get("flexibilities", []):
        if flex.get("type") != "bess":
            continue
        min_soc = flex.get("min_soc_pct")
        max_soc = flex.get("max_soc_pct")
        if min_soc is not None and max_soc is not None and min_soc >= max_soc:
            name = flex.get("name", "(unnamed)")
            raise ValueError(
                f"BESS flex '{name}': min_soc_pct ({min_soc}) must be strictly "
                f"less than max_soc_pct ({max_soc})."
            )


def get_schema() -> dict:
    """Return a copy of the portfolio JSON schema dictionary.

    Returns
    -------
    dict
        The portfolio schema as a plain Python dictionary compatible with
        ``jsonschema`` and any JSON Schema Draft 7 tool.
    """
    return PORTFOLIO_SCHEMA.copy()
