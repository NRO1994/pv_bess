"""Load and validate portfolio/Systemwert JSON configuration files.

Public API
----------
load_portfolio(path)           – Parse + validate a portfolio JSON file.
load_portfolio_dict(data)      – Validate a pre-loaded dict.
parse_portfolio_scenarios(cfg) – Extract PriceWeatherScenario list.

Reuses ``PriceWeatherScenario`` from ``config.loader`` for price-weather
scenario coupling.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pv_bess_model.config.defaults import (
    DEFAULT_BESS_DEGRADATION_RATE_PCT,
    DEFAULT_BESS_MAX_SOC_PCT,
    DEFAULT_BESS_MIN_SOC_PCT,
    DEFAULT_BESS_RTE_PCT,
    DEFAULT_BUNDESLAND,
    DEFAULT_COP_TEMP_COEFFICIENT,
    DEFAULT_EV_MIN_DEPARTURE_SOC_PCT,
    DEFAULT_EV_V2G_RTE_PCT,
    DEFAULT_FLEX_START_YEAR,
    DEFAULT_PERFECT_FORESIGHT_DISCOUNT,
    DEFAULT_PORTFOLIO_LIFETIME_YEARS,
    DEFAULT_PV_DEGRADATION_RATE_PCT,
    DEFAULT_SYSTEM_LOSS_PCT,
)
from pv_bess_model.config.loader import PriceWeatherScenario
from pv_bess_model.config.schema_portfolio import validate_portfolio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed result containers
# ---------------------------------------------------------------------------


@dataclass
class MetaModelConfig:
    """Top-level meta-model configuration.

    Attributes
    ----------
    name:
        Unique scenario name.
    baseline_year:
        Calendar year corresponding to project year 1.
    project_lifetime_years:
        Total simulation horizon in years.
    perfect_foresight_discount:
        Discount factor on grid-sell revenues (0–1).
    bundesland:
        German federal state code for holiday calendar.
    output_directory:
        Path to output directory.
    export_dispatch_sample:
        Whether to export quarter-hourly dispatch CSV.
    csv_separator:
        Output CSV delimiter.
    csv_decimal:
        Output CSV decimal separator.
    """

    name: str
    baseline_year: int
    project_lifetime_years: int = DEFAULT_PORTFOLIO_LIFETIME_YEARS
    perfect_foresight_discount: float = DEFAULT_PERFECT_FORESIGHT_DISCOUNT
    bundesland: str = DEFAULT_BUNDESLAND
    output_directory: str | None = None
    export_dispatch_sample: bool = True
    csv_separator: str = ";"
    csv_decimal: str = ","


@dataclass
class GenerationConfig:
    """Configuration for a single generation asset (PV plant).

    Attributes
    ----------
    type:
        Asset type (currently only ``"pv"``).
    name:
        Human-readable name for this generator.
    peak_power_kwp:
        Installed PV peak power in kWp.
    latitude:
        Geographic latitude of the PV plant.
    longitude:
        Geographic longitude of the PV plant.
    pvgis_database:
        PVGIS radiation database to use (e.g. ``"PVGIS-SARAH"``).
    degradation_rate_pct_per_year:
        Annual PV production degradation in percent.
    system_loss_pct:
        System losses at the grid connection point in percent.
    mounting_type:
        ``"free"`` (ground-mounted) or ``"building"`` (rooftop).
    azimuth_deg:
        Panel azimuth in degrees (-180 to 180, 0 = south).
    tilt_deg:
        Panel tilt angle in degrees (0 = horizontal, 90 = vertical).
    start_year:
        Simulation year (1-indexed) from which this generator is active.
    commissioning_year:
        Calendar year when the asset was commissioned.  Used to compute
        initial degradation at simulation start.  If ``None``, the asset
        is assumed to be commissioned at the baseline year (no initial
        degradation).
    lifetime_years:
        Technical lifetime of the asset in years (from commissioning).
        After ``commissioning_year + lifetime_years``, production is zero.
        If ``None``, the asset produces indefinitely.
    """

    type: str
    name: str
    peak_power_kwp: float
    latitude: float
    longitude: float
    pvgis_database: str = "PVGIS-SARAH2"
    degradation_rate_pct_per_year: float = DEFAULT_PV_DEGRADATION_RATE_PCT
    system_loss_pct: float = DEFAULT_SYSTEM_LOSS_PCT
    mounting_type: str = "free"
    azimuth_deg: float = 0.0
    tilt_deg: float = 30.0
    start_year: int = DEFAULT_FLEX_START_YEAR
    commissioning_year: int | None = None
    lifetime_years: int | None = None


@dataclass
class LoadGroupConfig:
    """Configuration for a single load group (e.g. household customers).

    Attributes
    ----------
    type:
        Load type (currently only ``"slp"``).
    name:
        Human-readable name for this load group.
    slp_type:
        BDEW standard load profile type (e.g. ``"H25"``).
    customer_count:
        Number of customers in this group.
    annual_consumption_kwh_per_customer:
        Average annual electricity consumption per customer (kWh).
    annual_growth_factor:
        Multiplicative annual load growth factor (e.g. 1.01 = +1 %/year).
    """

    type: str
    name: str
    slp_type: str
    customer_count: int
    annual_consumption_kwh_per_customer: float
    annual_growth_factor: float = 1.0


@dataclass
class BessFlexConfig:
    """Configuration for a BESS flexibility instance.

    Attributes
    ----------
    type:
        Always ``"bess"``.
    name:
        Human-readable name.
    annual_addition_kw:
        List of annual BESS power addition rates to enumerate (kW/year).
    e_to_p_ratio_hours:
        List of energy-to-power ratios to enumerate (hours).
    round_trip_efficiency_pct:
        BESS round-trip efficiency in percent.
    min_soc_pct:
        Minimum state of charge as percent of capacity.
    max_soc_pct:
        Maximum state of charge as percent of capacity.
    degradation_rate_pct_per_year:
        Annual capacity degradation in percent.
    start_year:
        Simulation year (1-indexed) from which annual additions begin.
    """

    type: str
    name: str
    annual_addition_kw: list[float]
    e_to_p_ratio_hours: list[float]
    round_trip_efficiency_pct: float = DEFAULT_BESS_RTE_PCT
    min_soc_pct: float = DEFAULT_BESS_MIN_SOC_PCT
    max_soc_pct: float = DEFAULT_BESS_MAX_SOC_PCT
    degradation_rate_pct_per_year: float = DEFAULT_BESS_DEGRADATION_RATE_PCT
    start_year: int = DEFAULT_FLEX_START_YEAR


@dataclass
class HeatPumpFlexConfig:
    """Configuration for a heat pump flexibility instance.

    Attributes
    ----------
    type:
        Always ``"heat_pump"``.
    name:
        Human-readable name.
    annual_addition_kw:
        List of annual WP power addition rates to enumerate (kW/year).
    cop_nominal:
        Nominal coefficient of performance at reference temperature.
    cop_reference_temp_c:
        Reference outdoor temperature for nominal COP (°C).
    annual_thermal_demand_mwh:
        Annual thermal demand in MWh.
    thermal_storage_kwh:
        Thermal storage capacity in kWh_th.
    start_year:
        Simulation year (1-indexed) from which annual additions begin.
    """

    type: str
    name: str
    annual_addition_kw: list[float]
    cop_nominal: float
    cop_reference_temp_c: float = 7.0
    annual_thermal_demand_mwh: float = 0.0
    thermal_storage_kwh: float = 0.0
    start_year: int = DEFAULT_FLEX_START_YEAR


@dataclass
class EVFlexConfig:
    """Configuration for an EV charging / V2G flexibility instance.

    Attributes
    ----------
    type:
        Always ``"ev_charging"``.
    name:
        Human-readable name.
    mean_kw_per_unit:
        Average charging power per EV unit (kW).
    annual_additional_units:
        List of annual unit addition rates to enumerate.
    daily_energy_demand_kwh_per_unit:
        Daily energy demand per EV unit (kWh).
    arrival_hour:
        Hour of day when EVs arrive (0–23).
    departure_hour:
        Hour of day when EVs depart (0–23).
    v2g_enabled:
        Whether vehicle-to-grid discharge is allowed.
    v2g_rte_pct:
        V2G round-trip efficiency in percent.
    min_departure_soc_pct:
        Minimum SoC at departure as percent of usable capacity.
    usable_battery_kwh_per_unit:
        Usable battery capacity per EV unit (kWh).
    start_year:
        Simulation year (1-indexed) from which annual additions begin.
    """

    type: str
    name: str
    mean_kw_per_unit: float
    annual_additional_units: list[int]
    daily_energy_demand_kwh_per_unit: float
    arrival_hour: int
    departure_hour: int
    v2g_enabled: bool = False
    v2g_rte_pct: float = DEFAULT_EV_V2G_RTE_PCT
    min_departure_soc_pct: float = DEFAULT_EV_MIN_DEPARTURE_SOC_PCT
    usable_battery_kwh_per_unit: float = 0.0
    start_year: int = DEFAULT_FLEX_START_YEAR


# Union type alias for all flex configs
FlexConfig = BessFlexConfig | HeatPumpFlexConfig | EVFlexConfig


@dataclass
class PortfolioConfig:
    """Fully validated, parsed portfolio/Systemwert configuration.

    Attributes
    ----------
    raw:
        The original validated dictionary as loaded from JSON.
    meta:
        Meta-model configuration.
    generation:
        List of generation asset configurations.
    load:
        List of load group configurations.
    flexibilities:
        List of flexibility configurations (BESS, heat pump, EV).
    price_scenarios:
        List of price-weather scenarios (reused from PV+BESS model).
    path:
        Absolute path to the source JSON file (``None`` if loaded from dict).
    """

    raw: dict[str, Any]
    meta: MetaModelConfig
    generation: list[GenerationConfig]
    load: list[LoadGroupConfig]
    flexibilities: list[FlexConfig]
    price_scenarios: list[PriceWeatherScenario]
    path: Path | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def load_portfolio(path: str | Path) -> PortfolioConfig:
    """Load and validate a portfolio JSON configuration file.

    Parameters
    ----------
    path:
        Path to the portfolio ``.json`` file.

    Returns
    -------
    PortfolioConfig
        Validated and parsed portfolio configuration.

    Raises
    ------
    FileNotFoundError
        When *path* does not exist.
    json.JSONDecodeError
        When the file contains invalid JSON.
    jsonschema.ValidationError
        When the JSON does not conform to the portfolio schema.
    ValueError
        When cross-field constraints are violated.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Portfolio config file not found: '{path}'. "
            "Check that the path is correct and the file exists."
        )

    logger.debug("Loading portfolio config from '%s'", path)

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(
            f"Invalid JSON in portfolio config file '{path}': {exc.msg}",
            exc.doc,
            exc.pos,
        ) from exc

    config = _parse_portfolio(data)
    config.path = path.resolve()

    logger.info(
        "Loaded portfolio config '%s' (lifetime=%d years, baseline=%d) from '%s'",
        config.meta.name,
        config.meta.project_lifetime_years,
        config.meta.baseline_year,
        path,
    )
    return config


def load_portfolio_dict(data: dict) -> PortfolioConfig:
    """Validate and wrap an already-parsed portfolio dictionary.

    Useful for testing or when the caller has already loaded the JSON.

    Parameters
    ----------
    data:
        Parsed portfolio configuration dictionary.

    Returns
    -------
    PortfolioConfig
        Validated and parsed portfolio configuration (``path=None``).

    Raises
    ------
    jsonschema.ValidationError
        When *data* does not conform to the portfolio schema.
    ValueError
        When cross-field constraints are violated.
    """
    return _parse_portfolio(data)


# ---------------------------------------------------------------------------
# Internal parsing
# ---------------------------------------------------------------------------


def _parse_portfolio(data: dict) -> PortfolioConfig:
    """Validate and parse a portfolio dictionary into typed dataclasses."""
    validate_portfolio(data)

    meta = _parse_meta(data["meta_model"])
    generation = [_parse_generation(g) for g in data["portfolio"]["generation"]]
    load = [_parse_load_group(lg) for lg in data["portfolio"]["load"]]
    flexibilities = [_parse_flex(f) for f in data["flexibilities"]]
    price_scenarios = _parse_price_scenarios(data["price_inputs"]["scenarios"])

    return PortfolioConfig(
        raw=data,
        meta=meta,
        generation=generation,
        load=load,
        flexibilities=flexibilities,
        price_scenarios=price_scenarios,
        path=None,
    )


def _parse_meta(raw: dict) -> MetaModelConfig:
    """Parse the meta_model block."""
    output = raw.get("output", {})
    return MetaModelConfig(
        name=raw["name"],
        baseline_year=raw["baseline_year"],
        project_lifetime_years=raw.get(
            "project_lifetime_years", DEFAULT_PORTFOLIO_LIFETIME_YEARS
        ),
        perfect_foresight_discount=raw.get(
            "perfect_foresight_discount", DEFAULT_PERFECT_FORESIGHT_DISCOUNT
        ),
        bundesland=raw.get("bundesland", DEFAULT_BUNDESLAND),
        output_directory=output.get("directory"),
        export_dispatch_sample=output.get("export_dispatch_sample", True),
        csv_separator=output.get("csv_separator", ";"),
        csv_decimal=output.get("csv_decimal", ","),
    )


def _parse_generation(raw: dict) -> GenerationConfig:
    """Parse a single generation entry."""
    loc = raw["location"]
    return GenerationConfig(
        type=raw["type"],
        name=raw["name"],
        peak_power_kwp=float(raw["peak_power_kwp"]),
        latitude=float(loc["latitude"]),
        longitude=float(loc["longitude"]),
        pvgis_database=loc.get("pvgis_database", "PVGIS-SARAH2"),
        degradation_rate_pct_per_year=float(
            raw.get("degradation_rate_pct_per_year", DEFAULT_PV_DEGRADATION_RATE_PCT)
        ),
        system_loss_pct=float(
            raw.get("system_loss_pct", DEFAULT_SYSTEM_LOSS_PCT)
        ),
        mounting_type=raw.get("mounting_type", "free"),
        azimuth_deg=float(raw.get("azimuth_deg", 0.0)),
        tilt_deg=float(raw.get("tilt_deg", 30.0)),
        start_year=int(raw.get("start_year", DEFAULT_FLEX_START_YEAR)),
        commissioning_year=(
            int(raw["commissioning_year"]) if "commissioning_year" in raw else None
        ),
        lifetime_years=(
            int(raw["lifetime_years"]) if "lifetime_years" in raw else None
        ),
    )


def _parse_load_group(raw: dict) -> LoadGroupConfig:
    """Parse a single load group entry."""
    return LoadGroupConfig(
        type=raw["type"],
        name=raw["name"],
        slp_type=raw["slp_type"],
        customer_count=int(raw["customer_count"]),
        annual_consumption_kwh_per_customer=float(
            raw["annual_consumption_kwh_per_customer"]
        ),
        annual_growth_factor=float(raw.get("annual_growth_factor", 1.0)),
    )


def _parse_flex(raw: dict) -> FlexConfig:
    """Parse a flexibility entry and dispatch to the correct typed config."""
    flex_type = raw["type"]
    if flex_type == "bess":
        return _parse_flex_bess(raw)
    elif flex_type == "heat_pump":
        return _parse_flex_heat_pump(raw)
    elif flex_type == "ev_charging":
        return _parse_flex_ev(raw)
    else:
        raise ValueError(f"Unknown flexibility type: '{flex_type}'")


def _parse_flex_bess(raw: dict) -> BessFlexConfig:
    """Parse a BESS flexibility entry."""
    return BessFlexConfig(
        type="bess",
        name=raw["name"],
        annual_addition_kw=[float(v) for v in raw["annual_addition_kw"]],
        e_to_p_ratio_hours=[float(v) for v in raw["e_to_p_ratio_hours"]],
        round_trip_efficiency_pct=float(
            raw.get("round_trip_efficiency_pct", DEFAULT_BESS_RTE_PCT)
        ),
        min_soc_pct=float(raw.get("min_soc_pct", DEFAULT_BESS_MIN_SOC_PCT)),
        max_soc_pct=float(raw.get("max_soc_pct", DEFAULT_BESS_MAX_SOC_PCT)),
        degradation_rate_pct_per_year=float(
            raw.get("degradation_rate_pct_per_year", DEFAULT_BESS_DEGRADATION_RATE_PCT)
        ),
        start_year=int(raw.get("start_year", DEFAULT_FLEX_START_YEAR)),
    )


def _parse_flex_heat_pump(raw: dict) -> HeatPumpFlexConfig:
    """Parse a heat pump flexibility entry."""
    return HeatPumpFlexConfig(
        type="heat_pump",
        name=raw["name"],
        annual_addition_kw=[float(v) for v in raw["annual_addition_kw"]],
        cop_nominal=float(raw["cop_nominal"]),
        cop_reference_temp_c=float(raw.get("cop_reference_temp_c", 7.0)),
        annual_thermal_demand_mwh=float(raw["annual_thermal_demand_mwh"]),
        thermal_storage_kwh=float(raw["thermal_storage_kwh"]),
        start_year=int(raw.get("start_year", DEFAULT_FLEX_START_YEAR)),
    )


def _parse_flex_ev(raw: dict) -> EVFlexConfig:
    """Parse an EV charging flexibility entry."""
    tw = raw["time_window"]
    return EVFlexConfig(
        type="ev_charging",
        name=raw["name"],
        mean_kw_per_unit=float(raw["mean_kw_per_unit"]),
        annual_additional_units=[int(v) for v in raw["annual_additional_units"]],
        daily_energy_demand_kwh_per_unit=float(
            raw["daily_energy_demand_kwh_per_unit"]
        ),
        arrival_hour=int(tw["arrival_hour"]),
        departure_hour=int(tw["departure_hour"]),
        v2g_enabled=bool(raw.get("v2g_enabled", False)),
        v2g_rte_pct=float(raw.get("v2g_rte_pct", DEFAULT_EV_V2G_RTE_PCT)),
        min_departure_soc_pct=float(
            raw.get("min_departure_soc_pct", DEFAULT_EV_MIN_DEPARTURE_SOC_PCT)
        ),
        usable_battery_kwh_per_unit=float(raw["usable_battery_kwh_per_unit"]),
        start_year=int(raw.get("start_year", DEFAULT_FLEX_START_YEAR)),
    )


def _parse_price_scenarios(
    scenarios_raw: list[dict],
) -> list[PriceWeatherScenario]:
    """Parse price-weather scenarios, reusing the existing dataclass."""
    result: list[PriceWeatherScenario] = []
    for s in scenarios_raw:
        result.append(
            PriceWeatherScenario(
                name=s["name"],
                label=s.get("label", s["name"]),
                csv_column=s["csv_column"],
                weather_year=int(s["weather_year"]),
                weight=float(s["weight"]),
                is_central=bool(s.get("is_central", False)),
                price_csv=s["price_csv"],
                inflation_on_input_data=s.get("inflation_on_input_data", False),
                csv_separator=s["csv_separator"],
                csv_decimal=s["csv_decimal"],
                csv_timestamp_column=s["csv_timestamp_column"],
                csv_timestamp_format=s["csv_timestamp_format"],
            )
        )

    logger.info(
        "Parsed %d price-weather scenario(s): %s",
        len(result),
        [s.name for s in result],
    )
    return result
