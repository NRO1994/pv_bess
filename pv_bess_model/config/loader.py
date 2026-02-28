"""Load and validate scenario JSON files and CSV price timeseries.

Public API
----------
load_scenario(path)      – Parse + validate a scenario JSON file.
load_price_csv(path, ...) – Load a price CSV and return per-column numpy arrays.

All error messages name the specific field or row that caused the problem so
the user can fix the JSON or CSV without guessing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pv_bess_model.config.defaults import (
    CSV_DELIMITER,
    CSV_INPUT_DECIMAL_SEPARATOR,
    CSV_TIMESTAMP_COLUMN,
    DEFAULT_DEBT_SIZING_DOWNSIDE_PCT,
    KWH_TO_MWH,
    MIN_PRICE_TIMESERIES_HOURS,
    PRICE_UNIT_EUR_PER_KWH,
    PRICE_UNIT_EUR_PER_MWH,
)
from pv_bess_model.config.schema import validate_scenario

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed result containers
# ---------------------------------------------------------------------------


@dataclass
class ScenarioConfig:
    """Fully validated, parsed scenario configuration.

    Attributes
    ----------
    raw:
        The original validated dictionary as loaded from JSON.  All other
        attributes are convenience accessors into ``raw`` to avoid repeated
        nested look-ups in calling code.
    name:
        Scenario name (``scenario.name``).
    operating_mode:
        ``"green"`` or ``"grey"``.
    lifetime_years:
        Project lifetime in years.
    path:
        Absolute path to the source JSON file (``None`` if loaded from a dict).
    """

    raw: dict[str, Any]
    name: str
    operating_mode: str
    lifetime_years: int
    commissioning_year: int
    path: Path | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Convenience properties (thin accessors into raw)
    # ------------------------------------------------------------------

    @property
    def project_settings(self) -> dict:
        """Shortcut to ``raw["project_settings"]``."""
        return self.raw["project_settings"]

    @property
    def technology(self) -> dict:
        """Shortcut to ``raw["project_settings"]["technology"]``."""
        return self.project_settings["technology"]

    @property
    def finance(self) -> dict:
        """Shortcut to ``raw["project_settings"]["finance"]``."""
        return self.project_settings["finance"]

    @property
    def pv(self) -> dict:
        """Shortcut to the PV technology block."""
        return self.technology["pv"]

    @property
    def bess(self) -> dict:
        """Shortcut to the BESS technology block."""
        return self.technology["bess"]

    @property
    def grid_connection(self) -> dict:
        """Shortcut to the grid-connection technology block."""
        return self.technology["grid_connection"]

    @property
    def monte_carlo(self) -> dict:
        """Shortcut to ``raw["scenario"]["monte_carlo"]`` (empty dict if absent)."""
        return self.raw["scenario"].get("monte_carlo", {})

    @property
    def mc_enabled(self) -> bool:
        """``True`` if Monte Carlo simulation is enabled."""
        return bool(self.monte_carlo.get("enabled", False))

    @property
    def price_csv_path(self) -> str:
        """Relative or absolute path to the day-ahead price CSV file."""
        return self.finance["price_inputs"]["day_ahead_csv"]

    @property
    def price_unit(self) -> str:
        """Price unit string: ``"eur_per_mwh"`` or ``"eur_per_kwh"``."""
        return self.finance["price_inputs"]["price_unit"]

    @property
    def pv_peak_kwp(self) -> float:
        """PV peak power in kWp."""
        return float(self.pv["design"]["peak_power_kwp"])

    @property
    def bess_scale_pct_list(self) -> list[float]:
        """BESS scale percentages for the grid search."""
        return [float(v) for v in self.bess["design_space"]["scale_pct_of_pv"]]

    @property
    def e_to_p_ratio_hours_list(self) -> list[float]:
        """Energy-to-power ratios (hours) for the grid search."""
        return [float(v) for v in self.bess["design_space"]["e_to_p_ratio_hours"]]

    @property
    def debt_sizing_downside_pct(self) -> float:
        """Downside percentage for debt sizing (replaces debt_uses_p90)."""
        return float(
            self.finance.get("debt_sizing_downside_pct", DEFAULT_DEBT_SIZING_DOWNSIDE_PCT)
        )


@dataclass
class PriceData:
    """Parsed and validated electricity price timeseries.

    Attributes
    ----------
    columns:
        Mapping of column name → numpy array (€/kWh, regardless of input unit).
        The ``"timestamp"`` column is excluded; only numeric price columns are
        included.
    n_hours:
        Number of hourly rows loaded.
    price_unit_input:
        Original price unit string from the scenario JSON (before conversion).
    """

    columns: dict[str, np.ndarray]
    n_hours: int
    price_unit_input: str

    def get_column(self, name: str) -> np.ndarray:
        """Return the price array for *name* (€/kWh).

        Raises
        ------
        KeyError
            If *name* is not a known column.
        """
        if name not in self.columns:
            available = ", ".join(sorted(self.columns))
            raise KeyError(
                f"Price column '{name}' not found. Available columns: {available}"
            )
        return self.columns[name]


@dataclass
class PriceWeatherScenario:
    """A single price-weather scenario mapping a price column to a weather year.

    Each scenario combines a specific price timeseries (CSV column) with a
    specific historical weather year.  The weight determines the sampling
    probability in Monte Carlo.

    Attributes
    ----------
    name:
        Unique identifier for this scenario (e.g. ``"mid_2017"``).
    label:
        Human-readable label for output/display.
    csv_column:
        Column name in the price CSV file.
    weather_year:
        Historical calendar year for PVGIS weather data.
    weight:
        Sampling probability weight (all weights must sum to 1.0).
    is_central:
        Whether this is the central scenario used for grid search.
    price_csv:
        Path to the price CSV file (inherited from parent if not set).
    price_unit:
        Unit of prices in CSV (inherited from parent if not set).
    inflation_on_input_data:
        Whether to apply inflation to price data (inherited if not set).
    csv_separator:
        CSV delimiter (inherited from parent if not set).
    csv_decimal:
        CSV decimal separator (inherited from parent if not set).
    csv_timestamp_column:
        Timestamp column name (inherited from parent if not set).
    csv_timestamp_format:
        Timestamp format string (inherited from parent if not set).
    pv_timeseries_15min:
        Quarter-hourly PV production array (35 040 values), set after
        PVGIS fetch + alignment + conversion.
    price_timeseries_extended:
        Extended price array covering the full project lifetime (€/kWh),
        set after price CSV loading.
    """

    name: str
    label: str
    csv_column: str
    weather_year: int
    weight: float
    is_central: bool = False
    price_csv: str | None = None
    price_unit: str | None = None
    inflation_on_input_data: bool | None = None
    csv_separator: str | None = None
    csv_decimal: str | None = None
    csv_timestamp_column: str | None = None
    csv_timestamp_format: str | None = None
    pv_timeseries_15min: np.ndarray | None = field(default=None, repr=False)
    price_timeseries_extended: np.ndarray | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def load_scenario(path: str | Path) -> ScenarioConfig:
    """Load and validate a scenario JSON file.

    Parameters
    ----------
    path:
        Path to the scenario ``.json`` file.

    Returns
    -------
    ScenarioConfig
        Validated and parsed scenario configuration.

    Raises
    ------
    FileNotFoundError
        When *path* does not exist.
    json.JSONDecodeError
        When the file contains invalid JSON.
    jsonschema.ValidationError
        When the JSON does not conform to the scenario schema.
    ValueError
        When cross-field constraints are violated (e.g. MC weight sum ≠ 1).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Scenario file not found: '{path}'. "
            "Check that the path is correct and the file exists."
        )

    logger.debug("Loading scenario from '%s'", path)

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(
            f"Invalid JSON in scenario file '{path}': {exc.msg}",
            exc.doc,
            exc.pos,
        ) from exc

    validate_scenario(data)  # raises ValidationError on schema violations

    ps = data["project_settings"]
    config = ScenarioConfig(
        raw=data,
        name=data["scenario"]["name"],
        operating_mode=ps["operating_mode"],
        lifetime_years=int(ps["lifetime_years"]),
        commissioning_year=int(ps["commissioning_year"]),
        path=path.resolve(),
    )

    logger.info(
        "Loaded scenario '%s' (mode=%s, lifetime=%d years, commissioning=%d) from '%s'",
        config.name,
        config.operating_mode,
        config.lifetime_years,
        config.commissioning_year,
        path,
    )
    return config


def load_scenario_dict(data: dict) -> ScenarioConfig:
    """Validate and wrap an already-parsed scenario dictionary.

    Useful for testing or when the caller has already loaded the JSON.

    Parameters
    ----------
    data:
        Parsed scenario dictionary.

    Returns
    -------
    ScenarioConfig
        Validated and parsed scenario configuration (``path=None``).

    Raises
    ------
    jsonschema.ValidationError
        When *data* does not conform to the scenario schema.
    ValueError
        When cross-field constraints are violated.
    """
    validate_scenario(data)
    ps = data["project_settings"]
    return ScenarioConfig(
        raw=data,
        name=data["scenario"]["name"],
        operating_mode=ps["operating_mode"],
        lifetime_years=int(ps["lifetime_years"]),
        commissioning_year=int(ps["commissioning_year"]),
        path=None,
    )


def parse_scenarios(scenario_config: ScenarioConfig) -> list[PriceWeatherScenario]:
    """Parse the price-weather scenarios array from a validated scenario config.

    Each scenario in the JSON ``price_inputs.scenarios`` array is parsed into a
    :class:`PriceWeatherScenario` dataclass.  Fields not set at the scenario
    level inherit defaults from the parent ``price_inputs`` block.

    Parameters
    ----------
    scenario_config:
        Validated scenario configuration.

    Returns
    -------
    list[PriceWeatherScenario]
        Parsed scenarios with inherited defaults.

    Raises
    ------
    ValueError
        When no scenarios are defined in the price_inputs block.
    """
    price_inputs = scenario_config.finance.get("price_inputs", {})
    scenarios_raw = price_inputs.get("scenarios", [])

    if not scenarios_raw:
        raise ValueError(
            "No scenarios defined in project_settings.finance.price_inputs.scenarios. "
            "At least one scenario is required."
        )

    # Parent-level defaults
    parent_csv = price_inputs.get("day_ahead_csv")
    parent_unit = price_inputs.get("price_unit")
    parent_inflation = price_inputs.get("inflation_on_input_data", False)
    parent_sep = price_inputs.get("csv_separator")
    parent_dec = price_inputs.get("csv_decimal")
    parent_ts_col = price_inputs.get("csv_timestamp_column")
    parent_ts_fmt = price_inputs.get("csv_timestamp_format")

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
                price_csv=s.get("price_csv", parent_csv),
                price_unit=s.get("price_unit", parent_unit),
                inflation_on_input_data=s.get("inflation_on_input_data", parent_inflation),
                csv_separator=s.get("csv_separator", parent_sep),
                csv_decimal=s.get("csv_decimal", parent_dec),
                csv_timestamp_column=s.get("csv_timestamp_column", parent_ts_col),
                csv_timestamp_format=s.get("csv_timestamp_format", parent_ts_fmt),
            )
        )

    logger.info(
        "Parsed %d price-weather scenario(s): %s",
        len(result),
        [s.name for s in result],
    )
    return result


def load_price_csv(
    path: str | Path,
    required_columns: list[str],
    price_unit: str,
    commissioning_year: int | None = None,
    delimiter: str = CSV_DELIMITER,
    decimal: str = CSV_INPUT_DECIMAL_SEPARATOR,
    timestamp_column: str = CSV_TIMESTAMP_COLUMN,
    timestamp_format: str | None = None,
) -> PriceData:
    """Load and validate an electricity price CSV file.

    The CSV must contain a timestamp column (ISO 8601) and at least one
    numeric price column. The function converts all price values to **€/kWh**
    regardless of the input unit declared in *price_unit*.

    Parameters
    ----------
    path:
        Path to the price CSV file.
    required_columns:
        List of column names that must be present (e.g. ``["MID"]``).
        The timestamp column is always required implicitly when
        *commissioning_year* is set.
    price_unit:
        Unit of the price values in the CSV: ``"eur_per_mwh"`` or
        ``"eur_per_kwh"``.
    commissioning_year:
        If provided, rows with timestamps before January 1st of the
        commissioning year are discarded before validation and conversion.
    delimiter:
        Column delimiter used in the CSV file (default: ``CSV_DELIMITER``).
    decimal:
        Decimal separator used for numeric values in the CSV
        (default: ``CSV_INPUT_DECIMAL_SEPARATOR`` = ``"."``).
    timestamp_column:
        Name of the column containing ISO 8601 timestamps
        (default: ``CSV_TIMESTAMP_COLUMN`` = ``"timestamp"``).
    timestamp_format:
        ``strftime``-compatible format string for parsing timestamps, e.g.
        ``"%Y-%m-%dT%H:%M:%S"``.  ``None`` (default) lets pandas
        auto-detect the format.

    Returns
    -------
    PriceData
        Validated price data with values in €/kWh.

    Raises
    ------
    FileNotFoundError
        When *path* does not exist.
    ValueError
        When the CSV fails any validation check (too few rows, NaN values,
        missing columns, unknown price unit, or no timestamp column
        when *commissioning_year* is set).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Price CSV file not found: '{path}'. "
            "Check the 'day_ahead_csv' path in the scenario JSON."
        )

    if price_unit not in (PRICE_UNIT_EUR_PER_MWH, PRICE_UNIT_EUR_PER_KWH):
        raise ValueError(
            f"Unknown price_unit '{price_unit}'. "
            f"Must be '{PRICE_UNIT_EUR_PER_MWH}' or '{PRICE_UNIT_EUR_PER_KWH}'."
        )

    logger.debug("Loading price CSV from '%s' (unit=%s)", path, price_unit)

    try:
        df = pd.read_csv(path, sep=delimiter, decimal=decimal)
    except Exception as exc:
        raise ValueError(f"Failed to parse price CSV '{path}': {exc}") from exc

    # --- filter by commissioning year --------------------------------------
    if commissioning_year is not None:
        df = _filter_from_commissioning_year(
            df, path, commissioning_year,
            timestamp_column=timestamp_column,
            timestamp_format=timestamp_format,
        )

    # --- column presence ---------------------------------------------------
    _check_required_columns(df, path, required_columns)

    # --- row count ---------------------------------------------------------
    n_rows = len(df)
    if n_rows < MIN_PRICE_TIMESERIES_HOURS:
        raise ValueError(
            f"Price CSV '{path}' has only {n_rows} rows. "
            f"A minimum of {MIN_PRICE_TIMESERIES_HOURS} hourly rows "
            "(one full year) is required."
        )

    # --- NaN check ---------------------------------------------------------
    price_cols = [c for c in required_columns]
    _check_no_nan(df, path, price_cols)

    # --- unit conversion ---------------------------------------------------
    if price_unit == PRICE_UNIT_EUR_PER_MWH:
        conversion_factor = KWH_TO_MWH  # €/MWh → €/kWh : divide by 1000
    else:
        conversion_factor = 1.0  # already €/kWh

    columns_out: dict[str, np.ndarray] = {}
    for col in required_columns:
        values = df[col].to_numpy(dtype=float)
        columns_out[col] = values * conversion_factor

    logger.info(
        "Loaded price CSV '%s': %d rows, columns=%s, unit=%s",
        path,
        n_rows,
        required_columns,
        price_unit,
    )
    return PriceData(
        columns=columns_out,
        n_hours=n_rows,
        price_unit_input=price_unit,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _filter_from_commissioning_year(
    df: pd.DataFrame,
    path: Path,
    commissioning_year: int,
    timestamp_column: str = CSV_TIMESTAMP_COLUMN,
    timestamp_format: str | None = None,
) -> pd.DataFrame:
    """Discard rows with timestamps before January 1st of *commissioning_year*.

    The function parses the timestamp column, drops all rows whose
    timestamp falls before the commissioning year, and resets the index.

    Parameters
    ----------
    df:
        DataFrame loaded from the price CSV.
    path:
        Original file path (for error messages only).
    commissioning_year:
        Rows before January 1st of this year are discarded.
    timestamp_column:
        Name of the column containing timestamps
        (default: ``CSV_TIMESTAMP_COLUMN``).
    timestamp_format:
        ``strftime``-compatible format string for parsing timestamps.
        ``None`` (default) lets pandas auto-detect the format.

    Raises
    ------
    ValueError
        If the CSV has no timestamp column or if none of its timestamps
        can be parsed.
    """
    if timestamp_column not in df.columns:
        raise ValueError(
            f"Price CSV '{path}' has no '{timestamp_column}' column, which is "
            "required when commissioning_year filtering is enabled."
        )

    try:
        if timestamp_format is not None:
            timestamps = pd.to_datetime(df[timestamp_column], format=timestamp_format)
        else:
            timestamps = pd.to_datetime(df[timestamp_column])
    except Exception as exc:
        raise ValueError(
            f"Failed to parse '{timestamp_column}' column in price CSV '{path}': {exc}"
        ) from exc

    cutoff = pd.Timestamp(year=commissioning_year, month=1, day=1)
    mask = timestamps >= cutoff
    n_before = len(df)
    df = df.loc[mask].reset_index(drop=True)
    n_dropped = n_before - len(df)

    if n_dropped > 0:
        logger.info(
            "Filtered price CSV '%s': dropped %d rows before %d-01-01 "
            "(%d rows remaining).",
            path,
            n_dropped,
            commissioning_year,
            len(df),
        )

    return df



def _check_required_columns(
    df: pd.DataFrame,
    path: Path,
    required_columns: list[str],
) -> None:
    """Raise ValueError listing all missing columns."""
    available = set(df.columns)
    missing = [c for c in required_columns if c not in available]
    if missing:
        raise ValueError(
            f"Price CSV '{path}' is missing required column(s): "
            f"{missing}. "
            f"Available columns: {sorted(available)}."
        )


def _check_no_nan(
    df: pd.DataFrame,
    path: Path,
    columns: list[str],
) -> None:
    """Raise ValueError naming each column that contains NaN values."""
    nan_cols = []
    for col in columns:
        if df[col].isna().any():
            n_nan = int(df[col].isna().sum())
            first_idx = int(df[col].isna().idxmax())
            nan_cols.append(f"'{col}' ({n_nan} NaN value(s), first at row {first_idx})")

    if nan_cols:
        raise ValueError(
            f"Price CSV '{path}' contains NaN values in the following "
            f"column(s): {'; '.join(nan_cols)}. "
            "No missing values are allowed."
        )


def extend_price_timeseries(
    prices: np.ndarray,
    target_years: int,
    hours_per_year: int,
) -> np.ndarray:
    """Extend a price timeseries to cover *target_years* full years.

    If the input covers more than one year, the **last full year** is repeated
    as many times as needed. If it covers exactly one year, that year is
    repeated. The returned array has exactly ``target_years × hours_per_year``
    elements.

    Parameters
    ----------
    prices:
        Input price array (at least *hours_per_year* elements, €/kWh).
    target_years:
        Number of project years to cover.
    hours_per_year:
        Hourly timesteps per year (typically 8 760).

    Returns
    -------
    np.ndarray
        Extended array of length ``target_years × hours_per_year``.

    Raises
    ------
    ValueError
        When *prices* is shorter than one full year.
    """
    if len(prices) < hours_per_year:
        raise ValueError(
            f"Price timeseries has {len(prices)} values, but at least "
            f"{hours_per_year} (one full year) are required."
        )

    n_full_years_available = len(prices) // hours_per_year
    last_year_start = (n_full_years_available - 1) * hours_per_year
    last_year = prices[last_year_start : last_year_start + hours_per_year]

    n_years_in_input = min(n_full_years_available, target_years)
    base = prices[: n_years_in_input * hours_per_year]

    if n_years_in_input >= target_years:
        return base[: target_years * hours_per_year]

    extra_years = target_years - n_years_in_input
    extension = np.tile(last_year, extra_years)
    return np.concatenate([base, extension])
