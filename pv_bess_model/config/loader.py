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
    DEFAULT_INFLATION_VALUE_COLUMN,
    DEFAULT_INFLATION_YEAR_COLUMN,
    MIN_PRICE_TIMESERIES_HOURS,
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
    def pv_peak_kwp(self) -> float:
        """PV peak power in kWp."""
        return float(self.pv["design"]["peak_power_kwp"])

    @property
    def pv_sub_arrays(self) -> list[dict] | None:
        """Optional dual-azimuth sub-array definitions.

        Returns ``None`` when no sub-arrays are configured (single-azimuth mode).
        """
        return self.pv["design"].get("sub_arrays")

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
        Mapping of column name → numpy array (€/kWh).
        The ``"timestamp"`` column is excluded; only numeric price columns are
        included.
    n_hours:
        Number of hourly rows loaded.
    """

    columns: dict[str, np.ndarray]
    n_hours: int

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
        Path to the price CSV file.
    inflation_on_input_data:
        Whether to apply inflation to price data.
    csv_separator:
        CSV delimiter.
    csv_decimal:
        CSV decimal separator.
    csv_timestamp_column:
        Timestamp column name.
    csv_timestamp_format:
        Timestamp format string.
    pv_timeseries_15min:
        Quarter-hourly PV production array (35 040 values), set after
        PVGIS fetch + alignment + conversion.
    price_per_year:
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
    inflation_on_input_data: bool | None = None
    csv_separator: str | None = None
    csv_decimal: str | None = None
    csv_timestamp_column: str | None = None
    csv_timestamp_format: str | None = None
    pv_timeseries_15min: np.ndarray | None = field(default=None, repr=False)
    price_per_year: np.ndarray | None = field(default=None, repr=False)


@dataclass
class InflationTimeseriesConfig:
    """Configuration for loading an inflation timeseries from CSV.

    Attributes
    ----------
    csv_path:
        Path to the inflation CSV file.
    year_column:
        Column name containing calendar years.
    inflation_column:
        Column name containing annual inflation rates in percent.
    csv_separator:
        CSV delimiter character.
    csv_decimal:
        CSV decimal separator character.
    """

    csv_path: str
    year_column: str = DEFAULT_INFLATION_YEAR_COLUMN
    inflation_column: str = DEFAULT_INFLATION_VALUE_COLUMN
    csv_separator: str = CSV_DELIMITER
    csv_decimal: str = CSV_INPUT_DECIMAL_SEPARATOR


# ---------------------------------------------------------------------------
# Cross-field validators
# ---------------------------------------------------------------------------

_SUB_ARRAY_POWER_TOLERANCE_KWP = 0.1
"""Maximum allowed difference between sum of sub-array powers and peak_power_kwp."""


def _validate_pv_sub_arrays(data: dict) -> None:
    """Validate that sub-array powers sum to peak_power_kwp (if sub_arrays present).

    Raises
    ------
    ValueError
        When the sum of sub-array ``power_kwp`` values deviates from
        ``peak_power_kwp`` by more than :data:`_SUB_ARRAY_POWER_TOLERANCE_KWP`.
    """
    pv_design = data["project_settings"]["technology"]["pv"]["design"]
    sub_arrays = pv_design.get("sub_arrays")
    if sub_arrays is None:
        return

    peak_power = float(pv_design["peak_power_kwp"])
    sub_sum = sum(float(sa["power_kwp"]) for sa in sub_arrays)

    if abs(sub_sum - peak_power) > _SUB_ARRAY_POWER_TOLERANCE_KWP:
        raise ValueError(
            f"Sum of sub-array powers ({sub_sum:.1f} kWp) does not match "
            f"peak_power_kwp ({peak_power:.1f} kWp). "
            f"Maximum allowed deviation: {_SUB_ARRAY_POWER_TOLERANCE_KWP} kWp."
        )


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
    _validate_pv_sub_arrays(data)

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
    _validate_pv_sub_arrays(data)
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


def load_price_csv(
    path: str | Path,
    required_columns: list[str],
    commissioning_year: int | None = None,
    delimiter: str = CSV_DELIMITER,
    decimal: str = CSV_INPUT_DECIMAL_SEPARATOR,
    timestamp_column: str = CSV_TIMESTAMP_COLUMN,
    timestamp_format: str | None = None,
) -> PriceData:
    """Load and validate an electricity price CSV file.

    The CSV must contain a timestamp column (ISO 8601) and at least one
    numeric price column. All price values are expected in **€/kWh**.

    Parameters
    ----------
    path:
        Path to the price CSV file.
    required_columns:
        List of column names that must be present (e.g. ``["MID"]``).
        The timestamp column is always required implicitly when
        *commissioning_year* is set.
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
        missing columns, or no timestamp column when *commissioning_year*
        is set).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Price CSV file not found: '{path}'. "
            "Check the 'price_csv' path in the scenario JSON."
        )

    logger.debug("Loading price CSV from '%s'", path)

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

    # --- read values (already in €/kWh, no conversion needed) --------------
    columns_out: dict[str, np.ndarray] = {}
    for col in required_columns:
        columns_out[col] = df[col].to_numpy(dtype=float)

    logger.info(
        "Loaded price CSV '%s': %d rows, columns=%s",
        path,
        n_rows,
        required_columns,
    )
    return PriceData(
        columns=columns_out,
        n_hours=n_rows,
    )


def parse_inflation_timeseries_config(
    scenario_config: ScenarioConfig,
) -> InflationTimeseriesConfig | None:
    """Parse the optional inflation timeseries config from the scenario.

    Parameters
    ----------
    scenario_config:
        Validated scenario configuration.

    Returns
    -------
    InflationTimeseriesConfig | None
        Parsed config, or ``None`` if no inflation timeseries is configured.
    """
    price_inputs = scenario_config.finance.get("price_inputs", {})
    ts_cfg = price_inputs.get("inflation_timeseries")
    if ts_cfg is None:
        return None

    return InflationTimeseriesConfig(
        csv_path=ts_cfg["csv_path"],
        year_column=ts_cfg.get("year_column", DEFAULT_INFLATION_YEAR_COLUMN),
        inflation_column=ts_cfg.get("inflation_column", DEFAULT_INFLATION_VALUE_COLUMN),
        csv_separator=ts_cfg.get("csv_separator", CSV_DELIMITER),
        csv_decimal=ts_cfg.get("csv_decimal", CSV_INPUT_DECIMAL_SEPARATOR),
    )


def load_inflation_csv(
    config: InflationTimeseriesConfig,
    scenario_path: Path | None = None,
) -> dict[int, float]:
    """Load an inflation timeseries CSV and return year → rate mapping.

    The CSV must contain a year column (integer) and an inflation column
    (percentage, e.g. 2.5 means 2.5 %).  The returned dictionary maps
    calendar year to inflation rate as a fraction (e.g. 2.5 → 0.025).

    Parameters
    ----------
    config:
        Inflation timeseries configuration.
    scenario_path:
        Path to the scenario JSON file (used to resolve relative CSV paths).

    Returns
    -------
    dict[int, float]
        Mapping of calendar year → annual inflation rate as fraction.

    Raises
    ------
    FileNotFoundError
        When the CSV file does not exist.
    ValueError
        When the CSV is malformed or contains invalid data.
    """
    csv_path = Path(config.csv_path)
    if not csv_path.is_absolute() and scenario_path is not None:
        csv_path = scenario_path.parent / csv_path

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Inflation timeseries CSV not found: '{csv_path}'. "
            "Check the 'csv_path' in price_inputs.inflation_timeseries."
        )

    logger.debug("Loading inflation timeseries from '%s'", csv_path)

    try:
        df = pd.read_csv(
            csv_path,
            sep=config.csv_separator,
            decimal=config.csv_decimal,
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to parse inflation CSV '{csv_path}': {exc}"
        ) from exc

    # Validate columns
    for col in [config.year_column, config.inflation_column]:
        if col not in df.columns:
            raise ValueError(
                f"Inflation CSV '{csv_path}' is missing column '{col}'. "
                f"Available columns: {sorted(df.columns.tolist())}."
            )

    # Validate no NaN
    if df[config.year_column].isna().any() or df[config.inflation_column].isna().any():
        raise ValueError(
            f"Inflation CSV '{csv_path}' contains NaN values. "
            "No missing values are allowed."
        )

    result: dict[int, float] = {}
    for _, row in df.iterrows():
        year = int(row[config.year_column])
        rate_pct = float(row[config.inflation_column])
        result[year] = rate_pct / 100.0

    logger.info(
        "Loaded inflation timeseries from '%s': %d years (%d–%d)",
        csv_path,
        len(result),
        min(result.keys()),
        max(result.keys()),
    )
    return result


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
