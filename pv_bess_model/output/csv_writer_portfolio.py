"""Write portfolio/Systemwert results to CSV files.

Four output files are produced per portfolio run:

1. ``{name}_baseline.csv``           – World A annual system costs (no flex).
2. ``{name}_system_value.csv``       – One row per enumeration point.
3. ``{name}_marginal_value.csv``     – Marginal value curve data.
4. ``{name}_dispatch_sample.csv``    – Quarter-hourly dispatch for one year.

All monetary values are in EUR, energy in kWh, prices in EUR/kWh.

Public API
----------
write_baseline_csv               – Write World A annual system costs.
write_system_value_csv           – Write enumeration results.
write_marginal_value_csv         – Write marginal value curves.
write_portfolio_dispatch_sample_csv – Write quarter-hourly dispatch data.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pv_bess_model.config.defaults import (
    CSV_DECIMAL_SEPARATOR,
    CSV_DELIMITER,
    CSV_TIMESTAMP_COLUMN,
    CSV_TIMESTAMP_FORMAT,
    INTERVALS_PER_DAY,
    _MAX_LOCK_RETRIES,
)
from pv_bess_model.dispatch.engine_portfolio import PortfolioAnnualResult
from pv_bess_model.dispatch.optimizer_portfolio import PortfolioDailyResult
from pv_bess_model.output.formatting import fmt_currency, fmt_float
from pv_bess_model.portfolio.marginal_value import MarginalValuePoint
from pv_bess_model.portfolio.system_value import SystemValueResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Baseline CSV (World A)
# ---------------------------------------------------------------------------


def write_baseline_csv(
    path: Path | str,
    world_a_annual_costs: list[float],
    baseline_year: int = 2027,
    delimiter: str = CSV_DELIMITER,
    decimal: str = CSV_DECIMAL_SEPARATOR,
) -> None:
    """Write World A annual system costs (no flexibility).

    One row per project year with the system cost and cumulative cost.

    Parameters
    ----------
    path:
        Destination file path.
    world_a_annual_costs:
        Annual system costs from World A, length = lifetime.
    baseline_year:
        Calendar year corresponding to project year 1.
    delimiter:
        CSV column delimiter.
    decimal:
        Decimal separator for numeric values.
    """
    rows: list[dict[str, str]] = []
    cumulative = 0.0

    for i, cost in enumerate(world_a_annual_costs):
        cumulative += cost
        rows.append({
            "year": str(baseline_year + i),
            "project_year": str(i + 1),
            "system_cost_eur": fmt_currency(cost, decimal=decimal),
            "cumulative_system_cost_eur": fmt_currency(cumulative, decimal=decimal),
        })

    _write_dicts(path, rows, delimiter=delimiter)
    logger.info("Wrote baseline CSV (%d rows): %s", len(rows), path)


# ---------------------------------------------------------------------------
# System value CSV
# ---------------------------------------------------------------------------


def write_system_value_csv(
    path: Path | str,
    system_value_result: SystemValueResult,
    delimiter: str = CSV_DELIMITER,
    decimal: str = CSV_DECIMAL_SEPARATOR,
) -> None:
    """Write system value enumeration results.

    One row per (flex_instance, addition_rate, e_to_p_ratio) point.

    Parameters
    ----------
    path:
        Destination file path.
    system_value_result:
        Complete enumeration result with World A costs and all points.
    delimiter:
        CSV column delimiter.
    decimal:
        Decimal separator for numeric values.
    """
    rows: list[dict[str, str]] = []

    for pt in system_value_result.points:
        rows.append({
            "flex_name": pt.flex_name,
            "flex_type": pt.flex_type,
            "annual_addition_kw": fmt_float(pt.annual_addition_kw, decimal=decimal),
            "e_to_p_ratio_h": fmt_float(
                pt.e_to_p_ratio, decimal=decimal
            ) if pt.e_to_p_ratio is not None else "",
            "cumulative_system_value_eur": fmt_currency(
                pt.cumulative_system_value_eur, decimal=decimal
            ),
            "marginal_value_eur_per_kw_a": fmt_float(
                pt.marginal_value_eur_per_kw_a, decimal=decimal
            ),
        })

    _write_dicts(path, rows, delimiter=delimiter)
    logger.info("Wrote system value CSV (%d rows): %s", len(rows), path)


# ---------------------------------------------------------------------------
# Marginal value CSV
# ---------------------------------------------------------------------------


def write_marginal_value_csv(
    path: Path | str,
    marginal_values: list[MarginalValuePoint],
    delimiter: str = CSV_DELIMITER,
    decimal: str = CSV_DECIMAL_SEPARATOR,
) -> None:
    """Write marginal value curve data.

    One row per step in the addition-rate curve, per flex instance.

    Parameters
    ----------
    path:
        Destination file path.
    marginal_values:
        Marginal value points from ``compute_marginal_values()``.
    delimiter:
        CSV column delimiter.
    decimal:
        Decimal separator for numeric values.
    """
    rows: list[dict[str, str]] = []

    for mv in marginal_values:
        rows.append({
            "flex_name": mv.flex_name,
            "flex_type": mv.flex_type,
            "annual_addition_kw": fmt_float(mv.annual_addition_kw, decimal=decimal),
            "e_to_p_ratio_h": fmt_float(
                mv.e_to_p_ratio, decimal=decimal
            ) if mv.e_to_p_ratio is not None else "",
            "cumulative_system_value_eur": fmt_currency(
                mv.cumulative_system_value_eur, decimal=decimal
            ),
            "marginal_value_eur_per_kw_a": fmt_float(
                mv.marginal_value_eur_per_kw_a, decimal=decimal
            ),
            "delta_kw": fmt_float(mv.delta_kw, decimal=decimal),
            "delta_value_eur": fmt_currency(mv.delta_value_eur, decimal=decimal),
        })

    _write_dicts(path, rows, delimiter=delimiter)
    logger.info("Wrote marginal value CSV (%d rows): %s", len(rows), path)


# ---------------------------------------------------------------------------
# Dispatch sample CSV
# ---------------------------------------------------------------------------


def write_portfolio_dispatch_sample_csv(
    path: Path | str,
    daily_results: list[PortfolioDailyResult],
    pv_profile: np.ndarray,
    load_profile: np.ndarray,
    spot_prices: np.ndarray,
    year: int = 2027,
    intervals_per_day: int = INTERVALS_PER_DAY,
    delimiter: str = CSV_DELIMITER,
    decimal: str = CSV_DECIMAL_SEPARATOR,
) -> None:
    """Write quarter-hourly dispatch sample for one simulation year.

    Produces 96 × 365 = 35,040 rows (or 24 × 365 for hourly resolution).

    Parameters
    ----------
    path:
        Destination file path.
    daily_results:
        List of 365 daily LP results from the simulation year.
    pv_profile:
        PV production for the year (kWh per interval).
    load_profile:
        Load demand for the year (kWh per interval).
    spot_prices:
        Spot prices for the year (EUR/kWh per interval).
    year:
        Calendar year for the timestamp column.
    intervals_per_day:
        Number of intervals per day (96 or 24).
    delimiter:
        CSV column delimiter.
    decimal:
        Decimal separator for numeric values.
    """
    n_days = len(daily_results)
    n_intervals = n_days * intervals_per_day

    # Determine timestamp frequency
    if intervals_per_day == 96:
        freq = "15min"
    else:
        freq = "h"

    timestamps = pd.date_range(
        start=f"{year}-01-01 00:00:00",
        periods=n_intervals,
        freq=freq,
    )

    rows: list[dict[str, str]] = []
    idx = 0

    for d, dr in enumerate(daily_results):
        T = intervals_per_day
        for t in range(T):
            global_t = d * T + t

            # Base fields
            row: dict[str, str] = {
                CSV_TIMESTAMP_COLUMN: timestamps[idx].strftime(CSV_TIMESTAMP_FORMAT),
                "pv_production_kwh": fmt_float(
                    float(pv_profile[global_t]) if global_t < len(pv_profile) else 0.0,
                    decimal=decimal,
                ),
                "load_demand_kwh": fmt_float(
                    float(load_profile[global_t]) if global_t < len(load_profile) else 0.0,
                    decimal=decimal,
                ),
                "spot_price_eur_per_kwh": fmt_float(
                    float(spot_prices[global_t]) if global_t < len(spot_prices) else 0.0,
                    precision=6,
                    decimal=decimal,
                ),
                "grid_sell_kwh": fmt_float(float(dr.grid_sell[t]), decimal=decimal),
                "grid_buy_kwh": fmt_float(float(dr.grid_buy[t]), decimal=decimal),
            }

            # BESS fields
            row["bess_charge_kwh"] = fmt_float(
                float(dr.bess_charge[t]), decimal=decimal
            )
            row["bess_discharge_kwh"] = fmt_float(
                float(dr.bess_discharge[t]), decimal=decimal
            )
            row["bess_soc_kwh"] = fmt_float(
                float(dr.bess_soc[t]), decimal=decimal
            )

            # WP fields
            if dr.wp_load is not None:
                row["wp_load_kwh"] = fmt_float(
                    float(dr.wp_load[t]), decimal=decimal
                )
            if dr.thermal_soc is not None:
                row["thermal_soc_kwh"] = fmt_float(
                    float(dr.thermal_soc[t]), decimal=decimal
                )

            # EV fields
            if dr.ev_charge is not None:
                row["ev_charge_kwh"] = fmt_float(
                    float(dr.ev_charge[t]), decimal=decimal
                )
            if dr.ev_discharge is not None:
                row["ev_discharge_kwh"] = fmt_float(
                    float(dr.ev_discharge[t]), decimal=decimal
                )
            if dr.ev_soc is not None:
                row["ev_soc_kwh"] = fmt_float(
                    float(dr.ev_soc[t]), decimal=decimal
                )

            rows.append(row)
            idx += 1

    _write_dicts(path, rows, delimiter=delimiter)
    logger.info("Wrote dispatch sample CSV (%d rows): %s", len(rows), path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict],
    delimiter: str,
) -> None:
    """Write rows to path as a CSV file."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def _write_dicts(
    path: Path | str,
    rows: list[dict],
    delimiter: str = CSV_DELIMITER,
) -> None:
    """Write a list of dicts to a CSV file, creating parent directories.

    Handles file-locking retries (e.g. when Excel has the file open).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    try:
        _write_csv(path, fieldnames, rows, delimiter)
    except PermissionError:
        for idx in range(1, _MAX_LOCK_RETRIES + 1):
            alt_path = path.with_stem(f"{path.stem}_{idx}")
            try:
                _write_csv(alt_path, fieldnames, rows, delimiter)
                logger.warning(
                    "File '%s' is locked; saved as '%s' instead.", path, alt_path
                )
                return
            except PermissionError:
                continue
        raise
