"""Write scenario results to CSV files.

Five output files are produced per scenario run:

1. ``{name}_summary.csv``        – Single row: key inputs + financial results.
2. ``{name}_cashflows.csv``       – One row per project year.
3. ``{name}_grid_search.csv``     – One row per (scale, E/P ratio) combination.
4. ``{name}_monte_carlo.csv``     – One row per MC iteration (optional).
5. ``{name}_dispatch_sample.csv`` – Dispatch rows for year 1 (8 760 hourly or 35 040 quarter-hourly, optional).

All monetary values are in EUR, energy in kWh (or MWh where noted), prices in
€/kWh.  None values are written as empty strings.

Public API
----------
CsvConfig                – Dataclass holding CSV formatting settings.
write_summary_csv        – Write the single-row summary file.
write_cashflows_csv      – Write the per-year cashflow table.
write_grid_search_csv    – Write the grid search results matrix.
write_monte_carlo_csv    – Write per-iteration MC results.
write_dispatch_sample_csv – Write hourly dispatch data for year 1.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from pv_bess_model.config.defaults import (
    CSV_DECIMAL_SEPARATOR,
    CSV_DELIMITER,
    CSV_TIMESTAMP_COLUMN,
    CSV_TIMESTAMP_FORMAT,
    HOURS_PER_YEAR,
    KWH_TO_MWH,
    _MAX_LOCK_RETRIES
)
from pv_bess_model.dispatch.engine import HourlySample
from pv_bess_model.finance.cashflow import CashflowProjection
from pv_bess_model.optimization.grid_search import GridSearchResult
from pv_bess_model.optimization.monte_carlo import MCResult
from pv_bess_model.output.formatting import fmt_currency, fmt_float, fmt_optional, fmt_pct

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class CsvConfig:
    """CSV output formatting settings, configurable via scenario JSON output block.

    All fields default to the project-wide constants defined in
    ``config/defaults.py`` so that callers need not specify anything unless
    they want to override the defaults.
    """

    delimiter: str = field(default_factory=lambda: CSV_DELIMITER)
    """Column delimiter written between fields (default: ``";"``).  Must match
    ``CSV_DELIMITER`` unless the user overrides it via the JSON ``csv_separator``
    key."""

    decimal: str = field(default_factory=lambda: CSV_DECIMAL_SEPARATOR)
    """Decimal separator used inside numeric strings (default: ``","``).  Set to
    ``"."`` for English-locale output."""

    timestamp_column: str = field(default_factory=lambda: CSV_TIMESTAMP_COLUMN)
    """Column header for the timestamp field in the dispatch sample CSV."""

    timestamp_format: str = field(default_factory=lambda: CSV_TIMESTAMP_FORMAT)
    """``strftime`` format string for timestamps in the dispatch sample CSV."""


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------


def write_summary_csv(
    path: Path | str,
    scenario_name: str,
    pv_peak_kwp: float,
    operating_mode: str,
    marketing_type: str,
    lifetime_years: int,
    grid_result: GridSearchResult,
    cashflow: CashflowProjection,
    equity_irr: float | None,
    project_irr: float | None,
    npv: float,
    dscr_min: float | None,
    dscr_avg: float | None,
    lcoe: float | None,
    payback_year: int | None,
    total_production_kwh: float,
    config: CsvConfig | None = None,
) -> None:
    """Write the single-row scenario summary CSV.

    Parameters
    ----------
    path:
        Destination file path.
    scenario_name:
        Human-readable scenario name.
    pv_peak_kwp:
        PV installed peak power in kWp.
    operating_mode:
        ``"green"`` or ``"grey"``.
    marketing_type:
        Marketing model identifier (e.g. ``"eeg"``, ``"ppa_floor"``).
    lifetime_years:
        Project lifetime in years.
    grid_result:
        Complete grid search result (used for optimal BESS sizing + CAPEX).
    cashflow:
        Cashflow projection for the optimal configuration.
    equity_irr:
        Post-leverage Equity IRR (or None).
    project_irr:
        Pre-leverage Project IRR (or None).
    npv:
        NPV at the configured discount rate in €.
    dscr_min:
        Minimum DSCR over loan tenor (or None).
    dscr_avg:
        Average DSCR over loan tenor (or None).
    lcoe:
        Levelized cost of energy in €/kWh (or None).
    payback_year:
        First year when cumulative equity CF turns positive (or None).
    total_production_kwh:
        Total PV production over the project lifetime in kWh.
    config:
        CSV formatting settings. Uses defaults when None.
    """
    cfg = config or CsvConfig()
    d = cfg.decimal

    opt = grid_result.optimal
    total_capex = opt.capex_total if opt else 0.0
    total_revenue = sum(y.revenue for y in cashflow.years)
    total_opex = sum(y.opex for y in cashflow.years)

    row = {
        "scenario_name": scenario_name,
        "pv_peak_kwp": fmt_float(pv_peak_kwp, decimal=d),
        "optimal_bess_scale_pct": fmt_float(opt.scale_pct if opt else None, decimal=d),
        "optimal_e_to_p_ratio_h": fmt_float(opt.e_to_p_ratio if opt else None, decimal=d),
        "optimal_bess_power_kw": fmt_float(opt.bess_power_kw if opt else None, decimal=d),
        "optimal_bess_capacity_kwh": fmt_float(opt.bess_capacity_kwh if opt else None, decimal=d),
        "operating_mode": operating_mode,
        "marketing_type": marketing_type,
        "lifetime_years": str(lifetime_years),
        "total_capex_eur": fmt_currency(total_capex, decimal=d),
        "total_revenue_eur": fmt_currency(total_revenue, decimal=d),
        "total_opex_eur": fmt_currency(total_opex, decimal=d),
        "total_production_mwh": fmt_float(total_production_kwh * KWH_TO_MWH, decimal=d),
        "equity_irr_pct": fmt_pct(equity_irr, decimal=d),
        "project_irr_pct": fmt_pct(project_irr, decimal=d),
        "npv_eur": fmt_currency(npv, decimal=d),
        "dscr_min": fmt_float(dscr_min, decimal=d),
        "dscr_avg": fmt_float(dscr_avg, decimal=d),
        "lcoe_eur_per_kwh": fmt_optional(lcoe, precision=6, decimal=d),
        "payback_year": str(payback_year) if payback_year is not None else "",
    }

    _write_dicts(path, [row], delimiter=cfg.delimiter)
    logger.info("Wrote summary CSV: %s", path)


# ---------------------------------------------------------------------------
# Cashflows CSV
# ---------------------------------------------------------------------------


def write_cashflows_csv(
    path: Path | str,
    cashflow: CashflowProjection,
    annual_pv_production_kwh: list[float],
    annual_bess_throughput_kwh: list[float],
    annual_dscr: list[float | None],
    commissioning_year: int | None = None,
    config: CsvConfig | None = None,
) -> None:
    """Write the per-year cashflow table.

    Parameters
    ----------
    path:
        Destination file path.
    cashflow:
        Complete cashflow projection (Year 1 through lifetime, no Year 0).
    annual_pv_production_kwh:
        PV production per year in kWh (length = lifetime_years, index 0 = year 1).
    annual_bess_throughput_kwh:
        BESS total throughput per year in kWh (same indexing).
    annual_dscr:
        Per-year DSCR values (same indexing, None outside loan tenor).
    commissioning_year:
        If provided, the ``year`` column shows calendar years
        (commissioning_year, commissioning_year+1, …) instead of project
        year indices (1, 2, …).
    config:
        CSV formatting settings. Uses defaults when None.
    """
    cfg = config or CsvConfig()
    d = cfg.decimal

    rows = []
    cumulative = 0.0

    for i, y in enumerate(cashflow.years):
        cumulative += y.equity_cf
        pv_mwh = (
            annual_pv_production_kwh[i] * KWH_TO_MWH
            if i < len(annual_pv_production_kwh)
            else None
        )
        bess_mwh = (
            annual_bess_throughput_kwh[i] * KWH_TO_MWH
            if i < len(annual_bess_throughput_kwh)
            else None
        )
        dscr_val = annual_dscr[i] if i < len(annual_dscr) else None

        if commissioning_year is not None:
            year_label = str(commissioning_year + y.year - 1)
        else:
            year_label = str(y.year)

        rows.append({
            "year": year_label,
            "capex_eur": fmt_currency(y.capex, decimal=d),
            "pv_production_mwh": fmt_float(pv_mwh, decimal=d),
            "bess_throughput_mwh": fmt_float(bess_mwh, decimal=d),
            "revenue_eur": fmt_currency(y.revenue, decimal=d),
            "opex_eur": fmt_currency(y.opex, decimal=d),
            "debt_interest_eur": fmt_currency(y.debt_interest, decimal=d),
            "debt_repayment_eur": fmt_currency(y.debt_repayment, decimal=d),
            "depreciation_eur": fmt_currency(y.depreciation, decimal=d),
            "gewerbesteuer_eur": fmt_currency(y.gewerbesteuer, decimal=d),
            "koerperschaftsteuer_eur": fmt_currency(y.koerperschaftsteuer, decimal=d),
            "solidaritaetszuschlag_eur": fmt_currency(y.solidaritaetszuschlag, decimal=d),
            "total_tax_eur": fmt_currency(y.total_tax, decimal=d),
            "project_cf_eur": fmt_currency(y.project_cf, decimal=d),
            "equity_cf_eur": fmt_currency(y.equity_cf, decimal=d),
            "cumulative_equity_cf_eur": fmt_currency(cumulative, decimal=d),
            "dscr": fmt_float(dscr_val, decimal=d),
        })

    _write_dicts(path, rows, delimiter=cfg.delimiter)
    logger.info("Wrote cashflows CSV (%d rows): %s", len(rows), path)


# ---------------------------------------------------------------------------
# Grid search CSV
# ---------------------------------------------------------------------------


def write_grid_search_csv(
    path: Path | str,
    grid_result: GridSearchResult,
    config: CsvConfig | None = None,
) -> None:
    """Write the grid search results matrix.

    One row per (scale, E/P ratio) combination, sorted by (scale_pct,
    e_to_p_ratio).

    Parameters
    ----------
    path:
        Destination file path.
    grid_result:
        Complete grid search result.
    config:
        CSV formatting settings. Uses defaults when None.
    """
    cfg = config or CsvConfig()
    d = cfg.decimal

    rows = []
    for pt in grid_result.points:
        rows.append({
            "scale_pct_of_pv": fmt_float(pt.scale_pct, decimal=d),
            "e_to_p_ratio_h": fmt_float(pt.e_to_p_ratio, decimal=d),
            "bess_power_kw": fmt_float(pt.bess_power_kw, decimal=d),
            "bess_capacity_kwh": fmt_float(pt.bess_capacity_kwh, decimal=d),
            "capex_total_eur": fmt_currency(pt.capex_total, decimal=d),
            "capex_pv_eur": fmt_currency(pt.capex_pv, decimal=d),
            "capex_bess_eur": fmt_currency(pt.capex_bess, decimal=d),
            "opex_base_eur": fmt_currency(pt.opex_base, decimal=d),
            "revenue_year1_eur": fmt_currency(pt.revenue_year1, decimal=d),
            "equity_irr_pct": fmt_pct(pt.equity_irr, decimal=d),
            "project_irr_pct": fmt_pct(pt.project_irr, decimal=d),
            "npv_eur": fmt_currency(pt.npv, decimal=d),
            "dscr_min": fmt_float(pt.dscr_min, decimal=d),
            "dscr_avg": fmt_float(pt.dscr_avg, decimal=d),
            "is_optimal": str(pt.is_optimal),
        })

    _write_dicts(path, rows, delimiter=cfg.delimiter)
    logger.info("Wrote grid search CSV (%d rows): %s", len(rows), path)


# ---------------------------------------------------------------------------
# Monte Carlo CSV
# ---------------------------------------------------------------------------


def write_monte_carlo_csv(
    path: Path | str,
    mc_result: MCResult,
    config: CsvConfig | None = None,
) -> None:
    """Write per-iteration Monte Carlo results.

    One row per iteration, sorted by iteration index.

    Parameters
    ----------
    path:
        Destination file path.
    mc_result:
        Complete Monte Carlo simulation result.
    config:
        CSV formatting settings. Uses defaults when None.
    """
    cfg = config or CsvConfig()
    d = cfg.decimal

    rows = []
    for it in mc_result.iterations:
        rows.append({
            "iteration": str(it.iteration),
            "price_scenario": it.price_scenario,
            "capex_factor_pv": fmt_float(it.capex_factor_pv, decimal=d),
            "capex_factor_bess": fmt_float(it.capex_factor_bess, decimal=d),
            "opex_factor_pv": fmt_float(it.opex_factor_pv, decimal=d),
            "opex_factor_bess": fmt_float(it.opex_factor_bess, decimal=d),
            "pv_availability_factor": fmt_float(it.pv_availability_factor, decimal=d),
            "bess_availability_factor": fmt_float(it.bess_availability_factor, decimal=d),
            "equity_irr_pct": fmt_pct(it.equity_irr, decimal=d),
            "project_irr_pct": fmt_pct(it.project_irr, decimal=d),
            "npv_eur": fmt_currency(it.npv, decimal=d),
            "dscr_min": fmt_float(it.dscr_min, decimal=d),
        })

    _write_dicts(path, rows, delimiter=cfg.delimiter)
    logger.info("Wrote Monte Carlo CSV (%d rows): %s", len(rows), path)


# ---------------------------------------------------------------------------
# Dispatch sample CSV
# ---------------------------------------------------------------------------


def write_dispatch_sample_csv(
    path: Path | str,
    hourly_sample: HourlySample,
    start_year: int = 2025,
    config: CsvConfig | None = None,
) -> None:
    """Write the dispatch sample for year 1.

    The number of rows is determined dynamically from the sample array length
    (8 760 for hourly or 35 040 for 15-min resolution).  The timestamp
    frequency is inferred accordingly.

    Parameters
    ----------
    path:
        Destination file path.
    hourly_sample:
        Dispatch arrays from the simulation (year 1).
    start_year:
        Calendar year for the timestamp column (default 2025).
    config:
        CSV formatting settings. Uses defaults when None.
    """
    cfg = config or CsvConfig()
    d = cfg.decimal

    n_intervals = len(hourly_sample.pv_production)

    # Determine timestamp frequency from array length
    if n_intervals == HOURS_PER_YEAR:
        freq = "h"
    else:
        freq = "15min"

    timestamps = pd.date_range(
        start=f"{start_year}-01-01 00:00:00",
        periods=n_intervals,
        freq=freq,
    )

    rows = []
    for h in range(n_intervals):
        rows.append({
            cfg.timestamp_column: timestamps[h].strftime(cfg.timestamp_format),
            "pv_production_kwh": fmt_float(float(hourly_sample.pv_production[h]), decimal=d),
            "price_spot_eur_per_kwh": fmt_float(
                float(hourly_sample.spot_prices[h]), precision=6, decimal=d
            ),
            "price_effective_eur_per_kwh": fmt_float(
                float(hourly_sample.effective_prices[h]), precision=6, decimal=d
            ),
            "bess_soc_kwh": fmt_float(float(hourly_sample.soc[h]), decimal=d),
            "bess_soc_green_kwh": fmt_float(float(hourly_sample.soc_green[h]), decimal=d),
            "bess_soc_grey_kwh": fmt_float(float(hourly_sample.soc_grey[h]), decimal=d),
            "bess_charge_pv_kwh": fmt_float(float(hourly_sample.charge_pv[h]), decimal=d),
            "bess_charge_grid_kwh": fmt_float(float(hourly_sample.charge_grid[h]), decimal=d),
            "bess_discharge_green_kwh": fmt_float(
                float(hourly_sample.discharge_green[h]), decimal=d
            ),
            "bess_discharge_grey_kwh": fmt_float(
                float(hourly_sample.discharge_grey[h]), decimal=d
            ),
            "pv_grid_export_kwh": fmt_float(float(hourly_sample.export_pv[h]), decimal=d),
            "grid_export_kwh": fmt_float(float(
                hourly_sample.export_pv[h] + hourly_sample.discharge_green[h] + hourly_sample.discharge_grey[h]
            ), decimal=d),
            "curtailed_kwh": fmt_float(float(hourly_sample.curtail[h]), decimal=d),
            "revenue_eur": fmt_float(float(hourly_sample.revenue[h]), decimal=d),
        })

    _write_dicts(path, rows, delimiter=cfg.delimiter)
    logger.info("Wrote dispatch sample CSV (%d rows): %s", n_intervals, path)


# ---------------------------------------------------------------------------
# Sensitivity analysis CSVs
# ---------------------------------------------------------------------------


def write_eeg_sensitivity_csv(
    path: Path | str,
    result: "SensitivityResult",
    config: CsvConfig | None = None,
) -> None:
    """Write EEG floor price sensitivity analysis results.

    One row per floor price point with MC summary statistics.

    Parameters
    ----------
    path:
        Destination file path.
    result:
        Complete EEG sensitivity result.
    config:
        CSV formatting settings. Uses defaults when None.
    """
    from pv_bess_model.optimization.analyses import SensitivityResult  # noqa: F811

    cfg = config or CsvConfig()
    d = cfg.decimal

    rows = []
    for pt in result.points:
        stats = pt.mc_result.overall_stats
        eq = stats.get("equity_irr")
        proj = stats.get("project_irr")
        npv_s = stats.get("npv")
        dscr_s = stats.get("dscr_min")

        rows.append({
            "floor_price_eur_per_kwh": fmt_float(
                pt.params.get("floor_price_eur_per_kwh"), precision=6, decimal=d
            ),
            "mc_iterations": str(len(pt.mc_result.iterations)),
            "equity_irr_mean": fmt_pct(eq.mean if eq else None, decimal=d),
            "equity_irr_std": fmt_pct(eq.std if eq else None, decimal=d),
            "equity_irr_p10": fmt_pct(eq.p10 if eq else None, decimal=d),
            "equity_irr_p50": fmt_pct(eq.p50 if eq else None, decimal=d),
            "equity_irr_p90": fmt_pct(eq.p90 if eq else None, decimal=d),
            "project_irr_mean": fmt_pct(proj.mean if proj else None, decimal=d),
            "npv_mean": fmt_currency(npv_s.mean if npv_s else 0.0, decimal=d),
            "dscr_min_mean": fmt_float(dscr_s.mean if dscr_s else None, decimal=d),
        })

    _write_dicts(path, rows, delimiter=cfg.delimiter)
    logger.info("Wrote EEG sensitivity CSV (%d rows): %s", len(rows), path)


def write_ppa_collar_csv(
    path: Path | str,
    result: "SensitivityResult",
    duration_years: int,
    config: CsvConfig | None = None,
) -> None:
    """Write PPA Collar sensitivity analysis results.

    One row per (floor, cap_spread) combination with MC summary statistics.

    Parameters
    ----------
    path:
        Destination file path.
    result:
        Complete PPA Collar sensitivity result.
    duration_years:
        PPA duration in years (written to each row for context).
    config:
        CSV formatting settings. Uses defaults when None.
    """
    from pv_bess_model.optimization.analyses import SensitivityResult  # noqa: F811

    cfg = config or CsvConfig()
    d = cfg.decimal

    rows = []
    for pt in result.points:
        stats = pt.mc_result.overall_stats
        eq = stats.get("equity_irr")
        proj = stats.get("project_irr")
        npv_s = stats.get("npv")

        rows.append({
            "floor_price_eur_per_mwh": fmt_float(
                pt.params.get("floor_price_eur_per_mwh"), decimal=d
            ),
            "cap_spread_eur_per_mwh": fmt_float(
                pt.params.get("cap_spread_eur_per_mwh"), decimal=d
            ),
            "cap_price_eur_per_mwh": fmt_float(
                pt.params.get("cap_price_eur_per_mwh"), decimal=d
            ),
            "duration_years": str(duration_years),
            "equity_irr_mean": fmt_pct(eq.mean if eq else None, decimal=d),
            "equity_irr_std": fmt_pct(eq.std if eq else None, decimal=d),
            "equity_irr_p10": fmt_pct(eq.p10 if eq else None, decimal=d),
            "equity_irr_p50": fmt_pct(eq.p50 if eq else None, decimal=d),
            "equity_irr_p90": fmt_pct(eq.p90 if eq else None, decimal=d),
            "project_irr_mean": fmt_pct(proj.mean if proj else None, decimal=d),
            "npv_mean": fmt_currency(npv_s.mean if npv_s else 0.0, decimal=d),
        })

    _write_dicts(path, rows, delimiter=cfg.delimiter)
    logger.info("Wrote PPA Collar CSV (%d rows): %s", len(rows), path)


def write_ppa_baseload_csv(
    path: Path | str,
    result: "SensitivityResult",
    duration_years: int,
    config: CsvConfig | None = None,
) -> None:
    """Write PPA Baseload sensitivity analysis results.

    One row per (ppa_price, baseload_mw) combination with MC summary statistics.

    Parameters
    ----------
    path:
        Destination file path.
    result:
        Complete PPA Baseload sensitivity result.
    duration_years:
        PPA duration in years (written to each row for context).
    config:
        CSV formatting settings. Uses defaults when None.
    """
    from pv_bess_model.optimization.analyses import SensitivityResult  # noqa: F811

    cfg = config or CsvConfig()
    d = cfg.decimal

    rows = []
    for pt in result.points:
        stats = pt.mc_result.overall_stats
        eq = stats.get("equity_irr")
        proj = stats.get("project_irr")
        npv_s = stats.get("npv")

        rows.append({
            "ppa_price_eur_per_mwh": fmt_float(
                pt.params.get("ppa_price_eur_per_mwh"), decimal=d
            ),
            "baseload_mw": fmt_float(
                pt.params.get("baseload_mw"), decimal=d
            ),
            "duration_years": str(duration_years),
            "equity_irr_mean": fmt_pct(eq.mean if eq else None, decimal=d),
            "equity_irr_std": fmt_pct(eq.std if eq else None, decimal=d),
            "equity_irr_p10": fmt_pct(eq.p10 if eq else None, decimal=d),
            "equity_irr_p50": fmt_pct(eq.p50 if eq else None, decimal=d),
            "equity_irr_p90": fmt_pct(eq.p90 if eq else None, decimal=d),
            "project_irr_mean": fmt_pct(proj.mean if proj else None, decimal=d),
            "npv_mean": fmt_currency(npv_s.mean if npv_s else 0.0, decimal=d),
        })

    _write_dicts(path, rows, delimiter=cfg.delimiter)
    logger.info("Wrote PPA Baseload CSV (%d rows): %s", len(rows), path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict],
    delimiter: str,
) -> None:
    """Write *rows* to *path* as a CSV file.

    Parameters
    ----------
    path:
        Destination file path (must already have its parent directory created).
    fieldnames:
        Ordered list of column names.
    rows:
        Row dicts to write.
    delimiter:
        Field delimiter character.

    Raises
    ------
    PermissionError
        When the file cannot be opened for writing (e.g. locked by Excel).
    """
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

    If the target file is locked (e.g. open in Excel), up to
    ``_MAX_LOCK_RETRIES`` alternative filenames are tried by appending an
    incrementing index to the stem (e.g. ``summary_1.csv``,
    ``summary_2.csv``).  A ``PermissionError`` is re-raised only when all
    alternatives are also locked.

    Parameters
    ----------
    path:
        Destination file path.
    rows:
        List of row dicts.  All dicts must have the same keys; the first
        dict determines the column order.
    delimiter:
        Field delimiter character (default: ``CSV_DELIMITER``).
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
