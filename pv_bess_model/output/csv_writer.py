"""Write scenario results to CSV files.

Five output files are produced per scenario run:

1. ``{name}_summary.csv``        – Single row: key inputs + financial results.
2. ``{name}_cashflows.csv``       – One row per project year.
3. ``{name}_grid_search.csv``     – One row per (scale, E/P ratio) combination.
4. ``{name}_monte_carlo.csv``     – One row per MC iteration (optional).
5. ``{name}_dispatch_sample.csv`` – 8 760 hourly rows for year 1 (optional).

All monetary values are in EUR, energy in kWh (or MWh where noted), prices in
€/kWh.  None values are written as empty strings.

Public API
----------
write_summary_csv        – Write the single-row summary file.
write_cashflows_csv      – Write the per-year cashflow table.
write_grid_search_csv    – Write the grid search results matrix.
write_monte_carlo_csv    – Write per-iteration MC results.
write_dispatch_sample_csv – Write hourly dispatch data for year 1.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import pandas as pd

from pv_bess_model.config.defaults import (
    CSV_DELIMITER,
    CSV_TIMESTAMP_FORMAT,
    HOURS_PER_YEAR,
    KWH_TO_MWH,
)
from pv_bess_model.dispatch.engine import HourlySample
from pv_bess_model.finance.cashflow import CashflowProjection
from pv_bess_model.optimization.grid_search import GridSearchResult
from pv_bess_model.optimization.monte_carlo import MCResult
from pv_bess_model.output.formatting import fmt_currency, fmt_float, fmt_optional, fmt_pct

logger = logging.getLogger(__name__)


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
    """
    opt = grid_result.optimal
    total_capex = opt.capex_total if opt else 0.0
    total_revenue = sum(y.revenue for y in cashflow.years if y.year > 0)
    total_opex = sum(y.opex for y in cashflow.years if y.year > 0)

    row = {
        "scenario_name": scenario_name,
        "pv_peak_kwp": fmt_float(pv_peak_kwp),
        "optimal_bess_scale_pct": fmt_float(opt.scale_pct if opt else None),
        "optimal_e_to_p_ratio_h": fmt_float(opt.e_to_p_ratio if opt else None),
        "optimal_bess_power_kw": fmt_float(opt.bess_power_kw if opt else None),
        "optimal_bess_capacity_kwh": fmt_float(opt.bess_capacity_kwh if opt else None),
        "operating_mode": operating_mode,
        "marketing_type": marketing_type,
        "lifetime_years": str(lifetime_years),
        "total_capex_eur": fmt_currency(total_capex),
        "total_revenue_eur": fmt_currency(total_revenue),
        "total_opex_eur": fmt_currency(total_opex),
        "total_production_mwh": fmt_float(total_production_kwh * KWH_TO_MWH),
        "equity_irr_pct": fmt_pct(equity_irr),
        "project_irr_pct": fmt_pct(project_irr),
        "npv_eur": fmt_currency(npv),
        "dscr_min": fmt_float(dscr_min),
        "dscr_avg": fmt_float(dscr_avg),
        "lcoe_eur_per_kwh": fmt_optional(lcoe, precision=6),
        "payback_year": str(payback_year) if payback_year is not None else "",
    }

    _write_dicts(path, [row])
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
) -> None:
    """Write the per-year cashflow table.

    Parameters
    ----------
    path:
        Destination file path.
    cashflow:
        Complete cashflow projection (includes year 0 CAPEX row).
    annual_pv_production_kwh:
        PV production per year in kWh (length = lifetime_years, index 0 = year 1).
    annual_bess_throughput_kwh:
        BESS total throughput per year in kWh (same indexing).
    annual_dscr:
        Per-year DSCR values (same indexing, None outside loan tenor).
    """
    rows = []

    # Year 0 – CAPEX row
    y0 = cashflow.years[0]
    rows.append({
        "year": "0",
        "pv_production_mwh": "",
        "bess_throughput_mwh": "",
        "revenue_eur": fmt_currency(y0.revenue),
        "opex_eur": fmt_currency(y0.opex),
        "debt_service_eur": fmt_currency(y0.debt_service),
        "depreciation_eur": fmt_currency(y0.depreciation),
        "gewerbesteuer_eur": fmt_currency(y0.gewerbesteuer),
        "koerperschaftsteuer_eur": fmt_currency(y0.koerperschaftsteuer),
        "solidaritaetszuschlag_eur": fmt_currency(y0.solidaritaetszuschlag),
        "total_tax_eur": fmt_currency(y0.total_tax),
        "project_cf_eur": fmt_currency(y0.project_cf),
        "equity_cf_eur": fmt_currency(y0.equity_cf),
        "cumulative_equity_cf_eur": fmt_currency(y0.equity_cf),
        "dscr": "",
    })

    cumulative = float(y0.equity_cf)
    n_operating = len(cashflow.years) - 1  # years 1..N

    for i in range(n_operating):
        y = cashflow.years[i + 1]
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

        rows.append({
            "year": str(y.year),
            "pv_production_mwh": fmt_float(pv_mwh),
            "bess_throughput_mwh": fmt_float(bess_mwh),
            "revenue_eur": fmt_currency(y.revenue),
            "opex_eur": fmt_currency(y.opex),
            "debt_service_eur": fmt_currency(y.debt_service),
            "depreciation_eur": fmt_currency(y.depreciation),
            "gewerbesteuer_eur": fmt_currency(y.gewerbesteuer),
            "koerperschaftsteuer_eur": fmt_currency(y.koerperschaftsteuer),
            "solidaritaetszuschlag_eur": fmt_currency(y.solidaritaetszuschlag),
            "total_tax_eur": fmt_currency(y.total_tax),
            "project_cf_eur": fmt_currency(y.project_cf),
            "equity_cf_eur": fmt_currency(y.equity_cf),
            "cumulative_equity_cf_eur": fmt_currency(cumulative),
            "dscr": fmt_float(dscr_val),
        })

    _write_dicts(path, rows)
    logger.info("Wrote cashflows CSV (%d rows): %s", len(rows), path)


# ---------------------------------------------------------------------------
# Grid search CSV
# ---------------------------------------------------------------------------


def write_grid_search_csv(
    path: Path | str,
    grid_result: GridSearchResult,
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
    """
    rows = []
    for pt in grid_result.points:
        rows.append({
            "scale_pct_of_pv": fmt_float(pt.scale_pct),
            "e_to_p_ratio_h": fmt_float(pt.e_to_p_ratio),
            "bess_power_kw": fmt_float(pt.bess_power_kw),
            "bess_capacity_kwh": fmt_float(pt.bess_capacity_kwh),
            "capex_total_eur": fmt_currency(pt.capex_total),
            "capex_pv_eur": fmt_currency(pt.capex_pv),
            "capex_bess_eur": fmt_currency(pt.capex_bess),
            "opex_base_eur": fmt_currency(pt.opex_base),
            "revenue_year1_eur": fmt_currency(pt.revenue_year1),
            "equity_irr_pct": fmt_pct(pt.equity_irr),
            "project_irr_pct": fmt_pct(pt.project_irr),
            "npv_eur": fmt_currency(pt.npv),
            "dscr_min": fmt_float(pt.dscr_min),
            "dscr_avg": fmt_float(pt.dscr_avg),
            "is_optimal": str(pt.is_optimal),
        })

    _write_dicts(path, rows)
    logger.info("Wrote grid search CSV (%d rows): %s", len(rows), path)


# ---------------------------------------------------------------------------
# Monte Carlo CSV
# ---------------------------------------------------------------------------


def write_monte_carlo_csv(
    path: Path | str,
    mc_result: MCResult,
) -> None:
    """Write per-iteration Monte Carlo results.

    One row per iteration, sorted by iteration index.

    Parameters
    ----------
    path:
        Destination file path.
    mc_result:
        Complete Monte Carlo simulation result.
    """
    rows = []
    for it in mc_result.iterations:
        rows.append({
            "iteration": str(it.iteration),
            "price_scenario": it.price_scenario,
            "pv_yield_factor": fmt_float(it.pv_yield_factor),
            "capex_factor": fmt_float(it.capex_factor),
            "opex_factor": fmt_float(it.opex_factor),
            "bess_availability_factor": fmt_float(it.bess_availability_factor),
            "equity_irr_pct": fmt_pct(it.equity_irr),
            "project_irr_pct": fmt_pct(it.project_irr),
            "npv_eur": fmt_currency(it.npv),
            "dscr_min": fmt_float(it.dscr_min),
        })

    _write_dicts(path, rows)
    logger.info("Wrote Monte Carlo CSV (%d rows): %s", len(rows), path)


# ---------------------------------------------------------------------------
# Dispatch sample CSV
# ---------------------------------------------------------------------------


def write_dispatch_sample_csv(
    path: Path | str,
    hourly_sample: HourlySample,
    start_year: int = 2025,
) -> None:
    """Write the 8 760-row hourly dispatch sample for year 1.

    Parameters
    ----------
    path:
        Destination file path.
    hourly_sample:
        Hourly dispatch arrays from the simulation (year 1).
    start_year:
        Calendar year for the timestamp column (default 2025).
    """
    timestamps = pd.date_range(
        start=f"{start_year}-01-01 00:00:00",
        periods=HOURS_PER_YEAR,
        freq="h",
    )

    rows = []
    for h in range(HOURS_PER_YEAR):
        rows.append({
            "timestamp": timestamps[h].strftime(CSV_TIMESTAMP_FORMAT),
            "pv_production_kwh": fmt_float(float(hourly_sample.pv_production[h])),
            "price_spot_eur_per_kwh": fmt_float(
                float(hourly_sample.spot_prices[h]), precision=6
            ),
            "price_effective_eur_per_kwh": fmt_float(
                float(hourly_sample.effective_prices[h]), precision=6
            ),
            "bess_soc_kwh": fmt_float(float(hourly_sample.soc[h])),
            "bess_soc_green_kwh": fmt_float(float(hourly_sample.soc_green[h])),
            "bess_soc_grey_kwh": fmt_float(float(hourly_sample.soc_grey[h])),
            "bess_charge_pv_kwh": fmt_float(float(hourly_sample.charge_pv[h])),
            "bess_charge_grid_kwh": fmt_float(float(hourly_sample.charge_grid[h])),
            "bess_discharge_green_kwh": fmt_float(
                float(hourly_sample.discharge_green[h])
            ),
            "bess_discharge_grey_kwh": fmt_float(
                float(hourly_sample.discharge_grey[h])
            ),
            "grid_export_kwh": fmt_float(float(hourly_sample.export_pv[h])),
            "curtailed_kwh": fmt_float(float(hourly_sample.curtail[h])),
            "revenue_eur": fmt_float(float(hourly_sample.revenue[h])),
        })

    _write_dicts(path, rows)
    logger.info("Wrote dispatch sample CSV (%d rows): %s", HOURS_PER_YEAR, path)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _write_dicts(path: Path | str, rows: list[dict]) -> None:
    """Write a list of dicts to a CSV file, creating parent directories.

    Parameters
    ----------
    path:
        Destination file path.
    rows:
        List of row dicts.  All dicts must have the same keys; the first
        dict determines the column order.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter=CSV_DELIMITER)
        writer.writeheader()
        writer.writerows(rows)
