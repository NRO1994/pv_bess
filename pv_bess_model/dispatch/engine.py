"""Hourly dispatch engine: yearly simulation loop over 365 daily LP solves.

The engine runs the full multi-year dispatch simulation for one BESS sizing
configuration.  It iterates over project years, applying PV/BESS degradation,
pricing regimes, and BESS offline days, then dispatches each day via the daily
LP optimizer.

The caller provides:

- Static configuration (BESS specs, grid limits, operating mode, degradation
  rates, replacement config).
- Pre-computed year-varying inputs (spot prices, fixed/floor prices, offline
  days).

The engine handles:

- PV degradation (applying ``(1 - rate)^year`` per year).
- BESS capacity degradation and optional mid-life replacement.
- SoC day-to-day and year-to-year coupling (including clipping after
  degradation-induced capacity changes).
- Revenue breakdown by stream (PV export, BESS green, BESS grey, grid import).

Unit conventions
----------------
Identical to the optimizer module:

========  ======  ====================================================
Quantity  Unit    Notes
========  ======  ====================================================
Energy    kWh     PV production, charge/discharge, SoC, throughput
Power     kW      Charge/discharge limits, grid export limit
Price     EUR/kWh  Spot prices and floor price
Revenue   EUR      Hourly and annual revenue
RTE       frac    Round-trip efficiency in (0, 1]
========  ======  ====================================================

Public API
----------
DispatchEngineConfig              - Static simulation configuration.
AnnualResult                      - Per-year aggregate results.
HourlySample                      - Per-hour dispatch data for one year.
SimulationResult                  - Full simulation output.
run_simulation                    - Execute the multi-year dispatch simulation.
compute_deterministic_offline_days - Evenly-spaced offline days from avail. %.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from pv_bess_model.bess.replacement import ReplacementConfig
from pv_bess_model.config.defaults import (
    DAYS_PER_YEAR,
    DEFAULT_START_SOC_FRACTION,
    DISPATCH_SAMPLE_YEAR,
    HOURS_PER_DAY,
    HOURS_PER_YEAR,
)
from pv_bess_model.dispatch.optimizer import (
    BessParams,
    OperatingMode,
    dispatch_offline_day,
    optimize_day,
)
from pv_bess_model.pv.degradation import degradation_factor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DispatchEngineConfig:
    """Static parameters for the dispatch simulation.

    All BESS parameters refer to nameplate (undegraded) values; the engine
    applies degradation internally.

    Attributes
    ----------
    mode:
        Operating mode: ``"green"`` or ``"grey"``.
    grid_max_kw:
        Maximum grid export power in kW.
    bess_nameplate_kwh:
        Nameplate (undegraded) BESS energy capacity in kWh.  Set to 0.0
        for PV-only scenarios.
    bess_max_charge_kw:
        Maximum BESS charging power in kW.
    bess_max_discharge_kw:
        Maximum BESS discharging power in kW.
    bess_rte:
        Round-trip efficiency as a fraction in (0, 1].
    bess_min_soc_pct:
        Minimum SoC as percentage of current (degraded) capacity.
    bess_max_soc_pct:
        Maximum SoC as percentage of current (degraded) capacity.
    bess_degradation_rate:
        Annual BESS capacity degradation as a fraction (e.g. 0.02 for 2 %).
    pv_degradation_rate:
        Annual PV production degradation as a fraction (e.g. 0.004 for 0.4 %).
    replacement:
        Mid-life BESS replacement configuration.
    lifetime_years:
        Number of project years to simulate.
    bess_power_kw:
        Rated BESS power in kW (used for replacement cost calculation).
    """

    mode: OperatingMode
    grid_max_kw: float
    bess_nameplate_kwh: float
    bess_max_charge_kw: float
    bess_max_discharge_kw: float
    bess_rte: float
    bess_min_soc_pct: float
    bess_max_soc_pct: float
    bess_degradation_rate: float
    pv_degradation_rate: float
    replacement: ReplacementConfig
    lifetime_years: int
    bess_power_kw: float


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class AnnualResult:
    """Aggregated dispatch results for one project year.

    All revenue/cost values in EUR.  All energy values in kWh.
    """

    year: int
    """Project year (1-indexed)."""

    # Revenue breakdown (EUR)
    revenue_pv_export: float
    """Revenue from PV exported directly to grid."""
    revenue_bess_green: float
    """Revenue from green BESS discharge (discharge_green x RTE x eff_price)."""
    revenue_bess_grey: float
    """Revenue from grey BESS discharge (discharge_grey x RTE x spot).  0.0 in green mode."""
    grid_import_cost: float
    """Cost of grid imports for BESS charging (charge_grid x spot).  0.0 in green mode."""
    total_revenue: float
    """Net revenue = pv_export + bess_green + bess_grey - grid_import_cost."""

    # Energy flows (kWh)
    pv_production: float
    """Total degraded PV production this year."""
    pv_export: float
    """PV energy exported directly to grid."""
    pv_curtailed: float
    """PV energy curtailed."""
    bess_charge_pv: float
    """Energy charged into BESS from PV surplus."""
    bess_charge_grid: float
    """Energy charged into BESS from grid.  0.0 in green mode."""
    bess_discharge_green: float
    """Energy removed from BESS green chamber (SoC change, before RTE)."""
    bess_discharge_grey: float
    """Energy removed from BESS grey chamber (SoC change, before RTE).  0.0 in green mode."""
    bess_throughput: float
    """Total kWh through BESS (charge_pv + charge_grid + discharge_green + discharge_grey)."""

    # BESS state
    bess_capacity_kwh: float
    """Effective BESS capacity at start of this year (after degradation/replacement)."""
    replacement_cost: float
    """Replacement cost booked as additional OPEX (EUR).  0.0 if no replacement."""


@dataclass
class HourlySample:
    """Hourly dispatch data for one project year (for ``dispatch_sample.csv``).

    All arrays have shape ``(HOURS_PER_YEAR,)`` = ``(8760,)``.
    Energy in kWh, prices in EUR/kWh, revenue in EUR.
    """

    pv_production: np.ndarray
    spot_prices: np.ndarray
    effective_prices: np.ndarray
    soc: np.ndarray
    soc_green: np.ndarray
    soc_grey: np.ndarray
    charge_pv: np.ndarray
    charge_grid: np.ndarray
    discharge_green: np.ndarray
    discharge_grey: np.ndarray
    export_pv: np.ndarray
    curtail: np.ndarray
    revenue: np.ndarray


@dataclass
class SimulationResult:
    """Complete output of a multi-year dispatch simulation."""

    annual_results: list[AnnualResult]
    """One entry per project year (index 0 = year 1)."""

    hourly_sample: HourlySample
    """Hourly dispatch data for the sample year (typically year 1)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_deterministic_offline_days(
    bess_availability_pct: float,
) -> set[int]:
    """Compute evenly-spaced BESS offline day indices for deterministic dispatch.

    Used by the grid search; Monte Carlo samples random offline days instead.

    Parameters
    ----------
    bess_availability_pct:
        BESS availability as a percentage (0--100).  100 % = always online.

    Returns
    -------
    set[int]
        Set of 0-indexed day indices (0--364) when the BESS is offline.
    """
    n_offline = round((1.0 - bess_availability_pct / 100.0) * DAYS_PER_YEAR)
    if n_offline <= 0:
        return set()
    if n_offline >= DAYS_PER_YEAR:
        return set(range(DAYS_PER_YEAR))

    # Distribute evenly: every Nth day
    step = DAYS_PER_YEAR / n_offline
    return {int(i * step) for i in range(n_offline)}


def _bess_params_for_capacity(
    config: DispatchEngineConfig,
    degraded_capacity_kwh: float,
) -> BessParams:
    """Build a :class:`BessParams` from engine config and a degraded capacity.

    Parameters
    ----------
    config:
        Engine configuration (provides power limits, RTE, SoC percentages).
    degraded_capacity_kwh:
        Current BESS capacity in kWh after degradation/replacement.

    Returns
    -------
    BessParams
    """
    return BessParams(
        max_charge_kw=config.bess_max_charge_kw,
        max_discharge_kw=config.bess_max_discharge_kw,
        round_trip_efficiency=config.bess_rte,
        soc_min_kwh=degraded_capacity_kwh * config.bess_min_soc_pct / 100.0,
        soc_max_kwh=degraded_capacity_kwh * config.bess_max_soc_pct / 100.0,
    )


def _clip_soc_to_limits(
    current_soc: float,
    current_soc_green: float,
    current_soc_grey: float,
    soc_min: float,
    soc_max: float,
) -> tuple[float, float, float]:
    """Clip total SoC to new limits after a capacity change.

    When BESS capacity decreases due to degradation, the SoC may exceed the
    new ``soc_max``.  Excess energy is removed from the grey chamber first
    (less valuable), then from the green chamber if necessary.

    Parameters
    ----------
    current_soc:
        Total SoC (kWh) before clipping.
    current_soc_green:
        Green-chamber SoC (kWh).
    current_soc_grey:
        Grey-chamber SoC (kWh).
    soc_min:
        New minimum SoC (kWh).
    soc_max:
        New maximum SoC (kWh).

    Returns
    -------
    tuple[float, float, float]
        (clipped_soc, clipped_soc_green, clipped_soc_grey)
    """
    new_soc = max(soc_min, min(current_soc, soc_max))
    if new_soc >= current_soc:
        return new_soc, current_soc_green, current_soc_grey

    # Need to reduce — remove from grey first, then green
    reduction = current_soc - new_soc
    grey_reduction = min(reduction, current_soc_grey)
    new_grey = current_soc_grey - grey_reduction
    remaining = reduction - grey_reduction
    new_green = current_soc_green - remaining

    return new_soc, new_green, new_grey


def _empty_hourly_sample() -> HourlySample:
    """Create an all-zeros hourly sample (fallback)."""
    z = np.zeros(HOURS_PER_YEAR)
    return HourlySample(
        pv_production=z.copy(),
        spot_prices=z.copy(),
        effective_prices=z.copy(),
        soc=z.copy(),
        soc_green=z.copy(),
        soc_grey=z.copy(),
        charge_pv=z.copy(),
        charge_grid=z.copy(),
        discharge_green=z.copy(),
        discharge_grey=z.copy(),
        export_pv=z.copy(),
        curtail=z.copy(),
        revenue=z.copy(),
    )


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


def run_simulation(
    config: DispatchEngineConfig,
    pv_base_timeseries: np.ndarray,
    spot_prices_yearly: list[np.ndarray],
    fixed_prices_yearly: list[float],
    offline_days_yearly: list[set[int]],
) -> SimulationResult:
    """Execute the full multi-year dispatch simulation.

    Iterates over all project years, running 365 daily LP solves per year.
    PV and BESS degradation are applied internally.  BESS replacement is
    handled at the configured year.

    Parameters
    ----------
    config:
        Static dispatch configuration.
    pv_base_timeseries:
        Undegraded PV production array, shape ``(HOURS_PER_YEAR,)`` in kWh.
    spot_prices_yearly:
        List of length ``lifetime_years``.  Each element is an 8760-array
        of spot prices in EUR/kWh (already inflation-adjusted if applicable).
    fixed_prices_yearly:
        List of length ``lifetime_years``.  Each element is the floor/fixed
        price in EUR/kWh for that year (0.0 when no floor is active).
    offline_days_yearly:
        List of length ``lifetime_years``.  Each element is a set of
        0-indexed day indices (0--364) when the BESS is offline.

    Returns
    -------
    SimulationResult
        Annual aggregates for all years and hourly dispatch sample.
    """
    annual_results: list[AnnualResult] = []
    hourly_sample: HourlySample | None = None

    has_bess = config.bess_nameplate_kwh > 0.0

    # BESS degradation: track years since start or last replacement
    bess_age = 0

    # SoC state carried between days and years
    current_soc = 0.0
    current_soc_green = 0.0
    current_soc_grey = 0.0
    soc_initialised = False

    for year in range(1, config.lifetime_years + 1):
        year_idx = year - 1

        # ---- 1. PV degradation ----
        pv_factor = degradation_factor(config.pv_degradation_rate, year)
        pv_year = pv_base_timeseries * pv_factor

        # ---- 2. BESS degradation + replacement ----
        replacement_cost = 0.0
        if has_bess:
            if config.replacement.enabled and year == config.replacement.year:
                # Replacement: reset capacity to nameplate, restart degradation
                bess_cap = config.bess_nameplate_kwh
                bess_age = 0
                replacement_cost = config.replacement.replacement_cost(
                    config.bess_power_kw,
                    config.bess_nameplate_kwh,
                )
                logger.info(
                    "BESS replacement at year %d: capacity reset to %.1f kWh, "
                    "cost = %.2f EUR.",
                    year,
                    bess_cap,
                    replacement_cost,
                )
            else:
                bess_age += 1
                bess_cap = config.bess_nameplate_kwh * degradation_factor(
                    config.bess_degradation_rate, bess_age
                )
        else:
            bess_cap = 0.0

        # ---- 3. BessParams for this year ----
        bess_params = _bess_params_for_capacity(config, bess_cap)

        # ---- 4. Initialise or clip SoC to new limits ----
        if not soc_initialised:
            current_soc = bess_params.soc_max_kwh * DEFAULT_START_SOC_FRACTION
            current_soc_green = current_soc
            current_soc_grey = 0.0
            soc_initialised = True
        else:
            current_soc, current_soc_green, current_soc_grey = _clip_soc_to_limits(
                current_soc,
                current_soc_green,
                current_soc_grey,
                bess_params.soc_min_kwh,
                bess_params.soc_max_kwh,
            )

        # ---- 5. Year-level price data ----
        spot_prices = spot_prices_yearly[year_idx]
        fixed_price = fixed_prices_yearly[year_idx]
        offline_days = offline_days_yearly[year_idx]

        # ---- 6. Prepare hourly sample arrays (year 1 only) ----
        is_sample_year = year == DISPATCH_SAMPLE_YEAR
        if is_sample_year:
            h_charge_pv = np.zeros(HOURS_PER_YEAR)
            h_charge_grid = np.zeros(HOURS_PER_YEAR)
            h_discharge_green = np.zeros(HOURS_PER_YEAR)
            h_discharge_grey = np.zeros(HOURS_PER_YEAR)
            h_export_pv = np.zeros(HOURS_PER_YEAR)
            h_curtail = np.zeros(HOURS_PER_YEAR)
            h_soc = np.zeros(HOURS_PER_YEAR)
            h_soc_green = np.zeros(HOURS_PER_YEAR)
            h_soc_grey = np.zeros(HOURS_PER_YEAR)
            h_revenue = np.zeros(HOURS_PER_YEAR)

        # ---- 7. Annual accumulators ----
        year_revenue_pv = 0.0
        year_revenue_green = 0.0
        year_revenue_grey = 0.0
        year_import_cost = 0.0
        year_pv_export = 0.0
        year_pv_curtailed = 0.0
        year_charge_pv = 0.0
        year_charge_grid = 0.0
        year_discharge_green = 0.0
        year_discharge_grey = 0.0

        # ---- 8. Day loop (365 days) ----
        for day in range(DAYS_PER_YEAR):
            h_start = day * HOURS_PER_DAY
            h_end = h_start + HOURS_PER_DAY

            pv_day = pv_year[h_start:h_end]
            spot_day = spot_prices[h_start:h_end]

            is_offline = (day in offline_days) or (not has_bess)

            if is_offline:
                result = dispatch_offline_day(
                    pv_production_kwh=pv_day,
                    spot_prices_eur_per_kwh=spot_day,
                    price_fixed_eur_per_kwh=fixed_price,
                    grid_max_kw=config.grid_max_kw,
                    start_soc_kwh=current_soc,
                    start_soc_green_kwh=current_soc_green,
                    start_soc_grey_kwh=current_soc_grey,
                )
            else:
                result = optimize_day(
                    pv_production_kwh=pv_day,
                    spot_prices_eur_per_kwh=spot_day,
                    price_fixed_eur_per_kwh=fixed_price,
                    bess=bess_params,
                    grid_max_kw=config.grid_max_kw,
                    mode=config.mode,
                    start_soc_kwh=current_soc,
                    start_soc_green_kwh=current_soc_green if config.mode == "grey" else None,
                    start_soc_grey_kwh=current_soc_grey if config.mode == "grey" else None,
                )

            # Update SoC carry-over
            current_soc = result["end_soc"]
            current_soc_green = result["end_soc_green"]
            current_soc_grey = result["end_soc_grey"]

            # Revenue breakdown for this day
            eff_day = (
                np.maximum(spot_day, fixed_price) if fixed_price > 0.0 else spot_day
            )

            day_rev_pv = float(np.sum(result["export_pv"] * eff_day))
            day_rev_green = float(
                np.sum(result["discharge_green"] * config.bess_rte * eff_day)
            )
            day_rev_grey = float(
                np.sum(result["discharge_grey"] * config.bess_rte * spot_day)
            )
            day_import = float(np.sum(result["charge_grid"] * spot_day))

            year_revenue_pv += day_rev_pv
            year_revenue_green += day_rev_green
            year_revenue_grey += day_rev_grey
            year_import_cost += day_import
            year_pv_export += float(np.sum(result["export_pv"]))
            year_pv_curtailed += float(np.sum(result["curtail"]))
            year_charge_pv += float(np.sum(result["charge_pv"]))
            year_charge_grid += float(np.sum(result["charge_grid"]))
            year_discharge_green += float(np.sum(result["discharge_green"]))
            year_discharge_grey += float(np.sum(result["discharge_grey"]))

            # Store hourly sample arrays
            if is_sample_year:
                h_charge_pv[h_start:h_end] = result["charge_pv"]
                h_charge_grid[h_start:h_end] = result["charge_grid"]
                h_discharge_green[h_start:h_end] = result["discharge_green"]
                h_discharge_grey[h_start:h_end] = result["discharge_grey"]
                h_export_pv[h_start:h_end] = result["export_pv"]
                h_curtail[h_start:h_end] = result["curtail"]
                h_soc[h_start:h_end] = result["soc"]
                h_soc_green[h_start:h_end] = result["soc_green"]
                h_soc_grey[h_start:h_end] = result["soc_grey"]
                h_revenue[h_start:h_end] = result["revenue"]

        # ---- 9. Compute annual aggregates ----
        total_pv = float(np.sum(pv_year))
        total_revenue = (
            year_revenue_pv + year_revenue_green + year_revenue_grey - year_import_cost
        )
        bess_throughput = (
            year_charge_pv + year_charge_grid + year_discharge_green + year_discharge_grey
        )

        annual_results.append(
            AnnualResult(
                year=year,
                revenue_pv_export=year_revenue_pv,
                revenue_bess_green=year_revenue_green,
                revenue_bess_grey=year_revenue_grey,
                grid_import_cost=year_import_cost,
                total_revenue=total_revenue,
                pv_production=total_pv,
                pv_export=year_pv_export,
                pv_curtailed=year_pv_curtailed,
                bess_charge_pv=year_charge_pv,
                bess_charge_grid=year_charge_grid,
                bess_discharge_green=year_discharge_green,
                bess_discharge_grey=year_discharge_grey,
                bess_throughput=bess_throughput,
                bess_capacity_kwh=bess_cap,
                replacement_cost=replacement_cost,
            )
        )

        # ---- 10. Build hourly sample ----
        if is_sample_year:
            eff_prices_year = (
                np.maximum(spot_prices, fixed_price)
                if fixed_price > 0.0
                else spot_prices.copy()
            )
            hourly_sample = HourlySample(
                pv_production=pv_year.copy(),
                spot_prices=spot_prices.copy(),
                effective_prices=eff_prices_year,
                soc=h_soc,
                soc_green=h_soc_green,
                soc_grey=h_soc_grey,
                charge_pv=h_charge_pv,
                charge_grid=h_charge_grid,
                discharge_green=h_discharge_green,
                discharge_grey=h_discharge_grey,
                export_pv=h_export_pv,
                curtail=h_curtail,
                revenue=h_revenue,
            )

        logger.debug(
            "Year %d: PV=%.0f kWh, Revenue=%.2f EUR, BESS cap=%.1f kWh",
            year,
            total_pv,
            total_revenue,
            bess_cap,
        )

    # Fallback if sample year was never reached
    if hourly_sample is None:
        hourly_sample = _empty_hourly_sample()

    return SimulationResult(
        annual_results=annual_results,
        hourly_sample=hourly_sample,
    )
