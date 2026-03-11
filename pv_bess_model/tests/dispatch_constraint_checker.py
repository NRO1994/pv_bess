"""Dispatch constraint validation for integration tests.

Validates that dispatch sample CSV output satisfies all physical and
operational constraints defined in the optimizer specification. Used by
the integration test suite to verify correctness across all scenario
combinations.

Public API
----------
ConstraintViolation  – Dataclass describing a single constraint violation.
check_dispatch_constraints – Validate all 9 constraint categories.
check_availability   – Count PV and BESS offline days from dispatch data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ConstraintViolation:
    """A single constraint violation found in dispatch data.

    Attributes
    ----------
    constraint:
        Name of the violated constraint category.
    timestep:
        Zero-based interval index where the violation occurred.
    expected:
        Human-readable description of the expected condition.
    actual:
        The actual value that caused the violation.
    severity:
        Either ``"error"`` or ``"warning"``.
    """

    constraint: str
    timestep: int
    expected: str
    actual: float
    severity: str


def check_dispatch_constraints(
        dispatch_df: pd.DataFrame,
        pv_peak_kwp: float,
        bess_power_kw: float,
        bess_capacity_kwh: float,
        grid_max_kw: float,
        rte: float,
        min_soc_pct: float,
        max_soc_pct: float,
        operating_mode: str,
        grid_loss_factor: float,
        tolerance: float = 0.01,
) -> list[ConstraintViolation]:
    """Validate dispatch output against all physical constraints.

    Parameters
    ----------
    dispatch_df:
        DataFrame loaded from the ``_dispatch_sample.csv`` output file.
    pv_peak_kwp:
        PV system peak power in kWp.
    bess_power_kw:
        BESS max charge/discharge power in kW.
    bess_capacity_kwh:
        BESS nameplate energy capacity in kWh.
    grid_max_kw:
        Maximum grid export power in kW.
    rte:
        Round-trip efficiency as a fraction (e.g. 0.88).
    min_soc_pct:
        Minimum SoC as percentage of capacity.
    max_soc_pct:
        Maximum SoC as percentage of capacity.
    operating_mode:
        ``"green"`` or ``"grey"``.
    grid_loss_factor:
        Grid loss factor (1.0 = no loss).
    tolerance:
        Absolute tolerance for floating-point comparisons (kWh).

    Returns
    -------
    list[ConstraintViolation]
        Empty list when all constraints are satisfied.
    """
    violations: list[ConstraintViolation] = []
    n = len(dispatch_df)
    has_bess = bess_power_kw > 0 and bess_capacity_kwh > 0

    # Extract columns as numpy arrays
    pv_prod = dispatch_df["pv_production_kwh"].values.astype(float)
    export_pv = dispatch_df["pv_grid_export_kwh"].values.astype(float)
    charge_pv = dispatch_df["bess_charge_pv_kwh"].values.astype(float)
    charge_grid = dispatch_df["bess_charge_grid_kwh"].values.astype(float)
    discharge_green = dispatch_df["bess_discharge_green_kwh"].values.astype(float)
    discharge_grey = dispatch_df["bess_discharge_grey_kwh"].values.astype(float)
    soc = dispatch_df["bess_soc_kwh"].values.astype(float)
    curtail = dispatch_df["curtailed_kwh"].values.astype(float)
    effective_price = dispatch_df["price_effective_eur_per_kwh"].values.astype(float)

    soc_min_kwh = bess_capacity_kwh * min_soc_pct / 100.0
    soc_max_kwh = bess_capacity_kwh * max_soc_pct / 100.0

    # Determine intervals per day from total length
    if n == 8760:
        ipd = 24
    elif n == 35040:
        ipd = 96
    else:
        ipd = 24  # fallback

    # Timestep duration in hours
    timestep_h = 24.0 / ipd

    # 1. Energy balance PV: export_pv + charge_pv + curtail ≈ pv_production
    for t in range(n):
        balance = (export_pv[t] + curtail[t]) / grid_loss_factor + charge_pv[t]
        diff = abs(balance - pv_prod[t])
        if diff > tolerance:
            violations.append(ConstraintViolation(
                constraint="energy_balance_pv",
                timestep=t,
                expected=f"export_pv + charge_pv + curtail = pv_production ({pv_prod[t]:.4f})",
                actual=balance,
                severity="error",
            ))

    if has_bess:
        # 2. SoC limits
        for t in range(n):
            if soc[t] < soc_min_kwh - tolerance:
                violations.append(ConstraintViolation(
                    constraint="soc_limits",
                    timestep=t,
                    expected=f"soc >= {soc_min_kwh:.2f}",
                    actual=soc[t],
                    severity="error",
                ))
            if soc[t] > soc_max_kwh + tolerance:
                violations.append(ConstraintViolation(
                    constraint="soc_limits",
                    timestep=t,
                    expected=f"soc <= {soc_max_kwh:.2f}",
                    actual=soc[t],
                    severity="error",
                ))
            if abs(soc[t] - dispatch_df["bess_soc_green_kwh"][t] - dispatch_df["bess_soc_grey_kwh"][t]) > tolerance:
                violations.append(ConstraintViolation(
                    constraint="soc_limits_sum_up",
                    timestep=t,
                    expected=f"soc = soc_green + soc_grey = {dispatch_df["bess_soc_green_kwh"][t] - dispatch_df["bess_soc_grey_kwh"][t]:.2f}",
                    actual=soc[t],
                    severity="error",
                ))

        # 3. Charge power limits
        for t in range(n):
            total_charge = charge_pv[t] + charge_grid[t]
            max_charge = bess_power_kw * timestep_h
            if total_charge > max_charge + tolerance:
                violations.append(ConstraintViolation(
                    constraint="charge_power_limit",
                    timestep=t,
                    expected=f"charge_pv + charge_grid <= {max_charge:.2f}",
                    actual=total_charge,
                    severity="error",
                ))

        # 4. Discharge power limits
        for t in range(n):
            total_discharge = (discharge_green[t] + discharge_grey[t]) / rte
            max_discharge = bess_power_kw * timestep_h
            if total_discharge > max_discharge + tolerance:
                violations.append(ConstraintViolation(
                    constraint="discharge_power_limit",
                    timestep=t,
                    expected=f"discharge_green + discharge_grey <= {max_discharge:.2f}",
                    actual=total_discharge,
                    severity="error",
                ))

    # 5. Grid connection limit
    grid_max_energy = grid_max_kw * timestep_h
    for t in range(n):
        total_export = export_pv[t] + (discharge_green[t] + discharge_grey[t])

        if total_export > grid_max_energy + tolerance:
            violations.append(ConstraintViolation(
                constraint="grid_connection_limit",
                timestep=t,
                expected=f"total grid export <= {grid_max_energy:.2f}",
                actual=total_export,
                severity="error",
            ))

    # 6. Non-negativity
    var_names = [
        ("pv_grid_export", export_pv),
        ("charge_pv", charge_pv),
        ("charge_grid", charge_grid),
        ("discharge_green", discharge_green),
        ("discharge_grey", discharge_grey),
        ("curtail", curtail),
    ]
    for var_name, arr in var_names:
        for t in range(n):
            if arr[t] < -tolerance:
                violations.append(ConstraintViolation(
                    constraint="non_negativity",
                    timestep=t,
                    expected=f"{var_name} >= 0",
                    actual=arr[t],
                    severity="error",
                ))

    # 7. Green mode restrictions
    if operating_mode == "green":
        for t in range(n):
            if charge_grid[t] > tolerance:
                violations.append(ConstraintViolation(
                    constraint="green_mode_no_grid_charge",
                    timestep=t,
                    expected="charge_grid = 0 in green mode",
                    actual=charge_grid[t],
                    severity="error",
                ))
            if discharge_grey[t] > tolerance:
                violations.append(ConstraintViolation(
                    constraint="green_mode_no_grey_discharge",
                    timestep=t,
                    expected="discharge_grey = 0 in green mode",
                    actual=discharge_grey[t],
                    severity="error",
                ))

    if has_bess:
        # 8. SoC charging/discharging
        n_days = n // ipd
        for t in range(n):
            previous_soc = soc[max(t-1, 0)]
            total_charge = charge_grid[t] + charge_pv[t]
            total_discharge = (discharge_green[t] / grid_loss_factor + discharge_grey[t]) / rte

            if (effective_price[t] < 0) and (total_discharge > 0):
                violations.append(ConstraintViolation(
                        constraint="discharging_at_neg_price",
                        timestep=t,
                        expected=f"only discharge, if effective price is > 0",
                        actual=total_discharge,
                        severity="error",))
            if (total_discharge > 0) and (total_charge > 0):
                violations.append(ConstraintViolation(
                    constraint="charging_discharging_simultaneously",
                    timestep=t,
                    expected=f"only discharge, or charge",
                    actual=total_discharge,
                    severity="warning", ))
            else:
                if (total_charge > 0) and (abs(previous_soc + total_charge - soc[t]) > tolerance):
                    violations.append(ConstraintViolation(
                        constraint="charging_soc_cumulative",
                        timestep=t,
                        expected=f"prev_soc + charge_grid + charge_pv = soc {previous_soc + total_charge}",
                        actual=soc[t],
                        severity="error",))

                if (total_discharge > 0) and (abs(previous_soc - total_discharge - soc[t]) > tolerance):
                    violations.append(ConstraintViolation(
                        constraint="discharging_soc_cumulative",
                        timestep=t,
                        expected=f"prev_soc - discharge_green + discharge_grey = soc {previous_soc - total_discharge}",
                        actual=soc[t],
                        severity="error",))

        # 9. BESS offline days: no charge/discharge, SoC constant
        for d in range(n_days):
            start = d * ipd
            end = (d + 1) * ipd
            day_charge = np.sum(charge_pv[start:end]) + np.sum(charge_grid[start:end])
            day_discharge = np.sum(discharge_green[start:end]) + np.sum(discharge_grey[start:end])
            is_offline = day_charge < tolerance and day_discharge < tolerance

            if is_offline and d > 0:
                # On offline days, SoC should be constant
                soc_day = soc[start:end]
                soc_spread = np.max(soc_day) - np.min(soc_day)
                if soc_spread > tolerance:
                    violations.append(ConstraintViolation(
                        constraint="bess_offline_soc_constant",
                        timestep=start,
                        expected="SoC constant on offline day",
                        actual=soc_spread,
                        severity="warning",
                    ))

    return violations


def check_availability(
        dispatch_df: pd.DataFrame,
        intervals_per_day: int = 24,
) -> tuple[int, int]:
    """Count PV and BESS offline days from dispatch data.

    Parameters
    ----------
    dispatch_df:
        DataFrame loaded from the ``_dispatch_sample.csv`` output file.
    bess_availability_pct:
        Expected BESS availability percentage (e.g. 97.0).
    has_bess:
        Whether the scenario includes a BESS.
    mc_enabled:
        Whether Monte Carlo simulation was enabled (affects offline day
        distribution).
    intervals_per_day:
        Number of intervals per day (24 for hourly, 96 for 15-min).

    Returns
    -------
    tuple[int, int]
        ``(pv_offline_days, bess_offline_days)``
    """
    n = len(dispatch_df)
    ipd = intervals_per_day
    n_days = n // ipd
    tolerance = 0.001

    pv_prod = dispatch_df["pv_production_kwh"].values.astype(float)
    charge_pv = dispatch_df["bess_charge_pv_kwh"].values.astype(float)
    charge_grid = dispatch_df["bess_charge_grid_kwh"].values.astype(float)
    discharge_green = dispatch_df["bess_discharge_green_kwh"].values.astype(float)
    discharge_grey = dispatch_df["bess_discharge_grey_kwh"].values.astype(float)

    # Intervals per hour
    iph = ipd // 24

    pv_offline_days = 0
    bess_offline_days = 0

    for d in range(n_days):
        start = d * ipd
        end = (d + 1) * ipd

        # PV offline: zero production during sunshine hours (8-17)
        sun_start = start + 8 * iph
        sun_end = start + 17 * iph
        if sun_end <= end:
            sun_production = np.sum(pv_prod[sun_start:sun_end])
            if sun_production < tolerance:
                pv_offline_days += 1

        # BESS offline: no charge and no discharge for the entire day
        day_charge = (
                np.sum(charge_pv[start:end])
                + np.sum(charge_grid[start:end])
        )
        day_discharge = (
                np.sum(discharge_green[start:end])
                + np.sum(discharge_grey[start:end])
        )
        if day_charge < tolerance and day_discharge < tolerance:
            bess_offline_days += 1

    return pv_offline_days, bess_offline_days


def check_price_dependencies(
        dispatch_df: pd.DataFrame,
) -> list[ConstraintViolation]:
    violations: list[ConstraintViolation] = []
    pv_prod = dispatch_df["pv_grid_export_kwh"].values.astype(float)
    eff_price = dispatch_df["price_effective_eur_per_kwh"].values.astype(float)

    mask = (pv_prod > 0) & (eff_price < 0)

    for idx in np.where(mask)[0]:
        violations.append(ConstraintViolation(
            constraint="no_pv_at_neg_price",
            timestep=idx,
            expected="0",
            actual=float(pv_prod[idx]),
            severity="error",
        ))

    return violations
