"""Unified CAPEX/OPEX calculation from scenario JSON using the three-component schema.

Each cost block (PV, BESS, Grid, Other) supports three additive components for CAPEX
and four for OPEX. Missing fields are treated as 0.

CAPEX_asset = fixed_eur + eur_per_kw × reference_kW + eur_per_kwh × reference_kWh
OPEX_asset  = fixed_eur + eur_per_kw × reference_kW + eur_per_kwh × reference_kWh
              + pct_of_capex × CAPEX_asset
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetCosts:
    """CAPEX and OPEX for a single asset."""

    capex: float
    opex: float


@dataclass(frozen=True)
class TotalCosts:
    """Aggregated CAPEX and OPEX across all assets."""

    capex_pv: float
    capex_bess: float
    capex_grid: float
    capex_other: float
    capex_total: float
    opex_pv: float
    opex_bess: float
    opex_grid: float
    opex_other: float
    opex_total: float


def calculate_asset_capex(
    cost_config: dict,
    reference_kw: float,
    reference_kwh: float = 0.0,
) -> float:
    """Calculate CAPEX for a single asset using the unified three-component schema.

    Args:
        cost_config: Dictionary with optional keys ``fixed_eur``, ``eur_per_kw``,
            ``eur_per_kwh``.  Missing keys are treated as 0.
        reference_kw: Reference power in kW (e.g. PV peak, BESS power, grid export).
        reference_kwh: Reference energy in kWh (e.g. BESS capacity). Ignored for
            assets without an energy dimension.

    Returns:
        CAPEX in euros.
    """
    fixed = cost_config.get("fixed_eur", 0.0)
    per_kw = cost_config.get("eur_per_kw", 0.0)
    per_kwh = cost_config.get("eur_per_kwh", 0.0)
    return fixed + per_kw * reference_kw + per_kwh * reference_kwh


def calculate_asset_opex(
    cost_config: dict,
    reference_kw: float,
    reference_kwh: float,
    asset_capex: float,
) -> float:
    """Calculate annual OPEX for a single asset using the unified four-component schema.

    Args:
        cost_config: Dictionary with optional keys ``fixed_eur``, ``eur_per_kw``,
            ``eur_per_kwh``, ``pct_of_capex``.  Missing keys are treated as 0.
        reference_kw: Reference power in kW.
        reference_kwh: Reference energy in kWh.
        asset_capex: CAPEX of the *same* asset (used for the ``pct_of_capex`` term).

    Returns:
        Annual OPEX in euros (base year, before inflation).
    """
    fixed = cost_config.get("fixed_eur", 0.0)
    per_kw = cost_config.get("eur_per_kw", 0.0)
    per_kwh = cost_config.get("eur_per_kwh", 0.0)
    pct = cost_config.get("pct_of_capex", 0.0)
    return fixed + per_kw * reference_kw + per_kwh * reference_kwh + pct * asset_capex


def calculate_replacement_cost(
    replacement_config: dict,
    bess_power_kw: float,
    bess_capacity_kwh: float,
    bess_capex: float,
) -> float:
    """Calculate BESS replacement cost using the same three-component schema.

    Args:
        replacement_config: Dictionary with optional keys ``fixed_eur``,
            ``eur_per_kw``, ``eur_per_kwh``, ``pct_of_capex``.
        bess_power_kw: BESS power rating in kW.
        bess_capacity_kwh: BESS capacity in kWh.
        bess_capex: Original BESS CAPEX (for the ``pct_of_capex`` term).

    Returns:
        Replacement cost in euros (added as OPEX in the replacement year).
    """
    fixed = replacement_config.get("fixed_eur", 0.0)
    per_kw = replacement_config.get("eur_per_kw", 0.0)
    per_kwh = replacement_config.get("eur_per_kwh", 0.0)
    pct = replacement_config.get("pct_of_capex", 0.0)
    return fixed + per_kw * bess_power_kw + per_kwh * bess_capacity_kwh + pct * bess_capex


def calculate_total_costs(
    capex_config: dict,
    opex_config: dict,
    pv_peak_kwp: float,
    bess_power_kw: float,
    bess_capacity_kwh: float,
    grid_max_export_kw: float,
) -> TotalCosts:
    """Calculate total CAPEX and base-year OPEX for all assets.

    Args:
        capex_config: Nested dict keyed by asset (``pv``, ``bess``, ``grid``,
            ``other``), each containing the unified cost fields.
        opex_config: Same structure as *capex_config* but for annual OPEX fields.
        pv_peak_kwp: PV peak power in kWp (= reference kW for PV).
        bess_power_kw: BESS power in kW.
        bess_capacity_kwh: BESS energy capacity in kWh.
        grid_max_export_kw: Grid connection power in kW.

    Returns:
        :class:`TotalCosts` dataclass with per-asset and total figures.
    """
    capex_pv = calculate_asset_capex(
        capex_config.get("pv", {}), reference_kw=pv_peak_kwp,
    )
    capex_bess = calculate_asset_capex(
        capex_config.get("bess", {}),
        reference_kw=bess_power_kw,
        reference_kwh=bess_capacity_kwh,
    )
    capex_grid = calculate_asset_capex(
        capex_config.get("grid", {}), reference_kw=grid_max_export_kw,
    )
    capex_other = calculate_asset_capex(
        capex_config.get("other", {}), reference_kw=0.0,
    )
    capex_total = capex_pv + capex_bess + capex_grid + capex_other

    opex_pv = calculate_asset_opex(
        opex_config.get("pv", {}),
        reference_kw=pv_peak_kwp,
        reference_kwh=0.0,
        asset_capex=capex_pv,
    )
    opex_bess = calculate_asset_opex(
        opex_config.get("bess", {}),
        reference_kw=bess_power_kw,
        reference_kwh=bess_capacity_kwh,
        asset_capex=capex_bess,
    )
    opex_grid = calculate_asset_opex(
        opex_config.get("grid", {}),
        reference_kw=grid_max_export_kw,
        reference_kwh=0.0,
        asset_capex=capex_grid,
    )
    opex_other = calculate_asset_opex(
        opex_config.get("other", {}),
        reference_kw=0.0,
        reference_kwh=0.0,
        asset_capex=capex_other,
    )
    opex_total = opex_pv + opex_bess + opex_grid + opex_other

    return TotalCosts(
        capex_pv=capex_pv,
        capex_bess=capex_bess,
        capex_grid=capex_grid,
        capex_other=capex_other,
        capex_total=capex_total,
        opex_pv=opex_pv,
        opex_bess=opex_bess,
        opex_grid=opex_grid,
        opex_other=opex_other,
        opex_total=opex_total,
    )
