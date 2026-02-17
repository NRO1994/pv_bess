"""Annual linear degradation applied to PV production timeseries.

CLAUDE.md specification
-----------------------
For project year *Y* (1-indexed, Y = 1 is the first operating year):

    production[Y] = base_production × (1 − degradation_rate) ^ Y

where ``degradation_rate`` is the fraction per year (e.g. 0.004 for 0.4 %/yr).

Year 0 (construction/commissioning) is not a simulation year and therefore not
part of the output.  The first element of the returned list corresponds to
project year 1.

Typical usage::

    from pv_bess_model.pv.degradation import apply_degradation
    yearly = apply_degradation(base_p50, degradation_rate=0.004, lifetime_years=25)
    # yearly[0] is year-1 production, yearly[24] is year-25 production
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def apply_degradation(
    base_timeseries: np.ndarray,
    degradation_rate: float,
    lifetime_years: int,
) -> list[np.ndarray]:
    """Return a degraded hourly timeseries for each project year.

    Parameters
    ----------
    base_timeseries:
        Undegraded hourly production array for year 0 (i.e. new-plant
        production in kWh/hour).  Typically the P50 or P90 from PVGIS.
    degradation_rate:
        Annual degradation as a fraction (e.g. ``0.004`` for 0.4 %/year).
        Must be in [0, 1).
    lifetime_years:
        Number of operating years to simulate (e.g. ``25``).

    Returns
    -------
    list[numpy.ndarray]
        List of length *lifetime_years*.  Element ``[i]`` (0-indexed) is the
        production array for project year ``i + 1``, scaled by
        ``(1 − degradation_rate) ^ (i + 1)``.

    Raises
    ------
    ValueError
        When *degradation_rate* is outside [0, 1) or *lifetime_years* < 1.

    Notes
    -----
    All output arrays share a reference to the *same* underlying data scaled
    by a scalar, so they consume minimal memory when the base array is large.
    """
    if not 0.0 <= degradation_rate < 1.0:
        raise ValueError(
            f"degradation_rate must be in [0, 1), got {degradation_rate}. "
            "For a 0.4 %/year rate pass 0.004, not 0.4."
        )
    if lifetime_years < 1:
        raise ValueError(f"lifetime_years must be ≥ 1, got {lifetime_years}.")

    base = np.asarray(base_timeseries, dtype=float)
    result: list[np.ndarray] = []

    for year in range(1, lifetime_years + 1):
        factor = (1.0 - degradation_rate) ** year
        result.append(base * factor)

    logger.debug(
        "Applied %.4f%% annual degradation over %d years "
        "(year-1 factor=%.6f, year-%d factor=%.6f)",
        degradation_rate * 100,
        lifetime_years,
        (1.0 - degradation_rate) ** 1,
        lifetime_years,
        (1.0 - degradation_rate) ** lifetime_years,
    )

    return result


def degradation_factor(degradation_rate: float, year: int) -> float:
    """Return the scalar multiplier for a single project year.

    Convenience function for callers that need just the factor without
    applying it to an array (e.g. BESS capacity degradation).

    Parameters
    ----------
    degradation_rate:
        Annual degradation fraction (e.g. ``0.004``).
    year:
        1-indexed project year (year 1 = first operating year).

    Returns
    -------
    float
        ``(1 − degradation_rate) ^ year``

    Raises
    ------
    ValueError
        When *degradation_rate* is outside [0, 1) or *year* < 1.
    """
    if not 0.0 <= degradation_rate < 1.0:
        raise ValueError(f"degradation_rate must be in [0, 1), got {degradation_rate}.")
    if year < 1:
        raise ValueError(f"year must be ≥ 1, got {year}.")
    return (1.0 - degradation_rate) ** year
