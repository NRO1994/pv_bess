"""P50/P90 timeseries construction from multi-year PVGIS data.

For each of the 8 760 hour indices (0 = 1 Jan 00:00, 8759 = 31 Dec 23:00) we
collect the corresponding production value from every available historical year
and compute the P50 (median) and P90 (10th percentile) of that distribution.

The P50 timeseries is used for equity-return optimisation (grid search).
The P90 timeseries is used for conservative debt-service analysis.

Leap-year handling: each input array must already be of length 8 760
(December 31st stripped).  :func:`compute_p50_p90` enforces this.

Typical usage::

    from pv_bess_model.pv.timeseries import compute_p50_p90
    p50, p90 = compute_p50_p90(yearly_data)   # yearly_data from PVGISClient
"""

from __future__ import annotations

import logging

import numpy as np

from pv_bess_model.config.defaults import HOURS_PER_YEAR

logger = logging.getLogger(__name__)

# P90 is the 10th percentile: only 10 % of historical years produced less.
_P90_PERCENTILE: float = 10.0
_P50_PERCENTILE: float = 50.0


def compute_p50_p90(
    yearly_data: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute hourly P50 and P90 production arrays from multi-year data.

    For each hour index ``h`` (0 … 8 759):

    * Collect ``values[h]`` from every year in *yearly_data*.
    * P50[h] = median of that cross-year distribution.
    * P90[h] = 10th percentile (conservative, only 10 % of years are worse).

    Parameters
    ----------
    yearly_data:
        Mapping ``{calendar_year: hourly_production_array}``.
        Each array must have exactly :data:`HOURS_PER_YEAR` (8 760) elements.
        Values are in **kWh** per hour.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(p50, p90)`` – two arrays of shape ``(8760,)`` in kWh.

    Raises
    ------
    ValueError
        When *yearly_data* is empty or any array has the wrong length.
    """
    if not yearly_data:
        raise ValueError(
            "yearly_data is empty. At least one historical year is required "
            "to compute P50/P90 timeseries."
        )

    years = sorted(yearly_data.keys())
    _validate_arrays(yearly_data, years)

    # Stack into shape (n_years, 8760)
    matrix = np.stack([yearly_data[y] for y in years], axis=0)

    logger.debug(
        "Computing P50/P90 from %d historical years (%d – %d)",
        len(years),
        years[0],
        years[-1],
    )

    p50 = np.percentile(matrix, _P50_PERCENTILE, axis=0)
    p90 = np.percentile(matrix, _P90_PERCENTILE, axis=0)

    # Production cannot be negative – clip to zero
    p50 = np.maximum(p50, 0.0)
    p90 = np.maximum(p90, 0.0)

    return p50, p90


def _validate_arrays(
    yearly_data: dict[int, np.ndarray],
    years: list[int],
) -> None:
    """Raise ``ValueError`` if any array has the wrong length."""
    wrong = {
        y: len(yearly_data[y]) for y in years if len(yearly_data[y]) != HOURS_PER_YEAR
    }
    if wrong:
        details = ", ".join(f"year {y}: {n} values" for y, n in sorted(wrong.items()))
        raise ValueError(
            f"All yearly arrays must have exactly {HOURS_PER_YEAR} hourly "
            f"values. The following years have incorrect lengths: {details}. "
            "Use pvgis_client._strip_leap_day() to normalise each year."
        )


def percentile_timeseries(
    yearly_data: dict[int, np.ndarray],
    percentile: float,
) -> np.ndarray:
    """Compute an arbitrary percentile timeseries from multi-year PVGIS data.

    This generalises :func:`compute_p50_p90` for custom percentile values.

    Parameters
    ----------
    yearly_data:
        Same format as in :func:`compute_p50_p90`.
    percentile:
        Percentile in [0, 100].  For example, ``10.0`` gives P90 (10th
        percentile), ``50.0`` gives P50 (median).

    Returns
    -------
    numpy.ndarray
        Array of shape ``(8760,)`` in kWh, clipped to non-negative values.

    Raises
    ------
    ValueError
        When *yearly_data* is empty, arrays have wrong lengths, or
        *percentile* is outside [0, 100].
    """
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}.")
    if not yearly_data:
        raise ValueError("yearly_data is empty.")

    years = sorted(yearly_data.keys())
    _validate_arrays(yearly_data, years)

    matrix = np.stack([yearly_data[y] for y in years], axis=0)
    result = np.percentile(matrix, percentile, axis=0)
    return np.maximum(result, 0.0)
