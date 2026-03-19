"""Heat demand profile from PVGIS temperature data.

Computes degree-day-based heat demand profiles and temperature-dependent
COP curves for heat pump flexibility modeling.

The degree-day (Gradtagszahl) approach distributes annual thermal demand
across hours proportionally to the heating degree hours, i.e. how far
the outdoor temperature falls below the heating threshold.

The COP (Coefficient of Performance) curve is a simplified linear model
where COP varies with outdoor temperature around a nominal operating point.

Typical usage::

    from pv_bess_model.portfolio.heat_demand import compute_heat_demand, compute_cop

    heat_qh = compute_heat_demand(temp_hourly, annual_thermal_demand_mwh=15000)
    cop_qh = compute_cop(temp_hourly, cop_nominal=3.5, cop_reference_temp_c=7.0)
"""

from __future__ import annotations

import logging

import numpy as np

from pv_bess_model.config.defaults import (
    DEFAULT_COP_TEMP_COEFFICIENT,
    DEFAULT_HEAT_DEMAND_HEIZGRENZE_C,
    HOURS_PER_YEAR,
    INTERVALS_PER_HOUR,
)

logger = logging.getLogger(__name__)

# Minimum COP value to prevent unrealistic results at very low temperatures
_MIN_COP: float = 1.0


def compute_heat_demand(
    temperature_hourly: np.ndarray,
    annual_thermal_demand_mwh: float,
    heizgrenze_c: float = DEFAULT_HEAT_DEMAND_HEIZGRENZE_C,
) -> np.ndarray:
    """Compute a quarter-hourly heat demand profile using degree-day method.

    The heating degree hours (Gradtagszahl) distribute the annual thermal
    demand across hours proportionally to the temperature deficit below
    the heating threshold::

        GTZ[h] = max(0, heizgrenze - T_outdoor[h])
        Q_th[h] = GTZ[h] / sum(GTZ) × Q_annual_total

    The hourly values are then expanded to quarter-hourly resolution by
    dividing each hourly value by 4 (energy conservation).

    Parameters
    ----------
    temperature_hourly:
        Hourly outdoor temperature array of length 8,760 in °C.
        Typically from PVGIS ``T2m`` field.
    annual_thermal_demand_mwh:
        Total annual thermal demand in MWh.
    heizgrenze_c:
        Heating threshold temperature in °C.  No heating demand is
        generated for hours where the outdoor temperature exceeds
        this value.

    Returns
    -------
    numpy.ndarray
        Quarter-hourly heat demand array of length 35,040 in kWh_th.

    Raises
    ------
    ValueError
        When *temperature_hourly* does not have 8,760 elements or
        *annual_thermal_demand_mwh* is negative.
    """
    if len(temperature_hourly) != HOURS_PER_YEAR:
        raise ValueError(
            f"temperature_hourly must have {HOURS_PER_YEAR} elements, "
            f"got {len(temperature_hourly)}."
        )
    if annual_thermal_demand_mwh < 0:
        raise ValueError(
            f"annual_thermal_demand_mwh must be >= 0, "
            f"got {annual_thermal_demand_mwh}."
        )

    temp = np.asarray(temperature_hourly, dtype=float)
    annual_kwh = annual_thermal_demand_mwh * 1000.0

    # Degree hours: max(0, heizgrenze - T)
    degree_hours = np.maximum(0.0, heizgrenze_c - temp)
    total_degree_hours = np.sum(degree_hours)

    if total_degree_hours == 0.0:
        logger.warning(
            "All temperatures >= %.1f°C – no heating demand generated.",
            heizgrenze_c,
        )
        return np.zeros(HOURS_PER_YEAR * INTERVALS_PER_HOUR, dtype=float)

    # Distribute annual demand proportionally to degree hours
    heat_hourly = degree_hours / total_degree_hours * annual_kwh

    # Expand to quarter-hourly: each hourly value / 4
    heat_qh = np.repeat(heat_hourly / INTERVALS_PER_HOUR, INTERVALS_PER_HOUR)

    logger.info(
        "Heat demand profile: %.0f MWh/a, %.0f degree-hours, "
        "peak hourly=%.1f kWh, mean temp=%.1f°C.",
        annual_thermal_demand_mwh,
        total_degree_hours,
        float(np.max(heat_hourly)),
        float(np.mean(temp)),
    )

    return heat_qh


def compute_cop(
    temperature_hourly: np.ndarray,
    cop_nominal: float,
    cop_reference_temp_c: float,
    temp_coefficient: float = DEFAULT_COP_TEMP_COEFFICIENT,
) -> np.ndarray:
    """Compute a quarter-hourly COP profile from hourly temperatures.

    The COP is a simplified linear function of outdoor temperature::

        COP(T) = COP_nominal × (1 + coefficient × (T - T_reference))

    The COP is clipped to a minimum of :data:`_MIN_COP` (1.0) to prevent
    unrealistic values at very low temperatures.

    Parameters
    ----------
    temperature_hourly:
        Hourly outdoor temperature array of length 8,760 in °C.
    cop_nominal:
        COP at the reference temperature (dimensionless).
    cop_reference_temp_c:
        Reference temperature at which ``COP = COP_nominal`` (°C).
    temp_coefficient:
        Linear temperature sensitivity coefficient.  Default is 0.025,
        meaning COP changes by 2.5% per °C deviation from reference.

    Returns
    -------
    numpy.ndarray
        Quarter-hourly COP array of length 35,040 (dimensionless).

    Raises
    ------
    ValueError
        When *temperature_hourly* has wrong length or *cop_nominal* <= 0.
    """
    if len(temperature_hourly) != HOURS_PER_YEAR:
        raise ValueError(
            f"temperature_hourly must have {HOURS_PER_YEAR} elements, "
            f"got {len(temperature_hourly)}."
        )
    if cop_nominal <= 0:
        raise ValueError(f"cop_nominal must be > 0, got {cop_nominal}.")

    temp = np.asarray(temperature_hourly, dtype=float)

    # COP(T) = COP_nominal × (1 + coeff × (T - T_ref))
    cop_hourly = cop_nominal * (1.0 + temp_coefficient * (temp - cop_reference_temp_c))

    # Clip to minimum COP
    cop_hourly = np.maximum(cop_hourly, _MIN_COP)

    # Expand to quarter-hourly (same COP for all 4 intervals in each hour)
    cop_qh = np.repeat(cop_hourly, INTERVALS_PER_HOUR)

    logger.info(
        "COP profile: nominal=%.2f @ %.1f°C, range=[%.2f, %.2f], mean=%.2f.",
        cop_nominal,
        cop_reference_temp_c,
        float(np.min(cop_qh)),
        float(np.max(cop_qh)),
        float(np.mean(cop_qh)),
    )

    return cop_qh


def compute_daily_heat_demand(
    heat_demand_qh: np.ndarray,
) -> np.ndarray:
    """Aggregate quarter-hourly heat demand to daily totals.

    Parameters
    ----------
    heat_demand_qh:
        Quarter-hourly heat demand array of length 35,040 in kWh_th.

    Returns
    -------
    numpy.ndarray
        Daily heat demand array of length 365 in kWh_th.

    Raises
    ------
    ValueError
        When *heat_demand_qh* does not have 35,040 elements.
    """
    expected = HOURS_PER_YEAR * INTERVALS_PER_HOUR
    if len(heat_demand_qh) != expected:
        raise ValueError(
            f"heat_demand_qh must have {expected} elements, "
            f"got {len(heat_demand_qh)}."
        )

    from pv_bess_model.config.defaults import INTERVALS_PER_DAY, DAYS_PER_YEAR

    return heat_demand_qh.reshape(DAYS_PER_YEAR, INTERVALS_PER_DAY).sum(axis=1)
