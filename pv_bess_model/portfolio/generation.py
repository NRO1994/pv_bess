"""PV generation profile aggregation for portfolio simulations.

Fetches PVGIS data for each generation asset, converts to quarter-hourly
resolution, applies system losses, and aggregates all assets into a single
combined production profile.

Typical usage::

    from pv_bess_model.portfolio.generation import build_aggregated_pv_profile

    pv_qh, temp_hourly = build_aggregated_pv_profile(
        generation_configs=[gen_config],
        weather_year=2018,
    )
"""

from __future__ import annotations

import logging

import numpy as np

from pv_bess_model.config.defaults import HOURS_PER_YEAR, INTERVALS_PER_HOUR
from pv_bess_model.config.loader_portfolio import GenerationConfig
from pv_bess_model.pv.pvgis_client import PVGISClient
from pv_bess_model.pv.timeseries import hourly_to_quarter_hourly

logger = logging.getLogger(__name__)


def build_aggregated_pv_profile(
    generation_configs: list[GenerationConfig],
    weather_year: int,
    pvgis_client: PVGISClient | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an aggregated PV production profile from multiple generators.

    For each :class:`GenerationConfig`, fetches PVGIS hourly data for the
    given weather year (production + temperature), applies system losses,
    converts to quarter-hourly, and sums all profiles.

    Temperature is taken from the **first** generation asset (all assets
    are assumed to be at similar locations for the portfolio model).

    Parameters
    ----------
    generation_configs:
        List of PV generation asset configurations.
    weather_year:
        Calendar year for PVGIS data (matched to price scenario).
    pvgis_client:
        Optional pre-configured PVGIS client.  If ``None``, a default
        client is created.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(pv_profile_qh, temperature_hourly)``

        - ``pv_profile_qh``: Aggregated quarter-hourly PV production in kWh
          (35,040 elements).
        - ``temperature_hourly``: Hourly temperature in °C from the first
          generation asset (8,760 elements).

    Raises
    ------
    ValueError
        When *generation_configs* is empty.
    """
    if not generation_configs:
        raise ValueError("At least one generation config is required.")

    if pvgis_client is None:
        pvgis_client = PVGISClient()

    expected_qh = HOURS_PER_YEAR * INTERVALS_PER_HOUR
    aggregated = np.zeros(expected_qh, dtype=float)
    temperature_hourly: np.ndarray | None = None

    for gen in generation_configs:
        # For the first asset, also extract temperature (same cached response)
        need_temp = temperature_hourly is None
        result = pvgis_client.fetch_single_year(
            year=weather_year,
            latitude=gen.latitude,
            longitude=gen.longitude,
            peak_power_kwp=gen.peak_power_kwp,
            mounting_type=gen.mounting_type,
            azimuth_deg=gen.azimuth_deg,
            tilt_deg=gen.tilt_deg,
            pvgis_database=gen.pvgis_database,
            include_temperature=need_temp,
        )

        if need_temp:
            assert isinstance(result, dict)
            production_hourly = result["production"]
            temperature_hourly = result["temperature"]
        else:
            assert isinstance(result, np.ndarray)
            production_hourly = result

        # Apply system losses at grid connection point
        loss_factor = 1.0 - gen.system_loss_pct / 100.0
        production_hourly = production_hourly * loss_factor

        # Convert to quarter-hourly
        production_qh = hourly_to_quarter_hourly(production_hourly)
        aggregated += production_qh

        logger.info(
            "PV asset '%s': %.0f kWp, weather year %d, "
            "system loss %.1f%%, annual yield %.0f MWh.",
            gen.name,
            gen.peak_power_kwp,
            weather_year,
            gen.system_loss_pct,
            float(np.sum(production_qh)) / 1000.0,
        )

    assert temperature_hourly is not None  # guaranteed by non-empty list
    return aggregated, temperature_hourly


def build_per_asset_pv_profiles(
    generation_configs: list[GenerationConfig],
    weather_year: int,
    pvgis_client: PVGISClient | None = None,
) -> dict[str, np.ndarray]:
    """Build individual PV production profiles for each generation asset.

    Parameters
    ----------
    generation_configs:
        List of PV generation asset configurations.
    weather_year:
        Calendar year for PVGIS data.
    pvgis_client:
        Optional pre-configured PVGIS client.

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping of asset name to quarter-hourly production profile (35,040).
    """
    if not generation_configs:
        return {}

    if pvgis_client is None:
        pvgis_client = PVGISClient()

    expected_qh = HOURS_PER_YEAR * INTERVALS_PER_HOUR
    profiles: dict[str, np.ndarray] = {}

    for gen in generation_configs:
        result = pvgis_client.fetch_single_year(
            year=weather_year,
            latitude=gen.latitude,
            longitude=gen.longitude,
            peak_power_kwp=gen.peak_power_kwp,
            mounting_type=gen.mounting_type,
            azimuth_deg=gen.azimuth_deg,
            tilt_deg=gen.tilt_deg,
            pvgis_database=gen.pvgis_database,
            include_temperature=False,
        )

        assert isinstance(result, np.ndarray)
        production_hourly = result

        # Apply system losses
        loss_factor = 1.0 - gen.system_loss_pct / 100.0
        production_hourly = production_hourly * loss_factor

        # Convert to quarter-hourly
        production_qh = hourly_to_quarter_hourly(production_hourly)
        profiles[gen.name] = production_qh

    return profiles
