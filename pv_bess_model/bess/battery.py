"""BESS state model: SoC tracking, charge/discharge, annual degradation.

The efficiency model applies losses on discharge only:
  - Charging is lossless: 1 kWh requested → 1 kWh SoC increase.
  - Discharge output = kWh removed from SoC × round-trip efficiency.

All power values are in kW (= kWh per hour for 1-hour timesteps).
All energy values are in kWh.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class BatteryState:
    """Models the instantaneous state of a battery energy storage system (BESS).

    Tracks current SoC, applies charge/discharge with physical limits, and
    accumulates cumulative energy throughput for reporting and degradation
    accounting.

    Efficiency convention
    ---------------------
    Charging is lossless (1 kWh in → 1 kWh SoC increase).
    Discharging applies the round-trip efficiency on the *output* side:
        grid_output_kwh = kWh_removed_from_soc × round_trip_efficiency
    """

    def __init__(
        self,
        max_capacity_kwh: float,
        max_charge_power_kw: float,
        max_discharge_power_kw: float,
        round_trip_efficiency_pct: float,
        min_soc_pct: float,
        max_soc_pct: float,
        initial_soc_kwh: float | None = None,
    ) -> None:
        """Initialise a BatteryState.

        Args:
            max_capacity_kwh: Current maximum energy capacity in kWh (nameplate
                at time of construction; may decrease via degradation).
            max_charge_power_kw: Maximum charging power in kW (kWh per hour).
            max_discharge_power_kw: Maximum discharging power in kW.
            round_trip_efficiency_pct: Round-trip efficiency as a percentage
                (e.g. 88 means 88 %).  Losses are applied on discharge only.
            min_soc_pct: Minimum allowable SoC as % of max_capacity_kwh (e.g.
                10 means the battery may not drop below 10 % of capacity).
            max_soc_pct: Maximum allowable SoC as % of max_capacity_kwh (e.g.
                90 means the battery may not be charged above 90 % of capacity).
            initial_soc_kwh: Starting SoC in kWh.  When *None* the SoC is
                initialised to the midpoint of the usable window
                (average of min and max SoC limits).

        Raises:
            ValueError: If any parameter is outside its valid range.
        """
        if max_capacity_kwh < 0.0:
            raise ValueError(
                f"max_capacity_kwh must be >= 0, got {max_capacity_kwh}"
            )
        if not (0.0 < round_trip_efficiency_pct <= 100.0):
            raise ValueError(
                f"round_trip_efficiency_pct must be in (0, 100], "
                f"got {round_trip_efficiency_pct}"
            )
        if not (0.0 <= min_soc_pct < max_soc_pct <= 100.0):
            raise ValueError(
                f"Require 0 <= min_soc_pct < max_soc_pct <= 100, "
                f"got min={min_soc_pct}, max={max_soc_pct}"
            )

        # Store nameplate for reset on replacement
        self._nameplate_kwh: float = max_capacity_kwh

        self.max_capacity_kwh: float = max_capacity_kwh
        self.max_charge_power_kw: float = max_charge_power_kw
        self.max_discharge_power_kw: float = max_discharge_power_kw
        self.round_trip_efficiency: float = round_trip_efficiency_pct / 100.0
        self.min_soc_pct: float = min_soc_pct
        self.max_soc_pct: float = max_soc_pct

        self.cumulative_throughput_kwh: float = 0.0

        if initial_soc_kwh is None:
            mid_fraction = (min_soc_pct + max_soc_pct) / 2.0 / 100.0
            self.current_soc_kwh: float = max_capacity_kwh * mid_fraction
        else:
            self.current_soc_kwh = float(initial_soc_kwh)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def min_soc_kwh(self) -> float:
        """Lower SoC limit in kWh (computed from current max_capacity_kwh)."""
        return self.max_capacity_kwh * self.min_soc_pct / 100.0

    @property
    def max_soc_kwh(self) -> float:
        """Upper SoC limit in kWh (computed from current max_capacity_kwh)."""
        return self.max_capacity_kwh * self.max_soc_pct / 100.0

    # ------------------------------------------------------------------
    # Charge / discharge
    # ------------------------------------------------------------------

    def charge(self, kwh: float) -> float:
        """Charge the battery by up to *kwh* kilowatt-hours.

        Charging is lossless: every kWh accepted increases SoC by 1 kWh.
        The actual amount charged is the minimum of:
          - the requested *kwh*
          - headroom remaining up to max_soc_kwh
          - max_charge_power_kw (energy limit per hour)

        Cumulative throughput is incremented by the actual charged amount.

        Args:
            kwh: Requested charge in kWh.  Must be >= 0.

        Returns:
            Actual kWh charged into the battery (0 ≤ result ≤ kwh).

        Raises:
            ValueError: If *kwh* is negative.
        """
        if kwh < 0.0:
            raise ValueError(f"Requested charge must be >= 0, got {kwh}")

        headroom = max(0.0, self.max_soc_kwh - self.current_soc_kwh)
        actual_kwh = min(kwh, headroom, self.max_charge_power_kw)

        self.current_soc_kwh += actual_kwh
        self.cumulative_throughput_kwh += actual_kwh
        return actual_kwh

    def discharge(self, kwh: float) -> float:
        """Discharge the battery, removing up to *kwh* kilowatt-hours from SoC.

        Losses are applied on discharge only:
            grid_output_kwh = kWh_removed_from_SoC × round_trip_efficiency

        The actual kWh removed from SoC is the minimum of:
          - the requested *kwh*
          - energy available above min_soc_kwh
          - max_discharge_power_kw

        Cumulative throughput is incremented by the kWh *removed from SoC*
        (not the grid output).

        Args:
            kwh: Requested discharge in kWh (energy to remove from SoC).
                Must be >= 0.

        Returns:
            Actual kWh delivered to the grid
            (= kWh removed from SoC × round_trip_efficiency).

        Raises:
            ValueError: If *kwh* is negative.
        """
        if kwh < 0.0:
            raise ValueError(f"Requested discharge must be >= 0, got {kwh}")

        available = max(0.0, self.current_soc_kwh - self.min_soc_kwh)
        actual_kwh = min(kwh, available, self.max_discharge_power_kw)

        self.current_soc_kwh -= actual_kwh
        self.cumulative_throughput_kwh += actual_kwh
        return actual_kwh * self.round_trip_efficiency

    # ------------------------------------------------------------------
    # Degradation
    # ------------------------------------------------------------------

    def apply_annual_degradation(self, rate: float) -> None:
        """Reduce the maximum capacity by *rate* fraction for one year.

        The formula applied is:
            max_capacity_kwh *= (1 - rate)

        After reducing the capacity, the current SoC is clipped to the new
        max_soc_kwh if it would otherwise exceed the updated limit.

        Args:
            rate: Annual degradation rate as a fraction (e.g. 0.02 for 2 %).
                Must be in [0, 1).

        Raises:
            ValueError: If *rate* is outside [0, 1).
        """
        if not (0.0 <= rate < 1.0):
            raise ValueError(
                f"Degradation rate must be in [0, 1), got {rate}"
            )

        self.max_capacity_kwh *= (1.0 - rate)
        # Clip SoC to the updated upper limit
        self.current_soc_kwh = min(self.current_soc_kwh, self.max_soc_kwh)
