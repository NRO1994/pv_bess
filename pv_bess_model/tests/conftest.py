"""Shared pytest fixtures for the pv_bess_model test suite.

All fixtures provide synthetic, deterministic data so tests run without
real PVGIS API calls or large CSV files. Numerical reference fixtures document
expected results for key calculations to enable regression testing.

Reference CAPEX breakdown (used by reference_capex_total and related fixtures)
-----------------------------------------------------------------------
PV   5 000 kWp  :  50 000 (fixed) + 800 €/kW × 5 000 kW      = 4 050 000 €
BESS 2 000 kW / 4 000 kWh:
                  50 000 (fixed) + 100 €/kW × 2 000 kW
                                 + 250 €/kWh × 4 000 kWh      = 1 250 000 €
Grid 4 000 kW  :  50 000 (fixed) + 100 €/kW × 4 000 kW        =   450 000 €
                                                          Total = 5 750 000 €

BESS sizing derives from: scale = 40 % of PV peak, E/P ratio = 2 h
  BESS_power    = 5 000 kWp × 0.40         = 2 000 kW
  BESS_capacity = 2 000 kW  × 2 h          = 4 000 kWh

Reference annuity: 75 % leverage on 5 750 000 € = 4 312 500 € loan
  r = 4.5 %, n = 18 years  →  computed via numpy_financial.pmt

Reference LP optimiser (4 h, Green Mode):
  PV   = [100, 200,  50,   0] kWh
  spot = [ 30,  10,  10,  80] €/MWh
  BESS 100 kW / 200 kWh, RTE 90 %, SoC 10–90 %, start = 100 kWh
  Grid max 150 kW
  Optimal:
    t=0: export 100 kWh, charge   0 kWh  (SoC 100 → 100)
    t=1: export 150 kWh, charge  50 kWh  (SoC 100 → 150, grid-limit fills gap)
    t=2: export  50 kWh, charge   0 kWh  (SoC stays 150)
    t=3: export   0 kWh, discharge 100 kWh → 90 kWh to grid  (SoC 150 → 50)
  Revenue: 3.00 + 1.50 + 0.50 + 7.20 = 12.20 €
"""

from __future__ import annotations

import math

import numpy as np
import numpy_financial as npf
import pytest

# ---------------------------------------------------------------------------
# PV timeseries fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_pv_timeseries_24h() -> np.ndarray:
    """24-hour synthetic PV profile using a half-sine (daylight hours 6–18).

    Returns an array of length 24 (kWh per hour) with a peak of 500 kWh at
    solar noon (hour 12). Hours outside 6–18 are zero.
    """
    hours = np.arange(24)
    peak_kw = 500.0
    # Sine arch spanning hours 6 to 18 (inclusive)
    production = np.where(
        (hours >= 6) & (hours <= 18),
        peak_kw * np.sin(np.pi * (hours - 6) / 12),
        0.0,
    )
    return production.astype(float)


@pytest.fixture
def sample_pv_timeseries_8760h() -> np.ndarray:
    """Synthetic full-year PV profile (8 760 hourly values, kWh).

    Constructed as a product of:
    - a daily half-sine arch (daylight hours 6–18)
    - a seasonal modulation (summer peak, winter trough)

    Peak hourly production is 1 000 kWh at solar noon in mid-summer.
    Winter production is approximately 30 % of the summer peak.
    """
    hours = np.arange(8760)
    day_of_year = hours // 24  # 0-indexed
    hour_of_day = hours % 24

    # Seasonal factor: cosine with max at summer solstice (day 172)
    seasonal = 0.65 + 0.35 * np.cos(2 * np.pi * (day_of_year - 172) / 365)

    # Daily arch: zero outside 6–18
    daylight = np.where(
        (hour_of_day >= 6) & (hour_of_day <= 18),
        np.sin(np.pi * (hour_of_day - 6) / 12),
        0.0,
    )

    peak_kw = 1000.0
    production = peak_kw * seasonal * daylight
    return production.astype(float)


# ---------------------------------------------------------------------------
# Price timeseries fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_price_timeseries_24h() -> np.ndarray:
    """24-hour spot price profile in €/MWh, including negative hours.

    Night hours (0–5, 22–23): low/slightly negative prices.
    Morning ramp (6–9): rising prices.
    Midday (10–16): moderately positive.
    Evening peak (17–20): high prices.
    Evening wind-down (21): declining.
    """
    prices = np.array(
        [
            -5.0,  # 00
            -8.0,  # 01
            -12.0,  # 02  (negative: excess renewable)
            -6.0,  # 03
            2.0,  # 04
            10.0,  # 05
            25.0,  # 06
            40.0,  # 07
            55.0,  # 08
            65.0,  # 09
            60.0,  # 10
            50.0,  # 11
            35.0,  # 12  (solar suppresses midday price)
            32.0,  # 13
            38.0,  # 14
            45.0,  # 15
            60.0,  # 16
            85.0,  # 17  (evening peak)
            95.0,  # 18
            90.0,  # 19
            75.0,  # 20
            55.0,  # 21
            20.0,  # 22
            5.0,  # 23
        ],
        dtype=float,
    )
    assert len(prices) == 24
    return prices


@pytest.fixture
def sample_price_timeseries_8760h(
    sample_price_timeseries_24h: np.ndarray,
) -> np.ndarray:
    """Full-year spot price profile (8 760 values) in €/MWh.

    Built by tiling the 24-hour profile 365 times and adding a small
    normally distributed noise (seed=42) so adjacent days differ slightly.
    """
    rng = np.random.default_rng(42)
    base = np.tile(sample_price_timeseries_24h, 365)
    noise = rng.normal(loc=0.0, scale=3.0, size=8760)
    return (base + noise).astype(float)


# ---------------------------------------------------------------------------
# CAPEX / OPEX config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_capex_config() -> dict:
    """Unified CAPEX configuration dict matching the reference CAPEX total.

    PV   : fixed=50 000 € + 800 €/kW  →  5 000 kWp  →  4 050 000 €
    BESS : fixed=50 000 € + 100 €/kW + 250 €/kWh → 2 000 kW / 4 000 kWh → 1 250 000 €
    Grid : fixed=50 000 € + 100 €/kW  →  4 000 kW  →    450 000 €
    Total: 5 750 000 €
    """
    return {
        "pv": {
            "fixed_eur": 50_000.0,
            "eur_per_kw": 800.0,
        },
        "bess": {
            "fixed_eur": 50_000.0,
            "eur_per_kw": 100.0,
            "eur_per_kwh": 250.0,
        },
        "grid": {
            "fixed_eur": 50_000.0,
            "eur_per_kw": 100.0,
        },
        "other": {},
    }


@pytest.fixture
def sample_opex_config() -> dict:
    """Annual OPEX configuration dict (base year, before inflation).

    PV   : fixed=5 000 € + 12 €/kW + 0.3 % of CAPEX_pv
    BESS : fixed=25 000 €          + 1.5 % of CAPEX_bess
    Grid : fixed=25 000 €          + 1.5 % of CAPEX_grid
    """
    return {
        "pv": {
            "fixed_eur": 5_000.0,
            "eur_per_kw": 12.0,
            "pct_of_capex": 0.003,
        },
        "bess": {
            "fixed_eur": 25_000.0,
            "pct_of_capex": 0.015,
        },
        "grid": {
            "fixed_eur": 25_000.0,
            "pct_of_capex": 0.015,
        },
        "other": {},
    }


# ---------------------------------------------------------------------------
# Battery state fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_battery_state() -> dict:
    """A representative mid-life BESS state dictionary.

    Corresponds to a 200 kWh (nameplate) battery at 90 % capacity retention,
    SoC limits 10–90 %, currently at 55 % SoC (within usable range).
    """
    nameplate_kwh = 200.0
    capacity_kwh = nameplate_kwh * 0.90  # after degradation
    min_soc_pct = 10.0
    max_soc_pct = 90.0
    usable_kwh = capacity_kwh * (max_soc_pct - min_soc_pct) / 100.0
    current_soc_kwh = capacity_kwh * 0.55  # 55 % of current capacity

    return {
        "capacity_kwh": capacity_kwh,
        "nameplate_kwh": nameplate_kwh,
        "max_power_kw": 100.0,
        "round_trip_efficiency_pct": 90.0,
        "min_soc_pct": min_soc_pct,
        "max_soc_pct": max_soc_pct,
        "min_soc_kwh": capacity_kwh * min_soc_pct / 100.0,
        "max_soc_kwh": capacity_kwh * max_soc_pct / 100.0,
        "usable_kwh": usable_kwh,
        "current_soc_kwh": current_soc_kwh,
        "cumulative_throughput_kwh": 12_500.0,
    }


# ---------------------------------------------------------------------------
# Scenario config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_scenario_config_green() -> dict:
    """Complete scenario config dict – Green Mode, EEG floor tariff.

    Dimensioned so that the reference CAPEX total of 5 750 000 € is reached:
      - PV peak 5 000 kWp
      - BESS scale 40 % of PV → 2 000 kW / 4 000 kWh (E/P = 2 h)
      - Grid connection 4 000 kW
    """
    return {
        "scenario": {
            "name": "EEG_Green_Test",
            "monte_carlo": {
                "enabled": False,
                "iterations": 100,
                "sigma_pv_yield_pct": 5.0,
                "sigma_capex_pct": 8.0,
                "sigma_opex_pct": 5.0,
                "sigma_bess_availability_pct": 2.0,
                "price_scenarios": {
                    "mid": {"csv_column": "MID", "weight": 1.0},
                },
            },
            "output": {
                "directory": "output/test/",
                "export_dispatch_sample": False,
            },
        },
        "project_settings": {
            "lifetime_years": 25,
            "discount_rate": 0.06,
            "operating_mode": "green",
            "location": {
                "latitude": 53.55,
                "longitude": 9.99,
                "pvgis_database": "PVGIS-SARAH2",
            },
            "technology": {
                "pv": {
                    "design": {
                        "peak_power_kwp": 5_000.0,
                        "mounting_type": "free",
                        "azimuth_deg": 0,
                        "tilt_deg": 30,
                    },
                    "performance": {
                        "system_loss_pct": 14.0,
                        "degradation_rate_pct_per_year": 0.4,
                    },
                    "costs": {
                        "capex": {
                            "fixed_eur": 50_000.0,
                            "eur_per_kw": 800.0,
                        },
                        "opex": {
                            "fixed_eur": 5_000.0,
                            "eur_per_kw": 12.0,
                            "pct_of_capex": 0.003,
                        },
                    },
                },
                "bess": {
                    "design_space": {
                        "scale_pct_of_pv": [0, 40],
                        "e_to_p_ratio_hours": [2],
                    },
                    "performance": {
                        "round_trip_efficiency_pct": 90.0,
                        "min_soc_pct": 10.0,
                        "max_soc_pct": 90.0,
                        "degradation_rate_pct_per_year": 2.0,
                        "bess_availability_pct": 97.0,
                    },
                    "costs": {
                        "capex": {
                            "fixed_eur": 50_000.0,
                            "eur_per_kw": 100.0,
                            "eur_per_kwh": 250.0,
                        },
                        "opex": {
                            "fixed_eur": 25_000.0,
                            "pct_of_capex": 0.015,
                        },
                        "replacement": {
                            "enabled": False,
                            "year": 12,
                            "fixed_eur": 0.0,
                            "eur_per_kw": 120.0,
                            "eur_per_kwh": 141.0,
                        },
                    },
                },
                "grid_connection": {
                    "max_export_kw": 4_000.0,
                    "costs": {
                        "capex": {
                            "fixed_eur": 50_000.0,
                            "eur_per_kw": 100.0,
                        },
                        "opex": {
                            "fixed_eur": 25_000.0,
                            "pct_of_capex": 0.015,
                        },
                    },
                },
            },
            "finance": {
                "leverage_pct": 75.0,
                "interest_rate_pct": 4.5,
                "loan_tenor_years": 18,
                "equity_irr_target": None,
                "debt_uses_p90": True,
                "inflation_rate": 0.02,
                "revenue_streams": {
                    "marketing": {
                        "type": "eeg",
                        "floor_price_eur_per_kwh": 0.0735,
                        "fixed_price_years": 20,
                        "eeg_inflation": False,
                    },
                    "ppa": {
                        "type": "none",
                        "pay_as_produced_price_eur_per_kwh": None,
                        "baseload_mw": None,
                        "floor_price_eur_per_kwh": None,
                        "cap_price_eur_per_kwh": None,
                        "duration_years": 10,
                        "inflation_on_ppa": False,
                        "guarantee_of_origin_eur_per_kwh": 0.005,
                    },
                },
                "price_inputs": {
                    "day_ahead_csv": "data/day_ahead_prices.csv",
                    "price_unit": "eur_per_mwh",
                    "inflation_on_input_data": True,
                },
                "tax": {
                    "afa_years_pv": 20,
                    "afa_years_bess": 10,
                    "gewerbesteuer_hebesatz": 400,
                    "gewerbesteuer_messzahl": 0.035,
                },
            },
        },
    }


@pytest.fixture
def sample_scenario_config_grey() -> dict:
    """Complete scenario config dict – Grey Mode, PPA floor tariff.

    Uses the same PV / BESS / grid sizing as sample_scenario_config_green so
    that CAPEX figures remain comparable. The only structural differences are:
      - operating_mode = "grey"
      - marketing type = "ppa" with ppa_floor
    """
    return {
        "scenario": {
            "name": "PPA_Floor_Grey_Test",
            "monte_carlo": {
                "enabled": False,
                "iterations": 100,
                "sigma_pv_yield_pct": 5.0,
                "sigma_capex_pct": 8.0,
                "sigma_opex_pct": 5.0,
                "sigma_bess_availability_pct": 2.0,
                "price_scenarios": {
                    "mid": {"csv_column": "MID", "weight": 1.0},
                },
            },
            "output": {
                "directory": "output/test/",
                "export_dispatch_sample": False,
            },
        },
        "project_settings": {
            "lifetime_years": 25,
            "discount_rate": 0.06,
            "operating_mode": "grey",
            "location": {
                "latitude": 53.55,
                "longitude": 9.99,
                "pvgis_database": "PVGIS-SARAH2",
            },
            "technology": {
                "pv": {
                    "design": {
                        "peak_power_kwp": 5_000.0,
                        "mounting_type": "free",
                        "azimuth_deg": 0,
                        "tilt_deg": 30,
                    },
                    "performance": {
                        "system_loss_pct": 14.0,
                        "degradation_rate_pct_per_year": 0.4,
                    },
                    "costs": {
                        "capex": {
                            "fixed_eur": 50_000.0,
                            "eur_per_kw": 800.0,
                        },
                        "opex": {
                            "fixed_eur": 5_000.0,
                            "eur_per_kw": 12.0,
                            "pct_of_capex": 0.003,
                        },
                    },
                },
                "bess": {
                    "design_space": {
                        "scale_pct_of_pv": [0, 40],
                        "e_to_p_ratio_hours": [2],
                    },
                    "performance": {
                        "round_trip_efficiency_pct": 90.0,
                        "min_soc_pct": 10.0,
                        "max_soc_pct": 90.0,
                        "degradation_rate_pct_per_year": 2.0,
                        "bess_availability_pct": 97.0,
                    },
                    "costs": {
                        "capex": {
                            "fixed_eur": 50_000.0,
                            "eur_per_kw": 100.0,
                            "eur_per_kwh": 250.0,
                        },
                        "opex": {
                            "fixed_eur": 25_000.0,
                            "pct_of_capex": 0.015,
                        },
                        "replacement": {
                            "enabled": False,
                            "year": 12,
                            "fixed_eur": 0.0,
                            "eur_per_kw": 120.0,
                            "eur_per_kwh": 141.0,
                        },
                    },
                },
                "grid_connection": {
                    "max_export_kw": 4_000.0,
                    "costs": {
                        "capex": {
                            "fixed_eur": 50_000.0,
                            "eur_per_kw": 100.0,
                        },
                        "opex": {
                            "fixed_eur": 25_000.0,
                            "pct_of_capex": 0.015,
                        },
                    },
                },
            },
            "finance": {
                "leverage_pct": 75.0,
                "interest_rate_pct": 4.5,
                "loan_tenor_years": 18,
                "equity_irr_target": None,
                "debt_uses_p90": True,
                "inflation_rate": 0.02,
                "revenue_streams": {
                    "marketing": {
                        "type": "ppa",
                        "floor_price_eur_per_kwh": None,
                        "fixed_price_years": None,
                        "eeg_inflation": False,
                    },
                    "ppa": {
                        "type": "ppa_floor",
                        "pay_as_produced_price_eur_per_kwh": None,
                        "baseload_mw": None,
                        "floor_price_eur_per_kwh": 0.055,
                        "cap_price_eur_per_kwh": None,
                        "duration_years": 15,
                        "inflation_on_ppa": False,
                        "guarantee_of_origin_eur_per_kwh": 0.005,
                    },
                },
                "price_inputs": {
                    "day_ahead_csv": "data/day_ahead_prices.csv",
                    "price_unit": "eur_per_mwh",
                    "inflation_on_input_data": True,
                },
                "tax": {
                    "afa_years_pv": 20,
                    "afa_years_bess": 10,
                    "gewerbesteuer_hebesatz": 400,
                    "gewerbesteuer_messzahl": 0.035,
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Numerical reference fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reference_capex_total() -> float:
    """Total CAPEX for the reference configuration in euros.

    PV (5 000 kWp)            : 50 000 + 800 × 5 000             = 4 050 000 €
    BESS (2 000 kW / 4 000 kWh): 50 000 + 100×2 000 + 250×4 000 = 1 250 000 €
    Grid (4 000 kW)            : 50 000 + 100 × 4 000             =   450 000 €
    Total                                                          = 5 750 000 €
    """
    capex_pv = 50_000.0 + 800.0 * 5_000.0
    capex_bess = 50_000.0 + 100.0 * 2_000.0 + 250.0 * 4_000.0
    capex_grid = 50_000.0 + 100.0 * 4_000.0
    total = capex_pv + capex_bess + capex_grid
    assert math.isclose(capex_pv, 4_050_000.0)
    assert math.isclose(capex_bess, 1_250_000.0)
    assert math.isclose(capex_grid, 450_000.0)
    assert math.isclose(total, 5_750_000.0)
    return total


@pytest.fixture
def reference_annuity(reference_capex_total: float) -> float:
    """Annual annuity payment for a loan covering 75 % of the reference CAPEX.

    Loan    = 5 750 000 € × 75 %  = 4 312 500 €
    Rate    = 4.5 % per annum
    Tenor   = 18 years
    Formula : A = PV × r(1+r)^n / ((1+r)^n - 1)
    Result  ≈ 354 617 € / year  (positive, representing cash outflow)

    numpy_financial.pmt returns a negative value (cash outflow convention);
    this fixture returns the absolute value (positive outflow magnitude).
    """
    loan_amount = reference_capex_total * 0.75  # 4 312 500 €
    rate = 0.045
    nper = 18
    # pmt() returns negative (outflow); we expose the positive magnitude
    annuity = abs(float(npf.pmt(rate, nper, loan_amount)))
    return annuity


@pytest.fixture
def reference_optimizer_4h() -> dict:
    """Input parameters and expected optimal dispatch for a 4-hour LP problem.

    Scenario
    --------
    PV production  : [100, 200,  50,   0] kWh
    Spot prices    : [ 30,  10,  10,  80] €/MWh  →  [0.030, 0.010, 0.010, 0.080] €/kWh
    BESS power     : 100 kW
    BESS capacity  : 200 kWh   (SoC limits 10–90 % → 20–180 kWh usable)
    RTE            : 90 %  (losses on discharge only)
    Start SoC      : 100 kWh
    Grid max export: 150 kW (= 150 kWh per hour)
    Mode           : Green (no grid charging)

    Optimal dispatch (LP-verified)
    ----------------------------------------
    The LP exploits an inter-temporal arbitrage the naive greedy strategy misses:
    discharging at t=0 (moderate price, 30 €/MWh) creates SoC headroom so that
    cheap PV surplus at t=1 can refill the BESS for the expensive t=3 peak.

    t=0  price=30: PV=100.  Export all 100 kWh PV.  Additionally discharge
         500/9 ≈ 55.56 kWh from SoC → grid output = 500/9 × 0.9 = 50 kWh.
         Grid total = 100 + 50 = 150 (limit binding).
         SoC: 100 → 100 − 500/9 = 400/9 ≈ 44.44.
         Revenue: 100×0.030 + 500/9×0.9×0.030 = 3.00 + 1.50 = 4.50 €

    t=1  price=10: PV=200.  Charge 680/9 ≈ 75.56 kWh (exactly enough to reach
         SoC = 120 kWh, enabling full 100 kWh discharge at t=3 down to SoC 20).
         Export 200 − 680/9 = 1120/9 ≈ 124.44 kWh (below grid limit).
         SoC: 400/9 + 680/9 = 1080/9 = 120.
         Revenue: 1120/9 × 0.010 ≈ 1.2444 €

    t=2  price=10: PV=50.  No charging needed (SoC 120 already supports full
         discharge at t=3).  Export 50 kWh.  SoC stays 120.
         Revenue: 50 × 0.010 = 0.50 €

    t=3  price=80: PV=0.  Discharge 100 kWh from SoC (power limited).
         Grid output = 100 × 0.9 = 90 kWh.  SoC: 120 → 20 = min SoC ✓
         Revenue: 100 × 0.9 × 0.080 = 7.20 €

    Total revenue: 4.50 + 1.2444 + 0.50 + 7.20 = 121/9 ≈ 13.4444 €

    Why this beats the naive 12.20 € strategy (no t=0 discharge):
    Discharging 500/9 kWh at t=0 earns 500/9 × 0.9 × 0.030 = 1.50 €.
    Recharging requires 680/9 − 50 = 230/9 ≈ 25.56 kWh of *additional* PV at t=1
    beyond the 50 kWh of free surplus (which would be curtailed anyway).
    Cost = 230/9 × 0.010 ≈ 0.2556 €.  Net arbitrage gain ≈ 1.24 €.
    """
    # Exact fractional values: discharge_t0 = 500/9, charge_t1 = 680/9,
    # export_t1 = 1120/9, total_revenue = 121/9.
    disch_t0 = 500.0 / 9.0
    charge_t1 = 680.0 / 9.0
    export_t1 = 1120.0 / 9.0

    return {
        # Inputs
        "pv_production_kwh": np.array([100.0, 200.0, 50.0, 0.0]),
        "spot_prices_eur_per_mwh": np.array([30.0, 10.0, 10.0, 80.0]),
        "bess_power_kw": 100.0,
        "bess_capacity_kwh": 200.0,
        "rte": 0.90,
        "soc_min_kwh": 20.0,  # 10 % of 200 kWh
        "soc_max_kwh": 180.0,  # 90 % of 200 kWh
        "start_soc_kwh": 100.0,
        "grid_max_kw": 150.0,
        "mode": "green",
        "price_fixed_eur_per_kwh": 0.0,  # no EEG floor active
        # Expected outputs
        "expected_charge_pv_kwh": np.array([0.0, charge_t1, 0.0, 0.0]),
        "expected_export_pv_kwh": np.array([100.0, export_t1, 50.0, 0.0]),
        "expected_curtail_kwh": np.array([0.0, 0.0, 0.0, 0.0]),
        "expected_discharge_green_kwh": np.array([disch_t0, 0.0, 0.0, 100.0]),
        # SoC at end of each hour (after charge/discharge applied)
        "expected_soc_kwh": np.array(
            [100.0 - disch_t0, 120.0, 120.0, 20.0]
        ),
        # Grid export per hour (PV direct + BESS discharge × RTE)
        "expected_grid_export_kwh": np.array(
            [100.0 + disch_t0 * 0.9, export_t1, 50.0, 90.0]
        ),
        "expected_total_revenue_eur": 121.0 / 9.0,  # ≈ 13.4444
    }
