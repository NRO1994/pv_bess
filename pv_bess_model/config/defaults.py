"""Global default values and constants.

All numeric constants used throughout the pv_bess_model package must be defined
here rather than as inline literals. Import from this module wherever a constant
is needed to ensure a single source of truth and full traceability.
"""

# ---------------------------------------------------------------------------
# Time constants
# ---------------------------------------------------------------------------

HOURS_PER_YEAR: int = 8760
"""Number of hours in a non-leap year (365 × 24)."""

DAYS_PER_YEAR: int = 365
"""Number of days used per simulation year (leap-day hours are discarded)."""

HOURS_PER_DAY: int = 24
"""Number of hourly timesteps per dispatch day."""

# ---------------------------------------------------------------------------
# PVGIS API
# ---------------------------------------------------------------------------

PVGIS_API_BASE_URL: str = "https://re.jrc.ec.europa.eu/api/v5_3/"
"""Base URL for the EU PVGIS REST API (version 5.3)."""

PVGIS_CACHE_DIR: str = "~/.pv_bess_cache"
"""Local directory for caching raw PVGIS JSON responses."""

PVGIS_SERIESCALC_ENDPOINT: str = "seriescalc"
"""PVGIS endpoint used for hourly historical production data."""

PVGIS_OUTPUT_FORMAT: str = "json"
"""Output format requested from PVGIS API."""

PVGIS_RETRY_MAX: int = 5
"""Maximum number of HTTP retry attempts for PVGIS API calls."""

PVGIS_RETRY_BACKOFF_FACTOR: float = 1.5
"""Exponential backoff factor (seconds) between PVGIS retries."""

PVGIS_REQUEST_TIMEOUT_S: int = 60
"""HTTP request timeout in seconds for PVGIS API calls."""

# ---------------------------------------------------------------------------
# Monte Carlo defaults
# ---------------------------------------------------------------------------

DEFAULT_MC_ITERATIONS: int = 1000
"""Default number of Monte Carlo iterations when not overridden by scenario JSON."""

DEFAULT_MC_SIGMA_PV_YIELD_PCT: float = 5.0
"""Default standard deviation for PV yield noise factor (% of 1.0)."""

DEFAULT_MC_SIGMA_CAPEX_PCT: float = 8.0
"""Default standard deviation for CAPEX noise factor (% of 1.0)."""

DEFAULT_MC_SIGMA_OPEX_PCT: float = 5.0
"""Default standard deviation for OPEX noise factor (% of 1.0)."""

DEFAULT_MC_SIGMA_BESS_AVAILABILITY_PCT: float = 2.0
"""Default standard deviation for BESS availability noise factor (% of 1.0)."""

MC_WEIGHT_TOLERANCE: float = 1e-6
"""Tolerance for checking that MC price scenario weights sum to 1.0."""

# ---------------------------------------------------------------------------
# BESS / dispatch defaults
# ---------------------------------------------------------------------------

DEFAULT_START_SOC_FRACTION: float = 0.50
"""Initial state-of-charge for the first day of simulation (fraction of usable capacity)."""

DEFAULT_BESS_AVAILABILITY_PCT: float = 100.0
"""Default BESS availability percentage (100 % = always online)."""

BESS_NOISE_CLIP_MIN: float = 0.0
"""Minimum clip value for sampled BESS availability noise factor."""

BESS_NOISE_CLIP_MAX: float = 1.0
"""Maximum clip value for sampled BESS availability noise factor."""

# ---------------------------------------------------------------------------
# LP solver
# ---------------------------------------------------------------------------

LP_SOLVER_METHOD: str = "highs"
"""scipy.optimize.linprog method selecting the HiGHS backend."""

LP_INFEASIBILITY_TOLERANCE: float = 1e-6
"""Tolerance below which an LP solution is considered feasible."""

# ---------------------------------------------------------------------------
# Financial defaults
# ---------------------------------------------------------------------------

DEFAULT_DISCOUNT_RATE: float = 0.06
"""Default equity discount rate (6 %) used for NPV calculation."""

DEFAULT_INFLATION_RATE: float = 0.02
"""Default annual inflation rate (2 %) applied to OPEX and optionally to prices."""

DEFAULT_LEVERAGE_PCT: float = 75.0
"""Default debt leverage as a percentage of total CAPEX."""

DEFAULT_INTEREST_RATE_PCT: float = 4.5
"""Default annual loan interest rate (4.5 %)."""

DEFAULT_LOAN_TENOR_YEARS: int = 18
"""Default loan tenor in years."""

DEFAULT_LIFETIME_YEARS: int = 25
"""Default project lifetime in years."""

IRR_MAX_ITERATIONS: int = 1000
"""Maximum iterations for IRR Newton-Raphson convergence (numpy_financial internal)."""

IRR_CONVERGENCE_TOLERANCE: float = 1e-7
"""Convergence tolerance for IRR calculation."""

DSCR_MINIMUM_THRESHOLD: float = 1.0
"""DSCR level below which a warning is emitted (debt cannot be serviced)."""

# ---------------------------------------------------------------------------
# Tax defaults (Germany)
# ---------------------------------------------------------------------------

DEFAULT_AFA_YEARS_PV: int = 20
"""Default linear depreciation period for PV assets (years)."""

DEFAULT_AFA_YEARS_BESS: int = 10
"""Default linear depreciation period for BESS assets (years)."""

DEFAULT_GEWERBESTEUER_HEBESATZ: int = 400
"""Default municipal trade-tax multiplier (Hebesatz) in percent."""

DEFAULT_GEWERBESTEUER_MESSZAHL: float = 0.035
"""Statutory trade-tax base rate (Messzahl) per § 11 GewStG."""

# ---------------------------------------------------------------------------
# Price / market defaults
# ---------------------------------------------------------------------------

PRICE_UNIT_EUR_PER_MWH: str = "eur_per_mwh"
"""Identifier for electricity prices denominated in €/MWh."""

PRICE_UNIT_EUR_PER_KWH: str = "eur_per_kwh"
"""Identifier for electricity prices denominated in €/kWh."""

MWH_TO_KWH: float = 1000.0
"""Conversion factor from MWh to kWh (multiply MWh value by this)."""

KWH_TO_MWH: float = 1.0 / 1000.0
"""Conversion factor from kWh to MWh (multiply kWh value by this)."""

MIN_PRICE_TIMESERIES_HOURS: int = 8760
"""Minimum required length of a price timeseries (one full year)."""

# ---------------------------------------------------------------------------
# Output defaults
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR: str = "output"
"""Default root directory for scenario result files."""

DISPATCH_SAMPLE_YEAR: int = 1
"""Project year exported to the dispatch sample CSV (1-indexed)."""

CSV_DELIMITER: str = ","
"""Delimiter used in all input and output CSV files."""

CSV_TIMESTAMP_FORMAT: str = "%Y-%m-%dT%H:%M:%S"
"""ISO 8601 timestamp format used in CSV files."""

FLOAT_PRECISION: int = 4
"""Number of decimal places for floating-point values in output CSVs."""

CURRENCY_PRECISION: int = 2
"""Number of decimal places for monetary values in output CSVs."""

# ---------------------------------------------------------------------------
# Grid search defaults
# ---------------------------------------------------------------------------

GRID_SEARCH_SCALE_ZERO_PCT: float = 0.0
"""Scale percentage representing the PV-only baseline (no BESS)."""
