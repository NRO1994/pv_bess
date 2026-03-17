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

INTERVALS_PER_HOUR: int = 4
"""Number of sub-hourly intervals per hour (4 = quarter-hourly resolution)."""

INTERVALS_PER_DAY: int = HOURS_PER_DAY * INTERVALS_PER_HOUR
"""Number of quarter-hourly intervals per day (24 × 4)."""

INTERVALS_PER_YEAR: int = HOURS_PER_YEAR * INTERVALS_PER_HOUR
"""Number of quarter-hourly intervals per non-leap year (365 × 96)."""

TIMESTEP_HOURS: float = 1 / INTERVALS_PER_HOUR
"""Duration of one sub-hourly interval in hours (0.25 = 15 minutes)."""

# ---------------------------------------------------------------------------
# PVGIS API
# ---------------------------------------------------------------------------

PVGIS_API_BASE_URL: str = "https://re.jrc.ec.europa.eu/api/v5_3/"
"""Base URL for the EU PVGIS REST API (version 5.3)."""

PVGIS_CACHE_DIR: str = ".data/pvgis_cache"
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

DEFAULT_MC_SIGMA_CAPEX_PV_PCT: float = 5.0
"""Default standard deviation for PV CAPEX noise factor (% of 1.0)."""

DEFAULT_MC_SIGMA_CAPEX_BESS_PCT: float = 10.0
"""Default standard deviation for BESS CAPEX noise factor (% of 1.0)."""

DEFAULT_MC_SIGMA_OPEX_PV_PCT: float = 3.0
"""Default standard deviation for PV OPEX noise factor (% of 1.0)."""

DEFAULT_MC_SIGMA_OPEX_BESS_PCT: float = 8.0
"""Default standard deviation for BESS OPEX noise factor (% of 1.0)."""

DEFAULT_MC_SIGMA_PV_AVAILABILITY_PCT: float = 2.0
"""Default standard deviation for PV availability noise factor (% of 1.0)."""

DEFAULT_MC_SIGMA_BESS_AVAILABILITY_PCT: float = 2.0
"""Default standard deviation for BESS availability noise factor (% of 1.0)."""

MC_WEIGHT_TOLERANCE: float = 1e-6
"""Tolerance for checking that MC price scenario weights sum to 1.0."""

# ---------------------------------------------------------------------------
# BESS / dispatch defaults
# ---------------------------------------------------------------------------

DEFAULT_DEBT_SIZING_DOWNSIDE_PCT: float = 10.0
"""Default downside percentage for debt sizing (replaces P90-based approach).
Applied as a reduction factor to PV production for conservative DSCR calculation."""

DEFAULT_BESS_AVAILABILITY_PCT: float = 100.0
"""Default BESS availability percentage (100 % = always online)."""

DEFAULT_BESS_REPLACEMENT_CAPACITY_FACTOR_PCT: float = 100.0
"""Default capacity upgrade factor for a mid-life BESS replacement (100 % = same
nameplate capacity as original; values > 100 model a technology-upgrade where
the replacement unit has a larger energy capacity)."""

DEFAULT_OPTIMIZATION_FEE_PCT: float = 0.0
"""Default BESS optimization service fee as percentage of BESS spot revenue."""

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

DEFAULT_COMMISSIONING_YEAR: int = 2027
"""Default commissioning (Inbetriebnahme) calendar year."""

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

MWH_TO_KWH: float = 1000.0
"""Conversion factor from MWh to kWh (multiply MWh value by this)."""

KWH_TO_MWH: float = 1.0 / 1000.0
"""Conversion factor from kWh to MWh (multiply kWh value by this)."""

MIN_PRICE_TIMESERIES_HOURS: int = 8760
"""Minimum required length of a price timeseries (one full year)."""

# ---------------------------------------------------------------------------
# Output defaults
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR: str = ".data/output"
"""Default root directory for scenario result files."""

DISPATCH_SAMPLE_YEAR: int = 1
"""Project year exported to the dispatch sample CSV (1-indexed)."""

_MAX_LOCK_RETRIES: int = 10
"""Maximum number of alternative filenames tried when the target is locked."""

CSV_DELIMITER: str = ";"
"""Delimiter used in all input and output CSV files."""

CSV_DECIMAL_SEPARATOR: str = ","
"""Decimal separator used in all output CSV files (German locale default)."""

CSV_INPUT_DECIMAL_SEPARATOR: str = "."
"""Decimal separator expected in input CSV files (standard market data format)."""

CSV_TIMESTAMP_COLUMN: str = "timestamp"
"""Column name for the timestamp field in price input and dispatch sample CSV files."""

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

DEFAULT_SKIP_BASELINE: bool = False
"""Whether to skip the automatic inclusion of the PV-only baseline (scale=0 %)
in the grid search.  When ``True``, the user takes responsibility for including
the baseline explicitly or accepting that no PV-only comparison point is produced.
Default is ``False`` (baseline always added)."""

# ---------------------------------------------------------------------------
# PPA type identifiers
# ---------------------------------------------------------------------------

PPA_TYPE_NONE: str = "none"
"""PPA type: no PPA active."""

PPA_TYPE_PAY_AS_PRODUCED: str = "ppa_pay_as_produced"
"""PPA type: buyer pays fixed price per kWh produced."""

PPA_TYPE_BASELOAD: str = "ppa_baseload"
"""PPA type: seller commits to flat baseload profile."""

PPA_TYPE_FLOOR: str = "ppa_floor"
"""PPA type: minimum price guaranteed (floor), seller keeps upside."""

PPA_TYPE_COLLAR: str = "ppa_collar"
"""PPA type: floor and cap price boundaries."""

# ---------------------------------------------------------------------------
# Marketing type identifiers
# ---------------------------------------------------------------------------

MARKETING_TYPE_EEG: str = "eeg"
"""Marketing type: EEG feed-in tariff (floor price)."""

MARKETING_TYPE_PPA: str = "ppa"
"""Marketing type: Power Purchase Agreement."""

MARKETING_TYPE_MARKET: str = "market"
"""Marketing type: pure market (spot) pricing."""

# ---------------------------------------------------------------------------
# Additional tax defaults (Germany)
# ---------------------------------------------------------------------------

DEFAULT_KOERPERSCHAFTSTEUER_PCT: float = 15.0
"""Default Körperschaftsteuer rate in percent (§ 23 KStG)."""

DEFAULT_SOLIDARITAETSZUSCHLAG_PCT: float = 5.5
"""Default Solidaritätszuschlag rate in percent (on KSt)."""

# ---------------------------------------------------------------------------
# BESS performance defaults
# ---------------------------------------------------------------------------

DEFAULT_BESS_RTE_PCT: float = 88.0
"""Default round-trip efficiency for BESS in percent."""

DEFAULT_BESS_MIN_SOC_PCT: float = 10.0
"""Default minimum SoC as percentage of capacity."""

DEFAULT_BESS_MAX_SOC_PCT: float = 90.0
"""Default maximum SoC as percentage of capacity."""

DEFAULT_BESS_DEGRADATION_RATE_PCT: float = 2.0
"""Default annual BESS capacity degradation rate in percent."""

# ---------------------------------------------------------------------------
# PV performance defaults
# ---------------------------------------------------------------------------

DEFAULT_PV_DEGRADATION_RATE_PCT: float = 0.4
"""Default annual PV production degradation rate in percent."""

DEFAULT_SYSTEM_LOSS_PCT: float = 14.0
"""Default system loss in percent."""

DEFAULT_PV_AVAILABILITY_PCT: float = 99.0
"""Default availability of PV asset"""

# ---------------------------------------------------------------------------
# Report defaults
# ---------------------------------------------------------------------------

REPORT_CORPORATE_COLORS: list[str] = [
    "#FF8200",
    "#F73E5E",
    "#A51BA7",
    "#00467A",
    "#006EB2",
    "#00BDDC",
]
"""Six-color corporate palette used for charts and report styling."""

REPORT_CHART_DPI: int = 150
"""Resolution (dots per inch) for exported chart PNG files."""

REPORT_CHART_WIDTH_INCHES: float = 10.0
"""Width of chart figures in inches."""

REPORT_CHART_HEIGHT_INCHES: float = 5.5
"""Height of chart figures in inches."""

REPORT_CHARTS_SUBDIR: str = "charts"
"""Sub-directory within the output directory for chart PNG files."""

REPORT_HTML_FILENAME_SUFFIX: str = "_report.html"
"""Suffix appended to the scenario name for the HTML report file."""

REPORT_LLM_PROMPT_FILENAME: str = "_llm_prompt.md"
"""Suffix for the rendered LLM prompt file saved in the output directory."""

REPORT_LLM_RESPONSE_FILENAME: str = "_llm_response.json"
"""Suffix for the LLM response JSON file expected in the output directory."""

REPORT_MODEL_VERSION: str = "1.0"
"""Version string displayed on the report cover page."""

PRICE_DATA_ORIGIN: str = "Prognos 2026"
"""Source attribution for the price input data."""
