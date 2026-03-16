# CLAUDE.md – PV + BESS Co-Location Financial Model

## Project Overview

A command-line Python tool for evaluating co-location of photovoltaic (PV) systems with battery energy storage systems (
BESS). The tool answers: What is the optimal BESS sizing relative to a PV plant? Which grid connection and marketing
strategies maximize equity returns? How do PPA structures compare to EEG feed-in tariffs?

The tool runs a **ratio-based grid search** over BESS sizing (as percentage of PV peak power, with configurable
energy-to-power ratios), executing a full multi-year financial model for each combination within a user-defined
scenario. The optimal configuration (max Equity IRR) is then subjected to Monte Carlo simulation for risk analysis.
Post-grid-search **sensitivity analyses** (EEG price sweep, PPA Collar sweep, PPA Baseload sweep) can be run on the
optimal configuration.

**User**: Single user (developer), CLI-only, with interactive HTML dashboard report for stakeholders.

---

## Architecture

### Module Structure

```
pv_bess_model/
├── main.py                     # CLI entrypoint, orchestrator (~1,300 lines)
├── config/
│   ├── schema.py               # JSON schema validation for scenario files
│   ├── loader.py               # Load & validate scenario JSON + CSV price files
│   │                           # Includes PriceWeatherScenario, PriceData dataclasses
│   └── defaults.py             # Global defaults and constants (~340 constants, NO magic numbers in code)
├── pv/
│   ├── pvgis_client.py         # PVGIS API client – fetch hourly historical data
│   ├── timeseries.py           # P50/P90 calculation from historical years (supports 35,040 intervals)
│   └── degradation.py          # Annual linear degradation applied to timeseries
├── bess/
│   └── replacement.py          # Mid-life BESS replacement config, cost calc, capacity upgrade
│   # NOTE: No battery.py – BESS state is managed implicitly by the LP optimizer
├── dispatch/
│   ├── engine.py               # Dispatch engine (yearly simulation loop, ~700 lines)
│   │                           # Supports configurable timestep (hourly or quarter-hourly)
│   └── optimizer.py            # Daily LP-based dispatch optimization (~1,000 lines)
│                               # Green/Grey mode, 24 or 96 timesteps per day
├── market/
│   ├── price_loader.py         # Load CSV price timeseries, extend to project lifetime
│   ├── eeg.py                  # EEG floor tariff logic
│   └── ppa.py                  # PPA models (pay-as-produced, baseload, floor, collar)
├── finance/
│   ├── cashflow.py             # Annual cashflow projection (revenue, OPEX, debt, equity)
│   ├── costs.py                # Unified CAPEX/OPEX calculation from scenario JSON
│   ├── debt.py                 # Annuity loan model + replacement debt
│   ├── tax.py                  # Tax treatment (AfA, GewSt, KSt, Soli) with Verlustvortrag
│   ├── metrics.py              # IRR, NPV, DSCR, LCOE, payback, capture rate
│   └── inflation.py            # Inflation escalation logic
├── optimization/
│   ├── grid_search.py          # Ratio-based grid search over BESS sizing
│   ├── monte_carlo.py          # MC simulation on optimal configuration (~1,000 lines)
│   │                           # Dispatch-once-per-scenario approach (~1000x speedup)
│   └── analyses.py             # Post-grid-search sensitivity analyses
│                               # (EEG sweep, PPA Collar 2D, PPA Baseload 2D)
├── output/
│   ├── csv_writer.py           # Write summary, cashflows, grid search, dispatch, analysis CSVs
│   ├── formatting.py           # Number/currency formatting helpers
│   └── report/
│       ├── data_collector.py   # HtmlReportData aggregation
│       ├── html_builder.py     # Template → HTML injection
│       ├── llm_prompt.py       # LLM prompt template rendering + response parsing
│       ├── charts.py           # Canvas-based chart rendering (JavaScript)
│       └── templates/
│           └── dashboard.html  # Single-file, offline-capable HTML template (~62 KB)
├── tests/
│   ├── conftest.py             # Shared fixtures, sample data
│   ├── dispatch_constraint_checker.py  # LP validation helper
│   └── test_*.py               # ~45 test files
└── scenarios/                  # Scenario JSON files
```

### Key Design Principles

1. **No magic numbers**: Every numeric value comes from the scenario JSON or from `config/defaults.py`. Constants in
   `config/defaults.py` must have descriptive names and docstrings.
2. **One scenario = one JSON file**: To compare scenarios, the user creates multiple JSON files and runs the tool
   separately. Comparison is done by the user via the output CSVs and the HTML dashboard.
3. **Scenario-driven determinism**: Grid search runs on the **central price-weather scenario** (marked
   `is_central: true`). MC samples across all price-weather scenarios with noise factors applied post-hoc.
4. **Quarter-hourly resolution**: The dispatch engine operates at **15-minute intervals** (35,040 steps per year,
   configurable). PVGIS data is fetched hourly and interpolated to quarter-hourly resolution.
5. **Immutable operating mode**: The BESS operating mode (green/grey) is fixed per scenario and cannot change during
   simulation.
6. **Price-weather scenario coupling**: Each price scenario is paired with a specific historical weather year, enabling
   realistic correlation between PV yield and electricity prices.

---

## Module Specifications

### 1. PV Module (`pv/`)

#### PVGIS API Client (`pvgis_client.py`)

- Fetch **hourly** historical radiation/production data from the EU PVGIS API (https://re.jrc.ec.europa.eu/api/v5_3/)
- Use the `seriescalc` endpoint with `outputformat=json`
- Parameters from scenario JSON: latitude, longitude, PV peak power (kWp), mounting type, azimuth, tilt
- **System loss is set to 0% at PVGIS download** – losses are applied separately at the grid connection point
- Fetch specific historical years as defined by price-weather scenario mapping (e.g., 2015, 2016, 2018)
- Handle API rate limits gracefully (retry with exponential backoff, max 5 retries)
- Cache downloaded data locally to avoid redundant API calls (cache in `.data/pvgis_cache/` relative to project root)

#### Timeseries Processing (`timeseries.py`)

- Input: Dictionary of {year: hourly_production_array} from PVGIS
- Supports both hourly (8,760) and quarter-hourly (35,040) interval arrays
- For each interval index across all historical years: compute P50 (median) and P90 (10th percentile)
- **Leap year handling**: Ignore December 31st for leap years, truncate all years to 8,760 hours / 35,040 intervals
- **Weekday alignment**: Weather year can be shifted so start day matches commissioning year (e.g., 2017 → 2030)
- Output: Per-scenario timeseries arrays (one per price-weather scenario)

#### Degradation (`degradation.py`)

- Apply linear annual degradation rate (user input, e.g., 0.4%/year) to the base timeseries
- For year Y of project: `production[Y] = base_production * (1 - degradation_rate) ^ Y`
- Return degraded timeseries for each project year

### 2. BESS Module (`bess/`)

#### BESS State Management

- **No separate `battery.py` class** – BESS state is managed implicitly within the LP optimizer
- SoC tracking is handled by LP decision variables in `optimizer.py`
- Annual degradation is computed in `engine.py` and passed to the optimizer as reduced capacity
- Parameters from scenario JSON: max capacity (kWh), max charge/discharge power (kW), round-trip efficiency (%), min
  SoC (%), max SoC (%)
- Efficiency model: **Losses are applied only on discharge.** Charging is lossless (1 kWh in = 1 kWh SoC increase).
  Discharge output = discharged kWh × round-trip efficiency.

#### Replacement (`replacement.py`)

- Optional: user specifies replacement year and cost in scenario JSON
- At replacement year: reset BESS capacity (optionally with capacity upgrade via `capacity_factor_pct`), add replacement
  cost as **CAPEX** in that year
- **Capacity upgrade**: `capacity_factor_pct` (default 100%) allows modeling technology improvements (e.g., 120% =
  replacement unit has 20% more capacity)
- Replacement cost follows unified cost schema (`fixed_eur`, `eur_per_kw`, `eur_per_kwh`)
- **New depreciation line** starts at replacement year for the replacement cost over `afa_years_bess`
- If no replacement specified: BESS degrades continuously over project lifetime

### 3. Dispatch Module (`dispatch/`)

#### Dispatch Engine (`engine.py`)

- Core simulation loop: iterate over **365 days per year**
- **Configurable timestep resolution** via `DispatchEngineConfig`:
    - `timestep_hours`: 1.0 (hourly) or 0.25 (quarter-hourly, default)
    - `intervals_per_day`: 24 or 96
    - `intervals_per_year`: 8,760 or 35,040
- **Grid loss factor**: Applied to green energy at grid connection point (
  `grid_loss_factor = 1 - system_loss_pct / 100`)
- For each project year:
    1. Get degraded PV timeseries for this year
    2. Get degraded BESS capacity for this year
    3. Get price timeseries for this year (with inflation if enabled)
    4. Determine pricing regime for this year (fixed-price phase or market phase)
    5. Determine BESS and PV offline days for this year
    6. For each day (d = 0..364):
       a. Extract `pv_production[d*N : (d+1)*N]` (N = intervals_per_day)
       b. Extract `prices[d*N : (d+1)*N]`
       c. If BESS is offline this day: skip LP, dispatch PV directly (`export = min(pv, P_grid_max)`, rest curtailed)
       d. If BESS is online: call daily LP optimizer with start SoC from previous day
       e. Record interval results into yearly arrays
       f. Carry over end-SoC to next day
    7. Aggregate annual results: total revenue, costs, energy flows
- Track all energy flows per interval for dispatch sample export and per year for cashflow

#### Daily LP Optimizer (`optimizer.py`)

The optimizer solves a linear program for each day (24 or 96 timesteps) to determine optimal dispatch decisions under
perfect foresight of day-ahead prices.

##### Decision Variables (per timestep t)

**Green Mode:**

- `charge_pv[t]` – kWh charged from PV surplus into BESS
- `discharge_green[t]` – kWh discharged from BESS (green energy)
- `export_pv[t]` – kWh PV directly exported to grid
- `curtail[t]` – kWh PV curtailed

**Grey Mode (additional variables):**

- `charge_grid[t]` – kWh charged from grid into BESS (at spot price)
- `discharge_grey[t]` – kWh discharged from BESS (grey energy)
- `soc_green[t]` – SoC tracking for green kWh in BESS
- `soc_grey[t]` – SoC tracking for grey kWh in BESS

##### Effective Price Pre-Computation

During the fixed-price PPA period, the effective price per kWh is `max(price_spot[t], price_fixed) + goo_premium`. Since
both spot prices and the floor price are known constants at solve time, the effective price is **pre-computed** before
the LP is built.

During the fixed-price EEG period, the effective price per kWh is `max(price_spot[t], price_fixed)`. Pre-computed like
in the PPA case, but without the goo_premium.

For PPA Collar: `clip(price_spot[t], floor_price, cap_price) + goo_premium`

```
PPA Floor case:
effective_green_price[t] = max(price_spot[t], price_fixed) + goo_premium

PPA Collar case:
effective_green_price[t] = clip(price_spot[t], floor_price, cap_price) + goo_premium

EEG case:
effective_green_price[t] = max(price_spot[t], price_fixed)
```

After the fixed-price period expires, `price_fixed` is set to 0, and the effective price equals
`price_spot[t] + goo_premium`.

##### Objective Function

**Green Mode – maximize daily revenue:**

```
max Σ_t [ export_pv[t] × effective_green_price[t] × grid_loss_factor
        + discharge_green[t] × RTE × effective_green_price[t] ]
```

**Grey Mode – maximize daily net revenue:**

```
max Σ_t [ export_pv[t] × effective_green_price[t] × grid_loss_factor
        + discharge_green[t] × RTE × effective_green_price[t]
        + discharge_grey[t] × RTE × price_spot[t]
        - charge_grid[t] × price_spot[t] ]
```

##### Constraints

**Energy balance PV (all modes):**

```
export_pv[t] + charge_pv[t] + curtail[t] = pv_production[t]   ∀t
```

**SoC tracking – Green Mode (single SoC track):**

```
soc[t+1] = soc[t] + charge_pv[t] - discharge_green[t]   ∀t
soc_min ≤ soc[t] ≤ soc_max                                ∀t
```

**SoC tracking – Grey Mode (dual chamber):**

```
soc_green[t+1] = soc_green[t] + charge_pv[t] - discharge_green[t]            ∀t
soc_grey[t+1]  = soc_grey[t]  + charge_grid[t] - discharge_grey[t]           ∀t
soc_green[t] + soc_grey[t] ≥ soc_min                                         ∀t
soc_green[t] + soc_grey[t] ≤ soc_max                                         ∀t
soc_green[t] ≥ 0                                                             ∀t
soc_grey[t] ≥ 0                                                              ∀t
```

**Power limits:**

```
charge_pv[t] + charge_grid[t] ≤ P_max_charge                                 ∀t
discharge_green[t] + discharge_grey[t] ≤ P_max_discharge                     ∀t
```

(In Green Mode: `charge_grid[t] = 0` and `discharge_grey[t] = 0` for all t)

**Grid connection limit (with grid loss factor):**

```
export_pv[t] × grid_loss_factor + (discharge_green[t] + discharge_grey[t]) × RTE ≤ P_grid_max   ∀t
```

**Non-negativity:**

```
All decision variables ≥ 0
```

##### SoC Day-to-Day Coupling

- Day 1 of the project: `soc[0] = soc_max / 2` (50% filled BESS)
- All subsequent days: `soc[0] = soc_end_previous_day` (end-SoC of previous day becomes start-SoC)
- No end-of-day SoC constraint – the optimizer is free to hold or empty the BESS across the day boundary
- In Grey Mode: `soc_green[0]` and `soc_grey[0]` are each carried over from the previous day

##### Implementation Details

- Solver: `scipy.optimize.linprog` with HiGHS backend
- Problem size per day (quarter-hourly):
    - Green Mode: ~384 variables (96 × 4 decision), ~480 constraints
    - Grey Mode: ~768 variables (96 × 8 decision), ~960 constraints
- Expected solve time: ~2-4ms per day with HiGHS (quarter-hourly)
- Total per year: 365 solves × ~3ms = ~1s per year
- The optimizer returns per-timestep dispatch decisions: `charge_pv[t]`, `charge_grid[t]`, `discharge_green[t]`,
  `discharge_grey[t]`, `export_pv[t]`, `curtail[t]`

##### BESS Availability (Offline Days)

BESS availability is modelled as whole-day outages (maintenance, faults). This applies in both grid search and Monte
Carlo, but with different logic:

**In Grid Search (deterministic):**

- Use the `bess_availability_pct` from the scenario JSON directly
- `n_offline_days = round((1 - bess_availability_pct / 100) × 365)`
- Offline days are distributed evenly across the year (every Nth day)

**In Monte Carlo (stochastic, post-hoc):**

- BESS availability factor is sampled and applied as a revenue scaling factor (not as actual offline days in dispatch)
- Scales BESS-related revenues (green + grey discharge) and grid import costs

**On offline days (grid search):**

- All BESS variables in the LP are fixed to 0 (charge = 0, discharge = 0)
- The LP reduces to pure PV dispatch: `export_pv[t] = min(pv_production[t], P_grid_max)`, remainder is curtailed
- SoC is frozen at the value from the end of the last online day

##### PV Availability

PV availability is modelled similarly to BESS availability:

- `pv_availability_pct` in scenario JSON (default: 99%)
- In grid search: deterministic offline days for PV (production = 0 on offline days)
- In MC: PV availability factor scales PV-related revenues post-hoc

### 4. Market Module (`market/`)

#### Price Loader (`price_loader.py`)

- Load CSV file with electricity price timeseries
- Expected format: prices in €/MWh (or €/kWh, configurable via `price_unit`)
- Supports both hourly and quarter-hourly resolution
- If project lifetime exceeds price timeseries length: repeat the **last full year** of the timeseries until project end
- Validate: timeseries must cover at least one full year
- Configurable CSV dialect per scenario: separator, decimal character, timestamp column, timestamp format

#### Price-Weather Scenarios

Each price scenario is defined as a `PriceWeatherScenario` object coupling price data with PV weather:

```json
"scenarios": [
{
"name": "Central",
"label": "Zentralszenario",
"csv_column": "price_central",
"weather_year": 2018,
"weight": 0.6,
"is_central": true,
"price_csv": "./inputs/day_ahead_prices.csv",
"inflation_on_input_data": false,
"csv_separator": ";",
"csv_decimal": ",",
"csv_timestamp_column": "timestamp",
"csv_timestamp_format": "%Y-%m-%dT%H:%M:%S"
}
]
```

- One scenario must be marked `is_central: true` – used for deterministic grid search
- All scenarios are used for MC sampling (weighted by `weight`, must sum to 1.0)
- Each scenario has its own `weather_year` for PV timeseries and `csv_column` for price data

#### EEG Module (`eeg.py`)

- EEG tariff acts as a **floor price (Mindestpreis)**, not a fixed price
- Effective price per kWh: `max(price_spot[t], price_eeg)`
- The floor applies for the first X years (both tariff level and duration from user input)
- After X years: pure market price (floor drops away)
- Inflation adjustment: optional, controlled by user flag (`eeg_inflation: true/false`)
- If inflation enabled: `eeg_price[year] = base_eeg_price * (1 + inflation_rate) ^ year`

#### PPA Module (`ppa.py`)

Four PPA structures, selectable per scenario:

1. **Pay-as-produced** (`ppa_pay_as_produced`)
    - Buyer pays fixed price per kWh actually produced
    - Price from user input, optional inflation escalation

2. **Baseload PPA** (`ppa_baseload`)
    - Seller commits to deliver a flat power profile (baseload MW)
    - Baseload level: explicit user input (`baseload_mw` required, no auto-calculation)
    - Profile cost: When PV < baseload → seller buys shortfall at market price. When PV > baseload → seller sells excess
      at market price
    - Net revenue = baseload_volume × ppa_price + excess_revenue - shortfall_cost
    - BESS can help shape the profile (reduce shortfall, shift excess)

3. **Floor PPA** (`ppa_floor`)
    - Minimum price guaranteed (floor), seller keeps upside above floor
    - `revenue_per_kwh = max(price_spot[t], ppa_price) + goo_premium`
    - Floor price from user input

4. **Collar PPA** (`ppa_collar`)
    - Floor price and cap price as boundaries
    - Revenue per kWh = `clip(market_price, floor_price, cap_price) + goo_premium`
    - Both prices from user input

All PPA models:

- Duration (years) from user input
- After PPA expires: switch to pure market price
- Inflation escalation: optional per user flag, effective on all price related data points (floor, cap, ppa_price)
- Guarantee of origin (GoO) premium added **after** the floor/clip operation for all PPA structures (user-defined €/kWh)

### 5. Finance Module (`finance/`)

#### Unified Cost Calculation (`costs.py`)

All CAPEX and OPEX fields follow a single, consistent schema. Each cost block (PV, BESS, Grid, Other) supports three
additive components. If a parameter is not set in the JSON, it is treated as 0 in the addition.

**CAPEX per asset:**

```
CAPEX_asset = fixed_eur
            + eur_per_kw × reference_kW
            + eur_per_kwh × reference_kWh
```

Reference sizes per asset:
| Asset | reference_kW | reference_kWh |
|-------|-------------|---------------|
| PV | peak_power_kwp | (not applicable, ignored if set) |
| BESS | bess_power_kw | bess_capacity_kwh |
| Grid | max_export_kw | (not applicable, ignored if set) |

**Total CAPEX:**

```
CAPEX_total = CAPEX_pv + CAPEX_bess + CAPEX_grid + CAPEX_other
```

**OPEX per asset (annual):**

```
OPEX_asset = fixed_eur
           + eur_per_kw × reference_kW
           + eur_per_kwh × reference_kWh
           + pct_of_capex × CAPEX_asset
```

Note: `pct_of_capex` refers to the CAPEX of the **same asset**, not the total project CAPEX.

**BESS Optimization Fee (additional OPEX component):**

```
optimization_fee = BESS_spot_revenue × optimization_fee_pct / 100
```

- Applied to BESS discharge revenue (spot-based, not effective price)
- NOT inflation-adjusted (already embedded in annual revenue)

**Total annual OPEX:**

```
OPEX_total = OPEX_pv + OPEX_bess + OPEX_grid + OPEX_other + optimization_fee
```

All OPEX (except optimization fee) is subject to inflation escalation over project lifetime.

BESS replacement cost follows the same three-component schema (`fixed_eur`, `eur_per_kw`, `eur_per_kwh`). Replacement
cost is treated as **CAPEX** in the replacement year (not OPEX), with a new depreciation line.

#### Cashflow Projection (`cashflow.py`)

- Build annual cashflow table for full project lifetime
- Revenue streams (from dispatch simulation):
    - PV direct feed-in revenue (EEG floor / PPA / market depending on year and config)
    - BESS discharge revenue green (EEG floor / PPA / market)
    - BESS discharge revenue grey (market price, Grey Mode only)
    - Minus: grid import costs (Grey Mode only)
- CAPEX: Year 1 (commissioning year, same year as first revenue and OPEX)
- BESS replacement CAPEX: In replacement year (if enabled)
- OPEX: Annual, inflated per year (inflation starts in year 2)
- Cashflow per year: Revenue - OPEX - Debt Service - Tax - Replacement CAPEX = Equity Cashflow

#### Debt Module (`debt.py`)

- Simple annuity loan
- Parameters: loan amount (% of total CAPEX), interest rate, loan tenor (years)
- Calculate annual annuity payment (constant over tenor)
- Separate interest and principal components per year
- Loan amount calculated from total CAPEX × leverage ratio
- **Debt sizing downside**: Uses `debt_sizing_downside_pct` to reduce revenue for conservative DSCR calculation (e.g.,
  10% downside = revenue × 0.9 for DSCR)
- DSCR per year = (Revenue × (1 - downside_pct/100) - OPEX) / Debt Service
- **Replacement debt**: Optional separate debt for mid-life BESS replacement (`replacement_leverage_pct`,
  `replacement_interest_rate`, `replacement_loan_tenor_years`)

#### Tax Module (`tax.py`)

- German tax treatment with four components:
    - **Linear depreciation (AfA)**: Separate depreciation periods for PV and BESS (user-defined, e.g., 20 years for PV,
      10 years for BESS). Depreciation base = CAPEX of respective asset. **BESS replacement** starts a new depreciation
      line from replacement year.
    - **Gewerbesteuer (GewSt)**: `GewSt = max(0, taxable_income) × Messzahl × Hebesatz / 100` where
      `taxable_income = Revenue - OPEX - AfA + Verlustvortrag_adjustment`
    - **Körperschaftsteuer (KSt)**: `KSt = max(0, taxable_income) × koerperschaftsteuer_pct / 100` (default: 15%)
    - **Solidaritätszuschlag (Soli)**: `Soli = KSt × solidaritaetszuschlag_pct / 100` (default: 5.5%)
- **Total tax per year**: `GewSt + KSt + Soli`
- **Verlustvortrag (loss carry-forward)**: If taxable income is negative in a year, the loss is carried forward
  indefinitely. Carried-forward losses offset future positive taxable income before tax is calculated.
- Tax reduces equity cashflow

#### Metrics (`metrics.py`)

- **Project IRR**: IRR on total project cashflows (pre-leverage)
- **Equity IRR**: IRR on equity cashflows (post-leverage, post-tax)
- **NPV**: At user-defined discount rate
- **DSCR**: Min DSCR and average DSCR over loan tenor, plus annual DSCR array
- **LCOE**: Levelized cost of energy (total costs / total production)
- **Payback period**: Year when cumulative equity cashflow turns positive
- **Capture Rate**: Average revenue per kWh of energy fed into the grid (€/kWh)
- Use `numpy_financial` for IRR/NPV calculation

#### Inflation (`inflation.py`)

- Single inflation rate from user input
- Applied to: OPEX (always), PPA price (if flag set), EEG price (if flag set), Day-Ahead price from CSV (if flag set per
  scenario)
- `inflated_value[year] = base_value * (1 + inflation_rate) ^ year`

### 6. Optimization Module (`optimization/`)

#### Grid Search (`grid_search.py`)

The grid search uses a **ratio-based parametrization** to efficiently explore BESS sizing. Instead of independently
varying power and capacity, the search space is defined by two user-configurable dimensions:

1. **BESS scale** (% of PV peak power): How large is the BESS relative to the PV plant?
2. **Energy-to-power ratio** (hours): What is the storage duration?

Both dimensions are specified as lists in the scenario JSON. The grid search evaluates all combinations.

**Deriving BESS power and capacity from ratios:**

```
BESS_power_kW   = pv_peak_kwp × scale_pct / 100
BESS_capacity_kWh = BESS_power_kW × e_to_p_ratio_hours
```

Example: PV = 5,000 kWp, scale = 40%, E/P = 2h → BESS = 2,000 kW / 4,000 kWh

**PV-only baseline**: By default, scale = 0% (PV-only) is automatically included. Can be skipped via
`skip_baseline: true` in scenario JSON.

**For each (scale, E/P ratio) combination:**

1. Derive BESS power and capacity from ratios
2. Calculate CAPEX using unified cost schema (PV + BESS + Grid + Other)
3. Calculate annual base OPEX using unified cost schema
4. **Run full multi-year dispatch** using the **central price-weather scenario**:
    - For each project year: apply PV degradation, BESS degradation, OPEX inflation, price evolution
    - Run 365 daily LP optimizations
    - BESS and PV offline days applied deterministically (see Availability)
5. Build complete cashflow projection (year-varying revenue, OPEX, debt, tax)
6. Calculate Equity IRR

**Output:** 2D matrix of Equity IRR indexed by (scale_pct, e_to_p_ratio). Identify optimum = max Equity IRR. Grid search
stores per-asset OPEX breakdowns (opex_pv, opex_bess, opex_grid, opex_other) for MC.

**Performance:** With typical inputs (e.g., 8 scale steps × 2 E/P ratios = 16 combinations, 25 years each), the grid
search requires 16 × 25 × 365 = ~146K LP solves. At ~3ms per solve (quarter-hourly): ~7 minutes total. Parallelizable
across combinations with `concurrent.futures`.

#### Monte Carlo (`monte_carlo.py`)

The MC simulation runs on the optimal (scale, E/P ratio) from grid search. It uses a **dispatch-once-per-scenario**
approach for ~1000x speedup over naive per-iteration dispatch.

##### Approach

1. **Dispatch phase**: Run full multi-year dispatch **once per price-weather scenario** with 100% PV and BESS
   availability. Results are parallelized across scenarios.
2. **MC sampling phase**: For each iteration, sample noise factors and apply them **post-hoc** to the pre-computed
   financial results. This runs in the main thread and is extremely fast.

##### Noise Factors (separate for PV and BESS)

| Factor                        | Distribution                       | Effect                       |
|-------------------------------|------------------------------------|------------------------------|
| `sigma_capex_pv_pct`          | N(1.0, σ)                          | Scales PV CAPEX              |
| `sigma_capex_bess_pct`        | N(1.0, σ)                          | Scales BESS CAPEX            |
| `sigma_opex_pv_pct`           | N(1.0, σ)                          | Scales PV OPEX               |
| `sigma_opex_bess_pct`         | N(1.0, σ)                          | Scales BESS OPEX             |
| `sigma_pv_availability_pct`   | N(1.0, σ), clipped [0,1]           | Scales PV-related revenues   |
| `sigma_bess_availability_pct` | N(μ_avail, σ), clipped [μ_avail,1] | Scales BESS-related revenues |

##### MC Iteration Logic

For each MC iteration:

1. **Sample price-weather scenario** according to weights
2. **Sample noise factors** from normal distributions (independent draws, no correlations)
3. **Apply factors post-hoc** to pre-computed dispatch results:
    - PV availability factor → scales PV export revenue + BESS green discharge revenue
    - BESS availability factor → scales BESS discharge revenues + grid import costs
    - CAPEX factors → scale asset-level CAPEX
    - OPEX factors → scale asset-level OPEX
4. Build complete cashflow projection, calculate metrics
5. Record: Equity IRR, Project IRR, NPV, Min DSCR, Capture Rate, all noise factors

##### MC Output

- Distribution statistics (mean, median, P10, P25, P50, P75, P90, std) for each metric
- Statistics broken down by price scenario (conditional distributions)
- Number of iterations: user input (default: 1,000)

#### Post-Grid-Search Sensitivity Analyses (`analyses.py`)

Three analysis types run after grid search on the optimal BESS configuration:

**1. EEG Sensitivity (`eeg_sensitivity`)**

- Sweep over user-defined EEG floor prices (e.g., [0.05, 0.06, ..., 0.09] €/kWh)
- For each price: run full MC simulation
- Output: mean/std/P10/P50/P90 IRR per floor price

**2. PPA Collar 2D Sweep (`ppa_collar`)**

- Sweep: floor_price × cap_spread (both user-defined lists)
- `cap_price = floor_price + cap_spread` for each combination
- Full MC for each combination
- Output: 2D results matrix

**3. PPA Baseload 2D Sweep (`ppa_baseload`)**

- Sweep: ppa_price × baseload_mw (both user-defined lists)
- Full MC for each combination
- Output: 2D results matrix

All analyses use the same MC framework (dispatch-once-per-scenario + post-hoc noise).

### 7. Output Module (`output/`)

#### CSV Writer (`csv_writer.py`)

Produce the following CSV files per scenario run:

1. **`{scenario_name}_summary.csv`**
    - One row with all key results:
        - Input parameters (scenario name, PV size, optimal BESS scale %, optimal E/P ratio, optimal BESS power, optimal
          BESS capacity, operating mode, marketing model, PPA type, fixed price, fixed price years, project lifetime,
          etc.)
        - Financial results (Equity IRR, Project IRR, NPV, Min DSCR, Avg DSCR, LCOE, Payback period, Capture Rate)
        - Total production (MWh lifetime), total revenue, total CAPEX, total OPEX

2. **`{scenario_name}_cashflows.csv`**
    - One row per project year (Year column shows calendar year starting from commissioning_year)
    - Columns: Year, PV Production (MWh), BESS Throughput (MWh), Revenue PV (€), Revenue BESS Green (€), Revenue BESS
      Grey (€), Grid Import Cost (€), Total Revenue (€), CAPEX (€), OPEX (€), Debt Service (€), Gewerbesteuer (€),
      Körperschaftsteuer (€), Solidaritätszuschlag (€), Depreciation (€), Equity CF (€), Cumulative Equity CF (€), DSCR

3. **`{scenario_name}_grid_search.csv`**
    - One row per (scale, E/P ratio) combination
    - Columns: Scale Pct of PV (%), E/P Ratio (h), BESS Power (kW), BESS Capacity (kWh), Total CAPEX (€), Total OPEX (
      €), Sum of revenue year 1 (€), Equity IRR (%), Project IRR (%), NPV (€), is_optimal (boolean)

4. **`{scenario_name}_monte_carlo.csv`** (only if MC is enabled)
    - One row per MC iteration
    - Columns: Iteration, Price Scenario, PV Availability Factor, BESS Availability Factor, CAPEX PV Factor, CAPEX BESS
      Factor, OPEX PV Factor, OPEX BESS Factor, Equity IRR (%), Project IRR (%), NPV (€), Min DSCR, Capture Rate

5. **`{scenario_name}_dispatch_sample.csv`** (optimal configuration, first year)
    - One row per interval (35,040 rows for quarter-hourly)
    - Columns: Timestamp, PV Production (kWh), Price Spot (€/MWh), Price Effective (€/MWh), BESS SoC (kWh), BESS SoC
      Green (kWh), BESS SoC Grey (kWh), BESS Charge PV (kWh), BESS Charge Grid (kWh), BESS Discharge Green (kWh), BESS
      Discharge Grey (kWh), Grid Export (kWh), Curtailed (kWh), Revenue (€)

6. **`{scenario_name}_analyses_eeg_sensitivity.csv`** (if EEG sensitivity enabled)
    - Floor price, IRR statistics per price point

7. **`{scenario_name}_analyses_ppa_collar.csv`** (if PPA Collar analysis enabled)
    - Floor price, cap spread, cap price, IRR statistics per combination

8. **`{scenario_name}_analyses_ppa_baseload.csv`** (if PPA Baseload analysis enabled)
    - PPA price, baseload MW, IRR statistics per combination

All output files go to a user-specified output directory (default: `.data/output/{scenario_name}/`).
CSV format: semicolon delimiter (`;`), comma decimal separator (`,`) by default (German locale).

#### HTML Dashboard Report (`output/report/`)

Interactive single-file HTML dashboard for stakeholder communication:

**Generation flow:**

1. `data_collector.py` aggregates all scenario results into `HtmlReportData` dataclass
2. `llm_prompt.py` renders a prompt template for AI-assisted report text generation
3. **Interactive pause**: Tool saves `{scenario}_llm_prompt.md`, prompts user to copy to Copilot/ChatGPT, save JSON
   response, enter path
4. `html_builder.py` injects data + LLM response into `dashboard.html` template

**Report tabs:**

1. Scenario overview (parameters, map)
2. Input timeseries (PV monthly yield, price scenarios)
3. Grid search results (multi-series IRR chart, if >1 grid point)
4. EEG sensitivity (if enabled)
5. PPA Collar sweep (if enabled)
6. PPA Baseload sweep (if enabled)
7. Cashflow analysis (stacked bar chart + KPIs)

**Features:**

- Offline-capable single HTML file
- Base64-embedded logos (tool + company)
- Canvas-based interactive charts (zoom, pan, tooltip)
- CSV/PNG download buttons
- Dark mode toggle
- German language

**CLI flags:**

- `--no-report`: Skip HTML report generation
- `--skip-llm-prompt`: Generate report with placeholder texts (no LLM pause)
- `--llm-response <PATH>`: Pre-specify LLM response JSON path (no interactive pause)

---

## Scenario JSON Schema

```json
{
  "scenario": {
    "name": "PV_BESS_Szenario_001",
    "skip_baseline": false,
    "monte_carlo": {
      "enabled": true,
      "iterations": 1000,
      "sigma_capex_pv_pct": 5.0,
      "sigma_capex_bess_pct": 7.5,
      "sigma_opex_pv_pct": 3.0,
      "sigma_opex_bess_pct": 4.0,
      "sigma_pv_availability_pct": 0.5,
      "sigma_bess_availability_pct": 0.8
    },
    "output": {
      "directory": "./outputs/pv_bess_szenario_001",
      "export_dispatch_sample": true,
      "csv_separator": ";",
      "csv_decimal": ",",
      "csv_timestamp_column": "timestamp",
      "csv_timestamp_format": "%Y-%m-%d %H:%M:%S",
      "report": {
        "enabled": true,
        "company_name": "Solar Storage GmbH",
        "logo_path": "./assets/logo.png"
      }
    },
    "analyses": {
      "eeg_sensitivity": {
        "enabled": true,
        "floor_prices_eur_per_kwh": [
          0.05,
          0.07,
          0.09
        ]
      },
      "ppa_collar": {
        "enabled": true,
        "floor_prices_eur_per_kwh": [
          0.040,
          0.055,
          0.070
        ],
        "cap_spreads_eur_per_kwh": [
          0.020,
          0.030
        ],
        "duration_years": 10,
        "inflation_on_ppa": true,
        "goo_premium_eur_per_kwh": 0.003
      },
      "ppa_baseload": {
        "enabled": true,
        "ppa_prices_eur_per_kwh": [
          0.060,
          0.070,
          0.080
        ],
        "baseload_levels_mw": [
          5.0,
          7.5
        ],
        "duration_years": 7,
        "inflation_on_ppa": false,
        "goo_premium_eur_per_kwh": 0.002
      }
    }
  },
  "project_settings": {
    "lifetime_years": 25,
    "commissioning_year": 2027,
    "discount_rate": 0.07,
    "operating_mode": "green",
    "location": {
      "latitude": 53.87,
      "longitude": 10.69,
      "pvgis_database": "PVGIS-SARAH"
    },
    "technology": {
      "pv": {
        "design": {
          "peak_power_kwp": 50000,
          "mounting_type": "free",
          "azimuth_deg": 0,
          "tilt_deg": 25
        },
        "performance": {
          "degradation_rate_pct_per_year": 0.5,
          "pv_availability_pct": 99.9
        },
        "costs": {
          "capex": {
            "fixed_eur": 250000.0,
            "eur_per_kw": 480.0,
            "eur_per_kwh": 0.0,
            "pct_of_capex": 0.0
          },
          "opex": {
            "fixed_eur": 120000.0,
            "eur_per_kw": 10.0,
            "eur_per_kwh": 0.0,
            "pct_of_capex": 0.01
          }
        }
      },
      "bess": {
        "design_space": {
          "scale_pct_of_pv": [
            25,
            50,
            100,
            150
          ],
          "e_to_p_ratio_hours": [
            2,
            4
          ]
        },
        "performance": {
          "round_trip_efficiency_pct": 88.0,
          "min_soc_pct": 10.0,
          "max_soc_pct": 95.0,
          "degradation_rate_pct_per_year": 2.0,
          "bess_availability_pct": 98.0
        },
        "costs": {
          "capex": {
            "fixed_eur": 150000.0,
            "eur_per_kw": 250.0,
            "eur_per_kwh": 200.0,
            "pct_of_capex": 0.0
          },
          "opex": {
            "fixed_eur": 60000.0,
            "eur_per_kw": 8.0,
            "eur_per_kwh": 2.0,
            "pct_of_capex": 0.015,
            "optimization_fee_pct": 3.0
          },
          "replacement": {
            "enabled": true,
            "year": 12,
            "fixed_eur": 50000.0,
            "eur_per_kw": 120.0,
            "eur_per_kwh": 90.0,
            "pct_of_capex": 0.0,
            "capacity_factor_pct": 80.0
          }
        }
      },
      "grid_connection": {
        "max_export_kw": 45000.0,
        "system_loss_pct": 2.5,
        "costs": {
          "capex": {
            "fixed_eur": 300000.0,
            "eur_per_kw": 30.0,
            "eur_per_kwh": 0.0,
            "pct_of_capex": 0.0
          },
          "opex": {
            "fixed_eur": 15000.0,
            "eur_per_kw": 1.0,
            "eur_per_kwh": 0.0,
            "pct_of_capex": 0.002
          }
        }
      }
    },
    "finance": {
      "leverage_pct": 60.0,
      "interest_rate_pct": 4.2,
      "loan_tenor_years": 12,
      "equity_irr_target": 12.0,
      "debt_sizing_downside_pct": 15.0,
      "inflation_rate": 0.02,
      "revenue_streams": {
        "marketing": {
          "type": "market",
          "floor_price_eur_per_kwh": null,
          "fixed_price_years": null,
          "eeg_inflation": false
        },
        "ppa": {
          "type": "ppa_collar",
          "pay_as_produced_price_eur_per_kwh": null,
          "baseload_mw": null,
          "floor_price_eur_per_kwh": 0.055,
          "cap_price_eur_per_kwh": 0.085,
          "duration_years": 10,
          "inflation_on_ppa": true,
          "guarantee_of_origin_eur_per_kwh": 0.003
        }
      },
      "price_inputs": {
        "scenarios": [
          {
            "name": "Central",
            "label": "Zentralszenario",
            "csv_column": "price_central",
            "weather_year": 2018,
            "weight": 0.6,
            "is_central": true,
            "price_csv": "./inputs/day_ahead_prices.csv",
            "inflation_on_input_data": false,
            "csv_separator": ";",
            "csv_decimal": ",",
            "csv_timestamp_column": "timestamp",
            "csv_timestamp_format": "%Y-%m-%dT%H:%M:%S"
          },
          {
            "name": "High",
            "label": "Hohes Preisniveau",
            "csv_column": "price_high",
            "weather_year": 2015,
            "weight": 0.25,
            "is_central": false,
            "price_csv": "./inputs/day_ahead_prices.csv",
            "inflation_on_input_data": false
          },
          {
            "name": "Low",
            "label": "Niedriges Preisniveau",
            "csv_column": "price_low",
            "weather_year": 2016,
            "weight": 0.15,
            "is_central": false,
            "price_csv": "./inputs/day_ahead_prices.csv",
            "inflation_on_input_data": false
          }
        ]
      },
      "tax": {
        "afa_years_pv": 20,
        "afa_years_bess": 15,
        "gewerbesteuer_hebesatz": 400.0,
        "gewerbesteuer_messzahl": 0.035,
        "koerperschaftsteuer_pct": 15.0,
        "solidaritaetszuschlag_pct": 5.5
      }
    }
  }
}
```

---

## CLI Interface

```bash
# Run a single scenario
python -m pv_bess_model.main --scenario scenarios/my_scenario.json

# Override output directory
python -m pv_bess_model.main --scenario scenarios/my_scenario.json --output results/run_01/

# Skip Monte Carlo (even if enabled in JSON)
python -m pv_bess_model.main --scenario scenarios/my_scenario.json --no-mc

# Skip grid search (use fixed BESS size from CLI override)
python -m pv_bess_model.main --scenario scenarios/my_scenario.json --bess-power 2000 --bess-capacity 8000

# Verbose logging
python -m pv_bess_model.main --scenario scenarios/my_scenario.json -v

# Dry run (validate JSON, no simulation)
python -m pv_bess_model.main --scenario scenarios/my_scenario.json --dry-run

# Skip HTML report generation
python -m pv_bess_model.main --scenario scenarios/my_scenario.json --no-report

# Generate report with placeholder texts (no LLM interaction)
python -m pv_bess_model.main --scenario scenarios/my_scenario.json --skip-llm-prompt

# Pre-specify LLM response JSON (no interactive pause)
python -m pv_bess_model.main --scenario scenarios/my_scenario.json --llm-response response.json
```

---

## Execution Flow

```
1.  Load & validate scenario JSON
2.  Fetch PVGIS data for each weather year (or load from cache)
3.  Build per-scenario PV timeseries (one per price-weather scenario)
4.  Load price CSV timeseries per scenario, extend to project lifetime
5.  Grid Search (ratio-based, full multi-year, central scenario):
    a. Build combinations from scale_pct_of_pv × e_to_p_ratio_hours
    b. For each (scale, E/P ratio) combination:
       i.    Derive BESS power and capacity from ratios
       ii.   Calculate total CAPEX and base OPEX (unified cost schema)
       iii.  Run full multi-year dispatch (central scenario, deterministic availability)
       iv.   Build complete cashflow projection
       v.    Calculate Equity IRR
    c. Record results in grid search matrix
    d. Identify optimum (max Equity IRR)
6.  Post-Grid-Search Analyses (on optimum, if enabled):
    a. EEG Sensitivity: sweep floor prices, full MC per point
    b. PPA Collar: 2D sweep (floor × cap_spread), full MC per combination
    c. PPA Baseload: 2D sweep (price × baseload_mw), full MC per combination
7.  Monte Carlo (on optimum, if enabled):
    a. Dispatch once per price-weather scenario (parallelized)
    b. For each iteration: sample noise factors, apply post-hoc, build cashflow
    c. Compute distribution statistics (overall and per price scenario)
8.  Write output CSVs
9.  Generate HTML dashboard report (with optional LLM-assisted text)
10. Print summary to stdout
```

---

## Implementation Guidelines

### Code Quality

- **Python 3.10+** with type hints on all function signatures
- **No magic numbers**: Every constant comes from scenario JSON or `config/defaults.py`
- `config/defaults.py` contains ~340 constants with docstrings covering:
    - Time constants (hours, days, intervals, quarter-hourly resolution)
    - PVGIS API settings
    - Monte Carlo defaults (separate sigmas for PV and BESS)
    - BESS/dispatch defaults
    - LP solver settings
    - Financial defaults (rates, tenors, tax)
    - Price/market constants
    - Output formatting (CSV delimiter, decimal, precision)
    - Grid search defaults
    - PPA/marketing type identifiers
    - Report styling constants
- **Docstrings**: Every module, class, and public function must have a docstring
- **Logging**: Use Python `logging` module, configurable verbosity via CLI
- Use `numpy` for array operations, `pandas` for timeseries handling
- Use `numpy_financial` for IRR/NPV (install via pip)
- Use `scipy.optimize.linprog` for daily LP dispatch optimization
- Use `requests` for PVGIS API calls
- Use `jsonschema` for scenario JSON validation

### Testing Strategy

- **Unit tests** for every module using `pytest` (~45 test files)
- Each test file mirrors the source module
- **Integration test suite**: 36-scenario matrix (3 tech setups × 2 modes × 6 marketing strategies)
    - `dispatch_constraint_checker.py` validates energy balance, SoC limits, power limits, grid constraints
    - KPI ranking tests confirm expected economic ordering
- Use fixtures in `conftest.py` for:
    - Sample PV timeseries (small, e.g., 24–48 intervals)
    - Sample price timeseries
    - Sample scenario configurations
- Test edge cases:
    - BESS SoC at limits (empty, full)
    - Grid export limit binding
    - Leap year handling
    - Price timeseries shorter than project lifetime (extension logic)
    - Zero BESS size (PV-only case)
    - Zero PV size (BESS-only case)
    - Negative electricity prices
    - Dual chamber in Grey Mode: 100% green, 100% grey, mixed
    - EEG floor price: spot above floor, spot below floor, spot equal to floor
    - PPA Collar: spot below floor, between floor/cap, above cap
    - PPA expiry mid-project
    - BESS replacement year + capacity upgrade
    - MC with σ = 0 (should equal deterministic result)
    - MC with single price scenario (no scenario sampling)
    - Unified cost schema: missing fields treated as 0
    - LP optimizer: verify SoC day-to-day coupling
    - LP optimizer: verify Green Mode blocks grid charging
    - LP optimizer: BESS offline day produces zero charge/discharge
    - Grid search: ratio-based BESS sizing derivation (power and capacity from scale + E/P)
    - Grid search: scale = 0% produces PV-only baseline
    - Verlustvortrag across multiple years
    - Grid loss factor applied correctly
    - PV availability offline days
- Test data: Use small synthetic datasets, not real PVGIS data (for speed)

### Performance Considerations

- Grid search is embarrassingly parallel → use `concurrent.futures`
- Ratio-based grid search produces fewer combinations (e.g., 8 scales × 2 E/P ratios = 16 combinations)
- Each grid search point runs full multi-year dispatch (25 years × 365 days = 9,125 LP solves)
- LP solver: `scipy.optimize.linprog` with HiGHS solves each 96-timestep LP in ~2-4ms
- **MC uses dispatch-once-per-scenario** approach: only S × 9,125 LP solves (not N × S × 9,125)
- Typical workload: 16 grid points × 9,125 LPs + 3 scenarios × 9,125 LPs = ~173K LP solves
- Target: Grid search completes in <10 minutes (parallelized). MC completes in <1 minute for 1,000 iterations.

### Error Handling

- Validate scenario JSON against schema before any computation
- Validate CSV price files (correct number of rows, no NaN values, correct delimiter)
- Validate MC price scenario weights sum to 1.0 (within tolerance)
- PVGIS API: Handle HTTP errors, timeout, rate limiting with retries (max 5, exponential backoff)
- Financial calculations: Handle edge cases (negative IRR, non-converging IRR, zero cashflows)
- LP solver: Handle infeasible LPs (log warning, fall back to zero-dispatch for that day)
- Log warnings for unusual but valid inputs (e.g., very high leverage, very long project lifetime)

---

## Agent Workflow for Claude Code

This project uses multiple Claude Code agents for development:

### Agent 1: Coding

- Set up project structure, `pyproject.toml`, dependencies
- Implement modules based on additional user input
- Do not write tests, this will be done by another agent

### Agent 2: Testing

- Write unit tests based on this specification and the implementation of the respective method calls
- Review all unit tests for completeness and correctness
- Add missing edge case tests
- Ensure test coverage > 90% for core modules (dispatch, finance, BESS)
- Create test fixtures with known-good results for validation

### General Rules for All Agents

- Read this CLAUDE.md before starting any implementation
- Never introduce magic numbers – use `config/defaults.py` or scenario JSON
- All functions must have type hints and docstrings
- Run `pytest` after every significant change
- Use `black` for formatting, `ruff` for linting
- Commit messages: `module: description` (e.g., `dispatch: implement daily LP optimizer`)

---

## Dependencies

```
numpy>=1.24
numpy-financial>=1.0
pandas>=2.0
scipy>=1.10
requests>=2.28
jsonschema>=4.17
pytest>=7.0
black>=23.0
ruff>=0.1
```

---

## Price CSV Format

The price CSV files must follow this format:

```csv
timestamp;price_central;price_high;price_low
2023-01-01T00:00:00;45.23;55.12;35.12
2023-01-01T00:15:00;44.89;54.23;34.90
2023-01-01T00:30:00;44.56;53.12;34.23
2023-01-01T00:45:00;45.01;54.01;35.01
...
```

- **timestamp**: ISO 8601 format, quarter-hourly or hourly resolution
- **Column names**: Match `csv_column` values from price-weather scenario definitions
- Delimiter: semicolon (`;`), configurable per scenario (`csv_separator`)
- Decimal separator: configurable per scenario (`csv_decimal`, default `.`)
- Header row required
- No missing values allowed
- Minimum: one full year (8,760 rows hourly or 35,040 rows quarter-hourly)
