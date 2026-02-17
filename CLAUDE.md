# CLAUDE.md – PV + BESS Co-Location Financial Model

## Project Overview

A command-line Python tool for evaluating co-location of photovoltaic (PV) systems with battery energy storage systems (BESS). The tool answers: What is the optimal BESS sizing relative to a PV plant? Which grid connection and marketing strategies maximize equity returns? How do PPA structures compare to EEG feed-in tariffs?

The tool runs a **ratio-based grid search** over BESS sizing (as percentage of PV peak power, with configurable energy-to-power ratios), executing a full multi-year financial model for each combination within a user-defined scenario. The optimal configuration (max Equity IRR) is then subjected to Monte Carlo simulation for risk analysis.

**User**: Single user (developer), CLI-only, no UI required.

---

## Architecture

### Module Structure

```
pv_bess_model/
├── main.py                     # CLI entrypoint, orchestrator
├── config/
│   ├── schema.py               # JSON schema validation for scenario files
│   ├── loader.py               # Load & validate scenario JSON + CSV price files
│   └── defaults.py             # Global defaults and constants (NO magic numbers in code)
├── pv/
│   ├── pvgis_client.py         # PVGIS API client – fetch hourly historical data
│   ├── timeseries.py           # P50/P90 calculation from historical years
│   └── degradation.py          # Annual linear degradation applied to timeseries
├── bess/
│   ├── battery.py              # BESS state model (SoC, charge/discharge, degradation)
│   └── replacement.py          # Optional mid-life BESS replacement logic
├── dispatch/
│   ├── engine.py               # Hourly dispatch engine (yearly simulation loop)
│   └── optimizer.py            # Daily LP-based dispatch optimization
├── market/
│   ├── price_loader.py         # Load CSV price timeseries, extend to project lifetime
│   ├── eeg.py                  # EEG floor tariff logic
│   ├── ppa.py                  # PPA models (pay-as-produced, baseload, floor, collar)
├── finance/
│   ├── cashflow.py             # Annual cashflow projection (revenue, OPEX, debt, equity)
│   ├── costs.py                # Unified CAPEX/OPEX calculation from scenario JSON
│   ├── debt.py                 # Annuity loan model
│   ├── tax.py                  # Simplified tax treatment (AfA, GewSt)
│   ├── metrics.py              # IRR, NPV, DSCR calculation
│   └── inflation.py            # Inflation escalation logic
├── optimization/
│   ├── grid_search.py          # Ratio-based grid search over BESS sizing
│   └── monte_carlo.py          # MC simulation on optimal configuration
├── output/
│   ├── csv_writer.py           # Write summary, cashflows, grid search, dispatch CSVs
│   └── formatting.py           # Number/currency formatting helpers
├── tests/
│   ├── test_pvgis_client.py
│   ├── test_timeseries.py
│   ├── test_battery.py
│   ├── test_optimizer.py
│   ├── test_costs.py
│   ├── test_ppa.py
│   ├── test_eeg.py
│   ├── test_cashflow.py
│   ├── test_debt.py
│   ├── test_metrics.py
│   ├── test_grid_search.py
│   ├── test_monte_carlo.py
│   └── conftest.py             # Shared fixtures, sample data
└── scenarios/                  # Example scenario JSON files
    ├── example_eeg_green.json
    ├── example_ppa_floor_grey.json
    └── example_regelenergie.json
```

### Key Design Principles

1. **No magic numbers**: Every numeric value comes from the scenario JSON or from `config/defaults.py`. Constants in `config/defaults.py` must have descriptive names and docstrings.
2. **One scenario = one JSON file**: To compare scenarios, the user creates multiple JSON files and runs the tool separately. Comparison is done by the user via the output CSVs.
3. **Deterministic before probabilistic**: Grid search runs deterministically (P50 for equity, P90 for debt) with full multi-year dispatch. MC adds stochastic noise factors on top of the grid search optimum.
4. **Hourly resolution**: The dispatch engine operates at 1-hour intervals (8,760 steps per year). PVGIS data arrives hourly and is used directly.
5. **Immutable operating mode**: The BESS operating mode (green/grey) is fixed per scenario and cannot change during simulation.

---

## Module Specifications

### 1. PV Module (`pv/`)

#### PVGIS API Client (`pvgis_client.py`)
- Fetch **hourly** historical radiation/production data from the EU PVGIS API (https://re.jrc.ec.europa.eu/api/v5_3/)
- Use the `seriescalc` endpoint with `outputformat=json`
- Parameters from scenario JSON: latitude, longitude, PV peak power (kWp), system loss (%), mounting type, azimuth, tilt
- Fetch **all available historical years** (typically 2005–2020 depending on database)
- Handle API rate limits gracefully (retry with backoff)
- Cache downloaded data locally to avoid redundant API calls (cache in `~/.pv_bess_cache/`)

#### Timeseries Processing (`timeseries.py`)
- Input: Dictionary of {year: hourly_production_array} from PVGIS
- For each hour index h (0–8759):
  - Collect the production value at hour h across all historical years
  - Compute P50 (median) and P90 (10th percentile) from this distribution
- **Leap year handling**: Ignore December 31st (hours 8760–8783) for leap years, truncate all years to 8760 hours
- Output: Two arrays of length 8760 (P50 and P90 hourly production in kWh)

#### Degradation (`degradation.py`)
- Apply linear annual degradation rate (user input, e.g., 0.4%/year) to the base timeseries
- For year Y of project: `production[Y] = base_production * (1 - degradation_rate) ^ Y`
- Return degraded timeseries for each project year

### 2. BESS Module (`bess/`)

#### Battery Model (`battery.py`)
- State: current SoC (kWh), max capacity (kWh), max charge/discharge power (kW), round-trip efficiency (%), min SoC (%), max SoC (%)
- All parameters from scenario JSON
- Efficiency model: **Losses are applied only on discharge.** Charging is lossless (1 kWh in = 1 kWh SoC increase). Discharge output = discharged kWh × round-trip efficiency.
- Methods:
  - `charge(kwh)` → actual kWh charged (respecting SoC limits, power limits). SoC increases by charged amount.
  - `discharge(kwh)` → actual kWh removed from SoC (respecting SoC limits, power limits). Grid output = discharged kWh × RTE.
  - `apply_annual_degradation(rate)` → reduce max capacity by rate per year
- Track cumulative throughput (kWh charged + discharged) for reporting

#### Replacement (`replacement.py`)
- Optional: user specifies replacement year(s) and cost in scenario JSON
- At replacement year: reset BESS capacity to original, add replacement cost as OPEX in that year, restart degradation
- Replacement cost follows the same unified cost schema (see Finance Module)
- If no replacement specified: BESS degrades continuously over project lifetime

### 3. Dispatch Module (`dispatch/`)

#### Dispatch Engine (`engine.py`)
- Core simulation loop: iterate over **365 days per year** (not 8,760 individual hours)
- For each project year:
  1. Get degraded PV timeseries for this year (8,760 values)
  2. Get degraded BESS capacity for this year
  3. Get price timeseries for this year (with inflation if enabled)
  4. Determine pricing regime for this year (fixed-price phase or market phase)
  5. Determine BESS offline days for this year (MC only, see BESS Availability below)
  6. For each day (d = 0..364):
     a. Extract `pv_production[d*24 : (d+1)*24]` (24 hourly values)
     b. Extract `prices[d*24 : (d+1)*24]` (24 hourly values)
     c. If BESS is offline this day: skip LP, dispatch PV directly (`export = min(pv, P_grid_max)`, rest curtailed)
     d. If BESS is online: call daily LP optimizer with start SoC from previous day
     e. Record 24 hourly results into yearly arrays
     f. Carry over end-SoC to next day
  7. Aggregate annual results: total revenue, costs, energy flows
- The engine receives the **operating mode** (green/grey) as configuration
- Track all energy flows per hour for dispatch sample export and per year for cashflow

#### Daily LP Optimizer (`optimizer.py`)

The optimizer solves a linear program for each day (24 hourly timesteps) to determine optimal dispatch decisions under perfect foresight of day-ahead prices.

##### Decision Variables (per hour t, t=0..23)

**Green Mode:**
- `charge_pv[t]` – kWh charged from PV surplus into BESS
- `discharge_green[t]` – kWh discharged from BESS (green energy)
- `export_pv[t]` – kWh PV directly exported to grid
- `curtail[t]` – kWh PV curtailed
- `revenue_export_pv[t]` – revenue helper variable for EEG floor linearization
- `revenue_discharge_green[t]` – revenue helper variable for EEG floor linearization

**Grey Mode (additional variables):**
- `charge_grid[t]` – kWh charged from grid into BESS (at spot price)
- `discharge_grey[t]` – kWh discharged from BESS (grey energy)
- `soc_green[t]` – SoC tracking for green kWh in BESS
- `soc_grey[t]` – SoC tracking for grey kWh in BESS

##### Objective Function

**Green Mode – maximize daily revenue:**
```
max Σ_t [ revenue_export_pv[t] + revenue_discharge_green[t] ]
```

**Grey Mode – maximize daily net revenue:**
```
max Σ_t [ revenue_export_pv[t]
        + revenue_discharge_green[t]
        + discharge_grey[t] × RTE × price_spot[t]
        - charge_grid[t] × price_spot[t] ]
```

##### EEG/PPA Floor Price Linearization

During the fixed-price period (EEG or PPA), the effective price per kWh is `max(price_spot[t], price_fixed)`. This is linearized via revenue helper variables:

```
revenue_export_pv[t] ≥ export_pv[t] × price_spot[t]
revenue_export_pv[t] ≥ export_pv[t] × price_fixed

revenue_discharge_green[t] ≥ discharge_green[t] × RTE × price_spot[t]
revenue_discharge_green[t] ≥ discharge_green[t] × RTE × price_fixed
```

Since the objective maximizes revenue, the solver will automatically select the higher value. After the fixed-price period expires, `price_fixed` is set to 0 (or removed), and revenue equals spot price.

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

**Grid connection limit:**
```
export_pv[t] + (discharge_green[t] + discharge_grey[t]) × RTE ≤ P_grid_max   ∀t
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
- Problem size per day:
  - Green Mode: ~72 variables (24h × 3 decision + revenue helpers), ~170 constraints
  - Grey Mode: ~168 variables (24h × 7 decision + revenue helpers), ~340 constraints
- Expected solve time: <1ms per day with HiGHS
- Total per year: 365 solves × <1ms = <0.4s per year
- The optimizer returns per-hour dispatch decisions: `charge_pv[t]`, `charge_grid[t]`, `discharge_green[t]`, `discharge_grey[t]`, `export_pv[t]`, `curtail[t]`

##### BESS Availability (Offline Days)

BESS availability is modelled as whole-day outages (maintenance, faults). This applies in both grid search and Monte Carlo, but with different logic:

**In Grid Search (deterministic):**
- Use the `bess_availability_pct` from the scenario JSON directly
- `n_offline_days = round((1 - bess_availability_pct / 100) × 365)`
- Offline days are distributed evenly across the year (every Nth day)

**In Monte Carlo (stochastic):**
- Sample `availability_factor` from `N(μ_avail, σ_avail)`, clipped to [0, 1]
- `n_offline_days = round((1 - availability_factor) × 365)`
- Offline days are drawn randomly (uniform, without replacement) from the 365 days of the year
- Offline days are redrawn independently for each project year within an MC iteration

**On offline days:**
- All BESS variables in the LP are fixed to 0 (charge = 0, discharge = 0)
- The LP reduces to pure PV dispatch: `export_pv[t] = min(pv_production[t], P_grid_max)`, remainder is curtailed
- SoC is frozen at the value from the end of the last online day

### 4. Market Module (`market/`)

#### Price Loader (`price_loader.py`)
- Load CSV file with electricity price timeseries
- Expected format: hourly prices in €/MWh (or €/kWh, configurable via `price_unit`)
- If project lifetime exceeds price timeseries length: repeat the **last full year** of the timeseries until project end
- Validate: timeseries must cover at least one full year (8,760 hourly values)
- Support loading multiple price CSVs for MC price scenarios (low/mid/high)

#### EEG Module (`eeg.py`)
- EEG tariff acts as a **floor price (Mindestpreis)**, not a fixed price
- Effective price per kWh: `max(price_spot[t], price_eeg)`
- The floor applies for the first X years (both tariff level and duration from user input)
- After X years: pure market price (floor drops away)
- Inflation adjustment: optional, controlled by user flag (`eeg_inflation: true/false`)
- If inflation enabled: `eeg_price[year] = base_eeg_price * (1 + inflation_rate) ^ year`

#### PPA Module (`ppa.py`)
Implement four PPA structures, selectable per scenario:

1. **Pay-as-produced** (`ppa_pay_as_produced`)
   - Buyer pays fixed price per kWh actually produced
   - Price from user input, optional inflation escalation

2. **Baseload PPA** (`ppa_baseload`)
   - Seller commits to deliver a flat power profile (baseload MW)
   - Baseload level: user input or calculated as annual production / 8760
   - Profile cost: When PV < baseload → seller buys shortfall at market price. When PV > baseload → seller sells excess at market price
   - Net revenue = baseload_volume × ppa_price + excess_revenue - shortfall_cost
   - BESS can help shape the profile (reduce shortfall, shift excess)

3. **Floor PPA** (`ppa_floor`)
   - Minimum price guaranteed (floor), seller keeps upside above floor
   - Same logic as EEG: `revenue_per_kwh = max(price_spot[t], ppa_price)`
   - Floor price from user input

4. **Collar PPA** (`ppa_collar`)
   - Floor price and cap price as boundaries
   - Revenue per kWh = clip(market_price, floor_price, cap_price)
   - Both prices from user input

All PPA models:
- Duration (years) from user input
- After PPA expires: switch to pure market price
- Inflation escalation: optional per user flag, effective on all price related data points (floor, cap, ppa_price)
- Guarantee of origin premium added to effective price for all PPA structures (user-defined €/kWh)

### 5. Finance Module (`finance/`)

#### Unified Cost Calculation (`costs.py`)

All CAPEX and OPEX fields follow a single, consistent schema. Each cost block (PV, BESS, Grid, Other) supports three additive components. If a parameter is not set in the JSON, it is treated as 0 in the addition.

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

**Total annual OPEX:**
```
OPEX_total = OPEX_pv + OPEX_bess + OPEX_grid + OPEX_other
```

All OPEX is subject to inflation escalation over project lifetime.

BESS replacement cost follows the same three-component schema (`fixed_eur`, `eur_per_kw`, `eur_per_kwh`, `pct_of_capex`). Replacement cost is added as additional OPEX in the replacement year.

#### Cashflow Projection (`cashflow.py`)
- Build annual cashflow table for full project lifetime
- Revenue streams (from dispatch simulation):
  - PV direct feed-in revenue (EEG floor / PPA / market depending on year and config)
  - BESS discharge revenue green (EEG floor / PPA / market)
  - BESS discharge revenue grey (market price, Grey Mode only)
  - Minus: grid import costs (Grey Mode only)
- Both grid search and Monte Carlo run full multi-year dispatch with degradation and price changes
- CAPEX: Year 0
- OPEX: Annual, inflated per year
- Cashflow per year: Revenue - OPEX - Debt Service - Tax = Equity Cashflow

#### Debt Module (`debt.py`)
- Simple annuity loan
- Parameters: loan amount (% of total CAPEX), interest rate, loan tenor (years)
- Calculate annual annuity payment (constant over tenor)
- Separate interest and principal components per year
- Loan amount calculated from total CAPEX × leverage ratio
- **P90 case**: Debt sizing uses P90 production timeseries for conservative revenue projection
- DSCR per year = (Revenue - OPEX) / Debt Service

#### Tax Module (`tax.py`)
- Simplified German tax treatment with two components:
  - **Linear depreciation (AfA)**: Separate depreciation periods for PV and BESS (user-defined, e.g., 20 years for PV, 10 years for BESS). Depreciation base = CAPEX of respective asset.
  - **Gewerbesteuer (GewSt)**: `GewSt = max(0, taxable_income) × Messzahl × Hebesatz / 100` where `taxable_income = Revenue - OPEX - AfA + Verlustvortrag_adjustment`
- **Verlustvortrag (loss carry-forward)**: If taxable income is negative in a year, the loss is carried forward indefinitely. Carried-forward losses offset future positive taxable income before GewSt is calculated.
- Tax reduces equity cashflow

#### Metrics (`metrics.py`)
- **Project IRR**: IRR on total project cashflows (pre-leverage)
- **Equity IRR**: IRR on equity cashflows (post-leverage, post-tax)
- **NPV**: At user-defined discount rate
- **DSCR**: Min DSCR and average DSCR over loan tenor
- **LCOE**: Levelized cost of energy (total costs / total production)
- **Payback period**: Year when cumulative equity cashflow turns positive
- Use `numpy_financial` for IRR/NPV calculation

#### Inflation (`inflation.py`)
- Single inflation rate from user input
- Applied to: OPEX (always), PPA price (if flag set), EEG price (if flag set), Day-Ahead price from CSV (if flag set)
- `inflated_value[year] = base_value * (1 + inflation_rate) ^ year`

### 6. Optimization Module (`optimization/`)

#### Grid Search (`grid_search.py`)

The grid search uses a **ratio-based parametrization** to efficiently explore BESS sizing. Instead of independently varying power and capacity, the search space is defined by two user-configurable dimensions:

1. **BESS scale** (% of PV peak power): How large is the BESS relative to the PV plant?
2. **Energy-to-power ratio** (hours): What is the storage duration?

Both dimensions are specified as lists in the scenario JSON. The grid search evaluates all combinations.

**Deriving BESS power and capacity from ratios:**
```
BESS_power_kW   = pv_peak_kwp × scale_pct / 100
BESS_capacity_kWh = BESS_power_kW × e_to_p_ratio_hours
```

Example: PV = 5,000 kWp, scale = 40%, E/P = 2h → BESS = 2,000 kW / 4,000 kWh

**Always include scale = 0% (PV-only baseline)** as the first entry.

**For each (scale, E/P ratio) combination:**
1. Derive BESS power and capacity from ratios
2. Calculate CAPEX using unified cost schema (PV + BESS + Grid + Other)
3. Calculate annual base OPEX using unified cost schema
4. **Run full multi-year dispatch** (all project years):
   - For each project year: apply PV degradation, BESS degradation, OPEX inflation, price evolution
   - Run 365 daily LP optimizations per year (P50 timeseries and `mid` price data)
   - BESS offline days applied deterministically (see BESS Availability)
5. Build complete cashflow projection (year-varying revenue, OPEX, debt, tax)
6. Calculate Equity IRR

**Output:** 2D matrix of Equity IRR indexed by (scale_pct, e_to_p_ratio). Identify optimum = max Equity IRR.

**Performance:** With typical inputs (e.g., 11 scale steps × 3 E/P ratios = 33 combinations, 25 years each), the grid search requires 33 × 25 × 365 = ~300K LP solves. At <1ms per solve: ~5 minutes total. Parallelizable across combinations with `multiprocessing.Pool` or `concurrent.futures`.

#### Monte Carlo (`monte_carlo.py`)

The MC simulation runs on the optimal (scale, E/P ratio) from grid search and adds **stochastic noise factors** on top of the full multi-year dispatch.

##### Price Scenarios

The MC supports multiple price timeseries (e.g., low/mid/high) with user-defined sampling weights. In each MC iteration, one price scenario is drawn randomly according to its weight.

```json
"price_scenarios": {
  "low":  {"csv_column": "LOW",  "weight": 0.25},
  "mid":  {"csv_column": "MID",  "weight": 0.50},
  "high": {"csv_column": "HIGH", "weight": 0.25}
}
```

If only one price CSV column is provided (no scenarios), all iterations use that single timeseries.

##### MC Iteration Logic

For each MC iteration:
1. **Sample price scenario**: Draw one of low/mid/high according to weights
2. **Sample noise factors** from normal distributions:
   - PV yield factor: `N(1.0, σ_pv)` – multiplied with entire production timeseries
   - CAPEX factor: `N(1.0, σ_capex)` – multiplied with total CAPEX
   - OPEX factor: `N(1.0, σ_opex)` – multiplied with annual OPEX
   - BESS availability factor: `N(μ_avail, σ_avail)` – determines number of offline days (see BESS Availability)
3. All σ values from user input. No correlations between parameters (independent draws).
4. **Run full multi-year simulation**:
   - For each project year: apply PV degradation, BESS degradation, BESS availability, inflation, and price evolution
   - Run daily LP optimizer for each year (365 × project_years LP solves per iteration)
   - Build complete cashflow projection
5. Record: Equity IRR, Project IRR, NPV, Min DSCR, selected price scenario, all noise factors

##### MC Output

- Distribution statistics (mean, median, P10, P25, P50, P75, P90, std) for each metric
- Statistics broken down by price scenario (conditional distributions)
- Number of iterations: user input (default: 1,000)

### 7. Output Module (`output/`)

#### CSV Writer (`csv_writer.py`)

Produce the following CSV files per scenario run:

1. **`{scenario_name}_summary.csv`**
   - One row with all key results:
     - Input parameters (scenario name, PV size, optimal BESS scale %, optimal E/P ratio, optimal BESS power, optimal BESS capacity, operating mode, marketing model, PPA type, fixed price, fixed price years, project lifetime, etc.)
     - Financial results (Equity IRR, Project IRR, NPV, Min DSCR, Avg DSCR, LCOE, Payback period)
     - Total production (MWh lifetime), total revenue, total CAPEX, total OPEX

2. **`{scenario_name}_cashflows.csv`**
   - One row per project year
   - Columns: Year, PV Production (MWh), BESS Throughput (MWh), Revenue PV (€), Revenue BESS Green (€), Revenue BESS Grey (€), Grid Import Cost (€), Total Revenue (€), CAPEX (€), OPEX (€), Debt Service (€), Gewerbesteuer (€), Decepriation (€), Equity CF (€), Cumulative Equity CF (€), DSCR

3. **`{scenario_name}_grid_search.csv`**
   - One row per (scale, E/P ratio) combination
   - Columns: Scale Pct of PV (%), E/P Ratio (h), BESS Power (kW), BESS Capacity (kWh), Total CAPEX (€), Total OPEX (€), Sum of revenue year 1 (€),  Equity IRR (%), Project IRR (%), NPV (€), is_optimal (boolean)

4. **`{scenario_name}_monte_carlo.csv`** (only if MC is enabled)
   - One row per MC iteration
   - Columns: Iteration, Price Scenario, PV Yield Factor, CAPEX Factor, OPEX Factor, BESS Availability Factor, Equity IRR (%), Project IRR (%), NPV (€), Min DSCR, all selected noise factors and price scenario

5. **`{scenario_name}_dispatch_sample.csv`** (optimal configuration, first year)
   - One row per hour (8,760 rows)
   - Columns: Timestamp, PV Production (kWh), Price Spot (€/MWh), Price Effective (€/MWh), BESS SoC (kWh), BESS SoC Green (kWh), BESS SoC Grey (kWh), BESS Charge PV (kWh), BESS Charge Grid (kWh), BESS Discharge Green (kWh), BESS Discharge Grey (kWh), Grid Export (kWh), Curtailed (kWh), Revenue (€)

All output files go to a user-specified output directory (default: `./output/{scenario_name}/`).

---

## Scenario JSON Schema

```json
{
  "scenario": {
    "name": "EEG_Green_Example",
    "monte_carlo": {
      "enabled": true,
      "iterations": 1000,
      "sigma_pv_yield_pct": 5.0,
      "sigma_capex_pct": 8.0,
      "sigma_opex_pct": 5.0,
      "sigma_bess_availability_pct": 2.0,
      "price_scenarios": {
        "low":  {"csv_column": "LOW",  "weight": 0.25},
        "mid":  {"csv_column": "MID",  "weight": 0.50},
        "high": {"csv_column": "HIGH", "weight": 0.25}
      }
    },
    "output": {
      "directory": "output/",
      "export_dispatch_sample": true
    }
  },
  "project_settings": {
    "lifetime_years": 25,
    "discount_rate": 0.06,
    "operating_mode": "green",
    "location": {
      "latitude": 53.55,
      "longitude": 9.99,
      "pvgis_database": "PVGIS-SARAH2"
    },
    "technology": {
      "pv": {
        "design": {
          "peak_power_kwp": 5000,
          "mounting_type": "free",
          "azimuth_deg": 0,
          "tilt_deg": 30
        },
        "performance": {
          "system_loss_pct": 14.0,
          "degradation_rate_pct_per_year": 0.4
        },
        "costs": {
          "capex": {
            "fixed_eur": 50000,
            "eur_per_kw": 800
          },
          "opex": {
            "fixed_eur": 5000,
            "eur_per_kw": 12,
            "pct_of_capex": 0.003
          }
        }
      },
      "bess": {
        "design_space": {
          "scale_pct_of_pv": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
          "e_to_p_ratio_hours": [1, 2, 4]
        },
        "performance": {
          "round_trip_efficiency_pct": 88.0,
          "min_soc_pct": 10.0,
          "max_soc_pct": 90.0,
          "degradation_rate_pct_per_year": 2.0,
          "bess_availability_pct": 97.0
        },
        "costs": {
          "capex": {
            "fixed_eur": 50000,
            "eur_per_kw": 100,
            "eur_per_kwh": 250
          },
          "opex": {
            "fixed_eur": 25000,
            "pct_of_capex": 0.015
          },
          "replacement": {
            "enabled": false,
            "year": 12,
            "fixed_eur": 0,
            "eur_per_kw": 120,
            "eur_per_kwh": 141
          }
        }
      },
      "grid_connection": {
        "max_export_kw": 4000,
        "costs": {
          "capex": {
            "fixed_eur": 50000,
            "eur_per_kw": 100
          },
          "opex": {
            "fixed_eur": 25000,
            "pct_of_capex": 0.015
          }
        }
      }
    },
    "finance": {
      "leverage_pct": 75,
      "interest_rate_pct": 4.5,
      "loan_tenor_years": 18,
      "equity_irr_target": null,
      "debt_uses_p90": true,
      "inflation_rate": 0.02,
      "revenue_streams": {
        "marketing": {
          "type": "eeg",
          "floor_price_eur_per_kwh": 0.0735,
          "fixed_price_years": 20,
          "eeg_inflation": false
        },
        "ppa": {
          "type": "none",
          "pay_as_produced_price_eur_per_kwh": null,
          "baseload_mw": null,
          "floor_price_eur_per_kwh": null,
          "cap_price_eur_per_kwh": null,
          "duration_years": 10,
          "inflation_on_ppa": false,
          "guarantee_of_origin_eur_per_kwh": 0.005
        }
      },
      "price_inputs": {
        "day_ahead_csv": "data/day_ahead_prices.csv",
        "price_unit": "eur_per_mwh",
        "inflation_on_input_data": true
      },
      "tax": {
        "afa_years_pv": 20,
        "afa_years_bess": 10,
        "gewerbesteuer_hebesatz": 400,
        "gewerbesteuer_messzahl": 0.035
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
```

---

## Execution Flow

```
1. Load & validate scenario JSON
2. Fetch PVGIS data (or load from cache)
3. Compute P50 and P90 hourly timeseries (8,760 values each)
4. Load price CSV timeseries, extend to project lifetime
5. Grid Search (ratio-based, full multi-year):
   a. Build combinations from scale_pct_of_pv × e_to_p_ratio_hours
   b. For each (scale, E/P ratio) combination:
      i.    Derive BESS power and capacity from ratios
      ii.   Calculate total CAPEX and base OPEX (unified cost schema)
      iii.  For each project year:
            - Apply PV degradation, BESS degradation, OPEX inflation, price evolution
            - Determine BESS offline days (deterministic from availability_pct)
            - Run 365 daily LP optimizations (P50 timeseries)
      iv.   Build complete cashflow projection
      v.    Calculate Equity IRR
   c. Record results in grid search matrix
   d. Identify optimum (max Equity IRR)
6. Monte Carlo (on optimum only, adds stochastic noise):
   a. For each iteration:
      i.    Sample price scenario (low/mid/high by weight)
      ii.   Sample noise factors (PV yield, CAPEX, OPEX, BESS availability)
      iii.  For each project year:
            - Apply degradation, inflation, sampled availability (random offline days)
            - Run 365 daily LPs
      iv.   Build complete cashflow, calculate metrics
   b. Compute distribution statistics (overall and per price scenario)
7. Write output CSVs
8. Print summary to stdout
```

---

## Implementation Guidelines

### Code Quality
- **Python 3.11+** with type hints on all function signatures
- **No magic numbers**: Every constant comes from scenario JSON or `config/defaults.py`
- `config/defaults.py` structure:
  ```python
  # config/defaults.py
  """Global default values. All numeric constants must be defined here, not inline."""

  HOURS_PER_YEAR: int = 8760
  DAYS_PER_YEAR: int = 365
  HOURS_PER_DAY: int = 24
  PVGIS_API_BASE_URL: str = "https://re.jrc.ec.europa.eu/api/v5_3/"
  PVGIS_CACHE_DIR: str = "~/.pv_bess_cache"
  DEFAULT_MC_ITERATIONS: int = 1000
  DEFAULT_START_SOC_FRACTION: float = 50.0  # Start at 50% bess capacity
  # ... etc.
  ```
- **Docstrings**: Every module, class, and public function must have a docstring
- **Logging**: Use Python `logging` module, configurable verbosity via CLI
- Use `numpy` for array operations, `pandas` for timeseries handling
- Use `numpy_financial` for IRR/NPV (install via pip)
- Use `scipy.optimize.linprog` for daily LP dispatch optimization
- Use `requests` for PVGIS API calls
- Use `jsonschema` for scenario JSON validation

### Testing Strategy
- **Unit tests** for every module using `pytest`
- Each test file mirrors the source module
- Use fixtures in `conftest.py` for:
  - Sample PV timeseries (small, e.g., 24–48 hours)
  - Sample price timeseries
  - Sample scenario configurations
  - Pre-built BESS state objects
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
  - PPA expiry mid-project
  - BESS replacement year
  - MC with σ = 0 (should equal deterministic result)
  - MC with single price scenario (no scenario sampling)
  - Unified cost schema: missing fields treated as 0
  - LP optimizer: verify SoC day-to-day coupling
  - LP optimizer: verify Green Mode blocks grid charging
  - LP optimizer: BESS offline day produces zero charge/discharge
  - Grid search: ratio-based BESS sizing derivation (power and capacity from scale + E/P)
  - Grid search: scale = 0% produces PV-only baseline
  - Verlustvortrag across multiple years
- **No integration tests in test suite** – integration testing done manually by user
- Test data: Use small synthetic datasets, not real PVGIS data (for speed)

### Performance Considerations
- Grid search is embarrassingly parallel → use `multiprocessing.Pool` or `concurrent.futures`
- Ratio-based grid search produces far fewer combinations than independent Power × Capacity grids (e.g., 11 scales × 3 E/P ratios = 33 vs. 11 × 11 = 121)
- Each grid search point runs full multi-year dispatch (25 years × 365 days = 9,125 LP solves) → target <10s per grid point
- LP solver performance: `scipy.optimize.linprog` with HiGHS solves each 24h LP in <1ms
- MC iterations are also parallelizable
- Typical workload: 33 grid points × 9,125 LPs + 1,000 MC iterations × 9,125 LPs = ~9.4M LP solves total
- Target: Grid search completes in <5 minutes (33 points, parallelized). MC completes in <30 minutes for 1,000 iterations (parallelized).

### Error Handling
- Validate scenario JSON against schema before any computation
- Validate CSV price files (correct number of rows, no NaN values, correct delimiter)
- Validate MC price scenario weights sum to 1.0 (within tolerance)
- PVGIS API: Handle HTTP errors, timeout, rate limiting with retries
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
timestamp,low,mid,high
2023-01-01T00:00:00,45.23,9.12,10.12
2023-01-01T01:00:00,44.89,-12.23,100.9
2023-01-01T02:00:00,44.56,88.12,91.23
2023-01-01T03:00:00,45.01,72.12,1.01
...
```

- **timestamp**: ISO 8601 format, hourly resolution
- **low/mid/high**: Electricity price in the unit specified by `price_unit` in scenario JSON
- Delimiter: comma
- Header row required
- No missing values allowed
- Minimum: one full year (8,760 rows)
