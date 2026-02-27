# Feature 09: Integration Test Suite

## Übersicht

Umfassende Integration-Test-Suite, die alle Kombinationen aus technischem Setup, Betriebsmodus und Vermarktungsstrategie als End-to-End-Tests abdeckt. Alle 36 Szenarien leiten sich aus einem einzigen Master-Szenario ab, sodass die Ergebnisse untereinander vergleichbar sind. Es wird **keine Grid-Search** durchgeführt – jedes Szenario verwendet das fest vordefinierte PV/BESS-Setup aus dem Master-Input. Zusätzlich validiert ein Dispatch-Constraint-Checker die physikalischen Grenzen, ein Availability-Checker zählt PV- und BESS-Offline-Tage, und ein KPI-Ranking prüft die erwartete Rangfolge der Finanz-Kennzahlen zwischen den Szenarien.

---

## Szenario-Matrix (36 Szenarien)

### Dimensionen

| Dimension | Werte | Anzahl |
|-----------|-------|--------|
| Technisches Setup | PV-Only, BESS-Only, PV+BESS | 3 |
| Betriebsmodus | Green, Grey | 2 |
| Vermarktung | Market, EEG, PPA-Pay-as-Produced, PPA-Baseload, PPA-Floor, PPA-Collar | 6 |

**3 × 2 × 6 = 36 Szenarien**

### Vollständige Matrix

| # | Setup | Modus | Vermarktung | Szenario-ID |
|---|-------|-------|-------------|-------------|
| 1 | PV-Only | Green | Market | `pv_only_green_market` |
| 2 | PV-Only | Green | EEG | `pv_only_green_eeg` |
| 3 | PV-Only | Green | PPA-Pay-as-Produced | `pv_only_green_ppa_pap` |
| 4 | PV-Only | Green | PPA-Baseload | `pv_only_green_ppa_baseload` |
| 5 | PV-Only | Green | PPA-Floor | `pv_only_green_ppa_floor` |
| 6 | PV-Only | Green | PPA-Collar | `pv_only_green_ppa_collar` |
| 7 | PV-Only | Grey | Market | `pv_only_grey_market` |
| 8 | PV-Only | Grey | EEG | `pv_only_grey_eeg` |
| 9 | PV-Only | Grey | PPA-Pay-as-Produced | `pv_only_grey_ppa_pap` |
| 10 | PV-Only | Grey | PPA-Baseload | `pv_only_grey_ppa_baseload` |
| 11 | PV-Only | Grey | PPA-Floor | `pv_only_grey_ppa_floor` |
| 12 | PV-Only | Grey | PPA-Collar | `pv_only_grey_ppa_collar` |
| 13 | BESS-Only | Green | Market | `bess_only_green_market` |
| 14 | BESS-Only | Green | EEG | `bess_only_green_eeg` |
| 15 | BESS-Only | Green | PPA-Pay-as-Produced | `bess_only_green_ppa_pap` |
| 16 | BESS-Only | Green | PPA-Baseload | `bess_only_green_ppa_baseload` |
| 17 | BESS-Only | Green | PPA-Floor | `bess_only_green_ppa_floor` |
| 18 | BESS-Only | Green | PPA-Collar | `bess_only_green_ppa_collar` |
| 19 | BESS-Only | Grey | Market | `bess_only_grey_market` |
| 20 | BESS-Only | Grey | EEG | `bess_only_grey_eeg` |
| 21 | BESS-Only | Grey | PPA-Pay-as-Produced | `bess_only_grey_ppa_pap` |
| 22 | BESS-Only | Grey | PPA-Baseload | `bess_only_grey_ppa_baseload` |
| 23 | BESS-Only | Grey | PPA-Floor | `bess_only_grey_ppa_floor` |
| 24 | BESS-Only | Grey | PPA-Collar | `bess_only_grey_ppa_collar` |
| 25 | PV+BESS | Green | Market | `pv_bess_green_market` |
| 26 | PV+BESS | Green | EEG | `pv_bess_green_eeg` |
| 27 | PV+BESS | Green | PPA-Pay-as-Produced | `pv_bess_green_ppa_pap` |
| 28 | PV+BESS | Green | PPA-Baseload | `pv_bess_green_ppa_baseload` |
| 29 | PV+BESS | Green | PPA-Floor | `pv_bess_green_ppa_floor` |
| 30 | PV+BESS | Green | PPA-Collar | `pv_bess_green_ppa_collar` |
| 31 | PV+BESS | Grey | Market | `pv_bess_grey_market` |
| 32 | PV+BESS | Grey | EEG | `pv_bess_grey_eeg` |
| 33 | PV+BESS | Grey | PPA-Pay-as-Produced | `pv_bess_grey_ppa_pap` |
| 34 | PV+BESS | Grey | PPA-Baseload | `pv_bess_grey_ppa_baseload` |
| 35 | PV+BESS | Grey | PPA-Floor | `pv_bess_grey_ppa_floor` |
| 36 | PV+BESS | Grey | PPA-Collar | `pv_bess_grey_ppa_collar` |

### Besonderheiten

- **Keine Grid-Search:** Alle Szenarien verwenden das fest definierte BESS-Setup aus dem Master-Input (`--bess-power` / `--bess-capacity` CLI-Override). Dadurch entfällt der Grid-Search-Overhead und die Ergebnisse sind direkt vergleichbar.
- **PV-Only:** BESS-Power und -Kapazität = 0, BESS-Kosten = 0. Green und Grey liefern identische Ergebnisse (kein BESS zum Laden aus dem Netz). Tests prüfen, dass Grey ≈ Green.
- **BESS-Only:** `peak_power_kwp: 0` (erfordert FIX-S2-03), PV-Kosten = 0. In Green Mode kann der BESS nicht geladen werden (keine PV-Produktion, kein Netzimport) → Revenue = 0, NPV negativ. Grey Mode ermöglicht Arbitrage.
- **PV+BESS:** Volle Funktionalität. Festes BESS-Setup: 500 kW / 1.000 kWh (50% der PV-Leistung, E/P = 2h).
- **PPA-Pay-as-Produced:** Fester Preis pro kWh, unabhängig vom Spotmarkt. Einfachste PPA-Struktur.
- **PPA-Baseload:** Erfordert `baseload_mw` als expliziten User-Input. Master setzt `baseload_mw: 0.3` (300 kW, ~30% der PV-Nennleistung).

---

## Master-Szenario

Alle 36 Szenarien leiten sich aus einem einzigen Master-Szenario ab. Die Abweichungen werden programmatisch über `copy.deepcopy()` + gezielte Feld-Modifikation erzeugt.

### Kein Grid-Search

Die Grid-Search wird übersprungen, indem jedes Szenario über die CLI-Argumente `--bess-power` und `--bess-capacity` mit festen BESS-Werten aufgerufen wird. Dadurch evaluiert das Modell genau einen Grid-Punkt.

| Tech-Setup | `--bess-power` | `--bess-capacity` |
|---|---|---|
| PV-Only | 0 | 0 |
| BESS-Only | 500 | 1000 |
| PV+BESS | 500 | 1000 |

### Master-Szenario-Parameter

```python
MASTER_SCENARIO = {
    "scenario": {
        "name": "integration_master",
        "monte_carlo": {"enabled": false},
        "output": {
            "directory": ".data/test/integration_suite/",
            "export_dispatch_sample": true
        }
    },
    "project_settings": {
        "commissioning_year": 2025,
        "lifetime_years": 3,           # Kurz für schnelle Tests
        "discount_rate": 0.06,
        "operating_mode": "green",      # Wird pro Szenario überschrieben
        "location": {
            "latitude": 53.55,
            "longitude": 9.99,
            "pvgis_database": "PVGIS-SARAH3"
        },
        "technology": {
            "pv": {
                "design": {
                    "peak_power_kwp": 1000,
                    "mounting_type": "free",
                    "azimuth_deg": 0,
                    "tilt_deg": 30
                },
                "performance": {
                    "degradation_rate_pct_per_year": 0.4
                },
                "costs": {
                    "capex": {"fixed_eur": 0, "eur_per_kw": 800},
                    "opex": {"fixed_eur": 5000, "eur_per_kw": 12}
                }
            },
            "bess": {
                "design_space": {
                    "scale_pct_of_pv": [50],
                    "e_to_p_ratio_hours": [2]
                },
                "performance": {
                    "round_trip_efficiency_pct": 90.0,
                    "min_soc_pct": 10.0,
                    "max_soc_pct": 90.0,
                    "degradation_rate_pct_per_year": 2.0,
                    "bess_availability_pct": 97.0
                },
                "costs": {
                    "capex": {"fixed_eur": 0, "eur_per_kw": 100, "eur_per_kwh": 250},
                    "opex": {"fixed_eur": 2000, "pct_of_capex": 0.015},
                    "replacement": {"enabled": false}
                }
            },
            "grid_connection": {
                "max_export_kw": 800,
                "costs": {
                    "capex": {"fixed_eur": 5000, "eur_per_kw": 50},
                    "opex": {"fixed_eur": 1000}
                },
                "system_loss_pct": 5.0
            }
        },
        "finance": {
            "leverage_pct": 60,
            "interest_rate_pct": 4.0,
            "loan_tenor_years": 3,
            "equity_irr_target": null,
            "debt_uses_p90": true,
            "inflation_rate": 0.02,
            "revenue_streams": {
                "marketing": {
                    "type": "market",       # Wird pro Szenario überschrieben
                    "floor_price_eur_per_kwh": 0.07,
                    "fixed_price_years": 3,
                    "eeg_inflation": false
                },
                "ppa": {
                    "type": "none",         # Wird pro Szenario überschrieben
                    "pay_as_produced_price_eur_per_kwh": 0.065,
                    "baseload_mw": 0.3,
                    "floor_price_eur_per_kwh": 0.06,
                    "cap_price_eur_per_kwh": 0.09,
                    "duration_years": 3,
                    "inflation_on_ppa": false,
                    "guarantee_of_origin_eur_per_kwh": 0.005
                }
            },
            "price_inputs": {
                "day_ahead_csv": "integration_suite_prices.csv",
                "price_unit": "eur_per_mwh",
                "inflation_on_input_data": false
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

### Preis-CSV

Eigene Preis-CSV `integration_suite_prices.csv` mit synthetischen, deterministischen Preisen.

- 3 Jahre × 8.760 Stunden = 26.280 Zeilen
- Spalten: `timestamp;MID` (eine Spalte reicht, MC ist deaktiviert)
- Preisprofil: Sinus-Tagesprofil mit saisonaler Variation, sodass BESS-Arbitrage möglich ist
- Preisspanne: 20–120 €/MWh, Durchschnitt ~60 €/MWh
- Negative Preise in einzelnen Nachtstunden (Stunden 2–4), um Curtailment zu provozieren

### Szenario-Modifikationen

```python
TECH_SETUPS = ["pv_only", "bess_only", "pv_bess"]
OPERATING_MODES = ["green", "grey"]
MARKETING_STRATEGIES = ["market", "eeg", "ppa_pap", "ppa_baseload", "ppa_floor", "ppa_collar"]

# Feste BESS-Konfigurationen pro Tech-Setup (kein Grid-Search)
BESS_CONFIGS = {
    "pv_only":  {"bess_power": 0,   "bess_capacity": 0},
    "bess_only": {"bess_power": 500, "bess_capacity": 1000},
    "pv_bess":  {"bess_power": 500, "bess_capacity": 1000},
}

def build_scenario(master, tech_setup, operating_mode, marketing):
    """Erzeuge Szenario aus Master durch gezielte Modifikation."""
    s = copy.deepcopy(master)
    name = f"{tech_setup}_{operating_mode}_{marketing}"
    s["scenario"]["name"] = name

    # --- Technisches Setup ---
    if tech_setup == "pv_only":
        # Kein BESS (wird über CLI --bess-power=0 --bess-capacity=0 gesteuert)
        pass
    elif tech_setup == "bess_only":
        s["project_settings"]["technology"]["pv"]["design"]["peak_power_kwp"] = 0
        s["project_settings"]["technology"]["pv"]["costs"]["capex"] = {
            "fixed_eur": 0, "eur_per_kw": 0
        }
        s["project_settings"]["technology"]["pv"]["costs"]["opex"] = {
            "fixed_eur": 0, "eur_per_kw": 0
        }
        # BESS-Only benötigt absolute Sizing (aus FIX-S2-03)
        s["project_settings"]["technology"]["bess"]["absolute_power_kw"] = 500
        s["project_settings"]["technology"]["bess"]["absolute_capacity_kwh"] = 1000
    # pv_bess: Master-Werte verwenden (500 kW / 1000 kWh)

    # --- Betriebsmodus ---
    s["project_settings"]["operating_mode"] = operating_mode

    # --- Vermarktung ---
    if marketing == "market":
        s["project_settings"]["finance"]["revenue_streams"]["marketing"]["type"] = "market"
        s["project_settings"]["finance"]["revenue_streams"]["ppa"]["type"] = "none"
    elif marketing == "eeg":
        s["project_settings"]["finance"]["revenue_streams"]["marketing"]["type"] = "eeg"
        s["project_settings"]["finance"]["revenue_streams"]["ppa"]["type"] = "none"
    elif marketing == "ppa_pap":
        s["project_settings"]["finance"]["revenue_streams"]["marketing"]["type"] = "ppa"
        s["project_settings"]["finance"]["revenue_streams"]["ppa"]["type"] = "ppa_pay_as_produced"
    elif marketing == "ppa_baseload":
        s["project_settings"]["finance"]["revenue_streams"]["marketing"]["type"] = "ppa"
        s["project_settings"]["finance"]["revenue_streams"]["ppa"]["type"] = "ppa_baseload"
    elif marketing == "ppa_floor":
        s["project_settings"]["finance"]["revenue_streams"]["marketing"]["type"] = "ppa"
        s["project_settings"]["finance"]["revenue_streams"]["ppa"]["type"] = "ppa_floor"
    elif marketing == "ppa_collar":
        s["project_settings"]["finance"]["revenue_streams"]["marketing"]["type"] = "ppa"
        s["project_settings"]["finance"]["revenue_streams"]["ppa"]["type"] = "ppa_collar"

    return s
```

---

## Dispatch-Constraint-Checker

Ein eigenes Modul `tests/dispatch_constraint_checker.py`, das den stündlichen Dispatch-Sample auf die Einhaltung aller physikalischen und vertraglichen Constraints prüft.

### Zu prüfende Constraints

#### Energiebilanz
```
∀t: export_pv[t] + charge_pv[t] + curtail[t] ≈ pv_production[t]    (Toleranz: 0.01 kWh)
```

#### SoC-Grenzen
```
∀t: soc_min ≤ soc[t] ≤ soc_max                                      (Toleranz: 0.01 kWh)
```

#### Leistungsgrenzen
```
∀t: charge_pv[t] + charge_grid[t] ≤ P_max_charge + ε                (ε = 0.01 kW)
∀t: discharge_green[t] + discharge_grey[t] ≤ P_max_discharge + ε
```

#### Netzanschlussgrenze
```
∀t: export_pv[t] × glf + (discharge_green[t] + discharge_grey[t]) × RTE ≤ P_grid_max + ε
```

#### Nicht-Negativität
```
∀t: charge_pv[t] ≥ -ε
∀t: charge_grid[t] ≥ -ε
∀t: discharge_green[t] ≥ -ε
∀t: discharge_grey[t] ≥ -ε
∀t: export_pv[t] ≥ -ε
∀t: curtail[t] ≥ -ε
```

#### Green-Mode-Restriktionen
```
Im Green Mode:
∀t: charge_grid[t] = 0                (kein Netzimport für BESS)
∀t: discharge_grey[t] = 0             (keine graue Entladung)
```

#### SoC-Kontinuität (Tag-zu-Tag)
```
∀d > 0: soc_start[d] ≈ soc_end[d-1]                                 (Toleranz: 0.01 kWh)
```

#### BESS-Offline-Tage
```
An Offline-Tagen:
∀t: charge_pv[t] = 0, charge_grid[t] = 0
∀t: discharge_green[t] = 0, discharge_grey[t] = 0
SoC bleibt konstant
```

#### PV-Offline-Tage
```
An PV-Offline-Tagen:
∀t: pv_production[t] = 0
∀t: export_pv[t] = 0, charge_pv[t] = 0, curtail[t] = 0
```

### Implementierung

```python
@dataclass
class ConstraintViolation:
    constraint: str       # Name des Constraints
    hour: int             # Stunde (0-8759)
    expected: str         # Erwarteter Wert / Bedingung
    actual: float         # Tatsächlicher Wert
    severity: str         # "error" oder "warning"

def check_dispatch_constraints(
    dispatch_df: pd.DataFrame,
    pv_peak_kwp: float,
    bess_power_kw: float,
    bess_capacity_kwh: float,
    grid_max_kw: float,
    rte: float,
    min_soc_pct: float,
    max_soc_pct: float,
    operating_mode: str,
    grid_loss_factor: float,
    tolerance: float = 0.01,
) -> list[ConstraintViolation]:
    """Prüfe alle Dispatch-Constraints. Gibt leere Liste zurück wenn alles OK."""
    violations = []
    # ... Constraint-Checks ...
    return violations
```

Der Checker wird im Test aufgerufen:
```python
violations = check_dispatch_constraints(dispatch_df, **params)
assert len(violations) == 0, f"Constraint-Verletzungen: {violations}"
```

---

## Availability-Checker (PV- und BESS-Offline-Tage)

Der Dispatch-Sample wird zusätzlich auf die Anzahl der PV- und BESS-Offline-Tage geprüft. Die gemessenen Offline-Tage müssen innerhalb des Rahmens der konfigurierten Availability liegen.

### Zählung der Offline-Tage

#### BESS-Offline-Tage
Ein Tag gilt als BESS-offline, wenn für alle 24 Stunden des Tages gilt:
```
charge_pv[t] = 0  AND  charge_grid[t] = 0  AND  discharge_green[t] = 0  AND  discharge_grey[t] = 0
```
**und** gleichzeitig der SoC über den Tag konstant bleibt (keine Ladung/Entladung).

Hinweis: An Tagen ohne PV und ohne Preisanreiz kann es vorkommen, dass der Optimizer ebenfalls null BESS-Dispatch wählt, obwohl der BESS online ist. Deshalb wird die Zählung über den Dispatch-Sample nur als Obergrenze verwendet.

#### PV-Offline-Tage
Ein Tag gilt als PV-offline, wenn für alle 24 Stunden des Tages gilt:
```
pv_production[t] = 0
```
Hierbei müssen Nachtstunden berücksichtigt werden — ein Tag zählt nur als PV-offline, wenn auch während der typischen Sonnenstunden (8–16 Uhr) die Produktion 0 ist.

### Erwartete Offline-Tage

```python
DAYS_PER_YEAR = 365

# BESS: deterministisch aus bess_availability_pct
expected_bess_offline = round((1.0 - bess_availability_pct / 100.0) * DAYS_PER_YEAR)
# Bei 97%: round((1 - 0.97) * 365) = round(10.95) = 11 Tage/Jahr

# PV: Im deterministischen Grid-Search-Pfad (kein MC) gibt es keine PV-Offline-Tage.
# PV-Offline-Tage werden nur im MC-Pfad über sigma_pv_availability erzeugt.
# Da MC in der Integration-Suite deaktiviert ist, erwarten wir 0 PV-Offline-Tage.
expected_pv_offline = 0
```

### Assertions

```python
def check_availability(
    dispatch_df: pd.DataFrame,
    bess_availability_pct: float,
    has_bess: bool,
    mc_enabled: bool,
) -> tuple[int, int]:
    """Zähle PV- und BESS-Offline-Tage im Dispatch-Sample (Jahr 1)."""
    bess_offline_days = 0
    pv_offline_days = 0

    for day in range(DAYS_PER_YEAR):
        h_start = day * 24
        h_end = h_start + 24
        day_data = dispatch_df.iloc[h_start:h_end]

        # PV-Offline: Produktion = 0 in Sonnenstunden (8-16h)
        sun_hours = day_data.iloc[8:17]  # Stunden 8-16
        if (sun_hours["pv_production_kwh"].abs() < 0.001).all():
            pv_offline_days += 1

        # BESS-Offline: Kein Charge/Discharge und SoC konstant
        if has_bess:
            bess_flow = (
                day_data["bess_charge_pv_kwh"].abs().sum()
                + day_data.get("bess_charge_grid_kwh", pd.Series(0)).abs().sum()
                + day_data["bess_discharge_green_kwh"].abs().sum()
                + day_data.get("bess_discharge_grey_kwh", pd.Series(0)).abs().sum()
            )
            soc_range = day_data["bess_soc_kwh"].max() - day_data["bess_soc_kwh"].min()
            if bess_flow < 0.01 and soc_range < 0.01:
                bess_offline_days += 1

    return pv_offline_days, bess_offline_days


class TestAvailability:
    """Prüfe, ob Offline-Tage innerhalb der konfigurierten Availability liegen."""

    def test_bess_offline_days_match_availability(self, all_results):
        """Anzahl BESS-Offline-Tage ≈ erwartete Tage aus bess_availability_pct."""
        bess_avail = 97.0  # Master-Wert
        expected = round((1.0 - bess_avail / 100.0) * DAYS_PER_YEAR)  # 11 Tage

        for name, result in all_results.items():
            if "pv_only" in name:
                continue  # Kein BESS → keine BESS-Offline-Tage
            # Gemessene Offline-Tage dürfen nicht weniger als erwartet sein
            # (mehr ist möglich, da Optimizer auch an Online-Tagen null-Dispatch wählen kann)
            assert result.bess_offline_days >= expected, (
                f"{name}: {result.bess_offline_days} BESS-Offline-Tage gemessen, "
                f"erwartet mindestens {expected}"
            )

    def test_pv_offline_days_zero_without_mc(self, all_results):
        """Ohne MC: Keine PV-Offline-Tage (PV-Availability wird nur in MC gesampelt)."""
        for name, result in all_results.items():
            if "bess_only" in name:
                continue  # Keine PV → nicht prüfbar
            assert result.pv_offline_days == 0, (
                f"{name}: {result.pv_offline_days} PV-Offline-Tage, erwartet 0 (MC deaktiviert)"
            )

    def test_bess_only_no_pv_production(self, all_results):
        """BESS-Only: PV-Produktion ist in allen Stunden 0."""
        for name, result in all_results.items():
            if "bess_only" not in name:
                continue
            # Alle 365 Tage zählen als "PV-offline" bei BESS-Only
            assert result.pv_offline_days == DAYS_PER_YEAR, (
                f"{name}: Erwartet {DAYS_PER_YEAR} PV-Offline-Tage bei BESS-Only"
            )
```

---

## KPI-Ranking zwischen Szenarien

### Erwartete Rangfolge (NPV / Equity IRR)

Die folgenden Rangfolgen ergeben sich aus der ökonomischen Logik der Vermarktungsstrukturen und werden als Assertions geprüft.

#### Innerhalb eines technischen Setups (gleicher Modus)

**PV-basierte Szenarien (PV-Only und PV+BESS):**

```
EEG > PPA-Floor > PPA-Collar > PPA-Pay-as-Produced > Market
```

Begründung:
- **EEG** bietet den höchsten Floor-Preis (0.07 €/kWh) ohne Cap → maximale Absicherung + volle Upside
- **PPA-Floor** bietet Floor (0.06 €/kWh) + GoO-Premium → Absicherung + Upside, aber niedrigerer Floor als EEG
- **PPA-Collar** begrenzt die Upside durch Cap (0.09 €/kWh), bietet aber Floor (0.06) + GoO → Floor-Absicherung mit begrenztem Gewinn
- **PPA-Pay-as-Produced** bietet festen Preis (0.065 €/kWh) + GoO → kein Upside bei hohen Spotpreisen, aber auch kein Downside
- **Market** hat keinen Floor → volles Preisrisiko, niedrigste durchschnittliche Erlöse bei volatilen Preisen

**PPA-Baseload** ist schwer in eine lineare Rangfolge einzuordnen, da der Erlös stark von der Korrelation zwischen PV-Produktion und Spotpreis abhängt. Baseload-PPAs können bei Unterdeckung Shortfall-Kosten verursachen. Die Rangfolge wird als **nicht strikt vorgegeben** behandelt – stattdessen wird nur geprüft, dass der Baseload-NPV plausibel ist (z.B. zwischen Market und EEG).

```python
assert npv["eeg"] >= npv["ppa_floor"], "EEG muss besser sein als PPA-Floor"
assert npv["ppa_floor"] >= npv["ppa_collar"], "PPA-Floor muss besser sein als PPA-Collar"
assert npv["ppa_collar"] >= npv["ppa_pap"], "PPA-Collar muss besser sein als PPA-Pay-as-Produced"
assert npv["ppa_pap"] >= npv["market"], "PPA-PaP muss besser sein als Market"
# Baseload: Plausibilitätscheck
assert npv["market"] <= npv["ppa_baseload"] <= npv["eeg"], \
    "PPA-Baseload NPV muss zwischen Market und EEG liegen"
```

**Hinweis:** Diese Rangfolge gilt bei den gewählten Master-Parametern (EEG-Floor > PPA-Floor > PPA-PaP-Preis, Durchschnittspreis unterhalb des Collar-Caps). Bei anderen Parametrisierungen könnte die Rangfolge abweichen. Die Tests validieren die Konsistenz bei gegebener Parametrisierung.

#### BESS-Only-Szenarien

Für BESS-Only gelten besondere Regeln:
- **Green Mode:** BESS kann nicht geladen werden (keine PV, kein Netzimport) → Revenue = 0 für alle Vermarktungsstrategien. NPV ist negativ und identisch (nur CAPEX/OPEX, kein Erlös).
- **Grey Mode:** BESS nutzt Arbitrage (Netzimport bei niedrigen Preisen, Entladung bei hohen). Die Vermarktungsstrategie beeinflusst nur den Entladepreis der grünen Komponente — bei BESS-Only gibt es keine grüne Ladung:

```
Grey: Market ≈ EEG ≈ PPA-Floor ≈ PPA-Collar ≈ PPA-PaP ≈ PPA-Baseload
```

Begründung: BESS-Entladung aus dem Netz ("Grey") wird im Grey Mode immer zum Spot-Preis vermarktet, unabhängig von EEG/PPA. Nur die grüne Entladung (aus PV-Ladung) profitiert von Floor/Cap — bei BESS-Only gibt es keine grüne Ladung.

#### Zwischen Betriebsmodi (gleicher Tech-Setup, gleiche Vermarktung)

**PV-Only:**
```
Green ≈ Grey    (kein BESS → identisches Verhalten)
```

**PV+BESS:**
```
Grey ≥ Green    (Grey hat Arbitrage-Optionen zusätzlich zu Green)
```

**BESS-Only:**
```
Grey > Green    (Green hat Revenue = 0)
```

#### Zwischen technischen Setups (gleicher Modus, gleiche Vermarktung)

**Green Mode:**
```
PV+BESS ≥ PV-Only > BESS-Only   (BESS-Only Green hat Revenue = 0)
```

**Grey Mode:**
```
PV+BESS ≥ max(PV-Only, BESS-Only)
```

### Implementierung KPI-Ranking

```python
@pytest.mark.integration
class TestKPIRanking:
    """Prüfe erwartete Rangfolge der Finanz-KPIs zwischen Szenarien."""

    @pytest.mark.parametrize("tech,mode", [
        ("pv_only", "green"), ("pv_only", "grey"),
        ("pv_bess", "green"), ("pv_bess", "grey"),
    ])
    def test_marketing_ranking(self, all_results, tech, mode):
        """EEG > PPA-Floor > PPA-Collar > PPA-PaP > Market."""
        r = all_results
        assert r[f"{tech}_{mode}_eeg"].npv >= r[f"{tech}_{mode}_ppa_floor"].npv
        assert r[f"{tech}_{mode}_ppa_floor"].npv >= r[f"{tech}_{mode}_ppa_collar"].npv
        assert r[f"{tech}_{mode}_ppa_collar"].npv >= r[f"{tech}_{mode}_ppa_pap"].npv
        assert r[f"{tech}_{mode}_ppa_pap"].npv >= r[f"{tech}_{mode}_market"].npv

    @pytest.mark.parametrize("tech,mode", [
        ("pv_only", "green"), ("pv_only", "grey"),
        ("pv_bess", "green"), ("pv_bess", "grey"),
    ])
    def test_baseload_plausibility(self, all_results, tech, mode):
        """PPA-Baseload NPV liegt zwischen Market und EEG."""
        r = all_results
        assert r[f"{tech}_{mode}_market"].npv <= r[f"{tech}_{mode}_ppa_baseload"].npv
        assert r[f"{tech}_{mode}_ppa_baseload"].npv <= r[f"{tech}_{mode}_eeg"].npv

    def test_pv_only_green_equals_grey(self, all_results):
        """PV-Only: Green ≈ Grey (kein BESS)."""
        for mkt in MARKETING_STRATEGIES:
            green = all_results[f"pv_only_green_{mkt}"].npv
            grey = all_results[f"pv_only_grey_{mkt}"].npv
            assert abs(green - grey) < 1.0, \
                f"PV-Only {mkt}: Green und Grey sollten identisch sein"

    def test_grey_geq_green_pv_bess(self, all_results):
        """PV+BESS: Grey ≥ Green."""
        for mkt in MARKETING_STRATEGIES:
            grey = all_results[f"pv_bess_grey_{mkt}"].npv
            green = all_results[f"pv_bess_green_{mkt}"].npv
            assert grey >= green - 1.0  # 1€ Toleranz

    def test_bess_only_green_negative(self, all_results):
        """BESS-Only Green: Revenue = 0, NPV negativ."""
        for mkt in MARKETING_STRATEGIES:
            npv = all_results[f"bess_only_green_{mkt}"].npv
            assert npv < 0, f"BESS-Only Green {mkt} sollte negativen NPV haben"

    def test_bess_only_green_all_equal(self, all_results):
        """BESS-Only Green: Alle Vermarktungen identisch (alle Revenue = 0)."""
        npvs = [all_results[f"bess_only_green_{mkt}"].npv for mkt in MARKETING_STRATEGIES]
        for npv in npvs[1:]:
            assert abs(npv - npvs[0]) < 1.0, \
                "BESS-Only Green: Alle NPVs sollten identisch sein"

    def test_bess_only_grey_all_approx_equal(self, all_results):
        """BESS-Only Grey: Alle Vermarktungen ≈ gleich (nur Spot-Arbitrage)."""
        npvs = [all_results[f"bess_only_grey_{mkt}"].npv for mkt in MARKETING_STRATEGIES]
        spread = max(npvs) - min(npvs)
        assert spread < abs(npvs[0]) * 0.05, \
            f"BESS-Only Grey: NPV-Spread ({spread:.0f}€) sollte < 5% sein"

    def test_pv_bess_geq_pv_only(self, all_results):
        """PV+BESS ≥ PV-Only (BESS kann nur verbessern)."""
        for mode in OPERATING_MODES:
            for mkt in MARKETING_STRATEGIES:
                combined = all_results[f"pv_bess_{mode}_{mkt}"].npv
                pv_only = all_results[f"pv_only_{mode}_{mkt}"].npv
                assert combined >= pv_only - 1.0

    def test_bess_only_grey_gt_green(self, all_results):
        """BESS-Only: Grey > Green (Grey ermöglicht Arbitrage)."""
        for mkt in MARKETING_STRATEGIES:
            grey = all_results[f"bess_only_grey_{mkt}"].npv
            green = all_results[f"bess_only_green_{mkt}"].npv
            assert grey > green, \
                f"BESS-Only {mkt}: Grey ({grey:.0f}) sollte besser als Green ({green:.0f}) sein"
```

---

## Test-Architektur

### Dateistruktur

```
pv_bess_model/
├── tests/
│   ├── test_integration_suite.py       # 36 Szenario-Tests + KPI-Ranking + Availability
│   ├── dispatch_constraint_checker.py  # Constraint-Validierung (Modul)
│   └── conftest.py                     # data_dir Fixture (existiert)
.data/
├── integration_test_inputs/
│   └── suite/
│       └── integration_suite_prices.csv  # Synthetische Preise
├── pvgis_cache/                          # Gecachte PVGIS-Daten (vorab gefetcht)
```

### Test-Klassen und Fixtures

```python
# test_integration_suite.py

import pytest
import copy
import argparse
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

TECH_SETUPS = ["pv_only", "bess_only", "pv_bess"]
OPERATING_MODES = ["green", "grey"]
MARKETING_STRATEGIES = ["market", "eeg", "ppa_pap", "ppa_baseload", "ppa_floor", "ppa_collar"]

BESS_CONFIGS = {
    "pv_only":  {"bess_power": 0,   "bess_capacity": 0},
    "bess_only": {"bess_power": 500, "bess_capacity": 1000},
    "pv_bess":  {"bess_power": 500, "bess_capacity": 1000},
}

@dataclass
class ScenarioResult:
    """Ergebnis eines Szenario-Durchlaufs."""
    name: str
    equity_irr: float | None
    project_irr: float | None
    npv: float
    dscr_min: float | None
    revenue_year1: float
    capex_total: float
    dispatch_violations: list       # ConstraintViolation
    pv_offline_days: int            # Gezählte PV-Offline-Tage (Jahr 1)
    bess_offline_days: int          # Gezählte BESS-Offline-Tage (Jahr 1)

@pytest.fixture(scope="module")
def price_csv_path(data_dir):
    """Pfad zur synthetischen Preis-CSV."""
    return data_dir / "integration_test_inputs" / "suite" / "integration_suite_prices.csv"

@pytest.fixture(scope="module")
def all_results(price_csv_path, tmp_path_factory):
    """Führe alle 36 Szenarien aus und sammle Ergebnisse.

    PVGIS-Daten werden aus dem lokalen Cache (.data/pvgis_cache/) geladen.
    Der Cache muss vorab befüllt sein (einmaliger PVGIS-Fetch).
    """
    results = {}
    output_base = tmp_path_factory.mktemp("integration_suite")

    for tech in TECH_SETUPS:
        for mode in OPERATING_MODES:
            for mkt in MARKETING_STRATEGIES:
                scenario = build_scenario(MASTER_SCENARIO, tech, mode, mkt)
                scenario["project_settings"]["finance"]["price_inputs"]["day_ahead_csv"] = \
                    str(price_csv_path)

                name = f"{tech}_{mode}_{mkt}"
                bess_cfg = BESS_CONFIGS[tech]
                result = run_scenario_programmatic(
                    scenario, output_base / name, bess_cfg
                )
                results[name] = result

    return results


@pytest.mark.integration
class TestScenarioExecution:
    """Jedes Szenario wird erfolgreich durchgeführt."""

    @pytest.mark.parametrize(
        "tech,mode,mkt",
        [(t, m, k) for t in TECH_SETUPS for m in OPERATING_MODES
         for k in MARKETING_STRATEGIES],
        ids=[f"{t}_{m}_{k}" for t in TECH_SETUPS for m in OPERATING_MODES
             for k in MARKETING_STRATEGIES],
    )
    def test_scenario_runs(self, all_results, tech, mode, mkt):
        """Szenario läuft ohne Fehler durch."""
        name = f"{tech}_{mode}_{mkt}"
        assert name in all_results

    @pytest.mark.parametrize(
        "tech,mode,mkt",
        [(t, m, k) for t in TECH_SETUPS for m in OPERATING_MODES
         for k in MARKETING_STRATEGIES],
        ids=[f"{t}_{m}_{k}" for t in TECH_SETUPS for m in OPERATING_MODES
             for k in MARKETING_STRATEGIES],
    )
    def test_dispatch_constraints(self, all_results, tech, mode, mkt):
        """Dispatch-Sample verletzt keine physikalischen Constraints."""
        name = f"{tech}_{mode}_{mkt}"
        violations = all_results[name].dispatch_violations
        assert len(violations) == 0, \
            f"{name}: {len(violations)} Constraint-Verletzungen: {violations[:5]}"


@pytest.mark.integration
class TestAvailability:
    """Prüfe Offline-Tage."""
    # ... (siehe Availability-Checker Abschnitt)


@pytest.mark.integration
class TestKPIRanking:
    """Prüfe Finanz-KPI Rangfolge."""
    # ... (siehe KPI-Ranking Abschnitt)


@pytest.mark.integration
class TestOutputCompleteness:
    """Prüfe, dass alle erwarteten Output-Dateien erzeugt werden."""

    @pytest.mark.parametrize(
        "tech,mode,mkt",
        [(t, m, k) for t in TECH_SETUPS for m in OPERATING_MODES
         for k in MARKETING_STRATEGIES],
    )
    def test_output_files_exist(self, all_results, output_dir, tech, mode, mkt):
        """Alle erwarteten CSV-Dateien werden erzeugt."""
        name = f"{tech}_{mode}_{mkt}"
        expected_files = [
            f"{name}_summary.csv",
            f"{name}_cashflows.csv",
            f"{name}_dispatch_sample.csv",
        ]
        for fname in expected_files:
            assert (output_dir / name / fname).exists(), f"Fehlende Datei: {fname}"
```

### Szenario-Ausführung (programmatisch)

```python
def run_scenario_programmatic(scenario_dict, output_dir, bess_cfg):
    """Führe ein Szenario programmatisch aus und gebe Ergebnis zurück.

    Grid-Search wird übersprungen durch --bess-power / --bess-capacity CLI-Override.
    PVGIS-Daten werden aus dem lokalen Cache geladen (kein Netzwerk-Zugriff).
    """
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Szenario als temporäre JSON-Datei schreiben
    scenario_json = output_dir / "scenario.json"
    scenario_json.write_text(json.dumps(scenario_dict, indent=2))

    # CLI-Argumente: --bess-power und --bess-capacity überspringen Grid-Search
    bess_power = bess_cfg["bess_power"] if bess_cfg["bess_power"] > 0 else None
    bess_capacity = bess_cfg["bess_capacity"] if bess_cfg["bess_capacity"] > 0 else None

    args = argparse.Namespace(
        scenario=str(scenario_json),
        output=str(output_dir),
        no_mc=True,
        bess_power=bess_power,
        bess_capacity=bess_capacity,
        verbose=False,
        dry_run=False,
    )

    exit_code = run(args)
    assert exit_code == 0, f"Szenario {scenario_dict['scenario']['name']} fehlgeschlagen"

    # Ergebnis aus Output-CSVs lesen
    result = parse_results_from_csvs(output_dir, scenario_dict)

    # Dispatch-Constraints und Availability prüfen
    scenario_name = scenario_dict["scenario"]["name"]
    dispatch_path = output_dir / scenario_name / f"{scenario_name}_dispatch_sample.csv"
    violations = []
    pv_offline = 0
    bess_offline = 0

    if dispatch_path.exists():
        dispatch_df = pd.read_csv(dispatch_path, delimiter=CSV_DELIMITER)
        violations = check_dispatch_constraints(
            dispatch_df, **extract_constraint_params(scenario_dict)
        )
        pv_offline, bess_offline = check_availability(
            dispatch_df,
            bess_availability_pct=scenario_dict["project_settings"]["technology"]
                ["bess"]["performance"]["bess_availability_pct"],
            has_bess=bess_cfg["bess_power"] > 0,
            mc_enabled=False,
        )

    return ScenarioResult(
        name=scenario_name,
        equity_irr=result.get("equity_irr"),
        project_irr=result.get("project_irr"),
        npv=result.get("npv"),
        dscr_min=result.get("dscr_min"),
        revenue_year1=result.get("revenue_year1"),
        capex_total=result.get("capex_total"),
        dispatch_violations=violations,
        pv_offline_days=pv_offline,
        bess_offline_days=bess_offline,
    )
```

---

## Synthetische Preis-CSV

### Anforderungen

- Deterministisch und reproduzierbar
- Tägliches Muster mit Morgen-/Abend-Peak und Nacht-Tal
- Saisonale Variation (Winter teurer als Sommer)
- Gelegentlich negative Preise (für Curtailment-Tests)
- Genug Spread für BESS-Arbitrage

### Generierung

```python
def generate_integration_prices(years=3, seed=42):
    """Generiere synthetische Preiszeitreihe für Integration-Tests."""
    rng = np.random.default_rng(seed)
    hours = years * HOURS_PER_YEAR
    t = np.arange(hours)

    # Basislevel: 60 €/MWh
    base = 60.0

    # Tägliches Muster: Peak morgens (8-10h) und abends (17-20h)
    hour_of_day = t % 24
    daily_pattern = np.where(
        (hour_of_day >= 8) & (hour_of_day <= 10), 25.0,
        np.where((hour_of_day >= 17) & (hour_of_day <= 20), 30.0,
        np.where((hour_of_day >= 1) & (hour_of_day <= 4), -15.0, 0.0))
    )

    # Saisonales Muster: Winter (+20), Sommer (-15)
    day_of_year = (t // 24) % 365
    seasonal = 20.0 * np.cos(2 * np.pi * day_of_year / 365)

    # Rauschen
    noise = rng.normal(0, 5, hours)

    prices = base + daily_pattern + seasonal + noise
    # Clip auf [-20, 200] €/MWh
    prices = np.clip(prices, -20.0, 200.0)

    return prices
```

---

## PVGIS-Daten

Die Integration-Tests verwenden **echte PVGIS-Daten aus dem lokalen Cache**. Kein Mock.

### Vorgehen

1. **Einmaliger Fetch:** Beim ersten Ausführen der Integration-Tests (oder manuell vorab) werden die PVGIS-Daten für die Master-Location (53.55°N, 9.99°E) gefetcht und im lokalen Cache `.data/pvgis_cache/` gespeichert.
2. **Nachfolgende Läufe:** Der PVGIS-Client lädt die Daten aus dem Cache – kein Netzwerkzugriff nötig.
3. **Determinismus:** Die PVGIS-Daten sind historisch und ändern sich nicht. Bei gleicher Location und gleicher Datenbank (`PVGIS-SARAH3`) sind die Ergebnisse reproduzierbar.

### Voraussetzung

- `.data/pvgis_cache/` muss beim ersten Lauf beschreibbar sein und Netzwerkzugriff auf `re.jrc.ec.europa.eu` bestehen.
- Die BESS-Only-Szenarien fetchen ebenfalls PVGIS-Daten (für die Location), auch wenn `peak_power_kwp = 0` – die PV-Timeseries wird dann als Nullen verwendet. Alternativ kann der PVGIS-Fetch bei `pv_peak_kwp = 0` übersprungen werden (siehe FIX-S2-03).

---

## Abhängigkeiten

### Voraussetzungen

| Abhängigkeit | Beschreibung | Status |
|---|---|---|
| FIX-S2-03 | BESS-Only Cases (`peak_power_kwp: 0`) | OFFEN – erforderlich für 12 BESS-Only-Szenarien |
| `@pytest.mark.integration` | Marker in `pyproject.toml` | ERLEDIGT (existiert) |
| PVGIS-Cache | PV-Daten für Location 53.55/9.99 im `.data/pvgis_cache/` | Einmaliger Fetch beim ersten Lauf |

---

## Performance-Abschätzung

| Komponente | Aufwand pro Szenario | Gesamt (36 Szenarien) |
|---|---|---|
| Dispatch (1 Punkt × 3 Jahre × 365 Tage) | ~1.095 LP-Solves | ~39.420 LP-Solves |
| LP-Solve-Zeit | ~1ms/Solve | ~40 Sekunden |
| Overhead (IO, Cashflow, etc.) | ~1s/Szenario | ~36 Sekunden |
| **Gesamt** | | **~1–2 Minuten** |

Kein Grid-Search-Overhead. Mit `max_workers=1` (seriell, für Determinismus in Tests).

---

## Zusammenfassung

| Aspekt | Detail |
|---|---|
| Anzahl Szenarien | 36 (3 Tech × 2 Modi × 6 Marketing) |
| Marketing-Strategien | Market, EEG, PPA-Pay-as-Produced, PPA-Baseload, PPA-Floor, PPA-Collar |
| Grid-Search | **Deaktiviert** – feste BESS-Konfiguration per CLI-Override |
| Master-Szenario | Einheitliche Basis, Modifikation per `deepcopy` |
| PVGIS-Daten | Echte Daten aus lokalem Cache (kein Mock) |
| Dispatch-Checker | Prüft 9 Constraint-Kategorien stündlich (inkl. PV-Offline-Tage) |
| Availability-Checker | Zählt PV- und BESS-Offline-Tage, prüft gegen konfigurierte Availability |
| KPI-Ranking | 9 Ranking-Tests mit ökonomischer Begründung |
| Laufzeit | ~1–2 Minuten (alle 36 Szenarien) |
| Test-Tag | `@pytest.mark.integration` |
| Abhängigkeit | FIX-S2-03 (BESS-Only) muss implementiert sein |
