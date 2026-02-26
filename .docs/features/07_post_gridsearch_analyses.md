# Feature 07: Post-Grid-Search Sensitivitätsanalysen

## Priorität: Mittel-Hoch
## Aufwand: Groß (8-12h)

## Beschreibung

Nach der Grid-Search und Identifikation der optimalen BESS-Dimensionierung sollen
drei weitere Analysen durchgeführt werden. Jede Analyse variiert einen bestimmten
Vermarktungsparameter und führt für jede Variante eine vollständige Monte-Carlo-Simulation
durch. Die Ergebnisse werden als CSV exportiert.

**Laufzeitbudget:** Solange die Gesamtlaufzeit aller Berechnungen < 60h bleibt, sind
keine weiteren Runtime-Optimierungen nötig. Falls die Berechnung länger dauert, kann
der User die MC-Iterationen reduzieren.

## Die drei Analysen

### Analyse 1: EEG-Zuschlagspreis-Sensitivität

**Fragestellung:** Um wie viel Prozent sinkt der IRR, wenn der EEG-Zuschlagspreis sinkt?

**Parameter-Sweep:**
- EEG-Zuschlagspreis variieren in einem Bereich um den Basis-Zuschlagspreis
- Z.B.: Basis = 7.35 ct/kWh → Sweep: [5.0, 5.5, 6.0, 6.5, 7.0, 7.35, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0] ct/kWh
- Sweep-Werte als User-Input (Liste in der JSON)

**Für jeden Zuschlagspreis:**
1. BESS-Dimensionierung aus Grid-Search-Optimum verwenden
2. Alle anderen Parameter unverändert
3. Volle MC-Simulation durchführen
4. Ergebnis: Mean IRR, Std IRR, P10, P50, P90

**JSON-Erweiterung:**
```json
"analyses": {
    "eeg_sensitivity": {
        "enabled": true,
        "floor_prices_eur_per_kwh": [0.050, 0.055, 0.060, 0.065, 0.070, 0.0735, 0.075, 0.080, 0.085, 0.090]
    }
}
```

**CSV-Output: `{scenario_name}_eeg_sensitivity.csv`**

| Spalte | Beschreibung |
|--------|-------------|
| floor_price_eur_per_kwh | EEG-Zuschlagspreis |
| mc_iterations | Anzahl MC-Iterationen |
| equity_irr_mean | Mittlerer Equity IRR |
| equity_irr_std | Standardabweichung |
| equity_irr_p10 | 10. Perzentil |
| equity_irr_p50 | Median |
| equity_irr_p90 | 90. Perzentil |
| project_irr_mean | Mittlerer Project IRR |
| npv_mean | Mittlerer NPV |
| dscr_min_mean | Mittlerer Min DSCR |

### Analyse 2: PPA-Collar-Optimierung

**Fragestellung:** Wie muss ein PPA-Collar-Modell designt sein, um sowohl dem Erzeuger
als auch dem Kunden den größten Mehrwert zu bieten?

**Parameter-Sweep (2D):**
- PPA Floor-Preis: Sweep-Werte aus User-Input
- PPA Cap-Preis: Floor + Aufschlag (z.B. +2, +5, +X EUR/MWh)
- PPA-Laufzeit: Fixer Parameter aus User-Input

**JSON-Erweiterung:**
```json
"analyses": {
    "ppa_collar": {
        "enabled": true,
        "floor_prices_eur_per_mwh": [40, 45, 50, 55, 60, 65, 70, 75, 80],
        "cap_spreads_eur_per_mwh": [2, 5, 10, 15, 20],
        "duration_years": 10,
        "inflation_on_ppa": false,
        "goo_premium_eur_per_kwh": 0.005
    }
}
```

**Cap-Berechnung:** `cap_price = floor_price + cap_spread`

**Für jede (Floor, Cap-Spread)-Kombination:**
1. BESS-Dimensionierung aus Grid-Search-Optimum
2. Marketing-Typ auf "PPA Collar" setzen (temporär, nicht persistent)
3. Volle MC-Simulation
4. Ergebnis: IRR-Statistiken

**CSV-Output: `{scenario_name}_ppa_collar.csv`**

| Spalte | Beschreibung |
|--------|-------------|
| floor_price_eur_per_mwh | PPA Floor-Preis |
| cap_spread_eur_per_mwh | Aufschlag Floor → Cap |
| cap_price_eur_per_mwh | Resultierender Cap-Preis |
| duration_years | PPA-Laufzeit |
| equity_irr_mean | Mittlerer Equity IRR |
| equity_irr_std | Standardabweichung |
| equity_irr_p10 | 10. Perzentil |
| equity_irr_p50 | Median |
| equity_irr_p90 | 90. Perzentil |
| project_irr_mean | Mittlerer Project IRR |
| npv_mean | Mittlerer NPV |

### Analyse 3: PPA-Baseload-Optimierung

**Fragestellung:** Wie muss der PPA-Baseload designt sein, damit er beiden Seiten
den größten Mehrwert bietet?

**Parameter-Sweep (2D):**
- PPA-Preis: Sweep-Werte aus User-Input
- Baseload-Level: Verschiedene MW-Werte aus User-Input
- PPA-Laufzeit: Fixer Parameter aus User-Input

**JSON-Erweiterung:**
```json
"analyses": {
    "ppa_baseload": {
        "enabled": true,
        "ppa_prices_eur_per_mwh": [40, 50, 55, 60, 65, 70, 75, 80, 90],
        "baseload_levels_mw": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "duration_years": 10,
        "inflation_on_ppa": false,
        "goo_premium_eur_per_kwh": 0.005
    }
}
```

**Für jede (PPA-Preis, Baseload-MW)-Kombination:**
1. BESS-Dimensionierung aus Grid-Search-Optimum
2. Marketing-Typ auf "PPA Baseload" setzen
3. Volle MC-Simulation
4. Ergebnis: IRR-Statistiken

**CSV-Output: `{scenario_name}_ppa_baseload.csv`**

| Spalte | Beschreibung |
|--------|-------------|
| ppa_price_eur_per_mwh | PPA-Preis |
| baseload_mw | Baseload-Level in MW |
| duration_years | PPA-Laufzeit |
| equity_irr_mean | Mittlerer Equity IRR |
| equity_irr_std | Standardabweichung |
| equity_irr_p10 | 10. Perzentil |
| equity_irr_p50 | Median |
| equity_irr_p90 | 90. Perzentil |
| project_irr_mean | Mittlerer Project IRR |
| npv_mean | Mittlerer NPV |

## Architektur

### Neues Modul: `optimization/analyses.py`

```python
"""Post-grid-search sensitivity analyses with MC simulations."""

@dataclass
class AnalysisResult:
    """Result of a single analysis point (one parameter combination + MC)."""
    params: dict[str, float]
    mc_result: MCResult

@dataclass
class SensitivityResult:
    """Complete result of a sensitivity analysis."""
    analysis_type: str  # "eeg_sensitivity", "ppa_collar", "ppa_baseload"
    points: list[AnalysisResult]

def run_eeg_sensitivity(
    base_config: GridSearchConfig,
    optimal: GridPointResult,
    mc_params: MCParams,
    scenario_prices: dict[str, list[np.ndarray]],
    floor_prices: list[float],
) -> SensitivityResult:
    """EEG-Zuschlagspreis-Sensitivitätsanalyse."""
    for price in floor_prices:
        # Modified fixed_prices_yearly mit neuem Floor
        modified_config = _modify_eeg_price(base_config, price)
        mc_result = run_monte_carlo(modified_config, optimal, mc_params, scenario_prices)
        # Speichere Ergebnis

def run_ppa_collar_analysis(
    base_config: GridSearchConfig,
    optimal: GridPointResult,
    mc_params: MCParams,
    scenario_prices: dict[str, list[np.ndarray]],
    floor_prices: list[float],
    cap_spreads: list[float],
    duration_years: int,
    inflation_on_ppa: bool,
    goo_premium: float,
) -> SensitivityResult:
    """PPA-Collar-Optimierungsanalyse."""
    for floor in floor_prices:
        for spread in cap_spreads:
            cap = floor + spread
            modified_config = _modify_ppa_collar(base_config, floor, cap, duration_years, ...)
            mc_result = run_monte_carlo(modified_config, optimal, mc_params, scenario_prices)

def run_ppa_baseload_analysis(...) -> SensitivityResult:
    """PPA-Baseload-Optimierungsanalyse."""
    # Analog zu Collar
```

### Parallelisierung

Jeder Analyse-Punkt (eine Parameterkombination) ist unabhängig.
Die MC-Simulation innerhalb jedes Punktes ist bereits parallelisiert.
Zusätzliche äußere Parallelisierung ist möglich, aber ggf. nicht nötig
(da MC bereits multiprocessing nutzt).

**Laufzeitabschätzung (Beispiel):**
- EEG: 10 Preise × 1.000 MC × 25 Jahre × 365 Tage × 96 Intervalle = ~33 Mrd. LP-Solves
  → Bei 2ms/Solve: ~18h (akzeptabel)
- PPA Collar: 9 Floors × 5 Spreads × 1.000 MC = 45 Punkte → ~82h (grenzwertig)
  → Lösung: MC-Iterationen auf 200-500 reduzieren, oder weniger Sweep-Punkte

### Integration in main.py

Nach dem regulären MC-Block (Schritt 6) einen neuen Block einfügen:

```python
# Step 6b: Post-Grid-Search Analyses
analyses_cfg = scenario.raw.get("analyses", {})

if analyses_cfg.get("eeg_sensitivity", {}).get("enabled", False):
    logger.info("Running EEG sensitivity analysis...")
    eeg_result = run_eeg_sensitivity(...)
    write_eeg_sensitivity_csv(output_dir / f"{scenario.name}_eeg_sensitivity.csv", eeg_result)

if analyses_cfg.get("ppa_collar", {}).get("enabled", False):
    logger.info("Running PPA Collar analysis...")
    collar_result = run_ppa_collar_analysis(...)
    write_ppa_collar_csv(output_dir / f"{scenario.name}_ppa_collar.csv", collar_result)

if analyses_cfg.get("ppa_baseload", {}).get("enabled", False):
    logger.info("Running PPA Baseload analysis...")
    baseload_result = run_ppa_baseload_analysis(...)
    write_ppa_baseload_csv(output_dir / f"{scenario.name}_ppa_baseload.csv", baseload_result)
```

## Betroffene Dateien

| Datei | Änderung |
|-------|----------|
| `config/schema.py` | `analyses`-Block im JSON-Schema |
| `optimization/analyses.py` | **NEU**: Analyse-Funktionen |
| `output/csv_writer.py` | 3 neue CSV-Writer-Funktionen |
| `main.py` | Analyse-Aufrufe nach Grid Search + MC |

## Abhängigkeiten

- Feature 01 (PPA Collar Bug Fix): Muss vorher implementiert sein
- Feature 06 (Szenario-Mapping): Wenn 9 Szenarien, dann MC mit passenden PV-Zeitreihen

## Tests

- EEG-Sensitivität: Höherer Floor → höherer IRR (monoton steigend)
- PPA Collar: Floor = 0, Cap = 0 → entspricht purem Marktpreis
- PPA Collar: Floor = Cap → entspricht fixem Preis
- PPA Baseload: Baseload = 0 MW → keine Baseload-Verpflichtung
- Alle Analysen: MC mit sigma=0 → deterministisches Ergebnis
- CSV-Output: Korrekte Spaltenanzahl und -namen
- Laufzeit: Stichprobenartig messen (< 60h für typische Konfiguration)
