# Feature 06: Preisszenario-Wetterjahr-Mapping mit 15min-Auflösung

## Priorität: Hoch
## Aufwand: Groß (8-12h)

## Beschreibung

Grundlegende Umstellung des Ertrags- und Preismodells:

1. **9 Szenarien** (statt 3): Von "Low Price / Bad Weather" bis "High Price / Good Weather"
2. **Feste Wetterjahre pro Szenario**: Jedes Preisszenario hat ein zugeordnetes Wetterjahr,
   das den PV-Ertrag bestimmt
3. **Wochentag-Alignment**: Die PV-Ertragszeitreihe des Wetterjahres wird so verschoben,
   dass der Startwochentag zum Prognosejahr passt
4. **15-Minuten-Auflösung** (35.040 Werte/Jahr): Die PV-Erträge werden gleichverteilt
   auf 15min-Intervalle aufgeteilt
5. **Kein P50/P90 mehr**: Es werden nur noch spezifische Wetterjahre heruntergeladen,
   keine historische Aggregation
6. **Ein Szenario als "Central"** markiert: Wird für die Grid-Search verwendet

## Ist-Zustand

### PV-Zeitreihe
- PVGIS liefert stündliche Daten (8.760 Werte/Jahr) für alle verfügbaren historischen Jahre
- P50/P90 wird aus allen Jahren berechnet
- P50 für Grid Search, P90 für Debt-Sizing

### Preisszenarien
- 3 Szenarien (low/mid/high) als CSV-Spalten
- Gewichtung für MC-Sampling
- Mid-Spalte für Grid Search

### Verbindung
- Keine direkte Kopplung zwischen Preisszenario und Wetterjahr
- PV-Yield-Faktor als stochastischer Noise im MC

## Soll-Zustand

### 1. JSON-Schema: Neue Szenario-Struktur

```json
"price_inputs": {
    "price_csv": "data/forward_curves.csv",
    "price_unit": "eur_per_mwh",
    "inflation_on_input_data": true,
    "forecast_year": 2030,
    "scenarios": [
        {
            "name": "low_bad",
            "label": "Low Price / Bad Weather",
            "csv_column": "LOW_BAD",
            "weather_year": 2021,
            "weight": 0.05,
            "is_central": false
        },
        {
            "name": "low_mid",
            "label": "Low Price / Mid Weather",
            "csv_column": "LOW_MID",
            "weather_year": 2018,
            "weight": 0.10,
            "is_central": false
        },
        {
            "name": "mid_mid",
            "label": "Mid Price / Mid Weather",
            "csv_column": "MID_MID",
            "weather_year": 2019,
            "weight": 0.20,
            "is_central": true
        },
        // ... weitere 6 Szenarien
    ]
}
```

**Validierung:**
- Genau ein Szenario muss `is_central: true` haben
- Weights müssen sich zu 1.0 summieren
- Alle `csv_column` müssen in der CSV-Datei vorhanden sein
- `weather_year` muss in PVGIS verfügbar sein
- `forecast_year` ist das Jahr, auf dem die Preisprogrose basiert (bestimmt den Startwochentag)

### 2. Wochentag-Alignment der PV-Ertragszeitreihe

**Algorithmus:**

```python
import datetime

def align_weather_to_forecast_year(
    weather_timeseries: np.ndarray,  # 8760 Werte (stündlich)
    weather_year: int,               # z.B. 2017
    forecast_year: int,              # z.B. 2030
) -> np.ndarray:
    """Verschiebe die Wetter-Zeitreihe, damit der Wochentag zum Prognosejahr passt."""

    # Wochentag des 01.01. ermitteln (0=Montag, 6=Sonntag)
    dow_weather = datetime.date(weather_year, 1, 1).weekday()
    dow_forecast = datetime.date(forecast_year, 1, 1).weekday()

    # Shift in Tagen: Wie viele Tage muss die Wetter-Zeitreihe nach vorne verschoben werden?
    # Wir suchen den ersten Tag im Wetterjahr, der denselben Wochentag wie der 01.01. des Prognosejahres hat
    shift_days = (dow_forecast - dow_weather) % 7

    # Schaltjahr-Handling: Immer auf 365 Tage (8760 Stunden) normieren
    # Falls weather_year ein Schaltjahr ist: 31.12. (letzte 24h) ignorieren
    ts = weather_timeseries[:8760]  # Auf 8760 kürzen falls nötig

    # Verschiebung: Die ersten shift_days Tage werden ans Ende angehängt
    shift_hours = shift_days * 24
    if shift_hours > 0:
        aligned = np.concatenate([ts[shift_hours:], ts[:shift_hours]])
    else:
        aligned = ts.copy()

    return aligned
```

**Beispiel (aus FEATURES.md):**
- Prognosejahr 2030: 01.01.2030 = Dienstag (weekday=1)
- Wetterjahr 2017: 01.01.2017 = Sonntag (weekday=6)
- shift_days = (1 - 6) % 7 = 2
- 03.01.2017 (Dienstag) wird auf 01.01.2030 gemappt
- 01./02.01.2017 werden auf 30./31.12.2030 gemappt

### 3. 15-Minuten-Auflösung

Die stündlichen PVGIS-Daten (8.760 Werte) werden auf 15-Minuten-Intervalle aufgeteilt:
```python
def hourly_to_quarter_hourly(hourly_ts: np.ndarray) -> np.ndarray:
    """Konvertiere stündliche PV-Zeitreihe in 15min-Intervalle.

    Jeder Stundenwert wird gleichverteilt auf 4 Intervalle aufgeteilt (/ 4).
    """
    return np.repeat(hourly_ts / 4.0, 4)  # 8760 → 35040
```

**Folgeänderungen für 15min-Auflösung:**

| Konstante | Alt | Neu |
|-----------|-----|-----|
| `HOURS_PER_YEAR` | 8760 | bleibt (aber neuer: `INTERVALS_PER_YEAR = 35040`) |
| `HOURS_PER_DAY` | 24 | bleibt (aber neuer: `INTERVALS_PER_DAY = 96`) |
| `TIMESTEP_HOURS` | 1.0 (implizit) | `0.25` |

**Wichtig für den LP-Optimizer:**
- Der LP löst pro Tag mit 96 statt 24 Timesteps
- Variablenanzahl vervierfacht sich: Green Mode ~288 statt ~72 Variablen
- Power-Limits müssen auf 15min angepasst werden:
  `charge_pv[t] ≤ P_max_charge × 0.25`  (kWh in 15min = kW × 0.25h)
- Solve-Time pro Tag steigt (geschätzt 2-4ms statt <1ms)
- Gesamtlaufzeit Grid Search: ~4× länger, aber noch akzeptabel

**Preisdaten:** Die CSV muss ebenfalls 35.040 Werte pro Jahr enthalten (15min-Auflösung).
Falls die Preisdaten stündlich sind, wird jeder Stundenwert auf 4 Intervalle kopiert
(gleicher Wert, NICHT geteilt - Preise sind pro kWh, nicht pro Intervall).

### 4. PVGIS-Download: Nur spezifische Wetterjahre

Statt aller verfügbaren Jahre werden nur die im JSON referenzierten Wetterjahre heruntergeladen:

```python
# Alt:
yearly_pvgis = client.fetch_hourly_production(...)  # Alle Jahre

# Neu:
required_weather_years = {s["weather_year"] for s in scenarios}
weather_timeseries = {}
for year in required_weather_years:
    weather_timeseries[year] = client.fetch_single_year(
        year=year,
        system_loss_pct=0.0,  # Feature 02!
        ...
    )
```

**PVGIS API:** Die `seriescalc`-Endpoint unterstützt `startyear` und `endyear` Parameter.
Für ein einzelnes Jahr: `startyear=2017&endyear=2017`.

**Cache:** Der Cache-Key muss das Jahr enthalten.

### 5. P50/P90 entfernt

- `pv/timeseries.py` (`compute_p50_p90`) wird für diesen Workflow nicht mehr verwendet
- Stattdessen: Pro Szenario ein festes Wetterjahr → eine feste PV-Zeitreihe
- **Kein P90 für Debt-Sizing:** Das Central-Szenario wird für die Debt-Berechnung verwendet
  (oder alternativ das konservativste Szenario)

### 6. Integration in Grid Search

- Grid Search verwendet das **Central-Szenario** (Preise + Wetterjahr)
- Eine PV-Zeitreihe: aligned + degradiert + 15min

### 7. Integration in Monte Carlo

- MC sampelt Szenarien (mit Gewichtung), wie bisher
- **Aber:** Jedes Szenario bringt seine eigene PV-Zeitreihe mit (aus dem Wetterjahr)
- Der PV-Yield-Faktor (`sigma_pv_yield`) entfällt → ersetzt durch PV-Verfügbarkeit (Feature 05)

```python
# In _run_mc_iteration:
scenario_name = rng.choice(scenario_names, p=weights)
spot_prices_yearly = scenario_prices[scenario_name]
pv_base_timeseries = scenario_pv_timeseries[scenario_name]  # NEU: pro Szenario
```

### 8. Schaltjahr-Handling

Wenn entweder Wetterjahr oder Prognosejahr ein Schaltjahr ist:
- 31.12. (letzte 24 Stunden / 96 Intervalle) wird ignoriert
- Alle Zeitreihen haben exakt 365 Tage = 8.760 Stunden = 35.040 Intervalle

## Datenfluss (Zusammenfassung)

```
JSON → Szenarien-Liste (9 Einträge)
  ├── Für jedes Szenario:
  │   ├── Preis-CSV laden (Spalte laut csv_column)
  │   ├── PVGIS-Download (weather_year, system_loss=0%)
  │   ├── Wochentag-Alignment (weather_year → forecast_year)
  │   └── 15min-Konvertierung (8760 → 35040)
  │
  ├── Central-Szenario → Grid Search
  │   └── 1 Preiszeitreihe + 1 PV-Zeitreihe → LP-Dispatch
  │
  └── Alle Szenarien → Monte Carlo
      └── Pro Iteration: 1 Szenario sampeln → Preis + PV → LP-Dispatch
```

## Betroffene Dateien

| Datei | Änderung |
|-------|----------|
| `config/schema.py` | Neues Szenario-Schema, `forecast_year`, `scenarios`-Array |
| `config/defaults.py` | `INTERVALS_PER_YEAR = 35040`, `INTERVALS_PER_DAY = 96`, `TIMESTEP_HOURS = 0.25` |
| `config/loader.py` | Neue Lade-Logik für Szenarien-Array, Validierung |
| `pv/pvgis_client.py` | `fetch_single_year()` Methode, `system_loss_pct=0` |
| `pv/timeseries.py` | `align_weather_to_forecast_year()`, `hourly_to_quarter_hourly()` |
| `dispatch/optimizer.py` | T=96 statt T=24, Timestep-Faktor 0.25 in Power-Limits |
| `dispatch/engine.py` | 35040 Intervalle, 96 pro Tag, PV-Zeitreihe pro Szenario |
| `market/price_loader.py` | 15min-Preisdaten laden, Validierung auf 35040 |
| `main.py` | Kompletter Umbau des Szenario-Handlings |
| `optimization/grid_search.py` | Central-Szenario für Grid Search |
| `optimization/monte_carlo.py` | PV-Zeitreihe pro Szenario, kein PV-Yield-Faktor |
| `output/csv_writer.py` | Dispatch-Sample mit 35040 Zeilen |

## Migration / Abwärtskompatibilität

Die alte JSON-Struktur (`price_scenarios` mit 3 Einträgen) soll weiterhin unterstützt werden.
Erkennung: Wenn `scenarios`-Array fehlt, wird die alte Logik verwendet (P50/P90, stündlich).

## Tests

- Wochentag-Alignment: Bekanntes Beispiel (2017 → 2030) verifizieren
- Schaltjahr-Handling: 2020 (Schaltjahr) → 2030, 31.12. ignoriert
- 15min-Konvertierung: Summe bleibt gleich (Σ hourly = Σ quarter-hourly)
- LP mit 96 Timesteps: Ergebnis konsistent mit 24-Timestep-Lösung (gleiche Tagesproduktion)
- Central-Szenario korrekt identifiziert
- MC sampelt alle 9 Szenarien laut Gewichtung
- PV-Zeitreihe pro Szenario korrekt zugeordnet
- Validierung: Genau ein `is_central`, Weights = 1.0
