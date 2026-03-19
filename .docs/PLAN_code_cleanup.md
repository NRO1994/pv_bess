# Code Cleanup Plan – Phase B (Redundanzen, toter Code, Performance)

## Grundregel

**Keine Logik-Änderungen.** Alle bestehenden Unit-Tests müssen vor UND nach jeder Änderung grün sein. Dies werde ICH
manuell prüfen, deine Aufgabe ist es nur die direkt betroffenen Unit-Tests anzupassen.

---

## Übersicht der Findings

| Kategorie                                       | Schwere  | Anzahl                       | Haupt-Dateien                        |
|-------------------------------------------------|----------|------------------------------|--------------------------------------|
| **Revenue-Berechnung 4x redundant + Bug**       | Kritisch | 4 Code-Stellen, 1 Bug        | `dispatch/optimizer.py`, `engine.py` |
| Toter Code (nur in Tests aufgerufen)            | Kritisch | 4 Funktionen                 | `market/eeg.py`, `market/ppa.py`     |
| Monolithische Funktion                          | Kritisch | 1 Funktion (694 Zeilen)      | `main.py:run()`                      |
| Duplizierter Code-Block                         | Hoch     | 1 identischer Block          | `main.py:917-926 vs 945-954`         |
| Vermarktungslogik in main.py statt Marktmodulen | Hoch     | 3 Funktionen (~155 Zeilen)   | `main.py:198-356`                    |
| Redundante Analyse-CSV-Writer                   | Mittel   | 3 fast-identische Funktionen | `output/csv_writer.py:543-701`       |
| Unnötige Array-Kopien                           | Mittel   | 5 Stellen                    | `optimizer.py`, `main.py`            |
| Tief verschachtelte Kontrollflüsse              | Mittel   | 1 Block                      | `main.py:668-728`                    |
| Test-only Hilfsfunktionen                       | Niedrig  | 2 Funktionen                 | `market/ppa.py`                      |

---

## B1: Toter Code entfernen

### B1.1: `market/eeg.py` – `apply_eeg_floor()` (Zeile 100-122)

**Status:** Wird nur in `tests/test_eeg.py` aufgerufen. Kein Aufruf in Produktionscode.

**Grund:** Die Floor-Pricing-Logik wird in `main.py:_build_fixed_prices_yearly()` vorab berechnet und
als `fixed_prices_yearly`-Array an den Dispatch-Optimizer übergeben. `apply_eeg_floor()` war die
ursprüngliche Post-hoc-Berechnung, die durch die Pre-Computation ersetzt wurde.

**Aktion:**

- `apply_eeg_floor()` aus `market/eeg.py` entfernen
- Zugehörige Tests in `tests/test_eeg.py` entfernen (Tests testen toten Code)
- Prüfen ob Imports in `test_eeg.py` angepasst werden müssen

**Risiko:** Gering. Keine Produktions-Aufrufe.

---

### B1.2: `market/ppa.py` – `apply_pay_as_produced()` (Zeile 174-198)

**Status:** Wird nur in `tests/test_ppa.py` aufgerufen. Kein Aufruf in Produktionscode.

**Grund:** Pay-as-Produced wird als fester Preis über `_build_fixed_prices_yearly()` in `main.py`
vorab berechnet. Die Funktion `apply_pay_as_produced()` multipliziert lediglich `production × (price + goo)`,
was der Optimizer bereits intern erledigt.

**Aktion:**

- `apply_pay_as_produced()` aus `market/ppa.py` entfernen
- Zugehörige Tests in `tests/test_ppa.py` entfernen
- Docstring am Modulanfang aktualisieren (Public API Section)

**Risiko:** Gering. Keine Produktions-Aufrufe.

---

### B1.3: `market/ppa.py` – `apply_floor_ppa()` (Zeile 322-356)

**Status:** Wird nur in `tests/test_ppa.py` aufgerufen. Kein Aufruf in Produktionscode.

**Grund:** Floor-PPA-Effektivpreis wird ebenfalls über `_build_fixed_prices_yearly()` vorab berechnet.
Die Funktion `apply_floor_ppa()` berechnet `max(spot, floor) + goo`, aber der Optimizer macht dies
intern über die pre-computed `effective_green_price`.

**Aktion:**

- `apply_floor_ppa()` aus `market/ppa.py` entfernen
- Zugehörige Tests in `tests/test_ppa.py` entfernen
- `effective_floor_price()` (Zeile 286-319) BLEIBT – wird von `effective_ppa_price_for_year()` genutzt

**Risiko:** Gering. Keine Produktions-Aufrufe.

---

### B1.4: `market/ppa.py` – `apply_collar_ppa()` (Zeile 405-449)

**Status:** Wird nur in `tests/test_ppa.py` aufgerufen. Kein Aufruf in Produktionscode.

**Grund:** Collar-PPA wird über `_build_fixed_prices_yearly()` + `_build_cap_prices_yearly()`
vorab berechnet. Der Optimizer wendet `clip(spot, floor, cap) + goo` intern an.

**Aktion:**

- `apply_collar_ppa()` aus `market/ppa.py` entfernen
- Zugehörige Tests in `tests/test_ppa.py` entfernen
- `effective_collar_prices()` (Zeile 364-402) BLEIBT – wird von `effective_ppa_price_for_year()` genutzt

**Risiko:** Gering. Keine Produktions-Aufrufe.

---

### B1.5: `market/price_loader.py` – `collect_scenario_columns`

**Status:** Bereits entfernt (laut Fehler in `test_price_loader.py`). Nur noch in der
`.docs/features/11_sanitizing_testing.md` erwähnt.

**Aktion:**

- Bestätigen, dass Funktion tatsächlich bereits entfernt ist
- Import-Referenz in `tests/test_price_loader.py` entfernen (Phase A, bereits dokumentiert)

---

## B2: Duplizierter Code-Block in `main.py`

### B2.1: Identische `baseline_market_config`-Konstruktion (Zeile 917-926 vs 945-954)

**Status:** Exakt identischer `dataclasses.replace()`-Aufruf in beiden Zweigen eines if/else.

**Code (identisch in beiden Blöcken):**

```python
baseline_market_config = _dc.replace(
    grid_search_config,
    scale_pct_of_pv=[optimal_setup.scale_pct],
    e_to_p_ratio_hours=[optimal_setup.e_to_p_ratio],
    fixed_prices_yearly=[0.0] * lifetime,
    goo_prices_yearly=[0.0] * lifetime,
    cap_prices_yearly=[0.0] * lifetime,
    baseload_mw=0,
    skip_baseline=True,
)
```

**Aktion:**

- `baseline_market_config`-Konstruktion VOR das if/else ziehen
- Nur die MC- vs Grid-Search-Logik im if/else belassen:
  ```python
  baseline_market_config = _build_baseline_market_config(grid_search_config, optimal_setup, lifetime)
  if need_mc_params and mc_params is not None:
      baseline_mc_result = run_monte_carlo(...)
  else:
      baseline_result = run_grid_search(baseline_market_config)
  ```

**Risiko:** Sehr gering. Rein strukturelle Änderung, keine Logik-Änderung.

---

### B2.2: Inline-Import `import dataclasses as _dc` (Zeile 911)

**Status:** `import dataclasses as _dc` steht mitten im Funktionskörper von `run()`.

**Aktion:**

- Import an den Modulanfang verschieben (PEP 8)

**Risiko:** Null.

---

## B3: Vermarktungslogik aus `main.py` in Marktmodule verschieben

### B3.1: `_build_fixed_prices_yearly()` (main.py Zeile 198-266, 69 Zeilen)

**Status:** Enthält if/elif-Kaskade für EEG, PPA_FLOOR, PPA_COLLAR, PPA_PAP, PPA_BASELOAD.
Ruft intern bereits `eeg_config_from_dict()`, `ppa_config_from_dict()`, `effective_eeg_price()`,
`pay_as_produced_price()`, `inflate_value()` auf.

**Problem:** `main.py` kennt die interne Logik aller Vermarktungstypen. Bei neuen PPA-Typen muss
`main.py` erweitert werden statt des zuständigen Marktmoduls.

**Aktion:**

- `eeg.py` erweitern: `get_floor_prices_yearly(eeg_config, lifetime, inflation_rate) -> list[float]`
- `ppa.py` erweitern: `get_fixed_prices_yearly(ppa_config, lifetime, inflation_rate) -> list[float]`
- `main.py` ruft nur noch die passende Funktion auf (Delegation statt Implementierung)
- Die if/elif-Kaskade wird zum einzeiligen Dispatch

**Betroffene Dateien:**

- `main.py:198-266` (Logik entfernen, durch Delegation ersetzen)
- `market/eeg.py` (neue Funktion hinzufügen)
- `market/ppa.py` (neue Funktion hinzufügen)

**Risiko:** Mittel. Logik wird 1:1 verschoben, aber Schnittstelle ändert sich. Tests müssen angepasst werden.

---

### B3.2: `_build_goo_prices_yearly()` (main.py Zeile 269-304, 36 Zeilen)

**Status:** Speziallogik für EEG-Jahre (keine GoO) und PPA-Jahre (mit GoO).

**Aktion:**

- `ppa.py` erweitern: `get_goo_prices_yearly(ppa_config, eeg_config, lifetime) -> list[float]`
- `main.py` delegiert

**Betroffene Dateien:**

- `main.py:269-304` (Logik verschieben)
- `market/ppa.py` (neue Funktion)

**Risiko:** Mittel. Gleiche Logik, andere Position.

---

### B3.3: `_build_cap_prices_yearly()` (main.py Zeile 307-356, 50 Zeilen)

**Status:** Speziallogik für Collar/PaP/Baseload Cap-Preise.

**Aktion:**

- `ppa.py` erweitern: `get_cap_prices_yearly(ppa_config, lifetime, inflation_rate) -> list[float]`
- `main.py` delegiert

**Betroffene Dateien:**

- `main.py:307-356` (Logik verschieben)
- `market/ppa.py` (neue Funktion)

**Risiko:** Mittel.

---

### B3.4: Zusammenführung aller drei Preisschedule-Funktionen

**Optionale Optimierung:** Statt 3 separate Funktionen könnte ein einziger Aufruf alle drei Listen bauen:

```python
# In market/pricing.py (oder ppa.py):
@dataclass
class YearlyPriceSchedules:
    fixed_prices: list[float]  # Floor/Fixed pro Jahr
    goo_prices: list[float]  # GoO-Prämie pro Jahr
    cap_prices: list[float]  # Cap-Preis pro Jahr


def build_yearly_price_schedules(scenario, inflation_rate) -> YearlyPriceSchedules:
    """Einmaliger Loop über alle Projektjahre, erzeugt alle 3 Listen."""
```

**Vorteil:** Ein Loop statt drei, sauberere API.
**Risiko:** Mittel – erfordert neue Tests für die zusammengeführte Funktion.

---

## B4: Redundante CSV-Writer zusammenführen

### B4.1: Drei fast-identische Analyse-CSV-Writer (csv_writer.py Zeile 543-701)

**Status:** `write_eeg_sensitivity_csv()`, `write_ppa_collar_csv()`, `write_ppa_baseload_csv()` folgen
exakt demselben Muster:

```python
def write_XXX_csv(path, result, config=None):
    cfg = config or CsvConfig()
    d = cfg.decimal
    rows = []
    for pt in result.points:
        stats = pt.mc_result.overall_stats
        eq = stats.get("equity_irr")
        proj = stats.get("project_irr")
        npv_s = stats.get("npv")
        rows.append({
            # Analyse-spezifische Spalten (2-4 Felder)
            # Identische Statistik-Spalten (8-9 Felder)
        })
    _write_dicts(path, rows, delimiter=cfg.delimiter)
```

**Unterschiede:**

- EEG: 1 Param-Spalte (`floor_price_eur_per_kwh`) + 1 Extra (`dscr_min_mean`)
- Collar: 3 Param-Spalten (`floor_price`, `cap_spread`, `cap_price`) + `duration_years`
- Baseload: 2 Param-Spalten (`ppa_price`, `baseload_mw`) + `duration_years`

**Aktion:**

- Gemeinsame Statistik-Spalten in Helper extrahieren:
  ```python
  def _mc_stats_columns(stats: dict, decimal: str) -> dict[str, str]:
      """Erzeugt die 8 identischen Statistik-Spalten."""
  ```
- Die drei Writer behalten ihre Signaturen (Abwärtskompatibilität), rufen aber intern den Helper auf
- Reduktion: ~160 Zeilen → ~80 Zeilen (Statistik-Block nur einmal definiert)

**Risiko:** Gering. Interne Refaktorierung, keine API-Änderung.

---

## B5: Unnötige Array-Kopien (Performance)

### B5.1: `optimizer.py` – Spot-Price-Kopien (Zeile 245, 771, 943)

**Status:** Drei Stellen mit `eff = spot_prices_eur_per_kwh.copy()`. Kopie wird nur benötigt, wenn
`price_fixed > 0` (dann wird `np.maximum()` berechnet) oder cap aktiv ist.

**Aktueller Code (Zeile 245):**

```python
eff = spot_prices_eur_per_kwh.copy()
if price_fixed_eur_per_kwh > 0.0:
    eff = np.maximum(eff, price_fixed_eur_per_kwh)
if goo_premium_eur_per_kwh > 0.0:
    eff = eff + goo_premium_eur_per_kwh
```

**Optimierung:**

```python
if price_fixed_eur_per_kwh > 0.0:
    eff = np.maximum(spot_prices_eur_per_kwh, price_fixed_eur_per_kwh)
else:
    eff = spot_prices_eur_per_kwh  # Referenz, keine Kopie
if goo_premium_eur_per_kwh > 0.0:
    eff = eff + goo_premium_eur_per_kwh  # erzeugt neues Array (kein in-place auf Referenz)
```

**Einsparung:** 1 Array-Allokation pro LP-Solve × 365 Tage × 25 Jahre × N Grid-Punkte.
Bei 96 Elementen pro Array (Quarter-hourly): vernachlässigbar, aber guter Code-Stil.

**Betroffene Stellen:**

- `optimizer.py:245` (Green-Mode effective price)
- `optimizer.py:771` (Baseload effective price)
- `optimizer.py:943` (PaP effective price)

**Risiko:** Gering, aber sorgfältig prüfen: Das neue `eff`-Array darf NICHT in-place modifiziert
werden, wenn es eine Referenz auf `spot_prices_eur_per_kwh` ist. Der `+ goo`-Operator erzeugt
ein neues Array, also ist dies sicher.

---

### B5.2: `optimizer.py` – Redundante `soc_green`-Kopie (Zeile 618)

**Status:** Im Green-Mode wird `soc_green=soc.copy()` gesetzt, obwohl `soc_green` und `soc`
identisch sind (es gibt nur eine SoC-Spur im Green-Mode).

**Code:**

```python
return DailyDispatchResult(
    ...
soc = soc,
soc_green = soc.copy(),  # Redundant: soc_green IS soc in Green Mode
soc_grey = np.zeros(n_steps + 1),
...
)
```

**Optimierung:** `soc_green=soc` (gleiche Referenz). Der Caller darf dann `soc_green` nicht
in-place modifizieren, aber das tut er bereits nicht (Werte werden nur gelesen).

**Prüfung erforderlich:** Suche nach `result.soc_green[...] = ` oder `result.soc_green +=` im
Caller-Code (engine.py). Wenn kein in-place Zugriff: sicher.

**Risiko:** Gering, aber Prüfung notwendig.

---

### B5.3: `main.py` – Price-Array-Kopie in Loop (Zeile 391)

**Status:** `year_prices = price_array[start:end].copy()`

**Kontext:**

```python
for y in range(1, lifetime_years + 1):
    start = (y - 1) * intervals_per_year
    end = y * intervals_per_year
    year_prices = price_array[start:end].copy()
    if apply_inflation:
        factor = inflate_value(1.0, inflation_rate, y)
        year_prices = year_prices * factor
    yearly.append(year_prices)
```

**Problem:** `.copy()` ist nur nötig, wenn `apply_inflation=True` (denn `* factor` erzeugt
sowieso ein neues Array). Wenn `apply_inflation=False`, ist die Kopie unnötig – ein Slice
reicht.

**Optimierung:**

```python
year_slice = price_array[start:end]
if apply_inflation:
    factor = inflate_value(1.0, inflation_rate, y)
    year_prices = year_slice * factor  # neues Array
else:
    year_prices = year_slice.copy()  # Kopie nötig, da Slice nur View ist
    # ODER: year_prices = year_slice (View reicht, wenn downstream nicht modifiziert wird)
```

**Ergebnis:** Keine Einsparung bei `apply_inflation=True`, aber die Intent-Klarheit verbessert sich.
Prüfen ob downstream `year_prices` in-place modifiziert wird.

**Risiko:** Gering.

---

## B6: Monolithische `run()`-Funktion aufteilen (main.py:476-1169)

### Status

694 Zeilen in einer einzigen Funktion. Enthält:

- Szenario-Laden und Validierung (Zeile 492-522)
- Parameter-Extraktion aus JSON (Zeile 524-637, **114 Zeilen** reine dict.get()-Aufrufe)
- PV-Daten Abruf und Zeitreihen-Setup (Zeile 642-728)
- Preisladen pro Szenario (Zeile 732-778)
- Grid-Search Konfiguration (Zeile 793-849)
- Grid-Search Ausführung (Zeile 851-870)
- MC/Baseline/Analyse-Orchestrierung (Zeile 875-1037)
- CSV-Output (Zeile 1041-1130)
- HTML-Report (Zeile 1132-1169)

### Vorgeschlagene Aufteilung

| Neue Funktion                                                | Zeilen aus `run()` | Beschreibung                                 |
|--------------------------------------------------------------|--------------------|----------------------------------------------|
| `_extract_scenario_params(scenario) -> ScenarioParams`       | 524-637            | Alle dict.get()-Aufrufe in Dataclass bündeln |
| `_fetch_pv_timeseries(params, scenarios) -> dict`            | 642-728            | PVGIS-Abruf + Quarter-hourly-Konvertierung   |
| `_load_price_data(params, scenarios) -> dict`                | 732-785            | Preisladen + Preisschedule-Erzeugung         |
| `_build_grid_search_config(params, ...) -> GridSearchConfig` | 793-849            | Config-Objekt zusammenbauen                  |
| `_run_baseline_market(config, optimal, mc_params) -> float`  | 908-962            | Baseline-Direktvermarktung (dedupliziert)    |
| `_write_all_outputs(results, output_dir, csv_config)`        | 1041-1130          | CSV-Schreiblogik                             |

### Neue Dataclass: `ScenarioParams`

Die 114 Zeilen Parameter-Extraktion (Zeile 524-637) erzeugen ~40 lokale Variablen.
Diese sollten in eine Dataclass zusammengefasst werden:

```python
@dataclass
class ScenarioParams:
    """Extrahierte Parameter aus einem validierten Szenario."""
    # Finance
    inflation_rate: float
    leverage_pct: float
    interest_rate_pct: float
    loan_tenor_years: int
    discount_rate: float
    debt_sizing_downside_pct: float
    # Tax
    afa_years_pv: int
    afa_years_bess: int
    gewerbesteuer_hebesatz: int
    gewerbesteuer_messzahl: float
    koerperschaftsteuer_pct: float
    solidaritaetszuschlag_pct: float
    # PV
    pv_peak_kwp: float
    pv_degradation_rate: float
    pv_availability_pct: float
    # BESS
    bess_rte: float
    bess_min_soc_pct: float
    bess_max_soc_pct: float
    bess_degradation_rate: float
    bess_availability_pct: float
    optimization_fee_pct: float
    # Replacement
    replacement_enabled: bool
    replacement_year: int
    replacement_config: ReplacementConfig | None
    # Grid
    grid_max_kw: float
    grid_loss_factor: float
    # Design Space
    scale_pct_list: list[float]
    e_to_p_list: list[float]
    skip_baseline: bool
    bess_absolute_power_kw: float | None
    bess_absolute_capacity_kwh: float | None
    # Timing
    lifetime: int
    commissioning_year: int
```

**Vorteil:**

- `run()` schrumpft um ~114 Zeilen
- Parameter werden als Einheit übergeben statt als 40 Einzelargumente
- Testbar: `_extract_scenario_params()` kann isoliert getestet werden

**Risiko:** Mittel. Alle Zugriffe auf die lokalen Variablen in `run()` müssen auf `params.xxx`
umgestellt werden. Viele kleine Änderungen, aber keine Logik-Änderung.

**Empfehlung:** Dies ist das größte Refactoring und sollte in einem eigenen Commit erfolgen.
Vorher alle Tests grün bestätigen, danach erneut bestätigen.

---

## B7: Tief verschachtelte Logik vereinfachen (main.py:668-728)

### Status

5 Verschachtelungsebenen für PV-Datenabruf:

```python
if pv_peak_kwp > 0 and scenarios_list:  # Level 1
    for wy in unique_weather_years:  # Level 2
        try:  # Level 3
            hourly_ts = client.fetch(...)
        except:  # Level 3
            ...
        qh_ts = hourly_to_quarter_hourly(...)
    for sc in scenarios_list:  # Level 2
        ...
    central_scenarios = [...]  # Level 2
    if central_scenarios:  # Level 3
        ...
    else:  # Level 3
        ...
else:  # Level 1
# BESS-Only fallback
```

### Aktion

Extrahieren in `_fetch_pv_timeseries()` (siehe B6). Die verschachtelte Logik wird zur
Hauptaufgabe dieser Funktion, und `run()` ruft nur noch auf:

```python
pv_data = _fetch_pv_timeseries(params, scenarios_list, client)
```

**Risiko:** Gering (Teil des B6-Refactorings).

---

## B8: Test-only Utility-Funktionen bewerten

### B8.1: `market/ppa.py` – `baseload_level_kwh()` (Zeile 206-233)

**Status:** Wird in `tests/test_ppa.py` aufgerufen und hat eine einzige Zeile Logik: `return baseload_mw * 1000.0`.

**Bewertung:** Triviale Konvertierung (MW → kW). Kann bleiben oder entfernt werden.
Der Dispatch-Optimizer berechnet dies inline.

**Empfehlung:** Behalten – ist dokumentiert und könnte in Post-MVP (PPA-Baseload im LP, Phase C3)
relevant werden.

---

### B8.2: `market/ppa.py` – `baseload_revenue()` (Zeile 236-278)

**Status:** Wird in `tests/test_ppa.py` aufgerufen. Berechnet
`baseload × (ppa_price + goo) + (export - baseload) × spot`.

**Bewertung:** Nützliche Validierungsfunktion für Tests. Könnte in Phase C3 (PPA-Baseload im LP)
wieder relevant werden.

**Empfehlung:** Behalten.

---

## B9: Revenue-Berechnung vereinheitlichen (optimizer.py + engine.py) ✅ DONE

### Status

Die tägliche Revenue-Berechnung wird an **vier** Stellen unabhängig voneinander implementiert:

1. `optimizer.py:_extract_green_result()` (Zeile 608-617) – **korrekt**
2. `optimizer.py:_extract_grey_result()` (Zeile 670-679) – **korrekt**
3. `optimizer.py:dispatch_offline_day()` (Zeile 978-987) – **korrekt**
4. `engine.py` (Zeile 605-622) – **FEHLERHAFT für Baseload PPA**

### Problem 1: Vierfache Redundanz

Alle vier Stellen berechnen eigenständig die Kennzahlen (Revenue PV, Revenue BESS Green, Revenue BESS Grey,
Import-Kosten, Shortfall-Kosten). Das verstößt gegen DRY und macht es unmöglich, eine konsistente Ergebniskalkulation zu
garantieren. Änderungen an der Revenue-Logik müssen an vier Stellen synchron durchgeführt werden.

### Problem 2: Falsche Baseload-Revenue in engine.py (Bug)

Die engine.py berechnet den Daily Revenue **immer** nach demselben Schema:

```python
# engine.py Zeile 606-608
day_rev_pv = float(np.sum(result["export_pv"] * result["effective_price"]))
day_rev_green = float(np.sum(result["discharge_green"] * result["effective_price"]))
day_rev_grey = float(np.sum(result["discharge_grey"] * spot_day))
```

Das ist korrekt für EEG, PPA Floor, PPA Collar und Market – aber **falsch für PPA Baseload**.

**Bei Baseload PPA** gilt (korrekte Berechnung im Optimizer):

```python
# optimizer.py _extract_green_result() Zeile 610-613
spot_revenue = max(export + discharge - baseload, 0) * spot_price
ppa_revenue = baseload_kwh * fixed_price
baseload_shortfall_cost = shortfall * spot_price
revenue = ppa_revenue + spot_revenue - baseload_shortfall_cost
```

Die engine.py wendet stattdessen den `effective_price` auf die **gesamte** Erzeugung an (`export_pv * eff` +
`discharge_green * eff`). Das ist falsch, weil:

- Nur der **Baseload-Anteil** zum PPA-Preis (`fixed_price`) verkauft wird
- Der **Überschuss** über dem Baseload wird zum **Spot-Preis** verkauft
- Ein **Shortfall** (Unterdeckung) muss zum **Spot-Preis** zugekauft werden

Der Optimizer berechnet das korrekt und gibt das Ergebnis als `result["revenue"]` zurück – die engine ignoriert diesen
Wert jedoch und berechnet den Revenue selbst nochmal (falsch).

### Auswirkung

Der Bug betrifft die **jährlichen Revenue-Breakdowns** (`revenue_pv_export`, `revenue_bess_green`) im `AnnualResult`.
Diese fließen in:

- Cashflow-Projektion (via `AnnualResult.total_revenue`)
- BESS Spot Revenue (für Optimization Fee Berechnung)
- Monte Carlo Post-hoc-Skalierung (PV vs. BESS Revenue-Aufteilung)

**Hinweis:** Der `total_revenue` in engine.py (Zeile 648-649) subtrahiert zwar `year_missing_baseload`, was den
Gesamtfehler teilweise kompensiert – die Einzelkomponenten (`revenue_pv_export`, `revenue_bess_green`) bleiben aber
falsch.

### Lösung

**Schritt 1:** Gemeinsame Revenue-Berechnungsfunktion erstellen:

```python
# optimizer.py (oder neues Modul dispatch/revenue.py)
def compute_daily_revenue(
        export_pv: np.ndarray,
        discharge_green: np.ndarray,
        discharge_grey: np.ndarray,
        charge_grid: np.ndarray,
        shortfall: np.ndarray,
        spot_prices: np.ndarray,
        eff_prices: np.ndarray,
        fixed_price: float,
        baseload_kwh: float,
) -> DailyRevenueBreakdown:
    """Einheitliche Revenue-Berechnung für alle Marketing-Typen.

    Returns
    -------
    DailyRevenueBreakdown
        revenue_pv, revenue_green, revenue_grey, import_cost,
        shortfall_cost, total_revenue, bess_spot_revenue
    """
```

**Schritt 2:** Alle vier Stellen auf diese Funktion umstellen:

- `_extract_green_result()` → ruft `compute_daily_revenue()` auf
- `_extract_grey_result()` → ruft `compute_daily_revenue()` auf
- `dispatch_offline_day()` → ruft `compute_daily_revenue()` auf
- `engine.py` Zeile 605-622 → nutzt `result["revenue_breakdown"]` aus dem Optimizer-Ergebnis statt eigener Berechnung

**Schritt 3:** `DailyDispatchResult` erweitern um strukturierte Revenue-Aufschlüsselung:

```python
@dataclass
class DailyRevenueBreakdown:
    revenue_pv: float  # PV-Einspeiseerlös
    revenue_green: float  # BESS Green Discharge Erlös
    revenue_grey: float  # BESS Grey Discharge Erlös
    import_cost: float  # Grid-Import-Kosten
    shortfall_cost: float  # Baseload-Unterdeckungskosten
    total_revenue: float  # Netto-Gesamterlös
    bess_spot_revenue: float  # BESS Spot-Revenue für Optimization Fee
```

**Schritt 4:** engine.py vereinfachen – die engine summiert nur noch die vom Optimizer gelieferten Breakdown-Werte auf,
ohne eigene Revenue-Kalkulation:

```python
# engine.py – NACH Refactoring
breakdown = result["revenue_breakdown"]
year_revenue_pv += breakdown.revenue_pv
year_revenue_green += breakdown.revenue_green
year_revenue_grey += breakdown.revenue_grey
year_import_cost += breakdown.import_cost
year_missing_baseload += breakdown.shortfall_cost
year_bess_spot_revenue += breakdown.bess_spot_revenue
```

### Betroffene Dateien

- `dispatch/optimizer.py` – Revenue-Berechnung in `_extract_green_result()`, `_extract_grey_result()`,
  `dispatch_offline_day()` durch gemeinsame Funktion ersetzen; `DailyDispatchResult` erweitern
- `dispatch/engine.py` – Revenue-Aufschlüsselung aus Optimizer-Ergebnis übernehmen statt eigene Berechnung
- `tests/test_optimizer.py` – Tests auf neue `DailyRevenueBreakdown`-Struktur anpassen
- `tests/test_dispatch_engine.py` – Revenue-Assertions anpassen

### Risiko

**Hoch.** Dies ist ein Bug-Fix + Refactoring in einem. Die Baseload-Revenue in engine.py ist falsch und muss korrigiert
werden. Gleichzeitig wird die Revenue-Berechnung zentralisiert, was die Code-Oberfläche für künftige Fehler reduziert.

### Priorität

**Kritisch** – höher als alle anderen Cleanup-Items, da ein tatsächlicher Berechnungsfehler vorliegt.

---

## B10: Simultanes Be-/Entladen verhindern (LP-Constraint)

### Status

Das LP erlaubt derzeit gleichzeitiges Laden (`charge_pv[t] > 0`) und Entladen (`discharge_green[t] > 0`) im selben
Zeitschritt. Die aktuelle Gegenmaßnahme – Entladung wird nur bei **negativen Spot-Preisen** auf 0 fixiert
(`optimizer.py:540`, `optimizer.py:711-722`) – hat sich als unzureichend herausgestellt. Simultanes Be-/Entladen
kann auch bei positiven Preisen auftreten.

### Warum passiert das?

In einem reinen LP gibt es kein "entweder-oder". Alle Variablen sind kontinuierlich, und der Solver kann für
denselben Zeitschritt `charge_pv[t] > 0` UND `discharge_green[t] > 0` setzen. Das passiert in folgenden Fällen:

1. **Degenerate Lösungen**: Wenn Laden+Entladen denselben Zielfunktionswert liefert wie der Netto-Fluss, kann der
   Solver eine beliebige der äquivalenten Lösungen wählen – inklusive einer mit simultanem Be-/Entladen.

2. **Floor-Pricing-Effekte**: Bei EEG/PPA-Floor ist `eff_price = max(spot, floor) + goo`. Wenn `spot < floor`, dann
   ist der effektive Preis für alle PV-Nutzungen gleich (`floor + goo`). Der Solver hat keinen Anreiz, zwischen
   direktem Export und Laden+Entladen zu unterscheiden.

3. **Grid-Constraint-Binding**: Wenn `export_pv × glf + discharge × RTE ≤ grid_max` bindet und `RTE ≠ glf`, kann
   der Solver durch gleichzeitiges Laden/Entladen eine andere Mischung von PV-Export und BESS-Discharge erzielen,
   die bei bestimmten Preiskonstellationen äquivalent ist.

4. **Grey Mode – Cross-Chamber**: Im Grey Mode kann `charge_pv[t] > 0` (Green-Kammer laden) und
   `discharge_grey[t] > 0` (Grey-Kammer entladen) gleichzeitig auftreten. Dies ist physikalisch fragwürdig,
   da die Batterie als Gesamtsystem nur eine Richtung gleichzeitig unterstützt.

### Warum reicht "discharge = 0 bei negativen Preisen" nicht?

- **Falsch-negativ**: Das Problem tritt auch bei positiven Preisen auf (siehe oben).
- **Falsch-positiv**: Bei negativen Preisen könnte Entladen durchaus sinnvoll sein – z.B. wenn der BESS voll ist und
  Platz für profitableres späteres Laden schaffen muss, oder um Baseload-Verpflichtungen zu erfüllen.
- Die aktuelle Logik blockiert also Entladung pauschal bei negativen Preisen, statt das eigentliche Problem
  (Simultaneität) zu adressieren.

### Lösungsansätze

#### Option A: Binärvariable (MILP) – **Empfohlen**

Einführung einer binären Indikatorvariable `δ[t] ∈ {0,1}` pro Zeitschritt:

```
charge_total[t]     ≤ δ[t] × M_charge          ∀t
discharge_total[t]  ≤ (1 - δ[t]) × M_discharge  ∀t
```

Wobei:
- `charge_total[t] = charge_pv[t]` (Green Mode) bzw. `charge_pv[t] + charge_grid[t]` (Grey Mode)
- `discharge_total[t] = discharge_green[t]` (Green Mode) bzw. `discharge_green[t] + discharge_grey[t]` (Grey Mode)
- `M_charge = max_charge_kw × timestep_hours` (Big-M Schranke für Ladeleistung)
- `M_discharge = max_discharge_kw × timestep_hours` (Big-M Schranke für Entladeleistung)
- `δ[t] = 1` → Laden erlaubt, Entladen gesperrt
- `δ[t] = 0` → Entladen erlaubt, Laden gesperrt

**Voraussetzung**: `scipy.optimize.milp` ist verfügbar (scipy 1.15.3 installiert, ≥ 1.9 erforderlich). ✅

**Vorteile:**
- Mathematisch exakte Lösung, keine Heuristik
- Kein Post-hoc-Patching nötig
- Funktioniert für alle Preiskonstellationen und Modi

**Nachteile:**
- MILP ist langsamer als LP. Erwarteter Overhead: ~2-5× pro Solve (HiGHS MILP-Solver ist effizient, Big-M-
  Formulierungen mit wenigen Binärvariablen sind gutmütig).
- Problemgröße steigt um T Variablen (T = 24 oder 96 Binärvariablen pro Tag).
- API-Umstellung: `scipy.optimize.linprog` → `scipy.optimize.milp` (andere Signatur und Constraint-Spezifikation).

**Performance-Abschätzung:**
- Aktuell: ~2-4ms pro LP-Solve (96 Zeitschritte, HiGHS LP)
- Erwartet: ~5-15ms pro MILP-Solve (96 Zeitschritte + 96 Binärvariablen, HiGHS MILP)
- 365 Tage × 25 Jahre = 9.125 Solves → aktuell ~27s, erwartet ~90s pro Grid-Punkt
- Bei 16 Grid-Punkten (parallelisiert auf 8 Kerne): ~3 Minuten statt ~1 Minute
- **Akzeptabel** – innerhalb des bestehenden Performance-Budgets (<10 Minuten)

#### Option B: Post-hoc Netting (Fallback / Zwischenlösung)

Nach jedem LP-Solve die Dispatch-Vektoren korrigieren:

```python
for t in range(T):
    if charge_pv[t] > 0 and discharge_green[t] > 0:
        net = charge_pv[t] - discharge_green[t]
        if net > 0:
            charge_pv[t] = net
            discharge_green[t] = 0.0
        else:
            charge_pv[t] = 0.0
            discharge_green[t] = -net
        # SoC-Update anpassen
```

**Vorteile:**
- Trivial zu implementieren, kein Solver-Wechsel
- Kein Performance-Overhead
- Korrekt, solange die LP-Lösung optimal ist (Netting ändert den Zielfunktionswert nicht)

**Nachteile:**
- Greift nur bei der Ergebnis-Interpretation, nicht im LP selbst
- SoC-Verlauf muss nach Netting rekonstruiert werden
- Grey Mode mit Dual-Chamber ist komplizierter (charge_pv + discharge_grey ist physikalisch möglich,
  charge_pv + discharge_green nicht)
- Kann zu Constraint-Verletzungen führen, wenn SoC-Bounds nach Netting nicht mehr passen

#### Option C: Penalty-Term in Zielfunktion

Kleinen Strafterm für Laden hinzufügen:

```
objective += ε × Σ_t charge_pv[t]    (mit ε ≈ 1e-6 €/kWh)
```

**Vorteile:** Bleibt LP, kein Solver-Wechsel, minimaler Performance-Overhead.
**Nachteile:** Garantiert Simultaneitäts-Verhinderung **nicht**. Verschiebt den Break-even-Punkt nur leicht.
Kann bei bestimmten Preiskonstellationen trotzdem versagen. **Nicht empfohlen.**

### Empfohlene Lösung: Option A (MILP)

#### Schritt 1: Migration von `linprog` zu `milp` (Infrastruktur)

Die `scipy.optimize.milp`-API unterscheidet sich grundlegend von `linprog`:

```python
# Bisherig (linprog):
result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

# Neu (milp):
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix

constraints = LinearConstraint(A, lb, ub)  # Vereint A_ub/A_eq in einer Matrix
integrality = np.zeros(n_vars)  # 0 = continuous, 1 = integer/binary
integrality[binary_indices] = 1
variable_bounds = Bounds(lb_array, ub_array)

result = milp(c, constraints=constraints, integrality=integrality, bounds=variable_bounds)
```

**Wichtige Unterschiede:**
- `milp` nutzt `LinearConstraint(A, lb, ub)` statt separatem `A_ub`/`A_eq`
- Equality: `lb[row] == ub[row]` (z.B. beide = `pv_production[t]`)
- Inequality ≤: `lb[row] = -np.inf`, `ub[row] = b_ub[row]`
- Variable bounds über `Bounds(lb, ub)` statt Liste von Tupeln
- Sparse Matrizen (`csc_matrix`) sind empfohlen für Performance
- `integrality`-Array markiert Binärvariablen

#### Schritt 2: Binärvariable `δ[t]` in Green-Mode LP (`_build_green_lp`)

**Aktuelle Variablen (5T bzw. 6T):**
```
[charge_pv(T), discharge_green(T), export_pv(T), curtail(T), soc(T), (shortfall(T))]
```

**Neue Variablen (6T bzw. 7T):**
```
[charge_pv(T), discharge_green(T), export_pv(T), curtail(T), soc(T), delta(T), (shortfall(T))]
```

**Neue Constraints (2T zusätzliche Zeilen):**
```
charge_pv[t]        - δ[t] × M_charge    ≤ 0     ∀t    (Laden nur wenn δ=1)
discharge_green[t]  + δ[t] × M_discharge ≤ M_discharge  ∀t    (Entladen nur wenn δ=0)
```

Äquivalent in Standardform:
```
charge_pv[t]        - M_charge × δ[t]    ≤ 0           ∀t
discharge_green[t]  + M_discharge × δ[t]  ≤ M_discharge  ∀t
```

**Integrality-Array:**
```python
integrality = np.zeros(n_vars)
integrality[5*T : 6*T] = 1  # δ[t] sind binär
```

**Bounds für δ:**
```python
# δ[t] ∈ {0, 1} → bounds (0, 1) + integrality = 1
```

**Entfernung der alten Logik:**
```python
# ALT (Zeile 540): discharge auf 0 fixieren bei negativen Preisen
ub = 0.0 if spot_prices[t] < 0 else max_discharge_energy

# NEU: Keine Sonderbehandlung negativer Preise nötig.
# Die Binärvariable verhindert Simultaneität für ALLE Preise.
ub = max_discharge_energy  # immer voll erlaubt
```

#### Schritt 3: Binärvariable `δ[t]` in Grey-Mode LP (`_build_grey_lp`)

**Aktuelle Variablen (8T bzw. 9T):**
```
[charge_pv(T), discharge_green(T), export_pv(T), curtail(T),
 charge_grid(T), discharge_grey(T), soc_green(T), soc_grey(T), (shortfall(T))]
```

**Neue Variablen (9T bzw. 10T):**
```
[charge_pv(T), discharge_green(T), export_pv(T), curtail(T),
 charge_grid(T), discharge_grey(T), soc_green(T), soc_grey(T), delta(T), (shortfall(T))]
```

**Neue Constraints (2T zusätzliche Zeilen):**
```
charge_pv[t] + charge_grid[t]           - M_charge × δ[t]    ≤ 0             ∀t
discharge_green[t] + discharge_grey[t]  + M_discharge × δ[t]  ≤ M_discharge   ∀t
```

Hier wird die **Gesamtladung** (PV + Grid) gegen die **Gesamtentladung** (Green + Grey) abgesichert.
Eine Batterie kann physikalisch nicht gleichzeitig laden und entladen, unabhängig von der "Kammer".

**Entfernung der alten Logik:**
```python
# ALT (Zeile 711, 720-721): discharge_green/grey auf 0 fixieren bei negativen Preisen
# NEU: Binärvariable übernimmt die Aufgabe für alle Preise.
```

#### Schritt 4: `optimize_day()` anpassen

```python
# ALT:
from scipy.optimize import linprog
result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=LP_SOLVER_METHOD)

# NEU:
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix

# _build_green_lp / _build_grey_lp liefern jetzt:
# (c, A_combined, lb_combined, ub_combined, var_lb, var_ub, integrality)
# statt (c, A_ub, b_ub, A_eq, b_eq, bounds)

constraints = LinearConstraint(csc_matrix(A_combined), lb_combined, ub_combined)
variable_bounds = Bounds(var_lb, var_ub)
result = milp(c, constraints=constraints, integrality=integrality, bounds=variable_bounds)
```

**Rückgabe-Kompatibilität:** `milp` gibt ein `OptimizeResult` mit `.x` und `.success` zurück –
identisch zu `linprog`. Die Ergebnis-Extraktion (`_extract_green_result`, `_extract_grey_result`)
muss nur die Index-Offsets für die neuen `δ`-Variablen berücksichtigen (δ-Werte werden verworfen).

#### Schritt 5: Konstanten in `config/defaults.py`

```python
# Solver-Konfiguration
LP_SOLVER_METHOD = "highs"  # Bestehend, wird für milp nicht mehr benötigt
MILP_SOLVER_OPTIONS = {"disp": False, "time_limit": 10.0}  # Optional: Timeout pro Solve
```

#### Schritt 6: Tests anpassen

**Neue Tests:**
- `test_optimizer.py`: Test, dass bei allen Preiskonstellationen (positiv, negativ, Floor-aktiv)
  niemals `charge[t] > 0 AND discharge[t] > 0` für denselben Zeitschritt gilt
- Parametrisierter Test über Green/Grey × alle Marketing-Typen
- Performance-Regression-Test: MILP-Solve darf nicht >50ms pro Tag dauern (96 Zeitschritte)

**Angepasste Tests:**
- Alle Tests, die die alte "discharge = 0 bei negativen Preisen"-Logik testen, müssen
  aktualisiert werden. Entladung bei negativen Preisen ist jetzt erlaubt (wenn nicht gleichzeitig geladen wird).
- `dispatch_constraint_checker.py` um Simultaneitäts-Check erweitern

**Bestehende Tests:**
- Alle anderen Dispatch-Tests sollten weiterhin grün sein (identische optimale Lösung,
  nur ohne phantom cycling)

### Betroffene Dateien

| Datei | Änderung |
|-------|----------|
| `dispatch/optimizer.py` | `_build_green_lp`, `_build_grey_lp`: Binärvariable + Constraints hinzufügen. `optimize_day`: `linprog` → `milp`. Alte "negative price → discharge=0"-Logik entfernen. |
| `config/defaults.py` | Optional: MILP-Solver-Optionen |
| `tests/test_optimizer.py` | Simultaneitäts-Tests hinzufügen, negative-Preis-Tests anpassen |
| `tests/dispatch_constraint_checker.py` | Simultaneitäts-Check hinzufügen |

### Risiko

**Hoch.** Fundamentale Änderung am LP-Kern. Die Umstellung von `linprog` auf `milp` betrifft die Solver-Schnittstelle
in der zentralen Funktion, die für jedes Tagesergebnis verantwortlich ist. Gründliches Testen ist zwingend erforderlich.

### Priorität

**Kritisch** – das Problem verfälscht Dispatch-Ergebnisse und damit Revenue-Berechnungen, Cashflows und IRR.
Sollte vor allen Cleanup-Items umgesetzt werden.

### Migrations-Strategie

Um das Risiko zu minimieren, wird die Umstellung in zwei Schritten empfohlen:

1. **Schritt A (Quick-Fix):** Post-hoc Netting (Option B) als sofortige Korrektur implementieren.
   Kann in <1h umgesetzt und sofort deployed werden. Entfernt die alte "negative price → discharge=0"-Logik.
2. **Schritt B (Ziellösung):** MILP-Umstellung (Option A). Ersetzt den Quick-Fix durch die saubere Lösung.
   Post-hoc Netting wird entfernt, sobald MILP validiert ist.

Alternativ kann Schritt B direkt umgesetzt werden, wenn ausreichend Zeit für Validierung vorhanden ist.

---

## Empfohlene Reihenfolge der Umsetzung

| #  | Aktion                                      | Dateien                                                                    | Risiko      | Aufwand |
|----|---------------------------------------------|----------------------------------------------------------------------------|-------------|---------|
| 0a | **B10: Simultanes Be-/Entladen (MILP)**    | `dispatch/optimizer.py`, `config/defaults.py`, Tests                       | **Hoch**    | Mittel  |
| 0b | **B9: Revenue vereinheitlichen (Bug-Fix)**  | `dispatch/optimizer.py`, `dispatch/engine.py`, Tests                       | **Hoch**    | Mittel  |
| 1  | B1.1-B1.4: Toten Code entfernen             | `market/eeg.py`, `market/ppa.py`, `tests/test_eeg.py`, `tests/test_ppa.py` | Gering      | Klein   |
| 2  | B2.1-B2.2: Duplizierung + Import            | `main.py`                                                                  | Sehr gering | Klein   |
| 3  | B4: CSV-Writer Stats-Helper                 | `output/csv_writer.py`                                                     | Gering      | Klein   |
| 4  | B5.1-B5.3: Array-Kopien                     | `optimizer.py`, `main.py`                                                  | Gering      | Klein   |
| 5  | B3.1-B3.4: Vermarktungslogik verschieben    | `main.py`, `market/eeg.py`, `market/ppa.py`                                | Mittel      | Mittel  |
| 6  | B6+B7: `run()` aufteilen + `ScenarioParams` | `main.py`                                                                  | Mittel      | Groß    |

**Gesamtaufwand:** ~6-9 Stunden (B10 allein ~2-3 Stunden inkl. Validierung)

