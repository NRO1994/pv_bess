# FIXES.md – Fehlerbehebungen und Anpassungen

## Integration

### FIX-01: BatteryState wird im LP-Optimizer nicht verwendet
**Status:** OFFEN
**Dateien:** `bess/battery.py`, `dispatch/optimizer.py`, `dispatch/engine.py`

**Problem:** Die Klasse `BatteryState` in `bess/battery.py` implementiert Charge/Discharge-Logik mit SoC-Tracking, Effizienz-Modell und Degradation. Der LP-Optimizer (`dispatch/optimizer.py`) verwendet stattdessen einen eigenen `BessParams`-Dataclass und modelliert SoC implizit über kumulative Constraints. Die `BatteryState`-Klasse wird nirgends im Dispatch-Pfad aufgerufen.

**Analyse:** Die LP-Optimierung *kann* `BatteryState.charge()`/`discharge()` konzeptionell nicht verwenden, da der Optimizer alle 24 Stunden simultan löst (keine sequenzielle Stunde-für-Stunde-Simulation). Die `BatteryState`-Klasse ist jedoch sinnvoll als:
- Validierungsreferenz in Tests (sequenzielle Simulation vs. LP-Ergebnis)
- State-Tracking für die Engine (SoC-Carryover zwischen Tagen)

**Empfohlene Änderung:**
1. Die Engine (`dispatch/engine.py`) verwendet bereits `BessParams` aus dem Optimizer statt `BatteryState`. Das ist korrekt für den LP-Ansatz.
2. `BatteryState` als **Validierungswerkzeug** beibehalten, aber dokumentieren, dass es *nicht* im Produktionspfad verwendet wird.
3. Alternative: `BatteryState` für die **Offline-Day-Dispatch** verwenden (sequenzielle Logik), um Konsistenz zu demonstrieren.

---

### FIX-02: Einheitliche Inflations-Klasse
**Status:** ERLEDIGT
**Dateien:** `finance/inflation.py`

**Analyse:** Das Modul `finance/inflation.py` bietet `inflate_value()`, `inflate_series()` und `build_inflation_factors()` als zentrale Schnittstelle. Alle Module (`cashflow.py`, `ppa.py`, `eeg.py`, `main.py`, `grid_search.py`, `monte_carlo.py`) verwenden diese Funktionen. Die Inflation beginnt korrekt erst in Jahr 2 (Faktor = `(1 + rate) ^ max(0, year - 1)`).

---

### FIX-03: BESS und PV teilen Degradation-Logik
**Status:** ERLEDIGT
**Dateien:** `pv/degradation.py`, `dispatch/engine.py`

**Analyse:** Sowohl PV als auch BESS verwenden `pv/degradation.py:degradation_factor()`. In `engine.py:run_simulation()` wird `degradation_factor(config.pv_degradation_rate, year)` für PV und `degradation_factor(config.bess_degradation_rate, bess_age)` für BESS aufgerufen. Gleiche Logik, gleiche Funktion.

---

## Logik

### FIX-04: Baseload-PPA – Baseload als reiner User-Input
**Status:** OFFEN
**Dateien:** `market/ppa.py`, `config/schema.py`

**Problem:** `ppa.py:baseload_level_kwh()` (Zeile 207-230) berechnet den Baseload automatisch aus der Jahresproduktion, wenn `baseload_mw is None`. Laut FIXES.md soll der Baseload ein reiner User-Input sein – keine Berechnungslogik nötig.

**Empfohlene Änderung:**
1. `baseload_level_kwh()` vereinfachen: Auto-Berechnung entfernen. Wenn `baseload_mw is None`, einen `ValueError` werfen.
2. In `config/schema.py`: `baseload_mw` als **required** markieren, wenn `ppa_type == "ppa_baseload"` (cross-field validation).
3. Validation hinzufügen: `baseload_mw * 1000 <= pv_peak_kwp` (Baseload darf PV-Nennleistung nicht überschreiten).

---

### FIX-05: Baseload-PPA – Überschuss zum Spotpreis verkaufen
**Status:** ERLEDIGT
**Dateien:** `market/ppa.py`

**Analyse:** `baseload_revenue()` (Zeile 233-275) berechnet korrekt:
```
revenue = baseload_kwh * (ppa_price + goo) + (export - baseload) * spot
```
Überschuss wird zum Spotpreis verkauft, Unterdeckung zum Spotpreis zugekauft.

---

### FIX-06: GoO-Premium auf den finalen Preis, nicht nur auf den Floor (Collar)
**Status:** ERLEDIGT
**Dateien:** `market/ppa.py`, `dispatch/optimizer.py`, `dispatch/engine.py`, `optimization/grid_search.py`, `optimization/monte_carlo.py`, `main.py`

**Problem:** In `effective_collar_prices()` (Zeile 346-380) wird die GoO-Prämie nur auf den Floor addiert (`floor += config.goo_premium_eur_per_kwh`), nicht auf den Cap. Der effektive Preis ist dann `clip(spot, floor+goo, cap)`. Laut FIXES.md soll die GoO-Prämie **immer auf den finalen Preis** addiert werden, d.h. `clip(spot, floor, cap) + goo`.

**Betrifft auch:**
- `effective_floor_price()` (Zeile 283-313): GoO wird korrekt zum Floor addiert → effektiv `max(spot, floor+goo)`. Das ist mathematisch äquivalent zu `max(spot, floor) + goo` nur wenn `spot >= floor+goo` – **das ist es NICHT**. Korrekt wäre: `max(spot, floor) + goo` = der Verkäufer bekommt mindestens `floor + goo`, bei höherem Spot bekommt er `spot + goo`.
- `pay_as_produced_price()`: GoO wird in `apply_pay_as_produced()` korrekt als Aufschlag auf den festen Preis behandelt.

**Empfohlene Änderung:**
1. **Collar:** `effective_collar_prices()` soll `(floor, cap)` ohne GoO zurückgeben. GoO wird **nach** dem Clip addiert: `effective = clip(spot, floor, cap) + goo`.
2. **Floor:** `effective_floor_price()` soll Floor ohne GoO zurückgeben. GoO wird **nach** dem Max addiert: `effective = max(spot, floor) + goo`.
3. **Pay-as-produced:** Bereits korrekt (GoO als Aufschlag auf festen Preis).
4. **Baseload:** GoO bereits korrekt auf den PPA-Preis addiert.
5. Den Optimizer (`dispatch/optimizer.py`) anpassen: Die `_effective_green_price()` muss die GoO getrennt behandeln – Floor/Cap für die Max/Clip-Berechnung, GoO als finaler Aufschlag.

---

### FIX-07: Inflation erst ab Jahr 2
**Status:** ERLEDIGT
**Dateien:** `finance/inflation.py`

**Analyse:** `inflate_value()` verwendet `max(0, year - 1)` als Exponent. Jahr 1 = Faktor 1.0 (keine Inflation), Jahr 2 = Faktor (1+r)^1. Korrekt implementiert.

---

### FIX-08: CAPEX/OPEX im ersten Jahr + Inbetriebnahmejahr als User-Input
**Status:** ERLEDIGT
**Dateien:** `finance/cashflow.py`, `config/schema.py`, `config/loader.py`, `config/defaults.py`, `main.py`, `output/csv_writer.py`, `optimization/grid_search.py`, `optimization/monte_carlo.py`

**Problem:** Aktuell wird CAPEX in "Jahr 0" und OPEX ab "Jahr 1" gebucht. Laut FIXES.md sollen **beide im ersten Jahr** (dem Inbetriebnahmejahr) anfallen. Zusätzlich soll das Inbetriebnahmejahr (z.B. 2027) als User-Input im Szenario-JSON aufgenommen werden.

**Empfohlene Änderung:**
1. **Neues Feld** in `project_settings`: `"commissioning_year": 2027`
2. **Schema** (`config/schema.py`): Feld als `required` mit `type: integer, minimum: 2020` aufnehmen.
3. **Cashflow** (`cashflow.py`): Jahr 0 eliminieren. CAPEX und OPEX fallen im selben Jahr 1 an (= Inbetriebnahmejahr). Equity CF in Jahr 1 = Revenue - OPEX - CAPEX_equity - Debt Service - Tax.
4. **CSV-Output** (`csv_writer.py`): Die `year`-Spalte im Cashflow-CSV soll das Kalenderjahr darstellen (z.B. 2027, 2028, ...) statt 0, 1, 2.
5. **Dispatch-Sample** (`csv_writer.py:write_dispatch_sample_csv`): `start_year` aus dem Szenario ableiten statt Hardcode `2025`.

---

### FIX-09: Preiszeitreihe ab Inbetriebnahmejahr filtern
**Status:** ERLEDIGT
**Dateien:** `config/loader.py`, `main.py`

**Problem:** Beginnt die CSV-Preiszeitreihe vor dem Inbetriebnahmejahr, sollen die früheren Jahre ignoriert werden.

**Empfohlene Änderung:**
1. In `load_price_csv()` oder `main.py`: Timestamp-Spalte parsen, den Start-Timestamp mit dem Inbetriebnahmejahr vergleichen.
2. Alle Zeilen vor dem 1.1. des Inbetriebnahmejahres verwerfen.
3. Abhängig von FIX-08 (Inbetriebnahmejahr als User-Input).

---

### FIX-10: Alle Preisszenarien verlängern (nicht nur MID)
**Status:** OFFEN
**Dateien:** `main.py`

**Problem:** In `main.py` (Zeilen 469-477) wird nur die `mid_column` für den Grid-Search-Pfad verlängert. Zwar werden im MC-Abschnitt (Zeilen 669-679) alle Szenarien verlängert, aber diese Logik greift nur, wenn MC aktiviert ist.

**Empfohlene Änderung:**
1. Alle `required_columns` (low/mid/high) direkt nach dem Laden verlängern – unabhängig davon, ob MC aktiviert ist.
2. Ein Dict `extended_prices: dict[str, np.ndarray]` für alle Spalten aufbauen.
3. Grid Search verwendet weiterhin nur "mid". MC greift auf das bereits vorbereitete Dict zu.

---

### FIX-11: Körperschaftsteuer aufnehmen
**Status:** ERLEDIGT
**Dateien:** `finance/tax.py`, `config/defaults.py`

**Analyse:** `calculate_koerperschaftsteuer()` ist implementiert. `DEFAULT_KOERPERSCHAFTSTEUER_PCT = 15.0` ist in `defaults.py` definiert. Wird in `calculate_tax_for_year()` korrekt aufgerufen.

---

### FIX-12: Solidaritätszuschlag aufnehmen
**Status:** ERLEDIGT
**Dateien:** `finance/tax.py`, `config/defaults.py`

**Analyse:** `calculate_solidaritaetszuschlag()` ist implementiert. `DEFAULT_SOLIDARITAETSZUSCHLAG_PCT = 5.5` ist in `defaults.py` definiert. Wird auf Basis der KSt berechnet. Korrekt.

---

## Kosmetik

### FIX-13: `timeseries.percentile_timeseries` entfernen
**Status:** OFFEN
**Dateien:** `pv/timeseries.py`

**Problem:** Die Funktion `percentile_timeseries()` (Zeile 108-145) ist eine Verallgemeinerung von `compute_p50_p90()`. Sie wird nirgends im Produktionscode aufgerufen (nur P50/P90 werden benötigt).

**Empfohlene Änderung:**
1. `percentile_timeseries()` aus `timeseries.py` entfernen.
2. Prüfen ob Tests die Funktion referenzieren → falls ja, Tests anpassen.

---

### FIX-14: Enums zentral in `config/defaults.py`
**Status:** ERLEDIGT
**Dateien:** `config/defaults.py`

**Analyse:** PPA-Typen (`PPA_TYPE_NONE`, `PPA_TYPE_PAY_AS_PRODUCED`, etc.) und Marketing-Typen (`MARKETING_TYPE_EEG`, `MARKETING_TYPE_PPA`, `MARKETING_TYPE_MARKET`) sind alle in `config/defaults.py` definiert (Zeilen 193-219). `ppa.py` und `eeg.py` importieren diese Konstanten korrekt.

---

### FIX-15: Magic Numbers in `main.py`
**Status:** OFFEN
**Dateien:** `main.py`

**Problem:** Mehrere Fallback-Werte in `main.py` sind als Literale statt als Referenzen auf `config/defaults.py` codiert:
- Zeile 323: `finance.get("inflation_rate", 0.02)` → sollte `DEFAULT_INFLATION_RATE` verwenden
- Zeile 325: `finance.get("interest_rate_pct", 4.5)` → `DEFAULT_INTEREST_RATE_PCT`
- Zeile 326: `finance.get("loan_tenor_years", 18)` → `DEFAULT_LOAN_TENOR_YEARS`
- Zeile 327: `scenario.project_settings.get("discount_rate", 0.06)` → `DEFAULT_DISCOUNT_RATE`
- Zeile 343: `pv_perf.get("degradation_rate_pct_per_year", 0.4)` → `DEFAULT_PV_DEGRADATION_RATE_PCT`
- Zeile 344: `pv_perf.get("system_loss_pct", 14.0)` → `DEFAULT_SYSTEM_LOSS_PCT`
- Zeile 349: `bess_perf.get("round_trip_efficiency_pct", 88.0)` → `DEFAULT_BESS_RTE_PCT`
- Zeile 350: `bess_perf.get("min_soc_pct", 10.0)` → `DEFAULT_BESS_MIN_SOC_PCT`
- Zeile 351: `bess_perf.get("max_soc_pct", 90.0)` → `DEFAULT_BESS_MAX_SOC_PCT`
- Zeile 352: `bess_perf.get("degradation_rate_pct_per_year", 2.0)` → `DEFAULT_BESS_DEGRADATION_RATE_PCT`
- Zeile 353: `bess_perf.get("bess_availability_pct", 100.0)` → `DEFAULT_BESS_AVAILABILITY_PCT`

**Empfohlene Änderung:** Alle Literal-Fallbacks durch die entsprechenden Konstanten aus `config/defaults.py` ersetzen. Die Konstanten sind bereits importiert (Zeile 32-62).

---

### FIX-16: Integrationstest für PVGIS-Fetch
**Status:** OFFEN
**Dateien:** Neuer Test `tests/test_pvgis_integration.py`

**Problem:** Es gibt keinen Integrationstest, der einen echten PVGIS-API-Aufruf testet.

**Empfohlene Änderung:**
1. Neuen Test `tests/test_pvgis_integration.py` erstellen.
2. Mit `@pytest.mark.integration` markieren.
3. Einen echten API-Aufruf mit bekannten Koordinaten durchführen.
4. Prüfen: Response-Format, Anzahl Jahre, Array-Länge 8760 pro Jahr.

---

### FIX-17: Pytest-Konfiguration für Integrationstests
**Status:** OFFEN
**Dateien:** `pyproject.toml`

**Problem:** Es gibt keine Pytest-Marker-Konfiguration. Integrationstests sollen standardmäßig übersprungen und nur separat ausgeführt werden.

**Empfohlene Änderung:**
1. In `pyproject.toml` unter `[tool.pytest.ini_options]`:
   ```toml
   markers = ["integration: Integrationstests (separat ausführen mit -m integration)"]
   addopts = "-m 'not integration'"
   ```
2. Integrationstests werden dann nur mit `pytest -m integration` ausgeführt.

---

### FIX-18: Testdaten im `.data`-Verzeichnis
**Status:** OFFEN
**Dateien:** Diverse Testdateien, Verzeichnisstruktur

**Problem:** Es gibt kein `.data`-Verzeichnis. Test-CSV-Dateien werden in Tests via `tmp_path` erstellt, was korrekt ist. Aber gemeinsame Testdaten (z.B. Referenz-CSVs) sollten in `.data/` liegen.

**Empfohlene Änderung:**
1. Verzeichnis `.data/` im Projektroot erstellen.
2. Testdaten (z.B. Referenz-Preiszeitreihen für Integrationstests) dort ablegen.
3. `conftest.py`-Fixtures anpassen, um aus `.data/` zu lesen, wo sinnvoll.

---

### FIX-19: CLAUDE.md aktualisieren
**Status:** OFFEN
**Dateien:** `CLAUDE.md`

**Problem:** Die CLAUDE.md reflektiert nicht alle Implementierungsentscheidungen:
- Revenue-Hilfsvariablen wurden im LP-Optimizer durch Pre-Computation von `max(spot, fixed)` ersetzt (dokumentiert im Optimizer-Docstring, aber nicht in CLAUDE.md)
- KSt und Soli wurden hinzugefügt (Tax Module erweitern)
- Inbetriebnahmejahr als neuer User-Input (nach FIX-08)
- GoO-Behandlung korrigiert (nach FIX-06)
- Baseload als reiner User-Input (nach FIX-04)
- CSV-Delimiter auf Semikolon geändert
- PV-Cache-Verzeichnis geändert (nach FIX-21)

**Empfohlene Änderung:** Nach Abschluss aller Fixes CLAUDE.md systematisch durchgehen und aktualisieren.

---

### FIX-20: CSV_DELIMITER in allen Unit Tests verwenden
**Status:** TEILWEISE ERLEDIGT
**Dateien:** `tests/test_loader.py`, `tests/test_price_loader.py`

**Analyse:** `CSV_DELIMITER` ist auf `";"` gesetzt (in `config/defaults.py`). Die Testdateien `test_loader.py` und `test_price_loader.py` verwenden bereits Semikolon als Delimiter (hardcodiert als `";"` bzw. `sep=";"`). Allerdings verwenden sie nicht die Konstante `CSV_DELIMITER` selbst.

**Empfohlene Änderung:**
1. In allen Testdateien: `from pv_bess_model.config.defaults import CSV_DELIMITER` importieren.
2. Alle Literal-`";"` durch `CSV_DELIMITER` ersetzen.

---

### FIX-21: PV-Cache im `.data`-Verzeichnis
**Status:** OFFEN
**Dateien:** `config/defaults.py`, `pv/pvgis_client.py`

**Problem:** `PVGIS_CACHE_DIR` zeigt auf `~/.pv_bess_cache` (Home-Verzeichnis). Soll stattdessen im Projekt unter `.data/pvgis_cache/` liegen.

**Empfohlene Änderung:**
1. In `config/defaults.py`: `PVGIS_CACHE_DIR = ".data/pvgis_cache"` (relativ zum Projektroot).
2. `.data/` in `.gitignore` aufnehmen.
3. `pvgis_client.py` anpassen: Pfad relativ zum Arbeitsverzeichnis auflösen.

---

### FIX-22: Cashflow-CSV Year-Index basierend auf Inbetriebnahmejahr
**Status:** OFFEN
**Dateien:** `output/csv_writer.py`

**Problem:** Die `year`-Spalte im Cashflow-CSV beginnt bei 0 statt beim Inbetriebnahmejahr. Abhängig von FIX-08.

**Empfohlene Änderung:**
1. Das Inbetriebnahmejahr (aus Szenario-JSON) an `write_cashflows_csv()` übergeben.
2. `year`-Spalte: `commissioning_year + y` statt `y`.

---

### FIX-23: Fortschrittslogging für Grid Search und MC (Debug-Mode)
**Status:** OFFEN
**Dateien:** `optimization/grid_search.py`, `optimization/monte_carlo.py`

**Problem:** Es gibt kein Fortschritts-Logging über die Iterationen der Grid Search und MC. Der Logger meldet nur Start und Ende.

**Empfohlene Änderung:**
1. **Grid Search** (`grid_search.py`): Nach jedem abgeschlossenen Grid-Point `logger.debug()` mit Fortschritt (z.B. `"Grid point 5/33: scale=40%%, E/P=2h → IRR=7.2%%"`).
2. **Monte Carlo** (`monte_carlo.py`): Alle N Iterationen (z.B. alle 100) `logger.debug()` mit Fortschritt (z.B. `"MC iteration 100/1000 complete"`).
3. Nur im Debug-Mode sichtbar (Level `DEBUG`), nicht im normalen `INFO`-Mode.

---

## Zusammenfassung

| # | Beschreibung | Status | Priorität |
|---|---|---|---|
| FIX-01 | BatteryState im LP nicht verwendet | OFFEN | Niedrig (Kosmetik) |
| FIX-02 | Einheitliche Inflation | ERLEDIGT | - |
| FIX-03 | Gemeinsame Degradation-Logik | ERLEDIGT | - |
| FIX-04 | Baseload als User-Input | OFFEN | Mittel |
| FIX-05 | Baseload Überschuss zum Spot | ERLEDIGT | - |
| FIX-06 | GoO auf finalen Preis | ERLEDIGT | - |
| FIX-07 | Inflation ab Jahr 2 | ERLEDIGT | - |
| FIX-08 | CAPEX/OPEX in Jahr 1 + Inbetriebnahmejahr | ERLEDIGT | - |
| FIX-09 | Preis-CSV ab Inbetriebnahmejahr filtern | ERLEDIGT | Mittel |
| FIX-10 | Alle Preisszenarien verlängern | OFFEN | Mittel |
| FIX-11 | Körperschaftsteuer | ERLEDIGT | - |
| FIX-12 | Solidaritätszuschlag | ERLEDIGT | - |
| FIX-13 | percentile_timeseries entfernen | OFFEN | Niedrig |
| FIX-14 | Enums zentral | ERLEDIGT | - |
| FIX-15 | Magic Numbers in main.py | OFFEN | Mittel |
| FIX-16 | PVGIS Integrationstest | OFFEN | Niedrig |
| FIX-17 | Pytest Marker für Integration | OFFEN | Niedrig |
| FIX-18 | Testdaten in .data/ | OFFEN | Niedrig |
| FIX-19 | CLAUDE.md aktualisieren | OFFEN | Mittel |
| FIX-20 | CSV_DELIMITER in Tests | OFFEN | Niedrig |
| FIX-21 | PV-Cache in .data/ | OFFEN | Niedrig |
| FIX-22 | Cashflow Year-Index | OFFEN | Mittel |
| FIX-23 | Fortschrittslogging | OFFEN | Niedrig |

**Empfohlene Reihenfolge der Bearbeitung:**
1. FIX-06 (GoO-Logik – beeinflusst Optimierungsergebnisse)
2. FIX-08 (Inbetriebnahmejahr – strukturelle Änderung, Voraussetzung für FIX-09 und FIX-22)
3. FIX-09 (Preis-CSV Filterung – abhängig von FIX-08)
4. FIX-22 (Cashflow Year-Index – abhängig von FIX-08)
5. FIX-04 (Baseload-Validierung)
6. FIX-10 (Alle Preisszenarien verlängern)
7. FIX-15 (Magic Numbers)
8. FIX-13, FIX-20, FIX-21, FIX-23 (Kosmetik-Fixes)
9. FIX-16, FIX-17, FIX-18 (Test-Infrastruktur)
10. FIX-01 (BatteryState-Dokumentation)
11. FIX-19 (CLAUDE.md – zuletzt, da alle Änderungen einfließen müssen)
