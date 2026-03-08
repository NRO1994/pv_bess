# Feature 11: Sanitizing & Testing – Offene Integration-Fixes (Session 3)

## Uebersicht

Dieses Dokument beschreibt den Plan fuer die noch offenen Punkte aus `FIXES-Session3.md`.
Es gibt drei Kategorien: **Integration/Testing**, **Logik** und **HTML-Report** (als eigenes Feature 10 bereits dokumentiert).

---

## Status Quo

| Metrik | Wert |
|--------|------|
| Unit-Tests bestanden | 470 |
| Unit-Tests fehlgeschlagen | 157 |
| Unit-Tests mit Sammel-Errors | 61 |
| Test-Dateien mit Fehlern | 14 (inkl. `test_price_loader.py` Import-Error) |

### Fehlerverteilung nach Testdatei

| # Failures | # Errors | Testdatei | Ursache (geschaetzt) |
|------------|----------|-----------|----------------------|
| 43 | 12 | `test_loader.py` | Schema-/Config-Aenderungen nicht nachgezogen |
| 35 | 0 | `test_schema.py` | Schema-Validierung veraendert (Preisfelder entfernt, Kosmetik-Fixes) |
| 17 | 0 | `test_engine.py` | Dispatch-Engine Signatur/Return-Aenderungen |
| 15 | 0 | `test_bess_only.py` | BESS-Only Logic refactored |
| 12 | 0 | `test_main_price_extension.py` | Price-Loader Umstrukturierung |
| 8 | 0 | `test_replacement.py` | Replacement als CAPEX + Upgrade-Faktor |
| 6 | 0 | `test_optimizer_bess_spot_pricing.py` | Optimizer-Signatur geaendert |
| 5 | 0 | `test_optimizer.py` | Optimizer-Signatur geaendert |
| 5 | 0 | `test_metrics.py` | Metriken erweitert/umbenannt |
| 4 | 24 | `test_monte_carlo.py` | MC-Framework Umstrukturierung |
| 4 | 20 | `test_grid_search.py` | Grid-Search refactored |
| 2 | 0 | `test_degradation.py` | Degradation-API geaendert |
| 1 | 5 | `test_csv_writer_cashflows.py` | Cashflow-Writer Spalten geaendert |
| Import-Error | 0 | `test_price_loader.py` | `collect_scenario_columns` entfernt |

---

## Plan

### Phase A: Unit-Tests reparieren (Hoechste Prioritaet)

**Ziel:** Alle Unit-Tests auf gruenen Stand bringen, ohne Produktions-Logik zu aendern.

**Reihenfolge:** Bottom-Up (abhaengigkeitsfreie Module zuerst)

#### A1: Triviale Fixes (Import-Errors und kleine API-Aenderungen)
- [ ] `test_price_loader.py`: Import-Error beheben (`collect_scenario_columns` existiert nicht mehr). Pruefen ob Tests noch relevant sind oder entfernt werden koennen.
- [ ] `test_degradation.py`: 2 Fehler – API-Signatur (year 0 / negative year) an neue Logik anpassen.

#### A2: Schema & Loader Tests
- [ ] `test_schema.py` (35 Failures): Schema-Validierung hat sich durch Kosmetik-Fixes stark geaendert (Preisfelder aus `price_inputs` entfernt, `optimization_fee_pct` verschoben, Preiseinheiten vereinheitlicht). Tests an aktuelles Schema anpassen.
- [ ] `test_loader.py` (43 Failures, 12 Errors): `ScenarioConfig`-Accessors pruefen ob sie noch zur aktuellen Datenstruktur passen. Insbesondere `price_unit`, `price_csv_path` und MC-bezogene Felder.

#### A3: Finance-Module Tests
- [ ] `test_metrics.py` (5 Failures): Metriken-API pruefen (evtl. neue Return-Felder oder umbenannte Keys).
- [ ] `test_replacement.py` (8 Failures): Replacement als CAPEX + Upgrade-Faktor nachziehen.
- [ ] `test_csv_writer_cashflows.py` (1 Failure, 5 Errors): Cashflow-Spalten an neuen Output anpassen.

#### A4: Dispatch-Module Tests
- [ ] `test_optimizer.py` (5 Failures): Optimizer-Funktionssignatur hat sich geaendert. Tests an neue Parameter anpassen.
- [ ] `test_optimizer_bess_spot_pricing.py` (6 Failures): Gleiche Ursache wie `test_optimizer.py`.
- [ ] `test_engine.py` (17 Failures): Engine Return-Struktur und Signatur anpassen.

#### A5: Uebergreifende Module Tests
- [ ] `test_bess_only.py` (15 Failures): BESS-Only Logik wurde refactored. Tests an neue Implementierung anpassen.
- [ ] `test_main_price_extension.py` (12 Failures): Price-Extension Logik hat sich durch Szenario-basierte CSV-Reads geaendert.
- [ ] `test_grid_search.py` (4 Failures, 20 Errors): Grid-Search API/Signaturen nachziehen.
- [ ] `test_monte_carlo.py` (4 Failures, 24 Errors): MC-Framework Umstrukturierung (Preisszenarien direkt aus `price_inputs`).

#### A6: Test-Coverage pruefen und erweitern
- [ ] Coverage-Bericht erstellen (`pytest --cov=pv_bess_model`)
- [ ] Kritische Funktionen mit fehlender Coverage identifizieren:
  - `timeseries.align_weather_to_forecast_year` – neue Funktion, vermutlich ungetestet
  - `optimizer._effective_green_price` – alle 6 Vermarktungsszenarien abdecken (Market, EEG, PPA-Floor, PPA-Collar, PPA-Baseload, PPA-Pay-as-Produced)
  - `cashflow.py` – Verlustvortrag ueber mehrere Jahre
  - `costs.py` – Unified Cost Schema mit fehlenden Feldern = 0
- [ ] Neue Unit-Tests schreiben fuer identifizierte Luecken

---

### Phase B: Code Clean-Up (Mittlere Prioritaet)

**Ziel:** Redundanzen entfernen, ungenutzten Code loeschen, Linting durchsetzen.

#### B1: Redundante Berechnungen identifizieren und vereinheitlichen
- [ ] Cashflow-Berechnung: Pruefen ob Revenue an mehreren Stellen unabhaengig berechnet wird (z.B. in `engine.py` vs `cashflow.py`)
- [ ] Kosten-Berechnung: Pruefen ob CAPEX/OPEX an mehreren Stellen berechnet wird
- [ ] Suche nach duplizierten Berechnungsmustern via Grep

#### B2: Ungenutzten Code entfernen
- [ ] `price_loader.py`: `collect_scenario_columns` und weitere nicht mehr genutzte Funktionen identifizieren und entfernen
- [ ] Pruefen ob `price_loader.py` selbst noch benoetigt wird oder ob die CSV-Logik vollstaendig in die Szenarien gewandert ist
- [ ] Weitere Module auf toten Code pruefen (insbesondere nach den Kosmetik-Fixes: `price_unit`-Konvertierung, alte `price_inputs`-Felder)
- [ ] Nicht mehr benoetigte Helper-Funktionen in Tests entfernen

#### B3: Linting & Formatting
- [ ] `ruff check pv_bess_model/` ausfuehren und Violations beheben
- [ ] `black --check pv_bess_model/` ausfuehren und formatieren
- [ ] Fehlende Type-Hints an geaenderten Funktionen ergaenzen (nur wo geaendert, kein Komplett-Refactoring)

#### B4: Tests nach Clean-Up verifizieren
- [ ] Alle Unit-Tests erneut laufen lassen nach jeder Clean-Up Aenderung
- [ ] Sicherstellen, dass Clean-Up keine Logik veraendert

---

### Phase C: Logik-Erweiterungen (Niedrigere Prioritaet, eigene Commits)

#### C1: Solver-Wechsel zu OR-Tools/HiGHS
- [ ] `scipy.optimize.linprog` durch `ortools.linear_solver.pywraplp` ersetzen
- [ ] Solver: `pywraplp.Solver.CreateSolver('HiGHS')`
- [ ] Betrifft: `pv_bess_model/dispatch/optimizer.py` (Hauptaenderung)
- [ ] Betrifft: `pv_bess_model/config/defaults.py` (Solver-Konstante)
- [ ] Alle bestehenden Unit-Tests und Integration-Tests muessen weiterhin bestehen
- [ ] Performance-Vergleich: Sicherstellen dass <1ms pro LP-Solve erhalten bleibt

#### C2: Simultanes Laden/Entladen verhindern
- [ ] Problem: BESS kann in einem Zeitschritt gleichzeitig laden und entladen
- [ ] Tritt nur bei negativen Preisen auf (Arbitrage durch Ineffizienz)
- [ ] Loesung: Bei negativen Preisen `discharge = 0` erzwingen (pragmatischer Ansatz)
- [ ] Alternative: Binary-Variable (MILP), aber deutlich langsamer
- [ ] Implementierung in `optimizer.py`: Zusaetzliche Constraint wenn `price_spot[t] < 0`
- [ ] Unit-Test: Verifizieren dass bei negativen Preisen kein simultanes Laden/Entladen

#### C3: PPA-Baseload korrekt im LP implementieren
- [ ] Aktuell: PPA-Baseload nicht vollstaendig im LP-Optimizer abgebildet
- [ ] Erforderlich: Baseload-Constraint im LP, damit BESS den Baseload verlaengern kann
- [ ] Revenue-Berechnung:
  - Bei ausreichender Einspeisung (PV + BESS >= baseload): `max(spot_price, effective_price)`
  - Bei Unterdeckung: Einkaufskosten `(baseload - grid_export) * (spot_price - effective_price)`
- [ ] Neue LP-Variablen: `shortfall[t]`, `excess[t]` relativ zum Baseload-Profil
- [ ] Neue Spalten im Dispatch-CSV und Cashflow fuer Baseload-bezogene Groessen
- [ ] Unit-Tests fuer alle PPA-Baseload Faelle (Ueberdeckung, Unterdeckung, genau Baseload)

---

### Phase D: HTML-Report (Eigenes Feature, separat dokumentiert)

Siehe [Feature 10: HTML-Report](10_html_report.md).

- [ ] Input-HTML (Szenario-Wizard)
- [ ] Output-HTML (Ergebnis-Dashboard mit 6 Tabs)
- [ ] LLM-Prompt-Template fuer Copilot-Workflow
- [ ] Integration in den Main-Flow

---

## Abhaengigkeiten

```
Phase A (Tests reparieren)
  |
  v
Phase B (Clean-Up)    ← haengt von A ab (Tests muessen gruen sein)
  |
  v
Phase C (Logik)       ← haengt von A+B ab (saubere Basis)
  |  |  |
  |  |  +-- C3 (PPA-Baseload) ← unabhaengig von C1/C2
  |  +-- C2 (Charge/Discharge) ← unabhaengig von C1/C3
  +-- C1 (Solver-Wechsel)      ← unabhaengig von C2/C3
  |
  v
Phase D (HTML-Report)  ← kann parallel zu C starten
```

## Aufwandsschaetzung

| Phase | Geschaetzter Aufwand |
|-------|---------------------|
| A: Unit-Tests reparieren | 6-10h |
| B: Code Clean-Up | 4-6h |
| C1: Solver-Wechsel | 3-4h |
| C2: Charge/Discharge Fix | 2-3h |
| C3: PPA-Baseload LP | 6-8h |
| D: HTML-Report | 16-24h (siehe Feature 10) |

## Empfohlene Reihenfolge der Bearbeitung

1. **A1-A5** – Tests reparieren (ein Commit pro Testdatei oder logische Gruppe)
2. **A6** – Coverage pruefen und neue Tests schreiben
3. **B1-B3** – Clean-Up (ein Commit pro logische Einheit)
4. **C1** – Solver-Wechsel (eigener Commit)
5. **C2** – Charge/Discharge Constraint (eigener Commit)
6. **C3** – PPA-Baseload (eigener Commit, groesster logischer Eingriff)
7. **D** – HTML-Report (mehrere Commits, siehe Feature 10)
