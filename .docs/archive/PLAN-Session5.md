# Bugfix-Plan Session 5

Basierend auf FIXES-Session4.md. Priorisiert nach Abhängigkeiten, Risiko und Aufwand.

---

## Phase 1: Kern-Logik (Höchste Priorität)

### 1. Solver-Wechsel: scipy → ortools/HiGHS -> ABGEBROCHEN, Solver ist wesentlich langsamer
- **Datei:** `dispatch/optimizer.py` + alle Optimizer-Tests
- **Aufwand:** Mittel-Hoch (834 Zeilen Optimizer umschreiben)
- **Begründung:** Grundlage für alle folgenden LP-Änderungen. Muss zuerst passieren, damit nachfolgende
  LP-Erweiterungen (Gleichzeitigkeits-Constraint, Baseload) direkt im neuen Solver gebaut werden.
- **Risiko:** Hoch – zentrale Komponente, alle Tests müssen grün bleiben.

### 2. Gleichzeitiges Laden/Entladen verhindern - ERLEDIGT
- **Datei:** `dispatch/optimizer.py`
- **Aufwand:** Gering
- **Begründung:** Baut auf neuem Solver auf. Vereinfachung: `discharge=0` bei negativen Preisen, vermeidet binäre
  Variablen.
- **Abhängigkeit:** → nach Schritt 1

### 3. PPA-Baseload in LP-Optimierung integrieren - OFFEN
- **Dateien:** `dispatch/optimizer.py`, `dispatch/engine.py`, `market/ppa.py`, `output/csv_writer.py`
- **Aufwand:** Hoch
- **Begründung:** Neue LP-Variablen (Baseload-Shortfall-Kosten), Revenue: `max(spot, effective)` bei ausreichender
  Einspeisung, Einkaufskosten `(baseload - grid_export) * (spot - effective)` bei Shortfall. Dispatch/Cashflow-Ausgabe
  anpassen.
- **Abhängigkeit:** → nach Schritt 1

### 4. MC-Framework Refactoring - ERLEDIGT
- **Datei:** `optimization/monte_carlo.py`
- **Aufwand:** Hoch (630 Zeilen, Architektur-Änderung)
- **Begründung:** Dispatch nur 1x pro Preisszenario (statt N×). PV/BESS-Availability auf 100% im Dispatch. MC-Noise auf
  Finanz-Ergebnisse sowie die PV/BESS Verfügbarkeit auf die entsprechende Revenues anwenden.
- **Ablauf:**
    1. Parallelisierte Simulation aller Preis-Szenarien einmalig (100% Verfügbarkeit)
    2. MC-Noise anwenden (sequenziell pro Szenario)
    3. Finale Zusammenfassung wie bisher, nur dass in den CSV Dateien die Zusammenfassung pro preiscenario vorgenommen
       werden soll. In den HTML Diagrammen, sollen so wie bisher alle MC-Runs in einer Linie dargestellt werden.
- **Abhängigkeit:** → nach Schritt 1

---

## Phase 2: Finanz-Bugfixes (Mittlere Priorität) (ABGESCHLOSSEN)

### 5. Abschreibungs-Bug PV-only (+100€ nach Jahr 10) - ERLEDIGT
- **Datei:** `finance/tax.py`
- **Aufwand:** Gering
- **Vermutung:** BESS-AfA wird auch bei BESS=0 berechnet oder Off-by-one bei AfA-Perioden.

### 6. Equity IRR Plausibilisierung - ERLEDIGT
- **Dateien:** `finance/metrics.py`, `finance/cashflow.py`
- **Aufwand:** Gering (Analyse), unklar (Fix)
- **Begründung:** Könnte durch Logik-Fixes (Baseload, AfA) bereits behoben werden. Daher am Ende prüfen.
- **Abhängigkeit:** → nach Schritt 3, 5

---

## Phase 3: CSV-Kosmetik (Niedrige Priorität, unabhängig) - ERLEDIGT

### 7. Zusätzliche Spalten im Cashflow-CSV - ERLEDIGT
- **Datei:** `output/csv_writer.py`, evtl. `finance/cashflow.py`
- **Aufwand:** Gering
- **Spalten:** BESS Green Revenue (EUR), BESS Grey Revenue (EUR), PV Revenue (EUR), PV Grid Export (MWh)
- **Begründung:** Daten existieren bereits, nur Durchreichung nötig.

---

## Phase 4: HTML/UI-Kosmetik (Niedrigste Priorität, unabhängig)

### 8. Input Wizard Anpassungen - ERLEDIGT
- **Datei:** `input/input_wizard.html`
- **Aufwand:** Mittel
- **Teilaufgaben:**
  - [x] MC-Parameter von Tab 1 → Tab 7 verschieben
  - [x] Betriebsmodus auf Tab 1, PV/BESS-Checkboxen mit bedingter Tab-Anzeige
  - [x] Diskontsatz von Tab 2 → Tab 6
  - [x] OSM-Kartenintegration (Leaflet.js, standalone HTML)
  - [x] Tab 7: graue Input-Felder fixen
  - [x] Tab 3+4: horizontale Trennlinien ergänzen
  - [x] Gesamtbreite erhöhen
  - [x] Preis-Szenario Inputs hardcoded aus full_input_example.json

### 9. Dashboard Report Anpassungen - ERLEDIGT
- **Datei:** `output/report/templates/dashboard.html`
- **Aufwand:** Mittel
- **Teilaufgaben:**
    - [x] Tooltip-Position näher an Cursor
    - [x] Multi-Line Tooltip: alle Datenreihen anzeigen
    - [x] OSM-Karte: interaktiv mit korrektem Pin (Leaflet)
    - [x] Header/Tab-Design vom Input Wizard übernehmen (ohne Grün)
    - [x] Das Hervorheben des ersten Jahres in der Cashflow Analyse ist unnötig, entferne den Orangen Kasten
    - [x] die Sensitivität im EEG-Tab soll nicht die Standardabweichung um den Mittelwert abbilden, sondern der orange
      Bereich zwischen P10 und P90 aufgespannt werden. Die Linie soll dann der Median sein. Alle Zahlen sind bereits in
      der Analyse vorhanden, es muss nur der data_collector entsprechend angepasst werden

---

## Reihenfolge & Abhängigkeiten

| Schritt | Fix                       | Abhängigkeit | Parallelisierbar?    |
|---------|---------------------------|--------------|----------------------|
| 1       | Solver → ortools          | –            | Nein (Basis)         |
| 2       | Laden/Entladen Constraint | → 1          | Nein                 |
| 3       | PPA-Baseload in LP        | → 1          | Nein                 |
| 4       | MC-Refactoring            | → 1          | Nein                 |
| 5       | AfA-Bug PV-only           | –            | Ja (parallel zu 1-4) |
| 6       | Equity IRR Prüfung        | → 3, 5       | Nein                 |
| 7       | Cashflow-CSV Spalten      | –            | Ja (parallel zu 1-4) |
| 8       | Input Wizard UI           | –            | Ja (parallel zu 1-4) |
| 9       | Dashboard Report UI       | –            | Ja (parallel zu 1-4) |
