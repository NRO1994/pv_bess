# Feature-Übersicht: PV + BESS Model Erweiterungen

## Implementierungsreihenfolge

Die Features und Fixes sind in Abhängigkeitsreihenfolge sortiert. Spätere Items können
auf früheren aufbauen.

### Phase 1: Bug Fixes & Kleine Anpassungen (abgeschlossen)

| # | Feature | Aufwand | Abhängigkeiten | Status |
|---|---------|---------|----------------|--------|
| 01 | [PPA Collar Bug Fix](01_ppa_collar_bugfix.md) | Klein (<1h) | Keine | ERLEDIGT |
| 03 | [BESS Optimierungs-OPEX](03_bess_optimization_opex.md) | Klein-Mittel (2-3h) | Keine | ERLEDIGT |
| 04 | [BESS Replacement als CAPEX](04_bess_replacement_as_capex.md) | Mittel (3-4h) | Keine | ERLEDIGT |

### Phase 2: Modell-Refactoring (abgeschlossen)

| # | Feature | Aufwand | Abhängigkeiten | Status |
|---|---------|---------|----------------|--------|
| 02 | [Systemverluste am Netzanschluss](02_system_losses_restructuring.md) | Mittel (2-4h) | Keine | ERLEDIGT |
| 05 | [MC Framework Erweiterungen](05_mc_framework_enhancements.md) | Mittel (3-4h) | Keine | ERLEDIGT |

### Phase 3: Fixes Session 2 – Quick Wins & Stabilisierung (abgeschlossen)

| # | Item | Aufwand | Abhängigkeiten | Status   |
|---|------|---------|----------------|----------|
| S2-17 | SoC Start = MIN_SOC | Klein (2 Zeilen) | Keine | ERLEDIGT |
| S2-14 | `-v` → `max_workers=1` | Klein (2 Zeilen) | Keine | ERLEDIGT |
| S2-07 | Output Directory aus JSON | Klein (3 Zeilen) | Keine | ERLEDIGT |
| S2-08 | Dezimalkomma Default | Klein-Mittel | Keine | ERLEDIGT |
| S2-06 | CSV User Input (Separator/Decimal/Timestamp) | Mittel | S2-08 | ERLEDIGT |
| S2-09 | Excel Lock Error Handling | Klein | Keine | ERLEDIGT |
| S2-10 | Debt Service Split (Interest + Repayment) | Mittel | Keine | ERLEDIGT |
| S2-11 | Grid Search Skip bei Einzel-Werten | Klein | Keine | ERLEDIGT |

### Phase 4: Fixes Session 2 – Finanzmodell & Strukturelle Änderungen

| # | Item | Aufwand | Abhängigkeiten | Status   |
|---|------|---------|----------------|----------|
| S2-12 | Collar Bug Verifikation | Klein (Diagnose) | S2-17 | OFFEN    |
| S2-15 | BESS-Replacement Upgrade-Faktor | Mittel | Keine | ERLEDIGT |
| S2-13 | BESS-Replacement fremdfinanzieren | Groß | S2-10, S2-15 | OFFEN    |
| S2-03 | BESS-Only Cases ermöglichen | Groß | Keine | OFFEN    |

### Phase 5: Großes Datenmodell-Refactoring

| # | Feature | Aufwand | Abhängigkeiten | Status |
|---|---------|---------|----------------|--------|
| 06 | [Preisszenario-Wetterjahr-Mapping](06_price_weather_scenario_mapping.md) | Groß (8-12h) | 02, 05, S2-17 | OFFEN |

### Phase 6: Neue Analysen

| # | Feature | Aufwand | Abhängigkeiten | Status |
|---|---------|---------|----------------|--------|
| 07 | [Post-Grid-Search Analysen](07_post_gridsearch_analyses.md) | Groß (8-12h) | 01, 06, S2-12, S2-13 | OFFEN |

### Phase 7: Integration & Qualitätssicherung

| # | Item | Aufwand | Abhängigkeiten | Status |
|---|------|---------|----------------|--------|
| 09 | [Integration Test Suite](09_integration_suite.md) | Groß (8-12h) | S2-03, S2-17 | OFFEN |
| S2-01 | Cashflow Benchmark-Test | Mittel | S2-08, S2-10 | OFFEN |
| S2-02 | Smoke Test | Mittel | Keine | OFFEN |

### Phase 8: Reporting

| # | Feature | Aufwand | Abhängigkeiten | Status |
|---|---------|---------|----------------|--------|
| 08 | [PDF Report](08_pdf_report.md) | Groß (12-16h) | 06, 07 | OFFEN |

## Fixes mit Überschneidungen zu Features

| Fix | Status | Begründung |
|-----|--------|------------|
| S2-04 (OPEX eur_per_kw/kwh) | BEREITS IMPLEMENTIERT | Code unterstützt dies bereits |
| S2-05 (Loan Tenor) | BEREITS IMPLEMENTIERT | Korrekt implementiert in `debt.py` |
| S2-16 (P90-DSCR entfernen) | ABGEDECKT DURCH FEATURE 06 | Feature 06 eliminiert P90 komplett und ersetzt es durch `debt_sizing_downside_percentage`. Separater Fix nicht nötig. |

## Gesamtaufwand: ~20-30 Arbeitstage

## Abhängigkeitsgraph

```
Phase 1-2 (ERLEDIGT)
  01 (Collar Fix)
  02 (Systemverluste)
  03 (BESS Opt. OPEX)
  04 (BESS Repl. CAPEX)
  05 (MC Erweiterungen)

Phase 3 (Quick Wins)
  S2-17 (SoC MIN_SOC) ────────────┐
  S2-14 (-v workers) ──┐          │
  S2-12 (Collar Verify) ◄─────────┘
  S2-07 (Output Dir)               │
  S2-08 (Dezimalkomma)─→ S2-06 (CSV Input)
  S2-09 (Excel Lock)               │
  S2-10 (Debt Split)               │
  S2-11 (GS Skip)                  │

Phase 4 (Finanzmodell)              │
  S2-15 (Repl. Upgrade)──┐         │
  S2-13 (Repl. Debt) ◄───┘         │
  S2-03 (BESS-Only)                 │

Phase 5                              │
  06 (Preis-Wetter-Mapping) ◄───────┘
       │
       ▼
Phase 6
  07 (Post-GS Analysen) ◄── S2-12, S2-13
       │
       ▼
Phase 7
  09 (Integration Suite) ◄── S2-03, S2-17
  S2-01 (Benchmark) ◄── S2-08, S2-10
  S2-02 (Smoke Test)
       │
       ▼
Phase 8
  08 (PDF Report) ◄── 06, 07
```

## Implementierungsplan nach Sessions

### Session 3: Quick Wins & Kosmetik
**Modell:** Sonnet (schnell, routinierte Änderungen)
**Dauer:** ~4-6h
**Inhalt:**
1. S2-17 – SoC Start = MIN_SOC (2 Zeilen)
2. S2-14 – `-v` → `max_workers=1` (2 Zeilen)
3. S2-12 – Collar Bug verifizieren (Diagnose + ggf. Logging)
4. S2-07 – Output Directory aus JSON (3 Zeilen)
5. S2-08 – Dezimalkomma Default
6. S2-06 – CSV User Input (Separator/Decimal/Timestamp)
7. S2-09 – Excel Lock Error Handling
8. S2-11 – Grid Search Skip bei Einzel-Werten
9. S2-02 – Smoke Test (JSON-Fix + Integrationstest)

### Session 4: Finanzmodell-Erweiterungen
**Modell:** Opus (komplexe Finanzlogik, viele Abhängigkeiten)
**Dauer:** ~6-8h
**Inhalt:**
1. S2-10 – Debt Service Split (Interest + Repayment)
2. S2-15 – BESS-Replacement Upgrade-Faktor
3. S2-13 – BESS-Replacement fremdfinanzieren
4. S2-03 – BESS-Only Cases ermöglichen
5. S2-01 – Cashflow Benchmark-Test + Analyse

### Session 5: Datenmodell-Refactoring (Feature 06)
**Modell:** Opus (fundamentale Architekturänderung)
**Dauer:** ~10-14h
**Inhalt:**
1. Feature 06 – Preisszenario-Wetterjahr-Mapping mit 15min-Auflösung
   - Eliminiert P90 (inkl. FIX-S2-16)
   - 9 Szenarien statt 3
   - 15min-Auflösung
   - Wochentag-Alignment

### Session 6: Post-Grid-Search Analysen (Feature 07)
**Modell:** Opus (komplexe MC-basierte Sweep-Analysen)
**Dauer:** ~8-12h
**Inhalt:**
1. Feature 07 – EEG Floor Sweep, PPA Collar 2D Sweep, PPA Baseload 2D Sweep

### Session 7: Integration Test Suite (Feature 09)
**Modell:** Sonnet (viele parallele, aber strukturell einfache Tests)
**Dauer:** ~8-10h
**Inhalt:**
1. Feature 09 – 36-Szenario-Matrix + Dispatch-Constraint-Checker + KPI-Ranking + Availability-Checker
2. Preis-CSV generieren
3. PVGIS-Cache vorbefüllen

### Session 8: PDF Report (Feature 08)
**Modell:** Opus (LLM-Integration, komplexes Layout)
**Dauer:** ~12-16h
**Inhalt:**
1. Feature 08 – PDF Report mit Charts, Tabellen und LLM-generiertem Text

## Hinweise zur Implementierung

- Jedes Feature/Fix soll als eigenständiger Commit implementiert werden
- Nach jedem Item: Tests laufen lassen (`pytest`)
- Die CLAUDE.md muss nach Abschluss aller Features aktualisiert werden
- Es bedarf keiner Backward-Compatibility, alles kann frei geändert und angepasst werden
- **Modell-Empfehlung:** Sonnet für routinierte, kleinere Änderungen; Opus für komplexe Architektur- und Finanz-Logik
