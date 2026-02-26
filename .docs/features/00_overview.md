# Feature-Übersicht: PV + BESS Model Erweiterungen

## Implementierungsreihenfolge

Die Features sind in Abhängigkeitsreihenfolge sortiert. Spätere Features können
auf früheren aufbauen.

### Phase 1: Bug Fixes & Kleine Anpassungen (1-2 Tage)

| # | Feature | Aufwand | Abhängigkeiten | Status |
|---|---------|---------|----------------|--------|
| 01 | [PPA Collar Bug Fix](01_ppa_collar_bugfix.md) | Klein (<1h) | Keine | ERLEDIGT |
| 03 | [BESS Optimierungs-OPEX](03_bess_optimization_opex.md) | Klein-Mittel (2-3h) | Keine | ERLEDIGT |
| 04 | [BESS Replacement als CAPEX](04_bess_replacement_as_capex.md) | Mittel (3-4h) | Keine | ERLEDIGT |

### Phase 2: Modell-Refactoring (2-3 Tage)

| # | Feature | Aufwand | Abhängigkeiten | Status   |
|---|---------|---------|----------------|----------|
| 02 | [Systemverluste am Netzanschluss](02_system_losses_restructuring.md) | Mittel (2-4h) | Keine | ERLEDIGT |
| 05 | [MC Framework Erweiterungen](05_mc_framework_enhancements.md) | Mittel (3-4h) | Keine | ERLEDIGT |

### Phase 3: Großes Datenmodell-Refactoring (3-5 Tage)

| # | Feature | Aufwand | Abhängigkeiten |
|---|---------|---------|----------------|
| 06 | [Preisszenario-Wetterjahr-Mapping](06_price_weather_scenario_mapping.md) | Groß (8-12h) | 02, 05 |

### Phase 4: Neue Analysen (3-5 Tage)

| # | Feature | Aufwand | Abhängigkeiten |
|---|---------|---------|----------------|
| 07 | [Post-Grid-Search Analysen](07_post_gridsearch_analyses.md) | Groß (8-12h) | 01, 06 |

### Phase 5: Reporting (3-5 Tage)

| # | Feature | Aufwand | Abhängigkeiten |
|---|---------|---------|----------------|
| 08 | [PDF Report](08_pdf_report.md) | Groß (12-16h) | 06, 07 |

## Gesamtaufwand: ~12-20 Arbeitstage

## Abhängigkeitsgraph

```
01 (Collar Fix)  ──────────────────────┐
02 (Systemverluste)  ──┐               │
03 (BESS Opt. OPEX)    │               │
04 (BESS Repl. CAPEX)  │               │
05 (MC Erweiterungen) ─┤               │
                       ▼               │
              06 (Preis-Wetter-        │
                  Mapping)             │
                       │               │
                       ▼               ▼
              07 (Post-GS Analysen) ◄──┘
                       │
                       ▼
              08 (PDF Report)
```

## Hinweise zur Implementierung

- Jedes Feature hat seine eigene Markdown-Datei mit allen Details
- Jedes Feature soll als eigenständiger Commit/PR implementiert werden
- Nach jedem Feature: Tests laufen lassen (`pytest`)
- Die CLAUDE.md muss nach Abschluss aller Features aktualisiert werden
- Es bedarf keiner Back-ward compability, alles kann frei geändert und angepasst werden
