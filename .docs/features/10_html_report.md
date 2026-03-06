# Feature 10: Interaktiver HTML-Report (Output-Dashboard)

## Prioritaet: Hoch
## Aufwand: Gross (16-24h, 4 Phasen)

## Uebersicht

Erstellung eines interaktiven, offline-faehigen HTML-Dashboards als Ergebnis-Report.
Der Report ersetzt den bisherigen PDF-basierten Report (`pdf_builder.py` + `weasyprint`).
Es handelt sich um eine einzelne, selbststaendige HTML-Datei ohne externe Abhaengigkeiten,
die im Browser (Edge/Chrome/Firefox) offline funktioniert.

**Zentrale Anforderungen:**
- Single-File HTML, kein Backend, keine Ports, keine externen CDNs
- Sprache: Deutsch
- Corporate Design (Farbpalette aus `08_pdf_report.md`)
- Tool-Logo (`.data/tool_logo.png`) + Unternehmenslogo (`.data/logo_stadtwerke_luebeck.png`)
  als Base64 eingebettet
- Alle Diagramme und Daten herunterladbar (PNG/CSV)
- LLM-generierte Texte via manuellem Copilot-Workflow (kein API-Zugriff)
- Tab-basiertes Layout mit konditionalen Tabs (nur anzeigen wenn Analyse durchgefuehrt)

## Referenz-Dateien (Bestand)

| Datei | Relevanz |
|-------|----------|
| `pv_bess_model/output/report/templates/result_dashboard.html` | Vorlage fuer interaktive Charts (Canvas-basiert, Zoom/Pan/Tooltip) |
| `pv_bess_model/output/report/templates/copilot_sample.html` | Vorlage fuer Input-Wizard (Stepper-Pattern) |
| `pv_bess_model/output/report/charts.py` | Bestehende matplotlib-Chart-Generierung (6 Chart-Typen) |
| `pv_bess_model/output/report/pdf_builder.py` | Bisheriger PDF-Report (wird durch HTML ersetzt) |
| `pv_bess_model/output/report/llm_client.py` | Bisheriger LLM-Client (Anthropic API, wird umgebaut) |
| `pv_bess_model/main.py:_generate_report()` | Integration in den Hauptfluss (Zeile 1146ff) |
| `pv_bess_model/output/csv_writer.py` | CSV-Output (parallel zum HTML-Report) |
| `pv_bess_model/finance/cashflow.py` | AnnualCashflow/CashflowProjection Datenstrukturen |
| `pv_bess_model/finance/metrics.py` | FinancialMetrics Datenstruktur |
| `pv_bess_model/dispatch/engine.py` | AnnualResult mit Revenue-Breakdown |
| `pv_bess_model/optimization/grid_search.py` | GridSearchResult/GridPointResult |
| `input_wizard.html` | Bestehender Input-Wizard (eigenstaendig) |

## Tab-Struktur des HTML-Reports

| Tab | Titel | Bedingung | Diagramm | LLM-Text |
|-----|-------|-----------|----------|----------|
| 1 | Szenario-Uebersicht | Immer | OpenStreetMap-Karte (wenn online) | Zusammenfassung der Schluesselparameter |
| 2 | Eingangszeitreihen | Immer | PV-Ertrag + Strompreisszenarien | Erklaerung der Eingangsdaten |
| 3 | EEG-Analyse | `marketing.type == "eeg"` | EEG-Sensitivitaet | Analyse der EEG-Ergebnisse |
| 4 | PPA-Collar-Analyse | Collar-Result vorhanden | PPA-Collar-Chart | Analyse der Collar-Ergebnisse |
| 5 | PPA-Baseload-Analyse | Baseload-Result vorhanden | PPA-Baseload-Chart | Analyse der Baseload-Ergebnisse |
| 6 | Cashflow-Analyse | Immer | Gestapeltes Saeulendiagramm | KPIs + Cashflow-Einschaetzung |

---

## Phase 1: Datenaufbereitung und LLM-Prompt-Template

### Ziel
Alle Simulationsergebnisse in eine JSON-Struktur ueberfu ehren, die als Datenbasis
fuer den HTML-Report dient. Ausserdem das LLM-Prompt-Template erstellen, das spaeter
manuell an Copilot uebergeben wird.

### 1.1 Report-Daten-Aggregator (`output/report/data_collector.py` – NEU)

Neues Modul, das alle Report-relevanten Daten in ein einzelnes Dictionary/Dataclass
zusammenfasst:

```python
@dataclass
class HtmlReportData:
    # Meta
    scenario_name: str
    scenario_json_filename: str
    creation_date: str  # DD.MM.YYYY
    commissioning_year: int

    # Input-Parameter (Tab 1)
    pv_peak_kwp: float
    pv_azimuth: float
    pv_tilt: float
    pv_degradation_pct: float
    bess_scale_range: list[float]
    bess_ep_ratios: list[float]
    bess_rte_pct: float
    grid_max_export_kw: float
    operating_mode: str  # "green" / "grey"
    marketing_type: str  # "eeg", "ppa_floor", "ppa_collar", "ppa_baseload", ...
    marketing_params: dict  # floor_price, cap_price, duration, etc.
    latitude: float
    longitude: float
    lifetime_years: int
    leverage_pct: float
    interest_rate_pct: float
    loan_tenor_years: int
    inflation_rate: float

    # Zeitreihen-Daten (Tab 2)
    pv_monthly_by_year: dict[int, list[float]]  # {weather_year: [jan..dec in GWh]}
    price_scenario_annual_means: list[dict]  # [{name, weather_year, means: [y1..yn]}]

    # Sensitivity-Ergebnisse (Tab 3-5, optional)
    eeg_sensitivity: list[dict] | None  # [{floor_ct_kwh, irr_mean, irr_std}]
    ppa_collar: list[dict] | None       # [{spread, floor, irr_mean}]
    ppa_baseload: list[dict] | None     # [{baseload_mw, ppa_price, irr_mean}]

    # Grid-Search (Tab 6 Kontext)
    grid_search_points: list[dict]  # [{scale, ep, irr, is_optimal}]
    optimal_scale_pct: float
    optimal_ep_ratio: float
    optimal_bess_power_kw: float
    optimal_bess_capacity_kwh: float

    # Cashflow (Tab 6)
    cashflow_years: list[dict]  # Pro Jahr: {year, rev_pv, rev_bess_green, rev_bess_grey,
                                #            grid_import_cost, capex, opex, debt_service,
                                #            tax_total, equity_cf, cumulative_equity_cf}

    # KPIs (Tab 6)
    metrics: dict  # {equity_irr, project_irr, npv, dscr_min, dscr_avg, lcoe, payback_year}

    # Logos (Base64-encoded)
    tool_logo_b64: str | None
    company_logo_b64: str | None

    # LLM-Texte (nach manuellem Copilot-Schritt)
    llm_texts: dict[str, str]  # {tab_1: "...", tab_2: "...", ...}
```

**Aufgaben:**
- [ ] Dataclass `HtmlReportData` definieren
- [ ] Factory-Funktion `collect_report_data(...)` implementieren, die aus den bestehenden
      Ergebnisstrukturen (`GridSearchResult`, `CashflowProjection`, `FinancialMetrics`,
      `AnnualResult`, Szenario-Config, etc.) die Daten extrahiert
- [ ] Base64-Encoding der Logo-PNGs (`.data/tool_logo.png`, `.data/logo_stadtwerke_luebeck.png`)
- [ ] Cashflow-Daten aufbereiten: Revenue-Breakdown (PV, BESS-Green, BESS-Grey) als positive
      Werte, Kosten (CAPEX, OPEX, Debt, Tax, Grid-Import, Baseload-Matching) als negative Werte
      – passend fuer das gestapelte Saeulendiagramm
- [ ] `HtmlReportData` als JSON serialisierbar machen (fuer Einbettung in HTML)

### 1.2 LLM-Prompt-Template (``.docs/llm_templates/report_prompt.md`` – NEU)

Ein einzelnes Prompt-Template, das mit den Ergebnisdaten befuellt wird und als **ein
Prompt** in Copilot eingefuegt werden kann. Das Template erzwingt eine strukturierte
JSON-Antwort, damit die Texte zuverlaessig den richtigen Tabs zugeordnet werden.

**Aufgaben:**
- [ ] Prompt-Template erstellen mit Platzhaltern (`{{scenario_name}}`, `{{metrics}}`, etc.)
- [ ] Klare Anweisung an die LLM, die Antwort als JSON-Objekt mit definierten Keys zu
      liefern:
      ```json
      {
        "tab_1_overview": "...",
        "tab_2_timeseries": "...",
        "tab_3_eeg": "..." | null,
        "tab_4_collar": "..." | null,
        "tab_5_baseload": "..." | null,
        "tab_6_cashflow": "..."
      }
      ```
- [ ] Jeder Text-Key enthaelt Anweisungen, welche Schluesselbegriffe **fett** markiert
      werden sollen (der HTML-Report rendert `**...**` dann als `<strong>`)
- [ ] Template soll ca. 500-800 Woerter Kontext enthalten (kompakte Datenzusammenfassung),
      nicht die Rohdaten
- [ ] Prompt-Rendering-Funktion: `render_prompt(data: HtmlReportData) -> str`
      befuellt das Template mit aktuellen Werten

### 1.3 Prompt-Generierung und Speicherung

**Aufgaben:**
- [ ] Funktion `save_rendered_prompt(data: HtmlReportData, output_dir: Path) -> Path`
      die den befuellten Prompt als `.md`-Datei im Output-Directory speichert
      (`{scenario_name}_llm_prompt.md`)
- [ ] Funktion `load_llm_response(output_dir: Path, scenario_name: str) -> dict[str, str] | None`
      die eine JSON-Datei (`{scenario_name}_llm_response.json`) liest und die Texte parst
- [ ] Validierung: Pruefen ob alle erwarteten Keys vorhanden sind, fehlende Keys mit
      Platzhaltertext ersetzen

---

## Phase 2: HTML-Template und Chart-Rendering (JavaScript)

### Ziel
Das HTML-Template erstellen, das alle Diagramme clientseitig via Canvas rendert
(analog zum bestehenden `result_dashboard.html`), aber mit der echten Datenstruktur
aus Phase 1.

### 2.1 HTML-Template (`output/report/templates/dashboard.html` – NEU)

Das Template ist eine vollstaendige HTML-Datei mit einem einzelnen `<script>`-Block
am Ende, der einen Platzhalter `const REPORT_DATA = {{REPORT_DATA_JSON}};` enthaelt.
Dieser Platzhalter wird zur Build-Zeit durch die serialisierten `HtmlReportData` ersetzt.

**Aufbau:**
```
<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8">
  <title>PV+BESS Ergebnis-Report – {{scenario_name}}</title>
  <style>
    /* Corporate Design CSS (Farben, Fonts, Layout) */
    /* Tab-System CSS */
    /* Chart-Container CSS */
    /* Responsive / Print CSS */
  </style>
</head>
<body>
  <!-- Header: Logos + Szenarioname + Datum -->
  <!-- Tab-Navigation (dynamisch basierend auf vorhandenen Daten) -->
  <!-- Tab 1-6: Content Panels -->
  <!-- Footer: Modellversion, Disclaimer -->

  <script>
    const REPORT_DATA = {{REPORT_DATA_JSON}};
    // Tab-Logik
    // Chart-Rendering (Canvas-basiert)
    // Download-Logik (PNG/CSV)
    // Dark-Mode Toggle
    // OpenStreetMap Tile (optional, wenn online)
  </script>
</body>
</html>
```

**Aufgaben:**
- [ ] HTML-Grundgeruest mit Corporate Design CSS erstellen
  - Farbpalette: `#FF8200`, `#F73E5E`, `#A51BA7`, `#00467A`, `#006EB2`, `#00BDDC`
  - Font-Stack: `-apple-system, Segoe UI, Roboto, Inter, system-ui, Arial, sans-serif`
  - Dark-Mode Support via `[data-theme="dark"]` CSS-Variablen
- [ ] Header-Bereich: Tool-Logo (links) + Unternehmenslogo (rechts), aus Base64
- [ ] Tab-Navigation: Dynamisch nur Tabs anzeigen, fuer die Daten vorhanden sind
      (Tab 3-5 konditional ueber `REPORT_DATA.eeg_sensitivity !== null` etc.)
- [ ] Footer: Erstellungsdatum, Modellversion, Szenarioname, Input-JSON-Dateiname

### 2.2 Canvas-basierte Chart-Klassen (JavaScript, inline im Template)

Die Charts werden direkt im HTML als JavaScript gerendert (kein Chart.js o.ae.),
analog zum bestehenden `result_dashboard.html`. Die bestehenden Klassen
`InteractiveCartesianChart` und `InteractiveHeatmap` werden als Basis verwendet
und erweitert.

**Neue Chart-Typen:**

| Chart | Tab | Typ | Beschreibung |
|-------|-----|-----|--------------|
| PV-Ertrag | 2 | Multi-Line | Monatliche Produktion pro Wetterjahr (x: Monat, y: GWh) |
| Strompreise | 2 | Multi-Line | Jahresmittelpreis pro Szenario ueber Projektlaufzeit |
| EEG-Sensitivitaet | 3 | Line + Band | IRR vs. Gebotspreis mit Std.Abw.-Band |
| PPA-Collar | 4 | Multi-Line | IRR vs. Floor-Preis, gruppiert nach Cap-Spread |
| PPA-Baseload | 5 | Multi-Line | IRR vs. PPA-Preis, gruppiert nach Baseload-Level |
| Cashflow | 6 | Stacked Bar | Gestapelte Saeulen (Revenue positiv, Kosten negativ) |

**Aufgaben:**
- [ ] `InteractiveCartesianChart` aus `result_dashboard.html` uebernehmen und anpassen:
  - Multi-Series-Support (mehrere Linien mit Legende)
  - Band/Fill-Support (fuer EEG Std.Abw.-Band)
- [ ] Neuen Chart-Typ `StackedBarChart` implementieren:
  - Positive Stapel: Revenue PV, Revenue BESS Green, Revenue BESS Grey
  - Negative Stapel: CAPEX, OPEX, Debt Service, Total Tax, Grid Import Cost,
    Baseload Matching Cost (sofern vorhanden)
  - X-Achse: Kalenderjahr (commissioning_year + offset)
  - Tooltip: Aufschluesselung aller Komponenten pro Jahr
  - Sekundaerlinie (optional): Kumulative Equity-CF als Linie ueberlagert
- [ ] OpenStreetMap-Kachel fuer Tab 1 (optional, nur wenn online):
  - `<img>` mit Tile-URL: `https://tile.openstreetmap.org/{z}/{x}/{y}.png`
  - Marker-Overlay via Canvas auf dem Tile-Image
  - Fallback wenn offline: nur Text "Standort: {lat}, {lon}"
- [ ] Download-Buttons pro Tab:
  - PNG: `canvas.toDataURL('image/png')` → Download
  - CSV: Daten als Semikolon-separierte CSV mit deutschem Dezimaltrennzeichen

### 2.3 LLM-Text-Rendering

**Aufgaben:**
- [ ] Pro Tab einen `<div class="llm-text">` Bereich, der den LLM-Text anzeigt
- [ ] Markdown-aehliches Rendering: `**fett**` → `<strong>`, Absaetze → `<p>`
      (einfacher Regex-basierter Parser, kein Markdown-Library noetig)
- [ ] Fallback-Text wenn kein LLM-Text vorhanden: "Textbeschreibung nicht verfuegbar.
      Bitte fuehren Sie den LLM-Prompt-Workflow durch."

---

## Phase 3: HTML-Builder (Python-seitig)

### Ziel
Python-Logik, die das HTML-Template mit den echten Daten befuellt und als
einzelne HTML-Datei speichert.

### 3.1 HTML-Builder (`output/report/html_builder.py` – NEU)

Ersetzt `pdf_builder.py` als primaerer Report-Generator.

```python
def build_html_report(
    data: HtmlReportData,
    output_dir: Path,
) -> Path:
    """Erstelle den interaktiven HTML-Report.

    1. Lade das Template aus templates/dashboard.html
    2. Serialisiere HtmlReportData als JSON
    3. Ersetze {{REPORT_DATA_JSON}} im Template
    4. Ersetze {{scenario_name}} im <title>
    5. Schreibe die fertige HTML-Datei nach output_dir/
    """
```

**Aufgaben:**
- [ ] Template laden und Platzhalter ersetzen
- [ ] JSON-Serialisierung der `HtmlReportData` (kompakt, keine NaN/Infinity –
      diese durch `null` ersetzen)
- [ ] Output-Datei: `{output_dir}/{scenario_name}_report.html`
- [ ] Sicherstellen, dass die HTML-Datei bei erneutem Lauf ueberschrieben wird
      (analog zu CSV-Verhalten)

### 3.2 Refactoring: `_generate_report()` in `main.py`

Die bestehende Funktion `_generate_report()` (main.py:1146) wird umgebaut:

**Bisheriger Flow:**
1. `create_all_charts()` → PNG-Dateien via matplotlib
2. LLM-Texte via Anthropic API (optional)
3. `build_report()` → PDF via weasyprint

**Neuer Flow:**
1. `collect_report_data()` → `HtmlReportData` Objekt
2. `save_rendered_prompt()` → `{scenario}_llm_prompt.md` im Output-Dir
3. `load_llm_response()` → Versuche `{scenario}_llm_response.json` zu laden
4. `build_html_report()` → `{scenario}_report.html`

**Aufgaben:**
- [ ] `_generate_report()` auf neuen Flow umbauen
- [ ] `charts.py` weiterhin fuer PNG-Export nutzen (Charts werden weiterhin als
      PNG gespeichert, zusaetzlich zum interaktiven HTML)
- [ ] `pdf_builder.py` entfernen (keine Backward-Compatibility noetig)
- [ ] `llm_client.py` umbauen: Anthropic-API-Code entfernen, stattdessen nur
      Prompt-Template-Rendering und Response-Parsing
- [ ] CLI-Flags anpassen:
  - `--no-report`: Kein HTML-Report (wie bisher kein PDF)
  - `--no-llm` entfernen (nicht mehr relevant, da kein API-Zugriff)
  - Neues optionales Flag: `--llm-response <path>` fuer explizite Angabe der
    LLM-Response-Datei (default: auto-detect im Output-Dir)

### 3.3 Aufraeum-Arbeiten

**Aufgaben:**
- [ ] `pdf_builder.py` loeschen
- [ ] `templates/report.html` loeschen (altes PDF-Template)
- [ ] `weasyprint` aus Dependencies entfernen (`pyproject.toml`)
- [ ] `anthropic` aus den zwingenden Dependencies entfernen (war vorher optional,
      wird jetzt gar nicht mehr benoetigt)
- [ ] `llm_client.py` reduzieren auf Prompt-Rendering + Response-Parsing
      (Klasse `LLMClient` entfernen, durch einfache Funktionen ersetzen)
- [ ] Alte Report-Konstanten in `config/defaults.py` pruefen und aufraeumen:
  - `REPORT_PDF_FILENAME_SUFFIX` → ersetzen durch `REPORT_HTML_FILENAME_SUFFIX`
  - `REPORT_LLM_CACHE_FILENAME` → entfernen
  - `REPORT_LLM_DEFAULT_MODEL` → entfernen
  - `REPORT_LLM_MAX_TOKENS` → entfernen
  - Neue Konstanten: `REPORT_LLM_PROMPT_FILENAME`, `REPORT_LLM_RESPONSE_FILENAME`

---

## Phase 4: Integration, Test, Feinschliff

### Ziel
End-to-End-Integration, manueller Test mit echtem Szenario, Feinschliff der
Darstellung.

### 4.1 End-to-End Integration

**Aufgaben:**
- [ ] Vollstaendigen Durchlauf mit einem echten Szenario testen
- [ ] Pruefen: HTML-Datei oeffnet korrekt in Edge, Chrome, Firefox
- [ ] Pruefen: Alle Tabs werden korrekt konditional angezeigt/versteckt
- [ ] Pruefen: Alle Charts rendern korrekt mit echten Daten
- [ ] Pruefen: PNG- und CSV-Downloads funktionieren
- [ ] Pruefen: Dark-Mode funktioniert
- [ ] Pruefen: OpenStreetMap-Karte zeigt korrekten Standort (online) / Fallback (offline)
- [ ] Pruefen: LLM-Prompt wird korrekt generiert und gespeichert
- [ ] Pruefen: LLM-Response wird korrekt geladen und in Tabs eingefuegt
- [ ] Pruefen: Report wird bei erneutem Lauf ueberschrieben

### 4.2 LLM-Workflow testen

Manueller Test des vollstaendigen LLM-Workflows:
1. Simulation laeuft → `{scenario}_llm_prompt.md` wird erzeugt
2. Prompt manuell in Copilot Chat einfuegen
3. Copilot-Antwort als `{scenario}_llm_response.json` speichern
4. Simulation erneut laufen (oder separates Skript) → HTML-Report mit LLM-Texten

**Aufgaben:**
- [ ] Workflow-Dokumentation in README oder separater .md-Datei
- [ ] Validierung: Was passiert bei fehlerhafter/unvollstaendiger LLM-Response?
- [ ] Fallback-Texte fuer alle Tabs wenn keine LLM-Response vorhanden

### 4.3 Cashflow-Saeulendiagramm Feinschliff

Das Cashflow-Diagramm (Tab 6) ist das komplexeste Chart und benoetigt besondere
Aufmerksamkeit:

**Aufgaben:**
- [ ] Farbzuordnung:
  - Revenue PV: `#FF8200` (Primaerfarbe)
  - Revenue BESS Green: `#00BDDC` (Hellblau)
  - Revenue BESS Grey: `#006EB2` (Mittelblau)
  - CAPEX: `#00467A` (Dunkelblau, negativ)
  - OPEX: `#A51BA7` (Lila, negativ)
  - Debt Service: `#F73E5E` (Rot, negativ)
  - Tax: `#94a3b8` (Grau, negativ)
  - Grid Import / Baseload Matching: Nur wenn > 0 (separater Negativbalken)
- [ ] Legende unterhalb des Diagramms
- [ ] Tooltip zeigt alle Komponenten fuer das jeweilige Jahr
- [ ] Kumulative Equity-CF als Linie (optional, mit eigener Y-Achse)
- [ ] Jahr 1 hervorheben (CAPEX-dominiert)

### 4.4 Unit Tests

**Aufgaben:**
- [ ] `test_data_collector.py`: Test der Datenaufbereitung
  - Korrekte Extraktion aus GridSearchResult, CashflowProjection, etc.
  - Base64-Encoding der Logos
  - JSON-Serialisierbarkeit
- [ ] `test_html_builder.py`: Test der HTML-Generierung
  - Template-Platzhalter werden korrekt ersetzt
  - Resultierende HTML-Datei ist valide (enthaelt `<!DOCTYPE html>`, `</html>`)
  - Szenarioname im `<title>`
- [ ] `test_llm_prompt.py`: Test des Prompt-Renderings
  - Platzhalter werden korrekt befuellt
  - Response-Parsing mit gueltigem/ungueltigem JSON
  - Fehlende Keys werden durch Fallback-Text ersetzt
- [ ] Bestehende Report-Tests anpassen (PDF-Tests entfernen)

---

## Aenderungen an bestehenden Dateien (Zusammenfassung)

| Datei | Aenderung |
|-------|-----------|
| `output/report/data_collector.py` | **NEU**: Daten-Aggregation |
| `output/report/html_builder.py` | **NEU**: HTML-Report-Generierung |
| `output/report/templates/dashboard.html` | **NEU**: Interaktives HTML-Template |
| `.docs/llm_templates/report_prompt.md` | **NEU**: LLM-Prompt-Template |
| `output/report/llm_client.py` | **UMBAU**: API-Code entfernen, nur Prompt-Rendering + Response-Parsing |
| `output/report/pdf_builder.py` | **LOESCHEN** |
| `output/report/templates/report.html` | **LOESCHEN** |
| `output/report/charts.py` | **BEHALTEN**: Weiterhin PNG-Export (optional neben HTML) |
| `main.py:_generate_report()` | **UMBAU**: Neuer Flow (collect → prompt → build HTML) |
| `config/defaults.py` | **ANPASSEN**: Alte PDF/LLM-Konstanten entfernen, neue HTML-Konstanten |
| `pyproject.toml` | **ANPASSEN**: weasyprint entfernen, anthropic optional machen |
| `tests/` | **ANPASSEN**: PDF-Tests entfernen, neue Tests fuer HTML-Report |

## User-Workflow (nach Implementation)

```
1. User fuehrt Simulation aus:
   $ python -m pv_bess_model.main --scenario my_scenario.json

2. Output-Directory enthaelt:
   - my_scenario_summary.csv
   - my_scenario_cashflows.csv
   - my_scenario_grid_search.csv
   - my_scenario_dispatch_sample.csv
   - my_scenario_llm_prompt.md          ← NEU
   - my_scenario_report.html            ← NEU (ohne LLM-Texte, mit Platzhaltern)
   - charts/*.png                       ← wie bisher

3. User kopiert Inhalt von my_scenario_llm_prompt.md in Copilot Chat

4. User speichert Copilot-Antwort als my_scenario_llm_response.json im Output-Dir

5. User fuehrt erneut aus (oder separates Kommando):
   $ python -m pv_bess_model.main --scenario my_scenario.json
   → HTML-Report wird mit LLM-Texten neu generiert

Alternative: Beim ersten Lauf wird der Report mit Platzhalter-Texten erzeugt,
der User kann die LLM-Response nachtraeglich hinzufuegen und den Report
separat neu bauen lassen.
```

## Offene Design-Entscheidungen

1. **Soll der HTML-Report auch ohne LLM-Texte vollstaendig nutzbar sein?**
   → Ja, Platzhalter-Texte wie "Beschreibung folgt nach LLM-Auswertung" anzeigen.

2. **Separates CLI-Kommando fuer Report-Rebuild?**
   → Optional: `--rebuild-report` Flag, das nur den HTML-Report neu generiert
   (ohne Simulation), nuetzlich nach Einfuegen der LLM-Response.

3. **Charts: matplotlib-PNG oder rein Canvas-basiert?**
   → Beides: matplotlib-PNGs werden weiterhin erzeugt (fuer standalone-Nutzung),
   die HTML-Datei rendert alle Charts interaktiv via Canvas/JavaScript.
   Die matplotlib-Charts sind damit optional und koennen mit `--no-charts` unterdrueckt werden.
