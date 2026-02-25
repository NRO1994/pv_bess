# Feature 08: PDF-Report im Präsentationsformat

## Priorität: Niedrig (nach allen anderen Features)
## Aufwand: Groß (12-16h)

## Beschreibung

Erstellung eines professionellen PDF-Reports im Präsentationsstil. Der Report fasst
alle Analyseergebnisse zusammen und enthält Diagramme + LLM-generierte Texte.

**Design-Prinzipien:**
- Professioneller Corporate-Look
- Texte werden per LLM-API generiert (kosteneffizient)
- Diagramme werden lokal als PNG erzeugt (matplotlib/plotly)
- LLM-API kann per CLI deaktiviert werden → dann nur CSV + PNG
- Template-basierter Aufbau (LLM liefert nur Text, Layout = Template)

## Corporate Design

**Farbpalette:**

| Level | Hex | Verwendung |
|-------|-----|-----------|
| 1 | `#FF8200` | Primärfarbe, Akzente, Überschriften |
| 2 | `#F73E5E` | Sekundärfarbe, Highlights |
| 3 | `#A51BA7` | Tertiär, Diagramm-Reihe 3 |
| 4 | `#00467A` | Dunkel, Text-Elemente |
| 5 | `#006EB2` | Mittel, Diagramm-Reihe 5 |
| 6 | `#00BDDC` | Hell, Hintergründe, Diagramm-Reihe 6 |

## Seiten-Struktur

### Deckblatt
- FIRMENNAME (User-Input aus JSON)
- Projekt-Name (= `scenario.name`)
- Datum (aktuelles Datum)
- Model-Version (aus `pyproject.toml` oder Konstante)
- Firmenlogo (PNG aus `data/`-Ordner, Pfad im JSON konfigurierbar)

### Seite 0: Modellbeschreibung
- Text: Allgemeine Beschreibung des Modells, Vorgehensweise, Finanzmodell
- Quelle: CLAUDE.md → LLM fasst die relevanten Abschnitte zusammen
- Rein textbasiert, kein Diagramm

### Seite 1: Input-Parameter
- Tabelle der wichtigsten Parameter aus dem JSON
- PV-Konfiguration, BESS-Design, Finanzparameter, Vermarktungsmodell
- LLM: Kurze Beschreibung der gewählten Konfiguration

### Seite 2: PV-Ertragsberechnung
- **Diagramm:** Ertragsprofil über das Jahr (x: Monat, y: kWh)
  - Alle Wetterjahre als einzelne Linien (zeigt Unsicherheit)
  - Oder: Monatliche Summen als Balken mit Fehlerbalken
- **Text (LLM):** Erklärung der PV-Ertragsberechnung, Interpretation der Variabilität

### Seite 3: Strompreisszenarien
- **Diagramm:** Alle Preisszenarien über die Zeit
  - x: Jahr (Projektlaufzeit)
  - y: Mittlerer Jahrespreis (EUR/MWh)
  - Eine Linie pro Szenario (9 Linien)
- **Text (LLM):** Beschreibung der Szenarien, wesentliche Unterschiede

### Seite 4: Grid-Search-Analyse
- **Diagramm:** Kurvenschar
  - x: BESS-Leistung in % der PV-Anlage
  - y: Equity IRR (%)
  - Eine Kurve pro E/P-Ratio (1h, 2h, 4h, ...)
  - Optimaler Punkt markiert
- **Text (LLM):** Erkenntnisse aus der Grid-Search

### Seite 5: EEG-Zuschlagspreis-Analyse (wenn enabled)
- **Diagramm:**
  - x: Zuschlagspreis (ct/kWh)
  - y: Equity IRR (%)
  - Linie: Mittlerer IRR
  - Schattiertes Band: ±1 Std. Abweichung
- **Text (LLM):** Wesentliche Erkenntnisse

### Seite 6: PPA-Collar-Analyse (wenn enabled)
- **Diagramm:**
  - x: PPA Floor (EUR/MWh)
  - y: Equity IRR (%)
  - Kurvenschar: Eine Kurve pro Cap-Aufschlag (+2, +5, +X EUR/MWh)
- **Text (LLM):** Wesentliche Erkenntnisse, empfohlene Collar-Struktur

### Seite 7: PPA-Baseload-Analyse (wenn enabled)
- **Diagramm:**
  - x: PPA-Preis (EUR/MWh)
  - y: Equity IRR (%)
  - Kurvenschar: Eine Kurve pro Baseload-Level (1, 2, X MW)
- **Text (LLM):** Wesentliche Erkenntnisse

### Seite 8: Fazit
- **Text (LLM):** Finale Betrachtung
  - Empfehlung für das beste Vermarktungsmodell
  - Vorschlag für weitere Analysen / andere Dimensionierungen
  - Basiert auf allen vorangegangenen Ergebnissen

## Architektur

### Neues Modul: `output/report/`

```
output/report/
├── __init__.py
├── charts.py          # Diagramm-Erstellung (matplotlib)
├── pdf_builder.py     # PDF-Assembly aus Template + Charts + Text
├── llm_client.py      # LLM-API-Client für Textgenerierung
└── templates/
    └── report.html    # HTML/CSS-Template (wird zu PDF konvertiert)
```

### Chart-Erstellung (`charts.py`)

```python
def create_pv_yield_chart(
    weather_timeseries: dict[str, np.ndarray],  # {weather_year: 35040 values}
    output_path: Path,
    colors: list[str],
) -> Path:
    """PV-Ertragsprofil über das Jahr, alle Wetterjahre."""

def create_price_scenario_chart(
    scenario_prices: dict[str, list[np.ndarray]],
    scenario_labels: dict[str, str],
    output_path: Path,
    colors: list[str],
) -> Path:
    """Strompreisszenarien: Jahresmittel über Projektlaufzeit."""

def create_grid_search_chart(
    grid_result: GridSearchResult,
    output_path: Path,
    colors: list[str],
) -> Path:
    """Kurvenschar: BESS-Scale vs. IRR, eine Kurve pro E/P-Ratio."""

def create_eeg_sensitivity_chart(
    eeg_result: SensitivityResult,
    output_path: Path,
    colors: list[str],
) -> Path:
    """EEG-Zuschlagspreis vs. IRR mit Std.-Band."""

def create_ppa_collar_chart(
    collar_result: SensitivityResult,
    output_path: Path,
    colors: list[str],
) -> Path:
    """PPA Floor vs. IRR, Kurvenschar für Cap-Spreads."""

def create_ppa_baseload_chart(
    baseload_result: SensitivityResult,
    output_path: Path,
    colors: list[str],
) -> Path:
    """PPA-Preis vs. IRR, Kurvenschar für Baseload-Level."""
```

Alle Charts werden als PNG gespeichert (`.data/output/{scenario_name}/charts/`).

### LLM-Client (`llm_client.py`)

```python
class LLMClient:
    """Kosteneffizienter LLM-API-Client für Report-Texte."""

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        """Verwende kleinstes/günstigstes Modell für Textgenerierung."""

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 500,
    ) -> str:
        """Generiere Text für eine Report-Seite."""
```

**Kosteneffizienz:**
- Verwende das günstigste verfügbare Modell (z.B. Claude Haiku)
- Kurze Prompts (max. 500 Tokens Output pro Seite)
- Relevante Daten als strukturierte Zusammenfassung im Prompt (nicht rohe Arrays)
- Cache: Generierte Texte in einer JSON-Datei speichern, um bei erneutem Lauf
  nicht erneut die API aufzurufen

### PDF-Builder (`pdf_builder.py`)

```python
def build_report(
    scenario_name: str,
    output_dir: Path,
    chart_paths: dict[str, Path],
    texts: dict[str, str],
    config: ReportConfig,
) -> Path:
    """Erstelle PDF-Report aus Charts + Texten."""
```

**PDF-Generierung:** `weasyprint` oder `reportlab` als Bibliothek.

**Empfehlung:** `weasyprint` (HTML → PDF), da:
- Einfacheres Layout via HTML/CSS
- Responsive Positionierung
- Corporate-CSS leicht anpassbar

### JSON-Erweiterung

```json
"scenario": {
    "output": {
        "directory": ".data/output/",
        "export_dispatch_sample": true,
        "report": {
            "enabled": true,
            "company_name": "Mein Unternehmen GmbH",
            "logo_path": "data/logo.png",
            "llm_api_key_env": "ANTHROPIC_API_KEY",
            "llm_model": "claude-haiku-4-5-20251001"
        }
    }
}
```

### CLI-Erweiterung

```bash
# Report mit LLM-Texten
python -m pv_bess_model.main --scenario my.json

# Report ohne LLM (nur CSV + PNG)
python -m pv_bess_model.main --scenario my.json --no-llm

# Kein Report (nur CSV, wie bisher)
python -m pv_bess_model.main --scenario my.json --no-report
```

## Abhängige Packages (neu)

```
matplotlib>=3.7
weasyprint>=59.0     # oder reportlab>=4.0
anthropic>=0.39      # für LLM-API (oder requests für generischen API-Call)
```

## Integration in main.py

Nach den CSV-Writes (Schritt 7):
```python
# Step 7b: Generate Report
report_cfg = scenario.raw.get("scenario", {}).get("output", {}).get("report", {})
if report_cfg.get("enabled", False) and not args.no_report:
    # 1. Charts erstellen (immer, auch ohne LLM)
    chart_paths = create_all_charts(output_dir / "charts", ...)

    # 2. LLM-Texte generieren (optional)
    if not args.no_llm:
        texts = generate_all_texts(llm_client, ...)
    else:
        texts = {}  # Leere Texte → Platzhalter im Report

    # 3. PDF bauen
    build_report(output_dir / f"{scenario.name}_report.pdf", chart_paths, texts, ...)
```

## Betroffene Dateien

| Datei                                 | Änderung |
|---------------------------------------|----------|
| `.data/output/report/__init__.py`     | **NEU** |
| `.data/output/report/charts.py`             | **NEU**: 6 Chart-Funktionen |
| `.data/output/report/pdf_builder.py`        | **NEU**: PDF-Assembly |
| `.data/output/report/llm_client.py`         | **NEU**: LLM-API-Client |
| `.data/output/report/templates/report.html` | **NEU**: HTML/CSS-Template |
| `config/schema.py`                    | Report-Block im Schema |
| `main.py`                             | Report-Erstellung nach CSV-Output, CLI-Flags |
| `pyproject.toml`                      | Neue Dependencies (matplotlib, weasyprint, anthropic) |

## Tests

- Chart-Erstellung: Alle 6 Chart-Typen erzeugen valides PNG
- PDF-Erstellung: weasyprint erzeugt valides PDF
- LLM-Client: Mock-Test (kein echter API-Call in Tests)
- `--no-llm`: Report ohne Texte (nur Charts)
- `--no-report`: Kein Report erzeugt
- Corporate-Farben korrekt angewendet
- Logo-Einbindung (PNG existiert / fehlt → Fallback)
