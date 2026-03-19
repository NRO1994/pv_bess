# Feature 13: Meta-Optimierungsframework – Regime-basierte Entscheidungsunterstützung

## Status: Implementierungsplan v1

---

## 1. Zielsetzung

### 1.1 Kernfrage

> Welche Kombination aus Vermarktungsstruktur, Betriebsmodus und BESS-Dimensionierung erreicht die
> Ziel-IRR zu minimalen Kosten?

### 1.2 Paradigmenwechsel

Das bestehende Modell maximiert die Equity-IRR innerhalb eines einzelnen, manuell gewählten Szenarios.
Das Meta-Optimierungsframework invertiert die Fragestellung:

| Bestehendes Modell                          | Meta-Optimierung                                          |
|---------------------------------------------|-----------------------------------------------------------|
| 1 Szenario-JSON → max(IRR)                  | Ziel-IRR → min(Kosten) über alle Regime                   |
| Vermarktung fix vorgegeben                  | Vermarktung als Entscheidungsvariable                     |
| Betriebsmodus fix                           | Green/Grey als Vergleichsdimension                        |
| Ergebnis: "beste BESS-Größe für Szenario X" | Ergebnis: "günstigstes Instrument für Ziel-IRR"           |

### 1.3 Nicht-Ziele (explizit ausgeschlossen)

- Kein ML-Surrogat im MVP (siehe Abschnitt 10)
- Keine Monte-Carlo-Integration im Regime-Vergleich (nur deterministische P50-Ergebnisse)
- Keine Änderung am bestehenden Dispatch-Optimizer oder Finanzmodell
- Keine neue CLI – Integration als zusätzlicher Modus im bestehenden `main.py`

---

## 2. Konzeptionelle Grundlagen

### 2.1 Entscheidungshierarchie

Die Entscheidungsvariablen des Modells lassen sich in zwei Kategorien trennen:

**Diskrete Regime (äußere Schleife – vorab enumerieren):**

| Regime               | Freie Parameter                                 |
|----------------------|-------------------------------------------------|
| Direktvermarktung    | –                                               |
| EEG-Floor            | Floor-Preis (€/kWh)                             |
| PPA-Floor            | Floor-Preis (€/kWh), GoO (€/kWh), Laufzeit (a) |
| PPA-Collar           | Floor (€/kWh), Cap (€/kWh), GoO, Laufzeit      |
| PPA Pay-as-Produced  | Fixpreis (€/kWh), GoO, Laufzeit                 |
| PPA Baseload         | Fixpreis (€/kWh), Baseload (MW), GoO, Laufzeit  |

Jeweils kombiniert mit Betriebsmodus: **green** oder **grey**.

**Stetige Design-Variablen (innere Optimierung – Grid Search):**

- BESS-Leistung (kW), abgeleitet aus: `scale_pct_of_pv`
- BESS-Kapazität (kWh), abgeleitet aus: `scale_pct_of_pv × e_to_p_ratio`

> **Merksatz:** *Regime wählen – Dimensionen optimieren.*

### 2.2 Capture Rate als vereinheitlichende Metrik

Die Capture Rate ist der mengengewichtete durchschnittliche Erlöspreis relativ zum Spot-Referenzpreis:

```
c = (Σ revenue_t) / (Σ production_t × spot_reference)
```

Sie wird bereits in `metrics.py:compute_all_metrics()` berechnet und im `FinancialMetrics`-Objekt
als `capture_rate` gespeichert. Die Capture Rate komprimiert alle Einflüsse (PV-Profil, Floor/Cap,
BESS-Arbitrage, Curtailment, GoO) in eine einzige Kennzahl.

**Zerlegung (konzeptionell):**

```
c_total = c_PV,raw + Δc_Marketing + Δc_Flexibilität
```

- `c_PV,raw`: Capture Rate der PV-only Direktvermarktung (Baseline)
- `Δc_Marketing`: Uplift durch Vermarktungsstruktur (EEG/PPA-Effekt)
- `Δc_Flexibilität`: Uplift durch BESS (Arbitrage, Curtailment-Vermeidung)

Diese Zerlegung ist **ableitbar aus den Grid-Search-Ergebnissen** – kein neuer Berechnungsschritt:
- `c_PV,raw` = Capture Rate des PV-only-Punktes (scale=0%) im Direktvermarktungs-Regime
- `Δc_Marketing` = Capture Rate PV-only im aktuellen Regime − `c_PV,raw`
- `Δc_Flexibilität` = Capture Rate mit BESS − Capture Rate PV-only (selbes Regime)

### 2.3 Bestimmung der Ziel-Capture-Rate c*

Die "minimale Capture Rate für Ziel-IRR" (`c*`) ist **kein Input**, sondern wird aus den
Grid-Search-Ergebnissen **interpoliert**:

Für jeden Grid-Punkt existiert ein Paar `(capture_rate, equity_irr)`. Der Punkt `c*` ist
der Capture-Rate-Wert, bei dem `equity_irr = target_irr`. Durch lineare Interpolation zwischen
den Grid-Punkten lässt sich `c*` für jedes Regime bestimmen.

Für den **PV-only-Fall** (kein BESS, also nur 1 Grid-Punkt pro Regime) ergibt sich `c*` nicht
aus einer Kurve, sondern ist einfach die beobachtete Capture Rate bei PV-only. Ob die Ziel-IRR
erreicht wird, ist dann eine Ja/Nein-Aussage.

**Offene Frage:** Im Originaldokument war die Idee, `c*` vorab analytisch zu berechnen, indem
man den Jahresertrag mit der Capture Rate skaliert und per Bisection die Ziel-IRR sucht. Das
funktioniert für PV-only (Revenue = Production × c × spot_ref), ist aber für PV+BESS ungenau,
weil der BESS gleichzeitig Kosten UND Produktion verändert. Der hier gewählte Ansatz umgeht
das Problem, indem `c*` post-hoc aus den vollständigen Simulationsergebnissen abgelesen wird.

---

## 3. Architektur

### 3.1 Einordnung in die bestehende Codebase

```
pv_bess_model/
├── optimization/
│   ├── grid_search.py          # Besteht, unverändert
│   ├── monte_carlo.py          # Besteht, unverändert
│   ├── analyses.py             # Besteht, unverändert
│   └── regime_search.py        # NEU: Meta-Optimierungsframework
├── output/
│   ├── csv_writer.py           # Erweitert: neue CSV-Ausgaben
│   └── report/
│       ├── data_collector.py   # Erweitert: Regime-Vergleichsdaten
│       ├── charts.js           # Erweitert: neuer Chart-Typ
│       └── templates/
│           └── dashboard.html  # Erweitert: neuer Tab
└── main.py                     # Erweitert: neuer CLI-Modus
```

### 3.2 Datenfluss

```
                          RegimeSearchConfig
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              Regime A      Regime B      Regime C       (N Regime)
           (EEG+green)  (Collar+green) (Floor+grey)
                    │           │           │
                    ▼           ▼           ▼
            run_grid_search  run_grid_search  run_grid_search
            (bestehend!)     (bestehend!)     (bestehend!)
                    │           │           │
                    ▼           ▼           ▼
           GridSearchResult GridSearchResult GridSearchResult
                    │           │           │
                    └───────────┼───────────┘
                                ▼
                    RegimeSearchResult
                    ├── Regime-Vergleichstabelle
                    ├── Capture-Rate-Zerlegung
                    ├── c*-Interpolation pro Regime
                    └── Visualisierungsdaten
```

**Zentrale Design-Entscheidung:** Das Framework ruft `run_grid_search()` **unverändert** auf.
Jedes Regime wird in eine `GridSearchConfig` übersetzt, die sich nur in den Preis-Schedules,
dem Betriebsmodus und ggf. dem Baseload-Level unterscheidet. Das gesamte Dispatch- und
Finanzmodell bleibt unberührt.

### 3.3 Beziehung zu bestehenden Analyse-Modulen

Das bestehende `analyses.py` führt Marketing-Parameter-Sweeps mit Monte Carlo durch
(EEG-Sensitivity, PPA-Collar-2D, PPA-Baseload-2D). Diese bleiben erhalten und werden
**nicht** durch das Regime-Framework ersetzt.

| Aspekt        | analyses.py (besteht)          | regime_search.py (neu)             |
|---------------|--------------------------------|------------------------------------|
| Zweck         | Sensitivität innerhalb 1 Regime | Vergleich über Regime hinweg       |
| Methode       | MC pro Parameterpunkt          | Deterministisch (P50 Grid Search)  |
| BESS-Größe    | Fix (Optimum aus Grid Search)  | Variabel (voller Grid Search)      |
| Ergebnis      | IRR-Verteilung pro Preispunkt  | Optimale BESS-Größe pro Regime     |
| Wann nutzen?  | Nach Regime-Wahl, für Feintuning | Vor Regime-Wahl, für Strategieentscheidung |

---

## 4. Datenmodell

### 4.1 Eingabe: `RegimeSearchConfig`

```python
@dataclass
class MarketingRegime:
    """Definition eines Vermarktungsregimes für den Regime-Vergleich."""
    name: str                              # z.B. "EEG Floor 7ct green"
    label: str                             # z.B. "EEG-Mindestpreis 0,07 €/kWh (Grünstrom)"
    marketing_type: str                    # "market" | "eeg" | "ppa_floor" | "ppa_collar" | "ppa_pap" | "ppa_baseload"
    operating_mode: str                    # "green" | "grey"

    # EEG-spezifisch
    eeg_floor_price_eur_per_kwh: float = 0.0
    eeg_duration_years: int = 20
    eeg_inflation: bool = False

    # PPA-spezifisch
    ppa_floor_price_eur_per_kwh: float = 0.0
    ppa_cap_price_eur_per_kwh: float = 0.0
    ppa_fixed_price_eur_per_kwh: float = 0.0
    ppa_duration_years: int = 10
    ppa_inflation: bool = False
    goo_premium_eur_per_kwh: float = 0.0
    baseload_mw: float = 0.0


@dataclass
class RegimeSearchConfig:
    """Konfiguration für den Regime-übergreifenden Vergleich."""
    regimes: list[MarketingRegime]
    target_irr: float                      # Ziel-IRR als Dezimalzahl (z.B. 0.08)

    # BESS Design Space (identisch für alle Regime)
    scale_pct_of_pv: list[float]
    e_to_p_ratio_hours: list[float]

    # Referenz auf das zentrale Preisszenario
    # (wird aus dem Scenario-JSON übernommen)
```

### 4.2 Ausgabe: `RegimeSearchResult`

```python
@dataclass
class RegimeResult:
    """Ergebnis eines einzelnen Regimes."""
    regime: MarketingRegime
    grid_search_result: GridSearchResult    # Vollständiges Grid-Search-Ergebnis

    # Abgeleitete Metriken
    capture_rate_pv_only: float | None      # c bei scale=0% (PV-only in diesem Regime)
    capture_rate_optimal: float | None      # c beim IRR-optimalen BESS-Punkt
    delta_c_marketing: float | None         # Δc gegenüber Direktvermarktung PV-only
    delta_c_flexibility: float | None       # Δc durch BESS (optimal - PV-only, selbes Regime)
    irr_pv_only: float | None              # IRR bei PV-only
    irr_optimal: float | None              # Beste IRR in diesem Regime
    min_bess_capex_for_target: float | None # Minimaler BESS-CAPEX für Ziel-IRR (oder None)
    target_irr_achievable: bool            # Wird Ziel-IRR in irgendeinem Grid-Punkt erreicht?


@dataclass
class RegimeSearchResult:
    """Gesamtergebnis des Regime-Vergleichs."""
    target_irr: float
    regime_results: list[RegimeResult]
    baseline_capture_rate: float            # c_PV,raw (Direktvermarktung, PV-only, green)

    # Sortierte Empfehlung: Regime, die Ziel-IRR erreichen, sortiert nach min BESS CAPEX
    feasible_regimes: list[RegimeResult]
```

---

## 5. Implementierungsplan

### 5.1 Schritt 1: `RegimeSearchConfig` und JSON-Schema (Szenario-Erweiterung)

**Datei:** `config/schema.py`, Szenario-JSON

Erweiterung des Szenario-JSON um einen optionalen `regime_search`-Block:

```json
{
  "scenario": {
    "regime_search": {
      "enabled": true,
      "target_irr": 8.0,
      "regimes": [
        {
          "name": "Direktvermarktung green",
          "marketing_type": "market",
          "operating_mode": "green"
        },
        {
          "name": "EEG 7ct green",
          "marketing_type": "eeg",
          "operating_mode": "green",
          "eeg_floor_price_eur_per_kwh": 0.07,
          "eeg_duration_years": 20
        },
        {
          "name": "PPA Collar 5.5/8.5ct green",
          "marketing_type": "ppa_collar",
          "operating_mode": "green",
          "ppa_floor_price_eur_per_kwh": 0.055,
          "ppa_cap_price_eur_per_kwh": 0.085,
          "ppa_duration_years": 10,
          "goo_premium_eur_per_kwh": 0.003
        },
        {
          "name": "PPA Floor 6ct grey",
          "marketing_type": "ppa_floor",
          "operating_mode": "grey",
          "ppa_floor_price_eur_per_kwh": 0.06,
          "ppa_duration_years": 10,
          "goo_premium_eur_per_kwh": 0.003
        }
      ]
    }
  }
}
```

Der `regime_search`-Block nutzt die **selbe BESS Design Space** (`scale_pct_of_pv`,
`e_to_p_ratio_hours`) wie der reguläre Grid Search. Technologie-, Finanz- und Preisparameter
kommen ebenfalls aus den bestehenden Szenario-Feldern.

**Aufwand:** Klein. Schema-Erweiterung + Validierung.

### 5.2 Schritt 2: Regime → GridSearchConfig Übersetzung

**Datei:** `optimization/regime_search.py`

Kernfunktion: Übersetzung eines `MarketingRegime` in die Preis-Schedules, die
`GridSearchConfig` erwartet:

```python
def _regime_to_price_schedules(
    regime: MarketingRegime,
    lifetime_years: int,
    inflation_rate: float,
) -> tuple[list[float], list[float], list[float]]:
    """Erzeuge (fixed_prices_yearly, goo_prices_yearly, cap_prices_yearly) für ein Regime.

    Nutzt die bestehende Logik aus market/eeg.py und market/ppa.py.
    """
```

Diese Funktion repliziert die Logik, die aktuell in `main.py:_build_fixed_prices_yearly()`,
`_build_goo_prices_yearly()`, `_build_cap_prices_yearly()` steckt. Nach Umsetzung von
Cleanup B3 (Vermarktungslogik in Marktmodule verschieben) kann hier direkt auf die neuen
`eeg.get_floor_prices_yearly()` und `ppa.get_fixed_prices_yearly()` delegiert werden.

**Abhängigkeit:** Idealerweise nach B3, aber auch ohne B3 umsetzbar (dann Logik aus main.py
kopieren, was temporär Duplikation erzeugt).

**Aufwand:** Mittel. Die Preis-Schedule-Erzeugung ist die komplexeste Übersetzung.

### 5.3 Schritt 3: `run_regime_search()` – Orchestrierung

**Datei:** `optimization/regime_search.py`

```python
def run_regime_search(
    base_config: GridSearchConfig,
    regime_config: RegimeSearchConfig,
) -> RegimeSearchResult:
    """Führe Grid Search für jedes Regime durch und vergleiche Ergebnisse.

    Parameters
    ----------
    base_config:
        Basis-GridSearchConfig mit allen technologie- und finanzspezifischen Parametern.
        Wird pro Regime modifiziert (Preis-Schedules, operating_mode, baseload_mw).
    regime_config:
        Liste der zu vergleichenden Regime + Ziel-IRR.
    """
```

**Algorithmus:**

```
1. Direktvermarktungs-Baseline berechnen:
   - Regime: marketing_type="market", operating_mode="green", scale=0%
   - → baseline_capture_rate = c_PV,raw

2. Für jedes Regime r in regime_config.regimes:
   a. Preis-Schedules erzeugen (Schritt 2)
   b. GridSearchConfig modifizieren:
      - fixed_prices_yearly, goo_prices_yearly, cap_prices_yearly überschreiben
      - operating_mode = r.operating_mode
      - baseload_mw = r.baseload_mw
   c. run_grid_search(modified_config)  ← BESTEHENDE FUNKTION
   d. Aus Ergebnis extrahieren:
      - PV-only-Punkt (scale=0%): irr_pv_only, capture_rate_pv_only
      - Optimaler Punkt (max IRR): irr_optimal, capture_rate_optimal
      - Δc_marketing = capture_rate_pv_only - baseline_capture_rate
      - Δc_flexibility = capture_rate_optimal - capture_rate_pv_only
      - target_irr_achievable = any(p.irr >= target_irr for p in points)
      - min_bess_capex_for_target = min(p.capex_bess for p in points if p.irr >= target_irr)

3. Feasible Regime sortieren: nach min_bess_capex_for_target aufsteigend

4. RegimeSearchResult zusammenbauen
```

**Parallelisierung:** Die Grid Searches pro Regime sind voneinander unabhängig. Zwei Ebenen
möglich:
- **Sequentiell über Regime, parallel innerhalb Grid Search** (einfach, bestehende
  Parallelisierung nutzen)
- **Parallel über Regime UND innerhalb Grid Search** (erfordert Resource-Management,
  Overkill für MVP)

Empfehlung für MVP: Sequentiell über Regime. Bei N=6 Regimen und ~7 Min pro Grid Search
sind das ~42 Minuten. Akzeptabel für eine strategische Analyse, die selten läuft.

**Aufwand:** Mittel. Die Orchestrierung ist straightforward, da `run_grid_search()`
als Black Box genutzt wird.

### 5.4 Schritt 4: CSV-Ausgabe

**Datei:** `output/csv_writer.py`

Zwei neue CSV-Dateien:

**`{scenario}_regime_comparison.csv`** – Eine Zeile pro Regime:

| Spalte                        | Beschreibung                                     |
|-------------------------------|--------------------------------------------------|
| Regime                        | Name des Regimes                                 |
| Marketing-Typ                 | market/eeg/ppa_floor/...                         |
| Betriebsmodus                 | green/grey                                       |
| IRR PV-only (%)               | Equity-IRR ohne BESS                             |
| IRR Optimal (%)               | Beste Equity-IRR mit BESS                        |
| Capture Rate PV-only          | c bei scale=0%                                   |
| Capture Rate Optimal          | c beim besten BESS-Punkt                         |
| Δc Marketing                  | Uplift durch Vermarktungsstruktur                |
| Δc Flexibilität               | Uplift durch BESS                                |
| Ziel-IRR erreichbar           | Ja/Nein                                          |
| Min BESS-CAPEX für Ziel (€)   | Minimaler BESS-CAPEX, der Ziel-IRR erreicht      |
| Optimale BESS-Leistung (kW)   | Bei min BESS-CAPEX für Ziel                      |
| Optimale BESS-Kapazität (kWh) | Bei min BESS-CAPEX für Ziel                      |

**`{scenario}_regime_grid_detail.csv`** – Alle Grid-Punkte aller Regime:

| Spalte         | Beschreibung                    |
|----------------|---------------------------------|
| Regime         | Name                            |
| Scale (%)      | BESS-Anteil                     |
| E/P (h)        | Ratio                           |
| BESS kW        | Leistung                        |
| BESS kWh       | Kapazität                       |
| CAPEX BESS (€) | Nur BESS-Anteil                 |
| IRR (%)        | Equity-IRR                      |
| Capture Rate   | c                               |
| ≥ Ziel-IRR     | Boolean                         |

**Aufwand:** Klein.

### 5.5 Schritt 5: Visualisierung (HTML-Dashboard)

**Dateien:** `output/report/charts.js`, `output/report/data_collector.py`, `dashboard.html`

Neuer Tab im HTML-Dashboard: **"Regime-Vergleich"**

**Chart 1: Capture Rate vs. BESS-CAPEX (Hauptchart)**

- X-Achse: Capture Rate
- Y-Achse: Zusätzlicher BESS-CAPEX (€) relativ zu PV-only
- Eine Linie pro Regime (unterschiedliche Farben)
- Jeder Punkt auf der Linie = ein Grid-Punkt (scale/E/P-Kombination)
- Vertikale gestrichelte Linie bei `c*`: "Ab hier wird Ziel-IRR erreicht"
- Punkte oberhalb der Linie und rechts von `c*` sind zulässig

Hinweis: `c*` variiert zwischen Regimen (da CAPEX/OPEX-Struktur unterschiedlich). Daher
wird `c*` als **Punkt auf jeder Linie** markiert, nicht als globale Vertikale. Alternativ
kann die PV-only-Schwelle (eine globale Capture Rate, ab der PV-only die Ziel-IRR erreicht)
als Referenzlinie dienen.

**Chart 2: IRR pro Regime (Balkendiagramm)**

- X-Achse: Regime (kategorisch)
- Y-Achse: Equity-IRR
- Zwei Balken pro Regime: PV-only und Optimal (mit BESS)
- Horizontale Linie bei Ziel-IRR

**Chart 3: Capture-Rate-Zerlegung (Stacked Bar)**

- X-Achse: Regime
- Y-Achse: Capture Rate
- Gestapelt: c_PV,raw | Δc_Marketing | Δc_Flexibilität

**Aufwand:** Mittel-Groß. Drei neue Charts + Tab-Logik im HTML-Template.

### 5.6 Schritt 6: CLI-Integration

**Datei:** `main.py`

Neuer CLI-Flag: `--regime-search` (aktiviert Regime-Vergleich statt normalem Grid Search)

```bash
# Normaler Modus (bestehend, unverändert):
python -m pv_bess_model.main --scenario scenario.json

# Regime-Vergleich:
python -m pv_bess_model.main --scenario scenario.json --regime-search
```

Wenn `--regime-search` aktiv:
1. Normaler Grid Search wird **übersprungen** (der Regime-Search ersetzt ihn)
2. Regime-Search liefert Ergebnisse für alle definierten Regime
3. Das "beste" Regime (Ziel-IRR erreichbar, minimaler BESS-CAPEX) wird als "Optimum" ausgewählt
4. Optional: MC auf dem Optimum des besten Regimes (wie bisher)
5. Analyse-Sweeps werden **nicht** ausgeführt (Regime-Search ersetzt sie konzeptionell)

Alternativ: Aktivierung über `regime_search.enabled: true` im JSON (ohne CLI-Flag).

**Aufwand:** Klein-Mittel.

---

## 6. Szenario-Beispiel: Erwarteter Output

### Eingabe

- PV: 50 MWp, NAP: 45 MW
- BESS Design Space: scale [0, 25, 50, 75, 100]%, E/P [2, 4]h
- Ziel-IRR: 8%
- 4 Regime definiert

### Erwarteter Output (`regime_comparison.csv`):

```
Regime;Marketing;Modus;IRR PV-only;IRR Optimal;c PV-only;c Optimal;Δc Mktg;Δc Flex;Ziel OK;Min BESS CAPEX
Direkt green;market;green;5,2%;7,1%;0,82;0,91;0,00;0,09;Nein;-
EEG 7ct green;eeg;green;7,8%;9,4%;0,95;1,03;0,13;0,08;Ja;2.400.000
Collar 5.5/8.5 green;ppa_collar;green;6,9%;8,6%;0,89;0,97;0,07;0,08;Ja;4.800.000
Floor 6ct grey;ppa_floor;grey;7,2%;9,1%;0,92;1,01;0,10;0,09;Ja;3.600.000
```

**Erkenntnis:** EEG 7ct mit BESS 25%/2h (CAPEX 2,4M€) ist die günstigste Lösung für 8% IRR.

---

## 7. Abgrenzung zum bestehenden Analyserahmen

### Was wird NICHT geändert?

- `grid_search.py`: Keinerlei Modifikation. Wird als Black Box aufgerufen.
- `monte_carlo.py`: Unverändert. Kann optional auf das Regime-Optimum angewendet werden.
- `analyses.py`: Unverändert. Kann weiterhin für Feintuning innerhalb eines Regimes genutzt werden.
- `optimizer.py`, `engine.py`: Unverändert. Der Dispatch bleibt identisch.
- Bestehendes Szenario-JSON: Abwärtskompatibel. `regime_search` ist ein optionaler Block.

### Was wird hinzugefügt?

| Neu                              | Datei                        |
|----------------------------------|------------------------------|
| `RegimeSearchConfig`             | `optimization/regime_search.py` |
| `MarketingRegime`                | `optimization/regime_search.py` |
| `RegimeResult`, `RegimeSearchResult` | `optimization/regime_search.py` |
| `run_regime_search()`            | `optimization/regime_search.py` |
| `_regime_to_price_schedules()`   | `optimization/regime_search.py` |
| `write_regime_comparison_csv()`  | `output/csv_writer.py`       |
| `write_regime_detail_csv()`      | `output/csv_writer.py`       |
| Regime-Tab + 3 Charts            | `output/report/`             |
| JSON-Schema-Erweiterung          | `config/schema.py`           |
| CLI-Flag `--regime-search`       | `main.py`                    |

---

## 8. Implementierungsreihenfolge

| #   | Schritt                                         | Abhängigkeit | Aufwand |
|-----|--------------------------------------------------|--------------|---------|
| 1   | `MarketingRegime` + `RegimeSearchConfig` Datenmodell | –          | Klein   |
| 2   | `_regime_to_price_schedules()` (Preis-Übersetzung) | 1, ideal nach B3 | Mittel |
| 3   | `run_regime_search()` Orchestrierung             | 1, 2         | Mittel  |
| 4   | JSON-Schema-Erweiterung + Validierung            | 1            | Klein   |
| 5   | CLI-Integration in `main.py`                     | 3, 4         | Klein   |
| 6   | CSV-Ausgabe                                      | 3            | Klein   |
| 7   | HTML-Dashboard: Regime-Tab + Charts              | 6            | Groß    |
| 8   | Tests                                            | 3            | Mittel  |

**Geschätzter Gesamtaufwand:** Mittel-Groß (ohne HTML-Charts: Mittel)

---

## 9. Tests

### Unit Tests (`tests/test_regime_search.py`)

- `_regime_to_price_schedules()`: Korrekter Output für jedes `marketing_type`
  (market, eeg, ppa_floor, ppa_collar, ppa_pap, ppa_baseload)
- `RegimeResult`: Korrekte Berechnung von `delta_c_marketing`, `delta_c_flexibility`
- `target_irr_achievable`: Korrekte Bestimmung (True wenn mindestens 1 Punkt ≥ target)
- `min_bess_capex_for_target`: Korrekter Wert, None wenn nicht erreichbar
- `feasible_regimes`: Korrekte Sortierung

### Integration Tests

- Kleines synthetisches Szenario mit 2 Regimen × 2 BESS-Größen
- Prüfen: Regime mit höherem Floor-Preis hat höhere Capture Rate
- Prüfen: Baseline (c_PV,raw) ist identisch für alle Regime bei scale=0% und Direktvermarktung
- Prüfen: Abwärtskompatibilität – Szenario ohne `regime_search` Block funktioniert wie bisher

---

## 10. ML-Surrogat (Zukunft, nicht im MVP)

Das Originaldokument erwähnt einen ML-basierten Surrogate-Ansatz. Bewertung:

**Sinnvoll als Beschleunigung**, wenn:
- Die Regime-Anzahl sehr groß wird (>20)
- Oder der BESS Design Space feingranular wird (>50 Punkte)
- Oder Echtzeit-Interaktivität gewünscht ist (Web-Frontend)

**Nicht sinnvoll für MVP**, weil:
- 6 Regime × 10 Grid-Punkte = 60 volle Simulationen → ~1h Rechenzeit, akzeptabel
- Das bestehende Grid-Search-Framework ist exakt, ein Surrogat nur approximativ
- Der Implementierungsaufwand für ML übersteigt den Nutzen bei dieser Problemgröße

**Möglicher späterer Ansatz:**
- Trainingsdaten: Alle Grid-Search-Ergebnisse aus vergangenen Läufen
- Modell: Gradient-Boosted Trees (z.B. LightGBM) – schnell, braucht wenig Daten
- Features: (marketing_type, mode, scale_pct, e_to_p, floor_price, cap_price, ...)
- Target: equity_irr
- Einsatz: Vorselektion vielversprechender Kandidaten → nur diese voll simulieren

---

## 11. Offene Fragen

### 11.1 Preisparameter-Sweeps innerhalb eines Regimes

Soll das Framework auch innerhalb eines Regimes über Preisparameter sweepen?

**Beispiel:** Für PPA-Collar nicht nur einen (Floor=5.5ct, Cap=8.5ct) Punkt evaluieren,
sondern automatisch Floor=[4,5,6,7]ct × Cap=[7,8,9,10]ct testen?

Das würde die Regime-Anzahl stark erhöhen (4 Marketing-Typen × 2 Modi × 4-16 Preiskombinationen
= 32-128 Regime), aber deutlich mehr Einsicht liefern. Alternativ kann das bestehende
`analyses.py` nach der Regime-Wahl für Feintuning genutzt werden.

**Meine Empfehlung:** Im MVP unterstützen, aber nicht erzwingen. Der Nutzer definiert
die Regime explizit im JSON – ob er 4 oder 40 anlegt, ist seine Entscheidung. Das Framework
skaliert linear.

### 11.2 Bisektionsverfahren für minimalen PPA-Preis

Soll das Framework automatisch den minimalen PPA-Preis finden, bei dem die Ziel-IRR
gerade noch erreicht wird?

**Ansatz:** Für ein gegebenes Regime + BESS-Größe den PPA-Preis per Bisection variieren,
bis `equity_irr(price) = target_irr`. Das erfordert wiederholte Grid-Search-Aufrufe
(jeweils mit 1 Grid-Punkt) und ist konzeptionell einfach, aber rechenintensiv.

**Meine Empfehlung:** Nicht im MVP. Die diskrete Regime-Liste gibt dem Nutzer genug
Kontrolle. Ein Bisektions-Feature kann als Schritt 2 ergänzt werden.

### 11.3 Soll ein Referenz-Regime immer "Direktvermarktung green" sein?

Für die Capture-Rate-Zerlegung wird ein Baseline-Regime benötigt (`c_PV,raw`). Aktuell
ist dies fest als "Direktvermarktung green, PV-only" definiert. Soll der Nutzer die
Baseline konfigurieren können, oder ist "Direktvermarktung green" immer korrekt?

### 11.4 Wie soll mit dem Fall umgegangen werden, dass kein Regime die Ziel-IRR erreicht?

Mögliche Reaktionen:
- Warnung ausgeben + trotzdem alle Ergebnisse berichten
- Das Regime mit der höchsten IRR als "nächstbestes" markieren
- Die Capture-Rate-Lücke quantifizieren ("X% mehr Capture Rate nötig")

### 11.5 Soll das Regime-Framework einen eigenen Entry Point bekommen?

Option A: Integration in `main.py` (wie oben beschrieben)
Option B: Separater Entry Point `main_regime.py` (analog zu `main_portfolio.py` aus Feature 12)

Vorteil von B: Sauberere Trennung, kein weiteres Aufblähen von `main.py`.
Nachteil von B: Geteilte Infrastruktur (Szenario-Laden, PV-Daten, Preise) müsste refactored werden.

### 11.6 Parallele Ausführung der Regime

Sollen die Grid Searches für verschiedene Regime parallel laufen? Bei sequentieller
Ausführung und 6 Regimen à 10 Grid-Punkte à 25 Jahre: ~60 × 9.125 ≈ 550K LP-Solves.
Bei ~3ms/Solve und 8 parallelen Grid-Punkten: ~35 Minuten. Akzeptabel?

Falls parallelisiert über Regime: ~6 Minuten (aber erfordert Prozess-Pool-Management
über zwei Ebenen).
