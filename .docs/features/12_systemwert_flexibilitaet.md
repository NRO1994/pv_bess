# Feature: Systemwert von Flexibilität im Stadtwerk-Portfolio

## Status: Konzeptphase / Entwurf v3

---

## 1. Zusammenfassung

Erweiterung der bestehenden pv_bess Codebase um ein Meta-Modell, das den ökonomischen Systemwert zusätzlicher
Flexibilität im Stadtwerk-Portfolio bewertet. Kernidee: Welt-A/Welt-B-Vergleich, wobei "Welt B" iterativ durch
Hinzufügen von Flexibilitäten erzeugt wird.

### Getroffene Entscheidungen (v3)

- **Zeitauflösung**: 15 Minuten (identische Logik wie bestehendes Modell)
- **Architektur**: Separates Modul `portfolio/` + separater Entry Point `main_portfolio.py`
- **MVP-Scope**: Rein marktbasierter Systemwert (kein CAPEX/OPEX/IRR). Isolierte Flex-Bewertung, keine
  Optimierung zwischen Flex-Typen. Alle Punkte werden berechnet und in Diagrammen dargestellt.
- **Suchmethode**: Vollständige Enumeration aller Punkte (kein Optimierungs-Grid-Search)
- **Flex-Typen**: Spezifische Technologien (BESS, WP, Wallbox/V2G). Mehrere Instanzen desselben Typs mit
  unterschiedlichen Parametern erlaubt.
- **Last**: Als Liste von Lastgruppen (MVP: nur Haushaltskunden SLP H0). Erweiterbar um Gewerbe etc.
- **Erzeugung**: Als Liste von Erzeugern (MVP: eine aggregierte PV-Anlage). Erweiterbar um Wind etc.
- **Flex-Wachstum**: Jährlicher Zubau von x kW innerhalb der 25-Jahres-Simulation
- **Lastwachstum**: Prozentualer jährlicher Faktor pro Lastgruppe
- **SLP-Aufbereitung**: Interne Methode aus BDEW-Excel, jahresabhängig generiert und gecacht
- **Preis-Inputs**: Identische Struktur wie bestehendes Modell (`price_inputs.scenarios[]`)
- **Monte Carlo**: Nicht im MVP
- **Output**: CSV + HTML-Dashboard (wie bestehendes Modell). HTML-Input-Formular basierend auf JSON-Schema.
- **WP-Modellierung**: Alle WP-Typen (Luft-Wasser, Wasser-Wasser) nutzen identische LP-Constraints;
  der Unterschied liegt nur in den Input-Parametern (COP-Kennlinie, thermischer Speicher).
- **Dezentrale WPs**: Werden im Modell als steuerbar angenommen (§14a EnWG)
- **Thermischer Speicher**: Als konfigurierbarer Parameter pro WP-Instanz (analog E/P-Ratio bei BESS)

---

## 2. Synergien mit bestehender Codebase

### Was direkt wiederverwendbar ist

| Bestehende Komponente                    | Nutzung im Meta-Modell                                      |
|------------------------------------------|-------------------------------------------------------------|
| `dispatch/optimizer.py` (Daily LP)       | Strukturvorlage für den neuen Portfolio-Optimizer           |
| `dispatch/engine.py` (Multi-Year Loop)   | Vorlage für 25-Jahres-Simulation                            |
| `market/price_loader.py`                 | Spotpreis-Zeitreihen laden und auf Projektlaufzeit strecken |
| `pv/pvgis_client.py`, `pv/timeseries.py` | PV-Erzeugungsprofile (P50/P90) + Temperaturdaten            |
| `finance/inflation.py`                   | Inflation auf Preise                                        |

### Was neu gebaut werden muss

- `portfolio/load_profiles.py` – BDEW-SLP aus Excel laden, jahresabhängig aufbereiten, cachen
- `portfolio/heat_demand.py` – Wärmelastprofil aus PVGIS-Temperaturdaten (Gradtagszahl)
- `dispatch/optimizer_portfolio.py` – Neuer LP mit Netto-Position, Load, Flex-Variablen (15min)
- `dispatch/engine_portfolio.py` – Multi-Year-Loop mit zeitvariierender Flex-Kapazität (jährl. Zubau)
- `portfolio/system_value.py` – Welt-A/Welt-B Vergleich, Delta-Berechnung
- `portfolio/marginal_value.py` – Grenznutzen-Kurven
- `main_portfolio.py` – Separater Entry Point

---

## 3. Vereinfachungen (MVP)

### 3.1 Erzeugung: Liste von Erzeugern

Erzeugung wird als Liste modelliert. Im MVP eine aggregierte PV-Anlage (~20 MWp). Später erweiterbar um
weitere PV-Anlagen (unterschiedliche Ausrichtung) oder Onshore-Wind.

Alle Erzeugerprofile werden zu einem aggregierten 15min-Profil summiert.

### 3.2 Last: Liste von Lastgruppen

Last wird als Liste modelliert. Im MVP nur Haushaltskunden (SLP H0). Später erweiterbar um Gewerbe (G0) etc.

- SLP-H0 × Anzahl Haushaltskunden × Jahresverbrauch pro Kunde
- Ergebnis: 35.040 Viertelstundenwerte aggregierte Last (kWh)
- Jährliches Lastwachstum: konfigurierbarer Faktor pro Lastgruppe

### 3.3 Welt A: Baseline ohne Flexibilität

Welt A = Aggregierte PV-Erzeugung vs. Aggregierte Last, rein marktbasiert:

- Viertelstunde t: `netto[t] = pv[t] - last[t]`
- `netto[t] > 0` → Überschuss, wird zu `spot[t]` verkauft
- `netto[t] < 0` → Unterdeckung, wird zu `spot[t]` eingekauft
- Systemkosten_A = Σ (Einkauf × spot) - Σ (Verkauf × spot)

### 3.4 Welt B: Baseline + Flexibilität X mit jährlichem Zubau

Wie Welt A, aber mit steigender Flex-Kapazität über die Projektlaufzeit:

- Jahr 1: Flex-Kapazität = x kW (Zubau-Rate)
- Jahr n: Flex-Kapazität = n × x kW (kumuliert)
- `Systemwert(X, x_kW_pa) = Σ_jahre [Systemkosten_A(j) - Systemkosten_B(X, j × x_kW_pa)]`

Alle definierten Zubau-Raten aller Flex-Instanzen werden vollständig durchgerechnet (kein Optimierer,
kein Early-Stopping). Die Ergebnisse werden in Diagrammen dargestellt, damit der Nutzer die Zusammenhänge
visuell erfassen kann. Eine automatische Optimierung zwischen Flex-Typen ist nicht Teil des MVP.

---

## 4. Integration der Last in den LP-Optimizer

### Konzept: Netto-Position

Die Last wird innerhalb des LP als fixe Nachfrage modelliert. Der Optimizer optimiert die Netto-Grid-Position
unter Berücksichtigung aller aktiven Flexibilitäten.

```
Netto-Grid-Position pro Viertelstunde t:
  grid_sell[t] - grid_buy[t] = export_pv[t] + discharge[t] × RTE - load[t]

  grid_sell[t] ≥ 0   (Netto-Einspeisung → Erlös)
  grid_buy[t]  ≥ 0   (Netto-Bezug → Kosten)

Zielfunktion:
  max Σ_t [ grid_sell[t] × spot[t] - grid_buy[t] × spot[t] ]
```

### LP-Formulierungen pro Flex-Typ

Jede Technologie hat ihre eigene LP-Formulierung. Mehrere Instanzen desselben Typs werden als separate
Variablen-Sets im LP abgebildet (z.B. `bess_1_charge[t]`, `bess_2_charge[t]`).

**BESS** (Vorlage aus bestehendem Optimizer):

- `charge[t]`, `discharge[t]`, `soc[t]`
- Constraints: SoC-Limits, Power-Limits, RTE
- Degradation: Kapazitätsverlust pro Jahr (Tranchenmodell bei gestaffeltem Zubau)

**Wärmepumpe (WP)**:

Luft-Wasser und Wasser-Wasser verwenden identische LP-Constraints. Der Unterschied liegt ausschließlich
in den Input-Parametern:

- COP-Nennwert und Referenztemperatur (bestimmt die temperaturabhängige COP-Kurve)
- Thermischer Speicher in kWh_th (bestimmt `max_shift_hours` implizit)

Die LP-Constraints sind unabhängig vom WP-Typ immer:

```
LP-Variablen:
  wp_load[t]          – elektrische Aufnahme in Viertelstunde t (kWh/15min)
  thermal_storage[t]  – thermischer Speicherstand (kWh_th)

Constraints:
  Σ wp_load[t] × COP(T_außen[t]) = Wärmebedarf_tag   (Tages-Energiebilanz)
  0 ≤ wp_load[t] ≤ P_wp_max / 4                       (Power-Limit, kW → kWh/15min)
  0 ≤ thermal_storage[t] ≤ thermal_storage_max         (Speicher-Limits)
  thermal_storage[t+1] = thermal_storage[t]
                         + wp_load[t] × COP(T_außen[t])
                         - heat_demand[t]               (Speicher-Bilanz)
```

Der thermische Speicher ist ein fixer Input-Parameter pro WP-Instanz (wie `thermal_storage_kwh` im JSON).
Im MVP wird er als fester Wert gesetzt, nicht als Suchdimension. Falls perspektivisch verschiedene
Speichergrößen verglichen werden sollen, können mehrere WP-Instanzen mit unterschiedlichen
`thermal_storage_kwh`-Werten definiert werden.

**Wallbox/E-Mobilität + V2G**:

```
LP-Variablen:
  ev_charge[t]        – Laden (Grid → EV), kWh/15min
  ev_discharge[t]     – Entladen / V2G (EV → Grid), kWh/15min  [nur wenn v2g_enabled]
  ev_soc[t]           – Batteriestand der EV-Flotte (kWh)

Constraints:
  ev_soc[t+1] = ev_soc[t] + ev_charge[t] - ev_discharge[t]
  ev_soc_min ≤ ev_soc[t] ≤ ev_soc_max
  ev_charge[t] ≤ P_charge_max / 4
  ev_discharge[t] ≤ P_discharge_max / 4 × RTE_v2g

  // Verfügbarkeitsfenster
  ev_charge[t] = 0       für t ∉ [arrival, departure]
  ev_discharge[t] = 0    für t ∉ [arrival, departure]

  // Mindest-SoC bei Abfahrt
  ev_soc[t_departure] ≥ E_min_departure

  // Ohne V2G: ev_discharge[t] = 0 ∀t
```

---

## 5. Zeitvariierende Flex-Kapazität (Jährlicher Zubau)

### Das Konzept

```
Bestehendes Modell (fix):     Jahr 1: 500 kW, Jahr 2: 500 kW, ..., Jahr 25: 500 kW
Portfolio-Modell (Zubau):     Jahr 1: 100 kW, Jahr 2: 200 kW, ..., Jahr 25: 2.500 kW
```

### Auswirkung auf die Engine

Die `engine_portfolio.py` muss für jedes Projektjahr:

1. Die aktuelle Flex-Kapazität berechnen: `capacity_year_n = zubau_rate × n`
2. Das LP mit dieser jahresspezifischen Kapazität lösen
3. Bei BESS: Degradation wirkt auf bereits installierte Kapazität (Tranchenmodell)

### Enumeration (kein Grid Search)

Alle definierten Zubau-Raten werden vollständig berechnet. Es gibt keine Optimierung im MVP:

```
BESS:     zubau_kw_pa = [0, 20, 50, 100, 200, 500] × e_to_p = [1, 2, 4]  → 18 Punkte
WP:       zubau_kw_pa = [0, 50, 100, 250, 500]                            → 5 Punkte
Wallbox:  zubau_kw_pa = [0, 20, 50, 100, 200]                             → 5 Punkte
```

Gesamt: 28 Punkte (bei je einer Instanz pro Typ).

### Performance-Implikation: 15min-Auflösung

- **Pro Tag**: 96 Zeitschritte → LP mit ~480 Variablen (BESS-only)
- HiGHS löst 96-Schritt-LP in <2ms → **~0.7s pro Jahr**
- **28 Punkte × 25 Jahre**: ~490s ≈ **~8 min** (parallelisiert: ~2-3 min)

### BESS-Degradation bei gestaffeltem Zubau

```
Jahr 5, Zubau-Rate = 100 kW/a:
  Tranche 1 (4 Jahre alt): 100 kW × (1 - 0.02)^4 = 92.2 kW
  Tranche 2 (3 Jahre alt): 100 kW × (1 - 0.02)^3 = 94.1 kW
  ...
  Tranche 5 (0 Jahre alt): 100 kW × 100% = 100.0 kW
  Gesamt-Kapazität = Σ [ 100 × (1 - 0.02)^(5-i) ] = 480.4 kW (statt naiv 500 kW)
```

Alle Tranchen werden im LP als ein aggregierter BESS modelliert. Die Gesamtkapazität wird vorab berechnet.

---

## 6. SLP-Aufbereitung (Interne Methode)

--> Anpassungen: Die SLP Daten liegen in .data/bdew_profile_2025.json. Die Profile müssen nicht mehr jährlich gebildet
werden, sondern nur einmalig, basierend auf dem Wetterjahr des Preisszenarios. Cache das Ergebnis in.data/bdew_cache um
die Berechnungen für neue Szenarien zu beschleunigen. Der Cache muss nur erneuert werden, wenn das Änderungsdatum der
bdew_profile_2025.json aktueller ist, als das des caches.
Die Vorgehensweise zur Dynamisierung und Anwendung ist in
.docs/2025-03-17_AWH_Aktuallisierte_SLP_Strom_2025_Veröffentlichung.pdf zu entnehmen.

### Anforderung

Die BDEW-SLP-Daten müssen **jahresabhängig** aufbereitet werden, da Tagestypen (Werktag/Samstag/Sonntag/
Feiertag) vom konkreten Kalenderjahr abhängen. Das SLP für 2027 hat andere Werktags-Verteilungen als 2028.

Die Aufbereitung muss an das Jahr des jeweiligen Preisszenarios gekoppelt sein (analog zur bestehenden
PVGIS-Zeitreihen-Zuordnung über `weather_year`).

### Vorgehen

```
Input:  BDEW-SLP Excel-Datei (unveränderlich, im Projekt-Repository)
        + Kalenderjahr (aus Preisszenario-Definition)

Schritte:
  1. Lese BDEW-Koeffizienten aus Excel (Tagestypen × Jahreszeiten × 96 Viertelstunden)
  2. Generiere Kalender für das Zieljahr (Werktag/Sa/So, Feiertage → So)
  3. Wende Dynamisierungsfunktion an (Polynomkoeffizienten × Temperatur aus PVGIS)
  4. Erzeuge 35.040 normierte Viertelstundenwerte (auf 1.000 kWh/a)
  5. Cache das Ergebnis: .data/slp_cache/h0_{year}.npy

Output: numpy-Array mit 35.040 normierten kWh-Werten

Skalierung im Modell:
  last[t] = slp_normiert[t] × (jahresverbrauch_kwh / 1000) × kundenanzahl
```

### Caching

- Cache-Key: `{slp_type}_{year}` (z.B. `h0_2027`)
- Invalidierung: Nur nötig wenn die BDEW-Excel-Datei sich ändert (sehr selten)
- Analoges Vorgehen zum bestehenden PVGIS-Cache in `.data/pvgis_cache/`

---

## 7. Wärmelastprofil aus PVGIS-Temperaturdaten

### Gradtagszahl-Ansatz

PVGIS liefert stündliche Temperaturwerte (`T2m` – 2m Außentemperatur). Daraus wird ein
temperaturabhängiges Wärmelastprofil abgeleitet:

```
Gradtagszahl (Heizgrenze 15°C):
  GTZ[t] = max(0, T_heiz - T_außen[t])

Wärmebedarf pro Zeitschritt:
  Q_th[t] = GTZ[t] / Σ_jahr(GTZ) × Q_jahres_gesamt

Elektrischer WP-Bedarf (Mindestwert, vor Flexibilisierung):
  P_wp_base[t] = Q_th[t] / COP(T_außen[t])

COP-Kennlinie (vereinfacht linear):
  COP(T) = COP_nenn × (1 + 0.025 × (T_außen - T_nenn))
```

`P_wp_base[t]` ist der "starre" WP-Bedarf ohne Flexibilisierung. Im LP wird dieser zum verschiebbaren
Bedarf: Die WP muss den Tages-Wärmebedarf decken, darf aber die zeitliche Verteilung optimieren
(begrenzt durch den thermischen Speicher).

---

## 8. Zur Frage: Imperfekte Voraussicht im LP

> Wie könnte man das LP anpassen, sodass es keine perfekte Voraussicht gibt? 6h perfekte Voraussicht,
> die anderen 18h mit Noise, und dann um 6h iterieren?

### Ansatz: Rolling-Horizon mit Forecast-Noise

Statt eines 24h-LP mit perfekter Preisvoraussicht kann man einen **Rolling-Horizon** implementieren:

```
Für jeden Tag:
  t = 0
  while t < 96:  // 96 Viertelstunden pro Tag
    1. Nimm die echten Preise für t..t+24  (6h perfekte Voraussicht)
    2. Für t+24..t+96: verwende Preis + N(0, σ_forecast)  (verrauschte Prognose)
    3. Löse LP über das gesamte Restfenster [t..96]
    4. Fixiere nur die Entscheidungen für t..t+24 (die nächsten 6h)
    5. t = t + 24  (rücke 6h vor)
```

**Auswirkung**:

- 4 LP-Solves pro Tag statt 1 → **4× mehr Rechenzeit**
- 28 Punkte × 25 Jahre × 365 Tage × 4 Solves = ~1M LP-Solves → ca. 30-35 min (parallelisiert: ~8-10 min)
- Das ist grenzwertig für den MVP, aber machbar

**Vorteil**: Realistischere Bewertung des Flexibilitätswerts, da Arbitrage-Gewinne nicht mehr auf perfekter
Preiskenntnis basieren.

**Nachteil**: Zusätzlicher Implementierungsaufwand, Parameter `σ_forecast` muss kalibriert werden.

**Empfehlung für MVP**: Perfekte Voraussicht beibehalten, aber als **Post-MVP-Feature** vormerken. Der
`perfect_foresight_discount`-Faktor (z.B. 0.8) ist eine einfachere Annäherung für den Anfang.
Alternativ: Rolling-Horizon als optionaler Modus im JSON (`"forecast_mode": "perfect" | "rolling_6h"`).

---

## 9. Architektur

### Modulstruktur

```
pv_bess_model/
├── ...bestehende Module (unverändert)...
├── portfolio/                           # NEU
│   ├── __init__.py
│   ├── load_profiles.py                 # BDEW-SLP aus Excel, jahresabhängig, gecacht
│   ├── heat_demand.py                   # Wärmelast aus PVGIS-Temperatur (Gradtagszahl)
│   ├── system_value.py                  # Welt-A/Welt-B Vergleich, Delta-Berechnung
│   └── marginal_value.py               # Grenznutzen-Kurven
├── dispatch/
│   ├── optimizer.py                     # UNVERÄNDERT (bestehendes PV+BESS Tool)
│   └── optimizer_portfolio.py           # NEU: LP mit Netto-Position + Flex (15min)
├── main.py                             # UNVERÄNDERT
└── main_portfolio.py                   # NEU: Separater Entry Point
```

### Entry Point

```bash
# Bestehendes PV+BESS Tool (unverändert)
python -m pv_bess_model.main --scenario scenarios/eeg_green.json

# Neues Portfolio/Systemwert-Tool
python -m pv_bess_model.main_portfolio --config portfolio/systemwert_2027.json
```

---

## 10. JSON-Input

Das JSON orientiert sich an der bestehenden Struktur, insbesondere `price_inputs.scenarios[]` aus dem
bestehenden Modell.

```json
{
  "meta_model": {
    "name": "Systemwert_Flex_2027",
    "baseline_year": 2027,
    "project_lifetime_years": 25,
    "perfect_foresight_discount": 0.8,
    "output": {
      "directory": "./outputs/systemwert_flex_2027",
      "export_dispatch_sample": true,
      "csv_separator": ";",
      "csv_decimal": ","
    }
  },
  "portfolio": {
    "generation": [
      {
        "type": "pv",
        "name": "PV Aggregiert",
        "peak_power_kwp": 19500,
        "location": {
          "latitude": 53.55,
          "longitude": 9.99
        },
        "degradation_rate_pct_per_year": 0.4,
        "system_loss_pct": 14.0,
        "mounting_type": "free",
        "azimuth_deg": 0,
        "tilt_deg": 30
      }
    ],
    "load": [
      {
        "type": "slp",
        "name": "Haushaltskunden",
        "slp_type": "H0",
        "bdew_excel": "data/bdew_slp_2024.xlsx",
        "customer_count": 8500,
        "annual_consumption_kwh_per_customer": 3200,
        "annual_growth_factor": 1.01
      }
    ]
  },
  "flexibilities": [
    {
      "type": "bess",
      "name": "Grossspeicher",
      "annual_addition_kw": [
        0,
        20,
        50,
        100,
        200,
        500
      ],
      "e_to_p_ratio_hours": [
        1,
        2,
        4
      ],
      "round_trip_efficiency_pct": 88.0,
      "min_soc_pct": 10.0,
      "max_soc_pct": 90.0,
      "degradation_rate_pct_per_year": 2.0
    },
    {
      "type": "heat_pump",
      "name": "Fernwaerme-WP Luft-Wasser",
      "annual_addition_kw": [
        0,
        50,
        100,
        250,
        500
      ],
      "cop_nominal": 3.5,
      "cop_reference_temp_c": 7.0,
      "annual_thermal_demand_mwh": 15000,
      "thermal_storage_kwh": 10000
    },
    {
      "type": "heat_pump",
      "name": "Dezentrale Haus-WP",
      "annual_addition_kw": [
        0,
        25,
        50,
        100
      ],
      "cop_nominal": 4.0,
      "cop_reference_temp_c": 7.0,
      "annual_thermal_demand_mwh": 5000,
      "thermal_storage_kwh": 2000
    },
    {
      "type": "ev_charging",
      "name": "Wallbox-Fleet V2G",
      "annual_addition_kw": [
        0,
        20,
        50,
        100,
        200
      ],
      "daily_energy_demand_kwh": 5000,
      "time_window": {
        "arrival_hour": 17,
        "departure_hour": 7
      },
      "v2g_enabled": true,
      "v2g_rte_pct": 90.0,
      "min_departure_soc_pct": 80.0,
      "usable_battery_kwh_per_kw": 3.0
    },
    {
      "type": "ev_charging",
      "name": "Wallbox-Fleet unidirektional",
      "annual_addition_kw": [
        0,
        20,
        50,
        100,
        200
      ],
      "daily_energy_demand_kwh": 5000,
      "time_window": {
        "arrival_hour": 17,
        "departure_hour": 7
      },
      "v2g_enabled": false,
      "min_departure_soc_pct": 80.0,
      "usable_battery_kwh_per_kw": 3.0
    }
  ],
  "price_inputs": {
    "inflation_timeseries": {
      "csv_path": "/test.csv",
      "inflation_column": "I",
      "year_column": "Y",
      "separator": ",",
      "decimal": "."
    },
    "scenarios": [
      {
        "name": "Central",
        "label": "Zentralszenario",
        "csv_column": "price_central",
        "weather_year": 2018,
        "weight": 0.6,
        "is_central": true,
        "price_csv": "./inputs/day_ahead_prices.csv",
        "inflation_on_input_data": false,
        "csv_separator": ";",
        "csv_decimal": ",",
        "csv_timestamp_column": "timestamp",
        "csv_timestamp_format": "%Y-%m-%dT%H:%M:%S"
      },
      {
        "name": "High",
        "label": "Hohes Preisniveau",
        "csv_column": "price_high",
        "weather_year": 2015,
        "weight": 0.25,
        "is_central": false,
        "price_csv": "./inputs/day_ahead_prices.csv",
        "inflation_on_input_data": false,
        "csv_separator": ";",
        "csv_decimal": ",",
        "csv_timestamp_column": "timestamp",
        "csv_timestamp_format": "%Y-%m-%dT%H:%M:%S"
      },
      {
        "name": "Low",
        "label": "Niedriges Preisniveau",
        "csv_column": "price_low",
        "weather_year": 2016,
        "weight": 0.15,
        "is_central": false,
        "price_csv": "./inputs/day_ahead_prices.csv",
        "inflation_on_input_data": false,
        "csv_separator": ";",
        "csv_decimal": ",",
        "csv_timestamp_column": "timestamp",
        "csv_timestamp_format": "%Y-%m-%dT%H:%M:%S"
      }
    ]
  }
}
```

--> Nutze ebenfalls die jährliche Inflation aus dem Basis Modell.
Beachte:

- `generation` und `load` sind Listen, erweiterbar um weitere Erzeuger/Lastgruppen
- `flexibilities` erlaubt mehrere Instanzen desselben Typs (z.B. zwei `heat_pump`-Einträge)
- `price_inputs.scenarios[]` ist identisch zur bestehenden Struktur aus dem PV+BESS-Modell
- Monte Carlo ist komplett entfernt (Post-MVP)
- `weather_year` in den Preisszenarien steuert, für welches Kalenderjahr das SLP generiert wird
  --> Die Listen von Generation, Load, und Flexibilities sollen ebenfalls die Möglichkeit besitzen, die Zubauraten erst
  ab Simulationsjahr X starten zu lassen.

---

## 11. Output

### CSV-Dateien

1. **`{name}_baseline.csv`** – Welt-A-Ergebnis: Jährliche Systemkosten ohne Flex
2. **`{name}_system_value.csv`** – Eine Zeile pro (Flex-Instanz, Zubau-Rate): kumulierter Systemwert (25J)
3. **`{name}_marginal_value.csv`** – Grenznutzenkurve: €/kW·a pro Ausbaustufe und Flex-Instanz
4. **`{name}_dispatch_sample.csv`** – Viertelstundenscharfer Dispatch (ausgewählte Konfiguration, 1 Jahr)

### HTML-Dashboard (wie bestehendes Modell)

- Grenznutzenkurven überlagert (alle Flex-Instanzen in einem Chart)
- Systemwert kumuliert über Zubau-Rate
- Vergleichstabelle: €/kW·a bei gleicher Zubau-Rate
- Ranking: "Bester nächster investierter Euro" (sobald Grenzkosten ergänzt werden)

### HTML-Input-Formular

Ein HTML-Formular basierend auf dem JSON-Schema der Input-Datei, analog zum bestehenden Modell.
Wird einmalig erstellt und erlaubt die komfortable Konfiguration der Szenarien im Browser.

---

## 12. Implementierungsplan (Detailliert)

### Übersicht: 7 Phasen

```
Phase 1: Fundament (Config, Loader, Entry Point)          ── Voraussetzung für alles
Phase 2: Datenquellen (SLP, PVGIS-Temperatur, Wärmelast)  ── Voraussetzung für LP
Phase 3: Welt A + Portfolio-LP (BESS-only)                 ── Kernstück
Phase 4: Multi-Year Engine mit Zubau + Enumeration         ── Systemwert-Berechnung
Phase 5: WP-Flex im LP                                     ── Erweiterung
Phase 6: EV/V2G-Flex im LP                                 ── Erweiterung
Phase 7: Output (CSV + HTML-Dashboard)                      ── Ergebnisdarstellung
```

Abhängigkeitsgraph:

```
Phase 1 ──→ Phase 2 ──→ Phase 3 ──→ Phase 4 ──→ Phase 7
                  │           │
                  └──→ Phase 5 ──→ Phase 4
                              │
                        Phase 6 ──→ Phase 4
```

Phase 5 und 6 können parallel entwickelt werden, sobald Phase 3 steht.
Phase 7 kann teilweise parallel zu Phase 4–6 entwickelt werden (CSV-Skeleton).

---

### Phase 1: Fundament (Config, Loader, Entry Point)

**Ziel:** JSON-Schema, Loader und CLI-Entry-Point, damit alle folgenden Phasen darauf aufbauen.

**Voraussetzungen:** Keine (erster Schritt).

**Wiederverwendung aus bestehender Codebase:**

- `config/schema.py` → Vorlage für JSON-Validierung (jsonschema-Library)
- `config/loader.py` → `PriceWeatherScenario`-Dataclass und `load_price_csv()` direkt wiederverwendbar
- `config/defaults.py` → Zeitkonstanten (INTERVALS_PER_DAY=96, TIMESTEP_HOURS=0.25) bereits vorhanden

#### Schritt 1.1: Portfolio-Defaults in `config/defaults.py` ergänzen

Neue Konstanten hinzufügen (bestehende Datei erweitern, NICHT neu anlegen):

```python
# ---------------------------------------------------------------------------
# Portfolio defaults
# ---------------------------------------------------------------------------
DEFAULT_PERFECT_FORESIGHT_DISCOUNT: float = 0.8
DEFAULT_PORTFOLIO_LIFETIME_YEARS: int = 25
DEFAULT_SLP_CACHE_DIR: str = ".data/slp_cache"
DEFAULT_HEAT_DEMAND_HEIZGRENZE_C: float = 15.0
DEFAULT_COP_TEMP_COEFFICIENT: float = 0.025
SLP_NORMIERUNG_KWH: float = 1000.0
```

**Dateien:** `pv_bess_model/config/defaults.py` (ergänzen)

#### Schritt 1.2: Portfolio-JSON-Schema erstellen

Neues Schema für die Portfolio-Input-Datei. Separates Schema, nicht das bestehende PV+BESS-Schema erweitern.

```python
# pv_bess_model/config/schema_portfolio.py
```

Validiert:

- `meta_model`-Block (name, baseline_year, lifetime, output)
- `portfolio.generation[]`-Liste (type=pv, peak_power_kwp, location, etc.)
- `portfolio.load[]`-Liste (type=slp, slp_type, customer_count, annual_consumption, growth_factor)
- `flexibilities[]`-Liste (type ∈ {bess, heat_pump, ev_charging}, jeweils typspezifische Felder)
- `price_inputs.scenarios[]` → identische Struktur zum bestehenden Modell

**Dateien:** `pv_bess_model/config/schema_portfolio.py` (neu)

#### Schritt 1.3: Portfolio-Loader erstellen

Lädt und validiert Portfolio-JSON. Wiederverwendet `load_price_csv()` und `PriceWeatherScenario` aus `config/loader.py`.

```python
# pv_bess_model/config/loader_portfolio.py

@dataclass
class GenerationConfig:
    type: str  # "pv"
    name: str
    peak_power_kwp: float
    latitude: float
    longitude: float
    # ... (alle PV-Felder aus JSON)


@dataclass
class LoadGroupConfig:
    type: str  # "slp"
    name: str
    slp_type: str  # "H0"
    bdew_excel: str
    customer_count: int
    annual_consumption_kwh_per_customer: float
    annual_growth_factor: float


@dataclass
class FlexConfig:
    type: str  # "bess" | "heat_pump" | "ev_charging"
    name: str
    annual_addition_kw: list[float]
    # ... typspezifische Felder


@dataclass
class PortfolioScenarioConfig:
    meta: MetaModelConfig
    generation: list[GenerationConfig]
    load: list[LoadGroupConfig]
    flexibilities: list[FlexConfig]
    price_scenarios: list[PriceWeatherScenario]  # Wiederverwendung!
```

**Dateien:** `pv_bess_model/config/loader_portfolio.py` (neu)
**Wiederverwendet:** `PriceWeatherScenario` aus `config/loader.py`

#### Schritt 1.4: Entry Point `main_portfolio.py`

Separater CLI-Entry-Point. Grundstruktur analog zu `main.py`, aber schlank:

```bash
python -m pv_bess_model.main_portfolio --config portfolio/systemwert_2027.json
python -m pv_bess_model.main_portfolio --config portfolio/systemwert_2027.json --dry-run
python -m pv_bess_model.main_portfolio --config portfolio/systemwert_2027.json -v
```

Erst nur Grundstruktur: argparse, JSON laden, validieren, Platzhalter für Simulation.

**Dateien:** `pv_bess_model/main_portfolio.py` (neu)

#### Schritt 1.5: Verzeichnisstruktur anlegen

```
pv_bess_model/portfolio/
├── __init__.py
├── load_profiles.py      # Platzhalter
├── heat_demand.py         # Platzhalter
├── system_value.py        # Platzhalter
└── marginal_value.py      # Platzhalter
```

**Tests für Phase 1:**

- `tests/test_schema_portfolio.py` – JSON-Validierung (gültige/ungültige Inputs)
- `tests/test_loader_portfolio.py` – Laden und Parsen einer Beispiel-JSON

---

### Phase 2: Datenquellen (SLP, Temperatur, Wärmelast)

**Ziel:** Alle Input-Zeitreihen bereitstellen, die der LP-Optimizer benötigt.

**Voraussetzungen:** Phase 1 (Loader kennt Pfade und Konfiguration).

**Wiederverwendung:**

- `pv/pvgis_client.py` → PVGIS-Abruf (muss um T2m-Temperatur erweitert werden)
- `pv/timeseries.py` → `hourly_to_quarter_hourly()` direkt wiederverwendbar
- `market/price_loader.py` → `load_price_csv()` direkt wiederverwendbar

#### Schritt 2.1: PVGIS-Client um Temperaturdaten erweitern

`pv/pvgis_client.py` erweitern: PVGIS liefert `T2m` (2m-Außentemperatur) als Teil der
`seriescalc`-Antwort. Aktuell wird nur die PV-Produktion (`P`) extrahiert.

Änderung: Auch `T2m` extrahieren und zurückgeben. Cache-Struktur erweitern.

```python
# Erweiterung in pvgis_client.py
def fetch_hourly_data(...) -> dict:
    """Returns dict with keys 'production' and 'temperature' (both hourly arrays)."""
```

**Dateien:** `pv_bess_model/pv/pvgis_client.py` (erweitern)
**Tests:** `tests/test_pvgis_client.py` (erweitern – T2m-Extraktion testen)

#### Schritt 2.2: SLP-Modul implementieren

```python
# pv_bess_model/portfolio/load_profiles.py

def load_bdew_slp(
        excel_path: str,
        slp_type: str,  # "H0"
        year: int,  # Kalenderjahr (für Tagestyp-Zuordnung)
        temperature: np.ndarray,  # 8760 Stunden-Temperaturen aus PVGIS
) -> np.ndarray:
    """
    Erzeugt 35.040 normierte Viertelstundenwerte (auf 1.000 kWh/a).

    Schritte:
    1. BDEW-Koeffizienten aus Excel lesen (Tagestyp × Jahreszeit × 96 VH)
    2. Kalender für Zieljahr generieren (Werktag/Sa/So, Feiertage → So)
    3. Dynamisierungsfunktion anwenden (Polynomkoeffizienten × Temperatur)
    4. Normierung auf 1.000 kWh/a
    5. Cache: .data/slp_cache/{slp_type}_{year}.npy
    """


def scale_slp(
        slp_normalized: np.ndarray,  # 35.040 Werte, normiert auf 1.000 kWh/a
        annual_consumption_kwh: float,
        customer_count: int,
) -> np.ndarray:
    """Skaliert SLP auf tatsächlichen Verbrauch."""
    return slp_normalized * (annual_consumption_kwh / 1000.0) * customer_count


def generate_calendar(year: int) -> list[str]:
    """Erzeugt Tagestyp-Kalender: 365 Einträge ('W', 'SA', 'SO')."""
    # Deutsche Feiertage → 'SO'
```

**Offene Frage (muss VOR Implementierung geklärt werden):**

- Welche BDEW-Excel-Datei? Format? Bereits vorhanden?
- Zeitzone: UTC oder MEZ/MESZ? (Preise sind UTC, SLP typisch MEZ)

**Dateien:** `pv_bess_model/portfolio/load_profiles.py` (neu)
**Tests:** `tests/test_load_profiles.py` (neu)

- Test: Normiertes SLP summiert sich auf 1.000 kWh
- Test: Korrekte Tagestyp-Zuordnung (Feiertag = Sonntag)
- Test: Skalierung mit customer_count und annual_consumption
- Test: Cache-Hit vermeidet Neuberechnung

#### Schritt 2.3: Wärmelast-Modul implementieren

```python
# pv_bess_model/portfolio/heat_demand.py

def compute_heat_demand(
        temperature_hourly: np.ndarray,  # 8760 Stunden-Temperaturen (°C)
        annual_thermal_demand_mwh: float,  # Jahres-Wärmebedarf in MWh
        heizgrenze_c: float = 15.0,  # Heizgrenztemperatur
) -> np.ndarray:
    """
    Gradtagszahl-basiertes Wärmelastprofil (35.040 Viertelstundenwerte in kWh_th).

    1. GTZ[h] = max(0, heizgrenze - T_außen[h])  pro Stunde
    2. Q_th[h] = GTZ[h] / Σ(GTZ) × Q_jahres_gesamt
    3. Expansion: Stunde → 4 Viertelstunden (÷4)
    """


def compute_cop(
        temperature_hourly: np.ndarray,
        cop_nominal: float,
        cop_reference_temp_c: float,
        temp_coefficient: float = 0.025,
) -> np.ndarray:
    """
    COP-Kennlinie (vereinfacht linear):
    COP(T) = cop_nominal × (1 + temp_coefficient × (T - T_referenz))

    Returns: 35.040 COP-Werte (Quarter-hourly, replicated from hourly).
    """
```

**Dateien:** `pv_bess_model/portfolio/heat_demand.py` (neu)
**Tests:** `tests/test_heat_demand.py` (neu)

- Test: Wärmebedarf = 0 bei T > Heizgrenze
- Test: Jahressumme = annual_thermal_demand (Energieerhaltung)
- Test: COP steigt mit Temperatur
- Test: COP bei Referenztemperatur = cop_nominal

#### Schritt 2.4: PV-Erzeugungsprofile für Portfolio

Wiederverwendung: `pv/pvgis_client.py` + `pv/timeseries.py` + `pv/degradation.py`.

Neue Hilfsfunktion in `portfolio/` die alle `generation[]`-Einträge aggregiert:

```python
# pv_bess_model/portfolio/generation.py  (oder in system_value.py)

def build_aggregated_pv_profile(
        generation_configs: list[GenerationConfig],
        weather_year: int,
) -> np.ndarray:
    """
    Für jeden Erzeuger: PVGIS-Daten holen, quarter-hourly konvertieren.
    Aggregation: Summe aller Erzeuger-Profile.
    Returns: 35.040 kWh-Werte.
    """
```

**Wiederverwendet:** `fetch_hourly_data()`, `hourly_to_quarter_hourly()`, `apply_degradation()`

---

### Phase 3: Welt A + Portfolio-LP (BESS-only)

**Ziel:** Welt-A-Berechnung (ohne Flex) und erster Portfolio-Optimizer mit BESS als Flexibilität.

**Voraussetzungen:** Phase 2 (alle Zeitreihen verfügbar).

**Wiederverwendung:**

- `dispatch/optimizer.py` → Strukturvorlage für LP-Aufbau (scipy.optimize.linprog, HiGHS)
- `dispatch/engine.py` → `DispatchEngineConfig`-Muster für Konfiguration

#### Schritt 3.1: Welt-A-Simulation (ohne Flex)

Einfachste Berechnung – kein LP nötig:

```python
# pv_bess_model/portfolio/system_value.py

def compute_world_a(
        pv_profile: np.ndarray,  # 35.040 kWh (aggregierte PV-Erzeugung)
        load_profile: np.ndarray,  # 35.040 kWh (aggregierte Last)
        spot_prices: np.ndarray,  # 35.040 €/kWh
        timestep_hours: float = 0.25,
) -> WorldAResult:
    """
    Netto-Position pro Viertelstunde:
      netto[t] = pv[t] - load[t]
      netto > 0 → Verkauf zu spot[t]
      netto < 0 → Einkauf zu spot[t]

    Returns:
      system_cost: float (Σ Einkauf × spot - Σ Verkauf × spot)
      hourly_netto: np.ndarray (für Dispatch-Sample)
    """
```

**Tests:**

- Test: Reine PV (keine Last) → System-Erlös = Σ(pv × spot)
- Test: Reine Last (keine PV) → Systemkosten = Σ(last × spot)
- Test: PV = Last → Systemkosten = 0
- Test: Energieerhaltung (Einkauf + Verkauf = |netto|)

#### Schritt 3.2: Portfolio-LP-Optimizer (BESS als erste Flex)

Neuer Optimizer, der den bestehenden `optimizer.py` als Strukturvorlage nutzt, aber eine
**andere Zielfunktion** hat (Netto-Position statt PV-Export + BESS-Entladung).

```python
# pv_bess_model/dispatch/optimizer_portfolio.py

@dataclass
class PortfolioLPConfig:
    """Statische LP-Konfiguration für einen Tag."""
    timestep_hours: float
    intervals_per_day: int  # 96
    perfect_foresight_discount: float


@dataclass
class BessFlexParams:
    """BESS-Parameter für das Portfolio-LP."""
    capacity_kwh: float
    power_kw: float
    rte: float
    min_soc_pct: float
    max_soc_pct: float
    start_soc_kwh: float


@dataclass
class PortfolioDailyResult(TypedDict):
    """Ergebnis eines täglichen LP-Solves."""
    grid_sell: np.ndarray  # 96 Werte
    grid_buy: np.ndarray  # 96 Werte
    bess_charge: np.ndarray  # 96 Werte
    bess_discharge: np.ndarray  # 96 Werte
    bess_soc: np.ndarray  # 97 Werte (inkl. End-SoC)
    system_cost: float  # Tages-Systemkosten
    daily_revenue: float  # Tages-Erlös


def optimize_portfolio_day(
        pv_production: np.ndarray,  # 96 kWh-Werte
        load_demand: np.ndarray,  # 96 kWh-Werte
        spot_prices: np.ndarray,  # 96 €/kWh-Werte
        bess_params: BessFlexParams | None,
        config: PortfolioLPConfig,
) -> PortfolioDailyResult:
    """
    LP-Formulierung:

    Entscheidungsvariablen (pro Viertelstunde t, t=0..95):
      grid_sell[t]      – Netto-Einspeisung (kWh/15min)
      grid_buy[t]       – Netto-Bezug (kWh/15min)
      bess_charge[t]    – BESS-Laden (kWh/15min)
      bess_discharge[t] – BESS-Entladen (kWh/15min)
      soc[t]            – BESS-Ladezustand (kWh)

    Zielfunktion:
      max Σ_t [ grid_sell[t] × spot[t] × discount
              - grid_buy[t] × spot[t] ]

    Nebenbedingungen:
      grid_sell[t] - grid_buy[t] = pv[t] - load[t]
                                   + bess_discharge[t] × RTE
                                   - bess_charge[t]               ∀t
      soc[t+1] = soc[t] + bess_charge[t] - bess_discharge[t]     ∀t
      soc_min ≤ soc[t] ≤ soc_max                                 ∀t
      bess_charge[t] ≤ P_max × timestep_hours                    ∀t
      bess_discharge[t] ≤ P_max × timestep_hours                 ∀t
      grid_sell[t], grid_buy[t] ≥ 0                              ∀t
    """
```

**Wichtiger Unterschied zum bestehenden Optimizer:**

- Bestehendes Modell: Netzanschluss-Limit, Green/Grey-Mode, Floor/Cap-Preise, GoO
- Portfolio-LP: Keine Netzanschluss-Limitierung, kein Green/Grey, keine PPA/EEG
- Portfolio-LP: `perfect_foresight_discount` als Skalierungsfaktor auf Sell-Erlöse
- Portfolio-LP: Bidirektionale Netto-Position (Kauf UND Verkauf möglich)

**Dateien:** `pv_bess_model/dispatch/optimizer_portfolio.py` (neu)
**Tests:** `tests/test_optimizer_portfolio.py` (neu)

- Test: Ohne BESS (bess_params=None) → Ergebnis = Welt-A
- Test: BESS vorhanden, alle Preise gleich → keine Arbitrage (charge=0, discharge=0)
- Test: BESS vorhanden, Preissprung → Arbitrage (laden bei niedrigem Preis, entladen bei hohem)
- Test: SoC-Grenzen werden eingehalten
- Test: Energiebilanz (netto = pv - last + discharge×RTE - charge)
- Test: SoC-Kopplung (end_soc = start_soc + Σcharge - Σdischarge)
- Test: perfect_foresight_discount < 1.0 reduziert Systemwert

---

### Phase 4: Multi-Year Engine + Enumeration + Systemwert

**Ziel:** 25-Jahres-Simulation mit jährlichem Flex-Zubau, Tranchenmodell, vollständige Enumeration.

**Voraussetzungen:** Phase 3 (Portfolio-LP funktioniert für einzelne Tage).

**Wiederverwendung:**

- `dispatch/engine.py` → Vorlage für Multi-Year-Loop (Degradation, Inflation, SoC-Kopplung)
- `pv/degradation.py` → `apply_degradation()` für PV
- `finance/inflation.py` → `inflate_value()` für Preise

#### Schritt 4.1: Portfolio-Engine (Multi-Year-Loop)

```python
# pv_bess_model/dispatch/engine_portfolio.py

@dataclass
class PortfolioEngineConfig:
    """Konfiguration für die Multi-Year Portfolio-Simulation."""
    lifetime_years: int
    baseline_year: int
    timestep_hours: float
    intervals_per_day: int
    intervals_per_year: int
    perfect_foresight_discount: float


@dataclass
class PortfolioAnnualResult:
    """Ergebnis eines Simulationsjahres."""
    year: int
    system_cost: float
    total_grid_sell_kwh: float
    total_grid_buy_kwh: float
    total_bess_throughput_kwh: float
    # ... je nach Flex-Typ


@dataclass
class FlexCapacityYear:
    """Flex-Kapazitäten für ein Projektjahr (nach Zubau + Degradation)."""
    bess_capacity_kwh: float
    bess_power_kw: float
    # wp_power_kw: float        # Phase 5
    # ev_power_kw: float        # Phase 6


def compute_bess_tranche_capacity(
        annual_addition_kw: float,
        e_to_p_ratio: float,
        project_year: int,  # 1-basiert
        degradation_rate: float,
) -> tuple[float, float]:
    """
    Tranchenmodell: Jede jährliche Zubau-Tranche degradiert unabhängig.

    Gesamt-Kapazität Jahr n:
      Σ_{i=1}^{n} [ addition_kw × e_to_p × (1 - deg_rate)^(n-i) ]

    Returns: (total_power_kw, total_capacity_kwh)
    """


def run_portfolio_simulation(
        config: PortfolioEngineConfig,
        pv_profile_base: np.ndarray,  # 35.040 kWh (Basis-Jahr)
        load_profile_base: np.ndarray,  # 35.040 kWh (Basis-Jahr)
        spot_prices_base: np.ndarray,  # 35.040 €/kWh (Basis-Jahr)
        flex_config: FlexConfig,
        annual_addition_kw: float,  # Konkreter Zubau-Punkt
        pv_degradation_rate: float,
        load_growth_factor: float,
        inflation_rate: float,
) -> list[PortfolioAnnualResult]:
    """
    25-Jahres-Simulation:
    Für jedes Jahr:
      1. PV-Degradation anwenden
      2. Last-Wachstum anwenden
      3. Preise inflationieren
      4. Flex-Kapazität berechnen (Zubau × Jahr, mit Tranchenmodell)
      5. 365 tägliche LP-Optimierungen
      6. Aggregation der Jahresergebnisse
    """
```

**Dateien:** `pv_bess_model/dispatch/engine_portfolio.py` (neu)
**Tests:** `tests/test_engine_portfolio.py` (neu)

- Test: 1-Jahres-Simulation ohne Flex = Welt A
- Test: Zubau 100 kW/a → Jahr 5 hat 500 kW (ohne Degradation)
- Test: Tranchenmodell: Gesamtkapazität < naive_kW bei Degradation > 0
- Test: PV-Degradation reduziert Erzeugung jährlich
- Test: Last-Wachstum erhöht Verbrauch jährlich
- Test: SoC-Kopplung über Tagesgrenzen funktioniert über 365 Tage

#### Schritt 4.2: Systemwert-Berechnung + Enumeration

```python
# pv_bess_model/portfolio/system_value.py (erweitern)

@dataclass
class SystemValuePoint:
    """Ergebnis für einen (Flex-Instanz, Zubau-Rate, E/P-Ratio)-Punkt."""
    flex_name: str
    flex_type: str
    annual_addition_kw: float
    e_to_p_ratio: float | None  # Nur BESS
    cumulative_system_value_eur: float  # Σ_jahre [cost_A(j) - cost_B(j)]
    annual_system_values: list[float]  # Pro-Jahr-Werte
    marginal_value_eur_per_kw_a: float  # Grenznutzen


@dataclass
class SystemValueResult:
    """Gesamtergebnis der Enumeration."""
    world_a_annual_costs: list[float]  # 25 Jahres-Systemkosten ohne Flex
    points: list[SystemValuePoint]  # Alle berechneten Punkte


def run_enumeration(
        config: PortfolioEngineConfig,
        pv_profiles: dict[str, np.ndarray],  # Pro Preisszenario
        load_profiles: dict[str, np.ndarray],  # Pro Preisszenario (SLP+Wetterjahr)
        spot_prices: dict[str, np.ndarray],  # Pro Preisszenario
        flexibilities: list[FlexConfig],
        central_scenario: str,
        **kwargs,
) -> SystemValueResult:
    """
    Enumeration aller Flex × Zubau-Rate × E/P-Ratio Kombinationen.

    Für jede Flex-Instanz:
      Für jede Zubau-Rate in annual_addition_kw:
        (Für BESS: zusätzlich für jede E/P-Ratio)
        1. Run 25-Jahres-Simulation (Welt B)
        2. Delta: Systemwert = cost_A - cost_B (pro Jahr, dann Summe)
        3. Grenznutzen = (Systemwert[rate] - Systemwert[rate-1]) / (rate - rate-1)

    Parallelisierung: Alle Punkte sind unabhängig → concurrent.futures
    """
```

#### Schritt 4.3: Grenznutzen-Kurven

```python
# pv_bess_model/portfolio/marginal_value.py

def compute_marginal_values(
        points: list[SystemValuePoint],
) -> list[MarginalValuePoint]:
    """
    Grenznutzen = ΔSystemwert / ΔKapazität

    Sortiert nach Zubau-Rate, berechnet diskrete Ableitung:
      marginal[i] = (value[i] - value[i-1]) / (kw[i] - kw[i-1])

    Ergebnis: €/kW·a pro Ausbaustufe
    """
```

**Dateien:**

- `pv_bess_model/portfolio/system_value.py` (erweitern)
- `pv_bess_model/portfolio/marginal_value.py` (neu)
- `pv_bess_model/dispatch/engine_portfolio.py` (erweitern)

**Tests:**

- Test: Systemwert bei Zubau=0 ist 0
- Test: Systemwert ist monoton steigend mit Zubau (bei positiven Preisspreads)
- Test: Grenznutzen ist monoton fallend (sinkender Grenzertrag)
- Test: Parallelisierung liefert gleiche Ergebnisse wie sequentielle Berechnung

---

### Phase 5: WP-Flex im LP

**Ziel:** Wärmepumpen als Flexibilität im Portfolio-Optimizer.

**Voraussetzungen:** Phase 3 (Portfolio-LP mit BESS läuft), Phase 2 (Wärmelast + COP verfügbar).

#### Schritt 5.1: WP-Parameter-Dataclass

```python
# In optimizer_portfolio.py ergänzen

@dataclass
class HeatPumpFlexParams:
    """Wärmepumpe-Parameter für das Portfolio-LP."""
    power_kw: float  # Elektrische Nennleistung
    cop_profile: np.ndarray  # 96 COP-Werte (temperaturabhängig, pro Tag)
    daily_heat_demand_kwh: float  # Tages-Wärmebedarf in kWh_th
    thermal_storage_kwh: float  # Thermischer Speicher in kWh_th
```

#### Schritt 5.2: WP-Variablen und Constraints im LP

Erweiterung von `optimize_portfolio_day()`:

```
Neue Entscheidungsvariablen:
  wp_load[t]          – elektrische WP-Aufnahme (kWh/15min)
  thermal_storage[t]  – thermischer Speicherstand (kWh_th)

Neue Constraints:
  Σ wp_load[t] × COP[t] = daily_heat_demand                          (Tages-Bilanz)
  0 ≤ wp_load[t] ≤ P_wp_max × timestep_hours                        (Power-Limit)
  0 ≤ thermal_storage[t] ≤ thermal_storage_max                       (Speicher-Limits)
  thermal_storage[t+1] = thermal_storage[t] + wp_load[t]×COP[t]
                         - heat_demand[t]                             (Speicher-Bilanz)

Anpassung der Netto-Position:
  grid_sell[t] - grid_buy[t] = pv[t] - load[t] - wp_load[t]
                               + bess_discharge[t]×RTE - bess_charge[t]
```

**Wichtig:** `wp_load[t]` erhöht den Strombezug (reduziert grid_sell / erhöht grid_buy).
Der Optimizer verschiebt WP-Laufzeiten in günstige Stunden.

**Dateien:** `pv_bess_model/dispatch/optimizer_portfolio.py` (erweitern)
**Tests:** `tests/test_optimizer_portfolio.py` (erweitern)

- Test: Ohne WP → Ergebnis wie Phase 3
- Test: WP ohne thermischen Speicher → WP läuft proportional zum Wärmebedarf (keine Verschiebung)
- Test: WP mit thermischem Speicher → WP verschiebt Last in günstige Stunden
- Test: Tages-Wärmebilanz eingehalten (Σ wp_load × COP = daily_heat_demand)
- Test: Thermischer Speicher bleibt in Grenzen

#### Schritt 5.3: WP-Zubau in Engine

Erweiterung von `engine_portfolio.py`:

- WP hat keinen Kapazitätsverlust (keine Degradation)
- Zubau: `wp_power_year_n = annual_addition_kw × n`
- Wärmebedarf skaliert mit installierter WP-Kapazität
  (Annahme: Zubau-Rate = zusätzliche Kunden/Gebäude angeschlossen)

---

### Phase 6: EV/V2G-Flex im LP

**Ziel:** Wallbox/E-Mobilität als Flexibilität, optional mit V2G (bidirektional).

**Voraussetzungen:** Phase 3 (Portfolio-LP läuft).

Kann **parallel zu Phase 5** entwickelt werden.

#### Schritt 6.1: EV-Parameter-Dataclass

```python
@dataclass
class EVFlexParams:
    """EV/Wallbox-Parameter für das Portfolio-LP."""
    power_kw: float  # Ladeleistung
    daily_energy_demand_kwh: float  # Tages-Energiebedarf der Flotte
    usable_battery_kwh: float  # Nutzbare Batterie-Kapazität der Flotte
    arrival_interval: int  # Ankunft (0-95)
    departure_interval: int  # Abfahrt (0-95)
    v2g_enabled: bool
    v2g_rte: float  # V2G Round-Trip-Efficiency
    min_departure_soc_pct: float  # Mindest-SoC bei Abfahrt
```

#### Schritt 6.2: EV-Variablen und Constraints im LP

```
Neue Entscheidungsvariablen:
  ev_charge[t]     – Laden Grid→EV (kWh/15min)
  ev_discharge[t]  – V2G-Entladen EV→Grid (kWh/15min), nur wenn v2g_enabled
  ev_soc[t]        – EV-Flotten-SoC (kWh)

Neue Constraints:
  ev_soc[t+1] = ev_soc[t] + ev_charge[t] - ev_discharge[t]           (SoC-Bilanz)
  ev_soc_min ≤ ev_soc[t] ≤ ev_soc_max                                (SoC-Grenzen)
  ev_charge[t] ≤ P_charge × timestep_hours                           (Power-Limit)
  ev_discharge[t] ≤ P_discharge × timestep_hours × v2g_rte           (V2G-Power)
  ev_charge[t] = 0 ∀t ∉ [arrival, departure]                         (Zeitfenster)
  ev_discharge[t] = 0 ∀t ∉ [arrival, departure]                      (Zeitfenster)
  ev_soc[departure] ≥ E_min_departure                                (Abfahrts-SoC)
  ev_discharge[t] = 0 ∀t  (wenn v2g_enabled = false)                 (Kein V2G)

Anpassung der Netto-Position:
  grid_sell[t] - grid_buy[t] = pv[t] - load[t] - wp_load[t]
                               + bess_discharge[t]×RTE - bess_charge[t]
                               + ev_discharge[t]×v2g_rte - ev_charge[t]
```

**Dateien:** `pv_bess_model/dispatch/optimizer_portfolio.py` (erweitern)
**Tests:** `tests/test_optimizer_portfolio.py` (erweitern)

- Test: Ohne V2G → nur Laden, kein Entladen
- Test: Mit V2G → Laden bei niedrigem Preis, Entladen bei hohem
- Test: Mindest-SoC bei Abfahrt eingehalten
- Test: Laden nur im Zeitfenster [arrival, departure]
- Test: Daily_energy_demand wird erfüllt (genug Energie geladen)
- Test: SoC-Grenzen eingehalten

#### Schritt 6.3: EV-Zubau in Engine

- Zubau: `ev_power_year_n = annual_addition_kw × n`
- `daily_energy_demand` skaliert proportional zum Zubau
- `usable_battery_kwh` skaliert mit `usable_battery_kwh_per_kw × power_kw`
- Keine Degradation (Vereinfachung MVP)

---

### Phase 7: Output (CSV + HTML-Dashboard)

**Ziel:** Alle Ergebnisse in CSV-Dateien und im HTML-Dashboard darstellen.

**Voraussetzungen:** Phase 4 (Systemwert-Ergebnisse vorhanden). Kann teilweise parallel entwickelt werden.

**Wiederverwendung:**

- `output/csv_writer.py` → Vorlage für CSV-Format, Delimiter, Dezimaltrennzeichen
- `output/formatting.py` → Number/Currency-Formatierung
- `output/report/` → HTML-Dashboard-Architektur (data_collector, html_builder, charts)

#### Schritt 7.1: CSV-Writer für Portfolio

```python
# pv_bess_model/output/csv_writer_portfolio.py

def write_baseline_csv(path, world_a_results):
    """Welt-A: Jährliche Systemkosten ohne Flex."""


def write_system_value_csv(path, system_value_result):
    """Eine Zeile pro (Flex, Zubau-Rate, E/P): kumulierter Systemwert."""


def write_marginal_value_csv(path, marginal_values):
    """Grenznutzenkurve: €/kW·a pro Ausbaustufe und Flex-Instanz."""


def write_portfolio_dispatch_sample_csv(path, daily_results, year):
    """Viertelstundenscharfer Dispatch (96 × 365 = 35.040 Zeilen)."""
```

**Dateien:** `pv_bess_model/output/csv_writer_portfolio.py` (neu)

#### Schritt 7.2: HTML-Dashboard für Portfolio

Neues Tab-Set im bestehenden Dashboard-Framework:

1. **Portfolio-Übersicht** – Eingabeparameter, Erzeuger, Lastgruppen, Flex-Typen
2. **Welt-A-Ergebnis** – Jährliche Systemkosten, Netto-Position-Profil
3. **Systemwert-Kurven** – Pro Flex-Instanz: Systemwert über Zubau-Rate
4. **Grenznutzen-Kurven** – Alle Flex-Instanzen überlagert (€/kW·a vs. kumulierte kW)
5. **Dispatch-Sample** – Viertelstunden-Dispatch eines ausgewählten Tages/Woche

**Dateien:**

- `pv_bess_model/output/report/data_collector_portfolio.py` (neu)
- `pv_bess_model/output/report/templates/dashboard_portfolio.html` (neu)
- `pv_bess_model/output/report/html_builder.py` (erweitern oder neuer Builder)

#### Schritt 7.3: Orchestrierung in `main_portfolio.py`

Entry Point vervollständigen:

```python
# Gesamtablauf in main_portfolio.py:
1.
JSON
laden
und
validieren
2.
PVGIS - Daten
abrufen(PV + Temperatur)
pro
weather_year
3.
SLP - Profile
generieren
pro
weather_year
4.
Wärmelast - Profile
berechnen
5.
Preis - Zeitreihen
laden
6.
Welt
A
berechnen(Zentralszenario)
7.
Enumeration: Alle
Flex × Zubau - Rate × E / P
durchrechnen(parallelisiert)
8.
Grenznutzen - Kurven
berechnen
9.
CSV - Dateien
schreiben
10.
HTML - Dashboard
generieren
11.
Zusammenfassung
auf
stdout
```

---

### Zusammenfassung: Dateien und Aufwand

| Phase | Neue Dateien                                                                                           | Geänderte Dateien                                             | Geschätzter Aufwand       |
|-------|--------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|---------------------------|
| 1     | schema_portfolio.py, loader_portfolio.py, main_portfolio.py, portfolio/__init__.py                     | defaults.py                                                   | Mittel                    |
| 2     | portfolio/load_profiles.py, portfolio/heat_demand.py                                                   | pv/pvgis_client.py                                            | Hoch (BDEW-Excel-Parsing) |
| 3     | dispatch/optimizer_portfolio.py, portfolio/system_value.py (Welt A)                                    | –                                                             | Hoch (neuer LP)           |
| 4     | dispatch/engine_portfolio.py, portfolio/marginal_value.py                                              | portfolio/system_value.py                                     | Hoch (Multi-Year + Enum)  |
| 5     | –                                                                                                      | dispatch/optimizer_portfolio.py, dispatch/engine_portfolio.py | Mittel (LP-Erweiterung)   |
| 6     | –                                                                                                      | dispatch/optimizer_portfolio.py, dispatch/engine_portfolio.py | Mittel (LP-Erweiterung)   |
| 7     | output/csv_writer_portfolio.py, report/data_collector_portfolio.py, templates/dashboard_portfolio.html | main_portfolio.py                                             | Mittel                    |

**Gesamtumfang:** ~12-15 neue Dateien, ~3-5 geänderte bestehende Dateien.

### Kritischer Pfad

```
Phase 1 (1-2 Tage) → Phase 2 (2-3 Tage) → Phase 3 (2-3 Tage) → Phase 4 (2-3 Tage) → Phase 7 (2-3 Tage)
                                                                                        ↑
                                            Phase 5 (1-2 Tage) ─────────────────────────┘
                                            Phase 6 (1-2 Tage) ─────────────────────────┘
```

**Minimaler MVP (nur BESS-Flex):** Phase 1 + 2 (ohne Wärmelast) + 3 + 4 + 7 = ~9-12 Tage
**Vollständiger MVP (alle 3 Flex-Typen):** + Phase 2 (Wärmelast) + 5 + 6 = ~13-18 Tage

### Empfohlene Reihenfolge für die Implementierung mit Claude Code

1. **Session A:** Phase 1 komplett + Phase 2.1 (PVGIS-Temperatur) + Phase 2.2 (SLP-Skeleton)
2. **Session B:** Phase 2.2 (SLP fertig) + Phase 2.3 (Wärmelast) + Phase 3.1 (Welt A)
3. **Session C:** Phase 3.2 (Portfolio-LP BESS) + Phase 4.1 (Engine)
4. **Session D:** Phase 4.2 (Enumeration + Systemwert) + Phase 4.3 (Grenznutzen)
5. **Session E:** Phase 5 (WP) + Phase 6 (EV/V2G) parallel
6. **Session F:** Phase 7 (Output CSV + HTML)

Jede Session endet mit lauffähigen Tests (`pytest`). Kein Code ohne Tests.

---

## 13. Risiken und Einschränkungen

- **Kupferplatte**: Netzrestriktionen ignoriert. Akzeptiert für strategische Planung.
- **SLP-Qualität**: Gut bei 8.500 Haushaltskunden. Fernwärme-WP hat eigene Messdaten.
- **Perfekte Voraussicht**: Überschätzt Spot-Arbitrage um ~20%. Adressiert über `perfect_foresight_discount`.
  Rolling-Horizon als Post-MVP-Feature vorgemerkt.
- **Fehlende Regelenergie**: Unterschätzt BESS-Systemwert. Pragmatische Arbeitshypothese: Überschätzung
  (perfekte Voraussicht) und Unterschätzung (fehlende Regelenergie) heben sich für BESS
  größenordnungsmäßig auf. Für WP/Wallbox gilt das nicht – dort bleibt die Überschätzung.

---

## 14. Post-MVP Features (Prioritätenliste)

1. **Monte Carlo**: Stochastische Bewertung auf ausgewählte Konfigurationen. Noise-Faktoren:
   PV-Yield, Preisszenario, Flex-Verfügbarkeit (pro Flex-Typ, nicht nur BESS).
2. **CAPEX/OPEX + Finanzlogik**: Grenzkosten pro Flex, Netto-Grenzwert-Kurve (Grenznutzen - Grenzkosten),
   IRR pro Flex-Konfiguration.
3. **Kombinatorik**: Mehrere Flex-Typen gleichzeitig optimieren (BESS + WP + EV).
4. **Rolling-Horizon**: Imperfekte Voraussicht (6h perfekt, Rest verrauscht).
5. **Weitere Erzeuger**: Onshore-Wind-Profile in der Erzeugungsliste.
6. **Weitere Lastgruppen**: Gewerbekunden (SLP G0), Großverbraucher.
7. **Optimierung**: Automatische Bestimmung der optimalen Zubau-Rate pro Flex-Typ.

---

## 15. Verbleibende offene Fragen

1. **BDEW-Excel-Datei**: Welche Version der BDEW-SLP-Daten soll verwendet werden? Hast du die Datei
   bereits, oder muss sie beschafft werden? Gibt es eine bestimmte Quelle/URL? --> Ja, siehe oben Punkt 6.

2. **Sommer-/Winterzeit im SLP**: Sollen die SLP-Profile in UTC oder in MEZ/MESZ aufbereitet werden?
   Die Preisszenarien sind vermutlich in UTC (da Day-Ahead Spotmarkt) – konsistente Zeitzone ist kritisch. --> UTC

3. **PVGIS-Temperatur für SLP-Dynamisierung**: Die BDEW-Dynamisierungsfunktion braucht eine
   Tages-Durchschnittstemperatur. Soll diese aus denselben PVGIS-Daten kommen, die auch für die
   PV-Erzeugung und Gradtagszahl verwendet werden? Das wäre konsistent, aber PVGIS liefert T2m
   (2m-Temperatur), nicht die für SLP übliche Temperatur der nächstgelegenen Wetterstation. --> Ja, zur konsistenz
   sollen die gleichen PVGIS Daten verwendet werden.

4. **EV-Flottengröße und Zubau**: Im JSON ist `annual_addition_kw` als Ladeleistungszubau definiert.
   Soll daraus die Flottengröße abgeleitet werden (z.B. 1 Wallbox = 11 kW → 100 kW/a = ~9 Wallboxen/a)?
   Oder soll die Flottengröße separat definiert werden, um den `daily_energy_demand_kwh` korrekt zu
   skalieren? --> es soll separat definiert werden. Diese Flexibilitätskategorie benötigt also 'mean_kw_per_unit' und 'annual_additional_units'