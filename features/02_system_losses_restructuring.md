# Feature 02: Systemverluste am Netzanschlusspunkt

## Priorität: Hoch
## Aufwand: Mittel (2-4h)

## Beschreibung

Aktuell werden Systemverluste (`system_loss_pct`) beim PVGIS-Download als PV-Verlust berücksichtigt.
Die Ertragszeitreihe enthält also bereits reduzierte Werte.

**Neues Modell:** Die Verluste werden am Netzanschlusspunkt modelliert, nicht im PV-Asset:

1. **PVGIS-Download mit 0% Verlusten** - Die Ertragszeitreihe repräsentiert die volle PV-Produktion
2. **PV-Systemverluste bei grüner Netzeinspeisung** - Nur die Energie, die tatsächlich ins Netz fließt (PV-Export + grüne BESS-Entladung), wird um die PV-Systemverluste reduziert
3. **BESS Round-Trip-Verluste bei grauer Rückeinspeisung** - Die RTE-Verluste beim Graustrom werden bei der Netzeinspeisung berücksichtigt (das ist schon der Fall: `discharge_grey × RTE`)

## Ist-Zustand

### PVGIS-Download (pvgis_client.py)
```python
client.fetch_hourly_production(
    system_loss_pct=system_loss_pct,  # z.B. 14.0%
    ...
)
```
Die Zeitreihe kommt bereits mit Verlusten zurück.

### Dispatch Engine (engine.py, Zeile 539-545)
```python
# Revenue-Berechnung
day_rev_pv = np.sum(result["export_pv"] * eff_day)
day_rev_green = np.sum(result["discharge_green"] * config.bess_rte * eff_day)
day_rev_grey = np.sum(result["discharge_grey"] * config.bess_rte * spot_day)
```
Aktuell: `export_pv` ist bereits verlustbehaftet (durch PVGIS). RTE wird nur auf Discharge angewendet.

## Soll-Zustand

### 1. PVGIS-Download: `system_loss_pct = 0`

In `main.py` beim PVGIS-Aufruf:
```python
yearly_pvgis = client.fetch_hourly_production(
    system_loss_pct=0.0,  # Keine Verluste im Download!
    ...
)
```

Der `system_loss_pct`-Parameter wird weiterhin aus dem JSON gelesen, aber NICHT an PVGIS übergeben.
Stattdessen wird er als Netz-Verlustfaktor gespeichert.

### 2. Verlustfaktor berechnen

```python
grid_loss_factor = 1.0 - system_loss_pct / 100.0  # z.B. 0.86 bei 14%
```

### 3. Engine: Verluste bei Netzeinspeisung anwenden

Die Verluste müssen an zwei Stellen greifen:

**a) Im LP-Optimizer (`optimizer.py`):**

Die Grid-Connection-Constraint muss den Verlustfaktor berücksichtigen:
```
export_pv[t] × grid_loss_factor + (discharge_green[t] + discharge_grey[t]) × RTE ≤ P_grid_max
```

**Hinweis:** Der Verlustfaktor muss auch in der Zielfunktion berücksichtigt werden,
da sonst die Optimierung falsche Revenue-Erwartungen hat:
```
revenue_pv[t] = export_pv[t] × grid_loss_factor × eff_price[t]
revenue_green[t] = discharge_green[t] × RTE × grid_loss_factor × eff_price[t]
revenue_grey[t] = discharge_grey[t] × RTE × spot_price[t]  # RTE bereits enthalten, KEIN grid_loss_factor für Graustrom
```

**Wichtig:** Graustrom-Revenue bekommt KEINEN PV-Systemverlust! Die PV-Systemverluste
betreffen nur "grüne" Energie (PV-Export und grüne BESS-Entladung).
Die BESS RTE-Verluste beim Graustrom sind bereits modelliert (`discharge_grey × RTE`).

**b) In der Revenue-Berechnung (`engine.py`):**
```python
day_rev_pv = np.sum(result["export_pv"] * grid_loss_factor * eff_day)
day_rev_green = np.sum(result["discharge_green"] * config.bess_rte * grid_loss_factor * eff_day)
day_rev_grey = np.sum(result["discharge_grey"] * config.bess_rte * spot_day)  # unverändert
```

### 4. JSON-Schema-Anpassung

Der `system_loss_pct`-Parameter bleibt im gleichen JSON-Block, wird aber semantisch uminterpretiert.
Optional: Umbenennung in `grid_loss_pct` (Breaking Change, evtl. mit Migration/Alias).

**Empfehlung:** Parameter im JSON-Block von `pv.performance` nach `grid_connection` verschieben:
```json
"grid_connection": {
    "max_export_kw": 4000,
    "system_loss_pct": 14.0,
    "costs": { ... }
}
```

## Parameter-Durchreichung

Der `grid_loss_factor` muss an folgende Stellen durchgereicht werden:

| Komponente | Neuer Parameter |
|-----------|----------------|
| `DispatchEngineConfig` | `grid_loss_factor: float` |
| `optimize_day()` | `grid_loss_factor: float` (default 1.0) |
| `dispatch_offline_day()` | `grid_loss_factor: float` (default 1.0) |
| `GridSearchConfig` | `grid_loss_factor: float` |
| `main.py` | Berechnung: `1.0 - system_loss_pct / 100.0` |

## Betroffene Dateien

| Datei | Änderung |
|-------|----------|
| `main.py` | PVGIS mit 0% loss, grid_loss_factor berechnen + durchreichen |
| `config/schema.py` | Optional: system_loss_pct in grid_connection verschieben |
| `dispatch/optimizer.py` | grid_loss_factor in Zielfunktion + Constraints |
| `dispatch/engine.py` | grid_loss_factor in Revenue-Berechnung, neues Config-Feld |
| `optimization/grid_search.py` | grid_loss_factor in GridSearchConfig |
| `optimization/monte_carlo.py` | grid_loss_factor durchreichen |

## Tests

- PV-Export mit grid_loss_factor = 0.86: Revenue sinkt um 14%
- Grüne BESS-Entladung: Revenue = discharge × RTE × grid_loss_factor × price
- Graue BESS-Entladung: Revenue = discharge × RTE × spot (KEIN grid_loss_factor)
- Grid-Constraint: export × grid_loss_factor + discharge × RTE ≤ P_grid_max
- grid_loss_factor = 1.0 (0% Verluste): Verhalten identisch zum aktuellen Zustand
- PVGIS-Cache: Neue Downloads mit 0% loss erzeugen neuen Cache-Key
