# Feature 05: Monte-Carlo-Framework Erweiterungen

## Priorität: Mittel
## Aufwand: Mittel (3-4h)

## Beschreibung

Drei zusammenhängende Änderungen am MC-Framework:

### A) Separate CAPEX/OPEX-Sigmas für PV und BESS

Aktuell gibt es je einen `sigma_capex` und `sigma_opex`, die auf den **gesamten** CAPEX/OPEX
angewendet werden. Da PV und BESS sehr unterschiedliche Kostenrisiken haben, sollen
die Sigmas getrennt werden.

### B) BESS-Verfügbarkeit: Minimum = User-Input, Maximum = 100%

Aktuell: `N(mu_avail, sigma_avail)`, geclippt auf [0, 1].
Begründung für Änderung: Bei Verfügbarkeit < 97% (oder User-definiertem Minimum) greift
eine Herstellergarantie/Entschädigung. Die MC-Simulation soll also nur zwischen
dem garantierten Minimum und 100% variieren.

### C) PV-Verfügbarkeit (ersetzt PV-Yield-Faktor)

Der PV-Yield-Faktor (`sigma_pv_yield`) wird durch einen PV-Verfügbarkeitsparameter ersetzt,
der dieselbe Offline-Logik wie die BESS-Verfügbarkeit verwendet. (Hinweis: Abhängigkeit zu
Feature 06 - dort wird der PV-Yield-Faktor ohnehin obsolet, da jedes Preisszenario ein
festes Wetterjahr hat.)

## Ist-Zustand

### MCParams (monte_carlo.py)
```python
@dataclass
class MCParams:
    sigma_pv_yield: float = 0.05
    sigma_capex: float = 0.08
    sigma_opex: float = 0.05
    mu_bess_availability: float = 0.97
    sigma_bess_availability: float = 0.02
```

### Noise Sampling (_run_mc_iteration)
```python
pv_yield_factor = rng.normal(1.0, mc.sigma_pv_yield)
capex_factor = rng.normal(1.0, mc.sigma_capex)
opex_factor = rng.normal(1.0, mc.sigma_opex)
bess_availability_factor = np.clip(rng.normal(mu_avail, sigma_avail), 0, 1)
```

### CAPEX/OPEX-Skalierung
```python
capex_total = optimal.capex_total * capex_factor
capex_pv = optimal.capex_pv * capex_factor
capex_bess = optimal.capex_bess * capex_factor
opex_base = optimal.opex_base * opex_factor
```

## Soll-Zustand

### A) Separate CAPEX/OPEX-Sigmas

#### JSON-Schema
```json
"monte_carlo": {
    "enabled": true,
    "iterations": 1000,
    "sigma_capex_pv_pct": 5.0,
    "sigma_capex_bess_pct": 10.0,
    "sigma_opex_pv_pct": 3.0,
    "sigma_opex_bess_pct": 8.0,
    "sigma_pv_availability_pct": 2.0,
    "mu_bess_availability_pct": 97.0,
    "sigma_bess_availability_pct": 2.0
}
```

**Abwärtskompatibilität:** 
Keine berücksichtigung der abwärtskompatibilität. Die Änderungen können ohne Rücksichtnahme auf bestehende Inputdateien 
getätigt werden - es gibt keine.

#### MCParams-Erweiterung
```python
@dataclass
class MCParams:
    # Ersetzen:
    sigma_capex_pv: float = 0.05       # war: sigma_capex
    sigma_capex_bess: float = 0.10     # war: sigma_capex
    sigma_opex_pv: float = 0.03        # war: sigma_opex
    sigma_opex_bess: float = 0.08      # war: sigma_opex
    sigma_pv_availability: float = 0.02  # war: sigma_pv_yield
    mu_bess_availability: float = 0.97
    sigma_bess_availability: float = 0.02
```

#### Noise-Sampling
```python
capex_factor_pv = rng.normal(1.0, mc.sigma_capex_pv)
capex_factor_bess = rng.normal(1.0, mc.sigma_capex_bess)
opex_factor_pv = rng.normal(1.0, mc.sigma_opex_pv)
opex_factor_bess = rng.normal(1.0, mc.sigma_opex_bess)
```

#### CAPEX/OPEX-Skalierung (getrennt)
```python
capex_pv = optimal.capex_pv * capex_factor_pv
capex_bess = optimal.capex_bess * capex_factor_bess
capex_grid = optimal.capex_grid  # Grid-CAPEX wird nicht variiert
capex_total = capex_pv + capex_bess + capex_grid + optimal.capex_other

# OPEX getrennt skalieren (braucht Zugriff auf Einzel-OPEX)
opex_pv = optimal.opex_pv * opex_factor_pv
opex_bess = optimal.opex_bess * opex_factor_bess
opex_grid = optimal.opex_grid  # nicht variiert
opex_base = opex_pv + opex_bess + opex_grid + optimal.opex_other
```

**Voraussetzung:** `GridPointResult` muss die Einzel-OPEX speichern (aktuell nur `opex_base` = Summe).
Neue Felder in `GridPointResult`:
```python
opex_pv: float
opex_bess: float
opex_grid: float
opex_other: float
```

### B) BESS-Verfügbarkeit Clipping

Aktuell: `np.clip(raw_avail, BESS_NOISE_CLIP_MIN, BESS_NOISE_CLIP_MAX)` mit `BESS_NOISE_CLIP_MIN=0, MAX=1`

Neu:
```python
min_availability = mc.mu_bess_availability  # z.B. 0.97 = garantiertes Minimum
bess_availability_factor = np.clip(
    rng.normal(mc.mu_bess_availability, mc.sigma_bess_availability),
    min_availability,  # Minimum = User-Input (Herstellergarantie)
    1.0                # Maximum = 100%
)
```

**Verteilung:** Da wir auf [min, 1.0] clippen und der Mean am unteren Rand liegt,
wird die Verteilung stark rechtsschief. Alternative: Uniform-Verteilung zwischen
min und 1.0. Die Normalverteilung mit Clipping ist aber konsistenter mit dem bestehenden
Framework.

**Anmerkung:** Der `mu_bess_availability` wird zum Minimum. Das Sampling-Zentrum sollte
eventuell höher liegen (z.B. Mitte zwischen min und 100%). Vorschlag:
```python
mu_sample = (min_availability + 1.0) / 2.0  # z.B. 0.985 bei min=0.97
bess_availability_factor = np.clip(
    rng.normal(mu_sample, mc.sigma_bess_availability),
    min_availability,
    1.0
)
```

### C) PV-Verfügbarkeit

Der PV-Yield-Faktor wird durch eine PV-Verfügbarkeit ersetzt, die dieselbe
Offline-Day-Logik wie BESS verwendet:

```python
# PV Availability Sampling
pv_availability_factor = np.clip(
    rng.normal(mu_pv_availability, mc.sigma_pv_availability),
    mu_pv_availability,  # Minimum
    1.0
)
n_pv_offline_days = round((1.0 - pv_availability_factor) * DAYS_PER_YEAR)
```

An PV-Offline-Tagen:
- PV-Produktion = 0 (alle 24 Stunden)
- BESS kann trotzdem operieren (Grey Mode: Grid-Charging möglich)

**Implementierung in der Engine:**
```python
for day in range(DAYS_PER_YEAR):
    pv_day = pv_year[h_start:h_end]
    if day in pv_offline_days:
        pv_day = np.zeros(HOURS_PER_DAY)
    # ... weiter wie bisher
```

Neue Parameter:
- `pv_offline_days_yearly: list[set[int]]` analog zu `offline_days_yearly` (BESS)

## MCIterationResult-Erweiterung

```python
@dataclass
class MCIterationResult:
    # Ersetzen / Erweitern:
    capex_factor_pv: float       # war: capex_factor
    capex_factor_bess: float     # neu
    opex_factor_pv: float        # war: opex_factor
    opex_factor_bess: float      # neu
    pv_availability_factor: float  # war: pv_yield_factor
    bess_availability_factor: float
```

## Betroffene Dateien

| Datei | Änderung |
|-------|----------|
| `config/schema.py` | Neue MC-Parameter im Schema |
| `config/defaults.py` | Neue Defaults für getrennte Sigmas |
| `optimization/monte_carlo.py` | MCParams, Sampling, CAPEX/OPEX-Skalierung, PV-Offline |
| `optimization/grid_search.py` | GridPointResult: Einzel-OPEX-Felder |
| `dispatch/engine.py` | PV-Offline-Days Parameter |
| `main.py` | Neue MC-Parameter aus JSON lesen |
| `output/csv_writer.py` | MC-CSV: Neue Spalten für getrennte Faktoren |

## Tests

### A) Separate Sigmas
- sigma_capex_pv = 0, sigma_capex_bess > 0: Nur BESS-CAPEX variiert
- sigma_opex_pv = 0, sigma_opex_bess > 0: Nur BESS-OPEX variiert
- Abwärtskompatibilität: Alte Felder setzen beide gleich

### B) BESS-Verfügbarkeit
- Alle Samples >= mu_bess_availability (z.B. >= 0.97)
- Alle Samples <= 1.0
- mu_bess_availability = 1.0: Keine Offline-Tage

### C) PV-Verfügbarkeit
- PV-Offline-Tag: PV-Produktion = 0
- PV-Offline + BESS Online: BESS kann aus Grid laden (Grey Mode)
- sigma_pv_availability = 0: Keine PV-Offline-Tage
