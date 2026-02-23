# Feature 01: PPA Collar Bug Fix

## Priorität: Hoch (Blocker)
## Aufwand: Klein (< 1h)

## Problem

Im PPA-Collar-Modus wird aktuell nur der Floor-Preis gesetzt, aber nicht der Cap-Preis.
Die `_build_fixed_prices_yearly()`-Funktion in `main.py` hat keinen `elif`-Zweig für `PPA_TYPE_COLLAR`.
Dadurch wird der Collar als reiner Floor behandelt - der Cap fehlt vollständig.

## Ist-Zustand (main.py, Zeile 190-218)

```python
def _build_fixed_prices_yearly(scenario, inflation_rate):
    # ...
    if marketing_type == "eeg":
        # EEG Floor - funktioniert
    elif ppa_type == PPA_TYPE_FLOOR:
        # PPA Floor - funktioniert
    elif ppa_type == PPA_TYPE_PAY_AS_PRODUCED:
        # Pay-as-produced - funktioniert
    # FEHLT: elif ppa_type == PPA_TYPE_COLLAR
```

## Soll-Zustand

### Option A: Cap-Preis in `_build_fixed_prices_yearly` berücksichtigen

Der Collar-PPA hat `floor_price` UND `cap_price`. Die Revenue pro kWh ist:
```
revenue_per_kwh = clip(spot_price, floor_price, cap_price)
```

Da die aktuelle Engine nur einen `fixed_price` pro Jahr kennt (als Floor), muss die Cap-Logik
an einer anderen Stelle greifen. Zwei Ansätze:

1. **Floor über `fixed_prices_yearly`, Cap über neuen `cap_prices_yearly` Parameter** (bevorzugt)
2. Effective-Price-Berechnung komplett in der Engine umbauen

### Implementierung (Ansatz 1)

#### 1. `main.py`: Neuen `elif`-Zweig für Collar in `_build_fixed_prices_yearly`

```python
elif ppa_type == PPA_TYPE_COLLAR:
    ppa_cfg = ppa_config_from_dict(ppa_dict)
    if year <= ppa_cfg.duration_years:
        base = ppa_cfg.floor_price_eur_per_kwh or 0.0
        if ppa_cfg.inflation_enabled:
            price = inflate_value(base, inflation_rate, year)
        else:
            price = base
```

#### 2. `main.py`: Neue Funktion `_build_cap_prices_yearly`

```python
def _build_cap_prices_yearly(scenario, inflation_rate) -> list[float]:
    """Cap-Preis pro Jahr. 0.0 = kein Cap (unbegrenzt nach oben)."""
    # Nur bei PPA_TYPE_COLLAR relevant
    # Gibt cap_price_eur_per_kwh zurück, ggf. inflationsbereinigt
    # Nach PPA-Ablauf: 0.0 (kein Cap)
```

#### 3. `dispatch/engine.py`: Cap-Logik in Revenue-Berechnung

Aktuell (Zeile 532-537):
```python
if fixed_price > 0.0:
    eff_day = np.maximum(spot_day, fixed_price)
else:
    eff_day = spot_day
```

Soll:
```python
if fixed_price > 0.0:
    eff_day = np.maximum(spot_day, fixed_price)
else:
    eff_day = spot_day
if cap_price > 0.0:
    eff_day = np.minimum(eff_day, cap_price)
```

#### 4. `dispatch/optimizer.py`: Cap-Preis im LP berücksichtigen

Die effective_prices-Berechnung im Optimizer muss ebenfalls den Cap anwenden:
```python
effective_prices = np.maximum(spot_prices, fixed_price)
if cap_price > 0.0:
    effective_prices = np.minimum(effective_prices, cap_price)
```

#### 5. Durchreichen des `cap_prices_yearly`-Parameters

- `GridSearchConfig`: Neues Feld `cap_prices_yearly: list[float]`
- `run_simulation()`: Neuer Parameter `cap_prices_yearly`
- `optimize_day()`: Neuer Parameter `price_cap_eur_per_kwh`
- `MCParams`/`run_monte_carlo()`: Cap-Preise durchreichen

## Betroffene Dateien

| Datei | Änderung |
|-------|----------|
| `main.py` | `_build_fixed_prices_yearly`: Collar-Zweig + `_build_cap_prices_yearly` |
| `dispatch/engine.py` | `cap_prices_yearly`-Parameter, Cap-Logik in Revenue |
| `dispatch/optimizer.py` | `price_cap_eur_per_kwh`-Parameter, effective price clipping |
| `optimization/grid_search.py` | `GridSearchConfig`: `cap_prices_yearly` Feld |
| `optimization/monte_carlo.py` | Cap-Preise durchreichen |

## Tests

- Collar mit Spot < Floor: Revenue = Floor
- Collar mit Floor < Spot < Cap: Revenue = Spot
- Collar mit Spot > Cap: Revenue = Cap
- Collar nach Ablauf: Revenue = Spot (kein Floor, kein Cap)
- Collar mit Inflation auf Floor und Cap
