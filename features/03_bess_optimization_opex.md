# Feature 03: BESS Optimierungsdienstleistung als OPEX

## Priorität: Mittel
## Aufwand: Klein-Mittel (2-3h)

## Beschreibung

Neuer OPEX-Kostenpunkt: Die Optimierungsdienstleistung für den BESS wird als Prozentsatz
des BESS-Netzerlöses berechnet. Der Prozentsatz ist User-Input.

**Definition BESS-Ertrag:** Einspeisung des BESS in das Netz × jeweiliger Spot-Preis.
Dies gilt für Grünstrom UND Graustrom.

**Formel:**
```
BESS_revenue_year = Σ_t (discharge_green[t] × RTE × spot[t]) + Σ_t (discharge_grey[t] × RTE × spot[t])
optimization_opex_year = BESS_revenue_year × optimization_fee_pct / 100
```

**Hinweis:** Hier wird der Spot-Preis verwendet (nicht der effective price mit Floor/GoO),
da die Optimierungsdienstleistung auf den tatsächlichen Markterlös des Speichers abzielt.

## Ist-Zustand

OPEX wird aktuell als fester Jahresbetrag (`base_opex`) mit Inflation berechnet:
```python
opex = inflate_value(base_opex, inflation_rate, y)
```

Es gibt keine erlösabhängige OPEX-Komponente.

## Soll-Zustand

### 1. JSON-Schema-Erweiterung

Im `bess.costs`-Block:
```json
"bess": {
    "costs": {
        "capex": { ... },
        "opex": { ... },
        "optimization_fee_pct": 5.0,
        "replacement": { ... }
    }
}
```

Neues Feld: `optimization_fee_pct` (float, default 0.0, optional)

### 2. Berechnung in der Dispatch Engine

Der BESS-Spot-Ertrag muss pro Jahr tracked werden. Die Engine liefert bereits:
- `revenue_bess_green`: Grüne BESS-Entladung × RTE × eff_price (enthält Floor + GoO)
- `revenue_bess_grey`: Graue BESS-Entladung × RTE × spot

Für die Optimierungs-Fee brauchen wir den **Spot-basierten** BESS-Ertrag:
```
bess_spot_revenue = Σ (discharge_green × RTE × spot) + Σ (discharge_grey × RTE × spot)
```

**Hinweis:** Der grüne BESS-Ertrag auf Spot-Basis unterscheidet sich vom `revenue_bess_green`,
das den Floor/GoO enthält.

### 3. Neue Felder in `AnnualResult` (engine.py)

```python
@dataclass
class AnnualResult:
    # ... bestehende Felder ...
    bess_spot_revenue: float
    """BESS grid revenue at spot price (green + grey), for optimization fee calculation."""
```

Im Day-Loop:
```python
day_bess_spot_rev = float(
    np.sum(result["discharge_green"] * config.bess_rte * spot_day)
    + np.sum(result["discharge_grey"] * config.bess_rte * spot_day)
)
year_bess_spot_revenue += day_bess_spot_rev
```

### 4. Cashflow-Berechnung

Die Optimierungs-Fee wird als zusätzlicher OPEX in der Cashflow-Berechnung berücksichtigt.

In `cashflow.py` (`build_cashflow_projection`):
```python
def build_cashflow_projection(
    ...
    optimization_fee_pct: float = 0.0,
    annual_bess_spot_revenues: list[float] | None = None,
):
    # ...
    for y in range(1, lifetime_years + 1):
        opex = inflate_value(base_opex, inflation_rate, y)

        # Optimization fee (not inflated - already based on current-year revenue)
        if optimization_fee_pct > 0.0 and annual_bess_spot_revenues:
            opex += annual_bess_spot_revenues[y - 1] * optimization_fee_pct / 100.0
```

**Hinweis:** Die Optimierungs-Fee wird NICHT inflationsbereinigt, da sie auf dem aktuellen
Jahreserlös basiert (der bereits die Preissteigerung enthält).

### 5. Durchreichung

| Komponente | Änderung |
|-----------|----------|
| `main.py` | `optimization_fee_pct` aus JSON lesen, an Cashflow durchreichen |
| `GridSearchConfig` | Neues Feld `optimization_fee_pct` |
| `grid_search.py` | Fee in Cashflow-Berechnung einbinden |
| `monte_carlo.py` | Fee in Cashflow-Berechnung einbinden |

## Betroffene Dateien

| Datei | Änderung |
|-------|----------|
| `config/schema.py` | `optimization_fee_pct` im BESS costs block |
| `config/defaults.py` | `DEFAULT_OPTIMIZATION_FEE_PCT = 0.0` |
| `dispatch/engine.py` | `bess_spot_revenue` in AnnualResult tracken |
| `finance/cashflow.py` | Optimization fee als zusätzlichen OPEX |
| `main.py` | Parameter lesen + durchreichen |
| `optimization/grid_search.py` | Fee durchreichen |
| `optimization/monte_carlo.py` | Fee durchreichen |

## Tests

- Fee = 0%: Kein Effekt auf OPEX
- Fee = 5%, BESS Spot Revenue = 100.000 EUR: Optimization OPEX = 5.000 EUR
- Fee gilt für Green + Grey Revenue (Spot-basiert)
- Fee wird NICHT inflationsbereinigt
- PV-only Szenario (kein BESS): Fee = 0 EUR
- Fee in Grid Search korrekt berücksichtigt (beeinflusst Equity IRR)
