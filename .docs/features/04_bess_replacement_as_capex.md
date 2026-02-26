# Feature 04: BESS Replacement als CAPEX mit Neustart der Abschreibung

## Priorität: Hoch
## Aufwand: Mittel (3-4h)

## Beschreibung

Aktuell wird die BESS-Replacement-Kosten als OPEX im Replacement-Jahr verbucht.
Das soll geändert werden:

1. **Replacement als CAPEX**: Die Kosten werden als Investition (CAPEX) behandelt
2. **Neustart der Abschreibung (AfA)**: Ab dem Replacement-Jahr beginnt eine neue
   lineare Abschreibung über `afa_years_bess` Jahre
3. **Cashflow-Auswirkung**: CAPEX-Auszahlung im Replacement-Jahr, neue AfA-Abschreibung
   in den Folgejahren (beeinflusst Steuerlast)

## Ist-Zustand

### cashflow.py (Zeile 130-132)
```python
# Add BESS replacement cost in the replacement year
if replacement_year is not None and y == replacement_year:
    opex += replacement_cost
```
Replacement wird als einmaliger OPEX-Aufschlag verbucht.

### tax.py (calculate_tax_for_year)
```python
depr_bess = calculate_annual_depreciation(capex_bess, afa_years_bess, project_year)
```
Nur eine Abschreibung auf den initialen BESS-CAPEX.

## Soll-Zustand

### 1. Replacement als CAPEX-Auszahlung

Im Cashflow soll die Replacement-Kosten als Investitionsauszahlung behandelt werden:
```python
# Replacement-Jahr:
# - Equity CF: -replacement_cost (Auszahlung)
# - Project CF: -replacement_cost (Auszahlung)
# - NICHT als OPEX verbucht
```

Die Finanzierung des Replacements muss geklärt werden:
- 100% Eigenkapital-finanziert (kein zusätzliches Fremdkapital)
- Die Replacement-Kosten werden direkt vom Equity-Cashflow abgezogen

### 2. Neue Abschreibung ab Replacement-Jahr

Ab dem Replacement-Jahr startet eine zweite AfA-Linie:
```
AfA_BESS_original[y] = capex_bess / afa_years_bess  (für y = 1..afa_years_bess)
AfA_BESS_replacement[y] = replacement_cost / afa_years_bess  (für y = replacement_year..replacement_year+afa_years_bess-1)
AfA_BESS_total[y] = AfA_BESS_original[y] + AfA_BESS_replacement[y]
```

**Beispiel:** BESS CAPEX = 1.000.000 EUR, Replacement = 500.000 EUR, AfA = 10 Jahre, Replacement in Jahr 12.
- Jahre 1-10: AfA = 100.000 EUR/Jahr (Original)
- Jahre 11-12: AfA = 0 EUR (Original abgeschrieben, Replacement noch nicht)
- Jahr 12: Replacement CAPEX-Auszahlung = 500.000 EUR
- Jahre 12-21: AfA = 50.000 EUR/Jahr (Replacement)

### 3. Änderungen in `tax.py`

Die Funktion `calculate_tax_for_year` braucht zusätzliche Parameter:
```python
def calculate_tax_for_year(
    ...
    capex_bess_replacement: float = 0.0,
    replacement_year: int | None = None,
    afa_years_bess_replacement: int | None = None,  # default = afa_years_bess
    ...
) -> TaxResult:
```

Neue Abschreibungslogik:
```python
# Original BESS AfA
depr_bess_original = calculate_annual_depreciation(capex_bess, afa_years_bess, project_year)

# Replacement BESS AfA (startet ab replacement_year)
depr_bess_replacement = 0.0
if replacement_year is not None and capex_bess_replacement > 0.0:
    years_since_replacement = project_year - replacement_year + 1
    if years_since_replacement >= 1:
        afa_yrs = afa_years_bess_replacement or afa_years_bess
        depr_bess_replacement = calculate_annual_depreciation(
            capex_bess_replacement, afa_yrs, years_since_replacement
        )

depr_bess = depr_bess_original + depr_bess_replacement
```

### 4. Änderungen in `cashflow.py`

```python
def build_cashflow_projection(
    ...
    replacement_cost: float = 0.0,
    replacement_year: int | None = None,
):
    # ...
    for y in range(1, lifetime_years + 1):
        opex = inflate_value(base_opex, inflation_rate, y)
        # ENTFERNT: opex += replacement_cost im Replacement-Jahr

        # Replacement als CAPEX-Auszahlung
        replacement_capex_this_year = 0.0
        if replacement_year is not None and y == replacement_year:
            replacement_capex_this_year = replacement_cost

        # Tax mit Replacement-AfA
        tax_result = calculate_tax_for_year(
            ...
            capex_bess_replacement=replacement_cost if replacement_year else 0.0,
            replacement_year=replacement_year,
        )

        # Project CF: Revenue - OPEX - Tax - Replacement-CAPEX
        proj_cf = revenue - opex - tax_result.total_tax - replacement_capex_this_year

        # Equity CF: Revenue - OPEX - Debt Service - Tax - Replacement-CAPEX
        eq_cf = revenue - opex - debt_svc - tax_result.total_tax - replacement_capex_this_year
```

### 5. Year-0-Analogie

Das Replacement im Jahr X verhält sich wie eine Mini-Version von Year 0:
- CAPEX-Auszahlung (reduziert Cash Flow)
- Neue Abschreibung startet
- Aber: KEIN neues Fremdkapital, KEIN neuer Debt Service

## Betroffene Dateien

| Datei | Änderung |
|-------|----------|
| `finance/tax.py` | Replacement-AfA: zweite Abschreibungslinie |
| `finance/cashflow.py` | Replacement als CAPEX statt OPEX, Replacement-AfA Parameter |
| `optimization/grid_search.py` | Replacement-Kosten korrekt an Cashflow durchreichen |
| `optimization/monte_carlo.py` | Replacement-Kosten korrekt an Cashflow durchreichen |
| `main.py` | Replacement-CAPEX an Cashflow durchreichen |
| `output/csv_writer.py` | Optional: Replacement-CAPEX als separate Spalte in Cashflows-CSV |

## Tests

- Replacement-Jahr: Equity CF sinkt um Replacement-Kosten
- AfA Original: Endet nach `afa_years_bess` Jahren
- AfA Replacement: Startet im Replacement-Jahr, Dauer `afa_years_bess`
- Überlappung: Falls Original-AfA noch läuft, addieren sich beide
- Keine Replacement: Verhalten identisch zum aktuellen Zustand
- Steuereffekt: Replacement-AfA reduziert taxable income
- Verlustvortrag: Negative taxable income durch Replacement korrekt verarbeitet
- Project IRR: Replacement-CAPEX als Auszahlung (nicht als OPEX)
