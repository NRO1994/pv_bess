# FIXES-Session2-Plan.md – Implementierungsplan

## Integration

### FIX-S2-01: Cashflow Benchmark-Test
**Status:** OFFEN
**Dateien:** `tests/test_integration_cashflow.py` (neu), `.docs/finance_benchmarking.md` (neu)

**Problem:** Es fehlt ein Integrationstest, der die vom Modell erzeugte `_cashflows.csv` mit einer manuell nachgebauten Referenzdatei (`integration_test_cashflows.csv`) vergleicht.

**Analyse:**
- Referenz-CSV: `.data/integration_test_inputs/finance/integration_test_cashflows.csv`
- Szenario-JSON: `.data/integration_test_inputs/finance/integration_test_cashflow.json`
- Preis-CSV: `.data/integration_test_inputs/finance/eeg_fixed_54_20y_22_5y.csv`
- Die Spalten der Referenz-CSV stimmen mit dem Modell-Output überein: `year;capex_eur;pv_production_mwh;bess_throughput_mwh;revenue_eur;opex_eur;debt_service_eur;depreciation_eur;gewerbesteuer_eur;koerperschaftsteuer_eur;solidaritaetszuschlag_eur;total_tax_eur;project_cf_eur;equity_cf_eur;cumulative_equity_cf_eur;dscr`
- **Wichtig:** Die Referenz-CSV verwendet deutsches Dezimalkomma (`,`), das Modell verwendet Dezimalpunkt (`.`). Das Parsing muss dies berücksichtigen.
- `cumulative_equity_cf_eur` und `dscr` enthalten `#NA` in der Referenz → diese Spalten beim Vergleich überspringen oder separat behandeln
- Toleranz: 1% prozentuale Abweichung ist akzeptabel
- **Kein Code wird geändert** – nur Analyse und Dokumentation der Differenzen in `.docs/finance_benchmarking.md`

**Empfohlene Änderung:**
1. Integrationstest erstellen (`@pytest.mark.integration`):
   - Modell mit dem Szenario-JSON laufen lassen (nur Grid-Search, PV-only mit scale=0%)
   - Erzeugte `_cashflows.csv` einlesen
   - Referenz-CSV einlesen (mit `decimal=","`)
   - Spaltenweise Vergleich mit 1% Toleranz (`np.allclose` mit `rtol=0.01`)
   - Output in `.data/test/integration_tests/` speichern
2. `.docs/finance_benchmarking.md` erstellen mit Analyse aller Abweichungen
3. Test mit `@pytest.mark.integration` taggen

**Abhängigkeiten:**
- Abhängig von FIX-S2-08 (Dezimalkomma) für korrektes Parsing der Referenz-CSV
- Abhängig von FIX-S2-07 (Output Directory) für korrekte Speicherung der Testergebnisse

---

### FIX-S2-02: Smoke Test (End-to-End)
**Status:** OFFEN
**Dateien:** `tests/test_integration_smoke.py` (neu), `.data/integration_test_inputs/smoke_test/smoke_test.json` (anpassen)

**Problem:** Es existieren Smoke-Test-Daten, aber kein End-to-End-Integrationstest.

**Analyse:**
- Smoke-Test-JSON: `.data/integration_test_inputs/smoke_test/smoke_test.json`
- Preis-CSV: `.data/integration_test_inputs/smoke_test/smoke_test_prices.csv`
- **Bekannte Probleme im Smoke-Test-JSON:**
  1. `commissioning_year` fehlt → Schema-Validierung schlägt fehl (seit FIX-08 aus Session 1 ist es `required`)
  2. Preis-CSV-Pfad ist `"data/smoke_test_prices.csv"` → Datei liegt aber direkt unter `smoke_test/smoke_test_prices.csv`
  3. `system_loss_pct` steht unter `grid_connection` statt unter `pv.performance` (das ist korrekt nach Feature-02, wo System-Verluste am Netzanschluss modelliert werden)
- JSON-Parameter: `lifetime_years: 3`, MC: 10 Iterationen, 3 Scales × 1 E/P = 3 Grid Points → schneller Test

**Empfohlene Änderung:**
1. `smoke_test.json` korrigieren:
   - `commissioning_year: 2025` hinzufügen
   - Preis-CSV-Pfad auf `"smoke_test_prices.csv"` ändern
2. Integrationstest erstellen (`@pytest.mark.integration`):
   - Modell mit Smoke-Test-JSON komplett durchlaufen (Grid Search + MC)
   - Prüfen: Alle erwarteten Output-CSVs werden erzeugt (summary, cashflows, grid_search, monte_carlo, dispatch_sample)
   - Prüfen: Grid-Search liefert ein Optimum
   - Prüfen: MC-Ergebnisse haben die erwartete Anzahl Iterationen (10)
   - Prüfen: Keine NaN/Inf-Werte in den Ergebnissen
   - Output in `.data/test/integration_tests/` speichern

**Abhängigkeiten:**
- Keine direkten Abhängigkeiten zu anderen Fixes
- Sollte als letzter Integrationstest laufen, um den End-to-End-Pfad zu verifizieren

---

### FIX-S2-03: BESS-Only Cases ermöglichen
**Status:** OFFEN
**Dateien:** `config/schema.py`, `main.py`, `optimization/grid_search.py`, `tests/test_integration_bess_only.py` (neu)

**Problem:** Das Modell unterstützt keine reinen BESS-Szenarien ohne PV. Die Eingabelogik erzwingt `peak_power_kwp > 0` und BESS-Sizing ist relativ zur PV-Leistung.

**Analyse:**
- `config/schema.py` Zeile 102: `"peak_power_kwp": {"type": "number", "exclusiveMinimum": 0}` → 0 nicht erlaubt
- `grid_search.py` Zeile 597-598: `bess_power_kw = config.pv_peak_kwp * scale_pct / 100.0` → BESS ist 0 wenn PV=0
- `main.py` Zeile 510: `scale_pct_list = [args.bess_power / pv_peak_kwp * 100.0]` → Division durch 0
- BESS-Only JSON (`.data/integration_test_inputs/bess_only/`) nutzt Workaround `peak_power_kwp: 1`
- Für BESS-Only muss eine alternative Sizing-Methode unterstützt werden: absolute kW/kWh-Angabe

**Empfohlene Änderung:**
1. **Schema** (`config/schema.py`):
   - `peak_power_kwp`: `exclusiveMinimum: 0` → `minimum: 0` (erlaube 0)
   - Optional: `pv`-Block als Ganzes optional machen
   - BESS: Neue optionale Felder `"absolute_power_kw"` und `"absolute_capacity_kwh"` neben dem `design_space`
2. **Grid Search** (`grid_search.py`):
   - Wenn `pv_peak_kwp == 0` und absolute BESS-Werte angegeben: verwende diese statt Ratio
   - Wenn `pv_peak_kwp == 0` und nur `scale_pct` vorhanden: alle Scales ergeben 0 → nur Baseline
3. **Main** (`main.py`):
   - Division-by-Zero-Guard bei `pv_peak_kwp = 0`
   - PVGIS-Fetch überspringen wenn `pv_peak_kwp == 0`
   - PV-Timeseries = `np.zeros(HOURS_PER_YEAR)` wenn keine PV
4. **Integrationstest** (`@pytest.mark.integration`):
   - BESS-Only-Szenario durchlaufen
   - Prüfen: PV-Produktion = 0, BESS-Revenue > 0 (Grey Mode) oder = 0 (Green Mode)
   - Prüfen: CAPEX enthält nur BESS + Grid

**Abhängigkeiten:**
- Unabhängig von anderen Fixes
- Hat großen Impact auf: `main.py`, `grid_search.py`, `schema.py`, `monte_carlo.py` (überall wo `pv_peak_kwp` verwendet wird)
- **Achtung:** In Green Mode ohne PV kann BESS nicht geladen werden → Revenue = 0. Nur Grey Mode ist sinnvoll für BESS-Only.

---

## Logik

### FIX-S2-04: OPEX eur_per_kw und eur_per_kwh
**Status:** BEREITS IMPLEMENTIERT
**Dateien:** `finance/costs.py`, `config/schema.py`

**Analyse:**
- `costs.py` Zeile 63-85: `calculate_asset_opex()` unterstützt bereits `fixed_eur`, `eur_per_kw`, `eur_per_kwh`, `pct_of_capex`
- `schema.py` Zeile 22-31: `_COST_COMPONENT` enthält alle vier Felder und wird für CAPEX und OPEX gemeinsam verwendet
- **Kein Fix erforderlich.** Code und Schema unterstützen dies bereits vollständig.

---

### FIX-S2-05: Loan Tenor in Debt Service
**Status:** BEREITS IMPLEMENTIERT
**Dateien:** `finance/debt.py`, `finance/cashflow.py`, `main.py`

**Analyse:**
- `debt.py` Zeile 54-104: `build_annuity_schedule()` akzeptiert `tenor_years` und berechnet die Annuität korrekt nur für diese Laufzeit
- `cashflow.py` Zeile 137: `get_debt_service()` gibt 0.0 zurück nach Ablauf des Tenors
- `main.py` Zeile 459: `loan_tenor_years` wird aus JSON gelesen und an `build_annuity_schedule()` durchgereicht
- **Kein Fix erforderlich.** Der Loan Tenor wird korrekt berücksichtigt.

---

## Kosmetik

### FIX-S2-06: CSV Timestamp/Separator/Decimal als User Input
**Status:** OFFEN
**Dateien:** `config/defaults.py`, `config/schema.py`, `output/csv_writer.py`, `output/formatting.py`, `main.py`

**Problem:** Timestamp-Spaltenname, Zeitformat, CSV-Separator und Dezimalzeichen sind nicht vom User konfigurierbar.

**Analyse:**
- `defaults.py`: Existieren bereits `CSV_DELIMITER = ";"` und `CSV_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"`, aber kein `CSV_DECIMAL_SEPARATOR` und kein `CSV_TIMESTAMP_COLUMN`
- `schema.py` Zeile 73-81: `_OUTPUT`-Block enthält nur `directory` und `export_dispatch_sample`
- `csv_writer.py` Zeile 384: Verwendet `CSV_DELIMITER` aus defaults, aber keine User-Überschreibung

**Empfohlene Änderung:**
1. **Defaults** (`defaults.py`):
   - `CSV_DECIMAL_SEPARATOR: str = ","` (Default: Komma, da deutsches Modell)
   - `CSV_TIMESTAMP_COLUMN: str = "timestamp"`
2. **Schema** (`schema.py` `_OUTPUT`):
   - Neue optionale Felder: `"csv_separator"`, `"csv_decimal"`, `"csv_timestamp_column"`, `"csv_timestamp_format"`
3. **Main** (`main.py`):
   - CSV-Einstellungen aus JSON lesen, Fallback auf Defaults
   - An CSV-Writer durchreichen
4. **CSV Writer** (`csv_writer.py`):
   - Alle Schreibfunktionen: Separator und Decimal als Parameter akzeptieren
5. **Formatting** (`formatting.py`):
   - `fmt_float()`, `fmt_currency()`, `fmt_pct()`: Dezimalzeichen als Parameter oder global konfigurierbar

**Abhängigkeiten:**
- FIX-S2-08 (Dezimalkomma) ist ein Subset dieses Fixes → zusammen implementieren
- Hat Impact auf alle Output-CSVs
- Tests müssen angepasst werden: `test_csv_writer*.py`, `test_formatting.py`

---

### FIX-S2-07: Output Directory aus JSON übernehmen
**Status:** OFFEN
**Dateien:** `main.py`

**Problem:** `scenario.output.directory` aus dem JSON wird komplett ignoriert. Es wird immer `DEFAULT_OUTPUT_DIR` oder `--output` CLI-Flag verwendet.

**Analyse:**
- `main.py` Zeile 448-451:
  ```python
  output_base = Path(args.output) if args.output else Path(DEFAULT_OUTPUT_DIR)
  ```
- JSON-Feld wird geladen und validiert (Schema erfordert `directory` im Output-Block), aber nie gelesen
- Die JSON-Dateien nutzen unterschiedliche Pfade: `".data/output/"`, `".data/test/integration_tests/"`

**Empfohlene Änderung:**
1. In `main.py` Zeile 448-451 ersetzen:
   ```python
   scenario_output_dir = scenario.raw.get("scenario", {}).get("output", {}).get("directory")
   if args.output:
       output_base = Path(args.output)
   elif scenario_output_dir:
       output_base = Path(scenario_output_dir)
   else:
       output_base = Path(DEFAULT_OUTPUT_DIR)
   ```
2. CLI `--output` überschreibt JSON, JSON überschreibt Default

**Abhängigkeiten:**
- Keine Abhängigkeiten zu anderen Fixes
- Kleiner, isolierter Fix mit minimalem Impact
- Tests: `main.py` Unit-Tests ggf. ergänzen

---

### FIX-S2-08: Dezimalkomma als Default
**Status:** OFFEN
**Dateien:** `config/defaults.py`, `output/formatting.py`, `output/csv_writer.py`

**Problem:** Der CSV-Writer gibt Dezimalpunkte aus. Für deutsche Nutzer soll Dezimalkomma der Default sein.

**Analyse:**
- `formatting.py` Zeile 39/62/93: Alle Format-Funktionen nutzen Python f-Strings → immer `.` als Dezimalzeichen
- Kein `CSV_DECIMAL_SEPARATOR` existiert in `defaults.py`
- Die Referenz-CSV für den Benchmark-Test verwendet bereits Dezimalkomma

**Empfohlene Änderung:**
1. `defaults.py`: `CSV_DECIMAL_SEPARATOR: str = ","` hinzufügen
2. `formatting.py`: Alle `fmt_*`-Funktionen erweitern:
   ```python
   def fmt_float(value: float, precision: int = FLOAT_PRECISION) -> str:
       result = f"{value:.{precision}f}"
       return result.replace(".", CSV_DECIMAL_SEPARATOR)
   ```
3. Alternative: In `csv_writer.py` `_write_dicts()` am Ende ein globales Replace durchführen

**Abhängigkeiten:**
- Wird von FIX-S2-06 (CSV User Input) umfasst – kann zusammen implementiert werden
- **Impact auf Tests:** Alle CSV-Writer-Tests und Formatting-Tests müssen angepasst werden
- **Impact auf FIX-S2-01:** Die Referenz-CSV nutzt Dezimalkomma → nach diesem Fix ist der Vergleich konsistent

---

### FIX-S2-09: Excel Lock Error Handling
**Status:** OFFEN
**Dateien:** `output/csv_writer.py`

**Problem:** Wenn eine CSV-Datei in Excel geöffnet ist, schlägt das Schreiben mit `PermissionError` fehl.

**Analyse:**
- `csv_writer.py` Zeile 374-386: `_write_dicts()` öffnet die Datei ohne Error-Handling
- Problem tritt primär unter Windows auf, kann aber auch unter Linux bei NFS-Locks auftreten
- Alle Schreibfunktionen (`write_summary_csv`, `write_cashflows_csv`, etc.) nutzen `_write_dicts()`

**Empfohlene Änderung:**
1. In `_write_dicts()` den `with path.open(...)` Block in try/except wrappen:
   ```python
   def _write_dicts(path: Path | str, rows: list[dict]) -> None:
       path = Path(path)
       path.parent.mkdir(parents=True, exist_ok=True)
       if not rows:
           path.write_text("", encoding="utf-8")
           return
       fieldnames = list(rows[0].keys())
       try:
           _write_csv(path, fieldnames, rows)
       except PermissionError:
           idx = 1
           while idx <= 10:  # Max 10 Versuche
               new_path = path.with_stem(f"{path.stem}_{idx}")
               try:
                   _write_csv(new_path, fieldnames, rows)
                   logger.warning("Datei %s gesperrt, gespeichert als %s", path, new_path)
                   break
               except PermissionError:
                   idx += 1
           else:
               raise
   ```
2. `_write_csv()` als Hilfsfunktion extrahieren

**Abhängigkeiten:**
- Keine Abhängigkeiten zu anderen Fixes
- Isolierter, defensiver Fix
- Test: Schwer zu testen unter Linux, ggf. mit Mock für `PermissionError`

---

### FIX-S2-10: Debt Service Split (Interest + Repayment)
**Status:** OFFEN
**Dateien:** `finance/cashflow.py`, `finance/debt.py`, `output/csv_writer.py`, `tests/test_cashflow.py`, `tests/test_csv_writer_cashflows.py`

**Problem:** Die `debt_service`-Spalte enthält nur den Gesamtwert. Gewünscht sind separate Spalten für Zinsanteil und Tilgungsanteil.

**Analyse:**
- `debt.py` `AnnuitySchedule` (Zeile 15-31): Enthält bereits `interest_payments: list[float]` und `principal_payments: list[float]`
- `cashflow.py` `AnnualCashflow` (Zeile 27-41): Nur `debt_service: float` als Gesamtwert
- `debt.py` `get_debt_service()` (Zeile 107-119): Gibt nur `schedule.annual_payment` zurück
- `csv_writer.py` Zeile 198-200: Schreibt nur `debt_service_eur`

**Empfohlene Änderung:**
1. **`finance/debt.py`**: Neue Funktion `get_debt_components(schedule, year) -> tuple[float, float, float]` (interest, principal, total)
2. **`finance/cashflow.py`** `AnnualCashflow`:
   - Neue Felder: `debt_interest: float`, `debt_repayment: float`
   - In `build_cashflow_projection()`: `debt_interest` und `debt_repayment` aus Schedule extrahieren
3. **`output/csv_writer.py`**:
   - Spalte `debt_service_eur` ersetzen durch `debt_interest_eur` und `debt_repayment_eur`
4. **Tests anpassen:**
   - `test_cashflow.py`: Prüfe `debt_interest` und `debt_repayment`
   - `test_csv_writer_cashflows.py`: `expected_cols` aktualisieren

**Abhängigkeiten:**
- Unabhängig von anderen Fixes
- **Impact:** Ändert das CSV-Ausgabeformat (breaking change für bestehende Nutzer)
- FIX-S2-01 (Benchmark-Test) muss dies berücksichtigen, falls die Referenz-CSV nur `debt_service_eur` hat

---

### FIX-S2-11: Grid Search Skip bei Einzel-Werten
**Status:** OFFEN
**Dateien:** `optimization/grid_search.py`, `main.py`

**Problem:** Die Grid Search läuft auch bei nur einem Wert je Array mit dem vollen Parallel-Overhead (ProcessPoolExecutor).

**Analyse:**
- `grid_search.py` Zeile 579-583: `scale = 0%` wird immer hinzugefügt → selbst bei `[40]` werden 2 Punkte evaluiert
- `grid_search.py` Zeile 628-696: Verwendet `ProcessPoolExecutor` für alle Fälle
- Bei einem einzigen Grid-Point ist der Overhead des Multiprocessing unnötig
- **Achtung:** Scale=0% (PV-only Baseline) ist konzeptionell wichtig als Vergleichsbasis

**Empfohlene Änderung:**
1. In `run_grid_search()`: Wenn genau 1 Kombination (nach Hinzufügen von scale=0%), direkt im Hauptprozess evaluieren statt über ProcessPoolExecutor
2. Alternativ: Wenn der User genau einen scale-Wert und einen E/P-Wert angibt, die automatische Hinzufügung von scale=0% optional machen (z.B. Flag `skip_baseline: true` im JSON)
3. Logging: "Grid search skipped – single configuration" im Info-Log

**Abhängigkeiten:**
- Keine Abhängigkeiten zu anderen Fixes
- Performanz-Optimierung, keine funktionalen Änderungen
- Relevant für FIX-S2-03 (BESS-Only), wo ggf. nur eine BESS-Konfiguration evaluiert wird

---

### FIX-S2-12: BESS Charging nur am ersten Tag des PPA/EEG-Jahres (Collar Bug)
**Status:** OFFEN
**Dateien:** `dispatch/engine.py`, `dispatch/optimizer.py`, `config/defaults.py`

**Problem:** In einem "Green PV+BESS"-Szenario mit Collar-PPA wird der BESS nur am ersten Tag jedes PPA-Jahres geladen/entladen. Erst nach Ablauf des PPA funktioniert der BESS an allen Tagen korrekt.

**Analyse:**
- Die Preiskonstruktion (`_build_fixed_prices_yearly`, `_build_cap_prices_yearly` in `main.py`) ist korrekt – Floor- und Cap-Preise werden für alle 8.760 Stunden pro Jahr gebaut.
- `_effective_green_price` in `optimizer.py` Zeile 219-226 ist mathematisch korrekt: `eff = min(max(spot, floor), cap)`.
- **Hauptursache:** Der BESS startet bei `soc_max_kwh * 0.50` (50% der oberen SoC-Grenze). Am ersten Tag wird entladen, weil der SoC über dem Optimum liegt. Danach bietet der Collar-Cap wenig Anreiz zum Nachladen, weil der Spread zwischen Lade- und Entladepreis (nach Cap-Begrenzung und RTE-Verlusten) marginal oder negativ ist.
- **Zusammenhang mit FIX-S2-17:** Wenn der Start-SoC auf `soc_min_kwh` gesetzt wird, beginnt der BESS leer und lädt über den Tag auf. Der Optimizer sieht dann über alle Tage einen Arbitrage-Anreiz (Laden zu Niedrigpreisen, Entladen zu Hochpreisen – auch innerhalb des Cap).
- **Zusätzlich prüfen:** Ob die effektiven Preise korrekt an den Optimizer übergeben werden. In `engine.py` Zeile 543-555 werden `price_fixed` und `price_cap` pro Jahr gesetzt und an `optimize_day()` durchgereicht. Diese Werte sollten für alle Tage identisch sein – Logging einbauen, um dies zu verifizieren.

**Empfohlene Änderung:**
1. **Primär:** FIX-S2-17 implementieren (Start-SoC = `soc_min_kwh`). Dies löst das beobachtete Symptom.
2. **Diagnose:** Debug-Logging in `engine.py` für die ersten 3 Tage einbauen: effektive Preise, SoC-Verlauf, Optimizer-Entscheidungen. Damit verifizieren, dass der Collar-Spread nach RTE-Verlusten den BESS-Einsatz nicht rechtfertigt.
3. **Falls nach FIX-S2-17 das Problem weiterhin besteht:** Detailanalyse der Preiskonstruktion mit konkreten Szenario-Daten (Spotpreise, Floor, Cap) durchführen.

**Abhängigkeiten:**
- Hängt stark von FIX-S2-17 (SoC Start = MIN_SOC) ab
- Betrifft alle PPA/EEG-Szenarien, nicht nur Collar

---

### FIX-S2-13: BESS-Replacement CAPEX fremdfinanzieren
**Status:** OFFEN
**Dateien:** `finance/cashflow.py`, `finance/debt.py`, `bess/replacement.py`, `optimization/grid_search.py`, `optimization/monte_carlo.py`

**Problem:** Der BESS-Replacement-CAPEX wird aktuell vollständig aus Eigenkapital finanziert. Er soll stattdessen (anteilig) fremdfinanziert werden – die Restschuld des bestehenden Kredits wird um den FK-Anteil des Replacements erhöht.

**Analyse:**
- `cashflow.py` Zeile 132-136: Kommentar bestätigt "equity-financed, no additional debt". `replacement_capex_this_year` wird vollständig von Equity-CF abgezogen.
- `debt.py`: `build_annuity_schedule()` baut einen fixen Tilgungsplan ab Jahr 1. Es gibt keine Funktion für Mid-Project-Debt.
- `AnnuitySchedule` (Zeile 15-31) enthält `interest_payments`, `principal_payments`, `remaining_balance` pro Jahr – die Restschuld ist verfügbar.
- `replacement.py` `ReplacementConfig` (Zeile 19-54): Enthält die Kostenkomponenten, aber kein Feld `pct_of_capex` (obwohl `main.py` es liest).

**Empfohlene Änderung:**
1. **`finance/debt.py`**: Neue Funktion `add_replacement_debt()`:
   ```python
   def add_replacement_debt(
       existing_schedule: AnnuitySchedule,
       replacement_cost: float,
       leverage_pct: float,
       annual_interest_rate: float,
       replacement_year: int,
       remaining_tenor_years: int,
   ) -> AnnuitySchedule:
       """Erhöhe die Restschuld ab replacement_year um den FK-Anteil des Replacements.

       Berechne eine neue Annuität für den Replacement-FK-Anteil über die Restlaufzeit
       und addiere diese auf die bestehenden Zahlungen.
       """
   ```
   - `replacement_debt = replacement_cost * leverage_pct / 100`
   - Neue Annuität über `remaining_tenor_years = loan_tenor_years - replacement_year + 1`
   - `interest_payments[y]` und `principal_payments[y]` für `y >= replacement_year` erhöhen
   - `annual_payment` wird variabel (vor Replacement: alte Annuität, danach: alte + neue)
2. **`finance/cashflow.py`**:
   - `replacement_capex_this_year` aufteilen in `replacement_equity` und `replacement_debt`
   - `replacement_equity = replacement_cost * (1 - leverage_pct / 100)`
   - Equity-CF-Abzug nur mit `replacement_equity`, nicht dem vollen Betrag
   - Debt Schedule muss **vor** der Cashflow-Schleife um den Replacement-Debt erweitert werden (oder in der Schleife bei `y == replacement_year` modifiziert werden)
3. **`bess/replacement.py`**: `pct_of_capex` als Feld zu `ReplacementConfig` hinzufügen (aktuell fehlt es)
4. **Tests:** `test_debt.py` um `add_replacement_debt` erweitern; `test_cashflow.py` Replacement-Szenario anpassen

**Abhängigkeiten:**
- FIX-S2-10 (Debt Split) sollte vorher implementiert sein, da die neuen Debt-Komponenten (Interest + Principal) auch für den Replacement-Kredit relevant sind
- Hat Impact auf alle Szenarien mit `replacement.enabled: true`
- Ändert Equity-CF und DSCR in Replacement-Szenarien signifikant

---

### FIX-S2-14: `-v` Flag setzt `max_workers=1`
**Status:** OFFEN
**Dateien:** `main.py`

**Problem:** Das `-v` / `--verbose` CLI-Flag setzt nur den Log-Level auf DEBUG, aber nicht `max_workers=1`. Multi-Processing erschwert Debugging erheblich.

**Analyse:**
- `main.py` Zeile 979: `log_level = logging.DEBUG if args.verbose else logging.INFO` – einzige Nutzung von `args.verbose`
- `GridSearchConfig` (Zeile 628-674): `max_workers` wird nie gesetzt → Default `None` (= alle CPU-Kerne)
- `MCParams` (Zeile 838-848): `max_workers` wird nie gesetzt → Default `None`
- Beide Dataclasses haben bereits ein `max_workers: int | None = None` Feld

**Empfohlene Änderung:**
1. In `main.py` bei der Konstruktion von `GridSearchConfig` (ca. Zeile 674):
   ```python
   max_workers=1 if args.verbose else None,
   ```
2. In `main.py` bei der Konstruktion von `MCParams` (ca. Zeile 848):
   ```python
   max_workers=1 if args.verbose else None,
   ```

**Abhängigkeiten:**
- Keine Abhängigkeiten
- Isolierter 2-Zeilen-Fix
- Kein Impact auf Testergebnisse (nur Performance)

---

### FIX-S2-15: BESS-Replacement Kapazitäts-Upgrade-Faktor
**Status:** OFFEN
**Dateien:** `config/schema.py`, `bess/replacement.py`, `dispatch/engine.py`, `main.py`, `optimization/grid_search.py`

**Problem:** Es soll möglich sein, dem BESS-Replacement einen Kapazitäts-Upgrade-Faktor mitzugeben, um Technologiesprünge zu simulieren (z.B. neuer BESS hat 120% der ursprünglichen Kapazität). Default = 1.0 (100%, kein Upgrade).

**Analyse:**
- `dispatch/engine.py` Zeile 442-445: Bei Replacement wird `bess_cap = config.bess_nameplate_kwh` gesetzt (100% der Ursprungskapazität)
- `bess/replacement.py` `ReplacementConfig` (Zeile 19-54): Kein Feld für Upgrade-Faktor
- `config/schema.py` `_BESS_REPLACEMENT` (Zeile 189-201): Kein Feld definiert
- Der Upgrade-Faktor muss auf die **Kapazität (kWh)** angewandt werden, die **Leistung (kW)** bleibt gleich (da der Netzanschluss und die Leistungselektronik bestehen bleiben)

**Empfohlene Änderung:**
1. **Schema** (`config/schema.py` `_BESS_REPLACEMENT`):
   ```python
   "capacity_factor_pct": {"type": "number", "minimum": 0, "default": 100}
   ```
2. **`bess/replacement.py`** `ReplacementConfig`:
   - Neues Feld: `capacity_factor_pct: float = 100.0`
   - In `replacement_config_from_dict()`: Feld lesen mit Default 100.0
3. **`dispatch/engine.py`** Zeile 444:
   ```python
   # Vorher:
   bess_cap = config.bess_nameplate_kwh
   # Nachher:
   upgrade = config.replacement.capacity_factor_pct / 100.0
   bess_cap = config.bess_nameplate_kwh * upgrade
   ```
4. **`config/defaults.py`**: `DEFAULT_BESS_REPLACEMENT_CAPACITY_FACTOR_PCT: float = 100.0`
5. **`optimization/grid_search.py`**: `_GridPointArgs` und `_evaluate_grid_point` um `replacement_capacity_factor_pct` erweitern
6. **`main.py`**: Feld aus JSON lesen und durchreichen
7. **Impact auf Kosten:** Der Upgrade-Faktor sollte auch die Replacement-Kosten beeinflussen – die `eur_per_kwh`-Komponente bezieht sich auf die **neue** Kapazität: `replacement_cost_kwh = eur_per_kwh × bess_capacity_kwh × upgrade_factor`

**Abhängigkeiten:**
- Kann unabhängig implementiert werden
- Sinnvoll in Kombination mit FIX-S2-13 (Replacement-CAPEX fremdfinanzieren), da höhere Kapazität → höhere Kosten → höherer Kredit
- Impact auf: Dispatch (höhere Kapazität → mehr Speicher), Kosten (höhere kWh-Basis), DSCR

---

### FIX-S2-16: DSCR auf P90-Basis entfernen
**Status:** ABGEDECKT DURCH FEATURE 06
**Dateien:** `main.py`, `optimization/grid_search.py`, `config/schema.py`

**Problem:** Die P90-Zeitreihe wird nur noch für eine separate DSCR-Berechnung verwendet. Da P90 aus der PV-Zeitreihe eliminiert wurde, ist dieser Aufwand (kompletter zweiter Dispatch-Lauf pro Grid-Punkt) nicht mehr gerechtfertigt.

**Analyse:**
- `main.py` Zeile 557-562: `p90_timeseries` wird weiterhin berechnet und geloggt
- `main.py` Zeile 670-674: `debt_uses_p90`, `pv_base_timeseries_p90`, `spot_prices_yearly_p90` werden an `GridSearchConfig` übergeben
- `grid_search.py` Zeile 139-146: `GridSearchConfig` enthält P90-Felder
- `grid_search.py` Zeile 441-454: **Vollständiger zweiter Dispatch-Lauf** (`run_simulation()`) mit P90-Timeseries – das ist der Performance-Overhead
- `grid_search.py` Zeile 518-524: P90-DSCR überschreibt die P50-DSCR
- `grid_search.py` Zeile 677-687: P90-Felder in `_GridPointArgs`
- `config/schema.py` Zeile 352, 361: `debt_uses_p90` als Schema-Feld

**Empfohlene Änderung:**
1. **`optimization/grid_search.py`**:
   - Entferne aus `GridSearchConfig`: `debt_uses_p90`, `pv_base_timeseries_p90`, `spot_prices_yearly_p90`
   - Entferne aus `_GridPointArgs`: `pv_base_timeseries_p90`, `spot_prices_yearly_p90`
   - Entferne den kompletten P90-Simulationsblock (Zeile 441-454)
   - Entferne die P90-DSCR-Überschreibung (Zeile 518-524)
2. **`main.py`**:
   - Entferne `debt_uses_p90` aus Szenario-Lesung
   - Entferne `p90_timeseries` aus `GridSearchConfig`-Konstruktion
   - `compute_p50_p90()` kann beibehalten werden (P90 wird noch für Logging/Info genutzt), aber die P90-Timeseries wird nicht mehr an den Grid-Search weitergegeben
3. **`config/schema.py`**: `debt_uses_p90` aus `_FINANCE` properties und required-Liste entfernen
4. **Performance-Gewinn:** Pro Grid-Punkt fällt ein kompletter Multi-Year-Dispatch weg → ~50% schneller

**Abhängigkeiten:**
- Unabhängig von anderen Fixes
- **Achtung:** Bestehende JSON-Dateien enthalten `"debt_uses_p90": true` → Schema-Validierung wird fehlschlagen, wenn das Feld entfernt wird. Option: Feld im Schema belassen aber ignorieren (mit `additionalProperties: true` oder einfach nicht mehr auswerten)
- Impact auf Testergebnisse: DSCR-Werte ändern sich (vorher P90-basiert, nachher P50-basiert)

---

### FIX-S2-17: SoC Start = MIN_SOC statt 50%
**Status:** OFFEN
**Dateien:** `config/defaults.py`, `dispatch/engine.py`

**Problem:** Der BESS startet die Simulation bei 50% des maximalen SoC. Ökonomisch sinnvoller (und realistischer) ist der Start bei MIN_SOC (leer), da der BESS am ersten Tag aus PV oder Netz geladen wird.

**Analyse:**
- `config/defaults.py` Zeile 78-79: `DEFAULT_START_SOC_FRACTION: float = 0.50`
- `dispatch/engine.py` Zeile 469-471:
  ```python
  current_soc = bess_params.soc_max_kwh * DEFAULT_START_SOC_FRACTION
  current_soc_green = current_soc
  current_soc_grey = 0.0
  ```
- `bess_params.soc_max_kwh` ist die absolute obere SoC-Grenze in kWh (z.B. 3.600 kWh bei 4.000 kWh Kapazität und 90% max_soc)
- Aktuell: Start-SoC = 3.600 × 0.50 = 1.800 kWh
- Gewünscht: Start-SoC = `bess_params.soc_min_kwh` (z.B. 400 kWh bei 10% min_soc)
- `DEFAULT_START_SOC_FRACTION` wird nur an dieser einen Stelle verwendet

**Empfohlene Änderung:**
1. **`dispatch/engine.py`** Zeile 470 ändern:
   ```python
   # Vorher:
   current_soc = bess_params.soc_max_kwh * DEFAULT_START_SOC_FRACTION
   # Nachher:
   current_soc = bess_params.soc_min_kwh
   ```
2. **`config/defaults.py`**: `DEFAULT_START_SOC_FRACTION` entfernen (nicht mehr benötigt)
3. **Grey Mode** Zeile 471-472: `current_soc_green = current_soc` und `current_soc_grey = 0.0` bleiben unverändert (BESS startet mit gesamtem SoC als "green")

**Abhängigkeiten:**
- FIX-S2-12 (Collar Bug) hängt von diesem Fix ab
- **Impact auf Tests:** `test_engine.py`, `test_optimizer.py` – alle Tests die den initialen SoC prüfen
- **Impact auf Ergebnisse:** Erste Tage des ersten Jahres haben andere Dispatch-Entscheidungen. Über das Gesamtjahr gleicht sich der Effekt aus, da der Optimizer den SoC schnell auf das ökonomische Optimum bringt.

---

## Abhängigkeiten zwischen Fixes

```
FIX-S2-06 (CSV User Input) ──umfasst──→ FIX-S2-08 (Dezimalkomma)
FIX-S2-08 (Dezimalkomma) ──────────→ FIX-S2-01 (Benchmark Test) [konsistentes Parsing]
FIX-S2-07 (Output Dir) ───────────→ FIX-S2-01 (Benchmark Test) [korrekter Output-Pfad]
FIX-S2-10 (Debt Split) ───────────→ FIX-S2-01 (Benchmark Test) [Spaltenänderung berücksichtigen]
FIX-S2-10 (Debt Split) ───────────→ FIX-S2-13 (Replacement Debt) [Debt-Komponenten benötigt]
FIX-S2-03 (BESS-Only) ────────────→ FIX-S2-11 (Grid Search Skip) [einzelne Konfiguration]
FIX-S2-17 (SoC Start MIN_SOC) ────→ FIX-S2-12 (Collar Bug) [löst Hauptursache]
FIX-S2-15 (Upgrade-Faktor) ───────→ FIX-S2-13 (Replacement Debt) [höhere Kapazität → höhere Kosten]
```

## Empfohlene Reihenfolge

1. **FIX-S2-17** – SoC Start = MIN_SOC (2 Zeilen, löst Collar-Bug)
2. **FIX-S2-14** – `-v` → `max_workers=1` (2 Zeilen, sofort nützlich für Debugging)
3. **FIX-S2-12** – Collar Bug verifizieren (nach FIX-S2-17, ggf. nur Diagnose)
4. **FIX-S2-07** – Output Directory aus JSON (kleiner, isolierter Fix)
5. **FIX-S2-08** – Dezimalkomma (Voraussetzung für Benchmark-Vergleich)
6. **FIX-S2-06** – CSV User Input (erweitert FIX-S2-08 um konfigurierbare Settings)
7. **FIX-S2-10** – Debt Service Split (ändert CSV-Format)
8. **FIX-S2-09** – Excel Lock Handling (defensiver Fix)
9. **FIX-S2-16** – P90-DSCR entfernen (Performance-Gewinn, Vereinfachung)
10. **FIX-S2-11** – Grid Search Skip (Performance-Optimierung)
11. **FIX-S2-15** – BESS-Replacement Upgrade-Faktor (neues Feature)
12. **FIX-S2-13** – BESS-Replacement fremdfinanzieren (komplexe Finanz-Änderung)
13. **FIX-S2-03** – BESS-Only (größte strukturelle Änderung)
14. **FIX-S2-02** – Smoke Test (validiert End-to-End nach allen Änderungen)
15. **FIX-S2-01** – Cashflow Benchmark (validiert Finanzergebnisse, nutzt finales CSV-Format)

## Zusammenfassung

| # | Beschreibung | Status | Priorität | Impact |
|---|---|---|---|---|
| FIX-S2-01 | Cashflow Benchmark-Test | OFFEN | Hoch | Neuer Test, keine Code-Änderung |
| FIX-S2-02 | Smoke Test | OFFEN | Hoch | Neuer Test + JSON-Fix |
| FIX-S2-03 | BESS-Only Cases | OFFEN | Hoch | Schema, Main, Grid Search |
| FIX-S2-04 | OPEX eur_per_kw/kwh | BEREITS IMPLEMENTIERT | – | – |
| FIX-S2-05 | Loan Tenor | BEREITS IMPLEMENTIERT | – | – |
| FIX-S2-06 | CSV User Input | OFFEN | Mittel | Defaults, Schema, CSV Writer |
| FIX-S2-07 | Output Dir aus JSON | OFFEN | Niedrig | Main.py (3 Zeilen) |
| FIX-S2-08 | Dezimalkomma | OFFEN | Mittel | Formatting, Defaults, Tests |
| FIX-S2-09 | Excel Lock Handling | OFFEN | Niedrig | CSV Writer |
| FIX-S2-10 | Debt Service Split | OFFEN | Mittel | Cashflow, Debt, CSV Writer |
| FIX-S2-11 | Grid Search Skip | OFFEN | Niedrig | Grid Search, Performance |
| FIX-S2-12 | Collar Bug (BESS Charging) | OFFEN | Hoch | Engine, Optimizer (Diagnose nach S2-17) |
| FIX-S2-13 | BESS-Replacement fremdfinanzieren | OFFEN | Hoch | Cashflow, Debt, Replacement |
| FIX-S2-14 | `-v` → `max_workers=1` | OFFEN | Niedrig | Main.py (2 Zeilen) |
| FIX-S2-15 | BESS-Replacement Upgrade-Faktor | OFFEN | Mittel | Schema, Replacement, Engine |
| FIX-S2-16 | P90-DSCR entfernen | ABGEDECKT DURCH FEATURE 06 | – | Feature 06 eliminiert P90 komplett |
| FIX-S2-17 | SoC Start = MIN_SOC | OFFEN | Hoch | Engine, Defaults (2 Zeilen) |
