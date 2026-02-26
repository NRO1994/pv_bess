# Bekannte Fehler und Inkonsistenzen

## Integration

- [ ] Cashflow Prüfung: Ich habe aus einem bestehenden Modell die CSV-Output Datei "_cashflows.csv" nachgebaut.
  Sie hat denselben Aufbau, wie die aus diesem Modell erzeugte "_cashflows.csv" Datei.
  Implementiere einen Integrationtest, der diese Datei mit der aus diesem Modell erzeugten vergleicht. Beachte dabei,
  die Ergebnisse im "test" Folder zu speichern, und nicht im regulären Output Ordner.
  Es ist eine prozentuale Abweichung von 1% tolerabel. Finalisiere den Vergleich, in dem du ein Protokoll und Analyse
  der Ergebnisse hier speicherst: .docs/finance_benchmarking.md.
  Beachte, dass auch die Base-Werte fehlerhaft sein können. Kommst du zu dem Schluss, dass die Betrachtung dieses
  Modells trotz größerer Abweichung korrekt ist, erkläre deine Sichtweise in dem Analyse-Part der finance_benchmarking.
  Bevor du Änderungen implementierst, möchte ich zunächst deine Sichtweise lesen. es bedarf nur deiner Feststellung und
  Erklärung der Differenzen, keine Codeanpassungen!
  Die Vergleichsdatei ist hier zu finden: .data/integration_test_inputs/finance/integration_test_cashflow_base.csv, das
  dazugehörige Szenario hier: .data/integration_test_inputs/finance/integration_test_cashflow.json
  Dieser Test soll ebenfalls mit dem "tag integration" versehen sein, also nur nach spezifischer Aufforderung laufen
- [ ] Es gibt bereits Daten für einen smoke test (.data/integration_tests_inputs/smoke_test). Erstelle einen Integration
  test, der auf die neusten Änderungen angepasst ist, und eine valide Aussage über den End-to-End Prozess machen kann.
  Dieser Test soll ebenfalls mit dem "tag integration" versehen sein, also nur nach spezifischer Aufforderung laufen
- [ ] ermögliche "BESS-only" cases. Dafür muss die Eingabelogik angepasst werden, sodass auch eine separate
  Leistung/Kapazität angegeben werden kann. Zudem muss es möglich sein das "pv" Attribut komplett wegzulassen, bzw. die
  Leistung auf 0 zu setzen. Entwickle hierfür ebenfalls ein integration test, der auf
  .data/integration_test_inputs/bess_only Dateien beruht

## Logik

- [ ] OPEX soll ebenfalls per "eur_pro_kw" und "eur_pro_kwh" berechnet werden können, passe auch das json-Schema
  entsprechend an
- [ ] loan tenor ist Teil des inputs, aber in der Berechnung des Debt Services nicht berücksichtigt

## Kosmetik

- [ ] timestamp column name ist user input, bzw. mit default wert in config/defaults.py.
  Ebenso soll das "Timeformat", so wie der "separator" und "decimal" auch aus dem json eingelesen werden können
- [ ] "scenario.output.directory" aus json wird nicht übernommen, es wird immer der default verwendet
- [ ] CSV writer soll als decimal komma verwenden. Dies soll ebenfalls im config/defaults.py definiert werden
- [ ] während des Testens kommt es immer wieder vor, das eine CSV Datei nicht gespeichert werden kann, da sie noch in
  Excel geöffnet ist. Implementiere für diesen Fehler ein catch, der dann dem Dateinamen einen Index hinzufügt
- [ ] die Spalte des "debt_service" soll in "debt_interest_rate" und "debt_repayment" aufgeteilt werden, und dann die
  entsprechenden Werte aus dem Finanzmodell beinhalten.
- [ ] ermögliche die GRid-Search zu überspringen, und zwar wenn für die "scale_pct_of_pv" und "e_to_p_ratio_hours" nur
  ein Wert im Array enthalten ist