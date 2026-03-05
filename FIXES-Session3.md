# Bekannte Fehler und Inkonsistenzen

## Integration

- [ ] price_loader wird nur in tests verwendet, aber nicht in produktiven Code. Entweder entferne die Klasse, oder
  stelle sicher, dass sie korrekt im produktiven Code genutzt wird.
- [ ] Ich habe manuell viele kleinere Fehler behoben, passt die Unit tests so an, dass sie alle wieder erfolgreich sind.
  Verändere nicht die Logik! Diese läuft nun korrekt

## Logik

- [ ] Die MC Parameter für die Preisszenarien soll direkt aus project_settings.finance.price_input.scenario stammen. Es
  bedarf keines extra Inputs in scenario.monte_carlo.price_scenario mehr
- [ ] es soll in dem daily optimization der solver from ortools.linear_solver import pywraplp
  ,pywraplp.Solver.CreateSolver('HiGHS') verwendet werden
- [ ] der BESS darf in einem Zeitpunkt nur entweder laden, oder entladen werden - aber nicht beides gleichzeitig. Dies
  triff nur auf, sofern der Preis negativ ist. Das heißt, es könnte auch die Nebenbedingung gelten, dass bei negativen
  strompreisen das "discharging" ausgeschlossen werden soll
- [ ] GoO sollen nur bei PV Cases berücksichtigt werden, nicht bei BESS only
- [ ] ppa baseload in LP optimization mit aufnehmen

## Kosmetik

- [ ] BESS.costs.optimization_fee_pct soll in BESS.costs.opex.optimization_fee_pct verschoben werden
- [ ] entferne die csv relevanten json attribute in project_settings.finance.price_inputs, da diese nun in den
  jeweiligen Szenarien direkt berücksichtigt werden
- [ ] implementiere Unit tests für _effective_green_price, der alle Market Szenarien (Market, EEG, PPA floor, PPA
  Collar, PPA baseload, PPA pay-as-produced) abdeckt
- Refactoring: Entferne allen Code, der nicht durch die Integration Tests abgedeckt ist, bzw. der Analyse und Output
  Generierung entspricht.
- Refactoring: Es gibt immer wieder im code alte variablen, die erzeugt werden aber nie genutzt sind, entferne diese.
  Das Gilt für die Unit Tests, ebenso wie für die integration test suite
- Prüfe die Code Coverage durch die Unit tests, und schlage weitere Unit tests vor, die diese verbessern
- MonteCarlo Simulation auf PriceWeatherScenario anpassen
- Analyse Module auf PriceWeatherScenario anpassen