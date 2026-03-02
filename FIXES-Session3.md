# Bekannte Fehler und Inkonsistenzen

## Integration

- [ ] price_loader wird nur in tests verwendet, aber nicht in produktiven Code. Entweder entferne die Klasse, oder
  stelle sicher, dass sie korrekt im produktiven Code genutzt wird.

## Logik

- [ ] Die MC Parameter für die Preisszenarien soll direkt aus project_settings.finance.price_input.scenario stammen. Es
  bedarf keines extra Inputs in scenario.monte_carlo.price_scenario mehr
- [ ] PV offline days fehlen in dem set up der grid search. BESS ist gegeben, das sollte auch für PV implementiert sein
- [ ] PV availability gar nicht berücksichtigt!?

## Kosmetik

- [ ] BESS.costs.optimization_fee_pct soll in BESS.costs.opex.optimization_fee_pct verschoben werden
- [ ] entferne die csv relevanten json attribute in project_settings.finance.price_inputs, da diese nun in den
  jeweiligen Szenarien direkt berücksichtigt werden