# Bekannte Fehler und Inkonsistenzen

## Integration

- [ ] price_loader wird nur in tests verwendet, aber nicht in produktiven Code. Entweder entferne die Klasse, oder
  stelle sicher, dass sie korrekt im produktiven Code genutzt wird.

## Logik

- [ ] Die MC Parameter für die Preisszenarien soll direkt aus project_settings.finance.price_input.scenario stammen. Es
  bedarf keines extra Inputs in scenario.monte_carlo.price_scenario mehr

## Kosmetik

- [ ] BESS.costs.optimization_fee_pct soll in BESS.costs.opex.optimization_fee_pct verschoben werden
- [ ] entferne die csv relevanten json attribute in project_settings.finance.price_inputs, da diese nun in den
  jeweiligen Szenarien direkt berücksichtigt werden
- [ ] implementiere Unit tests für _effective_green_price, der alle Market Szenarien (Market, EEG, PPA floor, PPA Collar, PPA baseload, PPA pay-as-produced) abdeckt
- [ ] Ich habe manuell viele kleinere Fehler behoben, passt die Unit tests so an, dass sie alle wieder erfolgreich sind. Verändere nicht die Logik! Diese läuft nun korrekt
- BESS: green discharge funktioniert bei Übertrag zwischen Tagen nicht.
  - Green --> Null
  - Grey --> Profitabel
  - EEG/PPA
    - Einkauf: Spot
    - Verkauf: Fixed