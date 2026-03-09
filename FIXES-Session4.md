# Bekannte Fehler und Inkonsistenzen

Dies sind MEINE Notizen, was ich MANUELL bearbeiten möchte. Vergiss nicht Financial Modell Integration Tests, und
Smoke-End2End Integration Test aus FIXES-Session2. Zu dem funktioniert weasyprint auf meinem Laptop nicht, das hängt
aber davon ab, wie EE die Outputs haben möchte.

## Integration

- [ ] Equity IRR scheint viel zu hoch - prüfen!

## Logik

- [ ] Die MC-Framework Logik muss überarbeitet werden: Es soll der Dispatch Optimizer über die Laufzeit für alle
  gegebenen Preisszenarien einmalig gerechnet. Da die MC-Parameter nur einzelne Tage (im Falle der PV/BESS
  Verfügbarkeit), oder kommerzielle Faktoren (CAPEX/OPEX Unsicherheiten) betreffen, ist es eine unnötige Berechnung
  jedes Mal den riesigen Overhead der vollumfänglichen Dispatch Optimierung zu durchlaufen. Die PV/BESS Availabilities
  sollen daher auf 100% für diesen ersten Durchlauf gesetzt werden. Im weiteren Verlauf sollen dann in den BESS-Offline
  Tagen die Dispatch Zeitreihen entsprechend ohne Optimierung durch "optimize_offline_day" nachgerechnet werden.
  Achtung: dies
  betrifft alle Jahre des Dispatches! Die CAPEX/OPEX Unsicherheiten haben nur auswirkung auf die Cashflow/KPI
  Berechnung. --> TODO!!! BESS bei PV offline Tag führt zu Inkonsistenzen, da dann im Zweifel alle folgenden Tage erneut
  berechnet werden müssten (Denn es fehlt ja an einem Tag die PV Einspeisung in den BESS)

## Kosmetik

- [ ] Der CSV Cashflow soll zudem die folgenden Spalten zusätzlich beinhalten. Alle Datenpunkte dazu existieren bereits,
  müssen nur an die entsprechende write-Method weitergeben werden:
    - BESS Green Revenue (EUR)
    - BESS Grey Revenue (EUR)
    - PV Revenue (EUR)
    - PV Grid Export (MWh)
- [ ] im HTML Input wizard soll es die folgenden Anpassungnen geben:
    - Die Monte Carlo Parameter sollen von Tab "1 - Szenario" zu "7 - Analysen", unterhalb der bestehenden Felder
    - Auf der "1- Szeanrio" soll der Betriebsmodus von Tab "2 - Projekt & Standort" hinüber gezogen werden. Zudem soll
      eine Checkbox für "PV-Case" und "BESS-Case" geben. Nur wenn die jeweilige Box ausgewählt ist, soll auch der
      dazugehörige Tab erscheinen. Ist er ausgeblendet, so soll im JSON File die default Werte mit 0 Leistung
      gespeichert werden. Hat eine JSON als Input 0 als power_kwp im PV oder 0 kw als Leistung im BESS, so soll es
      entsprechend als PV oder BESS only case gehandhabt werden. wird nur eines von beiden ausgewählt, soll die
    - Auf der "2 - Projekt & Standort" soll der Diskontsatz in die "6 - Finanzierung" geschoben werden. Der Standort
      soll auf einer OpenStreetMap Karte entweder auszuwählen sein, oder aber der Input aus den vorhandenen Feldern
      angezeigt werden. In der PVIGS-Datenbank soll es ein Drop-Down Menu sein, das aus den Feldern "PVGIS-SARAH3" und "
      ERA5" besteht.
    - Auf der "3 - PV-Anlage" sowie der "5 - Netzanschluss" soll das CAPEX und OPEX feld pro_kwh ausgeblendet sein, dies
      ist nur für BESS relevant. Im
      json soll es auf 0 gesetzt werden. Das Feld CAPEX % CAPEX existiert im json schema nicht, entferne es aus der UI.
    - Auf der "4 - Batteriespeicher" soll ebenfalls das CAPEX % of CAPEX Feld entfernt werden.
    - Auf der "6 - Finanzierung" soll die Inflation als Prozentzahl gegeben werden. Rechne sie dann für die JSON in eine
      dezimale Zahl um.
    - Im Allgemeinen ist die Breite etwas zu schmal. So wird die Leiste mit den tabs nicht optimal dargestellt. Passt
      die gesamte Breite etwas an, sodass etwas mehr Luft zwischen den einzelnen Feldern ist.
    - Der Dark-mode soll entfernt werden
    - in "7 - Analysen" sollen statt einzelner Werte zu den drei Analysen min, max, Stepsize und/oder anzahl steps
      auszuwählen sein. Stelle sicher, dass dann das JSON dem schema entspricht. So muss der Nutzer nicht mehr 10 Werte
      von Hand eingeben, sondern kann dies elegant beschleunigen, ohne dass es in einem falschen JSON mündet.
- [ ] dashboard_Report.html Anpassungen
    - Tool-Tip ist viel zu weit unten, muss dichter an den Cursor ren
    - Tool-Tip hat im Multi-Line Diagramm immer nur die erste Datenreihe als info, die anderen müssen ebenfalls
      erscheinen
- AfA im pv_only case ändert sich nach 10 jahren - warum!? -> Tax wird gar nicht getestet
- Gleiches Layout/Header zwischen input_wizard und dashboard -> Nutzen des Input wizards für Report. Input wizard so
  breit machen wie dashboard. Footer in input wizard, Versions Nummer und Creator aufnehmen. Dashboard: Gleiche Tab
  Formen, wie im Input Wizard, aber ohne Grün Färbung bei klick
- Dashboard: Pin in Karte ist nicht an der richtigen Stelle, kein Overlay, sondern direkt in OSM verankert, Soll
  interaktiv gestaltet sein, sodass man zoomen und verschieben kann. Input und Dashboard gleiches design
- Input: Analysen, gleicher aufbau wie andere Tabs. Aktuell sind die Eingabefelder ebenfalls grau
- Input/BESS und Input/PV: horizontale Trennlinien zwischen Bereichen fehlen

## Fragen:

- Decommissioning extra?
- OPEX feiner aufgliedern?
- Warum AfA in Cashflow Berechnung?