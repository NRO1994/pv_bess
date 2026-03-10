# Bekannte Fehler und Inkonsistenzen

## Integration

- [ ] Equity IRR scheint viel zu hoch - manuell von mir zu prüfen!

## Logik

- [ ] Die MC-Framework Logik muss überarbeitet werden: Es soll der Dispatch Optimizer über die Laufzeit für alle
  gegebenen Preisszenarien einmalig gerechnet werden. Da die MC-Parameter nur einzelne Tage (im Falle der PV/BESS
  Verfügbarkeit), oder kommerzielle Faktoren (CAPEX/OPEX Unsicherheiten) betreffen, ist es eine unnötige Berechnung
  jedes Mal den riesigen Overhead der vollumfänglichen Dispatch Optimierung zu durchlaufen. Die PV/BESS Availabilities
  sollen daher auf 100% für diesen ersten Durchlauf gesetzt werden. Ich habe in einer Studie herausgefunden, dass der
  Fehler nur gering ist, wenn man den Revenue der PV-/BESS-Anlage entsprechend um den Verfügbarkeitsfaktor kürzt. Da es
  sich nur um Prozentbruchteile handelt, ist diese Vereinfachung sinnvoll. Der MC-Framework Prozess soll daher wie folgt
  laufen:
    - Parallelisierte Simulation aller Preis-Szenarios einmalig mit 100% Verfügbarkeit von PV und BESS.
    - Anwenden aller MC-Parameter auf die nun anstehende Finanzbetrachtung. Da keine aufwendige Dispatch Optimierung
      mehr durchgeführt werden muss, können diese Betrachtungen sequenziell (pro Preis-Szenario) durchlaufen werden.
    - Gleiche Finale Zusammenfassung der Ergebnisse wie bisher
- [ ] Es soll in der daily-optimization der Solver aus ortools.linear_solver import pywraplp
  ,pywraplp.Solver.CreateSolver('HiGHS') verwendet werden. Beachte, dass alle Unit-Tests weiterhin erfolgreich sein
  sollen
- [ ] der BESS darf in einem Zeitpunkt nur entweder laden, oder entladen werden - aber nicht beides gleichzeitig.
  Überlege wie sich diese Nebenbedingung gut in das Problem einarbeiten lässt. Als mögliche Vereinfachung könnte bei
  negativen Preisen das `discharging = 0` festgelegt werden, da dieses Phänomen nur in solchen Fällen auftritt. Es
  erwirtschaftet dabei durch die Ineffizienzen des Speichers Gewinne, die aber nicht realisierbar sind.
- [ ] Der Vermarktungsmodus "PPA-Baseload" ist an diversen Stellen noch nicht korrekt implementiert. Er soll zunächst in
  die LP-Optimization mit aufgenommen werden. Dies ist notwendig, da im PV/BESS Case der Speicher dazu dient den
  Baseload länger zu halten, als es nur durch die PV Anlage möglich ist. Das bedeutet, die zusätzlichen Einkäufe, durch
  zu niedrigen Baseload müssen in der täglichen Optimierung berücksichtigt werden. Der Revenue berechnet sich dann in
  Zeitpunkten, bei denen genug Einspeisung (durch PV und/oder BESS) vorliegt durch: `max(spot_price, effective_price)`.
  In Zeiten wo nicht ausreichend Einspeisung vorliegt, müssen die Einkaufskosten berücksichtigt werden. Diese berechnen
  sich durch `(baseload - grid_export) * (spot_price - effective_price)`. Die Variablen dazu sollen dann ebenfalls im
  CSV Dispatch und Cashflow zu sehen sein.
- [ ] Prüfe, warum sich die Abschreibung im pv_only Case nach 10 Jahren ändert (+100EUR). Das darf nicht der Fall sein

## Kosmetik

- [ ] Der CSV Cashflow soll zudem die folgenden Spalten zusätzlich beinhalten. Alle Datenpunkte dazu existieren bereits,
  müssen nur an die entsprechende csv_write-Method weitergeben werden:
    - BESS Green Revenue (EUR)
    - BESS Grey Revenue (EUR)
    - PV Revenue (EUR)
    - PV Grid Export (MWh)
- [ ] im HTML Input wizard soll es die folgenden Anpassungen geben, achte dabei darauf, dass sich die Struktur der
  JSON-Datei nicht ändert!
    - Die Monte-Carlo Parameter sollen von Tab "1 - Szenario" zu "7 - Analysen", unterhalb der bestehenden Felder
    - Auf der "1- Szenario" soll der Betriebsmodus von Tab "2 - Projekt & Standort" hinüber gezogen werden. Zudem soll
      eine Checkbox für "PV-Case" und "BESS-Case" geben. Nur wenn die jeweilige Box ausgewählt ist, soll auch der
      dazugehörige Tab erscheinen. Ist er ausgeblendet, so soll im JSON File die default Werte mit 0 Leistung
      gespeichert werden. Hat eine JSON als Input 0 als power_kwp im PV oder 0 kw als Leistung im BESS, so soll es
      entsprechend als PV oder BESS only case gehandhabt werden.
    - Auf der "2 - Projekt & Standort" soll der Diskontsatz in die "6 - Finanzierung" geschoben werden. Der Standort
      soll auf einer OpenStreetMap Karte entweder auszuwählen sein, oder aber der Input aus den vorhandenen Feldern
      angezeigt werden. Dafür muss die OSM Integration in der Lage sein zu zoomen und zu verschieben. Achte dabei
      darauf, weiterhin das Prinzip der stand-alone HTML Datei aufrecht zu erhalten
    - Auf der "7 - Analysen" sind die Inputfelder grau hinterlegt, und nicht so wie sonst in weiß
    - Auf den "3 - PV Anlage" und "4 - Batteriespeicher" fehlen die horizontalen Trennlinien zwischen Bereichen
    - Im Allgemeinen ist die Breite etwas zu schmal. So wird die Leiste mit den tabs nicht optimal dargestellt. Passt
      die gesamte Breite etwas an, sodass etwas mehr Luft zwischen den einzelnen Feldern ist.
    - Die Preis-Szenario Inputs aus der datei .docs/full_input_example.json sollen als hardcoded in die Output
      JSON-datei aufgenommen werden
- [ ] dashboard_Report.html Anpassungen
    - Tool-Tip ist viel zu weit unten, muss dichter an den Cursor ren
    - Tool-Tip hat im Multi-Line Diagramm immer nur die erste Datenreihe als info, die anderen müssen ebenfalls
      erscheinen
    - Markierungspin in der Karte auf Tab "Szenario-Übersicht" ist nicht an der richtigen Stelle, kein Overlay, sondern
      direkt in OSM verankert. Es soll, so wie im Input, eine interaktive OSM Einbindung sein.
    - Verwende das Header und Tab-Design aus dem input_wizard auch für das Dashboard-Template, nur die Grün-Färbung soll
      nicht übernommen werden

 