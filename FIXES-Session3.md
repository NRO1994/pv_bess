# Bekannte Fehler und Inkonsistenzen

## Integration

- [ ] Einbettung in den User Flow stellt ein komplett neues Feature dar. Im Allgemeinen soll es eine stand-alone offline
  HTML Datei sein, die die Input JSON auf basis des Schemas (siehe .docs.input_schema.json) erstellt. Die HTML-Datei
  stelle ich allen KollegInnen zur Verfügung, die mit dem Tool arbeiten sollen. Sobald sie das Template ausgefüllt
  haben, lassen sie mir die resultierende JSON Datei zukommen. Ich führe die Simulation lokal durch, und teile den
  Report mit ihnen. Dieser Report soll ebenfalls eine einzelne HTML-Datei sein, die ein interaktives Dashboard enthält,
  in dem alle Ergebnisse einzusehen sind. Beide HTML-Dateien sollen kein Backend, keine Ports, keine externen
  Abhängigkeiten haben. Keine Azure App Registration. Es soll ein reiner Offline-Betrieb im Browser sein. Kompatibilität
  zu aktuellen Edge/Chrome/Firefox, Sprache Deutsch. Das Layout für beide HTML-Dateien soll professionell im corporate
  design (siehe .docs.features.08_pdf_report.md) gehalten sein. Keine Emojis, aber das Logo des Tools
  (.data/tool_logo.png) sowie das Unternehmenslogo (.data.stadtwerke_luebeck.png) sollen sichtbar sein.
    - Input HTML (Ein gutes Beispiel findest du hier: pv_bess_model.output.report.templates.copilot_sample.html): Die
      Input-Datei soll intuitiv, Nutzer zentriert und leicht in der Bedienung sein, es soll Spaß machen,
      sich mit diesem Thema zu beschäftigen. Gruppiere alle Inputparameter sinnvoll in verschiedene naheliegende
      Cluster. Diese Datei sollst du einmalig erstellen, ich verteile sie dann manuell weiter. Folgende Parameter werden
      direkt im Code gesetzt, auf diese soll der User keinen Zugriff haben:
        - `scenario.output`: Da die Simulation bei mir lokal läuft, sollen sie kein Output relevanten Felder ausfüllen.
          Das `directory`soll per default auf: `.data/outputs/`gesetzt werden
        - `project_settings.price_inputs`: Die CSV Dateien liegen auf meinem Laptop, die Pfade, sowie alle weiteren
          Parameter für die einzelnen Szenarien möchte ich manuell in die HTML-Datei einpflegen.
    - Output HTML (Ein gutes Beispiel findest du hier: pv_bess_model.output.report.templates.result_dashboard.html). Im
      Allgemeinen soll dieser Report die Ergebnisse der Simulation klar und transparent darlegen. Alle Diagramme, sowie
      die Daten sollen herunterzuladen sein. Diagramme als png, Daten als CSV. Es ist zwingend
      erforderlich, dass eine Verbindung zu dem Input JSON hergestellt wird. Zudem muss das Datum der Erstellung, sowie
      mindestens der Name der Input-JSON Datei vermerkt werden. Es soll durch eine LLM-generierte Beschreibung der
      Ergebnisse angereichert werden. Ich möchte, dass du dazu ein Template entwickelst, dass sich dann mit den
      aktuellen Werten befüllen lässt. Der zu implementierende Flow dazu soll wie folgt sein:
        - Erstellung der Prompts, die die Ergebnisse, Zusammenhänge und Erklärungen beschreibt. Dazu soll das
          Prompt-Template unter .docs/llm_templates/ erstellt werden, das dann mit den aktuellen Daten pro Durchlauf
          befüllt wird.
        - Speichern des Prompts in einer md Datei im output directory des Szenarios
        - MANUELLES kopieren in den Copilot Chat. Da ich im konzernumfeld agiere, und keinen API Zugriff habe
        - Ich erstelle eine Datei mit den Antworten des Copilot
        - Die HTML-Report Datei kann final erstellt werden.
        - Überlege dir genau, wie sichergestellt werden kann, dass die richtige Antwort zum richtigen Diagramm eingebaut
          wird. Es könnte zum Beispiel ein json Output der LLM erzwungen werden, der dann einfach zu parsen ist. Zu dem
          möchte ich nach Möglichkeit EINEN Prompt kopieren. Fal
        - Die Struktur der HTML-Datei soll sich wie folgt gestalten. Alle Diagramme werden bereits in der Codebase
          erzeugt, es sind also alle Daten bereits dafür aufgearbeitet. Es bedarf jedoch KEINER Backward-Compatibility,
          deswegen kannst du den Code direkt so ändern, dass er für die neue Ausgabe im Rahmen dieser interaktiven
          HTML-Dashboards passend ist:
            - Tab 1: Darstellung des Szenarioinputs. Fokus auf relevante Größen (PV Kapazität, BESS design space,
              Vermarktungsstrategien), Datum der Erstellung/Simulation. Wenn Internet verfügbar, Einblendung einer
              OpenStreetMap-Karte, in der eine Markierung auf den Standort der PV-Anlage gesetzt ist. Ansonsten der LLM
              Text, in dem Schlüsselparameter fett gedruckt sind
            - Tab 2: Darstellen der Inputzeitreihen aus den CSV. Nutze dafür die bereits vorhandenen charts zur PV
              Energie, sowie Strompreisszenarien. Ebenfalls soll der Input durch einen LLM Text erklärt werden
            - Tab 3: Analyse der EEG (Falls im Input angefragt). Diagramm, plus erklärenden Text.
            - Tab 4: Analyse der PPA-Collar (Falls im Input angefragt). Diagramm, plus erklärenden Text.
            - Tab 5: Analyse des PPA-Baseload (Falls im Input angefragt). Diagramm, plus erklärenden Text.
            - Tab 6: Cashflow Analyse des Grid-Search Optimums. Dies stellt den Case bei dem Central-Preis Szenario dar.
              Das Diagramm soll ein gestapeltes Säulendiagramm sein, bei dem der PV-, BESS-Green- und BESS Grau-Revenue
              positiv dargestellt ist, und alle Ausgaben (Capex, OPEX, Debt, Tax, sowie gegebenenfalls auch grid costs
              und ppa-balance costs) als negative Säulen. Der Text der LLM soll dabei alle KPI's des "metric" objects
              beinhalten, und beschreiben, wie der Cashflow einzuschätzen ist.
- [ ] Die höchste Priorität hat es, den Code auf ein Level zu bringen, in dem alle Unit-Tests erfolgreich laufen. Ich
  habe durch die Integration Test Suite manuell viele kleinere Fehler behoben, aber nicht die Unit Tests-angepasst.
  Deine Aufgabe ist es, die Unit-Tests so anzupassen, dass sie alle wieder erfolgreich sind. Verändere nicht die Logik!
  Diese läuft nun korrekt. Prüfe die Code Coverage durch die Unit-Tests, und schlage weitere Unit tests vor, um die
  Test-Coverage zu verbessern. Vor allem für kritische Funktionen wie zum Beispiel
  timeseries.align_weather_to_forecast_year, oder für _effective_green_price, der alle Market Szenarien (Market, EEG,
  PPA floor, PPA Collar, PPA baseload, PPA pay-as-produced) abdeckt
- [ ] Der nächste wichtige Schritt ist es ein generelles Clean-Up durchzuführen. Es gibt viele redundante
  Berechnungen, zum Beispiel für den Cashflow. Suche nach weiteren und vereinheitliche diese. Achte dabei besonders
  darauf, dass alle Tests weiterhin erfolgreich sind. Dies ist nur ein Refactoring bestehender Logik, und nicht die
  Erweiterung! Zudem gibt es vielen ungenutzten Code (zum Beispiel der price_loader). Prüfe auf weitere solcher Fälle.
  Prüfe jeweils, ob sich diese noch in die Logik integrieren lassen, oder ob sie gelöscht werden sollten. Überarbeite
  ebenfalls die Unit-Tests, falls sich dort ebenfalls Tests befinden, die entweder nicht mehr relevant sind, oder
  helper-functions die nicht mehr benötigt werden. Prüfe zu letzten noch das linting, und ergänze an Stellen, an denen
  es nicht eingehalten wird.

## Logik

- [ ] Es soll in der daily-optimization der Solver aus ortools.linear_solver import pywraplp
  ,pywraplp.Solver.CreateSolver('HiGHS') verwendet werden. Beachte, dass alle anderen Unit Tests sowie Integration Tests
  weiterhin erfolgreich sein sollen
- [ ] der BESS darf in einem Zeitpunkt nur entweder laden, oder entladen werden - aber nicht beides gleichzeitig.
  Überlege wie sich diese Nebenbedingung gut in das Problem einarbeiten lässt. Als mögliche Vereinfachung könnte bei
  negativen Preisen das `discharging = 0` festgelegt werden, da dieses Phänomen nur in solchen Fällen auftritt. Es
  erwirtschaftet dabei durch die Ineffizienzen des Speichers Gewinne, die aber nicht realisierbar sind.
- [ ] Der Vermarktungsmodus "PPA-Baseload" ist an diversen Stellen noch nicht korrekt implementiert. Er soll zunächst in
  LP optimization mit aufgenommen werden. Dies ist notwendig, da im PV/BESS Case der Speicher dient den Baseload länger
  zu halten, als es nur durch die PV Anlage möglich ist. Das bedeutet, die zusätzlichen Einkäufe, durch zu niedrigen
  Baseload müssen in der täglichen Optimierung berücksichtigt werden. Der Revenue berechnet sich dann in Zeitpunkten,
  bei denen genug Einspeisung (durch PV und/oder BESS) vorliegt durch: `max(spot_price, effective_price)`. In Zeiten wo
  nicht ausreichend Einspeisung vorliegt, müssen die Einkaufskosten berücksichtigt werden. Diese berechnen sich durch
  `(baseload - grid_export) * (spot_price - effective_price)`. Die Variablen dazu sollen dann ebenfalls im CSV Dispatch
  und Cashflow zu sehen sein.

## Kosmetik

- [ ] BESS.costs.optimization_fee_pct soll in BESS.costs.opex.optimization_fee_pct verschoben werden
- [ ] entferne die csv relevanten json attribute in project_settings.finance.price_inputs, da diese nun in den
  jeweiligen Szenarien direkt berücksichtigt werden, entferne ebenfalls die dazu passende Logik. Alle CSV-reads werden
  innerhalb der einzelnen Szenarien durchgeführt, es gibt keine weitere Notwendigkeit diese bestehende Logik
  beizubehalten.
- [ ] Die MC Parameter für die Preisszenarien soll direkt aus project_settings.finance.price_input.scenario stammen. Es
  bedarf keines extra Inputs in scenario.monte_carlo.price_scenario mehr
- [ ] Die Preisangaben im Input sind immer unterschiedlich (einmal Euro pro MWh, ein anderes Mal Euro pro KWh).
  Vereinheitliche dies auf pro kWh, sodass jegliche Konvertierung im Programm unnötig wird. Entferne diese Konvertierung
  ebenfalls.
- [ ] wenn im Output directory bereits die png Dateien enthalten sind, dann werden sie bei einem erneuten Durchlauf
  nicht ersetzt. Dies soll jedoch der Fall sein (So wie auch bei den CSV Outputs)