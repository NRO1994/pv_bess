- [x] (CLAUDE) MC Framework anpassen, Plan5 1-4
- [x] (CLAUDE) Smoke Test implementation (Requirements in FIXES), FIX-S2-02
- [x] (CLAUDE) Analyse soll einen Durchlauf mit zentralem Preis-Szenario und Direktvermarktung über die
  gesamte Laufzeit berechnen, ohne MC. Dies soll in jedem Diagramm (EEG, Grid-Search PPA-Collar, PPA-Baseload) als
  konstanter
  Vergleichswert angezeigt werden, um den Upside zu den Vermarktungsstrukturen besser identifizieren zu können. Den IRR
  daraus soll
  auch in dem LLM-Text der "Szenario-Übersicht" erwähnt werden. Zu dem soll (sofern gegeben im input json) eine
  kosntante Linie der internen mindes IRR Anforderung in jeden diagramm zu sehen sein.
- [x] Equity IRR passt nicht
- [x] Warum ist eine Vergrößerung der BESS Kapazität ein schlechterer IRR?
    - --> CAPEX Preisgestaltung ist Problematisch, sowie Grün/Grau sehr unterschiedlich
- [x] baseload MUSS in die LP Optimierung aufgenommen werden, wenn es sich um einen PV/BESS Case handelt. PLan5 1-3
- [x] PaP Preise in Integration suite stimmen nicht
- [x] Financial Model integration Test: Logik ist falsch, mit Steffen klären
- [x] Report
    - [x] Leaflet wird nicht richtig angezeigt
    - [x] Heder Input_wizard/Dashboard sind nicht identisch
    - [x] Format der Tabs soll vom Input_wizard übernommen werden
    - [x] Das Hervorheben des ersten Jahres in der Cashflow Analyse ist unnötig, entferne den Orangen Kasten
    - [x] die Sensitivität im EEG-Tab soll nicht die Standardabweichung um den Mittelwert abbilden, sondern der orange
      Bereich zwischen P10 und P90 aufgespannt werden. Die Linie soll dann der Median sein. Alle Zahlen sind bereits in
      der Analyse vorhanden, es muss nur der data_collector entsprechend angepasst werden
    - [x] Grid search Diagramm vergisst den letzten Punkt. (IRR War nicht gesetzt durch unprofitablen Case)
    - [x] LLM Text
        - [x] Wetterjahre miteinander vergleichen, nicht nur Sommer/winter
        - [x] EEG Standard Abweichung fehlt im Prompt
    - [x] Diagramm Datenreihen auf Deutsch
- [x] CSV Export
    - [x] pv production in CSV Cashflow ist falsch
    - [x] baseload constraints mit ausgeben
- [ ] LLM tags in input json entfernen
- [x] MC Einzel Runs in csv mit ausgeben
    - Warum ist der MEAN von eq.irr bei 20% der P50 aber nur bei 2%?
- [x] Baseload passt noch nicht
- [x] Direktvermarktungs-Baseline muss MC Run sein, Eq.IRR soll dann ebenfalls der P50 der MC results sein
- [x] chart creation entfernen
- [x] Smoke Test fehlende JSON Objekte hinzufügen
- [x] im optimizer.py, bei den methoden: dispatch_offline_day, extract_green_result und extract_grey_result berechnen
  alle jeweils eigenständig die Kennzahlen der Optimierung. Das MUSS vereinheitlicht werden, da sonst keine
  Gleichbehandlung der Ergebniskalkulation gewährleistet werden kann. Zusätzlich wird der daily/yearly revenue erneut in
  der engine gebildet, das heißt es gibt vier Codefragmente, die das gleiche tun sollen - das MUSS vereinheitlicht
  werden. Die Berechnung in den optimizer methoden ist korrekt. In der engine.py wird im Falle des Baseload der Revenue
  falsch berechnet, da der effektive Preis af die gesamte erzeugung angerechnet wird, und nicht nur auf den baseload -
  der rest wird zum Spotpreis verkauft. Prüfe, ob dies in .docs/PLAN_code_cleanup.md enthalten ist, falls nicht füge
  eine detaillierte Beschreibung dieses cleanups hinzu.
- [x] In LCoE Berechnung auch die "Produktion" vom BESS berücksichtigen
- [ ] Gleichzeitiges Be-/Entladen des Speichers muss im LP verhindert werden. Die aktuelle Logik, dass dies nur bei
  negativen Preisen der Fall ist, hat sich als falsch herausgestellt. Das bedeutet, dass diese Nebenbedingung
  umfassender in den Solver aufgenommen werden müssen. Könnte das Einführen einer zusätzlichen Variablen helfen?