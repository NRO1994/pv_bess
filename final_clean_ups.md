- [x] (CLAUDE) MC Framework anpassen, Plan5 1-4
- [ ] (CLAUDE) Smoke Test implementation (Requirements in FIXES), FIX-S2-02
- [x] (CLAUDE) Analyse soll einen Durchlauf mit zentralem Preis-Szenario und Direktvermarktung über die
  gesamte Laufzeit berechnen, ohne MC. Dies soll in jedem Diagramm (EEG, Grid-Search PPA-Collar, PPA-Baseload) als konstanter
  Vergleichswert angezeigt werden, um den Upside zu den Vermarktungsstrukturen besser identifizieren zu können. Den IRR daraus soll
  auch in dem LLM-Text der "Szenario-Übersicht" erwähnt werden. Zu dem soll (sofern gegeben im input json) eine
  kosntante Linie der internen mindes IRR Anforderung in jeden diagramm zu sehen sein.
- [x] Equity IRR passt nicht
- [ ] Warum ist eine Vergrößerung der BESS Kapazität ein schlechterer IRR?
- [x] baseload MUSS in die LP Optimierung aufgenommen werden, wenn es sich um einen PV/BESS Case handelt. PLan5 1-3
- [ ] PaP Preise in Integration suite stimmen nicht
- [x] Financial Model integration Test: Logik ist falsch, mit Steffen klären
- [x] Report
    - [x] Leaflet wird nicht richtig angezeigt
    - [x] Heder Input_wizard/Dashboard sind nicht identisch
    - [x] Format der Tabs soll vom Input_wizard übernommen werden
    - [x] Das hervorheben des ersten Jahres in der Cashflow Analyse ist unnötig, entferne den Orangen Kasten
    - [x] die Sensitivität im EEG-Tab soll nicht die Standardabweichung um den Mittelwert abbilden, sondern der orange
      Bereich zwischen P10 und P90 aufgespannt werden. Die Linie soll dann der Median sein. Alle Zahlen sind bereits in
      der Analyse vorhanden, es muss nur der data_collector entsprechend angepasst werden
    - [x] Grid search Diagramm vergisst den letzten Punkt. (IRR War nicht gesetzt durch unprofitablen Case)
    - [x] LLM Text
        - [x] Wetterjahre miteinander vergleichen, nicht nur Sommer/winter
        - [x] EEG Standard Abweichung fehlt im Prompt
- [ ] CSV Export
    - [x] pv production in CSV Cashflow ist falsch
    - [ ] baseload constraints mit ausgeben
- [ ] Smoke Test fehlende JSON Objekte hinzufügen