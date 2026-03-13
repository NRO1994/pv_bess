- [ ] (CLAUDE) MC Framework anpassen, Plan5 1-4
- [ ] (CLAUDE) Smoke Test implementation (Requirements in FIXES), FIX-S2-02
- [ ] (CLAUDE) Analyse soll einen Durchlauf mit zentralem Preis-Szenario und Direktvermarktung über die
  gesamte Laufzeit berechnen. Dies soll in jedem Diagramm (EEG, PPA-Collar, PPA-Baseload) als konstanter Vergleichswert
  angezeigt werden, um den Upside zu den Vermarktungsstrukturen besser identifizieren zu können. Den IRR daraus soll
  auch in dem LLM-Text der "Szenario-Übersicht" erwähnt werden
- [x] Equity IRR passt nicht
- [ ] Warum ist eine Vergrößerung der BESS Kapazität ein schlechterer IRR?
- [ ] baseload MUSS in die LP Optimierung aufgenommen werden, wenn es sich um einen PV/BESS Case handelt. PLan5 1-3
- [ ] PaP Preise in Integration suite stimmen nicht
- [x] Financial Model integration Test: Logik ist falsch, mit Steffen klären
- [ ] Report
    - [ ] Leaflet wird nicht richtig angezeigt
    - [ ] Heder Input_wizard/Dashboard sind nicht identisch
    - [ ] Format der Tabs soll vom Input_wizard übernommen werden
    - [ ] Das hervorheben des ersten Jahres in der Cashflow Analyse ist unnötig, entferne den Orangen Kasten
    - [ ] die Sensitivität im EEG-Tab soll nicht die Standardabweichung um den Mittelwert abbilden, sondern der orange
      Bereich zwischen P10 und P90 aufgespannt werden. Die Linie soll dann der Median sein. Alle Zahlen sind bereits in
      der Analyse vorhanden, es muss nur der data_collector entsprechend angepasst werden
    - [x] Grid search Diagramm vergisst den letzten Punkt. (IRR War nicht gesetzt durch unprofitablen Case)
    - [ ] LLM Text
        - [x] Wetterjahre miteinander vergleichen, nicht nur Sommer/winter
        - [ ] EEG Standard Abweichung fehlt im Prompt
- [ ] CSV Export
    - [x] pv production in CSV Cashflow ist falsch