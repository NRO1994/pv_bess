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