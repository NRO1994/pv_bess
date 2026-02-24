## Neue Features und andere Anpassungen
- Ertrags-/Preismodelling: Die Preisszenarien, hängen mit fest definierten Wetterjahren zusammen. So soll zum Beispiel 
im Low-Case das Jahr 2021 verwendet werden. Diese Abhängigkeiten sollen als User-Input Teil der Json Datei sein. Es 
bedarf daher keiner P50/P90 Berechnung aus historischen Daten, sondern nur noch dem Download der entsprechenden Jahre. 
Zudem ist die Auflösung von stündlich (8760 Werte) auf 15min Intervall verkleinert (35040 Werte). Die PV-Erträge 
sollen gleichverteilt auf diese Intervalle aufgeteilt werden (durch 4 teilen). Im MonteCarlo Framework ist nun der 
PV-Ertragsfaktor zu entfernen, und durch einen Verfügbarkeitsparameter zu ersetzen. Somit sollen BESS und PV dieselbe Offline-Logik teilen.
- Das Mapping der Preise auf das Wetterjahr ist wie folgt zu Implementieren: Das Preisprognosejahr dient als Basis. 
Beginnt dieses zum Beispiel an einem Dienstag, so ist die Ertragszeitreihe des Wetterjahres so zu verschieben, 
dass diese an dem ersten Dienstag des Jahres beginnt. Die Tage vorher sollen an das Ende des Jahres verschoben werden. 
Ist eines der beiden Jahre ein Schaltjahr, so ist der 31.12. zu ignorieren. 
Beispiel: Prognosejahr 2030, Wetterjahr 2017. 
01.01.2017 ist ein Sonntag, 01.01.2030 ist ein Dienstag.
03.01.2017 wird auf den 01.01.2030 gemappt. 
Entsprechend wird der 01./02.01.2017 auf den 30./31.12.2030 gemappt.
- Zusätzlich sind nun 9 (anstatt drei) Szenarien zu berücksichtigen. Von Low/Bad Weather, bis zu High/Good Weather gibt es unterschiedliche Forward-Curves,
jede hat ein definiertes Wetterjahr, das bei dessen Verwendung im Monte-Carlo Verfahren den Ertrag der PV Anlage bestimmt. 
Eines der Preisszenarien soll das Flag "Central" halten. Dieses ist als Basis für die Grid-Search zu verwenden
- Nach der Grid Search ist eine Logik zu implementieren, die die folgenden Fragestellungen beantwortet. Es sollen in 
allen Berechnungen Monte-Carlo Simulationen für jedes Set-Up ausgeführt werden. Jede Analyse methode soll in einer CSV Datei münden, 
die die Daten zur weiteren Verarbeitung bereitstellt. Solange die gesamte Laufzeit aller Berechnungen für alle Analysen kleiner als 60h ist, 
benötigt es keiner weiteren Runtime Optimierungen. Falls die Berechnung länger dauert, kann dies einfach über einen anderen User Input der MC Runs beschleunigt werden.
  1. Um wie viel Prozent sinkt der IRR, wenn der Zuschlagspreis im EEG sinkt?
  2. Wie muss ein PPA-Collar Modell designt sein, um sowohl uns (Dem Erzeuger) als auch unserem Kunden den größten Mehrwert bietet? 
  Wie hoch ist dann im besten Fall der IRR? Als PPA Laufzeit soll ein fixer Parameter (User Input) berücksichtigt werden.
  3. Wie muss der PPA-Baseload designt sein, damit uns (Dem Erzeuger) als auch unserem Kunden den größten Mehrwert bietet? 
  Wie hoch ist dann im besten Fall der IRR? Als PPA Laufzeit soll ein fixer Parameter (User Input) berücksichtigt werden.
- Der Output des Modells soll ebenfalls erweitert werden. Es sollen immer noch alle CSV-Dateien exportiert werden, 
um die Rohdaten für weitere Berechnung zur Verfügung zu haben. Zudem soll nun jedoch ein PDF-Report im Format einer Präsentation erstellt werden. 
Der Stil soll professionell und im corporate Look mit den Farben Level 1: #FF8200, Level 2: #F73E5E, Level 3: #A51BA7, Level 4: #00467A, Level 5: #006EB2, Level 6: #00BDDC. 
Im Allgemeinen sollen alle Texte durch eine LLM-API generiert werden. Der API-Call soll so kosteneffizient wie möglich gestaltet werden.
Alle Diagramme sollen auf der Maschine erzeugt werden und ebenfalls als PNG gespeichert werden. Die LLM-API soll nur den Text liefern, 
alles andere soll per Template auf der Maschine selbst stattfinden. die API soll per cli ausgeschaltet werden können, dann sollen nur CSV und PNG gespeichert werden. Der Report soll wie folgt zusammengestellt werden. 
  - Deckblatt: FIRMENNAME | Projekt-Name | Datum | Model version | Firmenlogo (local als png im Data Ordner zu finden)
  - Seite 0: Beschreibung des Modells, Erklärung der generellen Vorgehensweise. Zudem soll etwas detaillierter das Finanz-Modell erklärt werden. 
  Dies soll nur ein Text sein, der sich auf die CLAUDE.md beziehen soll. 
  - Seite 1: Beschreibung des relevanten Inputs. Dies sind zum einen die wichtigsten Parameter aus der JSON Datei.
  - Seite 2: Beschreibung der PV-Ertragsberechnung. Es soll ein Diagramm enthalten sein, dass das Ertragsprofil über das Jahr zeigt. 
  Es sollen alle berücksichtigten Wetterjahre dargestellt werden, sodass auch die Unsicherheit deutlich wird. Neben dem Diagramm soll ein erklärender Text zu sehen sein.
  - Seite 3: Beschreibung der Strompreisszenarien. Es soll ein Diagramm enthalten, dass alle Strompreisszenarien über den Zeitverlauf dargestellt werden. 
  Dazu soll der Mittelwert pro Jahr dargestellt werden. Hier soll ebenfalls ein erklärender Text die wesentlichen Unterschiede erklären.
  - Seite 4: Grid-Search Analyse. Es soll ein Diagramm enthalten, dass auf der x-Achse die Leistung prozentual von der PV Anlage darstellt. 
  Auf der Y-Achse soll der IRR dargestellt werden. Es soll eine Kurvenschar angezeigt werden, die eine Kurve pro Kapazität (1h, 2h, 4h, ...) enthält. 
  Auch hier soll ein erklärender Text die wesentlichen Erkenntnisse beschreiben.
  - Seite 5: EEG-Zuschlags Analyse: Diagramm x-Achse Zuschlagspreis, y-Achse IRR. eine Kurve für mittleren IRR pro Preis. 
  Zudem angedeutete Spreizung durch Std. Abweichung pro Preis. Auch hier ein erklärender Text, mit den wesentlichen Erkenntnissen der Analyse
  - Seite 6: PPA-Collar Modell Analyse. Diagramm, x-Achse: PPA-Floor, y-Achse: IRR, Kurven schar für Cap (+2/5/X EUR/MWh). 
  Auch hier ein erklärender Text, mit den wesentlichen Erkenntnissen der Analyse
  - Seite 7: PPA-Baseload Modell. Diagramm: x-Achse PPA-Preis, Y-Achse IRR, Kurvenschar für verschiedene Baseloads (1/2/X MW). 
  Auch hier soll ein erklärender Text der wesentlichen Erkenntnisse der Analyse enthalten sein.
  - Seite 8: Finale Betrachtung der Analyse, Vorschlag für das beste Vermarktungsmodell, Vorschlag für weitere Analysen oder Berechnungen mit anderen Größen
- Systemverluste im Netzanschluss und nicht im PV Asset. Die Ertragszeitreihe von PV-GIS soll mit 0% Verlusten heruntergeladen werden. 
Die Systemverluste der PV-Anlage sind in der grünen Netzeinspeisung zu berücksichtigen. Die Round-trip losses des BESS sollen im Graustrom der Rückeinspeisung vom BESS in das Netz berücksichtigt werden. 
- Als weiteren OPEX Kostenpunkt bei dem BESS soll die optimierungsdienstleistung berücksichtigt werden. Sie ist ein Prozentsatz auf den Ertrag des Speichers. 
Als Speicher Ertrag soll die Einspeisung vom BESS in das Netz angenommen werden, mulitpliziert mit dem jeweiligen Sport-Preis.
Dies trifft auf Grün sowie Grau strom zu. Der Prozentsatz ist User-Input.
- Das Replacement des BESS soll im cAPEX berücksichtigt werden. Zudem soll die Abschreibung dann erneut starten, und wiederum 
(wie zu Beginn beim initialien CAPEX) auch wieder in der Cashflow Berechnung berücksichtigt werden.
- Die OPEX Felder sollen zusätzlich auch die Felder "eur_pro_kw" und "eur_pro_kwh" berücksichtigen.
- BESS Verfügbarkeiten von weniger als 97% müssen auch im MC nicht berücksichtigt werden, da dann eine Entschädigung vom Hersteller aussteht. 
Die MC Simulation soll die Verfügbarkeit also zwischen 97% (Userinput) und 100% variieren.
- Die MC Framework soll erweitert werden: Die Capex und OPEX Sensitivität auf PV und BESS ist unterschiedlich. 
- PPA Collar funktioniert nicht, da nur der PPA Floor gesetzt wird und nicht der Cap. (main.py)
2 changes: 1 addition & 1 deletion 2
pv_bess_model/main.py
