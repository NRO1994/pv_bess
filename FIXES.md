# Bekannte Fehler und Inkonsistenzen

## Integration
- [ ] Die Klasse BessState beinhaltet die Logik der BESS. Diese wird jedoch in den LP-Optimierungsklassen nicht genutzt. #TODO!!!
- [ ] Es gibt unterschiedlichste Stellen in der codebase, die Inflation berechnen. Dies soll eine einheitliche klasse geben, die immer mit der gleichen Rate(user Input) die Inflationen im gesamten MOdell rechnet.
- [ ] BESS und PV sollen dieselbe Degradation Logik teilen

## Logik
- [ ] Bei einem PPA mit Baseload-Struktur ist der Baseload ein User-Input. Er bedarf keiner Berechnungslogik. Die einzig relevante Prüfung ist, ob er kleiner als die Nennleistung der PV Anlage ist.
- [ ] Bei einem PPA mit Baseload-Struktur, muss die Leistung, die über dem baseload liegt, für den Spotpreis verkauft werden. Ist die Baseload bei 5MW und der PPA Preis bei 50EUR/MWh, die Leistung zu der gegebenen Stunde jedoch bei 10MW, sollen die restlichen MWh für den Spotpreis (z.B. 60EUR/MWh) verkauft werden. Der Ertrag in dieser Situation ist also (5MWh * 50EUR/MWh + 5MW * 60EUR/MWh) = 550EUR
- [ ] Die Herkunftsnachweise (Guarantees of origin) werden immer auf den finalen PPA-Preis (oder Direktvermarktungspreis) hinzugerechnet. Nicht nur auf den floor wie aktuell beim collar mode
- [ ] Die Inflation soll erst im zweiten Jahr der Betrachtung angerechnet werden. Alle kommerziellen Inputwerte sind im Basisjahr gegeben.
- [ ] CAPEX und OPEX fallen beide im ersten Jahr an. Es wird ein Betriebsbeginn am 1.1. des Inbetriebnahmejahres angenommen. Dieses Jahr muss als Userinput mitaufgenommen werden.
- [ ] Beginnt die Preiszeitreihe aus der CSV Datei früher als das Inbetriebnahme Jahr, so sind alle vorherigen Jahre zu vernachlässigen.
- [ ] Es sollen alle drei Preiszeitreihen verlängert werden, nicht nur das "MID" Szenario.

## Kosmetik
- [ ] timeseries.percentile_timeseries ist nicht notwendig. Es werden außer P50/P90 keine weiteren Perzentile berechnet #TODO!!!
- [ ] Die Enums der PPA Strukturtags sind nicht teil der config/defaults.py Datei. Suche nach weiteren Enums, die ebenfalls nicht zentral gespeichert sind.
- [ ] main.py enthält magic numbers, die in config/defaults.py definiert sind.
- [ ] Erstelle einen Integrationtest, der den fetch vom PVGIS testet. 
- [ ] Passe den pytest Aufruf entsprechend an, dass die integrationstests nicht im Standardablauf enthalten sind. Diese sollen immer separat (und nur vom developer, nicht von claude!) getriggert werden.
- [ ] Alle Dateien mit (Test-)Daten sollen im Verzeichnis /.data abgelegt werden. Dies gilt insbesondere für die Testdateien, die im Rahmen der Unittests erstellt werden. Stelle zudem sicher, dass die Logik auch auf einem Windows PC läuft.
- [ ] Passe die CLAUDE.md mit deinen Änderungen an. Dies sind zum Beispiel die hier angesprochenen Anpassungen, aber auch Designentscheidungen wie das weglassen der revenue-hilfsvariablen in der LP Optimierung. Prüfe auch auf weitere Anpassungen, sodass die CLAUDE.md das korrekte Metadokument dieses Projektes bleibt
- [ ] Verwende den CSV_DELIMITER auch in allen Unit tests. Er soll zudem auf Semikolon umgestellt werden.
- [ ] der pv cache soll ebenfalls im .data Verzeichnis im project liegen
- [ ] der year index im CSV Cashflow output soll sich am Inbetriebnahmejahr orientieren und nicht von 0 starten
- [ ] Logge den Fortschritt über die Iterationen durch die grid-search und das MC ein. Dieser soll nur im Debug-Mode sichtbar sein