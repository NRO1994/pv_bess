# Benutzerhandbuch – PV + BESS Co-Location Finanzmodell

## 1. Was macht dieses Tool?

### 1.1 Zweck
- Bewertung der wirtschaftlichen Sinnhaftigkeit einer Kombination von Photovoltaik (PV) mit Batteriespeicher (BESS)
- Beantwortung der Fragen:
  - Welche Batteriegröße ist für eine gegebene PV-Anlage optimal?
  - Welche Vermarktungsstrategie (EEG, PPA, Markt) maximiert die Eigenkapitalrendite?
  - Wie wirken sich Risikofaktoren (Preisschwankungen, Verfügbarkeit, Kostenunsicherheit) auf die Rendite aus?

### 1.2 Grundprinzip
- Das Tool simuliert den Betrieb einer PV+BESS-Anlage über die gesamte Projektlaufzeit (z.B. 25 Jahre)
- Es berechnet für jede Viertelstunde des Jahres die optimale Einsatzstrategie des Batteriespeichers
- Alle Kosten, Erlöse, Steuern und Finanzierungsstrukturen werden jahresscharf abgebildet
- Am Ende steht eine Eigenkapitalrendite (Equity IRR) als zentrale Bewertungskennzahl

### 1.3 Ablauf einer Berechnung
- Der Nutzer erstellt eine Szenario-Datei (JSON) mit allen Eingabeparametern
- Das Tool führt folgende Schritte automatisch aus:
  1. Laden und Validieren der Eingabedaten
  2. Abruf historischer PV-Erzeugungsdaten von PVGIS (EU-Datenbank)
  3. Laden der Strompreiszeitreihen aus CSV-Dateien
  4. **Grid Search**: Systematische Suche nach der optimalen Batteriegröße
  5. **Sensitivitätsanalysen**: Analyse verschiedener Vermarktungsparameter (optional)
  6. **Monte Carlo**: Risikoanalyse mit 1.000+ Zufallsszenarien (optional)
  7. Erstellung von CSV-Ergebnisdateien und interaktivem HTML-Dashboard

---

## 2. Eingabedaten

### 2.1 Szenario-Datei (JSON)
- Zentrale Konfigurationsdatei, die alle Parameter eines Szenarios enthält
- Jedes Szenario = eine JSON-Datei
- Um Szenarien zu vergleichen: mehrere JSON-Dateien erstellen und separat ausführen
- Enthält alle Angaben zu:
  - Projektdaten (Laufzeit, Standort, Inbetriebnahmejahr)
  - Technische Parameter (PV-Leistung, Batterie-Suchraum, Netzanschluss)
  - Kosten (CAPEX, OPEX für jede Komponente)
  - Finanzierung (Fremdkapitalquote, Zinssatz, Laufzeit)
  - Vermarktung (EEG, PPA-Typ, Marktpreise)
  - Steuerliche Parameter
  - Monte-Carlo-Einstellungen
  - Sensitivitätsanalysen

### 2.2 Strompreis-CSV-Dateien
- Historische oder prognostizierte Day-Ahead-Strompreise
- Viertelstündliche Auflösung (35.040 Zeilen pro Jahr) oder stündlich (8.760 Zeilen)
- Mehrere Preisszenarien in einer Datei möglich (z.B. Spalten: `price_central`, `price_high`, `price_low`)
- Format: Semikolon-getrennt (`;`), mit Zeitstempel-Spalte
- Einheit: €/MWh (konfigurierbar)
- Mindestens ein volles Jahr erforderlich

### 2.3 PV-Erzeugungsdaten (PVGIS)
- Werden automatisch von der EU-Datenbank PVGIS heruntergeladen
- Basierend auf Standort (Breitengrad, Längengrad), Modulneigung, Ausrichtung
- Historische Wetterjahre werden pro Preisszenario zugeordnet (z.B. Zentralszenario = Wetterjahr 2018)
- Daten werden lokal zwischengespeichert (kein erneuter Download bei Wiederholung)

---

## 3. Technische Konfiguration

### 3.1 PV-Anlage
- **Spitzenleistung** (`peak_power_kwp`): Nennleistung der PV-Anlage in kWp
- **Montage** (`mounting_type`): Freiflächenanlage (`free`) oder Dachanlage (`building`)
- **Ausrichtung** (`azimuth_deg`): Kompassrichtung der Module (0° = Süd, -90° = Ost, 90° = West)
- **Neigung** (`tilt_deg`): Aufstellwinkel der Module in Grad
- **Degradation** (`degradation_rate_pct_per_year`): Jährlicher Leistungsverlust der Module in Prozent (z.B. 0,5%/Jahr)
- **Verfügbarkeit** (`pv_availability_pct`): Anteil der Zeit, in der die PV-Anlage tatsächlich in Betrieb ist (z.B. 99%)
  - Berücksichtigt Wartung, Störungen, Netzabschaltungen
  - An Ausfalltagen wird keine PV-Erzeugung simuliert

### 3.2 Batteriespeicher (BESS)

#### 3.2.1 Dimensionierungsraum (Design Space)
- Die Batteriegröße wird nicht fest vorgegeben, sondern als **Suchraum** definiert
- Zwei Dimensionen:
  - **Scale** (`scale_pct_of_pv`): Batterieleistung als Prozent der PV-Leistung
    - Beispiel: PV = 50.000 kWp, Scale = 50% → Batterieleistung = 25.000 kW
    - Liste von Werten, z.B. `[0, 25, 50, 75, 100, 150]`
    - 0% = Referenzfall nur PV, ohne Batterie
  - **E/P-Ratio** (`e_to_p_ratio_hours`): Verhältnis von Speicherkapazität zu Leistung in Stunden
    - Bestimmt die Speicherdauer (wie lange die Batterie bei voller Leistung entladen kann)
    - Beispiel: Leistung = 25.000 kW, E/P = 2h → Kapazität = 50.000 kWh
    - Liste von Werten, z.B. `[2, 4]`
- Das Tool berechnet alle Kombinationen aus Scale × E/P-Ratio

#### 3.2.2 Performance-Parameter
- **Round-Trip-Efficiency** (`round_trip_efficiency_pct`): Wirkungsgrad des Speichers über einen vollen Lade-/Entladezyklus (z.B. 88%)
  - Verluste fallen nur beim Entladen an
  - Laden ist verlustfrei (1 kWh geladen = 1 kWh im Speicher)
  - Beim Entladen: Netzeinspeisung = entladene kWh × Wirkungsgrad
- **SoC-Grenzen**: Minimaler und maximaler Ladezustand
  - `min_soc_pct`: z.B. 10% – Speicher wird nie unter 10% entladen
  - `max_soc_pct`: z.B. 95% – Speicher wird nie über 95% geladen
  - Schützt die Batterie vor Tiefentladung und Überladung
- **Degradation** (`degradation_rate_pct_per_year`): Jährlicher Kapazitätsverlust der Batterie (z.B. 2%/Jahr)
  - Reduziert die nutzbare Kapazität linear über die Projektlaufzeit
- **Verfügbarkeit** (`bess_availability_pct`): Anteil der Tage, an denen die Batterie einsatzbereit ist (z.B. 98%)
  - An Ausfalltagen: keine Lade-/Entladevorgänge, nur direkter PV-Export

#### 3.2.3 Batterieersatz (Replacement)
- Optional: Austausch der Batterie nach einer definierten Anzahl von Jahren (z.B. nach 12 Jahren)
- **Kapazitätsfaktor** (`capacity_factor_pct`): Kapazität der Ersatzbatterie relativ zum Original
  - 100% = gleiche Kapazität
  - 80% = kleinere Batterie (z.B. wegen besserer Technologie reicht weniger)
  - 120% = größere Batterie (Technologie-Upgrade)
- Kosten des Ersatzes folgen dem gleichen Schema wie die Erstinvestition
- Nach dem Ersatz beginnt die Degradation der neuen Batterie von vorne

### 3.3 Netzanschluss
- **Maximale Einspeiseleistung** (`max_export_kw`): Obergrenze der Leistung, die ins Netz eingespeist werden kann
  - Alles was darüber hinausgeht, wird abgeregelt (curtailed) oder im Speicher gespeichert
- **Systemverluste** (`system_loss_pct`): Verluste am Netzanschlusspunkt in Prozent (z.B. 2,5%)
  - Reduzieren die tatsächlich eingespeiste und vergütete Energiemenge
  - Werden nur auf grüne Energie angewendet (PV-Direkteinspeisung und grüne BESS-Entladung)

### 3.4 Betriebsmodus

#### 3.4.1 Green Mode (Grünstrom)
- Batterie wird **ausschließlich mit PV-Überschuss** geladen
- Kein Laden aus dem Netz
- Gesamte Entladung gilt als grüner Strom
- Einfachere Optimierung, eine SoC-Spur

#### 3.4.2 Grey Mode (Graustrom)
- Batterie kann **zusätzlich aus dem Netz** geladen werden (zu Marktpreisen)
- Getrennte Nachverfolgung von grünem und grauem Strom im Speicher (Dual-Chamber-Modell)
- Grüner Strom = aus PV geladen → erhält GoO-Prämie und ggf. PPA/EEG-Vergütung
- Grauer Strom = aus dem Netz geladen → wird nur zum Marktpreis verkauft
- Ermöglicht Arbitrage-Geschäfte (billig laden, teuer entladen)

---

## 4. Vermarktungsstrategien

### 4.1 Überblick
- Das Tool unterstützt verschiedene Vermarktungsmodelle, die kombiniert werden können
- Hauptunterscheidung: EEG-Einspeisevergütung vs. PPA (Power Purchase Agreement) vs. reiner Markt

### 4.2 EEG-Einspeisevergütung
- Funktioniert als **Mindestpreis (Floor Price)**, nicht als Festpreis
- In jeder Viertelstunde gilt: `Erlös = max(Spotpreis, EEG-Tarif)`
- Wenn der Marktpreis über dem EEG-Tarif liegt: Erlös zum Marktpreis
- Wenn der Marktpreis unter dem EEG-Tarif liegt: Erlös zum EEG-Tarif
- **Laufzeit begrenzt**: z.B. 20 Jahre, danach reine Marktvermarktung
- **Inflation optional**: EEG-Tarif kann jährlich mit der Inflationsrate angepasst werden

### 4.3 PPA-Modelle (Power Purchase Agreement)

#### 4.3.1 Pay-as-Produced PPA
- Käufer zahlt einen festen Preis pro tatsächlich erzeugter kWh
- Einfachstes PPA-Modell
- Preis kann optional jährlich inflationsangepasst werden

#### 4.3.2 Baseload PPA
- Verkäufer verpflichtet sich, ein gleichmäßiges Leistungsprofil zu liefern (z.B. 5 MW konstant)
- **Baseload-Level** muss vom Nutzer vorgegeben werden (in MW)
- Wenn PV-Erzeugung < Baseload → Fehlmenge wird am Markt zugekauft (Zusatzkosten)
- Wenn PV-Erzeugung > Baseload → Überschuss wird am Markt verkauft (Zusatzerlös)
- Batterie hilft beim Profil-Shaping (Überschuss speichern, Fehlmenge ausgleichen)

#### 4.3.3 Floor PPA (Mindestpreis-PPA)
- Garantierter Mindestpreis, Verkäufer behält Upside über dem Floor
- Gleiche Logik wie EEG: `Erlös = max(Spotpreis, PPA-Floor) + GoO-Prämie`
- GoO-Prämie = Herkunftsnachweis-Aufschlag für grünen Strom

#### 4.3.4 Collar PPA (Korridor-PPA)
- Preiskorridor mit Untergrenze (Floor) und Obergrenze (Cap)
- `Erlös = clip(Spotpreis, Floor, Cap) + GoO-Prämie`
- Spotpreis unter Floor → Erlös zum Floor-Preis
- Spotpreis zwischen Floor und Cap → Erlös zum Spotpreis
- Spotpreis über Cap → Erlös zum Cap-Preis
- Schützt vor Niedrigpreisen, begrenzt aber auch das Upside

#### 4.3.5 Gemeinsame PPA-Eigenschaften
- **Laufzeit** (`duration_years`): Vertragslaufzeit in Jahren; nach Ablauf → reine Marktvermarktung
- **Inflation**: Optional, jährliche Anpassung aller Preiskomponenten (Floor, Cap, PPA-Preis)
- **GoO-Prämie** (`guarantee_of_origin_eur_per_kwh`): Aufschlag für Herkunftsnachweise, wird nach Floor/Cap-Berechnung addiert

### 4.4 Marketing-Typ
- Bestimmt den Grundmechanismus der Vermarktung:
  - `eeg`: EEG-Einspeisevergütung als Mindestpreis
  - `ppa`: PPA-Vertrag (Typ wird separat gewählt)
  - `market`: Reine Marktvermarktung zum Day-Ahead-Spotpreis

---

## 5. Preisszenarien und Wetterjahre

### 5.1 Konzept
- Das Tool koppelt Strompreisszenarien mit PV-Wetterjahren
- Jedes Preisszenario hat ein zugehöriges Wetterjahr
- Beispiel:
  - Zentralszenario: Preise "Central" + Wetterjahr 2018 (normales Sonnenjahr)
  - Hochpreisszenario: Preise "High" + Wetterjahr 2015 (anderes Sonnenjahr)
  - Niedrigpreisszenario: Preise "Low" + Wetterjahr 2016 (schlechteres Sonnenjahr)

### 5.2 Gewichtung
- Jedem Szenario wird eine Wahrscheinlichkeit (Gewicht) zugeordnet
- Gewichte müssen sich auf 1,0 (= 100%) summieren
- Beispiel: Central 60%, High 25%, Low 15%
- In der Monte-Carlo-Simulation wird pro Iteration ein Szenario zufällig gemäß der Gewichte gezogen

### 5.3 Zentralszenario
- Ein Szenario muss als `is_central: true` markiert werden
- Das Zentralszenario wird für die deterministische Grid Search verwendet
- Alle Szenarien werden in der Monte-Carlo-Simulation berücksichtigt

### 5.4 CSV-Konfiguration pro Szenario
- Jedes Preisszenario kann eigene CSV-Einstellungen haben:
  - Spaltenname in der CSV-Datei (`csv_column`)
  - Pfad zur CSV-Datei (`price_csv`)
  - Trennzeichen (`csv_separator`)
  - Dezimaltrennzeichen (`csv_decimal`)
  - Inflation auf Eingabedaten (`inflation_on_input_data`)

---

## 6. Kosten und Finanzierung

### 6.1 Einheitliches Kostenschema
- Alle Kostenblöcke (PV, BESS, Netzanschluss) folgen dem gleichen Schema
- Jeder Block unterstützt vier additive Komponenten:
  - `fixed_eur`: Fester Betrag in Euro
  - `eur_per_kw`: Kosten pro kW Leistung
  - `eur_per_kwh`: Kosten pro kWh Kapazität (nur bei BESS relevant)
  - `pct_of_capex`: Prozentualer Anteil der CAPEX desselben Assets

### 6.2 CAPEX (Investitionskosten)
- Berechnung pro Asset:
  - `CAPEX = fixed_eur + eur_per_kw × Leistung + eur_per_kwh × Kapazität`
- Gesamte CAPEX = Summe aller Assets (PV + BESS + Netz)
- Fallen einmalig im Inbetriebnahmejahr an

### 6.3 OPEX (Betriebskosten)
- Berechnung pro Asset:
  - `OPEX = fixed_eur + eur_per_kw × Leistung + eur_per_kwh × Kapazität + pct_of_capex × CAPEX_Asset`
- Fallen jährlich an
- Werden ab Jahr 2 jährlich mit der Inflationsrate erhöht
- **BESS-Optimierungsgebühr** (`optimization_fee_pct`): Zusätzliche OPEX als Prozent der BESS-Spoterlöse
  - Deckt Kosten für Vermarktungs-/Optimierungsdienstleistung ab
  - Nicht inflationsbereinigt (ist bereits im Erlös enthalten)

### 6.4 Batterieersatzkosten
- Werden als **CAPEX** im Ersatzjahr behandelt (nicht als OPEX)
- Folgen dem gleichen Kostenschema (fixed_eur, eur_per_kw, eur_per_kwh)
- Erzeugen eine neue Abschreibungslinie (AfA beginnt ab Ersatzjahr)

### 6.5 Finanzierungsstruktur
- **Fremdkapitalquote** (`leverage_pct`): Anteil der CAPEX, der über Kredit finanziert wird (z.B. 60%)
- **Zinssatz** (`interest_rate_pct`): Jährlicher Zinssatz des Kredits (z.B. 4,2%)
- **Kreditlaufzeit** (`loan_tenor_years`): Laufzeit des Annuitätendarlehens in Jahren (z.B. 12)
- **Annuitätendarlehen**: Gleichbleibende jährliche Rate über die gesamte Kreditlaufzeit
  - Setzt sich zusammen aus Zins- und Tilgungsanteil
  - Zinsanteil sinkt über die Laufzeit, Tilgungsanteil steigt
- **Debt Sizing Downside** (`debt_sizing_downside_pct`): Konservativer Abschlag auf die Erlöse für die DSCR-Berechnung
  - Beispiel: 15% → DSCR wird mit nur 85% der prognostizierten Erlöse berechnet
  - Simuliert konservative Bankenperspektive

### 6.6 Inflation
- Einheitliche Inflationsrate für das gesamte Projekt (z.B. 2%/Jahr)
- Wird angewendet auf:
  - OPEX (immer)
  - PPA-Preise (wenn `inflation_on_ppa: true`)
  - EEG-Tarif (wenn `eeg_inflation: true`)
  - Eingabe-Strompreise (wenn `inflation_on_input_data: true` pro Szenario)
- Formel: `Wert[Jahr] = Basiswert × (1 + Inflationsrate)^Jahr`

---

## 7. Steuerliche Behandlung (Deutschland)

### 7.1 Lineare Abschreibung (AfA)
- **Getrennte Abschreibungszeiträume** für PV und BESS
  - PV: z.B. 20 Jahre
  - BESS: z.B. 15 Jahre
- Abschreibungsbasis = CAPEX des jeweiligen Assets
- Bei Batterieersatz: Neue AfA-Linie ab dem Ersatzjahr
- Reduziert das zu versteuernde Einkommen

### 7.2 Gewerbesteuer (GewSt)
- `GewSt = max(0, zu versteuerndes Einkommen) × Messzahl × Hebesatz / 100`
- **Messzahl**: Gesetzlich festgelegt (3,5%)
- **Hebesatz**: Gemeindeabhängig (z.B. 400%)
- Zu versteuerndes Einkommen = Erlöse - OPEX - AfA (bereinigt um Verlustvortrag)

### 7.3 Körperschaftsteuer (KSt)
- `KSt = max(0, zu versteuerndes Einkommen) × 15%`
- Pauschaler Steuersatz von 15%

### 7.4 Solidaritätszuschlag (Soli)
- `Soli = KSt × 5,5%`
- Wird auf die Körperschaftsteuer aufgeschlagen

### 7.5 Verlustvortrag
- Wenn das zu versteuernde Einkommen in einem Jahr negativ ist (z.B. wegen hoher AfA in den Anfangsjahren)
- Der Verlust wird unbegrenzt in die Zukunft vorgetragen
- Wird mit zukünftigen positiven Einkünften verrechnet, bevor Steuern berechnet werden
- Beispiel: Verlust Jahr 1 = -500.000 €, Gewinn Jahr 2 = 300.000 € → Steuerbasis Jahr 2 = 0 € (Restverlust 200.000 € wird weiter vorgetragen)

---

## 8. Dispatch-Optimierung (Einsatzplanung)

### 8.1 Grundprinzip
- Für jeden Tag des Jahres wird ein Optimierungsproblem gelöst
- **96 Zeitschritte pro Tag** (Viertelstundenauflösung)
- **Perfekte Voraussicht**: Der Optimierer kennt die Preise und PV-Erzeugung des gesamten Tages
- Ziel: Maximierung des täglichen Erlöses

### 8.2 Entscheidungsvariablen (pro Viertelstunde)
- Wieviel PV-Strom direkt ins Netz eingespeist wird
- Wieviel PV-Strom in die Batterie geladen wird
- Wieviel PV-Strom abgeregelt wird (Curtailment)
- Wieviel aus der Batterie entladen wird (grüner Strom)
- Im Grey Mode zusätzlich:
  - Wieviel aus dem Netz in die Batterie geladen wird
  - Wieviel grauer Strom entladen wird

### 8.3 Nebenbedingungen (Einschränkungen)
- **Energiebilanz**: Gesamte PV-Erzeugung muss aufgeteilt werden (Export + Laden + Abregelung = Erzeugung)
- **SoC-Grenzen**: Ladezustand muss zwischen min und max bleiben
- **Leistungsgrenzen**: Lade- und Entladeleistung der Batterie dürfen nicht überschritten werden
- **Netzanschlussgrenze**: Gesamte Einspeisung (PV + Batterie) darf die maximale Einspeiseleistung nicht überschreiten
- **Netzverluste**: Grüne Einspeisung wird um den Verlustfaktor reduziert
- **Green Mode**: Kein Laden aus dem Netz erlaubt

### 8.4 SoC-Kopplung zwischen Tagen
- Am ersten Tag des Projekts: Batterie startet bei 50% Ladezustand
- An allen Folgetagen: Startzustand = Endzustand des Vortags
- Kein Zwang zum Vollladen/Entladen am Tagesende – der Optimierer entscheidet frei

### 8.5 Ausfalltage
- An BESS-Ausfalltagen: Keine Lade-/Entladevorgänge, nur PV-Direkteinspeisung
- An PV-Ausfalltagen: Keine PV-Erzeugung
- Ladezustand der Batterie wird am Ausfalltag eingefroren

---

## 9. Grid Search (Optimierung der Batteriegröße)

### 9.1 Funktionsweise
- Systematische Durchrechnung aller Kombinationen aus Batterieleistung und Speicherdauer
- Für jede Kombination wird die gesamte Projektlaufzeit simuliert (z.B. 25 Jahre × 365 Tage × 96 Viertelstunden)
- Verwendet das Zentralszenario (Preise + Wetter)

### 9.2 Bewertungskriterium
- Zentrale Kennzahl: **Equity IRR** (Eigenkapitalrendite)
- Die Kombination mit der höchsten Equity IRR wird als optimal identifiziert
- Ergebnisse werden als 2D-Matrix dargestellt (Scale × E/P-Ratio)

### 9.3 PV-Only Baseline
- Standardmäßig wird Scale = 0% (nur PV, keine Batterie) als Referenzfall berechnet
- Kann mit `skip_baseline: true` übersprungen werden
- Dient zum Vergleich: Lohnt sich die Batterie überhaupt?

### 9.4 Parallelisierung
- Jede Kombination wird unabhängig berechnet
- Berechnung wird auf mehrere CPU-Kerne verteilt
- Typische Rechenzeit: 5–15 Minuten für 16 Kombinationen bei 25 Jahren Projektlaufzeit

---

## 10. Monte-Carlo-Simulation (Risikoanalyse)

### 10.1 Zweck
- Quantifizierung der Unsicherheit in den Ergebnissen
- Wie stark schwankt die Rendite bei Änderung der Eingabeparameter?
- Welche Bandbreite ist realistisch?

### 10.2 Ansatz
- Wird nur auf die optimale Konfiguration aus dem Grid Search angewendet
- **Effizientes Verfahren**: Dispatch wird nur einmal pro Preisszenario berechnet (nicht pro MC-Iteration)
- Unsicherheitsfaktoren werden nachträglich (post-hoc) auf die vorberechneten Ergebnisse angewendet
- Dadurch ~1.000× schneller als naive Berechnung

### 10.3 Unsicherheitsfaktoren
- **PV-CAPEX** (`sigma_capex_pv_pct`): Unsicherheit in den PV-Investitionskosten
- **BESS-CAPEX** (`sigma_capex_bess_pct`): Unsicherheit in den Batterie-Investitionskosten
- **PV-OPEX** (`sigma_opex_pv_pct`): Unsicherheit in den PV-Betriebskosten
- **BESS-OPEX** (`sigma_opex_bess_pct`): Unsicherheit in den Batterie-Betriebskosten
- **PV-Verfügbarkeit** (`sigma_pv_availability_pct`): Schwankung der PV-Verfügbarkeit
- **BESS-Verfügbarkeit** (`sigma_bess_availability_pct`): Schwankung der Batterie-Verfügbarkeit
- Alle Faktoren sind unabhängig voneinander (keine Korrelation)
- Werte als Standardabweichung einer Normalverteilung (z.B. 5% = ±5% um den Erwartungswert)

### 10.4 Preisszenario-Auswahl
- In jeder Iteration wird zufällig ein Preisszenario gezogen
- Gemäß den definierten Gewichten (z.B. Central 60%, High 25%, Low 15%)
- Unterschiedliche Preisszenarien bringen unterschiedliche Erlöse und PV-Erzeugungsprofile

### 10.5 Ergebnisse
- **Verteilungsstatistiken** für jede Kennzahl:
  - Mittelwert, Median, Standardabweichung
  - Perzentile: P10, P25, P50, P75, P90
- **Aufschlüsselung nach Preisszenario** (bedingte Verteilungen)
- Standard: 1.000 Iterationen (konfigurierbar)

---

## 11. Sensitivitätsanalysen (Post-Grid-Search)

### 11.1 Überblick
- Werden nach dem Grid Search auf die optimale Konfiguration angewendet
- Untersuchen die Auswirkung verschiedener Vermarktungsparameter auf die Rendite
- Für jeden getesteten Parameterwert wird eine vollständige Monte-Carlo-Simulation durchgeführt
- Drei Analysetypen verfügbar (jeweils optional aktivierbar)

### 11.2 EEG-Sensitivität
- Variation des EEG-Mindestpreises über einen Bereich (z.B. 5–9 ct/kWh)
- Ergebnis: Wie verändert sich die Equity IRR bei verschiedenen EEG-Tarifen?
- Nützlich zur Bewertung von Ausschreibungsergebnissen

### 11.3 PPA-Collar-Analyse (2D)
- Variation von zwei Parametern gleichzeitig:
  - **Floor-Preis**: Mehrere Mindestpreise (z.B. 4,0 / 5,5 / 7,0 ct/kWh)
  - **Cap-Spread**: Abstand zwischen Floor und Cap (z.B. 2,0 / 3,0 ct/kWh)
  - Cap = Floor + Spread
- Ergebnis: 2D-Matrix der Equity IRR für jede Kombination
- Nützlich zur Optimierung von PPA-Vertragsparametern

### 11.4 PPA-Baseload-Analyse (2D)
- Variation von zwei Parametern gleichzeitig:
  - **PPA-Preis**: Verschiedene Festpreise (z.B. 6,0 / 7,0 / 8,0 ct/kWh)
  - **Baseload-Level**: Verschiedene Bandlieferungsmengen (z.B. 5,0 / 7,5 MW)
- Ergebnis: 2D-Matrix der Equity IRR für jede Kombination

---

## 12. Ergebnisse und Kennzahlen

### 12.1 Finanzielle Kennzahlen

#### Equity IRR (Eigenkapitalrendite)
- **Wichtigste Kennzahl** des Modells
- Rendite auf das eingesetzte Eigenkapital nach Steuern und Schuldendienst
- Berücksichtigt den Zeitwert des Geldes (interner Zinsfuß)
- Interpretation: "Welche jährliche Verzinsung erzielt das eingesetzte Eigenkapital?"

#### Project IRR (Projektrendite)
- Rendite auf das Gesamtprojekt (vor Fremdkapital)
- Zeigt die wirtschaftliche Qualität des Projekts unabhängig von der Finanzierungsstruktur

#### NPV (Kapitalwert)
- Barwert aller zukünftigen Zahlungsströme, abgezinst mit dem Diskontierungssatz
- Positiver NPV = Projekt ist wirtschaftlich sinnvoll
- Negativer NPV = Projekt vernichtet Wert

#### DSCR (Debt Service Coverage Ratio)
- Verhältnis von verfügbarem Cashflow zum Schuldendienst
- `DSCR = (Erlöse - OPEX) / Schuldendienst`
- DSCR > 1,0: Schuldendienst ist gedeckt
- DSCR < 1,0: Erlöse reichen nicht für den Schuldendienst
- Wird mit konservativem Abschlag berechnet (debt_sizing_downside_pct)
- Ausgegeben als: Minimum-DSCR und Durchschnitts-DSCR über die Kreditlaufzeit

#### LCOE (Stromgestehungskosten)
- Gesamte Kosten geteilt durch gesamte Produktion über die Projektlaufzeit
- Einheit: €/kWh
- "Was kostet eine kWh Strom aus dieser Anlage?"

#### Payback Period (Amortisationszeit)
- Jahr, in dem der kumulierte Eigenkapital-Cashflow erstmals positiv wird
- "Wann hat sich das eingesetzte Eigenkapital amortisiert?"

#### Capture Rate (Erlösrate)
- Durchschnittlicher Erlös pro kWh eingespeister Energie
- Einheit: €/kWh
- Zeigt, wie gut die Vermarktungsstrategie funktioniert

### 12.2 Energiekennzahlen
- **PV-Produktion**: Gesamte Stromerzeugung der PV-Anlage (vor Verlusten) in MWh
- **BESS-Durchsatz**: Gesamte Energiemenge, die durch die Batterie geflossen ist (Laden + Entladen)
- **Curtailment**: Abregelungsmenge – Strom, der weder eingespeist noch gespeichert werden konnte
- **Netzimportkosten**: Kosten für Strom, der aus dem Netz in die Batterie geladen wurde (nur Grey Mode)

---

## 13. Ausgabedateien

### 13.1 CSV-Dateien

#### Summary CSV (`{Szenarioname}_summary.csv`)
- Eine Zeile mit allen wichtigen Ergebnissen
- Enthält: Eingabeparameter, optimale Konfiguration, alle Finanzkennzahlen

#### Cashflows CSV (`{Szenarioname}_cashflows.csv`)
- Eine Zeile pro Projektjahr
- Spalten: Kalenderjahr, PV-Produktion, BESS-Durchsatz, Erlöse (PV, BESS grün, BESS grau), Netzimportkosten, CAPEX, OPEX, Schuldendienst, Steuern (GewSt, KSt, Soli), AfA, Eigenkapital-Cashflow, kumulierter EK-CF, DSCR

#### Grid Search CSV (`{Szenarioname}_grid_search.csv`)
- Eine Zeile pro getesteter Batterie-Konfiguration
- Spalten: Scale %, E/P-Ratio, Leistung, Kapazität, CAPEX, OPEX, Erlös Jahr 1, Equity IRR, Project IRR, NPV, Ist-Optimal-Flag

#### Monte Carlo CSV (`{Szenarioname}_monte_carlo.csv`)
- Eine Zeile pro MC-Iteration
- Spalten: Iterationsnummer, Preisszenario, alle Noise-Faktoren, alle Finanzkennzahlen

#### Dispatch Sample CSV (`{Szenarioname}_dispatch_sample.csv`)
- Viertelstundenscharfe Einsatzdaten des ersten Projektjahres
- 35.040 Zeilen (ein volles Jahr in 15-Minuten-Schritten)
- Spalten: Zeitstempel, PV-Erzeugung, Spotpreis, Effektiv-Preis, SoC, Lade-/Entladevorgänge, Netzeinspeisung, Abregelung, Erlös

#### Analyse-CSVs (optional)
- `{Szenarioname}_analyses_eeg_sensitivity.csv`: EEG-Preissensitivität
- `{Szenarioname}_analyses_ppa_collar.csv`: PPA-Collar-Sweep
- `{Szenarioname}_analyses_ppa_baseload.csv`: PPA-Baseload-Sweep

### 13.2 HTML-Dashboard
- Interaktiver, einzelner HTML-Report
- Funktioniert offline (keine Internetverbindung nötig)
- Enthält alle Ergebnisse mit interaktiven Grafiken
- **7 Tabs:**
  1. Szenario-Übersicht (Parameter, Standortkarte)
  2. Eingangszeitreihen (monatliche PV-Erträge, Preisszenarien)
  3. Grid-Search-Ergebnisse (IRR-Kurven)
  4. EEG-Sensitivität (falls aktiviert)
  5. PPA-Collar-Sweep (falls aktiviert)
  6. PPA-Baseload-Sweep (falls aktiviert)
  7. Cashflow-Analyse (gestapeltes Balkendiagramm + KPIs)
- **Funktionen:**
  - Zoom, Pan und Tooltips in allen Grafiken
  - CSV- und PNG-Download-Buttons
  - Dark Mode
  - Sprache: Deutsch
- **KI-generierte Texte** (optional):
  - Das Tool erzeugt einen Prompt für ChatGPT/Copilot
  - Die KI-Antwort wird als Analysetext in den Report eingebettet
  - Kann übersprungen werden (`--skip-llm-prompt`)

---

## 14. Programmausführung

### 14.1 Voraussetzungen
- Python 3.10 oder höher
- Alle Abhängigkeiten installiert (numpy, scipy, pandas, etc.)
- Szenario-JSON-Datei erstellt
- Strompreis-CSV-Datei(en) vorhanden

### 14.2 Kommandozeilen-Befehle

#### Standard-Ausführung
```bash
python -m pv_bess_model.main --scenario pfad/zum/szenario.json
```

#### Ausgabeverzeichnis überschreiben
```bash
python -m pv_bess_model.main --scenario szenario.json --output ergebnisse/lauf_01/
```

#### Ohne Monte Carlo
```bash
python -m pv_bess_model.main --scenario szenario.json --no-mc
```

#### Feste Batteriegröße (kein Grid Search)
```bash
python -m pv_bess_model.main --scenario szenario.json --bess-power 20000 --bess-capacity 80000
```

#### Nur Validierung (kein Rechenlauf)
```bash
python -m pv_bess_model.main --scenario szenario.json --dry-run
```

#### Ausführliche Logausgabe
```bash
python -m pv_bess_model.main --scenario szenario.json -v
```

#### Ohne HTML-Report
```bash
python -m pv_bess_model.main --scenario szenario.json --no-report
```

#### Report ohne KI-Texte
```bash
python -m pv_bess_model.main --scenario szenario.json --skip-llm-prompt
```

### 14.3 Typischer Ablauf
1. Szenario-JSON und Preis-CSV vorbereiten
2. Tool starten
3. PVGIS-Daten werden automatisch heruntergeladen (beim ersten Mal)
4. Grid Search läuft (5–15 Minuten, Fortschrittsanzeige)
5. Sensitivitätsanalysen laufen (falls aktiviert)
6. Monte Carlo läuft (< 1 Minute dank optimiertem Verfahren)
7. Optional: KI-Prompt wird angezeigt → in ChatGPT/Copilot einfügen → Antwort speichern → Pfad eingeben
8. CSV-Dateien und HTML-Report werden geschrieben
9. Zusammenfassung wird auf der Konsole ausgegeben

---

## 15. Glossar

| Begriff | Erklärung |
|---------|-----------|
| **AfA** | Absetzung für Abnutzung – steuerliche Abschreibung |
| **BESS** | Battery Energy Storage System – Batteriespeichersystem |
| **CAPEX** | Capital Expenditure – Investitionskosten |
| **Collar** | Preiskorridor mit Floor (Untergrenze) und Cap (Obergrenze) |
| **Curtailment** | Abregelung – Erzeugter Strom, der weder eingespeist noch gespeichert wird |
| **Day-Ahead** | Strommarkt mit Lieferung am nächsten Tag, Handel am Vortag |
| **DSCR** | Debt Service Coverage Ratio – Schuldendienstdeckungsgrad |
| **E/P-Ratio** | Energy-to-Power Ratio – Verhältnis Speicherkapazität zu Leistung (in Stunden) |
| **EEG** | Erneuerbare-Energien-Gesetz – Vergütungsrahmen für erneuerbare Energien |
| **Equity IRR** | Eigenkapitalrendite (interner Zinsfuß auf EK-Cashflows) |
| **GewSt** | Gewerbesteuer |
| **GoO** | Guarantee of Origin – Herkunftsnachweis für grünen Strom |
| **Grid Search** | Systematische Suche durch Ausprobieren aller Kombinationen |
| **Grey Mode** | Betriebsmodus, bei dem die Batterie auch aus dem Netz geladen werden darf |
| **Green Mode** | Betriebsmodus, bei dem die Batterie nur aus PV geladen wird |
| **IRR** | Internal Rate of Return – interner Zinsfuß |
| **KSt** | Körperschaftsteuer |
| **LCOE** | Levelized Cost of Energy – Stromgestehungskosten |
| **LP** | Linear Program – mathematisches Optimierungsproblem |
| **Monte Carlo** | Simulationsverfahren mit vielen Zufallsdurchläufen zur Risikoanalyse |
| **NPV** | Net Present Value – Kapitalwert |
| **OPEX** | Operational Expenditure – Betriebskosten |
| **P50/P90** | Statistische Kennwerte: P50 = Median (50% Wahrscheinlichkeit), P90 = konservativ (90% Wahrscheinlichkeit, dass dieser Wert erreicht oder übertroffen wird) |
| **PPA** | Power Purchase Agreement – Stromliefervertrag |
| **PV** | Photovoltaik – Stromerzeugung aus Sonnenlicht |
| **PVGIS** | Photovoltaic Geographical Information System – EU-Datenbank für Solarstrahlungsdaten |
| **RTE** | Round-Trip Efficiency – Wirkungsgrad eines vollständigen Lade-/Entladezyklus |
| **Scale** | Batterieleistung als Prozent der PV-Spitzenleistung |
| **SoC** | State of Charge – Ladezustand der Batterie |
| **Soli** | Solidaritätszuschlag |
| **Verlustvortrag** | Steuerlicher Verlustvortrag – Verluste aus Vorjahren werden mit zukünftigen Gewinnen verrechnet |
