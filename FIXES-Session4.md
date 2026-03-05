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

# Spezifikation für eine **Single-File App** (HTML + JavaScript) zur JSON-Erfassung für OneDrive-Workflow

Diese Markdown-Datei dient als **kompletter Arbeitsauftrag** für eine LLM-/Entwicklungsumgebung, um eine **einzige**
HTML-Datei (`index.html`) zu erzeugen. Die Datei läuft **offline/lokal** im Browser, erfasst Eingaben des Fachbereichs,
validiert sie, und exportiert die Daten als **JSON-Datei**, die die Nutzer in ihren **lokalen OneDrive-Sync-Ordner**
speichern (damit die Datei automatisch mit OneDrive synchronisiert wird).

---

## 1) Ziel & Rahmen

- **Ziel:** Leichtgewichtige UI ohne Installation/Hosting. Eine einzelne Datei `colocation_input.html` reicht.
- **Austausch:**
    - Fachbereich → erfasst Eingaben → exportiert JSON → speichert in lokalen OneDrive-Ordner **Input**.
    - Python-Prozess (extern) → liest Input → berechnet Szenarien → legt Ergebnisse (Dateien) in **Output** ab.
- **IT-Rahmen:** Kein Backend, keine Ports, keine externen Abhängigkeiten. Keine Azure App Registration. **Reiner
  Offline-Betrieb** im Browser.
- **Kompatibilität:** Aktuelle Edge/Chrome/Firefox. Kein IE.

---

## 2) Funktionsumfang (MVP)

1. **Formular-Erfassung:** Es soll einen einfachen, intuitiven und geführten User Flow durch den Input geben. Kombiniere
   Inputs geschickt, sodass sich eine logische Reihenfolge für den Nutzer ergibt - dies muss NICHT der Reihenfolge des
   JSON-Schemas entsprechen.
    - Pflichtfelder & optionale Felder
    - Nummern, Dropdowns, Checkboxen, Text
    - dynamische Abschnitte (z.B. wiederholbare Parameterlisten)
2. **Client-seitige Validierung** mit klaren Fehlermeldungen.
3. **JSON-Export** (Datei-Download) mit sprechendem Dateinamen.
4. **Metadaten** automatisch einfügen: Version, Ersteller (frei), Timestamp in ISO 8601.
5. **UI-Feedback**: Erfolg-/Fehler-Banner, Validierungshighlights, Disabled-States beim Export.
6. **Konfigurationssektion** (im UI) mit:
    - Präfix für Dateiname
    - (Optional) Zielordnerhinweis (nur als Text, **kein** direktes Schreiben ohne API)
7. **Onboarding-Hinweis**: Kurze Anleitung, wohin JSON gespeichert werden soll (z.B. `OneDrive\Projekte\XYZ\Input`).
8. **„Vorlage laden“**: Nutzer kann vorhandene JSON importieren, Felder werden befüllt (Drag & Drop oder
   Datei-Selector).
9. **Persistenz (lokal):** Option „Eingaben lokal merken“ via `localStorage`.
10. **Barrierefreiheit & Tastaturbedienbarkeit** (WAI-ARIA minimal).
11. **Sprache:** Deutsch.

---

## 3) Nicht-Funktionsanforderungen

- **Single File**: Keine externen Bundles, kein CDN. Inline CSS & JS.
- **Performance**: < 200 KB Gesamtdokument (ohne Bilder) anstreben.
- **Security/Privacy**:
    - Keine Telemetrie, keine externen Calls.
    - Kein Tracking, keine Cookies (nur `localStorage`, wenn aktiviert).
    - Klare UX-Hinweise, dass Daten **nur lokal** bleiben, bis Nutzer sie speichern.
- **Robustheit**: Verständliche Fehlermeldungen, keine JS-Error-Leaks im UI.
- **Offline first**: Muss ohne Internet voll funktionieren.

---

## 4) Datenmodell (JSON-Schema)

Siehe `.docs/input_schema.json`

Dateinamensmuster (Vorschlag):
`{prefix}-{sanitized_scenarioName}-{YYYYMMDD-HHmmss}.json`

prefix: konfigurierbar im UI (Default: input)
sanitized_scenarioName: nur [a-z0-9_-], Kleinbuchstaben, Leerzeichen → -
Zeit: lokale Zeit, 24h, YYYYMMDD-HHmmss

## 5) UX-Design (ohne externe Frameworks)

Layout: zentrierte, responsive Spalte (max 860 px), großzügige Abstände, klare Labels.
Navigation: Scroll-Seite mit Sektionen: Allgemein, Parameter, Optionen (dynamisch), Anhänge (nur Namen/Notiz),
Konfiguration, Export.
Farben: Sollen sich am corporate design aus der chart generation orientieren
Interaktionsmuster:

Inline-Validierung beim Blur und erneut bei Export.
Fehlerhinweise direkt unter Feld (rot), Success-Banner nach Export (grün).
„Option hinzufügen“/„entfernen“ dynamisch.
Datei-Import (JSON) via Button + Drag & Drop.

Tastaturbedienung: Tab-Reihenfolge, Enter triggert Export, Buttons haben aria-label.
Tooltips für komplexe Felder.

## 6) Validierung (Details)

Synchron (JS):

Pflichtfelder: Name, paramA, paramB.
Grenzen: paramA ≥ 0.
Enums: category, paramB.
Eindeutige options.key innerhalb der Liste.
Max-Längen einhalten (Name 100, Description 1000).

Fehlermeldungen in DE, konkret (z.B. „ParamA muss ≥ 0 sein“).
Vor Export: Full-Form-Validation; bei Fehlern kein Export und Fokus auf erstes Fehlerfeld.
Import-Validierung: JSON-Struktur checken; bei Schema-Abweichung: Nutzerfreundlicher Hinweis + Option „trotzdem laden“ (
nicht empfohlen).

## 7) Export/Import

Export:

Blob + URL.createObjectURL() + <a download>.
Dateiname gemäß Muster (s.o.).
Metadaten füllen: schemaVersion="1.0.0", appVersion (hart im Code, z.B. 1.0.0), createdAt (ISO), createdBy (freies
Textfeld im UI, optional).

Import:

<input type="file" accept="application/json"> und Drag & Drop.
Parsen mit FileReader.
Validieren und Felder befüllen.
Bei Feldern, die nicht existieren (Schema-Drift): Ignorieren, aber Hinweis anzeigen.

## 8) Barrierefreiheit (A11y)

Labels verknüpfen mit for/id.
aria-invalid, aria-describedby für Fehlermeldungen.
Kontrast ≥ WCAG AA.
Fokus-Styling sichtbar (Outline).
Semantische Elemente (<main>, <section>, <form>, <button>).

## 9) Sicherheit & Datenschutz

Kein Netzwerkanruf, keine externen Skripte/Fonts.
Keine personenbezogenen Daten erheben außer optional createdBy (Freitext).
Das Dokument erklärt im Header eindeutig: „Daten bleiben lokal, bis Sie aktiv speichern.“
Sanitizing von Dateinamen & Strings für UI (XSS-Schutz in gefahrarmen Kontexten).

## 10) Qualität & Tests

Browser-Tests: Edge/Chrome/Firefox (aktuell).
Cases:

Leeres Formular → Fehler.
Min/Max-Values.
0/50 Optionen.
Nicht-ASCII im Namen → Sanitize richtig.
Import älterer Datei (gleiche schemaVersion) → passt.

Kein ESLint erforderlich, aber sauberer Code, Funktionen klar benannt, Kommentare DE.

11) Erweiterbarkeit (später)

Optionales „Ergebnis-Viewer“-Panel (liest Output-JSON/CSV via Datei-Auswahl).
Druck-/PDF-Ansicht der Eingaben (nur clientseitig).
Vorlagenverwaltung (lokal).
I18N via JSON-Dictionary (DE/EN).

12) Abnahmekriterien

Eine einzige Datei index.html, ausführbar lokal per Doppelklick.
Funktioniert offline ohne Fehlermeldungen in Konsole.
Erzeugt valide JSONs gemäß Schema oben.
Importierter JSON befüllt die Oberfläche korrekt.
Klare, deutsche Fehlermeldungen.
A11y-Basics erfüllt, Tastaturbedienung komplett möglich.
Kein externes Laden (CDN, Fonts, Bilder).

13) UI-Textbausteine (Deutsch)

Titel: „Szenario-Eingabe“
Sektionen: „Allgemein“, „Parameter“, „Optionen“, „Anhänge“, „Konfiguration“, „Export“
Felder (Beispiele):

Name des Szenarios (Pflicht)
Beschreibung (optional)
Kategorie (Baseline, Optimistisch, Pessimistisch, Benutzerdefiniert)
ParamA (Zahl ≥ 0)
ParamB (low | medium | high)
Optionen (Key/Value/Notiz)
Anhänge (Dateiname/Notiz – nur als Text)
Ersteller (optional)
Dateipräfix (Default: input)
Dark Mode (Schalter)

Buttons:

„Option hinzufügen“
„Option entfernen“
„JSON importieren“
„JSON exportieren“
„Eingaben lokal merken (Browser)“

Hinweise:

„Speichern Sie die Datei im OneDrive-Ordner: …\Input“
„Alle Daten bleiben lokal, bis Sie exportieren.“