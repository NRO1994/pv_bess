# Meta-Modell: Systemwert von Flexibilität im Stadtwerk-Portfolio

## Zweck dieses Dokuments

Dieses Dokument dient als Anweisung (Prompt/Spec) für eine Large Language Model (LLM), die ein Meta-Modell zur Bewertung
von Flexibilitätsoptionen im Stadtwerk-Portfolio erzeugen soll.

Ziel ist nicht die Detail-Implementierung einzelner Assets, sondern die konzeptionelle Modellarchitektur, die es
erlaubt, strategische Investitions- und Portfolioentscheidungen datenbasiert zu unterstützen.

---

## 1. Übergeordnete Zielsetzung des Meta-Modells

Das Meta-Modell soll beantworten:

> Wie viel ökonomischen Systemwert erzeugt ein zusätzlicher steuerbarer kW Flexibilität im Stadtwerk-Portfolio – 
> und welche Flexibilitätsoption liefert den höchsten Grenznutzen?

Der Fokus liegt auf:

- Portfolio- statt Projektlogik
- Grenznutzen statt Durchschnittsrenditen
- Vergleichbarkeit unterschiedlicher Flexibilitätsarten

---

## 2. Zentrale strategische Fragestellungen

Das Modell muss in der Lage sein, u. a. folgende Fragen zu beantworten:

1. Grenznutzen-Frage
    - Wie verändert sich das Systemergebnis durch +x kW zusätzlicher Flexibilität?
    - Wie hoch ist der Wert des *nächsten* kW Flex (€/kW·a)?

2. Priorisierungs-Frage
    - Welche Flexibilitätsoption (BESS, Wärmepumpe, Wallbox etc.) erzeugt bei gleichem kW den höchsten Systemwert?

3. Sättigungs-Frage
    - Ab welchem Ausbaugrad sinkt der Grenznutzen signifikant (abnehmender Grenznutzen)?

4. Portfolio-Frage
    - Wie verändert Flexibilität die Marktposition (Einkauf vs. Verkauf) des gesamten Stadtwerks?

5. Kapitalallokation
    - Wo sollte der nächste investierte Euro eingesetzt werden, um den größten System-Impact zu erzielen?

---

## 3. Grundlogik: Welt-A / Welt-B-Vergleich

Das Meta-Modell basiert auf einem kontrafaktischen Vergleich:

- Welt A (Baseline): Status quo des Portfolios
- Welt B (Intervention): Status quo + singuläre Änderung (z. B. +30 kW BESS Flex)

Der Systemwert ist definiert als:

Δ(Systemwert) = (Erlöse − Kosten)_B − (Erlöse − Kosten)_A

Wichtig: Es werden nur veränderliche, marktbezogene Größen betrachtet.

---

## 4. Definition von Erlösen und Kosten (Systemperspektive)

### 4.1 Erlöse

Als Erlöse gelten ausschließlich marktbasierte Zahlungsströme:

- Verkauf von Netto-Überschüssen am Strommarkt
- Bewertung zu zeitabhängigen Marktpreisen (Spot, PPA, EEG-Floor etc.)

Nicht als Erlöse zählen:

- interne Stromlieferungen an eigene Kunden
- Tarifumsätze oder Endkundenpreise

---

### 4.2 Kosten

Als Kosten gelten:

- Marktbasierte Beschaffungskosten bei Unterdeckung
- Bewertung ebenfalls zu zeitabhängigen Marktpreisen

Nicht berücksichtigen:

- CAPEX, fixe OPEX, Abschreibungen
- Fremdkapitaldienst, Steuern
- projektbezogene Finanzierungslogiken

Diese Größen sind im Welt-A/Welt-B-Vergleich konstant und würden den Systemwert verfälschen.

---

## 5. Opportunitätskosten eigener Erzeugung

Opportunitätskosten werden nicht explizit modelliert.

Sie sind implizit enthalten, da:

- jede interne Nutzung von Strom einen entgangenen Marktverkauf darstellt
- diese entgangenen Erlöse automatisch über geringere Verkaufsmengen abgebildet werden

Grundsatz:
> Opportunitätskosten sind Differenzen zwischen zwei zulässigen Marktpositionen, keine eigene Kostenart.

---

## 6. Zentrale Modellkomponenten, die die LLM erzeugen soll

Die LLM soll ein Modell entwerfen, das mindestens folgende Konzepte enthält:

### 6.1 Portfolio-Bilanz

- Aggregation aller Erzeugungsprofile
- Aggregation aller Lastprofile (starr + flexibel)
- Zeitauflösung (z. B. 15-min / 1h)

### 6.2 Flexibilitäts-Abstraktion

- Flex wird als steuerbare Leistung (kW) mit Nebenbedingungen modelliert
- Unterschiedliche Flexarten sind zulässig, müssen aber vergleichbar abstrahiert werden

Beispiele:

- BESS: Leistung + Energiekapazität (E/P-Ratio)
- Wärmepumpe: Verschiebefähige Last mit Energiebedarf
- Wallbox: Ladeleistung mit Zeitfenster

### 6.3 Optimierungslogik

- Ein zentrales Optimierungsproblem steuert alle Flexibilitäten gemeinsam
- Ziel: Minimierung der marktwertbasierten Systemkosten
- Ergebnis: optimierte Marktposition je Zeitschritt

### 6.4 Grenznutzen-Analyse

- Iterative Erhöhung der Flex (+x kW)
- Berechnung des Systemwerts je Ausbaustufe
- Ableitung von:
    - kumulierten Systemwertkurven
    - marginalen Systemwertkurven (€/kW)

---

## 7. Erwartete Outputs des Meta-Modells

Das Modell soll konzeptionell folgende Outputs liefern:

- Kurvenscharen: Systemwert vs. installierte Flex je Flex-Art
- Grenznutzenkurven: Wert des nächsten kW je Flex-Art
- Vergleichbarkeit zwischen Flexoptionen
- Sättigungspunkte für Investitionsentscheidungen

Diese Outputs dienen explizit der strategischen Entscheidungsunterstützung (Unternehmensentwicklung, Geschäftsführung),
nicht der Projektfinanzierung.

---

## 8. Abgrenzung

Das Meta-Modell soll nicht:

- detaillierte Netzrestriktionen abbilden
- regulatorische Abrechnung simulieren
- vollständige GuV- oder Cashflow-Rechnungen ersetzen

Diese Themen können in nachgelagerten Modellen berücksichtigt werden.

---

## 9. Leitprinzip

> Das Meta-Modell bewertet Flexibilität als systemische Ressource – nicht als einzelnes Asset.

Der Fokus liegt auf relativen Effekten, Grenzwerten und Entscheidungslogik.
