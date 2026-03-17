# Co-Location-Modell PV & BESS – Strukturierte Modellbeschreibung

---

## 1. Vermarktungstypen

### 1.1 PPA-Collar

- **Cap-Preis**: Float \[0 … P50-Strompreis-Szenario\]
- **Floor-Preis**: Float \[0 … Cap-Preis\]

### 1.2 PPA-/EEG-Floor

- **Floor-Preis**: Float \[0 … P50-Strompreis-Szenario\]

### 1.3 PPA Pay-as-Produced (PaP)

- **Fixpreis**: Float \[0 … P50-Strompreis-Szenario\]

### 1.4 PPA-Baseload

- **Baseload-Anteil**: Float \[0 … 100 % der PV-Leistung\]
- **Fixpreis**: Float \[0 … P50-Strompreis-Szenario\]

### 1.5 Betriebsmodus

- **Modus**: Kategorie \[green, grey\]

---

## 2. Technik

### 2.1 Photovoltaik (PV)

- PV-Leistung: **fix** (flächenbedingt)

### 2.2 Batteriespeicher (BESS)

- **E/P-Ratio**: Float \[0 … 1\]
- **Leistungsanteil relativ zur PV**: Float \[0 … 1\]

---

# Zusammenfassung: Optimierungsansatz für PV‑BESS‑Co‑Location

## 3. Ausgangsfrage

Ziel ist es, für eine **PV‑Freiflächenanlage mit fixer Leistung (X MW)** und begrenztem Netzanschlusspunkt zu bestimmen:

- welche **BESS-Dimensionierung** (Leistung / Kapazität) sinnvoll ist,
- welche **Vermarktungsstruktur** (EEG, PPA, Markt) erforderlich ist,

um eine **vorgegebene Ziel‑IRR** zu erreichen.

Der Lösungsraum ist groß, aber klar begrenzt:

- PV-Leistung ist fix.
- BESS-Leistung und -Kapazität sind technisch und netzseitig limitiert.
- Preise bewegen sich realistisch unterhalb bzw. in Relation zum langfristigen Spotpreisniveau.

---

## 4. Zentrale Modellierungserkenntnisse

### 4.1 Korrekte Zielfunktion

Kein klassisches *IRR-Maximierungsproblem*, sondern:

> **Finde alle Kombinationen aus Technik und Vermarktung, für die gilt:**  
> `IRR ≥ Ziel‑IRR`

Nebenbedingungen / Nebenoptimierungen:

- minimaler PPA-Preis,
- minimaler CAPEX (kleinster notwendiger BESS),
- maximale Robustheit (geringe Varianz der äquivalenten IRR).

---

### 4.2 Trennung der Entscheidungsarten

**Diskrete Regime (vorab fixieren):**

- Vermarktungsstruktur (EEG, PPA-Typen, Merchant)
- Betriebsmodus (green / grey)

**Stetige Design-Variablen (optimieren):**

- BESS-Leistung (kW)
- BESS-Kapazität (kWh)
- E/P-Ratio
- Preisparameter innerhalb des jeweiligen Regimes

> **Merksatz:** *Regime wählen – Dimensionen optimieren.*

---

## 5. EEG-Preis vs. PPA-Preis

### 5.1 EEG-Preis

- ökonomisch exogen (Ausschreibung),
- nicht frei optimierbar.

Einsatz im Modell:

- Schwellenanalysen („ab welchem EEG-Preis wirtschaftlich?“),
- strategische Meta-Analysen.

### 5.2 PPA-Preis

- verhandelbar → legitime Entscheidungsvariable.

Typischer Zielfall:
> Minimaler PPA-Preis, bei dem mit optimaler BESS-Dimensionierung die Ziel-IRR erreicht wird.

---

## 6. Klassischer Optimierungsansatz

### 6.1 Zweistufige Architektur

1. **Äußere Schleife:** Vermarktungsregime × Betriebsmodus
2. **Innere Optimierung:**
    - BESS-Leistung und -Kapazität
    - unter harten Nebenbedingungen:
        - PV-Leistung fix
        - Netzanschlusspunkt
        - technische BESS-Grenzen
3. Optional: Schleife über PPA-Preis (z. B. Bisektionsverfahren)

Ergebnis:

- mehrere zulässige Lösungen,
- gut vergleichbar,
- management-tauglich erklärbar.

---

## 7. ML‑Surrogat‑Ansatz (optional)

### 7.1 Grundidee

Simulation vieler Kombinationen erzeugt einen Trainingsdatensatz:

Input (Technik + Vermarktung) → Finanzmodell → IRR

Ein ML-Modell approximiert:

`IRR = f(Inputs)`

### 7.2 Bewertung

✅ sinnvoll als **Surrogate‑Assisted Optimization**:

- Regression statt reiner Klassifikation,
- separate Modelle pro Regime / Modus,
- schnelle Vorsortierung vielversprechender Kandidaten.

Grenzen:

- keine Garantie → finale Simulation notwendig,
- explizite Behandlung von Regime-Wechseln nötig.

---

# Top‑Down‑Optimierungsansatz über die Capture‑Rate

## 8. Motivation

Bottom‑Up‑Optimierungen sind hochdimensional und schwer interpretierbar.
Der Top‑Down‑Ansatz reduziert das Problem auf eine zentrale ökonomische Steuergröße:

> **Die Capture‑Rate trägt die Wirtschaftlichkeit (IRR).**

Technik und Vermarktung sind Instrumente zur Erreichung dieser Capture‑Rate.

---

## 9. Definition der Capture‑Rate

\[ c = rac{ ext{effektiver mengen‑gewichteter Erlöspreis}}{ ext{langjähriger Spot‑Referenzpreis}} \]

Sie beinhaltet bereits:

- Profileffekte der PV,
- Fixpreise, Floors, Caps,
- Speicherarbitrage,
- negative Preise und Curtailment,
- optionale Zusatzerlöse (z. B. GoO, Regelenergie).

---

## 10. Schritt 1: IRR → erforderliche Capture‑Rate

Fixe Größen:

- PV-Leistung und Ertrag
- CAPEX / OPEX von PV und BESS
- Finanzierung, Laufzeit, Steuern

Variable:

- Capture‑Rate \(c\)

Durch die Berechnung der beiden extrem Punkte (PV-only (basecase), und max BESS Dimension) werden zwei Capture Rates
definiert, die später die Grenze visualisieren, ab der das Projekt oberhalb des Ziel IRR's ist. Für den PV-Only case ist
es einfach zu rechnen: simples Finanzmodell, und den Jahresertrag so lange mit der durchschnittlichen Capture Rate
anpassen, bis die Ziel IRR erreicht wird. Doch wie macht man das für den max BESS Case? Einfach CAPEX und OPEX
erhöhen reicht nicht aus, da auch eine höhere Produktion durch den BESS gewährt wird, die nicht so einfach zu bestimmen
ist - oder doch?
Durch Variation von \(c\) wird bestimmt:

> **Minimale Capture‑Rate \(c^*\), die die Ziel‑IRR erfüllt.**

---

## 11. Schritt 2: Zerlegung der Capture‑Rate

\[ c = c_{ ext{PV,raw}} + \Delta c_{ ext{Vermarktung}} + \Delta c_{ ext{Flexibilität}} \]

- \(c_{ ext{PV,raw}}\): nackte PV ohne Absicherung
- \(\Delta c_{ ext{Vermarktung}}\): EEG / PPA-Strukturen
- \(\Delta c_{ ext{Flexibilität}}\): BESS-Effekte (abnehmender Grenznutzen)

---

## 12. Entscheidungsproblem

> Welche Kombination aus Vermarktung und minimaler Flexibilität erreicht \(c^*\) zu minimalen Kosten?

Formal:

\[ \min_{(Struktur, P, E)} CAPEX_{ ext{BESS}} + Risikoaufschläge \]

unter der Nebenbedingung:

\[ c_{ ext{Struktur}} + \Delta c_{ ext{Flex}}(P,E) \ge c^* \]

---

## 13. Visualisierbarer Entscheidungsraum

Empfohlene Darstellung:

- x‑Achse: Capture‑Rate
- y‑Achse: zusätzlicher CAPEX
- Linien: Vermarktungsstrukturen
- x-Achse Konstante: Ab Capture-Rate C erreicht das Projekt den Ziel IRR

Erkenntnisse:

- Wann EEG ausreicht,
- wann PPA genügt,
- wann Speicher überdimensioniert wäre.

---

## 14. Kernaussage

> **Die zentrale wirtschaftliche Frage lautet nicht „Welche Technik?“ oder „Welche Vermarktung?“ – sondern: Welche
Capture‑Rate braucht das Projekt, und welches Instrument liefert sie am günstigsten?**

