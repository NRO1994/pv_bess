"""
Strompreis-Daten Analyse-Skript.

Fuehrt eine tiefgehende Analyse von Day-Ahead Strompreisdaten durch:
- Deskriptive Statistik (pro Szenario, pro Jahr, gesamt)
- Konstante-Perioden-Analyse (Erkennung, Dauer, Verteilung, Heatmaps)
- Volatilitaets- & Verteilungsanalyse
- Saisonale & zeitliche Muster
- Szenario-Vergleich

Output: Markdown-Report + PNG-Diagramme
"""

import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ==============================================================================
# GLOBALE KONFIGURATION
# ==============================================================================

# Pfad zur CSV-Datei mit Strompreisdaten
CSV_PATH = ".data/integration_test_inputs/finance/eeg_fixed_54_20y_22_5y.csv"

# Spaltennamen
COL_TIMESTAMP = "timestamp"
COL_LOW = "Low"
COL_MID = "MID"
COL_HIGH = "High"
SCENARIO_COLS = [COL_LOW, COL_MID, COL_HIGH]

# CSV-Format
CSV_SEPARATOR = ";"
CSV_DECIMAL = ","

# Intervalle (Defaults fuer 15-min Aufloesung, werden bei load_data ueberschrieben)
INTERVALS_PER_YEAR = 35_040
INTERVALS_PER_DAY = 96
INTERVALS_PER_HOUR = 4
INTERVALS_PER_WEEK = 96 * 7

# Output
OUTPUT_DIR = Path(".data/price_analysis")
DPI = 150

# Schwellenwert: Ab wie vielen identischen aufeinanderfolgenden Werten gilt
# eine Periode als "konstant"? (2 = schon ab 2 gleichen Werten hintereinander)
CONST_MIN_LENGTH = 2

# Toleranz fuer Gleichheit (floating point)
CONST_TOLERANCE = 1e-10

# ==============================================================================
# HILFSFUNKTIONEN
# ==============================================================================


def load_data(csv_path: str) -> pd.DataFrame:
    """Lade CSV-Daten und parse Timestamps."""
    df = pd.read_csv(
        csv_path,
        sep=CSV_SEPARATOR,
        decimal=CSV_DECIMAL,
        parse_dates=[COL_TIMESTAMP],
    )
    n_rows = len(df)

    # Aufloesung automatisch erkennen anhand der Zeitdifferenz
    dt = df[COL_TIMESTAMP].iloc[1] - df[COL_TIMESTAMP].iloc[0]
    minutes_per_interval = int(dt.total_seconds() / 60)
    if minutes_per_interval == 15:
        intervals_per_day = 96
        intervals_per_year = 35_040
        intervals_per_hour = 4
    elif minutes_per_interval == 60:
        intervals_per_day = 24
        intervals_per_year = 8_760
        intervals_per_hour = 1
    else:
        raise ValueError(f"Unbekannte Aufloesung: {minutes_per_interval} min")

    n_years = n_rows // intervals_per_year

    # Kalender-Jahr aus Timestamp
    df["year"] = df[COL_TIMESTAMP].dt.year
    # Tag im Jahr (0-basiert)
    df["day_of_year"] = np.tile(
        np.repeat(np.arange(365), intervals_per_day), n_years
    )[:n_rows]
    # Intervall im Tag
    df["interval_of_day"] = np.tile(np.arange(intervals_per_day), 365 * n_years)[
        :n_rows
    ]
    # Stunde (float)
    df["hour"] = df["interval_of_day"] / intervals_per_hour
    # Monat aus Timestamp
    df["month"] = df[COL_TIMESTAMP].dt.month
    # Wochentag (0=Mo, 6=So)
    df["weekday"] = df[COL_TIMESTAMP].dt.weekday

    # Erkannte Aufloesung als Attribute speichern (fuer spaetere Nutzung)
    df.attrs["intervals_per_day"] = intervals_per_day
    df.attrs["intervals_per_year"] = intervals_per_year
    df.attrs["intervals_per_hour"] = intervals_per_hour
    df.attrs["minutes_per_interval"] = minutes_per_interval
    df.attrs["n_years"] = n_years

    return df


def find_constant_periods(
    values: np.ndarray, min_length: int = CONST_MIN_LENGTH, tol: float = CONST_TOLERANCE
) -> list[dict]:
    """
    Finde zusammenhaengende Perioden mit konstanten Werten.

    Returns:
        Liste von dicts mit: start_idx, end_idx, length, value
    """
    periods = []
    n = len(values)
    if n == 0:
        return periods

    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(values[j] - values[i]) <= tol:
            j += 1
        length = j - i
        if length >= min_length:
            periods.append(
                {
                    "start_idx": i,
                    "end_idx": j - 1,
                    "length": length,
                    "value": values[i],
                }
            )
        i = j

    return periods


def format_intervals_as_duration(intervals: int) -> str:
    """Formatiere Intervallanzahl als lesbare Dauer."""
    minutes_per = 60 // INTERVALS_PER_HOUR
    minutes = intervals * minutes_per
    if minutes < 60:
        return f"{minutes} min"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f} h"
    days = hours / 24
    return f"{days:.1f} Tage"


def ensure_output_dir() -> Path:
    """Erstelle Output-Verzeichnis falls noetig."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


# ==============================================================================
# 1. DESKRIPTIVE STATISTIK
# ==============================================================================


def descriptive_statistics(df: pd.DataFrame) -> str:
    """Berechne deskriptive Statistik pro Szenario und pro Jahr."""
    md = "## 1. Deskriptive Statistik\n\n"

    # --- Gesamt ---
    md += "### 1.1 Gesamtstatistik (alle Jahre)\n\n"
    md += "| Kennzahl | " + " | ".join(SCENARIO_COLS) + " |\n"
    md += "|----------|" + "|".join(["--------"] * len(SCENARIO_COLS)) + "|\n"

    stats = {
        "Anzahl Werte": lambda s: f"{len(s):,}",
        "Mittelwert (EUR/kWh)": lambda s: f"{s.mean():.5f}",
        "Median (EUR/kWh)": lambda s: f"{s.median():.5f}",
        "Std (EUR/kWh)": lambda s: f"{s.std():.5f}",
        "Min (EUR/kWh)": lambda s: f"{s.min():.5f}",
        "Max (EUR/kWh)": lambda s: f"{s.max():.5f}",
        "P5 (EUR/kWh)": lambda s: f"{s.quantile(0.05):.5f}",
        "P25 (EUR/kWh)": lambda s: f"{s.quantile(0.25):.5f}",
        "P75 (EUR/kWh)": lambda s: f"{s.quantile(0.75):.5f}",
        "P95 (EUR/kWh)": lambda s: f"{s.quantile(0.95):.5f}",
        "Anteil negativ (%)": lambda s: f"{(s < 0).mean() * 100:.2f}",
        "Anteil == 0 (%)": lambda s: f"{(s == 0).mean() * 100:.2f}",
        "Mittelwert (EUR/MWh)": lambda s: f"{s.mean() * 1000:.2f}",
    }

    for name, func in stats.items():
        vals = " | ".join(func(df[col]) for col in SCENARIO_COLS)
        md += f"| {name} | {vals} |\n"

    # --- Pro Jahr ---
    md += "\n### 1.2 Jaehrliche Statistik\n\n"
    for col in SCENARIO_COLS:
        md += f"\n#### Szenario: {col}\n\n"
        md += "| Jahr | Mean | Median | Std | Min | Max | P5 | P95 | Neg% |\n"
        md += "|------|------|--------|-----|-----|-----|----|----|------|\n"
        for year, grp in df.groupby("year"):
            s = grp[col]
            md += (
                f"| {year} "
                f"| {s.mean():.5f} "
                f"| {s.median():.5f} "
                f"| {s.std():.5f} "
                f"| {s.min():.5f} "
                f"| {s.max():.5f} "
                f"| {s.quantile(0.05):.5f} "
                f"| {s.quantile(0.95):.5f} "
                f"| {(s < 0).mean() * 100:.1f} |\n"
            )

    return md


def plot_annual_means(df: pd.DataFrame, output_dir: Path) -> str:
    """Jahresmittelwerte aller Szenarien als Liniendiagramm."""
    fig, ax = plt.subplots(figsize=(12, 5))
    annual = df.groupby("year")[SCENARIO_COLS].mean() * 1000  # EUR/MWh

    for col in SCENARIO_COLS:
        ax.plot(annual.index, annual[col], marker="o", markersize=4, label=col)

    ax.set_xlabel("Jahr")
    ax.set_ylabel("Durchschnittspreis (EUR/MWh)")
    ax.set_title("Jaehrliche Durchschnittspreise pro Szenario")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = output_dir / "01_annual_means.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return f"![Jahresmittelwerte]({path.name})\n\n"


def plot_annual_boxplots(df: pd.DataFrame, output_dir: Path) -> str:
    """Boxplots pro Jahr und Szenario."""
    years = sorted(df["year"].unique())
    fig, axes = plt.subplots(len(SCENARIO_COLS), 1, figsize=(16, 4 * len(SCENARIO_COLS)), sharex=True)

    for ax, col in zip(axes, SCENARIO_COLS):
        data_per_year = [df.loc[df["year"] == y, col].values * 1000 for y in years]
        bp = ax.boxplot(
            data_per_year,
            positions=range(len(years)),
            widths=0.6,
            showfliers=False,
            patch_artist=True,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
            patch.set_alpha(0.7)
        ax.set_ylabel("EUR/MWh")
        ax.set_title(f"Szenario: {col}")
        ax.grid(True, alpha=0.3, axis="y")

    axes[-1].set_xticks(range(len(years)))
    axes[-1].set_xticklabels(years, rotation=45)
    axes[-1].set_xlabel("Jahr")

    path = output_dir / "02_annual_boxplots.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return f"![Boxplots]({path.name})\n\n"


# ==============================================================================
# 2. KONSTANTE-PERIODEN-ANALYSE
# ==============================================================================


def constant_periods_analysis(df: pd.DataFrame) -> tuple[str, dict]:
    """Analysiere konstante Perioden in allen Szenarien."""
    md = "## 2. Konstante-Perioden-Analyse\n\n"
    md += (
        f"Definition: >= {CONST_MIN_LENGTH} aufeinanderfolgende identische Werte "
        f"(Toleranz: {CONST_TOLERANCE})\n\n"
    )

    all_periods: dict[str, list[dict]] = {}

    for col in SCENARIO_COLS:
        values = df[col].values
        periods = find_constant_periods(values)
        all_periods[col] = periods

        if not periods:
            md += f"### Szenario {col}: Keine konstanten Perioden gefunden.\n\n"
            continue

        lengths = np.array([p["length"] for p in periods])
        total_intervals = len(values)
        total_const = lengths.sum()

        md += f"### 2.1 Szenario: {col}\n\n"
        md += f"- **Anzahl konstanter Perioden**: {len(periods):,}\n"
        md += f"- **Gesamtanteil konstanter Intervalle**: {total_const:,} / {total_intervals:,} = **{total_const / total_intervals * 100:.2f}%**\n"
        md += f"- **Kuerzeste Periode**: {lengths.min()} Intervalle ({format_intervals_as_duration(lengths.min())})\n"
        md += f"- **Laengste Periode**: {lengths.max()} Intervalle ({format_intervals_as_duration(lengths.max())})\n"
        md += f"- **Mittlere Dauer**: {lengths.mean():.1f} Intervalle ({format_intervals_as_duration(int(lengths.mean()))})\n"
        md += f"- **Median Dauer**: {np.median(lengths):.0f} Intervalle ({format_intervals_as_duration(int(np.median(lengths)))})\n\n"

        # Verteilung nach Dauer-Klassen
        md += "#### Verteilung nach Dauer\n\n"
        md += "| Dauer-Klasse | Anzahl | Anteil (%) | Summe Intervalle |\n"
        md += "|-------------|--------|-----------|------------------|\n"

        bins = [
            (2, 4, "2-3 (30-45 min)"),
            (4, 12, "4-11 (1-2.75 h)"),
            (12, 48, "12-47 (3-11.75 h)"),
            (48, 96, "48-95 (12-23.75 h)"),
            (96, 192, "96-191 (1-1.99 Tage)"),
            (192, 672, "192-671 (2-6.99 Tage)"),
            (672, 2880, "672-2879 (1-4.29 Wochen)"),
            (2880, None, ">= 2880 (>= 30 Tage)"),
        ]

        for lo, hi, label in bins:
            if hi is not None:
                mask = (lengths >= lo) & (lengths < hi)
            else:
                mask = lengths >= lo
            count = mask.sum()
            if count > 0:
                sum_int = lengths[mask].sum()
                md += f"| {label} | {count} | {count / len(periods) * 100:.1f} | {sum_int:,} |\n"

        md += "\n"

        # Top 20 laengste Perioden
        sorted_periods = sorted(periods, key=lambda p: p["length"], reverse=True)
        top_n = min(20, len(sorted_periods))
        md += f"#### Top {top_n} laengste konstante Perioden\n\n"
        md += "| # | Start-Index | Dauer | Dauer (lesbar) | Wert (EUR/kWh) | Start-Timestamp |\n"
        md += "|---|------------|-------|----------------|---------------|----------------|\n"

        for i, p in enumerate(sorted_periods[:top_n]):
            ts = df[COL_TIMESTAMP].iloc[p["start_idx"]]
            md += (
                f"| {i+1} | {p['start_idx']:,} | {p['length']:,} "
                f"| {format_intervals_as_duration(p['length'])} "
                f"| {p['value']:.5f} | {ts} |\n"
            )

        md += "\n"

        # Bei welchen Preisniveaus treten konstante Perioden auf?
        const_values = np.array([p["value"] for p in periods])
        const_lengths = np.array([p["length"] for p in periods])
        unique_values = np.unique(const_values)

        md += f"#### Preisniveaus mit konstanten Perioden ({len(unique_values)} verschiedene Werte)\n\n"
        if len(unique_values) <= 30:
            md += "| Preis (EUR/kWh) | Anzahl Perioden | Gesamt-Intervalle | Laengste Periode |\n"
            md += "|----------------|----------------|-------------------|------------------|\n"
            for v in sorted(unique_values):
                mask = const_values == v
                n_per = mask.sum()
                sum_int = const_lengths[mask].sum()
                max_len = const_lengths[mask].max()
                md += f"| {v:.5f} | {n_per} | {sum_int:,} | {format_intervals_as_duration(max_len)} |\n"
        else:
            md += f"Zu viele verschiedene Werte ({len(unique_values)}). Top 15 nach Gesamtdauer:\n\n"
            md += "| Preis (EUR/kWh) | Anzahl Perioden | Gesamt-Intervalle | Laengste Periode |\n"
            md += "|----------------|----------------|-------------------|------------------|\n"
            value_total = {}
            for v in unique_values:
                mask = const_values == v
                value_total[v] = const_lengths[mask].sum()
            top_values = sorted(value_total, key=value_total.get, reverse=True)[:15]
            for v in top_values:
                mask = const_values == v
                n_per = mask.sum()
                sum_int = const_lengths[mask].sum()
                max_len = const_lengths[mask].max()
                md += f"| {v:.5f} | {n_per} | {sum_int:,} | {format_intervals_as_duration(max_len)} |\n"

        md += "\n"

    # --- Synchronitaet zwischen Szenarien ---
    md += "### 2.2 Synchronitaet konstanter Perioden zwischen Szenarien\n\n"

    # Markiere fuer jedes Intervall ob es in einer konstanten Periode liegt
    const_masks = {}
    for col in SCENARIO_COLS:
        mask = np.zeros(len(df), dtype=bool)
        for p in all_periods[col]:
            mask[p["start_idx"]:p["end_idx"] + 1] = True
        const_masks[col] = mask

    md += "| Szenario-Paar | Gleichzeitig konstant (%) | Nur A konstant (%) | Nur B konstant (%) |\n"
    md += "|-------------|--------------------------|-------------------|-------------------|\n"

    for i, col_a in enumerate(SCENARIO_COLS):
        for col_b in SCENARIO_COLS[i+1:]:
            both = (const_masks[col_a] & const_masks[col_b]).sum()
            only_a = (const_masks[col_a] & ~const_masks[col_b]).sum()
            only_b = (~const_masks[col_a] & const_masks[col_b]).sum()
            n = len(df)
            md += (
                f"| {col_a} / {col_b} "
                f"| {both / n * 100:.2f} "
                f"| {only_a / n * 100:.2f} "
                f"| {only_b / n * 100:.2f} |\n"
            )

    # Gleichzeitig konstant UND gleicher Wert?
    md += "\n#### Gleichzeitig konstant mit identischem Wert\n\n"
    for i, col_a in enumerate(SCENARIO_COLS):
        for col_b in SCENARIO_COLS[i+1:]:
            both_const = const_masks[col_a] & const_masks[col_b]
            if both_const.sum() > 0:
                same_value = both_const & (
                    np.abs(df[col_a].values - df[col_b].values) <= CONST_TOLERANCE
                )
                md += (
                    f"- **{col_a} / {col_b}**: "
                    f"{same_value.sum():,} Intervalle gleichzeitig konstant mit gleichem Wert "
                    f"({same_value.sum() / both_const.sum() * 100:.1f}% der gleichzeitig konstanten)\n"
                )

    md += "\n"

    # --- Jaehrliche Aufschluesselung ---
    md += "### 2.3 Konstante Perioden pro Jahr\n\n"
    for col in SCENARIO_COLS:
        md += f"#### Szenario: {col}\n\n"
        md += "| Jahr | Anz. Perioden | Anteil konstant (%) | Laengste (Intervalle) | Laengste (lesbar) |\n"
        md += "|------|--------------|--------------------|-----------------------|-------------------|\n"
        for year in sorted(df["year"].unique()):
            year_mask = df["year"] == year
            year_values = df.loc[year_mask, col].values
            year_periods = find_constant_periods(year_values)
            if year_periods:
                year_lengths = [p["length"] for p in year_periods]
                total_const = sum(year_lengths)
                md += (
                    f"| {year} | {len(year_periods)} "
                    f"| {total_const / len(year_values) * 100:.2f} "
                    f"| {max(year_lengths)} "
                    f"| {format_intervals_as_duration(max(year_lengths))} |\n"
                )
            else:
                md += f"| {year} | 0 | 0.00 | - | - |\n"
        md += "\n"

    return md, all_periods


def plot_constant_periods_heatmap(
    df: pd.DataFrame, all_periods: dict, output_dir: Path
) -> str:
    """Heatmap: Konstante Perioden ueber das Jahr (Tag x Stunde)."""
    n_years = len(df["year"].unique())
    fig, axes = plt.subplots(
        1, len(SCENARIO_COLS), figsize=(6 * len(SCENARIO_COLS), max(5, n_years * 0.25))
    )
    if len(SCENARIO_COLS) == 1:
        axes = [axes]

    images = []
    for ax, col in zip(axes, SCENARIO_COLS):
        # Erzeuge 2D-Matrix: Tag x Intervall-im-Tag
        # Aggregiert ueber alle Jahre
        heatmap = np.zeros((365, INTERVALS_PER_DAY))
        count_map = np.zeros((365, INTERVALS_PER_DAY))

        for p in all_periods[col]:
            for idx in range(p["start_idx"], p["end_idx"] + 1):
                if idx < len(df):
                    doy = df["day_of_year"].iloc[idx]
                    iod = df["interval_of_day"].iloc[idx]
                    heatmap[doy, iod] += 1

        # Normalisiere auf Anzahl Jahre
        heatmap /= n_years

        im = ax.imshow(
            heatmap.T,
            aspect="auto",
            cmap="YlOrRd",
            origin="lower",
            extent=[0, 365, 0, 24],
        )
        ax.set_xlabel("Tag im Jahr")
        ax.set_ylabel("Stunde")
        ax.set_title(f"{col}")
        images.append(im)

    # Gemeinsame Colorbar
    fig.colorbar(
        images[-1], ax=axes, label="Mittlere Haeufigkeit (pro Jahr)", shrink=0.8
    )
    fig.suptitle("Heatmap: Konstante Perioden (Tag x Stunde, gemittelt ueber alle Jahre)", y=1.02)

    path = output_dir / "03_constant_periods_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return f"![Heatmap konstante Perioden]({path.name})\n\n"


def plot_constant_periods_duration_hist(
    all_periods: dict, output_dir: Path
) -> str:
    """Histogramm der Dauern konstanter Perioden."""
    fig, axes = plt.subplots(1, len(SCENARIO_COLS), figsize=(6 * len(SCENARIO_COLS), 5))
    if len(SCENARIO_COLS) == 1:
        axes = [axes]

    for ax, col in zip(axes, SCENARIO_COLS):
        periods = all_periods[col]
        if not periods:
            ax.set_title(f"{col} - keine Perioden")
            continue
        lengths = [p["length"] for p in periods]
        # Log-skalierte Bins
        max_len = max(lengths)
        if max_len > 100:
            bins = np.logspace(np.log10(CONST_MIN_LENGTH), np.log10(max_len), 40)
            ax.set_xscale("log")
        else:
            bins = np.arange(CONST_MIN_LENGTH, max_len + 2)
        ax.hist(lengths, bins=bins, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Dauer (Intervalle, 15min)")
        ax.set_ylabel("Anzahl Perioden")
        ax.set_title(f"{col}")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Verteilung der Dauer konstanter Perioden")
    path = output_dir / "04_constant_periods_duration_hist.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return f"![Dauer-Histogramm]({path.name})\n\n"


def plot_constant_periods_by_price_level(
    all_periods: dict, output_dir: Path
) -> str:
    """Scatter: Preisniveau vs. Dauer der konstanten Perioden."""
    fig, axes = plt.subplots(1, len(SCENARIO_COLS), figsize=(6 * len(SCENARIO_COLS), 5))
    if len(SCENARIO_COLS) == 1:
        axes = [axes]

    for ax, col in zip(axes, SCENARIO_COLS):
        periods = all_periods[col]
        if not periods:
            continue
        values = [p["value"] * 1000 for p in periods]  # EUR/MWh
        lengths = [p["length"] for p in periods]
        ax.scatter(values, lengths, alpha=0.3, s=10)
        ax.set_xlabel("Preisniveau (EUR/MWh)")
        ax.set_ylabel("Dauer (Intervalle)")
        ax.set_title(f"{col}")
        ax.grid(True, alpha=0.3)
        if max(lengths) > 200:
            ax.set_yscale("log")

    fig.suptitle("Preisniveau vs. Dauer konstanter Perioden")
    path = output_dir / "05_constant_periods_price_vs_duration.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return f"![Preis vs Dauer]({path.name})\n\n"


def plot_constant_periods_timeline(
    df: pd.DataFrame, all_periods: dict, output_dir: Path
) -> str:
    """Zeitlicher Verlauf: Markierung konstanter Perioden auf der Preis-Zeitreihe (Ausschnitt erstes Jahr)."""
    first_year = df["year"].iloc[0]
    year_mask = df["year"] == first_year
    df_year = df.loc[year_mask].copy()

    fig, axes = plt.subplots(len(SCENARIO_COLS), 1, figsize=(16, 4 * len(SCENARIO_COLS)))
    if len(SCENARIO_COLS) == 1:
        axes = [axes]

    for ax, col in zip(axes, SCENARIO_COLS):
        prices = df_year[col].values * 1000  # EUR/MWh
        x = np.arange(len(prices))
        ax.plot(x, prices, linewidth=0.3, color="steelblue", alpha=0.7)

        # Konstante Perioden fuer dieses Jahr
        year_periods = find_constant_periods(df_year[col].values)
        for p in year_periods:
            if p["length"] >= 12:  # Nur Perioden >= 3h markieren
                ax.axvspan(
                    p["start_idx"], p["end_idx"],
                    alpha=0.3, color="red", linewidth=0
                )

        ax.set_ylabel("EUR/MWh")
        ax.set_title(f"{col} - Jahr {first_year} (rot = konstante Perioden >= 3h)")
        ax.grid(True, alpha=0.3)
        # X-Achse: Monate
        month_ticks = [0]
        days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        cum = 0
        for d in days_per_month:
            cum += d
            month_ticks.append(cum * INTERVALS_PER_DAY)
        month_labels = ["Jan", "Feb", "Mar", "Apr", "Mai", "Jun",
                        "Jul", "Aug", "Sep", "Okt", "Nov", "Dez", ""]
        ax.set_xticks(month_ticks)
        ax.set_xticklabels(month_labels)

    path = output_dir / "06_constant_periods_timeline_year1.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return f"![Timeline Jahr 1]({path.name})\n\n"


# ==============================================================================
# 3. VOLATILITAETS- & VERTEILUNGSANALYSE
# ==============================================================================


def volatility_analysis(df: pd.DataFrame) -> str:
    """Volatilitaets- und Verteilungsanalyse."""
    md = "## 3. Volatilitaets- & Verteilungsanalyse\n\n"

    # Intraday-Spread pro Tag
    md += "### 3.1 Intraday-Spread (Max - Min pro Tag)\n\n"
    md += "| Szenario | Mean Spread | Median Spread | Std Spread | Max Spread | Min Spread |\n"
    md += "|----------|-----------|--------------|-----------|-----------|----------|\n"

    for col in SCENARIO_COLS:
        daily_max = df.groupby(["year", "day_of_year"])[col].max()
        daily_min = df.groupby(["year", "day_of_year"])[col].min()
        spread = (daily_max - daily_min) * 1000  # EUR/MWh
        md += (
            f"| {col} "
            f"| {spread.mean():.2f} EUR/MWh "
            f"| {spread.median():.2f} EUR/MWh "
            f"| {spread.std():.2f} EUR/MWh "
            f"| {spread.max():.2f} EUR/MWh "
            f"| {spread.min():.2f} EUR/MWh |\n"
        )

    # Autokorrelation
    md += "\n### 3.2 Autokorrelation\n\n"
    md += "| Szenario | Lag 1 (15min) | Lag 4 (1h) | Lag 96 (1 Tag) | Lag 672 (1 Woche) |\n"
    md += "|----------|--------------|-----------|---------------|------------------|\n"

    for col in SCENARIO_COLS:
        s = df[col]
        lags = [1, 4, 96, 672]
        corrs = []
        for lag in lags:
            corr = s.autocorr(lag=lag)
            corrs.append(f"{corr:.4f}")
        md += f"| {col} | " + " | ".join(corrs) + " |\n"

    # Tage ohne Preisaenderung
    md += "\n### 3.3 Tage mit Null-Volatilitaet (identischer Preis den ganzen Tag)\n\n"
    for col in SCENARIO_COLS:
        daily_std = df.groupby(["year", "day_of_year"])[col].std()
        zero_vol_days = (daily_std < CONST_TOLERANCE).sum()
        total_days = len(daily_std)
        md += f"- **{col}**: {zero_vol_days} von {total_days} Tagen ({zero_vol_days / total_days * 100:.2f}%)\n"

    md += "\n"
    return md


def plot_price_distribution(df: pd.DataFrame, output_dir: Path) -> str:
    """Preisverteilung als Histogramm."""
    fig, axes = plt.subplots(1, len(SCENARIO_COLS), figsize=(6 * len(SCENARIO_COLS), 5))
    if len(SCENARIO_COLS) == 1:
        axes = [axes]

    for ax, col in zip(axes, SCENARIO_COLS):
        prices = df[col].values * 1000  # EUR/MWh
        ax.hist(prices, bins=100, edgecolor="black", alpha=0.7, density=True)
        ax.axvline(np.mean(prices), color="red", linestyle="--", label=f"Mean: {np.mean(prices):.1f}")
        ax.axvline(np.median(prices), color="green", linestyle="--", label=f"Median: {np.median(prices):.1f}")
        ax.set_xlabel("EUR/MWh")
        ax.set_ylabel("Dichte")
        ax.set_title(f"{col}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Preisverteilung")
    path = output_dir / "07_price_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return f"![Preisverteilung]({path.name})\n\n"


def plot_rolling_volatility(df: pd.DataFrame, output_dir: Path) -> str:
    """Rolling Volatility (7-Tage-Fenster)."""
    window = 7 * INTERVALS_PER_DAY  # 7 Tage
    fig, ax = plt.subplots(figsize=(16, 5))

    for col in SCENARIO_COLS:
        rolling_std = df[col].rolling(window=window, center=True).std() * 1000
        # Subsample fuer Performance (jeden Tag ein Wert)
        subsample = rolling_std.iloc[::INTERVALS_PER_DAY]
        ax.plot(
            df[COL_TIMESTAMP].iloc[subsample.index],
            subsample.values,
            linewidth=0.5,
            label=col,
            alpha=0.8,
        )

    ax.set_xlabel("Zeit")
    ax.set_ylabel("Rolling Std (7-Tage, EUR/MWh)")
    ax.set_title("Rolling Volatilitaet (7-Tage-Fenster)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    path = output_dir / "08_rolling_volatility.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return f"![Rolling Volatility]({path.name})\n\n"


# ==============================================================================
# 4. SAISONALE & ZEITLICHE MUSTER
# ==============================================================================


def seasonal_analysis(df: pd.DataFrame) -> str:
    """Saisonale und zeitliche Muster."""
    md = "## 4. Saisonale & Zeitliche Muster\n\n"

    # Monatliche Durchschnittspreise
    md += "### 4.1 Monatliche Durchschnittspreise (EUR/MWh, alle Jahre gemittelt)\n\n"
    md += "| Monat | " + " | ".join(SCENARIO_COLS) + " |\n"
    md += "|-------|" + "|".join(["--------"] * len(SCENARIO_COLS)) + "|\n"

    month_names = [
        "Jan", "Feb", "Mar", "Apr", "Mai", "Jun",
        "Jul", "Aug", "Sep", "Okt", "Nov", "Dez",
    ]
    monthly = df.groupby("month")[SCENARIO_COLS].mean() * 1000
    for m in range(1, 13):
        if m in monthly.index:
            vals = " | ".join(f"{monthly.loc[m, col]:.2f}" for col in SCENARIO_COLS)
            md += f"| {month_names[m-1]} | {vals} |\n"

    # Sommer vs Winter
    md += "\n### 4.2 Sommer (Apr-Sep) vs. Winter (Okt-Mar)\n\n"
    summer_mask = df["month"].isin([4, 5, 6, 7, 8, 9])
    md += "| Szenario | Sommer Mean (EUR/MWh) | Winter Mean (EUR/MWh) | Differenz |\n"
    md += "|----------|---------------------|---------------------|-----------|\n"
    for col in SCENARIO_COLS:
        s_mean = df.loc[summer_mask, col].mean() * 1000
        w_mean = df.loc[~summer_mask, col].mean() * 1000
        md += f"| {col} | {s_mean:.2f} | {w_mean:.2f} | {w_mean - s_mean:.2f} |\n"

    # Werktag vs Wochenende
    md += "\n### 4.3 Werktag (Mo-Fr) vs. Wochenende (Sa-So)\n\n"
    weekday_mask = df["weekday"] < 5
    md += "| Szenario | Werktag Mean | Wochenende Mean | Differenz |\n"
    md += "|----------|-------------|----------------|-----------|\n"
    for col in SCENARIO_COLS:
        wd_mean = df.loc[weekday_mask, col].mean() * 1000
        we_mean = df.loc[~weekday_mask, col].mean() * 1000
        md += f"| {col} | {wd_mean:.2f} | {we_mean:.2f} | {wd_mean - we_mean:.2f} |\n"

    md += "\n"
    return md


def plot_avg_daily_profile(df: pd.DataFrame, output_dir: Path) -> str:
    """Durchschnittliches Tagesprofil (96 Intervalle)."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for col in SCENARIO_COLS:
        profile = df.groupby("interval_of_day")[col].mean() * 1000  # EUR/MWh
        hours = profile.index / INTERVALS_PER_HOUR
        ax.plot(hours, profile.values, label=col, linewidth=1.5)

    ax.set_xlabel("Stunde")
    ax.set_ylabel("Durchschnittspreis (EUR/MWh)")
    ax.set_title("Durchschnittliches Tagesprofil (alle Jahre)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))

    path = output_dir / "09_avg_daily_profile.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return f"![Tagesprofil]({path.name})\n\n"


def plot_avg_weekly_profile(df: pd.DataFrame, output_dir: Path) -> str:
    """Durchschnittliches Wochenprofil."""
    fig, ax = plt.subplots(figsize=(14, 5))

    for col in SCENARIO_COLS:
        # Wochentag + Stunde kombiniert
        df_tmp = df[[col, "weekday", "interval_of_day"]].copy()
        df_tmp["week_interval"] = df_tmp["weekday"] * INTERVALS_PER_DAY + df_tmp["interval_of_day"]
        profile = df_tmp.groupby("week_interval")[col].mean() * 1000
        hours = profile.index / INTERVALS_PER_HOUR
        ax.plot(hours, profile.values, label=col, linewidth=0.8)

    day_labels = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
    day_ticks = [d * 24 for d in range(7)]
    ax.set_xticks(day_ticks)
    ax.set_xticklabels(day_labels)
    ax.set_xlabel("Wochentag")
    ax.set_ylabel("Durchschnittspreis (EUR/MWh)")
    ax.set_title("Durchschnittliches Wochenprofil (alle Jahre)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    for d in range(7):
        ax.axvline(d * 24, color="gray", linestyle=":", alpha=0.3)

    path = output_dir / "10_avg_weekly_profile.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return f"![Wochenprofil]({path.name})\n\n"


def plot_monthly_heatmap(df: pd.DataFrame, output_dir: Path) -> str:
    """Heatmap: Monatliche Durchschnittspreise pro Jahr."""
    fig, axes = plt.subplots(1, len(SCENARIO_COLS), figsize=(6 * len(SCENARIO_COLS), 6))
    if len(SCENARIO_COLS) == 1:
        axes = [axes]

    years = sorted(df["year"].unique())

    for ax, col in zip(axes, SCENARIO_COLS):
        monthly = df.groupby(["year", "month"])[col].mean() * 1000
        matrix = np.full((len(years), 12), np.nan)
        for i, y in enumerate(years):
            for m in range(1, 13):
                if (y, m) in monthly.index:
                    matrix[i, m - 1] = monthly[(y, m)]

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", origin="upper")
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years, fontsize=7)
        ax.set_xticks(range(12))
        ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
        ax.set_title(f"{col}")
        fig.colorbar(im, ax=ax, label="EUR/MWh", shrink=0.8)

    fig.suptitle("Monatliche Durchschnittspreise (EUR/MWh)")
    path = output_dir / "11_monthly_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return f"![Monatliche Heatmap]({path.name})\n\n"


# ==============================================================================
# 5. SZENARIO-VERGLEICH
# ==============================================================================


def scenario_comparison(df: pd.DataFrame) -> str:
    """Vergleich zwischen Szenarien."""
    md = "## 5. Szenario-Vergleich\n\n"

    # Korrelation
    md += "### 5.1 Korrelationsmatrix\n\n"
    corr = df[SCENARIO_COLS].corr()
    md += "| | " + " | ".join(SCENARIO_COLS) + " |\n"
    md += "|--|" + "|".join(["------"] * len(SCENARIO_COLS)) + "|\n"
    for col_a in SCENARIO_COLS:
        vals = " | ".join(f"{corr.loc[col_a, col_b]:.6f}" for col_b in SCENARIO_COLS)
        md += f"| {col_a} | {vals} |\n"

    # Spreads
    md += "\n### 5.2 Spreads zwischen Szenarien (EUR/MWh)\n\n"
    if len(SCENARIO_COLS) >= 2:
        pairs = [
            (COL_HIGH, COL_LOW, "High - Low"),
            (COL_HIGH, COL_MID, "High - MID"),
            (COL_MID, COL_LOW, "MID - Low"),
        ]
        md += "| Spread | Mean | Median | Std | Min | Max |\n"
        md += "|--------|------|--------|-----|-----|-----|\n"

        for col_a, col_b, label in pairs:
            if col_a in df.columns and col_b in df.columns:
                spread = (df[col_a] - df[col_b]) * 1000
                md += (
                    f"| {label} "
                    f"| {spread.mean():.3f} "
                    f"| {spread.median():.3f} "
                    f"| {spread.std():.3f} "
                    f"| {spread.min():.3f} "
                    f"| {spread.max():.3f} |\n"
                )

    # Jaehrliche Divergenz
    md += "\n### 5.3 Jaehrliche Divergenz (Spread High-Low, EUR/MWh)\n\n"
    if COL_HIGH in df.columns and COL_LOW in df.columns:
        md += "| Jahr | Mean Spread | Std Spread |\n"
        md += "|------|-----------|----------|\n"
        for year in sorted(df["year"].unique()):
            year_data = df[df["year"] == year]
            spread = (year_data[COL_HIGH] - year_data[COL_LOW]) * 1000
            md += f"| {year} | {spread.mean():.3f} | {spread.std():.3f} |\n"

    # Sind Szenarien nur verschobene Versionen voneinander?
    md += "\n### 5.4 Ist der Spread konstant oder variabel?\n\n"
    if len(SCENARIO_COLS) >= 2:
        for col_a, col_b, label in pairs:
            if col_a in df.columns and col_b in df.columns:
                spread = df[col_a] - df[col_b]
                unique_spreads = spread.nunique()
                const_spread_pct = spread.value_counts().iloc[0] / len(spread) * 100 if len(spread) > 0 else 0
                md += (
                    f"- **{label}**: {unique_spreads:,} verschiedene Spread-Werte. "
                    f"Haeufigster Wert: {spread.mode().iloc[0] * 1000:.3f} EUR/MWh "
                    f"(Anteil: {const_spread_pct:.1f}%)\n"
                )

    md += "\n"
    return md


def plot_spread_over_time(df: pd.DataFrame, output_dir: Path) -> str:
    """Spread zwischen Szenarien ueber Zeit (taeglich gemittelt)."""
    if COL_HIGH not in df.columns or COL_LOW not in df.columns:
        return ""

    fig, ax = plt.subplots(figsize=(16, 5))

    df_daily = df.set_index(COL_TIMESTAMP)[SCENARIO_COLS].resample("D").mean() * 1000

    if COL_HIGH in df_daily.columns and COL_LOW in df_daily.columns:
        spread = df_daily[COL_HIGH] - df_daily[COL_LOW]
        ax.plot(spread.index, spread.values, linewidth=0.5, color="purple", alpha=0.7)
        ax.axhline(spread.mean(), color="red", linestyle="--", label=f"Mean: {spread.mean():.2f} EUR/MWh")

    ax.set_xlabel("Zeit")
    ax.set_ylabel("Spread High-Low (EUR/MWh)")
    ax.set_title("Taeglicher Spread High - Low")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = output_dir / "12_spread_over_time.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return f"![Spread ueber Zeit]({path.name})\n\n"


# ==============================================================================
# 6. ZUSAETZLICHE KONSTANTE-PERIODEN DIAGRAMME
# ==============================================================================


def plot_constant_periods_per_year(
    df: pd.DataFrame, all_periods: dict, output_dir: Path
) -> str:
    """Anteil konstanter Intervalle pro Jahr als Balkendiagramm."""
    years = sorted(df["year"].unique())
    fig, ax = plt.subplots(figsize=(14, 5))

    bar_width = 0.25
    x = np.arange(len(years))

    for i, col in enumerate(SCENARIO_COLS):
        pcts = []
        for year in years:
            year_mask = df["year"] == year
            year_values = df.loc[year_mask, col].values
            year_periods = find_constant_periods(year_values)
            total_const = sum(p["length"] for p in year_periods)
            pcts.append(total_const / len(year_values) * 100)
        ax.bar(x + i * bar_width, pcts, bar_width, label=col, alpha=0.8)

    ax.set_xlabel("Jahr")
    ax.set_ylabel("Anteil konstanter Intervalle (%)")
    ax.set_title("Anteil konstanter Perioden pro Jahr und Szenario")
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(years, rotation=45, fontsize=7)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    path = output_dir / "13_constant_pct_per_year.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return f"![Konstant pro Jahr]({path.name})\n\n"


# ==============================================================================
# HAUPTPROGRAMM
# ==============================================================================


def main() -> None:
    """Fuehre alle Analysen durch und schreibe Ergebnisse."""
    print("=" * 60)
    print("STROMPREIS-DATEN ANALYSE")
    print("=" * 60)

    output_dir = ensure_output_dir()
    print(f"\nLade Daten aus: {CSV_PATH}")
    df = load_data(CSV_PATH)

    # Globale Intervall-Konstanten aus erkannter Aufloesung setzen
    global INTERVALS_PER_DAY, INTERVALS_PER_YEAR, INTERVALS_PER_HOUR, INTERVALS_PER_WEEK
    INTERVALS_PER_DAY = df.attrs["intervals_per_day"]
    INTERVALS_PER_YEAR = df.attrs["intervals_per_year"]
    INTERVALS_PER_HOUR = df.attrs["intervals_per_hour"]
    INTERVALS_PER_WEEK = INTERVALS_PER_DAY * 7
    minutes_per_interval = df.attrs["minutes_per_interval"]

    n_years = df.attrs["n_years"]
    print(f"Geladen: {len(df):,} Zeilen, {n_years} Jahre ({df['year'].min()}-{df['year'].max()})")
    print(f"Erkannte Aufloesung: {minutes_per_interval}-Minuten-Intervalle ({INTERVALS_PER_YEAR:,} pro Jahr)")
    print(f"Spalten: {list(df.columns[:4])}")
    print(f"Output-Verzeichnis: {output_dir}")

    # Markdown-Report aufbauen
    md = "# Strompreis-Daten Analyse\n\n"
    md += f"- **Datei**: `{CSV_PATH}`\n"
    md += f"- **Zeilen**: {len(df):,}\n"
    md += f"- **Jahre**: {n_years} ({df['year'].min()} - {df['year'].max()})\n"
    md += f"- **Szenarien**: {', '.join(SCENARIO_COLS)}\n"
    md += f"- **Aufloesung**: {minutes_per_interval}-Minuten-Intervalle ({INTERVALS_PER_YEAR:,} pro Jahr)\n\n"
    md += "---\n\n"

    # 1. Deskriptive Statistik
    print("\n[1/5] Deskriptive Statistik...")
    md += descriptive_statistics(df)
    md += plot_annual_means(df, output_dir)
    md += plot_annual_boxplots(df, output_dir)
    md += "---\n\n"

    # 2. Konstante-Perioden-Analyse
    print("[2/5] Konstante-Perioden-Analyse...")
    const_md, all_periods = constant_periods_analysis(df)
    md += const_md
    md += plot_constant_periods_heatmap(df, all_periods, output_dir)
    md += plot_constant_periods_duration_hist(all_periods, output_dir)
    md += plot_constant_periods_by_price_level(all_periods, output_dir)
    md += plot_constant_periods_timeline(df, all_periods, output_dir)
    md += plot_constant_periods_per_year(df, all_periods, output_dir)
    md += "---\n\n"

    # 3. Volatilitaet
    print("[3/5] Volatilitaetsanalyse...")
    md += volatility_analysis(df)
    md += plot_price_distribution(df, output_dir)
    md += plot_rolling_volatility(df, output_dir)
    md += "---\n\n"

    # 4. Saisonale Muster
    print("[4/5] Saisonale Analyse...")
    md += seasonal_analysis(df)
    md += plot_avg_daily_profile(df, output_dir)
    md += plot_avg_weekly_profile(df, output_dir)
    md += plot_monthly_heatmap(df, output_dir)
    md += "---\n\n"

    # 5. Szenario-Vergleich
    print("[5/5] Szenario-Vergleich...")
    md += scenario_comparison(df)
    md += plot_spread_over_time(df, output_dir)

    # Report schreiben
    report_path = output_dir / "analysis_report.md"
    report_path.write_text(md, encoding="utf-8")
    print(f"\n{'=' * 60}")
    print(f"FERTIG!")
    print(f"Report:     {report_path}")
    print(f"Diagramme:  {output_dir}/*.png")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
