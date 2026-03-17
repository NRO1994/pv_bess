import pandas as pd
import json
from pathlib import Path

# =========================
# Konfiguration
# =========================
EXCEL_FILE = "C:/Users/roescni/Downloads/Kopie_von_Repräsentative_Profile_BDEW_H25_G25_L25_P25_S25_Veröffentlichung.xlsx"
OUTPUT_JSON = ".data/bdew_profile_2025.json"

PROFILE_SHEETS = ["H25", "G25", "L25", "P25", "S25"]

MONTHS = [
    "Jan", "Feb", "Mrz", "Apr", "Mai", "Jun",
    "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"
]

DAY_TYPES = ["SA", "FT", "WT"]

# =========================
# Hilfsfunktion
# =========================
def parse_profile(sheet_name: str) -> dict:
    """
    Liest ein Profil-Sheet und erzeugt:
    Monat -> SA/FT/WT -> [96 Werte]
    """
    # Sheet roh einlesen
    df = pd.read_excel(
        EXCEL_FILE,
        sheet_name=sheet_name,
        header=None
    )

    # Die Zeitzeilen beginnen dort, wo die erste Spalte ein Zeitintervall enthält
    time_row_start = df[df.iloc[:, 0].astype(str).str.contains(":")].index[0]

    df = df.iloc[time_row_start: time_row_start + 96]

    # Werte ohne Zeitspalte
    values = df.iloc[:, 1:].reset_index(drop=True)

    profile_data = {}

    for month_idx, month in enumerate(MONTHS):
        profile_data[month] = {}

        for day_idx, day in enumerate(DAY_TYPES):
            col_idx = month_idx * 3 + day_idx

            series = (
                values.iloc[:, col_idx]
                .astype(float)
                .tolist()
            )

            if len(series) != 96:
                raise ValueError(
                    f"{sheet_name} – {month} – {day}: "
                    f"{len(series)} Werte gefunden"
                )

            profile_data[month][day] = series

    return profile_data


# =========================
# Hauptlogik
# =========================
result = {}

for profile in PROFILE_SHEETS:
    print(f"Lese Profil {profile} …")
    result[profile] = parse_profile(profile)

# =========================
# JSON schreiben
# =========================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"/nFertig ✅ → {OUTPUT_JSON}")