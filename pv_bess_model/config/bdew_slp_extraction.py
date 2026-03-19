import pandas as pd
import json

# =========================
# Konfiguration
# =========================
EXCEL_FILE = "C:/Users/roescni/Downloads/Kopie_von_Repräsentative_Profile_BDEW_H25_G25_L25_P25_S25_Veröffentlichung.xlsx"
OUTPUT_JSON = "../../.data/bdew_profile_2025.json"

PROFILE_SHEETS = ["H25", "G25", "L25", "P25", "S25"]

MONTHS = [
    "Jan", "Feb", "Mrz", "Apr", "Mai", "Jun",
    "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"
]

DAY_TYPES = ["SA", "FT", "WT"]


def find_data_start(df: pd.DataFrame) -> int:
    """
    Findet die erste Zeile mit echten numerischen Zeitreihenwerten.
    """
    for i in range(len(df)):
        numeric_count = pd.to_numeric(
            df.iloc[i, 1:], errors="coerce"
        ).notna().sum()

        if numeric_count >= 10:
            return i

    raise ValueError("Keine Datenzeile gefunden")


def parse_profile(sheet_name: str) -> dict:
    df = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name, header=None)

    data_start = find_data_start(df)

    # Rohdatenbereich (inkl. möglicher Zeitstrings)
    raw = df.iloc[data_start : data_start + 120, 1:]  # bewusst >96
    offset = 1
    profile = {}

    for m, month in enumerate(MONTHS):
        profile[month] = {}

        for d, day in enumerate(DAY_TYPES):
            col_idx = m * 3 + d + offset

            # ✅ NUR numerische Werte extrahieren
            series = (
                pd.to_numeric(raw.iloc[:, col_idx], errors="coerce")
                .dropna()
                .iloc[:96]
                .tolist()
            )

            if len(series) != 96:
                raise ValueError(
                    f"{sheet_name} {month} {day}: "
                    f"{len(series)} numerische Werte gefunden"
                )

            profile[month][day] = series

    return profile


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