import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pv_bess_model.config.defaults import CSV_DELIMITER, CSV_DECIMAL_SEPARATOR
from pv_bess_model.main import run

ABS_DELTA_LIMIT = 100
REL_DELTA_LIMIT = 0.01

# --- Pfad-Helpers ------------------------------------------------------------

def _project_root() -> Path:
    """
    Ermittelt robust den Projekt-Root relativ zu dieser Testdatei.
    Annahme: Datei liegt in .../pv_bess_model/tests/test_integration_cashflow.py
    -> Root ist typischerweise 3 Ebenen höher (repo/).
    Passe parents[x] an, falls dein Layout anders ist.
    """
    return Path(__file__).resolve().parents[2]


def _abs_path(p: str | Path) -> Path:
    """
    Macht aus einem ggf. relativen Pfad einen absoluten Pfad relativ zum Projekt-Root.
    """
    p = Path(p)
    return p if p.is_absolute() else (_project_root() / p).resolve()


# --- Vergleichslogik (KEIN pytest-Test!) -------------------------------------

def compare_csv_files(base_file: str | Path, file_to_test_path: str | Path) -> list[str]:
    """
    Vergleicht zwei CSV-Dateien:
      - prüft gleiche Spalten + gleiche Zeilenanzahl
      - vergleicht alle numerischen Werte
      - sammelt alle Stellen, wo |Δ| > ABS_DELTA_LIMIT

    Gibt eine Liste an Fehlerstrings zurück.
    """
    base_file = _abs_path(base_file)
    file_to_test_path = _abs_path(file_to_test_path)

    if not base_file.exists():
        raise FileNotFoundError(f"Base CSV nicht gefunden: {base_file}")
    if not file_to_test_path.exists():
        raise FileNotFoundError(f"Test CSV nicht gefunden: {file_to_test_path}")

    df_base = pd.read_csv(base_file, sep=CSV_DELIMITER, decimal=CSV_DECIMAL_SEPARATOR)
    df_totest = pd.read_csv(file_to_test_path, sep=CSV_DELIMITER, decimal=CSV_DECIMAL_SEPARATOR)

    # Grundlegende Strukturprüfung
    if list(df_base.columns) != list(df_totest.columns):
        raise AssertionError(
            "CSV-Dateien haben unterschiedliche Spalten:\n"
            f"{df_base.columns.tolist()} != {df_totest.columns.tolist()}\n"
            f"base_file={base_file}\n"
            f"test_file={file_to_test_path}"
        )

    if len(df_base) != len(df_totest):
        raise AssertionError(
            "CSV-Dateien haben unterschiedliche Zeilenanzahl: "
            f"{len(df_base)} != {len(df_totest)}\n"
            f"base_file={base_file}\n"
            f"test_file={file_to_test_path}"
        )

    errors: list[str] = []

    # Nur numerische Spalten vergleichen
    numeric_columns = df_base.select_dtypes(include=[np.number]).columns

    for col in numeric_columns:
        # abs delta
        delta =  1 - (df_base[col] / df_totest[col]).abs()

        failing_indices = delta[delta > REL_DELTA_LIMIT].index
        for idx in failing_indices:
            errors.append(
                f"Zeile {idx + 1}, Spalte '{col}': "
                f"{df_base.iloc[idx][col]:.2f} vs {df_totest.iloc[idx][col]:.2f} "
                f"(Δ={delta.iloc[idx]*100:.2f}%)"
            )

    return errors


# --- Pytest Integration Test --------------------------------------------------

@pytest.mark.integration
class TestIntegrationCashflow:
    def test_default_cashflow(self):
        # Eingabe-Dateien (relativ zum Projekt-Root)
        base_file_path = Path(".data/integration_test_inputs/finance/integration_test_cashflows_base.csv")
        base_input_path = Path(".data/integration_test_inputs/finance/integration_test_cashflow.json")

        # Output-Verzeichnis (relativ zum Projekt-Root, robust)
        output_dir = _abs_path(".data/test/integration_tests/")
        output_dir.mkdir(parents=True, exist_ok=True)

        args = argparse.Namespace(
            scenario=str(_abs_path(base_input_path)),
            output=str(output_dir),
            no_mc=True,
            bess_power=None,
            bess_capacity=None,
            verbose=False,
            dry_run=False,
        )

        # Run
        exit_code = run(args)
        assert exit_code == 0, f"Scenario Integrationtest-Cashflow failed with exit code {exit_code}"

        # Erwarteter Output-CSV Name (wie in deinem Code angenommen)
        produced_csv = output_dir / "Cashflow_Integrationtest/Cashflow_Integrationtest_cashflows.csv"

        errors = compare_csv_files(base_file_path, produced_csv)
        if errors:
            max_items = 20  # verhindert extrem lange Fehlermeldungen
            preview = "\n".join(errors[:max_items])
            raise AssertionError(
                f"Es gab {len(errors)} Fehler, in diesen Spalten und Zeilen:\n"
                f"{preview}"
                + ("\n..." if len(errors) > max_items else "")
            )