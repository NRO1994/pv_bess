# -*- coding: utf-8 -*-
"""
CART-Regeln (sklearn) aus einer Monte-Carlo-CSV extrahieren (ohne CLI).

Erwartung:
- CSV enthält Feature-Spalten (z.B. CAPEX, OPEX, Availability, CaptureRate, PriceScenario)
- CSV enthält IRR-Spalte (z.B. IRR)
- Target-IRR ist als Konstante hinterlegt (oder optional als Spalte möglich)

Output:
- out_rules/tree_rules.txt         (voller Baum als Text)
- out_rules/leaf_rules.csv         (Leaf-Regeln als IF-THEN inkl. Support/Probs)
- out_rules/leaf_rules.json
- out_rules/metrics.json
"""

import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score


# =========================
# CONFIG (HIER ANPASSEN)
# =========================
CSV_PATH = "C:/Users/roescni/Codebase/CoLocation_Analyse/.data/output/PVA LHG Nordlandkai/PVA LHG Nordlandkai_monte_carlo_1.csv"         # Pfad zu deiner CSV
CSV_SEP = ";"                        # z.B. "," oder ";"
CSV_DECIMAL = ","                    # z.B. "." oder "," (deutsche CSV oft ",")
ENCODING = "utf-8"                   # ggf. "cp1252" bei Windows-Exports

OUTDIR = "cart_rules_out"

# Feature-Spalten:
FEATURE_COLS = [
    "price_scenario",
    "capex_factor_pv",
    "opex_factor_pv",
    "pv_availability_factor",
    "capture_rate_eur_per_kwh"
]

# IRR-Spalte:
IRR_COL = "equity_irr_pct"

# Target IRR: (z.B. 7.5% => 0.075)
TARGET_IRR = 0.081

# Optional: Wenn Target-IRR in einer Spalte pro Run steht, setze:
TARGET_IRR_COL = None  # z.B. "IRR_Target" oder None für Konstante TARGET_IRR

# Klassifizierung vs Regression:
MODE = "classification"  # "classification" oder "regression"

# Toleranzband um Target für Klasse "erreicht":
# Beispiel: 0.001 = ±0.1 %-Punkte in IRR-Einheiten (wenn IRR als 0.075 notiert ist)
EPSILON = 0.005

# Missing-Handling:
DROPNA = False  # True: Zeilen mit NA droppen; False: simple Imputation (Median / "__MISSING__")

# Optional: kategoriale Features explizit angeben (sonst auto via dtype object/category)
CATEGORICAL_OVERRIDE = None  # z.B. ["PriceScenario"]

# Split:
TEST_SIZE = 0.2
RANDOM_STATE = 42

# CART-Parameter (entscheidend für "gute" Regeln):
MAX_DEPTH = 4
MIN_SAMPLES_LEAF = 300
MIN_SAMPLES_SPLIT = 800
CCP_ALPHA = 0.0              # Pruning; z.B. 0.0005 testen
MAX_LEAF_NODES = None        # z.B. 20, um Regelanzahl zu begrenzen
# =========================


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def infer_categoricals(df: pd.DataFrame, features, categorical_override=None):
    if categorical_override is not None and len(categorical_override) > 0:
        return [c for c in categorical_override if c in features]
    cats = []
    for c in features:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]):
            cats.append(c)
    return cats


def simple_missing_handling(df: pd.DataFrame, cols, dropna=False):
    if dropna:
        return df.dropna(subset=cols).copy()

    out = df.copy()
    for c in cols:
        if pd.api.types.is_numeric_dtype(out[c]):
            med = out[c].median(skipna=True)
            out[c] = out[c].fillna(med)
        else:
            out[c] = out[c].astype("object").fillna("__MISSING__")
    return out


def build_targets(df: pd.DataFrame):
    irr = df[IRR_COL].astype(float).values

    if TARGET_IRR_COL is not None:
        target = df[TARGET_IRR_COL].astype(float).values
    else:
        target = np.full_like(irr, TARGET_IRR, dtype=float)

    delta = irr - target

    if MODE == "classification":
        # -1 verfehlt, 0 erreicht, +1 übererfüllt
        y_raw = np.where(delta < -EPSILON, -1, np.where(delta > EPSILON, +1, 0))
        # map to 0..2 for sklearn
        class_order = [-1, 0, +1]
        y_map = {k: i for i, k in enumerate(class_order)}
        y = np.vectorize(y_map.get)(y_raw)
        class_names = ["verfehlt", "erreicht", "übererfüllt"]
        return y, delta, class_names
    else:
        # Regression auf ΔIRR
        return delta.astype(float), delta, None


def get_feature_names_after_ohe(preprocessor, numeric_features, categorical_features):
    feat_names = []
    feat_names.extend(list(numeric_features))
    if len(categorical_features) > 0:
        ohe = preprocessor.named_transformers_["cat"]
        ohe_names = ohe.get_feature_names_out(categorical_features).tolist()
        feat_names.extend(ohe_names)
    return feat_names


def extract_leaf_rules(tree, feature_names, is_classifier=True, class_names=None, labels_full=None):
    """
    Traversiere Baum und generiere Regeln pro Leaf.
    Bei Klassifikation werden class_counts / class_probs immer auf labels_full aufgefüllt,
    auch wenn der trainierte Baum nur eine Teilmenge der Klassen gesehen hat.
    """
    import numpy as np
    from sklearn.tree import _tree

    t = tree.tree_
    rules = []

    # Default für "vollständige" Labels (wenn nicht angegeben)
    if is_classifier and labels_full is None:
        labels_full = [0, 1, 2]  # verfehlt, erreicht, übererfüllt

    # Klassen, die der Baum tatsächlich kennt
    tree_classes = list(getattr(tree, "classes_", [])) if is_classifier else []
    tree_class_to_idx = {c: i for i, c in enumerate(tree_classes)}
    full_label_to_idx = {c: i for i, c in enumerate(labels_full)} if is_classifier else {}

    def recurse(node, conds):
        if t.feature[node] != _tree.TREE_UNDEFINED:
            fname = feature_names[t.feature[node]]
            thr = t.threshold[node]
            recurse(t.children_left[node], conds + [f"{fname} <= {thr:.6g}"])
            recurse(t.children_right[node], conds + [f"{fname} > {thr:.6g}"])
        else:
            n_samples = int(t.n_node_samples[node])

            if is_classifier:
                counts_partial = t.value[node][0]  # Länge = len(tree.classes_)
                counts_full = np.zeros(len(labels_full), dtype=float)

                for cls_label, idx_partial in tree_class_to_idx.items():
                    if cls_label in full_label_to_idx:
                        counts_full[full_label_to_idx[cls_label]] = counts_partial[idx_partial]

                total = counts_full.sum()
                probs_full = (counts_full / total) if total > 0 else counts_full

                pred_idx = int(np.argmax(counts_full)) if len(counts_full) else 0
                pred_label_name = class_names[pred_idx] if class_names else str(labels_full[pred_idx])

                rules.append({
                    "rule": " AND ".join(conds) if conds else "(ALL)",
                    "leaf_samples": n_samples,
                    "prediction": pred_label_name,
                    "class_counts": counts_full.tolist(),
                    "class_probs": probs_full.tolist(),
                })
            else:
                pred_value = float(t.value[node][0][0])
                rules.append({
                    "rule": " AND ".join(conds) if conds else "(ALL)",
                    "leaf_samples": n_samples,
                    "prediction": pred_value
                })

    recurse(0, [])
    rules.sort(key=lambda r: r["leaf_samples"], reverse=True)
    return rules

def main():
    ensure_outdir(OUTDIR)

    # 1) CSV lesen
    df = pd.read_csv(
        CSV_PATH,
        sep=CSV_SEP,
        decimal=CSV_DECIMAL,
        encoding=ENCODING,
        low_memory=False
    )

    needed = set(FEATURE_COLS + [IRR_COL])
    if TARGET_IRR_COL is not None:
        needed.add(TARGET_IRR_COL)

    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Folgende Spalten fehlen in der CSV: {missing}")

    df = df[list(needed)].copy()

    # 2) Missing-Handling
    df = simple_missing_handling(df, list(needed), dropna=DROPNA)

    # 3) Targets
    y, delta, class_names = build_targets(df)
    X = df[FEATURE_COLS].copy()

    # 4) Feature-Typen & Preprocessing
    categorical_features = infer_categoricals(X, FEATURE_COLS, CATEGORICAL_OVERRIDE)
    numeric_features = [c for c in FEATURE_COLS if c not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # 5) Train/Test split
    if MODE == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
        tree = DecisionTreeClassifier(
            criterion="gini",
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            min_samples_split=MIN_SAMPLES_SPLIT,
            ccp_alpha=CCP_ALPHA,
            max_leaf_nodes=MAX_LEAF_NODES,
            random_state=RANDOM_STATE,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        tree = DecisionTreeRegressor(
            criterion="squared_error",
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            min_samples_split=MIN_SAMPLES_SPLIT,
            ccp_alpha=CCP_ALPHA,
            max_leaf_nodes=MAX_LEAF_NODES,
            random_state=RANDOM_STATE,
        )

    pipe = Pipeline(steps=[("prep", preprocessor), ("tree", tree)])
    pipe.fit(X_train, y_train)

    # 6) Evaluation
    y_pred = pipe.predict(X_test)
    metrics = {"mode": MODE}

    if MODE == "classification":
        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        labels = [0, 1, 2]  # entspricht: verfehlt, erreicht, übererfüllt
        metrics["classification_report"] = classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=class_names,
            digits=4,
            output_dict=True,
            zero_division=0
        )
    else:
        metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
        metrics["r2"] = float(r2_score(y_test, y_pred))

    with open(os.path.join(OUTDIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 7) Regeln exportieren
    fitted_prep = pipe.named_steps["prep"]
    fitted_tree = pipe.named_steps["tree"]

    feature_names = get_feature_names_after_ohe(fitted_prep, numeric_features, categorical_features)

    # A) Baum als Text
    tree_text = export_text(fitted_tree, feature_names=feature_names, decimals=4)
    with open(os.path.join(OUTDIR, "tree_rules.txt"), "w", encoding="utf-8") as f:
        f.write(tree_text)

    # B) Leaf-Regeln als CSV/JSON
    leaf_rules = extract_leaf_rules(
        fitted_tree,
        feature_names=feature_names,
        is_classifier=(MODE == "classification"),
        class_names=class_names,
    )

    with open(os.path.join(OUTDIR, "leaf_rules.json"), "w", encoding="utf-8") as f:
        json.dump(leaf_rules, f, ensure_ascii=False, indent=2)

    rows = []
    if MODE == "classification":
        for r in leaf_rules:
            row = {
                "leaf_samples": r["leaf_samples"],
                "prediction": r["prediction"],
                "rule": r["rule"],
            }
            for i, name in enumerate(class_names):
                row[f"prob_{name}"] = r["class_probs"][i]
                row[f"count_{name}"] = r["class_counts"][i]
            rows.append(row)
    else:
        for r in leaf_rules:
            rows.append({
                "leaf_samples": r["leaf_samples"],
                "prediction_delta_irr": r["prediction"],
                "rule": r["rule"],
            })

    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "leaf_rules.csv"), index=False, encoding="utf-8")

    # 8) Kurzreport in Konsole
    print(f"[OK] Fertig. Output in: {OUTDIR}")
    if MODE == "classification":
        rep = metrics["classification_report"]
        print(f"Accuracy: {rep['accuracy']:.4f}")
        print("Top-5 Regeln nach Support (leaf_samples):")
        for r in leaf_rules[:5]:
            print(f"- n={r['leaf_samples']}: {r['prediction']} | {r['rule']}")
    else:
        print(f"MAE: {metrics['mae']:.6f} | R²: {metrics['r2']:.4f}")
        print("Top-5 Regeln nach Support (leaf_samples):")
        for r in leaf_rules[:5]:
            print(f"- n={r['leaf_samples']}: ΔIRR={r['prediction']:.6f} | {r['rule']}")


if __name__ == "__main__":
    main()