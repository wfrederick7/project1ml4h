from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path.home() / "project1ml4h"
DATA_DIR = BASE_DIR / "data" / "processed_derived"
RESULTS_DIR = BASE_DIR / "supervised_learning" / "results" / "q2_1_1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "a_linear": DATA_DIR / "set_a_linear.parquet",
    "b_linear": DATA_DIR / "set_b_linear.parquet",
    "c_linear": DATA_DIR / "set_c_linear.parquet",
    "a_ffill": DATA_DIR / "set_a_ffill.parquet",
    "b_ffill": DATA_DIR / "set_b_ffill.parquet",
    "c_ffill": DATA_DIR / "set_c_ffill.parquet",
}


# =========================================================
# Columns
# =========================================================

ID_COL = "PatientID"
TIME_COL = "Time"
TARGET_COL = "label"

STATIC_COLS = ["Age", "Gender", "Height", "Weight_static"]

DYNAMIC_COLS = [
    "ALP", "ALT", "AST", "Albumin", "BUN", "Bilirubin", "Cholesterol",
    "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT",
    "HR", "K", "Lactate", "MAP", "MechVent", "Mg", "NIDiasABP", "NIMAP",
    "NISysABP", "Na", "PaCO2", "PaO2", "Platelets", "RespRate", "SaO2",
    "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC", "Weight",
    "pH"
]

FEATURE_COLS = STATIC_COLS + DYNAMIC_COLS


# =========================================================
# Helpers
# =========================================================

def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_parquet(path)


def make_last_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one patient-level row by taking the final time step
    after causal preprocessing.
    """
    df = df.sort_values([ID_COL, TIME_COL]).copy()

    # Because each patient has 49 time steps, taking the last row
    # gives the hour-48 state after forward fill / scaling.
    last_df = df.groupby(ID_COL, as_index=False).last()

    out = last_df[[ID_COL] + FEATURE_COLS + [TARGET_COL]].copy()

    assert out[ID_COL].is_unique, "Expected exactly one row per patient."
    assert out.shape[1] == 1 + len(FEATURE_COLS) + 1
    return out


def split_X_y(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(int).copy()
    return X, y


def evaluate_probs(y_true, y_prob) -> dict:
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
    }


def fit_and_score_both(model, X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    return {
        "val": evaluate_probs(y_val, val_prob),
        "test": evaluate_probs(y_test, test_prob),
    }


def compute_scale_pos_weight(y: pd.Series) -> float:
    """
    Compute negative / positive ratio for class imbalance handling.
    Returns 1.0 if the positive class count is zero for safety.
    """
    y = pd.Series(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    if n_pos == 0:
        return 1.0

    return n_neg / n_pos


# =========================================================
# Logistic Regression
# =========================================================

def run_logistic_regression():
    print("\n=== Logistic Regression (linear + last) ===")

    train_df = make_last_features(load_df(FILES["a_linear"]))
    val_df   = make_last_features(load_df(FILES["b_linear"]))
    test_df  = make_last_features(load_df(FILES["c_linear"]))

    X_train, y_train = split_X_y(train_df)
    X_val, y_val     = split_X_y(val_df)
    X_test, y_test   = split_X_y(test_df)

    candidates = [
        {"C": 0.01, "class_weight": None},
        {"C": 0.1,  "class_weight": None},
        {"C": 1.0,  "class_weight": None},
        {"C": 10.0, "class_weight": None},
        {"C": 0.01, "class_weight": "balanced"},
        {"C": 0.1,  "class_weight": "balanced"},
        {"C": 1.0,  "class_weight": "balanced"},
        {"C": 10.0, "class_weight": "balanced"},
    ]

    rows = []
    best_score = (-np.inf, -np.inf)
    best_cfg = None
    best_metrics = None

    for cfg in candidates:
        model = LogisticRegression(
            C=cfg["C"],
            class_weight=cfg["class_weight"],
            max_iter=5000,
            solver="liblinear",
            random_state=42
        )

        metrics = fit_and_score_both(
            model, X_train, y_train, X_val, y_val, X_test, y_test
        )

        val_metrics = metrics["val"]
        test_metrics = metrics["test"]

        rows.append({
            "model": "LogisticRegression",
            "C": cfg["C"],
            "class_weight": str(cfg["class_weight"]),
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "test_auroc": test_metrics["auroc"],
            "test_auprc": test_metrics["auprc"],
        })

        score = (val_metrics["auprc"], val_metrics["auroc"])
        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_metrics = metrics

    results_df = pd.DataFrame(rows).sort_values(
        by=["val_auprc", "val_auroc"],
        ascending=False
    )
    results_df.to_csv(RESULTS_DIR / "logistic_regression_results.csv", index=False)

    print("Best Logistic Regression config:", best_cfg)
    print("Validation:", best_metrics["val"])
    print("Test:", best_metrics["test"])

    return best_cfg, best_metrics, results_df


# =========================================================
# Random Forest
# =========================================================

def run_random_forest():
    print("\n=== Random Forest (ffill + last) ===")

    train_df = make_last_features(load_df(FILES["a_ffill"]))
    val_df   = make_last_features(load_df(FILES["b_ffill"]))
    test_df  = make_last_features(load_df(FILES["c_ffill"]))

    X_train, y_train = split_X_y(train_df)
    X_val, y_val     = split_X_y(val_df)
    X_test, y_test   = split_X_y(test_df)

    candidates = [
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1, "class_weight": None},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1, "class_weight": None},
        {"n_estimators": 300, "max_depth": 10,   "min_samples_leaf": 1, "class_weight": None},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 5, "class_weight": None},
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1, "class_weight": "balanced"},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1, "class_weight": "balanced"},
        {"n_estimators": 300, "max_depth": 10,   "min_samples_leaf": 1, "class_weight": "balanced"},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 5, "class_weight": "balanced"},
    ]

    rows = []
    best_score = (-np.inf, -np.inf)
    best_cfg = None
    best_metrics = None

    for cfg in candidates:
        model = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            class_weight=cfg["class_weight"],
            random_state=42,
            n_jobs=-1
        )

        metrics = fit_and_score_both(
            model, X_train, y_train, X_val, y_val, X_test, y_test
        )

        val_metrics = metrics["val"]
        test_metrics = metrics["test"]

        rows.append({
            "model": "RandomForest",
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "min_samples_leaf": cfg["min_samples_leaf"],
            "class_weight": str(cfg["class_weight"]),
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "test_auroc": test_metrics["auroc"],
            "test_auprc": test_metrics["auprc"],
        })

        score = (val_metrics["auprc"], val_metrics["auroc"])
        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_metrics = metrics

    results_df = pd.DataFrame(rows).sort_values(
        by=["val_auprc", "val_auroc"],
        ascending=False
    )
    results_df.to_csv(RESULTS_DIR / "random_forest_results.csv", index=False)

    print("Best Random Forest config:", best_cfg)
    print("Validation:", best_metrics["val"])
    print("Test:", best_metrics["test"])

    return best_cfg, best_metrics, results_df


# =========================================================
# XGBoost
# =========================================================

def run_xgboost():
    print("\n=== XGBoost (ffill + last) ===")

    train_df = make_last_features(load_df(FILES["a_ffill"]))
    val_df   = make_last_features(load_df(FILES["b_ffill"]))
    test_df  = make_last_features(load_df(FILES["c_ffill"]))

    X_train, y_train = split_X_y(train_df)
    X_val, y_val     = split_X_y(val_df)
    X_test, y_test   = split_X_y(test_df)

    pos_weight = compute_scale_pos_weight(y_train)

    candidates = [
        {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 3,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 1.0,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 1.0,
        },
        {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 4,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 1.0,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 1.0,
        },
        {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 3,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": pos_weight,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": pos_weight,
        },
        {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 4,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": pos_weight,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": pos_weight,
        },
    ]

    rows = []
    best_score = (-np.inf, -np.inf)
    best_cfg = None
    best_metrics = None

    for cfg in candidates:
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            max_depth=cfg["max_depth"],
            min_child_weight=cfg["min_child_weight"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            scale_pos_weight=cfg["scale_pos_weight"],
            random_state=42,
            n_jobs=-1,
        )

        metrics = fit_and_score_both(
            model, X_train, y_train, X_val, y_val, X_test, y_test
        )

        val_metrics = metrics["val"]
        test_metrics = metrics["test"]

        rows.append({
            "model": "XGBoost",
            "n_estimators": cfg["n_estimators"],
            "learning_rate": cfg["learning_rate"],
            "max_depth": cfg["max_depth"],
            "min_child_weight": cfg["min_child_weight"],
            "subsample": cfg["subsample"],
            "colsample_bytree": cfg["colsample_bytree"],
            "scale_pos_weight": cfg["scale_pos_weight"],
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "test_auroc": test_metrics["auroc"],
            "test_auprc": test_metrics["auprc"],
        })

        score = (val_metrics["auprc"], val_metrics["auroc"])
        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_metrics = metrics

    results_df = pd.DataFrame(rows).sort_values(
        by=["val_auprc", "val_auroc"],
        ascending=False
    )
    results_df.to_csv(RESULTS_DIR / "xgboost_results.csv", index=False)

    print("Best XGBoost config:", best_cfg)
    print("Validation:", best_metrics["val"])
    print("Test:", best_metrics["test"])

    return best_cfg, best_metrics, results_df


# =========================================================
# Main
# =========================================================

def main():
    best_lr_cfg, best_lr_metrics, lr_results = run_logistic_regression()
    best_rf_cfg, best_rf_metrics, rf_results = run_random_forest()
    best_xgb_cfg, best_xgb_metrics, xgb_results = run_xgboost()

    summary = {
        "feature_type": "last",
        "description": "One row per patient using the final hourly state after causal preprocessing.",
        "logistic_regression": {
            "best_config": best_lr_cfg,
            "validation": best_lr_metrics["val"],
            "test": best_lr_metrics["test"],
        },
        "random_forest": {
            "best_config": best_rf_cfg,
            "validation": best_rf_metrics["val"],
            "test": best_rf_metrics["test"],
        },
        "xgboost": {
            "best_config": best_xgb_cfg,
            "validation": best_xgb_metrics["val"],
            "test": best_xgb_metrics["test"],
        },
    }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved all results to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()