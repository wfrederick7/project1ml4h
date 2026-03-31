from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path.home() / "project1ml4h"
DATA_DIR = BASE_DIR / "data" / "processed_derived"
RESULTS_DIR = BASE_DIR / "supervised_learning" / "results" / "q2_1_2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
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


# =========================================================
# Helpers
# =========================================================

def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_parquet(path)


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


def compute_slope(values: pd.Series, times: pd.Series) -> float:
    """
    Linear slope of values over time using least squares.
    Returns 0.0 if fewer than 2 non-missing points exist.
    """
    mask = values.notna()
    if mask.sum() < 2:
        return 0.0

    x = times[mask].to_numpy(dtype=float)
    y = values[mask].to_numpy(dtype=float)

    x_mean = x.mean()
    y_mean = y.mean()

    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0

    slope = np.sum((x - x_mean) * (y - y_mean)) / denom
    return float(slope)


def engineer_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create one row per patient with engineered summaries.
    """
    df = df.sort_values([ID_COL, TIME_COL]).copy()
    rows = []

    for patient_id, g in df.groupby(ID_COL):
        row = {ID_COL: patient_id}

        # Static features
        for col in STATIC_COLS:
            row[col] = float(g[col].iloc[0])

        times = g[TIME_COL]

        # Dynamic features
        for col in DYNAMIC_COLS:
            s = g[col]

            first_val = float(s.iloc[0])
            last_val = float(s.iloc[-1])

            row[f"{col}_first"] = first_val
            row[f"{col}_last"] = last_val
            row[f"{col}_mean"] = float(s.mean())
            row[f"{col}_min"] = float(s.min())
            row[f"{col}_max"] = float(s.max())
            row[f"{col}_std"] = float(s.std(ddof=0))
            row[f"{col}_delta"] = last_val - first_val
            row[f"{col}_slope"] = compute_slope(s, times)

        row[TARGET_COL] = int(g[TARGET_COL].iloc[0])
        rows.append(row)

    out = pd.DataFrame(rows)

    # Safety for any degenerate std/slope cases
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    assert out[ID_COL].is_unique, "Expected one row per patient."
    return out


def split_X_y(df: pd.DataFrame):
    X = df.drop(columns=[ID_COL, TARGET_COL]).copy()
    y = df[TARGET_COL].astype(int).copy()
    return X, y


def scale_for_logistic_regression(X_train, X_val, X_test):
    """
    Fit scaler on training features only, then transform val/test.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# =========================================================
# Logistic Regression
# =========================================================

def run_logistic_regression(train_feat, val_feat, test_feat):
    print("\n=== Logistic Regression (engineered features) ===")

    X_train, y_train = split_X_y(train_feat)
    X_val, y_val = split_X_y(val_feat)
    X_test, y_test = split_X_y(test_feat)

    X_train, X_val, X_test, scaler = scale_for_logistic_regression(X_train, X_val, X_test)

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
            max_iter=10000,
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
            "n_features": X_train.shape[1],
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
        by=["val_auprc", "val_auroc"], ascending=False
    )
    results_df.to_csv(RESULTS_DIR / "logistic_regression_engineered_results.csv", index=False)

    print("Best Logistic Regression config:", best_cfg)
    print("Validation:", best_metrics["val"])
    print("Test:", best_metrics["test"])

    return best_cfg, best_metrics, results_df


# =========================================================
# Random Forest
# =========================================================

def run_random_forest(train_feat, val_feat, test_feat):
    print("\n=== Random Forest (engineered features) ===")

    X_train, y_train = split_X_y(train_feat)
    X_val, y_val = split_X_y(val_feat)
    X_test, y_test = split_X_y(test_feat)

    candidates = [
        {"n_estimators": 300, "max_depth": 10,   "min_samples_leaf": 1, "class_weight": None},
        {"n_estimators": 300, "max_depth": 10,   "min_samples_leaf": 5, "class_weight": None},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 5, "class_weight": None},
        {"n_estimators": 500, "max_depth": 10,   "min_samples_leaf": 1, "class_weight": None},
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
            "n_features": X_train.shape[1],
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
        by=["val_auprc", "val_auroc"], ascending=False
    )
    results_df.to_csv(RESULTS_DIR / "random_forest_engineered_results.csv", index=False)

    print("Best Random Forest config:", best_cfg)
    print("Validation:", best_metrics["val"])
    print("Test:", best_metrics["test"])

    return best_cfg, best_metrics, results_df


# =========================================================
# XGBoost
# =========================================================

def run_xgboost(train_feat, val_feat, test_feat):
    print("\n=== XGBoost (engineered features) ===")

    X_train, y_train = split_X_y(train_feat)
    X_val, y_val = split_X_y(val_feat)
    X_test, y_test = split_X_y(test_feat)

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
            "n_features": X_train.shape[1],
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
        by=["val_auprc", "val_auroc"], ascending=False
    )
    results_df.to_csv(RESULTS_DIR / "xgboost_engineered_results.csv", index=False)

    print("Best XGBoost config:", best_cfg)
    print("Validation:", best_metrics["val"])
    print("Test:", best_metrics["test"])

    return best_cfg, best_metrics, results_df


# =========================================================
# Main
# =========================================================

def main():
    train_df = load_df(FILES["a_ffill"])
    val_df = load_df(FILES["b_ffill"])
    test_df = load_df(FILES["c_ffill"])

    print("Engineering features...")
    train_feat = engineer_patient_features(train_df)
    val_feat = engineer_patient_features(val_df)
    test_feat = engineer_patient_features(test_df)

    print(f"Engineered feature count: {train_feat.shape[1] - 2}")  # exclude ID + label

    best_lr_cfg, best_lr_metrics, lr_results = run_logistic_regression(
        train_feat, val_feat, test_feat
    )
    best_rf_cfg, best_rf_metrics, rf_results = run_random_forest(
        train_feat, val_feat, test_feat
    )
    best_xgb_cfg, best_xgb_metrics, xgb_results = run_xgboost(
        train_feat, val_feat, test_feat
    )

    summary = {
        "feature_type": "engineered_q2_1_2",
        "description": "Static variables plus first/last/mean/min/max/std/delta/slope for each dynamic variable.",
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

    # Save engineered tables too, useful for inspection
    train_feat.to_parquet(RESULTS_DIR / "set_a_engineered.parquet", index=False)
    val_feat.to_parquet(RESULTS_DIR / "set_b_engineered.parquet", index=False)
    test_feat.to_parquet(RESULTS_DIR / "set_c_engineered.parquet", index=False)

    print(f"\nSaved all results to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()