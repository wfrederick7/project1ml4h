# horn_preprocessing.py

from pathlib import Path
import json
import numpy as np
import pandas as pd

# =========================================================
# Paths
# =========================================================

PROJECT_PATH = Path.home() / "project1ml4h" / "data"
INPUT_DIR = PROJECT_PATH / "processed"
OUTPUT_DIR = PROJECT_PATH / "horn_processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SET_FILES = {
    "a": INPUT_DIR / "set_a.parquet",
    "b": INPUT_DIR / "set_b.parquet",
    "c": INPUT_DIR / "set_c.parquet",
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
BINARY_COLS = ["Gender", "MechVent"]

# Same skewed variables as in your existing preprocessing
LOG_COLS = [
    "ALP", "ALT", "AST", "Bilirubin", "BUN", "Creatinine", "Glucose",
    "Lactate", "Platelets", "TroponinI", "TroponinT", "Urine", "WBC"
]

# Same targeted outlier cleaning as before
VALID_RANGES = {
    "Height": (50, 250),
    "Temp": (30, 45),
    "Weight": (20, 300),
    "Weight_static": (20, 300),
    "pH": (6.5, 8.0),
}

# =========================================================
# Helpers
# =========================================================

def load_set(set_name: str) -> pd.DataFrame:
    path = SET_FILES[set_name]
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_parquet(path)


def sort_patient_time(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)


def clean_targeted_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set only clearly implausible values to NaN.
    """
    df = df.copy()

    for col, (low, high) in VALID_RANGES.items():
        if col not in df.columns:
            continue

        mask = pd.Series(False, index=df.index)

        if low is not None:
            mask |= df[col] < low
        if high is not None:
            mask |= df[col] > high

        df.loc[mask, col] = np.nan

    return df


def fit_value_scaler_on_training_observations(df_train: pd.DataFrame) -> dict:
    """
    Fit per-feature scaling parameters using observed values from training set only.

    Binary columns are kept as 0/1.
    LOG_COLS get log1p before robust scaling.
    Other numeric columns get robust scaling directly.
    """
    params = {}

    for col in FEATURE_COLS:
        s = df_train[col].dropna().astype(float)

        is_binary = col in BINARY_COLS
        use_log1p = col in LOG_COLS

        if is_binary:
            params[col] = {
                "is_binary": True,
                "use_log1p": False,
                "median": 0.0,
                "iqr": 1.0,
            }
            continue

        if use_log1p:
            s = np.log1p(s.clip(lower=0))

        if len(s) == 0:
            med = 0.0
            iqr = 1.0
        else:
            med = float(s.median())
            q1 = float(s.quantile(0.25))
            q3 = float(s.quantile(0.75))
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                iqr = 1.0

        params[col] = {
            "is_binary": False,
            "use_log1p": use_log1p,
            "median": med,
            "iqr": float(iqr),
        }

    return params


def transform_observed_value(value: float, col: str, scaler_params: dict) -> float:
    """
    Apply train-fitted transformation to one observed value.
    """
    if pd.isna(value):
        return np.nan

    p = scaler_params[col]

    if p["is_binary"]:
        return float(np.clip(np.round(float(value)), 0, 1))

    x = float(value)

    if p["use_log1p"]:
        x = np.log1p(max(x, 0.0))

    x = (x - p["median"]) / p["iqr"]
    return float(x)


def make_variable_ohe_columns(feature_cols: list[str]) -> list[str]:
    return [f"var_{c}" for c in feature_cols]


def tokenise_set(df: pd.DataFrame, scaler_params: dict) -> pd.DataFrame:
    """
    Convert regular hourly grid into Horn-style measurement tokens.

    Output columns:
      - PatientID
      - t_scaled
      - value_scaled
      - label
      - 41 one-hot variable columns

    Static variables are only emitted at time 0.
    Dynamic variables are emitted only when observed.
    """
    df = sort_patient_time(df.copy())

    ohe_cols = make_variable_ohe_columns(FEATURE_COLS)
    feature_to_ohe = {feat: f"var_{feat}" for feat in FEATURE_COLS}

    token_rows = []

    for patient_id, g in df.groupby(ID_COL, sort=False):
        patient_label = int(g[TARGET_COL].iloc[0]) if TARGET_COL in g.columns else np.nan

        for _, row in g.iterrows():
            hour = int(row[TIME_COL])
            t_scaled = float(hour / 48.0)

            for feat in FEATURE_COLS:
                raw_val = row[feat]

                # Only observed measurements become tokens
                if pd.isna(raw_val):
                    continue

                # Static variables should only appear once at admission
                if feat in STATIC_COLS and hour != 0:
                    continue

                scaled_val = transform_observed_value(raw_val, feat, scaler_params)

                token = {
                    ID_COL: int(patient_id),
                    "t_scaled": t_scaled,
                    "value_scaled": scaled_val,
                    TARGET_COL: patient_label,
                }

                for c in ohe_cols:
                    token[c] = 0
                token[feature_to_ohe[feat]] = 1

                token_rows.append(token)

    out = pd.DataFrame(token_rows)

    if len(out) == 0:
        raise ValueError("No tokens were created. Check input data.")

    # enforce column order
    out = out[[ID_COL, "t_scaled", "value_scaled", TARGET_COL] + ohe_cols]
    return out


def save_scaler_params(params: dict, filename: str) -> None:
    rows = []
    for col, p in params.items():
        rows.append({
            "column": col,
            "is_binary": p["is_binary"],
            "use_log1p": p["use_log1p"],
            "median": p["median"],
            "iqr": p["iqr"],
        })

    out_path = OUTPUT_DIR / filename
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def save_variable_map(filename_json: str, filename_csv: str) -> None:
    variable_map = {col: idx for idx, col in enumerate(FEATURE_COLS)}

    with open(OUTPUT_DIR / filename_json, "w") as f:
        json.dump(variable_map, f, indent=2)

    pd.DataFrame({
        "variable": FEATURE_COLS,
        "index": list(range(len(FEATURE_COLS))),
        "ohe_column": [f"var_{c}" for c in FEATURE_COLS]
    }).to_csv(OUTPUT_DIR / filename_csv, index=False)

    print(f"Saved: {OUTPUT_DIR / filename_json}")
    print(f"Saved: {OUTPUT_DIR / filename_csv}")


def save_token_counts(df_tokens: pd.DataFrame, filename: str) -> None:
    counts = (
        df_tokens.groupby(ID_COL)
        .size()
        .reset_index(name="n_tokens")
    )
    counts.to_csv(OUTPUT_DIR / filename, index=False)
    print(f"Saved: {OUTPUT_DIR / filename}")


def process_all_sets():
    # -----------------------------------------------------
    # 1) Load hourly-grid parquet files from Q1.1
    # -----------------------------------------------------
    train_df = load_set("a")
    val_df   = load_set("b")
    test_df  = load_set("c")

    # -----------------------------------------------------
    # 2) Apply same targeted outlier cleaning as Q1.3
    # -----------------------------------------------------
    train_df = clean_targeted_outliers(train_df)
    val_df   = clean_targeted_outliers(val_df)
    test_df  = clean_targeted_outliers(test_df)

    # -----------------------------------------------------
    # 3) Fit scaling parameters on set A observed values only
    # -----------------------------------------------------
    scaler_params = fit_value_scaler_on_training_observations(train_df)
    save_scaler_params(scaler_params, "horn_value_scaler_params.csv")
    save_variable_map("horn_variable_map.json", "horn_variable_map.csv")

    # -----------------------------------------------------
    # 4) Tokenise each split
    # -----------------------------------------------------
    print("\nTokenising set a ...")
    train_tokens = tokenise_set(train_df, scaler_params)
    train_tokens.to_parquet(OUTPUT_DIR / "set_a_horn.parquet", index=False)
    save_token_counts(train_tokens, "set_a_token_counts.csv")
    print(f"Saved: {OUTPUT_DIR / 'set_a_horn.parquet'}")

    print("\nTokenising set b ...")
    val_tokens = tokenise_set(val_df, scaler_params)
    val_tokens.to_parquet(OUTPUT_DIR / "set_b_horn.parquet", index=False)
    save_token_counts(val_tokens, "set_b_token_counts.csv")
    print(f"Saved: {OUTPUT_DIR / 'set_b_horn.parquet'}")

    print("\nTokenising set c ...")
    test_tokens = tokenise_set(test_df, scaler_params)
    test_tokens.to_parquet(OUTPUT_DIR / "set_c_horn.parquet", index=False)
    save_token_counts(test_tokens, "set_c_token_counts.csv")
    print(f"Saved: {OUTPUT_DIR / 'set_c_horn.parquet'}")

    # -----------------------------------------------------
    # 5) Quick summary
    # -----------------------------------------------------
    for name, df_tok in [("a", train_tokens), ("b", val_tokens), ("c", test_tokens)]:
        print(f"\nSet {name}:")
        print(f"  rows/tokens       : {len(df_tok)}")
        print(f"  patients          : {df_tok[ID_COL].nunique()}")
        print(f"  avg tokens/patient: {len(df_tok) / df_tok[ID_COL].nunique():.2f}")
        print(f"  t_scaled range    : [{df_tok['t_scaled'].min():.3f}, {df_tok['t_scaled'].max():.3f}]")

    print(f"\nDone. Horn-style files written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    process_all_sets()