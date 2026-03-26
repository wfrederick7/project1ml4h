from pathlib import Path
import pandas as pd
import numpy as np

# =========================================================
# Paths
# =========================================================

PROJECT_PATH = Path.home() / "project1ml4h" / "data"
INPUT_DIR = PROJECT_PATH / "processed"
OUTPUT_DIR = PROJECT_PATH / "processed_derived"
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

# Strongly right-skewed variables where log1p makes sense
LOG_COLS = [
    "ALP", "ALT", "AST", "Bilirubin", "BUN", "Creatinine", "Glucose",
    "Lactate", "Platelets", "TroponinI", "TroponinT", "Urine", "WBC"
]

# Numeric columns to scale for linear models
# (exclude identifier, target, and binary 0/1 columns)
SCALE_COLS = [c for c in FEATURE_COLS if c not in BINARY_COLS]

# =========================================================
# Small targeted outlier-cleaning table
# Values outside these bounds are set to NaN before imputation
# =========================================================

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

def compute_fill_defaults(df: pd.DataFrame) -> dict:
    """
    Defaults for leading missing values after forward fill.
    Binary/status vars get explicit defaults where appropriate.
    Other features get robust medians.
    """
    fill_defaults = {}

    # Binary columns
    fill_defaults["Gender"] = float(df["Gender"].mode(dropna=True).iloc[0]) if df["Gender"].notna().any() else 0.0
    fill_defaults["MechVent"] = 0.0

    # Everything else: median
    for col in FEATURE_COLS:
        if col in fill_defaults:
            continue

        non_missing = df[col].dropna()
        fill_defaults[col] = float(non_missing.median()) if len(non_missing) > 0 else 0.0

    return fill_defaults

def forward_impute_per_patient(df: pd.DataFrame, fill_defaults: dict) -> pd.DataFrame:
    """
    Within each patient:
    - forward fill all feature columns
    - fill remaining leading NaNs with defaults
    """
    df = sort_patient_time(df.copy())

    df[FEATURE_COLS] = df.groupby(ID_COL)[FEATURE_COLS].ffill()

    for col in FEATURE_COLS:
        df[col] = df[col].fillna(fill_defaults[col])

    # Keep binary columns as 0/1
    for col in BINARY_COLS:
        df[col] = df[col].round().clip(lower=0, upper=1)

    return df

def apply_log1p_transforms(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Apply log1p to selected right-skewed columns.
    Assumes values are non-negative after cleaning/imputation.
    """
    df = df.copy()

    for col in cols:
        if col in df.columns:
            # guard against accidental negatives
            df[col] = np.log1p(df[col].clip(lower=0))

    return df

def robust_scale(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Robust scaling: (x - median) / IQR
    """
    df = df.copy()
    params = {}

    for col in cols:
        s = df[col]
        med = s.median()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            iqr = 1.0

        df[col] = (s - med) / iqr
        params[col] = {"median": float(med), "iqr": float(iqr)}

    return df, params

def make_linear_ready(df_ffill: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Linear-model-ready pipeline:
    - keep binary vars as 0/1
    - log1p selected skewed vars
    - robust scale numeric non-binary feature columns
    """
    df = df_ffill.copy()

    # Ensure binary columns stay numeric 0/1
    for col in BINARY_COLS:
        df[col] = df[col].astype(float).round().clip(lower=0, upper=1)

    # Log-transform only selected skewed columns
    df = apply_log1p_transforms(df, LOG_COLS)

    # Robust scale non-binary feature columns
    df_scaled, scaler_params = robust_scale(df, SCALE_COLS)

    return df_scaled, scaler_params

def save_parquet(df: pd.DataFrame, filename: str) -> None:
    out_path = OUTPUT_DIR / filename
    df.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")

def save_scaler_params(params: dict, filename: str) -> None:
    out_path = OUTPUT_DIR / filename
    params_df = pd.DataFrame.from_dict(params, orient="index")
    params_df.index.name = "column"
    params_df.reset_index(inplace=True)
    params_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

# =========================================================
# Main processing per set
# =========================================================

def process_one_set(set_name: str) -> None:
    print(f"\n=== Processing set {set_name} ===")

    # Load
    df = load_set(set_name)
    df = sort_patient_time(df)

    # 1) Clean obvious implausible values to NaN
    df_cleaned = clean_targeted_outliers(df)

    # 2) Compute fill defaults on the cleaned data
    fill_defaults = compute_fill_defaults(df_cleaned)

    # 3) Forward-imputed version
    df_ffill = forward_impute_per_patient(df_cleaned, fill_defaults)
    save_parquet(df_ffill, f"set_{set_name}_ffill.parquet")

    # 4) Linear-model-ready version
    df_linear, scaler_params = make_linear_ready(df_ffill)
    save_parquet(df_linear, f"set_{set_name}_linear.parquet")
    save_scaler_params(scaler_params, f"set_{set_name}_linear_scaler_params.csv")

    print(f"Done with set {set_name}")

def process_all_sets():
    for set_name in ["a", "b", "c"]:
        process_one_set(set_name)

# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    process_all_sets()