"""
Produces one .parquet file per set (a, b, c).
Each parquet contains 49 rows per patient (hourly steps 0-48)
with 41 feature columns + PatientID + Time.

The 41 feature columns are:
  - 4 static  : Age, Gender, Height, Weight_static
                (admission values, broadcast to every row)
  - 37 dynamic: all time-series variables including Weight
                (Weight here tracks re-measurements during the stay)

WEIGHT DOES NOT SEEM TO BE STATIC

Timestamps are rounded UP (ceiling) to the nearest hour to
preserve temporal causality. No future information leaks back.
Missing values are left as NaN; no imputation is performed here.
"""

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = Path(os.path.expanduser("~/ml4h_data/p1"))
OUTPUT_DIR = Path(os.path.expanduser("~/ml4h_p1/data/processed"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SETS = ["a", "b", "c"]

# 4 static variables recorded at admission (t = 00:00), broadcast to all rows.
# ICUType is intentionally excluded per project instructions.
# Weight_static is the admission weight; "Weight" in DYNAMIC_VARS tracks
# re-measurements during the stay.
STATIC_VARS = ["Age", "Gender", "Height", "Weight_static"]

# 37 time-series variables from the Physionet 2012 dataset.
# Weight is included here because it can be re-measured over time.
DYNAMIC_VARS = [
    "ALP", "ALT", "AST", "Albumin", "BUN", "Bilirubin", "Cholesterol",
    "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT",
    "HR", "K", "Lactate", "MAP", "MechVent", "Mg", "NIDiasABP", "NIMAP",
    "NISysABP", "Na", "PaCO2", "PaO2", "Platelets", "RespRate", "SaO2",
    "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC", "Weight",
    "pH",
]

assert len(DYNAMIC_VARS) == 37, f"Expected 37 dynamic variables, got {len(DYNAMIC_VARS)}"
assert len(STATIC_VARS) == 4,   f"Expected 4 static variables, got {len(STATIC_VARS)}"
# Total feature columns: 37 dynamic + 4 static = 41
ALL_FEATURE_COLS = STATIC_VARS + DYNAMIC_VARS
assert len(ALL_FEATURE_COLS) == 41
assert len(set(ALL_FEATURE_COLS)) == 41, "Duplicate column names detected!"

# Parameters to skip entirely (not used as features)
SKIP_PARAMS = {"RecordID", "ICUType"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def time_str_to_minutes(time_str: str) -> int:
    """Convert 'HH:MM' to total minutes."""
    h, m = map(int, time_str.split(":"))
    return h * 60 + m


def ceil_hour(minutes: int) -> int:
    """
    Round minutes UP to the nearest whole hour.
    00:00 (0 min)  -> hour 0
    00:01 (1 min)  -> hour 1
    01:00 (60 min) -> hour 1
    01:01 (61 min) -> hour 2
    """
    return math.ceil(minutes / 60)


def parse_patient(filepath: Path) -> pd.DataFrame:
    """
    Parse one patient .txt file and return a DataFrame with 49 rows
    (one per hourly step 0-48) and 41 feature columns.

    Multiple measurements of the same variable within the same hour bin
    are resolved by keeping the last recorded value (last write wins).
    """
    raw = pd.read_csv(filepath)

    # ---- RecordID (patient ID) ----
    rec_rows = raw[raw["Parameter"] == "RecordID"]
    if rec_rows.empty:
        raise ValueError(f"No RecordID found in {filepath}")
    patient_id = int(float(rec_rows["Value"].iloc[0]))

    # ---- Static variables: read from t = 00:00 ----
    # Weight_static reads the "Weight" parameter at admission
    static_param_map = {
        "Age":           "Age",
        "Gender":        "Gender",
        "Height":        "Height",
        "Weight_static": "Weight",   # admission weight stored under new name
    }
    static_vals: dict[str, float] = {}
    for col_name, param_name in static_param_map.items():
        mask = (raw["Parameter"] == param_name) & (raw["Time"] == "00:00")
        rows = raw[mask]
        static_vals[col_name] = float(rows["Value"].iloc[0]) if not rows.empty else np.nan

    # ---- Dynamic variable grid (hour -> variable -> value) ----
    n_hours = 49  # hours 0 to 48
    grid: dict[str, list] = {var: [np.nan] * n_hours for var in DYNAMIC_VARS}

    # Parameters that are handled purely as static - skip in dynamic grid
    # Note: "Weight" is NOT skipped - it goes into the dynamic grid
    skip_as_dynamic = SKIP_PARAMS | {"Age", "Gender", "Height", "ICUType"}

    for _, row in raw.iterrows():
        param = row["Parameter"]

        if param in skip_as_dynamic:
            continue
        if param not in set(DYNAMIC_VARS):
            continue  # unknown / unused variable

        minutes = time_str_to_minutes(row["Time"])
        hour = ceil_hour(minutes)

        if hour > 48:
            continue  # discard measurements beyond the 48-hour window

        try:
            val = float(row["Value"])
        except (ValueError, TypeError):
            continue

        # Last measurement in the hour bin wins
        grid[param][hour] = val

    # ---- Assemble per-patient DataFrame (49 rows x 43 cols total) ----
    rows_out = []
    for h in range(n_hours):
        r: dict = {"PatientID": patient_id, "Time": h}
        for col_name in STATIC_VARS:
            r[col_name] = static_vals[col_name]
        for var in DYNAMIC_VARS:
            r[var] = grid[var][h]
        rows_out.append(r)

    df = pd.DataFrame(rows_out)
    # Enforce column order: PatientID, Time, 4 static, 37 dynamic
    df = df[["PatientID", "Time"] + ALL_FEATURE_COLS]
    return df


def load_labels(outcomes_path: Path) -> pd.Series:
    """
    Load In-hospital_death column from Outcomes-{a,b,c}.txt.
    Returns a Series indexed by RecordID (int).
    """
    outcomes = pd.read_csv(outcomes_path)
    outcomes["RecordID"] = outcomes["RecordID"].astype(int)
    outcomes = outcomes.set_index("RecordID")
    return outcomes["In-hospital_death"]


def process_set(set_name: str) -> None:
    """Process one complete set (a, b, or c) and save as .parquet."""
    set_dir = DATA_ROOT / f"set-{set_name}"
    outcomes_path = DATA_ROOT / f"Outcomes-{set_name}.txt"

    patient_files = sorted(set_dir.glob("*.txt"))
    if not patient_files:
        print(f"[{set_name}] No patient files found in {set_dir}. Skipping.")
        return

    print(f"\n[{set_name}] Processing {len(patient_files)} patients ...")

    dfs = []
    errors = []
    for fp in tqdm(patient_files, desc=f"Set {set_name}", unit="patient"):
        try:
            dfs.append(parse_patient(fp))
        except Exception as exc:
            errors.append((fp.name, str(exc)))

    if errors:
        print(f"  Skipped {len(errors)} files due to errors:")
        for fname, msg in errors[:10]:
            print(f"    {fname}: {msg}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more.")

    if not dfs:
        print(f"[{set_name}] No data produced. Aborting.")
        return

    set_df = pd.concat(dfs, ignore_index=True)

    # ---- Attach labels ----
    if outcomes_path.exists():
        labels = load_labels(outcomes_path)
        set_df["label"] = set_df["PatientID"].map(labels)
        n_labeled = set_df["label"].notna().sum() // 49
        print(f"  Labels loaded: {n_labeled} / {len(dfs)} patients have outcomes")
    else:
        print(f"  Warning: {outcomes_path} not found - no labels attached.")

    # ---- Sanity checks ----
    n_patients = set_df["PatientID"].nunique()
    expected_cols = 2 + 41 + (1 if "label" in set_df.columns else 0)
    assert set_df.shape[1] == expected_cols, \
        f"Expected {expected_cols} columns, got {set_df.shape[1]}"
    assert len(set_df) == n_patients * 49, \
        f"Expected {n_patients * 49} rows, got {len(set_df)}"

    # ---- Save ----
    out_path = OUTPUT_DIR / f"set_{set_name}.parquet"
    set_df.to_parquet(out_path, index=False)

    total_cells = len(set_df) * len(ALL_FEATURE_COLS)
    missing_cells = set_df[ALL_FEATURE_COLS].isna().sum().sum()
    print(f"  Patients : {n_patients}")
    print(f"  Rows     : {len(set_df)}  ({n_patients} x 49)")
    print(f"  Columns  : {set_df.shape[1]}  "
          f"(PatientID, Time, 41 features"
          f"{', label' if 'label' in set_df.columns else ''})")
    print(f"  Missing  : {missing_cells / total_cells:.1%} of feature values are NaN")
    print(f"  Saved ->  {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Physionet 2012 - preprocessing")
    print(f"  Data root : {DATA_ROOT}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Static vars  ({len(STATIC_VARS)}): {STATIC_VARS}")
    print(f"  Dynamic vars ({len(DYNAMIC_VARS)}): {DYNAMIC_VARS}")
    print(f"  Total feature columns: {len(ALL_FEATURE_COLS)}")

    for s in SETS:
        process_set(s)

    print("\nDone. Parquet files written to:", OUTPUT_DIR)