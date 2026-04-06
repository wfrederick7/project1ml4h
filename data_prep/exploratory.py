from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

# Use a non-interactive backend so plots can be rendered in sbatch jobs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ID_COLS = ["PatientID", "Time"]
BINARY_COLS = ["Gender", "MechVent", "label"]
STATIC_CONT_COLS = ["Age", "Height", "Weight_static"]
DYNAMIC_COLS = [
    "ALP", "ALT", "AST", "Albumin", "BUN", "Bilirubin", "Cholesterol",
    "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT",
    "HR", "K", "Lactate", "MAP", "Mg", "NIDiasABP", "NIMAP", "NISysABP",
    "Na", "PaCO2", "PaO2", "Platelets", "RespRate", "SaO2", "SysABP",
    "Temp", "TroponinI", "TroponinT", "Urine", "WBC", "Weight", "pH"
]


def get_project_paths():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "processed"
    fig_dir = repo_root / "figures" / "exploratory"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return repo_root, data_dir, fig_dir


def load_split(data_dir, split_name):
    split_path = data_dir / f"{split_name}.parquet"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing data file: {split_path}")
    return pd.read_parquet(split_path)


def save_figure(fig, output_path):
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_summary_table(df):
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "non_null": s.notna().sum(),
            "missing": s.isna().sum(),
            "missing_pct": 100 * s.isna().mean(),
            "n_unique": s.nunique(dropna=True),
            "min": s.min() if pd.api.types.is_numeric_dtype(s) else None,
            "q25": s.quantile(0.25) if pd.api.types.is_numeric_dtype(s) else None,
            "median": s.median() if pd.api.types.is_numeric_dtype(s) else None,
            "q75": s.quantile(0.75) if pd.api.types.is_numeric_dtype(s) else None,
            "max": s.max() if pd.api.types.is_numeric_dtype(s) else None,
            "mean": s.mean() if pd.api.types.is_numeric_dtype(s) else None,
            "std": s.std() if pd.api.types.is_numeric_dtype(s) else None,
        })
    summary = pd.DataFrame(rows).sort_values("missing_pct", ascending=False)
    return summary


def plot_missingness(summary_df, output_dir, split_name):
    temp = summary_df.sort_values("missing_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(temp) * 0.25)))
    ax.barh(temp["column"], temp["missing_pct"])
    ax.set_xlabel("Missingness (%)")
    ax.set_ylabel("Column")
    ax.set_title("Missingness by column")
    fig.tight_layout()
    save_figure(fig, output_dir / f"{split_name}_missingness.png")


def plot_binary_column(df, col, output_dir, split_name):
    counts = df[col].value_counts(dropna=False).sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.set_title(f"Value counts for {col}")
    fig.tight_layout()
    save_figure(fig, output_dir / f"{split_name}_binary_{col}.png")


def plot_continuous_distribution(df, col, output_dir, split_name, log_scale=False):
    s = df[col].dropna()

    if len(s) == 0:
        print(f"{col}: all values are missing")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(s, bins=40)
    axes[0].set_title(f"{col} histogram")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Count")

    axes[1].boxplot(s, vert=True)
    axes[1].set_title(f"{col} boxplot")
    axes[1].set_ylabel(col)

    if log_scale:
        positive_s = s[s > 0]
        if len(positive_s) > 0:
            axes[0].cla()
            axes[0].hist(np.log10(positive_s), bins=40)
            axes[0].set_title(f"{col} histogram (log10, positive values only)")
            axes[0].set_xlabel(f"log10({col})")
            axes[0].set_ylabel("Count")

    fig.tight_layout()
    suffix = "_log" if log_scale else ""
    save_figure(fig, output_dir / f"{split_name}_distribution_{col}{suffix}.png")


def plot_time_profile(df, col, output_dir, split_name):
    temp = df.groupby("Time")[col].agg(
        mean="mean",
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        count="count"
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(temp["Time"], temp["mean"], label="Mean")
    axes[0].plot(temp["Time"], temp["median"], label="Median")
    axes[0].fill_between(temp["Time"], temp["q25"], temp["q75"], alpha=0.3, label="IQR")
    axes[0].set_title(f"{col} over time")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel(col)
    axes[0].legend()

    axes[1].bar(temp["Time"], temp["count"])
    axes[1].set_title(f"{col} observed values by hour")
    axes[1].set_xlabel("Hour")
    axes[1].set_ylabel("Non-missing count")

    fig.tight_layout()
    save_figure(fig, output_dir / f"{split_name}_time_profile_{col}.png")


def plot_random_patient_trajectories(df, col, output_dir, split_name, n_patients=10, seed=42):
    valid_ids = df.loc[df[col].notna(), "PatientID"].unique()
    if len(valid_ids) == 0:
        print(f"{col}: no observed values")
        return

    rng = np.random.default_rng(seed)
    chosen = rng.choice(valid_ids, size=min(n_patients, len(valid_ids)), replace=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    for pid in chosen:
        sub = df[df["PatientID"] == pid]
        ax.plot(sub["Time"], sub[col], alpha=0.7)

    ax.set_xlabel("Hour")
    ax.set_ylabel(col)
    ax.set_title(f"{col}: random patient trajectories")
    fig.tight_layout()
    save_figure(fig, output_dir / f"{split_name}_trajectories_{col}.png")


def exploratory_analysis(df, output_dir, split_name):
    print(f"Shape ({split_name}):", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    summary = make_summary_table(df)
    print("\nSummary table:")
    print(summary)

    summary_path = output_dir / f"{split_name}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary table to: {summary_path}")

    return summary


def main():
    _, data_dir, fig_dir = get_project_paths()
    print(f"Saving figures under: {fig_dir}")

    df_a = load_split(data_dir, "set_a")

    summary_a = exploratory_analysis(df_a, fig_dir, split_name="set_a")

    plot_missingness(summary_a, fig_dir, split_name="set_a")

    for col in ["Gender", "MechVent", "label"]:
        plot_binary_column(df_a, col, fig_dir, split_name="set_a")

    for col in ["Age", "Height", "Weight_static"]:
        plot_continuous_distribution(df_a, col, fig_dir, split_name="set_a")

    for col in ["HR", "MAP", "Temp", "RespRate", "Creatinine"]:
        plot_continuous_distribution(df_a, col, fig_dir, split_name="set_a")
        plot_time_profile(df_a, col, fig_dir, split_name="set_a")
        plot_random_patient_trajectories(df_a, col, fig_dir, split_name="set_a", n_patients=8)

    for col in ["ALT", "AST", "Bilirubin", "Urine", "TroponinI", "TroponinT", "WBC"]:
        plot_continuous_distribution(df_a, col, fig_dir, split_name="set_a", log_scale=True)
        plot_time_profile(df_a, col, fig_dir, split_name="set_a")

    print("Exploratory figure generation complete.")


if __name__ == "__main__":
    main()