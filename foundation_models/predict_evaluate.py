import pandas as pd
import numpy as np
import ollama
import json
import re
import sys
from pathlib import Path
from pydantic import BaseModel, ValidationError, field_validator
from typing import Literal
from sklearn.metrics import roc_auc_score, average_precision_score, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

# =============================================================================
# KNOBS - set these to control what runs
# =============================================================================
RUN_ZERO_SHOT = False
RUN_FEW_SHOT = False
RUN_EMBEDDINGS = True
MODEL = 'llama3.1:latest'
# =============================================================================

client = ollama.Client(host='http://127.0.0.1:11435')
BASE_DIR = Path.home() / "project1ml4h"

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

# --- Pydantic ---
class MortalityPrediction(BaseModel):
    prediction: Literal["alive", "dead"]

    @field_validator("prediction", mode="before")
    @classmethod
    def normalize(cls, v):
        if isinstance(v, str):
            return v.strip().lower()
        return v

# =============================================================================
# SHARED: Patient prompt builder
# =============================================================================
def create_patient_prompt(patient_df: pd.DataFrame):
    static_lines = []

    age = patient_df["Age"].dropna()
    if not age.empty:
        static_lines.append(f"  - Age: {age.iloc[0]:.0f} years")

    gender = patient_df["Gender"].dropna()
    if not gender.empty:
        static_lines.append(f"  - Gender: {'Male' if gender.iloc[0] == 1 else 'Female'}")

    height = patient_df["Height"].dropna()
    if not height.empty:
        static_lines.append(f"  - Height: {height.iloc[0]:.1f} cm")

    weight_static = patient_df["Weight_static"].dropna()
    if not weight_static.empty:
        static_lines.append(f"  - Weight (admission): {weight_static.iloc[0]:.1f} kg")

    mechvent = patient_df["MechVent"].dropna()
    if not mechvent.empty and mechvent.max() == 1:
        static_lines.append("  - Mechanical Ventilation: Yes")

    dynamic_lines = []
    for col in DYNAMIC_COLS:
        series = patient_df[col].dropna()
        if series.empty:
            continue
        dynamic_lines.append(f"  - {col}: min={series.min():.2f}, max={series.max():.2f}")

    static_block = "\n".join(static_lines) if static_lines else "  (no static data available)"
    dynamic_block = "\n".join(dynamic_lines) if dynamic_lines else "  (no dynamic data available)"

    return static_block, dynamic_block


def build_zero_shot_prompt(static_block: str, dynamic_block: str) -> str:
    return f"""You are a clinical decision support system. Based on the following ICU patient data, predict whether the patient survived or died during their hospital stay.

## Patient Information
{static_block}

## Time-Series Measurements (min and max over ICU stay)
{dynamic_block}

## Instructions
- Analyze the provided clinical data carefully.
- Predict whether this patient is "alive" or "dead" at hospital discharge.
- Respond ONLY with a valid JSON object in exactly this format:
  {{"prediction": "alive"}} or {{"prediction": "dead"}}
- Do not include any explanation or additional text."""


def build_few_shot_prompt(static_block: str, dynamic_block: str, examples: str) -> str:
    return f"""You are a clinical decision support system. Based on the following ICU patient data, predict whether the patient survived or died during their hospital stay.

## Examples
{examples}

## Now predict for this patient:

### Patient Information
{static_block}

### Time-Series Measurements (min and max over ICU stay)
{dynamic_block}

## Instructions
- Analyze the provided clinical data carefully.
- Predict whether this patient is "alive" or "dead" at hospital discharge.
- Respond ONLY with a valid JSON object in exactly this format:
  {{"prediction": "alive"}} or {{"prediction": "dead"}}
- Do not include any explanation or additional text."""


def build_few_shot_examples(train_df: pd.DataFrame, n_per_class: int = 3, seed: int = 42) -> str:
    rng = np.random.default_rng(seed)
    example_lines = []

    for label, outcome in [(1, "dead"), (0, "alive")]:
        patient_ids = train_df[train_df["label"] == label]["PatientID"].unique()
        sampled_ids = rng.choice(patient_ids, size=n_per_class, replace=False)

        for pid in sampled_ids:
            patient_df = train_df[train_df["PatientID"] == pid]
            static_block, dynamic_block = create_patient_prompt(patient_df)
            example_lines.append(f"""### Example (outcome: {outcome})
Patient Information:
{static_block}

Time-Series Measurements:
{dynamic_block}

Prediction: {{"prediction": "{outcome}"}}
""")

    return "\n".join(example_lines)


# =============================================================================
# PREDICTION PIPELINE
# =============================================================================
def query_llm(prompt: str, retries: int = 3) -> MortalityPrediction | None:
    for attempt in range(retries):
        try:
            response = client.chat(
                model=MODEL,
                messages=[{'role': 'user', 'content': prompt}],
            )
            raw = response['message']['content'].strip()
            json_match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if json_match:
                raw = json_match.group()
            data = json.loads(raw)
            return MortalityPrediction(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"  Attempt {attempt + 1}/{retries} failed: {e}", flush=True)
            if attempt == retries - 1:
                return None


def evaluate(labels: list, predictions: list) -> dict:
    auroc = roc_auc_score(labels, predictions)
    auprc = average_precision_score(labels, predictions)
    return {"AUROC": auroc, "AUPRC": auprc}


def run_evaluation(test_df: pd.DataFrame, mode: str, few_shot_examples: str = None) -> dict:
    patient_ids = test_df["PatientID"].unique()
    print(f"\n=== {mode.upper()} | Evaluating {len(patient_ids)} patients ===\n", flush=True)

    all_labels, all_preds = [], []
    failed = 0

    for i, pid in enumerate(tqdm(patient_ids, desc=f"{mode.upper()}", file=sys.stdout)):
        patient_df = test_df[test_df["PatientID"] == pid]
        label = int(patient_df["label"].iloc[0])
        static_block, dynamic_block = create_patient_prompt(patient_df)

        if mode == "zero_shot":
            prompt = build_zero_shot_prompt(static_block, dynamic_block)
        else:
            prompt = build_few_shot_prompt(static_block, dynamic_block, few_shot_examples)

        result = query_llm(prompt)

        if result is None:
            print(f"  [{i+1}/{len(patient_ids)}] PatientID {pid} — FAILED, counting as wrong", flush=True)
            pred = 1 - label
            failed += 1
        else:
            pred = 1 if result.prediction == "dead" else 0

        all_labels.append(label)
        all_preds.append(pred)

    metrics = evaluate(all_labels, all_preds)
    print(f"\n--- {mode.upper()} Results ---")
    print(f"  AUROC: {metrics['AUROC']:.4f}")
    print(f"  AUPRC: {metrics['AUPRC']:.4f}")
    print(f"  Failed predictions: {failed}/{len(patient_ids)}")
    return metrics


# =============================================================================
# EMBEDDING PIPELINE
# =============================================================================
def get_embedding(text: str, retries: int = 3) -> np.ndarray | None:
    for attempt in range(retries):
        try:
            response = client.embeddings(model=MODEL, prompt=text)
            return np.array(response['embedding'])
        except Exception as e:
            print(f"  Embedding attempt {attempt+1}/{retries} failed: {e}", flush=True)
            if attempt == retries - 1:
                return None


def get_all_embeddings(df: pd.DataFrame, split_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    patient_ids = df["PatientID"].unique()
    embeddings, labels, valid_ids = [], [], []

    for pid in tqdm(patient_ids, desc=f"Embedding {split_name}", file=sys.stdout):
        patient_df = df[df["PatientID"] == pid]
        label = int(patient_df["label"].iloc[0])
        static_block, dynamic_block = create_patient_prompt(patient_df)
        prompt = build_zero_shot_prompt(static_block, dynamic_block)
        emb = get_embedding(prompt)

        if emb is not None:
            embeddings.append(emb)
            labels.append(label)
            valid_ids.append(pid)

    return np.array(embeddings), np.array(labels), np.array(valid_ids)


def train_linear_probe(train_embeddings, train_labels, test_embeddings, test_labels) -> dict:
    print("\nTraining linear classifier...", flush=True)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_embeddings, train_labels)
    test_probs = clf.predict_proba(test_embeddings)[:, 1]
    auroc = roc_auc_score(test_labels, test_probs)
    auprc = average_precision_score(test_labels, test_probs)
    print(f"  Linear Probe — AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")
    return {"AUROC": auroc, "AUPRC": auprc}


def compute_clustering_metrics(embeddings, labels, n_clusters: int = 2) -> dict:
    print("\nComputing clustering metrics...", flush=True)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    silhouette = silhouette_score(embeddings, cluster_labels)
    ari = adjusted_rand_score(labels, cluster_labels)
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Adjusted Rand Index: {ari:.4f}")
    return {"silhouette": silhouette, "ari": ari, "cluster_labels": cluster_labels}


def run_embedding_pipeline(train_df, test_df):
    train_embeddings, train_labels, _ = get_all_embeddings(train_df, "train")
    test_embeddings, test_labels, _ = get_all_embeddings(test_df, "test")

    # Save everything needed for plotting later
    np.save("train_embeddings.npy", train_embeddings)
    np.save("train_labels.npy", train_labels)
    np.save("test_embeddings.npy", test_embeddings)
    np.save("test_labels.npy", test_labels)
    print("Embeddings saved.", flush=True)

    linear_metrics = train_linear_probe(train_embeddings, train_labels, test_embeddings, test_labels)
    clustering_results = compute_clustering_metrics(test_embeddings, test_labels)

    # Save cluster labels for plotting later
    np.save("cluster_labels.npy", clustering_results["cluster_labels"])
    print("Cluster labels saved.", flush=True)

    return linear_metrics, clustering_results


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("Loading data...", flush=True)
    train_df = pd.read_parquet(BASE_DIR / "data" / "processed" / "set_a.parquet")
    test_df = pd.read_parquet(BASE_DIR / "data" / "processed" / "set_c.parquet")

    zero_shot_metrics = None
    few_shot_metrics = None

    # these two are for direct prediction/evaluation
    if RUN_ZERO_SHOT:
        zero_shot_metrics = run_evaluation(test_df, mode="zero_shot")

    if RUN_FEW_SHOT:
        print("\nBuilding few-shot examples...", flush=True)
        few_shot_examples = build_few_shot_examples(train_df)
        few_shot_metrics = run_evaluation(test_df, mode="few_shot", few_shot_examples=few_shot_examples)

    if RUN_EMBEDDINGS:
        linear_metrics, clustering_results = run_embedding_pipeline(train_df, test_df)

    # Final summary
    print("\n=== FINAL SUMMARY ===")
    if zero_shot_metrics:
        print(f"Zero-shot    — AUROC: {zero_shot_metrics['AUROC']:.4f} | AUPRC: {zero_shot_metrics['AUPRC']:.4f}")
    if few_shot_metrics:
        print(f"Few-shot     — AUROC: {few_shot_metrics['AUROC']:.4f} | AUPRC: {few_shot_metrics['AUPRC']:.4f}")
    if RUN_EMBEDDINGS:
        print(f"Linear Probe — AUROC: {linear_metrics['AUROC']:.4f} | AUPRC: {linear_metrics['AUPRC']:.4f}")
        print(f"Silhouette   — {clustering_results['silhouette']:.4f}")
        print(f"ARI          — {clustering_results['ari']:.4f}")