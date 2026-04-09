import pandas as pd
import numpy as np
import ollama
import json
import re
import sys
from pathlib import Path
from pydantic import BaseModel, ValidationError, field_validator
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    silhouette_score,
    precision_score,
    recall_score,
    confusion_matrix,
    adjusted_rand_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from tqdm import tqdm
from argparse import ArgumentParser

# =============================================================================
# KNOBS - set these to control what runs
# =============================================================================
RUN_ZERO_SHOT = False
RUN_FEW_SHOT = False
RUN_EMBEDDINGS = True
MODEL = "llama3.1:latest"

# Prediction mode:
#   "binary" -> {"prediction": "alive"} / {"prediction": "dead"}   [default]
#   "scale"  -> {"prediction": 1} ... {"prediction": 10}
PREDICTION_MODE = "binary"

# Only used when PREDICTION_MODE == "scale"
DEAD_THRESHOLD = 6
# =============================================================================

client = ollama.Client(host="http://127.0.0.1:11435")
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


# =============================================================================
# OUTPUT SCHEMAS
# =============================================================================
class MortalityPredictionBinary(BaseModel):
    prediction: str

    @field_validator("prediction", mode="before")
    @classmethod
    def normalize(cls, v):
        if isinstance(v, str):
            v = v.strip().lower()
        return v

    @field_validator("prediction")
    @classmethod
    def validate_label(cls, v):
        if v not in {"alive", "dead"}:
            raise ValueError("prediction must be 'alive' or 'dead'")
        return v


class MortalityPredictionScale(BaseModel):
    prediction: int

    @field_validator("prediction", mode="before")
    @classmethod
    def normalize(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if v.isdigit():
                v = int(v)
        return v

    @field_validator("prediction")
    @classmethod
    def validate_range(cls, v):
        if not isinstance(v, int) or not (1 <= v <= 10):
            raise ValueError("prediction must be an integer between 1 and 10")
        return v


# =============================================================================
# SHARED: PATIENT PROMPT BUILDER
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


def build_zero_shot_prompt(
    static_block: str,
    dynamic_block: str,
    prediction_mode: str = PREDICTION_MODE,
) -> str:
    if prediction_mode == "binary":
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
    elif prediction_mode == "scale":
        return f"""You are a clinical decision support system. Based on the following ICU patient data, estimate the patient's risk of in-hospital mortality.

## Patient Information
{static_block}

## Time-Series Measurements (min and max over ICU stay)
{dynamic_block}

## Instructions
- Analyze the provided clinical data carefully.
- Output an integer from 1 to 10:
  - 1 = very confident the patient is alive at hospital discharge
  - 10 = very confident the patient is dead at hospital discharge
  - 5 or 6 = uncertain / borderline
- Respond ONLY with a valid JSON object in exactly this format:
  {{"prediction": 7}}
- Do not include any explanation or additional text."""
    else:
        raise ValueError(f"Unsupported prediction_mode: {prediction_mode}")


def build_few_shot_prompt(
    static_block: str,
    dynamic_block: str,
    examples: str,
    prediction_mode: str = PREDICTION_MODE,
) -> str:
    if prediction_mode == "binary":
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
    elif prediction_mode == "scale":
        return f"""You are a clinical decision support system. Based on the following ICU patient data, estimate the patient's risk of in-hospital mortality.

## Examples
{examples}

## Now predict for this patient:

### Patient Information
{static_block}

### Time-Series Measurements (min and max over ICU stay)
{dynamic_block}

## Instructions
- Analyze the provided clinical data carefully.
- Output an integer from 1 to 10:
  - 1 = very confident the patient is alive at hospital discharge
  - 10 = very confident the patient is dead at hospital discharge
  - 5 or 6 = uncertain / borderline
- Respond ONLY with a valid JSON object in exactly this format:
  {{"prediction": 7}}
- Do not include any explanation or additional text."""
    else:
        raise ValueError(f"Unsupported prediction_mode: {prediction_mode}")


def build_few_shot_examples(
    train_df: pd.DataFrame,
    n_per_class: int = 3,
    seed: int = 42,
    prediction_mode: str = PREDICTION_MODE,
) -> str:
    rng = np.random.default_rng(seed)
    example_lines = []

    if prediction_mode == "binary":
        label_to_output = {1: "dead", 0: "alive"}
    elif prediction_mode == "scale":
        label_to_output = {1: 9, 0: 2}
    else:
        raise ValueError(f"Unsupported prediction_mode: {prediction_mode}")

    for label in [1, 0]:
        patient_ids = train_df[train_df["label"] == label]["PatientID"].unique()

        if len(patient_ids) < n_per_class:
            sampled_ids = patient_ids
        else:
            sampled_ids = rng.choice(patient_ids, size=n_per_class, replace=False)

        for pid in sampled_ids:
            patient_df = train_df[train_df["PatientID"] == pid]
            static_block, dynamic_block = create_patient_prompt(patient_df)
            target = label_to_output[label]

            if prediction_mode == "binary":
                pred_str = f'{{"prediction": "{target}"}}'
            else:
                pred_str = f'{{"prediction": {target}}}'

            example_lines.append(
                f"""### Example
Patient Information:
{static_block}

Time-Series Measurements:
{dynamic_block}

Prediction: {pred_str}
"""
            )

    return "\n".join(example_lines)


# =============================================================================
# PREDICTION PIPELINE
# =============================================================================
def query_llm(
    prompt: str,
    prediction_mode: str = PREDICTION_MODE,
    retries: int = 3,
):
    schema = MortalityPredictionBinary if prediction_mode == "binary" else MortalityPredictionScale

    for attempt in range(retries):
        try:
            response = client.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response["message"]["content"].strip()

            json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if json_match:
                raw = json_match.group()

            data = json.loads(raw)
            return schema(**data)

        except (json.JSONDecodeError, ValidationError) as e:
            print(f"  Attempt {attempt + 1}/{retries} failed: {e}", flush=True)
            if attempt == retries - 1:
                return None
        except Exception as e:
            print(f"  Attempt {attempt + 1}/{retries} failed with unexpected error: {e}", flush=True)
            if attempt == retries - 1:
                return None


def evaluate_binary(labels: list[int], pred_labels: list[int]) -> dict:
    labels = np.array(labels, dtype=int)
    pred_labels = np.array(pred_labels, dtype=int)

    auroc = roc_auc_score(labels, pred_labels)
    auprc = average_precision_score(labels, pred_labels)
    precision = precision_score(labels, pred_labels, zero_division=0)
    recall = recall_score(labels, pred_labels, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(labels, pred_labels, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    predicted_alive = int((pred_labels == 0).sum())
    predicted_dead = int((pred_labels == 1).sum())

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "PredictedAlive": predicted_alive,
        "PredictedDead": predicted_dead,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def evaluate_scale(labels: list[int], scores_1to10: list[int], threshold: int = DEAD_THRESHOLD) -> dict:
    labels = np.array(labels, dtype=int)
    scores_1to10 = np.array(scores_1to10, dtype=int)

    risk_scores = (scores_1to10 - 1) / 9.0
    pred_labels = (scores_1to10 >= threshold).astype(int)

    auroc = roc_auc_score(labels, risk_scores)
    auprc = average_precision_score(labels, risk_scores)
    precision = precision_score(labels, pred_labels, zero_division=0)
    recall = recall_score(labels, pred_labels, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(labels, pred_labels, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    predicted_alive = int((pred_labels == 0).sum())
    predicted_dead = int((pred_labels == 1).sum())

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "PredictedAlive": predicted_alive,
        "PredictedDead": predicted_dead,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def run_evaluation(
    test_df: pd.DataFrame,
    mode: str,
    few_shot_examples: str = None,
    prediction_mode: str = PREDICTION_MODE,
) -> dict:
    patient_ids = test_df["PatientID"].unique()
    print(
        f"\n=== {mode.upper()} | prediction_mode={prediction_mode} | Evaluating {len(patient_ids)} patients ===\n",
        flush=True,
    )

    all_labels = []
    all_outputs = []
    failed = 0

    for i, pid in enumerate(tqdm(patient_ids, desc=f"{mode.upper()}", file=sys.stdout)):
        patient_df = test_df[test_df["PatientID"] == pid]
        label = int(patient_df["label"].iloc[0])
        static_block, dynamic_block = create_patient_prompt(patient_df)

        if mode == "zero_shot":
            prompt = build_zero_shot_prompt(static_block, dynamic_block, prediction_mode=prediction_mode)
        elif mode == "few_shot":
            prompt = build_few_shot_prompt(
                static_block,
                dynamic_block,
                few_shot_examples,
                prediction_mode=prediction_mode,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        result = query_llm(prompt, prediction_mode=prediction_mode)

        if result is None:
            print(f"  [{i + 1}/{len(patient_ids)}] PatientID {pid} — FAILED, counting as wrong", flush=True)
            if prediction_mode == "binary":
                output = 1 - label
            else:
                output = 1 if label == 1 else 10
            failed += 1
        else:
            if prediction_mode == "binary":
                output = 1 if result.prediction == "dead" else 0
            else:
                output = int(result.prediction)

        all_labels.append(label)
        all_outputs.append(output)

    if prediction_mode == "binary":
        metrics = evaluate_binary(all_labels, all_outputs)
    else:
        metrics = evaluate_scale(all_labels, all_outputs, threshold=DEAD_THRESHOLD)

    metrics["Failed"] = failed
    metrics["N"] = len(patient_ids)

    print(f"\n--- {mode.upper()} Results ---")
    print(f"  AUROC:              {metrics['AUROC']:.4f}")
    print(f"  AUPRC:              {metrics['AUPRC']:.4f}")
    print(f"  Precision:          {metrics['Precision']:.4f}")
    print(f"  Recall:             {metrics['Recall']:.4f}")
    print(f"  Specificity:        {metrics['Specificity']:.4f}")
    print(f"  Predicted alive:    {metrics['PredictedAlive']}")
    print(f"  Predicted dead:     {metrics['PredictedDead']}")
    print(f"  TP:                 {metrics['TP']}")
    print(f"  FP:                 {metrics['FP']}")
    print(f"  TN:                 {metrics['TN']}")
    print(f"  FN:                 {metrics['FN']}")
    print(f"  Failed predictions: {failed}/{len(patient_ids)}")

    if prediction_mode == "scale":
        print(f"  Dead threshold:     {DEAD_THRESHOLD}")

    return metrics


# =============================================================================
# EMBEDDING PIPELINE
# =============================================================================
def get_embedding(text: str, retries: int = 3) -> np.ndarray | None:
    for attempt in range(retries):
        try:
            response = client.embeddings(model=MODEL, prompt=text)
            return np.array(response["embedding"])
        except Exception as e:
            print(f"  Embedding attempt {attempt + 1}/{retries} failed: {e}", flush=True)
            if attempt == retries - 1:
                return None


def get_all_embeddings(df: pd.DataFrame, split_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    patient_ids = df["PatientID"].unique()
    embeddings, labels, valid_ids = [], [], []

    for pid in tqdm(patient_ids, desc=f"Embedding {split_name}", file=sys.stdout):
        patient_df = df[df["PatientID"] == pid]
        label = int(patient_df["label"].iloc[0])
        static_block, dynamic_block = create_patient_prompt(patient_df)
        prompt = build_zero_shot_prompt(static_block, dynamic_block, prediction_mode="binary")
        emb = get_embedding(prompt)

        if emb is not None:
            embeddings.append(emb)
            labels.append(label)
            valid_ids.append(pid)

    return np.array(embeddings), np.array(labels), np.array(valid_ids)


def train_linear_probe(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    print("\nTraining linear classifier...", flush=True)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_embeddings, train_labels)

    test_probs = clf.predict_proba(test_embeddings)[:, 1]
    pred_labels = (test_probs >= threshold).astype(int)

    auroc = roc_auc_score(test_labels, test_probs)
    auprc = average_precision_score(test_labels, test_probs)
    precision = precision_score(test_labels, pred_labels, zero_division=0)
    recall = recall_score(test_labels, pred_labels, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(test_labels, pred_labels, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "PredictedAlive": int((pred_labels == 0).sum()),
        "PredictedDead": int((pred_labels == 1).sum()),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }

    print("\n--- LINEAR PROBE Results ---")
    print(f"  AUROC:            {metrics['AUROC']:.4f}")
    print(f"  AUPRC:            {metrics['AUPRC']:.4f}")
    print(f"  Precision:        {metrics['Precision']:.4f}")
    print(f"  Recall:           {metrics['Recall']:.4f}")
    print(f"  Specificity:      {metrics['Specificity']:.4f}")
    print(f"  Predicted alive:  {metrics['PredictedAlive']}")
    print(f"  Predicted dead:   {metrics['PredictedDead']}")
    print(f"  TP:               {metrics['TP']}")
    print(f"  FP:               {metrics['FP']}")
    print(f"  TN:               {metrics['TN']}")
    print(f"  FN:               {metrics['FN']}")

    return metrics


def compute_clustering_metrics(embeddings: np.ndarray, labels: np.ndarray, n_clusters: int = 2) -> dict:
    print("\nComputing clustering metrics...", flush=True)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    silhouette = silhouette_score(embeddings, cluster_labels)
    ari = adjusted_rand_score(labels, cluster_labels)

    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Adjusted Rand Index: {ari:.4f}")

    return {
        "silhouette": silhouette,
        "ari": ari,
        "cluster_labels": cluster_labels,
    }


def run_embedding_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_embeddings, train_labels, _ = get_all_embeddings(train_df, "train")
    test_embeddings, test_labels, _ = get_all_embeddings(test_df, "test")

    np.save("train_embeddings.npy", train_embeddings)
    np.save("train_labels.npy", train_labels)
    np.save("test_embeddings.npy", test_embeddings)
    np.save("test_labels.npy", test_labels)
    print("Embeddings saved.", flush=True)

    linear_metrics = train_linear_probe(train_embeddings, train_labels, test_embeddings, test_labels)
    clustering_results = compute_clustering_metrics(test_embeddings, test_labels)

    np.save("cluster_labels.npy", clustering_results["cluster_labels"])
    print("Cluster labels saved.", flush=True)

    return linear_metrics, clustering_results


# =============================================================================
# MAIN
# =============================================================================
def main(
    run_zero_shot: bool,
    run_few_shot: bool,
    run_embeddings: bool,
    prediction_mode: str = PREDICTION_MODE,
):
    print("Loading data...", flush=True)
    print(f"Prediction mode: {prediction_mode}", flush=True)

    train_df = pd.read_parquet(BASE_DIR / "data" / "processed" / "set_a.parquet")
    test_df = pd.read_parquet(BASE_DIR / "data" / "processed" / "set_c.parquet")

    zero_shot_metrics = None
    few_shot_metrics = None
    linear_metrics = None
    clustering_results = None

    if run_zero_shot:
        zero_shot_metrics = run_evaluation(
            test_df,
            mode="zero_shot",
            prediction_mode=prediction_mode,
        )

    if run_few_shot:
        print("\nBuilding few-shot examples...", flush=True)
        few_shot_examples = build_few_shot_examples(
            train_df,
            prediction_mode=prediction_mode,
        )
        few_shot_metrics = run_evaluation(
            test_df,
            mode="few_shot",
            few_shot_examples=few_shot_examples,
            prediction_mode=prediction_mode,
        )

    if run_embeddings:
        linear_metrics, clustering_results = run_embedding_pipeline(train_df, test_df)

    print("\n=== FINAL SUMMARY ===")

    if zero_shot_metrics is not None:
        print(
            f"Zero-shot    — "
            f"AUROC: {zero_shot_metrics['AUROC']:.4f} | "
            f"AUPRC: {zero_shot_metrics['AUPRC']:.4f} | "
            f"Precision: {zero_shot_metrics['Precision']:.4f} | "
            f"Recall: {zero_shot_metrics['Recall']:.4f} | "
            f"Specificity: {zero_shot_metrics['Specificity']:.4f} | "
            f"PredAlive: {zero_shot_metrics['PredictedAlive']} | "
            f"PredDead: {zero_shot_metrics['PredictedDead']} | "
            f"TP: {zero_shot_metrics['TP']} | "
            f"FP: {zero_shot_metrics['FP']} | "
            f"TN: {zero_shot_metrics['TN']} | "
            f"FN: {zero_shot_metrics['FN']} | "
            f"Failed: {zero_shot_metrics['Failed']}"
        )

    if few_shot_metrics is not None:
        print(
            f"Few-shot     — "
            f"AUROC: {few_shot_metrics['AUROC']:.4f} | "
            f"AUPRC: {few_shot_metrics['AUPRC']:.4f} | "
            f"Precision: {few_shot_metrics['Precision']:.4f} | "
            f"Recall: {few_shot_metrics['Recall']:.4f} | "
            f"Specificity: {few_shot_metrics['Specificity']:.4f} | "
            f"PredAlive: {few_shot_metrics['PredictedAlive']} | "
            f"PredDead: {few_shot_metrics['PredictedDead']} | "
            f"TP: {few_shot_metrics['TP']} | "
            f"FP: {few_shot_metrics['FP']} | "
            f"TN: {few_shot_metrics['TN']} | "
            f"FN: {few_shot_metrics['FN']} | "
            f"Failed: {few_shot_metrics['Failed']}"
        )

    if run_embeddings and linear_metrics is not None and clustering_results is not None:
        print(
            f"Linear Probe — "
            f"AUROC: {linear_metrics['AUROC']:.4f} | "
            f"AUPRC: {linear_metrics['AUPRC']:.4f} | "
            f"Precision: {linear_metrics['Precision']:.4f} | "
            f"Recall: {linear_metrics['Recall']:.4f} | "
            f"Specificity: {linear_metrics['Specificity']:.4f} | "
            f"PredAlive: {linear_metrics['PredictedAlive']} | "
            f"PredDead: {linear_metrics['PredictedDead']} | "
            f"TP: {linear_metrics['TP']} | "
            f"FP: {linear_metrics['FP']} | "
            f"TN: {linear_metrics['TN']} | "
            f"FN: {linear_metrics['FN']}"
        )
        print(f"Silhouette   — {clustering_results['silhouette']:.4f}")
        print(f"ARI          — {clustering_results['ari']:.4f}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run-zero-shot", action="store_true")
    parser.add_argument("--run-few-shot", action="store_true")
    parser.add_argument("--run-embeddings", action="store_true")
    parser.add_argument(
        "--prediction-mode",
        choices=["binary", "scale"],
        default="binary",
        help="binary = alive/dead output, scale = integer 1..10 output",
    )
    args = parser.parse_args()

    main(
        run_zero_shot=args.run_zero_shot,
        run_few_shot=args.run_few_shot,
        run_embeddings=args.run_embeddings,
        prediction_mode=args.prediction_mode,
    )