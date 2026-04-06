from multiprocessing import context
import sys
from pathlib import Path
from chronos import ChronosPipeline
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from chronos import ChronosPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# =============================================================================
# CONFIG
# =============================================================================
CHRONOS_MODEL = "amazon/chronos-t5-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3

BASE_DIR = Path.home() / "project1ml4h"
TRAIN_PATH = BASE_DIR / "data" / "processed_derived" / "set_a_ffill.parquet"
TEST_PATH = BASE_DIR / "data" / "processed_derived" / "set_c_ffill.parquet"
SCALER_PATH = BASE_DIR / "data" / "processed_derived" / "set_a_linear_scaler_params.csv"

DYNAMIC_COLS = [
    "ALP", "ALT", "AST", "Albumin", "BUN", "Bilirubin", "Cholesterol",
    "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT",
    "HR", "K", "Lactate", "MAP", "Mg", "NIDiasABP", "NIMAP", "NISysABP",
    "Na", "PaCO2", "PaO2", "Platelets", "RespRate", "SaO2", "SysABP",
    "Temp", "TroponinI", "TroponinT", "Urine", "WBC", "Weight", "pH"
]
STATIC_COLS = ["Age", "Height", "Weight_static", "Gender", "MechVent"]

# =============================================================================
# STEP 1: Load scaler params and normalize static features
# =============================================================================
def load_scaler(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def normalize_static(df: pd.DataFrame, scaler: pd.DataFrame) -> np.ndarray:
    result = []
    for col in STATIC_COLS:
        vals = df[col].values.astype(np.float32)
        if col in scaler.index:
            median = scaler.loc[col, "median"]
            iqr  = scaler.loc[col, "iqr"]
            vals = (vals - median) / (iqr + 1e-8)
        result.append(vals)
    return np.stack(result, axis=1)  # (n_patients, n_static)


# =============================================================================
# STEP 2: Build patient-level data
# =============================================================================
def get_patient_static(patient_df: pd.DataFrame) -> dict:
    """Extract one static value per patient."""
    return patient_df[STATIC_COLS].iloc[0].to_dict()


def build_dataset(df: pd.DataFrame, split_name: str) -> tuple:
    """
    Returns:
        dynamic_series: dict mapping col -> list of 1D np.arrays (one per patient)
        static_matrix: np.ndarray of shape (n_patients, n_static)
        labels: np.ndarray of shape (n_patients,)
        patient_ids: np.ndarray
    """
    patient_ids = df["PatientID"].unique()
    print(f"\nBuilding {split_name} dataset ({len(patient_ids)} patients)...", flush=True)

    # Initialize storage
    dynamic_series = {col: [] for col in DYNAMIC_COLS}
    static_rows = []
    labels = []

    for pid in tqdm(patient_ids, desc=f"Parsing {split_name}", file=sys.stdout):
        patient_df = df[df["PatientID"] == pid].sort_values("Time")
        label = int(patient_df["label"].iloc[0])

        # Dynamic: extract time series per variable, validate no NaNs
        for col in DYNAMIC_COLS:
            series = patient_df[col].values.astype(np.float32)
            if np.any(np.isnan(series)):
                raise ValueError(
                    f"NaN found in {col} for PatientID {pid} after ffill. "
                    f"This should not happen."
                )
            dynamic_series[col].append(series)

        # Static
        static_rows.append(get_patient_static(patient_df))
        labels.append(label)

    static_df = pd.DataFrame(static_rows, columns=STATIC_COLS)
    return dynamic_series, static_df, np.array(labels), patient_ids


# =============================================================================
# STEP 3: Generate Chronos embeddings
# =============================================================================
def get_chronos_embeddings(
    pipeline: ChronosPipeline,
    dynamic_series: dict,
    split_name: str
) -> np.ndarray:
    """
    For each dynamic variable, run all patients through the Chronos encoder.
    Mean-pool over time dimension to get one vector per (patient, variable).
    Returns: np.ndarray of shape (n_patients, n_vars, hidden_dim)
    """
    n_patients = len(dynamic_series[DYNAMIC_COLS[0]])
    all_var_embeddings = []

    for col in tqdm(DYNAMIC_COLS, desc=f"Chronos embeddings {split_name}", file=sys.stdout):
        series_list = dynamic_series[col]  # list of (T,) arrays

        # Process in batches
        var_embeddings = []
        for i in range(0, n_patients, BATCH_SIZE):
            batch = series_list[i:i + BATCH_SIZE]
            # Pad to same length within batch
            max_len = max(len(s) for s in batch)
            padded = np.zeros((len(batch), max_len), dtype=np.float32)
            for j, s in enumerate(batch):
                padded[j, :len(s)] = s

            context = torch.tensor(padded, dtype=torch.float32)

            with torch.no_grad():
                # Extract encoder hidden states
                token_ids, attention_mask, scale = pipeline.tokenizer._input_transform(context)                
                token_ids = token_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)

                encoder_out = pipeline.model.model.encoder(
                    input_ids=token_ids,
                    attention_mask=attention_mask
                )
                hidden = encoder_out.last_hidden_state  # (batch, seq_len, hidden_dim)

                # Mean pool over time (respecting attention mask)
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (batch, hidden_dim)
                var_embeddings.append(pooled.cpu().numpy())

        var_embeddings = np.concatenate(var_embeddings, axis=0)  # (n_patients, hidden_dim)
        all_var_embeddings.append(var_embeddings)

    # Stack: (n_patients, n_vars, hidden_dim)
    return np.stack(all_var_embeddings, axis=1)


# =============================================================================
# STEP 4: Aggregation strategies
# =============================================================================
def mean_aggregate(embeddings: np.ndarray, static: np.ndarray) -> np.ndarray:
    """
    Simple mean across variable dimension.
    embeddings: (n_patients, n_vars, hidden_dim)
    Returns: (n_patients, hidden_dim + n_static)
    """
    pooled = embeddings.mean(axis=1)  # (n_patients, hidden_dim)
    return np.concatenate([pooled, static], axis=1)


class LearnedChannelAggregator(nn.Module):
    """
    Learns a weighted combination of variable embeddings,
    then applies a linear classifier jointly.
    """
    def __init__(self, n_vars: int, hidden_dim: int, n_static: int):
        super().__init__()
        self.channel_weights = nn.Linear(n_vars, 1, bias=False)  # learned aggregation
        self.classifier = nn.Linear(hidden_dim + n_static, 1)    # linear probe

    def forward(self, embeddings: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        # embeddings: (batch, n_vars, hidden_dim)
        # Weighted sum across vars: (batch, hidden_dim)
        weights = torch.softmax(self.channel_weights.weight, dim=-1)  # (1, n_vars)
        pooled = (embeddings * weights.unsqueeze(-1)).sum(dim=1)       # (batch, hidden_dim)
        combined = torch.cat([pooled, static], dim=1)                  # (batch, hidden_dim + n_static)
        return self.classifier(combined).squeeze(-1)                   # (batch,)


def train_learned_aggregator(
    train_embeddings: np.ndarray,
    train_static: np.ndarray,
    train_labels: np.ndarray,
) -> LearnedChannelAggregator:
    n_vars, hidden_dim = train_embeddings.shape[1], train_embeddings.shape[2]
    n_static = train_static.shape[1]

    model = LearnedChannelAggregator(n_vars, hidden_dim, n_static).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    # Build tensors
    X_emb = torch.tensor(train_embeddings, dtype=torch.float32)
    X_sta = torch.tensor(train_static, dtype=torch.float32)
    y     = torch.tensor(train_labels, dtype=torch.float32)

    dataset = TensorDataset(X_emb, X_sta, y)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for emb_batch, sta_batch, y_batch in loader:
            emb_batch = emb_batch.to(DEVICE)
            sta_batch = sta_batch.to(DEVICE)
            y_batch   = y_batch.to(DEVICE)

            optimizer.zero_grad()
            logits = model(emb_batch, sta_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss/len(loader):.4f}", flush=True)

    return model


def evaluate_learned_aggregator(
    model: LearnedChannelAggregator,
    test_embeddings: np.ndarray,
    test_static: np.ndarray,
    test_labels: np.ndarray,
) -> dict:
    model.eval()
    X_emb = torch.tensor(test_embeddings, dtype=torch.float32).to(DEVICE)
    X_sta = torch.tensor(test_static, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = model(X_emb, X_sta)
        probs = torch.sigmoid(logits).cpu().numpy()

    auroc = roc_auc_score(test_labels, probs)
    auprc = average_precision_score(test_labels, probs)
    return {"AUROC": auroc, "AUPRC": auprc}


# =============================================================================
# STEP 5: Linear probe on mean-aggregated embeddings
# =============================================================================
def train_linear_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> dict:
    print("\nTraining linear probe (mean aggregation)...", flush=True)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_features, train_labels)
    probs = clf.predict_proba(test_features)[:, 1]
    auroc = roc_auc_score(test_labels, probs)
    auprc = average_precision_score(test_labels, probs)
    print(f"  Mean Agg Linear Probe — AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")
    return {"AUROC": auroc, "AUPRC": auprc}


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # --- Load data ---
    print("Loading data...", flush=True)
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df  = pd.read_parquet(TEST_PATH)
    scaler   = load_scaler(SCALER_PATH)

    # --- Build datasets ---
    train_dynamic, train_static_df, train_labels, _ = build_dataset(train_df, "train")
    test_dynamic,  test_static_df,  test_labels,  _ = build_dataset(test_df,  "test")

    # --- Normalize static features using train scaler params ---
    train_static = normalize_static(train_static_df, scaler)
    test_static  = normalize_static(test_static_df,  scaler)

    # --- Load Chronos ---
    print(f"\nLoading Chronos model: {CHRONOS_MODEL}...", flush=True)
    pipeline = ChronosPipeline.from_pretrained(
        CHRONOS_MODEL,
        device_map=DEVICE,
        torch_dtype=torch.float32,
    )
    pipeline.model.eval()

    # --- Generate embeddings ---
    train_embeddings = get_chronos_embeddings(pipeline, train_dynamic, "train")
    test_embeddings  = get_chronos_embeddings(pipeline, test_dynamic,  "test")

    # Save embeddings
    np.save("chronos_train_embeddings.npy", train_embeddings)
    np.save("chronos_train_labels.npy",     train_labels)
    np.save("chronos_test_embeddings.npy",  test_embeddings)
    np.save("chronos_test_labels.npy",      test_labels)
    np.save("chronos_train_static.npy",     train_static)
    np.save("chronos_test_static.npy",      test_static)
    print("Embeddings saved.", flush=True)

    # --- Strategy 1: Mean aggregation + logistic regression ---
    train_mean_features = mean_aggregate(train_embeddings, train_static)
    test_mean_features  = mean_aggregate(test_embeddings,  test_static)
    mean_metrics = train_linear_probe(
        train_mean_features, train_labels,
        test_mean_features,  test_labels
    )

    # --- Strategy 2: Learned aggregation + linear classifier (end-to-end) ---
    print("\nTraining learned channel aggregator...", flush=True)
    learned_model = train_learned_aggregator(
        train_embeddings, train_static, train_labels
    )
    learned_metrics = evaluate_learned_aggregator(
        learned_model, test_embeddings, test_static, test_labels
    )
    print(f"  Learned Agg — AUROC: {learned_metrics['AUROC']:.4f} | AUPRC: {learned_metrics['AUPRC']:.4f}")

    # --- Final summary ---
    print("\n=== FINAL SUMMARY ===")
    print(f"Mean Aggregation   — AUROC: {mean_metrics['AUROC']:.4f} | AUPRC: {mean_metrics['AUPRC']:.4f}")
    print(f"Learned Aggregation— AUROC: {learned_metrics['AUROC']:.4f} | AUPRC: {learned_metrics['AUPRC']:.4f}")