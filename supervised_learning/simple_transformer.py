# q2_3a_transformer.py

from pathlib import Path
import json
import copy
import math
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, average_precision_score


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path.home() / "project1ml4h"
DATA_DIR = BASE_DIR / "data" / "processed_derived"
RESULTS_DIR = BASE_DIR / "supervised_learning" / "results" / "q2_3a_transformer"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "a_linear": DATA_DIR / "set_a_linear.parquet",
    "b_linear": DATA_DIR / "set_b_linear.parquet",
    "c_linear": DATA_DIR / "set_c_linear.parquet",
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

EXPECTED_SEQ_LEN = 49
INPUT_DIM = len(FEATURE_COLS)


# =========================================================
# Helpers
# =========================================================

def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_parquet(path)


def make_sequence_arrays(df: pd.DataFrame):
    """
    Converts a dataframe with 49 rows per patient into:
      X: [N, 49, 41]
      y: [N]
      patient_ids: [N]
    """
    df = df.sort_values([ID_COL, TIME_COL]).copy()

    X_list = []
    y_list = []
    id_list = []

    for patient_id, g in df.groupby(ID_COL):
        g = g.sort_values(TIME_COL)

        if len(g) != EXPECTED_SEQ_LEN:
            raise ValueError(
                f"Patient {patient_id} has {len(g)} rows, expected {EXPECTED_SEQ_LEN}"
            )

        x = g[FEATURE_COLS].to_numpy(dtype=np.float32)
        y = int(g[TARGET_COL].iloc[0])

        X_list.append(x)
        y_list.append(y)
        id_list.append(int(patient_id))

    X = np.stack(X_list, axis=0)   # [N, 49, 41]
    y = np.array(y_list, dtype=np.float32)
    patient_ids = np.array(id_list, dtype=np.int64)

    return X, y, patient_ids


def evaluate_probs(y_true, y_prob) -> dict:
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
    }


# =========================================================
# Dataset
# =========================================================

class ICUSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================================================
# Positional Encoding
# =========================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# =========================================================
# Model
# =========================================================

class SimpleTransformerClassifier(nn.Module):
    """
    Simple encoder-only transformer for ICU mortality prediction.

    Pipeline:
    - project 41 raw features to d_model
    - add positional encoding
    - transformer encoder
    - mean pool over time
    - classifier head
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=EXPECTED_SEQ_LEN
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        """
        x: [B, T, F]
        returns:
            logits: [B]
        """
        x = self.input_proj(x)    # [B, T, D]
        x = self.pos_encoder(x)   # [B, T, D]
        x = self.encoder(x)       # [B, T, D]

        # Simple pooling across time
        x = x.mean(dim=1)         # [B, D]

        x = self.norm(x)
        x = self.dropout(x)

        logits = self.classifier(x).squeeze(-1)  # [B]
        return logits


# =========================================================
# Training / evaluation
# =========================================================

@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()

    all_probs = []
    all_y = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu().numpy())
        all_y.append(y_batch.cpu().numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_y)
    return y_true, y_prob


def run_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    n_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = X_batch.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

    return total_loss / n_samples


def train_model(
    train_loader,
    val_loader,
    input_dim,
    device,
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.2,
    lr=1e-3,
    weight_decay=1e-4,
    max_epochs=50,
    patience=8,
    pos_weight_value=1.0,
):
    model = SimpleTransformerClassifier(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )

    best_model_state = None
    best_val_metrics = None
    best_score = (-np.inf, -np.inf)
    epochs_no_improve = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device)

        y_val, val_prob = predict_probs(model, val_loader, device)
        val_metrics = evaluate_probs(y_val, val_prob)

        score = (val_metrics["auprc"], val_metrics["auroc"])

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
        })

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_auroc={val_metrics['auroc']:.4f} | "
            f"val_auprc={val_metrics['auprc']:.4f}"
        )

        if score > best_score:
            best_score = score
            best_model_state = copy.deepcopy(model.state_dict())
            best_val_metrics = val_metrics
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_metrics, history


def fit_and_evaluate_config(
    config_name,
    train_loader,
    val_loader,
    test_loader,
    input_dim,
    device,
    y_train,
    **model_kwargs
):
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = n_neg / max(n_pos, 1)

    print(f"\n=== Training {config_name} ===")
    print(f"Using pos_weight = {pos_weight:.4f}")
    print("Hyperparameters:", model_kwargs)

    model, best_val_metrics, history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=input_dim,
        device=device,
        pos_weight_value=pos_weight,
        **model_kwargs
    )

    y_val, val_prob = predict_probs(model, val_loader, device)
    y_test, test_prob = predict_probs(model, test_loader, device)

    val_metrics = evaluate_probs(y_val, val_prob)
    test_metrics = evaluate_probs(y_test, test_prob)

    print("Best validation:", best_val_metrics)
    print("Final validation:", val_metrics)
    print("Test:", test_metrics)

    model_path = RESULTS_DIR / f"{config_name}_best_model.pt"
    torch.save(model.state_dict(), model_path)

    pd.DataFrame({
        "y_true": y_test,
        "y_prob": test_prob
    }).to_csv(RESULTS_DIR / f"{config_name}_test_predictions.csv", index=False)

    pd.DataFrame(history).to_csv(
        RESULTS_DIR / f"{config_name}_history.csv",
        index=False
    )

    return {
        "config_name": config_name,
        "model_path": str(model_path),
        "validation": val_metrics,
        "test": test_metrics,
        "hyperparameters": model_kwargs,
    }


# =========================================================
# Main
# =========================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -----------------------------------------------------
    # Load processed/scaled sequence data
    # -----------------------------------------------------
    train_df = load_df(FILES["a_linear"])
    val_df   = load_df(FILES["b_linear"])
    test_df  = load_df(FILES["c_linear"])

    X_train, y_train, train_ids = make_sequence_arrays(train_df)
    X_val, y_val, val_ids       = make_sequence_arrays(val_df)
    X_test, y_test, test_ids    = make_sequence_arrays(test_df)

    print("Train shape:", X_train.shape, y_train.shape)
    print("Val shape:  ", X_val.shape, y_val.shape)
    print("Test shape: ", X_test.shape, y_test.shape)

    train_ds = ICUSequenceDataset(X_train, y_train)
    val_ds   = ICUSequenceDataset(X_val, y_val)
    test_ds  = ICUSequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)

    # -----------------------------------------------------
    # Small structured config search
    # -----------------------------------------------------
    configs = [
        {
            "config_name": "transformer_d64_h4_l1_ff128_do02",
            "d_model": 64,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 128,
            "dropout": 0.2,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "max_epochs": 50,
            "patience": 8,
        },
        {
            "config_name": "transformer_d64_h4_l2_ff128_do02",
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.2,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "max_epochs": 50,
            "patience": 8,
        },
        {
            "config_name": "transformer_d128_h4_l2_ff256_do02",
            "d_model": 128,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.2,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "max_epochs": 50,
            "patience": 8,
        },
        {
            "config_name": "transformer_d128_h8_l2_ff256_do02",
            "d_model": 128,
            "nhead": 8,
            "num_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.2,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "max_epochs": 50,
            "patience": 8,
        },
        {
            "config_name": "transformer_d64_h4_l2_ff128_do01",
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "max_epochs": 50,
            "patience": 8,
        },
        {
            "config_name": "transformer_d64_h4_l2_ff128_do03",
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.3,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "max_epochs": 50,
            "patience": 8,
        },
    ]

    all_results = []
    best_score = (-np.inf, -np.inf)
    best_result = None

    for cfg in configs:
        cfg = cfg.copy()
        config_name = cfg.pop("config_name")

        result = fit_and_evaluate_config(
            config_name=config_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            input_dim=INPUT_DIM,
            device=device,
            y_train=y_train,
            **cfg
        )

        all_results.append(result)

        score = (
            result["validation"]["auprc"],
            result["validation"]["auroc"]
        )

        if score > best_score:
            best_score = score
            best_result = result

    # -----------------------------------------------------
    # Save summary tables
    # -----------------------------------------------------
    summary_rows = []
    for r in all_results:
        row = {
            "config_name": r["config_name"],
            "val_auroc": r["validation"]["auroc"],
            "val_auprc": r["validation"]["auprc"],
            "test_auroc": r["test"]["auroc"],
            "test_auprc": r["test"]["auprc"],
        }
        row.update(r["hyperparameters"])
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["val_auprc", "val_auroc"],
        ascending=False
    )
    summary_df.to_csv(RESULTS_DIR / "all_config_results.csv", index=False)

    final_summary = {
        "task": "Q2.3a simple transformer on regular time-grid sequences",
        "selection_rule": "best validation AUPRC, break ties with validation AUROC",
        "all_results": all_results,
        "best_config": best_result,
    }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(final_summary, f, indent=2)

    print("\n====================================================")
    print("Best config selected:")
    print(best_result["config_name"])
    print("Validation:", best_result["validation"])
    print("Test:", best_result["test"])
    print(f"Saved results to: {RESULTS_DIR}")
    print("====================================================")


if __name__ == "__main__":
    main()