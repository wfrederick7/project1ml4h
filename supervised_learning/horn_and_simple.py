# q2_3_compare_transformers.py

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

GRID_DATA_DIR = BASE_DIR / "data" / "processed_derived"
HORN_DATA_DIR = BASE_DIR / "data" / "horn_processed"

RESULTS_DIR = BASE_DIR / "supervised_learning" / "results" / "q2_3_compare_transformers"
GRID_RESULTS_DIR = RESULTS_DIR / "time_grid"
HORN_RESULTS_DIR = RESULTS_DIR / "horn_tokens"

GRID_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
HORN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GRID_FILES = {
    "a": GRID_DATA_DIR / "set_a_linear.parquet",
    "b": GRID_DATA_DIR / "set_b_linear.parquet",
    "c": GRID_DATA_DIR / "set_c_linear.parquet",
}

HORN_FILES = {
    "a": HORN_DATA_DIR / "set_a_horn.parquet",
    "b": HORN_DATA_DIR / "set_b_horn.parquet",
    "c": HORN_DATA_DIR / "set_c_horn.parquet",
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

GRID_FEATURE_COLS = STATIC_COLS + DYNAMIC_COLS
GRID_INPUT_DIM = len(GRID_FEATURE_COLS)   # 41
GRID_EXPECTED_SEQ_LEN = 49

HORN_OHE_COLS = [f"var_{c}" for c in (STATIC_COLS + DYNAMIC_COLS)]
HORN_FEATURE_COLS = ["t_scaled", "value_scaled"] + HORN_OHE_COLS
HORN_INPUT_DIM = len(HORN_FEATURE_COLS)   # 43


# =========================================================
# Shared helpers
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


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()

    all_probs = []
    all_y = []

    for batch in loader:
        X_batch = batch["x"].to(device)
        y_batch = batch["y"].to(device)
        pad_mask = batch["pad_mask"].to(device)

        logits = model(X_batch, pad_mask=pad_mask)
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

    for batch in loader:
        X_batch = batch["x"].to(device)
        y_batch = batch["y"].to(device)
        pad_mask = batch["pad_mask"].to(device)

        optimizer.zero_grad()

        logits = model(X_batch, pad_mask=pad_mask)
        loss = criterion(logits, y_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = X_batch.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

    return total_loss / n_samples


# =========================================================
# Data preparation: time-grid
# =========================================================

def make_grid_sequence_arrays(df: pd.DataFrame):
    """
    Converts regular time-grid dataframe into:
      X: [N, 49, 41]
      y: [N]
      patient_ids: [N]
      lengths: [N] (always 49)
    """
    df = df.sort_values([ID_COL, TIME_COL]).copy()

    X_list = []
    y_list = []
    id_list = []
    len_list = []

    for patient_id, g in df.groupby(ID_COL):
        g = g.sort_values(TIME_COL)

        if len(g) != GRID_EXPECTED_SEQ_LEN:
            raise ValueError(
                f"Patient {patient_id} has {len(g)} rows, expected {GRID_EXPECTED_SEQ_LEN}"
            )

        x = g[GRID_FEATURE_COLS].to_numpy(dtype=np.float32)
        y = int(g[TARGET_COL].iloc[0])

        X_list.append(x)
        y_list.append(y)
        id_list.append(int(patient_id))
        len_list.append(GRID_EXPECTED_SEQ_LEN)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32)
    patient_ids = np.array(id_list, dtype=np.int64)
    lengths = np.array(len_list, dtype=np.int64)

    return X, y, patient_ids, lengths


# =========================================================
# Data preparation: Horn tokens
# =========================================================

def make_horn_sequence_list(df: pd.DataFrame):
    """
    Converts Horn token dataframe into per-patient variable-length sequences.

    Returns:
      X_list: list of [T_i, 43] arrays
      y: [N]
      patient_ids: [N]
      lengths: [N]
    """
    sort_cols = [ID_COL, "t_scaled"]
    df = df.sort_values(sort_cols).copy()

    missing_cols = [c for c in HORN_FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Horn dataframe missing columns: {missing_cols}")

    X_list = []
    y_list = []
    id_list = []
    len_list = []

    for patient_id, g in df.groupby(ID_COL, sort=False):
        g = g.sort_values("t_scaled")

        x = g[HORN_FEATURE_COLS].to_numpy(dtype=np.float32)
        y = int(g[TARGET_COL].iloc[0])

        X_list.append(x)
        y_list.append(y)
        id_list.append(int(patient_id))
        len_list.append(len(g))

    y = np.array(y_list, dtype=np.float32)
    patient_ids = np.array(id_list, dtype=np.int64)
    lengths = np.array(len_list, dtype=np.int64)

    return X_list, y, patient_ids, lengths


# =========================================================
# Dataset + collate
# =========================================================

class VariableLengthSequenceDataset(Dataset):
    def __init__(self, X_list, y, patient_ids):
        self.X_list = X_list
        self.y = np.asarray(y, dtype=np.float32)
        self.patient_ids = np.asarray(patient_ids, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "x": self.X_list[idx],
            "y": self.y[idx],
            "patient_id": self.patient_ids[idx],
        }


def pad_collate(batch):
    """
    Pads variable-length sequences to max length in batch.

    Returns:
      x: [B, T_max, F]
      y: [B]
      pad_mask: [B, T_max] with True where padded
      lengths: [B]
      patient_ids: [B]
    """
    lengths = [item["x"].shape[0] for item in batch]
    max_len = max(lengths)
    feat_dim = batch[0]["x"].shape[1]
    bs = len(batch)

    x_padded = torch.zeros((bs, max_len, feat_dim), dtype=torch.float32)
    pad_mask = torch.ones((bs, max_len), dtype=torch.bool)
    y = torch.zeros(bs, dtype=torch.float32)
    patient_ids = torch.zeros(bs, dtype=torch.int64)

    for i, item in enumerate(batch):
        x = torch.tensor(item["x"], dtype=torch.float32)
        L = x.shape[0]

        x_padded[i, :L] = x
        pad_mask[i, :L] = False
        y[i] = float(item["y"])
        patient_ids[i] = int(item["patient_id"])

    return {
        "x": x_padded,
        "y": y,
        "pad_mask": pad_mask,
        "lengths": torch.tensor(lengths, dtype=torch.int64),
        "patient_ids": patient_ids,
    }


# =========================================================
# Positional encoding
# =========================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
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
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds max positional length {self.pe.size(1)}"
            )
        return x + self.pe[:, :seq_len, :]


# =========================================================
# Transformer model
# =========================================================

class SimpleTransformerClassifier(nn.Module):
    """
    Same basic transformer idea as Q2.3a, but now supports padding masks.

    Pipeline:
    - project input features to d_model
    - add sinusoidal positional encoding
    - transformer encoder
    - masked mean pool over valid tokens
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
        max_len: int = 2048,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)

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

    def forward(self, x, pad_mask=None):
        """
        x: [B, T, F]
        pad_mask: [B, T] with True at padded positions
        """
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        if pad_mask is None:
            x = x.mean(dim=1)
        else:
            valid_mask = (~pad_mask).unsqueeze(-1).float()   # [B, T, 1]
            x = x * valid_mask
            denom = valid_mask.sum(dim=1).clamp(min=1.0)
            x = x.sum(dim=1) / denom

        x = self.norm(x)
        x = self.dropout(x)

        logits = self.classifier(x).squeeze(-1)
        return logits


# =========================================================
# Training
# =========================================================

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
    max_len=2048,
):
    model = SimpleTransformerClassifier(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=max_len,
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
    results_dir: Path,
    task_name: str,
    **model_kwargs
):
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = n_neg / max(n_pos, 1)

    print(f"\n=== Training {task_name} | {config_name} ===")
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

    model_path = results_dir / f"{config_name}_best_model.pt"
    torch.save(model.state_dict(), model_path)

    pd.DataFrame({
        "y_true": y_test,
        "y_prob": test_prob
    }).to_csv(results_dir / f"{config_name}_test_predictions.csv", index=False)

    pd.DataFrame(history).to_csv(results_dir / f"{config_name}_history.csv", index=False)

    return {
        "task_name": task_name,
        "config_name": config_name,
        "model_path": str(model_path),
        "validation": val_metrics,
        "test": test_metrics,
        "hyperparameters": model_kwargs,
    }


# =========================================================
# Config grid from Q2.3a
# =========================================================

TRANSFORMER_CONFIGS = [
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


# =========================================================
# Experiment runner
# =========================================================

def run_experiment(
    task_name: str,
    train_loader,
    val_loader,
    test_loader,
    y_train,
    input_dim: int,
    results_dir: Path,
    max_len: int,
    device,
):
    all_results = []
    best_score = (-np.inf, -np.inf)
    best_result = None

    for cfg in TRANSFORMER_CONFIGS:
        cfg = cfg.copy()
        config_name = cfg.pop("config_name")
        cfg["max_len"] = max_len

        result = fit_and_evaluate_config(
            config_name=config_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            input_dim=input_dim,
            device=device,
            y_train=y_train,
            results_dir=results_dir,
            task_name=task_name,
            **cfg,
        )

        all_results.append(result)

        score = (
            result["validation"]["auprc"],
            result["validation"]["auroc"]
        )
        if score > best_score:
            best_score = score
            best_result = result

    summary_rows = []
    for r in all_results:
        row = {
            "task_name": r["task_name"],
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
    summary_df.to_csv(results_dir / "all_config_results.csv", index=False)

    summary_json = {
        "task": task_name,
        "selection_rule": "best validation AUPRC, break ties with validation AUROC",
        "all_results": all_results,
        "best_config": best_result,
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    print("\n====================================================")
    print(f"Best config for {task_name}:")
    print(best_result["config_name"])
    print("Validation:", best_result["validation"])
    print("Test:", best_result["test"])
    print(f"Saved results to: {results_dir}")
    print("====================================================")

    return all_results, best_result, summary_df


# =========================================================
# Main
# =========================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -----------------------------------------------------
    # 1) Original Q2.3a regular time-grid data
    # -----------------------------------------------------
    print("\nLoading regular time-grid data...")
    grid_train_df = load_df(GRID_FILES["a"])
    grid_val_df   = load_df(GRID_FILES["b"])
    grid_test_df  = load_df(GRID_FILES["c"])

    X_train_grid, y_train_grid, train_ids_grid, len_train_grid = make_grid_sequence_arrays(grid_train_df)
    X_val_grid, y_val_grid, val_ids_grid, len_val_grid = make_grid_sequence_arrays(grid_val_df)
    X_test_grid, y_test_grid, test_ids_grid, len_test_grid = make_grid_sequence_arrays(grid_test_df)

    print("Grid train shape:", X_train_grid.shape, y_train_grid.shape)
    print("Grid val shape:  ", X_val_grid.shape, y_val_grid.shape)
    print("Grid test shape: ", X_test_grid.shape, y_test_grid.shape)

    grid_train_ds = VariableLengthSequenceDataset(list(X_train_grid), y_train_grid, train_ids_grid)
    grid_val_ds   = VariableLengthSequenceDataset(list(X_val_grid), y_val_grid, val_ids_grid)
    grid_test_ds  = VariableLengthSequenceDataset(list(X_test_grid), y_test_grid, test_ids_grid)

    grid_train_loader = DataLoader(grid_train_ds, batch_size=64, shuffle=True, collate_fn=pad_collate)
    grid_val_loader   = DataLoader(grid_val_ds, batch_size=256, shuffle=False, collate_fn=pad_collate)
    grid_test_loader  = DataLoader(grid_test_ds, batch_size=256, shuffle=False, collate_fn=pad_collate)

    grid_all_results, grid_best_result, grid_summary_df = run_experiment(
        task_name="q2_3a_regular_time_grid",
        train_loader=grid_train_loader,
        val_loader=grid_val_loader,
        test_loader=grid_test_loader,
        y_train=y_train_grid,
        input_dim=GRID_INPUT_DIM,
        results_dir=GRID_RESULTS_DIR,
        max_len=GRID_EXPECTED_SEQ_LEN,
        device=device,
    )

    # -----------------------------------------------------
    # 2) Q2.3b Horn-style tokenized measurement sequences
    # -----------------------------------------------------
    print("\nLoading Horn token data...")
    horn_train_df = load_df(HORN_FILES["a"])
    horn_val_df   = load_df(HORN_FILES["b"])
    horn_test_df  = load_df(HORN_FILES["c"])

    X_train_horn, y_train_horn, train_ids_horn, len_train_horn = make_horn_sequence_list(horn_train_df)
    X_val_horn, y_val_horn, val_ids_horn, len_val_horn = make_horn_sequence_list(horn_val_df)
    X_test_horn, y_test_horn, test_ids_horn, len_test_horn = make_horn_sequence_list(horn_test_df)

    print("Horn patients train/val/test:", len(X_train_horn), len(X_val_horn), len(X_test_horn))
    print("Horn avg tokens train:", float(np.mean(len_train_horn)))
    print("Horn max tokens train:", int(np.max(len_train_horn)))

    horn_train_ds = VariableLengthSequenceDataset(X_train_horn, y_train_horn, train_ids_horn)
    horn_val_ds   = VariableLengthSequenceDataset(X_val_horn, y_val_horn, val_ids_horn)
    horn_test_ds  = VariableLengthSequenceDataset(X_test_horn, y_test_horn, test_ids_horn)

    horn_train_loader = DataLoader(horn_train_ds, batch_size=64, shuffle=True, collate_fn=pad_collate)
    horn_val_loader   = DataLoader(horn_val_ds, batch_size=256, shuffle=False, collate_fn=pad_collate)
    horn_test_loader  = DataLoader(horn_test_ds, batch_size=256, shuffle=False, collate_fn=pad_collate)

    horn_max_len = int(max(
        np.max(len_train_horn),
        np.max(len_val_horn),
        np.max(len_test_horn),
    ))

    horn_all_results, horn_best_result, horn_summary_df = run_experiment(
        task_name="q2_3b_horn_measurement_tokens",
        train_loader=horn_train_loader,
        val_loader=horn_val_loader,
        test_loader=horn_test_loader,
        y_train=y_train_horn,
        input_dim=HORN_INPUT_DIM,
        results_dir=HORN_RESULTS_DIR,
        max_len=horn_max_len,
        device=device,
    )

    # -----------------------------------------------------
    # 3) Direct comparison table
    # -----------------------------------------------------
    comparison_rows = []
    for config_name in grid_summary_df["config_name"].unique():
        grid_row = grid_summary_df[grid_summary_df["config_name"] == config_name].iloc[0]
        horn_row = horn_summary_df[horn_summary_df["config_name"] == config_name].iloc[0]

        comparison_rows.append({
            "config_name": config_name,

            "grid_val_auroc": grid_row["val_auroc"],
            "grid_val_auprc": grid_row["val_auprc"],
            "grid_test_auroc": grid_row["test_auroc"],
            "grid_test_auprc": grid_row["test_auprc"],

            "horn_val_auroc": horn_row["val_auroc"],
            "horn_val_auprc": horn_row["val_auprc"],
            "horn_test_auroc": horn_row["test_auroc"],
            "horn_test_auprc": horn_row["test_auprc"],

            "delta_test_auroc_horn_minus_grid": horn_row["test_auroc"] - grid_row["test_auroc"],
            "delta_test_auprc_horn_minus_grid": horn_row["test_auprc"] - grid_row["test_auprc"],
        })

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        by="delta_test_auprc_horn_minus_grid",
        ascending=False
    )
    comparison_df.to_csv(RESULTS_DIR / "grid_vs_horn_comparison.csv", index=False)

    final_summary = {
        "regular_time_grid_best_config": grid_best_result,
        "horn_tokens_best_config": horn_best_result,
        "comparison_file": str(RESULTS_DIR / "grid_vs_horn_comparison.csv"),
    }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(final_summary, f, indent=2)

    print("\nDone.")
    print(f"Top-level results dir: {RESULTS_DIR}")
    print(f"Comparison table: {RESULTS_DIR / 'grid_vs_horn_comparison.csv'}")


if __name__ == "__main__":
    main()