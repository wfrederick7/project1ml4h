from __future__ import annotations

"""
Ablation script for Q2.2 LSTM experiments on Physionet 2012.

What it compares
----------------
1) Pooling choices over all hidden states:
   - last
   - mean
   - max
   - fixed recency-weighted mean
   - learned attention pooling
2) Directionality:
   - unidirectional
   - bidirectional
3) Input representation:
   - baseline linear-ready sequence inputs
   - baseline + observation mask + time-since-last-observation

Expected input files
--------------------
Raw hourly grid with NaNs (from preprocessing.py):
    ~/project1ml4h/data/processed/set_{a,b,c}.parquet
Linear-ready imputed/scaled sequences (from data_processing.py):
    ~/project1ml4h/data/processed_derived/set_{a,b,c}_linear.parquet

The script preserves the course split convention A/B/C = train/val/test.
It selects checkpoints by validation AUPRC first and validation AUROC second,
matching the project emphasis on AuROC and AuPRC.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import copy
import json
import math
import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path.home() / "project1ml4h"
RAW_DIR = BASE_DIR / "data" / "processed"
DERIVED_DIR = BASE_DIR / "data" / "processed_derived"
RESULTS_DIR = BASE_DIR / "supervised_learning" / "results" / "q2_2_lstm_ablation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILES = {
    "a": RAW_DIR / "set_a.parquet",
    "b": RAW_DIR / "set_b.parquet",
    "c": RAW_DIR / "set_c.parquet",
}

LINEAR_FILES = {
    "a": DERIVED_DIR / "set_a_linear.parquet",
    "b": DERIVED_DIR / "set_b_linear.parquet",
    "c": DERIVED_DIR / "set_c_linear.parquet",
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
BASE_INPUT_DIM = len(FEATURE_COLS)


# =========================================================
# Helpers
# =========================================================

def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_parquet(path)


def sort_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
    }


def compare_score(metrics: dict[str, float]) -> tuple[float, float]:
    return (metrics["auprc"], metrics["auroc"])


def compute_pos_weight(y: np.ndarray) -> float:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return n_neg / max(n_pos, 1)


# =========================================================
# Sequence construction
# =========================================================

def make_sequence_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = sort_df(df)
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    id_list: list[int] = []

    for patient_id, g in df.groupby(ID_COL):
        if len(g) != EXPECTED_SEQ_LEN:
            raise ValueError(
                f"Patient {patient_id} has {len(g)} rows, expected {EXPECTED_SEQ_LEN}"
            )
        X_list.append(g[FEATURE_COLS].to_numpy(dtype=np.float32))
        y_list.append(int(g[TARGET_COL].iloc[0]))
        id_list.append(int(patient_id))

    return (
        np.stack(X_list, axis=0),
        np.asarray(y_list, dtype=np.float32),
        np.asarray(id_list, dtype=np.int64),
    )


def make_mask_delta_arrays(raw_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build two extra per-timestep views from the raw hourly grid BEFORE imputation:
      mask[t, j]  = 1 if variable j was directly observed at time t, else 0
      delta[t, j] = hours since the last direct observation of variable j

    Static columns are treated as observed at t=0 when present and then carried with
    increasing delta afterwards.
    """
    raw_df = sort_df(raw_df)
    X_mask: list[np.ndarray] = []
    X_delta: list[np.ndarray] = []
    ids: list[int] = []

    for patient_id, g in raw_df.groupby(ID_COL):
        if len(g) != EXPECTED_SEQ_LEN:
            raise ValueError(
                f"Patient {patient_id} has {len(g)} rows, expected {EXPECTED_SEQ_LEN}"
            )

        obs = g[FEATURE_COLS].notna().to_numpy(dtype=np.float32)  # [T, F]
        delta = np.zeros_like(obs, dtype=np.float32)

        # Hours since last observation; if never observed, let it grow from 1,2,...
        last_seen = np.full(obs.shape[1], fill_value=-1, dtype=np.int32)
        for t in range(obs.shape[0]):
            seen_now = obs[t] > 0.5
            last_seen[seen_now] = t
            delta[t] = np.where(last_seen >= 0, t - last_seen, t + 1).astype(np.float32)

        X_mask.append(obs)
        X_delta.append(delta)
        ids.append(int(patient_id))

    return np.stack(X_mask, axis=0), np.stack(X_delta, axis=0), np.asarray(ids, dtype=np.int64)


def merge_features(
    base_X: np.ndarray,
    mask_X: Optional[np.ndarray],
    delta_X: Optional[np.ndarray],
    use_mask_delta: bool,
) -> np.ndarray:
    if not use_mask_delta:
        return base_X
    if mask_X is None or delta_X is None:
        raise ValueError("mask_X and delta_X must be provided when use_mask_delta=True")
    return np.concatenate([base_X, mask_X.astype(np.float32), delta_X.astype(np.float32)], axis=-1)


# =========================================================
# Dataset
# =========================================================

class ICUSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# =========================================================
# Model
# =========================================================

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # h: [B, T, H]
        attn_logits = self.score(h).squeeze(-1)      # [B, T]
        attn = torch.softmax(attn_logits, dim=1)     # [B, T]
        pooled = torch.sum(h * attn.unsqueeze(-1), dim=1)
        return pooled, attn


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        pooling: str = "recency",
        recency_strength: float = 2.0,
        mlp_hidden_dim: int = 0,
    ):
        super().__init__()
        self.pooling = pooling
        self.recency_strength = recency_strength

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.attn_pool = AttentionPooling(out_dim) if pooling == "attention" else None

        if mlp_hidden_dim > 0:
            self.head = nn.Sequential(
                nn.LayerNorm(out_dim),
                nn.Linear(out_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden_dim, 1),
            )
        else:
            self.head = nn.Linear(out_dim, 1)

    def _recency_weights(self, seq_len: int, device: torch.device) -> torch.Tensor:
        t = torch.linspace(0.0, 1.0, steps=seq_len, device=device)
        w = torch.exp(self.recency_strength * t)
        return w / w.sum()

    def _pool(self, outputs: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # outputs: [B, T, H]
        if self.pooling == "last":
            return outputs[:, -1, :], None
        if self.pooling == "mean":
            return outputs.mean(dim=1), None
        if self.pooling == "max":
            return outputs.max(dim=1).values, None
        if self.pooling == "recency":
            w = self._recency_weights(outputs.size(1), outputs.device)  # [T]
            pooled = torch.sum(outputs * w.view(1, -1, 1), dim=1)
            return pooled, w.unsqueeze(0).expand(outputs.size(0), -1)
        if self.pooling == "attention":
            assert self.attn_pool is not None
            return self.attn_pool(outputs)
        raise ValueError(f"Unsupported pooling: {self.pooling}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        outputs, _ = self.lstm(x)
        outputs = self.dropout(outputs)
        pooled, weights = self._pool(outputs)
        logits = self.head(pooled).squeeze(-1)
        return logits, weights


# =========================================================
# Training
# =========================================================

@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits, _ = model(X_batch)
        probs = torch.sigmoid(logits)
        probs_all.append(probs.cpu().numpy())
        y_all.append(y_batch.numpy())
    return np.concatenate(y_all), np.concatenate(probs_all)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits, _ = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = X_batch.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
    return total_loss / max(n, 1)


@dataclass
class ExperimentConfig:
    name: str
    bidirectional: bool
    pooling: str
    recency_strength: float = 2.0
    use_mask_delta: bool = False
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 50
    patience: int = 8
    mlp_hidden_dim: int = 0


def train_and_evaluate(
    cfg: ExperimentConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> dict:
    train_ds = ICUSequenceDataset(X_train, y_train)
    val_ds = ICUSequenceDataset(X_val, y_val)
    test_ds = ICUSequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = SequenceClassifier(
        input_dim=X_train.shape[-1],
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        bidirectional=cfg.bidirectional,
        pooling=cfg.pooling,
        recency_strength=cfg.recency_strength,
        mlp_hidden_dim=cfg.mlp_hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([compute_pos_weight(y_train)], dtype=torch.float32, device=device)
    )

    best_state = None
    best_val_metrics = None
    best_score = (-math.inf, -math.inf)
    epochs_without_improvement = 0
    history: list[dict] = []

    print(f"\n=== {cfg.name} ===")
    print(json.dumps(asdict(cfg), indent=2))

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device)
        y_val_true, y_val_prob = predict_probs(model, val_loader, device)
        val_metrics = evaluate_probs(y_val_true, y_val_prob)
        score = compare_score(val_metrics)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
        })

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_auroc={val_metrics['auroc']:.4f} | val_auprc={val_metrics['auprc']:.4f}"
        )

        if score > best_score:
            best_score = score
            best_val_metrics = val_metrics
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cfg.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_val_true, y_val_prob = predict_probs(model, val_loader, device)
    y_test_true, y_test_prob = predict_probs(model, test_loader, device)
    final_val_metrics = evaluate_probs(y_val_true, y_val_prob)
    test_metrics = evaluate_probs(y_test_true, y_test_prob)

    model_path = RESULTS_DIR / f"{cfg.name}_best.pt"
    hist_path = RESULTS_DIR / f"{cfg.name}_history.csv"
    pred_path = RESULTS_DIR / f"{cfg.name}_test_predictions.csv"
    torch.save(model.state_dict(), model_path)
    pd.DataFrame(history).to_csv(hist_path, index=False)
    pd.DataFrame({"y_true": y_test_true, "y_prob": y_test_prob}).to_csv(pred_path, index=False)

    return {
        "config": asdict(cfg),
        "input_dim": int(X_train.shape[-1]),
        "best_validation": best_val_metrics,
        "validation": final_val_metrics,
        "test": test_metrics,
        "artifacts": {
            "model_path": str(model_path),
            "history_path": str(hist_path),
            "test_predictions_path": str(pred_path),
        },
    }


# =========================================================
# Experiment grid
# =========================================================

def build_experiments() -> list[ExperimentConfig]:
    return [
        ExperimentConfig(name="uni_last", bidirectional=False, pooling="last"),
        ExperimentConfig(name="uni_mean", bidirectional=False, pooling="mean"),
        ExperimentConfig(name="uni_recency_2", bidirectional=False, pooling="recency", recency_strength=2.0),
        ExperimentConfig(name="bi_last", bidirectional=True, pooling="last"),
        ExperimentConfig(name="bi_mean", bidirectional=True, pooling="mean"),
        ExperimentConfig(name="bi_max", bidirectional=True, pooling="max"),
        ExperimentConfig(name="bi_recency_0", bidirectional=True, pooling="recency", recency_strength=0.0),
        ExperimentConfig(name="bi_recency_1", bidirectional=True, pooling="recency", recency_strength=1.0),
        ExperimentConfig(name="bi_recency_2", bidirectional=True, pooling="recency", recency_strength=2.0),
        ExperimentConfig(name="bi_recency_4", bidirectional=True, pooling="recency", recency_strength=4.0),
        ExperimentConfig(name="bi_attention", bidirectional=True, pooling="attention"),
        ExperimentConfig(name="bi_attention_mlp", bidirectional=True, pooling="attention", mlp_hidden_dim=64),
        ExperimentConfig(name="bi_mean_mask_delta", bidirectional=True, pooling="mean", use_mask_delta=True),
        ExperimentConfig(name="bi_attention_mask_delta", bidirectional=True, pooling="attention", use_mask_delta=True),
        ExperimentConfig(name="bi_attention_mask_delta_mlp", bidirectional=True, pooling="attention", use_mask_delta=True, mlp_hidden_dim=64),
    ]


# =========================================================
# Main
# =========================================================

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_linear = load_df(LINEAR_FILES["a"])
    val_linear = load_df(LINEAR_FILES["b"])
    test_linear = load_df(LINEAR_FILES["c"])

    X_train_base, y_train, train_ids = make_sequence_arrays(train_linear)
    X_val_base, y_val, val_ids = make_sequence_arrays(val_linear)
    X_test_base, y_test, test_ids = make_sequence_arrays(test_linear)

    raw_available = all(path.exists() for path in RAW_FILES.values())
    if raw_available:
        train_raw = load_df(RAW_FILES["a"])
        val_raw = load_df(RAW_FILES["b"])
        test_raw = load_df(RAW_FILES["c"])

        train_mask, train_delta, train_raw_ids = make_mask_delta_arrays(train_raw)
        val_mask, val_delta, val_raw_ids = make_mask_delta_arrays(val_raw)
        test_mask, test_delta, test_raw_ids = make_mask_delta_arrays(test_raw)

        if not (np.array_equal(train_ids, train_raw_ids) and np.array_equal(val_ids, val_raw_ids) and np.array_equal(test_ids, test_raw_ids)):
            raise ValueError("Patient order mismatch between linear and raw sequence tables.")
    else:
        print("Raw parquet files not found. Mask/delta experiments will be skipped.")
        train_mask = train_delta = val_mask = val_delta = test_mask = test_delta = None

    all_results: dict[str, dict] = {}
    ranking_rows: list[dict] = []

    for cfg in build_experiments():
        if cfg.use_mask_delta and not raw_available:
            continue

        X_train = merge_features(X_train_base, train_mask, train_delta, cfg.use_mask_delta)
        X_val = merge_features(X_val_base, val_mask, val_delta, cfg.use_mask_delta)
        X_test = merge_features(X_test_base, test_mask, test_delta, cfg.use_mask_delta)

        result = train_and_evaluate(cfg, X_train, y_train, X_val, y_val, X_test, y_test, device)
        all_results[cfg.name] = result
        ranking_rows.append({
            "name": cfg.name,
            "pooling": cfg.pooling,
            "bidirectional": cfg.bidirectional,
            "use_mask_delta": cfg.use_mask_delta,
            "input_dim": result["input_dim"],
            "val_auroc": result["validation"]["auroc"],
            "val_auprc": result["validation"]["auprc"],
            "test_auroc": result["test"]["auroc"],
            "test_auprc": result["test"]["auprc"],
        })

        with open(RESULTS_DIR / "summary.json", "w") as f:
            json.dump(all_results, f, indent=2)

    ranking_df = pd.DataFrame(ranking_rows).sort_values(["val_auprc", "val_auroc"], ascending=False)
    ranking_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    print("\nTop models by validation AUPRC:")
    print(ranking_df.head(10).to_string(index=False))
    print(f"\nSaved results to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
