from __future__ import annotations

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
    "ALP","ALT","AST","Albumin","BUN","Bilirubin","Cholesterol",
    "Creatinine","DiasABP","FiO2","GCS","Glucose","HCO3","HCT",
    "HR","K","Lactate","MAP","MechVent","Mg","NIDiasABP","NIMAP",
    "NISysABP","Na","PaCO2","PaO2","Platelets","RespRate","SaO2",
    "SysABP","Temp","TroponinI","TroponinT","Urine","WBC","Weight","pH"
]

FEATURE_COLS = STATIC_COLS + DYNAMIC_COLS
EXPECTED_SEQ_LEN = 49


# =========================================================
# Helpers
# =========================================================

def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_parquet(path)

def sort_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

def evaluate_probs(y_true, y_prob):
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
    }

def compare_score(m):
    return (m["auprc"], m["auroc"])

def compute_pos_weight(y):
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    return n_neg / max(n_pos, 1)


# =========================================================
# Sequence construction
# =========================================================

def make_sequence_arrays(df):
    df = sort_df(df)
    X_list, y_list, id_list = [], [], []

    for pid, g in df.groupby(ID_COL):
        if len(g) != EXPECTED_SEQ_LEN:
            raise ValueError(f"Patient {pid} has wrong length")

        X_list.append(g[FEATURE_COLS].to_numpy(dtype=np.float32))
        y_list.append(int(g[TARGET_COL].iloc[0]))
        id_list.append(int(pid))

    return np.stack(X_list), np.array(y_list), np.array(id_list)


def make_mask_delta_arrays(raw_df):
    raw_df = sort_df(raw_df)

    masks, deltas, ids = [], [], []

    for pid, g in raw_df.groupby(ID_COL):
        obs = g[FEATURE_COLS].notna().to_numpy(dtype=np.float32)
        delta = np.zeros_like(obs)

        last_seen = np.full(obs.shape[1], -1)

        for t in range(obs.shape[0]):
            seen = obs[t] > 0.5
            last_seen[seen] = t
            delta[t] = np.where(last_seen >= 0, t - last_seen, t + 1)

        masks.append(obs)
        deltas.append(delta)
        ids.append(pid)

    return np.stack(masks), np.stack(deltas), np.array(ids)


def merge_features(base, mask, delta, use_mask):
    if not use_mask:
        return base
    return np.concatenate([base, mask, delta], axis=-1)


# =========================================================
# Dataset
# =========================================================

class ICUSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================================================
# Model
# =========================================================

class SequenceClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        num_layers=1,
        dropout=0.2,
        bidirectional=False,
        pooling="recency",
        recency_strength=2.0,
    ):
        super().__init__()

        self.pooling = pooling
        self.recency_strength = recency_strength

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)

        self.dropout = nn.Dropout(dropout)

        # ONLY linear head
        self.head = nn.Linear(out_dim, 1)

    def recency_weights(self, T, device):
        t = torch.linspace(0, 1, T, device=device)
        w = torch.exp(self.recency_strength * t)
        return w / w.sum()

    def pool(self, h):
        if self.pooling == "last":
            return h[:, -1], None
        if self.pooling == "mean":
            return h.mean(1), None
        if self.pooling == "max":
            return h.max(1).values, None
        if self.pooling == "recency":
            w = self.recency_weights(h.size(1), h.device)
            return (h * w.view(1, -1, 1)).sum(1), None
        raise ValueError

    def forward(self, x):
        h, _ = self.lstm(x)
        h = self.dropout(h)
        pooled, weights = self.pool(h)
        logits = self.head(pooled).squeeze(-1)
        return logits, weights


# =========================================================
# Training
# =========================================================

@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    ys, ps = [], []
    for X, y in loader:
        X = X.to(device)
        logits, _ = model(X)
        probs = torch.sigmoid(logits)
        ys.append(y.numpy())
        ps.append(probs.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def run_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total, n = 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        opt.zero_grad()
        logits, _ = model(X)
        loss = loss_fn(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        total += loss.item() * len(X)
        n += len(X)

    return total / n


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


# =========================================================
# Experiments
# =========================================================

def build_experiments():
    return [
        ExperimentConfig("uni_last", False, "last"),
        ExperimentConfig("uni_mean", False, "mean"),
        ExperimentConfig("uni_recency_2", False, "recency", 2),
        ExperimentConfig("uni_recency_3", False, "recency", 3),
        ExperimentConfig("uni_recency_4", False, "recency", 4),

        ExperimentConfig("bi_last", True, "last"),
        ExperimentConfig("bi_mean", True, "mean"),
        ExperimentConfig("bi_max", True, "max"),

        ExperimentConfig("bi_recency_0", True, "recency", 0),
        ExperimentConfig("bi_recency_1", True, "recency", 1),
        ExperimentConfig("bi_recency_2", True, "recency", 2),
        ExperimentConfig("bi_recency_3", True, "recency", 3),
        ExperimentConfig("bi_recency_4", True, "recency", 4),
        ExperimentConfig("bi_recency_5", True, "recency", 5),
        ExperimentConfig("bi_recency_6", True, "recency", 6),

        ExperimentConfig("bi_mean_mask_delta", True, "mean", use_mask_delta=True),
    ]


# =========================================================
# Main
# =========================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_df = load_df(LINEAR_FILES["a"])
    val_df = load_df(LINEAR_FILES["b"])
    test_df = load_df(LINEAR_FILES["c"])

    X_train_base, y_train, train_ids = make_sequence_arrays(train_df)
    X_val_base, y_val, val_ids = make_sequence_arrays(val_df)
    X_test_base, y_test, test_ids = make_sequence_arrays(test_df)

    raw_available = all(p.exists() for p in RAW_FILES.values())

    if raw_available:
        train_raw = load_df(RAW_FILES["a"])
        val_raw = load_df(RAW_FILES["b"])
        test_raw = load_df(RAW_FILES["c"])

        train_mask, train_delta, train_raw_ids = make_mask_delta_arrays(train_raw)
        val_mask, val_delta, val_raw_ids = make_mask_delta_arrays(val_raw)
        test_mask, test_delta, test_raw_ids = make_mask_delta_arrays(test_raw)
    else:
        train_mask = train_delta = val_mask = val_delta = test_mask = test_delta = None

    results = {}
    ranking = []

    for cfg in build_experiments():
        if cfg.use_mask_delta and not raw_available:
            continue

        X_train = merge_features(X_train_base, train_mask, train_delta, cfg.use_mask_delta)
        X_val = merge_features(X_val_base, val_mask, val_delta, cfg.use_mask_delta)
        X_test = merge_features(X_test_base, test_mask, test_delta, cfg.use_mask_delta)

        train_loader = DataLoader(ICUSequenceDataset(X_train, y_train), cfg.batch_size, True)
        val_loader = DataLoader(ICUSequenceDataset(X_val, y_val), 256)
        test_loader = DataLoader(ICUSequenceDataset(X_test, y_test), 256)

        model = SequenceClassifier(
            X_train.shape[-1],
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            bidirectional=cfg.bidirectional,
            pooling=cfg.pooling,
            recency_strength=cfg.recency_strength
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([compute_pos_weight(y_train)], device=device)
        )

        best_score = (-math.inf, -math.inf)
        best_state = None
        history = []

        for epoch in range(1, cfg.max_epochs + 1):
            loss = run_epoch(model, train_loader, opt, loss_fn, device)
            yv, pv = predict_probs(model, val_loader, device)
            metrics = evaluate_probs(yv, pv)
            score = compare_score(metrics)

            history.append({"epoch": epoch, **metrics})

            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)

        yv, pv = predict_probs(model, val_loader, device)
        yt, pt = predict_probs(model, test_loader, device)

        val_metrics = evaluate_probs(yv, pv)
        test_metrics = evaluate_probs(yt, pt)

        results[cfg.name] = {
            "validation": val_metrics,
            "test": test_metrics,
        }

        ranking.append({
            "name": cfg.name,
            "val_auprc": val_metrics["auprc"],
            "test_auprc": test_metrics["auprc"],
        })

        torch.save(model.state_dict(), RESULTS_DIR / f"{cfg.name}.pt")
        pd.DataFrame(history).to_csv(RESULTS_DIR / f"{cfg.name}_history.csv", index=False)
        pd.DataFrame({"y_true": yt, "y_prob": pt}).to_csv(RESULTS_DIR / f"{cfg.name}_preds.csv", index=False)

    pd.DataFrame(ranking).sort_values("val_auprc", ascending=False).to_csv(
        RESULTS_DIR / "model_comparison.csv", index=False
    )

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Done")


if __name__ == "__main__":
    main()