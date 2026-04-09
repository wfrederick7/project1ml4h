"""
Q3.2 label-scarcity experiments.

Experiment A (from scratch, limited labels):
- Unidirectional LSTM (Q2.2-aligned)
- Bidirectional LSTM (Q2.2-aligned; same encoder family as Q3.1 pretraining)
- Transformer time-grid model (Q2.3a-aligned)

Experiment B:
- Linear probe (logistic regression) on frozen pretrained embeddings

All experiments use the exact same patient subsets for each label budget.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import run_metadata, save_json, seed_everything


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
    _CFG = yaml.safe_load(f)

_DATA = _CFG["data"]
_RL = _CFG["representation_learning"]
_M = _RL["model"]
_LP = _RL["linear_probe"]
_LS = _RL["label_scarce"]

ID_COL = _DATA["id_col"]
TIME_COL = _DATA["time_col"]
TARGET_COL = _DATA["target_col"]
FEATURE_COLS = _DATA["static_cols"] + _DATA["dynamic_cols"]
EXPECTED_SEQ_LEN = int(_DATA["n_hours"])

SEED = int(_RL["seed"])
seed_everything(SEED)

DATA_DIR = Path(os.path.expanduser(_CFG["paths"]["data_derived"]))
CKPT_DIR = Path(os.path.expanduser(_CFG["paths"]["checkpoints"]))
RESULTS_DIR = PROJECT_ROOT / "rep_learning" / "results" / "q3_2_label_scarce"
Q3_2_CKPT_DIR = CKPT_DIR / "q3_2_label_scarce"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
Q3_2_CKPT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_PATIENTS_LIST = [int(n) for n in _LS["n_patients"]]
SUPERVISED_EPOCHS = int(_LS["supervised_epochs"])
SUPERVISED_LR = float(_LS["supervised_lr"])
SUPERVISED_WD = float(_LS["supervised_wd"])
SUPERVISED_BATCH_SIZE = int(_LS["batch_size"])
SUPERVISED_PATIENCE = int(_LS["patience"])

# Best regular time-grid Transformer configuration from Q2.3 comparison.
_TR_CFG = _LS.get("transformer", {})
TRANSFORMER_CFG = {
    "d_model": int(_TR_CFG.get("d_model", 128)),
    "nhead": int(_TR_CFG.get("nhead", 8)),
    "num_layers": int(_TR_CFG.get("num_layers", 2)),
    "dim_feedforward": int(_TR_CFG.get("dim_feedforward", 256)),
    "dropout": float(_TR_CFG.get("dropout", 0.2)),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_sequence_arrays(parquet_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, patient_ids) with X shape (N, 49, 41)."""
    path = DATA_DIR / parquet_file
    df = pd.read_parquet(path).sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

    X_list, y_list, ids = [], [], []
    for pid, g in df.groupby(ID_COL, sort=False):
        if len(g) != EXPECTED_SEQ_LEN:
            raise ValueError(
                f"Patient {pid} has {len(g)} rows; expected {EXPECTED_SEQ_LEN}."
            )
        X_list.append(g[FEATURE_COLS].to_numpy(dtype=np.float32))
        label = g[TARGET_COL].iloc[0]
        y_list.append(float(label) if not pd.isna(label) else np.nan)
        ids.append(int(pid))

    return np.stack(X_list), np.array(y_list, dtype=np.float64), np.array(ids, dtype=np.int64)


def load_embeddings(split_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load precomputed embeddings. patient_ids may be None for old files."""
    data = np.load(CKPT_DIR / _DATA[split_key])
    ids = data["patient_ids"].astype(np.int64) if "patient_ids" in data else None
    return data["embeddings"], data["labels"], ids


def filter_labelled(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = ~np.isnan(y)
    return X[mask], y[mask].astype(np.int64), ids[mask]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
    }


def compute_pos_weight(y: np.ndarray) -> float:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return float(n_neg / max(n_pos, 1))


# ---------------------------------------------------------------------------
# Sequence datasets/models (Experiment A)
# ---------------------------------------------------------------------------
class ICUSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    """Q2.2-compatible LSTM classifier with recency-weighted pooling."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        recency_strength: float,
    ):
        super().__init__()
        self.recency_strength = recency_strength

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(out_dim, 1)

    def _recency_weights(self, T: int, device: torch.device) -> torch.Tensor:
        t = torch.linspace(0, 1, T, device=device)
        w = torch.exp(self.recency_strength * t)
        return w / w.sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        h = self.dropout(h)
        w = self._recency_weights(h.size(1), h.device)
        pooled = (h * w.view(1, -1, 1)).sum(dim=1)
        return self.head(pooled).squeeze(-1)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TransformerClassifier(nn.Module):
    """Q2.3a-compatible Transformer classifier on regular time-grid sequences."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, max_len=EXPECTED_SEQ_LEN)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.dropout(x)
        return self.classifier(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed_offset: int,
) -> DataLoader:
    ds = ICUSequenceDataset(X, y)
    generator = torch.Generator()
    generator.manual_seed(SEED + seed_offset)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_prob = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        probs = torch.sigmoid(logits)
        y_true.append(yb.numpy())
        y_prob.append(probs.cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_prob)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss, n_samples = 0.0, 0

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        n_samples += xb.size(0)

    return total_loss / max(n_samples, 1)


def train_supervised_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed_offset: int,
) -> tuple[nn.Module, dict, list[dict]]:
    train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True, seed_offset=seed_offset)
    val_loader = make_loader(X_val, y_val, batch_size=256, shuffle=False, seed_offset=seed_offset + 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([compute_pos_weight(y_train)], dtype=torch.float32, device=DEVICE)
    )

    best_state = None
    best_score = (-np.inf, -np.inf)  # (auprc, auroc)
    best_val = None
    history = []
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion)

        yv, pv = predict_probs(model, val_loader)
        val_metrics = evaluate_probs(yv, pv)
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
        }
        history.append(row)

        score = (val_metrics["auprc"], val_metrics["auroc"])
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_val = {
                "epoch": epoch,
                "auroc": val_metrics["auroc"],
                "auprc": val_metrics["auprc"],
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training produced no valid checkpoint.")

    model.load_state_dict(best_state)
    return model, best_val, history


# ---------------------------------------------------------------------------
# Linear probe (Experiment B)
# ---------------------------------------------------------------------------
def fit_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> dict:
    clf = LogisticRegression(
        C=_LP["C"],
        max_iter=_LP["max_iter"],
        solver=_LP["solver"],
        random_state=SEED,
    )
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_eval)[:, 1]
    return {
        "auroc": float(roc_auc_score(y_eval, y_prob)),
        "auprc": float(average_precision_score(y_eval, y_prob)),
        "n_train_labelled": int(len(y_train)),
        "n_eval_labelled": int(len(y_eval)),
    }


# ---------------------------------------------------------------------------
# XGBoost on engineered features (Experiment A1, matches Q2.1.2)
# ---------------------------------------------------------------------------
N_STATIC = len(_DATA["static_cols"])


def engineer_features_from_sequences(X: np.ndarray) -> np.ndarray:
    """Build per-patient engineered features from (N, T, F) sequences.

    Mirrors supervised_learning/classic_ML_2.engineer_patient_features:
    static values (constant across time) plus first / last / mean / min /
    max / std / delta / slope for each dynamic variable. Uses the hourly
    index 0..T-1 as the time axis for the slope computation.
    """
    N, T, F = X.shape
    times = np.arange(T, dtype=np.float64)
    t_centered = times - times.mean()
    denom = float((t_centered ** 2).sum())

    static = X[:, 0, :N_STATIC].astype(np.float64)
    dyn = X[:, :, N_STATIC:].astype(np.float64)

    first = dyn[:, 0, :]
    last = dyn[:, -1, :]
    mean = dyn.mean(axis=1)
    mn = dyn.min(axis=1)
    mx = dyn.max(axis=1)
    std = dyn.std(axis=1, ddof=0)
    delta = last - first

    if denom > 0:
        y_centered = dyn - mean[:, None, :]
        slope = (t_centered[None, :, None] * y_centered).sum(axis=1) / denom
    else:
        slope = np.zeros_like(mean)

    feats = np.concatenate(
        [static, first, last, mean, mn, mx, std, delta, slope], axis=1
    )
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats.astype(np.float32)


def fit_xgboost_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict, list[dict]]:
    """Exact Q2.1.2 XGBoost hyperparameter sweep, selected by val AuPRC."""
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = float(n_neg / max(n_pos, 1))

    candidates = [
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": 1.0},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": 1.0},
        {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": 1.0},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": 1.0},
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": pos_weight},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": pos_weight},
        {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": pos_weight},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": pos_weight},
    ]

    best_score = (-np.inf, -np.inf)
    best: dict | None = None
    sweep_rows: list[dict] = []

    for cfg in candidates:
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            max_depth=cfg["max_depth"],
            min_child_weight=cfg["min_child_weight"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            scale_pos_weight=cfg["scale_pos_weight"],
            random_state=SEED,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]
        val_m = evaluate_probs(y_val, val_prob)
        test_m = evaluate_probs(y_test, test_prob)
        sweep_rows.append({"config": cfg, "val": val_m, "test": test_m})

        score = (val_m["auprc"], val_m["auroc"])
        if score > best_score:
            best_score = score
            best = {"config": cfg, "val": val_m, "test": test_m}

    assert best is not None
    return best, sweep_rows


# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------
def subset_by_ids(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    subset_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    id_to_idx = {int(pid): i for i, pid in enumerate(ids)}
    missing = [int(pid) for pid in subset_ids if int(pid) not in id_to_idx]
    if missing:
        raise ValueError(f"Missing {len(missing)} patient IDs in target array. Example: {missing[:5]}")

    idx = np.array([id_to_idx[int(pid)] for pid in subset_ids], dtype=np.int64)
    return X[idx], y[idx]


def build_subsets(train_ids: np.ndarray) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(SEED)
    subsets: dict[str, np.ndarray] = {}

    for n in N_PATIENTS_LIST:
        chosen = rng.choice(train_ids, size=min(n, len(train_ids)), replace=False)
        subsets[str(n)] = np.sort(chosen.astype(np.int64))

    subsets["full"] = np.sort(train_ids.astype(np.int64))
    return subsets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading sequence data ...")
    X_train_seq_raw, y_train_seq_raw, train_ids_raw = load_sequence_arrays(_DATA["train_file"])
    X_val_seq_raw, y_val_seq_raw, val_ids_raw = load_sequence_arrays(_DATA["val_file"])
    X_test_seq_raw, y_test_seq_raw, test_ids_raw = load_sequence_arrays(_DATA["test_file"])

    X_train_seq, y_train_seq, train_ids = filter_labelled(X_train_seq_raw, y_train_seq_raw, train_ids_raw)
    X_val_seq, y_val_seq, val_ids = filter_labelled(X_val_seq_raw, y_val_seq_raw, val_ids_raw)
    X_test_seq, y_test_seq, test_ids = filter_labelled(X_test_seq_raw, y_test_seq_raw, test_ids_raw)

    print(
        f"Labelled sequence patients - Train: {len(train_ids)}, "
        f"Val: {len(val_ids)}, Test: {len(test_ids)}"
    )

    print("Loading pretrained embeddings ...")
    X_train_emb_raw, y_train_emb_raw, train_emb_ids_raw = load_embeddings("emb_train")
    X_val_emb_raw, y_val_emb_raw, val_emb_ids_raw = load_embeddings("emb_val")
    X_test_emb_raw, y_test_emb_raw, test_emb_ids_raw = load_embeddings("emb_test")

    # Backward compatibility for embedding files produced before patient_ids were stored.
    if train_emb_ids_raw is None:
        if len(X_train_emb_raw) != len(train_ids_raw):
            raise ValueError("Train embeddings do not contain patient_ids and cannot be aligned to sequence data.")
        train_emb_ids_raw = train_ids_raw.copy()
        print("Warning: emb_train has no patient_ids. Assuming same patient order as train sequence file.")
    if val_emb_ids_raw is None:
        if len(X_val_emb_raw) != len(val_ids_raw):
            raise ValueError("Val embeddings do not contain patient_ids and cannot be aligned to sequence data.")
        val_emb_ids_raw = val_ids_raw.copy()
        print("Warning: emb_val has no patient_ids. Assuming same patient order as val sequence file.")
    if test_emb_ids_raw is None:
        if len(X_test_emb_raw) != len(test_ids_raw):
            raise ValueError("Test embeddings do not contain patient_ids and cannot be aligned to sequence data.")
        test_emb_ids_raw = test_ids_raw.copy()
        print("Warning: emb_test has no patient_ids. Assuming same patient order as test sequence file.")

    X_train_emb, y_train_emb, train_emb_ids = filter_labelled(
        X_train_emb_raw, y_train_emb_raw, train_emb_ids_raw
    )
    X_val_emb, y_val_emb, val_emb_ids = filter_labelled(
        X_val_emb_raw, y_val_emb_raw, val_emb_ids_raw
    )
    X_test_emb, y_test_emb, test_emb_ids = filter_labelled(
        X_test_emb_raw, y_test_emb_raw, test_emb_ids_raw
    )

    print(
        f"Labelled embedding patients - Train: {len(train_emb_ids)}, "
        f"Val: {len(val_emb_ids)}, Test: {len(test_emb_ids)}"
    )

    subsets = build_subsets(train_ids)

    print("Engineering per-patient features for XGBoost ...")
    X_train_eng = engineer_features_from_sequences(X_train_seq)
    X_val_eng = engineer_features_from_sequences(X_val_seq)
    X_test_eng = engineer_features_from_sequences(X_test_seq)
    print(f"Engineered feature dim: {X_train_eng.shape[1]}")

    experiment_a: dict[str, dict[str, dict]] = {
        "xgboost_q2_1_2": {},
        "bi_lstm_q2_2": {},
        "transformer_q2_3a": {},
    }
    experiment_b: dict[str, dict] = {}

    table_rows = []

    for i, (label, subset_ids) in enumerate(subsets.items()):
        print(f"\n{'=' * 72}")
        print(f"N = {label} labelled patients")
        print(f"{'=' * 72}")

        # ------------------------------
        # Build supervised subset (same IDs for all models)
        # ------------------------------
        X_sub_seq, y_sub_seq = subset_by_ids(X_train_seq, y_train_seq, train_ids, subset_ids)

        # ------------------------------
        # Experiment A1: XGBoost on Q2.1.2 engineered features
        # ------------------------------
        X_sub_eng, y_sub_eng = subset_by_ids(
            X_train_eng, y_train_seq, train_ids, subset_ids
        )

        xgb_best, xgb_sweep = fit_xgboost_sweep(
            X_sub_eng, y_sub_eng,
            X_val_eng, y_val_seq,
            X_test_eng, y_test_seq,
        )
        xgb_val = xgb_best["val"]
        xgb_test = xgb_best["test"]

        xgb_dir = RESULTS_DIR / "experiment_a_from_scratch" / "xgboost_q2_1_2"
        xgb_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            xgb_dir / f"n_{label}_history.json",
            {
                "best_config": xgb_best["config"],
                "validation": xgb_val,
                "test": xgb_test,
                "sweep": xgb_sweep,
            },
        )

        experiment_a["xgboost_q2_1_2"][label] = {
            "best_config": xgb_best["config"],
            "validation": xgb_val,
            "test": xgb_test,
            "train_size": int(len(y_sub_eng)),
        }
        print(f"  XGBoost       | Test AuROC={xgb_test['auroc']:.4f}  AuPRC={xgb_test['auprc']:.4f}")

        # ------------------------------
        # Experiment A2: BiLSTM
        # ------------------------------
        bi_model = LSTMClassifier(
            input_dim=len(FEATURE_COLS),
            hidden_dim=_M["hidden_dim"],
            num_layers=_M["num_layers"],
            dropout=_M["dropout"],
            bidirectional=True,
            recency_strength=_M["recency_strength"],
        ).to(DEVICE)

        bi_model, bi_best_val, bi_hist = train_supervised_model(
            bi_model,
            X_sub_seq,
            y_sub_seq,
            X_val_seq,
            y_val_seq,
            lr=SUPERVISED_LR,
            weight_decay=SUPERVISED_WD,
            batch_size=SUPERVISED_BATCH_SIZE,
            max_epochs=SUPERVISED_EPOCHS,
            patience=SUPERVISED_PATIENCE,
            seed_offset=4000 + i,
        )

        yv_bi, pv_bi = predict_probs(bi_model, make_loader(X_val_seq, y_val_seq, 256, False, 5000 + i))
        yt_bi, pt_bi = predict_probs(bi_model, make_loader(X_test_seq, y_test_seq, 256, False, 6000 + i))
        bi_val = evaluate_probs(yv_bi, pv_bi)
        bi_test = evaluate_probs(yt_bi, pt_bi)

        bi_dir = RESULTS_DIR / "experiment_a_from_scratch" / "bi_lstm_q2_2"
        bi_dir.mkdir(parents=True, exist_ok=True)
        bi_ckpt_dir = Q3_2_CKPT_DIR / "bi_lstm_q2_2"
        bi_ckpt_dir.mkdir(parents=True, exist_ok=True)
        bi_model_path = bi_ckpt_dir / f"n_{label}_best.pt"
        torch.save(bi_model.state_dict(), bi_model_path)
        save_json(
            bi_dir / f"n_{label}_history.json",
            {
                "best_validation": bi_best_val,
                "final_validation": bi_val,
                "test": bi_test,
                "history": bi_hist,
            },
        )

        experiment_a["bi_lstm_q2_2"][label] = {
            "best_validation": bi_best_val,
            "final_validation": bi_val,
            "test": bi_test,
            "train_size": int(len(y_sub_seq)),
            "model_path": str(bi_model_path),
        }
        print(f"  BiLSTM        | Test AuROC={bi_test['auroc']:.4f}  AuPRC={bi_test['auprc']:.4f}")

        # ------------------------------
        # Experiment A3: Transformer
        # ------------------------------
        tr_model = TransformerClassifier(
            input_dim=len(FEATURE_COLS),
            d_model=TRANSFORMER_CFG["d_model"],
            nhead=TRANSFORMER_CFG["nhead"],
            num_layers=TRANSFORMER_CFG["num_layers"],
            dim_feedforward=TRANSFORMER_CFG["dim_feedforward"],
            dropout=TRANSFORMER_CFG["dropout"],
        ).to(DEVICE)

        tr_model, tr_best_val, tr_hist = train_supervised_model(
            tr_model,
            X_sub_seq,
            y_sub_seq,
            X_val_seq,
            y_val_seq,
            lr=SUPERVISED_LR,
            weight_decay=SUPERVISED_WD,
            batch_size=SUPERVISED_BATCH_SIZE,
            max_epochs=SUPERVISED_EPOCHS,
            patience=SUPERVISED_PATIENCE,
            seed_offset=7000 + i,
        )

        yv_tr, pv_tr = predict_probs(tr_model, make_loader(X_val_seq, y_val_seq, 256, False, 8000 + i))
        yt_tr, pt_tr = predict_probs(tr_model, make_loader(X_test_seq, y_test_seq, 256, False, 9000 + i))
        tr_val = evaluate_probs(yv_tr, pv_tr)
        tr_test = evaluate_probs(yt_tr, pt_tr)

        tr_dir = RESULTS_DIR / "experiment_a_from_scratch" / "transformer_q2_3a"
        tr_dir.mkdir(parents=True, exist_ok=True)
        tr_ckpt_dir = Q3_2_CKPT_DIR / "transformer_q2_3a"
        tr_ckpt_dir.mkdir(parents=True, exist_ok=True)
        tr_model_path = tr_ckpt_dir / f"n_{label}_best.pt"
        torch.save(tr_model.state_dict(), tr_model_path)
        save_json(
            tr_dir / f"n_{label}_history.json",
            {
                "best_validation": tr_best_val,
                "final_validation": tr_val,
                "test": tr_test,
                "history": tr_hist,
            },
        )

        experiment_a["transformer_q2_3a"][label] = {
            "best_validation": tr_best_val,
            "final_validation": tr_val,
            "test": tr_test,
            "train_size": int(len(y_sub_seq)),
            "model_path": str(tr_model_path),
        }
        print(f"  Transformer   | Test AuROC={tr_test['auroc']:.4f}  AuPRC={tr_test['auprc']:.4f}")

        # ------------------------------
        # Experiment B: Linear probe
        # ------------------------------
        X_sub_emb, y_sub_emb = subset_by_ids(X_train_emb, y_train_emb, train_emb_ids, subset_ids)
        lp_val = fit_linear_probe(X_sub_emb, y_sub_emb, X_val_emb, y_val_emb)
        lp_test = fit_linear_probe(X_sub_emb, y_sub_emb, X_test_emb, y_test_emb)

        experiment_b[label] = {
            "validation": lp_val,
            "test": lp_test,
            "train_size": int(len(y_sub_emb)),
        }
        print(f"  Linear probe  | Test AuROC={lp_test['auroc']:.4f}  AuPRC={lp_test['auprc']:.4f}")

        table_rows.extend([
            {
                "n_labelled": label,
                "experiment": "A_from_scratch",
                "model": "xgboost_q2_1_2",
                "val_auroc": xgb_val["auroc"],
                "val_auprc": xgb_val["auprc"],
                "test_auroc": xgb_test["auroc"],
                "test_auprc": xgb_test["auprc"],
            },
            {
                "n_labelled": label,
                "experiment": "A_from_scratch",
                "model": "bi_lstm_q2_2",
                "val_auroc": bi_val["auroc"],
                "val_auprc": bi_val["auprc"],
                "test_auroc": bi_test["auroc"],
                "test_auprc": bi_test["auprc"],
            },
            {
                "n_labelled": label,
                "experiment": "A_from_scratch",
                "model": "transformer_q2_3a",
                "val_auroc": tr_val["auroc"],
                "val_auprc": tr_val["auprc"],
                "test_auroc": tr_test["auroc"],
                "test_auprc": tr_test["auprc"],
            },
            {
                "n_labelled": label,
                "experiment": "B_linear_probe",
                "model": "logreg_on_pretrained_embeddings",
                "val_auroc": lp_val["auroc"],
                "val_auprc": lp_val["auprc"],
                "test_auroc": lp_test["auroc"],
                "test_auprc": lp_test["auprc"],
            },
        ])

    comparison_df = pd.DataFrame(table_rows)
    comparison_path = RESULTS_DIR / "comparison_table.csv"
    comparison_df.to_csv(comparison_path, index=False)

    summary = {
        "task": "Q3.2 label scarcity",
        "run": run_metadata(SEED),
        "device": str(DEVICE),
        "selection_rule_experiment_a": "early stopping by validation AuPRC, tie-break by AuROC",
        "experiment_a_from_scratch": experiment_a,
        "experiment_b_linear_probe": experiment_b,
        "table_csv": str(comparison_path),
        "checkpoint_dir": str(Q3_2_CKPT_DIR),
        "model_configs": {
            "xgboost_q2_1_2": {
                "feature_type": "engineered_q2_1_2",
                "n_features": int(X_train_eng.shape[1]),
                "sweep_size": 8,
                "selection": "best val AuPRC, tie-break AuROC",
            },
            "bi_lstm_q2_2": {
                "hidden_dim": _M["hidden_dim"],
                "num_layers": _M["num_layers"],
                "dropout": _M["dropout"],
                "recency_strength": _M["recency_strength"],
                "bidirectional": True,
            },
            "transformer_q2_3a": TRANSFORMER_CFG,
            "shared_training": {
                "lr": SUPERVISED_LR,
                "weight_decay": SUPERVISED_WD,
                "batch_size": SUPERVISED_BATCH_SIZE,
                "max_epochs": SUPERVISED_EPOCHS,
                "patience": SUPERVISED_PATIENCE,
            },
            "linear_probe": _LP,
        },
    }

    summary_path = RESULTS_DIR / "summary.json"
    save_json(summary_path, summary)
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved table:   {comparison_path}")


if __name__ == "__main__":
    main()
