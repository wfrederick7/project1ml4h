import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

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

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = _RL["seed"]
seed_everything(SEED)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CKPT_DIR = Path(os.path.expanduser(_CFG["paths"]["checkpoints"]))
RESULTS_DIR = PROJECT_ROOT / "rep_learning" / "results" / "q3_1_linear_probe"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_embeddings(model, dataset, device):
    """Return (N, hidden_dim*2) embeddings and (N,) labels from a dataset."""
    model.eval()
    loader = DataLoader(dataset, batch_size=512, shuffle=False,
                        collate_fn=lambda batch: batch)
    all_emb, all_lbl = [], []
    for items in loader:
        feats = torch.stack([f for f, _ in items]).to(device)
        labels = [l for _, l in items]
        emb = model.encode(feats)
        all_emb.append(emb.cpu())
        all_lbl.extend(labels)
    return torch.cat(all_emb, dim=0).numpy(), np.array(all_lbl, dtype=np.float64)


# ---------------------------------------------------------------------------
# Linear probe evaluation
# ---------------------------------------------------------------------------
def eval_linear_probe(model, ds_train, ds_val, device):
    """Fit logistic regression on train embeddings, return (AuROC, AuPRC) on val."""
    _lp = _RL["linear_probe"]

    X_train, y_train = extract_embeddings(model, ds_train, device)
    X_val, y_val = extract_embeddings(model, ds_val, device)
    model.train()  # restore after extract_embeddings set eval mode

    mask_train = ~np.isnan(y_train)
    mask_val = ~np.isnan(y_val)
    X_train, y_train = X_train[mask_train], y_train[mask_train]
    X_val, y_val = X_val[mask_val], y_val[mask_val]

    clf = LogisticRegression(
        max_iter=_lp["max_iter"],
        solver=_lp["solver"],
        C=_lp["C"],
        random_state=SEED,
    )
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_prob), average_precision_score(y_val, y_prob)


# ---------------------------------------------------------------------------
# Load precomputed embeddings
# ---------------------------------------------------------------------------
def load_embeddings(split_key):
    """Load precomputed embeddings from .npz file. split_key: 'emb_train'|'emb_val'|'emb_test'."""
    path = CKPT_DIR / _DATA[split_key]
    data = np.load(path)
    patient_ids = data["patient_ids"] if "patient_ids" in data else np.arange(len(data["labels"]))
    return data["embeddings"], data["labels"], patient_ids


# ---------------------------------------------------------------------------
# Standalone evaluation
# ---------------------------------------------------------------------------
def main():
    print("Loading precomputed embeddings ...")
    X_train, y_train, train_ids = load_embeddings("emb_train")
    X_val, y_val, val_ids = load_embeddings("emb_val")
    X_test, y_test, test_ids = load_embeddings("emb_test")
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    val_metrics = eval_linear_probe_from_embeddings(X_train, y_train, X_val, y_val)
    print(
        f"Linear probe (val):  AuROC={val_metrics['auroc']:.4f}  "
        f"AuPRC={val_metrics['auprc']:.4f}"
    )

    test_metrics = eval_linear_probe_from_embeddings(X_train, y_train, X_test, y_test)
    print(
        f"Linear probe (test): AuROC={test_metrics['auroc']:.4f}  "
        f"AuPRC={test_metrics['auprc']:.4f}"
    )

    summary = {
        "task": "Q3.1 linear probe on frozen pretrained embeddings",
        "run": run_metadata(SEED),
        "linear_probe_hyperparameters": _RL["linear_probe"],
        "files": {
            "train": str(CKPT_DIR / _DATA["emb_train"]),
            "val": str(CKPT_DIR / _DATA["emb_val"]),
            "test": str(CKPT_DIR / _DATA["emb_test"]),
        },
        "counts": {
            "train_total": int(len(train_ids)),
            "val_total": int(len(val_ids)),
            "test_total": int(len(test_ids)),
            "train_labelled": int(np.sum(~np.isnan(y_train))),
            "val_labelled": int(np.sum(~np.isnan(y_val))),
            "test_labelled": int(np.sum(~np.isnan(y_test))),
        },
        "validation": val_metrics,
        "test": test_metrics,
    }
    out_path = RESULTS_DIR / "summary.json"
    save_json(out_path, summary)
    print(f"Saved summary: {out_path}")


def eval_linear_probe_from_embeddings(X_train, y_train, X_eval, y_eval):
    """Fit logistic regression on precomputed embeddings and return metrics dict."""
    _lp = _RL["linear_probe"]

    mask_train = ~np.isnan(y_train)
    mask_eval = ~np.isnan(y_eval)
    X_train, y_train = X_train[mask_train], y_train[mask_train]
    X_eval, y_eval = X_eval[mask_eval], y_eval[mask_eval]

    clf = LogisticRegression(
        max_iter=_lp["max_iter"], solver=_lp["solver"],
        C=_lp["C"], random_state=SEED,
    )
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_eval)[:, 1]
    return {
        "auroc": float(roc_auc_score(y_eval, y_prob)),
        "auprc": float(average_precision_score(y_eval, y_prob)),
        "n_train_labelled": int(len(y_train)),
        "n_eval_labelled": int(len(y_eval)),
    }


if __name__ == "__main__":
    main()
