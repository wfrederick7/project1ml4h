import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bilstm import LSTMEncoder
from dataset import ICUPatientDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CFG = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
_DATA = _CFG["data"]
_RL = _CFG["representation_learning"]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = _RL["seed"]
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.path.expanduser(_CFG["paths"]["data_derived"]))
CKPT_DIR = Path(os.path.expanduser(_CFG["paths"]["checkpoints"]))

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
    """Fit logistic regression on train embeddings, return AuROC on val."""
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
    return roc_auc_score(y_val, y_prob)


# ---------------------------------------------------------------------------
# Load precomputed embeddings
# ---------------------------------------------------------------------------
def load_embeddings(split_key):
    """Load precomputed embeddings from .npz file. split_key: 'emb_train'|'emb_val'|'emb_test'."""
    path = CKPT_DIR / _DATA[split_key]
    data = np.load(path)
    return data["embeddings"], data["labels"]


# ---------------------------------------------------------------------------
# Standalone evaluation
# ---------------------------------------------------------------------------
def main():
    print("Loading precomputed embeddings ...")
    X_train, y_train = load_embeddings("emb_train")
    X_val, y_val = load_embeddings("emb_val")
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

    auroc = eval_linear_probe_from_embeddings(X_train, y_train, X_val, y_val)
    print(f"Linear probe AuROC: {auroc:.4f}")


def eval_linear_probe_from_embeddings(X_train, y_train, X_val, y_val):
    """Fit logistic regression on precomputed embeddings, return AuROC on val."""
    _lp = _RL["linear_probe"]

    mask_train = ~np.isnan(y_train)
    mask_val = ~np.isnan(y_val)
    X_train, y_train = X_train[mask_train], y_train[mask_train]
    X_val, y_val = X_val[mask_val], y_val[mask_val]

    clf = LogisticRegression(
        max_iter=_lp["max_iter"], solver=_lp["solver"],
        C=_lp["C"], random_state=SEED,
    )
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_prob)


if __name__ == "__main__":
    main()
