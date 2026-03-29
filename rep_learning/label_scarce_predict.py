"""
Label-scarcity experiment.

For each budget in {100, 500, 1000} labelled training patients:
  1. Linear probe (logistic regression) on frozen pretrained embeddings.
  2. MLP probe (2-layer) on frozen pretrained embeddings.

Reports full test-set (Set C) performance for every setting.
Uses precomputed embeddings saved by pretrain_nce.py.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CFG = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
_DATA = _CFG["data"]
_RL = _CFG["representation_learning"]
_M = _RL["model"]
_LS = _RL["label_scarce"]
_LP = _RL["linear_probe"]

SEED = _RL["seed"]
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

CKPT_DIR = Path(os.path.expanduser(_CFG["paths"]["checkpoints"]))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_PATIENTS_LIST = _LS["n_patients"]
SUP_EPOCHS = _LS["supervised_epochs"]
SUP_LR = _LS["supervised_lr"]
SUP_WD = _LS["supervised_wd"]


# ---------------------------------------------------------------------------
# Load precomputed embeddings
# ---------------------------------------------------------------------------
def load_embeddings(split_key):
    """Load .npz saved by pretrain_nce.py. Returns (embeddings, labels)."""
    data = np.load(CKPT_DIR / _DATA[split_key])
    return data["embeddings"], data["labels"]


def filter_labelled(X, y):
    mask = ~np.isnan(y)
    return X[mask], y[mask]


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------
def train_linear_probe(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(
        C=_LP["C"], max_iter=_LP["max_iter"],
        solver=_LP["solver"], random_state=SEED,
    )
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    return {
        "auroc": roc_auc_score(y_test, y_prob),
        "auprc": average_precision_score(y_test, y_prob),
    }


# ---------------------------------------------------------------------------
# MLP probe
# ---------------------------------------------------------------------------
class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_probe(X_train, y_train, X_val, y_val, X_test, y_test):
    emb_dim = X_train.shape[1]

    X_tr_t = torch.from_numpy(X_train).float().to(DEVICE)
    y_tr_t = torch.from_numpy(y_train).float().to(DEVICE)
    X_val_t = torch.from_numpy(X_val).float().to(DEVICE)

    model = MLPProbe(emb_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=SUP_LR, weight_decay=SUP_WD)
    criterion = nn.BCEWithLogitsLoss()

    best_auroc, best_state = 0.0, None
    for _ in range(SUP_EPOCHS):
        model.train()
        loss = criterion(model(X_tr_t), y_tr_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(X_val_t)).cpu().numpy()
        auroc = roc_auc_score(y_val, val_prob)
        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(DEVICE).eval()
    with torch.no_grad():
        test_prob = torch.sigmoid(
            model(torch.from_numpy(X_test).float().to(DEVICE))
        ).cpu().numpy()
    return {
        "auroc": roc_auc_score(y_test, test_prob),
        "auprc": average_precision_score(y_test, test_prob),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    print("Loading precomputed embeddings ...")

    X_train_all, y_train_all = load_embeddings("emb_train")
    X_val, y_val = load_embeddings("emb_val")
    X_test, y_test = load_embeddings("emb_test")

    # Keep only labelled patients
    X_train_all, y_train_all = filter_labelled(X_train_all, y_train_all)
    X_val, y_val = filter_labelled(X_val, y_val)
    X_test, y_test = filter_labelled(X_test, y_test)
    print(f"Train: {len(y_train_all)}, Val: {len(y_val)}, Test: {len(y_test)}")

    rng = np.random.RandomState(SEED)

    results = []
    for n in N_PATIENTS_LIST:
        print(f"\n{'='*60}")
        print(f"  N = {n} labelled patients")
        print(f"{'='*60}")

        idx = rng.choice(len(y_train_all), size=min(n, len(y_train_all)), replace=False)
        X_sub, y_sub = X_train_all[idx], y_train_all[idx]

        lp = train_linear_probe(X_sub, y_sub, X_test, y_test)
        print(f"  LinearProbe | AuROC={lp['auroc']:.4f}  AuPRC={lp['auprc']:.4f}")

        mlp = train_mlp_probe(X_sub, y_sub, X_val, y_val, X_test, y_test)
        print(f"  MLP Probe   | AuROC={mlp['auroc']:.4f}  AuPRC={mlp['auprc']:.4f}")

        results.append({"n": n, "lp": lp, "mlp": mlp})

    # Summary table
    print(f"\n{'='*60}")
    print(f"  Summary (test set)")
    print(f"{'='*60}")
    print(f"{'N':>6} | {'LP AuROC':>10} {'LP AuPRC':>10} | {'MLP AuROC':>10} {'MLP AuPRC':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['n']:>6} | {r['lp']['auroc']:>10.4f} {r['lp']['auprc']:>10.4f} "
              f"| {r['mlp']['auroc']:>10.4f} {r['mlp']['auprc']:>10.4f}")


if __name__ == "__main__":
    main()
