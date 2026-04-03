import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bilstm import LSTMEncoder
from dataset import ICUPatientDataset, contrastive_collate
from linear_probe import extract_embeddings, eval_linear_probe
from utils import run_metadata, save_json, seed_everything

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
    _CFG = yaml.safe_load(f)
_DATA = _CFG["data"]
_RL = _CFG["representation_learning"]
_PT = _RL["pretrain"]
_M  = _RL["model"]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = _RL["seed"]
seed_everything(SEED)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.path.expanduser(_CFG["paths"]["data_derived"]))
CKPT_DIR = Path(os.path.expanduser(_CFG["paths"]["checkpoints"]))
RESULTS_DIR = PROJECT_ROOT / "rep_learning" / "results" / "q3_1_pretrain"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
TAU            = _PT["tau"]
LR             = _PT["lr"]
WD             = _PT["weight_decay"]
BATCH_SIZE     = _PT["batch_size"]
EPOCHS         = _PT["epochs"]
WARMUP_EPOCHS  = _PT["warmup_epochs"]
EVAL_EVERY     = _PT["eval_every"]
NUM_WORKERS    = _PT["num_workers"]


# =========================================================================
# InfoNCE loss
# =========================================================================
def infonce_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = TAU) -> torch.Tensor:
    """
    z1, z2: (B, proj_dim) L2-normalised embeddings.
    """
    B = z1.shape[0]
    sim_12 = (z1 @ z2.T) / tau  # (B, B)
    labels = torch.arange(B, device=z1.device)
    loss_12 = nn.functional.cross_entropy(sim_12, labels)
    loss_21 = nn.functional.cross_entropy(sim_12.T, labels)
    return 0.5 * (loss_12 + loss_21)


# =========================================================================
# Training utilities
# =========================================================================
def get_lr(epoch: int, base_lr: float) -> float:
    if epoch < WARMUP_EPOCHS:
        return base_lr * (epoch + 1) / WARMUP_EPOCHS
    return base_lr


@torch.no_grad()
def eval_contrastive_loss(model, dataset, device):
    """Compute average InfoNCE loss on a dataset."""
    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=contrastive_collate, drop_last=False)
    total_loss, n_batches = 0.0, 0
    for v1, v2 in loader:
        if v1.size(0) < 2:
            continue
        v1, v2 = v1.to(device), v2.to(device)
        z1 = model.project(v1)
        z2 = model.project(v2)
        total_loss += infonce_loss(z1, z2).item()
        n_batches += 1
    model.train()  # restore after eval
    return total_loss / max(n_batches, 1)


# =========================================================================
# Main training loop
# =========================================================================
def train():
    print(f"Device: {DEVICE}")
    print(f"Data dir: {DATA_DIR}")

    ds_train = ICUPatientDataset(DATA_DIR / _DATA["train_file"])
    ds_val   = ICUPatientDataset(DATA_DIR / _DATA["val_file"])
    print(f"Train patients: {len(ds_train)}, Val patients: {len(ds_val)}")

    train_loader = DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=contrastive_collate,
        drop_last=False,
        num_workers=NUM_WORKERS,
    )

    model = LSTMEncoder(
        input_dim=_M["input_dim"],
        hidden_dim=_M["hidden_dim"],
        num_layers=_M["num_layers"],
        dropout=_M["dropout"],
        recency_strength=_M["recency_strength"],
        proj_dim=_M["proj_dim"],
    ).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    best_score = (-np.inf, -np.inf)  # (val_auprc, val_auroc)
    best_eval = None
    history = []
    eval_history = []

    for epoch in range(EPOCHS):
        model.train()

        # Linear warmup
        lr = get_lr(epoch, LR)
        for pg in optimiser.param_groups:
            pg["lr"] = lr

        epoch_loss, n_steps = 0.0, 0
        epoch_align, epoch_uniform, epoch_collapse = 0.0, 0.0, 0.0
        for v1, v2 in train_loader:
            if v1.size(0) < 2:
                continue
            v1, v2 = v1.to(DEVICE), v2.to(DEVICE)

            z1 = model.project(v1)
            z2 = model.project(v2)
            loss = infonce_loss(z1, z2)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()
            n_steps += 1

            # Monitoring metrics (detached)
            with torch.no_grad():
                # Alignment: mean squared distance between positive pairs
                epoch_align += (z1 - z2).pow(2).sum(dim=1).mean().item()
                # Uniformity: log of avg exp(-2 * sq_dist) over all pairs
                sq_dist = torch.cdist(z1, z1).pow(2)
                epoch_uniform += torch.log(torch.exp(-2 * sq_dist).mean()).item()
                # Collapse: avg cosine similarity across batch (z is L2-normed)
                B = z1.size(0)
                cos_sim = z1 @ z1.T
                mask = ~torch.eye(B, dtype=torch.bool, device=z1.device)
                epoch_collapse += cos_sim[mask].mean().item()

        avg_loss = epoch_loss / max(n_steps, 1)
        avg_align = epoch_align / max(n_steps, 1)
        avg_uniform = epoch_uniform / max(n_steps, 1)
        avg_collapse = epoch_collapse / max(n_steps, 1)
        history.append({
            "epoch": epoch + 1,
            "lr": float(lr),
            "train_nce": float(avg_loss),
            "alignment": float(avg_align),
            "uniformity": float(avg_uniform),
            "collapse": float(avg_collapse),
        })

        print(f"Epoch {epoch+1:3d}/{EPOCHS} | lr={lr:.6f} | train_nce={avg_loss:.4f} "
              f"| align={avg_align:.4f} | uniform={avg_uniform:.4f} | collapse={avg_collapse:.4f}")

        if (epoch + 1) % EVAL_EVERY == 0:
            val_nce = eval_contrastive_loss(model, ds_val, DEVICE)
            val_auroc, val_auprc = eval_linear_probe(model, ds_train, ds_val, DEVICE)
            eval_row = {
                "epoch": epoch + 1,
                "val_nce": float(val_nce),
                "val_auroc": float(val_auroc),
                "val_auprc": float(val_auprc),
            }
            eval_history.append(eval_row)
            print(f"  [eval] val_nce={val_nce:.4f} | val_auroc={val_auroc:.4f} | val_auprc={val_auprc:.4f}")

            score = (val_auprc, val_auroc)
            if score > best_score:
                best_score = score
                best_eval = eval_row
                # Save only encoder weights (lstm), not projection head
                state = {
                    "lstm": model.lstm.state_dict(),
                    "epoch": epoch + 1,
                    "val_auroc": float(val_auroc),
                    "val_auprc": float(val_auprc),
                }
                torch.save(state, CKPT_DIR / "encoder_best.pt")
                print(
                    f"  [ckpt] Saved encoder "
                    f"(val_auprc={val_auprc:.4f}, val_auroc={val_auroc:.4f})"
                )

    if best_eval is None:
        raise RuntimeError("No evaluation step was run; check EVAL_EVERY and EPOCHS.")

    print(
        f"\nDone. Best val metrics: "
        f"AuPRC={best_eval['val_auprc']:.4f}, AuROC={best_eval['val_auroc']:.4f} "
        f"(epoch {best_eval['epoch']})"
    )
    print(f"Checkpoint: {CKPT_DIR / 'encoder_best.pt'}")

    # Export embeddings from best encoder for downstream use
    export_manifest = export_embeddings()

    summary = {
        "task": "Q3.1 contrastive pretraining (InfoNCE)",
        "run": run_metadata(SEED),
        "device": str(DEVICE),
        "data": {
            "train_file": str(DATA_DIR / _DATA["train_file"]),
            "val_file": str(DATA_DIR / _DATA["val_file"]),
            "train_patients": len(ds_train),
            "val_patients": len(ds_val),
        },
        "model": {
            "input_dim": _M["input_dim"],
            "hidden_dim": _M["hidden_dim"],
            "num_layers": _M["num_layers"],
            "dropout": _M["dropout"],
            "recency_strength": _M["recency_strength"],
            "proj_dim": _M["proj_dim"],
        },
        "pretrain_hyperparameters": {
            "tau": TAU,
            "lr": LR,
            "weight_decay": WD,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "warmup_epochs": WARMUP_EPOCHS,
            "eval_every": EVAL_EVERY,
            "num_workers": NUM_WORKERS,
        },
        "selection_rule": "best validation linear-probe AuPRC, tie-break by AuROC",
        "best_eval": best_eval,
        "checkpoint": str(CKPT_DIR / "encoder_best.pt"),
        "train_history": history,
        "eval_history": eval_history,
        "embedding_exports": export_manifest,
    }
    summary_path = RESULTS_DIR / "summary.json"
    save_json(summary_path, summary)
    print(f"Saved summary: {summary_path}")


def export_embeddings():
    """Load best encoder and save embeddings for train/val/test as .npz files."""
    print("\nExporting embeddings ...")

    ckpt = torch.load(CKPT_DIR / "encoder_best.pt", map_location=DEVICE)
    model = LSTMEncoder(
        input_dim=_M["input_dim"], hidden_dim=_M["hidden_dim"],
        num_layers=_M["num_layers"], dropout=_M["dropout"],
        recency_strength=_M["recency_strength"], proj_dim=_M["proj_dim"],
    ).to(DEVICE)
    model.lstm.load_state_dict(ckpt["lstm"])
    model.eval()

    manifest = {}

    for split, file_key, emb_key in [
        ("train", "train_file", "emb_train"),
        ("val",   "val_file",   "emb_val"),
        ("test",  "test_file",  "emb_test"),
    ]:
        ds = ICUPatientDataset(DATA_DIR / _DATA[file_key])
        X, y = extract_embeddings(model, ds, DEVICE)
        patient_ids = np.asarray(ds.patient_ids)
        out = CKPT_DIR / _DATA[emb_key]
        np.savez(out, embeddings=X, labels=y, patient_ids=patient_ids)
        print(f"  {split}: {X.shape} -> {out}")

        manifest[split] = {
            "shape": [int(X.shape[0]), int(X.shape[1])],
            "labelled_patients": int(np.sum(~np.isnan(y))),
            "output_file": str(out),
            "contains_patient_ids": True,
        }

    return manifest


if __name__ == "__main__":
    train()
