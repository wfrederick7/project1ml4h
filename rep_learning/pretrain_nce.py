import os
import random
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CFG = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
_DATA = _CFG["data"]
_RL = _CFG["representation_learning"]
_PT = _RL["pretrain"]
_M  = _RL["model"]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = _RL["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.path.expanduser(_CFG["paths"]["data_derived"]))
CKPT_DIR = Path(os.path.expanduser(_CFG["paths"]["checkpoints"]))
CKPT_DIR.mkdir(parents=True, exist_ok=True)

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
    sim = (z1 @ z2.T) / tau  # (B, B)
    labels = torch.arange(B, device=z1.device)
    return nn.functional.cross_entropy(sim, labels)


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
                        collate_fn=contrastive_collate, drop_last=True)
    total_loss, n_batches = 0.0, 0
    for v1, v2 in loader:
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
        drop_last=True,
        num_workers=NUM_WORKERS,
    )

    model = LSTMEncoder(
        input_dim=_M["input_dim"],
        hidden_dim=_M["hidden_dim"],
        num_layers=_M["num_layers"],
        proj_dim=_M["proj_dim"],
    ).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    best_auroc = 0.0

    for epoch in range(EPOCHS):
        model.train()

        # Linear warmup
        lr = get_lr(epoch, LR)
        for pg in optimiser.param_groups:
            pg["lr"] = lr

        epoch_loss, n_steps = 0.0, 0
        for v1, v2 in train_loader:
            v1, v2 = v1.to(DEVICE), v2.to(DEVICE)

            z1 = model.project(v1)
            z2 = model.project(v2)
            loss = infonce_loss(z1, z2)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()
            n_steps += 1

        avg_loss = epoch_loss / max(n_steps, 1)
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | lr={lr:.6f} | train_nce={avg_loss:.4f}")

        if (epoch + 1) % EVAL_EVERY == 0:
            val_nce = eval_contrastive_loss(model, ds_val, DEVICE)
            val_auroc = eval_linear_probe(model, ds_train, ds_val, DEVICE)
            print(f"  [eval] val_nce={val_nce:.4f} | val_auroc={val_auroc:.4f}")

            if val_auroc > best_auroc:
                best_auroc = val_auroc
                # Save only encoder weights (lstm + fc), not projection head
                state = {
                    "lstm": model.lstm.state_dict(),
                    "fc":   model.fc.state_dict(),
                }
                torch.save(state, CKPT_DIR / "encoder_best.pt")
                print(f"  [ckpt] Saved encoder (val_auroc={best_auroc:.4f})")

    print(f"\nDone. Best val AuROC: {best_auroc:.4f}")
    print(f"Checkpoint: {CKPT_DIR / 'encoder_best.pt'}")

    # Export embeddings from best encoder for downstream use
    export_embeddings()


def export_embeddings():
    """Load best encoder and save embeddings for train/val/test as .npz files."""
    print("\nExporting embeddings ...")

    ckpt = torch.load(CKPT_DIR / "encoder_best.pt", map_location=DEVICE)
    model = LSTMEncoder(
        input_dim=_M["input_dim"], hidden_dim=_M["hidden_dim"],
        num_layers=_M["num_layers"], proj_dim=_M["proj_dim"],
    ).to(DEVICE)
    model.lstm.load_state_dict(ckpt["lstm"])
    model.fc.load_state_dict(ckpt["fc"])
    model.eval()

    for split, file_key, emb_key in [
        ("train", "train_file", "emb_train"),
        ("val",   "val_file",   "emb_val"),
        ("test",  "test_file",  "emb_test"),
    ]:
        ds = ICUPatientDataset(DATA_DIR / _DATA[file_key])
        X, y = extract_embeddings(model, ds, DEVICE)
        out = CKPT_DIR / _DATA[emb_key]
        np.savez(out, embeddings=X, labels=y)
        print(f"  {split}: {X.shape} -> {out}")


if __name__ == "__main__":
    train()
