import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CFG = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
_DATA = _CFG["data"]
_PT = _CFG["representation_learning"]["pretrain"]

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
STATIC_COLS = _DATA["static_cols"]
DYNAMIC_COLS = _DATA["dynamic_cols"]
FEATURE_COLS = STATIC_COLS + DYNAMIC_COLS

ID_COL     = _DATA["id_col"]
TIME_COL   = _DATA["time_col"]
TARGET_COL = _DATA["target_col"]

# ---------------------------------------------------------------------------
# Augmentation parameters
# ---------------------------------------------------------------------------
CROP_LEN       = _PT["crop_len"]
NOISE_STD      = _PT["noise_std"]
CHANNEL_DROP_P = _PT["channel_drop_p"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ICUPatientDataset(Dataset):
    """One item per patient: (n_hours, n_features) feature tensor and scalar label."""

    def __init__(self, parquet_path: str | Path):
        df = pd.read_parquet(parquet_path)
        df = df.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

        self.patient_ids = df[ID_COL].unique()
        self.features = {}
        self.labels = {}

        for pid in self.patient_ids:
            pat = df[df[ID_COL] == pid]
            feat = pat[FEATURE_COLS].values.astype(np.float32)
            self.features[pid] = torch.from_numpy(feat)
            label_vals = pat[TARGET_COL].values
            self.labels[pid] = float(label_vals[0]) if not np.isnan(label_vals[0]) else np.nan

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        return self.features[pid], self.labels[pid]


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def augment(x: torch.Tensor) -> torch.Tensor:
    """
    x: (n_hours, n_features) float tensor.
    Returns: (CROP_LEN, n_features) augmented tensor.
    """
    seq_len = x.shape[0]
    max_start = seq_len - CROP_LEN
    start = random.randint(0, max_start)
    crop = x[start : start + CROP_LEN].clone()

    # Gaussian noise
    crop = crop + torch.randn_like(crop) * NOISE_STD

    # Channel dropout: zero out entire channels independently
    mask = (torch.rand(crop.shape[1]) > CHANNEL_DROP_P).float()
    crop = crop * mask.unsqueeze(0)  # broadcast over time

    return crop


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------
def contrastive_collate(batch):
    """
    batch: list of (features, label) tuples.
    Returns two independently augmented views: (B, CROP_LEN, n_features) each.
    """
    v1, v2 = [], []
    for feat, _label in batch:
        v1.append(augment(feat))
        v2.append(augment(feat))
    return torch.stack(v1), torch.stack(v2)
