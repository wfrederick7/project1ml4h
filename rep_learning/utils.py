import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_metadata(seed: int) -> dict:
    """Minimal run metadata for report tracking."""
    return {
        "timestamp_utc": utc_now_iso(),
        "seed": int(seed),
    }


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_builtin(obj.tolist())
    if isinstance(obj, np.generic):
        return _to_builtin(obj.item())
    if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
        return None
    return obj


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = _to_builtin(payload)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(safe_payload, f, indent=2)
    tmp_path.replace(path)
