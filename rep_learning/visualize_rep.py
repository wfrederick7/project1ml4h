"""
Q3.3 — Embedding visualisation and clustering metrics.

1. Load precomputed embeddings for train/val/test.
2. Produce t-SNE and UMAP 2D scatter plots coloured by mortality label.
3. Compute ARI, Silhouette Score on real embeddings and a random baseline.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: umap-learn not installed. UMAP plots will be skipped. "
          "Install with: pip install umap-learn")

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

SEED = _RL["seed"]
seed_everything(SEED)

CKPT_DIR = Path(os.path.expanduser(_CFG["paths"]["checkpoints"]))
FIG_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "rep_learning" / "results" / "q3_3_visualization"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_embeddings(split_key):
    data = np.load(CKPT_DIR / _DATA[split_key])
    return data["embeddings"], data["labels"]


def filter_labelled(X, y):
    mask = ~np.isnan(y)
    return X[mask], y[mask].astype(int)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def scatter_2d(coords, labels, title, ax):
    colours = {0: "#3b82f6", 1: "#ef4444"}
    names = {0: "Survived", 1: "Died"}
    for c in [0, 1]:
        m = labels == c
        ax.scatter(coords[m, 0], coords[m, 1],
                   c=colours[c], label=names[c], s=8, alpha=0.5, edgecolors="none")
    ax.set_title(title)
    ax.legend(markerscale=3, frameon=True)
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Clustering metrics
# ---------------------------------------------------------------------------
def compute_clustering_metrics(X, y_true):
    """Compute KMeans-based ARI, NMI, and Silhouette Score."""
    km_labels = KMeans(n_clusters=2, random_state=SEED, n_init=10).fit_predict(X)
    ari = adjusted_rand_score(y_true, km_labels)
    nmi = normalized_mutual_info_score(y_true, km_labels)
    sil = silhouette_score(X, y_true)
    return {
        "ari": float(ari),
        "nmi": float(nmi),
        "silhouette": float(sil),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading precomputed embeddings ...")

    embs, lbls = [], []
    for name, key in [("Train", "emb_train"), ("Val", "emb_val"), ("Test", "emb_test")]:
        X, y = load_embeddings(key)
        X, y = filter_labelled(X, y)
        embs.append(X)
        lbls.append(y)
        print(f"  {name}: {len(y)} labelled patients")

    X_all = np.concatenate(embs)
    y_all = np.concatenate(lbls)
    print(f"\nTotal labelled: {len(y_all)}  (died={y_all.sum():.0f}, "
          f"survived={len(y_all) - y_all.sum():.0f})")

    # --- Clustering metrics on full-dimensional embeddings ---
    print("\nClustering metrics (full-dimensional embeddings):")
    metrics_real = compute_clustering_metrics(X_all, y_all)
    print(f"  ARI        = {metrics_real['ari']:.4f}")
    print(f"  NMI        = {metrics_real['nmi']:.4f}")
    print(f"  Silhouette = {metrics_real['silhouette']:.4f}")

    # --- Random baseline: random 2D projection ---
    rng = np.random.RandomState(SEED)
    W_rand = rng.randn(X_all.shape[1], 2)
    X_rand_2d = X_all @ W_rand

    metrics_rand = compute_clustering_metrics(X_rand_2d, y_all)
    print(f"\nRandom baseline (random 2D projection):")
    print(f"  ARI        = {metrics_rand['ari']:.4f}")
    print(f"  NMI        = {metrics_rand['nmi']:.4f}")
    print(f"  Silhouette = {metrics_rand['silhouette']:.4f}")

    # --- Dimensionality reduction ---
    print("\nRunning t-SNE (perplexity=30) ...")
    tsne_coords = TSNE(n_components=2, random_state=SEED, perplexity=30).fit_transform(X_all)
    metrics_tsne = compute_clustering_metrics(tsne_coords, y_all)

    umap_coords = None
    metrics_umap = None
    if HAS_UMAP:
        print("Running UMAP (n_neighbors=15) ...")
        umap_coords = umap.UMAP(
            n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1,
        ).fit_transform(X_all)
        metrics_umap = compute_clustering_metrics(umap_coords, y_all)

    # --- Figures ---
    n_cols = 2 if umap_coords is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    scatter_2d(
        tsne_coords, y_all,
        f"t-SNE (perp=30)\nARI={metrics_tsne['ari']:.3f}  Sil={metrics_tsne['silhouette']:.3f}",
        axes[0],
    )

    if umap_coords is not None:
        scatter_2d(
            umap_coords, y_all,
            f"UMAP (nn=15)\nARI={metrics_umap['ari']:.3f}  Sil={metrics_umap['silhouette']:.3f}",
            axes[1],
        )

    fig.suptitle("Pretrained Encoder Embeddings — Mortality Label", fontsize=14)
    fig.tight_layout()
    out_path = FIG_DIR / "embedding_visualisation.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close(fig)

    # --- Random baseline figure ---
    fig_rand, ax_rand = plt.subplots(1, 1, figsize=(7, 6))
    scatter_2d(
        X_rand_2d, y_all,
        f"Random 2D Projection (baseline)\n"
        f"ARI={metrics_rand['ari']:.3f}  Sil={metrics_rand['silhouette']:.3f}",
        ax_rand,
    )
    fig_rand.tight_layout()
    out_rand = FIG_DIR / "embedding_random_baseline.png"
    fig_rand.savefig(out_rand, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_rand}")
    plt.close(fig_rand)

    summary = {
        "task": "Q3.3 embedding visualisation and clustering metrics",
        "run": run_metadata(SEED),
        "input_files": {
            "train": str(CKPT_DIR / _DATA["emb_train"]),
            "val": str(CKPT_DIR / _DATA["emb_val"]),
            "test": str(CKPT_DIR / _DATA["emb_test"]),
        },
        "labelled_counts": {
            "total": int(len(y_all)),
            "died": int(y_all.sum()),
            "survived": int(len(y_all) - y_all.sum()),
        },
        "metrics": {
            "full_dim_embeddings": metrics_real,
            "random_2d_projection_baseline": metrics_rand,
            "tsne_2d": metrics_tsne,
            "umap_2d": metrics_umap,
        },
        "figures": {
            "embedding_visualisation": str(out_path),
            "embedding_random_baseline": str(out_rand),
        },
        "umap_available": HAS_UMAP,
    }
    summary_path = RESULTS_DIR / "summary.json"
    save_json(summary_path, summary)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
