"""
Q3.3 — Embedding visualisation and clustering metrics.

1. Load precomputed embeddings for train/val/test.
2. Produce t-SNE 2D scatter plots coloured by mortality label.
3. Compute ARI and NMI (KMeans k=2 vs true labels) as quantitative measures.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CFG = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
_DATA = _CFG["data"]
_RL = _CFG["representation_learning"]

SEED = _RL["seed"]
np.random.seed(SEED)

CKPT_DIR = Path(os.path.expanduser(_CFG["paths"]["checkpoints"]))
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


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

    # --- Dimensionality reduction ---
    print("\nRunning t-SNE (perplexity=30) ...")
    tsne30 = TSNE(n_components=2, random_state=SEED, perplexity=30).fit_transform(X_all)

    print("Running t-SNE (perplexity=50) ...")
    tsne50 = TSNE(n_components=2, random_state=SEED, perplexity=50).fit_transform(X_all)

    # --- Clustering metrics ---
    km_labels = KMeans(n_clusters=2, random_state=SEED, n_init=10).fit_predict(X_all)
    ari = adjusted_rand_score(y_all, km_labels)
    nmi = normalized_mutual_info_score(y_all, km_labels)
    print(f"\nKMeans (k=2) vs true labels:")
    print(f"  ARI = {ari:.4f}")
    print(f"  NMI = {nmi:.4f}")

    # --- Figures ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    scatter_2d(tsne30, y_all, f"t-SNE perp=30  (ARI={ari:.3f}, NMI={nmi:.3f})", axes[0])
    scatter_2d(tsne50, y_all, f"t-SNE perp=50  (ARI={ari:.3f}, NMI={nmi:.3f})", axes[1])

    fig.suptitle("Pretrained Encoder Embeddings — Mortality Label", fontsize=14)
    fig.tight_layout()
    out_path = FIG_DIR / "embedding_visualisation.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
