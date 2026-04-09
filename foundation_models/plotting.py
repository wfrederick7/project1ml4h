import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
import argparse
import random
import os

# -----------------------------
# Config
# -----------------------------
SEED = 42
EMB_DIR = "."  # same folder as your files


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


seed_everything(SEED)


# -----------------------------
# Load data
# -----------------------------
def load_data():
    X_train = np.load("chronos_train_embeddings.npy")
    y_train = np.load("chronos_train_labels.npy")

    X_test = np.load("chronos_test_embeddings.npy")
    y_test = np.load("chronos_test_labels.npy")

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    return X, y.astype(int)


# -----------------------------
# Aggregate embeddings to 2D
# -----------------------------
def aggregate_embeddings(X, method="mean"):
    """
    Convert embeddings to shape [n_samples, n_features].

    If X is already 2D, return as-is.
    If X is 3D, aggregate over axis 1.
    """
    X = np.asarray(X)

    if X.ndim == 2:
        return X

    if X.ndim == 3:
        if method == "mean":
            return X.mean(axis=1)
        elif method == "max":
            return X.max(axis=1)
        elif method == "flatten":
            return X.reshape(X.shape[0], -1)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    raise ValueError(f"Expected X to have 2 or 3 dimensions, got shape {X.shape}")


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(X, y):
    km = KMeans(n_clusters=2, random_state=SEED, n_init=10)
    km_labels = km.fit_predict(X)

    return {
        "silhouette_gt": float(silhouette_score(X, y)),
        "silhouette_kmeans": float(silhouette_score(X, km_labels)),
        "ari": float(adjusted_rand_score(y, km_labels)),
        "nmi": float(normalized_mutual_info_score(y, km_labels)),
    }


# -----------------------------
# Plot
# -----------------------------
def plot_tsne(X, y, out_name="t_sne_embeddings_LLM.png"):
    print("Running t-SNE...")
    coords = TSNE(n_components=2, perplexity=30, random_state=SEED).fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))

    colors = {0: "#3b82f6", 1: "#ef4444"}
    names = {0: "Alive", 1: "Dead"}

    for c in [0, 1]:
        mask = y == c
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=colors[c],
            label=names[c],
            s=8,
            alpha=0.5,
            edgecolors="none",
        )

    ax.set_title("t-SNE Embeddings — Mortality Label")
    ax.legend(markerscale=3)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_name, dpi=200, bbox_inches="tight")
    plt.show()


# -----------------------------
# Main
# -----------------------------
def main(score_only=False, aggregation="mean"):
    X_raw, y = load_data()

    print(f"\nRaw X shape: {X_raw.shape}")
    print(f"y shape: {y.shape}")

    X = aggregate_embeddings(X_raw, method=aggregation)

    print(f"Aggregated X shape: {X.shape}")
    print(f"Loaded {len(y)} patients")
    print(f"Dead: {int(y.sum())}, Alive: {int(len(y)-y.sum())}")

    metrics = compute_metrics(X, y)

    print("\n=== Metrics ===")
    print(f"Aggregation                = {aggregation}")
    print(f"Silhouette (GT labels)     = {metrics['silhouette_gt']:.4f}")
    print(f"Silhouette (KMeans labels) = {metrics['silhouette_kmeans']:.4f}")
    print(f"ARI (KMeans vs GT)         = {metrics['ari']:.4f}")
    print(f"NMI (KMeans vs GT)         = {metrics['nmi']:.4f}")

    if score_only:
        return

    plot_tsne(X, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument(
        "--aggregation",
        choices=["mean", "max", "flatten"],
        default="mean",
        help="How to convert 3D embeddings to 2D patient-level vectors.",
    )
    args = parser.parse_args()

    main(score_only=args.score_only, aggregation=args.aggregation)