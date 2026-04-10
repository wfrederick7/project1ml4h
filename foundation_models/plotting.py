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
    X_test = np.load("test_embeddings.npy")
    y_test = np.load("test_labels.npy")

    print(f"Embeddings shape: {X_test.shape}")
    print(f"Labels shape:     {y_test.shape}")
    return X_test, y_test.astype(int)



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
    coords = TSNE(n_components=2, perplexity=30, random_state=SEED, init="pca",learning_rate="auto", metric="euclidean").fit_transform(X)
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
def main():
    X_test, y_test = load_data()

    print(f"X shape: {X_test.shape}")
    print(f"y shape: {y_test.shape}")


    metrics = compute_metrics(X_test, y_test)

    print("\n=== Metrics ===")
    print(f"Silhouette (GT labels)     = {metrics['silhouette_gt']:.4f}")
    print(f"Silhouette (KMeans labels) = {metrics['silhouette_kmeans']:.4f}")
    print(f"ARI (KMeans vs GT)         = {metrics['ari']:.4f}")
    print(f"NMI (KMeans vs GT)         = {metrics['nmi']:.4f}")

    plot_tsne(X_test, y_test)


if __name__ == "__main__":
    main()