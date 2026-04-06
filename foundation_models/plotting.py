from pathlib import Path
import sys

import matplotlib
import numpy as np
from sklearn.manifold import TSNE

matplotlib.use("Agg")
from matplotlib import pyplot as plt

PROJECT_LIBS = Path.home() / "project1ml4h" / "libs"
if PROJECT_LIBS.exists():
    sys.path.insert(0, str(PROJECT_LIBS))

try:
    from umap import UMAP
except Exception:
    UMAP = None

BASE_DIR = Path(__file__).resolve().parent
TEST_EMBEDDINGS_PATH = BASE_DIR / "test_embeddings.npy"
TEST_LABELS_PATH = BASE_DIR / "test_labels.npy"
CLUSTER_LABELS_PATH = BASE_DIR / "cluster_labels.npy"


def load_required_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return np.load(path)


def plot_embeddings(reduced: np.ndarray, cluster_labels: np.ndarray, true_labels: np.ndarray, method: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{method} Embeddings", fontsize=14)

    scatter1 = axes[0].scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=cluster_labels,
        cmap="tab10",
        alpha=0.6,
        s=10,
    )
    axes[0].set_title("Colored by K-Means Cluster")
    axes[0].set_xlabel(f"{method} 1")
    axes[0].set_ylabel(f"{method} 2")
    plt.colorbar(scatter1, ax=axes[0], label="Cluster")

    scatter2 = axes[1].scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=true_labels,
        cmap="RdYlGn_r",
        alpha=0.6,
        s=10,
    )
    axes[1].set_title("Colored by Mortality Label")
    axes[1].set_xlabel(f"{method} 1")
    axes[1].set_ylabel(f"{method} 2")
    cbar = plt.colorbar(scatter2, ax=axes[1], label="Label")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Alive", "Dead"])

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    test_embeddings = load_required_array(TEST_EMBEDDINGS_PATH)
    test_labels = load_required_array(TEST_LABELS_PATH)
    cluster_labels = load_required_array(CLUSTER_LABELS_PATH)

    if test_embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {test_embeddings.shape}")

    if not (len(test_embeddings) == len(test_labels) == len(cluster_labels)):
        raise ValueError(
            "Input length mismatch: "
            f"embeddings={len(test_embeddings)}, labels={len(test_labels)}, clusters={len(cluster_labels)}"
        )

    print(f"Embeddings shape: {test_embeddings.shape}", flush=True)
    print(f"Labels shape:     {test_labels.shape}", flush=True)
    print(f"Cluster labels:   {cluster_labels.shape}", flush=True)

    perplexity = min(30, max(5, len(test_embeddings) - 1))
    print(f"Running t-SNE with perplexity={perplexity}...", flush=True)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    tsne_reduced = tsne.fit_transform(test_embeddings)
    print(f"t-SNE output shape: {tsne_reduced.shape}", flush=True)
    plot_embeddings(
        tsne_reduced,
        cluster_labels,
        test_labels,
        method="t-SNE",
        output_path=BASE_DIR / "t_sne_embeddings.png",
    )

    if UMAP is None:
        print("UMAP is not available in this environment. Skipping UMAP plot.", flush=True)
    else:
        print("Running UMAP...", flush=True)
        reducer = UMAP(n_components=2, random_state=42)
        umap_reduced = reducer.fit_transform(test_embeddings)
        print(f"UMAP output shape: {umap_reduced.shape}", flush=True)
        plot_embeddings(
            umap_reduced,
            cluster_labels,
            test_labels,
            method="UMAP",
            output_path=BASE_DIR / "umap_embeddings.png",
        )

    print("Plot generation completed.", flush=True)


if __name__ == "__main__":
    main()