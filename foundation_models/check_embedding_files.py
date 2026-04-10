#!/usr/bin/env python3

import os
import sys
import numpy as np
from collections import Counter


EMB_FILE = "test_embeddings.npy"
LABEL_FILE = "test_labels.npy"
TSNE_PERPLEXITY = 30
KMEANS_N_CLUSTERS = 2


def ok(msg):
    print(f"[OK]   {msg}")


def warn(msg):
    print(f"[WARN] {msg}")


def fail(msg):
    print(f"[FAIL] {msg}")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def try_load(path):
    try:
        arr = np.load(path, allow_pickle=False)
        return arr, None
    except Exception as e:
        return None, e


def main():
    failed = False

    section("1) FILE EXISTENCE")

    for path in [EMB_FILE, LABEL_FILE]:
        if os.path.exists(path):
            ok(f"Found file: {path}")
        else:
            fail(f"Missing file: {path}")
            failed = True

    if failed:
        print("\nStopping early because required files are missing.")
        sys.exit(1)

    section("2) LOAD FILES")

    X, err = try_load(EMB_FILE)
    if err is not None:
        fail(f"Could not load {EMB_FILE}: {repr(err)}")
        sys.exit(1)
    ok(f"Loaded {EMB_FILE}")

    y, err = try_load(LABEL_FILE)
    if err is not None:
        fail(f"Could not load {LABEL_FILE}: {repr(err)}")
        sys.exit(1)
    ok(f"Loaded {LABEL_FILE}")

    section("3) BASIC ARRAY INFO")

    print(f"{EMB_FILE}:")
    print(f"  shape   = {X.shape}")
    print(f"  dtype   = {X.dtype}")
    print(f"  ndim    = {X.ndim}")
    print(f"  size    = {X.size}")

    print(f"\n{LABEL_FILE}:")
    print(f"  shape   = {y.shape}")
    print(f"  dtype   = {y.dtype}")
    print(f"  ndim    = {y.ndim}")
    print(f"  size    = {y.size}")

    section("4) EMBEDDINGS CHECKS")

    if not isinstance(X, np.ndarray):
        fail("Embeddings file did not load as a NumPy array.")
        failed = True
    else:
        ok("Embeddings loaded as NumPy array.")

    if X.ndim == 2:
        ok(f"Embeddings are 2D: {X.shape}")
    else:
        fail(f"Embeddings must be 2D for your current script, got shape {X.shape}")
        failed = True

    if X.size > 0:
        ok("Embeddings array is non-empty.")
    else:
        fail("Embeddings array is empty.")
        failed = True

    if np.issubdtype(X.dtype, np.number):
        ok(f"Embeddings dtype is numeric: {X.dtype}")
    else:
        fail(f"Embeddings dtype must be numeric, got {X.dtype}")
        failed = True

    try:
        finite_mask = np.isfinite(X)
        if finite_mask.all():
            ok("Embeddings contain no NaN or inf values.")
        else:
            n_bad = (~finite_mask).sum()
            fail(f"Embeddings contain {n_bad} non-finite values (NaN or inf).")
            failed = True
    except Exception as e:
        fail(f"Could not check finiteness of embeddings: {repr(e)}")
        failed = True

    if X.ndim == 2 and X.shape[0] > 0 and X.shape[1] > 0:
        ok(f"Embeddings have {X.shape[0]} samples and {X.shape[1]} features.")
    elif X.ndim == 2:
        fail(f"Embeddings have invalid 2D shape: {X.shape}")
        failed = True

    section("5) LABEL CHECKS")

    if not isinstance(y, np.ndarray):
        fail("Labels file did not load as a NumPy array.")
        failed = True
    else:
        ok("Labels loaded as NumPy array.")

    if y.ndim == 1:
        ok(f"Labels are 1D: {y.shape}")
    else:
        fail(f"Labels must be 1D for your current script, got shape {y.shape}")
        failed = True

    if y.size > 0:
        ok("Labels array is non-empty.")
    else:
        fail("Labels array is empty.")
        failed = True

    try:
        y_int = y.astype(int)
        ok("Labels can be cast to int.")
    except Exception as e:
        fail(f"Labels cannot be cast to int: {repr(e)}")
        failed = True
        y_int = None

    if X.ndim == 2 and y.ndim == 1:
        if X.shape[0] == y.shape[0]:
            ok(f"Sample count matches: {X.shape[0]} samples")
        else:
            fail(
                f"Mismatch in number of samples: embeddings have {X.shape[0]}, "
                f"labels have {y.shape[0]}"
            )
            failed = True

    if y_int is not None and y_int.size > 0:
        unique, counts = np.unique(y_int, return_counts=True)
        label_counts = dict(zip(unique.tolist(), counts.tolist()))
        print(f"Label counts: {label_counts}")

        if len(unique) >= 2:
            ok(f"Labels contain at least 2 unique classes: {unique.tolist()}")
        else:
            fail(f"Labels must contain at least 2 unique classes, got {unique.tolist()}")
            failed = True

        if set(unique.tolist()) == {0, 1}:
            ok("Labels are exactly binary {0, 1}, matching your plot labels Alive/Dead.")
        else:
            warn(
                f"Labels are not exactly {{0,1}}; found {unique.tolist()}. "
                "Your plotting code assumes 0=Alive and 1=Dead."
            )

        if len(unique) < KMEANS_N_CLUSTERS:
            warn(
                f"Ground-truth labels have fewer than {KMEANS_N_CLUSTERS} classes. "
                "ARI/NMI may be uninformative."
            )

    section("6) CHECKS SPECIFIC TO YOUR CURRENT SCRIPT")

    if X.ndim == 2 and X.shape[0] >= KMEANS_N_CLUSTERS:
        ok(f"KMeans with n_clusters={KMEANS_N_CLUSTERS} is at least shape-feasible.")
    elif X.ndim == 2:
        fail(
            f"KMeans with n_clusters={KMEANS_N_CLUSTERS} is not feasible because "
            f"n_samples={X.shape[0]}"
        )
        failed = True

    if X.ndim == 2:
        n_samples = X.shape[0]
        if n_samples > TSNE_PERPLEXITY:
            ok(
                f"t-SNE perplexity={TSNE_PERPLEXITY} is valid because "
                f"n_samples={n_samples} > perplexity"
            )
        else:
            fail(
                f"t-SNE perplexity={TSNE_PERPLEXITY} will fail because "
                f"n_samples={n_samples} <= perplexity"
            )
            failed = True

    section("7) FINAL VERDICT")

    if failed:
        fail("One or more required conditions are not fulfilled.")
        print("\nYour current plotting/evaluation script is NOT guaranteed to run correctly.")
        sys.exit(1)
    else:
        ok("All critical conditions for your current script are fulfilled.")
        print("\nYour current plotting/evaluation script is likely to run.")
        sys.exit(0)


if __name__ == "__main__":
    main()
