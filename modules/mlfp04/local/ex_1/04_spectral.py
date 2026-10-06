# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 1.4: Spectral Clustering via the Graph Laplacian
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build an RBF affinity graph between points and its normalised Laplacian
#   - Embed points using the SMALLEST-k eigenvectors of the graph Laplacian
#   - Cluster in the spectral embedding space with K-means
#   - Recognise non-convex cluster shapes that K-means cannot separate, and
#     why Euclidean silhouette cannot be the judge of that
#
# PREREQUISITES: 01_kmeans.py (K-means as the embedding-space learner).
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — affinity, Laplacian, eigenvectors, embedding
#   2. Build — hand-built affinity → Laplacian → eigenvectors on two moons,
#      then sklearn spectral on a customer subsample for a few K values
#   3. Train — score partitions and pick the best K
#   4. Visualise — silhouette vs K
#   5. Apply — Singapore rail-network community detection, $ impact
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time

import numpy as np
from dotenv import load_dotenv
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score, silhouette_score

from kailash_ml import ModelVisualizer

from shared.mlfp04.ex_1 import (
    RANDOM_STATE,
    load_customers,
    out_path,
    setup_engines,
    standardise,
    subsample,
    teardown_engines,
    track_run,
)

load_dotenv()

# ── Kailash-ML ExperimentTracker — every clustering run logs here ─────────
tracker, exp_name = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Graph Laplacian Embedding
# ════════════════════════════════════════════════════════════════════════
# A_ij = exp(-||x_i - x_j||² / 2σ²)    (RBF affinity)
# D_ii = Σ_j A_ij                      (degree matrix)
# L = D - A                            (graph Laplacian)
# L_sym = I - D^-½ A D^-½              (normalised Laplacian)
# The SMALLEST k eigenvectors of L embed the graph in R^k; points that
# are connected through dense paths land near each other there. Run
# K-means on the embedding. Price: O(n^3). Small-to-medium data only.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Spectral by hand on two moons, then on customers
# ════════════════════════════════════════════════════════════════════════
# 2a. Build the pipeline yourself on data whose true groups we KNOW: two
# interlocking half-moons. They are non-convex, so K-means (which draws a
# straight boundary between two centroids) cannot separate them.

X_moons, y_moons = make_moons(n_samples=400, noise=0.06, random_state=RANDOM_STATE)
GAMMA_MOONS = 20.0


def spectral_embedding(
    X: np.ndarray, k: int, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """RBF affinity → normalised Laplacian → k smallest eigenvectors.

    Returns (U, eigenvalues) where U is the row-normalised n×k embedding
    (Ng-Jordan-Weiss) and eigenvalues are the k+1 smallest of L_sym.
    """
    sq_dists = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1)
    # TODO: 1. RBF affinity A = exp(-gamma * squared distance); zero the diagonal.
    A = ____
    np.fill_diagonal(A, 0.0)
    # TODO: 2. degree vector d (row sums of A) and 3. the normalised Laplacian
    # L_sym = I - D^-½ A D^-½ (broadcast 1/sqrt(d) over rows and columns).
    d = ____
    d_inv_sqrt = 1.0 / np.sqrt(d)
    L_sym = ____
    # TODO: eigendecompose the symmetric L_sym (ascending eigenvalues) and keep
    # the k eigenvectors with the SMALLEST eigenvalues.
    # Hint: np.linalg.eigh returns (eigenvalues, eigenvectors) in ascending order
    eigvals, eigvecs = ____
    U = ____
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    return U, eigvals[: k + 1]


U_moons, moon_eigvals = spectral_embedding(X_moons, k=2, gamma=GAMMA_MOONS)
# TODO: run KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10) on the
# spectral embedding U_moons, and separately on the raw X_moons.
moons_spectral = ____
moons_kmeans = ____
ari_spectral = adjusted_rand_score(y_moons, moons_spectral)
ari_kmeans = adjusted_rand_score(y_moons, moons_kmeans)
sil_moons_spectral = silhouette_score(X_moons, moons_spectral)
sil_moons_kmeans = silhouette_score(X_moons, moons_kmeans)

print("=" * 70)
print("  Spectral by hand on two moons (true labels known)")
print("=" * 70)
print(f"  Smallest Laplacian eigenvalues: {np.round(moon_eigvals, 5)}")
print("  (a near-zero eigenvalue per well-separated component, then a gap)")
print(f"  {'Method':<22} {'ARI vs truth':>13} {'Euclid. silhouette':>19}")
print(f"  {'Spectral (by hand)':<22} {ari_spectral:>13.3f} {sil_moons_spectral:>19.3f}")
print(f"  {'K-means on raw X':<22} {ari_kmeans:>13.3f} {sil_moons_kmeans:>19.3f}")
if ari_spectral > ari_kmeans and sil_moons_spectral < sil_moons_kmeans:
    print(
        "  Spectral recovers the moons far better (ARI), yet Euclidean silhouette\n"
        "  prefers K-means: silhouette rewards compact, convex blobs, so it\n"
        "  PENALISES a correct non-convex partition."
    )
else:
    print(
        "  Compare the two columns: ARI measures agreement with the truth;\n"
        "  Euclidean silhouette measures convex compactness. They need not agree."
    )

# 2b. On the real customers, sklearn's SpectralClustering runs the same
# pipeline. Subsample aggressively — the affinity matrix is n×n.

customers, feature_cols = load_customers()
X_scaled, _ = standardise(customers, feature_cols)
n_samples = X_scaled.shape[0]

# TODO: Subsample 2500 rows using the shared subsample() helper.
X_spec, idx_spec = ____
n_spec = X_spec.shape[0]

print("=" * 70)
print("  Spectral Clustering on Singapore E-commerce Customers")
print("=" * 70)
print(f"  Subsample: {n_spec:,} of {n_samples:,}")

K_CANDIDATES = [3, 4, 5]
spectral_results: dict[int, dict] = {}

print(f"\n  {'K':>3} {'Silhouette':>12} {'Time':>8}")
print("  " + "─" * 28)
for k in K_CANDIDATES:
    t0 = time.perf_counter()
    # TODO: Build a SpectralClustering model with n_clusters=k,
    # affinity="rbf", gamma=1.0, random_state=RANDOM_STATE,
    # assign_labels="kmeans". Fit_predict on X_spec.
    spec = ____
    labels = ____
    elapsed = time.perf_counter() - t0
    sil = silhouette_score(X_spec, labels)
    spectral_results[k] = {"labels": labels, "sil": sil, "time": elapsed}
    print(f"  {k:>3} {sil:>12.4f} {elapsed:>7.2f}s")


# ── Checkpoint 1 ──────────────────────────────────────────────────────────
assert U_moons.shape == (400, 2), "Task 2: moons embedding should be n x k"
assert moon_eigvals[0] < 1e-8, "Task 2: smallest L_sym eigenvalue should be ~0"
assert ari_spectral > ari_kmeans, "Task 2: spectral should beat K-means on the moons"
assert len(spectral_results) == len(K_CANDIDATES), "Task 2: spectral sweep incomplete"
print("\n  [ok] Checkpoint 1 passed — spectral embeddings fitted\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Pick the best K and compare to K-means on the same X
# ════════════════════════════════════════════════════════════════════════
# We score each K by silhouette in the ORIGINAL standardised feature space
# (not the spectral embedding) — and the moons above showed that this
# ruler is biased toward convex, K-means-like partitions.

# TODO: Pick the K with the best silhouette from spectral_results.
best_k_spec, best_stats = ____
print(f"  Best spectral K: {best_k_spec}  (silhouette={best_stats['sil']:.4f})")

km_compare = KMeans(n_clusters=best_k_spec, random_state=RANDOM_STATE, n_init=10)
km_labels_sub = km_compare.fit_predict(X_spec)
km_sil = silhouette_score(X_spec, km_labels_sub)
print(f"    K-means silhouette (same subsample): {km_sil:.4f}")
delta_sil = best_stats["sil"] - km_sil
print(f"    Δ (spectral − kmeans) = {delta_sil:+.4f}")
if delta_sil < 0:
    print(
        "    K-means scores higher — expected: it optimises (almost) what\n"
        "    Euclidean silhouette rewards. This is NOT evidence that spectral\n"
        "    is wrong; without ground truth, judge it on the business profile."
    )
else:
    print(
        "    Spectral scores higher even on a convex-biased ruler — the\n"
        "    customer groups are separated by more than straight-line distance."
    )

spec_labels = best_stats["labels"]


# ── Checkpoint 2 ──────────────────────────────────────────────────────────
assert best_k_spec in K_CANDIDATES, "Task 3: best K selection invalid"
assert len(set(spec_labels.tolist())) == best_k_spec, "Task 3: label count mismatch"
print("\n  [ok] Checkpoint 2 passed — spectral best-K selected\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: silhouette vs K
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()
fig = viz.training_history(
    {"Silhouette (spectral)": [spectral_results[k]["sil"] for k in K_CANDIDATES]},
    x_label="K",
)
fig.update_layout(title="Spectral Clustering: Silhouette vs K")
fig.write_html(str(out_path("04_spectral_silhouette.html")))
print(f"  Saved: {out_path('04_spectral_silhouette.html')}")


# ── Checkpoint 3 ──────────────────────────────────────────────────────────
assert out_path("04_spectral_silhouette.html").exists(), "Task 4: viz not saved"
print("\n  [ok] Checkpoint 3 passed — spectral visualisation rendered\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore Rail-Network Community Detection
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore rail operator's ~200 MRT/LRT stations form a
# natural graph with edge
# weights = inter-station rider flows. Spectral is the canonical
# community-detection method on graphs.
#
# BUSINESS IMPACT (illustrative assumptions, not reported figures):
# ~S$29M / year (capacity matching + station-group advertising + incident
# rerouting) on an assumed ~S$850M rail revenue.

print("  APPLY — Rail-Network Community Detection")
print("  ─────────────────────────────────────────────────────────────────")

# TODO: Compute sizes = np.bincount(spec_labels) and print each
# community's size + percentage of n_spec.
sizes = ____
for i, n in enumerate(sizes):
    print(f"    Community {i}: {n:>5,} customers ({n/n_spec:6.1%})")
print("    (In the rail scenario each node is a STATION, not a customer.)")
print("    Illustrative annual benefit: S$29M (capacity + ads + rerouting).")


# ── Checkpoint 4 ──────────────────────────────────────────────────────────
assert int(sizes.sum()) == n_spec, "Task 5: spectral partition size mismatch"
print("\n  [ok] Checkpoint 4 passed — spectral community partition valid\n")


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# scalar_metrics merges a base headline with per-K dicts derived from
# spectral_results — the same |-merge pattern you used in 02_hierarchical.

# TODO: call track_run with run_name "spectral_rbf". Headline scalars are
# spectral_best_silhouette (best_stats["sil"]), kmeans_baseline_silhouette
# (km_sil), and spectral_minus_kmeans_delta. Then |-merge in two per-K
# dicts: {f"spectral_k{k}_silhouette": float(r["sil"]) for k, r in
# spectral_results.items()} and the per-K times. The moons ARIs are given.
track_run(
    tracker,
    exp_name,
    run_name=____,
    params={
        "k_candidates": ",".join(str(k) for k in K_CANDIDATES),
        "best_k": best_k_spec,
        "affinity": "rbf",
        "gamma": 1.0,
        "n_subsample": n_spec,
    },
    scalar_metrics={
        "spectral_best_silhouette": float(best_stats["sil"]),
        "kmeans_baseline_silhouette": float(km_sil),
        "spectral_minus_kmeans_delta": float(best_stats["sil"] - km_sil),
        "moons_ari_spectral": float(ari_spectral),
        "moons_ari_kmeans": float(ari_kmeans),
    }
    | ____
    | ____,
)
print(f"  [tracked] spectral sweep + K-means baseline logged to {exp_name}\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — ClusteringEngine.fit(algorithm='spectral')
# ════════════════════════════════════════════════════════════════════════
# kailash-ml's ClusteringEngine wraps sklearn's SpectralClustering — the
# same affinity + Laplacian-embedding + KMeans-on-embedding pipeline you
# hand-built on the moons. One difference: the engine builds a
# k-nearest-neighbour affinity graph (affinity="nearest_neighbors")
# instead of the RBF kernel, so its labels can differ from the sweep above.
# The engine handles polars→numpy and returns
# ClusterResult — silhouette, CH, inertia, labels — in one call.

import polars as pl

from kailash_ml.engines.clustering import ClusteringEngine

spec_df = pl.from_numpy(X_spec, schema=feature_cols)

# TODO: instantiate ClusteringEngine and call .fit on spec_df with
# algorithm='spectral' and n_clusters=best_k_spec.
clustering = ____
fit_result = ____
print(
    f"  ClusteringEngine.fit(spectral, K={best_k_spec}): "
    f"silhouette={(fit_result.silhouette_score or 0.0):.4f}  "
    f"n_clusters={fit_result.n_clusters}"
)
print()
print(
    "  Three lessons, one ClusteringEngine surface — kmeans (lesson 01),"
    " dbscan (lesson 03), spectral (here). Same fit() signature across all"
    " three; the ExperimentTracker leaderboard makes the comparison trivial.\n"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Build an RBF affinity matrix and the normalised graph Laplacian
  [x] Embed points via the smallest k eigenvectors of L
  [x] Run K-means on the spectral embedding instead of the raw features
  [x] Recognise when spectral beats K-means (non-convex shapes, graph-
      structured data) — measured with ARI against known labels, because
      Euclidean silhouette is biased toward convex partitions
  [x] Mapped the method onto rail-network community detection for an
      illustrative ~S$29M / year capacity + ads + rerouting benefit

  KEY INSIGHT: When your data is NATURALLY a graph (stations, users,
  molecules), spectral is the default. When the similarity you care
  about is path-based rather than straight-line distance, spectral is
  the default. For everything else, prefer K-means or HDBSCAN because
  spectral is O(n^3).

  Next: 05_evaluation_profiling.py — stitch every method together and
  decide which one to ship.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
