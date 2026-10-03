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
# THEORY — Graph Laplacian Embedding in Plain English
# ════════════════════════════════════════════════════════════════════════
# Spectral clustering treats points as nodes in a graph. Edge weights
# encode "similarity". The algorithm:
#
#   1. Affinity matrix  A_ij = exp(-||x_i - x_j||² / 2σ²)   (RBF kernel)
#   2. Degree matrix    D_ii = Σ_j A_ij
#   3. Laplacian        L = D - A    (or normalised L_sym = I - D^-½ A D^-½)
#   4. Eigendecompose L and take the SMALLEST k eigenvectors → spectral
#      embedding in R^k
#   5. Run K-means on the k-dim embedding
#
# WHY it works: the k smallest eigenvectors of the graph Laplacian encode
# the graph's "connectivity structure". Points that are connected through
# many dense paths end up near each other in the embedding, even if they
# are FAR apart in the original input space. This is what lets spectral
# clustering separate non-convex shapes like concentric rings or two
# interlocking spirals — shapes K-means cannot resolve.
#
# The price: eigendecomposition is O(n³) time and O(n²) memory. Spectral
# clustering is strictly a small-to-medium data tool. For 50K+ points,
# use Nyström approximation or skip to HDBSCAN.


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
    A = np.exp(-gamma * sq_dists)  # 1. affinity
    np.fill_diagonal(A, 0.0)
    d = A.sum(axis=1)  # 2. degrees
    d_inv_sqrt = 1.0 / np.sqrt(d)
    L_sym = np.eye(X.shape[0]) - d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    eigvals, eigvecs = np.linalg.eigh(L_sym)  # 3. ascending eigenvalues
    U = eigvecs[:, :k]  # SMALLEST k eigenvectors
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    return U, eigvals[: k + 1]


U_moons, moon_eigvals = spectral_embedding(X_moons, k=2, gamma=GAMMA_MOONS)
moons_spectral = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10).fit_predict(
    U_moons
)
moons_kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10).fit_predict(
    X_moons
)
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

X_spec, idx_spec = subsample(X_scaled, n=2500, seed=RANDOM_STATE)
n_spec = X_spec.shape[0]

print("=" * 70)
print("  Spectral Clustering on Singapore E-commerce Customers")
print("=" * 70)
print(f"  Subsample: {n_spec:,} of {n_samples:,} (spectral is O(n^3))")

K_CANDIDATES = [3, 4, 5]
spectral_results: dict[int, dict] = {}

print(f"\n  {'K':>3} {'Silhouette':>12} {'Time':>8}")
print("  " + "─" * 28)
for k in K_CANDIDATES:
    t0 = time.perf_counter()
    spec = SpectralClustering(
        n_clusters=k,
        random_state=RANDOM_STATE,
        affinity="rbf",
        gamma=1.0,
        assign_labels="kmeans",
    )
    labels = spec.fit_predict(X_spec)
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
# TASK 3 — TRAIN: Pick the best K
# ════════════════════════════════════════════════════════════════════════
# Spectral has no training loop in the gradient-descent sense — the
# "training" is the eigendecomposition followed by a tiny K-means on the
# embedding. We score each K by silhouette in the ORIGINAL standardised
# feature space (not the spectral embedding) — and the moons above showed
# that this ruler is biased toward convex, K-means-like partitions.

best_k_spec, best_stats = max(spectral_results.items(), key=lambda x: x[1]["sil"])
print(f"  Best spectral K: {best_k_spec}  (silhouette={best_stats['sil']:.4f})")
print(f"  Compare with K-means silhouette on the SAME subsample:")

km_compare = KMeans(n_clusters=best_k_spec, random_state=RANDOM_STATE, n_init=10)
km_labels_sub = km_compare.fit_predict(X_spec)
km_sil = silhouette_score(X_spec, km_labels_sub)
print(f"    K-means silhouette: {km_sil:.4f}")
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
# TASK 4 — VISUALISE: silhouette vs K for the spectral sweep
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
assert out_path(
    "04_spectral_silhouette.html"
).exists(), "Task 4: visualisation not saved"
print("\n  [ok] Checkpoint 3 passed — spectral visualisation rendered\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore Rail-Network Community Detection
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore rail operator runs part of the MRT and LRT network.
# Station-to-station passenger flows form a natural graph: each station
# is a node, each tap-in→tap-out pair is an edge weighted by rider count.
# The operator's capacity planning team wants to discover "passenger communities"
# — groups of stations that trade riders densely with each other (CBD
# commuter cluster, Woodlands→Marina Bay commuter corridor, Orchard
# tourist cluster).
#
# Why spectral is the right tool here:
#   - The data is NATIVELY a graph — spectral is the canonical graph-
#     partitioning method
#   - Communities are non-convex in geographic space: a station in Jurong
#     may cluster with the CBD via commute flows, not with its neighbours
#   - The station count is ~200 — eigendecomposition is instant
#   - Normalised-cut (what spectral optimises) is the textbook community-
#     detection objective
#
# BUSINESS IMPACT (illustrative assumptions, not reported figures): assume
# ~S$850M annual rail revenue. A data-driven community taxonomy feeds
# three concrete operations:
#   1. Peak-hour train assignment — extra cars on the dense commute
#      corridor separated by the Fiedler vector (the eigenvector of the
#      second-smallest Laplacian eigenvalue). Assumed capacity-
#      matching lift: 3-5% ridership recovered from passengers who
#      currently cannot board during peak (~S$25M/year revenue recovery).
#   2. Advertising revenue — station-group media packages priced on
#      actual community co-occurrence, not geographic adjacency. Assumed
#      yield uplift 8-12% on ~S$40M ad revenue = ~S$4M/year.
#   3. Incident rerouting — when a line fails, the community graph tells
#      ops which bus bridges to prioritise.
# Total annual benefit ≈ S$29M vs. one-time modelling cost of a few
# engineer-hours (spectral on 200 nodes is 3 seconds).

print("  APPLY — Rail-Network Community Detection")
print("  ─────────────────────────────────────────────────────────────────")
sizes = np.bincount(spec_labels)
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

track_run(
    tracker,
    exp_name,
    run_name="spectral_rbf",
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
    | {
        f"spectral_k{k}_silhouette": float(r["sil"])
        for k, r in spectral_results.items()
    }
    | {f"spectral_k{k}_time_s": float(r["time"]) for k, r in spectral_results.items()},
)
print(
    f"  [tracked] spectral sweep + K-means baseline logged to {exp_name} run='spectral_rbf'\n"
)


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — ClusteringEngine.fit(algorithm='spectral')
# ════════════════════════════════════════════════════════════════════════
# kailash-ml's ClusteringEngine wraps sklearn's SpectralClustering — the
# same affinity + Laplacian-embedding + KMeans-on-embedding pipeline this
# lesson hand-built on the moons. One difference: the engine builds a
# k-nearest-neighbour affinity graph (affinity="nearest_neighbors")
# instead of the RBF kernel, so its labels can differ from the sweep above. The engine handles polars→numpy and
# returns ClusterResult — silhouette, CH, inertia, labels — in one call.

import polars as pl

from kailash_ml.engines.clustering import ClusteringEngine

spec_df = pl.from_numpy(X_spec, schema=feature_cols)
clustering = ClusteringEngine()
fit_result = clustering.fit(spec_df, algorithm="spectral", n_clusters=best_k_spec)
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
