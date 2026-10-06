# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 1.1: K-means with k-means++ Initialisation
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Apply K-means with k-means++ initialisation and understand why it
#     converges faster than random initialisation
#   - Use the elbow method, silhouette score and the gap statistic to
#     select K, and see where the three criteria disagree
#   - Read per-sample silhouette to spot mis-assigned points
#   - Interpret inertia (within-cluster sum of squares) as a loss value
#
# PREREQUISITES: MLFP03 complete (supervised ML, feature scaling).
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — why K-means works and how k-means++ fixes its weakness
#   2. Build — the elbow + silhouette sweep and the gap statistic across K
#   3. Train — fit K-means with k-means++ vs random and compare
#   4. Visualise — silhouette / gap curves vs K + per-sample silhouette
#   5. Apply — Singapore e-commerce loyalty segmentation, $ impact per tier
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time

import numpy as np
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

from kailash_ml import ModelVisualizer

from shared.mlfp04.ex_1 import (
    RANDOM_STATE,
    load_customers,
    out_path,
    setup_engines,
    standardise,
    teardown_engines,
    track_run,
)
from shared.mlfp004 import create_visualizer

load_dotenv()

# ── Kailash-ML ExperimentTracker — every clustering run logs here ─────────
tracker, exp_name = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why K-means Works and Why k-means++ Matters
# ════════════════════════════════════════════════════════════════════════
# K-means minimises the within-cluster sum of squares:
#     J = Σ_k Σ_{x in C_k} ||x - μ_k||²
# It alternates two steps: assign each point to the nearest centroid,
# then recompute each centroid as the mean of its assigned points.
# k-means++ seeds centroids far apart to avoid poor local minima — it
# usually helps a SINGLE start; with many restarts (n_init) random seeding
# often catches up, so measure, don't assume.
#
# Gap statistic: Gap(k) = E*[log W_k] - log W_k, where W_k is the
# within-cluster sum of squares and E* averages over B uniform reference
# datasets drawn from the data's bounding box. Pick the SMALLEST k with
# Gap(k) >= Gap(k+1) - s_{k+1}, where s_k = sd_k * sqrt(1 + 1/B).


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Load data and sweep K with silhouette scoring
# ════════════════════════════════════════════════════════════════════════

customers, feature_cols = load_customers()
X_scaled, _ = standardise(customers, feature_cols)
n_samples, n_features = X_scaled.shape

print("=" * 70)
print("  K-means on Singapore E-commerce Customers")
print("=" * 70)
print(f"  Samples={n_samples:,}  features={n_features}")


def sweep_k(X: np.ndarray, k_values: range) -> dict[str, list[float]]:
    """Fit K-means for each K and return per-K inertia + validity metrics."""
    inertias, sils, chs, dbs = [], [], [], []
    print(f"\n  {'K':>3} {'Inertia':>12} {'Silhouette':>12} {'CH':>10} {'DB':>8}")
    print("  " + "─" * 50)
    for k in k_values:
        # TODO: Build a KMeans instance with n_clusters=k, init='k-means++',
        # n_init=10, random_state=RANDOM_STATE. Fit_predict on X.
        # Hint: km = KMeans(n_clusters=____, random_state=____, n_init=10, init="k-means++")
        km = ____
        labels = ____

        # TODO: Append km.inertia_ to inertias and the three sklearn metrics
        # (silhouette_score, calinski_harabasz_score, davies_bouldin_score)
        # computed on (X, labels).
        inertias.append(____)
        sils.append(____)
        chs.append(____)
        dbs.append(____)
        print(
            f"  {k:>3} {km.inertia_:>12.0f} {sils[-1]:>12.4f} "
            f"{chs[-1]:>10.0f} {dbs[-1]:>8.4f}"
        )
    return {"inertia": inertias, "silhouette": sils, "ch": chs, "db": dbs}


K_RANGE = range(2, 11)
sweep = sweep_k(X_scaled, K_RANGE)

# TODO: Pick best_k as the K that MAXIMISES silhouette. Use np.argmax.
best_k = ____
print(f"\n  Best K by silhouette: {best_k} (score={max(sweep['silhouette']):.4f})")


def gap_statistic(
    X: np.ndarray, k_values: range, n_refs: int = 10, n_sub: int = 5000
) -> dict[str, list[float] | int]:
    """Tibshirani gap statistic with the 1-standard-error rule.

    Runs on a random subsample (n_sub rows) because every K needs n_refs
    extra K-means fits on uniform reference data.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(X.shape[0], min(n_sub, X.shape[0]), replace=False)
    X_sub = X[idx]
    lo, hi = X_sub.min(axis=0), X_sub.max(axis=0)
    # TODO: draw n_refs reference datasets uniformly inside [lo, hi],
    # each with the same shape as X_sub.
    # Hint: rng.uniform(low, high, size=shape) inside a list comprehension
    refs = ____

    gaps, s_k = [], []
    for k in k_values:
        log_wk = np.log(
            KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=3)
            .fit(X_sub)
            .inertia_
        )
        ref_log_wk = np.array(
            [
                np.log(
                    KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=3)
                    .fit(R)
                    .inertia_
                )
                for R in refs
            ]
        )
        # TODO: Gap(k) = mean of the reference log W_k minus the real log W_k;
        # s_k = std of the reference log W_k times sqrt(1 + 1/n_refs).
        gaps.append(____)
        s_k.append(____)

    ks = list(k_values)
    # 1-SE rule: smallest k whose gap is within one SE of the next k's gap
    chosen = ks[int(np.argmax(gaps))]
    for i in range(len(ks) - 1):
        # TODO: apply the 1-SE rule test to (gaps[i], gaps[i + 1], s_k[i + 1])
        if ____:
            chosen = ks[i]
            break
    return {"gap": gaps, "s_k": s_k, "best_k": chosen}


gap = gap_statistic(X_scaled, K_RANGE)
gap_k = int(gap["best_k"])
print(f"\n  {'K':>3} {'Gap(k)':>10} {'s_k':>8}")
for k, g, s in zip(K_RANGE, gap["gap"], gap["s_k"]):
    print(f"  {k:>3} {g:>10.4f} {s:>8.4f}")
print(f"  Best K by gap statistic (1-SE rule): {gap_k}")
if gap_k == best_k:
    print(f"  Silhouette and gap agree on K={best_k}.")
else:
    print(
        f"  Silhouette picks K={best_k}, gap picks K={gap_k}. They answer "
        "different questions\n  (separation between clusters vs tightness "
        "relative to no-structure data),\n  so disagreement means the data "
        "has no single 'true' K — K becomes a business choice."
    )


# ── Checkpoint 1 ──────────────────────────────────────────────────────────
assert 2 <= best_k <= 10, "Task 2: best_k must be in the tested range"
assert max(sweep["silhouette"]) > 0, "Task 2: best silhouette should be positive"
assert len(sweep["inertia"]) == len(list(K_RANGE)), "Task 2: sweep size mismatch"
assert len(gap["gap"]) == len(list(K_RANGE)), "Task 2: gap sweep size mismatch"
assert 2 <= gap_k <= 10, "Task 2: gap-statistic K must be in the tested range"
print("\n  [ok] Checkpoint 1 passed — silhouette + gap sweeps complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: k-means++ vs random initialisation head-to-head
# ════════════════════════════════════════════════════════════════════════

# TODO: Build TWO KMeans models with the SAME n_clusters=best_k and
# n_init=10, but different init values: "k-means++" and "random".
km_plus = ____
km_random = ____

t0 = time.perf_counter()
km_plus.fit(X_scaled)
t_plus = time.perf_counter() - t0

t0 = time.perf_counter()
km_random.fit(X_scaled)
t_random = time.perf_counter() - t0

print(f"  k-means++ vs Random Initialisation (K={best_k}):")
print(
    f"    k-means++: inertia={km_plus.inertia_:12.0f}  iters={km_plus.n_iter_:>3}  time={t_plus:.3f}s"
)
print(
    f"    Random:    inertia={km_random.inertia_:12.0f}  iters={km_random.n_iter_:>3}  time={t_random:.3f}s"
)


# With n_init=10 both runs keep their best of 10 restarts, which hides the
# seeding difference. A fair test of SEEDING compares single-start runs
# (n_init=1) averaged over several seeds.
single_plus, single_random = [], []
for seed in range(5):
    # TODO: fit one KMeans per init with n_clusters=best_k, random_state=seed,
    # n_init=1, and append each model's .inertia_ to its list.
    single_plus.append(____)
    single_random.append(____)
mean_plus, mean_random = float(np.mean(single_plus)), float(np.mean(single_random))
print(
    f"    Single-start mean over 5 seeds: k-means++={mean_plus:,.0f}  "
    f"random={mean_random:,.0f}"
)
if mean_plus < mean_random:
    print("    k-means++ seeding reached a lower inertia on average from one start.")
else:
    print(
        "    Random seeding matched or beat k-means++ here — on well-spread data\n"
        "    the seeding advantage can vanish; restarts (n_init) matter more."
    )

km_labels = km_plus.predict(X_scaled)


# ── Checkpoint 2 ──────────────────────────────────────────────────────────
assert np.isfinite(km_plus.inertia_) and np.isfinite(
    km_random.inertia_
), "Task 3: both initialisations should produce a finite inertia"
assert len(single_plus) == len(single_random) == 5, "Task 3: 5 single-start runs each"
assert len(set(km_labels.tolist())) == best_k, "Task 3: wrong cluster count"
print("\n  [ok] Checkpoint 2 passed — initialisation comparison complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Silhouette curves and per-sample silhouette
# ════════════════════════════════════════════════════════════════════════

viz = create_visualizer()
history = {
    "Silhouette": sweep["silhouette"],
    "Inertia (scaled)": [i / max(sweep["inertia"]) for i in sweep["inertia"]],
    "Gap statistic (scaled)": [g / max(gap["gap"]) for g in gap["gap"]],
}
fig = viz.training_history(history, x_label="K")
fig.update_layout(
    title=f"K-means: Silhouette, Inertia and Gap vs K "
    f"(silhouette K={best_k}, gap K={gap_k})"
)
fig.write_html(str(out_path("01_kmeans_elbow.html")))
print(f"  Saved: {out_path('01_kmeans_elbow.html')}")

# TODO: Compute per-sample silhouette using sklearn.metrics.silhouette_samples
# on (X_scaled, km_labels). Then for each cluster id, print its size, mean
# silhouette, and the number of points with s(i) < 0 (mis-assigned).
sil_samples = ____

print(f"\n  Per-Sample Silhouette (K={best_k}):")
for cid in range(best_k):
    mask = km_labels == cid
    s = sil_samples[mask]
    n_neg = int((s < 0).sum())
    print(
        f"    Cluster {cid}: n={int(mask.sum()):>5}  mean_sil={s.mean():.4f}  "
        f"mis-assigned={n_neg} ({n_neg/len(s):.1%})"
    )


# ── Checkpoint 3 ──────────────────────────────────────────────────────────
assert sil_samples.shape[0] == n_samples, "Task 4: per-sample silhouette missing points"
print("\n  [ok] Checkpoint 3 passed — visualisation and per-sample audit done\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore E-commerce Loyalty Segmentation
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A regional e-commerce platform's Singapore CRM team replaces
# its hand-coded Bronze/Silver/Gold tiers with data-driven K-means
# segments. K-means is a reasonable first tool: small K, standardised
# features, and the centroids ARE the segment profiles.
#
# BUSINESS IMPACT (illustrative assumptions, not a published figure): on an
# assumed 3M buyer base, a 20% lift on S$20/buyer incremental campaign
# revenue is ~S$12M / year. Training cost: seconds.

print("  APPLY — Singapore E-commerce Loyalty Segmentation")
print("  ─────────────────────────────────────────────────────────────────")

# TODO: Compute segment sizes via np.bincount(km_labels). Print each
# segment's size and its percentage of n_samples.
segment_sizes = ____
for i, n in enumerate(segment_sizes):
    pct = n / n_samples * 100
    print(f"    Segment {i}: {n:>5,} customers ({pct:5.1f}%)")
print("    Illustrative annual lift: S$12M (assumed 3M buyers × S$20 × 20%).")


# ── Checkpoint 4 ──────────────────────────────────────────────────────────
assert segment_sizes.min() > 0, "Task 5: every segment must have at least one customer"
assert int(segment_sizes.sum()) == n_samples, "Task 5: counts must sum to n_samples"
print("\n  [ok] Checkpoint 4 passed — segment sizes valid\n")


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# Every M4 ex_1 lesson logs into the SAME experiment ('m4_clustering_zoo')
# so you can compare K-means against hierarchical / DBSCAN / spectral /
# GMM later from one SQLite store. Sweep series = per-K curves; scalar
# metrics = the final-fit numbers.

# TODO: call track_run with the tracker + exp_name from setup_engines().
# run_name should identify the technique ("kmeans_pp" matches the solution
# leaderboard). Fill in the scalar_metrics value for the best silhouette
# from sweep["silhouette"], and the series_metrics dict with sweep_silhouette
# and sweep_inertia (already collected in `sweep`).
track_run(
    tracker,
    exp_name,
    run_name=____,
    params={
        "init": "k-means++",
        "n_init": 10,
        "random_state": RANDOM_STATE,
        "best_k": best_k,
        "n_features": n_features,
        "n_samples": n_samples,
    },
    scalar_metrics={
        "best_silhouette": ____,
        "kmeans_pp_inertia": float(km_plus.inertia_),
        "kmeans_random_inertia": float(km_random.inertia_),
        "kmeans_pp_iters": float(km_plus.n_iter_),
        "kmeans_random_iters": float(km_random.n_iter_),
        "kmeans_pp_time_s": float(t_plus),
        "kmeans_random_time_s": float(t_random),
        "gap_best_k": float(gap_k),
        "single_start_pp_inertia_mean": mean_plus,
        "single_start_random_inertia_mean": mean_random,
    },
    series_metrics={
        "sweep_silhouette": ____,
        "sweep_inertia": ____,
        "sweep_gap": gap["gap"],
    },
)
print(f"  [tracked] sweep + final-fit metrics logged to {exp_name}\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — the kailash-ml ClusteringEngine
# ════════════════════════════════════════════════════════════════════════
# You hand-rolled the K sweep, the silhouette/CH/DB metrics, the
# k-means++-vs-random comparison, and the per-sample silhouette audit —
# ~120 lines of structure to internalise the moving parts.
#
# kailash-ml ships a single engine that IS that pipeline. ClusteringEngine.
# `sweep_k` runs the K-vs-criterion sweep (silhouette or Calinski-Harabasz)
# for the algorithms that take a K (kmeans, gmm, spectral), and `fit`
# (kmeans, dbscan, gmm, spectral) returns a
# ClusterResult with labels + silhouette + Calinski-Harabasz + inertia.

import polars as pl

from kailash_ml.engines.clustering import ClusteringEngine

cluster_df = pl.from_numpy(X_scaled, schema=feature_cols)

# TODO: instantiate ClusteringEngine and call .sweep_k on cluster_df with
# range(2, 11), algorithm='kmeans', criterion='silhouette'. Print the
# returned sweep_result.optimal_k.
clustering = ____
sweep_result = ____
print(f"  ClusteringEngine.sweep_k(): optimal_k={sweep_result.optimal_k}")

# TODO: call clustering.fit on cluster_df with algorithm='kmeans' and
# n_clusters=best_k. The returned ClusterResult exposes .silhouette_score,
# .calinski_harabasz_score, and .inertia.
fit_result = ____
print(
    f"  ClusteringEngine.fit(K={best_k}): "
    f"silhouette={fit_result.silhouette_score:.4f}  "
    f"CH={fit_result.calinski_harabasz_score:.0f}  "
    f"inertia={fit_result.inertia:.0f}"
)
print()
print("  Every metric the lesson printed by hand — silhouette, CH, inertia,")
print("  cluster sizes — is a field on ClusterResult. ClusteringEngine IS")
print("  the destination this lesson walked you toward.\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] K-means minimises within-cluster sum of squares via alternating
      assignment/update steps — guaranteed to converge
  [x] Compared k-means++ and random seeding fairly (single starts
      averaged over seeds) instead of assuming k-means++ always wins
  [x] Silhouette score gives an objective criterion for choosing K
      (the elbow alone is subjective); the gap statistic compares the
      fit against structureless reference data
  [x] Per-sample silhouette exposes mis-assigned points for re-review
  [x] Mapped K={best_k} clusters onto an e-commerce loyalty tier system
      with an illustrative S$12M / year campaign revenue lift

  KEY INSIGHT: K-means gives you the centroids for free. The centroids
  ARE the segment profiles — no extra analysis needed before handing them
  to marketing. This is why K-means is the default first-pass clustering
  algorithm for customer segmentation.

  Next: 02_hierarchical.py — when you need a dendrogram instead of a K.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
