# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 1.5: Evaluation, AutoMLEngine, and Cluster Profiling
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Score every clustering method on the same internal metrics
#     (silhouette, Davies-Bouldin, Calinski-Harabasz) and external
#     agreement metrics (ARI, NMI)
#   - Run a kailash-ml AutoMLEngine search over clustering algorithm and K
#     (agent=False: no LLM; agent mode needs a double opt-in)
#   - Profile clusters into business-meaningful segment descriptions
#   - Use the algorithm selection guide to pick the right tool for the job
#
# PREREQUISITES: 01_kmeans.py through 04_spectral.py.
#
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — internal vs external metrics and why profiling matters
#   2. Build — fit five algorithms and collect labels
#   3. Train — AutoMLEngine search (agent=False + cost cap) over algorithm/K
#   4. Visualise — metric comparison bar chart and cluster profiles
#   5. Apply — Singapore retail-bank customer segmentation selection guide
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import time

import numpy as np
import polars as pl
from dotenv import load_dotenv
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from kailash_ml import AutoMLEngine, ModelVisualizer
from kailash_ml.automl import AutoMLConfig, ParamSpec, Trial, TrialOutcome
from kailash_ml.engines.clustering import ClusteringEngine

from shared.mlfp04.ex_1 import (
    RANDOM_STATE,
    agreement,
    load_customers,
    out_path,
    print_metric_row,
    score_partition,
    setup_engines,
    standardise,
    subsample,
    teardown_engines,
    track_run,
)
from shared.mlfp04 import create_visualizer

load_dotenv()

# ── Kailash-ML ExperimentTracker — clustering zoo shared store ───────────
tracker, exp_name = setup_engines()


try:
    import hdbscan as hdbscan_lib
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "05_evaluation_profiling.py compares HDBSCAN too: uv add hdbscan"
    ) from e


# ════════════════════════════════════════════════════════════════════════
# THEORY — Internal vs External Metrics and Profiling
# ════════════════════════════════════════════════════════════════════════
# Internal metrics (silhouette, Davies-Bouldin, Calinski-Harabasz) rank
# methods using only X + labels. External metrics (ARI, NMI) tell you
# how much two partitions AGREE — high agreement = real structure.
# Neither is sufficient without the profiling step, which converts
# statistical labels into actionable business segments via z-scores.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Fit every method and collect full-data labels
# ════════════════════════════════════════════════════════════════════════

customers, feature_cols = load_customers()
X_scaled, _ = standardise(customers, feature_cols)
n_samples = X_scaled.shape[0]

print("=" * 70)
print("  Clustering Evaluation + Profiling on Singapore E-commerce Customers")
print("=" * 70)
print(f"  Samples={n_samples:,}  features={len(feature_cols)}")

# Fixed for a like-for-like comparison. 01_kmeans.py showed silhouette and
# the gap statistic need not agree on K for this data, so K=5 is a
# business-granularity choice; the AutoMLEngine search in Task 3 lets the
# silhouette criterion pick its own K on a subsample.
BEST_K = 5
all_labels: dict[str, np.ndarray] = {}

# TODO: Fit K-means with BEST_K clusters (init='k-means++', n_init=10)
# and store all_labels["K-means"] = km.fit_predict(X_scaled).
km = ____
all_labels["K-means"] = ____

# TODO: Fit a GaussianMixture with n_components=BEST_K, covariance_type='full'
# and store all_labels["GMM"] = gmm.fit_predict(X_scaled).
gmm = ____
all_labels["GMM"] = ____

# --- Ward hierarchical (KNN-extend to full data) ---
X_hier, idx_hier = subsample(X_scaled, n=2000, seed=RANDOM_STATE)
Z = linkage(X_hier, method="ward")
ward_sub = fcluster(Z, t=BEST_K, criterion="maxclust") - 1
knn = KNeighborsClassifier(n_neighbors=5).fit(X_hier, ward_sub)
all_labels["Ward"] = knn.predict(X_scaled)

# --- DBSCAN with k-distance-selected epsilon ---
nn = NearestNeighbors(n_neighbors=10).fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
k_dist = np.sort(distances[:, -1])
# Kneedle elbow, as in 03_density_based.py: the point furthest from the
# chord of the normalised k-distance curve (the 2nd-derivative argmax
# latches onto the steepest tail jump and over-shoots).
_x = np.linspace(0.0, 1.0, k_dist.size)
_y = (k_dist - k_dist.min()) / (k_dist.max() - k_dist.min())
eps_suggested = float(k_dist[int(np.argmax(np.abs(_y - _x)))])
all_labels["DBSCAN"] = DBSCAN(eps=eps_suggested, min_samples=10, n_jobs=-1).fit_predict(
    X_scaled
)

# --- HDBSCAN ---
all_labels["HDBSCAN"] = hdbscan_lib.HDBSCAN(
    min_cluster_size=50, min_samples=10, cluster_selection_method="eom"
).fit_predict(X_scaled)

# --- Spectral (subsample + KNN-extend) ---
X_spec, idx_spec = subsample(X_scaled, n=2500, seed=RANDOM_STATE)
spec_sub = SpectralClustering(
    n_clusters=BEST_K,
    random_state=RANDOM_STATE,
    affinity="rbf",
    gamma=1.0,
    assign_labels="kmeans",
).fit_predict(X_spec)
knn_spec = KNeighborsClassifier(n_neighbors=5).fit(X_spec, spec_sub)
all_labels["Spectral"] = knn_spec.predict(X_scaled)

print("\n  Internal metrics per method:")
results: dict[str, dict] = {}
for name, labels in all_labels.items():
    # TODO: Use the shared score_partition(X_scaled, labels) helper and
    # store the result in results[name]. Then print via print_metric_row.
    m = ____
    results[name] = m
    print_metric_row(name, m)


# ── Checkpoint 1 ──────────────────────────────────────────────────────────
assert len(results) == 6, "Task 2: all six methods should be scored"
assert all("silhouette" in r for r in results.values()), "Task 2: metric gap"
print("\n  [ok] Checkpoint 1 passed — all methods scored\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: AutoMLEngine search over algorithm and K
# ════════════════════════════════════════════════════════════════════════
# The kailash-ml AutoMLEngine runs the comparison as a governed SEARCH: you
# declare the search space (ParamSpec) and a trial function that trains
# one candidate and returns its metric (TrialOutcome); the engine proposes
# trials, enforces the trial/time/cost budget and keeps the audit record.
# The trainer here is ClusteringEngine, so each trial is one .fit() call.
#
# agent=True would let an LLM propose trials — that costs money and is
# non-deterministic, so it is gated behind a DOUBLE opt-in (the flag plus
# an explicit cost cap / approval). We keep agent=False: no LLM runs.

AUTOML_SUBSAMPLE = 3000  # silhouette is O(n^2); search on a subsample
X_auto, _ = subsample(X_scaled, n=AUTOML_SUBSAMPLE, seed=RANDOM_STATE)
auto_df = pl.from_numpy(X_auto, schema=feature_cols)
automl_clustering = ClusteringEngine()

# TODO: Build an AutoMLConfig for a clustering search that MAXIMISES
# "silhouette" with search_strategy="grid", max_trials=12, agent=False and
# max_llm_cost_usd=1.0.
config = ____
# TODO: Declare the search space: a categorical "algorithm" over
# ("kmeans", "gmm") and an int "n_clusters" from 3 to 8.
# Hint: ParamSpec(name=..., kind="categorical", choices=(...)) / kind="int", low=, high=
search_space = ____


async def clustering_trial(trial: Trial) -> TrialOutcome:
    """Fit one (algorithm, K) candidate with ClusteringEngine; report silhouette."""
    t0 = time.perf_counter()
    # TODO: call automl_clustering.fit on auto_df with the trial's
    # "algorithm" and int("n_clusters") parameters.
    result = ____
    sil = result.silhouette_score
    return TrialOutcome(
        trial_number=trial.trial_number,
        params=trial.params,
        metric=float(sil) if sil is not None else float("nan"),
        metric_name="silhouette",
        direction="maximize",
        duration_seconds=time.perf_counter() - t0,
        error=None if sil is not None else "silhouette undefined (<2 clusters)",
    )


automl = AutoMLEngine(config=config, tenant_id="mlfp04", actor_id="student")
# TODO: run the search — automl.run is async; pass space= and trial_fn=.
automl_result = ____

print(f"  AutoMLEngine search ({config.search_strategy}, agent={config.agent}):")
print(
    f"    trials: {automl_result.completed_trials} completed, "
    f"{automl_result.failed_trials} failed, {automl_result.denied_trials} denied"
)
print(f"    {'#':>3} {'algorithm':<10} {'K':>3} {'silhouette':>11}")
for rec in automl_result.all_trials:
    print(
        f"    {rec.trial_number:>3} {rec.params['algorithm']:<10} "
        f"{int(rec.params['n_clusters']):>3} {rec.metric_value:>11.4f}"
    )
best_trial = automl_result.best_trial
if best_trial is not None:
    print(
        f"    Best: {best_trial.params['algorithm']} with "
        f"K={int(best_trial.params['n_clusters'])} "
        f"(silhouette={best_trial.metric_value:.4f}) on a "
        f"{AUTOML_SUBSAMPLE:,}-row subsample"
    )
print("  No governance engine or database connection is attached here, so the")
print("  engine logs that admission checks are skipped and trials are kept in")
print("  memory only. In production you pass both to AutoMLEngine(...).")

print("\n  External agreement (ARI / NMI):")
method_names = list(all_labels.keys())
for i in range(len(method_names)):
    for j in range(i + 1, len(method_names)):
        m1, m2 = method_names[i], method_names[j]
        # TODO: Call the shared agreement(labels_a, labels_b) helper.
        a = ____
        print(f"    {m1:<10} vs {m2:<10}  ARI={a['ari']:+.4f}  NMI={a['nmi']:+.4f}")


# ── Checkpoint 2 ──────────────────────────────────────────────────────────
assert config.agent is False, "Task 3: agent must default to False (double opt-in)"
assert config.max_llm_cost_usd > 0, "Task 3: cost cap must be positive"
assert automl_result.completed_trials > 0, "Task 3: the search should complete trials"
assert best_trial is not None, "Task 3: the search should return a best trial"
print("\n  [ok] Checkpoint 2 passed — AutoMLEngine search ran with guardrails\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Metric bar chart + cluster profiles
# ════════════════════════════════════════════════════════════════════════

viz = create_visualizer()
fig = viz.metric_comparison(
    {
        k: {"silhouette": v["silhouette"], "calinski_harabasz": v["calinski_harabasz"]}
        for k, v in results.items()
        if not np.isnan(v["silhouette"])
    }
)
fig.update_layout(title="Clustering Method Comparison (internal metrics)")
fig.write_html(str(out_path("05_method_comparison.html")))
print(f"  Saved: {out_path('05_method_comparison.html')}")

best_name = max(
    ((k, v) for k, v in results.items() if not np.isnan(v["silhouette"])),
    key=lambda x: x[1]["silhouette"],
)[0]
best_labels = all_labels[best_name]
print(f"\n  Best method by silhouette: {best_name}")

clustered = customers.with_columns(pl.Series("cluster", best_labels))
for cid in sorted(set(int(c) for c in best_labels.tolist() if c >= 0)):
    subset = clustered.filter(pl.col("cluster") == cid)
    pct = subset.height / clustered.height * 100
    print(f"\n  Cluster {cid} — n={subset.height:,} ({pct:.1f}%)")
    for col in feature_cols[:6]:
        mean_val = subset[col].mean()
        overall_mean = clustered[col].mean()
        overall_std = clustered[col].std()
        if overall_std and overall_std > 0:
            z = (mean_val - overall_mean) / overall_std
        else:
            z = 0.0
        flag = "HIGH" if z > 0.5 else ("LOW " if z < -0.5 else "    ")
        print(f"    {col:<28} mean={mean_val:>10.2f}  z={z:+.2f}  {flag}")


# ── Checkpoint 3 ──────────────────────────────────────────────────────────
assert out_path("05_method_comparison.html").exists(), "Task 4: chart not saved"
assert "cluster" in clustered.columns, "Task 4: cluster column missing"
print("\n  [ok] Checkpoint 3 passed — metric chart + cluster profiles rendered\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore Retail-Bank Segmentation Selection Guide
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore bank's consumer banking runs FIVE different
# segmentation programs — loyalty tiers, wealth desk affinity, fraud rings,
# cross-sell offers, RM-beat optimisation — each needs a DIFFERENT
# algorithm.
#
# BUSINESS IMPACT (illustrative assumptions, not reported figures):
# ~S$62M / year aggregate benefit.

print("  APPLY — Retail-Bank Consumer Segmentation Selection Guide")
print("  ─────────────────────────────────────────────────────────────────")
print(
    """
  ┌──────────────────┬───────────────────┬──────────────┬──────────────┬───────────────┐
  │ Algorithm        │ Requires K?       │ Cluster Shape│ Noise        │ Scalability   │
  ├──────────────────┼───────────────────┼──────────────┼──────────────┼───────────────┤
  │ K-means          │ Yes               │ Convex       │ None         │ O(nKI)        │
  │ Hierarchical     │ Yes (cut height)  │ Any          │ None         │ O(n^2 log n)  │
  │ DBSCAN           │ No (eps, minPts)  │ Arbitrary    │ Yes (-1)     │ O(n log n)    │
  │ HDBSCAN          │ No (auto)         │ Arbitrary    │ Yes (-1)     │ O(n log n)    │
  │ Spectral         │ Yes               │ Non-convex   │ None         │ O(n^3)        │
  │ GMM (full cov.)  │ Yes (BIC selects) │ Ellipsoidal  │ Soft         │ O(nKd^2)/iter │
  └──────────────────┴───────────────────┴──────────────┴──────────────┴───────────────┘
"""
)
print("  Illustrative annual benefit: S$62M across four segmentation programs.")


# ── Checkpoint 4 ──────────────────────────────────────────────────────────
assert best_name in results, "Task 5: best method must be in results"
print("\n  [ok] Checkpoint 4 passed — selection guide delivered\n")


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log the leaderboard to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# Method names (K-means / GMM / Ward / DBSCAN / HDBSCAN / Spectral) all
# match the tracker key regex [a-zA-Z_][a-zA-Z0-9_.\-]* — no _slug() is
# needed. silhouette CAN be NaN on a collapsed partition — track_run
# reports and skips undefined metrics rather than logging a fake 0.0.

per_method_scalars: dict[str, float] = {}
for name, m in results.items():
    # TODO: For each method, write three scalar entries — silhouette,
    # calinski_harabasz, davies_bouldin — cast with float(). Use keys
    # f"{name}_silhouette", f"{name}_calinski_harabasz",
    # f"{name}_davies_bouldin".
    per_method_scalars[f"{name}_silhouette"] = ____
    per_method_scalars[f"{name}_calinski_harabasz"] = ____
    per_method_scalars[f"{name}_davies_bouldin"] = ____

# TODO: call track_run with run_name="evaluation_profiling". Headline
# scalars: winner_silhouette (the best method's silhouette as a float)
# and n_methods_scored. |-merge with per_method_scalars.
track_run(
    tracker,
    exp_name,
    run_name=____,
    params={
        "best_k": BEST_K,
        "n_methods": len(results),
        "n_samples": n_samples,
        "automl_strategy": config.search_strategy,
        "automl_max_trials": config.max_trials,
        "automl_agent": config.agent,
        "automl_best": (
            f"{best_trial.params['algorithm']}_k{int(best_trial.params['n_clusters'])}"
        ),
    },
    scalar_metrics={
        "winner_silhouette": ____,
        "n_methods_scored": float(len(results)),
        "automl_best_silhouette": float(best_trial.metric_value),
        "automl_completed_trials": float(automl_result.completed_trials),
    }
    | per_method_scalars,
)
print(
    f"  [tracked] {len(results)}-method leaderboard logged to {exp_name} "
    f"(winner: {best_name})\n"
)


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — ClusteringEngine.fit(algorithm='kmeans')
# ════════════════════════════════════════════════════════════════════════
# kailash-ml's ClusteringEngine wraps four of the algorithms you fitted
# by hand — kmeans, gmm, dbscan, spectral — under one .fit() surface (Ward
# hierarchical and HDBSCAN stay with scipy / hdbscan). The AutoMLEngine
# search in Task 3 already drove that surface: it proposed (algorithm, K)
# trials and called ClusteringEngine.fit for each.

cluster_df = pl.from_numpy(X_scaled, schema=feature_cols)

# TODO: Instantiate ClusteringEngine and call .fit on cluster_df with
# algorithm='kmeans' and n_clusters=BEST_K.
clustering = ____
fit_result = ____
print(
    f"  ClusteringEngine.fit(kmeans, K={BEST_K}): "
    f"silhouette={(fit_result.silhouette_score or 0.0):.4f}  "
    f"n_clusters={fit_result.n_clusters}"
)
print(
    f"  Hand-rolled K-means silhouette (Task 2): "
    f"{results['K-means']['silhouette']:.4f} "
    f"— same algorithm, one-line vs ten-line interface.\n"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Scored five clustering methods on silhouette, DB, CH
  [x] Measured pairwise agreement via ARI and NMI — high agreement means
      the structure is real; low agreement means the domain expert must
      arbitrate
  [x] Ran an AutoMLEngine search over algorithm and K with agent=False
      + max_llm_cost_usd — the double opt-in pattern that makes LLM cost
      explicit
  [x] Profiled the best partition via per-feature z-scores to convert
      statistical labels into actionable business segments
  [x] Applied the selection guide to a retail bank: five use cases, five
      different right algorithms, illustrative S$62M / year benefit

  KEY INSIGHT: There is no universally best clustering algorithm. The
  choice depends on data size, cluster shape, need for noise detection,
  need for soft assignments, and the downstream decision. The job of the
  ML engineer is to match the algorithm to the problem — and to PROFILE
  the result so the marketing/ops team can act on it.

  Next: Exercise 2 digs into the EM algorithm behind GMM — implementing
  the E-step and M-step by hand to see the log-likelihood improve every
  iteration.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
