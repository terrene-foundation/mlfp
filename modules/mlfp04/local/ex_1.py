# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 1: Clustering
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Apply K-means, hierarchical, spectral, HDBSCAN, and GMM clustering
#   - Evaluate clusters using silhouette, Davies-Bouldin, and Calinski-Harabasz
#   - Select the appropriate clustering algorithm based on data shape
#   - Use AutoMLEngine for automated clustering comparison with a cost guardrail
#   - Profile clusters with business-meaningful descriptions
#
# PREREQUISITES:
#   - MLFP03 complete (supervised ML, feature engineering, preprocessing pipeline)
#   - MLFP02 complete (statistics, regression — for cluster evaluation intuition)
#
# ESTIMATED TIME: 60-90 minutes
#
# TASKS:
#   1. Load and prepare e-commerce customer features
#   2. Run K-means, spectral, HDBSCAN, GMM clustering
#   3. Evaluate with silhouette, Calinski-Harabasz, Davies-Bouldin
#   4. Use AutoMLEngine with agent=True double opt-in
#   5. Analyse cluster profiles
#   6. Visualise with ModelVisualizer
#
# DATASET: E-commerce customer data (from MLFP03)
#   Goal: segment customers without labels into behavioural groups
#   Business context: cluster = marketing segment (loyalty, spending, frequency)
#   USML bridge: In M3 you predicted labels. Now: what if there are no labels?
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import polars as pl
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.preprocessing import StandardScaler

from kailash_ml import PreprocessingPipeline, ModelVisualizer, DataExplorer
from kailash_ml.interop import to_sklearn_input

from shared import MLFPDataLoader

try:
    import hdbscan
except ImportError:
    hdbscan = None


# ── Data Loading ──────────────────────────────────────────────────────

loader = MLFPDataLoader()
customers = loader.load("mlfp03", "ecommerce_customers.parquet")

print(f"=== E-commerce Customer Data ===")
print(f"Shape: {customers.shape}")
print(f"Columns: {customers.columns}")

# TODO: Select numeric feature columns for clustering, excluding "customer_id"
feature_cols = [
    ____  # Hint: use zip(customers.columns, customers.dtypes) to filter pl.Float64/Float32/Int64/Int32
]

print(f"Clustering features ({len(feature_cols)}): {feature_cols}")

# Prepare data
X, _, col_info = to_sklearn_input(
    customers.drop_nulls(subset=feature_cols),
    feature_columns=feature_cols,
)

# TODO: Standardise features with StandardScaler — critical for distance-based clustering
scaler = ____  # Hint: StandardScaler()
# TODO: Fit the scaler to X and transform it
X_scaled = ____  # Hint: scaler.fit_transform(X)
print(f"Samples: {X_scaled.shape[0]:,}, Features: {X_scaled.shape[1]}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert X_scaled.shape[0] > 0, "Customer dataset should not be empty"
assert X_scaled.shape[1] > 0, "Should have at least one clustering feature"
assert len(feature_cols) > 0, "Feature column list should not be empty"
# INTERPRETATION: Standardisation (zero mean, unit variance) is mandatory
# for all distance-based algorithms (K-means, KNN, spectral, HDBSCAN).
# Without it, a feature measured in thousands (e.g., revenue) dominates
# distance calculations and makes all other features irrelevant.
print("\n✓ Checkpoint 1 passed — customer features loaded and standardised\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: K-means clustering
# ══════════════════════════════════════════════════════════════════════

# Elbow method: test K=2..10
inertias = []
sil_scores = []
for k in range(2, 11):
    # TODO: Create a KMeans model with n_clusters=k, random_state=42, n_init=10
    km = ____  # Hint: KMeans(n_clusters=k, ...)
    # TODO: Fit and predict cluster labels on X_scaled
    labels = ____  # Hint: km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    # TODO: Compute silhouette_score for this k
    sil_scores.append(____)  # Hint: silhouette_score(X_scaled, labels)

best_k = range(2, 11)[np.argmax(sil_scores)]
print(f"\n=== K-means ===")
print(f"Best K by silhouette: {best_k} (score={max(sil_scores):.4f})")

# TODO: Create the best KMeans model with n_clusters=best_k and fit_predict on X_scaled
km_best = ____  # Hint: KMeans(n_clusters=best_k, random_state=42, n_init=10)
km_labels = ____  # Hint: km_best.fit_predict(X_scaled)

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert 2 <= best_k <= 10, f"Best K should be between 2 and 10, got {best_k}"
assert max(sil_scores) > 0, "Best silhouette score should be positive"
assert len(set(km_labels)) == best_k, "K-means should produce exactly best_k clusters"
# INTERPRETATION: The elbow method in K-means is subjective; the silhouette
# score gives an objective criterion. Silhouette s(i) ∈ [-1, 1] measures
# how much closer point i is to its own cluster than to the nearest other
# cluster. High silhouette → compact, well-separated clusters.
print("\n✓ Checkpoint 2 passed — K-means optimal K selected by silhouette\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Spectral, HDBSCAN, GMM
# ══════════════════════════════════════════════════════════════════════

# Spectral clustering
n_spectral = best_k
# TODO: Create SpectralClustering with n_clusters=n_spectral, random_state=42, affinity="rbf"
spectral = ____  # Hint: SpectralClustering(n_clusters=n_spectral, ...)
# TODO: Fit and predict on a 10,000-sample slice (Spectral is O(n³))
spectral_labels = ____  # Hint: spectral.fit_predict(X_scaled[:10_000])

# HDBSCAN
if hdbscan is not None:
    # TODO: Create HDBSCAN with min_cluster_size=50, min_samples=10
    hdb = ____  # Hint: hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
    # TODO: Fit and predict labels
    hdbscan_labels = ____  # Hint: hdb.fit_predict(X_scaled)
    n_hdbscan_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
    noise_pct = (hdbscan_labels == -1).mean()
    print(f"\nHDBSCAN: {n_hdbscan_clusters} clusters, {noise_pct:.1%} noise")
else:
    hdbscan_labels = km_labels  # Fallback
    print("\nHDBSCAN not installed, using K-means labels as fallback")

# Gaussian Mixture Model
# TODO: Create GaussianMixture with n_components=best_k, random_state=42, covariance_type="full"
gmm = ____  # Hint: GaussianMixture(n_components=best_k, ...)
# TODO: Fit and predict hard cluster labels on X_scaled
gmm_labels = ____  # Hint: gmm.fit_predict(X_scaled)
# TODO: Get soft assignment probabilities from the fitted GMM
gmm_probs = ____  # Hint: gmm.predict_proba(X_scaled)

print(f"\nGMM: {best_k} components, BIC={gmm.bic(X_scaled):.0f}")
print(f"  Max assignment probability: {gmm_probs.max(axis=1).mean():.3f} (avg)")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(set(gmm_labels)) <= best_k, "GMM should not produce more clusters than specified"
assert gmm_probs.shape[1] == best_k, "GMM should produce probabilities for each component"
assert abs(gmm_probs.sum(axis=1).mean() - 1.0) < 1e-4, "GMM soft assignments should sum to 1"
# INTERPRETATION: GMM extends K-means by replacing hard cluster assignments
# with soft responsibilities: r_{ik} = P(cluster k | point i). This matters
# when clusters overlap — a customer near a segment boundary gets partial
# membership in both segments, which is more realistic than a hard label.
print("\n✓ Checkpoint 3 passed — GMM, spectral, HDBSCAN all fitted\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Evaluate all clustering methods
# ══════════════════════════════════════════════════════════════════════

results = {}
all_labels = {
    "K-means": km_labels,
    "Spectral": spectral_labels,
    "HDBSCAN": hdbscan_labels,
    "GMM": gmm_labels,
}

for name, labels in all_labels.items():
    # Filter noise points for HDBSCAN
    valid = labels != -1
    if valid.sum() < 2 or len(set(labels[valid])) < 2:
        continue

    data = X_scaled[: len(labels)]
    if not valid.all():
        data_clean = data[valid]
        labels_clean = labels[valid]
    else:
        data_clean = data
        labels_clean = labels

    # TODO: Compute all three clustering metrics for this method
    results[name] = {
        "n_clusters": len(set(labels_clean)),
        "silhouette": ____,  # Hint: silhouette_score(data_clean, labels_clean)
        "calinski_harabasz": ____,  # Hint: calinski_harabasz_score(data_clean, labels_clean)
        "davies_bouldin": ____,  # Hint: davies_bouldin_score(data_clean, labels_clean)
    }

print(f"\n=== Clustering Comparison ===")
print(f"{'Method':<12} {'K':>4} {'Silhouette':>12} {'CH':>12} {'DB':>8}")
print("─" * 52)
for name, r in results.items():
    print(
        f"{name:<12} {r['n_clusters']:>4} {r['silhouette']:>12.4f} "
        f"{r['calinski_harabasz']:>12.0f} {r['davies_bouldin']:>8.4f}"
    )

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert len(results) >= 2, "At least 2 clustering methods should be evaluated"
assert all("silhouette" in r for r in results.values()), \
    "All methods should report silhouette scores"
assert all(r["silhouette"] > -1.0 for r in results.values()), \
    "Silhouette scores should be > -1 (random assignment is 0)"
# INTERPRETATION: Davies-Bouldin measures cluster compactness vs separation
# (lower = better). Calinski-Harabasz measures between-cluster to within-cluster
# variance ratio (higher = better). Silhouette is [-1, 1] (higher = better).
# No single metric is perfect — use all three to triangulate the best K.
print("\n✓ Checkpoint 4 passed — all clustering methods evaluated on 3 metrics\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: AutoMLEngine comparison (agent=True double opt-in)
# ══════════════════════════════════════════════════════════════════════

# AutoMLEngine can automate the comparison above
# The agent=True flag enables LLM-guided algorithm selection
# max_llm_cost_usd caps the LLM usage — governance in action

from kailash_ml.engines.automl_engine import AutoMLEngine, AutoMLConfig


async def automl_comparison():
    """Use AutoMLEngine for automated clustering comparison."""

    # TODO: Create AutoMLConfig with task_type="clustering", metric_to_optimize="silhouette",
    #       direction="maximize", search_strategy="random", search_n_trials=20,
    #       agent=False, max_llm_cost_usd=1.0
    config = AutoMLConfig(
        task_type=____,  # Hint: "clustering"
        metric_to_optimize=____,  # Hint: "silhouette"
        direction=____,  # Hint: "maximize"
        search_strategy=____,  # Hint: "random"
        search_n_trials=____,  # Hint: 20
        # Agent guardrails — double opt-in pattern
        agent=____,  # Hint: False (set True + kailash-ml[agents] to enable)
        max_llm_cost_usd=____,  # Hint: 1.0 — hard budget cap
    )

    print(f"\n=== AutoMLEngine Config ===")
    print(f"Task: {config.task_type}")
    print(f"Metric: {config.metric_to_optimize}")
    print(f"Agent enabled: {config.agent}")
    print(f"LLM cost budget: ${config.max_llm_cost_usd}")
    print("\nNote: agent=True requires BOTH the flag AND kailash-ml[agents] installed.")
    print("This is the 'double opt-in' pattern — governance by design.")

    return config


config = asyncio.run(automl_comparison())

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert config is not None, "AutoMLEngine config should be created"
assert config.agent is False, "agent=False is the default (double opt-in for LLM)"
assert config.max_llm_cost_usd > 0, "LLM cost cap should be set"
# INTERPRETATION: The agent=True double opt-in pattern is governance by design.
# LLM-guided algorithm selection can be powerful but costs money and introduces
# non-determinism. Requiring BOTH the flag AND the kailash-ml[agents] package
# makes the cost and capability explicit — no accidental LLM calls.
print("\n✓ Checkpoint 5 passed — AutoMLEngine configured with governance guardrails\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Cluster profile analysis
# ══════════════════════════════════════════════════════════════════════

# Profile clusters using the best method
best_method = max(results.items(), key=lambda x: x[1]["silhouette"])
print(
    f"\nBest method: {best_method[0]} (silhouette={best_method[1]['silhouette']:.4f})"
)

best_labels = all_labels[best_method[0]]

# Add cluster labels to original data
clustered = customers.drop_nulls(subset=feature_cols).with_columns(
    pl.Series(
        "cluster", best_labels[: customers.drop_nulls(subset=feature_cols).height]
    )
)

# TODO: Profile each cluster — iterate over unique cluster IDs (skip -1),
#       for each: filter rows, print mean per feature vs overall mean
print(f"\n=== Cluster Profiles ===")
for cluster_id in sorted(clustered["cluster"].unique().to_list()):
    if cluster_id == -1:
        continue
    # TODO: Filter clustered to only rows where cluster == cluster_id
    subset = ____  # Hint: clustered.filter(pl.col("cluster") == cluster_id)
    print(f"\nCluster {cluster_id} (n={subset.height:,}):")
    for col in feature_cols[:6]:
        # TODO: Compute mean of col in this subset and in the full clustered df
        mean_val = ____  # Hint: subset[col].mean()
        overall_mean = ____  # Hint: clustered[col].mean()
        diff = (mean_val - overall_mean) / overall_mean * 100 if overall_mean else 0
        indicator = "▲" if diff > 10 else "▼" if diff < -10 else "─"
        print(f"  {col:<25} {mean_val:>10.2f} ({diff:+.1f}% vs avg) {indicator}")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Visualise
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# Method comparison
fig = viz.metric_comparison(results)
fig.update_layout(title="Clustering Method Comparison")
fig.write_html("ex1_clustering_comparison.html")
print("\nSaved: ex1_clustering_comparison.html")

# Elbow curve
elbow_data = {"Silhouette": sil_scores}
fig_elbow = viz.training_history(elbow_data, x_label="K")
fig_elbow.update_layout(title="K-means: Silhouette Score vs K")
fig_elbow.write_html("ex1_elbow.html")
print("Saved: ex1_elbow.html")

# ── Checkpoint 6 ─────────────────────────────────────────────────────
best_label = best_method[0]
assert best_label in results, "Best method must be in the results dict"
assert "cluster" in clustered.columns, "Clustered dataframe should have cluster column"
n_non_noise = clustered.filter(pl.col("cluster") >= 0).height
assert n_non_noise > 0, "At least some points should be assigned to valid clusters"
# INTERPRETATION: Cluster profiling converts statistical labels into business
# language. Cluster 0 is meaningless to a marketing manager; 'High-frequency
# low-spenders' is actionable. Always profile clusters before reporting results.
print("\n✓ Checkpoint 6 passed — cluster profiles computed\n")

print("\n✓ Exercise 1 complete — clustering comparison with AutoMLEngine")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(f"""
  ✓ K-means: elbow method + silhouette to select optimal K
  ✓ Spectral clustering: graph Laplacian for non-convex cluster shapes
  ✓ HDBSCAN: density-based, auto-detects noise points (label = -1)
  ✓ GMM: soft assignments via Gaussian mixture (extends K-means)
  ✓ Evaluation: silhouette (separation), DB (compactness), CH (ratio)
  ✓ AutoMLEngine: governance by design with agent double opt-in

  KEY INSIGHT: Clustering has no ground truth. You must use domain
  knowledge to decide: are these clusters meaningful? Do they correspond
  to real marketing segments, or are they statistical artefacts?

  ALGORITHM SELECTION GUIDE:
    K-means     → convex, spherical clusters, large datasets, fast
    Spectral    → non-convex (moons, rings), small datasets, slow
    HDBSCAN     → varying density, want noise detection, no K needed
    GMM         → overlapping clusters, need soft assignments

  NEXT: Exercise 2 digs into the mathematics of GMM — the EM algorithm.
  You'll implement the E-step and M-step by hand, proving that every
  iteration is guaranteed to improve the log-likelihood. This is the
  theoretical foundation for why GMM (and many other algorithms) work.
""")
print("═" * 70)
