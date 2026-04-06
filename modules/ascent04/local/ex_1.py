# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT04 — Exercise 1: Clustering Comparison with AutoMLEngine
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Compare K-means, spectral clustering, HDBSCAN, and GMM on
#   e-commerce customer data. Use AutoMLEngine for automated comparison.
#
# TASKS:
#   1. Load and prepare e-commerce customer features
#   2. Run K-means, spectral, HDBSCAN, GMM clustering
#   3. Evaluate with silhouette, Calinski-Harabasz, Davies-Bouldin
#   4. Use AutoMLEngine with agent=True double opt-in
#   5. Analyse cluster profiles
#   6. Visualise with ModelVisualizer
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

from shared import ASCENTDataLoader

try:
    import hdbscan
except ImportError:
    hdbscan = None


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
customers = loader.load("ascent04", "ecommerce_customers.parquet")

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

print("\n✓ Exercise 1 complete — clustering comparison with AutoMLEngine")
