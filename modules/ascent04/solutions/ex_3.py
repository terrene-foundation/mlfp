# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT04 — Exercise 3: Dimensionality Reduction
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Master PCA from SVD fundamentals — reconstruction error,
#   scree plots, loadings — then apply UMAP and t-SNE for nonlinear
#   embedding. Understand when each method is appropriate.
#
# TASKS:
#   1. PCA via SVD — explained variance, reconstruction error, loadings
#   2. Scree plot and cumulative variance — choosing n_components
#   3. PCA loadings — which features drive each principal component
#   4. Reconstruction error as a function of retained components
#   5. t-SNE — local structure, perplexity hyperparameter
#   6. UMAP — global structure, hyperparameter tuning, out-of-sample
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from kailash_ml import ModelVisualizer
from kailash_ml.interop import to_sklearn_input

from shared import ASCENTDataLoader

try:
    import umap as umap_lib
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("umap-learn not installed — UMAP tasks will use PCA fallback")


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
customers = loader.load("ascent04", "ecommerce_customers.parquet")

feature_cols = [
    c
    for c, d in zip(customers.columns, customers.dtypes)
    if d in (pl.Float64, pl.Float32, pl.Int64, pl.Int32) and c not in ("customer_id",)
]

X_raw, _, col_info = to_sklearn_input(
    customers.drop_nulls(subset=feature_cols),
    feature_columns=feature_cols,
)

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

n_samples, n_features = X.shape
print(f"=== E-commerce Customer Data ===")
print(f"Samples: {n_samples:,}, Features: {n_features}")
print(f"Feature names: {feature_cols}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: PCA via SVD — the connection explained
# ══════════════════════════════════════════════════════════════════════
# PCA finds directions of maximum variance (principal components).
# It is equivalent to computing the Singular Value Decomposition (SVD):
#
#   X = U S V'
#
# where:
#   U : (n, n) — left singular vectors (sample coordinates)
#   S : (n, p) — diagonal matrix of singular values σ_1 ≥ σ_2 ≥ ... ≥ 0
#   V : (p, p) — right singular vectors (principal directions in feature space)
#
# Connection to PCA:
#   - Columns of V = principal component directions (loadings)
#   - Scores (projected data) = U S  or equivalently  X V
#   - Explained variance for PC_k = σ_k² / (n - 1)
#   - Total variance = Σ_k σ_k² / (n - 1) = Σ_j Var(X_j)

print(f"\n=== PCA via SVD ===")

# Manual SVD on mean-centred data (X already standardised = mean-centred + scaled)
U, S, Vt = np.linalg.svd(X, full_matrices=False)

# Explained variance from singular values
explained_variance = S**2 / (n_samples - 1)
total_variance = explained_variance.sum()
explained_variance_ratio = explained_variance / total_variance
cumulative_evr = np.cumsum(explained_variance_ratio)

print(f"Total variance (should ≈ n_features={n_features}): {total_variance:.2f}")
print(f"\nTop 10 Principal Components:")
print(f"{'PC':>4} {'Singular Value':>16} {'Expl. Var':>12} {'Expl. Var %':>12} {'Cumulative %':>14}")
print("─" * 62)
for i in range(min(10, n_features)):
    print(
        f"{i+1:>4} {S[i]:>16.4f} {explained_variance[i]:>12.4f} "
        f"{explained_variance_ratio[i]:>11.2%} {cumulative_evr[i]:>13.2%}"
    )

# Verify against sklearn PCA
pca_full = PCA(n_components=n_features)
pca_full.fit(X)
max_diff = np.abs(pca_full.explained_variance_ratio_ - explained_variance_ratio).max()
print(f"\nSVD vs sklearn PCA max difference: {max_diff:.2e} (should be ≈ 0)")

# Principal directions (loadings): rows of Vt = columns of V
# PC_k direction = Vt[k, :] — a unit vector in feature space
print(f"\nPC1 direction (first {n_features} values):")
print(f"  {Vt[0].round(3)}")
print(f"  ||PC1|| = {np.linalg.norm(Vt[0]):.6f} (should be 1.0 — unit vector)")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Scree plot and choosing n_components
# ══════════════════════════════════════════════════════════════════════
# The scree plot shows explained variance ratio per component.
# "Elbow" = the point where marginal variance explained drops sharply.
# A common rule: retain components until cumulative variance ≥ 90-95%.

n_95 = np.searchsorted(cumulative_evr, 0.95) + 1
n_90 = np.searchsorted(cumulative_evr, 0.90) + 1
n_80 = np.searchsorted(cumulative_evr, 0.80) + 1

print(f"\n=== Variance Thresholds ===")
print(f"Components for 80% variance: {n_80}")
print(f"Components for 90% variance: {n_90}")
print(f"Components for 95% variance: {n_95}")
print(f"  (Original: {n_features} features → {n_95}x compression for 95% retention)")

# Kaiser criterion: retain components with eigenvalue > 1
# (only meaningful for standardised data where total variance = n_features)
n_kaiser = (explained_variance > 1.0).sum()
print(f"\nKaiser criterion (eigenvalue > 1): {n_kaiser} components")
print(f"  Eigenvalue = explained variance per component")
print(f"  Threshold 1.0 means the component captures more than one original feature")

viz = ModelVisualizer()

# Scree plot
fig_scree = viz.training_history(
    {
        "Explained Variance %": (explained_variance_ratio[:20] * 100).tolist(),
        "Cumulative %": (cumulative_evr[:20] * 100).tolist(),
    },
    x_label="Principal Component",
)
fig_scree.update_layout(title="Scree Plot: Explained Variance by Component")
fig_scree.write_html("ex3_scree_plot.html")
print("\nSaved: ex3_scree_plot.html")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: PCA loadings — feature contributions to each PC
# ══════════════════════════════════════════════════════════════════════
# Loadings = correlation between original features and principal components.
# Loading[j, k] = Vt[k, j] * sqrt(explained_variance[k]) / std(X_j)
# (for standardised data this simplifies to Vt[k, j] * sqrt(λ_k))
#
# |Loading| close to 1 → feature j strongly drives PC_k
# |Loading| close to 0 → feature j barely contributes to PC_k

loadings = Vt[:n_components_to_inspect := min(5, n_features), :].T  # (n_features, n_pcs)

print(f"\n=== PCA Loadings (top {n_components_to_inspect} PCs) ===")
print(f"{'Feature':<30}", end="")
for i in range(n_components_to_inspect):
    print(f"{'PC' + str(i+1):>10}", end="")
print()
print("─" * (30 + 10 * n_components_to_inspect))

for j, feat in enumerate(feature_cols):
    print(f"{feat:<30}", end="")
    for i in range(n_components_to_inspect):
        val = loadings[j, i]
        marker = " ★" if abs(val) > 0.4 else "  "
        print(f"{val:>9.3f}{marker[1]}", end="")
    print()

print("\n★ = strong loading (|loading| > 0.4)")
print("\nInterpretation:")
print("  PC1: the direction capturing most customer variation")
print("  Features with large |loading| on PC1 are the primary drivers")
print("  Features with near-zero loading on PC1 are orthogonal to it")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Reconstruction error as a function of retained components
# ══════════════════════════════════════════════════════════════════════
# Reconstruction: X̂ = X_proj @ Vt[:k, :]  (project and back-project)
# Reconstruction MSE = ||X - X̂||² / (n * p)
# = Σ_{i > k} σ_i² / (n * p)  (exactly the unexplained variance)

n_components_range = list(range(1, min(n_features + 1, 31)))
reconstruction_errors = []

for k in n_components_range:
    pca_k = PCA(n_components=k)
    X_proj = pca_k.fit_transform(X)
    X_recon = pca_k.inverse_transform(X_proj)
    mse = np.mean((X - X_recon) ** 2)
    reconstruction_errors.append(mse)

print(f"\n=== Reconstruction Error ===")
print(f"{'Components':>12} {'MSE':>12} {'% Variance Retained':>22}")
print("─" * 50)
for k, mse in zip(n_components_range[::3], reconstruction_errors[::3]):
    pct_retained = 1.0 - mse / np.mean(X ** 2)
    print(f"{k:>12} {mse:>12.4f} {pct_retained:>21.2%}")

fig_recon = viz.training_history(
    {"Reconstruction MSE": reconstruction_errors},
    x_label="Number of PCA Components",
)
fig_recon.update_layout(title="PCA: Reconstruction Error vs Components Retained")
fig_recon.write_html("ex3_reconstruction_error.html")
print("\nSaved: ex3_reconstruction_error.html")

# Recommended n_components for downstream tasks
n_for_embedding = n_90  # Retain 90% variance before applying t-SNE/UMAP
print(f"\nUsing {n_for_embedding} PCA components (90% variance) as t-SNE/UMAP input")
print("  Pre-reducing with PCA improves t-SNE/UMAP speed and removes noise")

pca_pre = PCA(n_components=n_for_embedding, random_state=42)
X_pca = pca_pre.fit_transform(X)
print(f"  Reduced: {X.shape} → {X_pca.shape}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: t-SNE — local structure, perplexity
# ══════════════════════════════════════════════════════════════════════
# t-SNE minimises KL divergence between:
#   - High-dimensional pairwise similarity (Gaussian kernel)
#   - Low-dimensional pairwise similarity (Student-t kernel)
#
# Properties:
#   - Preserves LOCAL structure (nearby points stay nearby)
#   - Global distances are NOT preserved (clusters can appear far apart)
#   - No out-of-sample extension (must refit for new points)
#   - O(n log n) with Barnes-Hut approximation
#
# Key hyperparameter: perplexity
#   - Roughly: effective number of nearest neighbours
#   - Low (5-10): focuses on very local structure, isolates small clusters
#   - High (50-100): more global perspective, smoother embeddings
#   - Rule of thumb: 5 ≤ perplexity ≤ n/3

# Subsample for speed (t-SNE is slow on large datasets)
rng = np.random.default_rng(42)
n_tsne = min(3000, n_samples)
idx_tsne = rng.choice(n_samples, n_tsne, replace=False)
X_tsne_input = X_pca[idx_tsne]

print(f"\n=== t-SNE (n={n_tsne}) ===")
print(f"{'Perplexity':>12} {'KL Divergence':>16} {'Time':>8}")
print("─" * 40)

import time
tsne_results = {}
for perplexity in [5, 30, 50]:
    t0 = time.time()
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=1000,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    embedding = tsne.fit_transform(X_tsne_input)
    elapsed = time.time() - t0

    tsne_results[perplexity] = {
        "embedding": embedding,
        "kl_divergence": tsne.kl_divergence_,
    }

    # Cluster quality in the embedding
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=4, random_state=42, n_init=5)
    labels_2d = km.fit_predict(embedding)
    sil = silhouette_score(embedding, labels_2d) if len(set(labels_2d)) > 1 else -1.0

    print(f"{perplexity:>12} {tsne.kl_divergence_:>16.4f} {elapsed:>7.1f}s")
    print(f"  Silhouette in 2D embedding: {sil:.4f}")

print("\nt-SNE perplexity guidance:")
print("  perplexity=5  : micro-clusters, many small tight groups")
print("  perplexity=30 : balanced (default recommendation)")
print("  perplexity=50 : smoother, fewer isolated clusters")
print("  NEVER interpret inter-cluster distances — not meaningful!")

# Key t-SNE warnings
print("\nt-SNE pitfalls:")
print("  1. Cluster sizes in t-SNE do NOT reflect real cluster sizes")
print("  2. Distances between clusters are NOT meaningful")
print("  3. Different runs give different layouts (random seed matters)")
print("  4. No out-of-sample extension — new points require full refit")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: UMAP — global structure, hyperparameter tuning
# ══════════════════════════════════════════════════════════════════════
# UMAP uses a fuzzy topological approach:
#   1. Build a weighted k-NN graph (high-dimensional)
#   2. Optimise a low-dimensional layout to match graph structure
#
# Advantages over t-SNE:
#   - Preserves BOTH local AND global structure
#   - Supports out-of-sample transform (transform new points)
#   - Faster: O(n) amortised
#   - Can embed into any dimensionality (not just 2D)
#
# Key hyperparameters:
#   n_neighbors: size of local neighbourhood (like perplexity in t-SNE)
#     - Small: focuses on local structure, captures fine detail
#     - Large: more global structure, smoother embedding
#   min_dist: minimum distance between points in embedding
#     - Small (0.0): tightly packed clusters
#     - Large (1.0): spread out, more continuous

print(f"\n=== UMAP Hyperparameter Comparison ===")

if UMAP_AVAILABLE:
    umap_configs = [
        {"n_neighbors": 5,  "min_dist": 0.1,  "label": "local (n_nbrs=5)"},
        {"n_neighbors": 15, "min_dist": 0.1,  "label": "default (n_nbrs=15)"},
        {"n_neighbors": 50, "min_dist": 0.5,  "label": "global (n_nbrs=50)"},
    ]

    umap_results = {}
    for cfg in umap_configs:
        t0 = time.time()
        reducer = umap_lib.UMAP(
            n_components=2,
            n_neighbors=cfg["n_neighbors"],
            min_dist=cfg["min_dist"],
            random_state=42,
            metric="euclidean",
        )
        # Fit on subsample, then transform full dataset (out-of-sample extension)
        reducer.fit(X_pca[idx_tsne])
        embedding_full = reducer.transform(X_pca)   # All samples
        elapsed = time.time() - t0

        km_labels = KMeans(n_clusters=4, random_state=42, n_init=5).fit_predict(
            embedding_full
        )
        sil = silhouette_score(embedding_full, km_labels) if len(set(km_labels)) > 1 else -1.0

        umap_results[cfg["label"]] = {
            "embedding": embedding_full,
            "silhouette": sil,
            "time": elapsed,
        }
        print(f"  {cfg['label']:<30}: silhouette={sil:.4f}, time={elapsed:.1f}s")

    print("\nOut-of-sample: UMAP supports reducer.transform(new_X)")
    print("  This is critical for production — new customers can be embedded")
    print("  without refitting the entire model")
    print("\nUMAP vs t-SNE decision guide:")
    print("  Use t-SNE:  exploratory visualisation, understanding local clusters")
    print("  Use UMAP:   need out-of-sample transform, large datasets, global structure")

else:
    # PCA fallback for environments without umap-learn
    pca_2d = PCA(n_components=2, random_state=42)
    embedding_2d = pca_2d.fit_transform(X_pca)
    km_labels = KMeans(n_clusters=4, random_state=42, n_init=5).fit_predict(embedding_2d)
    sil = silhouette_score(embedding_2d, km_labels) if len(set(km_labels)) > 1 else -1.0
    print(f"  PCA 2D fallback: silhouette={sil:.4f}")

    umap_results = {"PCA 2D": {"embedding": embedding_2d, "silhouette": sil, "time": 0.0}}

    print("\nInstall umap-learn to run UMAP: pip install umap-learn")
    print("UMAP benefits: preserves global structure, out-of-sample transform")


# ══════════════════════════════════════════════════════════════════════
# Method comparison summary
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Dimensionality Reduction Method Comparison ===")
print(f"""
┌──────────────────┬─────────────┬──────────────┬───────────────┬──────────────┐
│ Method           │ Linear?     │ Global Struct│ Out-of-Sample │ Speed        │
├──────────────────┼─────────────┼──────────────┼───────────────┼──────────────┤
│ PCA              │ Yes         │ Yes          │ Yes           │ O(np min(n,p))│
│ t-SNE            │ No          │ Local only   │ No            │ O(n log n)   │
│ UMAP             │ No          │ Both         │ Yes           │ O(n)         │
└──────────────────┴─────────────┴──────────────┴───────────────┴──────────────┘

Practical Recommendations:
  1. Always PCA first: removes noise, speeds up t-SNE/UMAP
  2. PCA for production: interpretable loadings, fast, invertible
  3. t-SNE for exploration: visualise cluster structure in 2D
  4. UMAP for production embedding: out-of-sample transform available
  5. Report n_components chosen and % variance retained
""")

# Visualise PCA 2D projection
pca_2d_final = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d_final.fit_transform(X)

fig_pca2d = viz.training_history(
    {"PC2 vs PC1": X_pca_2d[:, 1].tolist()},
    x_label="PC1 Score",
)
fig_pca2d.update_layout(title=f"PCA 2D Projection ({explained_variance_ratio[:2].sum():.1%} variance)")
fig_pca2d.write_html("ex3_pca_2d.html")

# Method silhouette comparison
method_silhouettes = {}
for perp, res in tsne_results.items():
    km_l = KMeans(n_clusters=4, random_state=42, n_init=5).fit_predict(res["embedding"])
    method_silhouettes[f"t-SNE p={perp}"] = {"Silhouette": silhouette_score(res["embedding"], km_l)}

for label, res in umap_results.items():
    method_silhouettes[f"UMAP {label}"] = {"Silhouette": res["silhouette"]}

km_pca_labels = KMeans(n_clusters=4, random_state=42, n_init=5).fit_predict(X_pca)
method_silhouettes[f"PCA {n_for_embedding}d"] = {
    "Silhouette": silhouette_score(X_pca, km_pca_labels)
}

fig_methods = viz.metric_comparison(method_silhouettes)
fig_methods.update_layout(title="Dimensionality Reduction: 2D Cluster Quality Comparison")
fig_methods.write_html("ex3_method_comparison.html")
print("Saved: ex3_scree_plot.html, ex3_reconstruction_error.html")
print("Saved: ex3_pca_2d.html, ex3_method_comparison.html")

print("\n✓ Exercise 3 complete — PCA via SVD + t-SNE + UMAP")
print("  Key takeaways:")
print("  1. PCA = SVD; explained variance = σ_k²/(n-1); scree plot → choose k")
print("  2. Loadings reveal which features drive each principal component")
print("  3. Reconstruction error = unexplained variance = Σ_{i>k} σ_i²")
print("  4. t-SNE: local structure only, no out-of-sample transform")
print("  5. UMAP: global + local, supports transform() — production-ready")
