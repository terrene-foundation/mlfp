---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 4.3: Dimensionality Reduction

### Module 4: Advanced ML

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Apply PCA for linear dimensionality reduction and feature compression
- Use UMAP and t-SNE for non-linear visualisation of high-dimensional data
- Choose the right technique based on the goal (compression vs visualisation)
- Interpret explained variance and embedding quality

---

## Recap: Lesson 4.2

- GMMs provide soft (probabilistic) cluster assignments
- EM alternates E-step (responsibilities) and M-step (parameter updates)
- BIC/AIC select the number of components
- EM is a general algorithm for hidden-variable models

---

## The Curse of Dimensionality

```
As dimensions increase:
  - Distances between points become similar (everything is "far")
  - Data becomes sparse (10 points in 100D = mostly empty space)
  - Models need exponentially more data to generalise
  - Visualisation becomes impossible (we see 2-3 dimensions)

50 features → 25 engineered → 200 interactions → too many!
```

Dimensionality reduction compresses information into fewer dimensions.

---

## Two Goals, Two Approaches

```
Goal 1: COMPRESSION (keep information, fewer features)
  → PCA: linear, fast, preserves variance
  → Use BEFORE modelling to reduce feature count

Goal 2: VISUALISATION (see structure in 2D/3D)
  → UMAP: preserves local + global structure
  → t-SNE: preserves local structure
  → Use for exploration, NOT as model input
```

---

## PCA: Principal Component Analysis

Find new axes (principal components) that capture maximum variance.

```
Original:                   After PCA:
  y │  · ·· ·                PC2 │  · ·
    │ · · ·· ·                   │ ··
    │· · ·· · · ·               │··    (most spread along PC1)
    │ · · · ·                    │·
    └──────────→ x               └──────────→ PC1

PC1 = direction of maximum variance
PC2 = perpendicular direction, next most variance
```

---

## PCA in Code

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import polars as pl

# Always scale before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.select(feature_cols).to_numpy())

# Fit PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# How much variance is captured?
for i, var in enumerate(pca.explained_variance_ratio_):
    cumulative = sum(pca.explained_variance_ratio_[:i+1])
    print(f"PC{i+1}: {var:.1%} (cumulative: {cumulative:.1%})")
```

---

## Explained Variance Plot

```
Cumulative Explained Variance
  100% │                    ─────────
       │               ╱───
       │           ╱──
   90% │        ╱─         ← 90% with 8 components
       │      ╱               (from 50 original features)
       │    ╱
       │  ╱
       │╱
    0% └──────────────────────→ # Components
       1  5  10  15  20  30  50
```

Rule of thumb: keep enough components for 90-95% variance.

---

## PCA for Compression

```python
# Choose components to keep 95% variance
pca_95 = PCA(n_components=0.95)
X_compressed = pca_95.fit_transform(X_scaled)

print(f"Original: {X_scaled.shape[1]} features")
print(f"Compressed: {X_compressed.shape[1]} features")
print(f"Variance kept: 95%")

# Use compressed features for modelling
from kailash_ml import TrainingPipeline

pipeline = TrainingPipeline()
pipeline.configure(
    dataset=pl.DataFrame(X_compressed),
    target_column="price",
    algorithm="lightgbm",
)
```

---

## PCA Loadings: What Do Components Mean?

```python
import polars as pl

loadings = pl.DataFrame(
    pca.components_[:3],
    schema=feature_cols,
)

# PC1's top contributing features
pc1 = loadings.row(0, named=True)
sorted_pc1 = sorted(pc1.items(), key=lambda x: abs(x[1]), reverse=True)
for feat, loading in sorted_pc1[:5]:
    print(f"  {feat}: {loading:+.3f}")
```

Loadings tell you what each component represents in terms of original features.

---

## t-SNE: Visualising Structure

t-SNE maps high-dimensional data to 2D, preserving **local neighbourhood** structure.

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Plot with cluster colours
import plotly.express as px
fig = px.scatter(
    x=X_tsne[:, 0], y=X_tsne[:, 1],
    color=cluster_labels,
    title="t-SNE Visualisation of HDB Clusters",
)
fig.show()
```

---

## t-SNE Caveats

```
⚠ Distances between clusters are MEANINGLESS
  → Only local structure is preserved
  → Two far-apart clusters may actually be similar

⚠ Perplexity matters
  → Low (5-10): tight, many small clusters
  → Medium (30): balanced (default)
  → High (50-100): more global structure

⚠ Non-deterministic
  → Different runs produce different layouts
  → Set random_state for reproducibility

⚠ Not for model input
  → Use for visualisation ONLY
```

---

## UMAP: Best of Both Worlds

UMAP preserves both local AND global structure. Faster than t-SNE.

```python
import umap

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                     random_state=42)
X_umap = reducer.fit_transform(X_scaled)

fig = px.scatter(
    x=X_umap[:, 0], y=X_umap[:, 1],
    color=cluster_labels,
    title="UMAP Visualisation of HDB Clusters",
)
fig.show()
```

---

## PCA vs t-SNE vs UMAP

| Aspect            | PCA             | t-SNE            | UMAP             |
| ----------------- | --------------- | ---------------- | ---------------- |
| Type              | Linear          | Non-linear       | Non-linear       |
| Preserves         | Global variance | Local structure  | Local + global   |
| Speed             | Very fast       | Slow             | Fast             |
| Deterministic     | Yes             | No               | Approximately    |
| Inverse transform | Yes             | No               | Yes              |
| Use for modelling | Yes             | No               | Cautiously       |
| Best for          | Compression     | 2D visualisation | 2D visualisation |

---

## Choosing a Method

```
What is your goal?
│
├─ Reduce features for a model?
│  └→ PCA (keep 90-95% variance)
│
├─ Visualise clusters in 2D?
│  └→ UMAP (preserves more structure than t-SNE)
│
├─ Explore local neighbourhoods?
│  └→ t-SNE (best local preservation)
│
└─ Interpret which features matter?
   └→ PCA loadings (components have meaning)
```

---

## Exercise Preview

**Exercise 4.3: Dimensionality Reduction for HDB Data**

You will:

1. Apply PCA and select components using explained variance
2. Interpret PCA loadings for the top components
3. Visualise clusters with UMAP and t-SNE, comparing results
4. Use PCA-compressed features in a model and compare performance

Scaffolding level: **Light+ (~40% code provided)**

---

## Common Pitfalls

| Mistake                                      | Fix                                                 |
| -------------------------------------------- | --------------------------------------------------- |
| Not scaling before PCA                       | PCA is variance-based -- unscaled features dominate |
| Using t-SNE output as model features         | t-SNE is for visualisation only                     |
| Interpreting t-SNE distances as meaningful   | Only local structure is preserved                   |
| Keeping too few/many PCA components          | Use explained variance ratio (90-95%)               |
| Running t-SNE on large data without sampling | Sample first or use UMAP (faster)                   |

---

## Summary

- PCA compresses features by finding maximum-variance axes (linear)
- t-SNE preserves local neighbourhoods for 2D visualisation (non-linear)
- UMAP preserves local and global structure, faster than t-SNE
- Use PCA for compression before modelling; UMAP/t-SNE for exploration
- Always scale data before any dimensionality reduction

---

## Next Lesson

**Lesson 4.4: Anomaly Detection and Ensembles**

We will learn:

- Isolation Forest and LOF for anomaly detection
- `EnsembleEngine` for combining multiple models
- Stacking, blending, and voting strategies
