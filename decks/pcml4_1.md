---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 4.1: Clustering

### Module 4: Advanced ML

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain when unsupervised learning is appropriate
- Apply K-means and HDBSCAN clustering algorithms
- Evaluate cluster quality with silhouette scores and elbow plots
- Use `AutoMLEngine` for automated clustering

---

## Recap: Module 3

- Built production ML pipelines: training, tuning, evaluation, persistence
- Gradient boosting with TrainingPipeline, HPO with HyperparameterSearch
- Explainability (SHAP/LIME), fairness, and calibration
- Workflow orchestration and DataFlow persistence

Module 4 moves beyond supervised learning.

---

## Supervised vs Unsupervised

```
Supervised:    "Here are examples with labels — learn the pattern"
               Input: (features, target) → Output: predictions
               Examples: price prediction, fraud detection

Unsupervised:  "Here is data with NO labels — find structure"
               Input: (features only) → Output: patterns, groups
               Examples: customer segments, anomaly detection
```

No "right answer" to evaluate against -- different challenge.

---

## Why Clustering?

```
10,000 HDB transactions — too many to understand individually.

Clustering groups similar transactions together:
  Cluster 1: "Budget central" — small flats, old lease, central
  Cluster 2: "Family suburban" — 4-5 room, long lease, suburban
  Cluster 3: "Premium mature" — large, mature estates, high price
```

Clustering reveals **natural segments** in the data.

---

## K-Means: The Workhorse

```
Algorithm:
  1. Pick K random centres
  2. Assign each point to nearest centre
  3. Move centres to mean of assigned points
  4. Repeat until stable

          ·  ·      ·  ·
        ·  ●  ·    ·  ●  ·
          ·  ·      ·  ·

     ·  ·         ·  ·
   ·  ●  ·      ·  ●  ·
     ·  ·         ·  ·

  ● = cluster centre    · = data points
```

---

## K-Means with AutoMLEngine

```python
from kailash_ml import AutoMLEngine

engine = AutoMLEngine()
engine.configure(
    dataset=df,
    task="clustering",
    algorithm="kmeans",
    params={
        "n_clusters": 5,
        "random_state": 42,
    },
)

result = engine.run()
df_clustered = result.labelled_data
print(f"Silhouette score: {result.metrics['silhouette']:.3f}")
```

---

## Choosing K: The Elbow Method

```
Inertia (within-cluster sum of squares)
  |╲
  | ╲
  |  ╲
  |    ╲
  |      ╲────────────────   ← Elbow here (diminishing returns)
  |
  └─────────────────────→ K
  1  2  3  4  5  6  7  8

Pick K at the "elbow" — where adding more clusters
stops improving significantly.
```

---

## Automated K Selection

```python
engine = AutoMLEngine()
engine.configure(
    dataset=df,
    task="clustering",
    algorithm="kmeans",
    params={
        "k_range": [2, 3, 4, 5, 6, 7, 8],
        "selection_metric": "silhouette",
    },
)

result = engine.run()
print(f"Optimal K: {result.best_k}")
print(f"Silhouette: {result.metrics['silhouette']:.3f}")
```

`AutoMLEngine` tests multiple K values and selects the best.

---

## Silhouette Score

Measures how similar a point is to its own cluster vs other clusters.

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))

a(i) = average distance to points in SAME cluster
b(i) = average distance to points in NEAREST other cluster

s = +1: perfectly assigned
s =  0: on the boundary
s = -1: probably in the wrong cluster
```

---

## HDBSCAN: Density-Based Clustering

K-means assumes spherical clusters. Real data is messier.

```
K-means sees 2 clusters:    HDBSCAN finds the shapes:
  ┌──────────────────┐        ┌──────────────────┐
  │  ···  ···        │        │  ···              │
  │ · ●·  · ●·      │        │ ·····  ····       │
  │  ···  ···        │        │  ···  ····        │
  │       ·····      │        │       ·····       │
  │      ·     ·     │        │      ·     ·      │
  │       ·····      │        │       ·····       │
  └──────────────────┘        └──────────────────┘
  (forced into circles)       (follows actual density)
```

---

## HDBSCAN Advantages

```python
engine = AutoMLEngine()
engine.configure(
    dataset=df,
    task="clustering",
    algorithm="hdbscan",
    params={
        "min_cluster_size": 50,
        "min_samples": 10,
    },
)

result = engine.run()
print(f"Clusters found: {result.n_clusters}")
print(f"Noise points: {result.n_noise}")
```

HDBSCAN advantages:

- No need to specify K (finds it automatically)
- Identifies noise points (outliers)
- Finds non-spherical clusters
- Handles varying densities

---

## K-Means vs HDBSCAN

| Aspect         | K-Means                               | HDBSCAN                         |
| -------------- | ------------------------------------- | ------------------------------- |
| Cluster shape  | Spherical                             | Arbitrary                       |
| Must specify K | Yes                                   | No                              |
| Noise handling | No                                    | Yes (labels outliers as -1)     |
| Speed          | Very fast                             | Moderate                        |
| Scalability    | Excellent                             | Good                            |
| Deterministic  | No (random init)                      | Yes                             |
| Best for       | Well-separated, similar-size clusters | Irregular, density-varying data |

---

## Interpreting Clusters

```python
import polars as pl

# Add cluster labels to data
df_result = df.with_columns(
    pl.Series("cluster", result.labels)
)

# Profile each cluster
cluster_profiles = df_result.group_by("cluster").agg(
    pl.col("price").mean().alias("avg_price"),
    pl.col("floor_area").mean().alias("avg_area"),
    pl.col("lease_years").mean().alias("avg_lease"),
    pl.col("price").count().alias("count"),
)
print(cluster_profiles.sort("cluster"))
```

Give clusters **meaningful names** based on their profiles.

---

## Visualising Clusters

```python
from kailash_ml import ModelVisualizer

viz = ModelVisualizer()
viz.configure(theme="professional")

# 2D scatter with cluster colours
viz.plot_clusters(
    data=df_result,
    x="floor_area",
    y="price",
    cluster_col="cluster",
    title="HDB Market Segments",
)
```

---

## Exercise Preview

**Exercise 4.1: HDB Market Segmentation**

You will:

1. Apply K-means with elbow method and silhouette analysis
2. Apply HDBSCAN and compare cluster quality
3. Profile and name each cluster meaningfully
4. Use `AutoMLEngine` for automated cluster selection

Scaffolding level: **Light+ (~40% code provided)**

---

## Common Pitfalls

| Mistake                                | Fix                                          |
| -------------------------------------- | -------------------------------------------- |
| Not scaling features before clustering | K-means is distance-based -- scale first     |
| Choosing K by intuition                | Use elbow method and silhouette score        |
| Ignoring noise points in HDBSCAN       | Noise (label -1) may be interesting outliers |
| Clusters without interpretation        | Always profile and name clusters             |
| Using K-means on non-spherical data    | Try HDBSCAN instead                          |

---

## Summary

- Unsupervised learning finds structure in unlabelled data
- K-means is fast and effective for well-separated spherical clusters
- HDBSCAN handles arbitrary shapes and identifies noise
- `AutoMLEngine` automates algorithm selection and parameter tuning
- Always interpret clusters with domain knowledge

---

## Next Lesson

**Lesson 4.2: EM Algorithm and GMMs**

We will learn:

- The Expectation-Maximisation algorithm
- Gaussian Mixture Models for soft clustering
- When GMMs outperform K-means
