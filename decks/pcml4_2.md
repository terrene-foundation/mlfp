---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 4.2: EM Algorithm and GMMs

### Module 4: Advanced ML

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain the Expectation-Maximisation (EM) algorithm step by step
- Apply Gaussian Mixture Models for soft (probabilistic) clustering
- Compare GMMs to K-means and choose appropriately
- Use model selection criteria (BIC/AIC) to choose the number of components

---

## Recap: Lesson 4.1

- K-means clusters by minimising within-cluster distances
- HDBSCAN finds arbitrary-shaped, density-based clusters
- `AutoMLEngine` automates clustering algorithm and K selection
- Always profile and interpret clusters with domain knowledge

---

## K-Means Limitation: Hard Assignment

```
K-means: each point belongs to EXACTLY one cluster.

But what about this transaction?
  Price: $490k, Floor area: 95 sqm, Town: QUEENSTOWN

  60% similar to "Family suburban" cluster
  40% similar to "Premium mature" cluster

K-means: forced into one. Information lost.
GMM:     belongs 60/40. Uncertainty preserved.
```

---

## Gaussian Mixture Models

A GMM assumes data comes from a mixture of K Gaussian distributions.

```
P(x) = Σ πₖ · N(x | μₖ, Σₖ)

Where:
  πₖ = mixing weight (how big is cluster k)
  μₖ = mean of cluster k
  Σₖ = covariance of cluster k (shape and orientation)
  K  = number of components
```

Each data point has a **probability** of belonging to each cluster.

---

## GMM vs K-Means Visual

```
K-means:                    GMM:
  Hard boundaries             Soft boundaries (probability)
  ┌─────┬────��┐              ┌────────────────┐
  │ A A │ B B │              │ A  AB   B      │
  │ A A │ B B │              │ A  AB   B      │
  │ A A │ B B │              │ A  AB   B      │
  └─────┴─────┘              └────────────────┘
  Sharp line                  Gradient transition
  between clusters            (uncertainty quantified)
```

---

## The EM Algorithm

EM finds the best GMM parameters by alternating two steps:

```
E-step (Expectation):
  "Given current parameters, compute probability
   each point belongs to each cluster"

M-step (Maximisation):
  "Given these probabilities, update cluster
   parameters (means, covariances, weights)"

Repeat until convergence.
```

---

## EM Step by Step

```
Initialise: Random μ₁, μ₂, Σ₁, Σ₂, π₁, π₂

E-step:
  For each point xᵢ:
    r(i,1) = π₁·N(xᵢ|μ₁,Σ₁) / [π₁·N(xᵢ|μ₁,Σ₁) + π₂·N(xᵢ|μ₂,Σ₂)]
    r(i,2) = 1 - r(i,1)

  r(i,k) = "responsibility" of cluster k for point i

M-step:
  μₖ = Σᵢ r(i,k)·xᵢ / Σᵢ r(i,k)     (weighted mean)
  Σₖ = weighted covariance
  πₖ = Σᵢ r(i,k) / N                   (fraction of points)

Repeat until log-likelihood converges.
```

---

## EM Convergence Visual

```
Log-likelihood
  |                     ────────── converged
  |                ╱───
  |            ╱──
  |         ╱─
  |       ╱
  |     ╱
  |   ╱
  |  ╱
  | ╱
  └──────────────────────→ Iteration
  1   5   10   15   20

EM guarantees the log-likelihood never decreases.
(But may find a local maximum, not global.)
```

---

## GMM in Code

```python
from sklearn.mixture import GaussianMixture
import polars as pl
import numpy as np

# Prepare features
features = ["price", "floor_area", "lease_years", "storey"]
X = df.select(features).to_numpy()

# Scale first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit GMM
gmm = GaussianMixture(n_components=4, random_state=42)
gmm.fit(X_scaled)

# Soft assignments (probabilities)
probs = gmm.predict_proba(X_scaled)
labels = gmm.predict(X_scaled)
```

---

## Soft Assignments: The GMM Advantage

```python
# Point 0's cluster probabilities
print(f"Point 0 belongs to:")
for k in range(4):
    print(f"  Cluster {k}: {probs[0, k]:.1%}")
```

```
Point 0 belongs to:
  Cluster 0: 62.3%
  Cluster 1: 31.5%
  Cluster 2:  5.8%
  Cluster 3:  0.4%
```

Soft assignments reveal **boundary cases** and **mixed-profile** observations.

---

## Choosing K: BIC and AIC

```python
bic_scores = []
aic_scores = []

for k in range(2, 10):
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_scaled)
    bic_scores.append({"k": k, "bic": gmm.bic(X_scaled)})
    aic_scores.append({"k": k, "aic": gmm.aic(X_scaled)})

bic_df = pl.DataFrame(bic_scores).sort("bic")
print(bic_df)
# Lower BIC/AIC = better model
```

BIC penalises complexity more than AIC. Prefer BIC for cluster selection.

---

## Covariance Types

```python
# Full: each cluster has its own shape and orientation
gmm_full = GaussianMixture(n_components=4, covariance_type="full")

# Tied: all clusters share the same covariance
gmm_tied = GaussianMixture(n_components=4, covariance_type="tied")

# Diagonal: axis-aligned ellipses (no rotation)
gmm_diag = GaussianMixture(n_components=4, covariance_type="diag")

# Spherical: like K-means (circular clusters)
gmm_sph = GaussianMixture(n_components=4, covariance_type="spherical")
```

```
Full:      Tied:      Diagonal:   Spherical:
 ╱╲        ╱╲         │ │         ○
╱  ╲      ╱  ╲        │ │         ○
(any)     (same shape) (no tilt)  (= K-means)
```

---

## EM Beyond GMMs

EM is a **general-purpose** algorithm for models with hidden variables:

| Application             | Hidden Variable    |
| ----------------------- | ------------------ |
| GMM clustering          | Cluster membership |
| Topic modelling (LDA)   | Topic assignments  |
| Hidden Markov Models    | Hidden states      |
| Missing data imputation | Missing values     |
| Factor analysis         | Latent factors     |

The E-step/M-step pattern applies to all of these.

---

## GMM with AutoMLEngine

```python
from kailash_ml import AutoMLEngine

engine = AutoMLEngine()
engine.configure(
    dataset=df,
    task="clustering",
    algorithm="gmm",
    params={
        "k_range": [2, 3, 4, 5, 6, 7],
        "covariance_type": "full",
        "selection_metric": "bic",
    },
)

result = engine.run()
print(f"Optimal K: {result.best_k}")
print(f"BIC: {result.metrics['bic']:,.0f}")
```

---

## Exercise Preview

**Exercise 4.2: Soft Clustering with GMMs**

You will:

1. Fit GMMs with different K values and select using BIC
2. Compare soft (GMM) vs hard (K-means) cluster assignments
3. Identify boundary cases using probability thresholds
4. Experiment with different covariance types

Scaffolding level: **Light+ (~40% code provided)**

---

## Common Pitfalls

| Mistake                                  | Fix                                         |
| ---------------------------------------- | ------------------------------------------- |
| Not scaling features before GMM          | Scale first -- EM is sensitive to magnitude |
| Ignoring local optima                    | Run multiple random initialisations         |
| Using AIC instead of BIC for K selection | BIC is more conservative and preferred      |
| Full covariance on small data            | Use diagonal or tied to avoid overfitting   |
| Treating soft assignments as hard        | Leverage the probabilities for uncertainty  |

---

## Summary

- GMMs model data as a mixture of Gaussian distributions
- EM alternates E-step (responsibilities) and M-step (parameter updates)
- Soft assignments preserve uncertainty about cluster membership
- BIC/AIC select the number of components with complexity penalty
- EM is a general algorithm applicable far beyond clustering

---

## Next Lesson

**Lesson 4.3: Dimensionality Reduction**

We will learn:

- PCA for linear dimensionality reduction
- UMAP and t-SNE for non-linear visualisation
- When and how to reduce dimensions before modelling
