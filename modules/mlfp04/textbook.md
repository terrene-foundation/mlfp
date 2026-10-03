# Module 4 — Unsupervised Machine Learning and Advanced Techniques for Insights

> _"What if the data could organise itself?"_

This chapter marks a turning point in the MLFP programme. In Modules 1 through 3 you built a complete supervised ML pipeline: hand-engineer features from domain knowledge, feed them to a model, predict a labelled outcome, evaluate, deploy, monitor. Everything you did required a target column — someone, somewhere, had to label each row. Now we remove the labels. Unsupervised machine learning discovers structure in data without being told what to look for. Clusters emerge. Dimensions collapse. Anomalies surface. Topics crystallise from raw text. And by the end of this chapter, you will see how matrix factorisation learns embeddings through optimisation — the same mechanism that powers every neural network you will build in Module 5.

The organising idea of this module is the Feature Engineering Spectrum. In Module 3, you designed features by hand using domain knowledge. In Lessons 4.1 through 4.6, unsupervised methods discover features independently — no labels, no error signal, just the geometry of the data. In Lesson 4.7, collaborative filtering introduces optimisation-driven feature discovery: embeddings learned by minimising a reconstruction loss. In Lesson 4.8, neural networks generalise this to arbitrary non-linear combinations with error feedback, completing the bridge from classical statistics to deep learning. That bridge is the intellectual backbone of the entire programme.

Everything in this chapter is engineering. You will implement K-means from scratch, derive the EM algorithm step by step, compute PCA via eigendecomposition and SVD, build anomaly detectors, extract topics from text, construct recommender systems, and train a neural network with backpropagation. Every derivation leads to running code. Every formula has a Polars DataFrame behind it.

---

## Learning Outcomes

By the end of this chapter you will be able to:

- Apply K-means, hierarchical, DBSCAN, and HDBSCAN clustering to real datasets, evaluate cluster quality using silhouette score, Davies-Bouldin index, and gap statistic, and interpret clusters with business meaning.
- Implement the EM algorithm from scratch for Gaussian Mixture Models, explain the difference between hard and soft clustering, and describe how Mixture of Experts extends mixture models to modern architectures.
- Perform PCA via both eigendecomposition and SVD, interpret scree plots and component loadings, apply t-SNE and UMAP for visualisation, and select the right dimensionality reduction method for a given task.
- Detect anomalies using statistical methods (Z-score, IQR), Isolation Forest, and Local Outlier Factor, blend scores from multiple detectors with the AnomalyDetectionEngine, and add a supervised second stage with the EnsembleEngine once labelled cases exist.
- Mine frequent itemsets with Apriori and FP-Growth, compute support, confidence, and lift, extract actionable business rules from transaction data, and use discovered patterns as features for supervised models.
- Derive TF-IDF from first principles, apply LDA and BERTopic for topic extraction, evaluate topic quality with coherence metrics, and use word embeddings as features.
- Build content-based and collaborative filtering recommender systems, implement matrix factorisation with ALS, visualise learned embeddings, and articulate the pivot: optimisation drives feature discovery.
- Construct a neural network from scratch — forward pass, loss, backpropagation, weight update — and explain how hidden layers are automated feature engineering with error feedback. Select activation functions, optimisers, and loss functions. Apply dropout, batch normalisation, and learning rate scheduling.

Those are the skills. Underneath them sits the deeper outcome: you will understand that every model you built in Module 3 relied on features you designed, and that the next three modules are about machines that design features for themselves.

---

## Prerequisites

**Module 3 complete.** This chapter assumes you can:

- Build a full supervised ML pipeline from feature engineering through evaluation and deployment.
- Work fluently with Polars DataFrames, NumPy arrays, and Kailash engines.
- Reason about bias-variance trade-offs, cross-validation, and model selection.
- Read and write mathematical notation for sums, products, derivatives, and matrix operations.
- Use gradient descent to optimise a loss function (from Module 2's linear regression).

**From Module 2 specifically:** Bayesian thinking (prior, likelihood, posterior), probability distributions (Gaussian, Bernoulli), maximum likelihood estimation, and the chain rule of calculus. These will be used in the EM derivation (Lesson 4.2), PCA (Lesson 4.3), and backpropagation (Lesson 4.8).

**Notation carried forward:**

- $\mathbf{x}$ is an input vector, $\mathbf{X}$ is a matrix $(n \times p)$.
- $\mu$ is a mean, $\sigma$ is a standard deviation, $\Sigma$ is a covariance matrix.
- $\|\mathbf{v}\|$ is the Euclidean norm of vector $\mathbf{v}$.
- $\nabla$ denotes the gradient operator.
- $\log$ without a base means natural log.

---

## How to Read This Chapter

This chapter has eight lessons that progress along the Feature Engineering Spectrum. Each lesson follows the same structure as Modules 1–3:

1. **Why This Matters** — a Singapore-contextualised motivation.
2. **Core Concepts** — plain-language explanations, then formal definitions, then code.
3. **Mathematical Foundations** — derivations from first principles. Marked THEORY or ADVANCED.
4. **The Kailash Engine** — the engine that implements the concept.
5. **Worked Example** — a complete walkthrough on real data.
6. **Try It Yourself** — five or more drills with solutions at the end of each lesson.
7. **Cross-References** — connections forward and backward.
8. **Reflection** — what you should now be able to do.

The three-layer depth markers continue:

| Marker           | Audience             | How to Read It                                                           |
| ---------------- | -------------------- | ------------------------------------------------------------------------ |
| **FOUNDATIONS:** | Zero background      | Plain language, analogies, no derivations. Read every word.              |
| **THEORY:**      | Practitioner         | Formal statement, derivation, working knowledge. Read to understand why. |
| **ADVANCED:**    | Masters / researcher | Paper references, frontier results. Skim on first read.                  |

**Estimated reading time per lesson:**

| Lesson | Title                                                      | Reading | Exercise | Total   |
| ------ | ---------------------------------------------------------- | ------- | -------- | ------- |
| 4.1    | Clustering                                                 | 100 min | 60 min   | ~2h 40m |
| 4.2    | EM Algorithm and Gaussian Mixture Models                   | 110 min | 65 min   | ~2h 55m |
| 4.3    | Dimensionality Reduction                                   | 120 min | 70 min   | ~3h 10m |
| 4.4    | Anomaly Detection and Ensembles                            | 100 min | 60 min   | ~2h 40m |
| 4.5    | Association Rules and Market Basket Analysis               | 90 min  | 55 min   | ~2h 25m |
| 4.6    | NLP — Text to Topics                                       | 110 min | 65 min   | ~2h 55m |
| 4.7    | Recommender Systems and Collaborative Filtering            | 120 min | 70 min   | ~3h 10m |
| 4.8    | Neural Networks, Backpropagation, and the Training Toolkit | 150 min | 90 min   | ~4h     |

Total: roughly 25 hours of focused work. Lesson 4.8 is the densest lesson in the entire programme — it bridges everything that came before to everything that comes after. Give it the time it needs.

---

# Lesson 4.1: Clustering

## Why This Matters

Consider an illustrative case, a composite of a common retail story rather than a report on one named company. A Singapore retailer with a large network of outlets across the island wants to personalise its loyalty programme. The marketing team had been segmenting customers by spending tier — bronze, silver, gold, platinum — using arbitrary thresholds set during a board meeting in 2018. Those thresholds had not changed in four years, even though the customer base had shifted dramatically during and after the pandemic. The gold tier contained stay-at-home parents who ordered groceries online every three days and executives who bought premium wine once a month. Their needs were entirely different, but the loyalty programme treated them identically because both spent between two hundred and five hundred dollars per month.

A data scientist on the team ran K-means clustering on the transaction data — not on spending alone, but on twelve features including purchase frequency, basket diversity, time-of-day preference, and category mix. Five clusters emerged. None of them aligned with the old bronze-silver-gold-platinum tiers. One cluster was "weeknight convenience shoppers" who bought ready meals and snacks between 6 and 9 PM. Another was "weekend entertainers" who bought large quantities of meat, beverages, and party supplies on Saturdays. The marketing team redesigned the loyalty programme around these five naturally occurring segments and could then measure whether targeted offers were redeemed more often than the old tier-based ones. (The segment names and numbers in this story are illustrative; the worked example below runs on the course's real customer data, where the structure turns out to be much less crisp.)

The lesson: domain-expert segmentation is a starting point, not a destination. When the data contains structure that your categories do not capture, unsupervised clustering can reveal it. But clustering is not magic — it is sensitive to your choice of algorithm, your choice of distance metric, your choice of the number of clusters, and whether the data has been properly scaled. This lesson teaches you to make those choices deliberately.

## Core Concepts

### FOUNDATIONS: What is clustering?

Clustering is the task of grouping data points so that points within the same group are more similar to each other than to points in other groups. There is no target variable — nobody has labelled the data. The algorithm discovers the groups on its own. This is the defining characteristic of unsupervised learning: structure discovered, not imposed.

The word "similar" does the heavy lifting. For numeric data, similarity usually means closeness in Euclidean space — points that are near each other in the feature space belong together. But closeness depends on scale. If one feature is measured in dollars (range 0 to 500,000) and another in kilometres (range 0 to 50), the dollar feature will dominate the distance calculation purely because its numbers are bigger. This is why you always standardise your features before clustering — subtract each column's mean and divide by its standard deviation (scikit-learn's `StandardScaler`, or one polars expression, as in the worked example below).

There are four families of clustering algorithms, each with different assumptions about what a "group" looks like:

**Centroid-based** (K-means): a cluster is defined by its centre point. Every data point belongs to the nearest centre. Clusters are convex and roughly spherical. Fast, but assumes you know how many clusters there are.

**Hierarchical** (agglomerative, divisive): builds a tree of nested clusters by progressively merging (or splitting) the closest groups. Does not require a pre-specified number of clusters. The tree, called a dendrogram, can be cut at any level.

**Density-based** (DBSCAN, HDBSCAN): a cluster is a dense region separated from other dense regions by sparser areas. Can find clusters of arbitrary shape. Does not require a pre-specified number of clusters. Naturally identifies noise points that do not belong to any cluster.

**Spectral**: constructs a graph from the data, computes the graph Laplacian, and clusters in the eigenspace of that Laplacian. Can find non-convex clusters that centroid methods miss.

### THEORY: The K-means objective

K-means partitions $n$ data points into $K$ clusters $C_1, C_2, \ldots, C_K$ by minimising the within-cluster sum of squares (WCSS):

$$J = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

where $\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i$ is the centroid of cluster $k$.

The algorithm alternates two steps:

1. **Assignment step:** assign each point to the cluster whose centroid is nearest: $C_k = \{\mathbf{x}_i : \|\mathbf{x}_i - \boldsymbol{\mu}_k\| \leq \|\mathbf{x}_i - \boldsymbol{\mu}_j\| \text{ for all } j\}$.
2. **Update step:** recompute each centroid as the mean of all points assigned to it: $\boldsymbol{\mu}_k = \frac{1}{|C_k|}\sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i$.

Why does this converge? Each step either decreases $J$ or leaves it unchanged. The assignment step reassigns points to a closer centroid, which cannot increase $J$. The update step moves the centroid to the mean of its assigned points, which is the value that minimises the sum of squared distances to those points (the same argument from Module 1, Lesson 1.1, where you proved the mean minimises squared error). Since $J$ is bounded below by zero and decreases monotonically, the algorithm must converge to a local minimum. Not the global minimum — K-means is sensitive to initialisation.

**K-means++ initialisation.** The standard K-means algorithm initialises centroids randomly, which can lead to poor local minima. K-means++ chooses initial centroids that are spread apart. The first centroid is chosen uniformly at random from the data. Each subsequent centroid is chosen with probability proportional to $D(\mathbf{x})^2$, where $D(\mathbf{x})$ is the distance from $\mathbf{x}$ to its nearest already-chosen centroid. This ensures centroids are not accidentally placed next to each other, and is provably $O(\log K)$-competitive with the optimal clustering.

### FOUNDATIONS: Hierarchical clustering

Agglomerative hierarchical clustering starts with each point as its own cluster and iteratively merges the two closest clusters until all points are in a single cluster. The result is a tree structure called a dendrogram. You choose the number of clusters by cutting the dendrogram at a chosen height.

The key decision is the linkage criterion — how you define the distance between two clusters:

| Linkage  | Definition                                                    | Tendency                               |
| -------- | ------------------------------------------------------------- | -------------------------------------- |
| Single   | Distance between the two closest points in the clusters       | Chains (elongated clusters)            |
| Complete | Distance between the two farthest points in the clusters      | Compact, spherical clusters            |
| Average  | Mean distance between all pairs of points across the clusters | Compromise between single and complete |
| Ward's   | Increase in WCSS if the clusters are merged                   | Minimises variance, similar to K-means |

Ward's linkage tends to produce clusters of similar size and is the most commonly used for general-purpose hierarchical clustering. Single linkage is useful when you expect irregular, elongated cluster shapes but suffers from the "chaining" effect — two clusters connected by a thin bridge of points will be merged prematurely.

Reading a dendrogram: the horizontal axis shows the data points (or clusters at lower levels), and the vertical axis shows the distance at which merges occur. A large gap in the vertical axis between two merge levels indicates a natural number of clusters — you cut just below the gap.

### FOUNDATIONS: DBSCAN and HDBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) defines clusters as connected regions of high density. It uses two parameters:

- $\varepsilon$ (epsilon): the radius of the neighbourhood around each point.
- $\text{minPts}$: the minimum number of points within the $\varepsilon$-neighbourhood to qualify as a core point.

A point is a **core point** if at least $\text{minPts}$ points (including itself) lie within its $\varepsilon$-neighbourhood. A point is a **border point** if it is within the $\varepsilon$-neighbourhood of a core point but does not itself have enough neighbours. A point is a **noise point** if it is neither core nor border. A cluster is a maximal set of density-connected core points plus their border points.

The strength of DBSCAN is that it can find clusters of arbitrary shape and naturally handles noise. The weakness is that the two parameters are hard to set, and DBSCAN struggles when clusters have very different densities.

**HDBSCAN** (Hierarchical DBSCAN) extends DBSCAN by varying $\varepsilon$ across the dataset, building a hierarchy of density-based clusters, and extracting the most stable clusters from that hierarchy. It requires only $\text{minPts}$ — the $\varepsilon$ parameter is effectively chosen automatically per region. This makes HDBSCAN far more practical for real data where cluster densities vary.

### THEORY: Spectral clustering

Spectral clustering constructs a similarity graph from the data, computes the graph Laplacian $\mathbf{L} = \mathbf{D} - \mathbf{W}$ (where $\mathbf{W}$ is the weighted adjacency matrix and $\mathbf{D}$ is the diagonal degree matrix), then clusters the data in the space of the $K$ smallest eigenvectors of $\mathbf{L}$. The intuition is that eigenvectors of the Laplacian separate the graph into loosely connected components. For data with non-convex clusters — two interleaved spirals, concentric rings — spectral clustering succeeds where K-means fails.

### FOUNDATIONS: Cluster evaluation

Since clustering has no ground-truth labels (in the general case), evaluation relies on internal metrics that measure cluster compactness and separation:

**Silhouette score.** For each point $i$, let $a(i)$ be the mean distance to all other points in the same cluster, and $b(i)$ be the mean distance to all points in the nearest neighbouring cluster. The silhouette coefficient is:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Values range from $-1$ to $+1$. A value near $+1$ means the point is well-clustered; near $0$ means it is on the boundary; near $-1$ means it may be in the wrong cluster. The overall silhouette score is the mean across all points.

**Davies-Bouldin Index.** For each cluster $i$, let $s_i$ be the average distance from points to the cluster centroid, and let $d(c_i, c_j)$ be the distance between centroids $i$ and $j$. The DB index is:

$$\text{DB} = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i} \frac{s_i + s_j}{d(c_i, c_j)}$$

Lower is better. A cluster with small intra-cluster distances and large inter-cluster distances scores well.

**Gap statistic** (Tibshirani, Walther and Hastie, 2001). Compare the within-cluster dispersion $W_K$ (the WCSS) of your clustering to its expected value under a null reference distribution with no clusters — data drawn uniformly over the bounding box of your features:

$$\text{Gap}(K) = \mathbb{E}^*\left[\log W_K^{\text{ref}}\right] - \log W_K$$

The expectation is estimated by clustering $B$ reference datasets; $s_K$ is the standard deviation of $\log W_K^{\text{ref}}$ across them, scaled by $\sqrt{1 + 1/B}$. The rule is not "take the largest gap": choose the **smallest** $K$ such that $\text{Gap}(K) \geq \text{Gap}(K+1) - s_{K+1}$ (the one-standard-error rule, which is what Exercise 1.1 implements). It is the most principled of the three methods for choosing $K$, and also the most expensive, because every $K$ is fitted $B + 1$ times.

**External metrics** apply when you do have ground-truth labels for evaluation:

- **ARI (Adjusted Rand Index):** measures agreement between predicted and true clusters, adjusted for chance. Ranges from $-1$ to $+1$; a random assignment scores near 0.
- **NMI (Normalised Mutual Information):** information-theoretic measure of agreement, normalised to $[0, 1]$.

### FOUNDATIONS: The elbow method

Plot WCSS (the K-means objective $J$) as a function of $K$. As $K$ increases, WCSS decreases — more clusters means each point is closer to its centroid. At some point the rate of decrease slows sharply, forming an "elbow" in the plot. The elbow is a reasonable (though subjective) choice for $K$. The gap statistic formalises this intuition.

## Mathematical Foundations

### THEORY: Why the centroid minimises within-cluster squared distance

This result underpins the K-means update step. For a cluster $C_k$ with points $\mathbf{x}_1, \ldots, \mathbf{x}_m$, we want the point $\boldsymbol{\mu}$ that minimises:

$$f(\boldsymbol{\mu}) = \sum_{i=1}^{m} \|\mathbf{x}_i - \boldsymbol{\mu}\|^2 = \sum_{i=1}^{m} (\mathbf{x}_i - \boldsymbol{\mu})^T(\mathbf{x}_i - \boldsymbol{\mu})$$

Take the gradient with respect to $\boldsymbol{\mu}$:

$$\nabla_{\boldsymbol{\mu}} f = \sum_{i=1}^{m} -2(\mathbf{x}_i - \boldsymbol{\mu}) = -2\left(\sum_{i=1}^{m} \mathbf{x}_i - m\boldsymbol{\mu}\right) = \mathbf{0}$$

Solving: $\boldsymbol{\mu} = \frac{1}{m}\sum_{i=1}^{m} \mathbf{x}_i$, which is the mean. This is the same result as Module 1's proof that the mean minimises squared error — extended to vectors. It is the foundation of every centroid-based method.

### ADVANCED: Connections to Gaussian Mixture Models

K-means is a special case of the EM algorithm for Gaussian Mixture Models with equal, spherical covariances and hard assignments. When you replace hard assignments (each point belongs to exactly one cluster) with soft assignments (each point has a probability of belonging to each cluster), you get the EM algorithm for GMMs, which is Lesson 4.2. This is a recurring pattern in ML: many algorithms are special cases of more general probabilistic frameworks.

## The Kailash Engine: ClusteringEngine

kailash-ml puts four of this lesson's algorithms — K-means, GMM (Lesson 4.2), DBSCAN and spectral clustering — behind one `ClusteringEngine.fit()` call that returns the labels together with the silhouette, Calinski-Harabasz and inertia values. `sweep_k()` runs the "try every $K$ and score it" loop you will write by hand in the worked example and reports the best $K$ for the criterion you choose. Ward/agglomerative clustering and HDBSCAN are not in the engine; the worked example and Exercise 1 use SciPy and the `hdbscan` package for those.

```python
import polars as pl
from shared import MLFPDataLoader
from kailash_ml.engines.clustering import ClusteringEngine

FEATURES = ["total_revenue", "order_count", "avg_order_value",
            "days_since_last_order", "customer_tenure_days",
            "satisfaction_score", "num_returns"]  # churned = outcome, excluded
customers = MLFPDataLoader().load("mlfp03", "ecommerce_customers.parquet")
X = customers.select(FEATURES).drop_nulls().sample(3000, seed=42)
X = X.select((pl.all() - pl.all().mean()) / pl.all().std())  # standardise

engine = ClusteringEngine()  # algorithms: kmeans | gmm | dbscan | spectral
sweep = engine.sweep_k(X, k_range=range(2, 9), criterion="silhouette")
fit = engine.fit(X, algorithm="kmeans", n_clusters=sweep.optimal_k)
print(f"K={fit.n_clusters}  silhouette={fit.silhouette_score:.3f}")
```

On this 3,000-customer sample the sweep picks $K = 3$ with a silhouette of about 0.18. `fit.labels` holds one cluster id per row, ready to join back onto the customer table. The same engine is what the module assessment's clustering task uses.

`AutoMLEngine` is a different tool: it does not know what clustering is. You give it a search space (for example, algorithm $\in$ {kmeans, gmm} and $K \in [3, 8]$) and an async trial function that fits one candidate — typically with `ClusteringEngine` — and returns its metric; the engine runs the search, enforces the trial, time and cost budgets, and records every trial. With `agent=False` no language model is called. Exercise 1.5 shows the full pattern.

## Worked Example: Singapore E-Commerce Customer Segmentation

We cluster the Singapore e-commerce customers you met in Module 3 (`mlfp03/ecommerce_customers.parquet`, 50,000 customers). Seven numeric columns describe behaviour: total revenue, order count, average order value, days since the last order, tenure in days, a 1–5 satisfaction score and the number of returns (0–6). The `churned` column is an **outcome**, not behaviour: it is deliberately left out of the features, so the segments are not partly a split on the answer, and used afterwards only to profile the segments. Hierarchical clustering and the silhouette score scale with $n^2$, so we work on the same 3,000-customer random sample as the engine example above; every step runs in seconds.

### Step 0: Load and standardise

```python
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
import hdbscan

from shared import MLFPDataLoader

FEATURES = ["total_revenue", "order_count", "avg_order_value",
            "days_since_last_order", "customer_tenure_days",
            "satisfaction_score", "num_returns"]

customers = MLFPDataLoader().load("mlfp03", "ecommerce_customers.parquet")
sample = customers.drop_nulls(subset=FEATURES).sample(3000, seed=42)
X_df = sample.select(FEATURES)
X_scaled = X_df.select((pl.all() - pl.all().mean()) / pl.all().std()).to_numpy()
print(X_scaled.shape)  # (3000, 7)
```

Standardisation is essential. Without it, `total_revenue` (S$0.01 to about S$3,600) and `customer_tenure_days` (hundreds to thousands) would dominate every distance calculation, and `satisfaction_score` (1–5) and `num_returns` (0–6) would be invisible to the algorithm. Those last two are small integer counts: after standardising they form a few stacked "bands" of points, which is worth remembering when you look at cluster scatter plots.

### Step 1: K-means with the elbow and silhouette

```python
wcss, sil_scores = [], []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    wcss.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

for k, w, s in zip(K_range, wcss, sil_scores):
    print(f"K={k:>2}  WCSS={w:>8,.0f}  silhouette={s:.3f}")
```

The output tells an honest story. WCSS falls by about 2,900 from $K = 2$ to $K = 3$, then by 1,300, 1,100, 850 and so on — a gentle curve with no sharp elbow after $K = 3$. The silhouette is 0.169 at $K = 2$, peaks at **0.182 at $K = 3$**, and drifts down to 0.145 at $K = 10$. A silhouette below about 0.25 means the clusters overlap heavily: real customer behaviour does not fall into crisp, well-separated groups. That does not make segmentation useless — it means the value lies in the business profile of each segment (Step 5), not in the score.

### Step 2: Hierarchical clustering with Ward's linkage

```python
Z = linkage(X_scaled, method="ward")
print("Last six merge heights:", np.round(Z[-6:, 2], 1))

labels_km = KMeans(n_clusters=3, init="k-means++", n_init=10,
                   random_state=42).fit_predict(X_scaled)
labels_ward = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X_scaled)
print(f"Agreement with K-means (ARI): {adjusted_rand_score(labels_km, labels_ward):.3f}")
```

The last six merges happen at heights of about 34.6, 37.1, 37.7, 54.9, 68.4 and 74.8. The final merge (two clusters into one) is at 74.8, the one before it (three into two) at 68.4. The largest jump is from 37.7 to 54.9 — between the five-to-four and four-to-three merges — so reading the dendrogram alone would suggest **four** clusters, while the silhouette preferred three. Cutting the Ward tree at three clusters gives an adjusted Rand index of about 0.49 with K-means: the two methods agree on the broad structure but disagree on many individual customers. Different methods, each reasonable, suggest different answers — a normal outcome when the data has no crisp clusters.

### Step 3: HDBSCAN for density-based comparison

```python
labels_hdb = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10).fit_predict(X_scaled)

n_clusters = len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0)
n_noise = int((labels_hdb == -1).sum())
print(f"HDBSCAN found {n_clusters} clusters and {n_noise} noise points "
      f"({n_noise / len(labels_hdb):.1%})")
```

HDBSCAN also finds three dense regions, but it refuses to assign 447 customers (14.9%) to any of them. Those noise points are customers whose combination of revenue, recency and tenure is not shared by a dense group — unusual high spenders, very long-tenured customers, one-off buyers. In a production system they are worth a look of their own rather than being forced into the nearest segment.

### Step 4: Evaluate and compare

```python
def report(name, X, labels):
    keep = labels != -1  # HDBSCAN noise is excluded from the scores
    sil = silhouette_score(X[keep], labels[keep])
    db = davies_bouldin_score(X[keep], labels[keep])
    print(f"{name:<9} n={int(keep.sum()):>5}  silhouette={sil:.3f}  DB={db:.3f}")

report("K-means", X_scaled, labels_km)
report("Ward", X_scaled, labels_ward)
report("HDBSCAN", X_scaled, labels_hdb)
```

| Method | Customers scored | Silhouette | Davies-Bouldin |
| --- | --- | --- | --- |
| K-means ($K=3$) | 3,000 | 0.182 | 1.71 |
| Ward ($K=3$) | 3,000 | 0.134 | 1.99 |
| HDBSCAN | 2,553 (noise excluded) | 0.095 | 2.77 |

K-means scores best on both internal metrics — unsurprisingly, because silhouette and Davies-Bouldin reward compact, convex, centroid-shaped clusters, which is exactly what K-means optimises. That is a bias of the metrics, not proof that K-means found the "true" segments (Exercise 1.4 shows silhouette preferring K-means on two interleaved moons that spectral clustering separates perfectly).

### Step 5: Interpret clusters with business meaning

```python
pl.Config.set_tbl_cols(-1)
profiles = (
    sample.with_columns(pl.Series("cluster", labels_km))
    .group_by("cluster")
    .agg(
        pl.len().alias("customers"),
        pl.col("total_revenue").mean().round(0),
        pl.col("avg_order_value").mean().round(0),
        pl.col("order_count").mean().round(1),
        pl.col("days_since_last_order").mean().round(0),
        pl.col("customer_tenure_days").mean().round(0),
        pl.col("churned").mean().round(2).alias("churn_rate"),  # outcome, profiling only
    )
    .sort("cluster")
)
print(profiles)
```

| Cluster | Customers | Revenue (S$) | Avg order (S$) | Orders | Days since last order | Tenure (days) | Churn rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 402 | 950 | 166 | 6.6 | 340 | 827 | 0.74 |
| 1 | 1,274 | 240 | 31 | 8.4 | 182 | 603 | 0.46 |
| 2 | 1,324 | 239 | 32 | 8.2 | 533 | 1,116 | 1.00 |

The cluster ids are arbitrary (rerun with another seed and they may be renumbered); the profiles are what matter. Cluster 0 is a small group of **high-value, big-basket customers** — four times the revenue and five times the average order of the rest. Cluster 1 is **recently active customers**: the shortest time since the last order, the shortest tenure and, by far, the lowest churn rate. Cluster 2 is **lapsed long-standing customers**: their last order was a year and a half ago on average, and every one of them in the sample is recorded as churned. Satisfaction and returns barely differ between the clusters (about 3.0 and 0.5 everywhere), so they are not what separates these customers. The business reading is immediate: protect cluster 0, keep cluster 1 engaged, and decide whether a win-back campaign for cluster 2 is worth its cost. Notice that `churned` was never a clustering input, yet the segments separate it sharply — recency is doing the work.

## Try It Yourself

**Drill 1.** Implement K-means from scratch in about twenty lines of Python. Use NumPy for distance computation. Start with random initialisation (not K-means++). Run it on a 2D synthetic dataset with three well-separated Gaussian blobs (use `sklearn.datasets.make_blobs`). Verify that your implementation produces the same partition as `sklearn.cluster.KMeans`.

**Solution:**

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)

def kmeans_scratch(X, K, max_iters=100, seed=0):
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), K, replace=False)].copy()
    for _ in range(max_iters):
        dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    wcss = float(((X - centroids[labels]) ** 2).sum())
    return labels, centroids, wcss

labels_sk = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X)
runs = []
for seed in range(5):
    labels, centroids, wcss = kmeans_scratch(X, 3, seed=seed)
    ari = adjusted_rand_score(labels, labels_sk)
    runs.append((wcss, seed, labels))
    print(f"seed={seed}: WCSS={wcss:8.1f}  ARI vs sklearn={ari:.3f}")

best_wcss, best_seed, best_labels = min(runs, key=lambda r: r[0])
print(f"Best of 5 restarts (seed {best_seed}): ARI = "
      f"{adjusted_rand_score(best_labels, labels_sk):.3f}")
```

The cluster *numbers* can differ between two implementations (your cluster 0 may be sklearn's cluster 2), so compare partitions with the adjusted Rand index, which ignores label names. The output shows the local-minimum problem in action: with seeds 0 and 1, random initialisation drops two starting centroids into the same blob, the algorithm converges to a WCSS of about 5,310, and the partition agrees poorly with sklearn (ARI ≈ 0.44). Seeds 2–4 reach WCSS ≈ 204 and reproduce sklearn's partition exactly (ARI = 1.0). Keeping the restart with the lowest WCSS fixes it — which is precisely what sklearn's `n_init=10` does, on top of the K-means++ initialisation that makes bad starts rare in the first place.

**Drill 2.** Apply agglomerative clustering with all four linkage methods (single, complete, average, Ward's) to the same blob dataset. Compare the dendrograms visually. Which linkage method produces the most balanced clusters? Which is most prone to chaining?

**Solution:**

```python
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, method in zip(axes.flat, ["single", "complete", "average", "ward"]):
    Z = linkage(X, method=method)
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=20)
    ax.set_title(f"{method.capitalize()} linkage")
    sizes = np.bincount(fcluster(Z, t=3, criterion="maxclust"))[1:]
    print(f"{method:<8} cluster sizes at K=3: {sizes.tolist()}")
plt.tight_layout()
plt.savefig("linkage_comparison.png")
```

On three well-separated blobs every linkage recovers the same three groups of 100 — easy data does not discriminate between methods. The dendrograms still differ: Ward's merges at heights that grow sharply once whole blobs are joined, giving the clearest cut, while single linkage merges points one at a time at small, similar heights — the chaining behaviour that makes it merge clusters joined by a thin bridge of points. Repeat the drill with `cluster_std=2.5` to see the methods disagree.

**Drill 3.** Run DBSCAN on the blob dataset with $\text{minPts} = 5$ and $\varepsilon \in \{0.3, 0.5, 2.0, 8.0\}$. For each, how many clusters and how many noise points? Explain the pattern. How large must $\varepsilon$ be before two blobs merge?

**Solution:**

```python
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

for eps in [0.3, 0.5, 2.0, 8.0]:
    labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"eps={eps}: {n_clusters} clusters, {n_noise} noise points")

# Smallest distance between points of different blobs
gaps = {(a, b): cdist(X[y_true == a], X[y_true == b]).min()
        for a in range(3) for b in range(a + 1, 3)}
print({pair: round(float(g), 2) for pair, g in gaps.items()})
```

| $\varepsilon$ | Clusters | Noise points |
| --- | --- | --- |
| 0.3 | 5 | 52 |
| 0.5 | 3 | 10 |
| 2.0 | 3 | 0 |
| 8.0 | 2 | 0 |

At $\varepsilon = 0.3$ the radius is too small: the sparser edges of the blobs fall apart into extra fragments and 52 points are noise. At 0.5 the three blobs are found with 10 noise points on their fringes. At 2.0 the result is **still three clusters** — the larger radius only absorbs the fringe points (0 noise); it does not merge the blobs, because the closest pair of points from two different blobs is about 7.5 units apart. Merging needs $\varepsilon$ larger than that gap: at $\varepsilon = 8.0$ two blobs join. The lesson: DBSCAN's $\varepsilon$ must be judged against the scale of the gaps between clusters (here, several units), not against the cluster spread (0.6). A common way to choose it is the "k-distance plot": sort every point's distance to its $\text{minPts}$-th neighbour and look for the knee.

**Drill 4.** Compute the silhouette score for K-means with $K = 2, \ldots, 8$ on the customer sample from the worked example, using **three different random 3,000-customer samples** (seeds 0, 1, 2). Does the best $K$ stay the same? What does that tell you?

**Solution:**

```python
for seed in [0, 1, 2]:
    sub = customers.drop_nulls(subset=FEATURES).sample(3000, seed=seed).select(FEATURES)
    Xs = sub.select((pl.all() - pl.all().mean()) / pl.all().std()).to_numpy()
    sils = {}
    for k in range(2, 9):
        lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Xs)
        sils[k] = silhouette_score(Xs, lab)
    best = max(sils, key=sils.get)
    print(f"seed={seed}: best K={best}  " +
          "  ".join(f"K{k}={s:.3f}" for k, s in sils.items()))
```

The silhouette curve is low (roughly 0.14–0.27) and flat, and the winning $K$ switches with the sample: seed 0 picks $K = 2$ (0.265), seeds 1 and 2 pick $K = 3$ (about 0.18). When the "optimal" $K$ is that sensitive to which customers you happened to sample, the data does not contain one obviously correct number of clusters. Choose $K$ by combining the metric with stability across samples and, above all, with whether the resulting profiles are distinct and actionable.

**Drill 5.** Using the three K-means clusters from the worked example, compute for each cluster the share of customers who ordered in the last 90 days and the churn rate. Which cluster is the highest churn risk among customers who are still reachable? What would you do for it?

**Solution:**

```python
churn_view = (
    sample.with_columns(
        pl.Series("cluster", labels_km),
        (pl.col("days_since_last_order") <= 90).alias("ordered_last_90d"),
    )
    .group_by("cluster")
    .agg(
        pl.len().alias("customers"),
        pl.col("ordered_last_90d").mean().round(3).alias("active_share"),
        pl.col("churned").mean().round(3).alias("churn_rate"),
    )
    .sort("churn_rate", descending=True)
)
print(churn_view)
```

Cluster 2 (lapsed long-standing customers) is already fully churned in this data, so it is a win-back question, not a retention one. Only 14% of the high-value cluster 0 ordered in the last 90 days (28% in cluster 1), and its churn rate is about 0.74 despite its spending — losing these customers costs the most revenue per head, so it is the segment where a retention offer (personal outreach, loyalty benefits) has the best expected return. Cluster 1 is the healthy core. `churned` was used only after clustering, never as an input.

## Cross-References

- **Module 3, Lesson 3.1** introduced feature engineering and feature selection. Clustering extends this: the cluster assignment itself becomes a feature you can feed back into a supervised model.
- **Lesson 4.2** generalises K-means to soft assignments using the EM algorithm. K-means is hard EM with spherical Gaussians.
- **Lesson 4.3** will use dimensionality reduction to visualise clusters in 2D when the original data has many features.
- **Lesson 4.7** introduces matrix factorisation, which can be viewed as clustering in a latent embedding space.
- **Module 5, Lesson 5.6** applies graph-based clustering ideas to GNNs, where the graph Laplacian from spectral clustering reappears as the propagation rule.

## Reflection

You should now be able to:

- Explain the four families of clustering algorithms and when to use each.
- Implement K-means from scratch and explain why it converges to a local minimum.
- Read a dendrogram and choose the number of clusters by identifying merge-level gaps.
- Set DBSCAN's $\varepsilon$ and minPts parameters and explain the consequences of setting them too large or too small.
- Compute silhouette score and Davies-Bouldin index and interpret their values.
- Interpret cluster profiles in business terms — not "cluster 0" and "cluster 1", but "weeknight convenience shoppers" and "weekend entertainers".

If the last point feels weak, go back to Step 5 of the worked example and spend fifteen minutes naming each cluster using the profile statistics. Naming is the skill that separates a clustering exercise from a clustering insight.

---

# Lesson 4.2: EM Algorithm and Gaussian Mixture Models

## Why This Matters

K-means assigns each customer to exactly one segment. In reality, a customer who buys groceries on weekdays and hosts dinner parties on weekends belongs partially to two segments. Forcing a hard assignment loses information. Gaussian Mixture Models solve this by assigning each point a probability of belonging to each cluster — soft clustering. The algorithm that fits GMMs is the Expectation-Maximisation (EM) algorithm, one of the most important algorithms in all of machine learning. EM is not limited to clustering; it is a general template for any model with latent (hidden) variables. You will see its echoes in variational autoencoders (Lesson 5.1), topic models (Lesson 4.6), and the training of hidden Markov models. Understanding EM here gives you a tool you will use repeatedly.

The mixture idea also has a modern descendant that is worth knowing about: the Mixture of Experts (MoE) architecture, used in several openly documented large language models such as Mixtral 8x7B. In an MoE model, a gating network selects which expert sub-network processes each input — a direct generalisation of the mixture model idea where the "assignment" of inputs to components is itself learned. We will touch on this briefly at the end of the lesson, and return to it in Module 6.

## Core Concepts

### FOUNDATIONS: Hard versus soft clustering

In K-means, each data point belongs to exactly one cluster. The assignment is binary: in or out. This is called hard clustering. It is simple and often sufficient, but it discards information at the boundaries. A point equidistant from two centroids is arbitrarily assigned to one, with no indication that the assignment is uncertain.

In soft clustering, each data point has a probability of belonging to each cluster. A customer might be 70% "weeknight convenience shopper" and 30% "weekend entertainer". These probabilities are called responsibilities (or posterior probabilities), and they encode uncertainty directly. Soft clustering is more informative than hard clustering at the cost of a more complex algorithm.

### THEORY: Gaussian Mixture Models

A Gaussian Mixture Model assumes the data is generated from a mixture of $K$ Gaussian distributions. Each component $k$ has:

- A mean $\boldsymbol{\mu}_k$ (the centre of the Gaussian).
- A covariance matrix $\boldsymbol{\Sigma}_k$ (the shape and orientation of the Gaussian).
- A mixing coefficient $\pi_k$ (the probability that a random point comes from component $k$), with $\sum_k \pi_k = 1$.

The probability of observing data point $\mathbf{x}_n$ is:

$$p(\mathbf{x}_n) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

where $\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$ is the multivariate Gaussian density.

The log-likelihood of the entire dataset is:

$$\mathcal{L} = \sum_{n=1}^{N} \log \left( \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right)$$

We cannot maximise this directly because the log of a sum does not simplify. The EM algorithm provides an iterative solution.

### THEORY: The EM algorithm — step by step

**E-step (Expectation):** compute the responsibility of each component $k$ for each data point $n$:

$$r_{nk} = \frac{\pi_k \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K} \pi_j \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$

This is Bayes' theorem: the numerator is the prior times the likelihood, and the denominator is the marginal likelihood. The responsibilities are the posterior probabilities of the latent variable (which component generated this point) given the observed data.

**M-step (Maximisation):** update the parameters using the responsibilities as weights:

$$N_k = \sum_{n=1}^{N} r_{nk}$$

$$\boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{n=1}^{N} r_{nk} \, \mathbf{x}_n$$

$$\boldsymbol{\Sigma}_k = \frac{1}{N_k} \sum_{n=1}^{N} r_{nk} \, (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^T$$

$$\pi_k = \frac{N_k}{N}$$

**Convergence:** the log-likelihood $\mathcal{L}$ is guaranteed to be non-decreasing at each iteration. The algorithm converges when the change in $\mathcal{L}$ falls below a threshold.

The derivation of the M-step update for $\boldsymbol{\mu}_k$ follows the same pattern as the K-means centroid update, but with responsibilities as weights. When $r_{nk} \in \{0, 1\}$ (hard assignments), the EM algorithm reduces to K-means. This is the formal sense in which K-means is a special case of EM.

### ADVANCED: Mixture of Experts

In a Mixture of Experts (MoE) model, the mixing coefficients $\pi_k$ are not constants — they are functions of the input. A gating network $g(\mathbf{x})$ produces a distribution over experts:

$$p(y \mid \mathbf{x}) = \sum_{k=1}^{K} g_k(\mathbf{x}) \, p_k(y \mid \mathbf{x})$$

where $g_k(\mathbf{x})$ is the probability that expert $k$ handles input $\mathbf{x}$, and $p_k(y \mid \mathbf{x})$ is expert $k$'s prediction. Some modern large language models (discussed in Module 6) use *sparse* MoE layers in which only the top few experts are activated per token, increasing model capacity without a proportional increase in computation. Mixtral 8x7B is a well-documented example (Jiang et al., 2024): each feed-forward block is replaced by 8 experts and a learned router sends every token to the top 2. Because the attention layers are shared, the model has about 46.7 billion parameters in total (not $8 \times 7 = 56$ billion) but uses only about 12.9 billion per token. Unlike a classical MoE fitted with EM, these routers are trained end-to-end by backpropagation (Lesson 4.8).

## Mathematical Foundations

### THEORY: Deriving the E-step from Bayes' theorem

The E-step computes the posterior probability that data point $\mathbf{x}_n$ was generated by component $k$. Let $z_n \in \{1, \ldots, K\}$ be the latent variable indicating which component generated $\mathbf{x}_n$. By Bayes' theorem:

$$p(z_n = k \mid \mathbf{x}_n) = \frac{p(\mathbf{x}_n \mid z_n = k) \, p(z_n = k)}{p(\mathbf{x}_n)} = \frac{\mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \cdot \pi_k}{\sum_{j=1}^{K} \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j) \cdot \pi_j} = r_{nk}$$

This is exactly the responsibility formula. The E-step is Bayesian inference on the latent variables, given the current parameter estimates.

### THEORY: Deriving the M-step for $\boldsymbol{\mu}_k$

We maximise the expected complete-data log-likelihood:

$$Q(\theta, \theta^{\text{old}}) = \sum_{n=1}^{N} \sum_{k=1}^{K} r_{nk} \left[ \log \pi_k + \log \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right]$$

Take the derivative with respect to $\boldsymbol{\mu}_k$ and set to zero:

$$\frac{\partial Q}{\partial \boldsymbol{\mu}_k} = \sum_{n=1}^{N} r_{nk} \, \boldsymbol{\Sigma}_k^{-1} (\mathbf{x}_n - \boldsymbol{\mu}_k) = \mathbf{0}$$

$$\sum_{n=1}^{N} r_{nk} \, \mathbf{x}_n = \boldsymbol{\mu}_k \sum_{n=1}^{N} r_{nk} = \boldsymbol{\mu}_k \, N_k$$

$$\boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{n=1}^{N} r_{nk} \, \mathbf{x}_n$$

This is a weighted mean, where the weights are the responsibilities. The M-step updates for $\boldsymbol{\Sigma}_k$ and $\pi_k$ follow similar derivations using Lagrange multipliers (for the constraint $\sum_k \pi_k = 1$).

## The Kailash Engine: ClusteringEngine (GMM)

The same `ClusteringEngine` from Lesson 4.1 fits a Gaussian mixture with `algorithm="gmm"`. It wraps scikit-learn's `GaussianMixture` (the same EM you derive in this lesson), passes extra keyword arguments such as `covariance_type` straight through, and returns the hard labels (each point's most probable component) plus BIC and AIC in `metrics`:

```python
import polars as pl
from shared import MLFPDataLoader
from kailash_ml.engines.clustering import ClusteringEngine

GMM_FEATURES = ["total_revenue", "avg_order_value",
                "days_since_last_order", "customer_tenure_days"]
customers = MLFPDataLoader().load("mlfp03", "ecommerce_customers.parquet")
X_gmm = customers.select(GMM_FEATURES).drop_nulls()
X_gmm = X_gmm.sample(5000, seed=42)  # the engine also scores silhouette, O(n^2)
X_gmm = X_gmm.select((pl.all() - pl.all().mean()) / pl.all().std())

engine = ClusteringEngine()
gmm_fit = engine.fit(X_gmm, algorithm="gmm", n_clusters=3, covariance_type="full")
print(gmm_fit.n_clusters, round(gmm_fit.metrics["bic"]), round(gmm_fit.metrics["aic"]))
```

What the engine does *not* return is the soft assignment itself: for the responsibilities $r_{nk}$ you call scikit-learn's `GaussianMixture.predict_proba` directly, as in Part C below. Choosing the number of components is your job — loop over $K$ and compare `metrics["bic"]` (Drill 2).

## Worked Example: EM on Synthetic and Real Data

### Part A: From-scratch EM on 2D synthetic data

The data in Part A is synthetic: 600 points drawn from three known Gaussians, so we can check whether EM recovers them.

```python
import numpy as np

rng = np.random.default_rng(42)
TRUE_MEANS = np.array([[0.0, 0.0], [5.0, 5.0], [2.0, 8.0]])
X = np.vstack([
    rng.multivariate_normal(TRUE_MEANS[0], [[1, 0.5], [0.5, 1]], 200),
    rng.multivariate_normal(TRUE_MEANS[1], [[1, -0.3], [-0.3, 1]], 200),
    rng.multivariate_normal(TRUE_MEANS[2], [[0.5, 0], [0, 2]], 200),
])

def gaussian_pdf(X, mu, sigma):
    D = X.shape[1]
    diff = X - mu
    inv_sigma = np.linalg.inv(sigma)
    exponent = -0.5 * np.sum(diff @ inv_sigma * diff, axis=1)
    norm = 1.0 / ((2 * np.pi) ** (D / 2) * np.linalg.det(sigma) ** 0.5)
    return norm * np.exp(exponent)

def log_likelihood(X, mu, sigma, pi):
    dens = sum(pi[k] * gaussian_pdf(X, mu[k], sigma[k]) for k in range(len(pi)))
    return float(np.log(dens).sum())

def em_gmm(X, K, n_iter=50, diagonal=False, seed=0):
    """Fit a K-component GMM by EM. Returns (mu, sigma, pi, resp, log-likelihoods)."""
    N, D = X.shape
    r = np.random.default_rng(seed)
    mu = X[r.choice(N, K, replace=False)].copy()
    sigma = np.array([np.eye(D)] * K)
    pi = np.ones(K) / K
    lls = []
    for _ in range(n_iter):
        # E-step: responsibilities r_nk
        resp = np.column_stack([pi[k] * gaussian_pdf(X, mu[k], sigma[k]) for k in range(K)])
        resp /= resp.sum(axis=1, keepdims=True)
        # M-step: weighted MLE
        Nk = resp.sum(axis=0)
        for k in range(K):
            mu[k] = (resp[:, k:k + 1] * X).sum(axis=0) / Nk[k]
            diff = X - mu[k]
            if diagonal:
                sigma[k] = np.diag((resp[:, k:k + 1] * diff**2).sum(axis=0) / Nk[k])
            else:
                sigma[k] = (resp[:, k:k + 1] * diff).T @ diff / Nk[k]
            sigma[k] += 1e-6 * np.eye(D)  # regularise against singular covariances
        pi = Nk / N
        lls.append(log_likelihood(X, mu, sigma, pi))
    return mu, sigma, pi, resp, lls

# EM finds a LOCAL maximum: run it from 5 initialisations, keep the best
fits = [em_gmm(X, K=3, seed=s) for s in range(5)]
for s, f in enumerate(fits):
    print(f"seed {s}: final log-likelihood = {f[4][-1]:.2f}")
mu, sigma, pi, resp, lls = max(fits, key=lambda f: f[4][-1])

for it in [0, 1, 2, 5, 10, 20, 49]:
    print(f"Iteration {it:>2}: log-likelihood = {lls[it]:.2f}")
order = np.argsort(mu[:, 0])
print("Recovered means:", np.round(mu[order], 2).tolist())
print("Mixing weights:  ", np.round(pi[order], 3).tolist())
```

The five restarts make the central caveat of EM visible. Seeds 1 and 4 reach a log-likelihood of −2283.13; seeds 0, 2 and 3 stop at −2526.75, −2374.99 and −2375.36 — local maxima where two components share one blob and a third straddles the other two. Every run is monotone (the EM guarantee), but monotone only means "never worse than the previous step", not "best possible". Keeping the run with the highest log-likelihood is exactly what scikit-learn's `n_init` does. For the best run the log-likelihood rises quickly in the first few iterations and then flattens, the recovered means are $(0.00, -0.04)$, $(5.04, 4.93)$ and $(2.03, 7.92)$ — close to the true $(0, 0)$, $(5, 5)$ and $(2, 8)$ — and the mixing weights are close to the true $1/3$ each.

### Part B: Verify responsibilities sum to 1

```python
row_sums = resp.sum(axis=1)
print("Responsibilities sum per point (should all be 1.0):")
print(f"  Min: {row_sums.min():.6f}")
print(f"  Max: {row_sums.max():.6f}")
```

### Part C: Soft clustering of real customers with scikit-learn

We now fit a three-component GMM to all 50,000 e-commerce customers. One modelling decision matters here: we use the four continuous behavioural features only. `satisfaction_score` (1–5) and `num_returns` (0–6) are small integer counts, and a full-covariance Gaussian can "collapse" onto a single integer value — its variance along that axis shrinks towards zero and its density, and so the likelihood, grows without bound. The fit then reports spectacular log-likelihoods that describe the integer grid, not customer segments (try it in Drill 2).

```python
import polars as pl
from sklearn.mixture import GaussianMixture
from shared import MLFPDataLoader

GMM_FEATURES = ["total_revenue", "avg_order_value",
                "days_since_last_order", "customer_tenure_days"]
customers = MLFPDataLoader().load("mlfp03", "ecommerce_customers.parquet")
X_real = customers.select(GMM_FEATURES).drop_nulls().to_numpy().astype(np.float64)
X_real_scaled = (X_real - X_real.mean(axis=0)) / X_real.std(axis=0)

gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
gmm.fit(X_real_scaled)
probs = gmm.predict_proba(X_real_scaled)  # the responsibilities r_nk

confidence = probs.max(axis=1)
n_boundary = int((confidence < 0.7).sum())
print(f"Mixing weights: {np.round(gmm.weights_, 3).tolist()}")
print(f"{n_boundary} customers ({n_boundary / len(X_real):.1%}) have no component above 0.7")
```

About 11% of customers (5,600 of 50,000) have no component with a responsibility above 0.7. These are the customers K-means would assign with false certainty; the GMM says, in effect, "this customer is 55% segment A and 40% segment B". For a marketing team that is useful information: a boundary customer can receive a blend of both segments' offers, or be excluded from a campaign whose targeting must be precise.

## Try It Yourself

**Drill 1.** Run the from-scratch EM with diagonal instead of full covariance matrices (`diagonal=True`). How does this change the number of covariance parameters per component? Compare the recovered covariances and the final log-likelihood with the full-covariance fit.

**Solution:**

```python
fits_d = [em_gmm(X, K=3, diagonal=True, seed=s) for s in range(5)]
mu_d, sigma_d, pi_d, resp_d, lls_d = max(fits_d, key=lambda f: f[4][-1])
print(f"Final log-likelihood  full: {lls[-1]:.2f}   diagonal: {lls_d[-1]:.2f}")
for k in np.argsort(mu[:, 0]):
    print(f"full  component at {np.round(mu[k], 1)}: off-diagonal = {sigma[k][0, 1]:+.2f}")
for k in np.argsort(mu_d[:, 0]):
    print(f"diag  component at {np.round(mu_d[k], 1)}: off-diagonal = {sigma_d[k][0, 1]:+.2f}")
```

A full covariance matrix has $D(D+1)/2$ free parameters per component — 3 in 2D (two variances and one covariance). A diagonal matrix has $D = 2$. The full fit recovers the tilted ellipses (off-diagonal terms near the true $+0.5$ and $-0.3$); the diagonal fit forces every off-diagonal term to exactly zero, so tilted clusters become axis-aligned ellipses, and its log-likelihood is lower because the restricted model cannot describe the correlations. In 7 dimensions the gap in parameter count is 28 versus 7 per component, which is why diagonal or tied covariances are often preferred when data are limited.

**Drill 2.** Implement model selection with the Bayesian Information Criterion, $\text{BIC} = -2\mathcal{L} + p \log N$, where $p$ is the number of free parameters. Run GMM with $K = 1, 2, \ldots, 8$ on the four-feature customer data and print BIC versus $K$. Which $K$ minimises BIC? Then repeat with all seven features, including the two integer counts. What goes wrong?

**Solution:**

```python
def bic_sweep(Xs, k_values=range(1, 9)):
    bics = {}
    for k in k_values:
        g = GaussianMixture(n_components=k, covariance_type="full", random_state=42).fit(Xs)
        bics[k] = g.bic(Xs)
    return bics

bics4 = bic_sweep(X_real_scaled)
print("4 continuous features:", {k: round(b) for k, b in bics4.items()})

ALL7 = GMM_FEATURES + ["order_count", "satisfaction_score", "num_returns"]
X7 = customers.select(ALL7).drop_nulls().to_numpy().astype(np.float64)
X7 = (X7 - X7.mean(axis=0)) / X7.std(axis=0)
bics7 = bic_sweep(X7)
print("7 features incl. counts:", {k: round(b) for k, b in bics7.items()})
```

On the four continuous features BIC falls at every step — from about 497,000 at $K = 1$ to 352,000 at $K = 4$, then only slowly to 325,000 at $K = 8$ — so the formal minimum is at the edge of the range. That is a common result with 50,000 points: the $p \log N$ penalty is small relative to the likelihood gain from using extra Gaussians to model the long right tail of revenue, so BIC keeps "improving" even though the new components describe the shape of a skewed distribution rather than new customer segments. Read the curve for where the gains become small (here after about $K = 4$) and judge the profiles. With all seven features the sweep is erratic — BIC drops from about 701,000 at $K = 3$ to 354,000 at $K = 4$, rises again to over 600,000 at $K = 6$, then plunges to about 106,000 at $K = 8$ — the signature of components collapsing onto integer values of the count columns, as described in Part C.

**Drill 3.** For the real customer data, compute each customer's "assignment confidence" $\max_k r_{nk}$ and plot a histogram of it. What fraction of customers have confidence below 0.6? How well do the GMM's hard labels agree with K-means at $K = 3$?

**Solution:**

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

confidences = probs.max(axis=1)
print(f"{(confidences < 0.6).mean():.1%} of customers have assignment confidence < 0.6")

labels_km = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X_real_scaled)
print(f"ARI between K-means and GMM hard labels: "
      f"{adjusted_rand_score(labels_km, probs.argmax(axis=1)):.3f}")

plt.hist(confidences, bins=30)
plt.xlabel("max responsibility")
plt.ylabel("customers")
plt.savefig("gmm_confidence.png")
```

About 6% of customers have confidence below 0.6, and the histogram is heavily skewed towards 1.0 — most customers sit clearly inside one component. The adjusted Rand index between the K-means partition and the GMM's hard labels is only about 0.42: the two methods carve the same data quite differently, because K-means assumes equal spherical clusters while the full-covariance GMM lets each component have its own size, orientation and elongation.

**Drill 4.** Verify empirically that the log-likelihood never decreases. Run EM for 100 iterations and assert that $\mathcal{L}_{t+1} \geq \mathcal{L}_t$ for all $t$. If you deliberately skip the M-step on one iteration (keep the old parameters), what happens to the log-likelihood?

**Solution:**

```python
_, _, _, _, lls_100 = em_gmm(X, K=3, n_iter=100, seed=1)
for t in range(1, len(lls_100)):
    assert lls_100[t] >= lls_100[t - 1] - 1e-9, f"log-likelihood decreased at iteration {t}"
print(f"Log-likelihood never decreased over {len(lls_100)} iterations "
      f"({lls_100[0]:.2f} -> {lls_100[-1]:.2f})")

# Skipping the M-step: an E-step alone does not change mu, sigma or pi
ll_before = log_likelihood(X, mu, sigma, pi)
resp_again = np.column_stack([pi[k] * gaussian_pdf(X, mu[k], sigma[k]) for k in range(3)])
resp_again /= resp_again.sum(axis=1, keepdims=True)
ll_after = log_likelihood(X, mu, sigma, pi)
print(f"E-step only: {ll_before:.4f} -> {ll_after:.4f}")
```

The assertion holds on every iteration (the tolerance only absorbs floating-point rounding). Skipping the M-step leaves the log-likelihood exactly unchanged: $\mathcal{L}$ depends only on the parameters $(\boldsymbol{\mu}, \boldsymbol{\Sigma}, \boldsymbol{\pi})$, and an E-step merely recomputes the responsibilities from them, so the iteration stalls rather than gets worse.

**Drill 5.** Explain in three sentences why Mixture of Experts is a generalisation of GMM. What plays the role of the responsibilities $r_{nk}$ in an MoE model? What plays the role of the component distributions?

**Solution:** In a GMM the mixing coefficients $\pi_k$ are constants — the same for every data point. In an MoE the mixing coefficients are produced by a gating network $g_k(\mathbf{x})$ that depends on the input, so different inputs are routed to different experts, and each expert models $p_k(y \mid \mathbf{x})$ rather than a density over $\mathbf{x}$. The gating probabilities $g_k(\mathbf{x})$ play the role of the prior $\pi_k$, the posterior over which expert produced an observed $(\mathbf{x}, y)$ plays the role of the responsibilities $r_{nk}$ when an MoE is fitted with EM, and the expert networks play the role of the component distributions.

## Cross-References

- **Lesson 4.1** introduced K-means as hard clustering. GMM generalises K-means to soft clustering — K-means is EM with spherical Gaussians and binary responsibilities.
- **Lesson 4.6** will use LDA (Latent Dirichlet Allocation) for topic modelling, which is another latent-variable model fitted with a variant of EM.
- **Lesson 5.1** introduces Variational Autoencoders, where the ELBO objective is derived using the same variational inference framework that underlies EM.
- **Module 6, Lesson 6.1** connects Mixture of Experts to modern LLM architectures.

## Reflection

You should now be able to:

- Write the E-step and M-step updates for a GMM from memory.
- Explain why the log-likelihood is non-decreasing under EM.
- Implement EM from scratch in under 30 lines of code.
- Distinguish hard clustering (K-means) from soft clustering (GMM) and name a situation where soft clustering provides more value.
- Describe the Mixture of Experts architecture as a generalisation of mixture models.

---

# Lesson 4.3: Dimensionality Reduction

## Why This Matters

The customer dataset in Lesson 4.1 had seven behavioural features. That is manageable. A genomics dataset might have 20,000 features (one per gene). A text dataset encoded with bag-of-words might have 50,000 features (one per unique word). You cannot visualise 20,000 dimensions. You cannot cluster effectively in 50,000 dimensions — the curse of dimensionality makes distance metrics meaningless when most of the volume of a high-dimensional hypercube is concentrated in its corners. You need to reduce the number of dimensions while preserving as much of the data's structure as possible.

Dimensionality reduction is not just a visualisation trick. It is feature extraction. The new, lower-dimensional features are combinations of the original features that capture the most important variation in the data. PCA, the simplest and most widely used method, finds the directions of maximum variance. Those directions are often interpretable: the first principal component of a housing dataset might capture "overall quality" (size, location, condition all moving together), and the second might capture the "urban vs suburban" trade-off (small-but-central versus large-but-remote). The reduced features can then be fed into any downstream model.

In this lesson you will derive PCA from first principles, connect it to the Singular Value Decomposition (SVD), and learn when to use non-linear alternatives like t-SNE and UMAP.

## Core Concepts

### FOUNDATIONS: The curse of dimensionality

As the number of dimensions increases, the volume of the space increases exponentially, and data points become increasingly isolated. Consider a unit hypercube in $d$ dimensions. The fraction of the volume within distance $\epsilon$ of the boundary is $1 - (1 - 2\epsilon)^d$. For $d = 100$ and $\epsilon = 0.01$, this is $1 - 0.98^{100} \approx 0.87$ — 87% of the volume is within 1% of the boundary. In high dimensions, almost all points are near the edge, distances between random points converge to the same value, and the concept of "nearest neighbour" becomes meaningless.

This has practical consequences: K-nearest-neighbours classifiers degrade, clustering algorithms produce spurious results, and density estimation becomes unreliable. Dimensionality reduction mitigates these effects by projecting the data onto a lower-dimensional subspace where distances are meaningful again.

### THEORY: PCA — the two-step process

PCA has two conceptual steps:

**Step 1: Decorrelate.** Rotate the coordinate axes so they align with the directions of maximum variance in the data. These new axes are called principal components. The first principal component is the direction along which the data varies the most. The second is the direction of maximum variance orthogonal to the first. And so on.

**Step 2: Reduce.** Keep only the top $k$ principal components, discarding the rest. The variance explained by the discarded components is the information lost.

Mathematically, PCA finds the linear projection that maximises the variance of the projected data.

### THEORY: PCA via eigendecomposition

Centre the data: $\tilde{\mathbf{X}} = \mathbf{X} - \bar{\mathbf{X}}$. Compute the covariance matrix:

$$\mathbf{C} = \frac{1}{n-1} \tilde{\mathbf{X}}^T \tilde{\mathbf{X}}$$

$\mathbf{C}$ is a $p \times p$ symmetric positive semi-definite matrix. Its eigenvectors are the principal component directions, and its eigenvalues are the variances along those directions.

Solve the eigenvalue problem: $\mathbf{C} \mathbf{v}_k = \lambda_k \mathbf{v}_k$, where $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_p \geq 0$.

The first principal component direction is $\mathbf{v}_1$ (the eigenvector with the largest eigenvalue). The projection of the data onto the first $k$ principal components is:

$$\mathbf{Z} = \tilde{\mathbf{X}} \mathbf{V}_k$$

where $\mathbf{V}_k = [\mathbf{v}_1, \ldots, \mathbf{v}_k]$ is the matrix of the top $k$ eigenvectors.

**Variance explained** by the first $k$ components:

$$\text{VE}(k) = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{p} \lambda_i}$$

A scree plot shows $\lambda_i$ versus $i$. You choose $k$ where the eigenvalues drop off sharply — the same "elbow" idea as in K-means.

### THEORY: The SVD connection

The Singular Value Decomposition of the centred data matrix $\tilde{\mathbf{X}}$ (with dimensions $n \times p$) is:

$$\tilde{\mathbf{X}} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$

where $\mathbf{U}$ is $n \times n$ (left singular vectors), $\boldsymbol{\Sigma}$ is $n \times p$ (diagonal matrix of singular values $\sigma_1 \geq \sigma_2 \geq \cdots$), and $\mathbf{V}$ is $p \times p$ (right singular vectors).

The connection: the columns of $\mathbf{V}$ are the eigenvectors of $\tilde{\mathbf{X}}^T \tilde{\mathbf{X}} = \mathbf{V} \boldsymbol{\Sigma}^T \boldsymbol{\Sigma} \mathbf{V}^T$, and the eigenvalues of the covariance matrix are $\lambda_i = \sigma_i^2 / (n-1)$.

So PCA via eigendecomposition of $\mathbf{C}$ and PCA via SVD of $\tilde{\mathbf{X}}$ give the same result. SVD is numerically more stable for large matrices and is what most implementations use internally.

**Reconstruction.** The rank-$k$ approximation of the data is:

$$\hat{\mathbf{X}} = \mathbf{Z} \mathbf{V}_k^T + \bar{\mathbf{X}}$$

The reconstruction error is:

$$\|\tilde{\mathbf{X}} - \hat{\tilde{\mathbf{X}}}\|_F^2 = \sum_{i=k+1}^{p} \sigma_i^2 = (n-1) \sum_{i=k+1}^{p} \lambda_i$$

The squared error is the sum of the discarded squared singular values — equivalently, $(n-1)$ times the sum of the discarded covariance eigenvalues, because $\lambda_i = \sigma_i^2/(n-1)$. Divided by the $n \times p$ entries of the matrix, the mean squared reconstruction error per entry is $\frac{n-1}{n} \cdot \frac{1}{p} \sum_{i>k} \lambda_i$: the discarded variance, spread over the features. As a fraction of the total variance it is $1 - \text{VE}(k)$. Drill 4 checks this to many decimal places.

### FOUNDATIONS: Component loadings

The loadings are the entries of the eigenvectors $\mathbf{v}_k$. Each loading tells you how much a particular original feature contributes to a principal component. If the first principal component has large positive loadings on "floor area", "number of rooms", and "price", and near-zero loadings on "storey" and "lease remaining", you can interpret it as "overall flat size and value". Loadings make PCA interpretable, not just a mathematical projection.

### FOUNDATIONS: t-SNE

t-SNE (t-distributed Stochastic Neighbour Embedding) is a non-linear dimensionality reduction method designed for visualisation. It preserves local structure: points that are close in high-dimensional space remain close in the 2D embedding. It does this by defining a probability distribution over pairs of points in high-dimensional space (based on Gaussian distances) and a corresponding distribution in the low-dimensional embedding (based on a Student's t-distribution), then minimising the KL divergence between them.

Key properties: t-SNE is excellent for visualisation but not suitable for feature extraction. It is non-deterministic (different runs produce different embeddings), it cannot place new points on an existing map without refitting (no `transform()`, and no inverse), and it does not preserve global structure (distances between distant clusters are not meaningful). The perplexity parameter controls the effective number of neighbours and typically ranges from 5 to 50.

### FOUNDATIONS: UMAP

UMAP (Uniform Manifold Approximation and Projection) is similar to t-SNE in spirit but grounded in a different mathematical framework (a fuzzy nearest-neighbour graph, motivated by topological data analysis). It is usually faster than t-SNE on large data and tends to keep more of the coarse arrangement of the data, and — unlike t-SNE — a fitted UMAP model can `transform()` new points, so it can be used for feature extraction, not just visualisation (it also offers an approximate `inverse_transform`). It shares t-SNE's main caveat: distances *between* well-separated groups and the apparent sizes of groups in a UMAP plot are not reliable, so never read "cluster A is twice as far from B as from C" off the picture. Results depend on `n_neighbors` (local versus global emphasis) and `min_dist` (how tightly points pack), and on the random seed. UMAP has become a common default for non-linear reduction in practice.

### ADVANCED: Kernel PCA

Standard PCA finds linear projections. Kernel PCA first maps the data into a higher-dimensional feature space via a kernel function (RBF, polynomial), then performs PCA in that space. This captures non-linear structure without explicitly computing the high-dimensional mapping — the kernel trick. Unlike PCA, kernel PCA has **no exact inverse**: a point in the implicit feature space generally has no exact pre-image in the input space (the "pre-image problem"), so reconstructing input-space points needs an approximation — fixed-point iteration, or scikit-learn's `KernelPCA(fit_inverse_transform=True)`, which learns an approximate inverse map by kernel ridge regression. Kernel PCA also needs the $n \times n$ kernel matrix, which limits it to a few thousand rows (Exercise 3.2 subsamples for this reason).

### FOUNDATIONS: Other manifold learners — a reference table

PCA, kernel PCA, t-SNE and UMAP are the methods you will use most. Three classical manifold learners are worth recognising; all are in `sklearn.manifold` and all scale poorly beyond tens of thousands of points.

| Method | What it preserves | When to reach for it | Main limitation |
| --- | --- | --- | --- |
| MDS (multidimensional scaling) | All pairwise distances, as well as possible | Visualising a given distance or dissimilarity matrix (e.g. survey similarity ratings) | $O(n^2)$ memory; with Euclidean distances, classical MDS equals PCA |
| Isomap | Geodesic distances — shortest paths along a $k$-nearest-neighbour graph | Data lying on one smooth, "unrolled" manifold (the classic Swiss roll) | Breaks if the neighbour graph short-circuits across folds or is disconnected; sensitive to noise |
| LLE (locally linear embedding) | Each point's reconstruction from its neighbours by linear weights | Smooth manifolds where local patches are nearly flat | Sensitive to $k$ and noise; can collapse points; no reliable global layout |

### THEORY: Intrinsic dimension

Data can live in $p$ ambient dimensions but vary along only $d \ll p$ independent directions; $d$ is its **intrinsic dimension**. A thousand photographs of a face rotating on a turntable have millions of pixel values each, but one intrinsic degree of freedom: the angle. Knowing $d$ tells you how far you can reduce without losing real structure. Two estimates are common:

- **PCA thresholds** — the number of components needed to reach 90% or 95% of the variance. This counts *linear* dimensions, so it overestimates $d$ for curved manifolds (a Swiss roll needs all 3 principal components but is intrinsically 2-dimensional).
- **The nearest-neighbour MLE of Levina and Bickel (2004).** Let $T_j(\mathbf{x})$ be the distance from $\mathbf{x}$ to its $j$-th nearest neighbour. If the data are locally uniform on a $d$-dimensional manifold, the number of neighbours within radius $r$ grows like $r^d$, and the maximum-likelihood estimate at $\mathbf{x}$ is

$$\hat{m}_k(\mathbf{x}) = \left[ \frac{1}{k-1} \sum_{j=1}^{k-1} \log \frac{T_k(\mathbf{x})}{T_j(\mathbf{x})} \right]^{-1}$$

averaged over the points (and, for stability, over several $k$). It is non-linear: it reports about 2 for a Swiss roll and about 8 for an 8-dimensional Gaussian cloud. Exact duplicate rows give $T_j = 0$ and must be removed first. The worked example estimates it for the customer data.

## Mathematical Foundations

### THEORY: Why PCA maximises variance

We want the direction $\mathbf{w}$ (unit vector, $\|\mathbf{w}\| = 1$) that maximises the variance of the projected data:

$$\text{Var}(\mathbf{w}^T \tilde{\mathbf{X}}^T) = \mathbf{w}^T \mathbf{C} \mathbf{w}$$

Subject to the constraint $\mathbf{w}^T \mathbf{w} = 1$, we form the Lagrangian:

$$L(\mathbf{w}, \lambda) = \mathbf{w}^T \mathbf{C} \mathbf{w} - \lambda(\mathbf{w}^T \mathbf{w} - 1)$$

Taking the derivative and setting to zero:

$$\frac{\partial L}{\partial \mathbf{w}} = 2\mathbf{C}\mathbf{w} - 2\lambda\mathbf{w} = \mathbf{0} \implies \mathbf{C}\mathbf{w} = \lambda\mathbf{w}$$

This is the eigenvalue equation. The variance of the projection is $\mathbf{w}^T \mathbf{C} \mathbf{w} = \mathbf{w}^T \lambda \mathbf{w} = \lambda$. So the maximum-variance direction is the eigenvector with the largest eigenvalue. The second principal component maximises variance subject to being orthogonal to the first, which gives the second eigenvector, and so on.

### THEORY: Reconstruction error and the Eckart-Young theorem

The Eckart-Young-Mirsky theorem states that the best rank-$k$ approximation of a matrix (in the Frobenius norm) is given by its truncated SVD. Since PCA is equivalent to truncated SVD, the PCA reconstruction is the best possible linear reconstruction with $k$ dimensions. No other linear method can achieve lower reconstruction error with the same number of components.

## The Kailash Engine: DimReductionEngine

`DimReductionEngine.reduce()` runs PCA, NMF (Lesson 4.6), t-SNE or UMAP behind one call and returns a result object with the embedding (`transformed`), the explained-variance ratio and the reconstruction error. `variance_analysis()` gives the numbers for a scree plot without choosing $k$ first. Kernel PCA, Isomap and LLE are not in the engine — use scikit-learn for those (Exercise 3.2 and 3.5). `ModelVisualizer` has no scree-plot helper; plot the cumulative variance with plotly directly.

```python
import plotly.graph_objects as go
import polars as pl
from shared import MLFPDataLoader
from kailash_ml.engines.dim_reduction import DimReductionEngine

FEATURES = ["total_revenue", "order_count", "avg_order_value",
            "days_since_last_order", "customer_tenure_days",
            "satisfaction_score", "num_returns"]
customers = MLFPDataLoader().load("mlfp03", "ecommerce_customers.parquet")
X = customers.select(FEATURES).drop_nulls().sample(3000, seed=42)
X = X.select((pl.all() - pl.all().mean()) / pl.all().std())

reducer = DimReductionEngine()  # algorithms: pca | nmf | tsne | umap
pca_res = reducer.reduce(X, algorithm="pca", n_components=5)
print([round(v, 3) for v in pca_res.explained_variance_ratio])
print(f"variance lost with 5 components: {pca_res.reconstruction_error:.3f}")

scree = reducer.variance_analysis(X)
fig = go.Figure(go.Scatter(y=scree["cumulative_variance"], mode="lines+markers"))
fig.update_layout(title="PCA scree (cumulative variance)",
                  xaxis_title="component", yaxis_title="cumulative variance")
fig.write_html("pca_scree.html")

umap_res = reducer.reduce(X, algorithm="umap", n_components=2, seed=42)
xy = umap_res.transformed  # 2-D coordinates to plot
```

On this 3,000-customer sample the five ratios are about 0.26, 0.23, 0.15, 0.14 and 0.14. Note what the engine calls `reconstruction_error`: the *fraction of variance discarded*, $1 - \text{VE}(k)$ (here about 0.08), not the squared error in the units of the data.

## Worked Example: PCA, t-SNE, and UMAP on E-Commerce Customers

PCA is cheap, so it runs on all 50,000 customers. t-SNE, UMAP and the neighbourhood-based quality score are $O(n^2)$ or close to it, so they run on a 3,000-customer sample.

```python
import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import NearestNeighbors
import umap

from shared import MLFPDataLoader

FEATURES = ["total_revenue", "order_count", "avg_order_value",
            "days_since_last_order", "customer_tenure_days",
            "satisfaction_score", "num_returns"]
customers = MLFPDataLoader().load("mlfp03", "ecommerce_customers.parquet")
X = customers.select(FEATURES).drop_nulls().to_numpy().astype(np.float64)
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

# PCA on all 50,000 customers
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
cum_var = np.cumsum(pca.explained_variance_ratio_)
for i, (ev, cv) in enumerate(zip(pca.explained_variance_, cum_var)):
    print(f"PC{i + 1}: eigenvalue = {ev:.3f}   cumulative variance = {cv:.3f}")

# Loadings: which features drive the first three components?
for i in range(3):
    top = np.argsort(np.abs(pca.components_[i]))[::-1][:3]
    print(f"PC{i + 1}: " + ", ".join(f"{FEATURES[j]} ({pca.components_[i][j]:+.2f})" for j in top))
```

| Component | Eigenvalue | Cumulative variance |
| --- | --- | --- |
| PC1 | 1.86 | 26.5% |
| PC2 | 1.59 | 49.3% |
| PC3 | 1.01 | 63.7% |
| PC4 | 1.00 | 78.0% |
| PC5 | 0.99 | 92.2% |
| PC6 | 0.41 | 98.0% |
| PC7 | 0.14 | 100% |

Five components are needed to pass 90% of the variance and six to pass 95%. The loadings explain why. PC1 loads on `avg_order_value` (+0.71) and `total_revenue` (+0.65): a **spend** axis. PC2 loads equally on `days_since_last_order` and `customer_tenure_days` (+0.71 each): a **customer-age** axis — long-standing customers are also the ones who last ordered long ago. Those two correlated pairs are the only real redundancy in the data. PC3–PC5 have eigenvalues of almost exactly 1.0: for standardised data that is the signature of features that are essentially uncorrelated with everything else (order count, satisfaction and returns), so each needs a component of its own. There is no sharp scree elbow — this dataset is genuinely about five-dimensional, not two-dimensional.

```python
# Reconstruction error with 4 components, checked against the eigenvalue formula
pca_4 = PCA(n_components=4)
X_recon = pca_4.inverse_transform(pca_4.fit_transform(X_scaled))
n, p = X_scaled.shape
mse = np.mean((X_scaled - X_recon) ** 2)
predicted = (n - 1) / n * pca.explained_variance_[4:].sum() / p
print(f"Reconstruction MSE (4 components): {mse:.4f}   predicted: {predicted:.4f}")

# Intrinsic dimension: Levina-Bickel nearest-neighbour MLE on a 2,000-row sample
rng = np.random.default_rng(42)
X_id = X_scaled[rng.choice(n, 2000, replace=False)]

def levina_bickel(X, k=20):
    dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    T = dist[:, 1:]                      # drop each point's distance to itself
    ok = np.all(T > 0, axis=1)           # duplicates would give log(0)
    return float((1.0 / np.log(T[ok, k - 1][:, None] / T[ok, : k - 1]).mean(axis=1)).mean())

print("Intrinsic dimension:", {k: round(levina_bickel(X_id, k), 1) for k in (10, 20, 30)})
```

The measured mean squared error with four components is 0.2204, matching the formula $\frac{n-1}{n} \cdot \frac{1}{p}\sum_{i>4}\lambda_i$ to four decimals: four components lose 22% of the (unit) variance of each standardised feature on average, the same as $1 - 0.780$. The Levina–Bickel estimate is about 5 (5.2, 4.9 and 4.8 for $k = 10, 20, 30$), agreeing with the five components PCA needed for 90%.

```python
# t-SNE and UMAP on a 3,000-customer sample, judged by trustworthiness
idx = rng.choice(n, 3000, replace=False)
X_s = X_scaled[idx]

emb = {
    "PCA (2D)": PCA(n_components=2).fit_transform(X_s),
    "t-SNE": TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_s),
    "UMAP": umap.UMAP(n_components=2, random_state=42).fit_transform(X_s),
}
for name, Z in emb.items():
    print(f"{name:<9} trustworthiness = {trustworthiness(X_s, Z, n_neighbors=10):.3f}")
```

Trustworthiness asks whether a point's neighbours *in the 2-D picture* were also its neighbours in the original 7-D space (1.0 = no false neighbours; about 0.5 = a random layout). PCA squeezed into two dimensions scores about 0.79 — it keeps only 49% of the variance, so many points that look close in the plot are not. t-SNE scores about 0.99 and UMAP about 0.97: both are far more faithful *locally*. That is exactly their job. It does not make their global layout trustworthy, and it does not make them feature extractors: the t-SNE map cannot embed a new customer without refitting.

## Try It Yourself

**Drill 1.** Implement PCA from scratch using NumPy's eigendecomposition. Compute the covariance matrix, find eigenvalues and eigenvectors, project onto the top 2 components. Verify your result matches `sklearn.decomposition.PCA`.

**Solution:**

```python
X_centred = X_scaled - X_scaled.mean(axis=0)
cov = X_centred.T @ X_centred / (len(X_centred) - 1)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
idx_sorted = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx_sorted]
eigenvectors = eigenvectors[:, idx_sorted]
X_my_pca = X_centred @ eigenvectors[:, :2]

pca_sk = PCA(n_components=2)
X_sk_pca = pca_sk.fit_transform(X_scaled)
print("Eigenvalues match sklearn:", np.allclose(eigenvalues[:2], pca_sk.explained_variance_))
# The sign of an eigenvector is arbitrary, so compare up to sign
for i in range(2):
    corr = np.abs(np.corrcoef(X_my_pca[:, i], X_sk_pca[:, i])[0, 1])
    print(f"PC{i + 1} |correlation| with sklearn: {corr:.6f}")  # 1.000000
```

**Drill 2.** Compute PCA using SVD instead of eigendecomposition. Verify that the singular values squared divided by $(n-1)$ equal the eigenvalues from Drill 1.

**Solution:**

```python
U, S, Vt = np.linalg.svd(X_centred, full_matrices=False)
eigenvalues_from_svd = S**2 / (len(X_centred) - 1)
print("Eigenvalues match:", np.allclose(eigenvalues, eigenvalues_from_svd))
print("Directions match (up to sign):",
      np.allclose(np.abs(Vt), np.abs(eigenvectors.T), atol=1e-6))
```

**Drill 3.** Run t-SNE with perplexity values of 5, 30, and 100 on the 3,000-customer sample. How does perplexity change the picture? Which setting preserves neighbourhoods best?

**Solution:**

```python
for perp in [5, 30, 100]:
    Z = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(X_s)
    tw = trustworthiness(X_s, Z, n_neighbors=10)
    print(f"perplexity={perp:>3}: trustworthiness={tw:.3f}  "
          f"x-range=[{Z[:, 0].min():.0f}, {Z[:, 0].max():.0f}]")
```

Perplexity is roughly the number of neighbours each point "pays attention to". Low perplexity (5) produces many small, tight fragments and the widest spread; high perplexity (100) produces a smoother, more compact map that reflects broader structure. All three keep local neighbourhoods well (trustworthiness 0.988, 0.993 and 0.990 at 10 neighbours for perplexity 5, 30 and 100); the x-range shrinks from about ±100 to about ±40 as perplexity grows. There is no single "clearest" setting — clusters that appear at one perplexity and vanish at another are a warning that the structure is not robust. Always look at more than one.

**Drill 4.** Demonstrate that PCA reconstruction error equals the discarded eigenvalues. Compute PCA with $k = 2$ components, reconstruct, and compare the squared Frobenius error with $(n-1)\sum_{i=3}^{p} \lambda_i$.

**Solution:**

```python
pca_2 = PCA(n_components=2)
X_recon2 = pca_2.inverse_transform(pca_2.fit_transform(X_scaled))
frob_sq = np.sum((X_scaled - X_recon2) ** 2)
predicted_sq = (len(X_scaled) - 1) * eigenvalues[2:].sum()
print(f"||X - X_hat||_F^2 = {frob_sq:,.2f}   (n-1) * sum(discarded eigenvalues) = {predicted_sq:,.2f}")
print(f"Mean squared error per entry: {frob_sq / X_scaled.size:.6f}")
```

The two numbers agree to rounding. Forgetting the factor $(n-1)$ — comparing the squared error with the bare sum of eigenvalues — is off by a factor of about 50,000 here, which is why the definition of $\lambda_i$ (eigenvalues of the covariance matrix, already divided by $n-1$) matters.

**Drill 5.** Apply UMAP with `n_components=3` to the 3,000-customer sample. Run K-means with $K = 4$ in the original 7-D space and in the 3-D UMAP space, and compare the silhouette scores. Which is higher? Does that mean UMAP "found better clusters"? Check trustworthiness too.

**Solution:**

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X_umap_3d = umap.UMAP(n_components=3, random_state=42).fit_transform(X_s)
km_orig = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X_s)
km_umap = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X_umap_3d)

print(f"Silhouette (original 7-D): {silhouette_score(X_s, km_orig.labels_):.3f}")
print(f"Silhouette (UMAP 3-D):     {silhouette_score(X_umap_3d, km_umap.labels_):.3f}")
print(f"UMAP 3-D trustworthiness:  {trustworthiness(X_s, X_umap_3d, n_neighbors=10):.3f}")
```

The silhouette in UMAP space (about 0.40) is far higher than in the original space (about 0.16), and the 3-D embedding is locally faithful (trustworthiness about 0.99). The silhouette gain is still a property of UMAP, not of the customers: UMAP deliberately pulls neighbouring points together and pushes the rest apart, so almost any K-means partition of its output looks well separated. A silhouette computed in the embedding answers "how blob-like is this picture?", not "how much real cluster structure is there?". Judge an embedding by neighbourhood preservation (trustworthiness) and judge clusters in the original feature space — this is why Exercise 3 ranks reducers by trustworthiness. Clustering on UMAP output can still be useful (BERTopic does it in Lesson 4.6), but the silhouette gain is not evidence for it.

## Cross-References

- **Module 2, Lesson 2.5** introduced linear algebra concepts. PCA is the direct application of eigendecomposition to data analysis.
- **Lesson 4.1** used clustering on the original features. Reducing with PCA before clustering can remove noise dimensions; clustering on UMAP output inflates internal scores (Drill 5), so judge the resulting clusters in the original feature space.
- **Lesson 4.7** introduces matrix factorisation. PCA is matrix factorisation via SVD; collaborative filtering is matrix factorisation via ALS. The connection is deep.
- **Module 5, Lesson 5.1** will use autoencoders for non-linear dimensionality reduction — a neural-network generalisation of PCA.

## Reflection

You should now be able to:

- Derive PCA from the variance-maximisation objective and connect it to eigendecomposition.
- Explain the SVD connection and why SVD is preferred computationally.
- Read a scree plot and choose the number of components.
- Interpret component loadings in domain terms.
- Distinguish when to use PCA (feature extraction), t-SNE (visualisation only), and UMAP (both, with care), and name the classical alternatives (MDS, Isomap, LLE).
- Estimate intrinsic dimension with PCA thresholds and the Levina–Bickel nearest-neighbour estimator.
- Judge an embedding by neighbourhood preservation (trustworthiness), not by clustering scores computed inside it.
- Compute reconstruction error and explain what information is lost.

---

# Lesson 4.4: Anomaly Detection and Ensembles

## Why This Matters

Consider an illustrative scenario (a composite, not a report on a named company). A Singapore digital-payments company notices something odd in its weekly fraud report: the number of flagged transactions had dropped by 40% even though total transaction volume was up 15%. Investigation revealed that a software update had changed the data pipeline's timestamp format, causing one of the three fraud detection models to silently return a default score of 0.5 for every transaction. The overall fraud score — an average of three models — was now systematically lower because one third of the signal had been replaced with noise. Transactions that would have been flagged at 0.72 were now scoring 0.58, just below the threshold.

The scenario illustrates two lessons. First, anomaly detection is not a single algorithm — it is a system. A single detector will miss things. Multiple detectors with blended scores are more robust. Second, the anomaly detection system itself needs monitoring; a detector that silently fails is worse than no detector at all, because it creates false confidence.

This lesson teaches you four anomaly detection methods — statistical, Isolation Forest, Local Outlier Factor, and ensemble blending — and the two Kailash engines involved: `AnomalyDetectionEngine` for the unsupervised detectors and their blend, and `EnsembleEngine` for the supervised second stage you can add once analysts have labelled some cases.

## Core Concepts

### FOUNDATIONS: What is an anomaly?

An anomaly (or outlier) is a data point that differs significantly from the majority. The word "significantly" requires a definition, and different definitions produce different methods:

**Statistical approach:** a point is anomalous if it falls far from the centre of a known distribution. Z-score and IQR methods assume the data is roughly normal (or at least unimodal).

**Distance- and density-based approach:** a point is anomalous if it is far from its neighbours or sits in a sparser region than they do. $k$-nearest-neighbour distance and LOF use this idea.

**Isolation-based approach:** a point is anomalous if random partitions of the feature space separate it from the rest quickly. Isolation Forest uses this idea — it never computes a distance.

**Model-based approach:** a point is anomalous if a model assigns it low probability. GMMs (from Lesson 4.2) can flag points in low-density regions.

### FOUNDATIONS: Z-score and IQR methods

**Z-score.** For each value $x$ in a column with mean $\bar{x}$ and standard deviation $s$:

$$z = \frac{x - \bar{x}}{s}$$

A point with $|z| > 3$ is more than three standard deviations from the mean, which for a normal distribution covers 99.7% of the data. Points beyond this threshold are flagged as anomalies.

**IQR method.** Compute the first quartile $Q_1$ and third quartile $Q_3$. The interquartile range is $\text{IQR} = Q_3 - Q_1$. A point is anomalous if:

$$x < Q_1 - 1.5 \times \text{IQR} \quad \text{or} \quad x > Q_3 + 1.5 \times \text{IQR}$$

The IQR method is more robust than the Z-score because the quartiles are resistant to outliers, while the mean and standard deviation are not.

### THEORY: Isolation Forest

Isolation Forest is based on a beautifully simple insight: anomalies are easier to isolate than normal points. The algorithm builds random binary trees by selecting a random feature and a random split value at each node. Normal points, which are surrounded by many similar points, require many splits to be isolated (deep tree path). Anomalies, which are rare and different, are isolated quickly (short tree path).

The anomaly score for a point $\mathbf{x}$ is:

$$s(\mathbf{x}, n) = 2^{-\frac{E[h(\mathbf{x})]}{c(n)}}$$

where $E[h(\mathbf{x})]$ is the average path length for $\mathbf{x}$ across all trees, and $c(n)$ is the average path length of an unsuccessful search in a binary search tree with $n$ elements:

$$c(n) = 2H(n-1) - \frac{2(n-1)}{n}$$

where $H(k) = \ln(k) + 0.5772\ldots$ (the Euler-Mascheroni constant). A score close to 1 indicates an anomaly; close to 0.5 indicates a normal point; close to 0 indicates a very normal point.

The intuition: the exponent is $-E[h(\mathbf{x})]/c(n)$. If $E[h(\mathbf{x})]$ is much smaller than $c(n)$ — the point is isolated in very few splits — the ratio is close to 0, so the exponent approaches 0 from below and $s \to 2^0 = 1$ (anomalous). If $E[h(\mathbf{x})] \approx c(n)$, the exponent is about $-1$ and $s \approx 0.5$ (no evidence either way). If $E[h(\mathbf{x})]$ is much larger than $c(n)$, the exponent is large and negative and $s \to 0$ (very normal). scikit-learn reports the related `decision_function`, where **lower** means more anomalous — negate it before blending it with scores where higher means more anomalous.

### THEORY: Local Outlier Factor (LOF)

LOF compares the local density of a point to the local densities of its neighbours. The local reachability density of point $\mathbf{x}$ is:

$$\text{lrd}_k(\mathbf{x}) = \left( \frac{\sum_{\mathbf{o} \in N_k(\mathbf{x})} \text{reach-dist}_k(\mathbf{x}, \mathbf{o})}{|N_k(\mathbf{x})|} \right)^{-1}$$

where $N_k(\mathbf{x})$ is the set of $k$ nearest neighbours and $\text{reach-dist}_k(\mathbf{x}, \mathbf{o}) = \max(d_k(\mathbf{o}), d(\mathbf{x}, \mathbf{o}))$ is the reachability distance. The LOF is the ratio of the average local density of the neighbours to the local density of the point itself:

$$\text{LOF}_k(\mathbf{x}) = \frac{\sum_{\mathbf{o} \in N_k(\mathbf{x})} \text{lrd}_k(\mathbf{o}) / \text{lrd}_k(\mathbf{x})}{|N_k(\mathbf{x})|}$$

A LOF near 1 means the point has similar density to its neighbours (normal). A LOF significantly greater than 1 means the point is in a sparser region than its neighbours (anomalous). LOF's strength is that it detects local anomalies — points that are normal globally but anomalous within their local neighbourhood.

### FOUNDATIONS: Score blending

No single anomaly detector is best in all cases. Z-score catches global outliers but misses local ones. LOF catches local anomalies but is sensitive to the choice of $k$. Isolation Forest is robust but has lower resolution in dense regions. Blending combines the strengths:

1. Put every detector on the same orientation (higher = more anomalous) and the same scale — min-max to $[0, 1]$, or ranks, which are robust to a detector with a few huge scores.
2. Compute a weighted average (or take the maximum).
3. Apply a threshold to the blended score.

The weights can be uniform, or tuned if labelled anomalies are available — but then tune them on one labelled sample and measure on another. Blending is insurance, not a guarantee: it protects you against relying on the one detector that is blind to the kind of anomaly you actually have, but an equal-weight blend can score *below* the best single detector (the worked example shows this).

### FOUNDATIONS: The masking trap

LOF compares a point with its $k$ nearest neighbours. If anomalies arrive as a *tight group* — say 40 near-identical applications from a coordinated ring — and $k$ is smaller than the group, every member's neighbours are the other members. Their local density matches their neighbours' density, LOF is close to 1, and the whole group looks normal. This is called **masking**. The cure is a neighbourhood larger than any plausible anomalous group, so the neighbourhood reaches the normal data around it. The worked example shows the effect directly.

## The Kailash Engines: AnomalyDetectionEngine and EnsembleEngine

`AnomalyDetectionEngine` fits Isolation Forest, LOF or a one-class SVM behind one `detect()` call. It flips scikit-learn's sign convention so that a higher score always means more anomalous, normalises the scores to $[0, 1]$ and applies the contamination threshold; `labels` follow scikit-learn's convention ($-1$ = anomaly, $1$ = normal). `ensemble_detect()` runs several detectors and combines them: `voting="score_average"` averages the normalised scores (the blend described above), the default `"majority"` votes on the labels. Extra keyword arguments such as `n_neighbors` pass through to the detector.

```python
import numpy as np
import polars as pl
from kailash_ml.engines.anomaly_detection import AnomalyDetectionEngine
from shared.mlfp04.ex_4 import load_dataset

X, y, cols, frame = load_dataset()     # standardised features; y used ONLY to evaluate
X_df = pl.from_numpy(X, schema=cols)

detector = AnomalyDetectionEngine()    # isolation_forest | lof | one_class_svm
lof_res = detector.detect(X_df, algorithm="lof", contamination=0.01, n_neighbors=50)
flagged = np.asarray(lof_res.labels) == -1
print(f"LOF flagged {lof_res.n_anomalies} rows; {int(y[flagged].sum())} are injected anomalies")

blend = detector.ensemble_detect(X_df, algorithms=["isolation_forest", "lof"],
                                 contamination=0.01, voting="score_average")
blend_scores = np.asarray(blend.combined_scores)  # normalised scores, averaged
```

`EnsembleEngine` is a different tool. Its `blend()`, `stack()`, `bag()` and `boost()` methods combine **supervised** models: `blend(models, data, target, weights=..., method="soft")` soft- or hard-votes fitted classifiers against a target column and does its own train/test split; `stack()` trains a meta-learner on their predictions. It cannot blend unsupervised anomaly scores — there is no target. It becomes useful once analysts have labelled a review sample: a classifier trained on the detector scores learns which detector to trust. Exercise 4.4 does exactly that, fitting the second stage on a labelled review sample and evaluating it on held-out rows.

## Worked Example: Credit Application Anomaly Detection

Real anomaly labels are rare, and a label made by thresholding a feature ("fraud = top 1% of returns") makes every evaluation circular — the detector that looks at that column wins by construction. Exercise 4 therefore follows the standard benchmark recipe: take 20,000 **real** applications from the course's Singapore credit-scoring dataset (`mlfp02/sg_credit_scoring.parquet`, 20 numeric fields) as the normal population and inject 200 anomalies of three known types, so the label comes from the injection, never from a feature threshold:

- **global** (80) — a real application with one field pushed 5–8 standard deviations above the mean (a fat-finger entry, an inflated declared balance);
- **dependency** (80) — every field copied from a *different* real application, a "synthetic identity" whose values are each plausible but whose combination is not;
- **clustered** (40) — a tight group of near-identical applications shifted 3 standard deviations on three fields, a coordinated application ring.

The detectors never see the label; it is used only to score them.

```python
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from shared.mlfp04.ex_4 import auc_by_type

# Method 1: Z-score — the most extreme standardised field of each row
z_scores = np.abs(X).max(axis=1)

# Method 2: Isolation Forest (negate decision_function: higher = more anomalous)
iforest = IsolationForest(n_estimators=200, contamination=0.01, random_state=42).fit(X)
iforest_scores = -iforest.decision_function(X)

# Method 3: LOF with a neighbourhood larger than any plausible ring
lof = LocalOutlierFactor(n_neighbors=50, contamination=0.01)
lof_labels = lof.fit_predict(X)
lof_scores = -lof.negative_outlier_factor_

def normalise(scores):
    return (scores - scores.min()) / (scores.max() - scores.min())

blended = (normalise(z_scores) + normalise(iforest_scores) + normalise(lof_scores)) / 3

for name, sc in [("Z-score", z_scores), ("Isolation Forest", iforest_scores),
                 ("LOF (k=50)", lof_scores), ("Equal blend", blended)]:
    top = sc > np.quantile(sc, 0.99)   # flag the top 1%, about 202 rows
    per_type = "  ".join(f"{t}={v:.2f}" for t, v in auc_by_type(frame, sc).items())
    print(f"{name:<17} AUC={roc_auc_score(y, sc):.3f}  AP={average_precision_score(y, sc):.3f}  "
          f"hits in top 1%={int(y[top].sum()):>3}  |  {per_type}")
```

| Detector | AUC-ROC | Avg. precision | Injected anomalies in top 1% | AUC: global | dependency | clustered |
| --- | --- | --- | --- | --- | --- | --- |
| Z-score | 0.81 | 0.15 | 61 | 1.00 | 0.55 | 0.97 |
| Isolation Forest | 0.77 | 0.03 | 3 | 0.78 | 0.67 | 0.98 |
| LOF ($k = 50$) | 0.97 | 0.63 | 121 | 1.00 | 0.98 | 0.89 |
| Equal blend | 0.91 | 0.19 | 56 | 0.99 | 0.80 | 0.98 |

Read the table by column, not just by the overall AUC.

- **Each detector sees different anomalies.** The Z-score finds the single-field extremes (global, AUC ≈ 1.00) but is blind to synthetic identities (0.55, barely better than chance) — no single field is unusual. LOF catches the synthetic identities (0.98), because their *combination* of values puts them in a sparse part of the space.
- **Isolation Forest ranks the injected anomalies above most applications (AUC 0.77) but almost never at the very top**: only 3 of its 202 highest-scoring rows are injected. Real credit data have heavy tails of their own — genuinely extreme but legitimate applications — and those are what Isolation Forest isolates first. "Statistically unusual" and "the anomaly you are looking for" are different questions.
- **Blending is insurance, not magic.** The equal blend's worst type is 0.80, far better than the Z-score's 0.55, but its overall AUC (0.91) is below LOF alone (0.97): averaging in a weaker detector dilutes a strong one. If labelled cases exist, tune the weights on one labelled sample and evaluate on another (Exercise 4.4); if they do not, the blend protects you from betting everything on the wrong detector.

## Try It Yourself

**Drill 1.** Apply the univariate Z-score method to the raw `loan_amount_sgd` column of `frame` with thresholds of 2, 3 and 4 standard deviations, and the IQR rule. How many rows does each flag, and how many of them are injected anomalies?

**Solution:**

```python
amount = frame["loan_amount_sgd"].to_numpy()
z_amount = np.abs((amount - amount.mean()) / amount.std())
for threshold in [2, 3, 4]:
    flag = z_amount > threshold
    print(f"|z| > {threshold}: {int(flag.sum()):>4} rows ({flag.mean():.2%}), "
          f"{int(y[flag].sum())} injected")

q1, q3 = np.percentile(amount, [25, 75])
iqr = q3 - q1
flag = (amount < q1 - 1.5 * iqr) | (amount > q3 + 1.5 * iqr)
print(f"IQR rule: {int(flag.sum())} rows, {int(y[flag].sum())} injected")
```

$|z| > 2$ flags 955 rows (4.7%) — far more than the 2.3% a normal distribution predicts beyond $\pm 2$, because loan amounts are right-skewed — but only 11 are injected. $|z| > 3$ flags 97 (3 injected), $|z| > 4$ flags 2 (both injected). The IQR rule flags 196 rows (4 injected). A univariate rule on one column catches only the anomalies that happen to involve that column, and on a skewed column it mostly flags legitimate large loans. Real anomaly detection has to look at all fields at once.

**Drill 2.** Run Isolation Forest with contamination 0.01, 0.02, 0.05 and 0.10. How does contamination change the number of flagged rows, the number of injected anomalies caught and the AUC?

**Solution:**

```python
for c in [0.01, 0.02, 0.05, 0.10]:
    model = IsolationForest(n_estimators=200, contamination=c, random_state=42).fit(X)
    flag = model.predict(X) == -1
    auc = roc_auc_score(y, -model.decision_function(X))
    print(f"contamination={c:.2f}: {int(flag.sum()):>5} flagged, "
          f"{int(y[flag].sum()):>3} injected caught, AUC={auc:.4f}")
```

The flagged count is simply contamination × 20,200 (202, 404, 1,010, 2,020), the catch rises (3, 10, 58, 70), and the AUC is identical (0.7742) every time. Contamination does not change the model or the ranking — it only moves the cut-off. Set it from your review capacity ("we can investigate 400 cases a week"), not as a tuning knob for accuracy.

**Drill 3.** Compare LOF with $k = 5, 20, 50$ neighbours. Report the overall AUC and the AUC per anomaly type. Which setting catches the coordinated ring (`clustered`)? Explain.

**Solution:**

```python
for k in [5, 20, 50]:
    scores = -LocalOutlierFactor(n_neighbors=k).fit(X).negative_outlier_factor_
    per_type = "  ".join(f"{t}={v:.3f}" for t, v in auc_by_type(frame, scores).items())
    print(f"k={k:>2}: AUC={roc_auc_score(y, scores):.3f}  |  {per_type}")
```

All three settings catch global and dependency anomalies (AUC 0.98–0.995). The ring is the difference: AUC 0.36 at $k = 5$, **0.22 at $k = 20$** — worse than random, LOF actively ranks the ring members as *more* normal than real applications — and 0.89 at $k = 50$. With 40 near-identical members, any $k$ below 40 finds only fellow members as neighbours: the masking trap. At $k = 50$ the neighbourhood reaches beyond the ring and its isolation becomes visible.

**Drill 4.** Implement a voting ensemble: flag a row if at least 2 of the 3 detectors flag it (each at its own top 1%). Compare its precision and recall on the injected anomalies with each single detector and with "any detector flags it".

**Solution:**

```python
z_flag = z_scores > np.quantile(z_scores, 0.99)
if_flag = iforest.predict(X) == -1
lof_flag = lof_labels == -1
votes = z_flag.astype(int) + if_flag.astype(int) + lof_flag.astype(int)

for name, flag in [("Z-score", z_flag), ("Isolation Forest", if_flag), ("LOF", lof_flag),
                   ("2-of-3 vote", votes >= 2), ("any detector", votes >= 1)]:
    hits = int(y[flag].sum())
    print(f"{name:<17} flagged={int(flag.sum()):>4}  precision={hits / max(flag.sum(), 1):.2f}  "
          f"recall={hits / y.sum():.2f}")
```

The 2-of-3 vote flags 115 rows with precision 0.52 and recall 0.30; LOF alone has precision 0.60 and recall 0.60; "any detector" reaches recall 0.61 at precision 0.25. Majority voting is conservative: a row must look odd to two detectors that see *different* kinds of oddity, so it misses anomalies only one detector can see (synthetic identities are LOF-only). Voting suits a high-cost-per-review setting; "any" suits a setting where missing an anomaly is the expensive mistake.

**Drill 5.** Simulate the silent failure from the introduction: replace the Isolation Forest scores with a constant 0.5. What happens to the blend? Write a health check that would catch this in production.

**Solution:**

```python
if_broken = np.full_like(iforest_scores, 0.5)
blended_broken = (normalise(z_scores) + if_broken + normalise(lof_scores)) / 3
threshold = np.quantile(blended, 0.99)
print(f"Rows above the old threshold: working={int((blended > threshold).sum())}, "
      f"broken={int((blended_broken > threshold).sum())}")

def check_detector_health(scores, name, min_std=1e-3, min_unique=10):
    scores = np.asarray(scores, dtype=float)
    if np.std(scores) < min_std or np.unique(scores).size < min_unique:
        raise ValueError(f"Detector '{name}' looks broken: std={np.std(scores):.2e}, "
                         f"unique values={np.unique(scores).size}")

check_detector_health(iforest_scores, "isolation_forest")      # passes
try:
    check_detector_health(if_broken, "isolation_forest")
except ValueError as err:
    print(err)
```

The broken blend still produces scores and still flags rows — nothing crashes — but against the old threshold it flags 111 rows instead of 202 and its ranking now ignores one third of the evidence. A constant or near-constant score is cheap to detect: check the spread and the number of distinct values of every detector's output on every batch, and alert on the flag *rate* over time, not only on individual flags.

## Cross-References

- **Module 2, Lesson 2.1** introduced the Z-score and the concept of statistical outliers. This lesson extends that to multivariate settings with ML-based detectors.
- **Module 3, Lesson 3.5** covered evaluation metrics. Anomaly detection is an extreme class-imbalance problem — precision-recall is more informative than accuracy.
- **Module 3, Lesson 3.8** introduced drift monitoring. The two are related but different: anomaly detection scores *individual rows* as unusual, while `DriftMonitor` compares whole *distributions* (PSI, Kolmogorov–Smirnov and similar tests) between a reference period and a new one. A rising anomaly-flag rate is often the first sign of drift.
- **Lesson 4.1** used clustering to find groups. Anomaly detection finds the points that do not belong to any group.

## Reflection

You should now be able to:

- Apply Z-score and IQR methods and explain their limitations for multivariate data.
- Explain how Isolation Forest works (shorter path = more anomalous) and derive its anomaly score formula.
- Explain how LOF compares local densities and why it catches anomalies that global methods miss.
- Blend scores from multiple detectors using normalisation and weighted averaging, and explain why a blend can be more robust yet score below the best single detector.
- Recognise the LOF masking trap and choose a neighbourhood size larger than any plausible anomalous group.
- Use `AnomalyDetectionEngine` for unsupervised detection and explain why `EnsembleEngine` is a supervised second stage.
- Design monitoring checks that detect when a detector has silently failed.

---

# Lesson 4.5: Association Rules and Market Basket Analysis

## Why This Matters

Walk into almost any supermarket and look at the shelf layout. Beer is near snacks. Nappies are near baby wipes. Fresh fruit is near yoghurt. Many such placements are informed by co-purchase patterns in transaction data. (The famous "nappies and beer" story is a retail-analytics anecdote whose original source is disputed — treat it as folklore that illustrates the idea, not as a documented finding.)

Association rule mining is the algorithm behind these discoveries. It takes a database of transactions (each transaction is a set of items) and finds rules of the form "if a customer buys X, they are likely to also buy Y". The rules are scored by support (how often X and Y appear together), confidence (how often Y appears when X is present), and lift (how much more likely Y is given X, compared to its baseline rate).

This lesson is not a dead end. Association rules discover co-occurrence patterns — features. Those features can be used as inputs to supervised models from Module 3. And the idea of "discovering patterns in co-occurrence data" is exactly what collaborative filtering does with embeddings in Lesson 4.7.

## Core Concepts

### FOUNDATIONS: Transaction data

A transaction database is a collection of transactions, where each transaction is a set of items. A supermarket receipt is a transaction; the items are the products purchased. A web session is a transaction; the items are the pages visited. A medical record is a transaction; the items are the diagnoses.

### THEORY: Support, confidence, and lift

Given items $X$ and $Y$:

**Support** measures how frequently the combination appears:

$$\text{supp}(X) = \frac{|\{t \in T : X \subseteq t\}|}{|T|}$$

**Confidence** measures the reliability of the rule $X \to Y$:

$$\text{conf}(X \to Y) = \frac{\text{supp}(X \cup Y)}{\text{supp}(X)}$$

This is the conditional probability $P(Y \mid X)$.

**Lift** measures the surprise factor — how much more likely $Y$ is given $X$ compared to $Y$'s baseline:

$$\text{lift}(X \to Y) = \frac{\text{conf}(X \to Y)}{\text{supp}(Y)} = \frac{P(X \cap Y)}{P(X) \cdot P(Y)}$$

A lift of 1 means $X$ and $Y$ are independent. Lift $> 1$ means positive association (buying $X$ makes $Y$ more likely). Lift $< 1$ means negative association.

### FOUNDATIONS: The Apriori algorithm

Apriori finds frequent itemsets — sets of items whose support exceeds a minimum threshold — using a bottom-up approach:

1. Find all items with support $\geq$ min_support (frequent 1-itemsets).
2. Generate candidate 2-itemsets from frequent 1-itemsets.
3. Count support of candidates, keep those above threshold.
4. Repeat, growing the itemset size by 1 each iteration.
5. Stop when no new frequent itemsets are found.

The key insight is the **Apriori principle**: if an itemset is infrequent, all its supersets are also infrequent. This allows aggressive pruning of the search space.

### FOUNDATIONS: FP-Growth

FP-Growth (Frequent Pattern Growth) avoids candidate generation entirely. It compresses the transaction database into a compact data structure called an FP-tree, then extracts frequent patterns directly from the tree. Because it never generates or counts candidate itemsets, FP-Growth is typically much faster than Apriori on large, dense transaction databases with many products and low support thresholds. On small problems the difference disappears and can even reverse (Drill 1).

## Mathematical Foundations

### THEORY: Why lift measures independence departure

Two events $X$ and $Y$ are independent if and only if $P(X \cap Y) = P(X) \cdot P(Y)$. The lift is the ratio:

$$\text{lift}(X \to Y) = \frac{P(X \cap Y)}{P(X) \cdot P(Y)}$$

Under independence this ratio is exactly 1. A lift of 2 means the joint occurrence is twice as frequent as independence would predict. A lift of 0.5 means it is half as frequent — the items are negatively associated (buying one makes the other less likely).

Lift is symmetric: $\text{lift}(X \to Y) = \text{lift}(Y \to X)$. Confidence is not symmetric: $\text{conf}(X \to Y) \neq \text{conf}(Y \to X)$ in general. This is an important distinction when interpreting rules.

## The Kailash Engine: none — and why that is fine

kailash-ml has no association-rule engine, and you should not go looking for one. The mining itself is a few lines of Python (the Apriori principle above), and for larger problems the open-source `mlxtend` library provides Apriori and FP-Growth. The Kailash value enters *after* mining: discovered rules become features for the supervised pipeline you built in Module 3 (Drill 5), and Exercise 5 logs every mining run to the `ExperimentTracker` so you can compare thresholds and algorithms.

## Worked Example: Singapore Mini-Mart Basket Analysis

The baskets in this example are **synthetic**. Exercise 5's generator (`shared.mlfp04.ex_5.generate_transactions`) simulates 2,500 transactions at a neighbourhood mini-mart with 25 products. Twelve "bundles" — kaya-toast breakfast (bread, butter, eggs), kopi (coffee, condensed milk, sugar), beer and chips, toiletries, household cleaning and so on — fire with known probabilities, each item in a firing bundle is dropped 15% of the time, and a few random impulse items are added. Because we know the bundles that generated the data, we can check whether the mining recovers them.

### Step 1: Pair rules from first principles

```python
import numpy as np
import polars as pl
from shared.mlfp04.ex_5 import generate_transactions, transactions_to_onehot

transactions = generate_transactions(n=2500, seed=42)   # list of sets of product names
basket = transactions_to_onehot(transactions)           # polars: one boolean column per product
print(basket.shape, f"avg basket = {np.mean([len(t) for t in transactions]):.2f} items")

B = basket.to_numpy().astype(np.float64)                # 2,500 x 25 indicator matrix
items = basket.columns
n = B.shape[0]
support_1 = B.mean(axis=0)                              # supp(X) for every single item
support_2 = (B.T @ B) / n                               # supp(X and Y) for every pair

pair_rules = pl.DataFrame(
    [
        (items[i], items[j], support_2[i, j],
         support_2[i, j] / support_1[i],                          # confidence
         support_2[i, j] / (support_1[i] * support_1[j]))         # lift
        for i in range(len(items)) for j in range(len(items))
        if i != j and support_2[i, j] >= 0.02
    ],
    schema=["antecedent", "consequent", "support", "confidence", "lift"],
    orient="row",
).sort("lift", descending=True)
print(pair_rules.head(6))
```

The product of the indicator matrix with itself counts every pair's co-occurrence in one line; support, confidence and lift follow directly from their definitions. The top pairs are shampoo ↔ toothpaste (lift 3.02), cooking oil ↔ fish (2.93), coffee ↔ condensed milk (2.92) and detergent ↔ tissue (2.91) — all pairs from bundles in the generator. Note the two directions of each pair: the lift is identical, the confidence is not (coffee → condensed milk 0.52, condensed milk → coffee 0.54).

### Step 2: All itemset sizes with FP-Growth

For itemsets larger than pairs we use `mlxtend`. It accepts only a pandas DataFrame, so we convert **at the call boundary only** and bring the result straight back into polars — all filtering and sorting stays in polars.

```python
from mlxtend.frequent_patterns import association_rules, fpgrowth

freq_items = fpgrowth(basket.to_pandas(), min_support=0.02, use_colnames=True)
rules_raw = association_rules(freq_items, metric="lift", min_threshold=1.2)

def rules_to_polars(raw):
    return pl.DataFrame({
        "antecedent": [", ".join(sorted(a)) for a in raw["antecedents"]],
        "consequent": [", ".join(sorted(c)) for c in raw["consequents"]],
        "support": raw["support"].to_numpy(),
        "confidence": raw["confidence"].to_numpy(),
        "lift": raw["lift"].to_numpy(),
    }).sort("lift", descending=True)

rules = rules_to_polars(rules_raw)
print(f"Frequent itemsets: {len(freq_items)}   rules with lift >= 1.2: {rules.height}")
print(rules.head(6))
```

At a minimum support of 2%, FP-Growth finds 336 frequent itemsets (25 single items, 226 pairs, 77 triples, 8 four-item sets) and 494 rules with lift of at least 1.2. The strongest are three-item bundles: {soap, tissue} → {detergent} with confidence 0.76 and lift 6.0, {soap, toothpaste} → {shampoo} (0.75, 5.6), {cooking oil, rice} → {fish} (0.69, 5.1).

### Interpreting the top rules

Read {soap, tissue} → {detergent} as: 3.8% of all baskets contain all three items (support); 76% of baskets with soap and tissue also contain detergent (confidence); and that is 6.0 times the rate at which detergent appears in baskets in general (lift). For a mini-mart this suggests shelving household cleaning together or bundling a "household restock" promotion. The reverse rule {detergent} → {soap, tissue} has the same lift but a confidence of only 0.30 — most detergent buyers do not buy the other two — so the promotion should be triggered by soap and tissue, not by detergent. And because these baskets are synthetic, we can confirm the method works: every top rule is a fragment of one of the generator's bundles.

## Try It Yourself

**Drill 1.** Run both Apriori and FP-Growth on the basket data with min_support = 0.02 and with 0.005. Compare execution time. Which is faster, and does that match the textbook claim that FP-Growth is faster?

**Solution:**

```python
import time
from mlxtend.frequent_patterns import apriori

basket_pd = basket.to_pandas()   # boundary conversion, once
for min_sup in [0.02, 0.005]:
    timings = {}
    for name, algo in [("Apriori", apriori), ("FP-Growth", fpgrowth)]:
        start = time.perf_counter()
        found = algo(basket_pd, min_support=min_sup, use_colnames=True)
        timings[name] = time.perf_counter() - start
    print(f"min_support={min_sup}: {len(found)} itemsets  " +
          "  ".join(f"{k}={v * 1000:.1f} ms" for k, v in timings.items()))
```

Both find identical itemsets, in milliseconds, and on this small problem (25 products, 2,500 baskets) Apriori is often the *faster* one — the exact timings vary from run to run and machine to machine. FP-Growth's advantage comes from never generating candidate itemsets, which pays off when there are thousands of products, millions of baskets and low support thresholds, where Apriori's candidate sets explode. Benchmarks on toy data do not transfer to scale; measure on data shaped like yours.

**Drill 2.** Find all rules with lift > 2 and confidence > 0.3. How many rules satisfy both conditions? What is the highest-lift rule, and does it make business sense?

**Solution:**

```python
strong_rules = rules.filter((pl.col("lift") > 2) & (pl.col("confidence") > 0.3))
print(f"Strong rules: {strong_rules.height}")
top = strong_rules.row(0, named=True)
print(f"Top rule: {{{top['antecedent']}}} -> {{{top['consequent']}}}  "
      f"lift={top['lift']:.2f}, confidence={top['confidence']:.2f}")
```

240 of the 494 rules pass both filters. The top one is {soap, tissue} → {detergent} (lift 6.02, confidence 0.76): a household-restocking trip, which is a sensible, actionable pattern. Many of the 240 are re-arrangements of the same few bundles — when presenting rules to a business audience, group them by the underlying itemset rather than listing every direction.

**Drill 3.** Demonstrate that lift is symmetric but confidence is not. Pick a rule $X \to Y$, find the reverse rule $Y \to X$, and compare both measures.

**Solution:**

```python
rule = rules.row(0, named=True)
reverse = rules.filter(
    (pl.col("antecedent") == rule["consequent"]) & (pl.col("consequent") == rule["antecedent"])
).row(0, named=True)

print(f"conf(X->Y)={rule['confidence']:.3f}   conf(Y->X)={reverse['confidence']:.3f}")
print(f"lift(X->Y)={rule['lift']:.3f}   lift(Y->X)={reverse['lift']:.3f}")
```

For {soap, tissue} ↔ {detergent} the confidences are 0.758 and 0.298 while both lifts are 6.016. Lift divides the joint support by the product of the two marginal supports, which does not depend on direction; confidence divides by the antecedent's support only.

**Drill 4.** Lower the minimum support threshold from 0.02 to 0.005. How many frequent itemsets are found now, and how are they distributed by size?

**Solution:**

```python
freq_low = fpgrowth(basket_pd, min_support=0.005, use_colnames=True)
sizes = pl.Series("size", [len(s) for s in freq_low["itemsets"]])
print(sizes.value_counts().sort("size"))
print(f"Total at 0.005: {len(freq_low)}, at 0.02: {len(freq_items)}")
```

The count jumps from 336 to 1,851 itemsets: 25 singles, 300 pairs (every possible pair of the 25 products), 1,007 triples, 448 four-item, 68 five-item and 3 six-item sets. Lowering support by a factor of four multiplied the output by more than five, and most of the new itemsets are combinations of random impulse purchases that occur in 13–50 baskets out of 2,500 — rules built on them are noise. Low support needs a stricter lift or confidence filter, or a statistical test, to stay useful.

**Drill 5.** Use association rules as features for a supervised model. Each shopper has two consecutive trips (`generate_shopper_trips`). Build features from the *first* trip — product presence alone (baseline), and product presence plus rule features — and predict whether the *next* trip contains at least two of bread, butter and eggs. Compare the test AUC of a logistic regression with and without the rule features. Why must the target come from a different trip?

**Solution:**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from shared.mlfp04.ex_5 import generate_shopper_trips

this_trip, next_trip = generate_shopper_trips(n_shoppers=2500, seed=42)
breakfast = {"bread", "butter", "eggs"}
y = np.array([int(len(nxt & breakfast) >= 2) for nxt in next_trip])

onehot = transactions_to_onehot(this_trip)
trip_rules = rules_to_polars(association_rules(
    fpgrowth(onehot.to_pandas(), min_support=0.03, use_colnames=True),
    metric="lift", min_threshold=1.5,
)).filter(pl.col("confidence") >= 0.4).head(20)

def has_all(itemset_text):
    return pl.all_horizontal([pl.col(item) for item in itemset_text.split(", ")])

rule_features = onehot.select(
    [has_all(r["antecedent"]).cast(pl.Int8).alias(f"rule{i}_antecedent")
     for i, r in enumerate(trip_rules.iter_rows(named=True))]
    + [(has_all(r["antecedent"]) & has_all(r["consequent"])).cast(pl.Int8).alias(f"rule{i}_full")
       for i, r in enumerate(trip_rules.iter_rows(named=True))]
)
X_base = onehot.to_numpy().astype(np.float64)
X_rules = np.hstack([X_base, rule_features.to_numpy().astype(np.float64)])

for name, X_feat in [("products only", X_base), ("products + rules", X_rules)]:
    X_tr, X_te, y_tr, y_te = train_test_split(X_feat, y, test_size=0.3,
                                              random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    print(f"{name:<17} test AUC = {roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]):.3f}")
print(f"positive rate: {y.mean():.1%}, rule features: {rule_features.width}")
```

The target must come from the next trip because a target computed from this trip's items — "the basket is big", "the basket contains the breakfast bundle" — is a function of the very columns the model sees, so a linear model recovers it almost perfectly and no feature can add anything (target leakage). With the honest next-trip target, the product-presence baseline has a test AUC of about 0.68 (Exercise 5.4 reports about 0.69 with its own rule miner) and adding the 40 rule features leaves it essentially unchanged (0.679 here). That is the realistic finding: rules mined from the same columns are mostly re-combinations of information a model already has; their value is interpretability and compact interaction terms for linear models, not a large accuracy gain. The same rules are not mined from the test rows' labels, but they are mined from all first trips — for a strict evaluation, mine them on the training rows only.

## Cross-References

- **Module 3, Lesson 3.1** introduced feature engineering. Association rules are a form of automated interaction-feature discovery.
- **Lesson 4.7** extends co-occurrence analysis to collaborative filtering, where the "co-occurrence" is user-item interactions and the output is learned embeddings.
- **Module 6, Lesson 6.4** uses retrieval methods (BM25, embedding similarity) that share the same mathematical foundation as co-occurrence statistics.

## Reflection

You should now be able to:

- Explain the Apriori principle and why it enables efficient pruning.
- Compute support, confidence, and lift from a transaction database.
- Distinguish between symmetric (lift) and asymmetric (confidence) measures.
- Use discovered rules as features for supervised models, with a target that is not a function of those features.
- Keep the mining pipeline in polars and convert to pandas only at a library boundary that requires it.
- Evaluate whether a discovered rule is actionable in a business context.

---

# Lesson 4.6: NLP — Text to Topics

## Why This Matters

Imagine a policy analyst facing 10,000 public submissions on a new housing regulation, or a newsroom archive of hundreds of thousands of articles. Nobody can read them all. Topic modelling can automatically discover the main themes — affordability concerns, construction quality, green building requirements, accessibility — and quantify how much attention each theme receives. Public consultations, customer-feedback inboxes and news archives all have this shape. The worked example uses a public news corpus with human-assigned section labels, so you can check what the unsupervised topics actually recover.

In this lesson you will learn to transform raw text into features that ML models can consume. The journey starts with the simplest representation (bag of words), moves through TF-IDF and BM25, touches word embeddings, and arrives at modern topic modelling with LDA and BERTopic. Along the way, you will derive TF-IDF from first principles and understand why it works.

## Core Concepts

### FOUNDATIONS: Text as data

Machines operate on numbers, not words. To apply ML to text, you must convert text into a numeric representation. The simplest representation is the bag of words: count how many times each word appears in each document. The result is a matrix where rows are documents, columns are unique words (the vocabulary), and values are counts.

The bag of words discards word order. "The dog bit the man" and "The man bit the dog" have the same representation. This is a severe limitation, but it is sufficient for many tasks, including topic modelling and document classification.

### THEORY: TF-IDF derivation

Term Frequency-Inverse Document Frequency weights each word by how important it is to a document, relative to the entire corpus.

**Term Frequency (TF):** how often word $t$ appears in document $d$:

$$\text{tf}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$

where $f_{t,d}$ is the raw count of $t$ in $d$, normalised by the total number of words in $d$.

**Inverse Document Frequency (IDF):** how rare the word is across the corpus:

$$\text{idf}(t) = \log \frac{N}{\text{df}(t)}$$

where $N$ is the total number of documents and $\text{df}(t)$ is the number of documents containing $t$.

**TF-IDF** is the product:

$$\text{tfidf}(t, d) = \text{tf}(t, d) \times \text{idf}(t)$$

Why does this work? A word that appears frequently in a document (high TF) is important to that document. But if that word also appears in every document (low IDF — e.g., "the", "is", "and"), it carries no discriminative information. The IDF term down-weights common words and up-weights rare, document-specific words.

**Variants matter in practice.** The formula above is the textbook form. Libraries differ: scikit-learn's `TfidfVectorizer` by default uses raw counts for tf, a *smoothed* idf $\ln\frac{1 + N}{1 + \text{df}(t)} + 1$ (as if one extra document contained every word, so no idf is zero or undefined), and then scales each document vector to unit length (L2 normalisation). The ranking intuition is the same, but the numbers are not — Drill 1 reproduces both.

### THEORY: BM25

BM25 (Best Matching 25) is an improved version of TF-IDF used in information retrieval. It introduces two refinements:

**Term frequency saturation.** In TF-IDF, doubling the word count doubles the score. In BM25, the term frequency component saturates:

$$\text{BM25}(t, d) = \text{idf}(t) \times \frac{f_{t,d} \times (k_1 + 1)}{f_{t,d} + k_1 \times (1 - b + b \times \frac{|d|}{|d_{\text{avg}}|})}$$

where $k_1$ (typically 1.2–2.0) controls saturation and $b$ (typically 0.75) controls document-length normalisation. The intuition: a word appearing 10 times versus 5 times in a document is not twice as important — there are diminishing returns. And longer documents are expected to have higher counts simply because they have more words, so the score is normalised by document length relative to the average.

BM25 will appear again in Module 6, Lesson 6.4, as the sparse retrieval component of RAG systems.

### FOUNDATIONS: Word embeddings (tools, not derivation)

Word embeddings represent each word as a dense vector in a continuous space, where semantically similar words are close together. Three major approaches:

- **Word2Vec (CBOW and Skip-gram):** learns embeddings by predicting a word from its context (CBOW) or context from a word (Skip-gram). The key insight: "words that appear in similar contexts have similar meanings."
- **GloVe:** learns embeddings from global word co-occurrence statistics.
- **FastText:** extends Word2Vec with subword embeddings (character n-grams), so it can handle out-of-vocabulary words.

We use these embeddings as features — we do not yet derive how they are trained. That derivation comes in Lesson 4.8, where you will see that Word2Vec is a shallow neural network, and the embedding vectors are its hidden layer weights. The connection to the Feature Engineering Spectrum: embeddings are features discovered by optimisation, not by hand.

### THEORY: LDA — Latent Dirichlet Allocation

LDA is a generative probabilistic model for topic discovery. It assumes each document is a mixture of topics, and each topic is a distribution over words:

$$p(\text{word} \mid \text{document}) = \sum_{k=1}^{K} p(\text{word} \mid \text{topic}_k) \times p(\text{topic}_k \mid \text{document})$$

The generative process for a document:

1. Choose a topic distribution $\theta_d \sim \text{Dirichlet}(\alpha)$.
2. For each word position in the document:
   a. Choose a topic $z \sim \text{Multinomial}(\theta_d)$.
   b. Choose a word $w \sim \text{Multinomial}(\phi_z)$.

The model is fitted either by variational inference — a *variational EM* procedure that generalises the EM of Lesson 4.2 by replacing the exact E-step with an approximate posterior; this is what scikit-learn's `LatentDirichletAllocation` uses — or by collapsed Gibbs sampling, which is a Markov chain Monte Carlo method, not EM.

Because LDA is a generative model of **word counts**, it must be fitted on a count matrix (`CountVectorizer`), not on TF-IDF weights: TF-IDF values are not counts, so feeding them to LDA violates its likelihood. NMF has no such requirement and is usually fitted on TF-IDF.

### FOUNDATIONS: BERTopic

BERTopic is a modern topic modelling approach that combines:

1. **Sentence embeddings** (from a pre-trained transformer like BERT) to represent documents as dense vectors.
2. **UMAP** (from Lesson 4.3) to reduce dimensionality.
3. **HDBSCAN** (from Lesson 4.1) to cluster the reduced embeddings.
4. **c-TF-IDF** (class-based TF-IDF) to extract topic labels from each cluster.

Because it starts from pre-trained sentence embeddings rather than raw word counts, BERTopic often produces more interpretable topics than LDA on short texts, and it chooses the number of topics itself (HDBSCAN). The costs: it needs an embedding model (downloaded once, then run for every document), it is slower, and HDBSCAN assigns some documents to an outlier topic $-1$.

### THEORY: Topic coherence — NPMI

Normalised Pointwise Mutual Information (NPMI) measures how often the top words in a topic co-occur in the corpus:

$$\text{NPMI}(w_i, w_j) = \frac{\log \frac{p(w_i, w_j)}{p(w_i) \cdot p(w_j)}}{-\log p(w_i, w_j)}$$

NPMI ranges from $-1$ (the two words never appear in the same document) through 0 (they co-occur exactly as often as chance predicts) to $+1$ (they always co-occur). A topic's coherence is the average NPMI across all pairs of its top words; higher means the topic's words genuinely belong together. The probabilities are usually document frequencies computed on a reference corpus (here, the corpus itself), and the absolute values depend on that corpus, the number of top words and how never-co-occurring pairs are scored — so compare models on the same setup rather than against published "typical" ranges. The course implementation is `shared.mlfp04.ex_6.compute_npmi`.

**UMass coherence** (Mimno et al., 2011) is the other common metric. For a topic's top words ordered by frequency, it sums over pairs with $i > j$

$$C_{\text{UMass}} = \sum_{i>j} \log \frac{D(w_i, w_j) + 1}{D(w_j)}$$

where $D(w)$ is the number of documents containing $w$ and $D(w_i, w_j)$ the number containing both; the $+1$ avoids $\log 0$. It is computed on the training corpus itself, is always $\le$ about 0, and values closer to 0 are better. UMass is cheap but less correlated with human judgements than NPMI.

## The Kailash Engine: DimReductionEngine (NMF)

There is no Kailash "topic modelling" engine. Topic extraction with NMF *is* dimensionality reduction of the document–term matrix, so it runs through the same `DimReductionEngine` you met in Lesson 4.3, with `algorithm="nmf"`. The engine takes a polars frame (one column per vocabulary term), checks that the input is non-negative, and returns the document–topic weights $\mathbf{W}$ in `transformed`. It does not return the topic–word matrix $\mathbf{H}$; when you need topic words, use scikit-learn's `NMF` directly (worked example) or rank terms by $\mathbf{W}^T \mathbf{X}$.

```python
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from kailash_ml.engines.dim_reduction import DimReductionEngine
from shared.mlfp04.ex_6 import NEWS_STOP_WORDS, corpus_as_lists, load_corpus

documents, categories = corpus_as_lists(load_corpus())
tfidf = TfidfVectorizer(max_features=2000, stop_words=NEWS_STOP_WORDS, min_df=3, max_df=0.95)
X_tfidf = tfidf.fit_transform(documents).toarray()          # dense: the engine takes a frame
matrix_df = pl.from_numpy(X_tfidf, schema=[f"t{i}" for i in range(X_tfidf.shape[1])])

nmf_res = DimReductionEngine().reduce(matrix_df, algorithm="nmf", n_components=4, seed=42)
W = np.asarray(nmf_res.transformed)                         # documents x topics
doc_topic = W.argmax(axis=1)
print(f"documents per topic: {np.bincount(doc_topic).tolist()}")

vocab = tfidf.get_feature_names_out()
for k, weights in enumerate(W.T @ X_tfidf):                  # term weight per topic
    print(f"topic {k}: {', '.join(vocab[np.argsort(weights)[::-1][:6]])}")
```

This is the engine the module assessment's topic task uses.

## Worked Example: Topics in a Public News Corpus

The corpus is AG News (`mlfp05/ag_news.parquet`): the title and lead sentence of 5,000 English news stories from 2004, each labelled by humans with one of four sections — world, sports, business and sci/tech. `load_corpus()` cleans encoding artefacts and removes duplicate stories, leaving 4,967 documents. The section labels are **never** used to fit a topic model, only afterwards to see what the topics line up with.

```python
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import normalized_mutual_info_score
from shared.mlfp04.ex_6 import compute_npmi

print(f"{len(documents):,} documents; sections: {sorted(set(categories))}")
N_TOPICS = 8

# NMF works on TF-IDF weights
tfidf_vec = TfidfVectorizer(max_features=3000, stop_words=NEWS_STOP_WORDS, min_df=3, max_df=0.95)
tfidf_matrix = tfidf_vec.fit_transform(documents)
tfidf_vocab = tfidf_vec.get_feature_names_out()
nmf = NMF(n_components=N_TOPICS, init="nndsvd", max_iter=400, random_state=42)
W_nmf = nmf.fit_transform(tfidf_matrix)       # document-topic matrix
nmf_topics = [[tfidf_vocab[j] for j in comp.argsort()[::-1][:10]] for comp in nmf.components_]

# LDA is a model of word COUNTS, so it gets a count matrix
count_vec = CountVectorizer(max_features=3000, stop_words=NEWS_STOP_WORDS, min_df=3, max_df=0.95)
count_matrix = count_vec.fit_transform(documents)
count_vocab = count_vec.get_feature_names_out()
lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=42)
theta_lda = lda.fit_transform(count_matrix)   # document-topic proportions
lda_topics = [[count_vocab[j] for j in comp.argsort()[::-1][:10]] for comp in lda.components_]

for name, topics, vec, doc_topic in [("NMF", nmf_topics, tfidf_vec, W_nmf),
                                     ("LDA", lda_topics, count_vec, theta_lda)]:
    npmi = compute_npmi(documents, topics, analyzer=vec.build_analyzer())
    nmi = normalized_mutual_info_score(categories, doc_topic.argmax(axis=1))
    print(f"\n{name}: mean NPMI = {np.mean(npmi):+.3f}   NMI with sections = {nmi:.3f}")
    for k, words in enumerate(topics):
        print(f"  topic {k}: {', '.join(words[:8])}")
```

The NMF topics are sharp and specific: an initial public offering by Google (`google, ipo, public, offering, initial, price, stock`), oil prices (`oil, prices, stocks, record, crude, barrel`), the Athens Olympics (`athens, gold, olympic, phelps, medal`), the fighting in Najaf (`najaf, iraq, sadr, cleric, shrine`), a Windows XP security update (`microsoft, windows, xp, update, sp2`), quarterly earnings, and Israeli politics (`sharon, minister, prime, gaza`); the eighth topic is a leftover mix (`court, apple, music, company, software, ibm, hurricane`). These are *stories*, not the four broad sections — on short news leads, word co-occurrence is dominated by the big stories of the week. Mean NPMI is about +0.41.

The LDA topics are broader and noisier: a technology-company topic, an Olympics topic, an oil-and-earnings topic, and several mixed topics that blend unrelated stories (Google's IPO with Najaf; a hurricane with police reports and bombings). Mean NPMI is about −0.02, because several top-word pairs never appear in the same document. Neither model recovers the four human sections well (normalised mutual information with the sections is about 0.28 for NMF and 0.23 for LDA): topic models find whatever co-occurrence structure is strongest, which need not be the categorisation a human editor would use.

```python
import os
from bertopic import BERTopic
from umap import UMAP
from shared.mlfp04.ex_6 import topic_embedding_model

# The sentence-embedding model name comes from TOPIC_EMBED_MODEL in .env (never hardcoded)
topic_model = BERTopic(
    embedding_model=topic_embedding_model(),
    umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42),
    vectorizer_model=CountVectorizer(stop_words=NEWS_STOP_WORDS, min_df=2),
    min_topic_size=15,
    nr_topics="auto",
)
bert_topics, _ = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(f"BERTopic: {int((info['Topic'] >= 0).sum())} topics, "
      f"{sum(t == -1 for t in bert_topics):,} outlier documents")
for _, row in info[info["Topic"] >= 0].head(6).iterrows():
    print(f"  topic {row['Topic']}: {row['Count']} docs  {row['Name']}")
```

BERTopic embeds each document with a pre-trained sentence-transformer, reduces the embeddings with UMAP to about five dimensions, clusters them with HDBSCAN and labels each cluster with class-based TF-IDF — the same pipeline as Exercise 6.4. It chooses the number of topics itself and puts documents that fit no dense cluster into topic $-1$. The exact topics depend on the embedding model you configure. With the small `sentence-transformers/all-MiniLM-L6-v2` model, one run produced 40 topics and put 1,223 documents (about 25%) in the outlier topic; the largest topics were the Olympics (921 documents), Najaf, Windows and online music, Google's IPO, quarterly profits and space exploration — the same stories NMF found, plus dozens of smaller ones. `get_topic_info()` returns a pandas table (BERTopic's own format), which is why the loop above uses pandas-style `iterrows()`; convert with `pl.from_pandas(info)` if you want to keep working in polars.

## Try It Yourself

**Drill 1.** Implement TF-IDF from scratch on a small corpus of 5 documents using the textbook formulas. Then reproduce scikit-learn's `TfidfVectorizer` output exactly by switching to its conventions (raw counts, smoothed idf, L2 normalisation), and verify with `np.allclose`.

**Solution:**

```python
from collections import Counter

corpus = [
    "singapore housing policy affordable homes",
    "affordable housing development singapore plan",
    "green building construction sustainability",
    "housing affordability young families singapore",
    "sustainable urban development green spaces",
]
vocab = sorted({word for doc in corpus for word in doc.split()})
counts = np.array([[Counter(doc.split())[w] for w in vocab] for doc in corpus], dtype=float)
N = len(corpus)
df_t = (counts > 0).sum(axis=0)

# Textbook TF-IDF: length-normalised tf x log(N / df)
tf = counts / counts.sum(axis=1, keepdims=True)
tfidf_textbook = tf * np.log(N / df_t)

# scikit-learn's default: raw counts x (ln((1+N)/(1+df)) + 1), then unit-length rows
tfidf_sklearn_style = counts * (np.log((1 + N) / (1 + df_t)) + 1)
tfidf_sklearn_style /= np.linalg.norm(tfidf_sklearn_style, axis=1, keepdims=True)

reference = TfidfVectorizer().fit_transform(corpus).toarray()
print("textbook formula == sklearn?     ", np.allclose(tfidf_textbook, reference))
print("sklearn conventions == sklearn?  ", np.allclose(tfidf_sklearn_style, reference))
print("'singapore' idf (textbook):", round(float(np.log(N / df_t[vocab.index('singapore')])), 3))
```

The textbook version does not match scikit-learn (`False`); the version with scikit-learn's conventions matches exactly (`True`). Both rank "singapore" (in 3 of 5 documents, idf $\ln(5/3) \approx 0.51$) below words that appear in one document only. When you compare TF-IDF numbers across tools, check which tf, idf and normalisation each one uses.

**Drill 2.** Compare NMF and LDA on the news corpus with NPMI coherence, using the course's `compute_npmi` (no extra library needed). Report the coherence of each topic, not just the mean. Which topics drag LDA's mean down?

**Solution:**

```python
for name, topics, vec in [("NMF", nmf_topics, tfidf_vec), ("LDA", lda_topics, count_vec)]:
    scores = compute_npmi(documents, topics, analyzer=vec.build_analyzer())
    print(f"{name}: mean NPMI {np.mean(scores):+.3f}")
    for words, score in sorted(zip(topics, scores), key=lambda t: t[1]):
        print(f"   {score:+.3f}  {', '.join(words[:5])}")
```

Seven of the eight NMF topics score between +0.38 and +0.63; the leftover mix (`court, apple, music, …`) scores −0.22. LDA's mean is pulled below zero by its three mixed topics — the worst (`world, south, buy, england, cup`, −0.46) combines world-news, sport and shopping words — whose top words rarely occur in the same short news lead, so many pairs score the minimum of $-1$. Topic-level scores show *which* topics to distrust; the mean hides that. Pass the same tokeniser (`vectorizer.build_analyzer()`) that built the topics, or words such as "sp2" or "47" will be tokenised differently and never found.

**Drill 3.** Vary the number of LDA topics over $K \in \{4, 6, 8, 10, 15\}$. For each $K$, fit on 80% of the documents and report the held-out perplexity on the other 20%, plus the NPMI of the topics. Do the two criteria agree on the best $K$?

**Solution:**

```python
from sklearn.model_selection import train_test_split

train_idx, test_idx = train_test_split(np.arange(count_matrix.shape[0]), test_size=0.2, random_state=42)
for k in [4, 6, 8, 10, 15]:
    lda_k = LatentDirichletAllocation(n_components=k, random_state=42).fit(count_matrix[train_idx])
    topics_k = [[count_vocab[j] for j in comp.argsort()[::-1][:10]] for comp in lda_k.components_]
    npmi_k = np.mean(compute_npmi(documents, topics_k, analyzer=count_vec.build_analyzer()))
    print(f"K={k:>2}: held-out perplexity = {lda_k.perplexity(count_matrix[test_idx]):,.0f}"
          f"   mean NPMI = {npmi_k:+.3f}")
```

Here held-out perplexity is lowest at $K = 4$ (about 3,500) and rises steadily to about 7,900 at $K = 15$, while NPMI peaks at $K = 6$ (+0.08). Perplexity measures how well the model predicts unseen text; NPMI measures whether each topic's top words belong together. They often disagree — the $K$ with the best perplexity is not necessarily the one whose topics a human finds clearest (Chang et al., 2009, showed that perplexity and human interpretability can even move in opposite directions). Perplexity must be computed on held-out documents: on the training documents it keeps improving with $K$. Use both numbers, then read the topics.

**Drill 4.** Using the BERTopic model from the worked example, compute the share of documents in each BERTopic topic (including the outlier topic $-1$) and compare it with the share of documents assigned to each LDA topic. Which model gives the more even split, and why?

**Solution:**

```python
bert = np.asarray(bert_topics)
bert_ids, bert_counts = np.unique(bert, return_counts=True)
print("BERTopic shares:", {int(t): f"{c / len(bert):.1%}" for t, c in zip(bert_ids, bert_counts)})

lda_assign = theta_lda.argmax(axis=1)
lda_counts = np.bincount(lda_assign, minlength=N_TOPICS)
print("LDA shares:     ", {k: f"{c / len(lda_assign):.1%}" for k, c in enumerate(lda_counts)})
```

LDA spreads documents over all $K$ topics because every document is a mixture and the argmax must pick one. BERTopic produces many topics of uneven size plus an outlier group: HDBSCAN creates a topic only where documents are densely similar and leaves the rest as $-1$, so a large outlier share is normal and is information, not failure. Which is more useful depends on the task — even coverage for routing every document somewhere, tight topics for "what are the distinct stories?".

**Drill 5.** Build a sentiment classifier using TF-IDF features and logistic regression on SST-2, a public set of English movie-review sentences labelled positive or negative by human annotators (`load_sentiment_reviews()`, as in Exercise 6.5). Report test accuracy and the most predictive words for each class.

**Solution:**

```python
from sklearn.linear_model import LogisticRegression
from shared.mlfp04.ex_6 import load_sentiment_reviews

train_df, test_df = load_sentiment_reviews()   # downloaded once from Hugging Face, then cached
sent_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X_train = sent_vec.fit_transform(train_df["text"].to_list())
X_test = sent_vec.transform(test_df["text"].to_list())

clf = LogisticRegression(max_iter=2000, C=4.0).fit(X_train, train_df["label"].to_numpy())
acc = clf.score(X_test, test_df["label"].to_numpy())
print(f"train={X_train.shape[0]:,} phrases  test={X_test.shape[0]} sentences  accuracy={acc:.3f}")

terms = sent_vec.get_feature_names_out()
order = np.argsort(clf.coef_[0])
print("Most negative:", ", ".join(terms[order[:10]]))
print("Most positive:", ", ".join(terms[order[::-1][:10]]))
```

The test set is 872 complete sentences from different reviews (sentences that also appear in the training data are removed), so accuracy is measured on unseen text. A bag of words and bigrams with a linear model lands in the low-80s percent range — a strong baseline, but word order is lost: bigrams such as "not good" help a little; phrases like "it's not that the film is bad" defeat it. The most predictive terms are the evaluative words you would expect ("bad", "dull", "worst" against "best", "beautiful", "fun"). Here it reaches 82.1%. Fine-tuned transformer models (Module 6) exceed 90% on this benchmark.

## Cross-References

- **Lesson 4.3** introduced dimensionality reduction. TF-IDF produces a very high-dimensional sparse representation; NMF and LDA reduce it to a low-dimensional dense topic space — the same idea as PCA, applied to text.
- **Lesson 4.7** introduces collaborative filtering, where the user-item matrix plays the same role as the document-term matrix. NMF on text is the same algorithm as NMF on user-item interactions.
- **Lesson 4.8** will explain how Word2Vec learns embeddings. The embeddings you used as tools in this lesson are features discovered by a shallow neural network.
- **Module 6, Lesson 6.4** uses BM25 as the sparse retrieval component of RAG systems.

## Reflection

You should now be able to:

- Derive TF-IDF from first principles and explain why IDF down-weights common words.
- Explain BM25's term-frequency saturation and document-length normalisation.
- Distinguish LDA (probabilistic, generative, fitted on counts) from NMF (matrix factorisation, usually on TF-IDF) and BERTopic (embedding- and clustering-based) and explain when each is preferred.
- Evaluate topic quality with NPMI and UMass coherence and held-out perplexity, and read coherence per topic, not only on average.
- Use word embeddings as features even though you cannot yet explain how they are trained (that comes in Lesson 4.8).

---

# Lesson 4.7: Recommender Systems and Collaborative Filtering

## Why This Matters

E-commerce marketplaces, food-delivery apps and streaming services all serve personalised recommendations. When you open a food-delivery app, the restaurants at the top are typically not a fixed list — a recommender system combines your past orders, the orders of similar users and the attributes of restaurants to predict what you are most likely to order next. The scale of the effect can be large: Netflix's own engineers reported that about 80% of the hours streamed on the service were influenced by recommendation (Gomez-Uribe and Hunt, 2015). For many platforms, the recommendation algorithm is the product.

In this lesson you will build three types of recommender systems — content-based, collaborative filtering, and matrix factorisation — and discover the concept that is the intellectual centrepiece of the entire programme: **optimisation drives feature discovery**. Matrix factorisation learns user and item embeddings by minimising reconstruction error. Those embeddings are dense vector representations that capture latent preferences. They are features, but nobody designed them. They emerged from the loss function.

This is THE PIVOT in the Feature Engineering Spectrum. Everything before this lesson summarised a fully observed data matrix — by distances, variance or co-occurrence — without being judged on predictions of values it had not seen. Everything after this lesson learns features by predicting a target (neural networks, deep learning). Matrix factorisation sits at the transition point: it uses optimisation against observed values, like supervised learning, to discover latent structure, like unsupervised learning. Understanding this bridge is understanding the rest of the programme.

## Core Concepts

### FOUNDATIONS: Three approaches to recommendation

**Content-based filtering** recommends items similar to what the user has liked before. If you watched three action movies, it recommends more action movies. It uses item features (genre, director, cast) and user preferences (ratings, watch history). Strength: does not need other users' data, and it can score a brand-new *item* as soon as the item has features (no item cold start). Weaknesses: limited to items similar to what the user has already seen — no surprise — and it still needs some history for a brand-new *user*, because without rated items there is no taste profile.

**User-based collaborative filtering** finds users similar to you and recommends what they liked. If User A and User B both rated the same five movies highly, and User B liked a sixth movie that User A has not seen, recommend that movie to User A. Strength: can recommend surprising items. Weakness: cold start — new users have no history to compare.

**Item-based collaborative filtering** finds items similar to the item the user liked. "Customers who bought this also bought that." It computes similarity between items based on who interacted with them. Strength: more stable than user-based (item similarities change slowly). Weakness: same cold-start problem for new items.

### THEORY: Matrix factorisation

Consider a user-item interaction matrix $\mathbf{R}$ of dimensions $m \times n$ (users $\times$ items), where $R_{ui}$ is user $u$'s rating of item $i$. Most entries are missing — users have interacted with only a tiny fraction of all items.

Matrix factorisation approximates $\mathbf{R}$ as the product of two low-rank matrices:

$$\mathbf{R} \approx \mathbf{U} \mathbf{V}^T$$

where $\mathbf{U}$ is $m \times k$ (user embeddings) and $\mathbf{V}$ is $n \times k$ (item embeddings), with $k \ll \min(m, n)$.

The predicted rating is:

$$\hat{R}_{ui} = \mathbf{u}_u^T \mathbf{v}_i = \sum_{f=1}^{k} U_{uf} V_{if}$$

The loss function to minimise is:

$$\mathcal{L} = \sum_{(u, i) \in \text{observed}} (R_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2 + \lambda(\|\mathbf{u}_u\|^2 + \|\mathbf{v}_i\|^2)$$

The regularisation term $\lambda(\|\mathbf{u}_u\|^2 + \|\mathbf{v}_i\|^2)$ prevents overfitting to observed ratings.

### THEORY: ALS — Alternating Least Squares

The loss function is not jointly convex in $\mathbf{U}$ and $\mathbf{V}$ (it is a product of unknowns). But if you fix $\mathbf{V}$, the problem becomes a standard regularised least squares problem in $\mathbf{U}$ — and vice versa. ALS alternates:

1. Fix $\mathbf{V}$, solve for $\mathbf{U}$: for each user $u$, $\mathbf{u}_u = (\mathbf{V}_u^T \mathbf{V}_u + \lambda \mathbf{I})^{-1} \mathbf{V}_u^T \mathbf{r}_u$
2. Fix $\mathbf{U}$, solve for $\mathbf{V}$: for each item $i$, $\mathbf{v}_i = (\mathbf{U}_i^T \mathbf{U}_i + \lambda \mathbf{I})^{-1} \mathbf{U}_i^T \mathbf{r}_i$

where $\mathbf{V}_u$ contains only the rows of $\mathbf{V}$ for items rated by user $u$, and $\mathbf{r}_u$ is the vector of user $u$'s observed ratings.

Each step has a closed-form solution (a linear solve), so ALS is simple to implement and parallelisable — every user's update is independent of every other user's. Because each half-step solves its sub-problem exactly, the regularised loss never increases, so ALS converges; like K-means and EM, it converges to a local minimum that depends on the initialisation, not necessarily the global one.

In practice the model also carries a global mean $\mu$ and bias terms: $\hat{R}_{ui} = \mu + b_u + b_i + \mathbf{u}_u^T \mathbf{v}_i$, where $b_u$ captures "this user rates generously" and $b_i$ "this item is simply good". The ALS update is the same ridge regression with a column of ones appended to the design matrix (the worked example does exactly this).

### FOUNDATIONS: Connection to PCA

Recall from Lesson 4.3 that PCA factorises $\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$ via SVD. Collaborative filtering factorises $\mathbf{R} \approx \mathbf{U} \mathbf{V}^T$. The difference: PCA observes the full matrix; collaborative filtering observes only a sparse subset. Both discover low-rank structure. Both learn embeddings (latent factors). The mechanism is the same: minimise a reconstruction error.

### FOUNDATIONS: THE PIVOT — optimisation drives feature discovery

In Lessons 4.1–4.6, unsupervised methods discovered features from the data's geometry. Several of them did minimise an objective — K-means minimises the within-cluster sum of squares, PCA the reconstruction error, NMF its factorisation error — but always over a *fully observed* matrix, as a summary of it. Clustering found groups by distance. PCA found directions by variance. LDA found topics by co-occurrence.

Matrix factorisation is the first time a model is fitted iteratively to a *partially observed* target — the ratings we have — and judged by how well it predicts the ratings we do not have, with its learned factors reused as embeddings. The user embedding $\mathbf{u}_u$ captures user $u$'s latent preferences. The item embedding $\mathbf{v}_i$ captures item $i$'s latent attributes. Nobody designed these features. They emerged from the optimisation process.

In Lesson 4.8, neural networks will generalise this. A hidden layer's activations are embeddings. They are discovered by minimising a loss function via backpropagation. The difference from matrix factorisation: neural networks can learn non-linear combinations through activation functions, not just linear ones.

The spectrum:

| Stage    | Method               | Features                  | Error signal from predicting a target? |
| -------- | -------------------- | ------------------------- | -------------------- |
| M3       | Manual               | Human-designed            | N/A                  |
| M4.1–4.6 | USML                 | Data geometry             | No (objectives summarise the observed data) |
| M4.7     | Matrix factorisation | Optimisation              | Yes (reconstruction) |
| M4.8     | Neural networks      | Backpropagation           | Yes (task loss)      |
| M5       | Deep learning        | Specialised architectures | Yes (task loss)      |

## Mathematical Foundations

### THEORY: Deriving the ALS update for $\mathbf{U}$

Fix $\mathbf{V}$. For user $u$, the loss restricted to that user's observed ratings is:

$$\mathcal{L}_u = \sum_{i \in \text{rated}_u} (R_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2 + \lambda \|\mathbf{u}_u\|^2$$

Let $\mathbf{V}_u$ be the matrix whose rows are $\mathbf{v}_i^T$ for items rated by user $u$, and $\mathbf{r}_u$ be the vector of observed ratings. Then:

$$\mathcal{L}_u = \|\mathbf{r}_u - \mathbf{V}_u \mathbf{u}_u\|^2 + \lambda \mathbf{u}_u^T \mathbf{u}_u$$

Take the derivative with respect to $\mathbf{u}_u$ and set to zero:

$$\nabla_{\mathbf{u}_u} \mathcal{L}_u = -2\mathbf{V}_u^T(\mathbf{r}_u - \mathbf{V}_u \mathbf{u}_u) + 2\lambda \mathbf{u}_u = \mathbf{0}$$

$$(\mathbf{V}_u^T \mathbf{V}_u + \lambda \mathbf{I}) \mathbf{u}_u = \mathbf{V}_u^T \mathbf{r}_u$$

$$\mathbf{u}_u = (\mathbf{V}_u^T \mathbf{V}_u + \lambda \mathbf{I})^{-1} \mathbf{V}_u^T \mathbf{r}_u$$

This is a regularised normal equation — the same structure as ridge regression from Module 2, but applied to learning embeddings instead of regression coefficients.

## The Kailash Engine: none — build it, then track it

kailash-ml has no recommender engine, and neither `AutoMLEngine` nor `ModelVisualizer` has a recommendation mode. Every method in this lesson is a few dozen lines of NumPy, which is the point: you see exactly how the embeddings arise. In production you would log each method's holdout metrics to the `ExperimentTracker` (as Exercise 7 does) and, for very large catalogues, use a dedicated open-source ALS library. Lesson 4.8 shows how the same idea — learn embeddings by minimising a loss — is written as a neural network and exported with `OnnxBridge`.

## Worked Example: An Electronics Marketplace Recommender

The ratings in this example are **synthetic**, generated by Exercise 7's `build_rating_dataset()` for a fictional Singapore electronics marketplace: 300 users, 120 products (SKUs), explicit 1–5 star ratings, about 30% of user–item pairs observed. The generator follows a known model — global mean + user bias + item bias + (user taste · item traits), with 5 hidden taste dimensions, plus noise — so we can check whether the learned embeddings recover the truth. Thirty per cent of the observed ratings are held out for evaluation, and 10 brand-new SKUs have *all* their ratings in the holdout: they are cold-start items that no collaborative method has seen. Every item also has 8 content features (a noisy view of its hidden traits and quality, like a spec sheet and price tier).

### Step 0: Data and baselines

```python
import numpy as np
from shared.mlfp04.ex_7 import build_rating_dataset, print_baselines, print_method_scores

data = build_rating_dataset()
R_obs, R_train = data["R_observed"], data["R_train"]
train_mask, holdout_mask = data["train_mask"], data["holdout_mask"]
cold_items, item_features = data["cold_items"], data["item_features"]
n_users, n_items = R_train.shape
print(f"{n_users} users x {n_items} items; {int(train_mask.sum()):,} training and "
      f"{int(holdout_mask.sum()):,} holdout ratings; {int(cold_items.sum())} cold-start SKUs")

baselines = print_baselines(R_train, train_mask, R_obs, holdout_mask)
```

Every recommender must beat three no-skill baselines: the global mean rating, each item's mean rating ("popularity"), and random scores. The scorecard reports holdout RMSE, coverage (the share of holdout pairs a method can score at all), precision@5 and mean average precision (MAP) — the last two judge the *ranking* of each user's held-out items, which is what a recommender is for. Here the global mean scores RMSE 1.03, the item mean (popularity) 0.97 with MAP 0.68, and random scores RMSE 1.60.

### Step 1: Content-based filtering

```python
def content_based(R, mask, features):
    """Score items by cosine similarity between each item's features and a user taste profile."""
    feats = features - features.mean(axis=0)
    unit = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    preds = np.full(R.shape, np.nan)
    for u in range(R.shape[0]):
        rated = np.where(mask[u])[0]
        if len(rated) < 2:
            continue                                   # a brand-new USER has no profile
        dev = R[u, rated] - R[u, rated].mean()         # liked items pull, disliked items push
        profile = dev @ unit[rated]
        if np.linalg.norm(profile) < 1e-10:
            continue
        cos = unit @ (profile / np.linalg.norm(profile))
        preds[u] = R[u, rated].mean() + 2 * R[u, rated].std() * cos
    return np.clip(preds, 1, 5)

pred_cb = content_based(R_train, train_mask, item_features)
m_cb = print_method_scores("Content-based", pred_cb, R_obs, holdout_mask)
```

The user's taste profile is the sum of the (centred) features of the items they rated, weighted by how far each rating is above or below their own average — so disliked items push the profile *away*. A new item is scored by how closely its features point in the profile's direction. Content-based filtering needs no other users and covers every item with features, including the 10 brand-new SKUs (coverage 100%); it scores RMSE 0.95 and MAP 0.73, a modest improvement on popularity. It does **not** solve the cold-start problem for a brand-new *user*: with no ratings there is no profile, and the function above returns nothing for them. Its weakness is that it can only recommend "more of the same features", and it is only as good as the features.

### Step 2: User- and item-based collaborative filtering

```python
def mean_centred(R, mask):
    user_mean = np.nanmean(np.where(mask, R, np.nan), axis=1)
    return np.where(mask, R - user_mean[:, None], 0.0), user_mean

def cosine_sim(M):
    unit = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-10)
    return unit @ unit.T

R_c, user_mean = mean_centred(R_train, train_mask)

def user_cf(k=30):
    sim = cosine_sim(R_c)
    np.fill_diagonal(sim, 0.0)
    preds = np.full(R_c.shape, np.nan)
    for u in range(n_users):
        nbrs = np.argsort(sim[u])[::-1][:k]                     # k most similar users
        w = sim[u, nbrs][:, None] * train_mask[nbrs]            # only neighbours who rated j count
        denom = np.abs(w).sum(axis=0)
        score = (w * R_c[nbrs]).sum(axis=0) / np.where(denom > 0, denom, np.nan)
        preds[u] = user_mean[u] + score
    return np.clip(preds, 1, 5)

def item_cf(k=20):
    sim = cosine_sim(R_c.T)                                     # item-item similarity
    np.fill_diagonal(sim, 0.0)
    preds = np.full(R_c.shape, np.nan)
    for j in range(n_items):
        nbrs = np.argsort(sim[j])[::-1][:k]                     # k most similar items
        w = sim[j, nbrs][None, :] * train_mask[:, nbrs]         # the user must have rated them
        denom = np.abs(w).sum(axis=1)
        score = (w * R_c[:, nbrs]).sum(axis=1) / np.where(denom > 0, denom, np.nan)
        preds[:, j] = user_mean + score
    preds[:, train_mask.sum(axis=0) == 0] = np.nan              # unseen items cannot be scored
    return np.clip(preds, 1, 5)

pred_ucf, pred_icf = user_cf(), item_cf()
m_ucf = print_method_scores("User-based CF", pred_ucf, R_obs, holdout_mask)
m_icf = print_method_scores("Item-based CF", pred_icf, R_obs, holdout_mask)
```

Both CF methods centre each user's ratings on their own mean first: a generous rater's 4 stars and a harsh rater's 3 stars can mean the same thing. User-based CF reaches RMSE 0.77 and item-based CF 0.82 — much better than the baselines on the pairs they can score — but neither can score the 10 cold-start SKUs (nobody has rated them, so they have no similarity to anything): coverage is only about 77%. Because MAP counts an unscored relevant item as a miss, their full-holdout MAP (0.61 and 0.58) is *below* the popularity baseline. Always read coverage next to accuracy.

### Step 3: Matrix factorisation with ALS

```python
def als(R, mask, k=5, lam=5.0, n_iter=30, seed=42):
    """Biased ALS: R[u, j] ~ mu + b_u + b_j + U[u] . V[j], each half-step a ridge regression."""
    rng = np.random.default_rng(seed)
    mu = float(R[mask].mean())
    U, V = rng.normal(0, 0.1, (R.shape[0], k)), rng.normal(0, 0.1, (R.shape[1], k))
    b_u, b_i = np.zeros(R.shape[0]), np.zeros(R.shape[1])
    R0, penalty, losses = np.nan_to_num(R), lam * np.eye(k + 1), []
    for _ in range(n_iter):
        for u in range(R.shape[0]):                    # fix items, solve each user
            js = np.where(mask[u])[0]
            X = np.hstack([V[js], np.ones((len(js), 1))])
            sol = np.linalg.solve(X.T @ X + penalty, X.T @ (R0[u, js] - mu - b_i[js]))
            U[u], b_u[u] = sol[:k], sol[k]
        for j in range(R.shape[1]):                    # fix users, solve each item
            us = np.where(mask[:, j])[0]
            if len(us) == 0:
                continue                               # cold item: nothing to learn from
            X = np.hstack([U[us], np.ones((len(us), 1))])
            sol = np.linalg.solve(X.T @ X + penalty, X.T @ (R0[us, j] - mu - b_u[us]))
            V[j], b_i[j] = sol[:k], sol[k]
        fit = mu + b_u[:, None] + b_i[None, :] + U @ V.T
        losses.append(float(((R0 - fit)[mask] ** 2).sum()
                            + lam * ((U**2).sum() + (V**2).sum() + (b_u**2).sum() + (b_i**2).sum())))
    preds = np.clip(mu + b_u[:, None] + b_i[None, :] + U @ V.T, 1, 5)
    preds[:, mask.sum(axis=0) == 0] = np.nan
    return preds, U, V, losses

pred_als, U, V, losses = als(R_train, train_mask)
print(f"regularised loss: {losses[0]:,.0f} -> {losses[-1]:,.0f} "
      f"(never increased: {all(b <= a + 1e-6 for a, b in zip(losses, losses[1:]))})")
m_als = print_method_scores("ALS (k=5)", pred_als, R_obs, holdout_mask)

# Do the learned item embeddings recover the hidden traits?  (R^2 of a linear map)
warm = ~cold_items
coef, *_ = np.linalg.lstsq(np.c_[V[warm], np.ones(warm.sum())], data["V_true"][warm], rcond=None)
resid = data["V_true"][warm] - np.c_[V[warm], np.ones(warm.sum())] @ coef
print(f"R^2 of true item traits explained by learned embeddings: "
      f"{1 - resid.var() / data['V_true'][warm].var():.2f}")
```

This is the biased form of the factorisation in the theory section: the global mean and the user and item biases absorb "this user rates generously" and "this product is simply good", leaving the factors to capture *taste*. Each half-step is an exact ridge-regression solve, so the regularised loss never increases. The regularised loss falls from about 4,190 to 3,360 without ever increasing. ALS reaches a holdout RMSE of 0.56 — far better than every baseline and both neighbourhood methods — with precision@5 of 0.66, but its coverage, like CF's, stops at 77% because of the cold SKUs. The last lines check the embeddings against the generator's hidden traits: a linear map from the 5 learned dimensions explains 96% of the variance of the 5 true trait dimensions. Nobody told ALS what the traits were — they emerged from minimising the reconstruction loss. That is the pivot of this lesson.

### Step 4: The cold-start split

```python
from shared.mlfp04.ex_7 import mean_average_precision

cold_pairs = holdout_mask & cold_items[None, :]
for name, preds in [("Content-based", pred_cb), ("Item-based CF", pred_icf), ("ALS", pred_als)]:
    covered = cold_pairs & ~np.isnan(preds)
    print(f"{name:<14} can score {int(covered.sum())} of {int(cold_pairs.sum())} cold-SKU ratings")
```

Only content-based filtering can say anything about the brand-new products. This is the motivation for **hybrid** recommenders (below): use collaborative signals where history exists and content where it does not.

## Further Concepts: SVD++, Implicit Feedback and Hybrids

**SVD++** (Koren, 2008) extends the biased factorisation with *implicit feedback*: the fact that a user rated (or viewed, or clicked) an item at all says something about their taste, regardless of the score. The user's vector becomes their explicit factor plus a normalised sum of factors of every item they interacted with:

$$\hat{R}_{ui} = \mu + b_u + b_i + \mathbf{v}_i^T \left( \mathbf{u}_u + |N(u)|^{-1/2} \sum_{j \in N(u)} \mathbf{y}_j \right)$$

where $N(u)$ is the set of items user $u$ interacted with and $\mathbf{y}_j$ is a second, learned embedding per item. A user with few ratings but many views is still placed sensibly. For purely implicit data (clicks, purchases, no stars), the standard method is weighted ALS on a binary preference matrix with confidence weights (Hu, Koren and Volinsky, 2008).

**Hybrid systems** combine content-based and collaborative scores. Common patterns: a *weighted* hybrid (a blend $\alpha \cdot \text{CF} + (1 - \alpha) \cdot \text{content}$, with $\alpha$ tuned on a validation split — never on the test set); a *switching* hybrid (content-based for cold items and new users with a few interactions, CF otherwise); and *feature-augmented* factorisation, where item features enter the model so that new items get an embedding from their features. Exercise 7.5 builds a validation-tuned weighted hybrid and measures its lift over each component.

## Try It Yourself

**Drill 1.** Using user-based CF, recommend the top 5 *unrated* items for three users (`sg_user_000`, `sg_user_010`, `sg_user_050`). For each, name the most similar user and how many items the two have both rated.

**Solution:**

```python
sim_users = cosine_sim(R_c)
np.fill_diagonal(sim_users, 0.0)
user_ids, item_ids = data["user_ids"], data["item_ids"]

for u in [0, 10, 50]:
    scores = np.where(train_mask[u], -np.inf, np.nan_to_num(pred_ucf[u], nan=-np.inf))
    top5 = np.argsort(scores)[::-1][:5]
    nb = int(np.argmax(sim_users[u]))
    overlap = int((train_mask[u] & train_mask[nb]).sum())
    print(f"{user_ids[u]}: recommend {[item_ids[j] for j in top5]}")
    print(f"   most similar: {user_ids[nb]} (cosine {sim_users[u, nb]:.2f}, {overlap} items in common)")
```

Notice how few items two users have in common when only 30% of pairs are observed — 4 to 6 for these pairs. Similarities computed on so few co-rated items are noisy, which is why neighbourhood methods use several neighbours and why matrix factorisation, which pools information across all users, usually does better on sparse data.

**Drill 2.** Compare user-based and item-based CF on warm items only (exclude the cold SKUs). Which ranks better (MAP), and why might item-based CF be preferred in production even when it does not win?

**Solution:**

```python
warm_holdout = holdout_mask & ~cold_items[None, :]
for name, preds in [("User-based CF", pred_ucf), ("Item-based CF", pred_icf),
                    ("ALS", pred_als), ("Content-based", pred_cb)]:
    print(f"{name:<14} warm-item MAP = {mean_average_precision(preds, R_obs, warm_holdout):.4f}")
```

On warm items ALS ranks best (MAP 0.92), then user-based CF (0.80), item-based CF (0.76) and content-based (0.74). Restricting to warm items separates ranking skill from coverage (MAP counts an unscored relevant item as a miss). In production item-based CF is often preferred regardless: item–item similarities change slowly and can be precomputed, a user's recommendations update instantly when they rate something new, and "because you bought X" is easy to explain.

**Drill 3.** Choose the number of latent factors $k$ for ALS with a validation split. Fit on `fit_mask`, measure RMSE on `val_mask` for $k \in \{1, 2, 3, 5, 10, 20\}$, pick the best $k$, then report its holdout RMSE once.

**Solution:**

```python
from shared.mlfp04.ex_7 import holdout_rmse

fit_mask, val_mask = data["fit_mask"], data["val_mask"]
R_fit = np.where(fit_mask, R_obs, np.nan)
val_scores = {}
for k in [1, 2, 3, 5, 10, 20]:
    preds_k, *_ = als(R_fit, fit_mask, k=k, lam=5.0, n_iter=20)
    train_rmse = float(np.sqrt(np.nanmean((preds_k - R_obs)[fit_mask] ** 2)))
    val_rmse, _ = holdout_rmse(preds_k, R_obs, val_mask)
    val_scores[k] = val_rmse
    print(f"k={k:>2}: train RMSE={train_rmse:.4f}  validation RMSE={val_rmse:.4f}")

best_k = min(val_scores, key=val_scores.get)
final, *_ = als(R_train, train_mask, k=best_k, lam=5.0)
print(f"best k={best_k}; holdout RMSE = {holdout_rmse(final, R_obs, holdout_mask)[0]:.4f}")
```

Training and validation RMSE both fall steeply up to $k = 5$ (validation 0.84 at $k = 1$, 0.63 at $k = 5$) and then flatten: with $\lambda = 5$, the penalty shrinks factors that have no real signal to fit towards zero, so $k = 10$ and $k = 20$ give almost the same model. With a weaker penalty the extra factors would fit noise and the validation error would rise. The validation curve, not the training curve, picks $k$; the holdout set is touched once, at the end, so its RMSE remains an honest estimate. The best $k$ is 5 — the number of true factors in the generator — and its holdout RMSE is 0.557.

**Drill 4.** Visualise the learned item embeddings in 2D with PCA, coloured by each item's mean training rating. Do items cluster by quality, by taste, or both?

**Solution:**

```python
import plotly.express as px
from sklearn.decomposition import PCA

V_warm = V[~cold_items]
V_2d = PCA(n_components=2).fit_transform(V_warm)
mean_rating = np.nanmean(np.where(train_mask, R_obs, np.nan), axis=0)[~cold_items]
fig = px.scatter(x=V_2d[:, 0], y=V_2d[:, 1], color=mean_rating,
                 labels={"x": "embedding PC1", "y": "embedding PC2", "color": "mean rating"},
                 title="ALS item embeddings")
fig.write_html("als_item_embeddings.html")
print(f"|corr(PC1, mean rating)| = {abs(np.corrcoef(V_2d[:, 0], mean_rating)[0, 1]):.2f}")
```

Because the biased model gives quality its own parameter ($b_i$), the factor embeddings mostly encode *taste* — which kind of user likes the item — and are only weakly correlated with the mean rating. Items close together in the plot are liked by the same users. Fit an unbiased model (drop the bias terms) and quality leaks into the first embedding dimension instead. Embeddings learned this way can be reused as item features in any downstream model.

**Drill 5.** Write a paragraph explaining the pivot concept: how does matrix factorisation bridge unsupervised feature discovery (Lessons 4.1–4.6) and supervised feature learning (Lesson 4.8 and Module 5)? Use the terms "embedding", "loss function", and "optimisation".

**Solution:** K-means, PCA and NMF already minimised objectives, but over a fully observed data matrix, and their outputs were mostly used as summaries of that matrix. Matrix factorisation fits user and item **embeddings** iteratively, by **optimisation**, to a *partially observed* target — the ratings we have — and is judged by how well it predicts the ratings we do not have. The **loss function** (squared error on observed ratings plus a penalty) is the only guidance; nobody specifies what the latent dimensions mean, yet they recover the hidden taste structure, and they can be reused as features elsewhere. Neural networks do the same thing with a task loss: their hidden-layer activations are embeddings discovered by backpropagation. The difference is that matrix factorisation combines its factors linearly (a dot product), while neural networks build non-linear combinations through activation functions.

## Cross-References

- **Lesson 4.3** derived PCA via SVD. With a fully observed matrix and no penalty, the best rank-$k$ factorisation *is* the truncated SVD (Eckart–Young). With most entries missing there is no closed form — SVD needs every entry — which is why collaborative filtering minimises the loss over observed entries only, iteratively.
- **Lesson 4.6** used NMF for topic modelling. NMF on a document-term matrix and NMF on a user-item matrix are the same algorithm.
- **Lesson 4.8** will generalise the idea: neural network hidden layers are embeddings learned by minimising a loss function, with the addition of non-linearity.
- **Module 5, Lesson 5.1** introduces autoencoders, which learn embeddings by reconstructing their input — the same objective as matrix factorisation, but with a neural network.

## Reflection

You should now be able to:

- Build content-based, user-based CF, and item-based CF recommenders.
- Implement ALS matrix factorisation from scratch and explain why it converges.
- Derive the ALS update rule as a regularised normal equation.
- Explain the pivot concept: optimisation drives feature discovery.
- Visualise learned embeddings and interpret what they capture.
- Articulate the connection between matrix factorisation and PCA.

---

# Lesson 4.8: Neural Networks, Backpropagation, and the DL Training Toolkit

## Why This Matters

This is the most important lesson in the MLFP programme. Not because neural networks are the most important algorithm (they are not — gradient boosting still wins most tabular competitions). But because this lesson completes the bridge from classical ML to deep learning, and everything in Modules 5 and 6 builds on it.

In Lesson 4.7 you saw that matrix factorisation learns embeddings by minimising a reconstruction loss. Those embeddings are linear combinations of the input. But real-world patterns are rarely linear. The relationship between a customer's purchase history and their next purchase involves interactions, thresholds, and non-linear effects that no linear model can capture.

A neural network with hidden layers learns non-linear combinations. Each hidden layer applies a linear transformation (multiply by weights, add bias) followed by a non-linear activation function. The activations of the hidden layers are the embeddings — features discovered by the network. The key insight: **hidden layers are automated feature engineering with error feedback**. The network writes its own features, guided by the loss function, through backpropagation.

This lesson covers the complete DL training toolkit: forward pass, backpropagation, gradient descent, activation functions, optimisers, loss functions, dropout, batch normalisation, weight initialisation, learning rate schedules, and gradient clipping. By the end, you will be able to build, train, and diagnose a neural network from scratch.

## Core Concepts

### FOUNDATIONS: From linear regression to neural networks

Linear regression predicts $\hat{y} = \mathbf{w}^T \mathbf{x} + b$. This is a single-layer network with no activation function. It can only learn linear relationships.

Add a hidden layer with an activation function:

$$\mathbf{h} = \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$$
$$\hat{y} = \mathbf{w}_2^T \mathbf{h} + b_2$$

where $\sigma$ is a non-linear activation function (like ReLU). The hidden layer $\mathbf{h}$ is a new set of features — not designed by a human, but learned from the data. Adding more hidden layers allows the network to compose non-linear transformations, learning increasingly abstract features.

The Universal Approximation Theorem states that a neural network with a single hidden layer of sufficient width can approximate any continuous function on a compact set to arbitrary precision. In practice, deeper networks (more layers) tend to learn more efficiently than very wide shallow networks.

### THEORY: The forward pass

For a network with $L$ layers:

$$\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)} \quad \text{(linear transformation)}$$
$$\mathbf{a}^{(l)} = f^{(l)}(\mathbf{z}^{(l)}) \quad \text{(activation function)}$$

where $\mathbf{a}^{(0)} = \mathbf{x}$ (the input), $\mathbf{W}^{(l)}$ is the weight matrix for layer $l$, $\mathbf{b}^{(l)}$ is the bias vector, and $f^{(l)}$ is the activation function.

The final output $\hat{y} = \mathbf{a}^{(L)}$ is compared to the true value $y$ using a loss function $\mathcal{L}(y, \hat{y})$.

### THEORY: Backpropagation — the chain rule through layers

Backpropagation computes the gradient of the loss with respect to every weight in the network, using the chain rule of calculus. For a single weight $W_{jk}^{(l)}$ in layer $l$:

$$\frac{\partial \mathcal{L}}{\partial W_{jk}^{(l)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(L)}} \cdot \frac{\partial \mathbf{a}^{(L)}}{\partial \mathbf{z}^{(L)}} \cdot \frac{\partial \mathbf{z}^{(L)}}{\partial \mathbf{a}^{(L-1)}} \cdots \frac{\partial \mathbf{z}^{(l)}}{\partial W_{jk}^{(l)}}$$

Define the error signal (delta) for layer $l$:

$$\boldsymbol{\delta}^{(l)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l)}}$$

For the output layer: $\boldsymbol{\delta}^{(L)} = \nabla_{\mathbf{a}^{(L)}} \mathcal{L} \odot f'^{(L)}(\mathbf{z}^{(L)})$

For hidden layers (propagating backward): $\boldsymbol{\delta}^{(l)} = (\mathbf{W}^{(l+1)T} \boldsymbol{\delta}^{(l+1)}) \odot f'^{(l)}(\mathbf{z}^{(l)})$

The gradient with respect to the weights: $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T$

The gradient with respect to the biases: $\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}$

### THEORY: Gradient descent

Update each weight in the direction that decreases the loss:

$$\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}}$$

where $\eta$ is the learning rate. Too large: overshoots and diverges. Too small: converges too slowly. The learning rate is the single most important hyperparameter in neural network training.

### FOUNDATIONS: Activation functions

| Function   | Formula                         | Use                        | Why                                        |
| ---------- | ------------------------------- | -------------------------- | ------------------------------------------ |
| ReLU       | $\max(0, z)$                    | Default hidden layer       | Simple, fast, mitigates vanishing gradient |
| Leaky ReLU | $\max(0.01z, z)$                | Hidden layer               | Avoids dead neurons                        |
| GELU       | $z \cdot \Phi(z)$               | Transformer hidden layers  | Smooth, used in BERT/GPT                   |
| Sigmoid    | $1/(1 + e^{-z})$                | Binary output              | Maps to $[0,1]$ probability                |
| Tanh       | $(e^z - e^{-z})/(e^z + e^{-z})$ | Hidden layer (less common) | Zero-centred, maps to $[-1,1]$             |
| Softmax    | $e^{z_i}/\sum_j e^{z_j}$        | Multi-class output         | Maps to probability distribution           |

ReLU is the default choice for hidden layers. Sigmoid and softmax are for output layers. GELU is the default in modern transformer architectures.

### THEORY: Loss functions taxonomy

| Loss          | Formula                                  | Use                         |
| ------------- | ---------------------------------------- | --------------------------- |
| MSE           | $\frac{1}{n}\sum(y - \hat{y})^2$         | Regression                  |
| MAE           | $\frac{1}{n}\sum\|y - \hat{y}\|$         | Robust regression           |
| Cross-entropy | $-\sum y_c \log \hat{y}_c$               | Multi-class classification  |
| Binary CE     | $-[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ | Binary classification       |
| Focal loss    | $-\alpha_t(1-p_t)^\gamma \log(p_t)$      | Imbalanced classification   |
| KL divergence | $\sum p \log(p/q)$                       | Distribution matching (VAE) |

### FOUNDATIONS: Dropout

Dropout randomly sets a fraction $p$ of the hidden layer activations to zero during training. This forces the network to learn redundant representations — no single neuron can be relied upon, so the knowledge must be distributed. During inference, dropout is turned off and activations are scaled by $(1-p)$ to compensate.

Dropout rate is typically 0.1–0.5. Higher rates provide stronger regularisation but slow convergence. It is the neural network equivalent of bagging — each training step uses a different random subset of neurons, effectively training an ensemble of networks.

### THEORY: Batch normalisation

Batch normalisation normalises the inputs to each layer to have zero mean and unit variance within each mini-batch:

$$\hat{z}_i = \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$$y_i = \gamma \hat{z}_i + \beta$$

where $\mu_B$ and $\sigma_B^2$ are the mini-batch mean and variance, $\gamma$ and $\beta$ are learned scale and shift parameters, and $\epsilon$ is a small constant for numerical stability.

Benefits: stabilises training (layers receive inputs with consistent statistics), enables higher learning rates, acts as a mild regulariser.

### FOUNDATIONS: Weight initialisation

If all weights are initialised to zero, all neurons compute the same output, all gradients are the same, and the network never breaks symmetry — it cannot learn. Random initialisation breaks symmetry, but the scale matters:

- **Xavier/Glorot** (for sigmoid/tanh): $W \sim \mathcal{N}(0, 2/(n_{\text{in}} + n_{\text{out}}))$
- **Kaiming/He** (for ReLU): $W \sim \mathcal{N}(0, 2/n_{\text{in}})$

Kaiming initialisation accounts for the fact that ReLU zeros out half the activations, so the variance must be doubled to compensate.

### THEORY: Optimisers

**SGD with momentum** maintains a running average of gradients:
$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \nabla \mathcal{L}$$
$$\mathbf{W} \leftarrow \mathbf{W} - \eta \mathbf{v}_t$$

**Adam** (Adaptive Moment Estimation) adapts the learning rate for each parameter:
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1-\beta_1^t), \quad \hat{v}_t = v_t / (1-\beta_2^t)$$
$$W \leftarrow W - \eta \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

Adam is the default optimiser for most deep learning tasks. AdamW adds decoupled weight decay, which is preferred for transformer training.

### FOUNDATIONS: Learning rate schedules

A fixed learning rate is rarely optimal. Common schedules:

- **Step decay:** reduce by a factor every $N$ epochs.
- **Cosine annealing:** $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t / T))$
- **Warmup + cosine:** start with a low learning rate, linearly increase to the peak, then follow cosine decay. Used in transformer training.
- **ReduceLROnPlateau:** reduce when validation loss stops improving.

### FOUNDATIONS: Gradient clipping and early stopping

**Gradient clipping** prevents exploding gradients by capping the gradient norm:

$$\text{if } \|\nabla \mathcal{L}\| > \text{max\_norm}: \quad \nabla \mathcal{L} \leftarrow \text{max\_norm} \cdot \frac{\nabla \mathcal{L}}{\|\nabla \mathcal{L}\|}$$

Essential for RNNs (Module 5) and transformers where gradients can grow explosively.

**Early stopping** monitors validation loss and stops training when it begins to increase (patience of $N$ epochs). This prevents overfitting — the model's performance on unseen data degrades even as training loss continues to decrease.

## Mathematical Foundations

### THEORY: Backpropagation derivation for a 2-layer network

Consider a network with one hidden layer:

$$\mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}$$
$$\mathbf{a}^{(1)} = \text{ReLU}(\mathbf{z}^{(1)})$$
$$\hat{y} = \mathbf{w}^{(2)T} \mathbf{a}^{(1)} + b^{(2)}$$
$$\mathcal{L} = \frac{1}{2}(y - \hat{y})^2$$

**Output layer gradients:**

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = -(y - \hat{y})$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}^{(2)}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \mathbf{a}^{(1)} = -(y - \hat{y}) \mathbf{a}^{(1)}$$

**Hidden layer gradients (chain rule):**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(1)}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \mathbf{w}^{(2)} = -(y - \hat{y}) \mathbf{w}^{(2)}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(1)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(1)}} \odot \text{ReLU}'(\mathbf{z}^{(1)})$$

where $\text{ReLU}'(z) = \mathbf{1}[z > 0]$ (1 if positive, 0 otherwise).

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(1)}} \mathbf{x}^T$$

This is backpropagation: compute the error at the output, propagate it backward through each layer using the chain rule, and use the result to compute the gradient for each weight.

### ADVANCED: Hidden layers as automated feature engineering

Consider a 2-hidden-layer network for HDB price prediction. The input is $\mathbf{x} = [\text{floor\_area}, \text{storey}, \text{lease\_remaining}, \text{town\_encoded}]$.

The first hidden layer might learn features like:

- $h_1$: "overall quality" (positive loading on area, storey, and lease)
- $h_2$: "location premium" (depends heavily on town encoding)
- $h_3$: "new vs old" (positive on lease remaining, negative on storey)

The second hidden layer combines these into more abstract features:

- $h'_1$: "premium mature estate flat" (combines location premium with overall quality)
- $h'_2$: "value new-build" (combines new-vs-old with moderate quality)

These features were not designed by anyone. They emerged from minimising the price prediction error via backpropagation. This is representation learning — the network discovers its own representations.

The connection to Module 4's journey: in Lesson 4.3, PCA found linear combinations that maximise variance. In Lesson 4.7, matrix factorisation found linear combinations that minimise reconstruction error. Here, neural networks find non-linear combinations that minimise task-specific loss. Each step adds more power.

## The Kailash Engine: OnnxBridge (model export)

```python
from kailash_ml import OnnxBridge

bridge = OnnxBridge()
bridge.export(model, input_shape=(1, 4), output_path="hdb_predictor.onnx")
# Load for inference
loaded = bridge.load("hdb_predictor.onnx")
prediction = loaded.predict(sample_input)
```

## Worked Example: Neural Network for HDB Price Prediction — from Scratch

We build a 3-layer network from scratch using NumPy, then add each training technique one by one to see its effect.

```python
import numpy as np

# Load HDB data
loader = MLFPDataLoader()
df = loader.load("mlfp04", "sg_hdb_prices.csv")
features = ["floor_area_sqm", "storey_range_mid", "remaining_lease_years", "town_encoded"]
X = df.select(features).to_numpy().astype(np.float64)
y = df["resale_price"].to_numpy().astype(np.float64).reshape(-1, 1)

# Standardise
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
y_mean, y_std = y.mean(), y.std()
X_norm = (X - X_mean) / X_std
y_norm = (y - y_mean) / y_std

# Split
n_train = int(0.8 * len(X_norm))
X_train, X_test = X_norm[:n_train], X_norm[n_train:]
y_train, y_test = y_norm[:n_train], y_norm[n_train:]

# Network architecture: 4 -> 64 -> 32 -> 1
np.random.seed(42)
W1 = np.random.randn(4, 64) * np.sqrt(2.0 / 4)   # Kaiming init
b1 = np.zeros((1, 64))
W2 = np.random.randn(64, 32) * np.sqrt(2.0 / 64)
b2 = np.zeros((1, 32))
W3 = np.random.randn(32, 1) * np.sqrt(2.0 / 32)
b3 = np.zeros((1, 1))

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

lr = 0.001
batch_size = 64
epochs = 100

for epoch in range(epochs):
    # Shuffle
    perm = np.random.permutation(n_train)
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0
    for start in range(0, n_train, batch_size):
        end = min(start + batch_size, n_train)
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]
        m = len(X_batch)

        # Forward pass
        z1 = X_batch @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        a2 = relu(z2)
        z3 = a2 @ W3 + b3
        y_hat = z3  # linear output for regression

        # Loss (MSE)
        loss = np.mean((y_batch - y_hat) ** 2)
        epoch_loss += loss * m

        # Backward pass
        dz3 = -2 * (y_batch - y_hat) / m
        dW3 = a2.T @ dz3
        db3 = dz3.sum(axis=0, keepdims=True)

        da2 = dz3 @ W3.T
        dz2 = da2 * relu_grad(z2)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)

        da1 = dz2 @ W2.T
        dz1 = da1 * relu_grad(z1)
        dW1 = X_batch.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        # Update weights
        W3 -= lr * dW3
        b3 -= lr * db3
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    epoch_loss /= n_train
    if epoch % 20 == 0:
        # Test loss
        z1_t = X_test @ W1 + b1; a1_t = relu(z1_t)
        z2_t = a1_t @ W2 + b2; a2_t = relu(z2_t)
        y_hat_t = a2_t @ W3 + b3
        test_loss = np.mean((y_test - y_hat_t) ** 2)
        print(f"Epoch {epoch}: train_loss={epoch_loss:.4f}, test_loss={test_loss:.4f}")
```

The training loss should decrease steadily. If the test loss begins to increase while training loss continues to decrease, you are overfitting — and that is where dropout, batch norm, and early stopping come in.

## Try It Yourself

**Drill 1.** Add dropout to the hidden layers (p=0.2). Implement it from scratch: during training, generate a binary mask from Bernoulli(1-p) and element-wise multiply the activations. Scale by 1/(1-p). During evaluation, do not apply dropout. Compare training curves with and without dropout.

**Solution:**

```python
def dropout(a, p=0.2, training=True):
    if not training:
        return a
    mask = (np.random.rand(*a.shape) > p).astype(float)
    return a * mask / (1 - p)

# In forward pass during training:
a1 = dropout(relu(z1), p=0.2, training=True)
a2 = dropout(relu(z2), p=0.2, training=True)
```

**Drill 2.** Implement batch normalisation from scratch for the first hidden layer. During training, normalise using the mini-batch statistics. Maintain running mean and variance for inference. Compare training convergence with and without batch norm.

**Solution:**

```python
gamma1 = np.ones((1, 64))
beta1 = np.zeros((1, 64))
running_mean = np.zeros((1, 64))
running_var = np.ones((1, 64))
momentum = 0.1

def batch_norm(z, gamma, beta, running_mean, running_var, training=True):
    if training:
        mu = z.mean(axis=0, keepdims=True)
        var = z.var(axis=0, keepdims=True)
        running_mean[:] = (1 - momentum) * running_mean + momentum * mu
        running_var[:] = (1 - momentum) * running_var + momentum * var
    else:
        mu = running_mean
        var = running_var
    z_hat = (z - mu) / np.sqrt(var + 1e-8)
    return gamma * z_hat + beta
```

**Drill 3.** Replace SGD with Adam. Implement Adam from scratch (maintain first and second moment estimates, apply bias correction). Compare convergence speed: how many epochs does SGD need versus Adam to reach the same test loss?

**Solution:**

```python
# Adam state for each parameter
m_W1 = np.zeros_like(W1); v_W1 = np.zeros_like(W1)
t = 0
beta1_adam, beta2_adam, eps_adam = 0.9, 0.999, 1e-8

def adam_update(param, grad, m, v, t, lr=0.001):
    m = beta1_adam * m + (1 - beta1_adam) * grad
    v = beta2_adam * v + (1 - beta2_adam) * grad**2
    m_hat = m / (1 - beta1_adam**t)
    v_hat = v / (1 - beta2_adam**t)
    param -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)
    return param, m, v
```

**Drill 4.** Implement cosine annealing for the learning rate. Start at $\eta = 0.001$, anneal to $\eta = 0.0001$ over 100 epochs. Plot the learning rate schedule and compare training curves with fixed vs cosine-annealed learning rate.

**Solution:**

```python
eta_max, eta_min = 0.001, 0.0001
T = 100
for epoch in range(T):
    lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * epoch / T))
    # Use lr for this epoch's updates
```

**Drill 5.** Implement gradient clipping with max_norm = 1.0. Compute the total gradient norm across all parameters. If it exceeds max_norm, scale all gradients down proportionally. When does gradient clipping activate during training? In which epochs?

**Solution:**

```python
def clip_gradients(grads, max_norm=1.0):
    total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        grads = [g * scale for g in grads]
    return grads, total_norm

# After computing all gradients:
[dW1, dW2, dW3, db1, db2, db3], norm = clip_gradients(
    [dW1, dW2, dW3, db1, db2, db3], max_norm=1.0
)
if norm > 1.0:
    print(f"Epoch {epoch}: gradient clipped (norm={norm:.2f})")
```

**Drill 6.** Extract the activations of the first hidden layer for all test data points. Apply PCA to reduce these 64-dimensional activations to 2D. Colour the scatter plot by the true resale price. Do the learned representations show meaningful structure (e.g., expensive flats clustered together)?

**Solution:**

```python
z1_test = X_test @ W1 + b1
a1_test = relu(z1_test)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(a1_test)
# Plot with colour = y_test (resale price)
```

## Cross-References

- **Module 2** introduced gradient descent for linear regression. Neural network training uses the same principle, extended to multiple layers via the chain rule.
- **Lesson 4.3** derived PCA as linear feature extraction. Hidden layer activations are non-linear feature extraction — a generalisation.
- **Lesson 4.7** introduced optimisation-driven feature discovery via matrix factorisation. Neural networks extend this with non-linearity and depth.
- **Module 5** builds on every concept in this lesson: autoencoders (5.1), CNNs (5.2), RNNs (5.3), transformers (5.4), GANs (5.5), GNNs (5.6), transfer learning (5.7), and RL (5.8).

## Reflection

You should now be able to:

- Build a neural network from scratch: forward pass, loss, backpropagation, weight update.
- Derive the backpropagation equations for a 2-layer network using the chain rule.
- Explain why hidden layers are automated feature engineering with error feedback.
- Select the appropriate activation function, optimiser, and loss function for a given task.
- Implement dropout and batch normalisation from scratch.
- Apply learning rate scheduling and gradient clipping.
- Articulate the complete Feature Engineering Spectrum from manual features (M3) to learned features (M4.8).

This lesson completes the Module 4 arc. You entered this chapter knowing how to design features by hand. You now understand that machines can design features for themselves, and the mechanism is optimisation. In Module 5 you will see specialised neural architectures that exploit the structure of images (CNNs), sequences (RNNs), graphs (GNNs), and attention (transformers). Each one is a variation on the same theme: learn features from data, guided by a loss function, via backpropagation. The vocabulary you built in this lesson — forward pass, chain rule, gradient descent, dropout, batch norm, Adam — will be on your fingertips for the next 16 lessons.

---

# Chapter Summary

Module 4 took you from the last row of labelled data in Module 3 into the territory of unlabelled data and beyond. The arc has a clear shape.

**The first half (Lessons 4.1–4.6)** was unsupervised machine learning: discovering structure in data without labels. Clustering found groups. PCA found directions. Topic modelling found themes. Anomaly detection found outliers. Association rules found co-occurrences. In every case, the features were discovered from the data's own geometry — no error signal, no loss function, no gradient.

**The pivot (Lesson 4.7)** introduced optimisation-driven feature discovery. Matrix factorisation learns embeddings by minimising reconstruction error. The embeddings are features, but nobody designed them — they emerged from the loss function. This is the bridge between unsupervised and supervised feature learning.

**The second half (Lesson 4.8)** generalised the pivot to neural networks. Hidden layers are automated feature engineering with error feedback. The activations are embeddings, learned by backpropagation. Non-linear activation functions allow the network to learn feature combinations that no linear method can capture.

## The Feature Engineering Spectrum — completed

| Stage        | Module   | Method                 | Features                | Error signal         |
| ------------ | -------- | ---------------------- | ----------------------- | -------------------- |
| Manual       | M3       | Domain expertise       | Human-designed          | N/A                  |
| Geometric    | M4.1–4.3 | Clustering, PCA        | Data structure          | No                   |
| Statistical  | M4.4–4.6 | Anomaly, topics, rules | Co-occurrence           | No                   |
| Optimisation | M4.7     | Matrix factorisation   | Embeddings (linear)     | Yes (reconstruction) |
| Learned      | M4.8     | Neural networks        | Embeddings (non-linear) | Yes (task loss)      |
| Specialised  | M5       | CNN, RNN, Transformer  | Architecture-specific   | Yes (task loss)      |
| Semantic     | M6       | LLMs                   | Language features       | Yes (pre-training)   |

This spectrum is the intellectual backbone of the MLFP programme. Every module from here forward is a variation on "learn features from data, guided by a loss function".

## What Module 5 builds on

Module 5 assumes you can:

- Implement a neural network from scratch (forward pass, backprop, gradient descent).
- Use dropout, batch normalisation, weight initialisation, and Adam.
- Read training curves and diagnose overfitting, underfitting, and gradient pathologies.
- Export models with OnnxBridge.
- Explain representation learning: hidden layers discover features.

Module 5 introduces specialised architectures: autoencoders for reconstruction, CNNs for spatial data, RNNs for sequential data, transformers for attention-based processing, GANs for generation, GNNs for graph data, transfer learning for reuse, and reinforcement learning for interaction. Each architecture imposes a structural bias that makes learning efficient for a specific data type. The DL training toolkit from Lesson 4.8 — activation functions, optimisers, loss functions, regularisation — applies to all of them.

---

# Glossary

**Activation function.** A non-linear function applied element-wise to a layer's output. Introduces non-linearity into the network. Common choices: ReLU, sigmoid, tanh, GELU, softmax.

**Adam.** Adaptive Moment Estimation. An optimiser that maintains per-parameter learning rates using first and second moment estimates of the gradient.

**Agglomerative clustering.** A hierarchical clustering method that starts with each point as its own cluster and iteratively merges the two closest clusters.

**ALS (Alternating Least Squares).** An optimisation algorithm for matrix factorisation that alternates between fixing user embeddings and solving for item embeddings, and vice versa.

**Anomaly.** A data point that differs significantly from the majority. Also called an outlier.

**Apriori algorithm.** A frequent itemset mining algorithm that generates candidates bottom-up and prunes using the Apriori principle: infrequent itemsets cannot have frequent supersets.

**ARI (Adjusted Rand Index).** An external cluster evaluation metric that measures agreement between predicted and true labels, adjusted for chance.

**Association rule.** A rule of the form $X \to Y$ discovered from transaction data, scored by support, confidence, and lift.

**Autoencoder.** A neural network that learns to reconstruct its input through a bottleneck, thereby learning compressed representations.

**Backpropagation.** The algorithm for computing gradients of the loss with respect to all weights in a neural network, using the chain rule of calculus.

**Batch normalisation.** A technique that normalises layer inputs to zero mean and unit variance within each mini-batch, stabilising training.

**BERTopic.** A modern topic modelling approach combining sentence embeddings, UMAP, HDBSCAN, and class-based TF-IDF.

**BM25.** An information retrieval scoring function that improves on TF-IDF with term frequency saturation and document length normalisation.

**Centroid.** The mean of all points in a cluster. Used in K-means as the cluster representative.

**Cluster.** A group of data points that are more similar to each other than to points in other groups.

**Collaborative filtering.** A recommendation technique that predicts a user's preferences based on the preferences of similar users or items.

**Confidence (association rules).** The conditional probability $P(Y \mid X)$ — how often $Y$ appears in transactions that contain $X$.

**Content-based filtering.** A recommendation technique that uses item features to recommend items similar to those the user has previously liked.

**Cosine annealing.** A learning rate schedule that follows a cosine curve from maximum to minimum over the training period.

**Cosine similarity.** A measure of similarity between two vectors based on the cosine of the angle between them: $\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$.

**Covariance matrix.** A symmetric matrix whose $(i,j)$ entry is the covariance between features $i$ and $j$. The eigenvalues and eigenvectors of the covariance matrix are the foundation of PCA.

**Cross-entropy loss.** A loss function for classification that measures the divergence between predicted class probabilities and true labels.

**Curse of dimensionality.** The phenomenon where high-dimensional spaces become increasingly sparse, making distance-based methods ineffective.

**Davies-Bouldin Index.** An internal cluster evaluation metric. Lower values indicate better-defined clusters.

**DBSCAN.** Density-Based Spatial Clustering of Applications with Noise. A clustering algorithm that defines clusters as dense regions separated by sparse regions.

**Dendrogram.** A tree diagram showing the hierarchy of cluster merges in hierarchical clustering.

**Dimensionality reduction.** Reducing the number of features while preserving important structure. PCA, t-SNE, and UMAP are dimensionality reduction methods.

**Dropout.** A regularisation technique that randomly zeros out a fraction of neurons during training, forcing the network to learn distributed representations.

**Early stopping.** Halting training when validation loss stops improving, to prevent overfitting.

**Eigenvalue.** A scalar $\lambda$ such that $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$ for some non-zero vector $\mathbf{v}$. In PCA, eigenvalues represent the variance along each principal component.

**Eigenvector.** A non-zero vector $\mathbf{v}$ such that $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$. In PCA, eigenvectors are the principal component directions.

**Elbow method.** A heuristic for choosing $K$ in K-means by plotting WCSS versus $K$ and identifying the "elbow" where the rate of decrease slows.

**EM algorithm.** Expectation-Maximisation. An iterative algorithm for fitting latent-variable models. Alternates between computing posterior probabilities of latent variables (E-step) and updating model parameters (M-step).

**Embedding.** A dense vector representation of a high-dimensional or discrete object (word, user, item) in a continuous low-dimensional space, learned through optimisation.

**EnsembleEngine.** Kailash ML engine for combining multiple models via blending, stacking, bagging, or boosting.

**Feature Engineering Spectrum.** The organising framework of the MLFP curriculum: from manual features (M3) through unsupervised discovery (M4.1–4.6) to optimisation-driven learning (M4.7) to neural representation learning (M4.8+).

**Forward pass.** Computing the output of a neural network by passing input through each layer sequentially.

**FP-Growth.** A frequent itemset mining algorithm that uses a compressed FP-tree to extract patterns without candidate generation.

**Gap statistic.** A method for choosing the number of clusters by comparing within-cluster dispersion to a null reference distribution.

**Gaussian Mixture Model (GMM).** A probabilistic model that represents data as a mixture of Gaussian distributions, fitted using the EM algorithm.

**GELU.** Gaussian Error Linear Unit. An activation function used in transformer architectures: $\text{GELU}(z) = z \cdot \Phi(z)$.

**Gradient clipping.** Limiting the magnitude of gradients during training to prevent exploding gradient problems.

**Gradient descent.** An optimisation algorithm that iteratively adjusts parameters in the direction that decreases the loss function.

**HDBSCAN.** Hierarchical DBSCAN. A density-based clustering algorithm that automatically selects the density threshold per cluster.

**Hidden layer.** A layer in a neural network between the input and output layers. Its activations are learned features.

**IDF (Inverse Document Frequency).** A measure of how rare a word is across a corpus: $\log(N / \text{df}(t))$.

**IQR (Interquartile Range).** The difference between the 75th and 25th percentiles. Used for outlier detection: values outside $Q_1 - 1.5 \times \text{IQR}$ to $Q_3 + 1.5 \times \text{IQR}$ are flagged.

**Isolation Forest.** An anomaly detection algorithm that isolates anomalies using random trees. Short path length indicates anomaly.

**K-means.** A clustering algorithm that partitions data into $K$ clusters by iteratively assigning points to the nearest centroid and recomputing centroids.

**K-means++.** An initialisation strategy for K-means that spreads initial centroids apart, improving convergence.

**Kaiming initialisation.** Weight initialisation designed for ReLU networks: $W \sim \mathcal{N}(0, 2/n_{\text{in}})$.

**LDA (Latent Dirichlet Allocation).** A generative probabilistic model for topic discovery that treats documents as mixtures of topics and topics as distributions over words.

**Learning rate.** The step size in gradient descent. Controls how much weights are updated per iteration.

**Lift (association rules).** The ratio of observed co-occurrence to expected co-occurrence under independence. Lift > 1 indicates positive association.

**Linkage.** The criterion for measuring distance between clusters in hierarchical clustering: single, complete, average, or Ward's.

**Loading (PCA).** The weight of an original feature in a principal component. Used for interpreting what each component represents.

**Local Outlier Factor (LOF).** An anomaly detection method that compares a point's local density to the local densities of its neighbours.

**Loss function.** A function that measures the discrepancy between model predictions and true values. Training minimises the loss.

**Matrix factorisation.** Decomposing a matrix into the product of two lower-rank matrices. Used in recommender systems and topic modelling.

**Mixture of Experts (MoE).** A model architecture where a gating network routes inputs to specialised sub-networks (experts).

**NMF (Non-negative Matrix Factorisation).** A matrix factorisation method that constrains both factor matrices to have non-negative entries. Used for topic modelling and recommender systems.

**NMI (Normalised Mutual Information).** An external cluster evaluation metric based on information theory.

**NPMI (Normalised Pointwise Mutual Information).** A coherence metric for topic models that measures word co-occurrence normalised by individual frequencies.

**PCA (Principal Component Analysis).** A dimensionality reduction method that finds the directions of maximum variance in the data via eigendecomposition of the covariance matrix.

**Perplexity (t-SNE).** A parameter controlling the effective number of neighbours in t-SNE. Typically 5–50.

**Reconstruction error.** The difference between the original data and its approximation from a reduced representation.

**ReLU.** Rectified Linear Unit. $f(z) = \max(0, z)$. The default activation function for hidden layers.

**Representation learning.** Learning data representations (features) that make downstream tasks easier. Hidden layer activations are learned representations.

**Responsibility (EM).** The posterior probability that a data point was generated by a particular component, computed in the E-step.

**Score blending.** Combining normalised scores from multiple detectors using a weighted average to improve anomaly detection robustness.

**Scree plot.** A plot of eigenvalues (or variance explained) versus component index, used to choose the number of PCA components.

**Silhouette score.** An internal cluster evaluation metric that measures how similar a point is to its own cluster versus other clusters. Range: $[-1, +1]$.

**Singular Value Decomposition (SVD).** The factorisation $\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$, where $\mathbf{U}$ and $\mathbf{V}$ are orthogonal and $\boldsymbol{\Sigma}$ is diagonal.

**Soft clustering.** Assigning each point a probability of belonging to each cluster, rather than a hard binary assignment. GMMs produce soft clusters.

**Spectral clustering.** A clustering method that uses eigenvalues of the graph Laplacian to partition data. Effective for non-convex clusters.

**Support (association rules).** The fraction of transactions containing an itemset: $\text{supp}(X) = |\{t : X \subseteq t\}| / |T|$.

**t-SNE.** t-distributed Stochastic Neighbour Embedding. A non-linear dimensionality reduction method for visualisation that preserves local structure.

**TF-IDF.** Term Frequency-Inverse Document Frequency. A text representation that weights words by their importance to a document relative to the corpus.

**Topic model.** A model that discovers latent themes (topics) in a collection of documents. LDA and BERTopic are topic models.

**UMAP.** Uniform Manifold Approximation and Projection. A non-linear dimensionality reduction method that preserves both local and global structure.

**Universal Approximation Theorem.** The theorem that a sufficiently wide single-hidden-layer neural network can approximate any continuous function on a compact set.

**Variance explained.** The proportion of total variance captured by a subset of principal components.

**Ward's linkage.** A hierarchical clustering linkage method that minimises the increase in total within-cluster variance at each merge.

**WCSS (Within-Cluster Sum of Squares).** The K-means objective function: the sum of squared distances from each point to its cluster centroid.

**Weight initialisation.** The method for setting initial values of neural network weights. Xavier/Glorot for sigmoid/tanh, Kaiming/He for ReLU.

**Xavier initialisation.** Weight initialisation designed for sigmoid and tanh networks: $W \sim \mathcal{N}(0, 2/(n_{\text{in}} + n_{\text{out}}))$.

**Z-score.** The number of standard deviations a value is from the mean: $z = (x - \bar{x}) / s$. Used for outlier detection with a threshold of $|z| > 3$.

---

# Further Reading

**On unsupervised learning**

- Hastie, T., Tibshirani, R., and Friedman, J. _The Elements of Statistical Learning._ Springer, 2009. Chapters 13 (prototypes and nearest-neighbours), 14 (unsupervised learning), and 8 (model inference and averaging) are directly relevant. Free online at `web.stanford.edu/~hastie/ElemStatLearn/`.

- Bishop, C. _Pattern Recognition and Machine Learning._ Springer, 2006. Chapter 9 (Mixture Models and EM) is the standard reference for the EM algorithm. Chapter 12 (Continuous Latent Variables) covers PCA and factor analysis.

**On clustering**

- Ester, M., et al. "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise." _KDD_, 1996. The original DBSCAN paper.

- McInnes, L., Healy, J., and Astels, S. "hdbscan: Hierarchical density based clustering." _JOSS_, 2017. The HDBSCAN reference.

- Arthur, D., and Vassilvitskii, S. "k-means++: The Advantages of Careful Seeding." _SODA_, 2007. The K-means++ initialisation paper with its $O(\log K)$ competitive guarantee.

**On dimensionality reduction**

- Jolliffe, I. _Principal Component Analysis._ Springer, 2002. The definitive PCA reference.

- van der Maaten, L., and Hinton, G. "Visualizing Data using t-SNE." _JMLR_, 2008. The original t-SNE paper.

- McInnes, L., Healy, J., and Melville, J. "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." _arXiv:1802.03426_, 2018.

**On anomaly detection**

- Liu, F., Ting, K., and Zhou, Z.-H. "Isolation Forest." _ICDM_, 2008. The original Isolation Forest paper.

- Breunig, M., et al. "LOF: Identifying Density-Based Local Outliers." _SIGMOD_, 2000. The original LOF paper.

**On topic modelling and NLP**

- Blei, D., Ng, A., and Jordan, M. "Latent Dirichlet Allocation." _JMLR_, 2003. The original LDA paper.

- Grootendorst, M. "BERTopic: Neural topic modeling with a class-based TF-IDF procedure." _arXiv:2203.05794_, 2022.

- Robertson, S., and Zaragoza, H. "The Probabilistic Relevance Framework: BM25 and Beyond." _Foundations and Trends in Information Retrieval_, 2009.

**On recommender systems**

- Koren, Y., Bell, R., and Volinsky, C. "Matrix Factorization Techniques for Recommender Systems." _Computer_, 2009. The Netflix Prize paper — the definitive introduction to collaborative filtering with matrix factorisation.

- Hu, Y., Koren, Y., and Volinsky, C. "Collaborative Filtering for Implicit Feedback Datasets." _ICDM_, 2008.

**On neural networks and deep learning foundations**

- Goodfellow, I., Bengio, Y., and Courville, A. _Deep Learning._ MIT Press, 2016. Chapters 6 (Deep Feedforward Networks), 7 (Regularization), and 8 (Optimization) are the standard reference for the material in Lesson 4.8. Free online at `deeplearningbook.org`.

- He, K., et al. "Delving Deep into Rectifiers." _ICCV_, 2015. The Kaiming initialisation paper.

- Kingma, D., and Ba, J. "Adam: A Method for Stochastic Optimization." _ICLR_, 2015.

- Ioffe, S., and Szegedy, C. "Batch Normalization: Accelerating Deep Network Training." _ICML_, 2015.

**On Singapore-specific data**

- `data.gov.sg` — Singapore government open data portal. HDB resale transactions, retail statistics, and economic indicators.

- Singapore Department of Statistics. Monthly and quarterly reports on retail sales, consumer prices, and economic activity.

---
