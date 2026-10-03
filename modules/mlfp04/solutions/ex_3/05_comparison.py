# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 3.5: Method comparison + intrinsic dimensionality
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Compare PCA, Kernel PCA, t-SNE, UMAP, Isomap on one ruler
#     (neighbourhood preservation: trustworthiness + kNN overlap)
#   - Estimate the intrinsic dimensionality of your data
#     (variance thresholds, Kaiser, broken-stick, NN MLE)
#   - Pick the right reducer given production vs visualisation goals
#
# PREREQUISITES: 01-04_*.py (all four previous technique files).
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — why intrinsic dim is the right number to target
#   2. Build — five reducers on the same data
#   3. Train — score every configuration on the same rows
#   4. Visualise — leaderboard + intrinsic-dimensionality summary
#   5. Apply — respondent segmentation for a public-sector forms platform
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap
from sklearn.neighbors import NearestNeighbors

from kailash_ml import ModelVisualizer

from shared.mlfp04.ex_3 import (
    OUTPUT_DIR,
    evaluate_embedding,
    holdout_indices,
    load_customer_matrix,
    setup_engines,
    subsample_indices,
    teardown_engines,
    track_run,
)

# ── Kailash-ML ExperimentTracker — every dim-reduction run logs here ─────
tracker, exp_name = setup_engines()

try:
    import umap as umap_lib  # type: ignore

    UMAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    umap_lib = None
    UMAP_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════════
# THEORY — what "intrinsic dimensionality" means
# ════════════════════════════════════════════════════════════════════════
# Your data may live in p ambient dimensions but actually only vary along
# d << p independent axes. d is the INTRINSIC dimensionality. Classic
# example: 1000 photos of a rotating face are ambient-dim 1000x1000x3,
# but intrinsic-dim 1 (just the rotation angle).
#
# Why it matters: if intrinsic d is small, every reducer above has a
# real target to hit. If d ≈ p, your data is genuinely high-dimensional
# and dim-reduction will hurt downstream accuracy. Estimating d tells
# you which regime you're in BEFORE you commit to any one method.
#
# Four estimators we'll compare:
#   1. PCA 80/90/95% variance thresholds
#   2. Kaiser: count eigenvalues > 1
#   3. Broken-stick: count eigenvalues beating a random partition share
#   4. Nearest-neighbour MLE (Levina & Bickel, 2004). With T_j(x) the
#      distance from x to its j-th nearest neighbour (x itself excluded):
#          m_k(x) = [ 1/(k-1) * sum_{j=1}^{k-1} log( T_k(x) / T_j(x) ) ]^-1
#      averaged over points x (and here over several k).
#
# How do we compare reducers fairly? Not by K-means silhouette in each
# embedding: t-SNE and UMAP pull points into blobs by construction, so
# silhouette rewards the picture, not the faithfulness. We rank by
# TRUSTWORTHINESS (are embedding neighbours genuine neighbours?) on the
# same rows, and keep silhouette only as a "clusterability" column.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: shared preprocessing + pre-reduction
# ════════════════════════════════════════════════════════════════════════

X, feature_cols, _ = load_customer_matrix()
n_samples, n_features = X.shape
print(f"=== E-commerce customers ===  n={n_samples:,}, p={n_features}")

# Baseline PCA — needed by t-SNE/UMAP pre-reduction AND by intrinsic-dim.
pca_full = PCA(n_components=n_features, random_state=42)
pca_full.fit(X)
evr = pca_full.explained_variance_ratio_
cum_evr = np.cumsum(evr)
explained_variance = pca_full.explained_variance_

n_80 = int(np.searchsorted(cum_evr, 0.80) + 1)
n_90 = int(np.searchsorted(cum_evr, 0.90) + 1)
n_95 = int(np.searchsorted(cum_evr, 0.95) + 1)

X_pca10 = pca_full.transform(X)[:, : min(10, n_features)]
idx = subsample_indices(n_samples, n_target=3000)
X_ref = X[idx]  # every method is scored against these original-space rows
# UMAP is fitted on idx and scored on a 10K slice of OTHER rows (a genuine
# out-of-sample transform); that slice excludes every fit row.
TRANSFORM_TARGET = 10_000
transform_idx = holdout_indices(n_samples, exclude=idx, n_target=TRANSFORM_TARGET)
X_pca10_oos = X_pca10[transform_idx]
KPCA_COMPONENTS = 3  # same compression as 02_kernel_pca.py


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: score every reducer on the same rows
# ════════════════════════════════════════════════════════════════════════

method_quality: dict[str, dict[str, float]] = {}


def record(label: str, X_high: np.ndarray, embedding: np.ndarray) -> None:
    """Score one configuration and store it with its output dimension."""
    method_quality[label] = {
        **evaluate_embedding(X_high, embedding),
        "dims": float(embedding.shape[1]),
    }


# (a) PCA at 2D, 80%, 90%, 95% variance (fit on all rows, scored on idx)
for n_comp in sorted({2, n_80, n_90, n_95}):
    pca_test = PCA(n_components=n_comp, random_state=42)
    X_test = pca_test.fit_transform(X)
    record(f"PCA {n_comp}d", X_ref, X_test[idx])

# (b) Kernel PCA — two RBF configs, one poly, on the subsample
for kernel, params, label in [
    ("rbf", {"gamma": 0.1}, "KernelPCA rbf g=0.1"),
    ("rbf", {"gamma": 1.0}, "KernelPCA rbf g=1.0"),
    ("poly", {"degree": 3, "gamma": 0.1}, "KernelPCA poly d=3"),
]:
    kpca = KernelPCA(
        n_components=KPCA_COMPONENTS, kernel=kernel, random_state=42, **params
    )
    record(label, X_ref, kpca.fit_transform(X[idx]))

# (c) t-SNE at a few perplexities
for perp in [15, 30, 50]:
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        max_iter=1000,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    record(f"t-SNE p={perp}", X_ref, tsne.fit_transform(X_pca10[idx]))

# (d) UMAP — three configs, fit on idx, scored on held-out rows
if UMAP_AVAILABLE:
    for n_nbr, min_d, label in [
        (15, 0.1, "UMAP default (OOS)"),
        (50, 0.5, "UMAP global (OOS)"),
        (15, 0.0, "UMAP tight (OOS)"),
    ]:
        reducer = umap_lib.UMAP(
            n_components=2,
            n_neighbors=n_nbr,
            min_dist=min_d,
            random_state=42,
            metric="euclidean",
        )
        reducer.fit(X_pca10[idx])
        record(label, X[transform_idx], reducer.transform(X_pca10_oos))
else:
    print("[warn] umap-learn missing — skipping UMAP rows in the leaderboard")

# (e) Isomap — manifold learning reference
iso = Isomap(n_components=2, n_neighbors=10)
record("Isomap k=10", X_ref, iso.fit_transform(X_pca10[idx]))


# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert len(method_quality) >= 10, (
    f"Expected ≥10 method configurations on the leaderboard, got "
    f"{len(method_quality)}"
)
for label, q in method_quality.items():
    assert 0.0 <= q["trustworthiness"] <= 1.0, f"{label}: trust out of range"
print(f"\n[ok] Checkpoint 1 — {len(method_quality)} reducer configurations scored\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: leaderboard + intrinsic dimensionality
# ════════════════════════════════════════════════════════════════════════

print("=== Leaderboard (descending trustworthiness) ===")
print(f"  {'method':<26}{'dims':>5}{'trust':>9}{'kNN ovl':>9}{'silhouette':>12}")
for name, q in sorted(method_quality.items(), key=lambda kv: -kv[1]["trustworthiness"]):
    print(
        f"  {name:<26}{int(q['dims']):>5}{q['trustworthiness']:>9.4f}"
        f"{q['knn_overlap']:>9.4f}{q['silhouette']:>12.4f}"
    )
print(
    "  Compare like with like: a 6-D PCA keeps more neighbourhoods than any"
    " 2-D map simply because it keeps more dimensions."
)

two_d = {k: v for k, v in method_quality.items() if v["dims"] == 2}
best_2d = max(two_d, key=lambda k: two_d[k]["trustworthiness"])
blobbiest_2d = max(two_d, key=lambda k: two_d[k]["silhouette"])
print(f"\n  Most faithful 2-D map : {best_2d}")
print(f"  Most blob-like 2-D map: {blobbiest_2d}")
if best_2d != blobbiest_2d:
    print(
        "  -> They differ: ranking by silhouette would have picked the"
        " prettier picture, not the more faithful one."
    )

viz = ModelVisualizer()
fig = viz.metric_comparison(
    {
        name: {
            "Trustworthiness": q["trustworthiness"],
            "kNN overlap": q["knn_overlap"],
            "Silhouette": q["silhouette"],
        }
        for name, q in method_quality.items()
    }
)
fig.update_layout(title="Dimensionality reduction: structure preservation leaderboard")
leaderboard_path = OUTPUT_DIR / "05_leaderboard.html"
fig.write_html(str(leaderboard_path))
print(f"\nSaved: {leaderboard_path}")


# Intrinsic dimensionality estimators
n_kaiser = int((explained_variance > 1.0).sum())
broken_stick = np.array(
    [
        sum(1.0 / j for j in range(i, n_features + 1)) / n_features
        for i in range(1, n_features + 1)
    ]
)
n_broken = int((evr > broken_stick).sum())


def estimate_intrinsic_dim_nn(
    X: np.ndarray, k_values: list[int], n_sub: int = 1000
) -> tuple[float, dict[int, float]]:
    """Levina-Bickel MLE estimator of intrinsic dimension.

    Returns (estimate averaged over k_values, per-k estimates).
    """
    rng = np.random.default_rng(42)
    sub = rng.choice(len(X), min(n_sub, len(X)), replace=False)
    X_s = X[sub]
    k_max = max(k_values)
    # Ask for k_max + 1 neighbours and drop column 0: querying the points
    # the index was built on returns each point as its OWN nearest
    # neighbour at distance 0, which would make every log-ratio infinite.
    dists, _ = NearestNeighbors(n_neighbors=k_max + 1).fit(X_s).kneighbors(X_s)
    T = dists[:, 1:]  # T[:, j-1] = distance to the j-th true neighbour
    per_k: dict[int, float] = {}
    for k in k_values:
        T_k = T[:, :k]
        valid = np.all(T_k > 0, axis=1)  # exact duplicates have T_j = 0
        if valid.sum() < 10:
            raise ValueError(
                f"Levina-Bickel: only {int(valid.sum())} points have {k} "
                "distinct neighbours — deduplicate the data first"
            )
        log_ratios = np.log(T_k[valid, k - 1][:, None] / T_k[valid, : k - 1])
        m_k = 1.0 / log_ratios.mean(axis=1)  # per-point estimate
        per_k[k] = float(m_k.mean())
    return float(np.mean(list(per_k.values()))), per_k


intrinsic_mle, intrinsic_per_k = estimate_intrinsic_dim_nn(X, k_values=[10, 20, 30])

print("\n=== Intrinsic dimensionality estimates ===")
print(f"  Ambient (p)          : {n_features}")
print(f"  PCA 80% variance     : {n_80}")
print(f"  PCA 90% variance     : {n_90}")
print(f"  PCA 95% variance     : {n_95}")
print(f"  Kaiser (eig > 1)     : {n_kaiser}")
print(f"  Broken-stick         : {n_broken}")
print(
    f"  NN MLE (Levina-Bickel): {intrinsic_mle:.1f}  "
    f"(per k: {', '.join(f'k={k}: {v:.1f}' for k, v in intrinsic_per_k.items())})"
)
print(f"\n  Practical recommendation: use {n_90} dims for downstream ML")


# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert n_90 <= n_features, "intrinsic dim cannot exceed ambient"
assert 1 <= n_kaiser <= n_features
assert np.isfinite(intrinsic_mle), "Levina-Bickel estimate must be finite"
assert 0.5 < intrinsic_mle < 2 * n_features, "MLE should be on the scale of p"
print("\n[ok] Checkpoint 2 — intrinsic dimensionality estimated via 4 methods\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Respondent Segmentation for a Public-Sector Forms Platform
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore public-sector digital forms
# platform handles millions of submissions a year across many agencies.
# Product analytics wants to segment respondents by completion behaviour:
# dozens of features per submission (time per field, back-navigation
# count, autofill usage, abandonment point, device class, validation
# error rate, accessibility mode). Three audiences need different cuts:
#
#   - Product managers (strategy deck)   -> a faithful 2-D picture
#   - Data science team (churn model)    -> PCA features (exact inverse,
#                                           stable, explainable loadings)
#   - Production ML (dropout detector)   -> an embedder with an
#                                           out-of-sample transform (UMAP
#                                           or PCA)
#
# WHY THE COMPARISON MATTERS: forcing all three audiences onto one reducer
# is a common way dim-reduction projects fail. PCA gives the DS team an
# exact inverse but can give a cluttered 2-D picture. t-SNE gives a vivid
# picture but cannot embed tomorrow's submissions. UMAP is a reasonable
# compromise for the picture and the embedder — check its trustworthiness.
#
# THE RIGHT WORKFLOW:
#   1. Run this comparison on the monthly snapshot.
#   2. Use the INTRINSIC DIM ESTIMATE to bound expectations — if d ~ 6,
#      compressing dozens of features to 6-10 loses little; if d ~ 40,
#      no method can go far below 40 without losing real signal.
#   3. Pick the reducer per audience from the leaderboard, comparing
#      methods at the SAME output dimension.
#
# BUSINESS IMPACT (illustrative assumptions, not reported figures): if
# segmentation-driven UX fixes cut abandonment on the busiest forms by a
# few percentage points, every completed online submission avoids a
# manual back-office follow-up. Multiply (extra completions) x (cost of a
# manual follow-up) for your own platform to size it.

top_method, top_q = max(
    method_quality.items(), key=lambda kv: kv[1]["trustworthiness"]
)
top_trust = top_q["trustworthiness"]
print("\n=== Forms-platform projection ===")
print(
    f"  Most faithful reducer overall : {top_method}  "
    f"(trustworthiness {top_trust:.4f}, {int(top_q['dims'])} dims)"
)
print(f"  Most faithful 2-D map       : {best_2d}")
print(f"  Intrinsic dim (NN MLE)      : {intrinsic_mle:.1f}")
print(f"  Ambient dim                 : {n_features}")
print(
    f"  Headroom for compression    : "
    f"{n_features - n_90} dimensions of noise available to discard"
)


# ════════════════════════════════════════════════════════════════════════
# DECISION GUIDE
# ════════════════════════════════════════════════════════════════════════
print(
    """

  +------------+---------+----------------+---------------+---------------+-----------+
  | Method     | Linear? | Global struct. | Out-of-sample | Inverse       | Speed     |
  +------------+---------+----------------+---------------+---------------+-----------+
  | PCA        | yes     | yes            | yes           | exact         | O(n p^2)  |
  | Kernel PCA | no      | partial        | yes           | approx (fit)  | O(n^2 p)  |
  | t-SNE      | no      | local only     | NO            | none          | O(n log n)|
  | UMAP       | no      | partly + local | yes           | approx        | ~O(n)     |
  | Isomap     | no      | geodesic       | yes           | none          | O(n^2)    |
  +------------+---------+----------------+---------------+---------------+-----------+

  PRODUCTION:   PCA first, UMAP if PCA is insufficient.
  VISUALISE:    t-SNE for dense micro-clusters, UMAP for mixed scales —
                check trustworthiness, not just how clean the blobs look.
  EXPLAIN:      PCA — its inverse_transform is exact linear algebra.
"""
)


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# This is the FINAL lesson in the M4 ex_3 dim-reduction block. After this,
# the m4_dimreduction_zoo experiment in mlfp04_ex3_dimreduction.db holds:
#   - pca_svd                  (ex_3/01)
#   - kernel_pca_<best_kernel> (ex_3/02)
#   - tsne_perp_<best_p>       (ex_3/03)
#   - umap_<best_config>       (ex_3/04)
#   - method_comparison        (this lesson)
#
# The leaderboard now lives in two places: this lesson's per-method dict,
# and the SQLite store on disk for cross-lesson comparison.


def _slug(s: str) -> str:
    """Tracker metric keys allow only [A-Za-z0-9_.-]; slugify the rest."""
    out = "".join(c if c.isalnum() or c in "_.-" else "_" for c in s)
    return out.lstrip("_") or "k"


track_run(
    tracker,
    exp_name,
    run_name="method_comparison",
    params={
        "algorithms_compared": "pca,kernel_pca,tsne,umap,isomap",
        "n_configs": len(method_quality),
        "n_features_ambient": n_features,
        "n_samples": n_samples,
        "intrinsic_dim_method": "levina_bickel_nn_mle",
    },
    scalar_metrics={
        "top_trustworthiness": float(top_trust),
        "intrinsic_dim_mle": float(intrinsic_mle),
        "n_components_80": float(n_80),
        "n_components_90": float(n_90),
        "n_components_95": float(n_95),
        "n_kaiser": float(n_kaiser),
        "n_broken_stick": float(n_broken),
    }
    | {
        f"trust_{_slug(name)}": float(q["trustworthiness"])
        for name, q in method_quality.items()
    }
    | {
        # NaN-guard silhouettes — a collapsed embedding can yield NaN.
        f"sil_{_slug(name)}": (
            float(q["silhouette"]) if q["silhouette"] == q["silhouette"] else 0.0
        )
        for name, q in method_quality.items()
    },
)
print(f"  [tracked] cross-method leaderboard logged to {exp_name}\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — DimReductionEngine across the four supported
# ════════════════════════════════════════════════════════════════════════
# kailash-ml's DimReductionEngine wraps pca / tsne / umap / nmf under one
# `reduce` surface. Run all four through the engine on the same rows and
# score them with the same trustworthiness ruler.

import polars as pl

from kailash_ml.engines.dim_reduction import DimReductionEngine

cust_df = pl.from_numpy(X_ref, schema=feature_cols)
# NMF needs non-negative input: shift each standardised column to >= 0.
cust_nonneg = pl.from_numpy(X_ref - X_ref.min(axis=0) + 1e-6, schema=feature_cols)
dimreduce = DimReductionEngine()

engine_leaderboard: dict[str, float] = {}
for alg in ("pca", "tsne", "umap", "nmf"):
    data = cust_nonneg if alg == "nmf" else cust_df
    r = dimreduce.reduce(data, algorithm=alg, n_components=2)
    emb = np.asarray(r.transformed)
    engine_leaderboard[f"engine.{alg}"] = evaluate_embedding(X_ref, emb)[
        "trustworthiness"
    ]

print("\n  DimReductionEngine.reduce leaderboard (2-D, trustworthiness):")
for name, trust in sorted(engine_leaderboard.items(), key=lambda x: -x[1]):
    print(f"    {name:<22}: {trust:.4f}")
print(
    "\n  Same ruler, four algorithms, one engine surface — open"
    " mlfp04_ex3_dimreduction.db for the full m4_dimreduction_zoo"
    " leaderboard across lessons 01-04 plus this comparison.\n"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Ran five reducer families on the same dataset with one metric
  [x] Built a trustworthiness leaderboard, with silhouette demoted to a
      clusterability column
  [x] Estimated intrinsic dimensionality with four methods, including a
      correct Levina-Bickel nearest-neighbour MLE
  [x] Picked per-audience reducers for a forms-platform scenario
      (illustrative)

  KEY INSIGHT: There is no "best" dimensionality reducer — there is only
  a best reducer FOR A SPECIFIC AUDIENCE AND DOWNSTREAM TASK. Compare on
  a neighbourhood-preservation ruler at the same output dimension,
  estimate intrinsic dim, and pick the reducer for the job.

  You have now completed Exercise 3. Next: Exercise 4 turns to anomaly
  detection — finding the rare rows that do not fit the structure.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
