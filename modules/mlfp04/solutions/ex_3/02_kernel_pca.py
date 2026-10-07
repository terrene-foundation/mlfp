# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 3.2: Kernel PCA (nonlinear dim reduction)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Apply the kernel trick to extend PCA to nonlinear manifolds
#   - Compare linear, RBF, and polynomial kernels
#   - Tune the RBF gamma hyperparameter (narrow vs wide kernel)
#   - Recognise Kernel PCA's memory wall (O(n^2) kernel matrix)
#
# PREREQUISITES: 01_pca.py (understand linear PCA first).
#
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Theory — the kernel trick in one paragraph
#   2. Build — fit Kernel PCA with linear, RBF, polynomial kernels
#   3. Train — sweep gamma for RBF, record neighbourhood preservation
#      (trustworthiness) and clusterability (silhouette) per config
#   4. Visualise — quality bar chart across kernel configurations
#   5. Apply — driver-behaviour fraud screening at a ride-hailing platform
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time

from sklearn.decomposition import KernelPCA

from kailash_ml import ModelVisualizer

import numpy as np

from shared.mlfp04.ex_3 import (
    OUTPUT_DIR,
    evaluate_embedding,
    load_customer_matrix,
    setup_engines,
    subsample_indices,
    teardown_engines,
    track_run,
)
from shared.mlfp04 import create_visualizer

# ── Kailash-ML ExperimentTracker — every dim-reduction run logs here ─────
tracker, exp_name = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# THEORY — the kernel trick
# ════════════════════════════════════════════════════════════════════════
# Linear PCA finds the axes of greatest variance in the original feature
# space. If your data lies on a curved manifold (a Swiss roll, a moon
# shape, an annulus of fraud vs honest behaviour), linear axes cannot
# unroll it. The fix is to pretend we ran PCA in a much richer feature
# space phi(x), without ever computing phi(x) explicitly:
#
#     K(x_i, x_j) = <phi(x_i), phi(x_j)>
#
# Do eigen-decomposition on the n x n kernel matrix K instead of the
# p x p covariance matrix. Popular kernels:
#   - linear:     K(x, y) = x . y   (equivalent to standard PCA)
#   - RBF:        K(x, y) = exp(-gamma * ||x - y||^2)
#   - polynomial: K(x, y) = (gamma * x . y + coef0)^degree
#
# COST: the kernel matrix is n x n. For n > ~10K rows, memory and wall
# time blow up. Subsample first.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: data + subsample for kernel cost
# ════════════════════════════════════════════════════════════════════════

X, feature_cols, _ = load_customer_matrix()
n_samples, n_features = X.shape
print(f"=== E-commerce customers ===  n={n_samples:,}, p={n_features}")

# Kernel PCA is O(n^2) in memory — subsample to a manageable size.
idx = subsample_indices(n_samples, n_target=3000)
X_sub = X[idx]
# Compress the 7 behavioural features to 3 components. (Keeping all 7
# would make linear PCA a pure rotation with perfect neighbourhood
# preservation, leaving nothing to compare.)
N_COMPONENTS = 3
print(f"Subsampled for kernel PCA: {X_sub.shape[0]:,} rows")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: sweep kernels + gamma values
# ════════════════════════════════════════════════════════════════════════
# For each kernel config we measure:
#   - wall time (kernel PCA is expensive; students should see this)
#   - trustworthiness: are embedding neighbours genuine original-space
#     neighbours? (structure preservation — the ranking metric)
#   - silhouette of 4-cluster KMeans in the embedding (clusterability
#     only — a blob-shaped picture is not proof of preserved structure)

kernel_configs = [
    {"kernel": "linear", "params": {}, "label": "linear"},
    {"kernel": "rbf", "params": {"gamma": 0.1}, "label": "rbf (gamma=0.1)"},
    {"kernel": "rbf", "params": {"gamma": 1.0}, "label": "rbf (gamma=1.0)"},
    {
        "kernel": "poly",
        "params": {"degree": 3, "gamma": 0.1},
        "label": "poly (deg=3)",
    },
]

kernel_results: dict[str, dict] = {}

print("\n=== Kernel PCA sweep ===")
print(f"{'kernel':<20}{'trust':>10}{'silhouette':>14}{'time (s)':>12}")
print("-" * 56)

for cfg in kernel_configs:
    t0 = time.time()
    kpca = KernelPCA(
        n_components=N_COMPONENTS,
        kernel=cfg["kernel"],
        random_state=42,
        **cfg["params"],
    )
    X_embed = kpca.fit_transform(X_sub)
    elapsed = time.time() - t0

    quality = evaluate_embedding(X_sub, X_embed)
    kernel_results[cfg["label"]] = {**quality, "time_s": elapsed}
    print(
        f"{cfg['label']:<20}{quality['trustworthiness']:>10.4f}"
        f"{quality['silhouette']:>14.4f}{elapsed:>12.2f}"
    )

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert len(kernel_results) == 4, "Must evaluate all 4 kernel configurations"
for label, res in kernel_results.items():
    assert 0.0 <= res["trustworthiness"] <= 1.0, f"{label}: trust out of range"
linear_trust = kernel_results["linear"]["trustworthiness"]
linear_sil = kernel_results["linear"]["silhouette"]
print(
    "\n[ok] Checkpoint 1 — linear trustworthiness="
    f"{linear_trust:.4f} establishes the PCA baseline to beat"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: silhouette across kernels
# ════════════════════════════════════════════════════════════════════════

viz = create_visualizer()
fig = viz.metric_comparison(
    {
        label: {
            "Trustworthiness": res["trustworthiness"],
            "kNN overlap": res["knn_overlap"],
            "Silhouette": res["silhouette"],
        }
        for label, res in kernel_results.items()
    }
)
fig.update_layout(title="Kernel PCA: structure preservation vs clusterability")
kernel_path = OUTPUT_DIR / "02_kernel_pca_silhouette.html"
fig.write_html(str(kernel_path))
print(f"\nSaved: {kernel_path}")

print("\nInterpretation:")
print("  - Linear is your baseline: if nothing beats it, use ordinary PCA.")
print("  - RBF with small gamma (wide kernel) = smooth global manifold.")
print("  - RBF with large gamma (narrow kernel) = local, more complex fit.")
print("  - Poly captures feature interactions at the cost of instability.")
print("  - Rank by trustworthiness (structure kept); silhouette only says")
print("    how blob-like the embedding is.")
print("  - Kernel PCA has no EXACT inverse. KernelPCA(fit_inverse_transform=True)")
print("    learns an APPROXIMATE pre-image by regression — unlike PCA's exact")
print("    linear reconstruction.")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Driver-Behaviour Fraud Screening at a Ride-Hailing Platform
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a ride-hailing platform's risk team in Singapore
# screens drivers for collusive behaviour (fake trips, rating manipulation,
# ghost cancellations). Each driver has ~50 behavioural features per week —
# trip counts, time-of-day distributions, rating patterns, cancellation
# geography, payment method mix. Suppose fraud rings form CURVED clusters
# in this space (a small group whose behaviour deforms smoothly into the
# legitimate majority) that linear PCA flattens into a thin crescent.
#
# WHY KERNEL PCA COULD HELP:
#   - An RBF kernel can follow a curved boundary without hand-built
#     features. Gamma sets the kernel width: large gamma = narrow, local
#     fit; small gamma = smooth, global fit.
#   - Subsampling is tolerable when the eigenbasis only needs to cover the
#     shape of normal behaviour.
#   - Polynomial kernels express interactions such as "high cancellation
#     rate AND short trip time AND uncommon payment method".
#
# BUSINESS IMPACT (illustrative assumptions, not reported figures): if a
# ring costs about S$45K a week in fake incentive payouts and the kernel
# view lets analysts catch rings one week earlier, each ring caught saves
# ~S$45K; at 30 rings a year that is ~S$1.35M against a few CPU-minutes per
# weekly batch. Whether the kernel view actually helps is an empirical
# question — compare its trustworthiness with linear PCA first (below).
#
# LIMITATIONS:
#   - Only an approximate, learned pre-image back to feature space, so
#     risk officers need another explanation tool (e.g. SHAP) downstream.
#   - The kernel matrix is O(n^2); scaling up requires per-city
#     subsampling or a switch to UMAP (Exercise 3.4).

print("\n=== Fraud-screening projection (ranked by trustworthiness) ===")
print(f"  Linear PCA trust      : {linear_trust:.4f}")
best_label, best = max(
    kernel_results.items(), key=lambda kv: kv[1]["trustworthiness"]
)
print(f"  Best kernel           : {best_label}")
print(f"  Best trustworthiness  : {best['trustworthiness']:.4f}")
lift = best["trustworthiness"] - linear_trust
print(f"  Lift over linear PCA  : {lift:+.4f}")
if best_label == "linear" or lift < 0.005:
    print("  -> No kernel beats linear PCA here: stay linear (see KEY INSIGHT).")


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# Per-kernel trustworthiness, silhouette and wall-time scalars + series go into the
# m4_dimreduction_zoo experiment for side-by-side comparison with PCA
# (lesson 01) and t-SNE / UMAP (lessons 03-04).

kernel_labels = list(kernel_results.keys())


def _slug(s: str) -> str:
    """Tracker metric keys allow only [A-Za-z0-9_.-]; slugify the rest."""
    out = "".join(c if c.isalnum() or c in "_.-" else "_" for c in s)
    return out.lstrip("_") or "k"


track_run(
    tracker,
    exp_name,
    run_name=f"kernel_pca_{_slug(best_label.split()[0])}",
    params={
        "algorithm": "kernel_pca",
        "n_components": N_COMPONENTS,
        "n_subsample": int(X_sub.shape[0]),
        "n_features": n_features,
        "best_kernel": best_label,
    },
    scalar_metrics={
        "linear_trustworthiness": float(linear_trust),
        "linear_silhouette": float(linear_sil),
        "best_trustworthiness": float(best["trustworthiness"]),
        "best_silhouette": float(best["silhouette"]),
        "lift_over_linear": float(lift),
    }
    | {
        f"{_slug(k)}_trustworthiness": float(v["trustworthiness"])
        for k, v in kernel_results.items()
    }
    | {
        f"{_slug(k)}_silhouette": float(v["silhouette"])
        for k, v in kernel_results.items()
    }
    | {f"{_slug(k)}_time_s": float(v["time_s"]) for k, v in kernel_results.items()},
    series_metrics={
        "kernel_trustworthiness": [
            float(kernel_results[k]["trustworthiness"]) for k in kernel_labels
        ],
        "kernel_silhouettes": [
            float(kernel_results[k]["silhouette"]) for k in kernel_labels
        ],
        "kernel_times_s": [float(kernel_results[k]["time_s"]) for k in kernel_labels],
    },
)
print(f"  [tracked] kernel sweep logged to {exp_name}\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — engine surface honesty for kernel PCA
# ════════════════════════════════════════════════════════════════════════
# kailash-ml's DimReductionEngine supports pca, tsne, umap and nmf — NOT
# kernel_pca, so the nonlinear kernels in this lesson stay on sklearn's
# KernelPCA with the ExperimentTracker as the record. What the engine CAN
# confirm is the baseline: a LINEAR-kernel Kernel PCA is ordinary PCA, so
# the engine's pca embedding should preserve neighbourhoods exactly as
# well as the "linear" row above.

import polars as pl

from kailash_ml.engines.dim_reduction import DimReductionEngine

engine_pca = DimReductionEngine().reduce(
    pl.from_numpy(X_sub, schema=feature_cols),
    algorithm="pca",
    n_components=N_COMPONENTS,
)
engine_quality = evaluate_embedding(X_sub, np.asarray(engine_pca.transformed))
print(
    f"  DimReductionEngine.reduce(pca) trustworthiness = "
    f"{engine_quality['trustworthiness']:.4f}  vs  linear-kernel KernelPCA = "
    f"{linear_trust:.4f}"
)
print(
    "  Engine-first take-away: linear kernel == PCA, which the engine owns;"
    " the nonlinear kernels are compared against it in"
    " mlfp04_ex3_dimreduction.db.\n"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Applied the kernel trick to lift PCA into nonlinear feature spaces
  [x] Compared linear, RBF, polynomial kernels on the same data
  [x] Ranked embeddings by trustworthiness, not by K-means silhouette
  [x] Swept the RBF gamma hyperparameter (narrow vs wide)
  [x] Measured the O(n^2) kernel-matrix cost firsthand
  [x] Framed Kernel PCA for a fraud-screening pipeline (illustrative)

  KEY INSIGHT: Kernel PCA gives you a curved coordinate system, but you
  pay for it in memory. Before reaching for the kernel trick, ask: does
  linear PCA already solve the problem? If yes, stop — linearity is a
  feature, not a weakness.

  Next: 03_tsne.py drops the linear-algebra framing entirely and uses a
  probabilistic neighbourhood model to find local structure.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
