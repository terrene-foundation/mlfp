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
#   3. Train — sweep gamma for RBF, record silhouette per config
#   4. Visualise — silhouette bar chart across kernel configurations
#   5. Apply — Grab Singapore driver-behaviour fraud screening
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time

from sklearn.decomposition import KernelPCA

from kailash_ml import ModelVisualizer

from shared.mlfp04.ex_3 import (
    OUTPUT_DIR,
    evaluate_embedding_silhouette,
    load_customer_matrix,
    subsample_indices,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — the kernel trick
# ════════════════════════════════════════════════════════════════════════
# Pretend we ran PCA in a much richer feature space phi(x), without ever
# computing phi(x) explicitly:  K(x_i, x_j) = <phi(x_i), phi(x_j)>.
# Do eigen-decomposition on the n x n kernel matrix K. Kernels:
#   - linear:     K(x, y) = x . y   (equivalent to standard PCA)
#   - RBF:        K(x, y) = exp(-gamma * ||x - y||^2)
#   - polynomial: K(x, y) = (gamma * x . y + coef0)^degree
# COST: O(n^2) in memory. Subsample first.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: data + subsample for kernel cost
# ════════════════════════════════════════════════════════════════════════

X, feature_cols, _ = load_customer_matrix()
n_samples, n_features = X.shape
print(f"=== E-commerce customers ===  n={n_samples:,}, p={n_features}")

# TODO: build a 3,000-row subsample of X. Use the shared helper.
# Hint: idx = subsample_indices(n_samples, n_target=3000)
idx = ____
X_sub = X[idx]
print(f"Subsampled for kernel PCA: {X_sub.shape[0]:,} rows")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: sweep kernels + gamma values
# ════════════════════════════════════════════════════════════════════════

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

print(f"\n=== Kernel PCA sweep ===")
print(f"{'kernel':<20}{'silhouette':>14}{'time (s)':>12}")
print("-" * 46)

for cfg in kernel_configs:
    t0 = time.time()
    # TODO: build a KernelPCA with n_components=8 and cfg['kernel'] and
    # **cfg['params']. Use random_state=42.
    # Hint: KernelPCA(n_components=8, kernel=..., random_state=42, **...)
    kpca = ____
    # TODO: fit_transform X_sub to produce the embedding.
    X_embed = ____
    elapsed = time.time() - t0

    # TODO: score the embedding using the shared silhouette helper.
    # Hint: evaluate_embedding_silhouette(X_embed)
    sil = ____
    kernel_results[cfg["label"]] = {"silhouette": sil, "time_s": elapsed}
    print(f"{cfg['label']:<20}{sil:>14.4f}{elapsed:>12.2f}")

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert len(kernel_results) == 4, "Must evaluate all 4 kernel configurations"
linear_sil = kernel_results["linear"]["silhouette"]
print(
    "\n[ok] Checkpoint 1 — linear silhouette="
    f"{linear_sil:.4f} establishes the PCA baseline to beat"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: silhouette across kernels
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()
fig = viz.metric_comparison(
    {label: {"Silhouette": res["silhouette"]} for label, res in kernel_results.items()}
)
fig.update_layout(title="Kernel PCA: cluster quality across kernels")
kernel_path = OUTPUT_DIR / "02_kernel_pca_silhouette.html"
fig.write_html(str(kernel_path))
print(f"\nSaved: {kernel_path}")

print("\nInterpretation:")
print("  - Linear is your baseline: if nothing beats it, use ordinary PCA.")
print("  - RBF with small gamma (wide kernel) = smooth global manifold.")
print("  - RBF with large gamma (narrow kernel) = local, more complex fit.")
print("  - Poly captures feature interactions at the cost of instability.")
print("  - Kernel PCA has no inverse_transform — you cannot reconstruct X.")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Grab Singapore driver-behaviour fraud screening
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Grab's risk team screens ~50 behavioural features/driver/week
# to find fraud rings. Rings form CURVED clusters — linear PCA misses
# them. Kernel PCA with an RBF kernel surfaces the rings as separable
# blobs. Catching rings 1 week earlier saves ~S$45K per ring, times ~30
# rings/yr = S$1.35M/yr in avoided payout leakage. Downside: no inverse
# transform, and the kernel matrix is O(n^2) so we subsample per city.

best_label, best = max(kernel_results.items(), key=lambda kv: kv[1]["silhouette"])
print(f"\n=== Grab-style fraud-screening projection ===")
print(f"  Linear PCA silhouette : {linear_sil:.4f}")
print(f"  Best kernel           : {best_label}")
print(f"  Best silhouette       : {best['silhouette']:.4f}")
lift = best["silhouette"] - linear_sil
print(f"  Lift over linear PCA  : {lift:+.4f}")


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
  [x] Swept the RBF gamma hyperparameter (narrow vs wide)
  [x] Measured the O(n^2) kernel-matrix cost firsthand
  [x] Sized Kernel PCA for a production fraud-screening pipeline

  KEY INSIGHT: Kernel PCA gives you a curved coordinate system, but you
  pay in memory. Before reaching for the kernel trick, ask: does linear
  PCA already solve the problem?

  Next: 03_tsne.py drops linear algebra for a probabilistic model.
"""
)
