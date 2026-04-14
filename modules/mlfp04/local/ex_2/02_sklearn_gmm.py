# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 2.2: sklearn GMM and BIC/AIC Model Selection
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Fit a GaussianMixture through kailash-ml's sklearn bridge
#   - Verify the library result matches the from-scratch EM from 2.1
#   - Sweep K using BIC and AIC to select the number of components
#   - Explain WHY BIC is more conservative than AIC (different penalties)
#   - Read a BIC/AIC curve and recognise the elbow
#
# PREREQUISITES: 01_em_from_scratch.py
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — BIC and AIC as log-likelihood minus complexity penalty
#   2. Build — fit_sklearn_gmm helper + bic_aic_sweep
#   3. Train — fit K=2..8 on Singapore e-commerce customers
#   4. Visualise — BIC/AIC curves + silhouette overlay
#   5. Apply — Shopee SEA customer segmentation at enterprise scale
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture

from kailash_ml import ModelVisualizer

from shared.mlfp04.ex_2 import (
    load_customers_scaled,
    out_path,
    safe_silhouette,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — BIC and AIC
# ════════════════════════════════════════════════════════════════════════
#   BIC = k * log(n) - 2 * log_lik   (Schwarz)
#   AIC = 2 * k      - 2 * log_lik   (Akaike)
#
# Lower is better for both. BIC's log(n) penalty grows with sample size,
# making it more conservative than AIC for large n. BIC is consistent
# (picks the true K as n -> infinity); AIC is efficient (minimises
# prediction error but may over-parameterise).
#
# PRACTICAL RULE: when BIC and AIC disagree, prefer BIC.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: fit_sklearn_gmm + bic_aic_sweep
# ════════════════════════════════════════════════════════════════════════


def fit_sklearn_gmm(
    X: np.ndarray,
    n_components: int,
    cov_type: str = "full",
    random_state: int = 42,
) -> GaussianMixture:
    """Fit a sklearn GaussianMixture and return the fitted estimator."""
    # TODO: construct GaussianMixture with n_components, covariance_type=cov_type,
    # random_state=random_state, max_iter=200 — then .fit(X) and return it.
    gmm = ____
    ____
    return gmm


def bic_aic_sweep(
    X: np.ndarray,
    k_range: range,
    cov_type: str = "full",
) -> dict[int, dict[str, float]]:
    """Fit a GMM for each K and record BIC, AIC, log-lik, silhouette."""
    results: dict[int, dict[str, float]] = {}
    for k in k_range:
        gmm = fit_sklearn_gmm(X, k, cov_type=cov_type)
        labels = gmm.predict(X)
        # TODO: fill the results dict with bic, aic, log_lik, silhouette
        # Hint: gmm.bic(X), gmm.aic(X), gmm.score(X) * X.shape[0], safe_silhouette(X, labels)
        results[k] = {
            "bic": ____,
            "aic": ____,
            "log_lik": ____,
            "silhouette": ____,
        }
    return results


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: fit K=2..8 on Singapore e-commerce customers
# ════════════════════════════════════════════════════════════════════════

X_scaled, customers, feature_cols, _ = load_customers_scaled()

print("=" * 70)
print("  Singapore e-commerce customers")
print("=" * 70)
print(f"Rows: {X_scaled.shape[0]}  Features: {X_scaled.shape[1]}")

print("\nSweeping K = 2..8 with sklearn GaussianMixture (cov_type=full)...")
# TODO: call bic_aic_sweep(X_scaled, range(2, 9))
sweep = ____

print(f"\n{'K':>4} {'BIC':>12} {'AIC':>12} {'log_lik':>12} {'silhouette':>12}")
print("─" * 56)
for k, v in sweep.items():
    print(
        f"{k:>4} {v['bic']:>12.0f} {v['aic']:>12.0f} "
        f"{v['log_lik']:>12.0f} {v['silhouette']:>12.4f}"
    )

# TODO: pick K that minimises BIC, K that minimises AIC, K that maximises silhouette
# Hint: min(sweep.items(), key=lambda kv: kv[1]["bic"])[0]
best_k_bic = ____
best_k_aic = ____
best_k_sil = ____

print(f"\nBest K by BIC:        {best_k_bic}")
print(f"Best K by AIC:        {best_k_aic}")
print(f"Best K by silhouette: {best_k_sil}")

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert best_k_bic in range(2, 9), "BIC-optimal K must be in the searched range"
assert best_k_aic in range(2, 9), "AIC-optimal K must be in the searched range"
bic_values = [v["bic"] for v in sweep.values()]
assert max(bic_values) - min(bic_values) > 1.0, "BIC should vary across K"
print("\n[ok] Checkpoint 1 passed — BIC/AIC sweep produced a usable ranking")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: BIC/AIC curves + silhouette overlay
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()
comparison = {
    f"K={k}": {"BIC": v["bic"], "AIC": v["aic"], "silhouette": v["silhouette"]}
    for k, v in sweep.items()
}
# TODO: call viz.metric_comparison(comparison)
fig = ____
fig.update_layout(title="GMM model selection: BIC, AIC, silhouette vs K")
fig.write_html(str(out_path("ex2_sklearn_bic_aic.html")))
print(f"\nSaved: {out_path('ex2_sklearn_bic_aic.html')}")

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert out_path("ex2_sklearn_bic_aic.html").exists(), "BIC/AIC plot must be written"
print("[ok] Checkpoint 2 passed — BIC/AIC visualisation written")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Shopee SEA customer segmentation at scale
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Shopee (Sea Group, Singapore-HQ) runs the largest e-commerce
# platform in SEA — ~80M monthly actives across 7 countries. The growth
# team segments customers for lifecycle campaigns: welcome, re-engagement,
# win-back, retention.
#
# Why BIC over "pick K=4 because marketing likes round numbers":
#   When true structure has 6-7 segments, K=4 merges genuinely different
#   buyers and creative targeted to the merged segment misfires on half
#   the members.
#
# BUSINESS IMPACT (from a Lazada 2024 study on SEA marketplaces):
#   - Monthly campaign spend: ~S$420,000
#   - Baseline open rate with K=4: ~18%
#   - BIC-selected K=6 lifted campaign engagement by ~23%
#   - 23% uplift on S$420K already-paid spend = S$97K/month = S$1.16M/year
#     in recovered ROI, zero extra infra cost.

# TODO: fit a GMM at best_k_bic and extract its mixing weights
best_gmm = ____
segment_weights = best_gmm.weights_

print("\n" + "=" * 70)
print(f"  APPLY — Shopee SEA segmentation (BIC-optimal K={best_k_bic})")
print("=" * 70)
for k, w in enumerate(segment_weights):
    print(f"  Segment {k}: weight={w:.3f}  (~{w * 100:.1f}% of customers)")

# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert len(segment_weights) == best_k_bic, "weights length must match K"
assert abs(segment_weights.sum() - 1.0) < 1e-6, "weights must sum to 1"
print("\n[ok] Checkpoint 3 passed — BIC-optimal segmentation produced")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Fit sklearn GaussianMixture as a drop-in for your from-scratch EM
  [x] Compute BIC and AIC to penalise model complexity
  [x] Select K by the BIC elbow (here: K={best_k_bic})
  [x] Shopee SEA scenario: BIC-guided K turns into S$1.16M/year in ROI

  Next: 03_covariance_types.py — same K, four cluster shapes, BIC picks.
"""
)
