# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 6.1: TreeSHAP and Global Feature Importance
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Compute exact TreeSHAP values in O(TLD²) for LightGBM models
#   - Verify the Shapley additivity axiom (SHAP sum + base = model output)
#   - Rank features globally by mean |SHAP|
#   - Read dependence plots to find the SIGN of each feature's effect
#   - Apply: a model-risk audit pack for a Singapore retail bank
#
# PREREQUISITES:
#   - MLFP03 Exercise 4 (LightGBM training)
#   - MLFP03 Exercise 5 (class imbalance — same model is explained here)
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — why Shapley values are the "right" attribution
#   2. Build — TreeExplainer on the trained LightGBM credit model
#   3. Train — no training; we EXPLAIN a pre-trained model
#   4. Visualise — additivity check + global importance ranking
#   5. Apply — model-risk audit pack (illustrative bank)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv

from shared.mlfp03.ex_6 import (
    OUTPUT_DIR,
    build_shap_explainer,
    print_section,
    rank_features_by_mean_abs_shap,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Shapley Values Are the "Right" Attribution
# ════════════════════════════════════════════════════════════════════════
# Shapley values come from cooperative game theory (Shapley 1953). For a
# prediction f(x), each feature is a "player" and the model output is the
# "payoff". The Shapley value φ_i is the average marginal contribution of
# player i across every possible coalition of the other players:
#
#     φ_i = Σ_S [|S|!(|F|-|S|-1)! / |F|!] · [f(S ∪ {i}) - f(S)]
#
# This is the UNIQUE attribution method satisfying four axioms:
#   1. Efficiency — φ values sum to f(x) - E[f(x)] (additivity)
#   2. Symmetry — features with identical marginal contributions get equal φ
#   3. Dummy — features with zero marginal contribution get φ = 0
#   4. Linearity — φ(f + g) = φ(f) + φ(g)
#
# For a general model, exact Shapley takes O(2^F) — infeasible. TreeSHAP
# (Lundberg & Lee 2018) exploits tree structure to compute exact Shapley
# in O(TLD²) where T=trees, L=max leaves, D=max depth. That's what we use.
#
# WHAT IS "THE MODEL OUTPUT" HERE? For a LightGBM classifier TreeSHAP
# explains the RAW score — the log-odds z, where P(default) = sigmoid(z).
# So additivity reads: base value + Σ φ_i = z (log-odds), NOT the
# probability. Comparing SHAP sums with predict_proba is a classic bug.
#
# WHY shap.TreeExplainer AND NOT kailash-ml's ModelExplainer: ModelExplainer
# wraps shap.Explainer with a background sample (interventional TreeSHAP).
# On this LightGBM model, in kailash-ml 2.2.2, that path fails shap's own
# additivity check, so this exercise uses path-dependent TreeSHAP directly
# (shared.mlfp03.ex_6.build_shap_explainer) and verifies additivity itself.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the TreeSHAP explainer
# ════════════════════════════════════════════════════════════════════════

bundle = build_shap_explainer()
model = bundle["model"]
X_test = bundle["X_test"]
y_test = bundle["y_test"]
feature_names: list[str] = bundle["feature_names"]
shap_vals: np.ndarray = bundle["shap_vals"]
expected_value: float = bundle["expected_value"]
auc = bundle["auc"]

print_section("TreeSHAP: Global Feature Attribution for Credit Default")
print(f"Model AUC-ROC:   {auc:.4f}")
print(f"SHAP shape:      {shap_vals.shape}  (samples x features)")
print(f"Expected value:  {expected_value:.4f}  (average raw score, in log-odds)")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — "TRAIN" = explain the trained model (SHAP has no training)
# ════════════════════════════════════════════════════════════════════════
# Verify the additivity axiom on EVERY test row: E[f(x)] + Σφ = f(x), where
# f(x) is the raw log-odds score (predict(..., raw_score=True)).

raw_scores = model.predict(X_test, raw_score=True)
shap_totals = shap_vals.sum(axis=1) + expected_value
additivity_gaps = np.abs(shap_totals - raw_scores)
mean_gap = float(additivity_gaps.mean())

print_section("Additivity Verification (log-odds)", char="─")
print(f"{'Sample':>8} {'base+ΣSHAP':>12} {'raw score':>12} {'P(default)':>11} {'|Gap|':>10}")
print("─" * 58)
for i in range(5):
    p_default = 1.0 / (1.0 + np.exp(-raw_scores[i]))
    print(f"{i:>8} {shap_totals[i]:>12.4f} {raw_scores[i]:>12.4f} {p_default:>11.4f} {additivity_gaps[i]:>10.2e}")
print(f"\nMean |gap| over {len(raw_scores):,} rows: {mean_gap:.2e}")

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert shap_vals.shape == (
    X_test.shape[0],
    X_test.shape[1],
), "Task 3: SHAP shape must be (n_samples, n_features)"
assert mean_gap < 1e-6, "Task 3: base value + sum of SHAP must equal the raw log-odds"
# INTERPRETATION: Additivity means every feature gets credit for EXACTLY
# its contribution to this prediction. Unlike gain-based importance
# (which is a global average), SHAP decomposes each individual prediction
# into per-feature contributions — the basis for the "right to explanation".
print("\n[ok] Checkpoint 1 — additivity verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE global importance ranking
# ════════════════════════════════════════════════════════════════════════

importance_ranking = rank_features_by_mean_abs_shap(shap_vals, feature_names)

print_section("Global Feature Importance (mean |SHAP|)")
print(f"{'Rank':>4} {'Feature':<30} {'mean|SHAP|':>12}")
print("─" * 60)
for rank, (name, imp) in enumerate(importance_ranking[:15], 1):
    bar = "#" * min(int(imp * 100), 60)
    print(f"{rank:>4} {name:<30} {imp:>12.4f}  {bar}")

top15 = importance_ranking[:15]
fig = go.Figure(
    go.Bar(
        x=[imp for _, imp in reversed(top15)],
        y=[name for name, _ in reversed(top15)],
        orientation="h",
        marker_color="#6366f1",
    )
)
fig.update_layout(
    title="Global importance: mean |SHAP| (log-odds) — Singapore credit default",
    xaxis_title="mean |SHAP value| (average shift in log-odds)",
    height=520,
)
html_out = OUTPUT_DIR / "ex6_01_shap_global_importance.html"
fig.write_html(str(html_out))
print(f"\nSaved: {html_out}")


# Dependence direction — SHAP vs feature value correlation
print_section("Dependence Direction (top 5 features)", char="─")
for feat_name, _ in importance_ranking[:5]:
    feat_idx = feature_names.index(feat_name)
    feat_vals = X_test[:, feat_idx]
    feat_shap = shap_vals[:, feat_idx]
    valid = ~(np.isnan(feat_vals) | np.isnan(feat_shap))
    if valid.sum() > 2:
        corr = float(np.corrcoef(feat_vals[valid], feat_shap[valid])[0, 1])
    else:
        corr = 0.0
    direction = "^ increases default risk" if corr > 0 else "^ decreases default risk"
    print(f"  {feat_name:<30} corr={corr:+.3f}  ({direction})")

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert len(importance_ranking) == len(feature_names), "Task 4: all features ranked"
top_name, top_imp = importance_ranking[0]
assert top_imp > 0, "Task 4: top feature must have positive importance"
print("\n[ok] Checkpoint 2 — global SHAP importance computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: A Model-Risk Audit Pack (illustrative bank)
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore retail bank deploys this LightGBM
# model as a pre-approval filter for unsecured personal loans. Its model-
# risk team (and supervisory guidance such as the MAS FEAT principles,
# which call for transparency in AI-driven decisions) expects:
#
#   1. A documented feature-contribution audit for every production model
#   2. A per-application explanation available when a customer asks
#   3. A periodic review of the top drivers of declined decisions
#
# Why SHAP fits:
#   - TreeSHAP is exact for LightGBM (no sampling noise)
#   - Additivity — which you just verified to ~1e-6 — makes every score
#     decomposable: base value + contributions = the model's log-odds
#   - The global ranking is the input to the periodic driver review
#
# BUSINESS VALUE (illustrative assumptions — replace with your own):
#   if documented explanations let the bank resolve even a handful of
#   escalated complaints a year without external dispute resolution, a
#   one-off pipeline costing a few engineer-weeks pays for itself.

top3 = ", ".join(name for name, _ in importance_ranking[:3])
print_section("Audit-pack summary (computed)", char="─")
print(f"  Model AUC-ROC:              {auc:.4f}")
print(f"  Additivity (mean |gap|):    {mean_gap:.2e} log-odds")
print(f"  Top-3 global drivers:       {top3}")
#
# LIMITATION: SHAP attributes the MODEL output. If the model is biased,
# SHAP explains the bias rather than eliminating it. That's why Exercise
# 6.5 (fairness audit) runs downstream of the SHAP pipeline.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print_section("WHAT YOU'VE MASTERED")
print(
    """
  [x] Built a TreeSHAP explainer against a trained LightGBM credit model
  [x] Verified the Shapley additivity axiom against the raw log-odds
  [x] Ranked features globally by mean |SHAP|
  [x] Read the sign of each feature's effect via SHAP-value correlation
  [x] Mapped the pipeline onto a bank's model-risk audit pack

  KEY INSIGHT: SHAP is not "yet another feature importance". It is the
  unique attribution method that satisfies the four Shapley axioms, and
  TreeSHAP computes it EXACTLY for tree models in polynomial time.

  Next: 02_permutation_importance.py — implement permutation importance
  from scratch and compare its ranking against SHAP.
"""
)
