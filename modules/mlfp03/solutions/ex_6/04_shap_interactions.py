# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 6.4: SHAP Interaction Effects
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Compute pairwise SHAP interaction values via TreeExplainer
#   - Distinguish main effects (diagonal) from interactions (off-diagonal)
#   - Rank feature pairs by mean |interaction|
#   - Separate NON-ADDITIVITY (interactions) from non-linear main effects
#   - Apply: an (illustrative) SME-lending cross-feature risk audit
#
# PREREQUISITES: 01_shap_global.py (same SHAP bundle + feature ranking).
#
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Theory — main effects vs interaction effects
#   2. Build — shap_interaction_values on a 500-row sample
#   3. Train — no training; MEASURE the trained model's interactions
#   4. Visualise — top-10 interaction table
#   5. Apply — SME cross-feature risk audit (illustrative bank)
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
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Main Effects vs Interaction Effects
# ════════════════════════════════════════════════════════════════════════
# The standard Shapley decomposition gives each FEATURE a scalar. The
# Shapley INTERACTION index (Grabisch 1997) decomposes each Shapley
# value further into main effects and pairwise interactions:
#
#     phi_i    = phi_ii + sum_{j != i} phi_ij
#
# where phi_ii is the "pure" main effect of feature i and phi_ij is the
# effect of features i and j acting TOGETHER. TreeSHAP splits each pair's
# joint effect equally between the two cells, phi_ij = phi_ji, so every
# ROW of the interaction matrix sums to that feature's ordinary SHAP value
# (you will verify this numerically below).
#
# TreeExplainer.shap_interaction_values() returns a tensor of shape
# (n_samples, n_features, n_features) where:
#   - the DIAGONAL is the main effect of each feature
#   - the OFF-DIAGONAL is the pairwise interaction (symmetric)
#
# An ADDITIVE model f(x) = g_1(x_1) + ... + g_d(x_d) has ZERO off-diagonal
# entries — even when each g_i is a wiggly non-linear curve. So the
# off-diagonal mass measures NON-ADDITIVITY (features modifying each
# other's effect), not "non-linearity". A tree can be strongly non-linear
# yet nearly additive. Auditing the off-diagonal tells you WHICH feature
# combinations the model actually exploits.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the interaction explainer (sample for speed)
# ════════════════════════════════════════════════════════════════════════

bundle = build_shap_explainer()
explainer = bundle["explainer"]
X_test = bundle["X_test"]
feature_names: list[str] = bundle["feature_names"]

sample_size = min(500, X_test.shape[0])
X_sample = X_test[:sample_size]

print_section("SHAP Interaction Values")
print(
    f"Computing interaction tensor for {sample_size} samples x "
    f"{len(feature_names)} features ..."
)

shap_interaction = explainer.shap_interaction_values(X_sample)
if isinstance(shap_interaction, list):
    shap_interaction = shap_interaction[1]

print(f"Interaction tensor shape: {shap_interaction.shape}")

# Verify: each row of the interaction matrix sums to the ordinary SHAP value
shap_sample = bundle["shap_vals"][:sample_size]
row_sum_error = float(np.abs(shap_interaction.sum(axis=2) - shap_sample).max())
print(f"max |row-sum of interactions - SHAP value| = {row_sum_error:.2e}")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — "TRAIN" = measure interactions
# ════════════════════════════════════════════════════════════════════════

n_features = len(feature_names)
interaction_strengths: list[tuple[str, str, float]] = []
for i in range(n_features):
    for j in range(i + 1, n_features):
        strength = float(np.abs(shap_interaction[:, i, j]).mean())
        interaction_strengths.append((feature_names[i], feature_names[j], strength))

interaction_strengths.sort(key=lambda t: t[2], reverse=True)

main_effects = np.abs(np.diagonal(shap_interaction, axis1=1, axis2=2)).mean(axis=0)
main_effects_ranked = sorted(
    zip(feature_names, main_effects), key=lambda t: t[1], reverse=True
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE top interactions + main effects
# ════════════════════════════════════════════════════════════════════════

print_section("Top 10 Main Effects (diagonal)")
print(f"{'Rank':>4} {'Feature':<30} {'Main |SHAP|':>14}")
print("─" * 52)
for rank, (name, val) in enumerate(main_effects_ranked[:10], 1):
    print(f"{rank:>4} {name:<30} {val:>14.4f}")


print_section("Top 10 Pairwise Interactions (off-diagonal)")
print(f"{'Rank':>4} {'Feature 1':<25} {'Feature 2':<25} {'Strength':>12}")
print("─" * 70)
for rank, (f1, f2, strength) in enumerate(interaction_strengths[:10], 1):
    print(f"{rank:>4} {f1:<25} {f2:<25} {strength:>12.4f}")

# How much of the attribution mass sits OFF the diagonal?
abs_mean = np.abs(shap_interaction).mean(axis=0)  # (features x features)
total_main = float(np.trace(abs_mean))
total_interaction = float(abs_mean.sum() - total_main)  # both phi_ij and phi_ji
interaction_share = total_interaction / (total_main + total_interaction)
print(f"\nMain-effect mass:     {total_main:.4f}")
print(f"Interaction mass:     {total_interaction:.4f}")
print(f"Interaction share:    {interaction_share:.1%}")

# ── Visual: SHAP interaction heatmap (top 10 features) ──────────────────
top_n = min(10, n_features)
top_feat_names = [name for name, _ in main_effects_ranked[:top_n]]
top_feat_idxs = [feature_names.index(n) for n in top_feat_names]
interaction_matrix = np.zeros((top_n, top_n))
for i_local, i_global in enumerate(top_feat_idxs):
    for j_local, j_global in enumerate(top_feat_idxs):
        interaction_matrix[i_local, j_local] = float(
            np.abs(shap_interaction[:, i_global, j_global]).mean()
        )

fig = go.Figure(
    data=go.Heatmap(
        z=interaction_matrix,
        x=top_feat_names,
        y=top_feat_names,
        colorscale="Viridis",
        text=np.round(interaction_matrix, 4),
        texttemplate="%{text:.4f}",
        showscale=True,
    )
)
fig.update_layout(
    title=f"SHAP Interaction Heatmap (top {top_n} features, diagonal = main effects)",
    height=550,
    width=650,
)
viz_path = OUTPUT_DIR / "ex6_04_shap_interaction_heatmap.html"
fig.write_html(str(viz_path))
print(f"\n  Saved: {viz_path}")

# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(interaction_strengths) > 0, "Task 4: interaction list must be non-empty"
assert (
    interaction_strengths[0][2] >= 0
), "Task 4: interaction strength must be non-negative"
assert row_sum_error < 1e-6, "Task 4: interaction rows must sum to the SHAP values"
# INTERPRETATION: The interaction share tells you how much attribution
# comes from features modifying each other. A low share means an ADDITIVE
# challenger (e.g. a GAM, or a linear model with good per-feature
# transforms) may be competitive and cheaper to govern; a high share means
# the model relies on combinations that such a challenger cannot express.
print("\n[ok] Checkpoint — interaction tensor computed and ranked\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: SME Cross-Feature Risk Audit (illustrative bank)
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore bank's SME-lending risk team has a
# hypothesis that INCOME x TENURE matters more than either alone: a
# high-income applicant with 6 months of history may be riskier than a
# modest-income applicant with 8 years of history.
#
# Why SHAP interaction values fit:
#   - The interaction tensor directly shows which PAIRS the model uses
#   - The team can check a hypothesis with a lookup instead of a
#     counterfactual experiment
#   - An unexpected pair involving a protected attribute (e.g. age x
#     debt_to_income) is an audit flag for proxy discrimination — check
#     whether any appears in the top-10 list printed above
#
# COST: the audit reuses the existing SHAP pipeline — the only extra cost
# is computing the interaction tensor on a sample, as you just did.

protected = {"age", "gender", "race"}
flagged = [(f1, f2, s) for f1, f2, s in interaction_strengths[:10] if f1 in protected or f2 in protected]
print_section("Interaction audit (computed)", char="─")
print(f"  Interaction share of attribution: {interaction_share:.1%}")
print(f"  Top pair: {interaction_strengths[0][0]} x {interaction_strengths[0][1]}")
if flagged:
    for f1, f2, s in flagged:
        print(f"  FLAG — protected attribute in a top-10 pair: {f1} x {f2} ({s:.4f})")
else:
    print("  No protected attribute appears in the top-10 interaction pairs")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print_section("WHAT YOU'VE MASTERED")
print(
    """
  [x] Computed the SHAP interaction tensor on a 500-sample slice
  [x] Separated main effects (diagonal) from interactions (off-diagonal)
  [x] Ranked feature PAIRS by mean |interaction|
  [x] Verified rows of the interaction matrix sum to the SHAP values
  [x] Quantified the interaction (non-additive) share of attribution
  [x] Mapped the audit to SME underwriting hypothesis testing

  KEY INSIGHT: Interactions measure NON-ADDITIVITY. If the interaction
  share is tiny, an additive challenger with good per-feature curves may
  match the tree and be easier to govern; if it is large, the model's
  value lives in feature combinations.

  Next: 05_fairness_audit.py — step out of accuracy and into FAIRNESS:
  disparate impact, equalized odds, and the impossibility theorem.
"""
)
