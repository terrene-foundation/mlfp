# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 6.2: From-Scratch Permutation Importance
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Implement permutation importance from scratch (no sklearn.inspection)
#   - Measure feature importance MODEL-AGNOSTICALLY (works for any model)
#   - Quantify estimator variance via repeated shuffles (mean +/- std)
#   - Compare the permutation ranking against SHAP for the same model
#   - See how correlated features distort BOTH permutation and SHAP rankings
#   - Draw an ALE curve: a feature's effect shape without impossible rows
#   - Apply: champion/challenger monitoring at an (illustrative) bank
#
# PREREQUISITES: 01_shap_global.py (we re-use its SHAP ranking for the
# comparison section).
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — why permutation importance is model-agnostic
#   2. Build — implement permutation_importance_manual() from scratch
#   3. Train — no training; we MEASURE the trained model
#   4. Visualise — ranking table, SHAP overlap, correlated twins, ALE curve
#   5. Apply — champion/challenger monitoring (illustrative bank)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
from sklearn.metrics import f1_score, roc_auc_score

from shared.mlfp03.ex_6 import (
    OUTPUT_DIR,
    build_shap_explainer,
    print_section,
    rank_features_by_mean_abs_shap,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Permutation Importance as a Model-Agnostic Probe
# ════════════════════════════════════════════════════════════════════════
# Permutation importance asks: "If I destroy feature j by shuffling its
# values across rows, how much does the model's performance drop?"
#
# Algorithm:
#   1. Score the model on the test set — call it the baseline
#   2. For each feature j:
#        a. Shuffle column j in the test matrix (break j <-> y link)
#        b. Score the model on the shuffled matrix
#        c. importance_j = baseline - shuffled_score
#   3. Repeat K times and average for stability
#
# PROPERTIES:
#   + Works for ANY model (tree, neural net, SVM, linear)
#   + Measures the model's REAL-WORLD reliance on each feature
#   + Variance across repeats gives a standard error
#   - Correlated features share information: shuffling one leaves the
#     other to carry the signal, so both get understated importance
#   - Extrapolation: shuffled rows may be outside the training manifold
#
# SHAP does NOT make correlation go away either. Path-dependent TreeSHAP
# follows the trees' own splits, so when two features carry the same
# information the credit goes mostly to whichever one the trees happened
# to split on — the twin can look unimportant. The two methods answer
# different questions: permutation = "how much does the model RELY on this
# column?"; SHAP = "how is this prediction ATTRIBUTED across columns?".
# With correlated features, report them as a group.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the from-scratch permutation importance function
# ════════════════════════════════════════════════════════════════════════


def permutation_importance_manual(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 5,
    scoring: str = "roc_auc",
    seed: int = 42,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Compute permutation importance from scratch.

    Returns (baseline_score, importances_dict) where importances_dict maps
    feature_name -> {"mean": float, "std": float}.
    """
    rng = np.random.default_rng(seed)

    # TODO: compute the un-shuffled baseline score (predict_proba -> roc_auc_score)
    y_proba = ____
    if scoring == "roc_auc":
        baseline = ____
    else:
        baseline = f1_score(y, model.predict(X))

    importances: dict[str, dict[str, float]] = {}
    for feat_idx, feat_name in enumerate(feature_names):
        drops: list[float] = []
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            # TODO: shuffle column feat_idx (break its link with y)
            # Hint: rng.permutation(...)
            X_shuffled[:, feat_idx] = ____
            y_p_shuffled = model.predict_proba(X_shuffled)[:, 1]
            if scoring == "roc_auc":
                shuffled_score = roc_auc_score(y, y_p_shuffled)
            else:
                shuffled_score = f1_score(y, (y_p_shuffled >= 0.5).astype(int))
            # TODO: record how much the score DROPPED
            ____
        importances[feat_name] = {
            "mean": float(np.mean(drops)),
            "std": float(np.std(drops)),
        }

    return float(baseline), importances


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — "TRAIN" = measure the already-trained model
# ════════════════════════════════════════════════════════════════════════

bundle = build_shap_explainer()
model = bundle["model"]
X_test = bundle["X_test"]
y_test = bundle["y_test"]
feature_names: list[str] = bundle["feature_names"]
shap_vals = bundle["shap_vals"]

print_section("From-Scratch Permutation Importance on Credit Model")
# TODO: call permutation_importance_manual with n_repeats=5, scoring="roc_auc", seed=42
baseline_score, perm_imp = ____
print(f"Baseline AUC-ROC: {baseline_score:.4f}")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the ranking + compare with SHAP
# ════════════════════════════════════════════════════════════════════════

# TODO: sort perm_imp by its "mean" value, descending -> list of (name, vals)
perm_ranking = ____

print(f"\n{'Rank':>4} {'Feature':<30} {'Imp (mean)':>12} {'+/- std':>10}")
print("─" * 58)
for rank, (name, vals) in enumerate(perm_ranking[:15], 1):
    print(f"{rank:>4} {name:<30} {vals['mean']:>12.4f} {vals['std']:>10.4f}")

shap_ranking = rank_features_by_mean_abs_shap(shap_vals, feature_names)

print_section("SHAP vs Permutation — Top 10 Overlap", char="─")
shap_top10 = {n for n, _ in shap_ranking[:10]}
perm_top10 = {n for n, _ in perm_ranking[:10]}
# TODO: features in BOTH top-10 sets
overlap = ____
print(f"Overlap size: {len(overlap)} / 10")
print(f"Shared features: {sorted(overlap)}")
only_shap = sorted(shap_top10 - perm_top10)
only_perm = sorted(perm_top10 - shap_top10)
if only_shap:
    print(f"SHAP-only top-10: {only_shap}")
if only_perm:
    print(f"Permutation-only top-10: {only_perm}")

# ── Visual: Permutation importance bar chart with std error bars ─────────
top_n = min(15, len(perm_ranking))
top_feats = perm_ranking[:top_n]
fig = go.Figure()
fig.add_trace(
    go.Bar(
        y=[name for name, _ in reversed(top_feats)],
        x=[vals["mean"] for _, vals in reversed(top_feats)],
        error_x=dict(
            type="data", array=[vals["std"] for _, vals in reversed(top_feats)]
        ),
        orientation="h",
        marker_color="#6366f1",
    )
)
fig.update_layout(
    title=f"Permutation Importance: Top {top_n} Features (mean +/- std, {5} repeats)",
    xaxis_title="Importance (AUC-ROC drop when shuffled)",
    yaxis_title="Feature",
    height=max(400, top_n * 28),
)
viz_path = OUTPUT_DIR / "ex6_02_permutation_importance.html"
fig.write_html(str(viz_path))
print(f"\n  Saved: {viz_path}")

# ── Correlated twins: how does each method treat them? ──────────────────
corr_matrix = np.corrcoef(X_test, rowvar=False)
np.fill_diagonal(corr_matrix, 0.0)
corr_matrix = np.nan_to_num(corr_matrix)
i_twin, j_twin = np.unravel_index(np.abs(corr_matrix).argmax(), corr_matrix.shape)
shap_lookup = dict(shap_ranking)
print_section("Most-correlated feature pair in the data", char="─")
print(f"  corr({feature_names[i_twin]}, {feature_names[j_twin]}) = {corr_matrix[i_twin, j_twin]:+.3f}")
for idx in (i_twin, j_twin):
    name = feature_names[idx]
    print(
        f"  {name:<28} permutation={perm_imp[name]['mean']:+.4f}  "
        f"mean|SHAP|={shap_lookup[name]:.4f}"
    )

# ── ALE: the effect SHAPE of a correlated feature, without impossible rows ──
# Partial-dependence plots set x_j to a value for EVERY row — producing
# rows like "employed 30 years, aged 25" when x_j is correlated with age.
# ALE (Apley & Zhu 2020) avoids this: within each quantile bin of x_j it
# moves ONLY the rows already in that bin from the bin's lower to upper
# edge, averages the change in the model's log-odds, and accumulates.


def ale_curve(model, X: np.ndarray, j: int, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """First-order ALE of feature j on the log-odds scale (centred)."""
    edges = np.unique(np.quantile(X[:, j], np.linspace(0, 1, n_bins + 1)))
    bin_of_row = np.clip(np.searchsorted(edges, X[:, j], side="right") - 1, 0, len(edges) - 2)
    local_effects = np.zeros(len(edges) - 1)
    counts = np.zeros(len(edges) - 1)
    for b in range(len(edges) - 1):
        rows = X[bin_of_row == b]
        if len(rows) == 0:
            continue
        lo_rows, hi_rows = rows.copy(), rows.copy()
        lo_rows[:, j], hi_rows[:, j] = edges[b], edges[b + 1]
        # TODO: change in raw log-odds when the bin's rows move from lower to upper edge
        # Hint: model.predict(..., raw_score=True) on hi_rows minus on lo_rows
        diff = ____
        local_effects[b], counts[b] = diff.mean(), len(rows)
    ale = np.concatenate([[0.0], np.cumsum(local_effects)])
    centre = np.sum((ale[:-1] + ale[1:]) / 2 * counts) / counts.sum()
    return edges, ale - centre


ale_feature = feature_names[i_twin]
ale_x, ale_y = ale_curve(model, X_test[:3000], i_twin)
print(f"\n  ALE of {ale_feature}: effect range {ale_y.min():+.3f} to {ale_y.max():+.3f} log-odds")
fig_ale = go.Figure(go.Scatter(x=ale_x, y=ale_y, mode="lines+markers", line=dict(color="#10b981")))
fig_ale.update_layout(
    title=f"ALE: effect of {ale_feature} on default log-odds (centred)",
    xaxis_title=ale_feature,
    yaxis_title="ALE (log-odds)",
    height=420,
)
ale_path = OUTPUT_DIR / "ex6_02_ale.html"
fig_ale.write_html(str(ale_path))
print(f"  Saved: {ale_path}")

# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(perm_imp) == len(feature_names), "Task 4: all features must be permuted"
assert (
    perm_ranking[0][1]["mean"] > 0
), "Task 4: top feature must have positive importance"
assert 0 <= len(overlap) <= 10, "Task 4: overlap is a count out of 10"
# INTERPRETATION: The overlap count says how far the two rankings agree.
# Then read the correlated pair: if one twin carries most of the SHAP
# credit while permutation scores BOTH low (shuffling one leaves its twin
# to carry the signal), you are seeing correlation distort each method in
# its own way — neither number alone is "the" importance of either column.
print(f"\n[ok] Checkpoint — permutation ranking computed; top-10 overlap with SHAP = {len(overlap)}/10\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Champion/Challenger Monitoring (illustrative bank)
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore bank runs a "champion / challenger"
# set-up: the production champion is a gradient-boosted tree, and each
# quarter a challenger (say, a neural network) is trained on the same
# features. The risk team wants feature-importance monitoring with the
# SAME meaning on both architectures, so rank-order drift between the two
# can be compared without one model's internal gain metric confounding it.
#
# Why permutation importance fits:
#   - TreeSHAP only works for trees; KernelSHAP on a neural net is far
#     slower for comparable stability
#   - Permutation is model-agnostic: one function, two models, same units
#     (here: drop in AUC-ROC when a column is scrambled)
#   - The per-feature std gives an uncertainty band to gate drift alerts
#
# LIMITATION (you measured it above): correlated columns share credit, so
# monitor correlated groups together — e.g. employment length in years
# and in months are one signal, not two.

print_section("Monitoring summary (computed)", char="─")
print(f"  Baseline AUC-ROC:                  {baseline_score:.4f}")
print(f"  Top permutation feature:           {perm_ranking[0][0]}")
print(f"  Top-10 overlap with SHAP ranking:  {len(overlap)}/10")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print_section("WHAT YOU'VE MASTERED")
print(
    """
  [x] Implemented permutation importance from scratch (no sklearn helper)
  [x] Estimated per-feature importance with a mean +/- std estimate
  [x] Compared permutation and SHAP rankings on the same credit model
  [x] Saw how a correlated feature pair distorts BOTH rankings
  [x] Mapped the pipeline to champion/challenger monitoring

  KEY INSIGHT: Permutation importance is the lingua franca of feature
  monitoring across model architectures. Use SHAP for exact attribution
  on trees, use permutation for cross-architecture comparability.

  Next: 03_lime_local.py — pivot from GLOBAL importance to LOCAL
  explanations for individual credit decisions.
"""
)
