# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 4.1: Boosting Theory (From-Scratch + XGBoost Split Gain)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Explain why boosting reduces BIAS while bagging reduces VARIANCE
#   - Run AdaBoost by hand: re-weight the rows the last stump got wrong
#   - Implement gradient boosting from scratch with shallow decision trees
#   - Derive the XGBoost split-gain formula from a 2nd-order Taylor
#     expansion of the log-loss
#   - Interpret λ (leaf-weight L2) and γ (min split loss) as structural
#     regularisers on trees
#   - Explain when a bank's credit team would refuse a split even if it
#     "looks" informative (the pruning condition: Gain < γ)
#
# PREREQUISITES: Exercise 3 (decision trees, Random Forest). Boosting
# extends the same decision-tree primitive into a sequential ensemble.
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — bias vs variance, sequential residual fitting
#   2. Build — AdaBoost warm-up, then a from-scratch gradient booster
#   3. Train — 10 rounds, watch residuals shrink; derive the split gain
#   4. Visualise — residual shrinkage + final probability surface (HTML)
#   5. Apply — an SME credit committee: when to refuse a split
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from shared.mlfp03.ex_4 import (
    OUTPUT_DIR,
    SEED,
    make_1d_demo,
    xgb_split_gain,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Boosting Works
# ════════════════════════════════════════════════════════════════════════
# Bagging (Random Forest) trains many independent trees on bootstrap
# samples and averages them. This reduces VARIANCE because each tree sees
# a different slice of the data, but every tree makes the SAME kind of
# systematic mistakes as the others (high variance, same bias).
#
# Boosting is the opposite idea: train one tree, look at where it is
# WRONG, train the next tree to correct THAT mistake, repeat. This
# attacks BIAS directly — each new tree is fitted to the residuals
# (the negative gradient of the loss), so the ensemble's systematic
# error shrinks round by round.
#
# The additive-model form:
#     F_0(x) = log-odds of the positive class
#     F_m(x) = F_{m-1}(x) + η · h_m(x)
#
# where h_m is a shallow tree fitted to the pseudo-residuals
#     r_i = y_i - sigmoid(F_{m-1}(x_i))
#
# and η (learning rate) is a step-size that keeps each correction small
# enough that the ensemble does not overshoot. Smaller η → slower but
# more robust convergence. This is the fundamental mechanism behind
# XGBoost, LightGBM, and CatBoost — all three use the same recipe, they
# just differ in how they find the best split at each round.


# ════════════════════════════════════════════════════════════════════════
# TASK 2a — BUILD: AdaBoost warm-up (boosting by RE-WEIGHTING rows)
# ════════════════════════════════════════════════════════════════════════
# AdaBoost (Freund & Schapire, 1997) was the first practical booster.
# Each round it fits a depth-1 tree ("stump") on WEIGHTED rows, then:
#     err_t   = Σ w_i over the rows the stump got wrong
#     α_t     = ½ ln((1 - err_t) / err_t)        (the stump's vote)
#     w_i    ← w_i · exp(+α_t) if wrong, w_i · exp(-α_t) if right; renormalise
# After the update the misclassified rows hold exactly HALF the total
# weight, so the next stump is forced to concentrate on them. Gradient
# boosting (below) generalises this: instead of re-weighting rows, each
# new tree is fit to the gradient of any differentiable loss.

print("\n" + "=" * 70)
print("  AdaBoost Warm-Up on a 1D Logistic Demo")
print("=" * 70)

x_demo, y_demo = make_1d_demo(n=200)
n_demo = len(y_demo)

w = np.full(n_demo, 1.0 / n_demo)
print(f"\n  {'Round':>6} {'weighted err':>13} {'alpha':>8} {'weight on missed rows':>22}")
print("  " + "─" * 54)
for t in range(1, 4):
    stump = DecisionTreeClassifier(max_depth=1, random_state=SEED)
    stump.fit(x_demo, y_demo, sample_weight=w)
    missed = stump.predict(x_demo) != y_demo
    # TODO: weighted error = total weight on the missed rows; the stump's
    # vote alpha = ½ ln((1 - err) / err).
    err = ____
    alpha = ____
    # TODO: multiply missed rows' weights by exp(+alpha), the rest by
    # exp(-alpha), then renormalise so the weights sum to 1.
    # Hint: np.where(missed, alpha, -alpha)
    w = ____
    w = w / w.sum()
    missed_share = float(w[missed].sum())
    print(f"  {t:>6} {err:>13.4f} {alpha:>8.4f} {missed_share:>22.4f}")

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=SEED
)
ada.fit(x_demo, y_demo)
ada_acc = list(ada.staged_score(x_demo, y_demo))
print(
    f"\n  sklearn AdaBoost (50 stumps): training accuracy after 1 stump = "
    f"{ada_acc[0]:.4f}, after {len(ada_acc)} = {ada_acc[-1]:.4f}"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 2b — BUILD a from-scratch gradient booster
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  From-Scratch Gradient Boosting on 1D Logistic Demo")
print("=" * 70)

# Hyperparameters for the demo run
learning_rate = 0.3
n_rounds = 10

# F_0 = log-odds of the positive rate (baseline score)
# TODO: Initialise F_0 as the log-odds of the positive rate, repeated n_demo times.
# Hint: pos_rate = y_demo.mean(); use np.full + np.log(pos_rate / (1 - pos_rate))
pos_rate = ____
F = ____

print(f"\n  Initial F_0 = log-odds = {F[0]:.4f}")
print(f"  Initial sigmoid(F_0) = {1 / (1 + np.exp(-F[0])):.4f}")
print(f"\n  {'Round':>6} {'MSE(resid)':>14} {'Mean|resid|':>14} {'Accuracy':>10}")
print("  " + "─" * 48)

history = []  # collect per-round (round, mse, mean_abs, acc)
for m in range(1, n_rounds + 1):
    # Current probabilities
    # TODO: current probabilities p = sigmoid(F) = 1 / (1 + exp(-F))
    p = ____

    # TODO: pseudo-residuals = negative gradient of log-loss = y_demo - p
    residuals = ____

    # TODO: fit DecisionTreeRegressor(max_depth=3, random_state=SEED) to
    # (x_demo, residuals) and predict h on x_demo.
    tree = ____
    tree.fit(x_demo, residuals)
    h = ____

    # TODO: additive update with the learning rate
    F = ____

    # Metrics
    p_new = 1 / (1 + np.exp(-F))
    preds = (p_new >= 0.5).astype(int)
    acc = float((preds == y_demo).mean())
    mse_resid = float(np.mean(residuals**2))
    mean_abs_resid = float(np.mean(np.abs(residuals)))
    history.append((m, mse_resid, mean_abs_resid, acc))
    print(f"  {m:>6} {mse_resid:>14.6f} {mean_abs_resid:>14.6f} {acc:>10.4f}")

final_acc = history[-1][3]


# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert abs(missed_share - 0.5) < 1e-9, "AdaBoost puts half the weight on missed rows"
assert final_acc > 0.6, "From-scratch boosting should converge above 60% accuracy"
assert history[-1][1] < history[0][1], "MSE of residuals must shrink across rounds"
# INTERPRETATION: Every round, the MSE of the residuals decreases — the
# ensemble is learning the systematic error and subtracting it out. The
# learning rate 0.3 is aggressive; production boosters use 0.01-0.05 and
# compensate with more rounds.
print("\n[ok] Checkpoint 1 passed — from-scratch gradient boosting converged\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — XGBoost Split-Gain Formula
# ════════════════════════════════════════════════════════════════════════
# XGBoost takes the additive model above and asks a sharper question: for
# a given split, how much does the loss decrease? It uses a second-order
# Taylor expansion:
#
#     L ≈ Σ [g_i · f(x_i) + ½ h_i · f(x_i)²] + Ω(f)
#
# where Ω(f) = γ·T + ½ λ Σ_j w_j² penalises a tree f with T leaves and
# leaf weights w_j.
# where g_i = ∂L/∂ŷ (first derivative) and h_i = ∂²L/∂ŷ² (second). For
# log-loss on a binary classification target:
#
#     g_i = p_i - y_i           (predicted minus actual)
#     h_i = p_i · (1 - p_i)     (prediction variance)
#
# Grouping the rows by the leaf j they fall in (G_j = Σ g_i, H_j = Σ h_i
# over that leaf), the loss is a quadratic in each w_j, minimised at
#
#     w_j* = - G_j / (H_j + λ)     with loss  - ½ G_j² / (H_j + λ) + γ
#
# Comparing the loss of one leaf (the parent) against two (the children),
# the split-gain formula falls out algebraically:
#
#     Gain = ½ [ G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ) ] - γ
#
# where G = Σ g_i, H = Σ h_i over each side. λ penalises large leaf
# weights (L2 regularisation on the tree's output), γ is a fixed cost of
# adding a leaf (pruning threshold). If Gain < 0 the split is refused.

print("\n" + "=" * 70)
print("  XGBoost Split-Gain Derivation — Worked Example")
print("=" * 70)

# Numerical example: a node with 100 defaults + 800 non-defaults
p_pred = 0.12  # current ensemble prediction (≈ population default rate)
# TODO: per-sample gradients and Hessian for log-loss.
# g = p - y for each class; h = p · (1 - p)
g_default = ____
g_no_default = ____
h_per_sample = ____

# Candidate split: left = 80 defaults + 50 non-defaults, right = the rest
g_left = 80 * g_default + 50 * g_no_default
h_left = 130 * h_per_sample
g_right = 20 * g_default + 750 * g_no_default
h_right = 770 * h_per_sample

# TODO: use the shared xgb_split_gain helper with lambda_reg=1.0, gamma=0.0
gain = ____
print(f"\n  Split gain (λ=1, γ=0): {gain:.4f}")
print(f"    Left:  G_L={g_left:>8.2f}  H_L={h_left:>8.2f}")
print(f"    Right: G_R={g_right:>8.2f}  H_R={h_right:>8.2f}")

# Sensitivity: how λ and γ change the pruning decision
print("\n  --- Regularisation Effect on Split Gain ---")
print(f"  {'λ':>6} {'γ':>6} {'Gain':>10}  {'Decision':<15}")
print("  " + "─" * 44)
for lam in [0.0, 1.0, 10.0, 100.0]:
    for gam in [0.0, 1.0, 5.0]:
        # TODO: call xgb_split_gain(g_left, h_left, g_right, h_right, lam, gam)
        g = ____
        decision = "accept" if g > 0 else "PRUNE (Gain<0)"
        print(f"  {lam:>6.1f} {gam:>6.1f} {g:>10.4f}  {decision:<15}")


# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert gain > 0, "A well-separated split should have positive gain"
assert (
    xgb_split_gain(g_left, h_left, g_right, h_right, 100.0, 5.0) < gain
), "Heavy regularisation (λ=100, γ=5) should reduce the gain"
# INTERPRETATION: The gain formula is what makes XGBoost different from
# vanilla gradient boosting. λ limits how much any single leaf can swing
# the prediction; γ introduces a fixed cost for every new leaf. Together
# they force the tree to justify every split against the cost of
# complexity — this is structural regularisation, not just early stopping.
print("\n[ok] Checkpoint 2 passed — XGBoost split gain formula verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the residual shrinkage
# ════════════════════════════════════════════════════════════════════════
# The headline proof of boosting is that the mean absolute residual
# decreases every round. We plot it against the round index; a smooth
# curve shows the additive model is absorbing the signal.

fig = go.Figure()
rounds = [h[0] for h in history]
mean_abs = [h[2] for h in history]
acc_series = [h[3] for h in history]

fig.add_trace(
    go.Scatter(
        x=rounds,
        y=mean_abs,
        mode="lines+markers",
        name="Mean |residual|",
        yaxis="y1",
    )
)
fig.add_trace(
    go.Scatter(
        x=rounds,
        y=acc_series,
        mode="lines+markers",
        name="Accuracy",
        yaxis="y2",
    )
)
fig.update_layout(
    title="From-Scratch Gradient Boosting — Residual Shrinkage per Round",
    xaxis_title="Boosting round",
    yaxis=dict(title="Mean |residual|", side="left"),
    yaxis2=dict(title="Accuracy", overlaying="y", side="right", range=[0, 1]),
    legend=dict(x=0.02, y=0.98),
)
viz_path = OUTPUT_DIR / "ex4_01_residual_shrinkage.html"
fig.write_html(viz_path)
print(f"  Saved: {viz_path}")

# Decision-surface plot: probability vs x after final round
p_final = 1 / (1 + np.exp(-F))
order = np.argsort(x_demo.ravel())
surface = go.Figure()
surface.add_trace(
    go.Scatter(
        x=x_demo.ravel()[order],
        y=p_final[order],
        mode="lines",
        name="Booster P(y=1|x)",
    )
)
surface.add_trace(
    go.Scatter(
        x=x_demo.ravel(),
        y=y_demo,
        mode="markers",
        name="True labels",
        marker=dict(size=6, opacity=0.5),
    )
)
surface.update_layout(
    title="Boosted Probability Surface After 10 Rounds",
    xaxis_title="x",
    yaxis_title="P(default)",
)
surface_path = OUTPUT_DIR / "ex4_01_decision_surface.html"
surface.write_html(surface_path)
print(f"  Saved: {surface_path}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: An SME Credit Committee — When To Refuse A Split
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore bank's SME credit committee uses a
# boosted model to pre-rank loan applications. A committee member asks:
# "Young companies are risky — why didn't the model split on
# 'company age ≤ 2 years'?"
#
# The gain formula answers it. Take the same node as above (100 defaults,
# 800 non-defaults, current prediction 0.12) and compare the strong split
# from Task 3 with a split that isolates a SMALL segment of 40 young
# companies, 8 of which defaulted (20% vs 12% overall).

# TODO: G and H for the 40-row young-company leaf (8 defaults, 32 not).
g_young = ____
h_young = ____
g_rest = 92 * g_default + 768 * g_no_default
h_rest = 860 * h_per_sample
gain_young = xgb_split_gain(g_young, h_young, g_rest, h_rest, lambda_reg=1.0, gamma=0.0)

print("\n" + "=" * 70)
print("  Small-segment split vs strong split (λ = 1)")
print("=" * 70)
print(f"  Strong split (Task 3)          gain before γ: {gain:.4f}")
print(f"  'company age ≤ 2y' (40 rows)   gain before γ: {gain_young:.4f}")
for gam in [0.0, 1.0, round(gain_young * 1.5, 2)]:
    verdict = (
        "accept"
        if xgb_split_gain(g_young, h_young, g_rest, h_rest, 1.0, gam) > 0
        else "PRUNE"
    )
    print(f"    with γ = {gam:<6} → young-company split: {verdict}")
print(
    f"\n  Any γ above {gain_young:.4f} refuses the young-company split while the "
    f"strong split (gain {gain:.4f}) survives γ values up to {gain:.1f}."
)

# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert 0 < gain_young < gain, "The small-segment split must earn far less gain"
print("\n[ok] Checkpoint 3 passed — γ separates weak splits from strong ones\n")

# INTERPRETATION: The committee's intuition ("young companies are riskier")
# is correct — the segment's default rate is higher — but with only 40
# rows the loss reduction is small and the leaf weight G/(H+λ) is shrunk
# hard by λ because H is small. γ sets the minimum improvement a split
# must buy. Tuned on held-out data (never on the test set), γ is in effect
# a policy decision: higher γ = refuse to carve out small segments =
# lower training fit but more stable out-of-sample behaviour.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Explained boosting as sequential residual fitting (bias reduction)
  [x] Implemented a from-scratch gradient booster with shallow trees
  [x] Watched MSE of residuals shrink round by round (the proof)
  [x] Derived the XGBoost split-gain formula from the 2nd-order Taylor
      expansion of log-loss
  [x] Interpreted λ and γ as structural regularisers that prevent the
      tree from memorising small, high-variance segments
  [x] Ran AdaBoost by hand: re-weighting puts half the weight on the
      rows the last stump missed
  [x] Connected γ to a credit committee's decision to refuse splits on
      small segments, with the gain computed rather than asserted

  KEY INSIGHT: Boosting is just gradient descent in function space. Every
  round, a new tree is the negative gradient direction in a space of
  shallow trees. The gain formula is what makes XGBoost competitive — it
  lets the tree justify every split against a complexity cost, turning
  "build a big tree" into "build only the splits that pay for themselves".

  Next: 02_xgboost.py screens the Singapore credit data for leakage,
  trains the full XGBoost classifier and ranks its feature importances.
"""
)
