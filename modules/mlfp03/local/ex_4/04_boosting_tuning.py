# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 4.4: Boosting Tuning — Sweeps, Heatmaps, Early Stopping
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Sweep learning rate and see how AUC-PR changes across η
#   - Build a learning_rate × max_depth heatmap and read the interaction
#   - Use early stopping to pick n_estimators automatically instead of
#     guessing — monitored on a VALIDATION split, never the test set
#   - Explain why "grid search over independent dials" is the wrong
#     mental model for boosting (Exercise 7 will replace it with Bayesian)
#   - Produce a final production-ready configuration
#
# PREREQUISITES: Exercise 4.2/4.3 (XGBoost and LightGBM/CatBoost).
#
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — hyperparameter interaction, why grids lie
#   2. Build — a learning-rate sweep and a 2-D depth×η heatmap
#   3. Train — sweeps + early stopping on validation; test the winner once
#   4. Visualise — heatmap + LR sweep + early-stopping comparison
#   5. Apply — a digital lender tunes for a business metric
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from shared.mlfp03.ex_4 import (
    OUTPUT_DIR,
    SEED,
    evaluate_classifier,
    make_lightgbm,
    make_xgboost,
    prepare_credit_split,
    print_metrics,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Hyperparameter Interaction Is Why Grid Search Lies
# ════════════════════════════════════════════════════════════════════════
# The naive approach to tuning boosting is "vary one knob at a time":
# hold max_depth fixed at 6, sweep learning_rate, pick the best, then
# hold learning_rate fixed and sweep max_depth. This is wrong because
# the knobs INTERACT.
#
# Example of interaction:
#   - At learning_rate=0.01, max_depth=10 is fine — the small step size
#     keeps the deep trees from overfitting.
#   - At learning_rate=0.2, max_depth=10 overfits badly — each big step
#     drives a deep tree far past the optimal loss.
#
# The right mental model is a 2-D surface over (η, depth). A 1-D sweep
# only sees a slice of that surface and can lead you to a local optimum
# that doesn't hold once the other knob moves.
#
# Two practical tools replace naive grid search:
#
#   1. Early stopping: set n_estimators to a large budget (2000+), let
#      the loss on a VALIDATION split (carved from the training data)
#      tell you when to stop. Using the test set here would leak it. This turns n_estimators
#      from a tuning knob into a self-tuning parameter.
#
#   2. 2-D heatmap of (learning_rate × max_depth). Small grids (3x4)
#      are enough to SEE the interaction surface. Beyond 2 dimensions,
#      grids explode combinatorially — Exercise 7 uses Bayesian
#      optimisation to scale this up.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the sweep grids
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Boosting Hyperparameter Tuning on Singapore Credit")
print("=" * 70)

data = prepare_credit_split()
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

# Every tuning decision below (learning rate, depth, number of rounds) is
# made on a VALIDATION split carved out of the training data. The test
# set is touched exactly once, at the very end, to report the chosen
# configuration. Tuning on the test set would make the final number an
# optimistic, selection-biased estimate.
# TODO: split the TRAINING data 80/20 into fit / validation, stratified on
# the target. Hint: train_test_split(X_train, y_train, test_size=0.2,
#       stratify=y_train, random_state=SEED)
X_fit, X_val, y_fit, y_val = ____

print(f"\n  Fit: {X_fit.shape} | Validation: {X_val.shape} | Test (held back): {X_test.shape}")
print(f"  Default rate: {data['default_rate']:.2%}")

learning_rates = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5]
depths_sweep = [3, 5, 6, 8, 10]
lr_sweep_for_heatmap = [0.01, 0.05, 0.1, 0.2]


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN the sweeps (all scored on the validation split)
# ════════════════════════════════════════════════════════════════════════

# --- 3a. 1-D learning-rate sweep (XGBoost, depth fixed at 6) ------------
print("\n  --- Learning-rate sweep (XGBoost, depth=6, 500 rounds; validation) ---")
print(f"  {'lr':>6}  {'AUC-ROC':>10}  {'AUC-PR':>10}")
print("  " + "─" * 32)

lr_curve: list[tuple[float, float, float]] = []  # (lr, auc_roc, auc_pr)
for lr in learning_rates:
    # TODO: make_xgboost(n_estimators=500, learning_rate=lr, max_depth=6),
    # fit on (X_fit, y_fit) with verbose=False, then evaluate_classifier on
    # the VALIDATION split's class-1 probabilities.
    m = ____
    m.fit(X_fit, y_fit, verbose=False)
    val_metrics = ____
    lr_curve.append((lr, val_metrics["auc_roc"], val_metrics["auc_pr"]))
    print(f"  {lr:>6.2f}  {val_metrics['auc_roc']:>10.4f}  {val_metrics['auc_pr']:>10.4f}")


# --- 3b. 2-D learning_rate × max_depth heatmap --------------------------
print("\n  --- Heatmap: validation AUC-PR, learning_rate × max_depth (300 rounds) ---")
print(f"  {'':>8}", end="")
for d in depths_sweep:
    print(f"  d={d:<3}   ", end="")
print()
print("  " + "─" * (10 + 10 * len(depths_sweep)))

heatmap = np.zeros((len(lr_sweep_for_heatmap), len(depths_sweep)))
for i, lr in enumerate(lr_sweep_for_heatmap):
    print(f"  lr={lr:<5}", end="")
    for j, d in enumerate(depths_sweep):
        # TODO: 300-round XGBoost at (lr, d); score AUC-PR on validation.
        # Hint: average_precision_score(y_val, m.predict_proba(X_val)[:, 1])
        m = ____
        m.fit(X_fit, y_fit, verbose=False)
        auc_pr = ____
        heatmap[i, j] = auc_pr
        print(f"  {auc_pr:>8.4f}", end="")
    print()

best_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
best_lr = lr_sweep_for_heatmap[best_idx[0]]
best_depth = depths_sweep[best_idx[1]]
print(
    f"\n  Best combo on validation: lr={best_lr}, depth={best_depth} "
    f"(AUC-PR={heatmap[best_idx]:.4f}); worst cell {heatmap.min():.4f}"
)


# --- 3c. Early stopping (2000-round budget, validation decides) ---------
print("\n  --- Early stopping (2000-round budget, η=0.05, depth=6) ---")
es_model = make_xgboost(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    early_stopping_rounds=50,
)
# TODO: fit es_model on (X_fit, y_fit), monitoring the VALIDATION split.
# Hint: eval_set=[(X_val, y_val)], verbose=False — never the test set.
____
best_iter_xgb = int(es_model.best_iteration)
es_metrics = evaluate_classifier(y_val, es_model.predict_proba(X_val)[:, 1])
print(f"  XGBoost: best_iteration = {best_iter_xgb}/2000")
print_metrics("XGB+ES (val)", es_metrics)

# Same idea, LightGBM API — plus a FIXED 500-round LightGBM at the same
# learning rate so the chart compares like with like.
lgb_es = make_lightgbm(n_estimators=2000, learning_rate=0.05, max_depth=6)
# TODO: same idea in LightGBM's API: eval_set on validation plus
# callbacks=[lgb.early_stopping(50, verbose=False)].
____
best_iter_lgb = int(lgb_es.best_iteration_)
lgb_es_metrics = evaluate_classifier(y_val, lgb_es.predict_proba(X_val)[:, 1])
print(f"  LightGBM: best_iteration = {best_iter_lgb}/2000")
print_metrics("LGB+ES (val)", lgb_es_metrics)

lgb_fixed = make_lightgbm(n_estimators=500, learning_rate=0.05, max_depth=6)
lgb_fixed.fit(X_fit, y_fit)
lgb_fixed_metrics = evaluate_classifier(y_val, lgb_fixed.predict_proba(X_val)[:, 1])
print_metrics("LGB fixed 500 (val)", lgb_fixed_metrics)
xgb_fixed_auc_pr = lr_curve[learning_rates.index(0.05)][2]  # 500 rounds, η=0.05

# --- 3d. Final configuration → ONE look at the test set ----------------
# TODO: XGBoost with the heatmap's best_lr / best_depth, a 2000-round
# budget and early_stopping_rounds=50.
final_model = ____
final_model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
final_test_metrics = evaluate_classifier(y_test, final_model.predict_proba(X_test)[:, 1])
print(
    f"\n  Final config: lr={best_lr}, depth={best_depth}, "
    f"rounds={int(final_model.best_iteration) + 1} (early-stopped on validation)"
)
print_metrics("FINAL on TEST", final_test_metrics)


# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert len(lr_curve) == len(learning_rates), "Every learning rate must be evaluated"
assert heatmap.shape == (len(lr_sweep_for_heatmap), len(depths_sweep))
assert heatmap.max() > 0, "At least one heatmap cell must have positive AUC-PR"
assert best_iter_xgb < 2000, "XGBoost early stopping must fire before the budget"
assert best_iter_lgb < 2000, "LightGBM early stopping must fire before the budget"
assert (
    final_test_metrics["auc_pr"] > 2 * data["default_rate"]
), "The tuned model's test AUC-PR should be at least twice the random baseline"
# INTERPRETATION: Early stopping fires well before 2000 rounds in almost
# every production setting. That's the signal the 2000-round budget was
# safely high enough — if best_iteration hits 2000, raise the budget.
print("\n[ok] Checkpoint 1 passed — sweeps + heatmap + early stopping all ran\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the tuning surface
# ════════════════════════════════════════════════════════════════════════

# 4a. LR curve
lr_fig = go.Figure()
lrs = [row[0] for row in lr_curve]
auc_prs = [row[2] for row in lr_curve]
lr_fig.add_trace(
    go.Scatter(x=lrs, y=auc_prs, mode="lines+markers", name="XGBoost AUC-PR")
)
lr_fig.update_layout(
    title="Learning-Rate Sensitivity — XGBoost (validation AUC-PR)",
    xaxis_title="learning_rate (η)",
    yaxis_title="AUC-PR",
    xaxis=dict(type="log"),
)
lr_path = OUTPUT_DIR / "ex4_04_lr_curve.html"
lr_fig.write_html(lr_path)
print(f"  Saved: {lr_path}")

# 4b. Heatmap
heat_fig = go.Figure(
    data=go.Heatmap(
        z=heatmap,
        x=[f"d={d}" for d in depths_sweep],
        y=[f"η={lr}" for lr in lr_sweep_for_heatmap],
        colorscale="Viridis",
        colorbar=dict(title="AUC-PR"),
    )
)
heat_fig.update_layout(
    title="Hyperparameter Interaction Heatmap — XGBoost (300 rounds, validation)",
    xaxis_title="max_depth",
    yaxis_title="learning_rate",
)
heat_path = OUTPUT_DIR / "ex4_04_heatmap.html"
heat_fig.write_html(heat_path)
print(f"  Saved: {heat_path}")

# 4c. Early-stopping comparison (fixed 500 vs early stop)
es_fig = go.Figure()
es_fig.add_trace(
    go.Bar(
        name="Fixed 500 rounds (η=0.05)",
        x=["XGBoost", "LightGBM"],
        y=[xgb_fixed_auc_pr, lgb_fixed_metrics["auc_pr"]],
    )
)
es_fig.add_trace(
    go.Bar(
        name="Early stopping (2000 budget)",
        x=["XGBoost", "LightGBM"],
        y=[es_metrics["auc_pr"], lgb_es_metrics["auc_pr"]],
    )
)
es_fig.update_layout(
    title="Early Stopping vs Fixed 500 Rounds — validation AUC-PR (η=0.05)",
    barmode="group",
    yaxis_title="AUC-PR",
)
es_path = OUTPUT_DIR / "ex4_04_early_stopping.html"
es_fig.write_html(es_path)
print(f"  Saved: {es_path}")


# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert es_metrics["auc_pr"] > 0, "Early-stopped XGBoost should have positive AUC-PR"
print(
    f"  Early stopping vs fixed 500 rounds (validation AUC-PR): "
    f"XGBoost {es_metrics['auc_pr']:.4f} vs {xgb_fixed_auc_pr:.4f} "
    f"({best_iter_xgb + 1} rounds used); LightGBM "
    f"{lgb_es_metrics['auc_pr']:.4f} vs {lgb_fixed_metrics['auc_pr']:.4f} "
    f"({best_iter_lgb} rounds used)"
)
# INTERPRETATION: The three figures give you three reads on the tuning
# surface. The LR curve is 1-D (easy to reason about but misleading).
# The heatmap is 2-D (hyperparameter interaction visible). Early stopping
# is the operational pattern that replaces guessing n_estimators. In
# production, you combine them: tune (η, depth) on a 2-D heatmap with
# early stopping driving n_estimators.
print("\n[ok] Checkpoint 2 passed — all tuning visualisations saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: A Digital Lender Tunes For A Business Metric
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a regional digital lender pre-approves app
# users for small credit lines. The model must
#   1. rank users by default risk so the top-X% get the offer,
#   2. produce probabilities good enough to price the offer from expected
#      loss (Lesson 3.5 calibrates them), and
#   3. re-train weekly as new repayment data arrives.
#
# The tuning objective is therefore NOT "maximise AUC-PR" but "maximise
# expected net revenue of the approved pool, subject to a ceiling on its
# default rate". The sweeps above tell the team where the cheap wins are:

flat_band = [row[2] for row in lr_curve if 0.03 <= row[0] <= 0.1]
print("\n  What the sweeps say (validation):")
print(
    f"    AUC-PR across η ∈ [0.03, 0.1]: {min(flat_band):.4f} - {max(flat_band):.4f} "
    f"(spread {max(flat_band) - min(flat_band):.4f})"
)
print(
    f"    AUC-PR at η=0.5: {lr_curve[-1][2]:.4f} — large steps overshoot"
)
print(
    f"    Heatmap optimum: depth={best_depth}, η={best_lr}; spread across all "
    f"cells {heatmap.max() - heatmap.min():.4f}"
)
print(f"    Early stopping chose {best_iter_xgb + 1} rounds at η=0.05")
# INTERPRETATION: when AUC-PR barely moves across a band of settings, the
# team is free to choose within that band on OTHER grounds — e.g. a lower
# learning rate and more rounds for smoother probabilities, or a stronger
# reg_lambda (λ) to shrink leaf weights — and to pick between candidates
# with the business metric itself. Exercise 7 formalises this with
# Bayesian optimisation over a chosen objective.


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — km.diagnose
# ════════════════════════════════════════════════════════════════════════
# This lesson tuned XGBoost by hand — learning-rate and depth sweeps,
# early stopping, one final look at the test set. The kailash-ml SDK packages the
# diagnostic surface (per-class metrics, class-balance severity,
# confusion matrix, accuracy heuristics) into a single call.
#
# Destination-first: when the journey is internalised, the SDK is one line.

from kailash_ml import diagnose

# `kind="classical_classifier"` dispatches to the sklearn ClassifierMixin
# adapter. XGBClassifier implements the ClassifierMixin interface.
report = diagnose(
    final_model, kind="classical_classifier", data=(X_test, y_test), show=False
)
print()
print(f"  km.diagnose model    : XGBoost final config (lr={best_lr}, depth={best_depth})")
print(f"  km.diagnose metrics  : {report.metrics}")
print(f"  km.diagnose severity : {report.severity}")
print()
print("km.diagnose: 1 call -> the same diagnostic surface the lesson body")
print("hand-rolled for the final configuration. Destination-first:")
print("when the journey is internalised, the SDK is one line.")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Made every tuning decision on a validation split carved from the
      training data — the test set was used once, for the final report
  [x] Swept learning_rate and saw AUC-PR change across 6 values
  [x] Built a 2-D learning_rate × max_depth heatmap and read the
      interaction surface (best: lr={best_lr}, depth={best_depth})
  [x] Used early stopping to pick n_estimators automatically
      (XGBoost best_iter={best_iter_xgb}, LightGBM best_iter={best_iter_lgb})
  [x] Final test AUC-PR {final_test_metrics['auc_pr']:.4f} vs random
      baseline {data['default_rate']:.4f}
  [x] Explained why grid search over independent dials is the wrong
      mental model and why Exercise 7 will use Bayesian optimisation
  [x] Mapped tuning knobs to a business metric using the lender scenario

  KEY INSIGHT: Hyperparameters interact. Never tune one at a time.
  Always use early stopping for n_estimators. And always tune against
  the business objective — AUC-PR is a proxy, not a destination.

  NEXT: Exercise 5 (class imbalance and calibration). The same ~13%
  default rate, now attacked head-on: metrics that respect imbalance,
  SMOTE vs cost-sensitive learning, focal loss, cost-based thresholds
  and probability calibration.
"""
)
