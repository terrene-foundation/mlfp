# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 5.3: Loss Functions — Focal Loss as a Custom Objective
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - The focal loss equation FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)
#   - Why MANY easy examples dominate cross-entropy even though each one
#     contributes very little
#   - How to plug a custom loss into LightGBM (gradient + Hessian)
#   - How to sweep gamma and read the ranking-vs-calibration trade-off
#
# PREREQUISITES: 01_metrics_and_baseline.py (the gamma=0 run must
#                reproduce its baseline — that is how we test our code)
# ESTIMATED TIME: ~35 min
#
# 5-PHASE STRUCTURE:
#   Theory   — focal loss derivation + gamma intuition
#   Build    — focal-loss gradient/Hessian as a LightGBM objective
#   Train    — sweep gamma in [0, 1, 2, 3]
#   Visualise — AUC-PR and Brier vs gamma, loss curves, easy-example share
#   Apply    — illustrative SME early-warning (long-tail hard cases)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import plotly.graph_objects as go
import polars as pl
from dotenv import load_dotenv

from shared.mlfp03.ex_5 import (
    OUTPUT_DIR,
    load_credit_splits,
    load_strategy_proba,
    metrics_row,
    print_metrics_table,
    save_strategy_proba,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Focal Loss and the Gamma Knob
# ════════════════════════════════════════════════════════════════════════
# Cross-entropy (CE) charges -log(p_t), where p_t is the probability the
# model gives to the TRUE class. A confident, correct example (p_t = 0.99)
# costs -log(0.99) = 0.010; an unsure one (p_t = 0.51) costs 0.673 — about
# 67x more. So each easy example is cheap. The problem is how MANY there
# are: in credit data most applicants are easy "obviously repays" cases,
# and thousands of small losses add up to a large share of the total loss
# and of the gradient the booster follows. (You will measure that share
# on this dataset below.)
#
# Focal Loss (Lin et al., 2017, ICCV) adds a modulating factor (1-p_t)^gamma:
#
#     FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
#
# where p_t = p if y=1 else 1-p, and alpha_t = alpha if y=1 else 1-alpha.
# When the model is already confident and correct, (1 - p_t)^gamma is tiny
# and the example nearly vanishes from the loss; hard examples keep most of
# their weight. gamma=0 recovers (alpha-weighted) cross-entropy; gamma=2 is
# the paper's default. alpha is a per-CLASS weight — the same idea as
# scale_pos_weight in 5.2. gamma is the new part: a per-EXAMPLE weight that
# depends on how hard the example currently is. That is something class
# reweighting cannot do.
#
# PLUGGING IT INTO LIGHTGBM: gradient boosting only needs, for every row,
# the gradient g and the curvature (Hessian) h of the loss with respect to
# the model's raw score z (the log-odds, p = sigmoid(z)). LightGBM accepts
# any callable objective(y_true, z) -> (g, h). We write g in closed form
# and let a central finite difference of g give h (clipped positive, since
# focal loss is not convex everywhere and LightGBM divides by h).
#
# With a custom objective LightGBM no longer knows the output is a
# probability, so (1) it starts from z = 0 unless we pass the base-rate
# log-odds as init_score, and (2) predict() returns raw scores — we apply
# the sigmoid (and add the same starting score back) ourselves.

GAMMAS = [0.0, 1.0, 2.0, 3.0]
ALPHA = 0.5  # 0.5 = no class reweighting, so we isolate the gamma effect


# ════════════════════════════════════════════════════════════════════════
# BUILD — focal loss value, gradient and Hessian for LightGBM
# ════════════════════════════════════════════════════════════════════════


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def focal_loss_value(
    y: np.ndarray, z: np.ndarray, gamma: float, alpha: float
) -> np.ndarray:
    """Per-row focal loss for raw scores z."""
    p = np.clip(sigmoid(z), 1e-7, 1 - 1e-7)
    p_t = np.where(y == 1, p, 1 - p)
    alpha_t = np.where(y == 1, alpha, 1 - alpha)
    return -alpha_t * (1 - p_t) ** gamma * np.log(p_t)


def focal_gradient(
    y: np.ndarray, z: np.ndarray, gamma: float, alpha: float
) -> np.ndarray:
    """dFL/dz for both classes (chain rule through p = sigmoid(z))."""
    p = np.clip(sigmoid(z), 1e-7, 1 - 1e-7)
    grad_pos = alpha * (gamma * p * (1 - p) ** gamma * np.log(p) - (1 - p) ** (gamma + 1))
    grad_neg = (1 - alpha) * (p ** (gamma + 1) - gamma * p**gamma * (1 - p) * np.log(1 - p))
    return np.where(y == 1, grad_pos, grad_neg)


def make_focal_objective(gamma: float, alpha: float, eps: float = 1e-4):
    """Return a LightGBM objective(y_true, raw_score) -> (grad, hess)."""

    def objective(y_true: np.ndarray, raw_score: np.ndarray):
        grad = focal_gradient(y_true, raw_score, gamma, alpha)
        grad_up = focal_gradient(y_true, raw_score + eps, gamma, alpha)
        grad_down = focal_gradient(y_true, raw_score - eps, gamma, alpha)
        hess = (grad_up - grad_down) / (2 * eps)
        return grad, np.maximum(hess, 1e-6)

    return objective


# Gradient check: our closed-form gradient must match a numerical
# derivative of the loss itself (the standard test for any custom loss).
z_grid = np.linspace(-4, 4, 41)
grad_errors = []
for label in (0, 1):
    y_grid = np.full_like(z_grid, label)
    numeric = (
        focal_loss_value(y_grid, z_grid + 1e-5, 2.0, 0.25)
        - focal_loss_value(y_grid, z_grid - 1e-5, 2.0, 0.25)
    ) / 2e-5
    grad_errors.append(np.abs(numeric - focal_gradient(y_grid, z_grid, 2.0, 0.25)).max())
max_grad_error = float(max(grad_errors))
print(f"\n  Gradient check (closed form vs numerical): max error = {max_grad_error:.2e}")


# ════════════════════════════════════════════════════════════════════════
# TRAIN — one LightGBM per gamma, same data, same trees
# ════════════════════════════════════════════════════════════════════════

X_train, y_train, X_test, y_test, pos_rate = load_credit_splits()
base_margin = float(np.log(pos_rate / (1 - pos_rate)))  # base-rate log-odds

print("\n" + "=" * 70)
print("  Exercise 5.3 — Focal Loss (custom LightGBM objective)")
print("=" * 70)
print(f"  Starting score (base-rate log-odds): {base_margin:.3f}")
print(f"  alpha = {ALPHA} (no class reweighting), gamma sweep = {GAMMAS}")

metric_rows: list[dict] = []
proba_by_gamma: dict[float, np.ndarray] = {}
for gamma in GAMMAS:
    model = lgb.LGBMClassifier(
        n_estimators=300,
        objective=make_focal_objective(gamma, ALPHA),
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train, init_score=np.full(len(y_train), base_margin))
    raw_score = model.predict(X_test, raw_score=True)
    y_proba = sigmoid(raw_score + base_margin)
    proba_by_gamma[gamma] = y_proba
    save_strategy_proba(f"focal_gamma_{gamma:.1f}", y_proba)
    row = metrics_row(f"gamma={gamma:.1f}", y_test, y_proba)
    row["gamma"] = float(gamma)
    row["mean_p"] = float(y_proba.mean())
    metric_rows.append(row)

baseline_auc = metrics_row("baseline", y_test, load_strategy_proba("baseline"))["auc_roc"]
gamma0_auc = metric_rows[0]["auc_roc"]


# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert max_grad_error < 1e-6, "Closed-form focal gradient disagrees with numerics"
assert len(metric_rows) == len(GAMMAS), "Must sweep every gamma"
assert all(0 <= r["auc_pr"] <= 1 for r in metric_rows), "AUC-PR in [0,1]"
assert all(0 <= r["brier"] <= 1 for r in metric_rows), "Brier in [0,1]"
assert (
    abs(gamma0_auc - baseline_auc) < 0.005
), "gamma=0 focal loss is cross-entropy: it must reproduce the 5.1 baseline"
print(
    f"[ok] Checkpoint 3 — gamma sweep complete; gamma=0 AUC {gamma0_auc:.4f} "
    f"matches the built-in baseline {baseline_auc:.4f}\n"
)


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — gamma sweep, loss curves, where the loss actually comes from
# ════════════════════════════════════════════════════════════════════════

print_metrics_table(metric_rows, "Focal loss gamma sweep (threshold=0.5)")
print(f"\n  {'gamma':>6} {'mean p':>8}   (real default rate {y_test.mean():.3f})")
for r in metric_rows:
    print(f"  {r['gamma']:>6.1f} {r['mean_p']:>8.3f}")

best_by_pr = max(metric_rows, key=lambda r: r["auc_pr"])
best_by_brier = min(metric_rows, key=lambda r: r["brier"])
print("\n  Sensitivity summary:")
print(f"    Highest AUC-PR: gamma={best_by_pr['gamma']:.1f} (AUC-PR={best_by_pr['auc_pr']:.4f})")
print(f"    Lowest Brier:   gamma={best_by_brier['gamma']:.1f} (Brier={best_by_brier['brier']:.4f})")

# Where does the loss come from? Use the gamma=0 (plain CE) probabilities.
p0 = np.clip(proba_by_gamma[0.0], 1e-7, 1 - 1e-7)
p_t = np.where(y_test == 1, p0, 1 - p0)
z0 = np.log(p0 / (1 - p0))
ce_rows = focal_loss_value(y_test, z0, 0.0, 0.5)
fl_rows = focal_loss_value(y_test, z0, 2.0, 0.5)
easy = p_t > 0.8
easy_share_rows = float(easy.mean())
easy_share_ce = float(ce_rows[easy].sum() / ce_rows.sum())
easy_share_fl = float(fl_rows[easy].sum() / fl_rows.sum())
print("\n  Easy examples (p_t > 0.8) in the test set:")
print(f"    share of rows:                 {easy_share_rows:.1%}")
print(f"    share of cross-entropy loss:   {easy_share_ce:.1%}")
print(f"    share of focal loss (gamma=2): {easy_share_fl:.1%}")

pl.DataFrame(metric_rows).write_parquet(OUTPUT_DIR / "focal_sweep_metrics.parquet")
print(f"\n  Saved: {OUTPUT_DIR / 'focal_sweep_metrics.parquet'}")

# ── Visual: AUC-PR and Brier vs gamma ───────────────────────────────────
gammas = [r["gamma"] for r in metric_rows]
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=gammas, y=[r["auc_pr"] for r in metric_rows], mode="lines+markers",
               name="AUC-PR", marker=dict(size=10), line=dict(color="#6366f1", width=3))
)
fig.add_trace(
    go.Scatter(x=gammas, y=[r["brier"] for r in metric_rows], mode="lines+markers",
               name="Brier score", marker=dict(size=10),
               line=dict(color="#f43f5e", width=3), yaxis="y2")
)
fig.update_layout(
    title="Focal Loss Gamma Sweep: ranking (AUC-PR) vs probability quality (Brier)",
    xaxis_title="gamma (0 = cross-entropy)",
    yaxis=dict(title="AUC-PR (higher = better ranking)"),
    yaxis2=dict(title="Brier (lower = better)", overlaying="y", side="right"),
    height=450,
    legend=dict(orientation="h", y=-0.2),
)
viz_path = OUTPUT_DIR / "ex5_03_focal_gamma_sweep.html"
fig.write_html(str(viz_path))
print(f"  Saved: {viz_path}")

# ── Visual: focal loss curve for different gamma values ─────────────────
p_grid = np.linspace(0.01, 0.99, 200)
fig2 = go.Figure()
for gamma in [0, 0.5, 1, 2, 5]:
    fl = -((1 - p_grid) ** gamma) * np.log(p_grid)
    fig2.add_trace(go.Scatter(x=p_grid, y=fl, mode="lines", name=f"gamma={gamma}", line=dict(width=2)))
fig2.update_layout(
    title="Focal Loss Curve: FL(p_t) = -(1-p_t)^gamma * log(p_t)",
    xaxis_title="p_t (model confidence for the correct class)",
    yaxis_title="Loss per example",
    height=450,
    legend=dict(orientation="h", y=-0.2),
)
viz_path2 = OUTPUT_DIR / "ex5_03_focal_loss_curves.html"
fig2.write_html(str(viz_path2))
print(f"  Saved: {viz_path2}")

# ── Visual: share of rows vs share of loss from easy examples ───────────
fig3 = go.Figure(
    go.Bar(
        x=["share of rows", "share of CE loss", "share of focal loss (gamma=2)"],
        y=[easy_share_rows, easy_share_ce, easy_share_fl],
        marker_color=["#9ca3af", "#6366f1", "#10b981"],
        text=[f"{v:.0%}" for v in (easy_share_rows, easy_share_ce, easy_share_fl)],
        textposition="outside",
    )
)
fig3.update_layout(title="Easy examples (p_t > 0.8): how much of the loss do they carry?",
                   yaxis_tickformat=".0%", height=420)
viz_path3 = OUTPUT_DIR / "ex5_03_easy_example_share.html"
fig3.write_html(str(viz_path3))
print(f"  Saved: {viz_path3}")

# INTERPRETATION: Read the bar chart first: focal loss shrinks the share
# of the loss carried by easy examples — that is the mechanism. Then read
# the sweep: whether that buys better RANKING (AUC-PR) on this data is an
# empirical question, and the mean-p column shows the side effect —
# focal loss is not a proper scoring rule, so its probabilities drift away
# from the real default rate as gamma grows (check the Brier column).


# ════════════════════════════════════════════════════════════════════════
# APPLY — SME early-warning (illustrative)
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore bank's SME portfolio has ~18,000
# business loans. The risk team wants an early-warning model that flags
# borrowers likely to default in the next 90 days.
#
# The imbalance is severe AND heterogeneous:
#   - Most defaults come from obvious distress signals (missed payroll,
#     rent arrears, bounced cheques) which any model classifies easily.
#   - The HARD cases are SMEs that look healthy until the final month —
#     the ones an early-warning system exists to catch.
#
# Why focal loss is worth TRYING here: it moves the training signal away
# from the many easy cases toward the borderline ones. Whether that
# improves ranking is data-dependent — that is why you sweep gamma and
# keep the result only if validation AUC-PR improves. Because focal loss
# distorts probabilities, a bank that adopted it would still recalibrate
# (5.5) before using the scores for provisioning or pricing.
#
# Value (illustrative assumption): if early notice lets the bank
# restructure a facility and avoid ~S$30,000 of loss per caught borrower,
# every extra default ranked into the review queue is worth that much.

VALUE_PER_EARLY_CATCH_SGD = 30_000  # illustrative assumption
ce_row = metric_rows[0]
print("\n  SME early-warning implication (computed from the sweep):")
print(f"    AUC-PR  gamma=0 (CE): {ce_row['auc_pr']:.4f}   best gamma={best_by_pr['gamma']:.1f}: {best_by_pr['auc_pr']:.4f}")
print(f"    Brier   gamma=0 (CE): {ce_row['brier']:.4f}   at that gamma:    {best_by_pr['brier']:.4f}")
print(f"    Recall @0.5 at the best-AUC-PR gamma: {best_by_pr['recall']:.2%}")
if best_by_pr["gamma"] == 0.0:
    print("    -> On this data focal loss did NOT beat plain cross-entropy on AUC-PR.")
else:
    print("    -> Focal loss improved ranking here; recalibrate before pricing.")
print(f"    Each extra early catch is worth ~S${VALUE_PER_EARLY_CATCH_SGD:,} (illustrative)")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — 5.3")
print("=" * 70)
print(
    """
  [x] Derived focal loss FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)
  [x] Measured how much of the cross-entropy loss easy examples carry
  [x] Wrote focal loss as a LightGBM custom objective (gradient + Hessian)
      and verified it: gradient check + gamma=0 reproduces the baseline
  [x] Swept gamma and read ranking (AUC-PR) against calibration (Brier)

  KEY INSIGHT: alpha re-weights CLASSES; gamma re-weights EXAMPLES by
  how hard they are. Both change what the model optimises, so both
  distort probabilities — keep a loss change only if it improves the
  metric you care about, and recalibrate afterwards.

  Next: 04_threshold_optimisation.py — instead of tuning the loss, tune
  the DECISION threshold from the business cost matrix directly.
"""
)
