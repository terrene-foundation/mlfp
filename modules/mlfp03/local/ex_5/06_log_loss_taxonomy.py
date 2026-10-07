# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 5.6: Log Loss — The Metric Behind Every Probabilistic
#                        Classifier
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Derive log loss from maximum likelihood (why "cross-entropy" and
#     "log loss" are the same formula with different names)
#   - Compute log loss by hand and with sklearn.metrics.log_loss
#   - See why a confident wrong answer costs MORE than a hedged wrong answer
#   - Compare log loss with Brier score and AUC on the same model — they
#     rank strategies DIFFERENTLY
#   - Place log loss in the metrics taxonomy: it is the only proper
#     scoring rule here that is also a TRAINING objective
#   - Apply to default-probability sign-off at a Singapore retail bank
#
# PREREQUISITES: 01_metrics_and_baseline.py (metrics taxonomy, baseline
#   probabilities saved under outputs/ex5_imbalance/strategy_probabilities.parquet)
# ESTIMATED TIME: ~30 min
#
# 5-PHASE STRUCTURE:
#   Theory   — derive log loss from maximum likelihood
#   Build    — compute log loss by hand on a tiny example
#   Train    — score the LightGBM strategies with log_loss
#   Visualise — per-row loss curve + strategy comparison
#   Apply    — probability sign-off for a bank's risk committee
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from shared.mlfp03.ex_5 import (
    OUTPUT_DIR,
    list_saved_strategies,
    load_credit_splits,
    load_strategy_proba,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — From Maximum Likelihood to a Number You Can Plot
# ════════════════════════════════════════════════════════════════════════
# Suppose you flip a biased coin that lands heads with probability p. After
# N flips you observe k heads. The LIKELIHOOD of p given the data is
#
#     L(p) = p^k · (1 − p)^(N − k)
#
# Maximum likelihood picks the p that maximises L(p). For a Bernoulli coin
# the answer is p* = k / N — the empirical frequency.
#
# Now replace the coin with a classifier that outputs p_i = P(default = 1)
# for each loan applicant. The likelihood of the OBSERVED outcomes under
# the model is
#
#     L = Π_i p_i^(y_i) · (1 − p_i)^(1 − y_i)
#
# Take the negative log (so products become sums) and divide by N to get
# an average per-applicant loss:
#
#     LOG LOSS = −(1/N) Σ_i [ y_i · log p_i + (1 − y_i) · log(1 − p_i) ]
#
# This is also called CROSS-ENTROPY (from information theory) or NEGATIVE
# LOG LIKELIHOOD (from statistics) — three names, one formula.
#
# WHY LOG LOSS IS SPECIAL IN THE TAXONOMY:
#   - It is the ONLY metric here that is also a TRAINING objective. When
#     LightGBM minimises binary_logloss it is directly minimising this
#     number. Accuracy, precision, recall, AUC are never what the optimiser
#     actually sees.
#   - It is a PROPER SCORING RULE: the unique minimiser is the true
#     probability. A model that outputs p=0.13 when the real rate is 13%
#     cannot be beaten on log loss by any other constant.
#   - It punishes OVERCONFIDENT mistakes far more than Brier score.
#     Confident wrong answer (p=0.99 on a non-default): log loss = 4.605.
#     Hedged wrong answer (p=0.60 on a non-default): log loss = 0.916.
#     Brier score on the same pair: 0.980 vs 0.360 — only a 2.7× gap,
#     where log loss shows a 5.0× gap. If your business penalty grows
#     super-linearly with overconfidence (risk-based pricing), log loss
#     is the metric that matches.
#
# WHEN LOG LOSS IS THE WRONG METRIC: when the decision is a yes/no, not a
# probability (deny vs approve at a fixed threshold). Log loss evaluates
# the probability VECTOR; threshold-based metrics evaluate the DECISION.
# A model can have the best log loss and still lose money if the operating
# threshold is wrong — see 04_threshold_optimisation.py.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: compute log loss by hand on a tiny example
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Exercise 5.6 — Log Loss in the Metrics Taxonomy")
print("=" * 70)

# A tiny 4-applicant example you can verify on paper.
y_tiny = np.array([0, 1, 0, 1])
p_tiny = np.array([0.10, 0.80, 0.60, 0.55])

# Hand computation: clip to avoid log(0), then apply the formula.
eps = 1e-15
p_clipped = np.clip(p_tiny, eps, 1 - eps)
# TODO: −mean( y·log p + (1−y)·log(1−p) ) over the tiny example
# Hint: np.mean of (y_tiny * np.log(p_clipped) + (1 - y_tiny) * np.log(1 - p_clipped))
hand_loss = ____
# TODO: the same number from sklearn.metrics
sklearn_loss = ____
print(f"  Tiny example y={y_tiny.tolist()}  p={p_tiny.tolist()}")
print(f"  Hand-computed log loss: {hand_loss:.6f}")
print(f"  sklearn log_loss:       {sklearn_loss:.6f}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert abs(hand_loss - sklearn_loss) < 1e-9, "Hand formula must match sklearn"
print("[ok] Checkpoint 1 — hand formula matches sklearn\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: score every saved strategy with log_loss
# ════════════════════════════════════════════════════════════════════════
# Exercise 5.1–5.5 saved per-strategy probability vectors to
# outputs/ex5_imbalance/strategy_probabilities.parquet. We score each one
# with three metrics — log_loss, Brier, AUC — and show they DO NOT always
# agree on the ranking.

X_train, y_train, X_test, y_test, pos_rate = load_credit_splits()
saved = list_saved_strategies()
if "baseline" not in saved:
    raise RuntimeError(
        "Run 01_metrics_and_baseline.py first — no saved strategies found."
    )

print(f"\n  Saved strategies: {saved}")
print(f"  Real default rate on test: {pos_rate:.3f}")

rows: list[dict] = []
for name in saved:
    p = load_strategy_proba(name)
    # TODO: score each strategy with all three metrics
    # Hint: log_loss(y_test, p), brier_score_loss(y_test, p),
    #       roc_auc_score(y_test, p) — wrap each in float(...)
    rows.append(
        {
            "strategy": name,
            "log_loss": ____,
            "brier": ____,
            "auc_roc": ____,
            "mean_p": float(p.mean()),
        }
    )

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert len(rows) >= 2, "Need at least 2 strategies for a meaningful comparison"
assert all(0 <= r["log_loss"] for r in rows), "Log loss is non-negative"
assert all(0 <= r["brier"] <= 1 for r in rows), "Brier in [0,1]"
print("[ok] Checkpoint 2 — log loss computed for every strategy\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: per-row loss curve + ranking disagreement
# ════════════════════════════════════════════════════════════════════════

# Per-row log-loss curve: for an actual defaulter (y=1), loss = −log(p).
# For a non-defaulter (y=0), loss = −log(1 − p). Plot both curves.
p_grid = np.linspace(0.001, 0.999, 400)
# TODO: per-row loss when the true label is 1
loss_when_y1 = ____
# TODO: per-row loss when the true label is 0
loss_when_y0 = ____

fig_curve = go.Figure()
fig_curve.add_trace(
    go.Scatter(x=p_grid, y=loss_when_y1, name="actual defaulter (y=1)", line=dict(color="#dc2626"))
)
fig_curve.add_trace(
    go.Scatter(x=p_grid, y=loss_when_y0, name="actual repayer (y=0)", line=dict(color="#2563eb"))
)
fig_curve.update_layout(
    title="Per-row log loss: confident mistakes are punished exponentially",
    xaxis_title="Predicted P(default)",
    yaxis_title="Log loss contribution (nats)",
    yaxis_range=[0, 5],
    height=420,
)
curve_path = OUTPUT_DIR / "ex5_06_logloss_curve.html"
fig_curve.write_html(str(curve_path))
print(f"  Saved: {curve_path}")

# Strategy comparison table — note how the ranking can disagree with AUC.
print(f"\n  {'Strategy':<28} {'log_loss':>10} {'brier':>8} {'AUC':>8} {'mean p':>8}")
print("  " + "─" * 68)
for r in sorted(rows, key=lambda r: r["log_loss"]):
    print(
        f"  {r['strategy']:<28} {r['log_loss']:>10.4f} {r['brier']:>8.4f} "
        f"{r['auc_roc']:>8.4f} {r['mean_p']:>8.3f}"
    )

fig_compare = go.Figure()
strategies = [r["strategy"] for r in rows]
fig_compare.add_trace(go.Bar(x=strategies, y=[r["log_loss"] for r in rows], name="log loss"))
fig_compare.add_trace(go.Bar(x=strategies, y=[r["brier"] for r in rows], name="Brier"))
fig_compare.update_layout(
    title="Log loss vs Brier across imbalance strategies",
    barmode="group",
    yaxis_title="Loss (lower is better)",
    height=420,
)
compare_path = OUTPUT_DIR / "ex5_06_logloss_vs_brier.html"
fig_compare.write_html(str(compare_path))
print(f"  Saved: {compare_path}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
# TODO: strategy name with the lowest log_loss (min over rows)
best_logloss_strategy = ____
# TODO: strategy name with the highest auc_roc (max over rows)
best_auc_strategy = ____
print(f"\n  Best by log loss: {best_logloss_strategy}")
print(f"  Best by AUC-ROC:  {best_auc_strategy}")
# INTERPRETATION: log loss rewards CALIBRATED probabilities; AUC rewards
# RANKING. A class-weighted model can win on AUC but lose on log loss
# because its probabilities are systematically inflated. After calibration
# (05_calibration.py), the same model often wins on BOTH.
print("[ok] Checkpoint 3 — rankings compared\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: probability sign-off at a Singapore retail bank
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore retail bank's risk committee must
# sign off on the default-probability model before it feeds three systems:
#
#   1. RISK-BASED PRICING — the interest rate is a deterministic function
#      of p(default). A 2x miscalibration doubles the quoted expected-loss
#      spread, which either over-prices good customers (they walk) or
#      under-prices bad ones (the bank eats the loss).
#
#   2. IFRS 9 EXPECTED CREDIT LOSS — the provision on the balance sheet
#      is Σ p · LGD · EAD across the portfolio. The auditor asks: "show me
#      a metric that would have caught a 5% probability error." AUC cannot;
#      log loss can.
#
#   3. STRESS TESTING — the 99th-percentile loss under a stressed default
#      rate is computed by sampling from p. If p is overconfident, the tail
#      estimate is too thin and the bank under-reserves.
#
# In all three systems the CONSUMER of the model is a probability, not a
# yes/no. That is exactly when log loss belongs on the scorecard next to
# the usual accuracy metrics.
#
# ILLUSTRATIVE ARITHMETIC (round numbers, not the bank's figures): if the
# portfolio carries S$2B of unsecured personal loans at a 1.4% expected
# loss rate, a 10% relative miscalibration in p shifts the provision by
# S$2B × 1.4% × 0.10 ≈ S$2.8M. The risk committee wants a metric that
# sees that error BEFORE the auditor does — log loss does, accuracy does
# not.
#
# LIMITATIONS:
#   - Log loss is unbounded: one confident mistake (p=0.999 on a defaulter
#     who repays) contributes 6.9 nats and can dominate the average on a
#     small test set. Report the 95th percentile of the per-row loss, not
#     just the mean.
#   - Log loss says nothing about WHICH threshold to operate at — that
#     decision belongs to 04_threshold_optimisation.py.
#   - For very rare events (p << 0.01), log loss and Brier score give
#     nearly identical rankings; the choice between them is about
#     communication, not correctness.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Derived log loss from maximum likelihood
  [x] Verified the hand formula against sklearn.metrics.log_loss
  [x] Scored every imbalance strategy with log loss
  [x] Saw that log loss and AUC can rank the same models differently
  [x] Placed log loss in the taxonomy: the only proper scoring rule here
      that is also a TRAINING objective

  KEY INSIGHT: when the downstream consumer is a probability (pricing,
  ECL, stress testing), evaluate with a metric that punishes overconfident
  mistakes super-linearly. That metric is log loss.

  Next: 07_regression_metrics.py — the regression side of the taxonomy
  (R², MAE, RMSE, MAPE) and why "percent error" is rarely what you want.
"""
)
