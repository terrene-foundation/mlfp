# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 2.1: Bias-Variance Trade-off
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Diagnose underfitting vs overfitting from train/test error gaps
#   - Decompose expected test error into Bias², Variance, and irreducible
#     noise by refitting on many independent training sets
#   - Read a bias-variance curve and pick the "sweet spot" complexity
#   - Connect the bias-variance picture to Singapore credit-risk decisions
#
# PREREQUISITES:
#   - MLFP03 Exercise 1 (feature engineering, sklearn basics)
#   - MLFP02 Module 2 (linear regression, expectation & variance)
#
# ESTIMATED TIME: ~35 minutes
#
# TASKS (5-phase R10):
#   1. Theory — why "more complex" is not always "better"
#   2. Build — polynomial pipelines at increasing degrees
#   3. Train — fit each degree, collect train/test MSE
#   4. Visualise — fitted curves + bias² / variance / noise curve
#   5. Apply — a Singapore retail bank's credit-scorecard choice
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error

from shared.mlfp03.ex_2 import (
    SEED,
    make_poly_pipeline,
    make_sine_dataset,
    print_header,
    sample_sine_training_set,
    save_html_plot,
    sine_truth,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — The Bias-Variance Decomposition
# ════════════════════════════════════════════════════════════════════════
# Every supervised prediction error decomposes into three additive terms:
#
#   E[(y - ŷ)²]  =  Bias²(ŷ)  +  Var(ŷ)  +  σ²
#                   ─────────    ──────     ───
#                   How wrong    How much   Irreducible
#                   the average  the model  noise in y
#                   prediction   wiggles    (we can't
#                   is           between    do better
#                                datasets   than this)
#
# The expectation is over TRAINING SETS: imagine collecting a fresh
# training set many times, refitting each time, and looking at the cloud
# of fitted curves. Bias² is how far the AVERAGE curve sits from the
# truth; Variance is how spread out the cloud is.
#
# INTUITION:
#   - A model that's too simple (degree 1 line fitting a sine wave) has
#     HIGH BIAS: even the average of many fits is wrong. More data won't
#     fix this — the model class can't express the truth.
#   - A model that's too complex (degree 15 polynomial on 40 points)
#     has HIGH VARIANCE: each training sample produces a different
#     curve. The model memorises noise.
#   - The "sweet spot" balances the two. Cross-validation FINDS this
#     automatically without knowing the true function.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the polynomial experiment
# ════════════════════════════════════════════════════════════════════════
# We use a 1D synthetic sine problem because (a) we KNOW the truth, which
# lets us measure bias directly, and (b) the effect is visually obvious.
# The training set is small (40 points) so overfitting is visible; the
# test set is large (1,000 points) so the test MSE is a stable estimate.

print_header("Polynomial Degree Experiment on sin(2πx)")

x_train, y_train, x_test, y_test, noise_variance = make_sine_dataset(
    n_train=40, n_test=1000, noise_sigma=0.2, seed=SEED
)
print(
    f"Train: {x_train.shape[0]} pts  "
    f"Test: {x_test.shape[0]} pts  "
    f"σ² = {noise_variance:.4f}"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN polynomial models at many degrees
# ════════════════════════════════════════════════════════════════════════
# For each degree we fit on the training set and record MSE on both the
# training set (how well the model memorised) and the held-out test set
# (how well it generalises). The GAP between the two is the overfit
# penalty — the price the model pays for fitting the training noise.

DEGREES = [1, 2, 4, 6, 9, 12, 15, 20]
degree_rows: dict[int, dict[str, float]] = {}
print(f"\n{'Degree':>6} {'Train MSE':>12} {'Test MSE':>12} {'Gap':>10}")
print("-" * 46)
for degree in DEGREES:
    model = make_poly_pipeline(degree)
    model.fit(x_train, y_train)
    train_mse = mean_squared_error(y_train, model.predict(x_train))
    test_mse = mean_squared_error(y_test, model.predict(x_test))
    gap = test_mse - train_mse

    degree_rows[degree] = {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "gap": gap,
    }
    print(f"{degree:>6} {train_mse:>12.4f} {test_mse:>12.4f} {gap:>10.4f}")

best_degree = min(degree_rows, key=lambda d: degree_rows[d]["test_mse"])
max_degree = DEGREES[-1]


# ── Checkpoint 1 ───────────────────────────────────────────────────────
assert (
    degree_rows[1]["test_mse"] > degree_rows[4]["test_mse"]
), "Degree=1 should have higher test error than degree=4 (underfit)"
assert (
    degree_rows[20]["train_mse"] < degree_rows[2]["train_mse"]
), "Degree=20 should fit the training data more closely than degree=2"
print("\n[ok] Checkpoint 1 passed — underfit/overfit pattern confirmed")

# INTERPRETATION — computed from the table above, not assumed:
print(
    f"""
Reading the table:
  - Degree 1 (a straight line): train MSE {degree_rows[1]['train_mse']:.4f}, test
    MSE {degree_rows[1]['test_mse']:.4f} — BOTH far above the noise floor.
    That is UNDERFITTING (high bias): the line cannot bend with the sine.
  - Lowest test MSE at degree {best_degree}: {degree_rows[best_degree]['test_mse']:.4f}
    (noise floor σ² = {noise_variance:.4f}).
  - Degree {max_degree}: train MSE {degree_rows[max_degree]['train_mse']:.4f} but test
    MSE {degree_rows[max_degree]['test_mse']:.4f} — the train/test gap grew from
    {degree_rows[best_degree]['gap']:.4f} to {degree_rows[max_degree]['gap']:.4f}.
    That widening gap is OVERFITTING (high variance).
"""
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the bias-variance decomposition
# ════════════════════════════════════════════════════════════════════════
# Because we KNOW the data-generating process, we can estimate Bias² and
# Variance directly from their definition:
#   1. Draw a FRESH training set of 40 points from the same process
#   2. Fit the polynomial
#   3. Predict on a fixed grid of 200 x-values
#   4. Repeat 200 times
#
# Bias²    = mean squared distance of the AVERAGE prediction from truth
# Variance = spread of predictions across the 200 refits
# Noise    = known σ² (0.04 in our synthetic setup)
#
# (On real data you cannot draw fresh training sets, so practitioners
# approximate step 1 with bootstrap resamples of the one training set.)
# The grid stays inside [0.05, 0.95] — where training data actually
# lives — so the estimate is not dominated by edge extrapolation.

N_REFITS = 200
X_GRID = np.linspace(0.05, 0.95, 200).reshape(-1, 1)
Y_GRID_TRUTH = sine_truth(X_GRID)
rng = np.random.default_rng(SEED)


def bias_variance_decomposition(
    degree: int, n_refits: int = N_REFITS
) -> dict[str, object]:
    """Estimate Bias², Variance, and expected test error for a polynomial."""
    all_preds = []
    for _ in range(n_refits):
        x_fresh, y_fresh = sample_sine_training_set(
            len(y_train), noise_sigma=0.2, rng=rng
        )
        model = make_poly_pipeline(degree)
        model.fit(x_fresh, y_fresh)
        all_preds.append(model.predict(X_GRID))

    preds = np.array(all_preds)  # (n_refits, n_grid)
    mean_pred = preds.mean(axis=0)

    bias_sq = float(np.mean((mean_pred - Y_GRID_TRUTH) ** 2))
    variance = float(np.mean(preds.var(axis=0)))
    expected = bias_sq + variance + noise_variance

    return {
        "bias_sq": bias_sq,
        "variance": variance,
        "noise": noise_variance,
        "expected_error": expected,
        "preds": preds,
    }


print_header("Bias-Variance Decomposition over 200 Fresh Training Sets")
print(
    f"{'Degree':>6} {'Bias²':>10} {'Variance':>10} {'Noise':>8} "
    f"{'Expected':>12}  Dominant"
)
print("-" * 60)

BV_DEGREES = [1, 2, 3, 4, 6, 8, 10, 12, 15]
bv_rows: dict[int, dict[str, object]] = {}
for degree in BV_DEGREES:
    bv = bias_variance_decomposition(degree)
    bv_rows[degree] = bv
    dominant = "Bias" if bv["bias_sq"] > bv["variance"] else "Variance"
    print(
        f"{degree:>6} {bv['bias_sq']:>10.4f} {bv['variance']:>10.4f} "
        f"{bv['noise']:>8.4f} {bv['expected_error']:>12.4f}  {dominant}"
    )

sweet_spot = min(BV_DEGREES, key=lambda d: bv_rows[d]["expected_error"])


# ── Checkpoint 2 ───────────────────────────────────────────────────────
assert (
    bv_rows[1]["bias_sq"] > bv_rows[10]["bias_sq"]
), "Degree=1 should have higher bias² than degree=10"
assert (
    bv_rows[1]["variance"] < bv_rows[15]["variance"]
), "Degree=1 should have lower variance than degree=15"
assert (
    bv_rows[1]["bias_sq"] > bv_rows[1]["variance"]
), "Degree=1 should be bias-dominated"
assert (
    bv_rows[15]["variance"] > bv_rows[15]["bias_sq"]
), "Degree=15 should be variance-dominated"
print("\n[ok] Checkpoint 2 passed — bias-variance decomposition valid")

print(
    f"""
Reading the decomposition:
  - Degree 1: Bias² {bv_rows[1]['bias_sq']:.4f} vs Variance {bv_rows[1]['variance']:.4f}
    — bias dominates; the average straight line is simply the wrong shape.
  - Degree {sweet_spot}: lowest expected error ({bv_rows[sweet_spot]['expected_error']:.4f}).
  - Degree 15: Bias² {bv_rows[15]['bias_sq']:.4f} vs Variance {bv_rows[15]['variance']:.4f}
    — variance dominates; each training sample gives a different curve.
  - No degree gets below σ² = {noise_variance:.4f}: that part is noise.
"""
)

# Figure 1 — train vs test MSE against degree (the classic U-curve)
fig_curve = go.Figure()
fig_curve.add_trace(
    go.Scatter(
        x=DEGREES,
        y=[degree_rows[d]["train_mse"] for d in DEGREES],
        mode="lines+markers",
        name="Train MSE",
    )
)
fig_curve.add_trace(
    go.Scatter(
        x=DEGREES,
        y=[degree_rows[d]["test_mse"] for d in DEGREES],
        mode="lines+markers",
        name="Test MSE",
    )
)
fig_curve.add_hline(y=noise_variance, line_dash="dot", annotation_text="σ² (noise)")
fig_curve.update_layout(
    title="Train vs test error by polynomial degree",
    xaxis_title="Polynomial degree",
    yaxis_title="MSE",
)
print(f"Saved: {save_html_plot(fig_curve, 'ex2_01_train_test_curve.html')}")

# Figure 2 — Bias², Variance and expected error against degree
fig_bv = go.Figure()
for key, label in [
    ("bias_sq", "Bias²"),
    ("variance", "Variance"),
    ("expected_error", "Expected error (Bias² + Var + σ²)"),
]:
    fig_bv.add_trace(
        go.Scatter(
            x=BV_DEGREES,
            y=[bv_rows[d][key] for d in BV_DEGREES],
            mode="lines+markers",
            name=label,
        )
    )
fig_bv.update_layout(
    title="Bias-variance decomposition (200 fresh training sets per degree)",
    xaxis_title="Polynomial degree",
    yaxis_title="Error",
)
print(f"Saved: {save_html_plot(fig_bv, 'ex2_01_bias_variance.html')}")

# Figure 3 — visual proof: 20 of the refitted curves at three degrees.
# Degree 1 curves agree with each other but miss the sine (bias); degree 15
# curves hug the sine on average but scatter (variance).
fig_fits = go.Figure()
for degree, colour in [(1, "firebrick"), (sweet_spot, "seagreen"), (15, "royalblue")]:
    for i in range(20):
        fig_fits.add_trace(
            go.Scatter(
                x=X_GRID.ravel(),
                y=bv_rows[degree]["preds"][i],
                mode="lines",
                line={"color": colour, "width": 1},
                opacity=0.35,
                name=f"degree {degree}",
                showlegend=(i == 0),
            )
        )
fig_fits.add_trace(
    go.Scatter(
        x=X_GRID.ravel(),
        y=Y_GRID_TRUTH,
        mode="lines",
        line={"color": "black", "width": 3},
        name="truth sin(2πx)",
    )
)
fig_fits.update_layout(
    title="20 refits per degree: bias = wrong shape, variance = spread",
    xaxis_title="x",
    yaxis_title="prediction",
    yaxis_range=[-2, 2],
)
print(f"Saved: {save_html_plot(fig_fits, 'ex2_01_refit_curves.html')}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: a Singapore retail bank's credit-scorecard choice
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore retail bank scores its credit-card
# customers every night to decide whether to extend additional credit.
# The risk team has a catalogue of ~180 candidate features (income,
# transaction velocity, bureau data, device signals, etc.).
#
# WHY BIAS-VARIANCE MATTERS HERE:
#   - A 5-feature linear scorecard is INTERPRETABLE but high-bias.
#     It systematically mis-ranks mid-risk customers, so the bank either
#     extends credit to future defaulters OR declines good customers.
#   - A 180-feature gradient-boosted model with no regularisation is
#     high-variance. On a new batch (e.g. customers acquired after a
#     change in the economy) its predictions swing because it memorised
#     correlations specific to the training window. This is why model
#     validation teams insist on OUT-OF-TIME testing before approval.
#
# ILLUSTRATIVE ARITHMETIC (round numbers, not a real bank's figures): on
# a S$10B card book, every 0.10 percentage-point reduction in the default
# rate is S$10M a year of avoided write-offs — so the "sweet spot" is
# worth real money.
#
# CONNECTION: The "sweet spot" in the bias-variance curve is what a
# model-validation reviewer looks for when vetting a scorecard. Too
# simple = systematic mispricing of risk. Too complex = instability on
# new customers.

print_header("Retail Credit Scoring — Bias/Variance in Context")
print(
    """
Stakeholder         | Concern                      | BV trade-off
--------------------|------------------------------|-----------------
Credit risk team    | Missed defaults              | Too much bias
Model validation    | Out-of-time instability      | Too much variance
Branch / CX         | Declined good customers      | Too much bias
Finance / capital   | Buffer volatility            | Too much variance

Takeaway: the bias-variance curve is not an academic exercise — every
stakeholder sits at a different point on it. The "right" complexity is
the one that minimises EXPECTED loss for the business, not training loss.
"""
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print(
    """
======================================================================
  WHAT YOU'VE MASTERED
======================================================================

  [x] Train/test error gap as an overfit diagnostic
  [x] The E[(y-ŷ)²] = Bias² + Variance + σ² decomposition
  [x] Empirical bias/variance estimation by refitting on fresh samples
  [x] Reading the "dominant" term to diagnose a model
  [x] Applying the trade-off to a Singapore credit-scoring decision

  KEY INSIGHT: "Complexity" is not a single number — it's the knob you
  turn to trade bias against variance. Regularisation (next file) is
  another way to turn that same knob without changing the model class.

  NEXT: 02_ridge_regression.py — shrink coefficients toward zero with
  L2 and meet its Bayesian alter-ego (Gaussian prior on weights).
"""
)
