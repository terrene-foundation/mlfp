# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 2.5: Learning Curves and Diagnostic Playbook
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Read a learning curve to decide "more data" vs "better model"
#   - Compare OLS, Ridge, and Lasso learning curves on one dataset
#   - Recognise the canonical shapes: large gap (high variance),
#     converged low (high bias), converged high (good fit)
#   - Use learning curves to justify (or reject) data-collection spend
#   - Tie the entire exercise together with a decision playbook
#
# PREREQUISITES:
#   - 01 through 04 in this exercise
#
# ESTIMATED TIME: ~35 minutes
#
# TASKS (5-phase R10):
#   1. Theory — learning-curve shapes and what they mean
#   2. Build — three models to compare (OLS, Ridge, Lasso)
#   3. Train — sklearn.learning_curve for each
#   4. Visualise — train-vs-validation curves on a real sample-size axis
#   5. Apply — a telco's churn-data purchase decision
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, learning_curve

from shared.mlfp03.ex_2 import (
    SEED,
    load_credit_data,
    print_header,
    save_html_plot,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — Reading a Learning Curve
# ════════════════════════════════════════════════════════════════════════
# A learning curve plots model performance (train and validation) as a
# function of the training-set size. The canonical readings:
#
#   1. LARGE GAP — train score well above validation score
#      Diagnosis: HIGH VARIANCE (overfitting). The model memorises its
#      training rows. Remedies: more data (if the validation curve is
#      still rising), stronger regularisation, or a simpler model.
#
#   2. CONVERGED LOW — train and validation meet, but at a poor score
#      Diagnosis: HIGH BIAS (underfitting). Adding rows WON'T help: the
#      curves are already together. Remedies: richer features or a more
#      flexible model class.
#
#   3. CONVERGED HIGH — train and validation meet at a good score
#      Diagnosis: good fit. More data gives marginal returns.
#
# The GAP between the curves reflects variance; the LEVEL at which they
# converge reflects bias (how good this model class can get here).


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the comparison models
# ════════════════════════════════════════════════════════════════════════
# A learning curve needs room to grow, so we draw a 4,000-row training
# pool (instead of the 300 rows used in files 02–04). The α values are
# the ones 5-fold CV chose on the 300-row sample in files 02 and 03.

print_header("Learning Curves — OLS vs Ridge vs Lasso")
# TODO: load_credit_data with a 4,000-row training pool and 1,000 test rows.
X_train, y_train, X_test, y_test, feature_names = ____
print(f"Training pool: {X_train.shape}")

models = {
    "OLS (unregularised)": LinearRegression(),
    "Ridge (α=100)": Ridge(alpha=100.0),
    "Lasso (α=0.1)": Lasso(alpha=0.1, max_iter=50_000),
}

TRAIN_SIZES = [50, 100, 200, 400, 800, 1600, 3200]
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)


def diagnose(train_curve: np.ndarray, val_curve: np.ndarray) -> str:
    """Read a learning curve from its LAST points (computed, not assumed)."""
    # TODO: gap = last train score minus last validation score;
    # still_rising = validation improved by more than 0.01 over the last step.
    gap = ____
    still_rising = ____
    if gap > 0.05:
        return (
            "HIGH VARIANCE — large train/validation gap"
            + ("; validation still rising, more data will help" if still_rising else "")
        )
    # TODO: which condition on the final validation score means "converged LOW"?
    if ____:
        return "HIGH BIAS — curves have met at a low score; more data won't help"
    return "GOOD FIT — curves have met at a good score"


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN each model across growing training-set sizes
# ════════════════════════════════════════════════════════════════════════

all_curves: dict[str, dict[str, np.ndarray]] = {}
for name, model in models.items():
    # TODO: sklearn learning_curve over TRAIN_SIZES with cv=cv, scoring="r2".
    # It returns (train_sizes, train_scores, validation_scores).
    train_sizes, tr_scores, te_scores = ____
    all_curves[name] = {
        "sizes": train_sizes,
        "train_mean": tr_scores.mean(axis=1),
        "test_mean": te_scores.mean(axis=1),
        "train_std": tr_scores.std(axis=1),
        "test_std": te_scores.std(axis=1),
    }

    print_header(name)
    print(f"{'N':>8} {'Train R²':>10} {'Val R²':>10} {'Gap':>10}")
    print("-" * 40)
    for n_, tr, te in zip(train_sizes, tr_scores.mean(axis=1), te_scores.mean(axis=1)):
        print(f"{n_:>8} {tr:>10.4f} {te:>10.4f} {(tr - te):>10.4f}")
    small_gap = all_curves[name]["train_mean"][0] - all_curves[name]["test_mean"][0]
    print(f"  Gap at N={train_sizes[0]}: {small_gap:.3f}")
    print(
        f"  Reading at N={train_sizes[-1]}: "
        f"{diagnose(all_curves[name]['train_mean'], all_curves[name]['test_mean'])}"
    )


# ── Checkpoint 1 ───────────────────────────────────────────────────────
assert len(train_sizes) == len(
    TRAIN_SIZES
), "Should have one entry per training-set size"
assert all(
    "test_mean" in c for c in all_curves.values()
), "Every model should record a test-mean curve"
print("\n[ok] Checkpoint 1 passed — learning curves computed for all models")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the three learning curves
# ════════════════════════════════════════════════════════════════════════
# One figure, the REAL training-set sizes on a log x-axis, solid lines
# for validation and dashed for train. This plot is the single best
# diagnostic to show a budget committee when asking for more data.

fig = go.Figure()
for name, curves in all_curves.items():
    # TODO: validation curve — x is the REAL training-set sizes.
    fig.add_trace(
        go.Scatter(
            x=____,
            y=____,
            mode="lines+markers",
            name=f"{name} — validation",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=curves["sizes"],
            y=curves["train_mean"],
            mode="lines",
            line={"dash": "dash"},
            name=f"{name} — train",
        )
    )
fig.update_layout(
    title="Learning curves — credit savings regression",
    xaxis_title="Training set size (samples, log scale)",
    yaxis_title="R²",
    xaxis_type="log",
    yaxis_range=[-0.5, 1.0],
)
plot_path = save_html_plot(fig, "learning_curves_ols_ridge_lasso.html")
print(f"\nSaved: {plot_path}")


# ── Checkpoint 2 ───────────────────────────────────────────────────────
assert plot_path.exists(), "Learning-curve plot should be saved"
print("\n[ok] Checkpoint 2 passed — learning-curve plot written")

ols = all_curves["OLS (unregularised)"]
ridge = all_curves["Ridge (α=100)"]
print(
    f"""
Reading the curves (computed from the tables above):
  - Small data (N={TRAIN_SIZES[0]}): OLS validation R² {ols['test_mean'][0]:+.3f} vs
    Ridge {ridge['test_mean'][0]:+.3f}. With few rows OLS overfits badly and
    regularisation is worth the most.
  - Large data (N={TRAIN_SIZES[-1]}): OLS {ols['test_mean'][-1]:+.3f}, Ridge
    {ridge['test_mean'][-1]:+.3f}. The models converge; regularisation matters
    less as data grows.
  - OLS at full size: {diagnose(ols['train_mean'], ols['test_mean'])}.
"""
)
if diagnose(ols["train_mean"], ols["test_mean"]).startswith("HIGH BIAS"):
    print(
        "  A linear model on these features has reached its ceiling — to do\n"
        "  better, change the features or the model class, not the row count."
    )


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: a telco's churn-data purchase decision
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore mobile operator is deciding
# whether to license 18 more months of historical call-detail records,
# growing its labelled churn training set from ~180K to ~420K
# subscribers, for a one-off cost of ~S$1.4M.
#
# WHY LEARNING CURVES ARE THE RIGHT TOOL:
#   - The CTO needs a defensible answer to "will more data reduce
#     churn?" BEFORE the cheque is written.
#   - If the validation curve has CONVERGED with the train curve (as our
#     credit curves did), extra rows will not move the needle — spend
#     the money on BETTER features or a more flexible model instead.
#   - If a LARGE GAP remains and the validation curve is still RISING,
#     extrapolate the curve to the new size, convert the expected lift
#     into retained revenue with the retention team, and compare that
#     with the licence cost.

print_header("Telco Churn — learning-curve data decision")
print(
    """
Learning curve shape        | Decision
----------------------------|----------------------------------------
Large gap, still rising     | BUY data (or regularise harder) — price
(high variance)             | the extrapolated lift against the cost
----------------------------|----------------------------------------
Converged at a low score    | PASS on data; invest in features or a
(high bias)                 | more flexible model
----------------------------|----------------------------------------
Converged at a good score   | PASS — the model is done
"""
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION — The Full Exercise in One Frame
# ════════════════════════════════════════════════════════════════════════
print(
    """
======================================================================
  WHAT YOU'VE MASTERED (ENTIRE EXERCISE)
======================================================================

  01. Bias-Variance     — why "more complex" is not always "better"
  02. Ridge (L2)        — shrinkage, Gaussian prior, stability
  03. Lasso + ElasticNet — sparsity, L1 diamond, feature selection
  04. Cross-validation  — nested / stratified / time-series / group
  05. Learning curves   — diagnose data hunger vs model weakness

  DECISION PLAYBOOK:
    Step 1. Start with a learning curve on a regularised baseline.
    Step 2. Large train/validation gap and validation still rising →
            more data and/or stronger regularisation (HIGH VARIANCE).
    Step 3. Curves converged close together at a poor score → a richer
            model or better features, NOT more data (HIGH BIAS).
    Step 4. If the data has temporal or group structure, DO NOT use
            shuffled k-fold — use TimeSeriesSplit or GroupKFold.
    Step 5. Choose hyperparameters with (nested) CV, never on the test set.

  KEY INSIGHT: Regularisation is how you encode your prior belief about
  the world. Cross-validation is how you audit that belief against
  reality. Learning curves tell you whether reality will change if you
  throw more data at it.

  NEXT: Exercise 3 — full supervised model zoo (SVM, KNN, Naive Bayes,
  Trees, Random Forests) on Singapore e-commerce churn data.
"""
)
