# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 1.4: Embedded Feature Selection (L1 / Lasso Path)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Use L1 regularisation to drive feature coefficients to exactly zero
#   - Walk a regularisation path (multiple C values) to understand the
#     sparsity / accuracy trade-off
#   - Read non-zero coefficients as a ranked feature importance
#   - Apply embedded selection in a low-latency production scenario
#     (fraud / anomaly scoring) where one-pass fit matters
#
# PREREQUISITES: 03_wrapper_selection.py (RFE understanding)
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Theory — why L1 produces sparse solutions
#   2. Build — scale features, define the C grid
#   3. Train — fit LogReg with L1 at each C
#   4. Visualise — sparsity path + top-15 surviving coefficients
#   5. Apply — card-fraud scoring at a Singapore retail bank
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from shared.mlfp03.ex_1 import (
    OUTPUT_DIR,
    build_full_feature_frame,
    load_icu_tables,
    log_selection_run,
    prepare_selection_inputs,
    setup_tracking,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — L1 Drives Coefficients To Zero
# ════════════════════════════════════════════════════════════════════════
# L1 (Lasso) regularisation adds a penalty proportional to the SUM of
# absolute coefficient values:
#
#     loss = data_loss + (1/C) * sum(|w_j|)
#
# (In scikit-learn ≥ 1.8 you request the pure L1 penalty with
# LogisticRegression(l1_ratio=1.0, solver="saga") — the older
# penalty="l1" spelling is deprecated.)
#
# The absolute-value penalty has a corner at zero, so the gradient
# pushes weak coefficients all the way to exactly 0 — not just "close
# to zero" like L2. Every zero coefficient is a feature that has been
# eliminated from the model.
#
# This is an EMBEDDED method: the selection happens INSIDE the training
# loop, not before it (filter) and not around it (wrapper). It is:
#   + a single fit instead of N refits (fast)
#   + natural to interpret (zero coefficient = eliminated)
#   + a Bayesian statement ("I believe only a few features matter",
#     i.e. a Laplace prior on coefficients)
#   - sensitive to feature scale — you MUST standardise first
#   - biased when features are highly correlated (ties broken
#     arbitrarily)
#
# The regularisation path: fit Lasso at decreasing C values (stronger
# regularisation). As C shrinks, more coefficients go to zero. The
# path shows the order in which the model "gives up" features — the
# features that survive the longest are the strongest predictors.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: scale inputs and define the C grid
# ════════════════════════════════════════════════════════════════════════

tables = load_icu_tables()
features = build_full_feature_frame(tables)
feature_cols, X_sel, y_binary = prepare_selection_inputs(features)

print("\n" + "=" * 70)
print("  Embedded Selection — L1 Logistic Regression")
print("=" * 70)
print(f"  Features: {len(feature_cols)}")
print(f"  Samples:  {X_sel.shape[0]}")

# TODO: Standardise X_sel before L1 fitting — L1 is scale-sensitive.
# Hint: StandardScaler()
X_scaled = ____
C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: walk the regularisation path
# ════════════════════════════════════════════════════════════════════════

print("\n--- Regularisation Path ---")
print(f"{'C':>8} {'Non-zero':>10} / {'Total':>6}")
print("-" * 30)

lasso_results: dict[float, dict] = {}
for c_val in C_VALUES:
    # TODO: L1 LogisticRegression at this C (l1_ratio=1.0, solver="saga",
    # max_iter=20000, random_state=42), then fit it on the scaled inputs.
    lasso = ____
    ____
    coefs = lasso.coef_[0].copy()
    n_nonzero = int((np.abs(coefs) > 1e-6).sum())
    lasso_results[c_val] = {"n_nonzero": n_nonzero, "coefs": coefs}
    print(f"  {c_val:>8.3f} {n_nonzero:>10} / {len(feature_cols):>6}")


LASSO_C = 0.1
lasso_coefs = lasso_results[LASSO_C]["coefs"]
# TODO: names of features with a non-zero coefficient (|coef| > 1e-6).
lasso_selected = ____
lasso_importance = sorted(
    [
        (name, float(abs(coef)))
        for name, coef in zip(feature_cols, lasso_coefs)
        if abs(coef) > 1e-6
    ],
    key=lambda x: x[1],
    reverse=True,
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert len(lasso_selected) > 0, "Task 3: L1 should retain SOME features"
assert len(lasso_selected) < len(
    feature_cols
), "Task 3: L1 should ELIMINATE some features — if none are zero, C is too high"
print("\n[ok] Checkpoint 1 passed — regularisation path + selection complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE sparsity trajectory + surviving coefficients
# ════════════════════════════════════════════════════════════════════════

print(f"\n--- L1 Selected Features (C={LASSO_C}, {len(lasso_selected)} total) ---")
print(f"{'Feature':<35} {'|coef|':>10}")
print("-" * 48)
max_abs = max((c for _, c in lasso_importance), default=1.0)
for name, imp in lasso_importance[:15]:
    bar = "#" * int((imp / max_abs) * 20)
    print(f"  {name:<33} {imp:>10.4f}  {bar}")

# Sparsity trajectory as ASCII sparkline
print("\n--- Sparsity Trajectory ---")
for c_val in C_VALUES:
    nz = lasso_results[c_val]["n_nonzero"]
    bar = "#" * int(nz / max(1, len(feature_cols)) * 40)
    print(f"  C={c_val:>7.3f}  {nz:>4}/{len(feature_cols):<4}  {bar}")

# --- Lasso coefficient path: regularisation strength vs coefficients ---
# Show how each feature's coefficient evolves as C increases (weaker
# regularisation). Features that survive low C are the strongest.
top_features = [name for name, _ in lasso_importance[:10]]
top_indices = [feature_cols.index(f) for f in top_features if f in feature_cols]

fig_path = go.Figure()
for idx in top_indices:
    # TODO: this feature's coefficient at every C in C_VALUES
    coef_path = ____
    fig_path.add_trace(
        go.Scatter(
            x=[np.log10(c) for c in C_VALUES],
            y=coef_path,
            mode="lines+markers",
            name=feature_cols[idx],
        )
    )
fig_path.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
fig_path.update_layout(
    title="Lasso Coefficient Path — Top 10 Features vs log10(C)",
    xaxis_title="log10(C)  (left = stronger regularisation)",
    yaxis_title="Coefficient value",
    height=500,
    legend=dict(font=dict(size=9)),
)
path_viz = OUTPUT_DIR / "ex1_04_lasso_coefficient_path.html"
fig_path.write_html(str(path_viz))
print(f"\n  Saved: {path_viz}")


# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert (
    lasso_results[0.001]["n_nonzero"] <= lasso_results[10.0]["n_nonzero"]
), "Task 4: stronger regularisation (smaller C) must yield fewer non-zero coefficients"
print("\n[ok] Checkpoint 2 passed — sparsity trajectory is monotonic\n")

# INTERPRETATION — computed from the path:
print(
    f"  C={C_VALUES[0]}: {lasso_results[C_VALUES[0]]['n_nonzero']} features survive; "
    f"C={C_VALUES[-1]}: {lasso_results[C_VALUES[-1]]['n_nonzero']} "
    "(practically unregularised)."
)
# The features that survive the smallest C are the ones the model gives
# up last. Where the count jumps between neighbouring C values is the
# "elbow"; pick C there by cross-validation, not by eye. (On this
# dataset 02 and 03 found no feature beating a noise baseline, so the
# survivors here are the least-noisy noise — L1 always returns SOME set.)


# ════════════════════════════════════════════════════════════════════════
# TASK 4b — LOG the embedded run
# ════════════════════════════════════════════════════════════════════════


async def log_embedded() -> str:
    conn, tracker, exp_id = await setup_tracking()
    run_id = await log_selection_run(
        tracker,
        exp_id,
        run_name=f"embedded_lasso_c{LASSO_C}",
        method="embedded",
        selected_features=lasso_selected,
        total_features=len(feature_cols),
        extra_params={
            "estimator": "LogisticRegression",
            "penalty": "l1",
            "C": str(LASSO_C),
            "solver": "saga",
            "max_iter": "20000",
        },
        extra_metrics={
            "top_abs_coef": lasso_importance[0][1] if lasso_importance else 0.0,
            "n_nonzero_c0p001": float(lasso_results[0.001]["n_nonzero"]),
            "n_nonzero_c10": float(lasso_results[10.0]["n_nonzero"]),
        },
    )
    await conn.close()
    return run_id


run_id = asyncio.run(log_embedded())
print(f"\n  ExperimentTracker run: {run_id}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: card-fraud scoring at a Singapore retail bank
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore retail bank scores millions of card
# transactions a day. The fraud team needs a model that:
#   - returns a decision within a ~25 ms point-of-sale budget
#   - uses a stable, auditable feature set its model-risk team reviews
#     every quarter
#   - chooses from ~400 candidate features (transaction velocity, geo
#     distance, merchant-category histograms, device signals)
#
# Why L1 is the right tool here:
#   - One fit instead of an RFE loop keeps nightly retraining cheap
#   - Zero-weight features can be dropped from the production scorer,
#     which helps the latency budget
#   - Reviewers can read a linear model's coefficients directly
#   - With standardised inputs the sparsity pattern tends to be stable
#     between runs, so the feature list can go into the model
#     documentation
#
# ILLUSTRATIVE ARITHMETIC (assumed values, not the bank's figures): if
# one percentage point of fraud recall at fixed precision is worth
# ~S$4M a year, a leaner feature set that frees latency for one more
# signal worth +0.8pp recall is worth ~0.8 × S$4M ≈ S$3.2M a year.
#
# LIMITATIONS:
#   - L1 breaks ties between correlated features arbitrarily — two
#     near-duplicate velocity features may randomly split weight on
#     each refit, which complicates the quarterly model-risk review
#   - Standardisation must be part of the production pipeline —
#     forgetting it silently destroys the coefficient interpretation
#   - L1 is biased toward sparse SUBSETS, not the minimum description
#     length; for truly minimum-feature-count goals, combine with
#     stability selection (bootstrap L1 many times, keep what survives
#     most runs)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Standardised features before L1 fitting (non-negotiable)
  [x] Walked the regularisation path across multiple C values
  [x] Read non-zero coefficients as an interpretable feature ranking
  [x] Logged the embedded run to ExperimentTracker
  [x] Applied L1 to card-fraud scoring where latency dominates

  KEY INSIGHT: Embedded methods give you selection for free as a side
  effect of training. When the model family is linear and the latency
  budget is tight, this is the default — no separate selection pass,
  no wrapper loop, no filter heuristic.

  Next: 05_validation_and_tracking.py — build a FeatureSchema contract,
  log the final consensus feature set, and run a leakage audit that
  any of the prior three selections must pass before production.
"""
)
