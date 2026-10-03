# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 1.3: Wrapper Feature Selection (Recursive Feature
#                         Elimination with Random Forest)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Run Recursive Feature Elimination (RFE) around a Random Forest
#   - Understand how wrapper methods capture feature INTERACTIONS
#   - Compare RFE's selection against the filter consensus
#   - Keep RFE inside the CV folds so the elimination curve is honest
#   - Apply wrapper selection in a setting where interactions matter
#     (cardiology risk models)
#
# PREREQUISITES: 02_filter_selection.py (filter consensus built)
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Theory — why wrappers see what filters miss
#   2. Build — assemble estimator + RFE
#   3. Train — fit RFE, get ranking + support mask
#   4. Visualise — ranked table + grouped-CV elimination curve
#   5. Apply — heart-failure readmission risk at a Singapore hospital
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline

from shared.mlfp03.ex_1 import (
    OUTPUT_DIR,
    build_full_feature_frame,
    load_icu_tables,
    log_selection_run,
    prepare_selection_inputs,
    setup_tracking,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Wrappers Capture Interactions
# ════════════════════════════════════════════════════════════════════════
# A wrapper method trains a MODEL, asks the model which features
# mattered, removes the weakest, and repeats. Because the model sees all
# features together, it can learn that feature A is only useful in the
# presence of feature B — an interaction a filter method ignores.
#
# Recursive Feature Elimination (RFE) is the canonical wrapper:
#     1. Fit a Random Forest on all features.
#     2. Rank features by the forest's feature_importances_.
#     3. Drop the lowest-ranked k features.
#     4. Refit, re-rank, repeat until we hit the target feature count.
#
# The Random Forest is a strong default inside RFE because it captures
# non-linear dependencies and interactions out of the box — linear
# wrappers (LogReg RFE) miss exactly the interactions we care about.
#
# COST TRADE-OFF:
#   + captures interactions
#   + works with any model that exposes feature importances
#   - much slower than filter methods (train N models, not one score)
#   - selection is specific to the estimator — an RFE-chosen set may
#     help a Random Forest but confuse a linear model


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: assemble the estimator and RFE object
# ════════════════════════════════════════════════════════════════════════

tables = load_icu_tables()
features = build_full_feature_frame(tables)
feature_cols, X_sel, y_binary = prepare_selection_inputs(features)

print("\n" + "=" * 70)
print("  Wrapper Selection — RFE with Random Forest")
print("=" * 70)
print(f"  Features: {len(feature_cols)}")
print(f"  Samples:  {X_sel.shape[0]}")

# TODO: Build a Random Forest estimator for RFE (100 trees, max_depth=5,
# random_state=42). Hint: RandomForestClassifier(...)
rf_estimator = ____

N_FEATURES_TO_SELECT = 15
# TODO: Wrap the estimator in RFE, keeping N_FEATURES_TO_SELECT and
# dropping 5 features per step.
rfe = ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: fit the RFE loop
# ════════════════════════════════════════════════════════════════════════

# TODO: Fit the RFE loop on the selection inputs.
____

rfe_selected = [name for name, selected in zip(feature_cols, rfe.support_) if selected]
rfe_ranking = sorted(
    [(name, int(rank)) for name, rank in zip(feature_cols, rfe.ranking_)],
    key=lambda x: x[1],
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert (
    len(rfe_selected) == N_FEATURES_TO_SELECT
), f"Task 3: RFE must select {N_FEATURES_TO_SELECT}, got {len(rfe_selected)}"
assert all(
    f in feature_cols for f in rfe_selected
), "Task 3: invalid feature in RFE output"
print("\n[ok] Checkpoint 1 passed — RFE fit complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the ranking
# ════════════════════════════════════════════════════════════════════════

print("\n--- RFE Ranking (top 20; rank=1 means SELECTED) ---")
print(f"{'Feature':<35} {'Rank':>6}")
print("-" * 44)
for name, rank in rfe_ranking[:20]:
    marker = "  <-- selected" if rank == 1 else ""
    print(f"  {name:<33} {rank:>6}{marker}")

print(f"\n  Total RFE-selected features: {len(rfe_selected)}")
print(f"  Selected: {rfe_selected}")

# --- RFE elimination curve: accuracy vs number of features ---
# The curve must be HONEST: if RFE chooses features on all rows and we
# then cross-validate on those same rows, the held-out folds helped pick
# the features (selection leakage). So RFE goes INSIDE a Pipeline and is
# re-fitted within every training fold. Folds are grouped by patient_id
# — one patient can have several admissions, and the same patient must
# not sit in both train and test.
groups = features["patient_id"].to_numpy()
group_cv = GroupKFold(n_splits=3)
# TODO: accuracy of always predicting the more common class
majority_acc = ____
n_features_range = [5, 8, 10, 12, 15, 18, 20, 25]
n_features_range = [n for n in n_features_range if n <= len(feature_cols)]
elim_scores = []
for n_feat in n_features_range:
    pipe = Pipeline(
        [
            (
                "rfe",
                RFE(
                    estimator=RandomForestClassifier(
                        n_estimators=50, max_depth=5, random_state=42
                    ),
                    n_features_to_select=n_feat,
                    step=5,
                ),
            ),
            ("rf", RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
        ]
    )
    # TODO: grouped CV accuracy of the WHOLE pipeline (RFE refits inside
    # each fold). Hint: cross_val_score(..., cv=group_cv, groups=groups,
    # scoring="accuracy").mean()
    cv_acc = ____
    elim_scores.append(cv_acc)
    print(f"  n_features={n_feat:<3}  grouped CV accuracy={cv_acc:.4f}")
print(f"  Majority-class baseline accuracy: {majority_acc:.4f}")

fig_rfe = go.Figure()
fig_rfe.add_trace(
    go.Scatter(
        x=n_features_range,
        y=elim_scores,
        mode="lines+markers",
        marker=dict(size=10, color="#2563eb"),
        line=dict(width=3),
        name="CV Accuracy",
    )
)
best_idx = int(np.argmax(elim_scores))
fig_rfe.add_annotation(
    x=n_features_range[best_idx],
    y=elim_scores[best_idx],
    text=f"Best: {n_features_range[best_idx]} features",
    showarrow=True,
    arrowhead=2,
)
fig_rfe.update_layout(
    title="RFE Elimination Curve — Accuracy vs Number of Features",
    xaxis_title="Number of Features Selected",
    yaxis_title="3-fold grouped CV accuracy (RFE inside each fold)",
    height=450,
)
fig_rfe.add_hline(y=majority_acc, line_dash="dot", annotation_text="majority-class baseline")
rfe_path = OUTPUT_DIR / "ex1_03_rfe_elimination_curve.html"
fig_rfe.write_html(str(rfe_path))
print(f"\n  Saved: {rfe_path}")


# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert rfe_ranking[0][1] == 1, "Task 4: top-ranked features should have rank=1"
print("\n[ok] Checkpoint 2 passed — RFE ranking is well-formed\n")

# INTERPRETATION — computed: does ANY feature subset beat guessing the
# majority class?
best_acc = max(elim_scores)
print(
    f"  Best grouped CV accuracy {best_acc:.4f} vs majority baseline "
    f"{majority_acc:.4f} ({best_acc - majority_acc:+.4f})."
)
if best_acc - majority_acc < 0.01:
    print(
        "  → No subset beats the baseline: RFE still returns 15 'selected'\n"
        "    features, but on this data they are an ordering of noise. A\n"
        "    wrapper's ranking means something only when the model it wraps\n"
        "    beats a trivial baseline under honest CV."
    )
else:
    print(
        "  → Compare the selected list with the filter consensus from 02:\n"
        "    features RFE keeps but filters ranked low are candidates for\n"
        "    interaction effects."
    )


# ════════════════════════════════════════════════════════════════════════
# TASK 4b — LOG the wrapper run
# ════════════════════════════════════════════════════════════════════════


async def log_wrapper() -> str:
    conn, tracker, exp_id = await setup_tracking()
    run_id = await log_selection_run(
        tracker,
        exp_id,
        run_name="wrapper_rfe_rf15",
        method="wrapper",
        selected_features=rfe_selected,
        total_features=len(feature_cols),
        extra_params={
            "estimator": "RandomForestClassifier",
            "n_estimators": "100",
            "max_depth": "5",
            "n_features_to_select": str(N_FEATURES_TO_SELECT),
            "step": "5",
        },
        extra_metrics={},
    )
    await conn.close()
    return run_id


run_id = asyncio.run(log_wrapper())
print(f"\n  ExperimentTracker run: {run_id}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: heart-failure readmission risk at a Singapore hospital
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore cardiac centre wants a 30-day
# readmission risk model for heart-failure patients. The training set has
# ~220 candidate features across demographics, lab panels, medication
# history and procedure codes. Clinicians expect interactions to matter:
#   - ejection fraction × diuretic dose (under-diuresed weak heart)
#   - creatinine × ACE inhibitor (renal contraindication)
#   - BNP × beta-blocker dose (titration window)
# Filter methods score each factor on its own and can miss these.
#
# Why RFE + Random Forest fits:
#   - Random Forest captures interactions natively through its splits
#   - RFE iteratively removes the weakest features, re-fitting so the
#     survivors can re-combine
#   - The final 15 can be reviewed by cardiologists against guidelines
#
# ILLUSTRATIVE ARITHMETIC (assumed values): if each prevented readmission
# saves ~S$16,000 and a model cuts readmissions by 4 percentage points
# on ~3,600 heart-failure discharges a year:
#     3,600 × 0.04 × S$16,000 ≈ S$2.3M/year
# — but only if the selected features beat a trivial baseline under
# honest (in-fold, grouped) cross-validation, as checked above.
#
# LIMITATIONS:
#   - RFE is estimator-specific: the 15 features that help a Random
#     Forest may not transfer cleanly to a logistic regression
#   - Compute cost scales with dataset size; at 500K rows and 500
#     features, RFE becomes impractical and you drop to embedded
#     methods (04_embedded_selection.py)
#   - The Random Forest's feature_importances_ are biased toward
#     high-cardinality features; permutation importance is more robust
#     if cardinality varies wildly across candidates


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Configured a Random Forest estimator inside sklearn's RFE loop
  [x] Fit RFE and extracted the selected-feature mask + ranking
  [x] Understood how wrappers promote interaction-rich features
  [x] Logged the wrapper run to ExperimentTracker
  [x] Measured the elimination curve with RFE inside grouped CV folds

  KEY INSIGHT: Wrappers see interactions but pay a compute tax. Use them
  when you can afford the training time AND when domain knowledge tells
  you interactions matter.

  Next: 04_embedded_selection.py — Lasso regularisation, which bakes
  feature selection into the model-fitting step itself.
"""
)
