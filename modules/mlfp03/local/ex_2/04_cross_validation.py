# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 2.4: Cross-Validation Strategies
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Run nested CV and measure the optimism of "tune and report on the
#     same folds"
#   - Use stratified k-fold to keep a rare outcome's rate equal across folds
#   - Apply TimeSeriesSplit for walk-forward validation on time-ordered data
#   - Use GroupKFold so repeat patients never sit on both sides of a split
#   - Pick the RIGHT CV strategy for a given deployment scenario
#
# PREREQUISITES:
#   - 02_ridge_regression.py and 03_lasso_elasticnet.py
#   - MLFP02 sampling theory (bias, variance of estimators)
#
# ESTIMATED TIME: ~45 minutes
#
# TASKS (5-phase R10):
#   1. Theory — why "one CV fits all" is wrong
#   2. Build — the splitters and the datasets whose structure they match
#   3. Train — nested CV (honest estimate after α selection)
#   4. Visualise — stratified, walk-forward and grouped CV on real structure
#   5. Apply — choosing the CV for a payments-fraud scorer
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
)

from shared.mlfp03.ex_2 import (
    ALPHAS,
    SEED,
    cross_val_auc_split_first,
    load_credit_data,
    load_credit_default_sample,
    load_icu_admissions_for_cv,
    print_header,
    save_html_plot,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Cross-Validation Is More Than Just k-fold
# ════════════════════════════════════════════════════════════════════════
# Standard k-fold implicitly assumes:
#   (a) Observations are INDEPENDENT — shuffling is harmless
#   (b) Observations are IDENTICALLY distributed — training-time
#       distribution equals prediction-time distribution
#   (c) Hyperparameter selection and performance estimation can share
#       the same splits (they can't — the chosen setting was picked
#       BECAUSE it looked good on those folds)
#
# Real datasets break these assumptions all the time:
#   - FINANCIAL / CLINICAL DATA is temporal: shuffling lets the model
#     train on later records to predict earlier ones (violates (b)).
#   - MEDICAL DATA has repeated admissions per patient: the same patient
#     can end up in both train and test (violates (a)).
#   - RARE OUTCOMES (defaults, fraud) make a random fold's positive rate
#     swing, so fold scores are noisier than they need to be.
#   - HYPERPARAMETER TUNING on the evaluation folds gives an optimistic
#     estimate (violates (c)).
#
# THE FIX — pick the CV strategy that MATCHES the deployment scenario:
#   i.i.d. data             → k-fold
#   rare binary outcome     → stratified k-fold
#   temporal data           → TimeSeriesSplit (walk-forward)
#   grouped data            → GroupKFold
#   hyperparameter tuning   → nested CV (outer for eval, inner for tune)


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: the datasets and their structure
# ════════════════════════════════════════════════════════════════════════
# Each strategy is demonstrated on data that ACTUALLY has the structure
# it is designed for:
#   - credit savings regression (300 rows)   → nested CV
#   - credit default, 600 applicants (~13%)  → stratified k-fold
#   - ICU admissions, time-ordered, with
#     repeat patients                        → TimeSeriesSplit, GroupKFold

print_header("Cross-Validation Strategies")
X_train, y_train, X_test, y_test, feature_names = load_credit_data()
print(f"Credit regression train: {X_train.shape}")

X_def, y_def, _ = load_credit_default_sample(n=600)
print(f"Credit default sample:   {X_def.shape}, default rate {y_def.mean():.1%}")

icu = load_icu_admissions_for_cv()
n_patients = len(np.unique(icu["groups"]))
print(
    f"ICU admissions:          {icu['X'].shape}, {n_patients} distinct patients, "
    f"{icu['admit_time'].min():%Y-%m-%d} → {icu['admit_time'].max():%Y-%m-%d}"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN with nested CV (honest estimate after α selection)
# ════════════════════════════════════════════════════════════════════════
# STANDARD CV uses the SAME folds to pick α AND report performance: the
# best mean CV score among the candidates. That number is optimistic —
# among several candidates, the winner is partly the one that got lucky
# on those particular folds.
#
# NESTED CV fixes this:
#   OUTER 5-fold: held out for performance reporting (never touched
#                 during α selection).
#   INNER 3-fold: used inside each outer fold to pick α from ALPHAS.
#
# The outer mean estimates how well the WHOLE procedure ("tune α by CV,
# then fit") generalises.

print_header("Nested Cross-Validation (Ridge on credit savings)")

# Standard (optimistic) CV: best mean CV score across the α grid
grid = GridSearchCV(
    Ridge(),
    {"alpha": ALPHAS},
    cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
    scoring="r2",
)
grid.fit(X_train, y_train)
# TODO: The standard (optimistic) score is the BEST mean CV score found by
# the grid search. Hint: a GridSearchCV attribute ending in an underscore.
standard_score = ____
print(
    f"Standard CV (tune + report on same folds): R² = {standard_score:.4f}, "
    f"selected α = {grid.best_params_['alpha']}"
)

outer_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=SEED)

nested_scores: list[float] = []
selected_alphas: list[float] = []
print("\nOuter folds:")
for fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X_train)):
    X_out_tr, X_out_te = X_train[tr_idx], X_train[te_idx]
    y_out_tr, y_out_te = y_train[tr_idx], y_train[te_idx]

    best_alpha = ALPHAS[0]
    best_inner = -np.inf
    for alpha in ALPHAS:
        # TODO: inner 3-fold CV of Ridge(alpha=alpha) on the OUTER-TRAIN rows
        # only (X_out_tr, y_out_tr), cv=inner_cv, scoring="r2".
        inner = ____
        if inner.mean() > best_inner:
            best_inner = inner.mean()
            best_alpha = alpha

    # TODO: refit Ridge at best_alpha on the outer-train rows and score R²
    # on the outer-test rows.
    ridge_selected = ____
    outer_score = ____
    nested_scores.append(outer_score)
    selected_alphas.append(best_alpha)
    print(
        f"  Fold {fold_idx + 1}: α = {best_alpha:<8.4f}  "
        f"outer R² = {outer_score:.4f}"
    )

nested_mean = float(np.mean(nested_scores))
nested_std = float(np.std(nested_scores))
optimism = standard_score - nested_mean
print(f"\nNested CV:   R² = {nested_mean:.4f} ± {nested_std:.4f}")
print(f"Standard CV: R² = {standard_score:.4f}")
print(f"Optimism (standard - nested): {optimism:+.4f}")


# ── Checkpoint 1 ───────────────────────────────────────────────────────
assert len(nested_scores) == 5, "Should have 5 outer fold scores"
assert all(isinstance(s, float) for s in nested_scores), "Scores must be floats"
print("\n[ok] Checkpoint 1 passed — nested CV estimate produced")
print(
    f"  The optimism here is {optimism:+.4f} R², against a fold-to-fold "
    f"spread of ±{nested_std:.4f}. "
    + (
        "It is larger than one standard deviation — report the nested number."
        if optimism > nested_std
        else "It is within the fold noise on this small sample, but the "
        "nested number is still the one to report."
    )
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE stratified, walk-forward and grouped CV
# ════════════════════════════════════════════════════════════════════════

# ── 4a. Stratified k-fold on a rare outcome ─────────────────────────────
print_header("Stratified vs plain k-fold — credit default (rare outcome)")

plain_cv = KFold(n_splits=10, shuffle=True, random_state=SEED)
# TODO: a 10-fold StratifiedKFold (shuffle=True, random_state=SEED)
strat_cv = ____
plain_rates = [float(y_def[te].mean()) for _, te in plain_cv.split(X_def, y_def)]
# TODO: default rate of each stratified test fold (mirror plain_rates).
strat_rates = ____
logit = LogisticRegression(max_iter=2000)
# X_def is RAW: each fold re-fits the imputer and scaler on its own
# training rows. Fitting them once on all 600 rows would let every test
# fold's statistics into its training fold.
plain_auc = cross_val_auc_split_first(logit, X_def, y_def, cv=plain_cv)
strat_auc = cross_val_auc_split_first(logit, X_def, y_def, cv=strat_cv)
print(
    f"""
                     default rate per fold          AUC (mean ± sd)
  KFold              {min(plain_rates):.3f} – {max(plain_rates):.3f}                 {plain_auc.mean():.3f} ± {plain_auc.std():.3f}
  StratifiedKFold    {min(strat_rates):.3f} – {max(strat_rates):.3f}                 {strat_auc.mean():.3f} ± {strat_auc.std():.3f}
"""
)

fig_strat = go.Figure()
fig_strat.add_trace(go.Bar(x=list(range(1, 11)), y=plain_rates, name="KFold"))
fig_strat.add_trace(go.Bar(x=list(range(1, 11)), y=strat_rates, name="StratifiedKFold"))
fig_strat.add_hline(y=float(y_def.mean()), line_dash="dot", annotation_text="overall rate")
fig_strat.update_layout(
    title="Default rate in each test fold",
    xaxis_title="Fold",
    yaxis_title="Default rate",
    barmode="group",
)
print(f"Saved: {save_html_plot(fig_strat, 'ex2_04_stratified_folds.html')}")

# ── 4b. Walk-forward (TimeSeriesSplit) on time-ordered ICU admissions ──
print_header("TimeSeriesSplit + GroupKFold — ICU length of stay")
X_icu, y_icu, groups = icu["X"], icu["y"], icu["groups"]
admit_time = icu["admit_time"]

# TODO: a 5-split walk-forward splitter
tscv = ____
print("\nWalk-forward splits (rows are sorted by admit_time):")
for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_icu)):
    assert admit_time[int(tr_idx[-1])] <= admit_time[int(te_idx[0])]
    print(
        f"  Fold {fold + 1}: train {admit_time[0]:%Y-%m} → "
        f"{admit_time[int(tr_idx[-1])]:%Y-%m} ({len(tr_idx)} adm.), "
        f"test {admit_time[int(te_idx[0])]:%Y-%m} → "
        f"{admit_time[int(te_idx[-1])]:%Y-%m} ({len(te_idx)} adm.)"
    )

# ── 4c. GroupKFold: same patient never in both train and test ──────────
shuffled_kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
group_cv = GroupKFold(n_splits=5)
leaky_patients = []
for (tr_k, te_k), (tr_g, te_g) in zip(
    shuffled_kfold.split(X_icu), group_cv.split(X_icu, groups=groups)
):
    # TODO: count patients present in BOTH the KFold train and test rows.
    leaky_patients.append(____)
    assert not (set(groups[tr_g]) & set(groups[te_g])), "GroupKFold leaked a patient"
print(
    f"\nShuffled KFold: on average {np.mean(leaky_patients):.0f} patients per fold "
    "appear in BOTH train and test."
)
print("GroupKFold:     0 patients shared between train and test (verified).")

model = Ridge(alpha=1.0)
icu_scores = {
    "KFold (shuffled)": cross_val_score(model, X_icu, y_icu, cv=shuffled_kfold, scoring="r2"),
    "TimeSeriesSplit": cross_val_score(model, X_icu, y_icu, cv=tscv, scoring="r2"),
    # TODO: GroupKFold needs the patient ids passed as groups=
    "GroupKFold": ____,
}
print(f"\n{'Strategy':<18} {'R² (mean ± sd)':>18}")
print("-" * 38)
for name, sc in icu_scores.items():
    print(f"{name:<18} {sc.mean():>+9.4f} ± {sc.std():.4f}")

fig_cv = go.Figure()
for name, sc in icu_scores.items():
    fig_cv.add_trace(go.Box(y=sc, name=name, boxpoints="all"))
fig_cv.update_layout(
    title="ICU length-of-stay R² per fold under three CV strategies",
    yaxis_title="R² on the held-out fold",
)
print(f"Saved: {save_html_plot(fig_cv, 'ex2_04_cv_strategies.html')}")


# ── Checkpoint 2 ───────────────────────────────────────────────────────
assert all(len(sc) == 5 for sc in icu_scores.values()), "Each strategy: 5 scores"
assert max(strat_rates) - min(strat_rates) <= max(plain_rates) - min(plain_rates), (
    "Stratified folds should keep the default rate at least as even as KFold"
)
assert np.mean(leaky_patients) > 0, "Shuffled KFold should split some patients"
print("[ok] Checkpoint 2 passed — strategies match the data structure")

best_icu = max(sc.mean() for sc in icu_scores.values())
print(
    "  "
    + (
        f"Every strategy gives R² ≤ {best_icu:.3f}: in this dataset, length of "
        "stay is essentially unpredictable from admission-time demographics, "
        "so the strategies cannot disagree much. That is itself a finding — "
        "and the honest walk-forward / grouped estimates are the ones you "
        "would report."
        if best_icu < 0.05
        else "Compare the means: a k-fold score clearly above the walk-forward "
        "or grouped score means shuffled CV was leaking time or patients."
    )
)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: choosing the CV for a payments-fraud scorer
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Southeast Asian digital-wallet operator
# builds a real-time fraud scorer that sees each transaction once. The
# data has STRONG structure:
#   - Fraud patterns shift every few weeks (new attack vectors)
#   - Merchant mix changes with marketing campaigns
#   - Seasonal peaks (11.11, year-end sales, Chinese New Year) create
#     regime shifts that a shuffled k-fold mixes together
#   - Large merchants and repeat users contribute thousands of rows each
#
# WHY TIME-SERIES CV: walk-forward validation mirrors deployment — train
# on the past, score the future. If walk-forward performance degrades
# across successive folds, the model is ageing and needs refreshing.
#
# WHY GROUPKFOLD TOO: if one merchant (or user) sits in both train and
# test, the model can memorise that merchant and look better than it
# will on NEW merchants. Grouping by merchant forces the estimate to
# reflect generalisation to unseen merchants.
#
# WHY STRATIFIED + NESTED: fraud is rare, so fold-level fraud rates must
# be held steady; and every threshold or hyperparameter tuned on the
# evaluation folds inflates the reported recall — nested CV removes that.
#
# THE RISK OF GETTING IT WRONG (illustrative): a team that validates with
# shuffled k-fold can report a recall that the live system never
# reaches, because the validation folds contained the same time period,
# merchants and users as the training folds. The gap shows up only after
# deployment, as fraud losses.

print_header("Payments Fraud — matching CV to deployment")
print(
    """
Data property                     | CV strategy
----------------------------------|----------------------------------
Scores tomorrow's transactions    | TimeSeriesSplit (walk-forward)
Many rows per merchant / user     | GroupKFold (group = merchant/user)
Fraud is rare                     | Stratified folds (or stratified
                                  |   time blocks)
Threshold / hyperparameters tuned | Nested CV, or a final untouched
                                  |   time-ordered holdout
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

  [x] Nested CV: outer for honest eval, inner for α selection
  [x] Measuring optimism as (best standard-CV score - nested score)
  [x] Stratified k-fold: equal outcome rates in every fold
  [x] TimeSeriesSplit: walk-forward validation, no future leakage
  [x] GroupKFold: repeat patients stay on one side of the split
  [x] Picking a CV strategy based on DEPLOYMENT, not data shape

  KEY INSIGHT: The CV strategy is a MODELLING DECISION, not a
  technicality. The "best" model under shuffled k-fold can be the
  worst model in production if the data has structure.

  NEXT: 05_learning_curves.py — learning curves diagnose "do I need
  more data or a better model?" and tie the whole exercise together.
"""
)
