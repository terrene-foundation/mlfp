# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 3.6: The Model Zoo — head-to-head comparison
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Compare all five classical model families on the SAME CV folds
#   - Tune each family inside the comparison (nested CV) so no model is
#     judged with hand-picked or test-tuned hyperparameters
#   - Read a comparison table as mean ± spread, not as a single number
#   - Overlay decision boundaries in 2D PCA space to see SHAPE differences
#   - Publish a "when to use which model" decision guide and quantify the
#     dollar impact of each model on the e-commerce churn scenario
#
# PREREQUISITES: 01_svm through 05_random_forest; Exercise 2.4 (nested CV)
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — why model selection is a multi-criteria decision
#   2. Build — one estimator + small tuning grid per family
#   3. Train — nested CV on shared outer folds; refit winners on train
#   4. Visualise — decision boundaries + metric comparison chart
#   5. Apply — when-to-use guide + dollars saved ranking
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from shared.mlfp03.ex_3 import (
    build_train_test_split,
    churn_saved_dollars,
    decision_boundary_mesh,
    fit_and_evaluate,
    project_2d,
    RANDOM_SEED,
    save_decision_boundaries,
    save_metric_comparison,
)

load_dotenv()

# ════════════════════════════════════════════════════════════════════════
# THEORY — Model selection is multi-criteria
# ════════════════════════════════════════════════════════════════════════
# No model wins on every axis at once. You trade off:
#   - Ranking quality (AUC) and accuracy vs the majority baseline
#   - Training time and prediction time
#   - Interpretability (can you explain a single prediction?)
#   - Robustness to drift (does small data shift the model dramatically?)
#
# A FAIR comparison has three rules:
#   1. Same data, same preprocessing, same CV folds for every model.
#   2. Every model gets its hyperparameters tuned the same way — inside
#      each training fold (nested CV), never on the folds used to score it
#      and never on the test set.
#   3. Report the spread across folds. If two models' means differ by
#      less than their fold-to-fold standard deviation, the data cannot
#      tell them apart — choose on speed or interpretability instead.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: one estimator + a small tuning grid per family
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP03 Exercise 3.6 — Classical ML Zoo — head-to-head")
print("=" * 70)

data = build_train_test_split()
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
outer_cv = data["cv"]  # the SAME 5 folds every earlier file used
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
print(f"Majority-class baseline accuracy (test): {data['majority_accuracy']:.4f}")

# The grids cover the ranges the earlier technique files found useful.
# GaussianNB has no hyperparameter worth tuning here, so its grid is empty.
zoo: dict[str, tuple[object, dict[str, list]]] = {
    "SVM (RBF)": (
        SVC(kernel="rbf", random_state=RANDOM_SEED),
        {"C": [0.01, 0.1, 1.0]},
    ),
    # TODO: KNeighborsClassifier() with n_neighbors grid [21, 51, 101]
    "KNN": ____,
    # TODO: GaussianNB() with an empty grid {}
    "Naive Bayes": ____,
    # TODO: DecisionTreeClassifier(random_state=RANDOM_SEED) with
    # max_depth grid [2, 3, 5, 7]
    "Decision Tree": ____,
    "Random Forest": (
        RandomForestClassifier(
            n_estimators=200, max_features="sqrt", random_state=RANDOM_SEED, n_jobs=-1
        ),
        {"min_samples_leaf": [1, 5, 20]},
    ),
}


def tuned(estimator: object, grid: dict[str, list]) -> object:
    """Wrap an estimator so it tunes itself (by AUC) on whatever data it is fit on."""
    if not grid:
        return estimator
    # TODO: GridSearchCV over `grid` using the INNER folds, scored by "roc_auc".
    return ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: nested CV on the shared outer folds
# ════════════════════════════════════════════════════════════════════════
# Outer loop (outer_cv, 5 folds): scores each family on data it never saw.
# Inner loop (inner_cv, 3 folds, inside each outer training fold): picks
# that family's hyperparameters. This is the nested CV from Exercise 2.4.

cv_rows: list[dict] = []
for name, (est, grid) in zoo.items():
    # TODO: cross_validate the TUNED estimator on the shared outer folds.
    # Hint: cross_validate(tuned(est, grid), X_train, y_train, cv=outer_cv,
    #       scoring=("accuracy", "f1", "roc_auc"), return_estimator=True)
    r = ____
    chosen = (
        [str(fold_est.best_params_) for fold_est in r["estimator"]] if grid else ["—"]
    )
    cv_rows.append(
        {
            "name": name,
            "auc": float(r["test_roc_auc"].mean()),
            "auc_std": float(r["test_roc_auc"].std()),
            "accuracy": float(r["test_accuracy"].mean()),
            "f1": float(r["test_f1"].mean()),
            "fit_time": float(r["fit_time"].mean()),
            "chosen": sorted(set(chosen)),
        }
    )

cv_rows.sort(key=lambda row: row["auc"], reverse=True)
print("\n--- Nested-CV comparison (5 shared outer folds) ---")
print(
    f"{'Model':<15} {'CV AUC (mean±std)':>18} {'CV Acc':>8} {'CV F1':>7} "
    f"{'fit s/fold':>11}  hyperparameters chosen per fold"
)
print("-" * 100)
for row in cv_rows:
    print(
        f"{row['name']:<15} {row['auc']:>10.4f} ± {row['auc_std']:.4f} "
        f"{row['accuracy']:>8.4f} {row['f1']:>7.4f} {row['fit_time']:>11.3f}  "
        f"{', '.join(row['chosen'])}"
    )

leader, runner_up = cv_rows[0], cv_rows[1]
gap = leader["auc"] - runner_up["auc"]
print(
    f"\nLeader by CV AUC: {leader['name']} ({leader['auc']:.4f}). "
    f"Gap to {runner_up['name']}: {gap:.4f} vs fold-to-fold std "
    f"{leader['auc_std']:.4f} — "
    + (
        "a real difference."
        if gap > leader["auc_std"]
        else "within noise: the data cannot separate them, so speed and "
        "interpretability should decide."
    )
)

# Refit every tuned family on the FULL training set, then score each ONCE
# on the untouched test set (needed for the dollar-impact table).
results: list[dict] = []
fitted: dict[str, object] = {}
for name, (est, grid) in zoo.items():
    # TODO: fit_and_evaluate(...) the tuned estimator on train, score on test.
    r = ____
    fitted[name] = r["model"].best_estimator_ if grid else r["model"]
    results.append(r)

print("\n--- Held-out test set (each tuned model scored once) ---")
print(f"{'Model':<15} {'Accuracy':>10} {'F1':>8} {'AUC-ROC':>9} {'refit s':>9}")
print("-" * 55)
for r in results:
    print(
        f"{r['name']:<15} {r['accuracy']:>10.4f} {r['f1']:>8.4f} "
        f"{r['auc_roc']:>9.4f} {r['train_time']:>9.2f}"
    )

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert len(cv_rows) == 5 and len(results) == 5, "All five classical models must run"
assert all(row["auc"] > 0.6 for row in cv_rows), "Every model must rank better than chance"
print("\n[ok] Checkpoint 1 passed — all 5 models compared on identical folds\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: decision boundaries + metric comparison
# ════════════════════════════════════════════════════════════════════════

pca_bundle = project_2d(X_train, X_test)
X_train_2d = pca_bundle["X_train_2d"]
print(
    f"PCA variance explained by the 2 plotted axes: "
    f"{pca_bundle['explained_variance'].sum():.2%}"
)

xx, yy = decision_boundary_mesh(X_train_2d)
grid_points = np.c_[xx.ravel(), yy.ravel()]
panels: dict[str, np.ndarray] = {}
for name, model in fitted.items():
    # Same family + same tuned hyperparameters, re-fit on the 2D projection
    # so every boundary is drawn on identical axes.
    # TODO: clone the model with type(model)(**model.get_params()), fit it on
    # X_train_2d, then predict grid_points and reshape to xx.shape.
    model_2d = ____
    panels[name] = ____

boundary_path = save_decision_boundaries(
    panels,
    xx,
    yy,
    X_train_2d,
    y_train,
    fname="ex3_06_zoo_boundaries.html",
    title="Five model families, one 2D PCA view (red = churned)",
)
print(f"Saved: {boundary_path}")

metric_dict = {
    row["name"]: {"CV AUC": row["auc"], "CV Accuracy": row["accuracy"], "CV F1": row["f1"]}
    for row in cv_rows
}
comparison_path = save_metric_comparison(metric_dict, "ex3_06_zoo_comparison.html")
print(f"Saved: {comparison_path}")

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert len(panels) == 5, "All 5 boundaries computed"
assert boundary_path.exists() and comparison_path.exists(), "Figures must be written"
print("[ok] Checkpoint 2 passed — boundaries and comparison chart rendered\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: when-to-use guide + dollars saved
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 76)
print("  WHEN TO USE EACH MODEL — e-commerce churn playbook")
print("=" * 76)
print(
    """
+-------------------+---------------------+---------------------+---------------+
| Model             | Best when           | Avoid when          | Key tradeoff  |
+-------------------+---------------------+---------------------+---------------+
| SVM (RBF)         | Small-to-mid n,     | Very large n        | Flexible, but |
|                   | curved boundary     | (O(n^2) kernel)     | slow + opaque |
+-------------------+---------------------+---------------------+---------------+
| KNN               | Small n, cold-start,| High-dim feature    | Zero training,|
|                   | low-ceremony        | space               | slow predict  |
+-------------------+---------------------+---------------------+---------------+
| Naive Bayes       | High volume,        | Correlated features | Tiny memory,  |
|                   | fast baseline       | need calibrated p   | strong bias   |
+-------------------+---------------------+---------------------+---------------+
| Decision Tree     | Every decision must | Noisy data          | Interpretable,|
|                   | be readable         | (high variance)     | unstable      |
+-------------------+---------------------+---------------------+---------------+
| Random Forest     | Default tabular     | Need per-prediction | Robust, but   |
|                   | workhorse           | explanations        | black-box     |
+-------------------+---------------------+---------------------+---------------+
(Business figures below are illustrative teaching assumptions.)
"""
)

print("\n--- Dollar impact ranking (held-out test set) ---")
print(f"{'Model':<15} {'TP':>6} {'S$ saved':>14} {'Monthly S$ @250K':>18}")
print("-" * 57)
impact_rows = []
for r in results:
    # TODO: count true positives and convert with churn_saved_dollars(...).
    tp = ____
    saved = ____
    monthly_scale = saved * (250_000 / len(y_test))
    impact_rows.append({"name": r["name"], "tp": tp, "monthly_scale": monthly_scale})
    print(f"{r['name']:<15} {tp:>6} S${saved:>11,.2f} S${monthly_scale:>15,.0f}")

best_dollar = max(impact_rows, key=lambda row: row["monthly_scale"])
print(
    f"\nHighest dollar impact: {best_dollar['name']} "
    f"(S${best_dollar['monthly_scale']:,.0f}/mo). Caution: this simple value "
    f"model counts caught churners but charges nothing for offers sent to "
    f"customers who would have stayed — a model that flags EVERYONE maximises "
    f"it. Lesson 3.5 replaces it with a full cost matrix."
)
print(f"Highest CV AUC: {leader['name']} (AUC={leader['auc']:.4f})")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — km.diagnose
# ════════════════════════════════════════════════════════════════════════
# This lesson built five classical models from primitives — tuning each,
# timing each, building a comparison table, mapping decision boundaries.
# The kailash-ml SDK packages the diagnostic surface (per-class metrics,
# class-balance severity, confusion matrix, accuracy heuristics) into a
# single call.

from kailash_ml import diagnose

# `kind="classical_classifier"` dispatches to the sklearn ClassifierMixin
# adapter; `data=(X, y)` is the held-out pair. Use the CV-AUC leader,
# already refit on the full training set above.
best_model = fitted[leader["name"]]
report = diagnose(
    best_model, kind="classical_classifier", data=(X_test, y_test), show=False
)
print()
print(f"  km.diagnose model    : {leader['name']}")
print(f"  km.diagnose metrics  : {report.metrics}")
print(f"  km.diagnose severity : {report.severity}")
print()
print("km.diagnose: 1 call -> the same diagnostic surface the lesson body")
print("hand-rolled. Destination-first: when the journey is internalised,")
print("the SDK is one line.")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Compared 5 classical model families on the same 5 outer CV folds
  [x] Tuned every family inside each training fold (nested CV) — no
      hand-picked or test-tuned hyperparameters
  [x] Read the table as mean ± std: leader {leader['name']} by
      {gap:.4f} AUC vs a fold std of {leader['auc_std']:.4f}
  [x] Mapped decision boundaries in 2D PCA space across all 5 models
  [x] Scored each tuned model once on the held-out test set
  [x] Published a "when to use which model" guide

  KEY INSIGHT: when the families are within noise of each other — as
  they often are on modest tabular data — the decision moves to cost,
  latency and explainability. Let the evidence, not habit, pick.

  NEXT: Exercise 4 — gradient boosting (XGBoost, LightGBM, CatBoost):
  tree ensembles that build each tree to correct the previous one's
  errors, usually the strongest family on tabular data.
"""
)
