# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 3.2: K-Nearest Neighbors
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Understand instance-based (lazy) learning: no training phase
#   - Sweep k and compare Euclidean / Manhattan / Cosine distance metrics
#   - Recognise the curse of dimensionality and why scaling matters
#   - Visualise the jagged KNN decision boundary in 2D PCA space
#   - Apply KNN to a cold-start e-commerce churn scenario
#
# PREREQUISITES: 01_svm.py (shared preprocessing pipeline, CV AUC,
#   majority-class baseline)
#
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Theory — lazy learning, distance metrics, curse of dimensionality
#   2. Build — k sweep, distance metric comparison (scored by CV AUC)
#   3. Train — final KNN with the best (k, metric) pair
#   4. Visualise — k-sweep curve + 2D decision boundary (jagged)
#   5. Apply — cold-start churn flagging for a new marketplace
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from dotenv import load_dotenv
from sklearn.neighbors import KNeighborsClassifier

from shared.mlfp03.ex_3 import (
    build_train_test_split,
    churn_saved_dollars,
    cv_scores,
    decision_boundary_mesh,
    fit_and_evaluate,
    OUTPUT_DIR,
    print_classification_report,
    project_2d,
    save_decision_boundaries,
    save_sweep_plot,
)

load_dotenv()

# ════════════════════════════════════════════════════════════════════════
# THEORY — Lazy Learning and Distance
# ════════════════════════════════════════════════════════════════════════
# KNN has NO training phase. Fit() just memorises the training set.
# Predict() searches the k closest training points and takes a majority
# vote (predict_proba = the fraction of the k neighbours that churned).
#
#   Prediction: y_hat = mode({y_j : j in k nearest neighbors of x})
#
# Distance metrics:
#     Euclidean  = sqrt(Σ (x_i - y_i)²)        — sphere of influence
#     Manhattan  = Σ |x_i - y_i|                — axis-aligned grid
#     Cosine     = 1 - (x . y) / (||x|| ||y||)  — direction, not magnitude
#
# CURSE OF DIMENSIONALITY: as the feature count grows, ALL pairs of
# points become roughly equidistant — the notion of "nearest" breaks
# down. KNN is therefore strong on small, low-dimensional data and
# weak once you pass ~50 features.
#
# SCALING IS MANDATORY: a feature with range [0, 100000] will dominate
# every distance computation. The shared preprocessing pipeline already
# applies z-score normalisation before we see the data.
#
# k IS A BIAS-VARIANCE KNOB: k=1 memorises every training point (high
# variance); very large k averages over half the dataset (high bias).


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: k sweep + distance metric comparison
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP03 Exercise 3.2 — K-Nearest Neighbors")
print("=" * 70)

data = build_train_test_split()
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
cv = data["cv"]

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
print(f"Majority-class baseline accuracy (test): {data['majority_accuracy']:.4f}")

K_VALUES = [1, 3, 5, 11, 21, 31, 51, 101]
METRICS = ["euclidean", "manhattan", "cosine"]

print("\n--- k sweep (euclidean distance, 5-fold CV) ---")
print(f"{'k':>6} {'CV Accuracy':>14} {'CV F1':>10} {'CV AUC':>10}")
print("-" * 44)
k_results: dict[int, dict[str, float]] = {}
for k in K_VALUES:
    scores = cv_scores(
        KNeighborsClassifier(n_neighbors=k, metric="euclidean"),
        X_train,
        y_train,
        cv,
    )
    k_results[k] = scores
    print(
        f"{k:>6} {scores['accuracy']:>14.4f} {scores['f1']:>10.4f} "
        f"{scores['auc_roc']:>10.4f}"
    )

best_k = max(k_results, key=lambda k: k_results[k]["auc_roc"])
print(f"\nBest k by CV AUC: {best_k}")
print(
    f"k=1 CV AUC {k_results[1]['auc_roc']:.4f} vs best {k_results[best_k]['auc_roc']:.4f} "
    f"— a single neighbour copies the noise of one customer."
)

print(f"\n--- distance metric sweep (k={best_k}) ---")
print(f"{'metric':<12} {'CV Accuracy':>14} {'CV F1':>10} {'CV AUC':>10}")
print("-" * 50)
metric_results: dict[str, dict[str, float]] = {}
for m in METRICS:
    scores = cv_scores(
        KNeighborsClassifier(n_neighbors=best_k, metric=m),
        X_train,
        y_train,
        cv,
    )
    metric_results[m] = scores
    print(
        f"{m:<12} {scores['accuracy']:>14.4f} {scores['f1']:>10.4f} "
        f"{scores['auc_roc']:>10.4f}"
    )

best_metric = max(metric_results, key=lambda m: metric_results[m]["auc_roc"])
print(f"\nBest metric by CV AUC: {best_metric}")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: final KNN
# ════════════════════════════════════════════════════════════════════════

knn_result = fit_and_evaluate(
    KNeighborsClassifier(n_neighbors=best_k, metric=best_metric),
    X_train,
    y_train,
    X_test,
    y_test,
    name=f"KNN (k={best_k}, {best_metric})",
)

print(
    f"\n{knn_result['name']}: 'trained' in {knn_result['train_time']:.4f}s | "
    f"accuracy={knn_result['accuracy']:.4f} | "
    f"F1={knn_result['f1']:.4f} | AUC={knn_result['auc_roc']:.4f}"
)
print(
    f"Accuracy lift over the majority baseline: "
    f"{knn_result['accuracy'] - data['majority_accuracy']:+.4f}"
)
print_classification_report(y_test, knn_result["pred"])

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert knn_result["auc_roc"] > 0.6, "KNN must rank churners above retained customers"
assert best_k > 1, "Best k should be > 1 (k=1 always overfits)"
print("[ok] Checkpoint 1 passed — KNN trained and evaluated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: k-sweep curve + 2D decision boundary (expect jagged)
# ════════════════════════════════════════════════════════════════════════

sweep_out = save_sweep_plot(
    K_VALUES,
    {
        "CV AUC": [k_results[k]["auc_roc"] for k in K_VALUES],
        "CV accuracy": [k_results[k]["accuracy"] for k in K_VALUES],
    },
    x_label="k (number of neighbours, log scale)",
    title="KNN: CV AUC / accuracy vs k",
    fname="ex3_02_knn_k_sweep.html",
    log_x=True,
)
print(f"Saved: {sweep_out}")

pca_bundle = project_2d(X_train, X_test)
X_train_2d = pca_bundle["X_train_2d"]

xx, yy = decision_boundary_mesh(X_train_2d)
grid = np.c_[xx.ravel(), yy.ravel()]
panels: dict[str, np.ndarray] = {}
for k_show in (1, best_k):
    knn_2d = KNeighborsClassifier(n_neighbors=k_show, metric=best_metric)
    knn_2d.fit(X_train_2d, y_train)
    panels[f"KNN k={k_show}"] = knn_2d.predict(grid).reshape(xx.shape)

boundary_out = save_decision_boundaries(
    panels,
    xx,
    yy,
    X_train_2d,
    y_train,
    fname="ex3_02_knn_boundary.html",
    title="KNN decision regions in 2D PCA space: k=1 vs the CV-best k",
)
print(f"Saved: {boundary_out}")
print(
    f"PCA variance captured: {pca_bundle['explained_variance'].sum():.2%}. "
    f"Compare the panels: k=1 carves an island around every training "
    f"customer; the CV-best k smooths those islands away."
)

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert all(Z.shape == xx.shape for Z in panels.values()), "Mesh must match grid"
assert boundary_out.exists(), "Decision-boundary figure must be written"
print("[ok] Checkpoint 2 passed — KNN 2D boundaries rendered\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: cold-start churn for a new marketplace
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A new regional marketplace (e.g. a niche B2C fashion
# platform) has only a few thousand labelled customers in its first six
# months. The data science team needs a churn model that works from
# day one, even before enough data exists to justify a deep model.
# (Business figures are illustrative teaching assumptions.)
#
# Why KNN fits:
#   - Zero training cost. Add new customers to the reference set and
#     predict immediately.
#   - Low ceremony: k and the distance metric are the only real choices.
#   - Low-dimensional (11 features) means the curse of dimensionality is
#     mild and distance still means something.
#
# LIMITATIONS:
#   - Prediction cost grows linearly with dataset size — at 250K
#     customers, every prediction scans 250K vectors (use an index).
#   - Anisotropic feature space: unevenly-scaled or correlated features
#     distort the distance metric. We mitigate via z-score normalisation.
#   - Hard to explain individual predictions. Retention teams prefer
#     "feature X was high" over "this customer resembled 31 past ones".

true_positives = int(((knn_result["pred"] == 1) & (y_test == 1)).sum())
dollars_saved = churn_saved_dollars(true_positives)
print(f"\nBusiness impact on held-out test set ({len(y_test)} customers):")
print(f"  True positives (churners caught): {true_positives}")
print(f"  Net retention value at 40% offer acceptance: S${dollars_saved:,.2f}")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Lazy learning: fit() is trivial, predict() does the work
  [x] k selection via CV AUC — k=1 overfits, very large k under-fits
  [x] Distance metric comparison (Euclidean / Manhattan / Cosine)
  [x] Held-out accuracy {knn_result['accuracy']:.4f} vs majority baseline
      {data['majority_accuracy']:.4f}; AUC {knn_result['auc_roc']:.4f}
  [x] Jagged k=1 vs smoothed best-k 2D decision boundaries
  [x] Cold-start churn business case — S${dollars_saved:,.0f} retained on test

  Next: 03_naive_bayes.py — Bayes theorem applied to classification.
"""
)
