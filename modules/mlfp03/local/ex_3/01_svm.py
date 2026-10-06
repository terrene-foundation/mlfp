# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 3.1: Support Vector Machines (Linear + RBF)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Understand margin maximisation and the soft-margin C parameter
#   - Train a linear SVM and an RBF-kernel SVM with sklearn
#   - Sweep C across orders of magnitude and pick the best via CV AUC
#   - Judge a classifier against the majority-class baseline, not "random"
#   - Visualise the RBF decision boundary in 2D PCA space
#   - Translate SVM output into e-commerce churn dollars saved
#
# PREREQUISITES: MLFP03 Exercise 2 (bias-variance, regularisation,
#   cross-validation)
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — margin maximisation and the kernel trick
#   2. Build — linear + RBF SVM, C parameter sweep scored by CV AUC
#   3. Train — final RBF SVM on the full training set, vs the baseline
#   4. Visualise — C-sweep curves + 2D decision boundary
#   5. Apply — e-commerce churn cost-benefit
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from dotenv import load_dotenv
from sklearn.svm import SVC

from shared.mlfp03.ex_3 import (
    build_train_test_split,
    churn_saved_dollars,
    cv_scores,
    decision_boundary_mesh,
    fit_and_evaluate,
    OUTPUT_DIR,
    print_classification_report,
    project_2d,
    RANDOM_SEED,
    save_decision_boundaries,
    save_sweep_plot,
)

load_dotenv()

# ════════════════════════════════════════════════════════════════════════
# THEORY — Margin Maximisation and the Kernel Trick
# ════════════════════════════════════════════════════════════════════════
# An SVM finds the hyperplane that separates two classes with the LARGEST
# possible gap (margin) between them. Intuition: a wider gap means any
# small perturbation in the data is less likely to flip a prediction.
#
#   Hard-margin primal:  minimise ||w||² / 2
#                        subject to y_i (w . x_i + b) >= 1 for all i
#
#   Soft-margin: add slack variables ξ_i that let the SVM misclassify
#   points at a cost C per unit of slack.
#     C -> infinity  : hard margin, no misclassification tolerated
#     C -> 0         : very wide margin, many misclassifications tolerated
#
# The KERNEL TRICK lets the SVM learn a nonlinear boundary without ever
# materialising the high-dimensional feature map φ(x). It computes inner
# products in that space via a kernel K(x, x'):
#     Linear : K(x, x') = x . x'
#     RBF    : K(x, x') = exp(-γ ||x - x'||²)
#
# RBF can bend the decision surface around arbitrary clusters — but it
# only beats a linear kernel when the true boundary is actually curved.
# The sweep below lets the data decide.
#
# BASELINES, NOT "RANDOM": about 74% of customers in this dataset churned.
# A "model" that answers "churned" for everyone is already ~74% accurate
# and has a churn F1 of ~0.85. So we (a) choose C by ROC AUC, which
# measures ranking quality and cannot be gamed by predicting one class,
# and (b) compare accuracy against the majority-class baseline.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Load data, sweep C for linear and RBF kernels
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP03 Exercise 3.1 — Support Vector Machines")
print("=" * 70)

data = build_train_test_split()
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
cv = data["cv"]
feature_names = data["feature_names"]

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
print(f"Churn rate (train): {data['churn_rate']:.2%}")
print(
    f"Majority-class baseline on test: accuracy={data['majority_accuracy']:.4f}, "
    f"churn F1 of 'everyone churns'={data['majority_f1']:.4f}"
)

C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]


def sweep_c(kernel: str) -> dict[float, dict[str, float]]:
    """Sweep C for a given kernel and return {C: cv_scores(...)}."""
    results: dict[float, dict[str, float]] = {}
    print(f"\n--- {kernel.upper()} SVM: C parameter sweep (5-fold CV) ---")
    print(f"{'C':>10} {'CV Accuracy':>14} {'CV F1':>10} {'CV AUC':>10}")
    print("-" * 48)
    for c_val in C_VALUES:
        # TODO: score SVC(kernel=kernel, C=c_val, random_state=RANDOM_SEED)
        # with the shared helper cv_scores(estimator, X, y, cv).
        # Hint: it returns a dict with accuracy / f1 / auc_roc keys.
        scores = ____
        results[c_val] = scores
        print(
            f"{c_val:>10.2f} {scores['accuracy']:>14.4f} "
            f"{scores['f1']:>10.4f} {scores['auc_roc']:>10.4f}"
        )
    return results


linear_results = sweep_c("linear")
rbf_results = sweep_c("rbf")

# TODO: pick the C with the highest CV AUC for each kernel.
# Hint: max(results_dict, key=lambda c: ...["auc_roc"])
best_c_linear = ____
best_c_rbf = ____
print(
    f"\nBest C by CV AUC — linear: {best_c_linear} "
    f"(AUC {linear_results[best_c_linear]['auc_roc']:.4f}), "
    f"RBF: {best_c_rbf} (AUC {rbf_results[best_c_rbf]['auc_roc']:.4f})"
)
kernel_gap = rbf_results[best_c_rbf]["auc_roc"] - linear_results[best_c_linear]["auc_roc"]
print(
    f"RBF minus linear CV AUC: {kernel_gap:+.4f} — "
    + (
        "the curved boundary earns its extra cost here."
        if kernel_gap > 0.01
        else "no meaningful gain from bending the boundary on this data."
    )
)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: final RBF SVM on the full training set
# ════════════════════════════════════════════════════════════════════════

# TODO: fit and score the final RBF SVM with fit_and_evaluate(...).
# Hint: SVC(kernel="rbf", C=best_c_rbf, random_state=RANDOM_SEED,
#       probability=True); name it f"SVM (RBF, C={best_c_rbf})".
svm_result = ____

print(
    f"\n{svm_result['name']}: trained in {svm_result['train_time']:.2f}s | "
    f"accuracy={svm_result['accuracy']:.4f} | "
    f"F1={svm_result['f1']:.4f} | AUC={svm_result['auc_roc']:.4f}"
)
print(
    f"Accuracy lift over the majority baseline: "
    f"{svm_result['accuracy'] - data['majority_accuracy']:+.4f}"
)
print_classification_report(y_test, svm_result["pred"])

svm_model = svm_result["model"]
n_sv = int(svm_model.support_vectors_.shape[0])
sv_pct = n_sv / len(y_train)
print(
    f"Support vectors: {n_sv} ({sv_pct:.1%} of training). "
    f"A large share means the classes overlap heavily — many points sit "
    f"inside or on the wrong side of the margin."
)

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert svm_result["auc_roc"] > 0.6, "SVM must rank churners above retained customers"
assert best_c_rbf in C_VALUES, "Best C must come from the sweep"
print("\n[ok] Checkpoint 1 passed — SVM trained and evaluated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: C-sweep curves + 2D decision boundary
# ════════════════════════════════════════════════════════════════════════

sweep_out = save_sweep_plot(
    C_VALUES,
    {
        "linear CV AUC": [linear_results[c]["auc_roc"] for c in C_VALUES],
        "RBF CV AUC": [rbf_results[c]["auc_roc"] for c in C_VALUES],
        "linear CV accuracy": [linear_results[c]["accuracy"] for c in C_VALUES],
        "RBF CV accuracy": [rbf_results[c]["accuracy"] for c in C_VALUES],
    },
    x_label="C (log scale)",
    title="SVM: linear vs RBF — CV AUC / accuracy across C",
    fname="ex3_01_svm_c_sweep.html",
    log_x=True,
)
print(f"Saved: {sweep_out}")

pca_bundle = project_2d(X_train, X_test)
X_train_2d = pca_bundle["X_train_2d"]

# TODO: fit an RBF SVC with the best C on the 2D projection X_train_2d.
# Hint: SVC(kernel="rbf", C=best_c_rbf, random_state=RANDOM_SEED), then .fit
svm_2d = ____
svm_2d.fit(X_train_2d, y_train)

xx, yy = decision_boundary_mesh(X_train_2d)
# TODO: predict a class for every mesh point and reshape to the grid.
# Hint: svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z = ____

boundary_out = save_decision_boundaries(
    {f"RBF SVM (C={best_c_rbf})": Z},
    xx,
    yy,
    X_train_2d,
    y_train,
    fname="ex3_01_svm_boundary.html",
    title="RBF SVM decision regions in 2D PCA space (red = churned)",
)
print(f"Saved: {boundary_out}")
print(
    f"PCA variance captured by the 2 plotted axes: "
    f"{pca_bundle['explained_variance'].sum():.2%} — the 2D picture is an "
    f"intuition aid, not the model the metrics above describe."
)

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert Z.shape == xx.shape, "Decision mesh must match the grid"
assert boundary_out.exists(), "Decision-boundary figure must be written"
print("[ok] Checkpoint 2 passed — 2D decision boundary rendered\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: e-commerce churn cost-benefit
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A mid-market regional e-commerce marketplace has ~250K active
# customers per month. The retention team runs a targeted promo campaign
# (S$18 offer) against customers the model flags as likely churners.
# (All business figures here are illustrative teaching assumptions.)
#
# Why SVM is a candidate:
#   - The feature space is small (11 behavioural / profile features),
#     where kernel methods are tractable and well-behaved.
#   - The SVM decision value says how far a customer sits from the
#     boundary, which can be used to RANK flags. It is NOT a calibrated
#     probability — that is why sklearn fits an extra Platt-scaling step
#     when you ask for probability=True (Lesson 3.5 covers calibration).
#   - Training cost is a one-off nightly batch job — O(n²) is tolerable
#     at the subsampled 4K training size.
#
# LIMITATIONS:
#   - An RBF SVM is hard to explain feature-by-feature; retention teams
#     usually want a reason they can act on, not a kernel distance.
#   - Scaling beyond ~50K training samples makes the O(n²) kernel matrix
#     impractical. For the full customer base, move to Random Forest or
#     gradient boosting.
#   - With ~74% of customers churning, flagging is only useful if the
#     model separates the uncertain middle — read the classification
#     report, not just the headline accuracy.

# TODO: count true positives (pred == 1 AND y_test == 1) and convert
# them to dollars with churn_saved_dollars(...).
# Hint: int(((svm_result["pred"] == 1) & (y_test == 1)).sum())
true_positives = ____
dollars_saved = ____
print(f"\nBusiness impact on held-out test set ({len(y_test)} customers):")
print(f"  True positives (churners caught): {true_positives}")
print(f"  Net retention value at 40% offer acceptance: S${dollars_saved:,.2f}")
print(
    f"  Extrapolated to the 250K monthly active base "
    f"(identical churn rate): "
    f"S${dollars_saved * (250_000 / len(y_test)):,.0f} / month"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Margin maximisation and the C parameter trade-off
  [x] Linear vs RBF kernels — CV AUC gap: {kernel_gap:+.4f}
  [x] CV-driven C selection across five orders of magnitude, by AUC
  [x] Held-out accuracy {svm_result['accuracy']:.4f} vs majority baseline
      {data['majority_accuracy']:.4f}; AUC {svm_result['auc_roc']:.4f}
  [x] 2D PCA decision boundary for visual intuition
  [x] Translated classifier output into S${dollars_saved:,.0f} of retained
      customer value on the held-out test fold

  Next: 02_knn.py — instance-based learning with no training phase.
"""
)
