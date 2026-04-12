# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 3: The Complete Supervised Model Zoo
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Train and compare SVM, KNN, Naive Bayes, Decision Trees, and
#     Random Forests using a consistent evaluation framework
#   - Compute Gini impurity and information gain by hand to understand
#     how decision trees choose splits
#   - Use OOB (out-of-bag) estimation as a free cross-validation proxy
#   - Select the appropriate algorithm based on data characteristics
#     (size, dimensionality, interpretability requirements)
#   - Build a model comparison table and justify the best choice
#
# PREREQUISITES:
#   - MLFP03 Exercise 2 (bias-variance, regularisation)
#   - MLFP02 Module (statistics, Bayesian priors — connects to Naive Bayes)
#
# ESTIMATED TIME: 75-90 minutes
#
# TASKS:
#   1. Load e-commerce data, set up binary classification (churn)
#   2. Preprocess: encode categoricals, scale numerics
#   3. Train SVM (linear + RBF), tune C parameter
#   4. Train KNN, experiment with k values and distance metrics
#   5. Train Naive Bayes (GaussianNB)
#   6. Train Decision Tree, compute Gini impurity manually
#   7. Train Random Forest, extract feature importance, OOB score
#   8. Compare all 5 model families: accuracy, F1, training time
#   9. Discuss when to use each model
#
# DATASET: E-commerce customer data (mlfp03/ecommerce_customers.parquet)
#   Target: churned (binary — 0=retained, 1=churned)
#   Rows: ~5,000 customers | Features: behavioural + demographic
#   Why this dataset: realistic churn rates, mixed feature types
#
# THEORY:
#   SVM margin: maximise 2/||w|| subject to y_i(w . x_i + b) >= 1
#   Kernel trick: K(x, x') = phi(x) . phi(x') without computing phi
#   Gini impurity: G = 1 - Sum(p_k^2) for k classes
#   Information gain: IG = H(parent) - Sum(w_j * H(child_j))
#   OOB estimation: ~36.8% of samples excluded per bootstrap sample
#     (probability of NOT being drawn in n trials = (1 - 1/n)^n -> 1/e)
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_auc_score,
)

from kailash_ml import PreprocessingPipeline, ModelVisualizer
from kailash_ml.interop import to_sklearn_input

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────

loader = MLFPDataLoader()
ecommerce = loader.load("mlfp03", "ecommerce_customers.parquet")

print("=== E-Commerce Customer Dataset ===")
print(f"Shape: {ecommerce.shape}")
print(f"Columns: {ecommerce.columns}")
print(f"Churn rate: {ecommerce['churned'].mean():.2%}")

# Drop text columns and high-cardinality strings not suitable for the zoo
ecommerce = ecommerce.drop("customer_id", "review_text", "product_categories")

print(f"\nAfter dropping text/ID columns: {ecommerce.shape}")
print(f"Remaining columns: {ecommerce.columns}")


# ════════════════════════════════════════════════════════════════════════
# TASK 1: Set up classification task
# ════════════════════════════════════════════════════════════════════════
# Target: churned (0 = retained, 1 = churned)
# This is a binary classification problem.

target_col = "churned"

# TODO: Set up PreprocessingPipeline with: normalize=True, normalize_method="zscore",
#       categorical_encoding="ordinal", imputation_strategy="median", train_size=0.8, seed=42
pipeline = PreprocessingPipeline()
result = pipeline.setup(
    data=ecommerce,
    target=target_col,
    train_size=____,  # Hint: 0.8
    seed=____,  # Hint: 42
    normalize=____,  # Hint: True — SVM and KNN require scaled features
    normalize_method=____,  # Hint: "zscore"
    categorical_encoding=____,  # Hint: "ordinal"
    imputation_strategy=____,  # Hint: "median"
)

print(f"\nTask type: {result.task_type}")
print(f"Train: {result.train_data.shape}, Test: {result.test_data.shape}")

feature_cols = [c for c in result.train_data.columns if c != target_col]

X_train, y_train, col_info = to_sklearn_input(
    result.train_data,
    feature_columns=feature_cols,
    target_column=target_col,
)
X_test, y_test, _ = to_sklearn_input(
    result.test_data,
    feature_columns=feature_cols,
    target_column=target_col,
)
feature_names = col_info["feature_columns"]

print(f"Features ({len(feature_names)}): {feature_names}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train churn rate: {y_train.mean():.2%}")

# Cross-validation strategy — same folds for all models
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert X_train.shape[0] > 0, "Training set is empty"
assert X_test.shape[0] > 0, "Test set is empty"
assert len(feature_names) > 0, "No feature columns found"
assert 0 < y_train.mean() < 1, "Target should be binary with mixed labels"
# INTERPRETATION: Churn prediction is a canonical binary classification task.
# You're predicting which customers will leave. The cost of a missed churner
# (False Negative) typically far exceeds the cost of wrongly flagging a loyal
# customer (False Positive), so F1 and AUC matter more than raw accuracy.
print("\n✓ Checkpoint 1 passed — data prepared for 5-model comparison\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2: Preprocessing verification
# ════════════════════════════════════════════════════════════════════════

print("\n=== Feature Scales (post-normalisation) ===")
print(f"{'Feature':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 62)
for i, name in enumerate(feature_names):
    col = X_train[:, i]
    print(f"{name:<30} {col.mean():>8.3f} {col.std():>8.3f} {col.min():>8.3f} {col.max():>8.3f}")

print("\nAll features normalised to z-scores (mean ~0, std ~1).")
print("This is essential for SVM (margin depends on scale) and KNN (distance).")


# ════════════════════════════════════════════════════════════════════════
# TASK 3: SVM — Support Vector Machines
# ════════════════════════════════════════════════════════════════════════
# THEORY: SVM finds the hyperplane that maximises the margin between classes.
#   Hard-margin SVM: maximise 2/||w|| s.t. y_i(w . x_i + b) >= 1
#   Soft-margin SVM: allow misclassification via slack variables xi_i,
#     controlled by C (penalty parameter).
#     - C -> inf: hard margin (no misclassification allowed)
#     - C -> 0: wide margin, many misclassifications tolerated

print("\n=== SVM: Support Vector Machines ===")

# 3a: Linear SVM — tune C parameter
print("\n--- Linear SVM: C parameter sweep ---")
print(f"{'C':>10} {'CV Accuracy':>14} {'CV F1':>10}")
print("-" * 38)

c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
linear_svm_results = {}
for c_val in c_values:
    # TODO: Create SVC with kernel="linear", C=c_val, random_state=42
    svm_lin = ____  # Hint: SVC(kernel="linear", C=c_val, random_state=42)
    # TODO: Cross-validate accuracy and F1 on X_train, y_train using cv
    acc_scores = ____  # Hint: cross_val_score(svm_lin, X_train, y_train, cv=cv, scoring="accuracy")
    f1_scores = ____  # Hint: cross_val_score(svm_lin, X_train, y_train, cv=cv, scoring="f1")
    linear_svm_results[c_val] = {
        "accuracy": acc_scores.mean(),
        "f1": f1_scores.mean(),
    }
    print(f"{c_val:>10.2f} {acc_scores.mean():>14.4f} {f1_scores.mean():>10.4f}")

best_c_linear = max(linear_svm_results, key=lambda c: linear_svm_results[c]["f1"])
print(f"\nBest linear SVM: C={best_c_linear} (F1={linear_svm_results[best_c_linear]['f1']:.4f})")

# 3b: RBF SVM — tune C parameter
print("\n--- RBF SVM: C parameter sweep ---")
print(f"{'C':>10} {'CV Accuracy':>14} {'CV F1':>10}")
print("-" * 38)

rbf_svm_results = {}
for c_val in c_values:
    # TODO: Create SVC with kernel="rbf", C=c_val, random_state=42
    svm_rbf = ____  # Hint: SVC(kernel="rbf", C=c_val, random_state=42)
    # TODO: Cross-validate accuracy and F1
    acc_scores = ____  # Hint: cross_val_score(svm_rbf, X_train, y_train, cv=cv, scoring="accuracy")
    f1_scores = ____  # Hint: cross_val_score(svm_rbf, X_train, y_train, cv=cv, scoring="f1")
    rbf_svm_results[c_val] = {
        "accuracy": acc_scores.mean(),
        "f1": f1_scores.mean(),
    }
    print(f"{c_val:>10.2f} {acc_scores.mean():>14.4f} {f1_scores.mean():>10.4f}")

best_c_rbf = max(rbf_svm_results, key=lambda c: rbf_svm_results[c]["f1"])
print(f"\nBest RBF SVM: C={best_c_rbf} (F1={rbf_svm_results[best_c_rbf]['f1']:.4f})")

# Train final SVM on full training set
t0 = time.perf_counter()
# TODO: Train final SVC with kernel="rbf", C=best_c_rbf, random_state=42, probability=True
svm_final = ____  # Hint: SVC(kernel="rbf", C=best_c_rbf, random_state=42, probability=True)
# TODO: Fit svm_final on X_train, y_train
____  # Hint: svm_final.fit(X_train, y_train)
svm_train_time = time.perf_counter() - t0

# TODO: Predict classes and probabilities on X_test
svm_pred = ____  # Hint: svm_final.predict(X_test)
svm_prob = ____  # Hint: svm_final.predict_proba(X_test)[:, 1]

print(f"\nSVM Final (RBF, C={best_c_rbf}): trained in {svm_train_time:.2f}s")
print(classification_report(y_test, svm_pred, target_names=["Retained", "Churned"]))

print("SVM Insights:")
print(f"  Support vectors: {svm_final.n_support_} (per class)")
print(f"  Total support vectors: {svm_final.support_vectors_.shape[0]} "
      f"({svm_final.support_vectors_.shape[0] / len(y_train):.1%} of training data)")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
svm_acc = accuracy_score(y_test, svm_pred)
assert svm_acc > 0.5, f"SVM accuracy {svm_acc:.4f} should be above random baseline"
# INTERPRETATION: The number of support vectors tells you how complex the
# boundary is. If 80% of training samples are support vectors, the model
# has learned almost nothing general — it has memorised the training set.
# A well-trained SVM typically uses 10-30% of training points as support vectors.
print("\n✓ Checkpoint 2 passed — SVM trained and evaluated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4: KNN — K-Nearest Neighbors
# ════════════════════════════════════════════════════════════════════════

print("\n=== KNN: K-Nearest Neighbors ===")

# 4a: Vary k
print("\n--- KNN: k value sweep (Euclidean distance) ---")
print(f"{'k':>6} {'CV Accuracy':>14} {'CV F1':>10}")
print("-" * 34)

k_values = [1, 3, 5, 7, 11, 15, 21, 31]
knn_k_results = {}
for k in k_values:
    # TODO: Create KNeighborsClassifier with n_neighbors=k, metric="euclidean"
    knn = ____  # Hint: KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    # TODO: Cross-validate accuracy and F1
    acc_scores = ____  # Hint: cross_val_score(knn, X_train, y_train, cv=cv, scoring="accuracy")
    f1_scores = ____  # Hint: cross_val_score(knn, X_train, y_train, cv=cv, scoring="f1")
    knn_k_results[k] = {
        "accuracy": acc_scores.mean(),
        "f1": f1_scores.mean(),
    }
    print(f"{k:>6} {acc_scores.mean():>14.4f} {f1_scores.mean():>10.4f}")

best_k = max(knn_k_results, key=lambda k: knn_k_results[k]["f1"])
print(f"\nBest k={best_k} (F1={knn_k_results[best_k]['f1']:.4f})")

# 4b: Compare distance metrics at best k
print(f"\n--- KNN: distance metric comparison (k={best_k}) ---")
print(f"{'Metric':<12} {'CV Accuracy':>14} {'CV F1':>10}")
print("-" * 40)

metrics = ["euclidean", "manhattan", "cosine"]
knn_metric_results = {}
for metric in metrics:
    knn = KNeighborsClassifier(n_neighbors=best_k, metric=metric)
    acc_scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring="f1")
    knn_metric_results[metric] = {
        "accuracy": acc_scores.mean(),
        "f1": f1_scores.mean(),
    }
    print(f"{metric:<12} {acc_scores.mean():>14.4f} {f1_scores.mean():>10.4f}")

best_metric = max(knn_metric_results, key=lambda m: knn_metric_results[m]["f1"])
print(f"\nBest metric: {best_metric} (F1={knn_metric_results[best_metric]['f1']:.4f})")

# Train final KNN
t0 = time.perf_counter()
# TODO: Create and fit KNeighborsClassifier with best_k neighbours and best_metric
knn_final = ____  # Hint: KNeighborsClassifier(n_neighbors=best_k, metric=best_metric)
# TODO: Fit knn_final on X_train, y_train
____  # Hint: knn_final.fit(X_train, y_train)
knn_train_time = time.perf_counter() - t0

# TODO: Predict classes and probabilities on X_test
knn_pred = ____  # Hint: knn_final.predict(X_test)
knn_prob = ____  # Hint: knn_final.predict_proba(X_test)[:, 1]

print(f"\nKNN Final (k={best_k}, {best_metric}): trained in {knn_train_time:.4f}s")
print(classification_report(y_test, knn_pred, target_names=["Retained", "Churned"]))

# ── Checkpoint 3 ─────────────────────────────────────────────────────
knn_acc = accuracy_score(y_test, knn_pred)
assert knn_acc > 0.5, f"KNN accuracy {knn_acc:.4f} should be above random baseline"
assert best_k > 1, "Best k should be > 1 (k=1 always overfits)"
# INTERPRETATION: The best k reflects the neighbourhood size where local
# majority vote is most reliable. k=1 memorises noise; k=n votes globally
# and ignores local structure. The cross-validated k balances both.
print("\n✓ Checkpoint 3 passed — KNN trained and optimal k found\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5: Naive Bayes (GaussianNB)
# ════════════════════════════════════════════════════════════════════════
# THEORY: Naive Bayes applies Bayes' theorem with the naive assumption
#   that features are conditionally independent given the class:
#   P(y|x_1,...,x_n) proportional to P(y) * Product(P(x_i|y))
#   GaussianNB assumes each P(x_i|y) ~ N(mu_iy, sigma_iy^2).

print("\n=== Naive Bayes (GaussianNB) ===")

t0 = time.perf_counter()
# TODO: Create a GaussianNB model, fit it, then predict on test set
nb = ____  # Hint: GaussianNB()
# TODO: Fit nb on X_train, y_train
____  # Hint: nb.fit(X_train, y_train)
nb_train_time = time.perf_counter() - t0

# TODO: Predict classes and probabilities on X_test
nb_pred = ____  # Hint: nb.predict(X_test)
nb_prob = ____  # Hint: nb.predict_proba(X_test)[:, 1]

print(f"GaussianNB: trained in {nb_train_time:.4f}s")
print(classification_report(y_test, nb_pred, target_names=["Retained", "Churned"]))

# Cross-validation
nb_cv_acc = cross_val_score(nb, X_train, y_train, cv=cv, scoring="accuracy")
nb_cv_f1 = cross_val_score(nb, X_train, y_train, cv=cv, scoring="f1")
print(f"CV Accuracy: {nb_cv_acc.mean():.4f} (+/- {nb_cv_acc.std():.4f})")
print(f"CV F1:       {nb_cv_f1.mean():.4f} (+/- {nb_cv_f1.std():.4f})")

# Inspect class-conditional parameters
print(f"\nClass priors: P(retained)={nb.class_prior_[0]:.4f}, P(churned)={nb.class_prior_[1]:.4f}")
print(f"\nClass-conditional means (mu_iy):")
print(f"{'Feature':<30} {'Retained':>10} {'Churned':>10} {'Diff':>10}")
print("-" * 64)
for i, name in enumerate(feature_names):
    mu_0 = nb.theta_[0, i]
    mu_1 = nb.theta_[1, i]
    print(f"{name:<30} {mu_0:>10.4f} {mu_1:>10.4f} {abs(mu_1 - mu_0):>10.4f}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
nb_acc = accuracy_score(y_test, nb_pred)
assert nb_acc > 0.5, f"Naive Bayes accuracy {nb_acc:.4f} should be above random baseline"
assert abs(nb.class_prior_[0] + nb.class_prior_[1] - 1.0) < 1e-6, \
    "Class priors must sum to 1"
# INTERPRETATION: Naive Bayes class priors directly reflect the training set
# class balance. If prior[1] (churned) is 0.25, the model starts every
# prediction with a 25% base rate for churn — then updates based on features.
print("\n✓ Checkpoint 4 passed — Naive Bayes trained and class priors inspected\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6: Decision Tree — Gini impurity and tree structure
# ════════════════════════════════════════════════════════════════════════
# THEORY: Decision trees recursively partition the feature space by
#   selecting the split that best separates classes.
#   Gini impurity: G(node) = 1 - Sum(p_k^2) for k classes

print("\n=== Decision Tree ===")

# 6a: Manual Gini impurity computation
n_retained = (y_train == 0).sum()
n_churned = (y_train == 1).sum()
n_total = len(y_train)

p_retained = n_retained / n_total
p_churned = n_churned / n_total

# TODO: Compute gini_root using the formula: 1 - (p_retained**2 + p_churned**2)
gini_root = ____  # Hint: 1 - (p_retained**2 + p_churned**2)

print(f"\n--- Manual Gini Impurity at Root Node ---")
print(f"Training data: {n_retained} retained, {n_churned} churned (n={n_total})")
print(f"p(retained) = {p_retained:.4f}")
print(f"p(churned)  = {p_churned:.4f}")
print(f"Gini = 1 - (p_retained^2 + p_churned^2)")
print(f"     = 1 - ({p_retained:.4f}^2 + {p_churned:.4f}^2)")
print(f"     = 1 - ({p_retained**2:.4f} + {p_churned**2:.4f})")
print(f"     = {gini_root:.4f}")

# Simulate one split manually to show information gain
feat_idx = 0
feat_name = feature_names[feat_idx]
threshold = np.median(X_train[:, feat_idx])

left_mask = X_train[:, feat_idx] <= threshold
right_mask = ~left_mask

n_left = left_mask.sum()
n_right = right_mask.sum()

p_churn_left = y_train[left_mask].mean()
p_retain_left = 1 - p_churn_left
# TODO: Compute gini_left = 1 - (p_retain_left**2 + p_churn_left**2)
gini_left = ____  # Hint: 1 - (p_retain_left**2 + p_churn_left**2)

p_churn_right = y_train[right_mask].mean()
p_retain_right = 1 - p_churn_right
gini_right = 1 - (p_retain_right**2 + p_churn_right**2)

# TODO: Compute weighted Gini after split: (n_left/n_total)*gini_left + (n_right/n_total)*gini_right
gini_split = ____  # Hint: (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
# TODO: Compute gini_gain = gini_root - gini_split
gini_gain = ____  # Hint: gini_root - gini_split

print(f"\n--- Example Split: {feat_name} <= {threshold:.4f} ---")
print(f"Left child:  n={n_left}, churn rate={p_churn_left:.4f}, Gini={gini_left:.4f}")
print(f"Right child: n={n_right}, churn rate={p_churn_right:.4f}, Gini={gini_right:.4f}")
print(f"Weighted Gini after split: ({n_left}/{n_total})*{gini_left:.4f} + ({n_right}/{n_total})*{gini_right:.4f} = {gini_split:.4f}")
print(f"Gini gain: {gini_root:.4f} - {gini_split:.4f} = {gini_gain:.4f}")

# 6b: Train Decision Tree with depth tuning
print(f"\n--- Decision Tree: max_depth sweep ---")
print(f"{'Depth':>8} {'CV Accuracy':>14} {'CV F1':>10}")
print("-" * 36)

depths = [2, 3, 5, 7, 10, 15, None]
dt_results = {}
for depth in depths:
    # TODO: Create and cross-validate a DecisionTreeClassifier with max_depth=depth
    dt = ____  # Hint: DecisionTreeClassifier(max_depth=depth, random_state=42)
    acc_scores = cross_val_score(dt, X_train, y_train, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(dt, X_train, y_train, cv=cv, scoring="f1")
    label = str(depth) if depth is not None else "None"
    dt_results[label] = {
        "depth": depth,
        "accuracy": acc_scores.mean(),
        "f1": f1_scores.mean(),
    }
    print(f"{label:>8} {acc_scores.mean():>14.4f} {f1_scores.mean():>10.4f}")

best_depth_key = max(dt_results, key=lambda d: dt_results[d]["f1"])
best_depth = dt_results[best_depth_key]["depth"]
print(f"\nBest max_depth={best_depth_key} (F1={dt_results[best_depth_key]['f1']:.4f})")

# Train final Decision Tree
t0 = time.perf_counter()
# TODO: Train final DecisionTreeClassifier with best_depth and fit on X_train, y_train
dt_final = ____  # Hint: DecisionTreeClassifier(max_depth=best_depth, random_state=42)
# TODO: Fit dt_final
____  # Hint: dt_final.fit(X_train, y_train)
dt_train_time = time.perf_counter() - t0

# TODO: Predict classes and probabilities on X_test
dt_pred = ____  # Hint: dt_final.predict(X_test)
dt_prob = ____  # Hint: dt_final.predict_proba(X_test)[:, 1]

print(f"\nDecision Tree (depth={best_depth_key}): trained in {dt_train_time:.4f}s")
print(classification_report(y_test, dt_pred, target_names=["Retained", "Churned"]))

# Visualise tree structure (top 4 levels)
print("--- Tree Structure (first 4 levels) ---")
tree_text = export_text(dt_final, feature_names=feature_names, max_depth=4)
print(tree_text)

# Feature importance from the tree
dt_importances = dict(zip(feature_names, dt_final.feature_importances_))
dt_importances_sorted = dict(sorted(dt_importances.items(), key=lambda x: x[1], reverse=True))

print("--- Decision Tree Feature Importance ---")
print(f"{'Feature':<30} {'Importance':>12}")
print("-" * 44)
for name, imp in list(dt_importances_sorted.items())[:10]:
    bar = "#" * int(imp * 50)
    print(f"{name:<30} {imp:>12.4f}  {bar}")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
dt_acc = accuracy_score(y_test, dt_pred)
assert dt_acc > 0.5, f"Decision Tree accuracy {dt_acc:.4f} should be above random baseline"
assert abs(gini_root - (1 - (p_retained**2 + p_churned**2))) < 1e-9, \
    "Gini impurity formula should match manual calculation"
# INTERPRETATION: Gini impurity is the expected error if you randomly label
# a sample from the node's distribution. A pure node (Gini=0) makes perfect
# predictions. The tree algorithm maximises REDUCTION in Gini with each split —
# greedy local optimization, which may not yield the globally optimal tree.
print("\n✓ Checkpoint 5 passed — Decision Tree trained and Gini computed manually\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 7: Random Forest — bagging, feature importance, OOB
# ════════════════════════════════════════════════════════════════════════
# THEORY: Random Forest = bagging + feature subsampling.
#   Bagging: train B decision trees on bootstrap samples, aggregate by vote.
#   Feature subsampling: at each split, consider only sqrt(p) random features.
#   OOB: ~36.8% of training data excluded per tree — used as free validation.

print("\n=== Random Forest ===")

# 7a: Train with OOB estimation
t0 = time.perf_counter()
# TODO: Create RandomForestClassifier with n_estimators=200, max_features="sqrt",
#       oob_score=True, random_state=42, n_jobs=-1. Then fit on X_train, y_train.
rf = ____  # Hint: RandomForestClassifier(n_estimators=200, max_features="sqrt", oob_score=True, random_state=42, n_jobs=-1)
# TODO: Fit rf on X_train, y_train
____  # Hint: rf.fit(X_train, y_train)
rf_train_time = time.perf_counter() - t0

# TODO: Predict classes and probabilities on X_test
rf_pred = ____  # Hint: rf.predict(X_test)
rf_prob = ____  # Hint: rf.predict_proba(X_test)[:, 1]

print(f"Random Forest (200 trees): trained in {rf_train_time:.2f}s")
print(f"OOB Score: {rf.oob_score_:.4f}")
print(classification_report(y_test, rf_pred, target_names=["Retained", "Churned"]))

rf_cv_acc = cross_val_score(rf, X_train, y_train, cv=cv, scoring="accuracy")
rf_cv_f1 = cross_val_score(rf, X_train, y_train, cv=cv, scoring="f1")
print(f"CV Accuracy: {rf_cv_acc.mean():.4f} (+/- {rf_cv_acc.std():.4f})")
print(f"CV F1:       {rf_cv_f1.mean():.4f} (+/- {rf_cv_f1.std():.4f})")

# 7b: Feature importance
rf_importances = dict(zip(feature_names, rf.feature_importances_))
rf_importances_sorted = dict(sorted(rf_importances.items(), key=lambda x: x[1], reverse=True))

print(f"\n--- Random Forest Feature Importance ---")
print(f"{'Feature':<30} {'Importance':>12}")
print("-" * 44)
for name, imp in rf_importances_sorted.items():
    bar = "#" * int(imp * 50)
    print(f"{name:<30} {imp:>12.4f}  {bar}")

# 7c: OOB convergence
print(f"\n--- OOB Convergence vs Number of Trees ---")
oob_scores = []
n_trees_list = [10, 25, 50, 75, 100, 150, 200]
for n_trees in n_trees_list:
    rf_temp = RandomForestClassifier(
        n_estimators=n_trees,
        max_features="sqrt",
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )
    rf_temp.fit(X_train, y_train)
    oob_scores.append(rf_temp.oob_score_)

print(f"{'Trees':>8} {'OOB Score':>12}")
print("-" * 24)
for n, oob in zip(n_trees_list, oob_scores):
    print(f"{n:>8} {oob:>12.4f}")

# ── Checkpoint 6 ─────────────────────────────────────────────────────
rf_acc = accuracy_score(y_test, rf_pred)
assert rf_acc > 0.5, f"Random Forest accuracy {rf_acc:.4f} should be above random baseline"
assert rf.oob_score_ > 0.5, "OOB score should be above random baseline"
assert abs(rf.oob_score_ - rf_cv_acc.mean()) < 0.10, \
    "OOB score and CV accuracy should be reasonably close (within 10pp)"
# INTERPRETATION: OOB score and CV accuracy being close is a sign that the
# model generalises consistently. A large gap (OOB >> CV) would suggest
# data leakage or distribution shift between folds.
print("\n✓ Checkpoint 6 passed — Random Forest trained with OOB validation\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 8: Model Comparison
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 76)
print("MODEL COMPARISON: All 5 Supervised Learning Families")
print("=" * 76)

models = {
    "SVM (RBF)": {"model": svm_final, "pred": svm_pred, "prob": svm_prob, "train_time": svm_train_time},
    "KNN": {"model": knn_final, "pred": knn_pred, "prob": knn_prob, "train_time": knn_train_time},
    "Naive Bayes": {"model": nb, "pred": nb_pred, "prob": nb_prob, "train_time": nb_train_time},
    "Decision Tree": {"model": dt_final, "pred": dt_pred, "prob": dt_prob, "train_time": dt_train_time},
    "Random Forest": {"model": rf, "pred": rf_pred, "prob": rf_prob, "train_time": rf_train_time},
}

comparison_rows = []
for name, info in models.items():
    acc = accuracy_score(y_test, info["pred"])
    f1 = f1_score(y_test, info["pred"])
    auc = roc_auc_score(y_test, info["prob"])
    comparison_rows.append({
        "Model": name,
        "Accuracy": acc,
        "F1": f1,
        "AUC-ROC": auc,
        "Train Time (s)": info["train_time"],
    })

comparison_df = pl.DataFrame(comparison_rows)

print("\n--- Test Set Performance ---")
print(f"{'Model':<18} {'Accuracy':>10} {'F1':>10} {'AUC-ROC':>10} {'Time (s)':>10}")
print("-" * 62)
for row in comparison_rows:
    print(
        f"{row['Model']:<18} {row['Accuracy']:>10.4f} {row['F1']:>10.4f} "
        f"{row['AUC-ROC']:>10.4f} {row['Train Time (s)']:>10.4f}"
    )

ranked = sorted(comparison_rows, key=lambda r: r["F1"], reverse=True)
print(f"\nRanking by F1 Score:")
for i, row in enumerate(ranked, 1):
    print(f"  {i}. {row['Model']} (F1={row['F1']:.4f})")


# ════════════════════════════════════════════════════════════════════════
# TASK 8b: Visualise with ModelVisualizer
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

metric_dict = {
    row["Model"]: {"Accuracy": row["Accuracy"], "F1": row["F1"], "AUC-ROC": row["AUC-ROC"]}
    for row in comparison_rows
}
fig_compare = viz.metric_comparison(metric_dict)
fig_compare.update_layout(title="Model Zoo: Performance Comparison")
fig_compare.write_html("ex3_model_comparison.html")
print("\nSaved: ex3_model_comparison.html")

fig_importance = viz.metric_comparison({"Importance": rf_importances_sorted})
fig_importance.update_layout(title="Random Forest: Feature Importance")
fig_importance.write_html("ex3_rf_importance.html")
print("Saved: ex3_rf_importance.html")

fig_oob = viz.training_history({"OOB Score": oob_scores}, x_label="Number of Trees")
fig_oob.update_layout(title="Random Forest: OOB Score vs Number of Trees")
fig_oob.write_html("ex3_oob_convergence.html")
print("Saved: ex3_oob_convergence.html")


# ════════════════════════════════════════════════════════════════════════
# TASK 9: When to Use Each Model
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 76)
print("WHEN TO USE EACH MODEL — Decision Guide")
print("=" * 76)

print("""
+-------------------+---------------------+---------------------+---------------+
| Model             | Best When           | Avoid When          | Key Tradeoff  |
+-------------------+---------------------+---------------------+---------------+
| SVM               | Clear margin of     | Very large datasets | High accuracy |
|                   | separation, high-   | (O(n^2) kernel      | but slow to   |
|                   | dimensional data    | computation)        | train at scale|
+-------------------+---------------------+---------------------+---------------+
| KNN               | Small datasets,     | High dimensions     | No training,  |
|                   | non-linear boundary | (curse of           | but slow at   |
|                   |                     | dimensionality)     | prediction    |
+-------------------+---------------------+---------------------+---------------+
| Naive Bayes       | Fast baseline, NLP, | Features are highly | Lightning     |
|                   | well-calibrated     | correlated          | fast, rarely  |
|                   | probabilities needed|                     | best accuracy |
+-------------------+---------------------+---------------------+---------------+
| Decision Tree     | Need interpretable  | Complex data with   | Fully         |
|                   | rules, regulatory   | many features       | interpretable |
|                   | requirements        |                     | but overfits  |
+-------------------+---------------------+---------------------+---------------+
| Random Forest     | General purpose     | Need predictions    | Robust        |
|                   | default, tabular    | explained in rules  | accuracy, but |
|                   | data, missing values|                     | black box     |
+-------------------+---------------------+---------------------+---------------+
""")

print("\n✓ Exercise 3 complete — the complete supervised model zoo")
