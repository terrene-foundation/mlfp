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

# SVM with RBF kernel is O(n²) — subsample to keep training time reasonable
# (5000 samples is enough to demonstrate all model behaviours)
ecommerce = ecommerce.sample(n=5000, seed=42)

print("=== E-Commerce Customer Dataset ===")
print(f"Shape: {ecommerce.shape} (subsampled for SVM tractability)")
print(f"Columns: {ecommerce.columns}")
print(f"Churn rate: {ecommerce['churned'].mean():.2%}")

# Drop text columns and high-cardinality strings not suitable for the zoo
# product_categories has 2000+ unique values — not useful without NLP.
# review_text is free-form — belongs in an NLP exercise, not here.
# customer_id is an identifier, not a feature.
ecommerce = ecommerce.drop("customer_id", "review_text", "product_categories")

print(f"\nAfter dropping text/ID columns: {ecommerce.shape}")
print(f"Remaining columns: {ecommerce.columns}")


# ════════════════════════════════════════════════════════════════════════
# TASK 1: Set up classification task
# ════════════════════════════════════════════════════════════════════════
# Target: churned (0 = retained, 1 = churned)
# This is a binary classification problem.

target_col = "churned"

pipeline = PreprocessingPipeline()
result = pipeline.setup(
    data=ecommerce,
    target=target_col,
    train_size=0.8,
    seed=42,
    normalize=True,  # SVM and KNN require scaled features
    normalize_method="zscore",
    categorical_encoding="ordinal",  # Low-cardinality categoricals
    imputation_strategy="median",
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
# Confirm scaling — SVM and KNN are distance-based and sensitive to scale.
# Decision trees and Random Forests are invariant to monotone transforms,
# but we scale everything uniformly for fair comparison.

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
#
#   Kernel trick: map data to higher-dimensional space where linear
#   separation is possible. K(x, x') = phi(x) . phi(x') computes
#   the dot product in feature space without explicit transformation.
#     - Linear: K(x, x') = x . x'
#     - RBF: K(x, x') = exp(-gamma ||x - x'||^2)
#     - Polynomial: K(x, x') = (gamma * x . x' + r)^d

print("\n=== SVM: Support Vector Machines ===")

# 3a: Linear SVM — tune C parameter
print("\n--- Linear SVM: C parameter sweep ---")
print(f"{'C':>10} {'CV Accuracy':>14} {'CV F1':>10}")
print("-" * 38)

c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
linear_svm_results = {}
for c_val in c_values:
    svm_lin = SVC(kernel="linear", C=c_val, random_state=42)
    acc_scores = cross_val_score(svm_lin, X_train, y_train, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(svm_lin, X_train, y_train, cv=cv, scoring="f1")
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
    svm_rbf = SVC(kernel="rbf", C=c_val, random_state=42)
    acc_scores = cross_val_score(svm_rbf, X_train, y_train, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(svm_rbf, X_train, y_train, cv=cv, scoring="f1")
    rbf_svm_results[c_val] = {
        "accuracy": acc_scores.mean(),
        "f1": f1_scores.mean(),
    }
    print(f"{c_val:>10.2f} {acc_scores.mean():>14.4f} {f1_scores.mean():>10.4f}")

best_c_rbf = max(rbf_svm_results, key=lambda c: rbf_svm_results[c]["f1"])
print(f"\nBest RBF SVM: C={best_c_rbf} (F1={rbf_svm_results[best_c_rbf]['f1']:.4f})")

# Train final SVM on full training set
t0 = time.perf_counter()
svm_final = SVC(kernel="rbf", C=best_c_rbf, random_state=42, probability=True)
svm_final.fit(X_train, y_train)
svm_train_time = time.perf_counter() - t0

svm_pred = svm_final.predict(X_test)
svm_prob = svm_final.predict_proba(X_test)[:, 1]

print(f"\nSVM Final (RBF, C={best_c_rbf}): trained in {svm_train_time:.2f}s")
print(classification_report(y_test, svm_pred, target_names=["Retained", "Churned"]))

print("SVM Insights:")
print(f"  Support vectors: {svm_final.n_support_} (per class)")
print(f"  Total support vectors: {svm_final.support_vectors_.shape[0]} "
      f"({svm_final.support_vectors_.shape[0] / len(y_train):.1%} of training data)")
print("  RBF kernel maps data into infinite-dimensional feature space.")
print("  C controls the bias-variance tradeoff:")
print("    Large C -> low bias, high variance (tight fit to training data)")
print("    Small C -> high bias, low variance (wider margin, more generalisation)")

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
# THEORY: KNN is an instance-based (lazy) learner — no training phase.
#   Prediction: for a query point x, find the k closest training points
#   and take a majority vote.
#
#   Distance metrics:
#     Euclidean: d(x, x') = sqrt(Sum((x_i - x'_i)^2))
#     Manhattan: d(x, x') = Sum(|x_i - x'_i|)
#     Cosine:    d(x, x') = 1 - (x . x') / (||x|| * ||x'||)
#
#   Curse of dimensionality: as dimensions grow, all points become
#   equidistant. Distances concentrate around the mean, making KNN
#   unreliable in high-dimensional spaces without feature selection.

print("\n=== KNN: K-Nearest Neighbors ===")

# 4a: Vary k
print("\n--- KNN: k value sweep (Euclidean distance) ---")
print(f"{'k':>6} {'CV Accuracy':>14} {'CV F1':>10}")
print("-" * 34)

k_values = [1, 3, 5, 7, 11, 15, 21, 31]
knn_k_results = {}
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    acc_scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring="f1")
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
knn_final = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric)
knn_final.fit(X_train, y_train)
knn_train_time = time.perf_counter() - t0

knn_pred = knn_final.predict(X_test)
knn_prob = knn_final.predict_proba(X_test)[:, 1]

print(f"\nKNN Final (k={best_k}, {best_metric}): trained in {knn_train_time:.4f}s")
print(classification_report(y_test, knn_pred, target_names=["Retained", "Churned"]))

print("KNN Insights:")
print("  k=1: memorises training data (high variance, overfitting)")
print(f"  k={best_k}: smoothed decision boundary (bias-variance sweet spot)")
print("  KNN has NO training phase — all computation happens at prediction time.")
print(f"  With {X_train.shape[1]} features, distances are meaningful (not too high-dimensional).")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
knn_acc = accuracy_score(y_test, knn_pred)
assert knn_acc > 0.5, f"KNN accuracy {knn_acc:.4f} should be above random baseline"
assert best_k > 1, "Best k should be > 1 (k=1 always overfits)"
# INTERPRETATION: The best k reflects the neighbourhood size where local
# majority vote is most reliable. k=1 memorises noise; k=n votes globally
# and ignores local structure. The cross-validated k balances both.
# Notice that training time is near-zero (KNN is lazy) but prediction
# requires computing distances to all training points — the cost is deferred.
print("\n✓ Checkpoint 3 passed — KNN trained and optimal k found\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5: Naive Bayes (GaussianNB)
# ════════════════════════════════════════════════════════════════════════
# THEORY: Naive Bayes applies Bayes' theorem with the naive assumption
#   that features are conditionally independent given the class:
#
#   P(y|x_1,...,x_n) proportional to P(y) * Product(P(x_i|y))
#
#   GaussianNB assumes each P(x_i|y) ~ N(mu_iy, sigma_iy^2).
#   The model estimates class-conditional means and variances from data.
#
#   Despite the naive independence assumption (rarely true), Naive Bayes
#   often performs well in practice because the decision boundary only
#   needs the RANKING of posteriors to be correct, not the exact values.
#
#   Variants:
#     GaussianNB: continuous features, assumes Gaussian likelihood
#     MultinomialNB: discrete counts (bag-of-words text, word frequencies)
#     BernoulliNB: binary features (word present/absent)

print("\n=== Naive Bayes (GaussianNB) ===")

t0 = time.perf_counter()
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_train_time = time.perf_counter() - t0

nb_pred = nb.predict(X_test)
nb_prob = nb.predict_proba(X_test)[:, 1]

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

print("\nNaive Bayes Insights:")
print("  Fastest training of all models (single pass through data).")
print("  No hyperparameters to tune (var_smoothing is rarely critical).")
print("  Independence assumption is violated (features ARE correlated),")
print("  but NB still provides a useful baseline — calibration may help.")
print("  Best suited for: text classification (MultinomialNB), fast baseline.")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
nb_acc = accuracy_score(y_test, nb_pred)
assert nb_acc > 0.5, f"Naive Bayes accuracy {nb_acc:.4f} should be above random baseline"
assert nb.class_prior_[0] + nb.class_prior_[1] == pytest.approx(1.0) if False else \
    abs(nb.class_prior_[0] + nb.class_prior_[1] - 1.0) < 1e-6, \
    "Class priors must sum to 1"
# INTERPRETATION: Naive Bayes class priors directly reflect the training set
# class balance. If prior[1] (churned) is 0.25, the model starts every
# prediction with a 25% base rate for churn — then updates based on features.
# This is exactly Bayes' theorem in action, connecting back to M2 Bayesian thinking.
print("\n✓ Checkpoint 4 passed — Naive Bayes trained and class priors inspected\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6: Decision Tree — Gini impurity and tree structure
# ════════════════════════════════════════════════════════════════════════
# THEORY: Decision trees recursively partition the feature space by
#   selecting the split that best separates classes.
#
#   Gini impurity: G(node) = 1 - Sum(p_k^2) for k classes
#     Pure node: G = 0 (all one class)
#     Maximum impurity (binary): G = 0.5 (50/50 split)
#
#   Information gain (entropy-based):
#     H(node) = -Sum(p_k * log2(p_k))
#     IG = H(parent) - Sum(w_j * H(child_j))
#
#   Pruning controls overfitting:
#     Pre-pruning: max_depth, min_samples_split, min_samples_leaf
#     Post-pruning: cost-complexity pruning (ccp_alpha)

print("\n=== Decision Tree ===")

# 6a: Manual Gini impurity computation
# Demonstrate the formula on the root node before any split
n_retained = (y_train == 0).sum()
n_churned = (y_train == 1).sum()
n_total = len(y_train)

p_retained = n_retained / n_total
p_churned = n_churned / n_total

gini_root = 1 - (p_retained**2 + p_churned**2)

print(f"\n--- Manual Gini Impurity at Root Node ---")
print(f"Training data: {n_retained} retained, {n_churned} churned (n={n_total})")
print(f"p(retained) = {p_retained:.4f}")
print(f"p(churned)  = {p_churned:.4f}")
print(f"Gini = 1 - (p_retained^2 + p_churned^2)")
print(f"     = 1 - ({p_retained:.4f}^2 + {p_churned:.4f}^2)")
print(f"     = 1 - ({p_retained**2:.4f} + {p_churned**2:.4f})")
print(f"     = {gini_root:.4f}")

# Simulate one split manually to show information gain
# Pick the first feature and a threshold at the median
feat_idx = 0
feat_name = feature_names[feat_idx]
threshold = np.median(X_train[:, feat_idx])

left_mask = X_train[:, feat_idx] <= threshold
right_mask = ~left_mask

n_left = left_mask.sum()
n_right = right_mask.sum()

p_churn_left = y_train[left_mask].mean()
p_retain_left = 1 - p_churn_left
gini_left = 1 - (p_retain_left**2 + p_churn_left**2)

p_churn_right = y_train[right_mask].mean()
p_retain_right = 1 - p_churn_right
gini_right = 1 - (p_retain_right**2 + p_churn_right**2)

gini_split = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
gini_gain = gini_root - gini_split

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
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
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
dt_final = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt_final.fit(X_train, y_train)
dt_train_time = time.perf_counter() - t0

dt_pred = dt_final.predict(X_test)
dt_prob = dt_final.predict_proba(X_test)[:, 1]

print(f"\nDecision Tree (depth={best_depth_key}): trained in {dt_train_time:.4f}s")
print(classification_report(y_test, dt_pred, target_names=["Retained", "Churned"]))

# 6c: Visualise tree structure (top 4 levels)
print("--- Tree Structure (first 4 levels) ---")
tree_text = export_text(
    dt_final,
    feature_names=feature_names,
    max_depth=4,
)
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

print("\nDecision Tree Insights:")
print(f"  Tree depth: {dt_final.get_depth()}, Leaves: {dt_final.get_n_leaves()}")
print("  Fully interpretable: you can trace any prediction through the tree.")
print("  Prone to overfitting without pruning (depth=None memorises training data).")
print("  Non-linear boundaries: can capture complex feature interactions.")

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
#   Bagging (bootstrap aggregating):
#     1. Draw B bootstrap samples (with replacement) from training data
#     2. Train one decision tree on each bootstrap sample
#     3. Aggregate predictions by majority vote (classification)
#
#   Feature subsampling: at each split, consider only sqrt(p) random
#   features (not all p). This decorrelates the trees, reducing variance.
#
#   Out-of-Bag (OOB) estimation:
#     Each bootstrap sample excludes ~36.8% of training data.
#     P(sample i NOT in bootstrap) = (1 - 1/n)^n -> 1/e ~ 0.368
#     Use excluded samples as a free validation set per tree.
#     OOB error ~ cross-validation error without extra computation.

print("\n=== Random Forest ===")

# 7a: Train with OOB estimation
t0 = time.perf_counter()
rf = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    oob_score=True,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
rf_train_time = time.perf_counter() - t0

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

print(f"Random Forest (200 trees): trained in {rf_train_time:.2f}s")
print(f"OOB Score: {rf.oob_score_:.4f}")
print(classification_report(y_test, rf_pred, target_names=["Retained", "Churned"]))

# Cross-validation for comparison
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

# 7c: OOB convergence — how many trees do we actually need?
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

print("\nRandom Forest Insights:")
print("  OOB score provides a free estimate of generalisation error.")
print(f"  OOB ({rf.oob_score_:.4f}) ~ CV accuracy ({rf_cv_acc.mean():.4f}) — confirms consistency.")
print(f"  Feature subsampling (max_features='sqrt'={int(np.sqrt(len(feature_names)))}) decorrelates trees.")
print("  Robust default: rarely needs extensive hyperparameter tuning.")
print("  NOT interpretable: 200 trees cannot be inspected like a single tree.")

# ── Checkpoint 6 ─────────────────────────────────────────────────────
rf_acc = accuracy_score(y_test, rf_pred)
assert rf_acc > 0.5, f"Random Forest accuracy {rf_acc:.4f} should be above random baseline"
assert rf.oob_score_ > 0.5, "OOB score should be above random baseline"
assert abs(rf.oob_score_ - rf_cv_acc.mean()) < 0.10, \
    "OOB score and CV accuracy should be reasonably close (within 10pp)"
# INTERPRETATION: OOB score and CV accuracy being close is a sign that the
# model generalises consistently. A large gap (OOB >> CV) would suggest
# data leakage or distribution shift between folds. The OOB mechanism is
# essentially cross-validation built into the training process — free quality control.
print("\n✓ Checkpoint 6 passed — Random Forest trained with OOB validation\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 8: Model Comparison
# ════════════════════════════════════════════════════════════════════════
# Compare all 5 model families on: accuracy, F1, AUC-ROC, training time.
# Use consistent evaluation on the SAME test set.

print("\n" + "=" * 76)
print("MODEL COMPARISON: All 5 Supervised Learning Families")
print("=" * 76)

# Collect all results
models = {
    "SVM (RBF)": {
        "model": svm_final,
        "pred": svm_pred,
        "prob": svm_prob,
        "train_time": svm_train_time,
    },
    "KNN": {
        "model": knn_final,
        "pred": knn_pred,
        "prob": knn_prob,
        "train_time": knn_train_time,
    },
    "Naive Bayes": {
        "model": nb,
        "pred": nb_pred,
        "prob": nb_prob,
        "train_time": nb_train_time,
    },
    "Decision Tree": {
        "model": dt_final,
        "pred": dt_pred,
        "prob": dt_prob,
        "train_time": dt_train_time,
    },
    "Random Forest": {
        "model": rf,
        "pred": rf_pred,
        "prob": rf_prob,
        "train_time": rf_train_time,
    },
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

# Build comparison table in polars
comparison_df = pl.DataFrame(comparison_rows)

print("\n--- Test Set Performance ---")
print(f"{'Model':<18} {'Accuracy':>10} {'F1':>10} {'AUC-ROC':>10} {'Time (s)':>10}")
print("-" * 62)
for row in comparison_rows:
    print(
        f"{row['Model']:<18} {row['Accuracy']:>10.4f} {row['F1']:>10.4f} "
        f"{row['AUC-ROC']:>10.4f} {row['Train Time (s)']:>10.4f}"
    )

# Rank by F1
ranked = sorted(comparison_rows, key=lambda r: r["F1"], reverse=True)
print(f"\nRanking by F1 Score:")
for i, row in enumerate(ranked, 1):
    print(f"  {i}. {row['Model']} (F1={row['F1']:.4f})")


# ════════════════════════════════════════════════════════════════════════
# TASK 8b: Visualise comparison with ModelVisualizer
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# Metric comparison across all models
metric_dict = {
    row["Model"]: {
        "Accuracy": row["Accuracy"],
        "F1": row["F1"],
        "AUC-ROC": row["AUC-ROC"],
    }
    for row in comparison_rows
}
fig_compare = viz.metric_comparison(metric_dict)
fig_compare.update_layout(title="Model Zoo: Performance Comparison")
fig_compare.write_html("ex3_model_comparison.html")
print("\nSaved: ex3_model_comparison.html")

# Random Forest feature importance bar chart
fig_importance = viz.metric_comparison(
    {"Importance": rf_importances_sorted}
)
fig_importance.update_layout(title="Random Forest: Feature Importance")
fig_importance.write_html("ex3_rf_importance.html")
print("Saved: ex3_rf_importance.html")

# Training time comparison
time_dict = {
    row["Model"]: {"Train Time (s)": row["Train Time (s)"]}
    for row in comparison_rows
}
fig_time = viz.metric_comparison(time_dict)
fig_time.update_layout(title="Model Zoo: Training Time Comparison")
fig_time.write_html("ex3_training_time.html")
print("Saved: ex3_training_time.html")

# OOB convergence plot
fig_oob = viz.training_history(
    {"OOB Score": oob_scores},
    x_label="Number of Trees",
)
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
|                   | non-linear          | (curse of           | but slow at   |
|                   | boundaries, quick   | dimensionality),    | prediction    |
|                   | prototyping         | large n (O(n) pred) | time          |
+-------------------+---------------------+---------------------+---------------+
| Naive Bayes       | Text classification,| Complex feature     | Extremely     |
|                   | fast baseline,      | interactions,       | fast but      |
|                   | very large data,    | correlated features | strong        |
|                   | real-time systems   | (violates naive     | assumptions   |
|                   |                     | assumption badly)   |               |
+-------------------+---------------------+---------------------+---------------+
| Decision Tree     | Interpretability    | High variance       | Fully         |
|                   | required (audit,    | (single tree        | interpretable |
|                   | regulation),        | overfits easily),   | but unstable  |
|                   | non-linear data     | noisy data          | (small data   |
|                   |                     |                     | change = new  |
|                   |                     |                     | tree)         |
+-------------------+---------------------+---------------------+---------------+
| Random Forest     | Robust default,     | Need interpretable  | Robust,       |
|                   | mixed feature       | model, extreme      | accurate,     |
|                   | types, minimal      | speed constraints,  | but black-box |
|                   | tuning available    | memory-constrained  | (200 trees)   |
+-------------------+---------------------+---------------------+---------------+
""")

print("--- Complexity vs Interpretability Trade-off ---")
print("  Most interpretable:  Naive Bayes > Decision Tree > KNN > Random Forest > SVM (RBF)")
print("  Typically most accurate: Random Forest > SVM > KNN > Decision Tree > Naive Bayes")
print("  Fastest training:    Naive Bayes > Decision Tree > KNN > Random Forest > SVM")
print("  Fastest prediction:  Naive Bayes > Decision Tree > Random Forest > SVM > KNN")

print("\n--- Key Takeaways ---")
print("  1. No single model wins on ALL criteria — model selection is a tradeoff.")
print("  2. Random Forest is the safest default for tabular data (robust, minimal tuning).")
print("  3. SVM excels in high-dimensional spaces but does not scale to millions of rows.")
print("  4. Naive Bayes is the speed champion — use it for baselines and real-time systems.")
print("  5. Decision Trees are the ONLY fully interpretable model in this zoo.")
print("  6. KNN is conceptually simple but suffers from the curse of dimensionality.")
print("  7. Always compare models on the SAME evaluation protocol (same CV splits, same metrics).")

# ── Checkpoint 7 ─────────────────────────────────────────────────────
all_accuracies = [svm_acc, knn_acc, nb_acc, dt_acc, rf_acc]
assert all(a > 0.5 for a in all_accuracies), \
    "All 5 models should beat random baseline (accuracy > 0.5)"
assert len(comparison_rows) == 5, "Comparison table should have exactly 5 models"
best_by_f1 = max(comparison_rows, key=lambda r: r["F1"])
# INTERPRETATION: If Random Forest wins by a large margin, the task has
# complex feature interactions that simpler models can't capture. If Naive Bayes
# is competitive, features are mostly independent and the simple model is enough.
# Always report the FULL table — picking one metric hides trade-offs.
print("\n✓ Checkpoint 7 passed — all 5 models compared\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(f"""
  ✓ SVM: margin maximisation, kernel trick, C parameter (bias-variance)
  ✓ KNN: instance-based, distance metrics, k selection via CV
  ✓ Naive Bayes: Bayes theorem, independence assumption, speed vs accuracy
  ✓ Decision Tree: Gini impurity by hand, depth control, interpretability
  ✓ Random Forest: bagging, OOB estimation, feature importance

  BEST MODEL BY F1: {best_by_f1['Model']} (F1={best_by_f1['F1']:.4f})

  KEY INSIGHT: No model wins on all criteria simultaneously.
  Interpretability and accuracy often conflict: the tree is explainable
  but fragile; the forest is robust but opaque. Regulatory contexts
  (credit, healthcare, hiring) often require interpretable models even
  at the cost of some accuracy.

  NEXT: Exercise 4 dives deep into gradient boosting — XGBoost, LightGBM,
  and CatBoost. These algorithms typically outperform the model zoo on
  tabular data by building trees sequentially, each correcting the errors
  of the previous one.
""")

print("\n" + "-" * 76)
print("Exercise 3 complete — the supervised model zoo with 5 algorithm families")
print("-" * 76)
