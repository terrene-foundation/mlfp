# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 3.3: Naive Bayes (Gaussian, Multinomial, Bernoulli)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Apply Bayes theorem to a classification problem
#   - Understand the "naive" conditional independence assumption
#   - Read class priors and class-conditional means off a GaussianNB
#   - Match the NB variant (Gaussian / Multinomial / Bernoulli) to the
#     shape of the features
#   - Visualise the Gaussian decision boundary in 2D PCA space
#   - Use Naive Bayes as a lightning-fast baseline for high-volume
#     e-commerce triage
#
# PREREQUISITES: 01_svm.py, MLFP02 Bayesian thinking
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — Bayes theorem, conditional independence, three likelihoods
#   2. Build — fit a GaussianNB and compare the three NB variants by CV
#   3. Train — inspect class priors and class-conditional parameters
#   4. Visualise — class-mean profile + 2D decision boundary
#   5. Apply — high-volume triage for a regional marketplace
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

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
)

load_dotenv()

# ════════════════════════════════════════════════════════════════════════
# THEORY — Bayes Theorem and the Naive Assumption
# ════════════════════════════════════════════════════════════════════════
# Bayes theorem, applied to a class y given features x_1, ..., x_n:
#
#     P(y | x_1, ..., x_n)  proportional to  P(y) * prod_i P(x_i | y)
#
# The "naive" step is the independence assumption:
#     P(x_1, x_2, ..., x_n | y) = prod_i P(x_i | y)
# i.e. we pretend every feature is independent of every other feature
# given the class label. This is almost always false in practice (a
# customer's order_count and total_revenue are correlated), yet Naive
# Bayes still produces strong baselines — its RANKING of customers is
# often good even when its probabilities are over-confident.
#
# The three sklearn variants differ only in the likelihood P(x_i | y):
#   GaussianNB    — continuous features: Normal(μ_iy, σ²_iy)
#   MultinomialNB — non-negative counts / frequencies (word counts in
#                   text classification is the classic use)
#   BernoulliNB   — binary present/absent features
# Pick the one whose likelihood matches how your features are measured.
#
# Why it is fast: training is a single pass over the data to compute
# per-class means and variances (or counts). Prediction is a vectorised
# log-likelihood scoring. No gradient descent, no iterative optimisation.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: fit GaussianNB, then compare the three NB variants
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP03 Exercise 3.3 — Naive Bayes")
print("=" * 70)

data = build_train_test_split()
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
cv = data["cv"]
feature_names = data["feature_names"]

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
print(f"Majority-class baseline accuracy (test): {data['majority_accuracy']:.4f}")

nb_result = fit_and_evaluate(
    GaussianNB(),
    X_train,
    y_train,
    X_test,
    y_test,
    name="GaussianNB",
)
nb_model = nb_result["model"]

print(
    f"\n{nb_result['name']}: trained in {nb_result['train_time']:.4f}s | "
    f"accuracy={nb_result['accuracy']:.4f} | "
    f"F1={nb_result['f1']:.4f} | AUC={nb_result['auc_roc']:.4f}"
)
print_classification_report(y_test, nb_result["pred"])

# The features are z-scored (can be negative), so each variant gets the
# input shape its likelihood expects:
#   Multinomial — rescale every feature to [0, 1] (non-negative "amounts")
#   Bernoulli   — binarise at 0, i.e. "above vs below the average customer"
variants = {
    "GaussianNB": GaussianNB(),
    "MultinomialNB (min-max scaled)": make_pipeline(MinMaxScaler(), MultinomialNB()),
    "BernoulliNB (above/below mean)": BernoulliNB(binarize=0.0),
}
variant_scores: dict[str, dict[str, float]] = {}
print("\n--- NB variants, 5-fold CV on the same folds ---")
print(f"{'variant':<32} {'CV Accuracy':>12} {'CV AUC':>8} {'fit (s)':>9}")
print("-" * 64)
for name, est in variants.items():
    variant_scores[name] = cv_scores(est, X_train, y_train, cv)
    s = variant_scores[name]
    print(f"{name:<32} {s['accuracy']:>12.4f} {s['auc_roc']:>8.4f} {s['fit_time']:>9.4f}")
best_variant = max(variant_scores, key=lambda n: variant_scores[n]["auc_roc"])
print(
    f"\nBest variant by CV AUC: {best_variant}. Bernoulli throws away the size "
    f"of each feature (only above/below average survives) — the AUC gap "
    f"shows how much information that costs on continuous data."
)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN phase: inspect the learned parameters
# ════════════════════════════════════════════════════════════════════════

print(
    f"\nClass priors: "
    f"P(retained)={nb_model.class_prior_[0]:.4f}, "
    f"P(churned)={nb_model.class_prior_[1]:.4f}"
)
mean_gap = np.abs(nb_model.theta_[1] - nb_model.theta_[0])
order = np.argsort(-mean_gap)
print("\nClass-conditional means (z-scored), sorted by |difference|:")
print(f"{'Feature':<30} {'Retained':>10} {'Churned':>10} {'|Diff|':>10}")
print("-" * 64)
for i in order:
    print(
        f"{feature_names[i]:<30} {nb_model.theta_[0, i]:>10.4f} "
        f"{nb_model.theta_[1, i]:>10.4f} {mean_gap[i]:>10.4f}"
    )
print(
    f"\nThe feature whose class means differ most — {feature_names[order[0]]} — "
    f"carries most of the evidence GaussianNB uses."
)

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert nb_result["auc_roc"] > 0.6, "GaussianNB must rank churners above retained customers"
assert (
    abs(nb_model.class_prior_[0] + nb_model.class_prior_[1] - 1.0) < 1e-6
), "Priors must sum to 1"
assert len(variant_scores) == 3, "All three NB variants must be cross-validated"
print("\n[ok] Checkpoint 1 passed — Naive Bayes trained and parameters inspected\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: class-mean profile + 2D decision boundary
# ════════════════════════════════════════════════════════════════════════

fig = go.Figure()
fig.add_trace(go.Bar(x=feature_names, y=nb_model.theta_[0], name="retained"))
fig.add_trace(go.Bar(x=feature_names, y=nb_model.theta_[1], name="churned"))
fig.update_layout(
    title="GaussianNB: class-conditional means per feature (z-scored)",
    barmode="group",
    xaxis_title="feature",
    yaxis_title="mean (standard deviations from overall mean)",
)
out = OUTPUT_DIR / "ex3_03_nb_means.html"
fig.write_html(str(out))
print(f"Saved: {out}")

pca_bundle = project_2d(X_train, X_test)
X_train_2d = pca_bundle["X_train_2d"]

nb_2d = GaussianNB()
nb_2d.fit(X_train_2d, y_train)
xx, yy = decision_boundary_mesh(X_train_2d)
Z = nb_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

boundary_out = save_decision_boundaries(
    {"GaussianNB": Z},
    xx,
    yy,
    X_train_2d,
    y_train,
    fname="ex3_03_nb_boundary.html",
    title="GaussianNB decision regions in 2D PCA space (red = churned)",
)
print(f"Saved: {boundary_out}")
print(
    f"PCA variance captured: {pca_bundle['explained_variance'].sum():.2%}. "
    f"With class-specific variances the GaussianNB boundary is a smooth "
    f"quadratic curve (an ellipse or parabola), never jagged."
)

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert Z.shape == xx.shape, "Decision mesh must match the grid"
assert boundary_out.exists(), "Decision-boundary figure must be written"
print("[ok] Checkpoint 2 passed — GaussianNB 2D boundary rendered\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: high-volume marketplace triage
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A regional marketplace wants to score every daily session
# with a churn-risk flag in near real time. With tens of millions of
# events per day, inference latency and cost both matter.
# (Business figures are illustrative teaching assumptions.)
#
# Why Naive Bayes fits:
#   - Training is O(n): re-train nightly from scratch on the full day's
#     data, no gradient descent loop.
#   - Prediction is a vectorised log-likelihood — microseconds per row.
#   - Compare its CV AUC above with the other families in 06_model_zoo:
#     if it is within noise of them, the cheapest model wins.
#   - Transparent: priors and class means are human-readable artefacts
#     a reviewer can sign off.
#
# LIMITATIONS:
#   - Conditional independence is false when features are correlated
#     (order_count, total_revenue, avg_order_value). The model double-
#     counts the shared signal, so its probabilities are over-confident
#     even when its ranking is good (calibrate before using them as
#     probabilities — Lesson 3.5).
#   - The Gaussian assumption fails on long-tailed counts (orders,
#     dollars). A log-transform or discretisation helps.

true_positives = int(((nb_result["pred"] == 1) & (y_test == 1)).sum())
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
  [x] Bayes theorem rewritten as a classifier
  [x] The independence assumption and its failure modes
  [x] Class priors and per-feature Gaussian parameters
  [x] Gaussian vs Multinomial vs Bernoulli — best here: {best_variant}
  [x] Held-out accuracy {nb_result['accuracy']:.4f} vs majority baseline
      {data['majority_accuracy']:.4f}; AUC {nb_result['auc_roc']:.4f}
  [x] Smooth, quadratic 2D decision boundary
  [x] High-volume triage business case — S${dollars_saved:,.0f} retained

  Next: 04_decision_tree.py — Gini impurity from scratch + sklearn trees.
"""
)
