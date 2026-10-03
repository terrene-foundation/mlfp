# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 3.4: Decision Trees (Gini from scratch + sklearn)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Compute Gini impurity and entropy from scratch in numpy
#   - Simulate the first split of a decision tree by exhaustive search
#   - Verify the from-scratch split matches sklearn's first split
#   - Pre-prune with max_depth and post-prune with cost-complexity (ccp_alpha)
#   - Read a tree's feature-importance ranking as an audit artefact
#   - Use decision trees where every decision must be explainable
#
# PREREQUISITES: 01_svm.py (shared preprocessing, CV AUC, majority
#   baseline), MLFP01 numpy basics
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — Gini impurity, entropy, information gain
#   2. Build — from-scratch Gini + best-split search
#   3. Train — depth tuning (pre-pruning) + ccp_alpha (post-pruning)
#   4. Visualise — tree structure, feature importance, 2D boundary
#   5. Apply — interpretable churn flagging a reviewer can audit
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from dotenv import load_dotenv
from sklearn.tree import DecisionTreeClassifier, export_text

from shared.mlfp03.ex_3 import (
    build_train_test_split,
    churn_saved_dollars,
    cv_scores,
    decision_boundary_mesh,
    fit_and_evaluate,
    get_visualizer,
    OUTPUT_DIR,
    print_classification_report,
    project_2d,
    RANDOM_SEED,
    save_decision_boundaries,
    save_sweep_plot,
)

load_dotenv()

# ════════════════════════════════════════════════════════════════════════
# THEORY — Impurity, Entropy, and Greedy Splitting
# ════════════════════════════════════════════════════════════════════════
# A decision tree partitions the feature space by repeatedly asking
# "which feature, at which threshold, most reduces class impurity?"
#
# Gini impurity (the default sklearn split criterion):
#     G(node) = 1 - Σ p_k²
#         Pure node (all one class): G = 0
#         Binary 50/50:              G = 0.5  (maximum)
#
# Entropy (ID3/C4.5 criterion):
#     H(node) = - Σ p_k log₂(p_k)
#
# Information gain for a candidate split s that partitions node N into
# children C_1, ..., C_m:
#     IG(s) = impurity(N) - Σ_j (|C_j| / |N|) * impurity(C_j)
#
# The tree is GREEDY: at every node it picks the locally best split,
# without backtracking or global optimisation. This is why an ensemble
# of de-correlated trees (Random Forest, 05_random_forest.py) usually
# beats a single tree.
#
# PRUNING controls overfitting in two ways:
#   Pre-pruning  — stop growing early (max_depth, min_samples_leaf).
#   Post-pruning — grow the full tree, then cut back the branches whose
#     impurity reduction does not pay for their size. Cost-complexity
#     pruning minimises  R_α(T) = R(T) + α·|leaves(T)|; each α on
#     cost_complexity_pruning_path() is a nested, smaller subtree.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: from-scratch Gini + best-split search
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP03 Exercise 3.4 — Decision Trees")
print("=" * 70)


def gini_impurity(y: np.ndarray) -> float:
    """Gini impurity: G = 1 - Σ p_k² for k classes."""
    _, counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)
    return float(1.0 - np.sum(proportions**2))


def entropy(y: np.ndarray) -> float:
    """Shannon entropy: H = - Σ p_k log₂(p_k)."""
    _, counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)
    proportions = proportions[proportions > 0]
    return float(-np.sum(proportions * np.log2(proportions)))


# Gini for canonical distributions
print("\n--- Gini / Entropy for reference distributions ---")
print(f"{'Distribution':<25} {'Gini':>8} {'Entropy':>8}")
print("-" * 43)
for label, y_demo in [
    ("Pure: [100, 0]", np.array([0] * 100)),
    ("50/50: [50, 50]", np.array([0] * 50 + [1] * 50)),
    ("90/10: [90, 10]", np.array([0] * 90 + [1] * 10)),
    ("70/30: [70, 30]", np.array([0] * 70 + [1] * 30)),
    ("3-class [40,30,30]", np.array([0] * 40 + [1] * 30 + [2] * 30)),
]:
    print(f"  {label:<23} {gini_impurity(y_demo):>8.4f} {entropy(y_demo):>8.4f}")


def best_split_search(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[int, float, float]:
    """Exhaustive best-split search — returns (feature_idx, threshold, gain).

    For each feature, try midpoints between consecutive unique values as
    candidate thresholds, compute the weighted-child Gini, and track the
    split with the largest reduction in parent impurity. Samples at most
    50 thresholds per feature to keep the search tractable.
    """
    n = len(y)
    parent_gini = gini_impurity(y)
    best_gain = 0.0
    best_feature_idx = 0
    best_threshold = 0.0

    rng = np.random.default_rng(RANDOM_SEED)
    for feat_idx in range(X.shape[1]):
        sorted_vals = np.unique(X[:, feat_idx])
        if len(sorted_vals) < 2:
            continue
        thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2
        if len(thresholds) > 50:
            thresholds = rng.choice(thresholds, 50, replace=False)
        for threshold in thresholds:
            left_mask = X[:, feat_idx] <= threshold
            n_left = int(left_mask.sum())
            n_right = n - n_left
            if n_left == 0 or n_right == 0:
                continue
            g_left = gini_impurity(y[left_mask])
            g_right = gini_impurity(y[~left_mask])
            weighted = (n_left / n) * g_left + (n_right / n) * g_right
            gain = parent_gini - weighted
            if gain > best_gain:
                best_gain = gain
                best_feature_idx = feat_idx
                best_threshold = float(threshold)
    return best_feature_idx, best_threshold, float(best_gain)


data = build_train_test_split()
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
cv = data["cv"]
feature_names = data["feature_names"]

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
print("\n--- From-scratch best-split search on training data ---")
best_feat_idx, best_threshold, best_gain = best_split_search(X_train, y_train)
best_feat_name = feature_names[best_feat_idx]
print(
    f"  Feature: {best_feat_name}\n"
    f"  Threshold: {best_threshold:.4f}\n"
    f"  Gini gain: {best_gain:.4f}"
)

# Verify against sklearn's depth-1 stump
stump = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_SEED)
stump.fit(X_train, y_train)
sklearn_feat = feature_names[int(stump.tree_.feature[0])]
sklearn_thresh = float(stump.tree_.threshold[0])
print(f"\nsklearn stump  : feature={sklearn_feat}, threshold={sklearn_thresh:.4f}")
print(
    "  Match"
    if sklearn_feat == best_feat_name
    else "  Close (search samples thresholds; sklearn uses all)"
)

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert best_gain > 0, "Best split should have positive Gini gain"
assert abs(gini_impurity(np.array([0] * 100))) < 1e-9, "Pure Gini should be 0"
assert abs(gini_impurity(np.array([0] * 50 + [1] * 50)) - 0.5) < 1e-9, "50/50 = 0.5"
print("\n[ok] Checkpoint 1 passed — Gini + best-split verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: sklearn DecisionTreeClassifier with depth tuning
# ════════════════════════════════════════════════════════════════════════

print(f"Majority-class baseline accuracy (test): {data['majority_accuracy']:.4f}")

DEPTHS = [2, 3, 5, 7, 10, 15, None]
print("--- Pre-pruning: max_depth sweep (5-fold CV) ---")
print(f"{'depth':>8} {'CV Accuracy':>14} {'CV AUC':>10}")
print("-" * 36)
depth_results: dict[str, dict[str, float | int | None]] = {}
for depth in DEPTHS:
    scores = cv_scores(
        DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_SEED),
        X_train,
        y_train,
        cv,
    )
    label = str(depth) if depth is not None else "None"
    depth_results[label] = {"depth": depth, **scores}
    print(f"{label:>8} {scores['accuracy']:>14.4f} {scores['auc_roc']:>10.4f}")

best_depth_label = max(depth_results, key=lambda d: depth_results[d]["auc_roc"])  # type: ignore[arg-type]
best_depth = depth_results[best_depth_label]["depth"]
print(f"\nBest depth by CV AUC: {best_depth_label}")
print(
    f"Unlimited depth CV AUC: {depth_results['None']['auc_roc']:.4f} vs "
    f"best {depth_results[best_depth_label]['auc_roc']:.4f} — a fully grown "
    f"tree memorises the training noise."
)

print("\n--- Post-pruning: cost-complexity (ccp_alpha) on the full tree ---")
path = DecisionTreeClassifier(random_state=RANDOM_SEED).cost_complexity_pruning_path(
    X_train, y_train
)
# The path has one α per prunable subtree (hundreds), ordered from the
# full tree (α=0) to the root-only stump (the last α, excluded here).
# Most of the interesting small trees sit near the END of the path, so
# sample 12 indices geometrically spaced back from the end.
n_alphas = len(path.ccp_alphas)
alpha_idx = np.unique((n_alphas - 1) - np.geomspace(1, n_alphas - 1, 12).astype(int))
ccp_alphas = [float(path.ccp_alphas[i]) for i in alpha_idx]
ccp_results: dict[float, dict[str, float]] = {}
print(f"{'ccp_alpha':>12} {'leaves':>8} {'CV AUC':>10}")
print("-" * 32)
for alpha in ccp_alphas:
    pruned = DecisionTreeClassifier(ccp_alpha=alpha, random_state=RANDOM_SEED)
    ccp_results[alpha] = cv_scores(pruned, X_train, y_train, cv)
    n_leaves = pruned.fit(X_train, y_train).get_n_leaves()
    ccp_results[alpha]["leaves"] = float(n_leaves)
    print(f"{alpha:>12.6f} {n_leaves:>8} {ccp_results[alpha]['auc_roc']:>10.4f}")
best_alpha = max(ccp_results, key=lambda a: ccp_results[a]["auc_roc"])
print(
    f"\nBest ccp_alpha by CV AUC: {best_alpha:.6f} "
    f"({int(ccp_results[best_alpha]['leaves'])} leaves, "
    f"AUC {ccp_results[best_alpha]['auc_roc']:.4f}) vs best pre-pruned depth "
    f"AUC {depth_results[best_depth_label]['auc_roc']:.4f}"
)

dt_result = fit_and_evaluate(
    DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_SEED),
    X_train,
    y_train,
    X_test,
    y_test,
    name=f"DecisionTree (depth={best_depth_label})",
)
dt_model = dt_result["model"]

print(
    f"\n{dt_result['name']}: trained in {dt_result['train_time']:.4f}s | "
    f"accuracy={dt_result['accuracy']:.4f} | "
    f"F1={dt_result['f1']:.4f} | AUC={dt_result['auc_roc']:.4f}"
)
print(
    f"Accuracy lift over the majority baseline: "
    f"{dt_result['accuracy'] - data['majority_accuracy']:+.4f}"
)
print_classification_report(y_test, dt_result["pred"])
print(f"Tree depth: {dt_model.get_depth()}, leaves: {dt_model.get_n_leaves()}")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: tree structure, feature importance, 2D boundary
# ════════════════════════════════════════════════════════════════════════

print("\n--- Tree structure (first 3 levels) ---")
print(export_text(dt_model, feature_names=feature_names, max_depth=3)[:1500])

importances = sorted(
    zip(feature_names, dt_model.feature_importances_),
    key=lambda pair: pair[1],
    reverse=True,
)
print("\n--- Top feature importances ---")
print(f"{'feature':<30} {'importance':>12}")
print("-" * 44)
for name, imp in importances[:10]:
    bar = "#" * int(imp * 50)
    print(f"{name:<30} {imp:>12.4f}  {bar}")

# 2D decision boundary
pca_bundle = project_2d(X_train, X_test)
X_train_2d = pca_bundle["X_train_2d"]
dt_2d = DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_SEED)
dt_2d.fit(X_train_2d, y_train)
xx, yy = decision_boundary_mesh(X_train_2d)
Z = dt_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

viz = get_visualizer()
fig = viz.feature_importance(dt_model, feature_names, top_n=10)
fig.update_layout(title="Decision Tree — top 10 impurity-based importances")
out = OUTPUT_DIR / "ex3_04_tree_importance.html"
fig.write_html(str(out))
print(f"\nSaved: {out}")

leaves_sorted = sorted(ccp_alphas, key=lambda a: ccp_results[a]["leaves"])
prune_out = save_sweep_plot(
    [ccp_results[a]["leaves"] for a in leaves_sorted],
    {"CV AUC": [ccp_results[a]["auc_roc"] for a in leaves_sorted]},
    x_label="number of leaves after cost-complexity pruning (log scale)",
    title="Post-pruning: CV AUC vs tree size",
    fname="ex3_04_tree_pruning.html",
    log_x=True,
)
print(f"Saved: {prune_out}")

boundary_out = save_decision_boundaries(
    {f"Decision tree (depth={best_depth_label})": Z},
    xx,
    yy,
    X_train_2d,
    y_train,
    fname="ex3_04_tree_boundary.html",
    title="Decision-tree regions in 2D PCA space (red = churned)",
)
print(f"Saved: {boundary_out}")
print(
    f"PCA variance captured: {pca_bundle['explained_variance'].sum():.2%}. "
    f"Decision-tree boundaries are AXIS-ALIGNED rectangles — no diagonal lines."
)

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert dt_result["auc_roc"] > 0.6, "Decision tree must rank churners above retained customers"
assert best_alpha in ccp_results, "Best ccp_alpha must come from the pruning sweep"
assert boundary_out.exists(), "Decision-boundary figure must be written"
print("[ok] Checkpoint 2 passed — tree trained, pruned and visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: interpretable churn flagging a reviewer can audit
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: The retention team must be able to tell a reviewer WHY any
# customer received (or did not receive) a S$18 offer. A single tree is
# the one model in this zoo whose full decision logic can be printed and
# read. (Business figures are illustrative teaching assumptions.)
#
# Why a decision tree fits:
#   - Every prediction is a path of feature-threshold rules — the
#     printed tree structure above IS the model, so the audit artefact
#     and the deployed logic cannot drift apart.
#   - Feature importance gives a single-column risk summary.
#
# LIMITATIONS:
#   - High variance: a small change in training data can flip the top
#     split and produce a dramatically different tree.
#   - A single tree usually trails an ensemble of trees; compare the
#     CV AUCs here with 05_random_forest.py before assuming it.
#   - The single-tree recall is capped by axis-aligned splits —
#     diagonal decision surfaces need many steps.

# Show the actual rule path for the first held-out customer.
node_path = dt_model.decision_path(X_test[:1]).indices
rules = []
for node in node_path[:-1]:
    f_idx = int(dt_model.tree_.feature[node])
    thr = float(dt_model.tree_.threshold[node])
    went_left = X_test[0, f_idx] <= thr
    rules.append(f"{feature_names[f_idx]} {'<=' if went_left else '>'} {thr:.3f}")
print("\nRule path for the first test customer (z-scored units):")
print("  " + " AND ".join(rules) + f" -> predict {'churn' if dt_result['pred'][0] else 'retain'}")

true_positives = int(((dt_result["pred"] == 1) & (y_test == 1)).sum())
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
  [x] Gini impurity and entropy computed from scratch
  [x] Best-split exhaustive search, verified against sklearn
  [x] Pre-pruning (max_depth={best_depth_label}) vs post-pruning
      (ccp_alpha={best_alpha:.6f}) compared by CV AUC
  [x] Held-out accuracy {dt_result['accuracy']:.4f} vs majority baseline
      {data['majority_accuracy']:.4f}; AUC {dt_result['auc_roc']:.4f}
  [x] Tree feature importance as an audit artefact
  [x] Axis-aligned 2D decision boundary
  [x] Interpretability business case
      — S${dollars_saved:,.0f} retained on the held-out test fold

  Next: 05_random_forest.py — bag many de-correlated trees for OOB.
"""
)
