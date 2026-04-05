# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT3 — Exercise 4: SHAP, LIME, and Fairness
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Full SHAP and LIME analysis on the credit scoring model —
#   global and local interpretability, interaction effects, and fairness
#   audit across protected attributes.
#
# TASKS:
#   1. Compute TreeSHAP values for the best gradient boosting model
#   2. Global interpretation: summary plot and feature importance
#   3. Dependence plots: how individual features affect predictions
#   4. LIME: model-agnostic local linear approximations
#   5. Local interpretation: explain individual predictions
#   6. Fairness audit: check SHAP values across protected attributes
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
import shap
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from kailash_ml import PreprocessingPipeline, ModelVisualizer
from kailash_ml.interop import to_sklearn_input

from shared import ASCENTDataLoader


# ── Data Loading & Model Training ─────────────────────────────────────

loader = ASCENTDataLoader()
credit = loader.load("ascent03", "sg_credit_scoring.parquet")

pipeline = PreprocessingPipeline()
result = pipeline.setup(
    credit, target="default", seed=42, normalize=False, categorical_encoding="ordinal"
)

X_train, y_train, col_info = to_sklearn_input(
    result.train_data,
    feature_columns=[c for c in result.train_data.columns if c != "default"],
    target_column="default",
)
X_test, y_test, _ = to_sklearn_input(
    result.test_data,
    feature_columns=[c for c in result.test_data.columns if c != "default"],
    target_column="default",
)
feature_names = col_info["feature_columns"]

# Train best model from Exercise 1
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    scale_pos_weight=(1 - y_train.mean()) / y_train.mean(),
    random_state=42,
    verbose=-1,
)
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]
print(f"Model AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Compute TreeSHAP values
# ══════════════════════════════════════════════════════════════════════
# TreeSHAP computes exact Shapley values in polynomial time (O(TLD²))
# vs exponential time for exact Shapley.
# Key insight: tree structure allows efficient path-based computation.

explainer = shap.TreeExplainer(model)

# SHAP values for test set
shap_values = explainer.shap_values(X_test)

# For binary classification, shap_values may be a list [class_0, class_1]
if isinstance(shap_values, list):
    shap_vals = shap_values[1]  # Use positive class (default)
else:
    shap_vals = shap_values

print(f"\n=== SHAP Values ===")
print(f"Shape: {shap_vals.shape} (samples × features)")
print(f"Expected value (base rate): {explainer.expected_value}")

# Verify additivity: sum of SHAP values + expected = model output
sample_idx = 0
shap_sum = shap_vals[sample_idx].sum() + (
    explainer.expected_value[1]
    if isinstance(explainer.expected_value, list)
    else explainer.expected_value
)
model_output = model.predict_proba(X_test[sample_idx : sample_idx + 1])[:, 1][0]
print(f"Sample 0: SHAP sum = {shap_sum:.4f}, model output = {model_output:.4f}")
print(
    f"  Additivity check: {'✓ PASS' if abs(shap_sum - model_output) < 0.01 else '✗ FAIL'}"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Global interpretation — feature importance ranking
# ══════════════════════════════════════════════════════════════════════

# Mean absolute SHAP values = global feature importance
mean_abs_shap = np.abs(shap_vals).mean(axis=0)
importance_ranking = sorted(
    zip(feature_names, mean_abs_shap),
    key=lambda x: x[1],
    reverse=True,
)

print(f"\n=== Global Feature Importance (SHAP) ===")
for name, imp in importance_ranking[:15]:
    bar = "█" * int(imp * 200)
    print(f"  {name:<30} {imp:.4f} {bar}")

# Visualise with ModelVisualizer
viz = ModelVisualizer()
fig = viz.feature_importance(model, feature_names, top_n=15)
fig.update_layout(title="SHAP Feature Importance: Credit Default Prediction")
fig.write_html("ex3_shap_importance.html")
print("Saved: ex3_shap_importance.html")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Dependence plots — individual feature effects
# ══════════════════════════════════════════════════════════════════════

# Top 5 features — how does each influence default probability?
top_features = [name for name, _ in importance_ranking[:5]]

print(f"\n=== Dependence Analysis ===")
for feat in top_features:
    feat_idx = feature_names.index(feat)
    feat_vals = X_test[:, feat_idx]
    feat_shap = shap_vals[:, feat_idx]

    # Correlation between feature value and SHAP value
    corr = np.corrcoef(
        feat_vals[~np.isnan(feat_vals)], feat_shap[~np.isnan(feat_vals)]
    )[0, 1]
    direction = "↑ increases default risk" if corr > 0 else "↑ decreases default risk"
    print(f"  {feat}: correlation = {corr:.3f} ({direction})")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Interaction effects
# ══════════════════════════════════════════════════════════════════════

# SHAP interaction values (O(TL²D²) — more expensive)
# For efficiency, use a sample
sample_size = min(1000, X_test.shape[0])
X_sample = X_test[:sample_size]

shap_interaction = explainer.shap_interaction_values(X_sample)
if isinstance(shap_interaction, list):
    shap_interaction = shap_interaction[1]

# Find strongest interactions
n_features = len(feature_names)
interaction_strengths = []
for i in range(n_features):
    for j in range(i + 1, n_features):
        strength = np.abs(shap_interaction[:, i, j]).mean()
        interaction_strengths.append((feature_names[i], feature_names[j], strength))

interaction_strengths.sort(key=lambda x: x[2], reverse=True)

print(f"\n=== Top Feature Interactions ===")
for f1, f2, strength in interaction_strengths[:10]:
    print(f"  {f1} × {f2}: {strength:.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4 (LIME): Model-agnostic local linear approximations
# ══════════════════════════════════════════════════════════════════════
# LIME (Local Interpretable Model-agnostic Explanations):
# For a single prediction x, LIME:
#   1. Generates perturbed samples around x
#   2. Weights them by proximity to x (exponential kernel)
#   3. Fits a sparse linear model on the weighted samples
# The linear model coefficients are the local feature importances.
#
# Key difference from SHAP:
#   SHAP = global consistency via Shapley values (game theory)
#   LIME = local fidelity only; explanations may vary across calls

try:
    from lime.lime_tabular import LimeTabularExplainer  # type: ignore

    lime_explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=["no_default", "default"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    # Explain the highest-risk prediction with LIME
    risk_order_lime = np.argsort(y_proba)
    high_risk_idx_lime = risk_order_lime[-1]

    lime_exp = lime_explainer.explain_instance(
        X_test[high_risk_idx_lime],
        model.predict_proba,
        num_features=10,
        top_labels=1,
    )

    print(f"\n=== LIME Explanation (highest-risk sample) ===")
    print(f"P(default) = {y_proba[high_risk_idx_lime]:.4f}")
    print(f"\nLIME local feature importances:")
    for feat_desc, weight in lime_exp.as_list():
        direction = "↑risk" if weight > 0 else "↓risk"
        print(f"  {feat_desc:<45} {weight:+.4f} ({direction})")

    print("\nLIME vs SHAP:")
    print("  LIME: fast, model-agnostic, but can be unstable (different perturbations)")
    print("  SHAP: theoretically grounded (Shapley values), exact for trees")
    print(
        "  Production recommendation: use TreeSHAP for tree models; LIME for black-boxes"
    )

except ImportError:
    print("\n=== LIME (lime not installed) ===")
    print("Install with: pip install lime")
    print("LIME: generates perturbed samples near x, fits sparse local linear model")
    print("Coefficient of the local model = feature importance for that prediction")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Local interpretation — explain individual predictions
# ══════════════════════════════════════════════════════════════════════

# Explain the highest-risk and lowest-risk predictions
risk_order = np.argsort(y_proba)

high_risk_idx = risk_order[-1]
low_risk_idx = risk_order[0]

for label, idx in [("Highest Risk", high_risk_idx), ("Lowest Risk", low_risk_idx)]:
    print(f"\n=== {label} (predicted P(default) = {y_proba[idx]:.4f}) ===")
    # Top contributing features for this prediction
    sample_shap = shap_vals[idx]
    sorted_contrib = sorted(
        zip(feature_names, sample_shap, X_test[idx]),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    for name, shap_val, feat_val in sorted_contrib[:8]:
        direction = "↑risk" if shap_val > 0 else "↓risk"
        print(f"  {name} = {feat_val:.2f} → SHAP = {shap_val:+.4f} ({direction})")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Fairness audit — SHAP across protected attributes
# ══════════════════════════════════════════════════════════════════════
# Governance question: does the model treat protected groups fairly?

# Check if protected attributes are in the features
protected_candidates = ["age", "gender", "ethnicity", "marital_status"]
protected_in_model = [f for f in protected_candidates if f in feature_names]

if protected_in_model:
    print(f"\n=== Fairness Audit ===")
    print(f"Protected attributes in model: {protected_in_model}")
    print("⚠ These features may encode bias. Check SHAP contributions:")

    for attr in protected_in_model:
        attr_idx = feature_names.index(attr)
        attr_shap = shap_vals[:, attr_idx]
        print(f"\n  {attr}:")
        print(f"    Mean |SHAP|: {np.abs(attr_shap).mean():.4f}")
        print(
            f"    Rank: #{[n for n, _ in importance_ranking].index(attr) + 1} / {len(feature_names)}"
        )

        # Distribution of SHAP values by feature value
        attr_vals = X_test[:, attr_idx]
        unique_vals = np.unique(attr_vals[~np.isnan(attr_vals)])
        if len(unique_vals) <= 10:
            for val in sorted(unique_vals):
                mask = attr_vals == val
                mean_shap = attr_shap[mask].mean()
                print(
                    f"    Value={val:.0f}: mean SHAP = {mean_shap:+.4f} (n={mask.sum()})"
                )

    print("\n  Recommendation: if protected attributes rank in top 10,")
    print("  consider disparate impact testing before deployment.")
else:
    print("\n✓ No protected attributes found in feature set")

print("\n✓ Exercise 3 complete — SHAP interpretability + fairness audit")
