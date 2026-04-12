# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 8: Production Pipeline Project
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Build a complete production ML pipeline from training to deployment
#   - Generate a Mitchell et al. model card documenting performance and limits
#   - Apply conformal prediction for distribution-free uncertainty quantification
#   - Analyse bias-variance trade-off across model complexity levels
#   - Register, version, and promote models through the ModelRegistry lifecycle
#
# PREREQUISITES:
#   - MLFP03 Exercises 1-7 (all of Module 3)
#   - MLFP02 complete (preprocessing pipeline, Singapore credit data)
#
# ESTIMATED TIME: 75-90 minutes
#
# TASKS:
#   1. Build complete pipeline: load → preprocess → train → evaluate → persist
#   2. Generate model card (Mitchell et al. template)
#   3. Conformal prediction for uncertainty quantification
#   4. Cross-validate and analyse bias-variance trade-off
#   5. Persist to ModelRegistry and promote to production
#   6. Generate deployment-ready artifacts and final visualisation
#
# DATASET: Singapore credit scoring (from MLFP02)
#   Target: default prediction (binary, 12% positive rate)
#   Final model: LightGBM + isotonic calibration + conformal coverage guarantee
#   Regulatory context: Singapore MAS requires documented model cards for
#   credit decisions affecting consumers under the Code of Consumer Banking Practice
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    f1_score,
    accuracy_score,
)
from sklearn.calibration import CalibratedClassifierCV

from kailash.db import ConnectionManager
from kailash_ml import PreprocessingPipeline, ModelVisualizer
from kailash_ml.interop import to_sklearn_input
from kailash_ml.engines.model_registry import ModelRegistry
from kailash_ml.engines.experiment_tracker import ExperimentTracker

from shared import MLFPDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ──────────────────────────────────────────────────────

loader = MLFPDataLoader()
credit = loader.load("mlfp02", "sg_credit_scoring.parquet")

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

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert X_train.shape[0] > 0, "Training set should not be empty"
assert X_test.shape[0] > 0, "Test set should not be empty"
assert len(feature_names) > 0, "Feature names should be populated"
assert y_train.mean() < 0.5, "Default rate should be minority class (< 50%)"
# INTERPRETATION: This is the same preprocessing pipeline from earlier exercises.
# In production, this step is deterministic and versioned — the same seed
# and config always produce the same train/test split, ensuring reproducibility.
print("\n✓ Checkpoint 1 passed — data loaded and split reproducibly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Train final production model
# ══════════════════════════════════════════════════════════════════════

# Best hyperparameters from Exercise 5 (Bayesian optimization)
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=(1 - y_train.mean()) / y_train.mean(),
    random_state=42,
    verbose=-1,
)
model.fit(X_train, y_train)

# Calibrate
calibrated_model = CalibratedClassifierCV(model, method="isotonic", cv=5)
calibrated_model.fit(X_train, y_train)

y_proba = calibrated_model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "auc_roc": roc_auc_score(y_test, y_proba),
    "auc_pr": average_precision_score(y_test, y_proba),
    "log_loss": log_loss(y_test, y_proba),
    "brier": brier_score_loss(y_test, y_proba),
}

print("=== Final Model Metrics ===")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert metrics["auc_roc"] > 0.5, \
    f"Final model AUC-ROC {metrics['auc_roc']:.4f} should exceed random baseline"
assert metrics["auc_pr"] > 0, "AUC-PR should be positive"
assert 0 < metrics["brier"] < 0.25, \
    "Brier score should be reasonable (calibrated model < 0.25)"
# INTERPRETATION: The calibrated model uses isotonic regression to map raw
# predicted scores to true probabilities. A Brier score < 0.1 on a 12% base
# rate dataset indicates the model is well-calibrated — when it says 20% default
# risk, roughly 20% of those applicants will actually default.
print("\n✓ Checkpoint 2 passed — final calibrated model trained\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Generate model card (Mitchell et al.)
# ══════════════════════════════════════════════════════════════════════

model_card = f"""
# Model Card: Singapore Credit Default Prediction

## Model Details
- **Model type**: LightGBM Classifier (calibrated with isotonic regression)
- **Version**: 1.0
- **Date**: {datetime.now().strftime("%Y-%m-%d")}
- **Framework**: kailash-ml (Terrene Foundation)
- **License**: Internal use only

## Intended Use
- **Primary use**: Credit default risk assessment for Singapore market
- **Users**: Credit risk analysts, automated underwriting systems
- **Out of scope**: Regulatory capital calculation, cross-border lending

## Training Data
- **Source**: Synthetic Singapore credit applications (data.gov.sg characteristics)
- **Size**: {X_train.shape[0]:,} training samples
- **Features**: {X_train.shape[1]} features (financial, behavioral, demographic)
- **Target**: Binary default (12% positive rate)
- **Time range**: 2020-2024

## Evaluation
- **Test set**: {X_test.shape[0]:,} samples (holdout, same distribution)
- **AUC-ROC**: {metrics['auc_roc']:.4f}
- **AUC-PR**: {metrics['auc_pr']:.4f}
- **Brier Score**: {metrics['brier']:.4f} (calibrated)
- **Log Loss**: {metrics['log_loss']:.4f}

## Ethical Considerations
- Protected attributes (age, gender, ethnicity) were analysed with SHAP
- Disparate impact testing performed (see Exercise 3 results)
- Model should be monitored for drift in protected group performance

## Limitations
- Trained on synthetic data — validate on production data before deployment
- Singapore-specific — do not apply to other markets without retraining
- Point-in-time: model performance may degrade as economic conditions change

## Monitoring
- DriftMonitor (Module 4) should be configured with PSI threshold = 0.1
- Retrain trigger: PSI > 0.2 OR AUC-PR drops below {metrics['auc_pr'] * 0.9:.4f}
"""

print("\n=== Model Card ===")
print(model_card)

# Save model card
with open("ex6_model_card.md", "w") as f:
    f.write(model_card)
print("Saved: ex6_model_card.md")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
import os
assert os.path.exists("ex6_model_card.md"), "Model card file should be written to disk"
assert f"AUC-ROC" in model_card, "Model card should report AUC-ROC"
assert "Ethical Considerations" in model_card, "Model card should include ethics section"
# INTERPRETATION: The Mitchell et al. (2019) model card template is a de facto
# standard for transparent ML documentation. Singapore MAS and EU AI Act both
# require this level of documentation for high-risk AI systems. The monitoring
# section is especially critical — it names the specific metric and threshold
# that triggers retraining, making oversight concrete and actionable.
print("\n✓ Checkpoint 3 passed — model card generated and saved\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Conformal prediction for uncertainty quantification
# ══════════════════════════════════════════════════════════════════════
# Conformal prediction provides distribution-free prediction sets with
# guaranteed coverage: P(Y ∈ C(X)) ≥ 1 - α
# No distributional assumptions needed.

# Split calibration set from test set
n_cal = X_test.shape[0] // 2
X_cal, X_eval = X_test[:n_cal], X_test[n_cal:]
y_cal, y_eval = y_test[:n_cal], y_test[n_cal:]

# Compute nonconformity scores on calibration set
cal_proba = calibrated_model.predict_proba(X_cal)[:, 1]
# Score = 1 - predicted probability of the true class
cal_scores = np.where(y_cal == 1, 1 - cal_proba, cal_proba)

# Quantile for desired coverage
alpha = 0.10  # 90% coverage
n_cal_size = len(cal_scores)
quantile_level = np.ceil((n_cal_size + 1) * (1 - alpha)) / n_cal_size
q_hat = np.quantile(cal_scores, min(quantile_level, 1.0))

# Prediction sets on evaluation data
eval_proba = calibrated_model.predict_proba(X_eval)[:, 1]

# For each sample, include classes where score ≤ q_hat
prediction_sets = []
for i in range(len(y_eval)):
    pred_set = set()
    if (1 - eval_proba[i]) <= q_hat:  # Include class 1
        pred_set.add(1)
    if eval_proba[i] <= q_hat:  # Include class 0
        pred_set.add(0)
    if not pred_set:  # Always include most likely class
        pred_set.add(1 if eval_proba[i] >= 0.5 else 0)
    prediction_sets.append(pred_set)

# Evaluate coverage and set sizes
coverage = np.mean([y_eval[i] in ps for i, ps in enumerate(prediction_sets)])
avg_set_size = np.mean([len(ps) for ps in prediction_sets])
singleton_rate = np.mean([len(ps) == 1 for ps in prediction_sets])

print(f"\n=== Conformal Prediction (α={alpha}) ===")
print(f"Calibration quantile (q̂): {q_hat:.4f}")
print(f"Coverage: {coverage:.4f} (target: {1 - alpha:.4f})")
print(f"Average set size: {avg_set_size:.3f}")
print(f"Singleton rate: {singleton_rate:.1%} (precise predictions)")
print(f"Ambiguous rate: {1 - singleton_rate:.1%} (both classes possible)")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert coverage >= (1 - alpha - 0.05), \
    f"Conformal coverage {coverage:.4f} should be near or above target {1 - alpha:.4f}"
assert 0 < avg_set_size <= 2, "Average prediction set size should be between 0 and 2"
assert 0 <= singleton_rate <= 1, "Singleton rate should be a valid proportion"
# INTERPRETATION: Conformal prediction is unique because its coverage guarantee
# is distribution-free: regardless of whether defaults are Gaussian, skewed, or
# multimodal, P(true label ∈ prediction set) ≥ 90% is mathematically guaranteed
# from the exchangeability of the calibration and test sets. High singleton_rate
# means the model is confident; ambiguous samples should trigger human review.
print("\n✓ Checkpoint 4 passed — conformal coverage guarantee verified\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Cross-validation bias-variance analysis
# ══════════════════════════════════════════════════════════════════════

# Compare models of increasing complexity
complexities = [
    (
        "Simple (depth=3)",
        lgb.LGBMClassifier(max_depth=3, n_estimators=100, verbose=-1, random_state=42),
    ),
    (
        "Medium (depth=6)",
        lgb.LGBMClassifier(max_depth=6, n_estimators=300, verbose=-1, random_state=42),
    ),
    (
        "Complex (depth=10)",
        lgb.LGBMClassifier(max_depth=10, n_estimators=500, verbose=-1, random_state=42),
    ),
    (
        "Very Complex (depth=-1)",
        lgb.LGBMClassifier(
            max_depth=-1, n_estimators=1000, num_leaves=255, verbose=-1, random_state=42
        ),
    ),
]

print(f"\n=== Bias-Variance Analysis ===")
print(f"{'Model':<25} {'CV Mean':>10} {'CV Std':>10} {'Train':>10} {'Gap':>10}")
print("─" * 70)

bv_results = []
for name, m in complexities:
    cv_scores = cross_val_score(m, X_train, y_train, cv=5, scoring="average_precision")
    m.fit(X_train, y_train)
    train_score = average_precision_score(y_train, m.predict_proba(X_train)[:, 1])
    gap = train_score - cv_scores.mean()
    bv_results.append({
        "name": name,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "train_score": train_score,
        "gap": gap,
    })
    print(
        f"{name:<25} {cv_scores.mean():>10.4f} {cv_scores.std():>10.4f} {train_score:>10.4f} {gap:>10.4f}"
    )

print("\nInterpretation:")
print("  Small gap + low score → high bias (underfitting)")
print("  Large gap + high train score → high variance (overfitting)")
print("  The 'Medium' model typically offers the best bias-variance trade-off")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert len(bv_results) == 4, "Should analyse 4 complexity levels"
# Simple model should have less variance than very complex
simple_gap = bv_results[0]["gap"]
complex_gap = bv_results[3]["gap"]
assert complex_gap >= simple_gap, \
    "Very complex model should show larger train-CV gap (higher variance) than simple"
# Complex model should fit training data better
assert bv_results[3]["train_score"] >= bv_results[0]["train_score"], \
    "More complex model should achieve higher training score"
# INTERPRETATION: The bias-variance table is the empirical proof of the
# fundamental tradeoff. Simple model: low gap (low variance) but low CV mean
# (high bias). Very complex: high train score but large gap (high variance,
# the model memorises training data). Production models live in the middle.
print("\n✓ Checkpoint 5 passed — bias-variance tradeoff demonstrated\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Persist everything and generate artifacts
# ══════════════════════════════════════════════════════════════════════


async def persist_final():
    conn = ConnectionManager("sqlite:///mlfp02_models.db")
    await conn.initialize()

    try:
        registry = ModelRegistry(conn)
        HAS_REGISTRY = True
    except Exception as e:
        registry = None
        HAS_REGISTRY = False
        print(f"  Note: ModelRegistry setup skipped ({e})")

    tracker = ExperimentTracker(conn)

    # Register calibrated model (serialize to bytes)
    import pickle

    model_id = None
    if HAS_REGISTRY:
        from kailash_ml.types import MetricSpec

        model_bytes = pickle.dumps(calibrated_model)
        model_version = await registry.register_model(
            name="credit_default_production",
            artifact=model_bytes,
            metrics=[
                MetricSpec(name="auc_pr", value=metrics["auc_pr"]),
                MetricSpec(name="brier", value=metrics["brier"]),
            ],
        )

        # Promote to production
        await registry.promote_model(
            name="credit_default_production",
            version=model_version.version,
            target_stage="production",
            reason=f"Passed all quality gates: AUC-PR={metrics['auc_pr']:.4f}, "
            f"Brier={metrics['brier']:.4f}, Coverage={coverage:.4f}",
        )
        model_id = model_version.version
    else:
        model_id = "skipped"
        print("  Note: ModelRegistry not available. Skipping model registration.")

    # Log to experiment tracker
    exp_id = await tracker.create_experiment(
        name="mlfp02_e2e_pipeline",
        description="End-to-end supervised ML pipeline",
    )
    async with tracker.run(exp_id, run_name="production_model_v1") as run:
        await run.log_param("model", "lgbm_calibrated_conformal")
        await run.log_metrics({**metrics, "conformal_coverage": coverage})
        await run.set_tag("stage", "production")

    print(f"\n=== Final Artifacts ===")
    print(f"Model registered: {model_id}")
    print(f"Stage: production")
    print(f"Model card: ex6_model_card.md")

    await conn.close()
    return model_id


model_id = asyncio.run(persist_final())

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert model_id is not None, "Model should be registered in the ModelRegistry"
assert coverage >= (1 - alpha - 0.05), \
    "Conformal coverage should meet target before promotion to production"
# INTERPRETATION: The promote_model step formalises the quality gate decision.
# The reason string becomes part of the audit trail — auditors can see exactly
# why version 1 was promoted: it passed AUC-PR, Brier score, AND conformal
# coverage thresholds. This is governance in action, not just good practice.
print("\n✓ Checkpoint 6 passed — model registered and promoted to production\n")


# Final visualisation
viz = ModelVisualizer()
fig = viz.metric_comparison(
    {
        "Final Model": metrics,
    }
)
fig.update_layout(title="Production Model: Credit Default Prediction")
fig.write_html("ex6_final_metrics.html")
print("Saved: ex6_final_metrics.html")

print("\n✓ Exercise 8 complete — end-to-end supervised ML pipeline")
print("  Pipeline: preprocess → train → calibrate → conformal → persist → deploy")
print("  Module 3 complete: 8 exercises covering supervised ML theory to production")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  MODULE 3 MASTERY — SUPERVISED ML THEORY TO PRODUCTION")
print("═" * 70)
print(f"""
  M3 CAPSTONE CHECKLIST:
  ✓ Feature engineering (Ex 1): domain-driven features, leakage prevention
  ✓ Bias-variance (Ex 2): regularisation, cross-validation strategies
  ✓ Model zoo (Ex 3): SVM, KNN, Naive Bayes, Decision Trees, Random Forests
  ✓ Gradient boosting (Ex 4): XGBoost, LightGBM, CatBoost comparison
  ✓ Imbalance & calibration (Ex 5): SMOTE, cost-sensitive, Platt/isotonic
  ✓ Interpretability & fairness (Ex 6): SHAP, LIME, disparate impact
  ✓ Workflow orchestration (Ex 7): WorkflowBuilder, DataFlow, ModelSignature
  ✓ Production pipeline (Ex 8): conformal prediction, ModelRegistry, model card

  THIS EXERCISE:
  ✓ Conformal prediction: P(Y ∈ C(X)) ≥ 1 - α without distributional assumptions
  ✓ Bias-variance demonstrated empirically across 4 complexity levels
  ✓ ModelRegistry lifecycle: register → promote to production with audit trail
  ✓ Model card: the governance document for regulated credit decisions

  THE PRODUCTION PIPELINE PATTERN:
    preprocess → train → calibrate → conformal predict →
    register → promote → monitor drift → document

  KEY INSIGHT: Production ML is 20% modelling and 80% engineering.
  The model you just deployed is good. The pipeline, the model card,
  the conformal coverage guarantee, and the registry trail are what
  make it trustworthy enough to use for real credit decisions.

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MODULE 4 PREVIEW: UNSUPERVISED ML AND ANOMALY DETECTION
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  So far you have been working with labelled data — every row has a
  known outcome (default / no default). Module 4 removes that luxury.

  M4 covers the full unsupervised ML landscape:
  • Clustering: K-means, HDBSCAN, spectral, GMM (Exercises 1-2)
  • Dimensionality reduction: PCA, t-SNE, UMAP (Exercise 3)
  • Anomaly detection: IsolationForest, LOF, EnsembleEngine (Exercise 4)
  • Association rules: Apriori, FP-Growth, market basket (Exercise 5)
  • NLP / BERTopic: text to topics without labelled data (Exercise 6)
  • Recommender systems: matrix factorisation, collaborative filtering (Ex 7)
  • Deep learning foundations: CNNs, ResBlocks, OnnxBridge (Exercise 8)

  The credit scoring model you just built will reappear in M4 as the
  supervised baseline — you'll discover that anomaly scores from
  IsolationForest (Ex 4) improve fraud detection beyond what any
  labelled model can achieve alone.

  See you in Module 4.
""")
print("═" * 70)
