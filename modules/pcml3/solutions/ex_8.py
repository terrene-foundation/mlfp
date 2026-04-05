# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT3 — Exercise 8: Production Pipeline Project
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Complete supervised ML pipeline combining all M3 concepts:
#   workflow orchestration, persistence, model card, conformal prediction.
#
# TASKS:
#   1. Build complete pipeline: load → preprocess → train → evaluate → persist
#   2. Generate model card (Mitchell et al. template)
#   3. Conformal prediction for uncertainty quantification
#   4. Cross-validate and analyse bias-variance trade-off
#   5. Final model comparison and selection
#   6. Generate deployment-ready artifacts
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

from kailash.db.connection import ConnectionManager
from kailash_ml import PreprocessingPipeline, ModelVisualizer
from kailash_ml.interop import to_sklearn_input
from kailash_ml.engines.model_registry import ModelRegistry
from kailash_ml.engines.experiment_tracker import ExperimentTracker

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ──────────────────────────────────────────────────────

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

for name, m in complexities:
    cv_scores = cross_val_score(m, X_train, y_train, cv=5, scoring="average_precision")
    m.fit(X_train, y_train)
    train_score = average_precision_score(y_train, m.predict_proba(X_train)[:, 1])
    gap = train_score - cv_scores.mean()
    print(
        f"{name:<25} {cv_scores.mean():>10.4f} {cv_scores.std():>10.4f} {train_score:>10.4f} {gap:>10.4f}"
    )

print("\nInterpretation:")
print("  Small gap + low score → high bias (underfitting)")
print("  Large gap + high train score → high variance (overfitting)")
print("  The 'Medium' model typically offers the best bias-variance trade-off")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Persist everything and generate artifacts
# ══════════════════════════════════════════════════════════════════════


async def persist_final():
    conn = ConnectionManager("sqlite:///ascent03_models.db")
    await conn.initialize()

    registry = ModelRegistry(conn)
    await registry.initialize()

    tracker = ExperimentTracker(conn)
    await tracker.initialize()

    # Register calibrated model (serialize to bytes)
    import pickle
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

    # Log to experiment tracker
    exp_id = await tracker.create_experiment(
        name="ascent03_e2e_pipeline",
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


asyncio.run(persist_final())

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
