# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT3 — Exercise 7: Model Registry and Hyperparameter Search
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Use HyperparameterSearch for Bayesian optimization, then
#   register the best model in ModelRegistry with staging → production
#   promotion as a governance gate.
#
# TASKS:
#   1. Define search space with SearchSpace and ParamDistribution
#   2. Run HyperparameterSearch with Bayesian optimization
#   3. Analyse search results and convergence
#   4. Register best model in ModelRegistry
#   5. Promote from staging to production (governance gate)
#   6. Retrieve and compare model versions
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import polars as pl
from kailash.db.connection import ConnectionManager
from kailash_ml import PreprocessingPipeline, ModelVisualizer
from kailash_ml.interop import to_sklearn_input
from kailash_ml.engines.hyperparameter_search import (
    HyperparameterSearch,
    SearchSpace,
    SearchConfig,
    ParamDistribution,
)
from kailash_ml.engines.model_registry import ModelRegistry

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
# TASK 1: Define search space
# ══════════════════════════════════════════════════════════════════════

search_space = SearchSpace(
    params=[
        ParamDistribution(
            name="n_estimators",
            distribution="int_uniform",
            low=100,
            high=1000,
        ),
        ParamDistribution(
            name="learning_rate",
            distribution="log_uniform",
            low=0.01,
            high=0.5,
        ),
        ParamDistribution(
            name="max_depth",
            distribution="int_uniform",
            low=3,
            high=10,
        ),
        ParamDistribution(
            name="num_leaves",
            distribution="int_uniform",
            low=15,
            high=127,
        ),
        ParamDistribution(
            name="min_child_samples",
            distribution="int_uniform",
            low=5,
            high=100,
        ),
        ParamDistribution(
            name="subsample",
            distribution="uniform",
            low=0.5,
            high=1.0,
        ),
        ParamDistribution(
            name="colsample_bytree",
            distribution="uniform",
            low=0.5,
            high=1.0,
        ),
        ParamDistribution(
            name="reg_alpha",
            distribution="log_uniform",
            low=1e-8,
            high=10.0,
        ),
        ParamDistribution(
            name="reg_lambda",
            distribution="log_uniform",
            low=1e-8,
            high=10.0,
        ),
    ]
)

print("=== Search Space ===")
for p in search_space.params:
    print(f"  {p.name}: {p.distribution} [{p.low}, {p.high}]")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Run HyperparameterSearch
# ══════════════════════════════════════════════════════════════════════

config = SearchConfig(
    model_class="lightgbm.LGBMClassifier",
    metric="average_precision",  # AUC-PR (better for imbalanced)
    direction="maximize",
    n_trials=50,
    cv_folds=5,
    seed=42,
    early_stopping_rounds=10,
    timeout_seconds=300,
)


async def run_search():
    search = HyperparameterSearch()

    search_result = await search.run(
        X=X_train,
        y=y_train,
        search_space=search_space,
        config=config,
    )

    print(f"\n=== Search Results ===")
    print(f"Best score (AUC-PR): {search_result.best_score:.4f}")
    print(f"Best params:")
    for k, v in search_result.best_params.items():
        print(f"  {k}: {v}")
    print(f"Total trials: {search_result.n_trials}")
    print(f"Time: {search_result.elapsed_seconds:.1f}s")

    # Trial history
    print(f"\nTop 5 trials:")
    sorted_trials = sorted(search_result.trials, key=lambda t: t["score"], reverse=True)
    for i, trial in enumerate(sorted_trials[:5]):
        print(
            f"  #{i+1}: score={trial['score']:.4f}, "
            f"lr={trial['params'].get('learning_rate', '?'):.4f}, "
            f"depth={trial['params'].get('max_depth', '?')}"
        )

    return search_result


search_result = asyncio.run(run_search())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Analyse convergence
# ══════════════════════════════════════════════════════════════════════

# Plot optimisation convergence
scores = [t["score"] for t in search_result.trials]
best_so_far = np.maximum.accumulate(scores)

viz = ModelVisualizer()
fig = viz.training_history(
    {"Best Score": best_so_far.tolist()},
    x_label="Trial",
)
fig.update_layout(title="Hyperparameter Search Convergence")
fig.write_html("ex5_search_convergence.html")
print("\nSaved: ex5_search_convergence.html")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Register best model in ModelRegistry
# ══════════════════════════════════════════════════════════════════════


async def register_model():
    conn = ConnectionManager("sqlite:///ascent03_models.db")
    await conn.initialize()

    registry = ModelRegistry(conn)
    await registry.initialize()

    # Train final model with best params
    import lightgbm as lgb

    best_model = lgb.LGBMClassifier(
        **search_result.best_params,
        random_state=42,
        verbose=-1,
    )
    best_model.fit(X_train, y_train)

    # Register in ModelRegistry (serialize model to bytes)
    import pickle
    from kailash_ml.types import MetricSpec

    model_bytes = pickle.dumps(best_model)
    model_version = await registry.register_model(
        name="credit_default_lgbm",
        artifact=model_bytes,
        metrics=[MetricSpec(name="auc_pr", value=search_result.best_score)],
    )
    model_id = model_version.version

    print(f"\n=== Model Registered ===")
    print(f"Model ID: {model_id}")
    print(f"Status: staging (not yet production)")

    return conn, registry, model_id, best_model


conn, registry, model_id, best_model = asyncio.run(register_model())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Promote staging → production (governance gate)
# ══════════════════════════════════════════════════════════════════════


async def promote_model():
    """
    Promotion is a governance gate:
    - A model cannot reach production without explicit promotion
    - This creates an audit trail: who promoted, when, why
    - EATP concept: model provenance chain
    """

    # Verify model quality before promotion
    from sklearn.metrics import roc_auc_score, average_precision_score

    y_proba = best_model.predict_proba(X_test)[:, 1]
    auc_pr = average_precision_score(y_test, y_proba)
    auc_roc = roc_auc_score(y_test, y_proba)

    print(f"\n=== Pre-Promotion Validation ===")
    print(f"Test AUC-PR:  {auc_pr:.4f}")
    print(f"Test AUC-ROC: {auc_roc:.4f}")

    # Quality gate: must exceed thresholds
    min_auc_pr = 0.30  # Reasonable for 12% default rate
    if auc_pr >= min_auc_pr:
        await registry.promote_model(
            name="credit_default_lgbm",
            version=model_id,
            target_stage="production",
            reason=f"Passed quality gate: AUC-PR={auc_pr:.4f} >= {min_auc_pr}",
        )
        print(f"✓ Model promoted to PRODUCTION")
        print(f"  Reason: AUC-PR={auc_pr:.4f} >= {min_auc_pr} threshold")
    else:
        print(f"✗ Model REJECTED: AUC-PR={auc_pr:.4f} < {min_auc_pr} threshold")

    return auc_pr, auc_roc


auc_pr, auc_roc = asyncio.run(promote_model())


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Query and compare model versions
# ══════════════════════════════════════════════════════════════════════


async def compare_versions():
    """List all registered models and their stages."""
    models = await registry.list_models()
    print(f"\n=== Model Registry ===")
    for m in models:
        print(
            f"  {m.get('name', '?')} (v{m.get('version', '?')}): "
            f"stage={m.get('stage', '?')}, "
            f"AUC-PR={m.get('metrics', {}).get('auc_pr', '?')}"
        )

    # Get production model
    prod_model = await registry.get_production("credit_default_lgbm")
    if prod_model:
        print(
            f"\nProduction model: {prod_model.get('name')} v{prod_model.get('version')}"
        )
        print(f"  Params: {prod_model.get('params', {})}")
    else:
        print("\nNo production model found")

    await conn.close()


asyncio.run(compare_versions())

print("\n✓ Exercise 5 complete — HyperparameterSearch + ModelRegistry promotion")
print("  Key concept: staging → production promotion as governance gate")
