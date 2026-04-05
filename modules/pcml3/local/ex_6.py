# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT3 — Exercise 6: DataFlow and Persistence
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Persist ML evaluation results to a DataFlow database using
#   @db.model, db.express CRUD, and async context. Query and compare
#   model runs to demonstrate reproducible experiment tracking.
#
# TASKS:
#   1. Design @db.model schema for ML evaluation results
#   2. Train multiple model variants to generate experiment data
#   3. Persist all results with db.express.create
#   4. Query, filter, and compare runs with db.express.list
#   5. Update records: mark the best model as production candidate
#   6. Explore async patterns: async with, context managers, connection lifecycle
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

from kailash_dataflow import DataFlow, field
from kailash_ml import PreprocessingPipeline, ModelVisualizer
from kailash_ml.interop import to_sklearn_input

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
credit = loader.load("ascent03", "sg_credit_scoring.parquet")

pipeline = PreprocessingPipeline()
result = pipeline.setup(
    credit,
    target="default",
    seed=42,
    normalize=False,
    categorical_encoding="ordinal",
    imputation_strategy="median",
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

print(f"=== Singapore Credit Data ===")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Default rate (train): {y_train.mean():.2%}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Design @db.model schema
# ══════════════════════════════════════════════════════════════════════
# DataFlow uses declarative models — define once, get CRUD for free.
# The @db.model decorator registers the class with the DataFlow engine.
# field(primary_key=True) marks the auto-increment primary key.

DB_URL = "sqlite:///ascent03_ex6_results.db"

# TODO: Create a DataFlow instance with the DB_URL
db = ____  # Hint: DataFlow(DB_URL)


# TODO: Decorate ModelRun with @db.model
____


class ModelRun:
    """A single training run with its configuration and metrics."""

    id: int = field(primary_key=True)
    run_name: str = field()
    model_family: str = field()
    dataset: str = field()
    hyperparams_json: str = field(default="{}")
    accuracy: float = field(default=0.0)
    f1_score: float = field(default=0.0)
    auc_roc: float = field(default=0.0)
    auc_pr: float = field(default=0.0)
    log_loss_val: float = field(default=0.0)
    brier_score: float = field(default=0.0)
    train_samples: int = field(default=0)
    test_samples: int = field(default=0)
    feature_count: int = field(default=0)
    is_production_candidate: bool = field(default=False)
    notes: str = field(default="")


# TODO: Decorate FeatureImportance with @db.model
____


class FeatureImportance:
    """Top feature importances for a model run — linked by run_id."""

    id: int = field(primary_key=True)
    run_id: int = field()
    feature_name: str = field()
    importance: float = field()
    rank: int = field()


print("\n=== DataFlow Schema Defined ===")
print("ModelRun: evaluation results + hyperparams per training run")
print("FeatureImportance: top features per run (linked by run_id)")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Train multiple model variants
# ══════════════════════════════════════════════════════════════════════

model_configs = [
    {
        "run_name": "lgbm_default",
        "hyperparams": {
            "n_estimators": 300,
            "learning_rate": 0.1,
            "max_depth": 6,
            "scale_pos_weight": (1 - y_train.mean()) / y_train.mean(),
        },
        "notes": "Baseline with cost-sensitive weight",
    },
    {
        "run_name": "lgbm_shallow",
        "hyperparams": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 3,
            "scale_pos_weight": (1 - y_train.mean()) / y_train.mean(),
        },
        "notes": "Shallow trees — higher bias, lower variance",
    },
    {
        "run_name": "lgbm_deep",
        "hyperparams": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 8,
            "num_leaves": 63,
            "scale_pos_weight": (1 - y_train.mean()) / y_train.mean(),
        },
        "notes": "Deeper trees — lower bias, higher variance",
    },
    {
        "run_name": "lgbm_regularised",
        "hyperparams": {
            "n_estimators": 400,
            "learning_rate": 0.08,
            "max_depth": 6,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_samples": 30,
            "scale_pos_weight": (1 - y_train.mean()) / y_train.mean(),
        },
        "notes": "L1+L2 regularisation applied",
    },
]

trained_runs = []

for config in model_configs:
    print(f"\nTraining {config['run_name']}...")
    model = lgb.LGBMClassifier(**config["hyperparams"], random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_proba),
        "auc_pr": average_precision_score(y_test, y_proba),
        "log_loss_val": log_loss(y_test, y_proba),
        "brier_score": brier_score_loss(y_test, y_proba),
    }

    importances = model.feature_importances_
    top_features = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    trained_runs.append(
        {
            "config": config,
            "metrics": metrics,
            "top_features": top_features,
            "model": model,
        }
    )

    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}  AUC-PR: {metrics['auc_pr']:.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Persist results with db.express.create
# ══════════════════════════════════════════════════════════════════════
# db.express is the zero-config CRUD layer.
# await db.express.create("ModelName", {field: value}) inserts a row.
# Pattern: always await db.initialize() before first use.

run_ids = []


async def persist_all_runs():
    """Persist all training runs and feature importances to DataFlow."""
    # TODO: Initialize the DataFlow connection
    await db.____()  # Hint: await db.initialize()

    print("\n=== Persisting Results ===")
    for run_data in trained_runs:
        config = run_data["config"]
        metrics = run_data["metrics"]

        # TODO: Create a ModelRun record using db.express.create
        run_record = await db.express.create(
            ____,  # Hint: "ModelRun"
            {
                "run_name": config["run_name"],
                "model_family": "lightgbm",
                "dataset": "sg_credit_scoring",
                "hyperparams_json": json.dumps(config["hyperparams"]),
                "accuracy": metrics["accuracy"],
                "f1_score": metrics["f1_score"],
                "auc_roc": metrics["auc_roc"],
                "auc_pr": metrics["auc_pr"],
                "log_loss_val": metrics["log_loss_val"],
                "brier_score": metrics["brier_score"],
                "train_samples": X_train.shape[0],
                "test_samples": X_test.shape[0],
                "feature_count": X_train.shape[1],
                "is_production_candidate": False,
                "notes": config["notes"],
            },
        )
        run_id = run_record["id"]
        run_ids.append(run_id)
        print(f"  Persisted {config['run_name']}: ID={run_id}")

        # Persist top feature importances for this run
        for rank, (feat_name, importance) in enumerate(
            run_data["top_features"], start=1
        ):
            # TODO: Create a FeatureImportance record linked to this run_id
            await db.express.create(
                ____,  # Hint: "FeatureImportance"
                {
                    "run_id": run_id,
                    "feature_name": feat_name,
                    "importance": float(importance),
                    "rank": rank,
                },
            )

    print(
        f"\nPersisted {len(trained_runs)} runs, {len(trained_runs) * 10} feature records"
    )


asyncio.run(persist_all_runs())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Query, filter, and compare runs
# ══════════════════════════════════════════════════════════════════════
# db.express.list("ModelName", filter_dict) returns matching records.
# db.express.get("ModelName", id) retrieves a single record by primary key.


async def query_and_compare():
    """Retrieve and compare all stored model runs."""

    # TODO: List all ModelRun records using db.express.list
    all_runs = await db.express.list(____)  # Hint: "ModelRun"
    print(f"\n=== All Stored Runs ({len(all_runs)}) ===")
    print(f"{'Run Name':<25} {'AUC-ROC':>10} {'AUC-PR':>10} {'Brier':>8} {'Notes':<35}")
    print("─" * 95)
    for run in sorted(all_runs, key=lambda r: r["auc_pr"], reverse=True):
        print(
            f"{run['run_name']:<25} {run['auc_roc']:>10.4f} {run['auc_pr']:>10.4f} "
            f"{run['brier_score']:>8.4f} {run['notes']:<35}"
        )

    high_quality = [r for r in all_runs if r["auc_pr"] > 0.30]
    print(f"\nRuns with AUC-PR > 0.30: {len(high_quality)}")

    # TODO: List ModelRun records filtered by model_family and notes
    regularised = await db.express.list(
        ____,  # Hint: "ModelRun"
        {"model_family": "lightgbm", "notes": "L1+L2 regularisation applied"},
    )
    print(f"Regularised runs: {len(regularised)}")

    # TODO: Retrieve the first run by its ID using db.express.get
    if run_ids:
        first_run = await db.express.get(____, str(run_ids[0]))  # Hint: "ModelRun"
        print(f"\nFirst run retrieved by ID={run_ids[0]}:")
        hyperparams = json.loads(first_run["hyperparams_json"])
        print(f"  Hyperparams: {hyperparams}")

    return all_runs


all_runs = asyncio.run(query_and_compare())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Update records — mark the best model
# ══════════════════════════════════════════════════════════════════════
# db.express.update("ModelName", id, {field: new_value}) updates a record.


async def promote_best():
    """Find the best run by AUC-PR and mark it as production candidate."""
    all_runs = await db.express.list("ModelRun")

    best = max(all_runs, key=lambda r: r["auc_pr"])
    print(f"\n=== Promoting Best Run ===")
    print(f"Best run: {best['run_name']} (AUC-PR={best['auc_pr']:.4f})")

    # TODO: Update the best run to set is_production_candidate=True
    updated = await db.express.update(
        ____,  # Hint: "ModelRun"
        str(best["id"]),
        {
            "is_production_candidate": True,
            "notes": best["notes"]
            + f" | Promoted {datetime.now().strftime('%Y-%m-%d')}",
        },
    )
    print(f"Updated ID={updated['id']}: is_production_candidate = True")

    # TODO: Verify the update by retrieving the record with db.express.get
    confirmed = await db.express.get(____, str(best["id"]))  # Hint: "ModelRun"
    print(
        f"Confirmed: {confirmed['run_name']} → production_candidate={confirmed['is_production_candidate']}"
    )

    # TODO: List only production candidates (is_production_candidate=True)
    candidates = await db.express.list(
        ____, {"is_production_candidate": True}
    )  # Hint: "ModelRun"
    print(f"\nProduction candidates: {[c['run_name'] for c in candidates]}")


asyncio.run(promote_best())


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Async patterns — connection lifecycle
# ══════════════════════════════════════════════════════════════════════
# Key async patterns:
# 1. await db.initialize()         — opens connection pool
# 2. await db.express.create(...)  — single insert
# 3. await db.express.list(...)    — query with optional filter
# 4. await db.express.get(...)     — retrieve by primary key
# 5. await db.express.update(...)  — partial update
# 6. await db.express.delete(...)  — delete by primary key
# 7. await db.close()              — release connection pool


async def demonstrate_async_context():
    """Show async context manager pattern for connection management."""

    print("\n=== Async Connection Lifecycle ===")
    print("1. db.initialize()  — open pool (done once per process)")
    print("2. db.express.*     — use CRUD operations")
    print("3. db.close()       — release pool (done at shutdown)")

    print("\nPattern: async with db.connect() as conn:")
    print("  # conn is the underlying connection object")
    print("  # automatically released when block exits")
    print("  # even on exception — guaranteed cleanup")

    # Feature importance retrieval — demonstrate join pattern
    if run_ids:
        print("\n=== Top Feature per Run ===")
        for run_id in run_ids:
            # TODO: List FeatureImportance records for this run_id with rank=1
            fi_records = await db.express.list(
                ____,  # Hint: "FeatureImportance"
                {"run_id": run_id, "rank": 1},
            )
            # TODO: Retrieve the ModelRun record for this run_id
            run = await db.express.get(____, str(run_id))  # Hint: "ModelRun"
            if fi_records:
                top_feat = fi_records[0]
                print(
                    f"  {run['run_name']}: top feature = {top_feat['feature_name']} "
                    f"(importance={top_feat['importance']:.2f})"
                )

    # TODO: Close the DataFlow connection pool
    await db.____()  # Hint: await db.close()
    print("\nConnection pool closed.")


asyncio.run(demonstrate_async_context())


# ══════════════════════════════════════════════════════════════════════
# Final comparison visualisation
# ══════════════════════════════════════════════════════════════════════

metrics_by_run = {
    r["run_name"]: {
        "AUC_ROC": r["auc_roc"],
        "AUC_PR": r["auc_pr"],
        "Brier_Score": r["brier_score"],
    }
    for r in all_runs
}

viz = ModelVisualizer()
fig = viz.metric_comparison(metrics_by_run)
fig.update_layout(title="Model Run Comparison — Stored in DataFlow")
fig.write_html("ex6_run_comparison.html")
print("\nSaved: ex6_run_comparison.html")

print("\n✓ Exercise 6 complete — DataFlow persistence for ML experiments")
print("  Patterns learned:")
print("  • @db.model = declarative schema → auto CRUD")
print("  • db.express.create/list/get/update = zero-boilerplate operations")
print("  • async/await = non-blocking I/O for concurrent ML pipelines")
print("  • Persistence = reproducible experiments + governance audit trail")
