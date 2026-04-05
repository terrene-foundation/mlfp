# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT3 — Exercise 4: Workflow Orchestration with Kailash Core SDK
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build an ML pipeline using Kailash WorkflowBuilder — load,
#   preprocess, train, evaluate, persist to DataFlow. Learn the
#   runtime.execute(workflow.build()) pattern.
#
# TASKS:
#   1. Build a Kailash workflow for the ML pipeline
#   2. Define @db.model for evaluation results (DataFlow)
#   3. Execute the workflow with LocalRuntime
#   4. Persist results with db.express
#   5. Define ModelSignature (input/output schema for trained models)
#   6. Query persisted results
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl
from dotenv import load_dotenv

from kailash.workflow.builder import WorkflowBuilder
from kailash.runtime import LocalRuntime
from kailash_dataflow import DataFlow, field
from kailash_ml import PreprocessingPipeline, ModelVisualizer
from kailash_ml.interop import to_sklearn_input
from kailash_ml.engines.training_pipeline import TrainingPipeline, ModelSpec, EvalSpec

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
credit = loader.load("ascent03", "sg_credit_scoring.parquet")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Build a Kailash ML workflow
# ══════════════════════════════════════════════════════════════════════
# WorkflowBuilder: nodes, connections, runtime.execute(workflow.build())
# This is the Kailash Core SDK pattern for orchestrating multi-step operations.

# TODO: Create a WorkflowBuilder named "credit_scoring_pipeline"
workflow = ____  # Hint: WorkflowBuilder("credit_scoring_pipeline")

# Node 1: Data loading and preprocessing
# TODO: Add a "DataPreprocessNode" with node_id "preprocess" and the given config dict
workflow.add_node(
    ____,  # Hint: "DataPreprocessNode" (node type as string literal)
    ____,  # Hint: "preprocess" (node id as string literal)
    {
        "data_source": "sg_credit_scoring",
        "target": "default",
        "train_size": 0.8,
        "seed": 42,
        "normalize": False,
        "categorical_encoding": "ordinal",
        "imputation_strategy": "median",
    },
)

# Node 2: Model training
# TODO: Add a "ModelTrainNode" with node_id "train", the given config, and connections=["preprocess"]
workflow.add_node(
    ____,  # Hint: "ModelTrainNode"
    ____,  # Hint: "train"
    {
        "model_class": "lightgbm.LGBMClassifier",
        "hyperparameters": {
            "n_estimators": 500,
            "learning_rate": 0.1,
            "max_depth": 6,
            "scale_pos_weight": 7.3,
        },
    },
    connections=____,  # Hint: ["preprocess"]
)

# Node 3: Model evaluation
# TODO: Add a "ModelEvalNode" with node_id "evaluate", the given config, and connections=["train"]
workflow.add_node(
    ____,  # Hint: "ModelEvalNode"
    ____,  # Hint: "evaluate"
    {
        "metrics": ["accuracy", "f1", "auc_roc", "auc_pr", "log_loss"],
    },
    connections=____,  # Hint: ["train"]
)

# Node 4: Persist results
# TODO: Add a "PersistNode" with node_id "persist", the given config, and connections=["evaluate"]
workflow.add_node(
    ____,  # Hint: "PersistNode"
    ____,  # Hint: "persist"
    {
        "storage": "sqlite:///ascent03_models.db",
    },
    connections=____,  # Hint: ["evaluate"]
)

# Build and execute
# TODO: Create a LocalRuntime instance
runtime = ____  # Hint: LocalRuntime()
print("=== Executing Workflow ===")
# TODO: Execute the workflow — MUST call .build() on the workflow before passing to runtime
results, run_id = ____  # Hint: runtime.execute(workflow.build())

print(f"Run ID: {run_id}")
print(f"Node results: {list(results.keys())}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Define @db.model for evaluation results
# ══════════════════════════════════════════════════════════════════════
# DataFlow lets us persist structured data to a database using
# declarative models. This is how ML artifacts get stored.

# TODO: Create a DataFlow instance pointing to "sqlite:///ascent03_models.db"
db = ____  # Hint: DataFlow("sqlite:///ascent03_models.db")


# TODO: Decorate this class with @db.model to register it as a DataFlow model
____


class ModelEvaluation:
    """Stores evaluation results for trained models."""

    # TODO: Define id as primary key using field(primary_key=True)
    id: int = ____  # Hint: field(primary_key=True)
    model_name: str = field()
    dataset: str = field()
    accuracy: float = field()
    f1_score: float = field()
    auc_roc: float = field()
    auc_pr: float = field()
    log_loss: float = field()
    train_size: int = field()
    test_size: int = field()
    feature_count: int = field()


# TODO: Decorate this class with @db.model
____


class ModelArtifact:
    """Stores model metadata and serialisation path."""

    # TODO: Define id as primary key
    id: int = ____  # Hint: field(primary_key=True)
    model_name: str = field()
    version: int = field()
    artifact_path: str = field()
    is_production: bool = field(default=False)
    created_by: str = field(default="ascent03")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Train and evaluate manually (parallel to workflow)
# ══════════════════════════════════════════════════════════════════════
# The workflow orchestrates the pipeline. Here we also do it manually
# to understand what each node does internally.

import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
)

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

model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    scale_pos_weight=(1 - y_train.mean()) / y_train.mean(),
    random_state=42,
    verbose=-1,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

eval_metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "auc_roc": roc_auc_score(y_test, y_proba),
    "auc_pr": average_precision_score(y_test, y_proba),
    "log_loss": log_loss(y_test, y_proba),
}

print(f"\n=== Manual Evaluation ===")
for metric, value in eval_metrics.items():
    print(f"  {metric}: {value:.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Persist results with db.express
# ══════════════════════════════════════════════════════════════════════


async def persist_results():
    """Store evaluation results and model metadata in DataFlow."""
    await db.initialize()

    # TODO: Use db.express.create() to store a "ModelEvaluation" record
    #       with the fields from eval_metrics plus train_size, test_size, feature_count
    eval_record = (
        await ____
    )  # Hint: db.express.create("ModelEvaluation", {"model_name": ..., "accuracy": eval_metrics["accuracy"], ...})
    print(f"\nPersisted evaluation: ID={eval_record['id']}")

    # TODO: Use db.express.create() to store a "ModelArtifact" record
    #       with model_name, version=1, artifact_path, is_production=False, created_by
    artifact_record = (
        await ____
    )  # Hint: db.express.create("ModelArtifact", {"model_name": "lgbm_credit_v1", ...})
    print(f"Persisted artifact: ID={artifact_record['id']}")

    return eval_record, artifact_record


eval_record, artifact_record = asyncio.run(persist_results())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Define ModelSignature
# ══════════════════════════════════════════════════════════════════════
# ModelSignature is the input/output contract for a trained model.
# It specifies what features are required and what outputs are produced.

from kailash_ml.types import ModelSignature, FeatureSchema, FeatureField

# TODO: Create a FeatureSchema named "credit_model_input" with a FeatureField
#       for each column in col_info["feature_columns"] with dtype="float64",
#       and entity_id_column="application_id"
input_schema = ____  # Hint: FeatureSchema(name="credit_model_input", features=[FeatureField(name=f, dtype="float64") for f in col_info["feature_columns"]], entity_id_column="application_id")

# TODO: Create a ModelSignature with input_schema, output_columns, output_dtypes, model_type
signature = ____  # Hint: ModelSignature(input_schema=input_schema, output_columns=["default_probability", "default_label"], output_dtypes=["float64", "int64"], model_type="classifier")

print(f"\n=== ModelSignature ===")
print(f"Input features: {len(signature.input_schema.features)}")
print(f"Output: {signature.output_columns}")
print(f"Model type: {signature.model_type}")
print(f"\nModelSignature is the contract between model and deployment.")
print("InferenceServer (M4) validates inputs against this signature.")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Query persisted results
# ══════════════════════════════════════════════════════════════════════


async def query_results():
    """Query stored evaluations to compare models."""
    # TODO: List all "ModelEvaluation" records using db.express.list()
    evals = await ____  # Hint: db.express.list("ModelEvaluation")
    print(f"\n=== Persisted Evaluations ({len(evals)}) ===")
    for e in evals:
        print(
            f"  {e['model_name']}: AUC-ROC={e['auc_roc']:.4f}, AUC-PR={e['auc_pr']:.4f}"
        )

    # TODO: List all "ModelArtifact" records using db.express.list()
    artifacts = await ____  # Hint: db.express.list("ModelArtifact")
    print(f"\nModel Artifacts ({len(artifacts)}):")
    for a in artifacts:
        status = "PRODUCTION" if a["is_production"] else "staging"
        print(f"  {a['model_name']} v{a['version']}: {status}")

    await db.close()


asyncio.run(query_results())

print("\n✓ Exercise 4 complete — Kailash workflow orchestration + DataFlow persistence")
print("  Pattern: WorkflowBuilder → runtime.execute(workflow.build()) → db.express")
