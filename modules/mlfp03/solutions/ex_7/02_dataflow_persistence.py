# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 7.2: DataFlow Persistence with @db.model
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Declare a database table with `@db.model` (schema = Python class)
#   - Let DataFlow auto-migrate the schema — no hand-written DDL
#   - Run the full CRUD cycle with `db.express`: create, read
#     (find_one / read / list / count), update and delete
#   - Persist a TrainingPipeline evaluation next to its registry reference
#   - Understand why governed ML needs a first-class result store
#
# PREREQUISITES: 01_workflow_builder.py
# ESTIMATED TIME: ~35 min
#
# 5-PHASE R10:
#   1. Theory     — why "write metrics to a database" matters
#   2. Build      — @db.model classes for evaluations + artefacts
#   3. Train      — train once, then create / update / delete rows
#   4. Visualise  — read the persisted records back
#   5. Apply      — a bank's model evidence store
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from dataflow import DataFlow
from kailash_ml import TrainingPipeline
from kailash_ml.engines.training_pipeline import EvalSpec, ModelSpec
from kailash_ml.interop import to_sklearn_input

from shared.mlfp03.ex_7 import (
    DATASET_NAME,
    DB_URL,
    ENGINE_METRICS,
    RANDOM_SEED,
    TARGET_COLUMN,
    build_training_registry,
    compute_classification_metrics,
    credit_feature_schema,
    headline_roi_text,
    load_registered_model,
    prepare_credit_frames,
    print_metric_block,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why persist metrics?
# ════════════════════════════════════════════════════════════════════════
# The `print()` statement is the cheapest, worst form of observability
# for ML. Months after a loan was declined, a model validator, internal
# auditor or supervisor can ask:
#
#   "Show me the model that was used for this decision, the version,
#    the exact metrics on the evaluation set at the time of promotion,
#    and the hyperparameters the training run used."
#
# If your answer is "let me grep some log files", you have already lost.
# The defensible answer is a database table — queryable, indexed by
# model name + version — that records every evaluation, every artefact
# and every review decision. DataFlow turns that table from a week of
# SQL-migration work into a decorated Python class, and `db.express`
# gives you the whole CRUD cycle as one-line async calls.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD @db.model tables
# ════════════════════════════════════════════════════════════════════════

db = DataFlow(DB_URL)


@db.model
class CreditEvaluation:
    """Evaluation metrics for every trained credit-model version.

    ``@db.model`` auto-generates the primary key ``id`` from the plain
    ``id: int`` annotation (plus created_at / updated_at timestamps).
    Plain class annotations ARE the schema; defaults are ordinary
    Python defaults.
    """

    id: int
    model_name: str
    model_version: int
    dataset: str
    accuracy: float
    f1_score: float
    auc_roc: float
    auc_pr: float
    log_loss_val: float
    train_rows: int
    feature_count: int
    hyperparameters: str = "{}"


@db.model
class CreditModelArtifact:
    """Registry reference + human review status for each model version."""

    id: int
    model_name: str
    version: int
    registry_ref: str
    review_status: str = "pending"
    created_by: str = "mlfp03_ex7"


MODEL_NAME = "credit_default_dataflow"
HYPERPARAMETERS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "num_leaves": 31,
    "min_child_samples": 40,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN once, then CREATE / UPDATE / DELETE rows
# ════════════════════════════════════════════════════════════════════════
# Training goes through TrainingPipeline (fit + holdout evaluation +
# registry entry in one engine call) on the dev frame; the registered
# model is then scored on the test frame. Every DataFlow call below is
# awaited inside ONE event loop — the DataFlow connection pool belongs to
# the loop that opened it, so one asyncio.run() drives the whole file.


async def train_with_engine() -> tuple[dict[str, float], int, int, int]:
    """Train via TrainingPipeline, then score the registered model on test.

    Returns (test metrics, registry version, training rows, n_features).
    """
    dev, test, feature_cols = prepare_credit_frames()
    registry, conn = await build_training_registry()
    try:
        pipeline = TrainingPipeline(feature_store=None, registry=registry)
        result = await pipeline.train(
            data=dev,
            schema=credit_feature_schema(feature_cols),
            model_spec=ModelSpec(
                model_class="lightgbm.LGBMClassifier",
                framework="lightgbm",
                hyperparameters=HYPERPARAMETERS,
            ),
            eval_spec=EvalSpec(
                metrics=ENGINE_METRICS, split_strategy="holdout", test_size=0.2
            ),
            experiment_name=MODEL_NAME,
        )
        if result.model_version is None:
            raise RuntimeError("TrainingPipeline did not register the model")
        version = int(result.model_version.version)
        model = await load_registered_model(registry, MODEL_NAME, version)
    finally:
        await conn.close()

    X_test, y_test, _ = to_sklearn_input(
        test, feature_columns=feature_cols, target_column=TARGET_COLUMN
    )
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_classification_metrics(y_test, (y_proba >= 0.5).astype(int), y_proba)
    return metrics, version, dev.height, len(feature_cols)


async def persist_review_cleanup(
    metrics: dict[str, float], version: int, rows: int, n_features: int
) -> dict[str, Any]:
    """CREATE the evaluation + artefact rows, UPDATE the review, DELETE a scratch row."""
    await db.initialize()

    # CREATE — returns the written fields plus rows_affected (no id echo)
    created_eval = await db.express.create(
        "CreditEvaluation",
        {
            "model_name": MODEL_NAME,
            "model_version": version,
            "dataset": DATASET_NAME,
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1"],
            "auc_roc": metrics["auc_roc"],
            "auc_pr": metrics["auc_pr"],
            "log_loss_val": metrics["log_loss"],
            "train_rows": rows,
            "feature_count": n_features,
            "hyperparameters": json.dumps(HYPERPARAMETERS),
        },
    )
    created_artifact = await db.express.create(
        "CreditModelArtifact",
        {
            "model_name": MODEL_NAME,
            "version": version,
            "registry_ref": f"model_registry://{MODEL_NAME}/v{version}",
        },
    )

    # READ the row we just wrote (find_one filters on any columns)
    artifact = await db.express.find_one(
        "CreditModelArtifact", {"model_name": MODEL_NAME, "version": version}
    )

    # UPDATE — a reviewer signs off this version
    reviewed = await db.express.update(
        "CreditModelArtifact", artifact["id"], {"review_status": "approved"}
    )

    # DELETE — a smoke-test row written by a CI check must not linger
    await db.express.create(
        "CreditEvaluation",
        {
            "model_name": "ci_smoke_test",
            "model_version": 0,
            "dataset": "synthetic",
            "accuracy": 0.0,
            "f1_score": 0.0,
            "auc_roc": 0.5,
            "auc_pr": 0.0,
            "log_loss_val": 0.0,
            "train_rows": 0,
            "feature_count": 0,
        },
    )
    smoke = await db.express.find_one("CreditEvaluation", {"model_name": "ci_smoke_test"})
    deleted = await db.express.delete("CreditEvaluation", smoke["id"])
    smoke_left = await db.express.count("CreditEvaluation", {"model_name": "ci_smoke_test"})

    return {
        "created_eval": created_eval,
        "created_artifact": created_artifact,
        "artifact_id": artifact["id"],
        "reviewed": reviewed,
        "deleted": deleted,
        "smoke_left": smoke_left,
    }


async def read_back(version: int) -> dict[str, Any]:
    """READ: by id, by filter, the full list, and a count."""
    evaluation = await db.express.find_one(
        "CreditEvaluation", {"model_name": MODEL_NAME, "model_version": version}
    )
    artifact = await db.express.find_one(
        "CreditModelArtifact", {"model_name": MODEL_NAME, "version": version}
    )
    by_id = await db.express.read("CreditModelArtifact", artifact["id"])
    return {
        "evaluation": evaluation,
        "artifact": by_id,
        "all_evaluations": await db.express.list(
            "CreditEvaluation", filter={"model_name": MODEL_NAME}
        ),
        "n_artifacts": await db.express.count(
            "CreditModelArtifact", {"model_name": MODEL_NAME}
        ),
    }


async def run_all() -> dict[str, Any]:
    metrics, version, rows, n_features = await train_with_engine()
    try:
        crud = await persist_review_cleanup(metrics, version, rows, n_features)
        stored = await read_back(version)
    finally:
        await db.close_async()
    return {"metrics": metrics, "version": version, **crud, **stored}


out = asyncio.run(run_all())
print_metric_block("Test-frame metrics of the registered model — ready to persist", out["metrics"])

# ── Checkpoint ──────────────────────────────────────────────────────────
# db.express.create returns {<fields...>, "rows_affected": 1}; the
# auto-generated id is visible when the row is read back.
assert out["created_eval"].get("rows_affected") == 1, "Task 3: one evaluation row"
assert out["created_artifact"].get("rows_affected") == 1, "Task 3: one artefact row"
assert out["reviewed"]["review_status"] == "approved", "Task 3: update must persist"
assert out["deleted"] is True and out["smoke_left"] == 0, "Task 3: delete must remove the row"
print("\n[ok] Checkpoint passed — create / update / delete verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE persisted records by reading them back
# ════════════════════════════════════════════════════════════════════════
# Any persistence code you cannot READ BACK is a black hole. The audit
# test is "I wrote it, I can find it, I can filter it, I can count it."

ev, art = out["evaluation"], out["artifact"]
print(f"=== {MODEL_NAME} v{out['version']} as stored ===")
print(
    f"  CreditEvaluation id={ev['id']}: AUC-ROC={ev['auc_roc']:.4f} "
    f"AUC-PR={ev['auc_pr']:.4f} train_rows={ev['train_rows']:,} features={ev['feature_count']}"
)
print(
    f"  CreditModelArtifact id={art['id']}: {art['registry_ref']} "
    f"review_status={art['review_status']}"
)
print(f"\n=== Evaluation history for {MODEL_NAME} ({len(out['all_evaluations'])} runs) ===")
for row in out["all_evaluations"]:
    print(
        f"  v{row['model_version']:<3} AUC-PR={row['auc_pr']:.4f}  "
        f"written {row.get('created_at', '?')}"
    )
print(f"  artefact rows for {MODEL_NAME}: {out['n_artifacts']}")

# ── Checkpoint ──────────────────────────────────────────────────────────
assert abs(ev["auc_pr"] - out["metrics"]["auc_pr"]) < 1e-9, (
    "Task 4: the stored AUC-PR must equal the engine's value"
)
assert art["review_status"] == "approved", "Task 4: read-back must see the update"
print("\n[ok] Checkpoint passed — persisted rows read back intact\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: A Bank's Model Evidence Store
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): before this file, a Singapore bank's model
# validation team rebuilt an "evidence pack" by hand for every retrain —
# copying metrics out of notebooks, chasing which hyperparameters were
# used, emailing for sign-off. With the two tables above, the evidence
# pack is a query: evaluation metrics + registry reference + review
# status for any model version, written at the moment it happened.
print("\n" + "=" * 70)
print("  APPLY: Model Evidence Store")
print("=" * 70)
print(headline_roi_text())
print(
    "\n  The evidence-pack line above is what persistence buys: the rows"
    "\n  you just wrote replace the manual reconstruction."
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Declared database tables with the @db.model decorator
  [x] Ran the full db.express CRUD cycle: create, find_one/read/list/count,
      update, delete
  [x] Stored a TrainingPipeline evaluation next to its registry reference
  [x] Recorded a human review decision as an UPDATE, not an email

  KEY INSIGHT: If your metrics only exist in a print() call, they don't
  exist from an auditor's perspective. DataFlow turns "write it to a DB"
  from a week of schema work into a decorated class and a few awaits.

  Next: 03_hyperparameter_search.py — use Bayesian search to improve the
  metrics you just learned how to persist.

  DB URL: {DB_URL}
"""
)
