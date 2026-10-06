# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 7.5: Orchestrated Pipeline (Workflow + DataFlow +
#                        Hyperparameter Search + Model Registry)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Chain every previous technique (workflow, persistence, search,
#     registry) into ONE executed workflow
#   - Branch the workflow on a quality gate with a SwitchNode
#   - Write an audit trail per run_id into a DataFlow table
#   - Verify reproducibility: same params + same seed -> same test metrics
#
# PREREQUISITES: 01-04 of this exercise
# ESTIMATED TIME: ~45 min
#
# 5-PHASE R10:
#   1. Theory     — why reproducibility is the ML-ops contract
#   2. Build      — nodes for prepare / search / train / evaluate / gate
#   3. Train      — one runtime.execute(); audit rows written per stage
#   4. Visualise  — the audit trail and the reproducibility check
#   5. Apply      — the production model, diagnosed in one call
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from typing import Any

import polars as pl

from dataflow import DataFlow
from kailash.nodes.base import Node, NodeParameter, register_node
from kailash.nodes.base_async import AsyncNode
from kailash.runtime import LocalRuntime
from kailash.workflow.builder import WorkflowBuilder
from kailash_ml import HyperparameterSearch, TrainingPipeline, diagnose
from kailash_ml.engines.hyperparameter_search import (
    ParamDistribution,
    SearchConfig,
    SearchSpace,
)
from kailash_ml.engines.training_pipeline import EvalSpec, ModelSpec
from kailash_ml.interop import to_sklearn_input

from shared.mlfp03.ex_7 import (
    DB_URL,
    ENGINE_METRICS,
    OUTPUT_DIR,
    PROMOTION_MIN_AP_MULTIPLE,
    PROMOTION_MIN_AUC,
    RANDOM_SEED,
    TARGET_COLUMN,
    audit_trail_row,
    build_training_registry,
    compute_classification_metrics,
    credit_feature_schema,
    headline_roi_text,
    load_registered_model,
    prepare_credit_frames,
    print_metric_block,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Reproducibility as the ML-ops contract
# ════════════════════════════════════════════════════════════════════════
# You have four pieces:
#   1. A workflow DAG           (file 01)
#   2. A persistence layer      (file 02)
#   3. A hyperparameter search  (file 03)
#   4. A model registry         (file 04)
#
# An orchestrated pipeline ties them together so the answer to "can we
# rebuild exactly the model we shipped on March 12?" is YES:
#   - Same data + same hyperparameters + same seed = same model
#   - Every stage leaves an audit row tagged with the same run_id
#   - A branching quality gate decides promote vs hold
#
# What is and isn't reproducible here: the FINAL model is (LightGBM with a
# fixed seed on fixed data). The SEARCH is not guaranteed to be — Optuna's
# sampler is unseeded in this engine — which is exactly why the audit
# trail records the chosen hyperparameters: they are what you need to
# rebuild the model, not the search.
#
# Model-risk teams care about this because a decision you cannot
# reproduce is a decision you cannot explain.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the nodes, the DAG and the audit table
# ════════════════════════════════════════════════════════════════════════

MODEL_NAME = "credit_default_orchestrated"
DEV_PATH = OUTPUT_DIR / "ex7_05_dev.parquet"
TEST_PATH = OUTPUT_DIR / "ex7_05_test.parquet"
FIXED_PARAMS: dict[str, Any] = {"random_state": RANDOM_SEED, "verbose": -1, "n_jobs": 8}
EVAL_SPEC = EvalSpec(metrics=ENGINE_METRICS, split_strategy="holdout", test_size=0.2)

db = DataFlow(DB_URL)


@db.model
class PipelineAuditEntry:
    """One row per stage of an orchestrated pipeline run."""

    id: int
    run_id: str
    stage: str
    detail: str


def lgbm_spec(params: dict[str, Any]) -> ModelSpec:
    return ModelSpec(
        model_class="lightgbm.LGBMClassifier",
        framework="lightgbm",
        hyperparameters={**FIXED_PARAMS, **params},
    )


@register_node()
class PrepareCreditNode(Node):
    """Load + preprocess → dev/test parquet artefacts."""

    def get_parameters(self) -> dict[str, NodeParameter]:
        return {}

    def run(self, **kwargs: Any) -> dict[str, Any]:
        dev, test, feature_columns = prepare_credit_frames()
        dev.write_parquet(DEV_PATH)
        test.write_parquet(TEST_PATH)
        return {
            "dev_path": str(DEV_PATH),
            "test_path": str(TEST_PATH),
            "feature_columns": feature_columns,
            "default_rate": float(dev[TARGET_COLUMN].mean()),
            "rows": dev.height + test.height,
        }


@register_node()
class BayesianSearchNode(AsyncNode):
    """HyperparameterSearch over TrainingPipeline on the dev frame."""

    def get_parameters(self) -> dict[str, NodeParameter]:
        return {
            "dev_path": NodeParameter(name="dev_path", type=str, required=True),
            "feature_columns": NodeParameter(
                name="feature_columns", type=list, required=True
            ),
            "n_trials": NodeParameter(name="n_trials", type=int, required=True),
        }

    async def async_run(self, **kwargs: Any) -> dict[str, Any]:
        dev = pl.read_parquet(kwargs["dev_path"])
        # Bayesian search is a teaching demo: cap its rows so the run finishes
        # in minutes on a free Colab / fleet slot, not hours. Same lesson, same mechanics.
        if dev.height > 15000:
            dev = dev.sample(15000, seed=RANDOM_SEED)
        registry, conn = await build_training_registry()
        try:
            searcher = HyperparameterSearch(
                pipeline=TrainingPipeline(feature_store=None, registry=registry)
            )
            result = await searcher.search(
                data=dev,
                schema=credit_feature_schema(kwargs["feature_columns"]),
                base_model_spec=lgbm_spec({}),
                search_space=SearchSpace(
                    params=[
                        ParamDistribution("n_estimators", "int_uniform", low=100, high=500),
                        ParamDistribution("learning_rate", "log_uniform", low=0.01, high=0.3),
                        ParamDistribution("max_depth", "int_uniform", low=3, high=10),
                        ParamDistribution("num_leaves", "int_uniform", low=15, high=127),
                        ParamDistribution("min_child_samples", "int_uniform", low=5, high=100),
                    ]
                ),
                config=SearchConfig(
                    strategy="bayesian",
                    n_trials=kwargs["n_trials"],
                    metric_to_optimize="auc",
                    direction="maximize",
                    register_best=False,
                ),
                eval_spec=EVAL_SPEC,
                experiment_name=f"{MODEL_NAME}_search",
            )
        finally:
            await conn.close()
        return {
            "best_params": dict(result.best_params),
            "best_val_auc": float(result.best_metrics["auc"]),
            "n_trials": len(result.all_trials),
        }


@register_node()
class TrainFinalNode(AsyncNode):
    """Train + register one model with the given hyperparameters."""

    def get_parameters(self) -> dict[str, NodeParameter]:
        return {
            "dev_path": NodeParameter(name="dev_path", type=str, required=True),
            "feature_columns": NodeParameter(
                name="feature_columns", type=list, required=True
            ),
            "best_params": NodeParameter(name="best_params", type=dict, required=True),
            "experiment_name": NodeParameter(
                name="experiment_name", type=str, required=True
            ),
        }

    async def async_run(self, **kwargs: Any) -> dict[str, Any]:
        dev = pl.read_parquet(kwargs["dev_path"])
        # Bayesian search is a teaching demo: cap its rows so the run finishes
        # in minutes on a free Colab / fleet slot, not hours. Same lesson, same mechanics.
        if dev.height > 15000:
            dev = dev.sample(15000, seed=RANDOM_SEED)
        registry, conn = await build_training_registry()
        try:
            result = await TrainingPipeline(feature_store=None, registry=registry).train(
                data=dev,
                schema=credit_feature_schema(kwargs["feature_columns"]),
                model_spec=lgbm_spec(kwargs["best_params"]),
                eval_spec=EVAL_SPEC,
                experiment_name=kwargs["experiment_name"],
            )
        finally:
            await conn.close()
        if result.model_version is None:
            raise RuntimeError("TrainingPipeline did not register the model")
        return {
            "model_name": kwargs["experiment_name"],
            "model_version": int(result.model_version.version),
        }


@register_node()
class EvaluateTestNode(AsyncNode):
    """Score a registered model on the untouched test frame."""

    def get_parameters(self) -> dict[str, NodeParameter]:
        return {
            "test_path": NodeParameter(name="test_path", type=str, required=True),
            "feature_columns": NodeParameter(
                name="feature_columns", type=list, required=True
            ),
            "model_name": NodeParameter(name="model_name", type=str, required=True),
            "model_version": NodeParameter(name="model_version", type=int, required=True),
        }

    async def async_run(self, **kwargs: Any) -> dict[str, Any]:
        X_test, y_test, _ = to_sklearn_input(
            pl.read_parquet(kwargs["test_path"]),
            feature_columns=kwargs["feature_columns"],
            target_column=TARGET_COLUMN,
        )
        registry, conn = await build_training_registry()
        try:
            model = await load_registered_model(
                registry, kwargs["model_name"], kwargs["model_version"]
            )
        finally:
            await conn.close()
        y_proba = model.predict_proba(X_test)[:, 1]
        return {
            "test_metrics": compute_classification_metrics(
                y_test, (y_proba >= 0.5).astype(int), y_proba
            )
        }


@register_node()
class PromoteNode(AsyncNode):
    """staging → production, with the gate evidence as the reason."""

    def get_parameters(self) -> dict[str, NodeParameter]:
        return {"payload": NodeParameter(name="payload", type=dict, required=True)}

    async def async_run(self, **kwargs: Any) -> dict[str, Any]:
        p = kwargs["payload"]
        registry, conn = await build_training_registry()
        try:
            promoted = await registry.promote_model(
                p["model_name"],
                p["model_version"],
                "production",
                reason=(
                    f"Orchestrated gate passed: AUC-ROC {p['auc_roc']:.4f} >= "
                    f"{p['min_auc_roc']}, AUC-PR {p['auc_pr']:.4f} >= "
                    f"{p['min_auc_pr']:.4f}"
                ),
            )
        finally:
            await conn.close()
        return {"stage": promoted.stage, "model_version": p["model_version"]}


GATE_CODE = f"""
min_ap = {PROMOTION_MIN_AP_MULTIPLE} * default_rate
auc = test_metrics['auc_roc']
ap = test_metrics['auc_pr']
result = {{'auc_roc': auc, 'auc_pr': ap,
          'min_auc_roc': {PROMOTION_MIN_AUC}, 'min_auc_pr': min_ap,
          'promote': auc >= {PROMOTION_MIN_AUC} and ap >= min_ap,
          'model_name': model_name, 'model_version': model_version}}
"""

wf = WorkflowBuilder()
wf.add_node("PrepareCreditNode", "prepare", {})
wf.add_node("BayesianSearchNode", "search", {"n_trials": 12})
wf.add_node("TrainFinalNode", "train", {"experiment_name": MODEL_NAME})
wf.add_node("TrainFinalNode", "retrain", {"experiment_name": f"{MODEL_NAME}_repro"})
wf.add_node("EvaluateTestNode", "evaluate", {})
wf.add_node("EvaluateTestNode", "evaluate_repro", {})
wf.add_node("PythonCodeNode", "gate_check", {"code": GATE_CODE})
wf.add_node("SwitchNode", "gate", {"condition_field": "promote", "operator": "==", "value": True})
wf.add_node("PromoteNode", "promote", {})
wf.add_node("PythonCodeNode", "hold", {"code": "result = dict(payload, stage='staging')"})

for target in ("search", "train", "retrain"):
    wf.add_connection("prepare", "dev_path", target, "dev_path")
    wf.add_connection("prepare", "feature_columns", target, "feature_columns")
for target in ("train", "retrain"):
    wf.add_connection("search", "best_params", target, "best_params")
for train_id, eval_id in (("train", "evaluate"), ("retrain", "evaluate_repro")):
    wf.add_connection("prepare", "test_path", eval_id, "test_path")
    wf.add_connection("prepare", "feature_columns", eval_id, "feature_columns")
    wf.add_connection(train_id, "model_name", eval_id, "model_name")
    wf.add_connection(train_id, "model_version", eval_id, "model_version")
wf.add_connection("evaluate", "test_metrics", "gate_check", "test_metrics")
wf.add_connection("prepare", "default_rate", "gate_check", "default_rate")
wf.add_connection("train", "model_name", "gate_check", "model_name")
wf.add_connection("train", "model_version", "gate_check", "model_version")
wf.add_connection("gate_check", "result", "gate", "input_data")
wf.add_connection("gate", "true_output", "promote", "payload")
wf.add_connection("gate", "false_output", "hold", "payload")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: one runtime.execute(), then persist the audit trail
# ════════════════════════════════════════════════════════════════════════
# "retrain" re-fits the winning hyperparameters as a separate registered
# model; comparing its test metrics with "train" is the reproducibility
# check. skip_branches: only the SwitchNode branch that received data
# runs (kailash 2.44.1 also logs one harmless internal-metrics error
# line in this mode).

print("\n" + "=" * 70)
print("  Orchestrated pipeline — one workflow, every stage audited")
print("=" * 70)
with LocalRuntime(conditional_execution="skip_branches") as runtime:
    results, run_id = runtime.execute(wf.build())

prep, search = results["prepare"], results["search"]
metrics = results["evaluate"]["test_metrics"]
repro_metrics = results["evaluate_repro"]["test_metrics"]
gate_in = results["gate_check"]["result"]
branch = "promote" if results.get("promote") else "hold"
drift = abs(metrics["auc_pr"] - repro_metrics["auc_pr"])
final_stage = (
    results["promote"]["stage"] if branch == "promote" else results["hold"]["result"]["stage"]
)

audit_rows = [
    audit_trail_row(stage="prepare", run_id=run_id,
                    detail=f"rows={prep['rows']} features={len(prep['feature_columns'])}"),
    audit_trail_row(stage="hyperparameter_search", run_id=run_id,
                    detail=f"bayesian n_trials={search['n_trials']} "
                           f"best_val_auc={search['best_val_auc']:.4f} "
                           f"params={search['best_params']}"),
    audit_trail_row(stage="train", run_id=run_id,
                    detail=f"{MODEL_NAME} v{results['train']['model_version']}"),
    audit_trail_row(stage="evaluate", run_id=run_id,
                    detail=f"test auc_roc={metrics['auc_roc']:.4f} auc_pr={metrics['auc_pr']:.4f}"),
    audit_trail_row(stage="quality_gate", run_id=run_id,
                    detail=f"promote={gate_in['promote']} -> branch {branch}"),
    audit_trail_row(stage=branch, run_id=run_id,
                    detail=f"{MODEL_NAME} v{results['train']['model_version']} -> {final_stage}"),
    audit_trail_row(stage="reproducibility", run_id=run_id,
                    detail=f"retrain auc_pr diff={drift:.2e}"),
]


async def persist_audit() -> list[dict[str, Any]]:
    await db.initialize()
    try:
        for row in audit_rows:
            await db.express.create("PipelineAuditEntry", row)
        return await db.express.list("PipelineAuditEntry", filter={"run_id": run_id})
    finally:
        await db.close_async()


stored_audit = asyncio.run(persist_audit())

# ── Checkpoint ──────────────────────────────────────────────────────────
assert run_id is not None, "Task 3: the workflow must return a run_id"
assert drift < 1e-9, f"Task 3: same params + seed must reproduce (diff={drift})"
assert (branch == "promote") == gate_in["promote"], "Task 3: branch must follow the gate"
assert len(stored_audit) == len(audit_rows), "Task 3: every stage must be persisted"
print(f"\n  run_id={run_id}")
print("\n[ok] Checkpoint passed — orchestrated run executed, audited, reproduced\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the audit trail + reproducibility certificate
# ════════════════════════════════════════════════════════════════════════

print(f"=== Audit trail (read back from PipelineAuditEntry, run {run_id[:8]}) ===")
for row in sorted(stored_audit, key=lambda r: r["id"]):
    print(f"  [{row['stage']:<22}] {row['detail']}")

print_metric_block("Final model — test frame", metrics)
print_metric_block("Re-trained twin — test frame", repro_metrics)
print(f"\n  AUC-PR difference between the two fits: {drift:.2e}")
print(
    "\nExecuted DAG:"
    "\n  prepare -> search -> train   -> evaluate -> gate_check -> gate -> promote|hold"
    "\n                   \\-> retrain -> evaluate_repro  (reproducibility twin)"
    f"\n  branch taken: {branch}; {MODEL_NAME} is in {final_stage}"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: diagnose the model this run produced, in one call
# ════════════════════════════════════════════════════════════════════════
# The pipeline above wired WorkflowBuilder, DataFlow, HyperparameterSearch,
# TrainingPipeline and ModelRegistry from primitives. kailash-ml packages
# the diagnostic surface a reviewer asks for next — per-class metrics,
# class-balance severity — into km.diagnose. We load the exact registered
# artefact the gate judged and diagnose it on the same test frame.


async def load_final_model() -> Any:
    registry, conn = await build_training_registry()
    try:
        return await load_registered_model(
            registry, MODEL_NAME, results["train"]["model_version"]
        )
    finally:
        await conn.close()


final_model = asyncio.run(load_final_model())
X_test, y_test, _ = to_sklearn_input(
    pl.read_parquet(prep["test_path"]),
    feature_columns=prep["feature_columns"],
    target_column=TARGET_COLUMN,
)
report = diagnose(
    final_model, kind="classical_classifier", data=(X_test, y_test), show=False
)
print("\n" + "=" * 70)
print("  APPLY: The Run's Model, Diagnosed")
print("=" * 70)
print(f"  km.diagnose metrics  : {report.metrics}")
print(f"  km.diagnose severity : {report.severity}")
print(headline_roi_text())


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Ran search, training, evaluation, gating and promotion as ONE
      executed Kailash workflow (run_id {run_id[:8]})
  [x] Wrote every stage into a DataFlow audit table and read it back
  [x] Branched with a SwitchNode on a test-set quality gate
      (AUC-ROC >= {PROMOTION_MIN_AUC}, AUC-PR >= {PROMOTION_MIN_AP_MULTIPLE:g} x default rate)
  [x] Proved the final model reproduces: AUC-PR difference {drift:.1e}

  KEY INSIGHT: Orchestration is not about running scripts in order — it
  is about producing an audit trail someone else can replay. Every stage
  is named, every run is tagged, every transition is a row in a table
  that outlives the analyst who wrote the pipeline.

  Next: Exercise 8 adds conformal prediction, DriftMonitor and a
  production monitoring view on top of everything you just built.
"""
)
