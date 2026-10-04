# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 7.1: Kailash WorkflowBuilder + Custom Nodes
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Write custom nodes with `@register_node()` (Node and AsyncNode)
#   - Add a PythonCodeNode for a small inline step and a SwitchNode for
#     a conditional branch
#   - Wire node outputs to node inputs with `add_connection(...)`
#   - Execute the DAG with `runtime.execute(workflow.build())` and read
#     every node's real output plus the run_id
#   - Promote a model to production only when the quality gate passes
#
# PREREQUISITES: MLFP03 Exercises 1-6, MLFP02 preprocessing
# ESTIMATED TIME: ~45 min
#
# 5-PHASE R10:
#   1. Theory     — why workflows beat hand-rolled scripts
#   2. Build      — custom nodes + load→preprocess→train→evaluate→gate
#   3. Train      — LocalRuntime.execute(workflow.build())
#   4. Visualise  — trace every node's output and the branch taken
#   5. Apply      — monthly credit-model retraining at a Singapore bank
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from typing import Any

import polars as pl

from kailash.nodes.base import Node, NodeParameter, register_node
from kailash.nodes.base_async import AsyncNode
from kailash.runtime import LocalRuntime
from kailash.workflow.builder import WorkflowBuilder
from kailash_ml import TrainingPipeline
from kailash_ml.engines.training_pipeline import EvalSpec, ModelSpec
from kailash_ml.interop import to_sklearn_input

from shared.mlfp03.ex_7 import (
    ENGINE_METRICS,
    OUTPUT_DIR,
    PROMOTION_MIN_AP_MULTIPLE,
    PROMOTION_MIN_AUC,
    RANDOM_SEED,
    TARGET_COLUMN,
    build_training_registry,
    compute_classification_metrics,
    credit_feature_schema,
    headline_roi_text,
    load_credit_frame,
    load_registered_model,
    prepare_credit_frames,
    print_metric_block,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why WorkflowBuilder?
# ════════════════════════════════════════════════════════════════════════
# A hand-rolled training script is "just Python" until the day it isn't.
# When you come back six months later and ask "which model went into
# production on March 12?", you need four things the script cannot give
# you without re-implementation:
#
#   1. A NAME for each step ("preprocess", "train", "gate") so you can
#      reference it in an audit row.
#   2. A RUN_ID for every execution so the June run is distinguishable
#      from the March run.
#   3. DEPENDENCIES between steps expressed as edges, not as "the order
#      functions happen to appear in the file."
#   4. A SINGLE EXECUTION ENTRYPOINT: `runtime.execute(workflow.build())`.
#
# Three kinds of node build this exercise's DAG:
#   - CUSTOM nodes: a class decorated with `@register_node()` that
#     subclasses `Node` (sync `run`) or `AsyncNode` (`async_run`, for
#     engines such as TrainingPipeline that talk to a database).
#   - PythonCodeNode: a few lines of inline code; its output is the dict
#     you assign to `result`, so downstream nodes read `result`.
#   - SwitchNode: routes its `input_data` to `true_output` or
#     `false_output`; only the branch that receives data runs.
#
# One rule shapes the design: node outputs must be JSON-serialisable.
# Heavy objects therefore travel as REFERENCES — a parquet path for the
# data, a registry name + version for the model — exactly like a
# production pipeline that passes artefact pointers, not objects.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the custom nodes and the DAG
# ════════════════════════════════════════════════════════════════════════

EXPERIMENT_NAME = "credit_default_workflow"
RAW_PATH = OUTPUT_DIR / "ex7_01_raw.parquet"
DEV_PATH = OUTPUT_DIR / "ex7_01_dev.parquet"
TEST_PATH = OUTPUT_DIR / "ex7_01_test.parquet"


@register_node()
class CreditLoadNode(Node):
    """Data-loading node: leak-free credit table → parquet artefact."""

    def get_parameters(self) -> dict[str, NodeParameter]:
        return {}

    def run(self, **kwargs: Any) -> dict[str, Any]:
        raw = load_credit_frame()
        raw.write_parquet(RAW_PATH)
        return {
            "raw_path": str(RAW_PATH),
            "rows": raw.height,
            "default_rate": float(raw["default"].mean()),
        }


@register_node()
class CreditPreprocessNode(Node):
    """Preprocessing node: PreprocessingPipeline → dev + test parquet files.

    The test rows are held out BEFORE the pipeline is fitted, so the
    imputation and encoding are learned from the dev rows only. The dev
    frame is for training (TrainingPipeline takes its own holdout from
    it); the test frame is touched only by the evaluation node.
    """

    def get_parameters(self) -> dict[str, NodeParameter]:
        return {
            "raw_path": NodeParameter(name="raw_path", type=str, required=True),
            "seed": NodeParameter(
                name="seed", type=int, required=False, default=RANDOM_SEED
            ),
        }

    def run(self, **kwargs: Any) -> dict[str, Any]:
        raw = pl.read_parquet(kwargs["raw_path"])
        # TODO: split the raw frame into (dev, test, feature_columns) with the
        # shared helper that runs PreprocessingPipeline, passing the seed.
        # Hint: prepare_credit_frames(<frame>, seed=...)
        dev, test, feature_columns = ____
        dev.write_parquet(DEV_PATH)
        test.write_parquet(TEST_PATH)
        return {
            "dev_path": str(DEV_PATH),
            "test_path": str(TEST_PATH),
            "feature_columns": feature_columns,
        }


@register_node()
class TrainingPipelineNode(AsyncNode):
    """Training node: one kailash-ml TrainingPipeline.train() call.

    TrainingPipeline holds out 20% of the dev frame, fits LightGBM on the
    rest, scores that holdout and registers the model (stage: staging).
    The node returns the metrics and the registry reference — not the
    model object.
    """

    def get_parameters(self) -> dict[str, NodeParameter]:
        return {
            "dev_path": NodeParameter(name="dev_path", type=str, required=True),
            "feature_columns": NodeParameter(
                name="feature_columns", type=list, required=True
            ),
            "hyperparameters": NodeParameter(
                name="hyperparameters", type=dict, required=True
            ),
            "experiment_name": NodeParameter(
                name="experiment_name", type=str, required=True
            ),
        }

    async def async_run(self, **kwargs: Any) -> dict[str, Any]:
        dev = pl.read_parquet(kwargs["dev_path"])
        registry, conn = await build_training_registry()
        try:
            # TODO: build the TrainingPipeline engine (no feature store here).
            # Hint: TrainingPipeline(feature_store=..., registry=...)
            pipeline = ____
            result = await pipeline.train(
                data=dev,
                schema=credit_feature_schema(kwargs["feature_columns"]),
                model_spec=ModelSpec(
                    model_class="lightgbm.LGBMClassifier",
                    framework="lightgbm",
                    hyperparameters=kwargs["hyperparameters"],
                ),
                eval_spec=EvalSpec(
                    metrics=ENGINE_METRICS, split_strategy="holdout", test_size=0.2
                ),
                experiment_name=kwargs["experiment_name"],
            )
        finally:
            await conn.close()
        if result.model_version is None:
            raise RuntimeError("TrainingPipeline did not register the model")
        return {
            "holdout_metrics": {k: float(v) for k, v in result.metrics.items()},
            "model_name": kwargs["experiment_name"],
            "model_version": int(result.model_version.version),
        }


@register_node()
class EvaluateNode(AsyncNode):
    """Evaluation node: score the registered model on the untouched test frame."""

    def get_parameters(self) -> dict[str, NodeParameter]:
        return {
            "test_path": NodeParameter(name="test_path", type=str, required=True),
            "feature_columns": NodeParameter(
                name="feature_columns", type=list, required=True
            ),
            "model_name": NodeParameter(name="model_name", type=str, required=True),
            "model_version": NodeParameter(
                name="model_version", type=int, required=True
            ),
        }

    async def async_run(self, **kwargs: Any) -> dict[str, Any]:
        test = pl.read_parquet(kwargs["test_path"])
        X_test, y_test, _ = to_sklearn_input(
            test,
            feature_columns=kwargs["feature_columns"],
            target_column=TARGET_COLUMN,
        )
        registry, conn = await build_training_registry()
        try:
            # TODO: load the model the train node registered (name + version).
            # Hint: the shared load_registered_model(registry, name, version) is async
            model = await ____
        finally:
            await conn.close()
        # TODO: probability of default (class 1) for every test row
        # Hint: predict_proba returns one column per class
        y_proba = ____
        y_pred = (y_proba >= 0.5).astype(int)
        return {"test_metrics": compute_classification_metrics(y_test, y_pred, y_proba)}


@register_node()
class PromoteNode(AsyncNode):
    """Promotion node: moves the registered version staging → production."""

    def get_parameters(self) -> dict[str, NodeParameter]:
        return {"payload": NodeParameter(name="payload", type=dict, required=True)}

    async def async_run(self, **kwargs: Any) -> dict[str, Any]:
        p = kwargs["payload"]
        registry, conn = await build_training_registry()
        try:
            promoted = await registry.promote_model(
                p["model_name"],
                p["model_version"],
                # TODO: the target stage for a model that passed the gate
                # Hint: registry stages are staging / shadow / production / archived
                ____,
                reason=(
                    f"Quality gate passed on the test frame: "
                    f"AUC-ROC {p['auc_roc']:.4f} >= {p['min_auc_roc']}, "
                    f"AUC-PR {p['auc_pr']:.4f} >= {p['min_auc_pr']:.4f}"
                ),
            )
        finally:
            await conn.close()
        return {"stage": promoted.stage, "model_version": p["model_version"]}


# The quality gate is a PythonCodeNode: a few inline lines whose `result`
# dict becomes the SwitchNode's input. `test_metrics`, `default_rate`,
# `model_name` and `model_version` arrive through connections.
GATE_CODE = f"""
min_ap = {PROMOTION_MIN_AP_MULTIPLE} * default_rate
auc = test_metrics['auc_roc']
ap = test_metrics['auc_pr']
result = {{'auc_roc': auc, 'auc_pr': ap,
          'min_auc_roc': {PROMOTION_MIN_AUC}, 'min_auc_pr': min_ap,
          'promote': auc >= {PROMOTION_MIN_AUC} and ap >= min_ap,
          'model_name': model_name, 'model_version': model_version}}
"""

HYPERPARAMETERS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "num_leaves": 31,
    "min_child_samples": 40,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}

workflow = WorkflowBuilder()
workflow.add_node("CreditLoadNode", "load", {})
workflow.add_node("CreditPreprocessNode", "preprocess", {"seed": RANDOM_SEED})
workflow.add_node(
    "TrainingPipelineNode",
    "train",
    {"hyperparameters": HYPERPARAMETERS, "experiment_name": EXPERIMENT_NAME},
)
workflow.add_node("EvaluateNode", "evaluate", {})
workflow.add_node("PythonCodeNode", "gate_check", {"code": GATE_CODE})
# TODO: configure the SwitchNode to route on the gate_check verdict.
# Hint: {"condition_field": <key in the input dict>, "operator": "==", "value": ...}
workflow.add_node("SwitchNode", "gate", ____)
workflow.add_node("PromoteNode", "promote", {})
workflow.add_node(
    "PythonCodeNode",
    "hold",
    {"code": "result = dict(payload, stage='staging', held=True)"},
)

# Edges: add_connection(from_node, from_output, to_node, to_input)
workflow.add_connection("load", "raw_path", "preprocess", "raw_path")
workflow.add_connection("preprocess", "dev_path", "train", "dev_path")
workflow.add_connection("preprocess", "feature_columns", "train", "feature_columns")
workflow.add_connection("preprocess", "test_path", "evaluate", "test_path")
workflow.add_connection("preprocess", "feature_columns", "evaluate", "feature_columns")
workflow.add_connection("train", "model_name", "evaluate", "model_name")
workflow.add_connection("train", "model_version", "evaluate", "model_version")
workflow.add_connection("evaluate", "test_metrics", "gate_check", "test_metrics")
workflow.add_connection("load", "default_rate", "gate_check", "default_rate")
workflow.add_connection("train", "model_name", "gate_check", "model_name")
workflow.add_connection("train", "model_version", "gate_check", "model_version")
# TODO: feed the PythonCodeNode's output dict into the SwitchNode.
# Hint: PythonCodeNode outputs live under "result"; SwitchNode reads "input_data"
workflow.add_connection(____)
# TODO: which SwitchNode output should reach the promote node?
# Hint: SwitchNode emits "true_output" and "false_output"
workflow.add_connection("gate", ____, "promote", "payload")
workflow.add_connection("gate", "false_output", "hold", "payload")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN via runtime.execute(workflow.build())
# ════════════════════════════════════════════════════════════════════════
# LocalRuntime validates the DAG, runs each node in dependency order and
# returns (results, run_id). `results[node_id]` is that node's output
# dict; a node on the branch the SwitchNode did NOT take has no entry.
# If any node fails, execute() raises — there is no silent fallback.

print("\n" + "=" * 70)
print("  Executing the credit-scoring workflow")
print("=" * 70)

# conditional_execution="skip_branches" makes the runtime SKIP every node
# on the SwitchNode branch that received no data (the default
# "route_data" mode would still try to feed it None). In kailash 2.44.1
# this mode also logs one "Error tracking conditional execution
# performance" line from its internal metrics hook; execution itself is
# unaffected. The context manager releases the runtime's resources.
with LocalRuntime(conditional_execution="skip_branches") as runtime:
    # TODO: execute the BUILT workflow
    # Hint: runtime.execute(...) takes workflow.build(), not the builder
    results, run_id = runtime.execute(____)
print(f"  run_id: {run_id}")

train_out = results["train"]
test_metrics = results["evaluate"]["test_metrics"]
gate_in = results["gate_check"]["result"]
branch_taken = "promote" if results.get("promote") is not None else "hold"

# ── Checkpoint ──────────────────────────────────────────────────────────
assert run_id is not None, "Task 3: runtime.execute must return a run_id"
assert results["load"]["rows"] > 0, "Task 3: the load node must read rows"
assert train_out["model_version"] >= 1, "Task 3: the model must be registered"
assert 0.5 < test_metrics["auc_roc"] <= 1.0, "Task 3: test AUC must beat random"
assert (results.get("promote") is None) != (
    results.get("hold") is None
), "Task 3: exactly one SwitchNode branch must run"
assert (branch_taken == "promote") == gate_in["promote"], (
    "Task 3: the branch taken must match the gate verdict"
)
print("\n[ok] Checkpoint passed — custom nodes, PythonCodeNode and SwitchNode ran\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the executed DAG
# ════════════════════════════════════════════════════════════════════════

print("Executed DAG (node -> output):")
print(
    f"  load        -> {results['load']['rows']:,} rows, "
    f"default rate {results['load']['default_rate']:.2%}"
)
print(
    f"  preprocess  -> {len(results['preprocess']['feature_columns'])} features; "
    f"dev + test frames written to {OUTPUT_DIR}"
)
print(
    f"  train       -> {train_out['model_name']} v{train_out['model_version']} "
    "(registered: staging)"
)
print(
    f"  evaluate    -> test AUC-ROC {test_metrics['auc_roc']:.4f}, "
    f"AUC-PR {test_metrics['auc_pr']:.4f}"
)
print(
    f"  gate_check  -> needs AUC-ROC >= {gate_in['min_auc_roc']} and "
    f"AUC-PR >= {gate_in['min_auc_pr']:.4f}: promote={gate_in['promote']}"
)
print(f"  gate        -> condition_result={results['gate']['condition_result']}")
if branch_taken == "promote":
    print(f"  promote     -> stage={results['promote']['stage']}")
    print("  hold        -> skipped (branch not taken)")
else:
    print("  promote     -> skipped (branch not taken)")
    print(f"  hold        -> stays in {results['hold']['result']['stage']}")

print_metric_block(
    "TrainingPipeline holdout metrics (train node, from the dev frame)",
    train_out["holdout_metrics"],
)
print_metric_block("Untouched test-frame metrics (evaluate node)", test_metrics)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Monthly Credit-Model Retraining at a Singapore Bank
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore retail bank's credit-risk team
# retrains its default model on the first Monday of every month. Today
# the retraining is a notebook owned by one analyst; the bank's model-risk
# policy asks for every production model to be reproducible and for each
# promotion to be justified — and a notebook cannot show either.
#
# Converting the notebook to this workflow gives:
#   - A single named entrypoint the retraining scheduler can invoke
#   - A run_id per execution, which maps 1:1 to an audit row
#   - A promotion that happens ONLY when the gate node says so, with the
#     gate's numbers written into the registry's promotion reason
#   - No hidden state — every node's input is a config value or an edge
print("\n" + "=" * 70)
print("  APPLY: Monthly Retraining as a Workflow")
print("=" * 70)
print(headline_roi_text())


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Wrote custom nodes with @register_node() — Node and AsyncNode
  [x] Used a PythonCodeNode for the gate and a SwitchNode to branch
  [x] Wired outputs to inputs with add_connection(...)
  [x] Executed with runtime.execute(workflow.build()) — run_id {run_id[:8]}
  [x] Branch taken this run: {branch_taken}

  KEY INSIGHT: The workflow is documentation you can execute. Every edge
  is machine-readable, every node's output is inspectable, and the
  promotion decision is a node — not a line someone forgot to run.

  Next: 02_dataflow_persistence.py — stop returning metrics as a dict
  and start writing them to a DataFlow-managed database table.
"""
)
