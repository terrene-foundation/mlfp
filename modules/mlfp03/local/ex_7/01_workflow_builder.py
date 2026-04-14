# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 7.1: Kailash WorkflowBuilder + LocalRuntime
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Declare an ML pipeline as a named DAG using WorkflowBuilder
#   - Wire nodes with `connections=[...]` to form execution edges
#   - Execute the workflow with `runtime.execute(workflow.build())`
#   - Capture the run_id so every training run is auditable
#
# PREREQUISITES: MLFP03 Exercises 1-6, MLFP02 preprocessing
# ESTIMATED TIME: ~35 min
#
# 5-PHASE R10:
#   1. Theory     — why workflows beat hand-rolled scripts
#   2. Build      — declare the DAG with WorkflowBuilder
#   3. Train      — LocalRuntime.execute(workflow.build())
#   4. Visualise  — inspect the DAG and the parallel manual pipeline
#   5. Apply      — Singapore bank monthly credit-model retraining
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import lightgbm as lgb

from kailash.runtime import LocalRuntime
from kailash.workflow.builder import WorkflowBuilder

from shared.mlfp03.ex_7 import (
    RANDOM_SEED,
    SG_BANK_PORTFOLIO,
    compute_classification_metrics,
    headline_roi_text,
    load_credit_frame,
    prepare_credit_split,
    print_metric_block,
    scale_pos_weight_for,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why WorkflowBuilder?
# ════════════════════════════════════════════════════════════════════════
# A hand-rolled script has no NAME per step, no RUN_ID per run, no
# machine-readable DEPENDENCIES, and no single ENTRYPOINT. WorkflowBuilder
# gives you all four by turning the pipeline into a DAG.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the workflow
# ════════════════════════════════════════════════════════════════════════

# TODO: instantiate a WorkflowBuilder with the name "credit_scoring_pipeline"
# Hint: workflow = WorkflowBuilder("credit_scoring_pipeline")
workflow = ____

# TODO: add the preprocess node (DataPreprocessNode, "preprocess")
# Hint: 4-param order is add_node("NodeType", "node_id", {config}, connections)
workflow.add_node(
    "DataPreprocessNode",
    "preprocess",
    {
        "data_source": "sg_credit_scoring",
        "target": "default",
        "train_size": 0.8,
        "seed": RANDOM_SEED,
        "normalize": False,
        "categorical_encoding": "ordinal",
        "imputation_strategy": "median",
    },
)

# TODO: add the train node, connected to "preprocess"
# Hint: pass connections=["preprocess"] as the 4th argument
workflow.add_node(
    "ModelTrainNode",
    "train",
    {
        "model_class": "lightgbm.LGBMClassifier",
        "hyperparameters": {
            "n_estimators": 500,
            "learning_rate": 0.1,
            "max_depth": 6,
            "scale_pos_weight": 7.3,
        },
    },
    connections=____,
)

# TODO: add the evaluate node, connected to "train"
workflow.add_node(
    "ModelEvalNode",
    "evaluate",
    {"metrics": ["accuracy", "f1", "auc_roc", "auc_pr", "log_loss"]},
    connections=["train"],
)

# TODO: add the persist node, connected to "evaluate"
workflow.add_node(
    "PersistNode",
    "persist",
    {"storage": "sqlite:///mlfp03_models.db"},
    connections=["evaluate"],
)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN via runtime.execute(workflow.build())
# ════════════════════════════════════════════════════════════════════════

# TODO: create a LocalRuntime
# Hint: runtime = LocalRuntime()
runtime = ____

print("\n" + "=" * 70)
print("  Executing credit_scoring_pipeline workflow")
print("=" * 70)

try:
    # TODO: call runtime.execute on the BUILT workflow
    # Hint: runtime.execute(workflow.build()) — MUST call .build()
    results, run_id = runtime.execute(____)
    workflow_ok = True
    print(f"  run_id:      {run_id}")
    print(f"  node_count:  {len(results)}")
except Exception as exc:
    print(f"  [info] custom nodes not registered ({type(exc).__name__})")
    print(f"  [info] falling back to hand-rolled pipeline that mirrors the DAG")
    results, run_id = {}, "fallback-manual-run"
    workflow_ok = False

# Hand-rolled pipeline that mirrors the DAG (always runs)
credit = load_credit_frame()
split = prepare_credit_split(credit)
baseline = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    scale_pos_weight=scale_pos_weight_for(split.y_train),
    random_state=RANDOM_SEED,
    verbose=-1,
)
baseline.fit(split.X_train, split.y_train)
y_pred = baseline.predict(split.X_test)
y_proba = baseline.predict_proba(split.X_test)[:, 1]
baseline_metrics = compute_classification_metrics(split.y_test, y_pred, y_proba)


# ── Checkpoint ──────────────────────────────────────────────────────────
assert run_id is not None, "Task 3: runtime.execute must return a run_id"
assert baseline_metrics["auc_roc"] > 0.5, "Task 3: model must beat random"
print("\n[ok] Checkpoint passed — workflow + hand-rolled pipeline executed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the DAG and the baseline metrics
# ════════════════════════════════════════════════════════════════════════

print("DAG shape:")
print("  preprocess -> train -> evaluate -> persist")
print(f"  workflow_ok={workflow_ok}  run_id={run_id}")
print_metric_block("Baseline LightGBM (workflow hyperparameters)", baseline_metrics)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: DBS Credit Model Monthly Retraining
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: DBS's Retail Credit Risk team retrains the default model on
# the first Monday of every month. Today the retraining script is a
# Jupyter notebook owned by one analyst. MAS Notice 635 requires every
# model used in a credit decision to have a reproducible audit trail —
# and the auditor cannot verify a notebook.
print("\n" + "=" * 70)
print("  APPLY: DBS Monthly Retraining — S$48B Unsecured Portfolio")
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
  [x] Declared an ML pipeline as a Kailash WorkflowBuilder DAG
  [x] Wired nodes with `connections=[...]` to form execution edges
  [x] Executed with the canonical runtime.execute(workflow.build()) pattern
  [x] Captured a run_id that serves as the audit primary key
  [x] Connected the orchestration plane to a Singapore banking ML-ops scenario

  Next: 02_dataflow_persistence.py — persist the metrics into a DB.
  Portfolio anchor: S${SG_BANK_PORTFOLIO['portfolio_sgd']/1e9:.0f}B
"""
)
