# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 7.5: Orchestrated Pipeline (Workflow + DataFlow +
#                        Hyperparameter Search + Model Registry)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Chain every previous technique into one reproducible run
#   - Build a branching workflow with a conditional quality gate
#   - Write an audit trail per run_id covering every stage transition
#   - Verify reproducibility: same seed + same code -> same metrics
#
# PREREQUISITES: 01-04 of this exercise
# ESTIMATED TIME: ~45 min
#
# 5-PHASE R10:
#   1. Theory     — reproducibility is the ultimate ML-ops contract
#   2. Build      — orchestrated pipeline with a branching workflow
#   3. Train      — run end-to-end: preprocess -> search -> register -> promote
#   4. Visualise  — audit trail and reproducibility check
#   5. Apply      — full Singapore banking ML-ops ROI
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import json
import pickle
import uuid

import lightgbm as lgb

from kailash.dataflow import DataFlow, field
from kailash.db import ConnectionManager
from kailash.runtime import LocalRuntime
from kailash.workflow.builder import WorkflowBuilder
from kailash_ml.engines.hyperparameter_search import (
    HyperparameterSearch,
    ParamDistribution,
    SearchConfig,
    SearchSpace,
)
from kailash_ml.engines.model_registry import ModelRegistry
from kailash_ml.types import FeatureField, FeatureSchema, MetricSpec, ModelSignature

from shared.mlfp03.ex_7 import (
    DB_URL,
    RANDOM_SEED,
    SG_BANK_PORTFOLIO,
    audit_trail_row,
    compute_classification_metrics,
    headline_roi_text,
    prepare_credit_split,
    print_metric_block,
    scale_pos_weight_for,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Reproducibility as the ML-ops contract
# ════════════════════════════════════════════════════════════════════════
# Under MAS Notice 635, reproducibility IS the burden of proof. Same
# data + same seed + same code MUST produce the same output, every
# stage tagged with the same run_id, every transition replayable.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the orchestrated pipeline
# ════════════════════════════════════════════════════════════════════════

# TODO: generate a fresh UUID run_id for this pipeline execution
# Hint: RUN_ID = str(uuid.uuid4())
RUN_ID = ____

db = DataFlow(DB_URL)


@db.model
class PipelineAuditEntry:
    """One row per stage transition for an orchestrated pipeline run."""

    id: int = field(primary_key=True)
    run_id: str = field()
    stage: str = field()
    detail: str = field()


branching_workflow = WorkflowBuilder("credit_scoring_orchestrated")
branching_workflow.add_node(
    "DataPreprocessNode",
    "preprocess",
    {"data_source": "sg_credit_scoring", "target": "default"},
)
branching_workflow.add_node(
    "ModelTrainNode",
    "train_primary",
    {"model_class": "lightgbm.LGBMClassifier"},
    connections=["preprocess"],
)
branching_workflow.add_node(
    "ModelEvalNode",
    "evaluate",
    {"metrics": ["auc_pr", "brier_score"]},
    connections=["train_primary"],
)
# TODO: add a ConditionalNode that fires "register" when auc_pr > 0.5
# Hint: {"condition": "auc_pr > 0.5", "true_output": "register", "false_output": "retrain"}
branching_workflow.add_node(
    "ConditionalNode",
    "quality_gate",
    ____,
    connections=["evaluate"],
)
branching_workflow.add_node(
    "PersistNode",
    "register",
    {"storage": DB_URL, "stage": "staging"},
    connections=["quality_gate"],
)


runtime = LocalRuntime()


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: run the full pipeline
# ════════════════════════════════════════════════════════════════════════


async def orchestrated_run() -> dict:
    await db.initialize()

    audit: list[dict] = []

    async def log_stage(stage: str, detail: str) -> None:
        row = audit_trail_row(stage=stage, detail=detail, run_id=RUN_ID)
        audit.append(row)
        await db.express.create("PipelineAuditEntry", row)

    # Stage 1: declarative DAG execution
    try:
        _, wf_run_id = runtime.execute(branching_workflow.build())
        await log_stage("workflow.run", f"runtime.execute ok wf_run_id={wf_run_id}")
    except Exception as exc:
        await log_stage(
            "workflow.run",
            f"declarative-only (custom nodes unregistered): {type(exc).__name__}",
        )

    # Stage 2: preprocess
    split = prepare_credit_split()
    await log_stage(
        "preprocess",
        f"train={split.train_size} test={split.test_size} features={split.feature_count}",
    )
    pos_weight = scale_pos_weight_for(split.y_train)

    # Stage 3: Bayesian hyperparameter search
    search_space = SearchSpace(
        params=[
            ParamDistribution("n_estimators", "int", low=100, high=1000),
            ParamDistribution("learning_rate", "float", low=0.01, high=0.3, log=True),
            ParamDistribution("max_depth", "int", low=3, high=10),
            ParamDistribution("num_leaves", "int", low=15, high=127),
            ParamDistribution("min_child_samples", "int", low=5, high=50),
        ]
    )
    search_config = SearchConfig(
        n_trials=20,
        metric="average_precision",
        direction="maximize",
        cv_folds=5,
        random_state=RANDOM_SEED,
    )
    searcher = HyperparameterSearch(search_space, search_config)
    best_params, best_score, _ = searcher.search(
        estimator_class=lgb.LGBMClassifier,
        X=split.X_train,
        y=split.y_train,
        fixed_params={
            "random_state": RANDOM_SEED,
            "verbose": -1,
            "scale_pos_weight": pos_weight,
        },
    )
    await log_stage(
        "hyperparameter_search",
        f"bayesian n_trials=20 best_cv_auc_pr={best_score:.4f}",
    )

    # Stage 4: train final, evaluate
    final_model = lgb.LGBMClassifier(
        **best_params,
        random_state=RANDOM_SEED,
        verbose=-1,
        scale_pos_weight=pos_weight,
    )
    final_model.fit(split.X_train, split.y_train)
    y_pred = final_model.predict(split.X_test)
    y_proba = final_model.predict_proba(split.X_test)[:, 1]
    metrics = compute_classification_metrics(split.y_test, y_pred, y_proba)
    await log_stage(
        "evaluate",
        f"auc_pr={metrics['auc_pr']:.4f} auc_roc={metrics['auc_roc']:.4f}",
    )

    # Stage 5: quality gate
    # TODO: set gate_passes True iff AUC-PR > 0.5
    # Hint: gate_passes = metrics["auc_pr"] > 0.5
    gate_passes = ____
    await log_stage(
        "quality_gate", f"auc_pr>0.5 -> {'register' if gate_passes else 'retrain'}"
    )

    version_id: int | None = None
    if gate_passes:
        await db.express.create(
            "PipelineAuditEntry",
            audit_trail_row(
                stage="persist.evaluation",
                detail=json.dumps({k: round(v, 6) for k, v in metrics.items()}),
                run_id=RUN_ID,
            ),
        )

        input_schema = FeatureSchema(
            name="credit_model_input",
            features=[
                FeatureField(name=f, dtype="float64") for f in split.feature_columns
            ],
            entity_id_column="application_id",
        )
        signature = ModelSignature(
            input_schema=input_schema,
            output_columns=["default_probability", "default_label"],
            output_dtypes=["float64", "int64"],
            model_type="classifier",
        )
        conn = ConnectionManager(DB_URL)
        await conn.initialize()
        try:
            registry = ModelRegistry(conn)
            # TODO: register the model with its metrics, then promote to production
            version = await registry.register_model(
                name="credit_default_v2",
                artifact=pickle.dumps(final_model),
                metrics=[
                    MetricSpec(name="auc_pr", value=metrics["auc_pr"]),
                    MetricSpec(name="auc_roc", value=metrics["auc_roc"]),
                    MetricSpec(name="f1_score", value=metrics["f1"]),
                ],
            )
            version_id = version.version
            await log_stage("register", f"credit_default_v2 v{version_id} in staging")
            await registry.promote_model(
                name="credit_default_v2",
                version=version_id,
                target_stage="production",
                reason=(
                    f"Orchestrated run {RUN_ID[:8]}: passed AUC-PR gate, "
                    f"Bayesian-optimised hyperparameters."
                ),
            )
            await log_stage(
                "promote",
                f"credit_default_v2 v{version_id} staging->production",
            )
        finally:
            await conn.close()
        _ = signature

    # Stage 6: reproducibility check
    repro_model = lgb.LGBMClassifier(
        **best_params,
        random_state=RANDOM_SEED,
        verbose=-1,
        scale_pos_weight=pos_weight,
    )
    repro_model.fit(split.X_train, split.y_train)
    y_proba_repro = repro_model.predict_proba(split.X_test)[:, 1]
    repro_metrics = compute_classification_metrics(
        split.y_test, repro_model.predict(split.X_test), y_proba_repro
    )
    # TODO: drift = absolute difference of AUC-PR between the two runs
    # Hint: abs(repro_metrics["auc_pr"] - metrics["auc_pr"])
    drift = ____
    await log_stage("reproducibility", f"drift_auc_pr={drift:.6f} (must be <1e-3)")

    return {
        "metrics": metrics,
        "repro_metrics": repro_metrics,
        "best_params": best_params,
        "best_cv_score": best_score,
        "version_id": version_id,
        "drift": drift,
        "audit": audit,
    }


print("\n" + "=" * 70)
print(f"  Orchestrated pipeline run — run_id={RUN_ID}")
print("=" * 70)

orchestration = asyncio.run(orchestrated_run())


# ── Checkpoint ──────────────────────────────────────────────────────────
assert (
    orchestration["drift"] < 1e-3
), f"Task 3: same seed must reproduce same metrics (drift={orchestration['drift']:.6f})"
assert (
    orchestration["metrics"]["auc_pr"] > 0.5
), "Task 3: final model must clear the quality gate"
assert len(orchestration["audit"]) >= 6, "Task 3: audit trail must record every stage"
print("\n[ok] Checkpoint passed — orchestrated pipeline + reproducibility verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the audit trail + reproducibility certificate
# ════════════════════════════════════════════════════════════════════════

print("=== Audit Trail ===")
for row in orchestration["audit"]:
    print(f"  [{row['stage']:<22}] {row['detail']}")

print_metric_block("Final metrics (orchestrated run)", orchestration["metrics"])
print_metric_block("Reproducibility re-run metrics", orchestration["repro_metrics"])
print(f"\n  AUC-PR drift: {orchestration['drift']:.6f}  (threshold: 1e-3 — PASS)")
print(
    f"\nPipeline DAG:"
    f"\n  preprocess -> hyperparameter_search -> train -> evaluate"
    f"\n       -> quality_gate -> [register -> promote]"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Full Singapore Banking ML-Ops ROI
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  APPLY: End-to-End Credit Risk ML Ops (S$48B Portfolio)")
print("=" * 70)
print(headline_roi_text())
print(
    f"\n  Plus a reproducibility certificate for run_id={RUN_ID[:8]}"
    f"\n  that maps 1:1 to the PipelineAuditEntry rows above."
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Tied WorkflowBuilder, DataFlow, HyperparameterSearch, and
      ModelRegistry into ONE orchestrated Kailash pipeline
  [x] Wrote every stage transition into a DataFlow-managed audit table
  [x] Branched on a quality gate (AUC-PR > 0.5 -> register, else retrain)
  [x] Proved reproducibility: same seed + same code -> drift < 1e-3
  [x] Mapped the pipeline to a full Singapore banking ML-ops ROI

  Exercise 7 complete. Next: MLFP03 Exercise 8 — conformal prediction,
  DriftMonitor, production monitoring.

  Version promoted to production: credit_default_v2 v{orchestration['version_id']}
  Portfolio anchor: S${SG_BANK_PORTFOLIO['portfolio_sgd']/1e9:.0f}B
"""
)
