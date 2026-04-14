# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 7.4: ModelRegistry Lifecycle (Staging → Production)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Define a ModelSignature (input schema + output contract)
#   - Register a model with kailash-ml's ModelRegistry
#   - Promote through `staging -> production` with an audit reason
#   - Understand why ModelSignature is the contract InferenceServer enforces
#
# PREREQUISITES: 03_hyperparameter_search.py
# ESTIMATED TIME: ~35 min
#
# 5-PHASE R10:
#   1. Theory     — why a registry beats "the pickle on S3"
#   2. Build      — ModelSignature + register_model
#   3. Train      — promote through the lifecycle with a reason
#   4. Visualise  — inspect the registered version + signature
#   5. Apply      — audit-grade lineage for MAS on-site inspection
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import pickle

import lightgbm as lgb

from kailash.db import ConnectionManager
from kailash_ml.engines.model_registry import ModelRegistry
from kailash_ml.types import FeatureField, FeatureSchema, MetricSpec, ModelSignature

from shared.mlfp03.ex_7 import (
    DB_URL,
    RANDOM_SEED,
    SG_BANK_PORTFOLIO,
    compute_classification_metrics,
    headline_roi_text,
    prepare_credit_split,
    print_metric_block,
    scale_pos_weight_for,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why a ModelRegistry?
# ════════════════════════════════════════════════════════════════════════
# Every ML-outage postmortem traces to "the pickle on S3 with no
# signature, no version, no promotion reason." The registry is the
# antidote: typed versions, captured metrics, audit-grade transitions.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: train, define ModelSignature + register
# ════════════════════════════════════════════════════════════════════════

split = prepare_credit_split()
pos_weight = scale_pos_weight_for(split.y_train)

best_params = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 63,
    "min_child_samples": 20,
}

best_model = lgb.LGBMClassifier(
    **best_params,
    random_state=RANDOM_SEED,
    verbose=-1,
    scale_pos_weight=pos_weight,
)
best_model.fit(split.X_train, split.y_train)

y_pred = best_model.predict(split.X_test)
y_proba = best_model.predict_proba(split.X_test)[:, 1]
metrics = compute_classification_metrics(split.y_test, y_pred, y_proba)
print_metric_block("Model to register", metrics)


# TODO: build a FeatureSchema with one FeatureField per training column
# Hint: features=[FeatureField(name=f, dtype="float64") for f in split.feature_columns]
input_schema = FeatureSchema(
    name="credit_model_input",
    features=____,
    entity_id_column="application_id",
)

# TODO: build the ModelSignature — outputs are default_probability + default_label
# Hint: output_columns=["default_probability", "default_label"]
signature = ModelSignature(
    input_schema=input_schema,
    output_columns=____,
    output_dtypes=["float64", "int64"],
    model_type="classifier",
)


async def register_and_promote() -> tuple[int, str]:
    conn = ConnectionManager(DB_URL)
    await conn.initialize()
    try:
        registry = ModelRegistry(conn)
        artefact_bytes = pickle.dumps(best_model)

        # TODO: register the model with its metrics
        # Hint: await registry.register_model(name=..., artifact=..., metrics=[...])
        version = await registry.register_model(
            name="credit_default_v2",
            artifact=artefact_bytes,
            metrics=[
                MetricSpec(name="auc_pr", value=metrics["auc_pr"]),
                MetricSpec(name="auc_roc", value=metrics["auc_roc"]),
                MetricSpec(name="f1_score", value=metrics["f1"]),
            ],
        )

        # TODO: promote the model from staging to production with a reason
        # Hint: await registry.promote_model(name=..., version=..., target_stage="production", reason="...")
        await registry.promote_model(
            name="credit_default_v2",
            version=version.version,
            target_stage=____,
            reason=(
                f"Quality gates passed: AUC-PR={metrics['auc_pr']:.4f}, "
                f"AUC-ROC={metrics['auc_roc']:.4f}, F1={metrics['f1']:.4f}. "
                f"Hyperparameters optimised via kailash-ml Bayesian search."
            ),
        )
        return version.version, "production"
    finally:
        await conn.close()


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: run the async registration + promotion
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Registering credit_default_v2 and promoting to production")
print("=" * 70)

version_id, stage = asyncio.run(register_and_promote())


# ── Checkpoint ──────────────────────────────────────────────────────────
assert version_id is not None, "Task 3: register_model must return a version"
assert stage == "production", "Task 3: promotion should land in production"
assert (
    len(signature.input_schema.features) == split.feature_count
), "Task 3: ModelSignature features must match training features"
print(f"\n  credit_default_v2 version={version_id} stage={stage}")
print("\n[ok] Checkpoint passed — registration + promotion complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE signature + promotion reason
# ════════════════════════════════════════════════════════════════════════

print("=== ModelSignature ===")
print(f"  input features : {len(signature.input_schema.features)}")
print(f"  entity id col  : {signature.input_schema.entity_id_column}")
print(f"  output columns : {signature.output_columns}")
print(f"  output dtypes  : {signature.output_dtypes}")
print(f"  model type     : {signature.model_type}")

print("\n=== Lifecycle ===")
print("  experiment -> register (staging) -> promote (production) -> retire")
print(f"  credit_default_v2 v{version_id} is now PRODUCTION")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: MAS On-Site Inspection Lineage
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  APPLY: MAS On-Site Inspection Lineage")
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
  [x] Built a ModelSignature with FeatureSchema + FeatureField
  [x] Registered a LightGBM model in kailash-ml's ModelRegistry
  [x] Promoted through staging -> production with an audit reason
  [x] Tied the registry to a real MAS on-site inspection scenario

  Next: 05_orchestrated_pipeline.py — one end-to-end run, fully audited.
  Version now in production: credit_default_v2 v{version_id}
  Portfolio anchor: S${SG_BANK_PORTFOLIO['portfolio_sgd']/1e9:.0f}B
"""
)
