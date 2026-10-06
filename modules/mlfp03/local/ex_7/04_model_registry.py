# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 7.4: ModelRegistry Lifecycle (Staging → Production)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Read the signature TrainingPipeline attaches when it registers a model
#   - Define a richer ModelSignature (input schema + output contract) and
#     register a version WITH it via ModelRegistry.register_model
#   - Record test-set metrics on the version with MetricSpec
#   - Promote staging -> production only when the quality gate passes,
#     with the evidence in the promotion reason
#
# PREREQUISITES: 03_hyperparameter_search.py
# ESTIMATED TIME: ~35 min
#
# 5-PHASE R10:
#   1. Theory     — why a registry beats "the pickle on S3"
#   2. Build      — ModelSignature for the production contract
#   3. Train      — train a candidate, register + gate + promote
#   4. Visualise  — inspect versions, stages and signatures
#   5. Apply      — answering "which model was live on that day?"
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from typing import Any

from kailash_ml import TrainingPipeline
from kailash_ml.engines.training_pipeline import EvalSpec, ModelSpec
from kailash_ml.interop import to_sklearn_input
from kailash_ml.types import MetricSpec, ModelSignature

from shared.mlfp03.ex_7 import (
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
    promotion_gate,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why a ModelRegistry?
# ════════════════════════════════════════════════════════════════════════
# The worst production handoff is:
#
#   "The new model is in s3://bucket/models/model_final_v2_FINAL.pkl —
#    I think Alice uploaded it last Tuesday, ping her if it breaks."
#
# No signature, no version, no promotion reason, no rollback path, no
# record of which metric the model was judged on.
#
# kailash-ml's ModelRegistry gives every model:
#   - register_model(name, artefact, metrics=, signature=) -> a numbered
#     version, starting in stage "staging"
#   - promote_model(name, version, target_stage, reason=) — stages are
#     staging / shadow / production / archived, and only some moves are
#     allowed (e.g. archived -> production is not; it goes via staging)
#   - get_model(name, version) / get_model(name, stage=...) to look up
#     exactly what was registered, with its metrics and signature
#
# A ModelSignature is the I/O contract stored with the version: which
# feature columns (and dtypes) go in, which columns come out. It is the
# thing a serving layer can check requests against.
#
# TrainingPipeline already registers every model it trains, with an
# automatic signature whose single output column is "prediction". For the
# production entry we want the contract the credit system actually
# consumes — a default probability AND a 0/1 decision — so we register
# that version ourselves.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the production signature
# ════════════════════════════════════════════════════════════════════════

CANDIDATE_NAME = "credit_default_candidates"
PRODUCTION_NAME = "credit_default"
HYPERPARAMETERS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "num_leaves": 31,
    "min_child_samples": 40,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}

dev, test, feature_cols = prepare_credit_frames()
input_schema = credit_feature_schema(feature_cols)
default_rate = float(dev[TARGET_COLUMN].mean())

# TODO: declare the production contract — the input schema above, two
# outputs (a float64 default probability and an int64 0/1 label), and
# the model type.
# Hint: ModelSignature(input_schema=..., output_columns=[...],
#                      output_dtypes=[...], model_type=...)
signature = ____
print(
    f"\nProduction signature: {len(signature.input_schema.features)} inputs -> "
    f"{signature.output_columns}"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN a candidate, register it with the signature, gate, promote
# ════════════════════════════════════════════════════════════════════════


async def register_and_promote() -> dict[str, Any]:
    registry, conn = await build_training_registry()
    try:
        # 1) Candidate: TrainingPipeline fits, evaluates its holdout of the
        #    dev frame and registers the model with an automatic signature.
        pipeline = TrainingPipeline(feature_store=None, registry=registry)
        candidate = await pipeline.train(
            data=dev,
            schema=input_schema,
            model_spec=ModelSpec(
                model_class="lightgbm.LGBMClassifier",
                framework="lightgbm",
                hyperparameters=HYPERPARAMETERS,
            ),
            eval_spec=EvalSpec(
                metrics=ENGINE_METRICS, split_strategy="holdout", test_size=0.2
            ),
            experiment_name=CANDIDATE_NAME,
        )
        if candidate.model_version is None:
            raise RuntimeError("TrainingPipeline did not register the candidate")
        cand_version = candidate.model_version.version
        # TODO: look up the candidate version and keep its signature
        # Hint: await registry.get_model(<name>, <version>) returns a ModelVersion
        auto_signature = ____

        # 2) Score the candidate on the untouched test frame.
        model = await load_registered_model(registry, CANDIDATE_NAME, cand_version)
        X_test, y_test, _ = to_sklearn_input(
            test, feature_columns=feature_cols, target_column=TARGET_COLUMN
        )
        y_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = compute_classification_metrics(
            y_test, (y_proba >= 0.5).astype(int), y_proba
        )

        # 3) Production entry: same artefact bytes, the richer signature and
        #    the test-set metrics recorded as MetricSpec rows.
        artifact = await registry.load_artifact(CANDIDATE_NAME, cand_version)
        # TODO: register the artefact under PRODUCTION_NAME with metrics + signature
        # Hint: registry.register_model(name, artifact, metrics=[...], signature=...)
        registered = await registry.____(
            PRODUCTION_NAME,
            artifact,
            metrics=[
                MetricSpec(name=k, value=v, split="test", higher_is_better=(k != "log_loss"))
                for k, v in test_metrics.items()
            ],
            signature=signature,
        )

        # 4) Quality gate decides whether this version may serve.
        # TODO: run the shared quality gate on the test metrics
        # Hint: promotion_gate(<metrics dict>, <default rate>)
        gate = ____
        if gate["promote"]:
            await registry.promote_model(
                PRODUCTION_NAME,
                registered.version,
                # TODO: target stage for a version that passed the gate
                # Hint: stages are staging / shadow / production / archived
                ____,
                reason=(
                    f"Gate passed on {len(y_test):,} test rows: "
                    f"AUC-ROC {gate['auc_roc']:.4f} >= {gate['min_auc_roc']}, "
                    f"AUC-PR {gate['auc_pr']:.4f} >= {gate['min_auc_pr']:.4f}; "
                    f"artefact from {CANDIDATE_NAME} v{cand_version}"
                ),
            )
        stored = await registry.get_model(PRODUCTION_NAME, registered.version)
        versions = await registry.get_model_versions(PRODUCTION_NAME)
    finally:
        await conn.close()

    return {
        "candidate_version": cand_version,
        "candidate_holdout": {k: float(v) for k, v in candidate.metrics.items()},
        "auto_signature": auto_signature,
        "test_metrics": test_metrics,
        "gate": gate,
        "stored": stored,
        "versions": versions,
    }


print("\n" + "=" * 70)
print(f"  Training a candidate and registering {PRODUCTION_NAME}")
print("=" * 70)
out = asyncio.run(register_and_promote())
stored = out["stored"]
print_metric_block("Candidate — test-frame metrics", out["test_metrics"])

# ── Checkpoint ──────────────────────────────────────────────────────────
assert out["auto_signature"].output_columns == ["prediction"], (
    "Task 3: TrainingPipeline's automatic signature has one 'prediction' output"
)
assert stored.signature.output_columns == signature.output_columns, (
    "Task 3: the production version must carry OUR signature"
)
assert len(stored.signature.input_schema.features) == len(feature_cols), (
    "Task 3: signature inputs must match the training features"
)
expected_stage = "production" if out["gate"]["promote"] else "staging"
assert stored.stage == expected_stage, (
    f"Task 3: stage should be {expected_stage} given the gate verdict"
)
print(f"\n  {PRODUCTION_NAME} v{stored.version} stage={stored.stage}")
print("\n[ok] Checkpoint passed — signature attached, gate respected\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE versions, stages and the two signatures
# ════════════════════════════════════════════════════════════════════════

auto = out["auto_signature"]
print("=== Signatures ===")
print(f"  {CANDIDATE_NAME} v{out['candidate_version']} (automatic):")
print(f"    outputs {auto.output_columns} {auto.output_dtypes}")
print(f"  {PRODUCTION_NAME} v{stored.version} (ours):")
print(f"    outputs {stored.signature.output_columns} {stored.signature.output_dtypes}")
print(f"    inputs  {len(stored.signature.input_schema.features)} features, "
      f"entity id '{stored.signature.input_schema.entity_id_column}'")

gate = out["gate"]
print("\n=== Quality gate (test frame) ===")
print(f"  AUC-ROC {gate['auc_roc']:.4f}  (needs >= {gate['min_auc_roc']})")
print(f"  AUC-PR  {gate['auc_pr']:.4f}  (needs >= {gate['min_auc_pr']:.4f} = 2 x default rate)")
print(f"  verdict: {'PROMOTE' if gate['promote'] else 'stay in staging'}")

print(f"\n=== {PRODUCTION_NAME}: every registered version ===")
for v in out["versions"]:
    recorded = {m.name: round(m.value, 4) for m in (v.metrics or [])}
    print(f"  v{v.version:<3} stage={v.stage:<11} auc_pr={recorded.get('auc_pr', 'n/a')}")
print("\n  Lifecycle stages: staging -> shadow -> production -> archived")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: "Which model was live on that day?"
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a customer disputes a declined loan from last
# quarter. The bank's model validation team must show which model made
# the recommendation and why that model was allowed to be in production.
#
# With the registry, the answer is a lookup: the version that held the
# "production" stage at that date, its signature (what it was fed), its
# recorded test metrics, and the promotion reason containing the gate
# numbers. Without it, the team reconstructs history from file names,
# chat threads and memory — and usually cannot prove anything.
print("\n" + "=" * 70)
print("  APPLY: Model Lineage on Demand")
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
  [x] Read the automatic signature TrainingPipeline registers
  [x] Registered a version with your own ModelSignature + MetricSpec rows
  [x] Let the quality gate decide staging vs production
  [x] Listed every version of a model with its stage and metrics

  KEY INSIGHT: Models don't fail because they're wrong — they fail
  because nobody remembers which one was in production on the day of
  the incident. The registry makes that a lookup, not a forensics
  project.

  Next: 05_orchestrated_pipeline.py — stitch files 01-04 into one
  run that audits itself end-to-end.

  This run: {PRODUCTION_NAME} v{stored.version} is in {stored.stage}
"""
)
