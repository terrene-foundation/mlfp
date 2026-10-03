# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 8.4: Deployment Pipeline — Registry, Promotion, Rollback
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Register a trained model in kailash-ml's ModelRegistry with metrics
#   - Log the run (params, metrics, tags) with ExperimentTracker
#   - Promote staging → production only when computed gates pass
#   - Rehearse a rollback and verify the restored model is the same one
#   - Read the registry back as a production dashboard
#
# PREREQUISITES: 01_conformal_prediction.py; 03_model_card.py (8.5 reads
# the card this exercise's model is described by).
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory     — what a registry buys you vs "pickle + upload"
#   2. Build      — train, measure, compute the promotion gates
#   3. Train      — register, track, promote, rollback drill
#   4. Visualise  — production dashboard from registry metadata
#   5. Apply      — a rollback service level for a Singapore lender
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import pickle
import time
from typing import Any

import numpy as np
import plotly.graph_objects as go

from kailash_ml import ExperimentTracker
from kailash_ml.types import MetricSpec

from shared.mlfp03.ex_8 import (
    DECISION_THRESHOLD,
    OUTPUT_DIR,
    PRODUCTION_MODEL_NAME,
    conformal_on_test,
    evaluate_classification,
    load_credit_split,
    open_registry,
    train_calibrated_model,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why a model registry is non-negotiable
# ════════════════════════════════════════════════════════════════════════
# The temptation: pickle the model and upload the file. One line. Ships
# today. Fails the day something goes wrong in production and you need to:
#
#   - Prove WHICH version was live when the incident occurred
#   - Roll back to the last known-good version in minutes, not hours
#   - Compare the live model's metrics with the candidate's
#   - Show a reviewer the full promotion history with reasons
#
# A MODEL REGISTRY gives you:
#   1. Numbered, immutable versions (no "which model.pkl is live?")
#   2. Stages: staging / shadow / production / archived
#   3. Every stage transition recorded with a reason string
#   4. Metrics stored with each version
#   5. Rollback as two promotions: promote_model(name, old_version,
#      "staging") then promote_model(name, old_version, "production").
#      Promoting a version to production archives the one it replaces.
#
# Pair it with ExperimentTracker for run-level lineage (which run, which
# parameters produced this version) and you have the "who / what / when /
# why" every incident review needs.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: train, measure, compute the promotion gates
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP03 Exercise 8.4 — Deployment Pipeline")
print("=" * 70)

split = load_credit_split()
X_train, y_train = split["X_train"], split["y_train"]
X_test, y_test = split["X_test"], split["y_test"]
feature_names = split["feature_names"]

# Hint: train_calibrated_model(<X>, <y>, <feature names>)
calibrated_model = ____
y_proba = calibrated_model.predict_proba(X_test)[:, 1]
metrics = evaluate_classification(y_test, y_proba, threshold=DECISION_THRESHOLD)
conformal = conformal_on_test(y_test, y_proba, alpha=0.10)
no_skill_brier = float(y_test.mean() * (1 - y_test.mean()))

# Promotion gates: each one is computed from this model's measurements.
# The AUC floor is a policy choice for this example, not a standard.
AUC_FLOOR = 0.70
gates = {
    f"AUC-ROC >= {AUC_FLOOR}": metrics["auc_roc"] >= AUC_FLOOR,
    # Hint: compare the model's Brier score with no_skill_brier
    "Brier beats the base-rate forecast": ____,
    "Conformal coverage within 2 pts of target": (
        conformal["coverage"] >= 1 - conformal["alpha"] - 0.02
    ),
}
print(
    f"\nCandidate: AUC-ROC={metrics['auc_roc']:.4f}  AUC-PR={metrics['auc_pr']:.4f}  "
    f"Brier={metrics['brier']:.4f} (base rate {no_skill_brier:.4f})  "
    f"Coverage={conformal['coverage']:.3f}"
)
for gate, passed in gates.items():
    print(f"  {'[pass]' if passed else '[FAIL]'}  {gate}")


# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert metrics["auc_roc"] > 0.5, "Task 2: Model should beat random"
assert all(isinstance(v, bool) for v in gates.values()), "Task 2: gates are computed booleans"
print("\n[ok] Checkpoint 1 — candidate measured, gates computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: register, track, promote, rehearse a rollback
# ════════════════════════════════════════════════════════════════════════
# The tracker gets its own database file; the registry lives in the
# exercise's registry database (shared.mlfp03.ex_8.DB_URL).

TRACKING_DB_URL = f"sqlite:///{(OUTPUT_DIR / 'ex8_tracking.db').resolve().as_posix()}"


async def register_promote_rollback() -> dict[str, Any]:
    registry, conn = await open_registry()
    tracker = await ExperimentTracker.create(store_url=TRACKING_DB_URL)
    try:
        # 1. Track the run that produced this candidate.
        async with tracker.track("mlfp03_ex8_production") as run:
            await run.log_params(
                {"model": "lightgbm+isotonic", "decision_threshold": f"{DECISION_THRESHOLD:.4f}"}
            )
            await run.log_metrics({**metrics, "conformal_coverage": conformal["coverage"]})
            await run.add_tag("market", "singapore")
            run_id = run.run_id

        # 2. Register the calibrated model with its metrics (lands in staging).
        # Hint: registry.register_model(<name>, <pickled model bytes>,
        #       metrics=[MetricSpec(name=..., value=...), ...]) — register
        #       auc_roc, auc_pr, brier (higher_is_better=False) and
        #       conformal_coverage
        registered = await ____
        version = registered.version

        # 3. Promote ONLY if every gate passed; the reason quotes the numbers.
        if not all(gates.values()):
            failed = [g for g, ok in gates.items() if not ok]
            raise RuntimeError(f"Promotion blocked by gates: {failed}")
        # Hint: registry.promote_model(<name>, <version>, "production",
        #       reason=<a string quoting the run id and the measured metrics>)
        promoted = await ____

        # 4. Rollback DRILL. Publish a second release (here the same artefact,
        #    so the drill is safe), then roll back to the version above.
        drill = await registry.register_model(PRODUCTION_MODEL_NAME, pickle.dumps(calibrated_model))
        await registry.promote_model(
            PRODUCTION_MODEL_NAME, drill.version, "production", reason="rollback drill: release"
        )
        t0 = time.perf_counter()
        # Hint: an archived version cannot jump straight to production —
        # promote_model it to "staging" first, then to "production"
        await ____
        restored = await ____
        rollback_seconds = time.perf_counter() - t0

        # 5. Verify: the model now served from "production" is the one we tested.
        # Hint: registry.get_model(<name>, stage=<the live stage>)
        live = await ____
        # Unpickling executes code: only load artefacts you trained yourself.
        live_model = pickle.loads(await registry.load_artifact(PRODUCTION_MODEL_NAME, live.version))
        max_pred_diff = float(np.abs(live_model.predict_proba(X_test)[:, 1] - y_proba).max())

        versions = await registry.get_model_versions(PRODUCTION_MODEL_NAME)
    finally:
        await tracker.close()
        await conn.close()
    return {
        "run_id": run_id,
        "version": version,
        "promoted_stage": promoted.stage,
        "drill_version": drill.version,
        "restored_stage": restored.stage,
        "live_version": live.version,
        "rollback_seconds": rollback_seconds,
        "max_pred_diff": max_pred_diff,
        "versions": [(v.version, v.stage) for v in versions],
    }


result = asyncio.run(register_promote_rollback())
model_version = result["version"]
print(f"[tracker]  run {result['run_id']}")
print(f"[registry] {PRODUCTION_MODEL_NAME} v{model_version} → {result['promoted_stage']}")
print(
    f"[drill]    v{result['drill_version']} released, then rolled back to "
    f"v{result['live_version']} in {result['rollback_seconds']:.3f} s"
)
print(f"[verify]   max |Δp| between live artefact and tested model: {result['max_pred_diff']:.2e}")


# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert result["promoted_stage"] == "production", "Task 3: candidate should reach production"
assert result["live_version"] == model_version, "Task 3: rollback must restore the tested version"
assert result["max_pred_diff"] < 1e-12, "Task 3: the live artefact must be the model we tested"
assert (result["drill_version"], "archived") in result["versions"], (
    "Task 3: the replaced release should be archived"
)
print("\n[ok] Checkpoint 2 — registered, promoted, rollback rehearsed and verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: production dashboard from the registry
# ════════════════════════════════════════════════════════════════════════
# What an operator sees at 9am: which version is live, what it measured,
# and the stage of every version (the registry keeps them all).

stage_counts: dict[str, int] = {}
for _, stage in result["versions"]:
    stage_counts[stage] = stage_counts.get(stage, 0) + 1
print("=== Production dashboard ===")
print(f"  Live: {PRODUCTION_MODEL_NAME} v{result['live_version']}")
print(
    f"  AUC-ROC {metrics['auc_roc']:.4f} | AUC-PR {metrics['auc_pr']:.4f} | "
    f"Brier {metrics['brier']:.4f} | coverage {conformal['coverage']:.1%}"
)
print(f"  Versions by stage: {stage_counts}")

fig = go.Figure()
labels = ["AUC-ROC", "AUC-PR", "1 - Brier", "Coverage", f"Recall @ {DECISION_THRESHOLD:.2f}"]
values = [
    metrics["auc_roc"],
    metrics["auc_pr"],
    1 - metrics["brier"],
    conformal["coverage"],
    metrics["recall"],
]
fig.add_trace(
    go.Bar(
        x=labels,
        y=values,
        marker_color=["#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"],
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
    )
)
fig.add_hline(
    y=1 - conformal["alpha"],
    line_dash="dash",
    line_color="#8b5cf6",
    annotation_text=f"coverage target {1 - conformal['alpha']:.0%}",
)
fig.update_layout(
    title=(
        f"Production dashboard — {PRODUCTION_MODEL_NAME} v{result['live_version']} "
        f"({', '.join(f'{k}: {v}' for k, v in stage_counts.items())})"
    ),
    yaxis_title="Score (higher is better)",
    yaxis_range=[0, 1.1],
    height=500,
)
viz_path = OUTPUT_DIR / "ex8_04_dashboard.html"
fig.write_html(str(viz_path))
print(f"Saved: {viz_path}")


# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert viz_path.exists(), "Task 4: Dashboard should be written"
assert stage_counts.get("production") == 1, "Task 4: exactly one live version"
print("\n[ok] Checkpoint 3 — dashboard rendered from registry state\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: a rollback service level for a Singapore lender
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a lender scores ~600 applications an hour at
# peak. A bad release is live until someone restores the previous one.
# Without a registry, restoring means finding the old file, re-deploying
# and restarting — assume ~2 hours end to end. With the registry, the
# restore itself is the two promotions you just timed; the people part
# (noticing, deciding) still dominates, so we assume 15 minutes total.

apps_per_hour = 600  # illustrative volume
manual_window_h = 2.0  # illustrative manual restore
registry_window_h = 0.25  # illustrative: alert + decision + timed restore
print("=== Applications scored by a bad release before it is replaced ===")
for label, hours in (("manual redeploy", manual_window_h), ("registry rollback", registry_window_h)):
    print(f"  {label:<18} ~{hours * 60:>4.0f} min  → {apps_per_hour * hours:>6,.0f} applications")
print(f"  (the registry operation itself took {result['rollback_seconds']:.3f} s in the drill)")

# LIMITATION: the registry makes the restore fast; it does not tell you
# WHEN to restore. That comes from monitoring (8.2) and from readiness
# gates that stop bad releases in the first place (8.5).


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Registered {PRODUCTION_MODEL_NAME} v{model_version} with its metrics
  [x] Logged the run (params + metrics + tags) via ExperimentTracker
  [x] Promoted to production only after {len(gates)} computed gates passed
  [x] Rehearsed a rollback ({result['rollback_seconds']:.3f} s) and proved the
      restored model predicts exactly like the one you tested
  [x] Built the operator's dashboard from registry state

  KEY INSIGHT: The registry is not storage. It's the record that lets
  your team act on an incident in minutes and explain it afterwards.

  Next: 05_production_readiness.py — the final gate before the model
  goes live, and the capstone wrap for Module 3.
"""
)
