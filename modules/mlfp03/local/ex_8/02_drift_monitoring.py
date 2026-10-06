# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 8.2: Drift Monitoring with DriftMonitor + DataFlow
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Explain covariate, label and concept drift
#   - Set up kailash-ml's DriftMonitor with PSI + KS alerting thresholds
#   - Store a reference distribution and check new batches against it
#   - Inject gradual and sudden drift and verify the monitor catches both
#   - Persist every drift check with DataFlow and acknowledge alerts
#
# PREREQUISITES: 01_conformal_prediction.py
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory     — drift types, PSI and KS, and the false-alarm problem
#   2. Build      — reference model, DriftMonitor, DataFlow DriftCheck table
#   3. Train      — set the reference, check clean / gradual / sudden batches
#   4. Visualise  — PSI per feature per scenario + AUC under drift
#   5. Apply      — a retraining rule for a Singapore lender
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score

from dataflow import DataFlow
from kailash.db import ConnectionManager
from kailash_ml import DriftMonitor

from shared.mlfp03.ex_8 import (
    DRIFT_FEATURES,
    MODEL_NAME,
    OUTPUT_DIR,
    load_credit_split,
    simulate_gradual_drift,
    simulate_sudden_drift,
    to_frame,
    train_calibrated_model,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why drift is the silent killer of production ML
# ════════════════════════════════════════════════════════════════════════
# A model is trained on a photograph of the world. The world keeps moving.
#
# COVARIATE DRIFT:  p(x) changes — e.g. rates rise and applicants'
#                   debt-to-income ratios shift. Same rules, new inputs.
# LABEL DRIFT:      p(y) changes — e.g. the default rate jumps from 12% to
#                   17% in a downturn.
# CONCEPT DRIFT:    p(y | x) changes — the same debt ratio that was safe
#                   last year is risky this year.
#
# Input-distribution checks (this file) catch covariate drift early,
# before labels arrive. Label and concept drift need outcomes, so they
# are caught by tracking live performance once defaults are observed.
#
# TWO COMPLEMENTARY TESTS, run per feature:
#   PSI (Population Stability Index) = Σ (p_new - p_ref) · ln(p_new / p_ref)
#       over bins; a common credit-industry rule of thumb reads
#       < 0.1 stable, 0.1-0.2 moderate, > 0.2 significant.
#   KS (Kolmogorov-Smirnov) — largest gap between the two empirical CDFs,
#       with a p-value.
#
# THE FALSE-ALARM PROBLEM: with tens of thousands of rows, KS flags tiny,
# harmless differences; and with 33 features tested at p < 0.05 you
# expect 1-2 alarms from chance alone even when nothing moved. So we set
# the KS threshold near 0.05 / 33 ≈ 0.0015 (a Bonferroni-style
# correction) and read PSI for "how big", KS for "is it real".


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: reference model, DriftMonitor, DataFlow table
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP03 Exercise 8.2 — Drift Monitoring")
print("=" * 70)

split = load_credit_split()
X_train, y_train = split["X_train"], split["y_train"]
X_test, y_test = split["X_test"], split["y_test"]
feature_names = split["feature_names"]

# Hint: train_calibrated_model(<X>, <y>, <feature names>)
calibrated_model = ____
auc_ref = float(roc_auc_score(y_test, calibrated_model.predict_proba(X_test)[:, 1]))
print(f"\nReference model AUC-ROC on the clean test batch: {auc_ref:.4f}")

# Alert thresholds (see Theory): PSI > 0.2 = "significant" shift; the KS
# p-value cut-off is Bonferroni-tightened for 33 simultaneous tests.
PSI_THRESHOLD = 0.2
KS_THRESHOLD = 0.001

# Three incoming "batches" — one clean, two with injected drift.
drift_idx = [feature_names.index(f) for f in DRIFT_FEATURES]
income_idx = feature_names.index("income_sgd")
batches = {
    "clean": X_test,
    # Hint: simulate_gradual_drift(<reference X>, <batch X>, <column indices>, shift=0.5)
    "gradual": ____,
    "sudden": simulate_sudden_drift(X_train, X_test, income_idx, sigma_shift=3.0),
}

# Two monitoring stores, both separate from the model registry's file:
#   - DriftMonitor keeps its references and reports in its own tables.
#     (The registry database already has a table with the same name but a
#     different layout, so sharing that file fails.)
#   - DataFlow keeps our DriftCheck audit table: a monitor nobody can
#     query later is a log line, not an audit trail.
DRIFT_DB_URL = f"sqlite:///{(OUTPUT_DIR / 'ex8_drift.db').resolve().as_posix()}"
# RUN_ID tags this run's rows; earlier runs stay in the table as history.
RUN_ID = uuid.uuid4().hex[:8]
MONITORING_DB_URL = f"sqlite:///{(OUTPUT_DIR / 'ex8_monitoring.db').resolve().as_posix()}"
db = DataFlow(MONITORING_DB_URL)


@db.model
class DriftCheck:
    """One row per DriftMonitor.check_drift() call."""

    id: int
    run_id: str
    model_name: str
    batch: str
    overall_drift: bool
    max_psi: float
    severity: str
    n_flagged: int
    flagged_features: str
    status: str = "open"


# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert auc_ref > 0.5, "Task 2: Reference AUC should beat random"
assert set(batches) == {"clean", "gradual", "sudden"}, "Task 2: three batches"
print("\n[ok] Checkpoint 1 — reference model, batches and monitor config ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: set the reference, check three batches, persist results
# ════════════════════════════════════════════════════════════════════════
# The reference distribution is the TRAINING data the model learned from.
# check_drift() runs PSI + KS (and Jensen-Shannon) on every feature and
# stores a report in the monitor's own table. We also write one DriftCheck
# row per batch with DataFlow, then run the alert lifecycle on those rows:
# READ them back, UPDATE the alerts to "acknowledged", and DELETE a
# smoke-test row a CI health check wrote.


async def monitor_and_persist() -> dict[str, Any]:
    conn = ConnectionManager(DRIFT_DB_URL)
    await conn.initialize()
    await db.initialize()
    try:
        # Hint: DriftMonitor(<connection>, tenant_id="_single", psi_threshold=..., ks_threshold=...)
        monitor = ____
        # Hint: monitor.set_reference_data(<model name>, <polars frame of the
        # training rows>, <feature column names>) — to_frame() builds the frame
        await ____

        reports = {}
        for name, X_batch in batches.items():
            # Hint: monitor.check_drift(<model name>, <polars frame of the batch>)
            report = await ____
            reports[name] = report
            flagged = [f.feature_name for f in report.feature_results if f.drift_detected]
            # CREATE — one audit row per check
            await db.express.create(
                "DriftCheck",
                {
                    "run_id": RUN_ID,
                    "model_name": MODEL_NAME,
                    "batch": name,
                    "overall_drift": bool(report.overall_drift_detected),
                    "max_psi": max(f.psi for f in report.feature_results),
                    "severity": str(report.overall_severity),
                    "n_flagged": len(flagged),
                    "flagged_features": ",".join(flagged),
                },
            )

        # READ — every check from this run
        # Hint: db.express.list(<model>, filter={...}) — filter on this run's id
        stored = await ____

        # UPDATE — the risk team acknowledges each open alert
        acknowledged = []
        for row in stored:
            if bool(row["overall_drift"]):
                # Hint: db.express.update(<model>, <row id>, {<field>: <new value>})
                updated = await ____
                acknowledged.append(updated)

        # DELETE — a CI health check writes a smoke row; it must not linger
        await db.express.create(
            "DriftCheck",
            {
                "run_id": RUN_ID,
                "model_name": "ci_smoke_test",
                "batch": "smoke",
                "overall_drift": False,
                "max_psi": 0.0,
                "severity": "none",
                "n_flagged": 0,
                "flagged_features": "",
            },
        )
        smoke = await db.express.find_one(
            "DriftCheck", {"run_id": RUN_ID, "model_name": "ci_smoke_test"}
        )
        # Hint: db.express.delete(<model>, <row id>) returns True on success
        deleted = await ____
        rows_after = await db.express.list("DriftCheck", filter={"run_id": RUN_ID})

        # The monitor keeps its own report history too (latest first).
        history = await monitor.get_drift_history(MODEL_NAME, limit=3)
    finally:
        await db.close_async()
        await conn.close()
    return {
        "reports": reports,
        "acknowledged": acknowledged,
        "deleted": deleted,
        "rows": rows_after,
        "history": history,
    }


out = asyncio.run(monitor_and_persist())
reports = out["reports"]

print(f"{'Batch':<9} {'Drift?':>7} {'Severity':>9} {'#flagged':>9}  Flagged features")
print("─" * 78)
for name, report in reports.items():
    flagged = [f.feature_name for f in report.feature_results if f.drift_detected]
    print(
        f"{name:<9} {str(report.overall_drift_detected):>7} "
        f"{report.overall_severity:>9} {len(flagged):>9}  {', '.join(flagged) or '-'}"
    )

print(f"\nDriftCheck rows stored for run {RUN_ID}:")
for row in out["rows"]:
    print(
        f"  id={row['id']:<4} batch={row['batch']:<8} max_psi={row['max_psi']:>7.3f} "
        f"status={row['status']}"
    )
print(f"Monitor's own history: {len(out['history'])} most recent reports retrieved")

# ── Checkpoint 2 ────────────────────────────────────────────────────────
flagged_gradual = {f.feature_name for f in reports["gradual"].feature_results if f.drift_detected}
flagged_sudden = {f.feature_name for f in reports["sudden"].feature_results if f.drift_detected}
assert not reports["clean"].overall_drift_detected, (
    "Task 3: the clean batch comes from the same distribution — it must not alarm"
)
assert set(DRIFT_FEATURES) <= flagged_gradual, "Task 3: every shifted feature must be flagged"
assert "income_sgd" in flagged_sudden, "Task 3: the sudden income shift must be flagged"
assert len(out["acknowledged"]) == 2, "Task 3: two alerts acknowledged"
assert all(r["status"] == "acknowledged" for r in out["acknowledged"]), "Task 3: update persisted"
assert out["deleted"] is True and len(out["rows"]) == 3, "Task 3: smoke row deleted"
print("\n[ok] Checkpoint 2 — drift caught, clean batch quiet, CRUD verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: PSI per feature per batch + AUC under drift
# ════════════════════════════════════════════════════════════════════════
# The simulation keeps the true labels, so we can also measure what the
# shifted inputs do to the model's ranking quality. (In production the
# outcomes arrive months later — which is why input drift is watched.)

auc_by_batch = {
    name: float(roc_auc_score(y_test, calibrated_model.predict_proba(X_batch)[:, 1]))
    for name, X_batch in batches.items()
}
for name, auc in auc_by_batch.items():
    print(f"  AUC-ROC on {name:<8} batch: {auc:.4f}  (change {auc - auc_ref:+.4f})")

psi_matrix = [
    [next(f.psi for f in reports[name].feature_results if f.feature_name == feat) for feat in feature_names]
    for name in batches
]
fig = make_subplots(
    rows=2,
    cols=1,
    row_heights=[0.65, 0.35],
    subplot_titles=("PSI per feature (DriftMonitor)", "AUC-ROC on each batch"),
    vertical_spacing=0.25,
)
fig.add_trace(
    go.Heatmap(
        z=psi_matrix,
        x=feature_names,
        y=list(batches),
        colorscale="YlOrRd",
        zmin=0,
        zmax=1.0,
        colorbar=dict(title="PSI", len=0.6, y=0.75),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Bar(
        x=list(auc_by_batch),
        y=list(auc_by_batch.values()),
        marker_color=["#10b981", "#f59e0b", "#ef4444"],
        text=[f"{v:.3f}" for v in auc_by_batch.values()],
        textposition="outside",
        showlegend=False,
    ),
    row=2,
    col=1,
)
fig.update_yaxes(range=[0.5, 1.0], row=2, col=1)
fig.update_layout(
    title="Drift monitoring: where the inputs moved, and what it cost the model",
    height=720,
)
viz_path = OUTPUT_DIR / "ex8_02_drift_psi.html"
fig.write_html(str(viz_path))
print(f"\nSaved: {viz_path}")

# INTERPRETATION: the heatmap rows for the drifted batches light up only
# on the features we shifted — the monitor localises the problem. Read
# the AUC bars alongside: a large PSI does not always mean a large AUC
# loss (the model may lean little on that feature), and a small AUC
# change is no proof the inputs are fine.

# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert len(psi_matrix) == 3 and len(psi_matrix[0]) == len(feature_names), "Task 4: PSI grid"
assert auc_by_batch["sudden"] <= auc_ref + 0.005, (
    "Task 4: corrupting income should not materially improve AUC"
)
print("\n[ok] Checkpoint 3 — drift and its performance cost visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: a retraining rule for a Singapore lender
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore lender scores ~600 unsecured-loan
# applications a day. Defaults are only known 6-12 months later, so the
# rule must act on INPUT drift first and on performance once labels land.


def retraining_action(report, live_auc: float | None, max_auc_drop: float = 0.02) -> str:
    """Turn one DriftReport (+ live AUC when labels exist) into an action."""
    # Hint: names of the flagged features whose PSI is above 0.25
    severe = ____
    if live_auc is not None and auc_ref - live_auc > max_auc_drop:
        return "RETRAIN NOW — measured performance has fallen"
    if severe:
        return f"INVESTIGATE + prepare retrain — severe shift in {', '.join(severe)}"
    if report.overall_drift_detected:
        return "WATCH — moderate shift, re-check the next batch"
    return "NO ACTION"


print("=== Retraining decisions ===")
for name, report in reports.items():
    print(f"  {name:<8} labels not yet known : {retraining_action(report, None)}")
    print(f"  {'':<8} labels known (AUC {auc_by_batch[name]:.3f}): "
          f"{retraining_action(report, auc_by_batch[name])}")

# Why daily checks: drifted applications scored before anyone notices.
daily_apps = 600  # illustrative volume
for cadence, delay_days in (("daily DriftMonitor check", 1), ("quarterly manual review", 90)):
    print(f"  {cadence:<26} worst-case {delay_days * daily_apps:>7,} applications scored on drifted inputs")

# LIMITATIONS: input drift says the world moved, not that the model is
# wrong; only outcomes confirm that. And a rule that freezes decisions
# on every alarm will be switched off — that is why the KS threshold
# was tightened and severe PSI, not any flag, triggers a retrain.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Explained covariate, label and concept drift
  [x] Set up DriftMonitor (PSI > {PSI_THRESHOLD}, KS p < {KS_THRESHOLD}) against
      the training distribution
  [x] Clean batch: no alarm; gradual + sudden injected drift: caught
      ({len(flagged_gradual)} and {len(flagged_sudden)} features flagged)
  [x] Persisted every check with DataFlow and acknowledged the alerts
  [x] Measured the cost: AUC {auc_ref:.3f} → {auc_by_batch['sudden']:.3f} under
      the sudden income shift

  KEY INSIGHT: Monitor the inputs because the outcomes arrive too late;
  tune the thresholds so an alarm still means something.

  Next: 03_model_card.py — document what the model does, how well, and
  for whom, with numbers you measured.
"""
)
