# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 8.5: Production Readiness + Module 3 Capstone
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Turn a readiness checklist into gates computed from real artefacts
#   - Read evidence left by earlier steps (model card, registry, monitor)
#   - Separate what code can gate from what needs a human decision
#   - Emit a deployment config that records the evidence
#   - See which gates fail when the incoming data shifts
#
# PREREQUISITES: 02_drift_monitoring.py, 03_model_card.py and
# 04_deployment_pipeline.py must have run — their outputs are the
# evidence this file checks. Missing evidence FAILS a gate; nothing is
# faked to make it pass.
#
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory     — why readiness gates must be computed, not asserted
#   2. Build      — compute every gate from evidence
#   3. Train      — (no new model) emit the deployment config
#   4. Visualise  — readiness traffic light + final metrics
#   5. Apply      — re-run the performance gates on shifted data
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import json
import pickle
from datetime import datetime
from typing import Any

import numpy as np
import plotly.graph_objects as go
import polars as pl

from kailash.db import ConnectionManager
from kailash_ml import DriftMonitor, diagnose
from kailash_ml.engines.model_registry import ModelNotFoundError

from shared.mlfp03.ex_8 import (
    CARD_EVIDENCE_PATH,
    CARD_PATH,
    DECISION_THRESHOLD,
    MODEL_NAME,
    OUTPUT_DIR,
    PRODUCTION_MODEL_NAME,
    conformal_on_test,
    evaluate_classification,
    fairness_report,
    fairness_summary,
    load_credit_split,
    open_registry,
    simulate_sudden_drift,
    to_frame,
    train_calibrated_model,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — The readiness checklist is structural, not paperwork
# ════════════════════════════════════════════════════════════════════════
# The temptation at this stage: "everything works, just ship it."
#
# A model that passes every offline test can still ship broken because:
#   - its probabilities are miscalibrated (no Brier check)
#   - the fairness audit was skipped "for time reasons"
#   - conformal coverage was never verified on held-out data
#   - drift monitoring was never armed, so alerts stay silent
#   - the model card describes a different model from the one deployed
#   - the registry has no previous version, so rollback is impossible
#
# Each gate below is an EXPRESSION over evidence: a measurement, a file
# another step wrote, or registry state. A gate written as `True` checks
# nothing — and a gate that creates its own evidence (e.g. writing a stub
# model card so "card exists" passes) is worse, because it looks checked.
#
# Some questions are not code's to answer. Whether an age-band disparity
# driven by different default rates is acceptable is a POLICY decision.
# The checklist's job there is to surface the measured fact for a human
# sign-off, not to hard-code a verdict either way.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: compute every gate from evidence
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP03 Exercise 8.5 — Production Readiness + Capstone")
print("=" * 70)

split = load_credit_split()
X_train, y_train = split["X_train"], split["y_train"]
X_test, y_test = split["X_test"], split["y_test"]
feature_names = split["feature_names"]

# Re-train the model 8.4 registered. Training is seeded, so if the
# pipeline is reproducible this model predicts exactly like the live one.
calibrated_model = train_calibrated_model(X_train, y_train, feature_names)
y_proba = calibrated_model.predict_proba(X_test)[:, 1]
metrics = evaluate_classification(y_test, y_proba, threshold=DECISION_THRESHOLD)
conformal = conformal_on_test(y_test, y_proba, alpha=0.10)
no_skill_brier = float(y_test.mean() * (1 - y_test.mean()))
fair = fairness_summary(
    fairness_report(y_test, y_proba, split["test_groups"], DECISION_THRESHOLD)
)

# Evidence from 8.3: the card and the measurements it was rendered from.
evidence = json.loads(CARD_EVIDENCE_PATH.read_text()) if CARD_EVIDENCE_PATH.exists() else None
card_auc = evidence["metrics"]["auc_roc"] if evidence else float("nan")
card_attributes = {r["attribute"] for r in evidence["fairness_summary"]} if evidence else set()

# Thresholds used by 8.2's monitor (KS cut-off tightened for 33 features).
PSI_THRESHOLD, KS_THRESHOLD = 0.2, 0.001


async def registry_and_monitor_evidence() -> dict[str, Any]:
    """Live model + version stages from the registry; a drift-monitor self-test."""
    out: dict[str, Any] = {"live_version": None, "live_max_diff": float("nan")}
    registry, conn = await open_registry()
    try:
        try:
            live = await registry.get_model(PRODUCTION_MODEL_NAME, stage="production")
        except ModelNotFoundError as exc:
            print(f"[evidence] no production model in the registry — run 8.4 first ({exc})")
        else:
            # Unpickling executes code: only load artefacts you trained yourself.
            live_model = pickle.loads(
                await registry.load_artifact(PRODUCTION_MODEL_NAME, live.version)
            )
            out["live_version"] = live.version
            out["live_max_diff"] = float(
                np.abs(live_model.predict_proba(X_test)[:, 1] - y_proba).max()
            )
        try:
            versions = await registry.get_model_versions(PRODUCTION_MODEL_NAME)
        except ModelNotFoundError:
            versions = []
        out["n_archived"] = sum(1 for v in versions if v.stage == "archived")
    finally:
        await conn.close()

    # Monitor self-test with 8.2's store and thresholds: quiet on clean
    # data, alarms on an injected income shift.
    drift_conn = ConnectionManager(f"sqlite:///{(OUTPUT_DIR / 'ex8_drift.db').resolve().as_posix()}")
    await drift_conn.initialize()
    try:
        monitor = DriftMonitor(
            drift_conn, tenant_id="_single", psi_threshold=PSI_THRESHOLD, ks_threshold=KS_THRESHOLD
        )
        await monitor.set_reference_data(MODEL_NAME, to_frame(X_train, feature_names), feature_names)
        clean = await monitor.check_drift(MODEL_NAME, to_frame(X_test, feature_names))
        X_shift = simulate_sudden_drift(X_train, X_test, feature_names.index("income_sgd"))
        shifted = await monitor.check_drift(MODEL_NAME, to_frame(X_shift, feature_names))
        out["monitor_clean_quiet"] = not clean.overall_drift_detected
        out["monitor_catches_shift"] = shifted.overall_drift_detected
    finally:
        await drift_conn.close()
    return out


ev = asyncio.run(registry_and_monitor_evidence())

di = dict(zip(fair["attribute"], fair["disparate_impact"]))
checklist = {
    "AUC-ROC meets the 0.70 policy floor": metrics["auc_roc"] >= 0.70,
    "Calibrated: Brier beats the base-rate forecast": metrics["brier"] < no_skill_brier,
    "Conformal coverage within 2 pts of 90%": conformal["coverage"] >= 0.88,
    "Model card written by 8.3": CARD_PATH.exists(),
    "Card describes THIS model (|ΔAUC| < 0.005)": bool(abs(card_auc - metrics["auc_roc"]) < 0.005),
    "Card reports race, gender and age fairness": card_attributes == {"race", "gender", "age_band"},
    "Four-fifths rule: race and gender": bool(di["race"] >= 0.8 and di["gender"] >= 0.8),
    "Calibrated within every group (gap < 0.02)": bool((fair["max_calibration_gap"] < 0.02).all()),
    "Drift monitor quiet on clean data, alarms on shift": bool(
        ev["monitor_clean_quiet"] and ev["monitor_catches_shift"]
    ),
    "Reproducible: live registry model = retrained model": bool(ev["live_max_diff"] < 1e-9),
    "Rollback target exists (archived version)": ev["n_archived"] >= 1,
}
# Not gated — needs a human decision (see the model card, section 8).
review_items = [
    f"{attr}: disparate impact {value:.2f} — needs model-risk sign-off"
    for attr, value in di.items()
    if attr not in ("race", "gender") and value < 0.8
]

print("\n=== Deployment Readiness Checklist ===")
for item, passed in checklist.items():
    print(f"  {'[pass]' if passed else '[FAIL]'}  {item}")
all_pass = all(checklist.values())
print(f"\n  Overall: {'READY FOR DEPLOYMENT' if all_pass else 'BLOCKED — fix failures'}")
print("  For human sign-off:", *(review_items or ["none"]), sep="\n    ")


# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert all(isinstance(v, bool) for v in checklist.values()), "Task 2: every gate is computed"
assert all_pass, (
    "Task 2: a gate failed — read the [FAIL] lines; if evidence is missing, "
    "run 02_drift_monitoring.py, 03_model_card.py and 04_deployment_pipeline.py first"
)
print("\n[ok] Checkpoint 1 — readiness gates all green on real evidence\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — emit the deployment config (no new model)
# ════════════════════════════════════════════════════════════════════════
# The config records what will run AND the evidence it was approved on.

deployment_config = {
    "model_name": PRODUCTION_MODEL_NAME,
    "model_version": ev["live_version"],
    "framework": "lightgbm",
    "calibration": "isotonic",
    "decision_threshold": DECISION_THRESHOLD,
    "conformal_alpha": conformal["alpha"],
    "conformal_qhat": conformal["q_hat"],
    "feature_names": feature_names,
    "drift_psi_threshold": PSI_THRESHOLD,
    "drift_ks_threshold": KS_THRESHOLD,
    "retrain_auc_pr_floor": float(metrics["auc_pr"] * 0.9),
    "metrics": metrics,
    "coverage": conformal["coverage"],
    "readiness": {item: passed for item, passed in checklist.items()},
    "human_signoff_required": review_items,
    "created": datetime.now().isoformat(),
}
config_path = OUTPUT_DIR / "ex8_05_deployment_config.json"
config_path.write_text(json.dumps(deployment_config, indent=2))
print(f"Saved: {config_path}")


# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert config_path.exists(), "Task 3: deployment_config.json must exist"
loaded = json.loads(config_path.read_text())
assert "conformal_qhat" in loaded, "Task 3: config must include conformal state"
assert "drift_psi_threshold" in loaded, "Task 3: config must include drift thresholds"
assert loaded["model_version"] is not None, "Task 3: config must name the live registry version"
print("\n[ok] Checkpoint 2 — deployment artifacts emitted\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE readiness as a traffic light + capstone metrics
# ════════════════════════════════════════════════════════════════════════

fig = go.Figure()
items = list(checklist.keys())
vals = [1 if checklist[k] else 0 for k in items]
fig.add_trace(
    go.Bar(
        y=items,
        x=[1] * len(items),
        orientation="h",
        marker_color=["#10b981" if v else "#ef4444" for v in vals],
        text=["PASS" if v else "FAIL" for v in vals],
        textposition="inside",
        insidetextanchor="middle",
        showlegend=False,
    )
)
fig.update_layout(
    title=f"Production Readiness — {PRODUCTION_MODEL_NAME} v{ev['live_version']}",
    xaxis=dict(showticklabels=False, range=[0, 1]),
    yaxis=dict(autorange="reversed"),
    height=520,
)
viz_path = OUTPUT_DIR / "ex8_05_readiness_traffic_light.html"
fig.write_html(str(viz_path))
print(f"Saved: {viz_path}")

# Final metrics panel (log loss is unbounded, so it is left off this 0-1 chart)
panel = {k: v for k, v in metrics.items() if k != "log_loss"}
fig2 = go.Figure()
fig2.add_trace(
    go.Bar(
        x=list(panel.keys()),
        y=list(panel.values()),
        marker_color="#2563eb",
        text=[f"{v:.3f}" for v in panel.values()],
        textposition="outside",
    )
)
fig2.update_layout(
    title=f"Final metrics at threshold {DECISION_THRESHOLD:.3f} — {PRODUCTION_MODEL_NAME}",
    yaxis_title="Value",
    yaxis_range=[0, 1.1],
    height=480,
)
metrics_viz_path = OUTPUT_DIR / "ex8_05_final_metrics.html"
fig2.write_html(str(metrics_viz_path))
print(f"Saved: {metrics_viz_path}")


# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert (
    viz_path.exists() and metrics_viz_path.exists()
), "Task 4: both readiness and metrics visuals must exist"
print("\n[ok] Checkpoint 3 — capstone visuals rendered\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: what the performance gates say when the data shifts
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): six months after launch a Singapore lender's
# income feed changes (a sudden +3σ shift, as in 8.2), and the outcomes
# for those applicants eventually arrive. Re-run the PERFORMANCE gates on
# that batch: the ones that flip are the ones that would have caught it.


def performance_gates(p: np.ndarray, y: np.ndarray) -> dict[str, bool]:
    m = evaluate_classification(y, p, threshold=DECISION_THRESHOLD)
    cov = conformal_on_test(y, p, alpha=0.10)["coverage"]
    base = float(y.mean() * (1 - y.mean()))
    return {
        "AUC-ROC >= 0.70": m["auc_roc"] >= 0.70,
        "Brier < base rate": m["brier"] < base,
        "Coverage >= 88%": cov >= 0.88,
        "Recall within 5 pts of launch": m["recall"] >= metrics["recall"] - 0.05,
    }


X_shift = simulate_sudden_drift(X_train, X_test, feature_names.index("income_sgd"))
p_shift = calibrated_model.predict_proba(X_shift)[:, 1]
launch_gates = performance_gates(y_proba, y_test)
shift_gates = performance_gates(p_shift, y_test)
print(f"{'Gate':<32} {'launch':>8} {'shifted':>9}")
for gate in launch_gates:
    print(
        f"  {gate:<30} {'pass' if launch_gates[gate] else 'FAIL':>8} "
        f"{'pass' if shift_gates[gate] else 'FAIL':>9}"
    )
flipped = [g for g in launch_gates if launch_gates[g] and not shift_gates[g]]
print(f"\nGates that flipped under the income shift: {flipped or 'none'}")
# INTERPRETATION: if nothing (or little) flips, the performance gates
# alone would have let the shifted feed through — which is exactly why
# 8.2 watches the inputs directly instead of waiting for outcomes.


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — km.diagnose
# ════════════════════════════════════════════════════════════════════════
# This capstone wired conformal prediction, drift monitoring, model
# cards, deployment and readiness gates from primitives. kailash-ml also
# packages a diagnostic summary (per-class metrics, class balance,
# confusion matrix) into a single call.

# `kind="classical_classifier"` dispatches to the sklearn-classifier
# adapter; the calibrated model implements that interface.
report = diagnose(
    calibrated_model,
    kind="classical_classifier",
    data=(X_test, y_test),
    show=False,
)
print()
print("  diagnose model    : capstone calibrated credit-default classifier")
print(f"  diagnose metrics  : {report.metrics}")
print(f"  diagnose severity : {report.severity}")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION — Module 3 Capstone
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  MODULE 3 CAPSTONE — SUPERVISED ML FROM THEORY TO PRODUCTION")
print("=" * 70)
print(
    f"""
  M3 CAPSTONE CHECKLIST:
  [x] Feature engineering (Ex 1): domain features, leakage prevention,
      mutual information, chi-squared, RFE, L1 selection
  [x] Bias-variance (Ex 2): regularisation, nested CV, time-series CV,
      GroupKFold, learning curves
  [x] Model zoo (Ex 3): SVM, KNN, Naive Bayes, decision trees (Gini from
      scratch), random forests, decision boundaries
  [x] Gradient boosting (Ex 4): XGBoost split gain, LightGBM, CatBoost,
      tuning on a validation split, early stopping
  [x] Imbalance + calibration (Ex 5): SMOTE, cost-sensitive learning,
      focal loss, threshold optimisation, isotonic / Platt calibration
  [x] Interpretability + fairness (Ex 6): SHAP, LIME, permutation
      importance, disparate impact, equalised odds, impossibility theorem
  [x] Workflow orchestration (Ex 7): WorkflowBuilder with branching,
      DataFlow, HyperparameterSearch, ModelRegistry

  EXERCISE 8:
  [x] 8.1 Conformal prediction: marginal coverage under exchangeability
  [x] 8.2 DriftMonitor + DataFlow: injected drift caught, checks persisted
  [x] 8.3 Model card with measured, disaggregated fairness
  [x] 8.4 ModelRegistry: gated promotion and a verified rollback drill
  [x] 8.5 Readiness: {sum(checklist.values())}/{len(checklist)} gates computed from evidence,
      {len(review_items)} item(s) routed to a human

  THE PRODUCTION PIPELINE PATTERN:
    preprocess → train → calibrate → conformal predict →
    document → register → gate → promote → monitor drift

  KEY INSIGHT: Production ML is mostly engineering discipline. The
  model is the easy part; the gates, the card, the coverage guarantee
  and the registry trail are what make it trustworthy — and only if
  every one of them is measured rather than asserted.

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MODULE 4 PREVIEW: UNSUPERVISED ML AND DEEP LEARNING FOUNDATIONS
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Module 3 worked with labelled data — every row had a known outcome.
  Module 4 removes that luxury:
    - Clustering (Ex 1) and EM / Gaussian mixtures (Ex 2)
    - Dimensionality reduction: PCA, kernel PCA, t-SNE, UMAP (Ex 3)
    - Anomaly detection and ensembles (Ex 4)
    - Association rules: Apriori, FP-Growth (Ex 5)
    - NLP: text to topics (Ex 6)
    - Recommender systems (Ex 7)
    - Deep learning foundations: neural networks and backpropagation (Ex 8)
"""
)
print("=" * 70)
