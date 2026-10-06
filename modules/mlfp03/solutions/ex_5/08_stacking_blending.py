# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 5.8: Stacking and Blending — Combining Models
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Why three weak-but-DIFFERENT models beat one strong model
#   - Stacking: a meta-learner trained on out-of-fold base predictions
#   - Blending: a (weighted) vote over base predictions
#   - When the ensemble win is real and when it is complexity theatre
#   - kailash-ml's EnsembleEngine.stack / .blend as the production path
#
# PREREQUISITES: Exercise 5.1–5.7 (metrics taxonomy, calibration)
# ESTIMATED TIME: ~30 min
#
# 5-PHASE STRUCTURE:
#   Theory   — diversity beats strength; stack vs blend
#   Build    — three diverse base models on the credit dataset
#   Train    — EnsembleEngine.stack and .blend
#   Visualise — every model + ensemble, one AUC/log-loss panel
#   Apply    — a lender's model-inventory committee decision
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from kailash_ml import EnsembleEngine
from shared import MLFPDataLoader
from shared.kailash_helpers import split_then_preprocess
from shared.mlfp03.ex_5 import CREDIT_NON_FEATURE_COLUMNS, OUTPUT_DIR

# ── THEORY — diversity beats strength ────────────────────────────────────
# One strong model makes ONE kind of mistake everywhere it is wrong. Three
# different models (a linear one, a bagged one, a boosted one) make
# DIFFERENT mistakes — so a combination can cancel them. The catch: the
# errors must be genuinely different. Three gradient-boosted forests with
# different seeds are correlated and stack to nothing.
#
#   BLENDING — hold out a slice, fit each base model on the rest, average
#   the probabilities (optionally weighted). Simple, fast, and safe: the
#   combiner is just a weighted mean, so it cannot overfit much. This is
#   VotingClassifier with soft voting.
#
#   STACKING — the combiner is itself a MODEL (usually logistic regression)
#   trained on OUT-OF-FOLD base predictions: each training row's feature for
#   the meta-learner is the prediction a base model made on a fold that did
#   NOT contain that row. This is what stops the meta-learner from simply
#   copying the most overfit base model.
#
# Rule of thumb: blending first (nearly free), stacking when the base models
# disagree a lot and you have the rows to afford the extra fit.

loader = MLFPDataLoader()
credit = loader.load("mlfp02", "sg_credit_scoring.parquet").drop(
    CREDIT_NON_FEATURE_COLUMNS
)

# Same leak-free preprocessing discipline as the rest of ex_5 — split FIRST,
# fit the pipeline on training rows only — with two deltas: keep the polars
# frames (EnsembleEngine takes a frame with a named target column and does
# its own internal holdout), and NORMALIZE — the logistic base model cannot
# converge on raw ordinal scales, and the trees do not care either way.
result = split_then_preprocess(
    credit,
    target="default",
    test_size=0.2,
    seed=42,
    normalize=True,
    categorical_encoding="ordinal",
)
train_frame, test_frame = result.train_data, result.test_data
feature_cols = [c for c in train_frame.columns if c != "default"]
X_test = test_frame.select(feature_cols).to_numpy()
y_test = test_frame["default"].to_numpy()

# ── BUILD — three models with three different inductive biases ───────────
bases = {
    "logistic (linear)": LogisticRegression(max_iter=2000, random_state=42),
    "random forest (bagging)": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=8
    ),
    "lightgbm (boosting)": lgb.LGBMClassifier(
        n_estimators=300, random_state=42, verbose=-1
    ),
}

X_train = train_frame.select(feature_cols).to_numpy()
y_train = train_frame["default"].to_numpy()

print("  Base models (fitted on the shared leak-free train split):")
fitted = {}
base_rows = []
for name, model in bases.items():
    model.fit(X_train, y_train)
    fitted[name] = model
    p = model.predict_proba(X_test)[:, 1]
    base_rows.append(
        {"model": name, "auc": roc_auc_score(y_test, p), "logloss": log_loss(y_test, p)}
    )
    print(f"    {name:<26} AUC {base_rows[-1]['auc']:.4f}  log loss {base_rows[-1]['logloss']:.4f}")

# ── TRAIN — EnsembleEngine stack + blend ─────────────────────────────────
# PRODUCTION GOTCHA the engine cannot save you from: it evaluates each PASSED
# model on its internal holdout (component_contributions). If the models you
# pass were fitted on the SAME frame you hand in, those contribution metrics
# are contaminated — a random forest that memorised its training rows shows a
# perfect 1.0 AUC there. The ENSEMBLE metrics are clean (sklearn clones and
# refits every base model on the internal split). Below we therefore read
# per-base performance from the honest shared test split, and only the
# weights from the engine's contributions.
engine = EnsembleEngine()

stack_res = engine.stack(
    list(fitted.values()), train_frame, "default", fold=5, seed=42
)
blend_res = engine.blend(list(fitted.values()), train_frame, "default", seed=42)

p_stack = stack_res.ensemble_model.predict_proba(X_test)[:, 1]
p_blend = blend_res.ensemble_model.predict_proba(X_test)[:, 1]
ens_rows = [
    {"model": "STACK (meta-learner)", "auc": roc_auc_score(y_test, p_stack), "logloss": log_loss(y_test, p_stack)},
    {"model": "BLEND (soft vote)", "auc": roc_auc_score(y_test, p_blend), "logloss": log_loss(y_test, p_blend)},
]
for r in ens_rows:
    print(f"    {r['model']:<26} AUC {r['auc']:.4f}  log loss {r['logloss']:.4f}")

all_rows = base_rows + ens_rows

# ── Checkpoint 1 ──────────────────────────────────────────────────────────
best_base = max(r["auc"] for r in base_rows)
best_ens = max(r["auc"] for r in ens_rows)
assert stack_res.metrics and blend_res.metrics, "engine must return eval metrics"
assert len(stack_res.component_contributions) == 3, "one contribution row per base"
assert best_ens >= best_base - 0.01, (
    "ensembles must be at least competitive with the best base model"
)
print("[ok] Checkpoint 1 — stack + blend fitted; ensembles competitive with best base")

# ── VISUALISE — every model on one panel + blend weights ─────────────────
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=[r["model"] for r in all_rows],
        y=[r["auc"] for r in all_rows],
        marker_color=["#64748B"] * 3 + ["#6366F1", "#0D9488"],
        text=[f"{r['auc']:.4f}" for r in all_rows],
        textposition="auto",
        name="AUC",
    )
)
fig.update_layout(
    title="Diversity beats strength: two ensembles vs three base models (held-out AUC)",
    yaxis_title="AUC-ROC (held-out 20%)",
    yaxis_range=[0.5, 1.0],
)
fig.write_html(str(OUTPUT_DIR / "ex5_08_stacking_blending.html"))
print(f"  Saved: {OUTPUT_DIR / 'ex5_08_stacking_blending.html'}")

print("\n  Blend weights (equal by default) vs honest shared-test AUC:")
for c, (name, model) in zip(blend_res.component_contributions, fitted.items()):
    honest_auc = next(r["auc"] for r in base_rows if r["model"] == name)
    engine_auc = c["metrics"]["auc"]
    flag = "  ← contaminated (trained on this frame)" if engine_auc > honest_auc + 0.05 else ""
    print(
        f"    {name:<26} weight {c['weight']:.2f}  "
        f"engine-eval AUC {engine_auc:.4f}  shared-test AUC {honest_auc:.4f}{flag}"
    )
print("[ok] Checkpoint 2 — comparison panel + blend weights saved")

# ── APPLY — a lender's model-inventory committee ─────────────────────────
gain = best_ens - best_base
verdict = (
    f"stacking {'gained' if gain >= 0 else 'LOST'} {abs(gain):.4f} AUC versus "
    f"the best single model"
)
print(f"\n  APPLY: {verdict} — measured, not assumed.")
print(
    "  A lender's model-risk committee reads that number against the cost of "
    "operating FOUR models in production (monitoring, drift, retraining, "
    "explainability sign-off). If the gain is under ~0.01 AUC, the honest "
    "answer is usually: keep the best base model, revisit blending when the "
    "inventory genuinely disagrees. Ensemble FIRST when the base models are "
    "diverse and the decision is expensive (credit limits, fraud interdiction); "
    "ensemble LAST when the gain is cosmetic."
)

# REFLECTION
print(
    """
  What you've mastered:
    ✓ Diversity is the requirement — three models with different inductive
      biases, not three seeds of the same one
    ✓ Stacking trains a meta-learner on OUT-OF-FOLD predictions; blending
      is a (weighted) soft vote — EnsembleEngine.stack / .blend do both
    ✓ Measuring the ensemble against the best base on the SAME held-out
      split — and what to do when the gain is cosmetic
    ✓ Where ensembles sit in the taxonomy: they move AUC and log loss,
      not the threshold economics of 5.4

  This is the preview of M4's EnsembleEngine coverage — there you'll take
  ensembles into AutoML and drift monitoring.
"""
)
