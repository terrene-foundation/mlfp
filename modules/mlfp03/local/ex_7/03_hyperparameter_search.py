# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 7.3: Bayesian Hyperparameter Search
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Declare a search space with ParamDistribution
#     (int_uniform / log_uniform / uniform)
#   - Configure a Bayesian search run with SearchConfig
#   - Drive the search through kailash-ml's HyperparameterSearch engine,
#     wired on top of TrainingPipeline (no raw .fit() in user code)
#   - Compare Bayesian search against a small grid FAIRLY: same
#     validation rows and same metric for selection, one untouched test
#     set for the verdict
#   - Turn a measured test-set difference into a business number
#
# PREREQUISITES: 01_workflow_builder.py, 02_dataflow_persistence.py
# ESTIMATED TIME: ~45 min
#
# 5-PHASE R10:
#   1. Theory     — why Bayesian search beats grid search (usually)
#   2. Build      — SearchSpace + SearchConfig + TrainingPipeline base
#   3. Train      — async .search() and the grid baseline on the dev frame
#   4. Visualise  — search trajectory vs grid, then the test-set verdict
#   5. Apply      — defaults caught at a fixed review budget, in S$
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import plotly.graph_objects as go

from kailash_ml import HyperparameterSearch, TrainingPipeline
from kailash_ml.engines.hyperparameter_search import (
    ParamDistribution,
    SearchConfig,
    SearchSpace,
)
from kailash_ml.engines.training_pipeline import EvalSpec, ModelSpec
from kailash_ml.interop import to_sklearn_input

from shared.mlfp03.ex_7 import (
    ENGINE_METRICS,
    ILLUSTRATIVE_BANK,
    OUTPUT_DIR,
    RANDOM_SEED,
    TARGET_COLUMN,
    annual_loss_avoided,
    build_training_registry,
    compute_classification_metrics,
    credit_feature_schema,
    defaults_caught_at_budget,
    headline_roi_text,
    load_registered_model,
    prepare_credit_frames,
    print_metric_block,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Bayesian vs grid search
# ════════════════════════════════════════════════════════════════════════
# Grid search is the "brute-force every combination" approach. A 5x5x5
# grid is 125 trials at 3 hyperparameters; every extra dimension
# multiplies the cost, and the grid only ever tries the values you
# listed.
#
# Bayesian optimisation is smarter about where to look next:
#   1. Fit a cheap surrogate model to the (hyperparameters -> score)
#      pairs seen so far.
#   2. Use it to pick the next candidate, balancing EXPLORATION (regions
#      the surrogate is unsure about) and EXPLOITATION (regions that
#      already look good).
#   3. Evaluate that candidate, add it to the history, repeat.
#
# kailash-ml's strategy="bayesian" runs Optuna's TPE sampler (a
# Tree-structured Parzen Estimator — it models good vs bad regions as two
# densities instead of fitting a Gaussian process). Each trial is a real
# TrainingPipeline.train() call, so no raw .fit() appears in user code.
#
# THE FAIRNESS RULE for comparing search strategies:
#   - SELECTION happens on validation data only. TrainingPipeline's
#     holdout split is seeded, so every trial AND every grid point is
#     scored on the very same validation rows of the dev frame.
#   - The VERDICT happens once, on a test frame nobody selected on.
# Comparing a search's best validation score with a grid's CV score — or
# letting the search see the reporting rows — measures nothing.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the search space, the grid and the configuration
# ════════════════════════════════════════════════════════════════════════
# ParamDistribution type strings:
#   - "int_uniform"  integer bounds
#   - "log_uniform"  floats sampled on a log scale (good for learning rates)
#   - "uniform"      floats sampled linearly
#   - "categorical"  discrete choices (pass `choices=[...]`)

search_space = SearchSpace(
    params=[
        ParamDistribution("n_estimators", "int_uniform", low=100, high=800),
        # TODO: learning rate between 0.01 and 0.3, sampled on a LOG scale
        # Hint: ParamDistribution(<name>, <type string>, low=..., high=...)
        ____,
        ParamDistribution("max_depth", "int_uniform", low=3, high=10),
        ParamDistribution("num_leaves", "int_uniform", low=15, high=127),
        ParamDistribution("min_child_samples", "int_uniform", low=5, high=100),
    ]
)

# SearchConfig fields:
#   - strategy="bayesian" picks the Optuna TPE backend
#   - n_trials caps the search budget
#   - metric_to_optimize must be a key TrainingPipeline emits in
#     result.metrics — "auc" (ROC) is in ENGINE_METRICS
#   - register_best=False: we register the winner explicitly below
search_config = SearchConfig(
    # TODO: choose the strategy that runs Optuna's TPE sampler
    # Hint: SearchConfig strategies include "grid", "random", "bayesian"
    strategy=____,
    n_trials=20,
    metric_to_optimize="auc",
    direction="maximize",
    register_best=False,
)

# The baseline a team would hand-tune: four sensible grid points.
GRID: list[dict[str, Any]] = [
    {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5},
    {"n_estimators": 500, "learning_rate": 0.10, "max_depth": 6},
    {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 7},
    {"n_estimators": 700, "learning_rate": 0.03, "max_depth": 8},
]

FIXED_PARAMS: dict[str, Any] = {"random_state": RANDOM_SEED, "verbose": -1, "n_jobs": 8}
EVAL_SPEC = EvalSpec(metrics=ENGINE_METRICS, split_strategy="holdout", test_size=0.2)

dev, test, feature_cols = prepare_credit_frames()
schema = credit_feature_schema(feature_cols)
print(f"\nDev frame: {dev.height:,} rows (search + grid)   Test frame: {test.height:,} rows")


def lgbm_spec(params: dict[str, Any]) -> ModelSpec:
    """LightGBM ModelSpec with the fixed reproducibility params merged in."""
    return ModelSpec(
        model_class="lightgbm.LGBMClassifier",
        framework="lightgbm",
        hyperparameters={**FIXED_PARAMS, **params},
    )


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Bayesian search and grid baseline on the dev frame,
#                 then score both winners once on the test frame
# ════════════════════════════════════════════════════════════════════════


async def run_search_and_grid() -> dict[str, Any]:
    registry, conn = await build_training_registry()
    try:
        pipeline = TrainingPipeline(feature_store=None, registry=registry)

        # 1) Bayesian search — 20 TrainingPipeline.train() calls on `dev`
        # TODO: wrap the TrainingPipeline in the search engine
        # Hint: HyperparameterSearch(pipeline=...)
        searcher = ____
        # TODO: launch the search (it is async)
        # Hint: the engine method that takes data, schema, base_model_spec,
        # search_space, config, eval_spec and experiment_name
        result = await searcher.____(
            data=dev,
            schema=schema,
            base_model_spec=lgbm_spec({}),
            search_space=search_space,
            config=search_config,
            eval_spec=EVAL_SPEC,
            experiment_name="credit_default_hp_search",
        )

        # 2) Grid baseline — same engine, same dev frame, same validation rows
        grid_scores: list[float] = []
        for i, params in enumerate(GRID):
            g = await pipeline.train(
                data=dev,
                schema=schema,
                model_spec=lgbm_spec(params),
                eval_spec=EVAL_SPEC,
                experiment_name=f"credit_default_grid_{i}",
            )
            # TODO: record this grid point's validation AUC
            # Hint: TrainingResult.metrics uses the ENGINE_METRICS names
            grid_scores.append(____)
        # TODO: the grid configuration with the highest validation AUC
        # Hint: np.argmax over grid_scores indexes into GRID
        best_grid = ____

        # 3) Re-train each winner once (registered), score on the test frame
        X_test, y_test, _ = to_sklearn_input(
            test, feature_columns=feature_cols, target_column=TARGET_COLUMN
        )
        test_proba: dict[str, np.ndarray] = {}
        for label, params in (("bayesian", result.best_params), ("grid", best_grid)):
            final = await pipeline.train(
                data=dev,
                schema=schema,
                model_spec=lgbm_spec(dict(params)),
                eval_spec=EVAL_SPEC,
                experiment_name=f"credit_default_{label}_winner",
            )
            if final.model_version is None:
                raise RuntimeError(f"{label} winner was not registered")
            model = await load_registered_model(
                registry, f"credit_default_{label}_winner", final.model_version.version
            )
            test_proba[label] = model.predict_proba(X_test)[:, 1]
    finally:
        await conn.close()

    return {
        "best_params": dict(result.best_params),
        "best_val_auc": float(result.best_metrics["auc"]),
        "trials": list(result.all_trials),
        "grid_scores": grid_scores,
        "best_grid": best_grid,
        "y_test": np.asarray(y_test),
        "test_proba": test_proba,
    }


print("\n" + "=" * 70)
print("  Bayesian search (20 trials) vs 4-point grid — validation AUC on dev")
print("=" * 70)
out = asyncio.run(run_search_and_grid())

y_test = out["y_test"]
test_metrics = {
    label: compute_classification_metrics(y_test, (p >= 0.5).astype(int), p)
    for label, p in out["test_proba"].items()
}
best_grid_val = max(out["grid_scores"])
print(f"\n  Bayesian best validation AUC: {out['best_val_auc']:.4f}")
print(f"  Grid best validation AUC:     {best_grid_val:.4f}")
print(f"  Bayesian best params: {out['best_params']}")
print(f"  Grid best params:     {out['best_grid']}")

# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(out["trials"]) == search_config.n_trials, (
    f"Task 3: expected {search_config.n_trials} trials, got {len(out['trials'])}"
)
assert 0.5 < out["best_val_auc"] <= 1.0, "Task 3: search should beat random"
assert len(out["grid_scores"]) == len(GRID), "Task 3: every grid point scored"
for label, m in test_metrics.items():
    assert 0.5 < m["auc_roc"] <= 1.0, f"Task 3: {label} winner must beat random on test"
print("\n[ok] Checkpoint passed — search, grid and test verdict complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the search trajectory, then the test-set verdict
# ════════════════════════════════════════════════════════════════════════
# all_trials is a list[TrialResult] with .trial_number, .params, .metrics
# and .training_time_seconds.

trials = sorted(out["trials"], key=lambda t: t.trial_number)
trial_auc = [float(t.metrics.get("auc", np.nan)) for t in trials]
running_best = np.maximum.accumulate(trial_auc)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=[t.trial_number for t in trials],
        y=trial_auc,
        mode="markers",
        name="Bayesian trial (validation AUC)",
    )
)
fig.add_trace(
    go.Scatter(
        x=[t.trial_number for t in trials],
        y=running_best,
        mode="lines",
        name="Bayesian best so far",
    )
)
fig.add_hline(
    y=best_grid_val,
    line_dash="dash",
    annotation_text=f"best of {len(GRID)} grid points",
)
fig.update_layout(
    title="Bayesian search vs grid — same validation rows of the dev frame",
    xaxis_title="Trial number",
    yaxis_title="Validation AUC-ROC",
)
plot_path = OUTPUT_DIR / "ex7_03_search_trajectory.html"
fig.write_html(str(plot_path))
print(f"Saved: {plot_path}")

top5 = sorted(trials, key=lambda t: t.metrics.get("auc", 0.0), reverse=True)[:5]
print("\nTop 5 Bayesian trials (validation AUC):")
for rank, t in enumerate(top5, 1):
    print(f"  {rank}. trial #{t.trial_number}  AUC={t.metrics['auc']:.4f}  {t.params}")

budget = ILLUSTRATIVE_BANK["review_budget"]
caught = {
    # TODO: defaults among the top `budget` share of test rows by score
    # Hint: the shared defaults_caught_at_budget(y_true, y_proba, budget)
    label: ____
    for label, p in out["test_proba"].items()
}
print(
    f"\nTest-set verdict ({len(y_test):,} applications, "
    f"{int(y_test.sum()):,} defaults):"
)
print(f"  {'winner':<10} {'AUC-ROC':>8} {'AUC-PR':>8} {'defaults in top slice':>22}")
for label in ("bayesian", "grid"):
    m = test_metrics[label]
    print(f"  {label:<10} {m['auc_roc']:>8.4f} {m['auc_pr']:>8.4f} {caught[label]:>22,}")
for label in ("bayesian", "grid"):
    print_metric_block(f"{label} winner — test frame", test_metrics[label])

val_gap = out["best_val_auc"] - best_grid_val
test_gap = test_metrics["bayesian"]["auc_roc"] - test_metrics["grid"]["auc_roc"]
print(
    f"\n  Validation edge of the search: {val_gap:+.4f} AUC"
    f"\n  Test edge of the search:       {test_gap:+.4f} AUC"
)
# INTERPRETATION: the search's best validation score is the maximum of
# 20 noisy draws, the grid's of only 4, so part of any validation edge
# is selection luck. The test row is the honest comparison.
if test_gap > 0 and val_gap > test_gap:
    print("  The search kept part of its edge on test; the rest was selection luck.")
elif test_gap > 0:
    print("  The search's edge held up on unseen data.")
else:
    print(
        "  On unseen data the search did NOT beat the hand-picked grid — the"
        "\n  extra trials bought nothing on this dataset this time."
    )


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: what the search is worth to the credit team
# ════════════════════════════════════════════════════════════════════════
# A credit team can only review a fixed share of applications (here the
# illustrative 10% budget). The model that puts more true defaults into
# that reviewed slice saves money. We scale the MEASURED test-set
# difference to a year of applications using the ILLUSTRATIVE
# assumptions in shared.mlfp03.ex_7 (exposure, LGD, volume).

extra_caught = caught["bayesian"] - caught["grid"]
loss_avoided = annual_loss_avoided(extra_caught, len(y_test))
print("\n" + "=" * 70)
print("  APPLY: Bayesian Search Lift = Defaults Caught at a Fixed Budget")
print("=" * 70)
print(
    f"  Defaults caught in the top {budget:.0%}: Bayesian {caught['bayesian']:,} "
    f"vs grid {caught['grid']:,} ({extra_caught:+,} on {len(y_test):,} applications)"
)
print(headline_roi_text(loss_avoided_sgd=loss_avoided))
if extra_caught <= 0:
    print("  -> No measured gain: keep the cheaper grid model and save the compute.")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Declared a 5-dimensional SearchSpace with int_uniform / log_uniform
  [x] Ran SearchConfig(strategy="bayesian") through HyperparameterSearch
      on top of TrainingPipeline — every trial is an engine train() call
  [x] Compared search and grid on the SAME validation rows and metric
  [x] Judged both winners once on an untouched test frame:
      test AUC edge of the search = {test_gap:+.4f}
  [x] Turned the measured difference into defaults caught and S$

  KEY INSIGHT: a search is only "better" if its edge survives on data it
  never selected on. Best-of-20 always looks better than best-of-4 on the
  validation rows it was chosen on.

  Next: 04_model_registry.py — promote the winning model through the
  ModelRegistry lifecycle so production can safely pick it up.
"""
)
