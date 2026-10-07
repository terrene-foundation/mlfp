# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP03 Exercise 7 — Kailash Workflows, DataFlow
Persistence, Hyperparameter Search, and Model Registry.

Contains: leak-free dataset loading, fixed dev/test frames, FeatureSchema
construction, registry setup and artefact loading, metric computation, the
production quality gate, illustrative ROI helpers, DB URL resolution and
pipeline-audit utilities. Technique-specific code (workflow node wiring, search space
definitions, registry lifecycle transitions) lives in the per-technique files.

Available after ``uv sync`` from any directory.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)

from kailash_ml.types import FeatureField, FeatureSchema

from shared import MLFPDataLoader
from shared.kailash_helpers import setup_environment, split_then_preprocess

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()
load_dotenv()

RANDOM_SEED: int = 42
TARGET_COLUMN: str = "default"
DATASET_NAME: str = "sg_credit_scoring"
DATASET_FILE: str = "sg_credit_scoring.parquet"

# Columns that MUST NOT be model inputs (Lesson 3.1 leakage rule):
#   customer_id              — a row identifier, not a property of the applicant
#   future_default_indicator — recorded AFTER the loan outcome is known; it
#                              agrees with ``default`` on ~99% of rows, so a
#                              model that sees it "predicts" default by
#                              reading the answer (Exercise 4 screens for it).
CREDIT_NON_FEATURE_COLUMNS: tuple[str, ...] = ("customer_id", "future_default_indicator")

# Output directory for artefacts (audit trails, evaluation tables)
OUTPUT_DIR = Path("outputs") / "mlfp03_ex7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# DataFlow persistence URL. SQLite is hermetic — every student gets a fresh
# DB file per run. In production we would read this from the environment.
#
# NOTE: The modern dataflow sqlite adapter interprets `sqlite:///relative`
# as an absolute `/relative` path (breaking old sqlite URL conventions).
# We therefore resolve to an absolute path and use the `sqlite:////abs`
# four-slash form so the behaviour is identical on every working dir.
_DB_ABS_PATH = (OUTPUT_DIR / "mlfp03_models.db").resolve()
DB_URL: str = os.environ.get(
    "MLFP03_EX7_DB_URL", f"sqlite:///{_DB_ABS_PATH.as_posix()}"
)


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — Singapore credit scoring (from MLFP02)
# ════════════════════════════════════════════════════════════════════════


def load_credit_frame() -> pl.DataFrame:
    """Load the Singapore credit scoring dataset as a polars DataFrame.

    Columns: demographic + bureau features, with ``default`` (0/1) target.
    ``CREDIT_NON_FEATURE_COLUMNS`` (row ID + post-outcome leak) are dropped
    here so every downstream split, schema and pipeline inherits the fix.
    """
    loader = MLFPDataLoader()
    return loader.load("mlfp02", DATASET_FILE).drop(CREDIT_NON_FEATURE_COLUMNS)


def prepare_credit_frames(
    credit: pl.DataFrame | None = None, *, seed: int = RANDOM_SEED
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Return ``(dev_frame, test_frame, feature_columns)`` for honest model selection.

    ``dev_frame`` (80%) is everything model SELECTION may touch —
    hyperparameter search, grid baselines, early stopping; TrainingPipeline
    carves its own validation holdout out of it. ``test_frame`` (20%) is
    touched exactly once, to report the chosen model. The 20% is held out
    BEFORE the preprocessing is fitted, so its rows shape no imputation or
    encoding statistic. Both carry a unique
    ``application_id`` so they satisfy the same ``FeatureSchema``.
    """
    if credit is None:
        credit = load_credit_frame()

    # Split FIRST, then fit imputation/encoding on the training rows only:
    # PreprocessingPipeline.setup() on the whole frame would fit them on the
    # test rows too (it splits only after fitting).
    result = split_then_preprocess(
        credit,
        target=TARGET_COLUMN,
        test_size=0.2,
        seed=seed,
        normalize=False,
        categorical_encoding="ordinal",
    )
    n_dev = result.train_data.height
    dev = result.train_data.with_columns(
        pl.int_range(0, n_dev, dtype=pl.Int64).alias("application_id")
    )
    test = result.test_data.with_columns(
        pl.int_range(n_dev, n_dev + result.test_data.height, dtype=pl.Int64).alias(
            "application_id"
        )
    )
    feature_columns = [
        c for c in dev.columns if c not in (TARGET_COLUMN, "application_id")
    ]
    return dev, test, feature_columns


def credit_feature_schema(feature_columns: list[str]) -> FeatureSchema:
    """Build a FeatureSchema matching ``prepare_credit_frames`` output."""
    return FeatureSchema(
        name="credit_model_input",
        features=[FeatureField(name=f, dtype="float64") for f in feature_columns],
        entity_id_column="application_id",
    )


async def build_training_registry(db_url: str | None = None):
    """Create + initialise a kailash-ml ModelRegistry for TrainingPipeline.

    Returns ``(registry, connection_manager)``. Caller owns ``connection_manager.close()``.
    """
    from kailash.db import ConnectionManager
    from kailash_ml import ModelRegistry

    conn = ConnectionManager(db_url or DB_URL)
    await conn.initialize()
    registry = ModelRegistry(conn)
    return registry, conn


# ════════════════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════════════════


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    """Return accuracy, f1, auc_roc, auc_pr, log_loss."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "auc_pr": float(average_precision_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
    }


def print_metric_block(title: str, metrics: dict[str, float]) -> None:
    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


# ════════════════════════════════════════════════════════════════════════
# QUALITY GATE — what a candidate model must clear before production
# ════════════════════════════════════════════════════════════════════════
# Both thresholds are relative to what "no skill" looks like on THIS data:
#   - AUC-ROC 0.5 is random ranking; we demand a clear margin above it.
#   - AUC-PR (average precision) of a random scorer equals the default
#     rate (~13% here); we demand at least twice that.
# Leak-free LightGBM on this dataset lands around AUC 0.77 / AUC-PR 0.37,
# so a sound model passes and a broken one (e.g. shuffled labels) fails.

PROMOTION_MIN_AUC: float = 0.75
PROMOTION_MIN_AP_MULTIPLE: float = 2.0  # × default rate


def promotion_gate(metrics: dict[str, float], default_rate: float) -> dict[str, Any]:
    """Evaluate the production quality gate.

    ``metrics`` uses ``compute_classification_metrics`` names
    (``auc_roc``, ``auc_pr``) computed on rows the model never trained on.
    Returns a JSON-serialisable dict: thresholds, observed values, verdict.
    """
    min_ap = PROMOTION_MIN_AP_MULTIPLE * default_rate
    auc = float(metrics["auc_roc"])
    ap = float(metrics["auc_pr"])
    return {
        "auc_roc": auc,
        "auc_pr": ap,
        "min_auc_roc": PROMOTION_MIN_AUC,
        "min_auc_pr": min_ap,
        "promote": bool(auc >= PROMOTION_MIN_AUC and ap >= min_ap),
    }


async def load_registered_model(registry: Any, name: str, version: int) -> Any:
    """Load a model artefact that THIS course's pipeline registered.

    The registry stores the fitted estimator as pickle bytes. Unpickling
    executes code, so only ever load artefacts you trained yourself.
    """
    import pickle

    artifact = await registry.load_artifact(name, version)
    return pickle.loads(artifact)


# Metrics TrainingPipeline can compute from its holdout predictions. In
# kailash-ml 2.2.2 the engine's evaluator does not pass predicted
# probabilities to the metric registry, so "average_precision",
# "log_loss" and "brier_score_loss" are skipped there; the exercises
# compute those with ``compute_classification_metrics`` on the
# registered model's probabilities instead.
ENGINE_METRICS: list[str] = ["accuracy", "f1", "auc"]


# ════════════════════════════════════════════════════════════════════════
# APPLY-PHASE BUSINESS ASSUMPTIONS (R9B) — ILLUSTRATIVE
# ════════════════════════════════════════════════════════════════════════
# Round numbers for a HYPOTHETICAL Singapore retail bank's unsecured
# lending book. They are teaching assumptions, not figures from any real
# lender, regulator or report — replace them with your institution's own
# numbers. Only the model-quality inputs (defaults caught, metrics) are
# measured by the exercise code itself.

ILLUSTRATIVE_BANK: dict[str, Any] = {
    "applications_per_year": 200_000,
    "avg_exposure_sgd": 18_000.0,  # average principal per approved loan
    "lgd": 0.65,  # loss given default on unsecured retail
    "review_budget": 0.10,  # share of applications the credit team can review
    "evidence_hours_per_retrain": 160.0,  # manual evidence-pack assembly
    "analyst_hourly_sgd": 120.0,
    "retrains_per_year": 12,
}


def defaults_caught_at_budget(
    y_true: np.ndarray, y_proba: np.ndarray, budget: float | None = None
) -> int:
    """Defaults among the top ``budget`` share of applications by model score.

    This is how a credit team actually uses a score: it can only review a
    fixed share of applications, so a better model is one that puts more
    true defaults into that reviewed slice.
    """
    budget = ILLUSTRATIVE_BANK["review_budget"] if budget is None else budget
    k = max(1, int(round(budget * len(y_true))))
    top = np.argsort(-np.asarray(y_proba))[:k]
    return int(np.asarray(y_true)[top].sum())


def annual_loss_avoided(extra_defaults_caught: float, n_scored: int) -> float:
    """Scale extra defaults caught on ``n_scored`` test rows to a year (S$)."""
    b = ILLUSTRATIVE_BANK
    per_year = extra_defaults_caught * b["applications_per_year"] / n_scored
    return float(per_year * b["avg_exposure_sgd"] * b["lgd"])


def evidence_prep_savings() -> float:
    """Annual analyst cost of hand-assembling model evidence packs (S$)."""
    b = ILLUSTRATIVE_BANK
    return float(
        b["evidence_hours_per_retrain"] * b["analyst_hourly_sgd"] * b["retrains_per_year"]
    )


def headline_roi_text(loss_avoided_sgd: float | None = None) -> str:
    """Plain-text ROI block for the Apply phases (ILLUSTRATIVE assumptions).

    ``loss_avoided_sgd`` must come from a measured comparison (see
    03_hyperparameter_search.py); when it is not supplied the line says so
    instead of inventing a number.
    """
    b = ILLUSTRATIVE_BANK
    audit = evidence_prep_savings()
    lines = [
        "  (illustrative assumptions for a hypothetical Singapore retail bank)",
        f"  Applications scored:  {b['applications_per_year']:,} / yr",
        f"  Retrains:             {b['retrains_per_year']} / yr",
        f"  Evidence-pack prep:   ~S${audit/1e3:,.0f}k / yr of analyst time that a",
        "                        persisted, replayable pipeline removes",
    ]
    if loss_avoided_sgd is None:
        lines.append(
            "  Loss avoided:         not claimed here — measured in "
            "03_hyperparameter_search.py"
        )
    else:
        lines.append(
            f"  Loss avoided:         ~S${loss_avoided_sgd/1e6:,.2f}M / yr "
            "(measured lift × assumptions)"
        )
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# PIPELINE AUDIT HELPERS
# ════════════════════════════════════════════════════════════════════════


def audit_trail_row(
    *,
    stage: str,
    detail: str,
    run_id: str,
) -> dict[str, Any]:
    """Structured audit row used by the orchestrated pipeline."""
    return {"stage": stage, "detail": detail, "run_id": run_id}
