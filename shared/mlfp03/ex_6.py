# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP03 Exercise 6 — Interpretability and Fairness.

Contains: Singapore credit scoring data load, LightGBM model training
(via kailash-ml TrainingPipeline),
TreeSHAP explainer setup, output directory, and common helper utilities.

Technique-specific code (permutation importance loops, LIME wrappers,
fairness audit reports) lives in the per-technique files under
`modules/mlfp03/solutions/ex_6/`.

Import pattern (solutions and local both):

    from shared.mlfp03.ex_6 import (
        FEATURE_NAMES,
        OUTPUT_DIR,
        load_credit_scoring,
        train_credit_model,
        build_shap_explainer,
    )
"""
from __future__ import annotations

import asyncio
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

import shap
from sklearn.metrics import roc_auc_score

from kailash_ml.interop import to_sklearn_input

from shared.data_loader import MLFPDataLoader
from shared.kailash_helpers import split_then_preprocess


# ════════════════════════════════════════════════════════════════════════
# PATHS / CONSTANTS
# ════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path("outputs") / "mlfp03_ex6_interpretability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Singapore credit scoring: this synthetic dataset simulates a retail-bank
# default prediction task used throughout MLFP02/MLFP03. Banks are expected
# to be able to explain credit decisions (e.g. under the MAS FEAT
# principles, which are non-binding guidance), which is why Exercise 6
# explains this model.
DATASET_MODULE = "mlfp02"
DATASET_FILE = "sg_credit_scoring.parquet"
TARGET_COLUMN = "default"
RANDOM_SEED = 42

# Columns that MUST NOT be model inputs (Lesson 3.1 leakage rule):
#   customer_id              — a row identifier, not a property of the applicant
#   future_default_indicator — recorded AFTER the loan outcome is known; it
#                              agrees with ``default`` on ~99% of rows, so a
#                              model that sees it "predicts" default by
#                              reading the answer (Exercise 4 screens for it).
CREDIT_NON_FEATURE_COLUMNS: tuple[str, ...] = ("customer_id", "future_default_indicator")

# Protected attributes audited in 05_fairness_audit.py. These are the
# REAL column names in sg_credit_scoring.parquet (race, not "ethnicity").
# Categorical ones are ordinal-encoded by PreprocessingPipeline and are
# decoded back to labels with ``decode_group``; age is banded.
PROTECTED_ATTRIBUTES: list[str] = ["race", "gender", "age"]
AGE_BANDS: list[tuple[str, int, int]] = [
    ("21-34", 21, 34),
    ("35-49", 35, 49),
    ("50-64", 50, 64),
    ("65+", 65, 200),
]


# ════════════════════════════════════════════════════════════════════════
# DATA LOAD + MODEL TRAIN
# ════════════════════════════════════════════════════════════════════════

# Populated on first call so every technique file sees the same split.
_CACHE: dict[str, Any] = {}


def load_credit_scoring() -> dict[str, Any]:
    """Load the Singapore credit scoring dataset and run the M3 preprocessing
    pipeline. Returns a dict with X_train, y_train, X_test, y_test, feature_names.

    The return value is cached so repeated calls from different technique
    files re-use the same split (essential for interpretability comparisons).
    """
    if _CACHE:
        return _CACHE

    loader = MLFPDataLoader()
    credit: pl.DataFrame = loader.load(DATASET_MODULE, DATASET_FILE).drop(
        CREDIT_NON_FEATURE_COLUMNS
    )

    # Split FIRST, then fit imputation/encoding on the training rows only:
    # PreprocessingPipeline.setup() on the whole frame would fit them on the
    # test rows too (it splits only after fitting).
    result = split_then_preprocess(
        credit,
        target=TARGET_COLUMN,
        test_size=0.2,
        seed=RANDOM_SEED,
        normalize=False,
        categorical_encoding="ordinal",
    )

    feature_columns = [c for c in result.train_data.columns if c != TARGET_COLUMN]
    X_train, y_train, col_info = to_sklearn_input(
        result.train_data,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
    )
    X_test, y_test, _ = to_sklearn_input(
        result.test_data,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
    )
    feature_names: list[str] = col_info["feature_columns"]

    _CACHE.update(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        ordinal_mappings=result.transformers["ordinal_mappings"],
    )
    return _CACHE


# The model is trained through kailash-ml's TrainingPipeline (fit + holdout
# evaluation + registry entry in one call) and loaded back from the
# registry, so SHAP explains exactly the registered artefact. Class
# weighting (scale_pos_weight) is kept from the original exercise design:
# it raises recall on the 12% default class at the cost of inflated
# probabilities (Exercise 5) — 05_fairness_audit.py discusses the effect.
MODEL_NAME = "credit_default_ex6"
_DB_ABS_PATH = (OUTPUT_DIR / "ex6_models.db").resolve()
DB_URL: str = os.environ.get("MLFP03_EX6_DB_URL", f"sqlite:///{_DB_ABS_PATH.as_posix()}")


async def _train_via_pipeline(
    X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]
) -> Any:
    from kailash.db import ConnectionManager
    from kailash_ml import ModelRegistry, TrainingPipeline
    from kailash_ml.engines.training_pipeline import EvalSpec, ModelSpec
    from kailash_ml.types import FeatureField, FeatureSchema

    frame = pl.DataFrame(X_train, schema=feature_names, orient="row").with_columns(
        pl.Series(TARGET_COLUMN, y_train),
        pl.int_range(0, len(y_train), dtype=pl.Int64).alias("row_id"),
    )
    schema = FeatureSchema(
        name="ex6_credit_input",
        features=[FeatureField(name=f, dtype="float64") for f in feature_names],
        entity_id_column="row_id",
    )
    conn = ConnectionManager(DB_URL)
    await conn.initialize()
    try:
        registry = ModelRegistry(conn)
        pipeline = TrainingPipeline(feature_store=None, registry=registry)
        result = await pipeline.train(
            data=frame,
            schema=schema,
            model_spec=ModelSpec(
                model_class="lightgbm.LGBMClassifier",
                framework="lightgbm",
                hyperparameters={
                    "n_estimators": 500,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "scale_pos_weight": float((1 - y_train.mean()) / y_train.mean()),
                    "random_state": RANDOM_SEED,
                    "verbose": -1,
                },
            ),
            eval_spec=EvalSpec(metrics=["auc"], split_strategy="holdout", test_size=0.2),
            experiment_name=MODEL_NAME,
        )
        if result.model_version is None:
            raise RuntimeError("TrainingPipeline did not register the model")
        version = int(result.model_version.version)
        # Unpickling executes code: only load artefacts you trained yourself.
        return pickle.loads(await registry.load_artifact(MODEL_NAME, version))
    finally:
        await conn.close()


def train_credit_model() -> dict[str, Any]:
    """Train the LightGBM credit default model via TrainingPipeline. Cached per-process.

    Returns a dict with model, y_proba, y_pred, auc, and all data from
    `load_credit_scoring()`.
    """
    if "model" in _CACHE:
        return _CACHE

    data = load_credit_scoring()
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    model = asyncio.run(_train_via_pipeline(X_train, y_train, data["feature_names"]))

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_proba)

    _CACHE.update(model=model, y_proba=y_proba, y_pred=y_pred, auc=auc)
    return _CACHE


# ════════════════════════════════════════════════════════════════════════
# SHAP EXPLAINER
# ════════════════════════════════════════════════════════════════════════


def build_shap_explainer() -> dict[str, Any]:
    """Construct the TreeSHAP explainer and compute SHAP values for X_test.

    Returns the full bundle: model, data, explainer, shap_vals, expected_value.
    """
    if "shap_vals" in _CACHE:
        return _CACHE

    bundle = train_credit_model()
    explainer = shap.TreeExplainer(bundle["model"])
    shap_values = explainer.shap_values(bundle["X_test"])

    # TreeSHAP for binary classifiers may return [class_0, class_1]
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    expected_value = (
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, list)
        else explainer.expected_value
    )

    _CACHE.update(
        explainer=explainer,
        shap_vals=shap_vals,
        expected_value=expected_value,
    )
    return _CACHE


# ════════════════════════════════════════════════════════════════════════
# REUSABLE UTILITIES
# ════════════════════════════════════════════════════════════════════════


def rank_features_by_mean_abs_shap(
    shap_vals: np.ndarray, feature_names: list[str]
) -> list[tuple[str, float]]:
    """Return [(feature, mean_abs_shap), ...] sorted descending."""
    mean_abs = np.abs(shap_vals).mean(axis=0)
    return sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)


def feature_index(feature_names: list[str], name: str) -> int:
    """Lookup a feature column index by name, raising a clear error."""
    if name not in feature_names:
        raise KeyError(
            f"Feature '{name}' not found. Available: {feature_names[:10]}..."
        )
    return feature_names.index(name)


def decode_group(
    X: np.ndarray,
    feature_names: list[str],
    attribute: str,
    ordinal_mappings: dict[str, dict[str, int]],
) -> np.ndarray:
    """Return a string label per row for a protected attribute.

    Categorical attributes are mapped back from their ordinal codes using
    the PreprocessingPipeline's fitted ``ordinal_mappings``; ``age`` is
    binned into ``AGE_BANDS``.
    """
    values = X[:, feature_index(feature_names, attribute)]
    if attribute == "age":
        labels = np.full(values.shape[0], "unknown", dtype=object)
        for name, lo, hi in AGE_BANDS:
            labels[(values >= lo) & (values <= hi)] = name
        return labels
    if attribute not in ordinal_mappings:
        raise KeyError(f"No ordinal mapping for '{attribute}' — is it categorical?")
    code_to_label = {code: label for label, code in ordinal_mappings[attribute].items()}
    return np.array([code_to_label.get(int(v), "unknown") for v in values], dtype=object)


def print_section(title: str, char: str = "=") -> None:
    """Print a standardised section banner."""
    line = char * 70
    print(f"\n{line}")
    print(f"  {title}")
    print(line)
