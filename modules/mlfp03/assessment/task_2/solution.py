# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP03 — Assessment Task 2: Model Selection You Can Defend (Reference Solution)

Instructors only. Graded by grader.py on training samples and applications the
student never sees.

Decisions this reference makes (one defensible route, not the only one):

1. Inputs: every numeric application field except the key and any field that
   predicts the outcome almost perfectly on its own (the post-outcome leak),
   median-imputed and standardised. Extra numeric fields are used as given.
2. Comparison: seven families, each with a small hyperparameter grid, scored by
   the SAME stratified 5-fold split (ROC-AUC). Imputation and scaling sit
   inside every fold, so no fold's validation rows leak into its fitting. Each
   family's score is its best grid point's mean out-of-fold AUC.
3. Regularisation matters when data is scarce: the logistic grid spans strong
   to weak L2 penalties, the trees have depth / leaf-size limits, and boosting
   uses shallow trees with a small learning rate.
4. The winner is refitted on all training rows through kailash-ml
   (``PreprocessingPipeline`` for imputation/scaling, ``TrainingPipeline`` for
   the model, registered in a throwaway ``ModelRegistry``).
"""
from __future__ import annotations

import asyncio
import os
import pickle
import tempfile
import uuid
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier

from kailash_ml import ModelRegistry, PreprocessingPipeline, TrainingPipeline
from kailash_ml.engines.training_pipeline import EvalSpec, ModelSpec
from kailash_ml.types import FeatureField, FeatureSchema
from shared import MLFPDataLoader

warnings.filterwarnings("ignore")

ID = "customer_id"
TARGET = "default"
SEED = 42

# family -> list of (model_class path, framework, hyperparameters)
GRID: dict[str, list[tuple[str, str, dict]]] = {
    "logistic_regression": [
        ("sklearn.linear_model.LogisticRegression", "sklearn", {"C": c, "max_iter": 3000})
        for c in (0.003, 0.01, 0.03, 0.1, 1.0)
    ],
    "svm": [("sklearn.svm.SVC", "sklearn", {"C": 0.3, "kernel": "rbf", "random_state": SEED})],
    "knn": [("sklearn.neighbors.KNeighborsClassifier", "sklearn", {"n_neighbors": k}) for k in (25, 75)],
    "naive_bayes": [("sklearn.naive_bayes.GaussianNB", "sklearn", {})],
    "decision_tree": [
        ("sklearn.tree.DecisionTreeClassifier", "sklearn", {"max_depth": d, "min_samples_leaf": 20, "random_state": SEED})
        for d in (3, 5)
    ],
    "random_forest": [
        ("sklearn.ensemble.RandomForestClassifier", "sklearn",
         {"n_estimators": 200, "min_samples_leaf": leaf, "max_features": "sqrt", "n_jobs": 2, "random_state": SEED})
        for leaf in (5, 20)
    ],  # fmt: skip
    "gradient_boosting": [
        ("lightgbm.LGBMClassifier", "lightgbm",
         {"n_estimators": n, "learning_rate": 0.03, "num_leaves": 7, "min_child_samples": 30,
          "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "random_state": SEED, "verbose": -1})
        for n in (100, 300)
    ],  # fmt: skip
}
CLASSES = {
    "sklearn.linear_model.LogisticRegression": LogisticRegression,
    "sklearn.svm.SVC": SVC,
    "sklearn.neighbors.KNeighborsClassifier": KNeighborsClassifier,
    "sklearn.naive_bayes.GaussianNB": GaussianNB,
    "sklearn.tree.DecisionTreeClassifier": DecisionTreeClassifier,
    "sklearn.ensemble.RandomForestClassifier": RandomForestClassifier,
    "lightgbm.LGBMClassifier": LGBMClassifier,
}


def load_history() -> pl.DataFrame:
    """The labelled development file (for local runs only)."""
    return MLFPDataLoader().load("mlfp02", "sg_credit_scoring.parquet")


def _input_columns(train: pl.DataFrame) -> list[str]:
    y = train[TARGET].to_numpy()
    n1 = y.sum()
    cols = []
    for c, t in train.schema.items():
        if c in (ID, TARGET) or not t.is_numeric():
            continue
        x = train[c].cast(pl.Float64)
        x = x.fill_null(x.median() if x.null_count() < x.len() else 0.0).to_numpy()
        r = rankdata(x)
        a = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))
        if max(a, 1 - a) <= 0.9:  # near-perfect single fields are post-outcome leaks
            cols.append(c)
    return cols


def _compare(X: np.ndarray, y: np.ndarray) -> tuple[dict[str, float], dict[str, tuple]]:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores: dict[str, float] = {}
    best: dict[str, tuple] = {}
    for family, grid in GRID.items():
        for spec in grid:
            est = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), CLASSES[spec[0]](**spec[2]))
            s = float(cross_val_score(est, X, y, cv=folds, scoring="roc_auc").mean())
            if s > scores.get(family, -1.0):
                scores[family], best[family] = s, spec
    return scores, best


async def _fit_final(train: pl.DataFrame, cols: list[str], spec: tuple):
    """Refit the chosen model on all training rows through kailash-ml."""
    pre = PreprocessingPipeline()
    pre.setup(train.select(cols + [TARGET]), target=TARGET, normalize=True, imputation_strategy="median", seed=SEED)
    frame = pl.concat([train.select(ID), pre.transform(train.select(cols)), train.select(TARGET)], how="horizontal")

    from kailash.db import ConnectionManager

    db = Path(tempfile.gettempdir()) / f"mlfp03_t2_{os.getpid()}_{uuid.uuid4().hex[:8]}.db"
    conn = ConnectionManager(f"sqlite:///{db.as_posix()}")
    await conn.initialize()
    try:
        registry = ModelRegistry(conn)
        schema = FeatureSchema(
            name="credit_default",
            features=[FeatureField(name=c, dtype="float64") for c in cols],
            entity_id_column=ID,
        )
        result = await TrainingPipeline(feature_store=None, registry=registry).train(
            data=frame,
            schema=schema,
            model_spec=ModelSpec(model_class=spec[0], framework=spec[1], hyperparameters=spec[2]),
            eval_spec=EvalSpec(metrics=["auc"], split_strategy="holdout", test_size=0.1),
            experiment_name="credit_default",
        )
        mv = result.model_version
        model = pickle.loads(await registry.load_artifact(mv.name, mv.version))
    finally:
        await conn.close()
        db.unlink(missing_ok=True)
    return pre, model


def select_and_fit(train: pl.DataFrame) -> dict:
    cols = _input_columns(train)
    X = train.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy()
    y = train[TARGET].to_numpy()
    scores, best = _compare(X, y)
    chosen = max(scores, key=scores.get)
    spec = best[chosen]
    if spec[0] == "sklearn.svm.SVC":  # ROC-AUC in CV used the margin; serving needs probabilities
        spec = (spec[0], spec[1], {**spec[2], "probability": True})
    pre, model = asyncio.run(_fit_final(train, cols, spec))

    def predict_proba(applications: pl.DataFrame) -> np.ndarray:
        Z = pre.transform(applications.select(cols)).select(cols).to_numpy()
        return model.predict_proba(Z)[:, 1]

    return {
        "cv_auc": scores,
        "chosen": chosen,
        "estimated_auc": scores[chosen],
        "predict_proba": predict_proba,
    }


if __name__ == "__main__":
    data = load_history().sample(5_000, seed=1)
    out = select_and_fit(data)
    for fam, s in sorted(out["cv_auc"].items(), key=lambda kv: -kv[1]):
        print(f"{fam:<22} cv AUC {s:.4f}")
    print(f"chosen: {out['chosen']}  (estimate {out['estimated_auc']:.4f})")
