# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP03 — Assessment Task 3: Decisions Priced in Dollars (Reference Solution)

Instructors only. Graded by grader.py on applications the student never sees.

Decisions this reference makes (one defensible route, not the only one):

1. Inputs: every numeric application field except the key and any field that
   predicts the outcome almost perfectly on its own (the post-outcome leak),
   plus loan-to-income and savings-to-income. Missing income is imputed from
   age with a line fitted on the fitting rows.
2. Split first: 75% of the history fits the model, 25% is held back. The
   held-back rows are used twice and only for things that are not fitting the
   model: Platt scaling (``TrainingPipeline.calibrate``) and the cost estimate.
3. Model and imbalance: the risk in this book is close to additive in the
   (log-)odds, so an L2 logistic regression ranks as well as boosting on a few
   thousand rows (Task 2's comparison shows the same). It is trained with
   ``class_weight="balanced"`` so the 13% defaulters weigh as much as the
   payers. Re-weighting inflates every probability, which is why step 2
   recalibrates on rows the model never saw — calibration removes the
   weight shift, it does not count the cost asymmetry twice.
4. Decision: with calibrated probabilities, approving costs p × c_missed and
   declining costs (1 − p) × c_declined, so approve exactly when
   p < c_declined / (c_declined + c_missed).
5. Deployment: a ``CreditScorer`` (preprocessing + calibrated model) is
   registered in the given ``ModelRegistry`` and promoted to production, so the
   serving layer loads exactly the object that produced these numbers.
"""
from __future__ import annotations

import asyncio
import pickle
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

from kailash.db import ConnectionManager
from kailash_ml import ModelRegistry, PreprocessingPipeline, TrainingPipeline
from kailash_ml.engines.model_registry import LocalFileArtifactStore
from kailash_ml.engines.training_pipeline import EvalSpec, ModelSpec
from kailash_ml.types import FeatureField, FeatureSchema
from shared import MLFPDataLoader

warnings.filterwarnings("ignore")

ID = "customer_id"
TARGET = "default"
SEED = 42
MODEL_NAME = "credit_default_decision"
ENGINEERED = ["loan_to_income", "savings_to_income"]


def load_history() -> pl.DataFrame:
    """The labelled development file (for local runs only)."""
    return MLFPDataLoader().load("mlfp02", "sg_credit_scoring.parquet")


def _leak_free_numeric(train: pl.DataFrame) -> list[str]:
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


class CreditScorer:
    """Everything needed to turn an application frame into a calibrated
    default probability. This object is what goes into the registry."""

    def __init__(self, cols, slope, intercept, pre, model):
        self.cols, self.slope, self.intercept = cols, slope, intercept
        self.pre, self.model = pre, model

    def features(self, applications: pl.DataFrame) -> pl.DataFrame:
        income = pl.col("income_sgd").cast(pl.Float64)
        filled = income.fill_null(pl.lit(self.intercept) + pl.lit(self.slope) * pl.col("age").cast(pl.Float64))
        return applications.with_columns(
            (pl.col("loan_amount_sgd") / filled).alias("loan_to_income"),
            (pl.col("savings_balance") / filled).alias("savings_to_income"),
        ).select(self.cols)

    def predict_proba(self, applications: pl.DataFrame) -> np.ndarray:
        Z = self.pre.transform(self.features(applications)).select(self.cols).to_numpy()
        return self.model.predict_proba(Z)[:, 1]


async def _train(fit: pl.DataFrame, cal: pl.DataFrame, cols, slope, intercept, registry_dir: Path):
    probe = CreditScorer(cols, slope, intercept, None, None)
    fit_x = probe.features(fit)
    pre = PreprocessingPipeline()
    pre.setup(fit_x.with_columns(fit[TARGET]), target=TARGET, normalize=True, imputation_strategy="median", seed=SEED)
    frame = pl.concat(
        [fit.select(ID), pre.transform(fit_x).select(cols), fit.select(pl.col(TARGET).cast(pl.Int64))],
        how="horizontal",
    )

    registry_dir.mkdir(parents=True, exist_ok=True)
    conn = ConnectionManager(f"sqlite:///{(registry_dir / 'registry.db').as_posix()}")
    await conn.initialize()
    try:
        registry = ModelRegistry(conn, LocalFileArtifactStore(registry_dir / "artifacts"))
        pipeline = TrainingPipeline(feature_store=None, registry=registry)
        result = await pipeline.train(
            data=frame,
            schema=FeatureSchema(
                name="credit_default",
                features=[FeatureField(name=c, dtype="float64") for c in cols],
                entity_id_column=ID,
            ),
            model_spec=ModelSpec(
                model_class="sklearn.linear_model.LogisticRegression",
                framework="sklearn",
                hyperparameters={"C": 0.3, "class_weight": "balanced", "max_iter": 3000},
            ),
            eval_spec=EvalSpec(metrics=["auc"], split_strategy="holdout", test_size=0.1),
            experiment_name=MODEL_NAME,
        )
        mv = result.model_version
        base = pickle.loads(await registry.load_artifact(mv.name, mv.version))
        cal_x = pre.transform(probe.features(cal)).select(cols)
        calibrated = await pipeline.calibrate(base, cal_x, cal[TARGET].cast(pl.Int64), method="sigmoid")
        scorer = CreditScorer(cols, slope, intercept, pre, calibrated)
        version = await registry.register_model(MODEL_NAME, pickle.dumps(scorer))
        await registry.promote_model(
            MODEL_NAME, version.version, "production", reason="calibrated scorer with cost-based decision rule"
        )
    finally:
        await conn.close()
    return scorer


def build_decision_model(history: pl.DataFrame, costs: dict, registry_dir: str) -> dict:
    c_missed = float(costs["missed_default"])
    c_declined = float(costs["declined_good"])

    rng = np.random.default_rng(SEED)
    y_all = history[TARGET].to_numpy()
    is_cal = np.zeros(history.height, bool)
    for cls in (0, 1):  # stratified 25% hold-back
        idx = np.flatnonzero(y_all == cls)
        is_cal[rng.choice(idx, size=len(idx) // 4, replace=False)] = True
    fit, cal = history.filter(pl.Series(~is_cal)), history.filter(pl.Series(is_cal))

    cols = [c for c in _leak_free_numeric(fit) if c != ID] + ENGINEERED
    known = fit.filter(pl.col("income_sgd").is_not_null())
    slope, intercept = np.polyfit(
        known["age"].to_numpy().astype(float), known["income_sgd"].to_numpy().astype(float), 1
    )
    scorer = asyncio.run(_train(fit, cal, cols, float(slope), float(intercept), Path(registry_dir)))

    threshold = c_declined / (c_declined + c_missed)

    def predict_proba(applications: pl.DataFrame) -> np.ndarray:
        return scorer.predict_proba(applications)

    def decide(applications: pl.DataFrame) -> np.ndarray:
        return predict_proba(applications) < threshold

    # Cost estimate on the held-back rows with their real outcomes.
    approve = decide(cal)
    y_cal = cal[TARGET].to_numpy()
    realised = np.where(approve, y_cal * c_missed, (1 - y_cal) * c_declined)

    return {
        "predict_proba": predict_proba,
        "decide": decide,
        "expected_cost": float(realised.mean()),
        "model_name": MODEL_NAME,
    }


if __name__ == "__main__":
    import tempfile

    data = load_history().sample(8_000, seed=1)
    with tempfile.TemporaryDirectory() as d:
        out = build_decision_model(data, {"missed_default": 10_000, "declined_good": 1_500}, d)
        print("estimated cost per application:", round(out["expected_cost"], 2))
        print("approval rate:", out["decide"](data.drop(TARGET)).mean())
