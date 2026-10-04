# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: the split-first helper never lets test rows shape the fit.

kailash-ml's ``PreprocessingPipeline.setup()`` fits imputation, encoding and
scaling on every row before it splits. ``shared.kailash_helpers`` holds the
test rows out first; these tests pin that every fitted statistic comes from
the training rows only.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from shared.kailash_helpers import (
    preprocess_train_test,
    split_raw_train_test,
    split_then_preprocess,
)


@pytest.fixture()
def frame() -> pl.DataFrame:
    rng = np.random.RandomState(0)
    n = 400
    df = pl.DataFrame(
        {
            "x1": rng.normal(50, 10, n),
            "x2": rng.normal(0, 1, n),
            "cat": rng.choice(["a", "b", "c"], n),
            "y": (rng.rand(n) < 0.25).astype(int),
        }
    )
    # every 7th x1 missing so the imputer has work to do
    return df.with_columns(
        pl.when(pl.int_range(pl.len()) % 7 == 0)
        .then(None)
        .otherwise(pl.col("x1"))
        .alias("x1")
    )


def test_split_is_disjoint_complete_and_stratified(frame: pl.DataFrame) -> None:
    tr, te, stratified = split_raw_train_test(
        frame.with_row_index("rid"), "y", test_size=0.2, seed=42
    )
    assert stratified
    assert set(tr["rid"]).isdisjoint(set(te["rid"]))
    assert tr.height + te.height == frame.height
    assert abs(tr["y"].mean() - te["y"].mean()) < 0.01


def test_fitted_statistics_come_from_train_rows_only(frame: pl.DataFrame) -> None:
    kwargs = dict(normalize=True, categorical_encoding="target", imputation_strategy="median")
    result = split_then_preprocess(frame, "y", test_size=0.2, seed=42, **kwargs)
    train_raw, test_raw, _ = split_raw_train_test(frame, "y", test_size=0.2, seed=42)

    median = train_raw["x1"].median()
    assert result.transformers["imputer_stats"]["x1"] == median
    expected_mean = [train_raw["x1"].fill_null(median).mean(), train_raw["x2"].mean()]
    assert np.allclose(result.transformers["scaler"].mean_, expected_mean)
    expected_te = dict(train_raw.group_by("cat").agg(pl.col("y").mean()).iter_rows())
    fitted_te = result.transformers["target_mappings"]["cat"]
    assert all(abs(fitted_te[k] - v) < 1e-12 for k, v in expected_te.items())

    # Wildly different test rows must not move any fitted statistic.
    wild = test_raw.with_columns(pl.col("x1") * 1000, pl.col("x2") + 500, pl.lit(1).alias("y"))
    again = preprocess_train_test(train_raw, wild, "y", seed=42, **kwargs)
    assert np.allclose(again.transformers["scaler"].mean_, result.transformers["scaler"].mean_)
    assert again.transformers["target_mappings"] == result.transformers["target_mappings"]
    assert again.transformers["imputer_stats"] == result.transformers["imputer_stats"]

    assert result.train_data.height == train_raw.height
    assert result.test_data.height == test_raw.height
    assert result.train_data.columns == result.test_data.columns


def test_test_only_gaps_are_filled_with_training_statistic(frame: pl.DataFrame) -> None:
    train_raw, test_raw, _ = split_raw_train_test(frame, "y", seed=42)
    gappy = test_raw.with_columns(
        pl.when(pl.int_range(pl.len()) < 3).then(None).otherwise(pl.col("x2")).alias("x2")
    )
    result = preprocess_train_test(
        train_raw, gappy, "y", normalize=False, categorical_encoding="ordinal"
    )
    assert result.test_null_fills["x2"] == pytest.approx(train_raw["x2"].mean())
    assert result.test_data["x2"].null_count() == 0


def test_row_changing_options_touch_training_rows_only(frame: pl.DataFrame) -> None:
    reg = frame.with_columns((pl.col("x2") * 3).alias("t")).drop("y")
    result = split_then_preprocess(reg, "t", test_size=100, seed=1, remove_outliers=True)
    assert not result.stratified
    assert result.test_data.height == 100
    assert result.train_only_steps and result.train_only_steps[0].startswith("remove_outliers")


@pytest.mark.parametrize("bad", [{"train_size": 0.7}, {"seed_typo": 1}])
def test_rejects_options_that_would_bypass_the_split(frame: pl.DataFrame, bad: dict) -> None:
    with pytest.raises(TypeError):
        split_then_preprocess(frame, "y", **bad)


def test_stratify_true_requires_classification_target(frame: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="classification"):
        split_then_preprocess(frame, "x2", stratify=True)
