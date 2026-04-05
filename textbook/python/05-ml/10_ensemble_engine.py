# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — ML / EnsembleEngine
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Combine multiple models using blend(), stack(), bag(), and
#            boost().  EnsembleEngine accepts polars DataFrames and
#            converts at the sklearn boundary.  Task type (classification
#            vs regression) is auto-detected from the target column.
# LEVEL: Intermediate
# PARITY: Python-only (kailash-ml has no Rust counterpart)
# VALIDATES: EnsembleEngine, BlendResult, StackResult, BagResult,
#            BoostResult — blend(), stack(), bag(), boost()
#
# Run: uv run python textbook/python/05-ml/10_ensemble_engine.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from kailash_ml.engines.ensemble import (
    BagResult,
    BlendResult,
    BoostResult,
    EnsembleEngine,
    StackResult,
)

# ── 1. Create synthetic classification data ─────────────────────────

df = pl.DataFrame(
    {
        "feat_a": [float(i % 10) for i in range(200)],
        "feat_b": [float(i * 0.3) for i in range(200)],
        "feat_c": [float((i * 7) % 13) for i in range(200)],
        "target": [i % 2 for i in range(200)],
    }
)

assert isinstance(df, pl.DataFrame)

# ── 2. Train base models (sklearn) ──────────────────────────────────
# EnsembleEngine takes pre-fitted sklearn-compatible estimators.

import numpy as np

feature_cols = ["feat_a", "feat_b", "feat_c"]
X = df.select(feature_cols).to_numpy()
y = df["target"].to_numpy()

rf = RandomForestClassifier(n_estimators=20, random_state=42)
rf.fit(X, y)

gb = GradientBoostingClassifier(n_estimators=20, random_state=42)
gb.fit(X, y)

lr = LogisticRegression(max_iter=200, random_state=42)
lr.fit(X, y)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X, y)

# ── 3. Instantiate EnsembleEngine ───────────────────────────────────

engine = EnsembleEngine()
assert isinstance(engine, EnsembleEngine)

# ── 4. blend() — soft voting (weighted averaging) ────────────────────

blend_result = engine.blend(
    models=[rf, gb, lr],
    data=df,
    target="target",
    method="soft",
    test_size=0.2,
    seed=42,
)

assert isinstance(blend_result, BlendResult)
assert blend_result.n_models == 3
assert blend_result.method == "soft"
assert "accuracy" in blend_result.metrics
assert len(blend_result.weights) == 3
assert blend_result.ensemble_model is not None

# Component contributions show per-model metrics
assert len(blend_result.component_contributions) == 3
for contrib in blend_result.component_contributions:
    assert "model_index" in contrib
    assert "model_class" in contrib
    assert "weight" in contrib
    assert "metrics" in contrib

# ── 5. blend() — hard voting ─────────────────────────────────────────

hard_blend = engine.blend(
    models=[rf, gb, lr],
    data=df,
    target="target",
    method="hard",
    test_size=0.2,
    seed=42,
)

assert isinstance(hard_blend, BlendResult)
assert hard_blend.method == "hard"

# ── 6. blend() — custom weights ──────────────────────────────────────

weighted_blend = engine.blend(
    models=[rf, gb, lr],
    data=df,
    target="target",
    weights=[2.0, 3.0, 1.0],  # Weight GradientBoosting highest
    method="soft",
    test_size=0.2,
    seed=42,
)

assert weighted_blend.weights == [2.0, 3.0, 1.0]

# ── 7. stack() — meta-learner on CV predictions ─────────────────────

stack_result = engine.stack(
    models=[rf, gb, lr],
    data=df,
    target="target",
    meta_model_class="sklearn.linear_model.LogisticRegression",
    fold=3,
    test_size=0.2,
    seed=42,
)

assert isinstance(stack_result, StackResult)
assert stack_result.meta_model_class == "sklearn.linear_model.LogisticRegression"
assert stack_result.n_base_models == 3
assert stack_result.fold == 3
assert "accuracy" in stack_result.metrics
assert stack_result.ensemble_model is not None

# Component contributions for stacking
assert len(stack_result.component_contributions) == 3

# ── 8. bag() — bootstrap aggregating ────────────────────────────────

bag_result = engine.bag(
    model=dt,
    data=df,
    target="target",
    n_estimators=15,
    max_samples=0.8,
    max_features=0.9,
    test_size=0.2,
    seed=42,
)

assert isinstance(bag_result, BagResult)
assert bag_result.n_estimators == 15
assert bag_result.max_samples == 0.8
assert bag_result.max_features == 0.9
assert "accuracy" in bag_result.metrics
assert "DecisionTreeClassifier" in bag_result.base_model_class
assert bag_result.ensemble_model is not None

# ── 9. boost() — AdaBoost ───────────────────────────────────────────

boost_result = engine.boost(
    model=dt,
    data=df,
    target="target",
    n_estimators=30,
    learning_rate=0.1,
    test_size=0.2,
    seed=42,
)

assert isinstance(boost_result, BoostResult)
assert boost_result.n_estimators == 30
assert boost_result.learning_rate == 0.1
assert "accuracy" in boost_result.metrics
assert "DecisionTreeClassifier" in boost_result.base_model_class
assert boost_result.ensemble_model is not None

# ── 10. Regression auto-detection ────────────────────────────────────

regression_df = pl.DataFrame(
    {
        "x1": [float(i) for i in range(100)],
        "x2": [float(i * 0.5) for i in range(100)],
        "price": [float(50000 + i * 1000 + (i % 7) * 500) for i in range(100)],
    }
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

X_reg = regression_df.select(["x1", "x2"]).to_numpy()
y_reg = regression_df["price"].to_numpy()

rfr = RandomForestRegressor(n_estimators=20, random_state=42)
rfr.fit(X_reg, y_reg)

dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(X_reg, y_reg)

# blend auto-detects regression (many unique target values)
reg_blend = engine.blend(
    models=[rfr],
    data=regression_df,
    target="price",
    method="soft",
    test_size=0.2,
    seed=42,
)
assert isinstance(reg_blend, BlendResult)
# Regression metrics include r2, rmse, etc.
assert any(m in reg_blend.metrics for m in ["r2", "rmse", "mae"])

# bag works for regression too
reg_bag = engine.bag(
    model=dtr,
    data=regression_df,
    target="price",
    n_estimators=10,
    test_size=0.2,
    seed=42,
)
assert isinstance(reg_bag, BagResult)

# ── 11. Serialization round-trips ────────────────────────────────────

# BlendResult
br_dict = blend_result.to_dict()
br_restored = BlendResult.from_dict(br_dict)
assert br_restored.n_models == blend_result.n_models
assert br_restored.method == blend_result.method

# StackResult
sr_dict = stack_result.to_dict()
sr_restored = StackResult.from_dict(sr_dict)
assert sr_restored.meta_model_class == stack_result.meta_model_class
assert sr_restored.n_base_models == stack_result.n_base_models

# BagResult
bag_dict = bag_result.to_dict()
bag_restored = BagResult.from_dict(bag_dict)
assert bag_restored.n_estimators == bag_result.n_estimators

# BoostResult
boost_dict = boost_result.to_dict()
boost_restored = BoostResult.from_dict(boost_dict)
assert boost_restored.n_estimators == boost_result.n_estimators
assert boost_restored.learning_rate == boost_result.learning_rate

# ── 12. Edge case: empty model list ──────────────────────────────────

try:
    engine.blend(models=[], data=df, target="target")
    assert False, "Should raise ValueError for empty models"
except ValueError:
    pass  # Expected

try:
    engine.stack(models=[], data=df, target="target")
    assert False, "Should raise ValueError for empty models"
except ValueError:
    pass  # Expected

# ── 13. Edge case: invalid blend method ──────────────────────────────

try:
    engine.blend(models=[rf], data=df, target="target", method="invalid")
    assert False, "Should raise ValueError for invalid method"
except ValueError:
    pass  # Expected

# ── 14. Edge case: weight length mismatch ────────────────────────────

try:
    engine.blend(
        models=[rf, gb, lr],
        data=df,
        target="target",
        weights=[1.0, 2.0],  # 2 weights for 3 models
    )
    assert False, "Should raise ValueError for weight mismatch"
except ValueError:
    pass  # Expected

# ── 15. Edge case: invalid meta_model_class ──────────────────────────

try:
    engine.stack(
        models=[rf],
        data=df,
        target="target",
        meta_model_class="os.system",  # Not in allowlist
    )
    assert False, "Should reject non-allowlisted meta model"
except ValueError:
    pass  # Expected

print("PASS: 05-ml/10_ensemble_engine")
