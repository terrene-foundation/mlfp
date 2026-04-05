# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — ML / HyperparameterSearch
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Optimize hyperparameters using grid, random, and Bayesian
#            search strategies.  Covers SearchSpace, ParamDistribution,
#            SearchConfig, and the search() entry point.
# LEVEL: Intermediate
# PARITY: Python-only (kailash-ml has no Rust counterpart)
# VALIDATES: HyperparameterSearch, SearchSpace, ParamDistribution,
#            SearchConfig, TrialResult, SearchResult — search(),
#            sample_grid(), sample_random()
#
# Run: uv run python textbook/python/05-ml/08_hyperparameter_search.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import tempfile

import polars as pl

from kailash.db.connection import ConnectionManager
from kailash_ml import FeatureField, FeatureSchema
from kailash_ml.engines.feature_store import FeatureStore
from kailash_ml.engines.hyperparameter_search import (
    HyperparameterSearch,
    ParamDistribution,
    SearchConfig,
    SearchResult,
    SearchSpace,
    TrialResult,
)
from kailash_ml.engines.model_registry import LocalFileArtifactStore, ModelRegistry
from kailash_ml.engines.training_pipeline import EvalSpec, ModelSpec, TrainingPipeline


async def main() -> None:
    # ── 1. Set up infrastructure ────────────────────────────────────────

    conn = ConnectionManager("sqlite:///:memory:")
    await conn.initialize()

    with tempfile.TemporaryDirectory() as artifact_dir:
        artifact_store = LocalFileArtifactStore(artifact_dir)
        registry = ModelRegistry(conn, artifact_store)
        fs = FeatureStore(conn)
        await fs.initialize()

        pipeline = TrainingPipeline(feature_store=fs, registry=registry)
        search = HyperparameterSearch(pipeline)
        assert isinstance(search, HyperparameterSearch)

        # ── 2. Create synthetic data ────────────────────────────────────

        n_samples = 150
        df = pl.DataFrame(
            {
                "entity_id": [f"e{i}" for i in range(n_samples)],
                "feat_a": [float(i % 8) for i in range(n_samples)],
                "feat_b": [float(i * 0.3) for i in range(n_samples)],
                "target": [i % 2 for i in range(n_samples)],
            }
        )

        schema = FeatureSchema(
            name="hp_search_demo",
            features=[
                FeatureField(name="feat_a", dtype="float64"),
                FeatureField(name="feat_b", dtype="float64"),
            ],
            entity_id_column="entity_id",
        )

        # ── 3. Define a SearchSpace ─────────────────────────────────────
        # ParamDistribution supports: uniform, log_uniform, int_uniform,
        # categorical.

        space = SearchSpace(
            params=[
                ParamDistribution(
                    name="n_estimators",
                    type="int_uniform",
                    low=5,
                    high=30,
                ),
                ParamDistribution(
                    name="max_depth",
                    type="categorical",
                    choices=[3, 5, 7],
                ),
            ]
        )

        # ── 4. SearchSpace.sample_grid() — exhaustive grid ──────────────
        # For int_uniform with low=5, high=30 this generates all integers.
        # For categorical it uses all choices.

        grid = space.sample_grid()
        assert isinstance(grid, list)
        assert len(grid) > 0
        # Each element is a dict of param_name -> value
        assert "n_estimators" in grid[0]
        assert "max_depth" in grid[0]

        # ── 5. SearchSpace.sample_random() — random samples ──────────────

        random_samples = space.sample_random(n=10)
        assert isinstance(random_samples, list)
        assert len(random_samples) == 10
        for sample in random_samples:
            assert "n_estimators" in sample
            assert "max_depth" in sample
            # int_uniform produces integers
            assert isinstance(sample["n_estimators"], int)
            assert 5 <= sample["n_estimators"] <= 30
            # categorical draws from choices
            assert sample["max_depth"] in [3, 5, 7]

        # ── 6. ParamDistribution types ──────────────────────────────────

        # uniform: continuous float between low and high
        uniform_param = ParamDistribution("lr", "uniform", low=0.001, high=0.1)

        # log_uniform: log-scale sampling (useful for learning rates)
        log_param = ParamDistribution("lr_log", "log_uniform", low=0.001, high=0.1)

        mixed_space = SearchSpace(params=[uniform_param, log_param])
        mixed_samples = mixed_space.sample_random(n=5)
        for s in mixed_samples:
            assert 0.001 <= s["lr"] <= 0.1
            assert 0.001 <= s["lr_log"] <= 0.1

        # ── 7. Configure a random search ─────────────────────────────────

        config = SearchConfig(
            strategy="random",
            n_trials=3,  # Small for tutorial speed
            metric_to_optimize="accuracy",
            direction="maximize",
        )

        assert config.strategy == "random"
        assert config.direction == "maximize"

        base_model_spec = ModelSpec(
            model_class="sklearn.ensemble.RandomForestClassifier",
            hyperparameters={"random_state": 42},
            framework="sklearn",
        )

        eval_spec = EvalSpec(metrics=["accuracy"])

        # ── 8. Run random search ─────────────────────────────────────────

        result = await search.search(
            data=df,
            schema=schema,
            base_model_spec=base_model_spec,
            search_space=space,
            config=config,
            eval_spec=eval_spec,
            experiment_name="random_search_demo",
        )

        assert isinstance(result, SearchResult)
        assert result.strategy == "random"
        assert len(result.all_trials) == 3
        assert result.best_trial_number >= 0
        assert "accuracy" in result.best_metrics
        assert isinstance(result.best_params, dict)
        assert result.total_time_seconds > 0

        # All trials have results
        for trial in result.all_trials:
            assert isinstance(trial, TrialResult)
            assert isinstance(trial.params, dict)
            assert "accuracy" in trial.metrics
            assert trial.training_time_seconds > 0

        # ── 9. Run grid search ───────────────────────────────────────────
        # Use a small categorical-only space so grid is manageable.

        small_space = SearchSpace(
            params=[
                ParamDistribution(
                    name="max_depth",
                    type="categorical",
                    choices=[3, 5],
                ),
            ]
        )

        grid_config = SearchConfig(
            strategy="grid",
            n_trials=10,  # Grid ignores n_trials; uses full grid
            metric_to_optimize="accuracy",
            direction="maximize",
        )

        grid_result = await search.search(
            data=df,
            schema=schema,
            base_model_spec=base_model_spec,
            search_space=small_space,
            config=grid_config,
            eval_spec=eval_spec,
            experiment_name="grid_search_demo",
        )

        assert isinstance(grid_result, SearchResult)
        assert grid_result.strategy == "grid"
        assert len(grid_result.all_trials) == 2  # 2 choices for max_depth

        # ── 10. Minimization direction ───────────────────────────────────

        min_config = SearchConfig(
            strategy="random",
            n_trials=2,
            metric_to_optimize="accuracy",
            direction="minimize",
        )

        min_result = await search.search(
            data=df,
            schema=schema,
            base_model_spec=base_model_spec,
            search_space=small_space,
            config=min_config,
            eval_spec=eval_spec,
            experiment_name="min_search_demo",
        )

        # Best should be the trial with lowest accuracy
        best_acc = min_result.best_metrics.get("accuracy", 0)
        for trial in min_result.all_trials:
            assert best_acc <= trial.metrics.get("accuracy", float("inf"))

        # ── 11. Serialization round-trips ────────────────────────────────

        # ParamDistribution
        pd_dict = space.params[0].to_dict()
        pd_restored = ParamDistribution.from_dict(pd_dict)
        assert pd_restored.name == "n_estimators"
        assert pd_restored.type == "int_uniform"

        # SearchSpace
        ss_dict = space.to_dict()
        ss_restored = SearchSpace.from_dict(ss_dict)
        assert len(ss_restored.params) == len(space.params)

        # SearchConfig
        sc_dict = config.to_dict()
        sc_restored = SearchConfig.from_dict(sc_dict)
        assert sc_restored.strategy == "random"
        assert sc_restored.n_trials == 3

        # TrialResult
        tr_dict = result.all_trials[0].to_dict()
        tr_restored = TrialResult.from_dict(tr_dict)
        assert tr_restored.trial_number == result.all_trials[0].trial_number

        # SearchResult
        sr_dict = result.to_dict()
        sr_restored = SearchResult.from_dict(sr_dict)
        assert sr_restored.strategy == result.strategy
        assert len(sr_restored.all_trials) == len(result.all_trials)

        # ── 12. Edge case: invalid strategy ──────────────────────────────

        try:
            bad_config = SearchConfig(strategy="invalid_strategy")
            await search.search(
                data=df,
                schema=schema,
                base_model_spec=base_model_spec,
                search_space=space,
                config=bad_config,
                eval_spec=eval_spec,
                experiment_name="bad_strategy",
            )
            assert False, "Should raise ValueError for invalid strategy"
        except ValueError:
            pass  # Expected

    # ── 13. Clean up ─────────────────────────────────────────────────────

    await conn.close()

    print("PASS: 05-ml/08_hyperparameter_search")


asyncio.run(main())
