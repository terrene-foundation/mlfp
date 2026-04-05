# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — ML / AutoMLEngine
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Automated model selection and hyperparameter optimization
#            with agent double opt-in.  AutoMLEngine quick-trains multiple
#            model families, ranks by metric, then runs deep
#            HyperparameterSearch on the top candidate.
# LEVEL: Intermediate
# PARITY: Python-only (kailash-ml has no Rust counterpart)
# VALIDATES: AutoMLEngine, AutoMLConfig, AutoMLResult, CandidateResult,
#            LLMCostTracker, LLMBudgetExceededError — run()
#
# Run: uv run python textbook/python/05-ml/09_automl_engine.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math
import tempfile

import polars as pl

from kailash.db.connection import ConnectionManager
from kailash_ml import FeatureField, FeatureSchema
from kailash_ml.engines.automl_engine import (
    AutoMLConfig,
    AutoMLEngine,
    AutoMLResult,
    CandidateResult,
    LLMBudgetExceededError,
    LLMCostTracker,
)
from kailash_ml.engines.feature_store import FeatureStore
from kailash_ml.engines.hyperparameter_search import HyperparameterSearch
from kailash_ml.engines.model_registry import LocalFileArtifactStore, ModelRegistry
from kailash_ml.engines.training_pipeline import EvalSpec, TrainingPipeline


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
        hp_search = HyperparameterSearch(pipeline)

        engine = AutoMLEngine(pipeline, hp_search, registry=registry)
        assert isinstance(engine, AutoMLEngine)

        # ── 2. Create synthetic classification data ─────────────────────

        n_samples = 200
        df = pl.DataFrame(
            {
                "entity_id": [f"e{i}" for i in range(n_samples)],
                "feat_a": [float(i % 10) for i in range(n_samples)],
                "feat_b": [float(i * 0.5) for i in range(n_samples)],
                "feat_c": [float((i * 3) % 11) for i in range(n_samples)],
                "target": [i % 2 for i in range(n_samples)],
            }
        )

        schema = FeatureSchema(
            name="automl_demo",
            features=[
                FeatureField(name="feat_a", dtype="float64"),
                FeatureField(name="feat_b", dtype="float64"),
                FeatureField(name="feat_c", dtype="float64"),
            ],
            entity_id_column="entity_id",
        )

        # ── 3. AutoMLConfig — classification without agents ──────────────

        config = AutoMLConfig(
            task_type="classification",
            metric_to_optimize="accuracy",
            direction="maximize",
            search_strategy="random",
            search_n_trials=2,  # Small for tutorial speed
            agent=False,  # No agent augmentation
        )

        assert config.task_type == "classification"
        assert config.agent is False
        assert config.auto_approve is False  # Default: human approval required

        eval_spec = EvalSpec(metrics=["accuracy", "f1"])

        # ── 4. Run AutoML ────────────────────────────────────────────────

        result = await engine.run(
            data=df,
            schema=schema,
            config=config,
            eval_spec=eval_spec,
            experiment_name="automl_classification",
        )

        assert isinstance(result, AutoMLResult)
        assert isinstance(result.best_model, CandidateResult)
        assert "accuracy" in result.best_metrics
        assert len(result.all_candidates) >= 1
        assert result.total_time_seconds > 0

        # Candidates are ranked (rank 1 = best)
        assert result.best_model.rank == 1
        for i, candidate in enumerate(result.all_candidates):
            assert candidate.rank == i + 1

        # Baseline recommendation is populated (Guardrail 4)
        assert isinstance(result.baseline_recommendation, list)
        assert len(result.baseline_recommendation) >= 1

        # Agent recommendation is None when agent=False
        assert result.agent_recommendation is None

        # ── 5. AutoMLConfig for regression ───────────────────────────────

        regression_df = pl.DataFrame(
            {
                "entity_id": [f"r{i}" for i in range(n_samples)],
                "sqft": [float(500 + i * 10) for i in range(n_samples)],
                "bedrooms": [float(1 + i % 5) for i in range(n_samples)],
                "price": [float(100000 + i * 5000) for i in range(n_samples)],
            }
        )

        reg_schema = FeatureSchema(
            name="automl_regression",
            features=[
                FeatureField(name="sqft", dtype="float64"),
                FeatureField(name="bedrooms", dtype="float64"),
            ],
            entity_id_column="entity_id",
        )

        reg_config = AutoMLConfig(
            task_type="regression",
            metric_to_optimize="r2",
            direction="maximize",
            search_strategy="random",
            search_n_trials=2,
        )

        reg_eval = EvalSpec(metrics=["r2", "rmse"])

        reg_result = await engine.run(
            data=regression_df,
            schema=reg_schema,
            config=reg_config,
            eval_spec=reg_eval,
            experiment_name="automl_regression",
        )

        assert isinstance(reg_result, AutoMLResult)
        assert len(reg_result.all_candidates) >= 1

        # ── 6. LLMCostTracker — budget guardrail ────────────────────────

        tracker = LLMCostTracker(max_budget_usd=0.10)
        assert tracker.total_spent == 0.0

        # Record a small call
        tracker.record("test-model", input_tokens=100, output_tokens=50)
        assert tracker.total_spent > 0.0
        assert len(tracker.calls) == 1

        # Budget exceeded raises LLMBudgetExceededError
        try:
            tracker.record("test-model", input_tokens=100000, output_tokens=100000)
            assert False, "Should raise LLMBudgetExceededError"
        except LLMBudgetExceededError:
            pass  # Expected

        # ── 7. LLMCostTracker — validation ───────────────────────────────

        try:
            LLMCostTracker(max_budget_usd=-1.0)
            assert False, "Should reject negative budget"
        except ValueError:
            pass  # Expected

        try:
            LLMCostTracker(max_budget_usd=float("inf"))
            assert False, "Should reject infinite budget"
        except ValueError:
            pass  # Expected

        try:
            LLMCostTracker(max_budget_usd=float("nan"))
            assert False, "Should reject NaN budget"
        except ValueError:
            pass  # Expected

        # ── 8. AutoMLConfig — financial field validation ─────────────────

        try:
            AutoMLConfig(max_llm_cost_usd=float("nan"))
            assert False, "Should reject NaN cost"
        except ValueError:
            pass  # Expected

        try:
            AutoMLConfig(max_llm_cost_usd=float("inf"))
            assert False, "Should reject infinite cost"
        except ValueError:
            pass  # Expected

        try:
            AutoMLConfig(max_llm_cost_usd=-5.0)
            assert False, "Should reject negative cost"
        except ValueError:
            pass  # Expected

        try:
            AutoMLConfig(approval_timeout_seconds=0)
            assert False, "Should reject zero timeout"
        except ValueError:
            pass  # Expected

        # ── 9. Agent double opt-in pattern ───────────────────────────────
        # Agent augmentation requires BOTH:
        #   1. agent=True in AutoMLConfig
        #   2. pip install kailash-ml[agents]
        #
        # Without the [agents] extra installed, agent features are
        # simply not available.  The engine falls back to algorithmic
        # mode gracefully.

        agent_config = AutoMLConfig(
            task_type="classification",
            agent=True,  # Opt-in 1: enable agent
            auto_approve=False,  # Human approval gate (default)
            max_llm_cost_usd=5.0,  # Cost budget
        )

        assert agent_config.agent is True
        assert agent_config.auto_approve is False
        assert agent_config.max_llm_cost_usd == 5.0

        # ── 10. Serialization round-trips ────────────────────────────────

        # AutoMLConfig
        cfg_dict = config.to_dict()
        cfg_restored = AutoMLConfig.from_dict(cfg_dict)
        assert cfg_restored.task_type == config.task_type
        assert cfg_restored.agent == config.agent
        assert cfg_restored.search_n_trials == config.search_n_trials

        # CandidateResult
        cr_dict = result.best_model.to_dict()
        cr_restored = CandidateResult.from_dict(cr_dict)
        assert cr_restored.model_class == result.best_model.model_class
        assert cr_restored.rank == result.best_model.rank

        # AutoMLResult
        ar_dict = result.to_dict()
        ar_restored = AutoMLResult.from_dict(ar_dict)
        assert len(ar_restored.all_candidates) == len(result.all_candidates)
        assert ar_restored.total_time_seconds == result.total_time_seconds

        # ── 11. Edge case: invalid task type ─────────────────────────────

        try:
            bad_config = AutoMLConfig(task_type="unsupervised")
            await engine.run(
                data=df,
                schema=schema,
                config=bad_config,
                eval_spec=eval_spec,
                experiment_name="bad_task",
            )
            assert False, "Should raise ValueError for unknown task type"
        except ValueError:
            pass  # Expected

    # ── 12. Clean up ─────────────────────────────────────────────────────

    await conn.close()

    print("PASS: 05-ml/09_automl_engine")


asyncio.run(main())
