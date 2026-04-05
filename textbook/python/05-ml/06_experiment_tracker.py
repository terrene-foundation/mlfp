# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — ML / ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Track experiments, runs, parameters, step-based metrics,
#            tags, and compare runs.  Uses the async context manager
#            pattern: `async with tracker.run(name) as run:`.
# LEVEL: Intermediate
# PARITY: Python-only (kailash-ml has no Rust counterpart)
# VALIDATES: ExperimentTracker, Experiment, Run, RunContext,
#            MetricEntry, RunComparison — run(), log_params(),
#            log_metrics(), log_metric(), set_tag(), compare_runs(),
#            get_metric_history(), search_runs()
#
# Run: uv run python textbook/python/05-ml/06_experiment_tracker.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from kailash.db.connection import ConnectionManager
from kailash_ml.engines.experiment_tracker import (
    Experiment,
    ExperimentNotFoundError,
    ExperimentTracker,
    MetricEntry,
    Run,
    RunComparison,
    RunContext,
    RunNotFoundError,
)


async def main() -> None:
    # ── 1. Set up ConnectionManager + ExperimentTracker ─────────────────

    conn = ConnectionManager("sqlite:///:memory:")
    await conn.initialize()

    with tempfile.TemporaryDirectory() as artifact_dir:
        tracker = ExperimentTracker(conn, artifact_root=artifact_dir)
        assert isinstance(tracker, ExperimentTracker)

        # ── 2. Context manager pattern: async with tracker.run() ────────
        # The preferred way to track runs.  Auto-creates the experiment
        # if it does not exist.  Marks COMPLETED on normal exit, FAILED
        # on exception.

        async with tracker.run("churn_experiment", run_name="baseline") as ctx:
            assert isinstance(ctx, RunContext)
            assert isinstance(ctx.run_id, str)
            assert ctx.run.status == "RUNNING"

            # Log parameters (values are strings)
            await ctx.log_params(
                {
                    "model_class": "RandomForestClassifier",
                    "n_estimators": "100",
                    "max_depth": "10",
                }
            )

            # Log metrics at specific steps (training curve)
            await ctx.log_metric("train_loss", 0.65, step=0)
            await ctx.log_metric("train_loss", 0.42, step=1)
            await ctx.log_metric("train_loss", 0.28, step=2)

            # Log multiple metrics at once
            await ctx.log_metrics({"accuracy": 0.92, "f1": 0.89}, step=0)

            # Set tags for categorization
            await ctx.set_tag("team", "ml-platform")
            await ctx.set_tag("dataset", "churn_v2")

        baseline_run_id = ctx.run_id

        # ── 3. Verify run was marked COMPLETED ──────────────────────────

        completed_run = await tracker.get_run(baseline_run_id)
        assert isinstance(completed_run, Run)
        assert completed_run.status == "COMPLETED"
        assert completed_run.end_time is not None
        assert completed_run.params["model_class"] == "RandomForestClassifier"
        assert completed_run.params["n_estimators"] == "100"
        assert completed_run.metrics["accuracy"] == 0.92
        assert completed_run.metrics["f1"] == 0.89
        assert completed_run.tags["team"] == "ml-platform"

        # ── 4. Run a second experiment run for comparison ────────────────

        async with tracker.run("churn_experiment", run_name="improved") as ctx2:
            await ctx2.log_params(
                {
                    "model_class": "GradientBoostingClassifier",
                    "n_estimators": "200",
                    "learning_rate": "0.05",
                }
            )
            await ctx2.log_metrics({"accuracy": 0.95, "f1": 0.93})

        improved_run_id = ctx2.run_id

        # ── 5. Failed run — exception marks FAILED ───────────────────────

        failed_run_id = None
        try:
            async with tracker.run("churn_experiment", run_name="failed_run") as ctx3:
                failed_run_id = ctx3.run_id
                await ctx3.log_params({"model_class": "BrokenModel"})
                raise RuntimeError("Training crashed")
        except RuntimeError:
            pass  # Expected

        assert failed_run_id is not None
        failed_run = await tracker.get_run(failed_run_id)
        assert failed_run.status == "FAILED"

        # ── 6. Get experiment metadata ───────────────────────────────────

        exp = await tracker.get_experiment("churn_experiment")
        assert isinstance(exp, Experiment)
        assert exp.name == "churn_experiment"
        assert isinstance(exp.id, str)
        assert isinstance(exp.created_at, str)

        # ── 7. List experiments ──────────────────────────────────────────

        experiments = await tracker.list_experiments()
        assert len(experiments) >= 1
        assert any(e.name == "churn_experiment" for e in experiments)

        # ── 8. List runs for an experiment ───────────────────────────────

        all_runs = await tracker.list_runs("churn_experiment")
        assert len(all_runs) == 3  # baseline, improved, failed

        # Filter by status
        completed_runs = await tracker.list_runs("churn_experiment", status="COMPLETED")
        assert len(completed_runs) == 2

        failed_runs = await tracker.list_runs("churn_experiment", status="FAILED")
        assert len(failed_runs) == 1

        # ── 9. compare_runs() — tabular metric/param comparison ──────────

        comparison = await tracker.compare_runs([baseline_run_id, improved_run_id])
        assert isinstance(comparison, RunComparison)
        assert len(comparison.run_ids) == 2
        assert len(comparison.run_names) == 2
        assert "accuracy" in comparison.metrics
        assert "f1" in comparison.metrics
        assert comparison.metrics["accuracy"][0] == 0.92  # baseline
        assert comparison.metrics["accuracy"][1] == 0.95  # improved
        assert "model_class" in comparison.params

        # ── 10. get_metric_history() — step-based training curves ────────

        history = await tracker.get_metric_history(baseline_run_id, "train_loss")
        assert isinstance(history, list)
        assert len(history) == 3  # steps 0, 1, 2
        assert all(isinstance(e, MetricEntry) for e in history)
        assert history[0].step == 0
        assert history[0].value == 0.65
        assert history[1].step == 1
        assert history[1].value == 0.42
        assert history[2].step == 2
        assert history[2].value == 0.28

        # ── 11. search_runs() — filter by parameters ────────────────────

        search_results = await tracker.search_runs(
            "churn_experiment",
            filter_params={"model_class": "GradientBoostingClassifier"},
        )
        assert len(search_results) == 1
        assert search_results[0].params["model_class"] == "GradientBoostingClassifier"

        # ── 12. Serialization round-trips ────────────────────────────────

        # Experiment
        exp_dict = exp.to_dict()
        exp_restored = Experiment.from_dict(exp_dict)
        assert exp_restored.name == exp.name
        assert exp_restored.id == exp.id

        # Run
        run_dict = completed_run.to_dict()
        run_restored = Run.from_dict(run_dict)
        assert run_restored.id == completed_run.id
        assert run_restored.status == "COMPLETED"

        # MetricEntry
        me_dict = history[0].to_dict()
        me_restored = MetricEntry.from_dict(me_dict)
        assert me_restored.key == "train_loss"
        assert me_restored.value == 0.65

        # RunComparison
        comp_dict = comparison.to_dict()
        comp_restored = RunComparison.from_dict(comp_dict)
        assert comp_restored.run_ids == comparison.run_ids

        # ── 13. Edge case: NaN/Inf metric values rejected ────────────────

        async with tracker.run("churn_experiment", run_name="nan_test") as ctx4:
            try:
                await ctx4.log_metric("loss", float("nan"))
                assert False, "Should reject NaN metric"
            except ValueError:
                pass  # Expected: NaN not allowed

            try:
                await ctx4.log_metric("loss", float("inf"))
                assert False, "Should reject Inf metric"
            except ValueError:
                pass  # Expected: Inf not allowed

        # ── 14. Edge case: experiment not found ──────────────────────────

        try:
            await tracker.get_experiment("nonexistent")
            assert False, "Should raise ExperimentNotFoundError"
        except ExperimentNotFoundError:
            pass  # Expected

        # ── 15. Edge case: run not found ─────────────────────────────────

        try:
            await tracker.get_run("nonexistent-run-id")
            assert False, "Should raise RunNotFoundError"
        except RunNotFoundError:
            pass  # Expected

        # ── 16. delete_run() ─────────────────────────────────────────────

        await tracker.delete_run(failed_run_id)
        try:
            await tracker.get_run(failed_run_id)
            assert False, "Deleted run should not be found"
        except RunNotFoundError:
            pass  # Expected

        # ── 17. delete_experiment() ──────────────────────────────────────

        # Create a throwaway experiment to delete
        await tracker.create_experiment("throwaway")
        async with tracker.run("throwaway", run_name="temp") as ctx5:
            await ctx5.log_metric("x", 1.0)

        await tracker.delete_experiment("throwaway")
        try:
            await tracker.get_experiment("throwaway")
            assert False, "Deleted experiment should not be found"
        except ExperimentNotFoundError:
            pass  # Expected

    # ── 18. Clean up ─────────────────────────────────────────────────────

    await conn.close()

    print("PASS: 05-ml/06_experiment_tracker")


asyncio.run(main())
