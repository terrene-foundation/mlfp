# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — ML / ExperimentTracker (kailash-ml 1.1.1)
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Track experiments, runs, parameters, step-based metrics,
#            tags, and compare runs.  Uses the async factory + context
#            manager pattern: `tracker = await ExperimentTracker.create(...)`
#            then `async with tracker.track(experiment=, run_name=) as run:`.
# LEVEL: Intermediate
# PARITY: Python-only (kailash-ml has no Rust counterpart)
# VALIDATES: ExperimentTracker.create / .track / .start_run / .end_run /
#            .get_run / .list_runs / .list_experiments / .search_runs /
#            .diff_runs / .list_metrics / .close
#            Run.log_params / .log_metric / .log_metrics / .add_tag / .run_id
#
# Run: uv run python textbook/python/05-ml/06_experiment_tracker.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math
import tempfile
from pathlib import Path

from kailash_ml import ExperimentTracker


async def main() -> None:
    # ── 1. Set up ExperimentTracker via the async factory ───────────────
    # The factory replaces the pre-1.0 positional constructor.  ``store_url``
    # accepts SQLite/Postgres/MySQL URLs; passing ``None`` would resolve via
    # the ``KAILASH_ML_STORE_URL`` env var or fall back to the default
    # ``~/.kailash_ml/ml.db``.

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "experiments.db"
        tracker = await ExperimentTracker.create(store_url=f"sqlite:///{db_path}")
        assert isinstance(tracker, ExperimentTracker)

        # ── 2. Async-with run context: tracker.track(...) ────────────────
        # The preferred pattern.  Auto-creates the experiment on first use.
        # On normal exit the run is marked FINISHED; on exception, FAILED.

        async with tracker.track(
            experiment="churn_experiment", run_name="baseline"
        ) as run:
            assert isinstance(run.run_id, str)

            # Log parameters (values are strings)
            await run.log_params(
                {
                    "model_class": "RandomForestClassifier",
                    "n_estimators": "100",
                    "max_depth": "10",
                }
            )

            # Log step-based metrics (training curve over steps 0, 1, 2)
            await run.log_metric("train_loss", 0.65, step=0)
            await run.log_metric("train_loss", 0.42, step=1)
            await run.log_metric("train_loss", 0.28, step=2)

            # Log multiple metrics at once
            await run.log_metrics({"accuracy": 0.92, "f1": 0.89}, step=0)

            # Tags use add_tag (renamed from set_tag in 1.0)
            await run.add_tag("team", "ml-platform")
            await run.add_tag("dataset", "churn_v2")

        baseline_run_id = run.run_id

        # ── 3. Verify the run was marked FINISHED ────────────────────────
        # Status vocabulary in 1.0+ is exactly {RUNNING, FINISHED, FAILED, KILLED}
        # — the legacy SUCCESS / COMPLETED tokens were hard-migrated away.

        completed = await tracker.get_run(baseline_run_id)
        assert completed.status == "FINISHED"
        assert completed.wall_clock_end is not None
        assert completed.params["model_class"] == "RandomForestClassifier"
        assert completed.params["n_estimators"] == "100"
        assert completed.error_type is None

        # ── 4. A second run for comparison ──────────────────────────────

        async with tracker.track(
            experiment="churn_experiment", run_name="improved"
        ) as run2:
            await run2.log_params(
                {
                    "model_class": "GradientBoostingClassifier",
                    "n_estimators": "200",
                    "learning_rate": "0.05",
                }
            )
            await run2.log_metrics({"accuracy": 0.95, "f1": 0.93})

        improved_run_id = run2.run_id

        # ── 5. Failed run — exception inside the block marks FAILED ─────

        failed_run_id: str | None = None
        try:
            async with tracker.track(
                experiment="churn_experiment", run_name="broken"
            ) as run3:
                failed_run_id = run3.run_id
                await run3.log_params({"model_class": "BrokenModel"})
                raise RuntimeError("Training crashed")
        except RuntimeError:
            pass  # Expected

        assert failed_run_id is not None
        failed = await tracker.get_run(failed_run_id)
        assert failed.status == "FAILED"
        assert failed.error_type == "RuntimeError"
        assert "Training crashed" in failed.error_message

        # ── 6. Manual lifecycle: start_run / end_run (no async-with) ────
        # When you cannot wrap the work in an ``async with`` block (e.g. a
        # long-running training driver that hands the run id to other code),
        # ``start_run`` returns the run object so you can pair it with a
        # later ``end_run``.  The caller MUST end every run they start.

        manual_run = await tracker.start_run(
            experiment="churn_experiment", run_name="manual"
        )
        await manual_run.log_metric("accuracy", 0.88)
        await tracker.end_run(manual_run, status="FINISHED")

        manual_record = await tracker.get_run(manual_run.run_id)
        assert manual_record.status == "FINISHED"

        # ── 7. List experiments — returns a polars DataFrame ────────────

        experiments_df = await tracker.list_experiments()
        assert experiments_df.height >= 1
        exp_names = experiments_df.get_column("experiment").to_list()
        assert "churn_experiment" in exp_names

        # ── 8. List runs for an experiment — returns a polars DataFrame ─

        all_runs = await tracker.list_runs(experiment="churn_experiment")
        # baseline + improved + broken + manual
        assert all_runs.height == 4

        finished = all_runs.filter(all_runs["status"] == "FINISHED")
        assert finished.height == 3  # baseline, improved, manual

        failed_only = all_runs.filter(all_runs["status"] == "FAILED")
        assert failed_only.height == 1

        # ── 9. search_runs() — MLflow-compatible filter strings ─────────
        # Filter expressions reference ``params.<key>`` or ``metrics.<key>``.

        search_df = await tracker.search_runs(
            filter="params.model_class = 'GradientBoostingClassifier'",
        )
        assert search_df.height == 1
        assert search_df.get_column("run_id").to_list() == [improved_run_id]

        # ── 10. list_metrics() — step-based history per run ─────────────

        history = await tracker.list_metrics(baseline_run_id)
        # 3 train_loss steps + 1 accuracy + 1 f1 = 5 metric rows
        loss_history = history.filter(history["key"] == "train_loss").sort("step")
        assert loss_history.height == 3
        steps = loss_history.get_column("step").to_list()
        values = loss_history.get_column("value").to_list()
        assert steps == [0, 1, 2]
        assert math.isclose(values[0], 0.65)
        assert math.isclose(values[1], 0.42)
        assert math.isclose(values[2], 0.28)

        # ── 11. diff_runs() — pairwise param + metric comparison ────────

        diff = await tracker.diff_runs(baseline_run_id, improved_run_id)
        assert diff.run_id_a == baseline_run_id
        assert diff.run_id_b == improved_run_id
        # model_class changed between baseline and improved
        assert diff.params["model_class"].changed is True
        # accuracy improved by 0.95 - 0.92 = 0.03
        assert math.isclose(diff.metrics["accuracy"].delta, 0.03)
        # Both runs share the same kailash-ml version, host, etc.
        assert diff.environment["kailash_ml_version"].changed is False

        # ── 12. Edge case: NaN/Inf metrics rejected at log time ─────────

        async with tracker.track(
            experiment="churn_experiment", run_name="finiteness_check"
        ) as run4:
            try:
                await run4.log_metric("loss", float("nan"))
                raise AssertionError("Should reject NaN metric")
            except ValueError:
                pass  # Expected: 1.1.1 raises MetricValueError(ValueError)

            try:
                await run4.log_metric("loss", float("inf"))
                raise AssertionError("Should reject Inf metric")
            except ValueError:
                pass  # Expected

        # ── 13. Clean shutdown ──────────────────────────────────────────
        # ExperimentTracker holds an aiosqlite connection pool — close it
        # explicitly so background workers exit cleanly.

        await tracker.close()

    print("PASS: 05-ml/06_experiment_tracker")


asyncio.run(main())
