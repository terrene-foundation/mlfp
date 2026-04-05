# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — ML / DriftMonitor
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Detect feature drift using PSI and KS-test, monitor
#            performance degradation, and configure scheduled monitoring.
#            Covers set_reference(), check_drift(), check_performance(),
#            schedule_monitoring(), and DriftSpec.
# LEVEL: Intermediate
# PARITY: Python-only (kailash-ml has no Rust counterpart)
# VALIDATES: DriftMonitor, DriftReport, DriftSpec, FeatureDriftResult,
#            PerformanceDegradationReport — set_reference(), check_drift(),
#            check_performance(), get_drift_history(), schedule_monitoring()
#
# Run: uv run python textbook/python/05-ml/12_drift_monitor.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from datetime import timedelta

import polars as pl

from kailash.db.connection import ConnectionManager
from kailash_ml.engines.drift_monitor import (
    DriftMonitor,
    DriftReport,
    DriftSpec,
    FeatureDriftResult,
    PerformanceDegradationReport,
)


async def main() -> None:
    # ── 1. Set up ConnectionManager + DriftMonitor ──────────────────────

    conn = ConnectionManager("sqlite:///:memory:")
    await conn.initialize()

    monitor = DriftMonitor(
        conn,
        psi_threshold=0.2,  # PSI > 0.2 triggers drift alert
        ks_threshold=0.05,  # KS p-value < 0.05 triggers drift alert
        performance_threshold=0.1,  # Absolute metric degradation threshold
    )
    assert isinstance(monitor, DriftMonitor)

    # ── 2. Create reference data (training distribution) ─────────────────

    reference_df = pl.DataFrame(
        {
            "age": [25.0 + (i % 50) for i in range(500)],
            "income": [30000.0 + i * 100.0 for i in range(500)],
            "tenure": [float(1 + i % 24) for i in range(500)],
        }
    )

    # ── 3. set_reference() — store per-feature reference distribution ────

    await monitor.set_reference(
        model_name="churn_model",
        reference_data=reference_df,
        feature_columns=["age", "income", "tenure"],
    )

    # ── 4. check_drift() — no drift (same distribution) ──────────────────

    # Current data drawn from same distribution
    stable_df = pl.DataFrame(
        {
            "age": [26.0 + (i % 50) for i in range(300)],
            "income": [30500.0 + i * 100.0 for i in range(300)],
            "tenure": [float(2 + i % 24) for i in range(300)],
        }
    )

    report = await monitor.check_drift("churn_model", stable_df)

    assert isinstance(report, DriftReport)
    assert report.model_name == "churn_model"
    assert len(report.feature_results) == 3
    assert report.sample_size_reference == 500
    assert report.sample_size_current == 300
    assert report.checked_at is not None
    assert report.reference_set_at is not None

    # Each feature has drift statistics
    for fr in report.feature_results:
        assert isinstance(fr, FeatureDriftResult)
        assert fr.psi >= 0.0
        assert fr.ks_statistic >= 0.0
        assert 0.0 <= fr.ks_pvalue <= 1.0
        assert fr.drift_type in ("none", "moderate", "severe")

    # Stable data should not trigger drift
    assert report.overall_severity in ("none", "moderate")

    # drifted_features property
    assert isinstance(report.drifted_features, list)

    # ── 5. check_drift() — significant drift (shifted distribution) ──────

    # Dramatically different distribution
    drifted_df = pl.DataFrame(
        {
            "age": [80.0 + (i % 10) for i in range(300)],  # Much older
            "income": [150000.0 + i * 500.0 for i in range(300)],  # Much higher
            "tenure": [float(50 + i % 5) for i in range(300)],  # Much longer
        }
    )

    drift_report = await monitor.check_drift("churn_model", drifted_df)

    assert drift_report.overall_drift_detected is True
    assert drift_report.overall_severity in ("moderate", "severe")
    assert len(drift_report.drifted_features) > 0

    # At least some features should show drift
    drifted_results = [f for f in drift_report.feature_results if f.drift_detected]
    assert len(drifted_results) > 0

    # PSI should be high for shifted distributions
    for fr in drifted_results:
        assert fr.psi > 0.1

    # ── 6. get_drift_history() — stored reports ──────────────────────────

    history = await monitor.get_drift_history("churn_model", limit=10)
    assert isinstance(history, list)
    assert len(history) >= 2  # stable + drifted checks

    # ── 7. check_performance() — no baseline stored yet ──────────────────
    # First call stores current metrics as the baseline.

    predictions = pl.DataFrame({"pred": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})
    actuals = pl.DataFrame({"actual": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})

    perf_report = await monitor.check_performance(
        "churn_model",
        predictions,
        actuals,
    )

    assert isinstance(perf_report, PerformanceDegradationReport)
    assert perf_report.model_name == "churn_model"
    assert isinstance(perf_report.baseline_metrics, dict)
    assert isinstance(perf_report.current_metrics, dict)
    assert "accuracy" in perf_report.current_metrics
    assert perf_report.current_metrics["accuracy"] == 1.0  # Perfect predictions
    assert perf_report.degraded is False

    # ── 8. check_performance() — with degradation ────────────────────────

    # Provide explicit baseline showing higher performance
    degraded_preds = pl.DataFrame({"pred": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]})
    degraded_actuals = pl.DataFrame({"actual": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})

    degraded_report = await monitor.check_performance(
        "churn_model",
        degraded_preds,
        degraded_actuals,
        baseline_metrics={"accuracy": 0.95, "f1": 0.93},
    )

    assert degraded_report.degraded is True
    assert degraded_report.degradation["accuracy"] > 0  # Positive = degraded

    # ── 9. DriftSpec — custom thresholds for scheduled monitoring ────────

    spec = DriftSpec(
        feature_columns=["age", "income"],
        psi_threshold=0.15,
        ks_threshold=0.01,
    )

    assert spec.psi_threshold == 0.15
    assert spec.ks_threshold == 0.01
    assert spec.feature_columns == ["age", "income"]

    # ── 10. schedule_monitoring() + cancel_monitoring() ──────────────────
    # schedule_monitoring() creates an asyncio background task.

    call_count = 0

    async def get_current_data() -> pl.DataFrame:
        nonlocal call_count
        call_count += 1
        return stable_df

    await monitor.schedule_monitoring(
        model_name="churn_model",
        interval=timedelta(seconds=60),  # 60s interval
        data_fn=get_current_data,
        spec=spec,
    )

    # Verify schedule is active
    assert "churn_model" in monitor.active_schedules

    # Cancel it (we do not want to wait 60s in a tutorial)
    cancelled = await monitor.cancel_monitoring("churn_model")
    assert cancelled is True
    assert "churn_model" not in monitor.active_schedules

    # Cancel non-existent returns False
    cancelled_again = await monitor.cancel_monitoring("nonexistent_model")
    assert cancelled_again is False

    # ── 11. shutdown() — cancel all schedules ────────────────────────────

    await monitor.schedule_monitoring(
        "churn_model", timedelta(seconds=60), get_current_data
    )
    await monitor.shutdown()
    assert len(monitor.active_schedules) == 0

    # ── 12. Serialization round-trips ────────────────────────────────────

    # FeatureDriftResult
    fdr_dict = drift_report.feature_results[0].to_dict()
    fdr_restored = FeatureDriftResult.from_dict(fdr_dict)
    assert fdr_restored.feature_name == drift_report.feature_results[0].feature_name
    assert fdr_restored.psi == drift_report.feature_results[0].psi

    # DriftReport
    dr_dict = drift_report.to_dict()
    dr_restored = DriftReport.from_dict(dr_dict)
    assert dr_restored.model_name == drift_report.model_name
    assert dr_restored.overall_drift_detected == drift_report.overall_drift_detected
    assert len(dr_restored.feature_results) == len(drift_report.feature_results)

    # PerformanceDegradationReport
    pdr_dict = degraded_report.to_dict()
    pdr_restored = PerformanceDegradationReport.from_dict(pdr_dict)
    assert pdr_restored.degraded == degraded_report.degraded

    # DriftSpec
    ds_dict = spec.to_dict()
    ds_restored = DriftSpec.from_dict(ds_dict)
    assert ds_restored.feature_columns == spec.feature_columns
    assert ds_restored.psi_threshold == spec.psi_threshold

    # ── 13. Edge case: check_drift without reference ─────────────────────

    try:
        await monitor.check_drift("no_reference_model", stable_df)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "no reference" in str(e).lower()

    # ── 14. Edge case: schedule without reference ────────────────────────

    try:
        await monitor.schedule_monitoring(
            "no_reference_model",
            timedelta(seconds=60),
            get_current_data,
        )
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "no reference" in str(e).lower()

    # ── 15. Edge case: too-short monitoring interval ─────────────────────

    try:
        await monitor.schedule_monitoring(
            "churn_model",
            timedelta(seconds=0),
            get_current_data,
        )
        assert False, "Should raise ValueError for interval < 1s"
    except ValueError:
        pass  # Expected

    # ── 16. Edge case: non-finite threshold values ───────────────────────

    try:
        DriftMonitor(conn, psi_threshold=float("nan"))
        assert False, "Should reject NaN threshold"
    except ValueError:
        pass  # Expected

    try:
        DriftMonitor(conn, ks_threshold=float("inf"))
        assert False, "Should reject Inf threshold"
    except ValueError:
        pass  # Expected

    # ── 17. Clean up ─────────────────────────────────────────────────────

    await conn.close()

    print("PASS: 05-ml/12_drift_monitor")


asyncio.run(main())
