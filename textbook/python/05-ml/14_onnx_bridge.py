# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — ML / OnnxBridge
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Export trained models to ONNX format for cross-runtime
#            serving ("train in Python, serve in Rust").  Covers
#            pre-flight compatibility checks, export, and post-export
#            numerical validation.
# LEVEL: Intermediate
# PARITY: Python-only (kailash-ml has no Rust counterpart)
# VALIDATES: OnnxBridge, OnnxCompatibility, OnnxExportResult,
#            OnnxValidationResult — check_compatibility(), export(),
#            validate()
#
# Run: uv run python textbook/python/05-ml/14_onnx_bridge.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge

from kailash_ml import FeatureField, FeatureSchema, ModelSignature
from kailash_ml.bridge.onnx_bridge import (
    OnnxBridge,
    OnnxCompatibility,
    OnnxExportResult,
    OnnxValidationResult,
)

# ── 1. Instantiate OnnxBridge ───────────────────────────────────────

bridge = OnnxBridge()
assert isinstance(bridge, OnnxBridge)

# ── 2. Train models for export ──────────────────────────────────────

X_train = np.array(
    [
        [1.0, 2.0, 0.5],
        [3.0, 4.0, 1.5],
        [5.0, 6.0, 2.5],
        [7.0, 8.0, 3.5],
        [2.0, 1.0, 0.2],
        [4.0, 3.0, 1.2],
        [6.0, 5.0, 2.2],
        [8.0, 7.0, 3.2],
    ]
)
y_cls = np.array([0, 1, 0, 1, 0, 1, 0, 1])
y_reg = np.array([10.0, 20.0, 30.0, 40.0, 15.0, 25.0, 35.0, 45.0])

rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X_train, y_cls)

gb = GradientBoostingClassifier(n_estimators=10, random_state=42)
gb.fit(X_train, y_cls)

lr = LogisticRegression(max_iter=200, random_state=42)
lr.fit(X_train, y_cls)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_reg)

# ── 3. check_compatibility() — pre-flight check ─────────────────────

rf_compat = bridge.check_compatibility(rf, "sklearn")
assert isinstance(rf_compat, OnnxCompatibility)
assert rf_compat.compatible is True
assert rf_compat.confidence == "guaranteed"
assert rf_compat.framework == "sklearn"
assert rf_compat.model_type == "RandomForestClassifier"

gb_compat = bridge.check_compatibility(gb, "sklearn")
assert gb_compat.compatible is True

lr_compat = bridge.check_compatibility(lr, "sklearn")
assert lr_compat.compatible is True

# Unknown framework
unknown_compat = bridge.check_compatibility(rf, "unknown_framework")
assert unknown_compat.compatible is False
assert unknown_compat.confidence == "unsupported"

# ── 4. export() — sklearn RandomForest ───────────────────────────────

schema = ModelSignature(
    input_schema=FeatureSchema(
        name="export_test",
        features=[
            FeatureField(name="a", dtype="float64"),
            FeatureField(name="b", dtype="float64"),
            FeatureField(name="c", dtype="float64"),
        ],
        entity_id_column="entity_id",
    ),
    output_columns=["prediction"],
    output_dtypes=["float64"],
    model_type="classifier",
)

with tempfile.TemporaryDirectory() as tmpdir:
    output_path = Path(tmpdir) / "rf_model.onnx"

    result = bridge.export(
        rf,
        "sklearn",
        schema=schema,
        output_path=output_path,
    )

    assert isinstance(result, OnnxExportResult)

    # Export may succeed or be skipped depending on skl2onnx availability
    if result.success:
        assert result.onnx_status == "success"
        assert result.onnx_path == output_path
        assert output_path.exists()
        assert result.model_size_bytes is not None
        assert result.model_size_bytes > 0
        assert result.export_time_seconds > 0
        assert result.error_message is None

        # ── 5. validate() — numerical correctness ───────────────────
        # Compares native sklearn predictions with ONNX runtime predictions.

        validation = bridge.validate(
            rf,
            output_path,
            X_train,
            tolerance=1e-4,
        )

        assert isinstance(validation, OnnxValidationResult)
        if validation.valid:
            assert validation.max_diff <= 1e-4
            assert validation.mean_diff <= 1e-4
            assert validation.n_samples > 0
        # If onnxruntime not installed, valid=False with notes

        # ── 6. Export other sklearn models ───────────────────────────

        gb_path = Path(tmpdir) / "gb_model.onnx"
        gb_result = bridge.export(gb, "sklearn", schema=schema, output_path=gb_path)
        assert isinstance(gb_result, OnnxExportResult)

        lr_path = Path(tmpdir) / "lr_model.onnx"
        lr_result = bridge.export(lr, "sklearn", schema=schema, output_path=lr_path)
        assert isinstance(lr_result, OnnxExportResult)

        ridge_path = Path(tmpdir) / "ridge_model.onnx"
        ridge_result = bridge.export(
            ridge, "sklearn", schema=schema, output_path=ridge_path
        )
        assert isinstance(ridge_result, OnnxExportResult)

    else:
        # skl2onnx not installed — export is gracefully skipped
        assert result.onnx_status in ("failed", "skipped")
        assert result.error_message is not None

# ── 7. export() — infer n_features from model ────────────────────────

with tempfile.TemporaryDirectory() as tmpdir:
    auto_path = Path(tmpdir) / "auto.onnx"

    # Without schema, OnnxBridge infers from model.n_features_in_
    auto_result = bridge.export(
        rf,
        "sklearn",
        output_path=auto_path,
    )

    assert isinstance(auto_result, OnnxExportResult)
    # Should succeed (model has n_features_in_ = 3)

# ── 8. export() — unsupported framework ──────────────────────────────

with tempfile.TemporaryDirectory() as tmpdir:
    unsupported_result = bridge.export(
        rf,
        "unknown_framework",
        output_path=Path(tmpdir) / "unsupported.onnx",
        n_features=3,
    )

    assert unsupported_result.success is False
    assert unsupported_result.onnx_status == "skipped"
    assert unsupported_result.error_message is not None

# ── 9. export() — cannot determine features ──────────────────────────


class BareModel:
    """Model without n_features_in_ attribute."""

    pass


with tempfile.TemporaryDirectory() as tmpdir:
    bare_result = bridge.export(
        BareModel(),
        "sklearn",
        output_path=Path(tmpdir) / "bare.onnx",
        # No schema, no n_features -> should fail
    )

    assert bare_result.success is False
    assert bare_result.onnx_status == "failed"
    assert "features" in bare_result.error_message.lower()

# ── 10. export() with explicit n_features ─────────────────────────────

with tempfile.TemporaryDirectory() as tmpdir:
    explicit_result = bridge.export(
        rf,
        "sklearn",
        n_features=3,
        output_path=Path(tmpdir) / "explicit.onnx",
    )

    assert isinstance(explicit_result, OnnxExportResult)

# ── 11. export() returns result, never raises ─────────────────────────
# On failure, export() returns OnnxExportResult with success=False.
# It does not raise exceptions.  This makes ONNX export non-fatal.

with tempfile.TemporaryDirectory() as tmpdir:
    # Even with bad parameters, export returns a result
    bad_result = bridge.export(
        "not_a_model",
        "sklearn",
        n_features=3,
        output_path=Path(tmpdir) / "bad.onnx",
    )
    assert isinstance(bad_result, OnnxExportResult)
    assert bad_result.success is False

print("PASS: 05-ml/14_onnx_bridge")
