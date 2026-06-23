#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP05 Assessment Task 3 — GRU Time-Series Forecasting.

Usage:
    python grader.py starter.py     # grade your attempt
    python grader.py solution.py    # verify the reference passes

The grader re-derives the exact same series and test split, re-runs the returned
model on it (so a hand-tuned `test_pred` array fails the anti-faking check), and
verifies the model is genuinely recurrent and beats the naive last-value baseline.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SEQ_LEN = 20
SEED = 13
RATIO_CEILING = 0.97  # model MSE must be <= 0.97 * naive MSE


def _reference_dataset():
    """Re-derive (X_test, y_test, naive_pred) identically to the starter."""
    rng = np.random.default_rng(SEED)
    n = 3000
    a1, a2 = 1.35, -0.55
    series = np.zeros(n, dtype=np.float64)
    noise = rng.normal(0.0, 0.5, size=n)
    for t in range(2, n):
        season = 0.6 * np.sin(2.0 * np.pi * t / 11.0)
        series[t] = a1 * series[t - 1] + a2 * series[t - 2] + season + noise[t]
    series = series.astype(np.float32)

    xs, ys = [], []
    for i in range(len(series) - SEQ_LEN - 1):
        xs.append(series[i : i + SEQ_LEN])
        ys.append(series[i + SEQ_LEN])
    X = np.array(xs, dtype=np.float32)[:, :, None]
    y = np.array(ys, dtype=np.float32)
    split = int(len(X) * 0.8)
    X_test = X[split:]
    y_test = y[split:]
    naive_pred = X_test[:, -1, 0].astype(np.float32)
    return X_test, y_test, naive_pred


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_task3", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def grade(student_path: Path) -> dict:
    score: dict = {"passed": False, "checks": {}, "total": 0, "max": 0}
    try:
        student = load_student_module(student_path)
    except Exception as e:
        score["error"] = f"Failed to import: {type(e).__name__}: {e}"
        return score
    if not hasattr(student, "solve"):
        score["error"] = "Module does not define a solve() function"
        return score
    try:
        r = student.solve()
    except Exception as e:
        score["error"] = f"Runtime error in solve(): {type(e).__name__}: {e}"
        return score

    c = score["checks"]
    c["returns_dict"] = isinstance(r, dict)
    if not c["returns_dict"]:
        return _finalize(score)

    required = {"model", "test_pred", "y_test", "naive_pred", "uses_recurrent"}
    c["has_required_keys"] = required.issubset(r.keys())
    if not c["has_required_keys"]:
        return _finalize(score)

    model = r["model"]
    c["model_is_nn_module"] = isinstance(model, nn.Module)

    # Genuine recurrent net: at least one GRU/LSTM/RNN, declared flag matches.
    if c["model_is_nn_module"]:
        actual_rec = any(
            isinstance(m, (nn.GRU, nn.LSTM, nn.RNN)) for m in model.modules()
        )
        c["is_recurrent"] = actual_rec and bool(r["uses_recurrent"]) == actual_rec
    else:
        c["is_recurrent"] = False

    X_test_ref, y_test_ref, naive_ref = _reference_dataset()

    try:
        test_pred = np.asarray(r["test_pred"], dtype=np.float32).ravel()
        y_test = np.asarray(r["y_test"], dtype=np.float32).ravel()
        naive_pred = np.asarray(r["naive_pred"], dtype=np.float32).ravel()
        c["shapes_match"] = (
            test_pred.shape == y_test.shape == naive_pred.shape == y_test_ref.shape
        )
    except Exception:
        c["shapes_match"] = False
        test_pred = y_test = naive_pred = np.array([])

    # Model beats the naive baseline by a clear margin.
    if c["shapes_match"]:
        model_mse = float(((test_pred - y_test) ** 2).mean())
        naive_mse = float(((naive_pred - y_test) ** 2).mean())
        c["beats_naive_baseline"] = bool(
            naive_mse > 0 and model_mse <= RATIO_CEILING * naive_mse
        )
    else:
        c["beats_naive_baseline"] = False

    # Anti-faking: re-run the returned model on the re-derived X_test and require it
    # to reproduce the submitted predictions.
    if c["model_is_nn_module"] and c["shapes_match"]:
        try:
            model.eval()
            with torch.no_grad():
                rerun = model(torch.tensor(X_test_ref)).cpu().numpy().ravel()
            max_abs = float(np.max(np.abs(rerun - test_pred)))
            tol = 1e-3 * (float(np.std(y_test_ref)) + 1e-9)
            c["preds_reproduced_by_model"] = bool(max_abs <= max(tol, 1e-4))
        except Exception:
            c["preds_reproduced_by_model"] = False
    else:
        c["preds_reproduced_by_model"] = False

    # Naive baseline is the correct last-value baseline (matches re-derivation).
    if c["shapes_match"]:
        c["naive_baseline_correct"] = bool(
            np.allclose(naive_pred, naive_ref, atol=1e-5)
        )
    else:
        c["naive_baseline_correct"] = False

    return _finalize(score)


def _finalize(score: dict) -> dict:
    score["total"] = sum(1 for v in score["checks"].values() if v)
    score["max"] = len(score["checks"])
    score["passed"] = score["max"] > 0 and score["total"] == score["max"]
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("student", type=Path)
    args = parser.parse_args()
    result = grade(args.student)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)
