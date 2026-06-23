#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP04 Assessment Task 4 — Neural Network Foundations.

Usage:
    python grader.py starter.py     # grade your attempt
    python grader.py solution.py    # verify the reference passes

The grader regenerates the concentric-circles dataset and the same train/test
split, recomputes the submission's accuracy from its predictions, and certifies
that the problem genuinely requires a non-linear model (the classes share a
centre and an independent linear classifier is stuck near chance). All ten
checks must pass.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

SEED = 20260404
N = 800
SPLIT = 600
FEATURES = ["x1", "x2"]
TARGET = "label"
N_TEST = N - SPLIT
ACC_FLOOR = 0.90
LINEAR_CEILING = 0.70
CENTROID_MAX = 0.50


def _reference() -> pl.DataFrame:
    """Regenerate the exact concentric-circles dataset."""
    rng = np.random.default_rng(SEED)
    m = N // 2

    def ring(radius: float, noise: float) -> np.ndarray:
        theta = rng.uniform(0, 2 * np.pi, m)
        r = radius + rng.normal(0, noise, m)
        return np.c_[r * np.cos(theta), r * np.sin(theta)]

    X = np.vstack([ring(1.0, 0.18), ring(3.0, 0.30)])
    y = np.r_[np.zeros(m, dtype=int), np.ones(m, dtype=int)]
    perm = rng.permutation(N)
    X, y = X[perm], y[perm]
    return pl.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "label": y})


def _linear_baseline_test_acc(df: pl.DataFrame) -> float:
    """Pure-numpy least-squares linear classifier — the linear ceiling oracle."""
    tr, te = df.head(SPLIT), df.tail(N_TEST)
    Xtr = tr.select(FEATURES).to_numpy()
    ytr = tr[TARGET].to_numpy().astype(float)
    Xte = te.select(FEATURES).to_numpy()
    yte = te[TARGET].to_numpy()
    A = np.c_[Xtr, np.ones(len(Xtr))]
    w, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    pred = ((np.c_[Xte, np.ones(len(Xte))] @ w) > 0.5).astype(int)
    return float((pred == yte).mean())


def _centroid_distance(df: pl.DataFrame) -> float:
    """Inter-class centroid distance in z-space (≈0 ⇒ linearly inseparable)."""
    X = df.select(FEATURES).to_numpy()
    y = df[TARGET].to_numpy()
    Xz = (X - X.mean(0)) / X.std(0)
    return float(np.linalg.norm(Xz[y == 0].mean(0) - Xz[y == 1].mean(0)))


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_t4", path)
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

    c["keys_present"] = all(
        k in r for k in ("test_predictions", "test_accuracy", "train_accuracy")
    )
    if not c["keys_present"]:
        return _finalize(score)

    preds = r["test_predictions"]
    c["test_predictions_length"] = isinstance(preds, list) and len(preds) == N_TEST
    if not c["test_predictions_length"]:
        return _finalize(score)

    yhat = np.asarray(preds, dtype=int)
    c["predictions_binary"] = bool(np.isin(yhat, [0, 1]).all())
    c["predictions_non_degenerate"] = len(np.unique(yhat)) == 2

    df = _reference()
    y_test = df.tail(N_TEST)[TARGET].to_numpy()
    recomputed_acc = float((yhat == y_test).mean())
    c["test_accuracy_above_floor"] = recomputed_acc >= ACC_FLOOR

    try:
        c["accuracy_self_report_honest"] = (
            abs(float(r["test_accuracy"]) - recomputed_acc) <= 0.03
        )
    except Exception:
        c["accuracy_self_report_honest"] = False

    try:
        ta = float(r["train_accuracy"])
        c["train_accuracy_sane"] = ACC_FLOOR <= ta <= 1.0
    except Exception:
        c["train_accuracy_sane"] = False

    # Certify the task genuinely needs a non-linear model.
    c["problem_is_nonlinear"] = _centroid_distance(df) < CENTROID_MAX
    c["beats_linear_ceiling"] = (
        recomputed_acc >= ACC_FLOOR and _linear_baseline_test_acc(df) < LINEAR_CEILING
    )

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
