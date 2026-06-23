#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP04 Assessment Task 2 — Dim Reduction & Anomaly.

Usage:
    python grader.py starter.py     # grade your attempt
    python grader.py solution.py    # verify the reference passes

The grader regenerates the planted sensor matrix (with the hidden anomaly flags
it never gave the student), re-derives the PCA intrinsic dimensionality and the
anomaly-detection ROC-AUC, and checks the submission against strict invariants.
All ten checks must pass.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from kailash_ml.engines.dim_reduction import DimReductionEngine

SEED = 20260402
N_NORMAL = 975
N_ANOM = 25
D = 24
K_LATENT = 3
N_TOTAL = N_NORMAL + N_ANOM
AUC_FLOOR = 0.85
PRECISION_FLOOR = 0.5


def _reference() -> tuple[pl.DataFrame, np.ndarray]:
    """Regenerate the sensor matrix and the hidden anomaly flags (1 = anomaly)."""
    rng = np.random.default_rng(SEED)
    Z = rng.normal(0, 1, (N_NORMAL, K_LATENT))
    W = rng.normal(0, 1, (K_LATENT, D)) * 3.5
    X_normal = Z @ W + rng.normal(0, 0.5, (N_NORMAL, D))
    X_anom = rng.normal(12.0, 4.0, (N_ANOM, D)) * rng.choice([-1, 1], (N_ANOM, D))
    X = np.vstack([X_normal, X_anom])
    y = np.r_[np.zeros(N_NORMAL, int), np.ones(N_ANOM, int)]
    perm = rng.permutation(X.shape[0])
    X, y = X[perm], y[perm]
    cols = [f"f{i:02d}" for i in range(D)]
    return pl.DataFrame({c: X[:, j] for j, c in enumerate(cols)}), y


def _reference_pca() -> tuple[int, float]:
    """Independent PCA reference: (n_components_90, reconstruction_error)."""
    df, _ = _reference()
    dre = DimReductionEngine()
    full = dre.reduce(df, algorithm="pca", n_components=df.width)
    cum = np.cumsum(np.asarray(full.explained_variance_ratio))
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    recon = float(
        dre.reduce(df, algorithm="pca", n_components=n90).reconstruction_error
    )
    return n90, recon


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_t2", path)
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

    required = (
        "n_components_90",
        "reconstruction_error",
        "anomaly_scores",
        "anomaly_labels",
        "n_anomalies",
    )
    c["keys_present"] = all(k in r for k in required)
    if not c["keys_present"]:
        return _finalize(score)

    _, y_true = _reference()
    ref_n90, ref_recon = _reference_pca()

    c["n_components_90_correct"] = r["n_components_90"] == ref_n90
    c["compression_is_real"] = (
        isinstance(r["n_components_90"], int) and 0 < r["n_components_90"] < D
    )

    try:
        re_val = float(r["reconstruction_error"])
        c["reconstruction_error_matches"] = re_val > 0.0 and abs(
            re_val - ref_recon
        ) <= max(0.02, 0.02 * ref_recon)
    except Exception:
        c["reconstruction_error_matches"] = False

    scores = r["anomaly_scores"]
    labels = r["anomaly_labels"]
    c["scores_length_correct"] = isinstance(scores, list) and len(scores) == N_TOTAL
    c["labels_length_correct"] = isinstance(labels, list) and len(labels) == N_TOTAL
    if not (c["scores_length_correct"] and c["labels_length_correct"]):
        return _finalize(score)

    sc = np.asarray(scores, dtype=float)
    lab = np.asarray(labels, dtype=int)

    try:
        auc = roc_auc_score(y_true, sc)
        c["anomaly_auc_above_floor"] = bool(sc.std() > 0 and auc >= AUC_FLOOR)
    except Exception:
        c["anomaly_auc_above_floor"] = False

    flagged = lab == 1
    c["n_anomalies_consistent"] = (
        int(r["n_anomalies"]) == int(flagged.sum()) and 10 <= int(flagged.sum()) <= 60
    )

    try:
        prec = float(y_true[flagged].mean()) if flagged.sum() else 0.0
        c["flagged_precision_above_floor"] = prec >= PRECISION_FLOOR
    except Exception:
        c["flagged_precision_above_floor"] = False

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
