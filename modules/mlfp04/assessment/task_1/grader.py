#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP04 Assessment Task 1 — Customer Segmentation.

Usage:
    python grader.py starter.py     # grade your attempt
    python grader.py solution.py    # verify the reference passes

The grader regenerates the planted cohort (with the hidden persona labels it
never gave the student), re-derives the z-scored space, and checks the
submission's partition against strict recovery invariants. All ten checks must
pass.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
)

SEED = 20260401
N_EXPECTED = 1200
SIL_FLOOR = 0.55
ARI_FLOOR = 0.90
AMI_FLOOR = 0.85


def _reference() -> tuple[np.ndarray, np.ndarray]:
    """Regenerate the planted cohort: returns (z-scored X, hidden labels)."""
    rng = np.random.default_rng(SEED)
    centers = np.array(
        [
            [5.0, 50.0, 2000.0, 60.0, 180.0],
            [40.0, 10.0, 400.0, 12.0, 60.0],
            [90.0, 3.0, 150.0, 48.0, 45.0],
            [15.0, 25.0, 1200.0, 36.0, 220.0],
        ]
    )
    spreads = np.array([3.0, 3.0, 120.0, 5.0, 18.0])
    sizes = [320, 300, 280, 300]
    blocks, ys = [], []
    for i, (c, n) in enumerate(zip(centers, sizes)):
        blocks.append(c + rng.normal(0, 1, (n, 5)) * spreads)
        ys.append(np.full(n, i))
    X = np.vstack(blocks)
    y = np.concatenate(ys)
    perm = rng.permutation(X.shape[0])
    X, y = X[perm], y[perm]
    # z-score (population std, ddof=0 to match polars default std? polars uses
    # ddof=1; silhouette is scale-invariant per-column up to a constant, and we
    # only need the SAME space the student clustered in — use ddof=1 to mirror
    # Polars .std()).
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    Xz = (X - mu) / sd
    return Xz, y


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_t1", path)
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

    c["keys_present"] = all(k in r for k in ("labels", "n_clusters", "silhouette"))
    if not c["keys_present"]:
        return _finalize(score)

    labels = r["labels"]
    c["labels_length_1200"] = isinstance(labels, list) and len(labels) == N_EXPECTED
    if not c["labels_length_1200"]:
        return _finalize(score)

    Xz, y_true = _reference()
    lab = np.asarray(labels)

    c["n_clusters_is_4"] = r["n_clusters"] == 4

    uniq = np.unique(lab)
    sizes = np.bincount(lab.astype(int), minlength=int(uniq.max()) + 1)
    c["four_nonempty_clusters"] = len(uniq) == 4 and bool((sizes[uniq] > 0).all())

    try:
        c["labels_valid_ints"] = (
            bool(
                np.issubdtype(lab.dtype, np.integer)
                or np.allclose(lab, lab.astype(int))
            )
            and int(lab.min()) >= 0
        )
    except Exception:
        c["labels_valid_ints"] = False

    try:
        sil_grader = float(silhouette_score(Xz, lab))
        c["silhouette_above_floor"] = sil_grader >= SIL_FLOOR
    except Exception:
        sil_grader = float("nan")
        c["silhouette_above_floor"] = False

    try:
        c["silhouette_self_report_honest"] = (
            abs(float(r["silhouette"]) - sil_grader) <= 0.05
        )
    except Exception:
        c["silhouette_self_report_honest"] = False

    try:
        c["ari_recovers_personas"] = adjusted_rand_score(y_true, lab) >= ARI_FLOOR
    except Exception:
        c["ari_recovers_personas"] = False

    try:
        c["ami_recovers_personas"] = (
            adjusted_mutual_info_score(y_true, lab) >= AMI_FLOOR
        )
    except Exception:
        c["ami_recovers_personas"] = False

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
