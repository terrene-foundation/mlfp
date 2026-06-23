#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP05 Assessment Task 1 — Autoencoder Anomaly Detection.

Usage:
    python grader.py starter.py     # grade your attempt
    python grader.py solution.py    # verify the reference passes

The grader regenerates fresh healthy/anomalous telemetry independently and re-runs
the returned model on it, so a submission that returns a pre-baked `scores` array
without a genuinely-trained autoencoder fails the anti-faking check.
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
from sklearn.metrics import roc_auc_score

INPUT_DIM = 12
AUC_FLOOR = 0.90
SEP_FLOOR = 1.5


def _fresh_eval_batches() -> tuple[np.ndarray, np.ndarray]:
    """Independent healthy + anomalous batches with a DIFFERENT seed.

    Used to verify the returned model itself ranks anomalies above healthy —
    a faked `scores` array cannot pass this.
    """
    rng = np.random.default_rng(20260624)
    basis = rng.normal(size=(3, INPUT_DIM))
    z = rng.normal(size=(200, 3))
    healthy = (z @ basis + 0.15 * rng.normal(size=(200, INPUT_DIM))).astype(np.float32)
    anom = (2.5 * rng.normal(size=(200, INPUT_DIM))).astype(np.float32)
    return healthy, anom


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_task1", path)
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

    required = {"model", "scores", "y_test", "input_dim", "latent_dim"}
    c["has_required_keys"] = required.issubset(r.keys())
    if not c["has_required_keys"]:
        return _finalize(score)

    model = r["model"]
    c["model_is_nn_module"] = isinstance(model, nn.Module)

    try:
        latent = int(r["latent_dim"])
        c["undercomplete_bottleneck"] = 0 < latent < int(r["input_dim"]) == INPUT_DIM
    except Exception:
        c["undercomplete_bottleneck"] = False

    try:
        scores = np.asarray(r["scores"], dtype=float).ravel()
        y_test = np.asarray(r["y_test"]).ravel().astype(int)
        c["scores_shape_matches"] = scores.shape == y_test.shape and scores.size > 0
    except Exception:
        c["scores_shape_matches"] = False
        scores, y_test = np.array([]), np.array([])

    # AUC of the submitted scores against the labels.
    if c["scores_shape_matches"] and len(set(y_test.tolist())) == 2:
        try:
            auc = roc_auc_score(y_test, scores)
            c["auc_at_least_0p90"] = bool(auc >= AUC_FLOOR)
        except Exception:
            c["auc_at_least_0p90"] = False
        try:
            sep = scores[y_test == 1].mean() / max(scores[y_test == 0].mean(), 1e-12)
            c["separation_at_least_1p5x"] = bool(sep >= SEP_FLOOR)
        except Exception:
            c["separation_at_least_1p5x"] = False
    else:
        c["auc_at_least_0p90"] = False
        c["separation_at_least_1p5x"] = False

    # Anti-faking: re-run the RETURNED model on fresh, independently-seeded data.
    # A genuine AE trained on the healthy manifold must assign higher recon error
    # to off-manifold anomalies than to fresh healthy cycles.
    if c["model_is_nn_module"]:
        try:
            healthy, anom = _fresh_eval_batches()
            model.eval()
            with torch.no_grad():
                h = torch.tensor(healthy)
                a = torch.tensor(anom)
                he = ((h - model(h)) ** 2).mean(dim=1).mean().item()
                ae = ((a - model(a)) ** 2).mean(dim=1).mean().item()
            c["model_ranks_anomalies_higher"] = bool(ae > he * SEP_FLOOR)
        except Exception:
            c["model_ranks_anomalies_higher"] = False
    else:
        c["model_ranks_anomalies_higher"] = False

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
