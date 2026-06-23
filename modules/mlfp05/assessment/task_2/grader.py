#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP05 Assessment Task 2 — Tiny CNN Image Classification.

Usage:
    python grader.py starter.py     # grade your attempt
    python grader.py solution.py    # verify the reference passes

The grader re-derives the exact same test split, re-runs the returned model on it
(so a hand-tuned `preds` array fails the anti-faking check), and verifies the model
is genuinely convolutional and generalises on a held-out slice.
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
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

N_CLASSES = 10
SEED = 42
ACC_FLOOR = 0.90
HELDOUT_FLOOR = 0.88


def _reference_test_split() -> tuple[np.ndarray, np.ndarray]:
    """Re-derive the exact (X_test, y_test) the student trained against."""
    digits = load_digits()
    X = (digits.images / 16.0).astype(np.float32)[:, None, :, :]
    y = digits.target.astype(int)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    return X_test, y_test


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_task2", path)
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

    required = {"model", "preds", "y_test", "n_conv"}
    c["has_required_keys"] = required.issubset(r.keys())
    if not c["has_required_keys"]:
        return _finalize(score)

    model = r["model"]
    c["model_is_nn_module"] = isinstance(model, nn.Module)

    # Genuine CNN: at least one Conv2d, and declared n_conv matches introspection.
    if c["model_is_nn_module"]:
        actual_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
        try:
            declared = int(r["n_conv"])
        except Exception:
            declared = -1
        c["is_convolutional"] = actual_conv >= 1 and declared == actual_conv
    else:
        c["is_convolutional"] = False

    X_test_ref, y_test_ref = _reference_test_split()

    try:
        preds = np.asarray(r["preds"]).ravel().astype(int)
        y_test = np.asarray(r["y_test"]).ravel().astype(int)
        c["preds_shape_matches"] = preds.shape == y_test.shape == y_test_ref.shape
    except Exception:
        c["preds_shape_matches"] = False
        preds, y_test = np.array([]), np.array([])

    if c["preds_shape_matches"]:
        c["test_accuracy_at_least_0p90"] = bool((preds == y_test).mean() >= ACC_FLOOR)
    else:
        c["test_accuracy_at_least_0p90"] = False

    # Anti-faking: re-run the returned model on the re-derived test set and require
    # the predictions to match what was submitted (within a tiny tolerance).
    model_preds = None
    if c["model_is_nn_module"] and c["preds_shape_matches"]:
        try:
            model.eval()
            with torch.no_grad():
                logits = model(torch.tensor(X_test_ref))
                model_preds = logits.argmax(dim=1).cpu().numpy().astype(int)
            agree = (model_preds == preds).mean()
            c["preds_reproduced_by_model"] = bool(agree >= 0.99)
        except Exception:
            c["preds_reproduced_by_model"] = False
    else:
        c["preds_reproduced_by_model"] = False

    # Generalisation: accuracy of the model's OWN predictions on a held-out slice.
    if model_preds is not None:
        held = model_preds[::3]  # every third test row
        held_y = y_test_ref[::3]
        c["heldout_accuracy_at_least_0p88"] = bool(
            (held == held_y).mean() >= HELDOUT_FLOOR
        )
    else:
        c["heldout_accuracy_at_least_0p88"] = False

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
