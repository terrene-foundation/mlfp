#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grade MLFP05 Task 1 — Fashion-MNIST CNN classifier."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torchvision

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "data" / "mlfp05" / "fashion_mnist"

N_TEST = 5000
ACCURACY_THRESHOLD = 0.85


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_task1", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _finalize(score: dict) -> None:
    score["total"] = sum(1 for v in score["checks"].values() if v)
    score["max"] = len(score["checks"])
    score["passed"] = score["total"] == score["max"] and score["max"] > 0


def grade(student_path: Path) -> dict:
    score: dict = {"passed": False, "checks": {}, "metrics": {}, "total": 0, "max": 0}
    try:
        student = load_student_module(student_path)
    except Exception as e:
        score["error"] = f"Failed to import: {type(e).__name__}: {e}"
        return score

    if not hasattr(student, "solve"):
        score["error"] = "Module does not define a solve() function"
        return score

    try:
        out = student.solve()
    except Exception as e:
        score["error"] = f"Runtime error in solve(): {type(e).__name__}: {e}"
        return score

    if not (isinstance(out, tuple) and len(out) == 2):
        score["error"] = "solve() must return a 2-tuple (model, predict)"
        return score

    model, predict = out

    score["checks"]["returns_module"] = isinstance(model, torch.nn.Module)
    if not score["checks"]["returns_module"]:
        _finalize(score)
        return score

    score["checks"]["model_in_eval_mode"] = model.training is False

    n_params = sum(p.numel() for p in model.parameters())
    score["metrics"]["model_parameters"] = n_params
    score["checks"]["parameter_count_reasonable"] = 5_000 <= n_params <= 5_000_000

    try:
        demo = predict(torch.zeros(2, 1, 28, 28))
        shape_ok = isinstance(demo, torch.Tensor) and demo.shape == (2,)
        dtype_ok = demo.dtype == torch.int64
        range_ok = bool(((demo >= 0) & (demo <= 9)).all().item())
        score["checks"]["predict_interface_ok"] = shape_ok and dtype_ok and range_ok
    except Exception as e:
        score["checks"]["predict_interface_ok"] = False
        score["error"] = f"predict() raised: {type(e).__name__}: {e}"

    if not score["checks"]["predict_interface_ok"]:
        _finalize(score)
        return score

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    test_set = torchvision.datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    X_test = torch.stack([test_set[i][0] for i in range(N_TEST)])
    y_test = torch.tensor([test_set[i][1] for i in range(N_TEST)], dtype=torch.int64)

    try:
        preds = predict(X_test)
        accuracy = float((preds.cpu() == y_test).float().mean().item())
    except Exception as e:
        score["error"] = f"predict() raised on test set: {type(e).__name__}: {e}"
        _finalize(score)
        return score

    score["metrics"]["test_accuracy"] = round(accuracy, 4)
    score["checks"]["test_accuracy_above_threshold"] = accuracy >= ACCURACY_THRESHOLD

    _finalize(score)
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("student", type=Path)
    args = parser.parse_args()
    result = grade(args.student)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)
