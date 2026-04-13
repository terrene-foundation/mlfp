#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grade MLFP05 Task 2 — transfer learning + ONNX."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torchvision

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "data" / "mlfp05" / "cifar10"

N_TEST = 1000
N_PARITY = 100
ACCURACY_THRESHOLD = 0.55
PARITY_THRESHOLD = 0.95
MAX_ONNX_MB = 60.0


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_task2", path)
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

    onnx_path = student_path.parent / "student.onnx"
    if onnx_path.exists():
        onnx_path.unlink()

    try:
        student = load_student_module(student_path)
    except Exception as e:
        score["error"] = f"Failed to import: {type(e).__name__}: {e}"
        return score

    if not hasattr(student, "solve"):
        score["error"] = "Module does not define a solve(onnx_path) function"
        return score

    try:
        out = student.solve(onnx_path)
    except Exception as e:
        score["error"] = f"Runtime error in solve(): {type(e).__name__}: {e}"
        return score

    if not (isinstance(out, tuple) and len(out) == 2):
        score["error"] = "solve(onnx_path) must return a 2-tuple (model, predict)"
        return score

    model, predict = out

    score["checks"]["returns_module"] = isinstance(model, torch.nn.Module)
    if not score["checks"]["returns_module"]:
        _finalize(score)
        return score

    score["checks"]["onnx_file_exists"] = onnx_path.exists()
    if onnx_path.exists():
        mb = onnx_path.stat().st_size / (1024 * 1024)
        score["metrics"]["onnx_size_mb"] = round(mb, 2)
        score["checks"]["onnx_size_under_limit"] = mb <= MAX_ONNX_MB
    else:
        score["checks"]["onnx_size_under_limit"] = False
        _finalize(score)
        return score

    resnet_key = "layer4.1.conv2.weight"
    state_keys = list(model.state_dict().keys())
    score["checks"]["model_is_resnet18_backbone"] = any(k.endswith(resnet_key) for k in state_keys)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    test_set = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    X_test = torch.stack([test_set[i][0] for i in range(N_TEST)])
    y_test = torch.tensor([test_set[i][1] for i in range(N_TEST)], dtype=torch.int64)

    try:
        onnx_preds = predict(X_test)
    except Exception as e:
        score["error"] = f"predict() raised on test set: {type(e).__name__}: {e}"
        _finalize(score)
        return score

    if not isinstance(onnx_preds, torch.Tensor) or onnx_preds.shape != (N_TEST,):
        score["checks"]["predict_interface_ok"] = False
        _finalize(score)
        return score
    score["checks"]["predict_interface_ok"] = onnx_preds.dtype == torch.int64

    acc = float((onnx_preds.cpu() == y_test).float().mean().item())
    score["metrics"]["test_accuracy"] = round(acc, 4)
    score["checks"]["test_accuracy_above_threshold"] = acc >= ACCURACY_THRESHOLD

    model.eval()
    with torch.no_grad():
        torch_preds = model(X_test[:N_PARITY]).argmax(dim=1).cpu().to(torch.int64)
    agreement = float((torch_preds == onnx_preds[:N_PARITY]).float().mean().item())
    score["metrics"]["pytorch_onnx_agreement"] = round(agreement, 4)
    score["checks"]["pytorch_onnx_parity"] = agreement >= PARITY_THRESHOLD

    _finalize(score)
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("student", type=Path)
    args = parser.parse_args()
    result = grade(args.student)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)
