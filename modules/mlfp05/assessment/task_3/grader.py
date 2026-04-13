#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grade MLFP05 Task 3 — STI walk-forward LSTM regression."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[4]
STI_CACHE = REPO_ROOT / "data" / "mlfp05" / "sti" / "sti_close.parquet"

SEQ_LEN = 20
HORIZON = 5
FEATURES = ["Close", "High", "Low", "Volume"]


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_task3", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _finalize(score: dict) -> None:
    score["total"] = sum(1 for v in score["checks"].values() if v)
    score["max"] = len(score["checks"])
    score["passed"] = score["total"] == score["max"] and score["max"] > 0


def _ensure_sti() -> pl.DataFrame:
    if STI_CACHE.exists():
        return pl.read_parquet(STI_CACHE)
    import yfinance as yf

    STI_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df = yf.download(
        "^STI", start="2010-01-01", end="2024-12-31", progress=False, auto_adjust=True
    )
    if df is None or len(df) == 0:
        df = yf.download(
            "AAPL",
            start="2010-01-01",
            end="2024-12-31",
            progress=False,
            auto_adjust=True,
        )
    df = df.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index()
    out = pl.from_pandas(df)
    out.write_parquet(STI_CACHE)
    return out


def _has_recurrent_child(model: nn.Module) -> bool:
    for m in model.modules():
        if isinstance(m, (nn.LSTM, nn.GRU)):
            return True
    return False


def grade(student_path: Path) -> dict:
    score: dict = {"passed": False, "checks": {}, "metrics": {}, "total": 0, "max": 0}

    try:
        source = student_path.read_text()
    except Exception as e:
        score["error"] = f"Failed to read source: {type(e).__name__}: {e}"
        return score

    score["checks"]["uses_gradient_clipping"] = (
        "clip_grad_norm_" in source or "clip_grad_value_" in source
    )

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

    score["checks"]["returns_module"] = isinstance(model, nn.Module)
    if not score["checks"]["returns_module"]:
        _finalize(score)
        return score

    score["checks"]["contains_lstm_or_gru"] = _has_recurrent_child(model)

    df = _ensure_sti()
    try:
        preds = predict(df)
    except Exception as e:
        score["error"] = f"predict() raised: {type(e).__name__}: {e}"
        _finalize(score)
        return score

    expected_len = len(df) - SEQ_LEN - HORIZON + 1
    score["checks"]["prediction_shape_correct"] = (
        isinstance(preds, torch.Tensor)
        and preds.dim() == 1
        and preds.shape[0] == expected_len
    )
    if not score["checks"]["prediction_shape_correct"]:
        _finalize(score)
        return score

    data = df.select(FEATURES).to_numpy().astype(np.float32)
    split = int(0.8 * len(data))
    train = data[:split]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-8
    data_norm = (data - mean) / std

    n_windows = len(data_norm) - SEQ_LEN - HORIZON + 1
    y_true_all = data_norm[SEQ_LEN + HORIZON - 1 : SEQ_LEN + HORIZON - 1 + n_windows, 0]
    # Naive = "close HORIZON days ahead equals close at end of window"
    window_end_vals = data_norm[SEQ_LEN - 1 : SEQ_LEN - 1 + n_windows, 0]

    n_train_w = split - SEQ_LEN - HORIZON + 1
    val_true = y_true_all[n_train_w:]
    naive_preds_val = window_end_vals[n_train_w:]
    naive_mse = float(((naive_preds_val - val_true) ** 2).mean())

    student_val = preds[n_train_w:].detach().cpu().numpy().astype(np.float32)
    student_mse = float(((student_val - val_true) ** 2).mean())

    score["metrics"]["naive_mse"] = round(naive_mse, 6)
    score["metrics"]["student_mse"] = round(student_mse, 6)
    score["checks"]["beats_naive_baseline"] = student_mse < naive_mse

    _finalize(score)
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("student", type=Path)
    args = parser.parse_args()
    result = grade(args.student)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)
