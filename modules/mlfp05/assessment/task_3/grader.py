#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grader for MLFP05 Assessment Task 3 — One-Step-Ahead Load Forecast.

    python grader.py starter.py          # grade a submission
    python grader.py solution.py         # verify the reference passes
    python grader.py solution.py --seed 123   # replay a grading run

Ground truth the student cannot influence: the grader regenerates the load
series from the documented process with fresh secret seeds (derived from
--seed, never the student's training seed), builds the windows itself, runs
the RETURNED model, and computes the naive last-value and mean forecasters
itself. Self-reported metrics are never read.

Anti-stub: an untrained recurrent net loses to the naive forecast; a constant
predictor fails the mean-forecaster and variance checks; a feed-forward-only
model fails the recurrent-forward-hook check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grading_harness import Checks, finalize, load_student_module, main  # noqa: E402

WEIGHT = 25
NAIVE_RATIO = 0.97
GATES = ("returns_model", "output_contract")


def make_series(seed: int, n: int = 1200) -> np.ndarray:
    """The documented generator — the grader drives it with fresh seeds."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=np.float64)
    x[0], x[1] = rng.normal(0.0, 0.3, size=2)
    for t in range(2, n):
        x[t] = (
            0.75 * x[t - 1]
            - 0.20 * x[t - 2]
            + 0.35 * np.sin(2 * np.pi * t / 24)
            + 0.15 * rng.normal()
        )
    return x.astype(np.float32)


def _windows(series: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(series, dtype=np.float32)
    xs = np.stack([s[i : i + window] for i in range(len(s) - window)])
    return xs.reshape(-1, window, 1), s[window:]


def _predict(model, X: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X)).reshape(-1).numpy()


def grade(student_path: Path, seed: int) -> dict:
    checks = Checks()
    try:
        st = load_student_module(student_path, "student_m5_task3")
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"Failed to import: {type(e).__name__}: {e}", GATES)
    if not callable(getattr(st, "solve", None)):
        return finalize(checks, WEIGHT, seed, "Module does not define solve()", GATES)
    try:
        r = st.solve()
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"solve() raised {type(e).__name__}: {e}", GATES)

    import torch
    import torch.nn as nn

    torch.set_num_threads(2)
    rng = np.random.default_rng(seed)

    model = r.get("model") if isinstance(r, dict) else None
    try:
        window = int(r.get("window")) if isinstance(r, dict) else 0
    except (TypeError, ValueError):
        window = 0
    checks.add(
        "returns_model",
        isinstance(model, nn.Module) and window >= 8,
        "solve() must return {'model': nn.Module, 'window': int >= 8}",
    )
    if not checks.results["returns_model"]:
        return finalize(checks, WEIGHT, seed, None, GATES)

    # Fresh series the student never saw (fresh seeds from the secret seed).
    seed1, seed2 = (int(rng.integers(1, 2**31 - 1)) for _ in range(2))
    X1, y1 = _windows(make_series(seed1), window)
    X2, y2 = _windows(make_series(seed2), window)

    def contract():
        fired: list[str] = []
        hooks = []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.RNNBase):
                hooks.append(mod.register_forward_hook(lambda m, i, o, n=name: fired.append(n)))
        try:
            out = _predict(model, X1[:3])
        finally:
            for h in hooks:
                h.remove()
        return {
            "output_contract": (
                out.shape == (3,),
                f"model((3,{window},1)) returned shape {out.shape}; expected (3,) or (3,1)",
            ),
            "recurrent_fires": (
                len(fired) > 0,
                "no nn.GRU/LSTM/RNN fired during the forward pass — a feed-forward-only model is not this task",
            ),
        }

    checks.guarded(["output_contract", "recurrent_fires"], contract)
    if not checks.results["output_contract"]:
        return finalize(checks, WEIGHT, seed, None, GATES)

    def scores():
        p1 = _predict(model, X1)
        p1b = _predict(model, X1)
        naive1 = X1.reshape(len(X1), window)[:, -1]
        naive2 = X2.reshape(len(X2), window)[:, -1]
        mse1 = float(((p1 - y1) ** 2).mean())
        naive_mse1 = float(((naive1 - y1) ** 2).mean())
        p2 = _predict(model, X2)
        mse2 = float(((p2 - y2) ** 2).mean())
        naive_mse2 = float(((naive2 - y2) ** 2).mean())
        mean_mse1 = float(((float(y1.mean()) - y1) ** 2).mean())
        return {
            "deterministic_eval": (
                bool(np.array_equal(p1, p1b)),
                "two eval-mode forward passes on identical input differ",
            ),
            "beats_naive_fresh_series_1": (
                mse1 <= NAIVE_RATIO * naive_mse1,
                f"fresh series 1: your MSE {mse1:.4f} vs naive {naive_mse1:.4f} (ratio {mse1 / naive_mse1:.3f} > {NAIVE_RATIO})",
            ),
            "beats_naive_fresh_series_2": (
                mse2 <= NAIVE_RATIO * naive_mse2,
                f"fresh series 2: your MSE {mse2:.4f} vs naive {naive_mse2:.4f} (ratio {mse2 / naive_mse2:.3f} > {NAIVE_RATIO})",
            ),
            "beats_mean_forecast": (
                mse1 <= mean_mse1,
                f"your MSE {mse1:.4f} does not beat predicting the series mean ({mean_mse1:.4f})",
            ),
            "predictions_vary": (
                float(p1.std()) > 1e-4 and float(p2.std()) > 1e-4,
                f"forecast std {p1.std():.2e} — a constant forecaster cannot track a series",
            ),
        }

    checks.guarded(
        [
            "deterministic_eval",
            "beats_naive_fresh_series_1",
            "beats_naive_fresh_series_2",
            "beats_mean_forecast",
            "predictions_vary",
        ],
        scores,
    )
    return finalize(checks, WEIGHT, seed, None, GATES)


if __name__ == "__main__":
    main(grade)
