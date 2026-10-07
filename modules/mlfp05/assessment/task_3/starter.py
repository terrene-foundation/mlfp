# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP05 — Assessment Task 3: One-Step-Ahead Load Forecast

Implement `solve()`. problem.md holds the model contract and the acceptance
criteria. The grader regenerates fresh series from the same process with
fresh seeds and scores your RETURNED model against its own naive forecast —
an untrained net or a constant predictor fails.

    python starter.py               # (you) train + smoke-test your model
    python grader.py starter.py     # (instructor) grade an attempt
"""
from __future__ import annotations

import numpy as np
import torch

torch.set_num_threads(2)  # tiny CPU model; two threads is plenty


def make_series(seed: int, n: int = 1200) -> np.ndarray:
    """The load-series generator: damped AR(2) + daily seasonality + noise.

    x_t = 0.75 x_{t-1} - 0.20 x_{t-2} + 0.35 sin(2 pi t / 24) + 0.15 e_t

    Train on `make_series(7)`. The grader uses fresh seeds of its own.
    """
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


def make_windows(series: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window pairs: X[i] = series[i:i+window], y[i] = series[i+window].

    Returns X as (N, window, 1) float32 and y as (N,) float32.
    """
    s = np.asarray(series, dtype=np.float32)
    xs = np.stack([s[i : i + window] for i in range(len(s) - window)])
    ys = s[window:]
    return xs.reshape(-1, window, 1), ys


def solve() -> dict:
    """Train a recurrent one-step-ahead forecaster.

    Returns:
        {"model": nn.Module, "window": int} where the model maps
        (N, window, 1) float32 to (N,) or (N, 1) next-reading forecasts.
    """
    raise NotImplementedError("Implement solve() — see problem.md")


if __name__ == "__main__":
    series = make_series(7)
    print(f"Training series: {series.shape}, mean {series.mean():.3f}, std {series.std():.3f}")
    out = solve()
    model, window = out["model"], int(out["window"])
    dev = make_series(99)  # your own development series — NOT the grader's
    X, y = make_windows(dev, window)
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X)).reshape(-1).numpy()
    mse = float(((preds - y) ** 2).mean())
    naive = float(((X.reshape(len(X), window)[:, -1] - y) ** 2).mean())
    print(f"Your dev MSE {mse:.4f} vs naive {naive:.4f}  (ratio {mse / naive:.3f})")
