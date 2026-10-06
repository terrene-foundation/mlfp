# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP05 — Assessment Task 1: Handwritten Postcode Reader

Implement `solve()`. problem.md holds the model contract and the acceptance
criteria. The grader scores your RETURNED model on held-out data it splits
itself — an untrained network or a constant predictor fails.

    python starter.py               # (you) train + smoke-test your model
    python grader.py starter.py     # (instructor) grade an attempt
"""
from __future__ import annotations

import numpy as np
import torch

torch.set_num_threads(2)  # tiny CPU model; two threads is plenty


def load_digits_train() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The training mail: a fixed, documented stratified split of the bundled
    8x8 digits (seed 7, 80/20). Pixels are scaled to [0, 1].

    Returns:
        (x_train, y_train, x_val, y_val) — x arrays are (N, 1, 8, 8) float32,
        y arrays are int64 class labels 0-9.

    The grader evaluates on ITS OWN split, drawn with a fresh secret seed, so
    do not tune to this particular validation slice.
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    x, y = load_digits(return_X_y=True)
    x = (x / 16.0).astype(np.float32).reshape(-1, 1, 8, 8)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y.astype(np.int64), test_size=0.2, stratify=y, random_state=7
    )
    return x_train, y_train, x_val, y_val


def solve() -> dict:
    """Train a convolutional classifier on the digits training data.

    Returns:
        {"model": nn.Module, "history": {"train_loss": [...], "val_loss": [...]}}
        where the model maps (N, 1, 8, 8) float32 in [0, 1] to (N, 10) logits.
    """
    raise NotImplementedError("Implement solve() — see problem.md")


if __name__ == "__main__":
    x_train, y_train, x_val, y_val = load_digits_train()
    print(f"Train {x_train.shape}, val {x_val.shape}")
    out = solve()
    model = out["model"]
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_val))
    acc = (logits.argmax(1).numpy() == y_val).mean()
    print(f"Your val accuracy: {acc:.3f}  (grader uses its own split)")
