# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP05 — Assessment Task 2: Triage a Ward of Failing Training Runs

Implement `diagnose_model()`. problem.md defines the four labels and what
each means. The grader builds its own ward of patients with planted
pathologies — a fixed answer, or "healthy" for everything, fails.

    python starter.py               # (you) smoke-test on example patients
    python grader.py starter.py     # (instructor) grade an attempt
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)

LABELS = ("healthy", "dead_neurons", "vanishing_gradients", "diverging_loss")


def diagnose_model(
    model: torch.nn.Module,
    loader,
    loss_fn,
    *,
    train_losses: list[float] | None = None,
    val_losses: list[float] | None = None,
) -> str:
    """Return the single label from LABELS that names this run's condition.

    Args:
        model: the trained model handed to you (do not mutate it).
        loader: yields (x_batch, y_batch) batches of the run's data.
        loss_fn: loss_fn(model, (x_batch, y_batch)) -> scalar loss.
        train_losses: per-epoch training loss recorded by the run.
        val_losses: per-epoch validation loss, when recorded.
    """
    raise NotImplementedError("Implement diagnose_model() — see problem.md")


# ── example patients for your own smoke-test (NOT the grader's ward) ──────
def _example_loader(seed: int = 3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(256, 20)).astype(np.float32)
    y = (X @ rng.normal(size=(20, 4))).argmax(1).astype(np.int64)
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y)),
        batch_size=64,
    )


def _ce(model, batch):
    xb, yb = batch
    return F.cross_entropy(model(xb), yb)


if __name__ == "__main__":
    loader = _example_loader()
    torch.manual_seed(0)
    demo = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 4))
    print("example patient ->", diagnose_model(demo, loader, _ce, train_losses=[1.4, 0.9, 0.6]))
