# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP05 — Assessment Task 2: Triage a Ward of Failing Training Runs
(Reference Solution)

Withheld from students. Verified to pass grader.py across seeds.

Decision rule, applied to the kailash-ml DLDiagnostics readings (each
instrument reports HEALTHY / WARNING / CRITICAL / UNKNOWN):

  1. dead neurons fire      -> "dead_neurons"     (the ward's clearest signal;
     a zeroed layer also starves gradients downstream, so read this first)
  2. gradient flow fires    -> "vanishing_gradients"
  3. loss trend fires AND the recorded loss ends higher than it started
                            -> "diverging_loss"
  4. otherwise              -> "healthy"

The increasing-history clause separates a diverged run from a plateaued one
(plateau is a *symptom* of vanishing gradients, caught at step 2).

Raw torch.nn for model handling per the module's DL rules; the measuring
instruments are kailash-ml's.
"""
from __future__ import annotations

import torch
import torch.nn as nn

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
    from kailash_ml.diagnostics import run_diagnostic_checkpoint

    _, findings = run_diagnostic_checkpoint(
        model,
        loader,
        loss_fn,
        title="triage",
        show=False,
        n_batches=4,
        train_losses=list(train_losses) if train_losses is not None else None,
        val_losses=list(val_losses) if val_losses is not None else None,
    )

    def severity(instrument: str) -> str:
        reading = findings.get(instrument, {})
        if isinstance(reading, dict):
            return str(reading.get("severity", "UNKNOWN"))
        return "UNKNOWN"

    if severity("dead_neurons") in ("WARNING", "CRITICAL"):
        return "dead_neurons"
    if severity("gradient_flow") in ("WARNING", "CRITICAL"):
        return "vanishing_gradients"
    if severity("loss_trend") in ("WARNING", "CRITICAL"):
        losses = [float(v) for v in (train_losses or [])]
        if len(losses) >= 2 and losses[-1] > losses[0]:
            return "diverging_loss"
    return "healthy"


if __name__ == "__main__":
    import numpy as np
    import torch.nn.functional as F

    rng = np.random.default_rng(3)
    X = rng.normal(size=(256, 20)).astype(np.float32)
    y = (X @ rng.normal(size=(20, 4))).argmax(1).astype(np.int64)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y)),
        batch_size=64,
    )

    def ce(m, batch):
        xb, yb = batch
        return F.cross_entropy(m(xb), yb)

    torch.manual_seed(0)
    demo = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 4))
    opt = torch.optim.Adam(demo.parameters(), lr=2e-3)
    losses = []
    for _ in range(20):
        for xb, yb in loader:
            loss = ce(demo, (xb, yb))
            opt.zero_grad()
            loss.backward()
            opt.step()
        losses.append(float(loss))
    print("trained example ->", diagnose_model(demo, loader, ce, train_losses=losses))
