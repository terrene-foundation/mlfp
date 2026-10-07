#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grader for MLFP05 Assessment Task 2 — Triage a Ward of Failing Training Runs.

    python grader.py starter.py          # grade a submission
    python grader.py solution.py         # verify the reference passes
    python grader.py solution.py --seed 123   # replay a grading run

The grader builds the ward itself with a fresh secret seed: small MLPs on
synthetic Gaussian-blob data, trained briefly, with the pathology planted
by the grader (a zeroed layer / a deep tiny-init tanh stack / an increasing
recorded loss history). The student's diagnose_model() labels each patient;
labels are compared against the planted truth. A constant-label stub fails
the pathology checks and the distinct-labels check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grading_harness import Checks, finalize, load_student_module, main, quiet  # noqa: E402

WEIGHT = 25
VALID = ("healthy", "dead_neurons", "vanishing_gradients", "diverging_loss")


def _blobs(rng: np.random.Generator, n: int = 256, d: int = 20, k: int = 4):
    x = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=(d, k)).astype(np.float32)
    y = (x @ w + 0.5 * rng.normal(size=(n, k))).argmax(1).astype(np.int64)
    return x, y


def _loader(x: np.ndarray, y: np.ndarray):
    import torch

    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(x), torch.tensor(y)),
        batch_size=64,
    )


def _ce(model, batch):
    import torch.nn.functional as F

    xb, yb = batch
    return F.cross_entropy(model(xb), yb)


def _train(model, loader, epochs: int, lr: float) -> list[float]:
    import torch

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        for xb, yb in loader:
            loss = _ce(model, (xb, yb))
            opt.zero_grad()
            loss.backward()
            opt.step()
        losses.append(float(loss))
    return losses


def _train_to_converge(model, loader, lr: float = 5e-3, max_epochs: int = 90,
                       target: float = 0.02) -> list[float]:
    """Train until the loss is genuinely small (or max_epochs).

    A mid-training net keeps gradient RMS high enough for the flow
    instrument to read "exploding"; a converged net reads HEALTHY. The
    ward's healthy/dead/diverging patients must be converged so only the
    planted pathology fires.
    """
    import torch

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    for _ in range(max_epochs):
        total, n = 0.0, 0
        for xb, yb in loader:
            loss = _ce(model, (xb, yb))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
            n += 1
        losses.append(total / n)
        if losses[-1] < target:
            break
    return losses


def _ward(seed: int):
    """Build the ten patients. Returns a list of
    (truth, model, loader, loss_fn, train_losses)."""
    import torch
    import torch.nn as nn

    rng = np.random.default_rng(seed)
    ward: list[tuple[str, nn.Module, object, object, list[float]]] = []

    # 3 healthy: shallow ReLU MLPs on separable blobs, trained to
    # convergence — the loss really falls and gradients settle small.
    for _ in range(3):
        x, y = _blobs(rng)
        loader = _loader(x, y)
        width = int(rng.choice([24, 32, 48]))
        torch.manual_seed(int(rng.integers(1_000_000_000)))
        m = nn.Sequential(nn.Linear(20, width), nn.ReLU(), nn.Linear(width, 4))
        losses = _train_to_converge(m, loader)
        ward.append(("healthy", m, loader, _ce, losses))

    # 3 dead-layer: trained as above, then one hidden layer's weights and
    # biases are zeroed (post-training damage) — its units output all zeros.
    for _ in range(3):
        x, y = _blobs(rng)
        loader = _loader(x, y)
        width = int(rng.choice([24, 32, 48]))
        torch.manual_seed(int(rng.integers(1_000_000_000)))
        m = nn.Sequential(nn.Linear(20, width), nn.ReLU(), nn.Linear(width, 4))
        losses = _train_to_converge(m, loader)
        with torch.no_grad():
            m[0].weight.zero_()
            m[0].bias.zero_()
        ward.append(("dead_neurons", m, loader, _ce, losses))

    # 2 vanishing-gradient: deep tanh stacks with tiny init, trained briefly —
    # gradients collapse long before the input layer; the loss barely moves.
    for _ in range(2):
        x, y = _blobs(rng)
        loader = _loader(x, y)
        torch.manual_seed(int(rng.integers(1_000_000_000)))
        layers: list[nn.Module] = []
        for _ in range(6):
            layers += [nn.Linear(20, 20), nn.Tanh()]
        layers.append(nn.Linear(20, 4))
        m = nn.Sequential(*layers)
        with torch.no_grad():
            for lyr in m:
                if isinstance(lyr, nn.Linear):
                    lyr.weight.mul_(0.01)
                    lyr.bias.zero_()
        losses = _train(m, loader, epochs=8, lr=1e-3)
        ward.append(("vanishing_gradients", m, loader, _ce, losses))

    # 2 diverging-loss: the model is fine (trained healthy MLP), but the
    # recorded training log increases monotonically — the run went wrong.
    for _ in range(2):
        x, y = _blobs(rng)
        loader = _loader(x, y)
        torch.manual_seed(int(rng.integers(1_000_000_000)))
        m = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 4))
        _train_to_converge(m, loader)
        base = float(rng.uniform(0.5, 0.9))
        growth = float(rng.uniform(1.4, 1.8))
        losses = [
            base * growth**e * (1.0 + 0.03 * float(rng.normal())) for e in range(12)
        ]
        ward.append(("diverging_loss", m, loader, _ce, losses))

    return ward


def grade(student_path: Path, seed: int) -> dict:
    checks = Checks()
    try:
        st = load_student_module(student_path, "student_m5_task2")
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"Failed to import: {type(e).__name__}: {e}")
    fn = getattr(st, "diagnose_model", None)
    if not callable(fn):
        return finalize(checks, WEIGHT, seed, "Module does not define diagnose_model()")

    import torch

    torch.set_num_threads(2)
    ward = _ward(seed)

    labels: list[str] = []
    errors: list[str] = []
    for truth, model, loader, loss_fn, train_losses in ward:
        try:
            with quiet():
                lab = fn(model, loader, loss_fn, train_losses=list(train_losses))
        except Exception as e:
            lab = f"__raised_{type(e).__name__}"
            errors.append(f"{truth}: {type(e).__name__}: {e}")
        labels.append(str(lab))

    checks.add(
        "valid_labels",
        all(l in VALID for l in labels),
        f"returned labels {labels}; each must be one of {VALID}"
        + (f"; first error: {errors[0]}" if errors else ""),
    )

    def class_check(truth: str) -> tuple[bool, str]:
        got = [l for l, (t, *_rest) in zip(labels, ward) if t == truth]
        ok = bool(got) and all(l == truth for l in got)
        return ok, f"{truth}: expected {truth} x{len(got)}, got {got}"

    for truth in ("healthy", "dead_neurons", "vanishing_gradients", "diverging_loss"):
        ok, note = class_check(truth)
        checks.add(f"{truth}_correct", ok, note)

    checks.add(
        "distinct_labels",
        len(set(labels)) >= 3,
        f"only {len(set(labels))} distinct label(s) across ten patients — a one-label stub cannot triage",
    )

    def repeat():
        truth, model, loader, loss_fn, tl = ward[0]
        with quiet():
            a = fn(model, loader, loss_fn, train_losses=list(tl))
            b = fn(model, loader, loss_fn, train_losses=list(tl))
        return {
            "consistent_repeat": (
                str(a) == str(b),
                f"same patient labelled {a!r} then {b!r} — the ward runs are deterministic; your function must be too",
            )
        }

    checks.guarded(["consistent_repeat"], repeat)
    return finalize(checks, WEIGHT, seed)


if __name__ == "__main__":
    main(grade)
