#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grader for MLFP05 Assessment Task 1 — Handwritten Postcode Reader.

    python grader.py starter.py          # grade a submission
    python grader.py solution.py         # verify the reference passes
    python grader.py solution.py --seed 123   # replay a grading run

Ground truth the student cannot influence:
  * the grader draws its OWN stratified split of the bundled digits with a
    fresh secret seed, and scores the RETURNED MODEL's predictions against its
    own labels — self-reported metrics are never read;
  * the grader trains its own classical baseline (logistic regression on raw
    pixels) on its own training slice and requires the CNN to be competitive
    with it;
  * a grader-built intensity-jittered variant of the held-out mail checks the
    model generalises beyond the exact pixels it was shown.

Anti-stub: an untrained network scores ~10% on the held-out split; a constant
predictor fails the majority and distinct-class checks; a model whose Conv2d
is declared but unwired fails the forward-hook check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grading_harness import Checks, finalize, load_student_module, main  # noqa: E402

WEIGHT = 25
ACC_FLOOR = 0.90
JITTER_FLOOR = 0.80
BASELINE_SLACK = 0.05
GATES = ("returns_model", "output_contract")


def _grader_split(seed: int):
    """Grader-held stratified split of the bundled digits (fresh seed)."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    x, y = load_digits(return_X_y=True)
    x = (x / 16.0).astype(np.float32).reshape(-1, 1, 8, 8)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y.astype(np.int64), test_size=0.3, stratify=y, random_state=seed
    )
    return x_tr, y_tr, x_te, y_te


def _classical_baseline_acc(x_tr, y_tr, x_te, y_te, seed: int) -> float:
    """The legacy pipeline: logistic regression on flattened pixels."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=1500, random_state=seed)
    clf.fit(x_tr.reshape(len(x_tr), -1), y_tr)
    return float(clf.score(x_te.reshape(len(x_te), -1), y_te))


def _jitter(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Intensity-jittered variant: global gain + per-pixel noise, clipped."""
    gain = float(rng.uniform(0.75, 1.25))
    noise = rng.normal(0.0, 0.05, size=x.shape).astype(np.float32)
    return np.clip(x * gain + noise, 0.0, 1.0).astype(np.float32)


def _predict(model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(logits, predictions) of the student's model on grader inputs."""
    import torch

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x))
    return logits.numpy(), logits.argmax(1).numpy()


def grade(student_path: Path, seed: int) -> dict:
    checks = Checks()
    try:
        st = load_student_module(student_path, "student_m5_task1")
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
    checks.add(
        "returns_model",
        isinstance(model, nn.Module),
        f"solve() must return a dict with a torch.nn.Module under 'model'; got {type(model).__name__}",
    )
    if not checks.results["returns_model"]:
        return finalize(checks, WEIGHT, seed, None, GATES)

    x_tr, y_tr, x_te, y_te = _grader_split(seed)

    def contract():
        probe = torch.tensor(x_te[:4])
        fired: list[str] = []
        hooks = []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d):
                hooks.append(mod.register_forward_hook(lambda m, i, o, n=name: fired.append(n)))
        try:
            model.eval()
            with torch.no_grad():
                out = model(probe)
        finally:
            for h in hooks:
                h.remove()
        ok_shape = isinstance(out, torch.Tensor) and tuple(out.shape) == (4, 10)
        return {
            "output_contract": (
                bool(ok_shape),
                f"model((4,1,8,8)) returned {getattr(out, 'shape', type(out).__name__)}; expected (4, 10) logits",
            ),
            "conv_fires": (
                len(fired) > 0,
                "no Conv2d module fired during the forward pass — a declared-but-unused convolution does not count",
            ),
        }

    checks.guarded(["output_contract", "conv_fires"], contract)
    if not checks.results["output_contract"]:
        return finalize(checks, WEIGHT, seed, None, GATES)

    def scores():
        logits1, pred1 = _predict(model, x_te)
        logits2, _ = _predict(model, x_te)
        acc = float((pred1 == y_te).mean())
        majority = float(np.bincount(y_tr, minlength=10).max() / len(y_tr))
        baseline = _classical_baseline_acc(x_tr, y_tr, x_te, y_te, seed)
        xj = _jitter(x_te, rng)
        _, predj = _predict(model, xj)
        acc_j = float((predj == y_te).mean())
        return {
            "deterministic_eval": (
                bool(np.array_equal(logits1, logits2)),
                "two eval-mode forward passes on identical input gave different logits (dropout or sampling left on?)",
            ),
            "heldout_accuracy_at_least_0p90": (
                acc >= ACC_FLOOR,
                f"accuracy on the grader-held split is {acc:.3f} (floor {ACC_FLOOR})",
            ),
            "beats_majority": (
                acc > majority + 0.05,
                f"accuracy {acc:.3f} does not clear the majority rate {majority:.3f}",
            ),
            "competitive_with_classical_baseline": (
                acc >= baseline - BASELINE_SLACK,
                f"accuracy {acc:.3f} is below the grader's logistic-regression baseline {baseline:.3f} minus {BASELINE_SLACK}",
            ),
            "jittered_accuracy_at_least_0p80": (
                acc_j >= JITTER_FLOOR,
                f"accuracy on the intensity-jittered variant is {acc_j:.3f} (floor {JITTER_FLOOR})",
            ),
            "predictions_vary": (
                len(set(pred1.tolist())) >= 5,
                f"only {len(set(pred1.tolist()))} distinct predicted classes on held-out mail",
            ),
        }

    checks.guarded(
        [
            "deterministic_eval",
            "heldout_accuracy_at_least_0p90",
            "beats_majority",
            "competitive_with_classical_baseline",
            "jittered_accuracy_at_least_0p80",
            "predictions_vary",
        ],
        scores,
    )
    return finalize(checks, WEIGHT, seed, None, GATES)


if __name__ == "__main__":
    main(grade)
