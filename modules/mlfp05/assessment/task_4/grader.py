#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grader for MLFP05 Assessment Task 4 — Ship the Postcode Reader as ONNX.

    python grader.py starter.py          # grade a submission
    python grader.py solution.py         # verify the reference passes
    python grader.py solution.py --seed 123   # replay a grading run

Ground truth the student cannot influence: the grader loads the returned
.onnx artefact with onnxruntime itself, feeds it grader-held inputs (its own
stratified split of the digits with a fresh secret seed), and checks (a)
numerical parity with the returned torch model and (b) accuracy against
grader-held labels. Self-reported metrics are never read.

Anti-stub: an untrained export passes parity but fails the accuracy floor;
a constant artefact fails the variety and accuracy checks; a stale artefact
that does not match the returned torch model fails parity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grading_harness import Checks, finalize, load_student_module, main, quiet  # noqa: E402

WEIGHT = 25
ACC_FLOOR = 0.88
PARITY_TOL = 1e-4
GATES = ("returns_contract", "export_succeeded")


def _grader_split(seed: int):
    """Grader-held stratified split of the bundled digits (fresh seed)."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    x, y = load_digits(return_X_y=True)
    x = (x / 16.0).astype(np.float32)
    _x_tr, x_te, y_tr, y_te = train_test_split(
        x, y.astype(np.int64), test_size=0.3, stratify=y, random_state=seed
    )
    return y_tr, x_te, y_te


def grade(student_path: Path, seed: int) -> dict:
    checks = Checks()
    try:
        st = load_student_module(student_path, "student_m5_task4")
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"Failed to import: {type(e).__name__}: {e}", GATES)
    if not callable(getattr(st, "solve", None)):
        return finalize(checks, WEIGHT, seed, "Module does not define solve()", GATES)
    try:
        with quiet():
            r = st.solve()
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"solve() raised {type(e).__name__}: {e}", GATES)

    import torch
    import torch.nn as nn

    torch.set_num_threads(2)
    rng = np.random.default_rng(seed)

    model = r.get("model") if isinstance(r, dict) else None
    onnx_path = r.get("onnx_path") if isinstance(r, dict) else None
    export_result = r.get("export_result") if isinstance(r, dict) else None
    checks.add(
        "returns_contract",
        isinstance(model, nn.Module) and onnx_path is not None,
        "solve() must return {'model': nn.Module, 'onnx_path': Path, 'export_result': ...}",
    )
    if not checks.results["returns_contract"]:
        return finalize(checks, WEIGHT, seed, None, GATES)

    p = Path(str(onnx_path))
    checks.add(
        "export_succeeded",
        bool(getattr(export_result, "success", False)) and p.exists() and p.suffix == ".onnx",
        f"export_result.success={getattr(export_result, 'success', None)!r}, file exists={p.exists()} ({p})",
    )
    if not checks.results["export_succeeded"]:
        return finalize(checks, WEIGHT, seed, None, GATES)

    import onnxruntime as ort

    y_tr, x_te, y_te = _grader_split(seed)

    def serving():
        try:
            sess = ort.InferenceSession(str(p))
        except Exception as e:
            return {"onnx_loads": (False, f"InferenceSession raised {type(e).__name__}: {e}")}
        inp = sess.get_inputs()[0]
        shape_ok = len(inp.shape) == 2 and inp.shape[-1] == 64
        type_ok = "float" in inp.type
        if not (shape_ok and type_ok):
            return {
                "onnx_loads": (
                    False,
                    f"input contract violated: name={inp.name!r} shape={inp.shape} type={inp.type}; expected float32 (batch, 64)",
                )
            }
        outs = sess.get_outputs()
        if len(outs) != 1:
            return {"onnx_loads": (False, f"expected 1 output, got {len(outs)}")}

        batches = []
        for bs in (1, 7, 16, 33, 64):
            batches.append(
                rng.normal(0.5, 0.3, size=(bs, 64)).clip(0, 1).astype(np.float32)
            )
        model.eval()
        worst = 0.0
        for xb in batches:
            with torch.no_grad():
                t_out = model(torch.tensor(xb)).numpy()
            o_out = sess.run(None, {inp.name: xb})[0]
            if t_out.shape != o_out.shape:
                return {
                    "onnx_loads": (True, ""),
                    "torch_onnx_parity": (
                        False,
                        f"shape mismatch: torch {t_out.shape} vs onnx {o_out.shape}",
                    ),
                }
            worst = max(worst, float(np.abs(t_out - o_out).max()))
        return {
            "onnx_loads": (True, ""),
            "torch_onnx_parity": (
                worst <= PARITY_TOL,
                f"max |torch - onnx| over five grader batches = {worst:.2e} (tolerance {PARITY_TOL})",
            ),
        }

    checks.guarded(["onnx_loads", "torch_onnx_parity"], serving)
    if not checks.results.get("onnx_loads"):
        return finalize(checks, WEIGHT, seed, None, GATES)

    def accuracy():
        sess = ort.InferenceSession(str(p))
        inp_name = sess.get_inputs()[0].name
        o1 = sess.run(None, {inp_name: x_te})[0]
        o2 = sess.run(None, {inp_name: x_te})[0]
        pred = o1.argmax(1)
        acc = float((pred == y_te).mean())
        majority = float(np.bincount(y_tr, minlength=10).max() / len(y_tr))
        return {
            "heldout_accuracy_at_least_0p88": (
                acc >= ACC_FLOOR,
                f"artefact accuracy on the grader-held split is {acc:.3f} (floor {ACC_FLOOR})",
            ),
            "beats_majority": (
                acc > majority + 0.05,
                f"accuracy {acc:.3f} does not clear the majority rate {majority:.3f}",
            ),
            "outputs_vary": (
                len(set(pred.tolist())) >= 5,
                f"only {len(set(pred.tolist()))} distinct predicted classes — a constant artefact serves nobody",
            ),
            "artefact_deterministic": (
                bool(np.array_equal(o1, o2)),
                "two runs of the artefact on identical input differ",
            ),
        }

    checks.guarded(
        ["heldout_accuracy_at_least_0p88", "beats_majority", "outputs_vary", "artefact_deterministic"],
        accuracy,
    )
    return finalize(checks, WEIGHT, seed, None, GATES)


if __name__ == "__main__":
    main(grade)
