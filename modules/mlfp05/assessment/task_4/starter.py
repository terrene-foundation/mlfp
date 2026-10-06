# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP05 — Assessment Task 4: Ship the Postcode Reader as an ONNX Artefact

Implement `solve()`. problem.md holds the serving contract and the acceptance
criteria. The grader loads your .onnx artefact with onnxruntime itself and
scores it against grader-held labels — an untrained export or a constant
artefact fails.

    python starter.py               # (you) train, export + smoke-test
    python grader.py starter.py     # (instructor) grade an attempt
"""
from __future__ import annotations

import numpy as np
import torch

torch.set_num_threads(2)  # tiny CPU model; two threads is plenty


def load_digits_flat() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The training mail: fixed documented stratified split (seed 7, 80/20)
    of the bundled 8x8 digits, flattened to 64 features, scaled to [0, 1].

    Returns (x_train, y_train, x_val, y_val): x arrays (N, 64) float32,
    y arrays int64. The grader scores your artefact on ITS OWN split.
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    x, y = load_digits(return_X_y=True)
    x = (x / 16.0).astype(np.float32)
    return train_test_split(
        x, y.astype(np.int64), test_size=0.2, stratify=y, random_state=7
    )


def solve() -> dict:
    """Train a classifier on the flattened digits and export it to ONNX.

    Returns:
        {"model": nn.Module, "onnx_path": Path, "export_result": ...}
        where the artefact honours the serving contract in problem.md:
        float32 (batch, 64) in [0, 1] -> float32 (batch, 10) logits.
    """
    raise NotImplementedError("Implement solve() — see problem.md")


if __name__ == "__main__":
    x_train, y_train, x_val, y_val = load_digits_flat()
    print(f"Train {x_train.shape}, val {x_val.shape}")
    out = solve()
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out["onnx_path"]))
    name = sess.get_inputs()[0].name
    onnx_logits = sess.run(None, {name: x_val})[0]
    model = out["model"]
    model.eval()
    with torch.no_grad():
        torch_logits = model(torch.tensor(x_val)).numpy()
    print(f"parity max diff: {float(np.abs(torch_logits - onnx_logits).max()):.2e}")
    print(f"val accuracy (artefact): {(onnx_logits.argmax(1) == y_val).mean():.3f}")
