#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP05 Assessment Task 4 — Tiny Transformer Text Classification.

Usage:
    python grader.py starter.py     # grade your attempt
    python grader.py solution.py    # verify the reference passes

The grader re-derives the exact same encoded test split, re-runs the returned model
on it (so a hand-tuned `preds` array fails the anti-faking check), and verifies the
model genuinely uses self-attention and clears the accuracy floor.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from shared import MLFPDataLoader

MAX_LEN = 40
MAX_VOCAB = 8000
SEED = 5
ACC_FLOOR = 0.72
MAJORITY_MARGIN = 0.15
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _reference_test_split():
    """Re-derive (X_test, y_test) identically to the starter's make_dataset."""
    train_df = MLFPDataLoader().load("mlfp05", "ag_news.parquet")
    test_df = MLFPDataLoader().load("mlfp05", "ag_news_test.parquet")
    train_texts = train_df["text"].to_list()
    test_texts = test_df["text"].to_list()
    y_test = test_df["label"].to_numpy().astype(np.int64)

    counts: Counter = Counter()
    for t in train_texts:
        counts.update(_tokenize(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, _ in counts.most_common(MAX_VOCAB - 2):
        vocab[tok] = len(vocab)

    out = np.zeros((len(test_texts), MAX_LEN), dtype=np.int64)
    for i, t in enumerate(test_texts):
        for j, tok in enumerate(_tokenize(t)[:MAX_LEN]):
            out[i, j] = vocab.get(tok, 1)
    return out, y_test


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_task4", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def grade(student_path: Path) -> dict:
    score: dict = {"passed": False, "checks": {}, "total": 0, "max": 0}
    try:
        student = load_student_module(student_path)
    except Exception as e:
        score["error"] = f"Failed to import: {type(e).__name__}: {e}"
        return score
    if not hasattr(student, "solve"):
        score["error"] = "Module does not define a solve() function"
        return score
    try:
        r = student.solve()
    except Exception as e:
        score["error"] = f"Runtime error in solve(): {type(e).__name__}: {e}"
        return score

    c = score["checks"]
    c["returns_dict"] = isinstance(r, dict)
    if not c["returns_dict"]:
        return _finalize(score)

    required = {"model", "preds", "y_test", "uses_attention"}
    c["has_required_keys"] = required.issubset(r.keys())
    if not c["has_required_keys"]:
        return _finalize(score)

    model = r["model"]
    c["model_is_nn_module"] = isinstance(model, nn.Module)

    # Genuine attention: at least one attention module, declared flag matches.
    if c["model_is_nn_module"]:
        actual_attn = any(
            isinstance(
                m,
                (
                    nn.MultiheadAttention,
                    nn.TransformerEncoderLayer,
                    nn.TransformerEncoder,
                ),
            )
            for m in model.modules()
        )
        c["uses_self_attention"] = (
            actual_attn and bool(r["uses_attention"]) == actual_attn
        )
    else:
        c["uses_self_attention"] = False

    X_test_ref, y_test_ref = _reference_test_split()

    try:
        preds = np.asarray(r["preds"]).ravel().astype(int)
        y_test = np.asarray(r["y_test"]).ravel().astype(int)
        c["preds_shape_matches"] = preds.shape == y_test.shape == y_test_ref.shape
    except Exception:
        c["preds_shape_matches"] = False
        preds, y_test = np.array([]), np.array([])

    if c["preds_shape_matches"]:
        acc = float((preds == y_test).mean())
        c["accuracy_at_least_floor"] = bool(acc >= ACC_FLOOR)
        majority = float(np.bincount(y_test).max() / len(y_test))
        c["beats_majority_baseline"] = bool(acc - majority >= MAJORITY_MARGIN)
    else:
        c["accuracy_at_least_floor"] = False
        c["beats_majority_baseline"] = False

    # Anti-faking: re-run the returned model on the re-derived test set.
    if c["model_is_nn_module"] and c["preds_shape_matches"]:
        try:
            model.eval()
            with torch.no_grad():
                logits = model(torch.tensor(X_test_ref))
                model_preds = logits.argmax(dim=1).cpu().numpy().astype(int)
            agree = float((model_preds == preds).mean())
            c["preds_reproduced_by_model"] = bool(agree >= 0.99)
        except Exception:
            c["preds_reproduced_by_model"] = False
    else:
        c["preds_reproduced_by_model"] = False

    return _finalize(score)


def _finalize(score: dict) -> dict:
    score["total"] = sum(1 for v in score["checks"].values() if v)
    score["max"] = len(score["checks"])
    score["passed"] = score["max"] > 0 and score["total"] == score["max"]
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("student", type=Path)
    args = parser.parse_args()
    result = grade(args.student)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)
