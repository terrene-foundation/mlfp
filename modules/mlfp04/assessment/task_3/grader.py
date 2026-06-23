#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP04 Assessment Task 3 — NLP Topic Discovery (NMF).

Usage:
    python grader.py starter.py     # grade your attempt
    python grader.py solution.py    # verify the reference passes

The grader re-derives the canonical document order and the TRUE domain labels
(which the student never sees) and scores the submission's topic assignment for
agreement with the real domains. All ten checks must pass.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from shared import MLFPDataLoader

CATEGORIES = ["finance", "food", "geography", "transport"]
PURITY_FLOOR = 0.65
ARI_FLOOR = 0.45
NMI_FLOOR = 0.55
MAX_TOPIC_SHARE = 0.65


def _reference_truth() -> np.ndarray:
    """True domain id per document, in the canonical (grader-aligned) order."""
    df = MLFPDataLoader().load("mlfp04", "sg_domain_qa.parquet")
    df = df.filter(pl.col("category").is_in(CATEGORIES)).sort(
        ["category", "instruction"]
    )
    cat_to_id = {c: i for i, c in enumerate(CATEGORIES)}
    return np.array([cat_to_id[c] for c in df["category"].to_list()])


def _purity(true: np.ndarray, pred: np.ndarray) -> float:
    return float(
        sum(
            np.unique(true[pred == c], return_counts=True)[1].max()
            for c in np.unique(pred)
        )
        / len(true)
    )


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_t3", path)
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

    c["keys_present"] = all(k in r for k in ("doc_topics", "n_topics", "topic_purity"))
    if not c["keys_present"]:
        return _finalize(score)

    true = _reference_truth()
    n_docs = len(true)
    dt = r["doc_topics"]
    c["doc_topics_length_correct"] = isinstance(dt, list) and len(dt) == n_docs
    if not c["doc_topics_length_correct"]:
        return _finalize(score)

    pred = np.asarray(dt, dtype=int)
    c["n_topics_is_4"] = r["n_topics"] == 4

    uniq, counts = np.unique(pred, return_counts=True)
    c["four_nonempty_topics"] = len(uniq) == 4 and bool((counts > 0).all())
    c["no_degenerate_topic"] = bool(counts.max() <= MAX_TOPIC_SHARE * n_docs)

    try:
        pur = _purity(true, pred)
        c["purity_above_floor"] = pur >= PURITY_FLOOR
    except Exception:
        pur = float("nan")
        c["purity_above_floor"] = False

    try:
        c["ari_above_floor"] = adjusted_rand_score(true, pred) >= ARI_FLOOR
    except Exception:
        c["ari_above_floor"] = False

    try:
        c["nmi_above_floor"] = normalized_mutual_info_score(true, pred) >= NMI_FLOOR
    except Exception:
        c["nmi_above_floor"] = False

    try:
        c["purity_self_report_honest"] = abs(float(r["topic_purity"]) - pur) <= 0.05
    except Exception:
        c["purity_self_report_honest"] = False

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
