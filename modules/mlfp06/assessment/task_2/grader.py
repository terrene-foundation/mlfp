#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP06 Assessment Task 2 — RAG Pipeline with Evaluation.

Usage:
    python grader.py starter.py     # grade your attempt
    python grader.py solution.py    # verify the reference passes

Retrieval is deterministic (embedding cosine over a fixed corpus): the grader
re-derives the gold passage index for each query independently from the SQuAD
parquet and grades recall@1 / recall@3. Generated answers are graded by
GROUNDED FACT containment (does the answer contain a content token from the
gold answer), never by exact text — LLM phrasing is not bit-stable, but at
temperature 0 the grounded outcome is. Floors tolerate at most one drift.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import polars as pl

from shared import MLFPDataLoader

TOP_K = 3
N_CORPUS = 30
N_QUERIES = 6
RECALL1_FLOOR = 5  # of 6
GROUNDED_FLOOR = 5  # of 6

_STOP = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "and",
    "or",
    "for",
    "is",
    "are",
    "was",
    "were",
    "by",
    "at",
    "as",
    "with",
    "that",
    "this",
    "near",
    "present",
    "day",
}


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower())


def _content_tokens(s) -> list[str]:
    return [t for t in _norm(s).split() if t not in _STOP and len(t) >= 3]


def _reference() -> tuple[list[str], list[str], list[int], list[str]]:
    """Independently re-derive (corpus, questions, gold_idx, gold_answer)."""
    df = MLFPDataLoader().load("mlfp06", "squad/squad_v2_300.parquet")
    answerable = df.filter(
        (pl.col("answer").is_not_null()) & (pl.col("answer").str.len_chars() > 0)
    )
    seen: dict[str, int] = {}
    corpus: list[str] = []
    questions: list[str] = []
    gold_idx: list[int] = []
    gold_answer: list[str] = []
    for row in answerable.iter_rows(named=True):
        ctx = row["text"]
        if ctx not in seen:
            seen[ctx] = len(corpus)
            corpus.append(ctx)
        if len(questions) < N_QUERIES and row["question"]:
            if 1 <= len(_content_tokens(row["answer"])) <= 3:
                questions.append(row["question"])
                gold_idx.append(seen[ctx])
                gold_answer.append(row["answer"])
        if len(corpus) >= N_CORPUS and len(questions) >= N_QUERIES:
            break
    return corpus, questions, gold_idx, gold_answer


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_task2", path)
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

    _corpus, _questions, gold_idx, gold_answer = _reference()
    n = len(gold_idx)

    c = score["checks"]
    c["returns_dict"] = isinstance(r, dict)
    if not c["returns_dict"]:
        return _finalize(score)

    retrieved = r.get("retrieved")
    answers = r.get("answers")

    c["retrieved_shape"] = (
        isinstance(retrieved, list)
        and len(retrieved) == n
        and all(isinstance(x, list) for x in retrieved)
    )
    c["answers_shape"] = (
        isinstance(answers, list)
        and len(answers) == n
        and all(isinstance(x, str) and x.strip() for x in answers)
    )
    if not (c["retrieved_shape"] and c["answers_shape"]):
        return _finalize(score)

    # top-k size + valid indices.
    c["topk_size_correct"] = all(len(x) >= TOP_K for x in retrieved)
    c["indices_in_range"] = all(
        all(isinstance(i, int) and 0 <= i < N_CORPUS for i in x) for x in retrieved
    )

    # Retrieval quality (deterministic).
    rec1 = sum(1 for x, g in zip(retrieved, gold_idx) if x and x[0] == g)
    rec3 = sum(1 for x, g in zip(retrieved, gold_idx) if g in x[:TOP_K])
    c["recall_at_1"] = rec1 >= RECALL1_FLOOR
    c["recall_at_3"] = rec3 == n  # gold must be retrieved in top-3 for every query

    # Grounded answers (fact containment, not exact text).
    grounded = 0
    for ans, gold in zip(answers, gold_answer):
        gtok = _content_tokens(gold)
        na = _norm(ans)
        if gtok and any(t in na for t in gtok):
            grounded += 1
    c["answers_grounded"] = grounded >= GROUNDED_FLOOR

    # Answers are not just echoes of the corpus dump — sanity on brevity.
    c["answers_nontrivial"] = all(1 <= len(a.split()) <= 60 for a in answers)

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
