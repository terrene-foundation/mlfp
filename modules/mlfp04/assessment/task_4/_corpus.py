# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Instructor-only corpus sampler and topic-quality references for MLFP04
Task 4 (never shipped to students)."""
from __future__ import annotations

import re
from collections import Counter
from itertools import combinations

import numpy as np
import polars as pl

from shared import MLFPDataLoader

SECTIONS = {0: "world", 1: "sports", 2: "business", 3: "sci_tech"}
_TOKEN = re.compile(r"[a-z]+")


def news_pool() -> pl.DataFrame:
    loader = MLFPDataLoader()
    frames = [loader.load("mlfp05", f) for f in ("ag_news.parquet", "ag_news_test.parquet")]
    return pl.concat([f.select("text", "label") for f in frames]).unique("text")


def secret_sample(rng: np.random.Generator, pool: pl.DataFrame) -> tuple[list[str], np.ndarray, int]:
    """Random subset of 3 or 4 sections, ~350 documents each (raw text)."""
    k = int(rng.choice([3, 4]))
    sections = sorted(rng.choice(4, k, replace=False).tolist())
    parts = [pool.filter(pl.col("label") == s).sample(int(rng.integers(300, 400)), seed=int(rng.integers(1 << 31)))
             for s in sections]
    df = pl.concat(parts).sample(fraction=1.0, shuffle=True, seed=int(rng.integers(1 << 31)))
    return df["text"].to_list(), df["label"].to_numpy(), k


def tokens(doc: str) -> set[str]:
    return set(_TOKEN.findall(doc.lower()))


def npmi(doc_tokens: list[set[str]], topic: list[str]) -> float:
    """Mean NPMI over word pairs, document co-occurrence; never co-occurring
    pairs score -1."""
    n = len(doc_tokens)
    words = set(topic)
    wc: Counter = Counter()
    pc: Counter = Counter()
    for t in doc_tokens:
        present = sorted(t & words)
        wc.update(present)
        pc.update(combinations(present, 2))
    vals = []
    for a, b in combinations(topic, 2):
        pij = pc[tuple(sorted((a, b)))] / n
        if pij == 0:
            vals.append(-1.0)
        elif pij == 1:
            vals.append(1.0)
        else:
            vals.append(float(np.log(pij / ((wc[a] / n) * (wc[b] / n))) / -np.log(pij)))
    return float(np.mean(vals)) if vals else -1.0
