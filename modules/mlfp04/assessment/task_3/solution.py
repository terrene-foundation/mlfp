# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP04 — Assessment Task 3: NLP Topic Discovery with NMF (Reference)

Reference implementation. Withheld from students. Verified to pass grader.py.
Framework-first: topic factorisation via kailash-ml DimReductionEngine (NMF).
Dataset: real data/mlfp04/sg_domain_qa.parquet (four most distinct domains).
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import polars as pl

from kailash_ml.engines.dim_reduction import DimReductionEngine
from shared import MLFPDataLoader

CATEGORIES = ["finance", "food", "geography", "transport"]
N_TOPICS = 4

STOPWORDS = set(
    "the a an and or but of to in on at for with as is are was were be been being "
    "this that these those it its they them their from by we you your our he she "
    "his her not have has had do does did will would can could should about into "
    "over under more most some any all than then so such which who what when where "
    "how also use used using many much each both other one two new high low first "
    "second within across per via etc".split()
)
STOPWORDS |= {
    "singapore",
    "singapores",
    "singaporean",
    "singaporeans",
    "country",
    "city",
    "include",
    "including",
    "main",
    "known",
    "provides",
    "provide",
    "offers",
    "offer",
}


def load_documents() -> pl.DataFrame:
    """Load the four distinct domains in the canonical, grader-aligned order."""
    df = MLFPDataLoader().load("mlfp04", "sg_domain_qa.parquet")
    df = df.filter(pl.col("category").is_in(CATEGORIES)).sort(
        ["category", "instruction"]
    )
    return df.with_columns(
        (pl.col("instruction") + " " + pl.col("response")).alias("text")
    )


def build_tfidf(docs: list[str], *, min_df: int = 5, max_df_frac: float = 0.15):
    """Deterministic TF-IDF: sublinear TF, df-pruned vocab, L2-normalised rows."""
    tokens = [
        [t for t in re.findall(r"[a-z]{3,}", d.lower()) if t not in STOPWORDS]
        for d in docs
    ]
    n = len(tokens)
    dfreq: Counter[str] = Counter()
    for ts in tokens:
        for w in set(ts):
            dfreq[w] += 1
    vocab = sorted(w for w, cnt in dfreq.items() if min_df <= cnt <= max_df_frac * n)
    vidx = {w: i for i, w in enumerate(vocab)}
    idf = np.array([np.log((1 + n) / (1 + dfreq[w])) + 1 for w in vocab])
    M = np.zeros((n, len(vocab)))
    for i, ts in enumerate(tokens):
        for w, cnt in Counter(ts).items():
            if w in vidx:
                M[i, vidx[w]] = 1 + np.log(cnt)  # sublinear TF
    M = M * idf
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return M / norms, vocab


def _purity(true: np.ndarray, pred: np.ndarray) -> float:
    return float(
        sum(
            np.unique(true[pred == c], return_counts=True)[1].max()
            for c in np.unique(pred)
        )
        / len(true)
    )


def solve() -> dict:
    """Discover four topics via TF-IDF + NMF — kailash-ml DimReductionEngine."""
    frame = load_documents()
    M, _vocab = build_tfidf(frame["text"].to_list())

    matrix_df = pl.from_numpy(M, schema=[f"t{i}" for i in range(M.shape[1])])
    reduced = DimReductionEngine().reduce(
        matrix_df, algorithm="nmf", n_components=N_TOPICS, seed=42
    )
    W = np.asarray(reduced.transformed)
    doc_topics = W.argmax(axis=1).astype(int)

    # Self-reported purity (the grader recomputes it from the true domains).
    cat_to_id = {c: i for i, c in enumerate(CATEGORIES)}
    true = np.array([cat_to_id[c] for c in frame["category"].to_list()])

    return {
        "doc_topics": [int(v) for v in doc_topics],
        "n_topics": N_TOPICS,
        "topic_purity": _purity(true, doc_topics),
    }


if __name__ == "__main__":
    out = solve()
    sizes = np.bincount(np.array(out["doc_topics"]), minlength=4)
    print(f"n_topics     : {out['n_topics']}")
    print(f"topic_purity : {out['topic_purity']:.4f}")
    print(f"topic sizes  : {sizes.tolist()}")
