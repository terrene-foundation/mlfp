# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP04 — Assessment Task 4: Topics from News Text (Reference)

Instructor-only reference. Withheld from students.

Decisions: strip the wire-service artefacts (HTML entities, "(Reuters)
Reuters -" bylines) and boilerplate words before vectorising; sublinear
TF-IDF with document-frequency limits; NMF through DimReductionEngine; the
engine returns document weights only, so topic-word weights are recovered by
non-negative regression of the TF-IDF matrix on those weights.
"""
from __future__ import annotations

import re

import numpy as np
import polars as pl
from scipy.optimize import nnls
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from kailash_ml.engines.dim_reduction import DimReductionEngine
from shared import MLFPDataLoader

BOILERPLATE = {
    "said", "says", "reuters", "ap", "afp", "new", "york", "year", "years", "today",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "week", "quot", "lt", "gt", "href", "http", "www", "com", "font", "39", "36",
}
STOP = sorted(ENGLISH_STOP_WORDS | BOILERPLATE)


def load_news(n: int = 1200, seed: int = 0) -> list[str]:
    raw = MLFPDataLoader().load("mlfp05", "ag_news.parquet")
    return raw.sample(n, seed=seed)["text"].to_list()


def clean(text: str) -> str:
    text = text.replace("\\", " ")
    text = re.sub(r"&lt;.*?&gt;", " ", text)
    text = re.sub(r"&?#?[a-z0-9]{1,6};", " ", text)
    text = re.sub(r"\([^()]{1,40}\)\s+[^-]{1,40}?\s+-{1,2}\s", " ", text, count=1)
    return re.sub(r"\s+", " ", text).strip()


def discover_topics(docs: list[str], n_topics: int) -> dict:
    vec = TfidfVectorizer(stop_words=STOP, min_df=3, max_df=0.4, sublinear_tf=True,
                          token_pattern=r"(?u)\b[a-z][a-z]+\b", lowercase=True)
    M = vec.fit_transform([clean(d) for d in docs]).toarray()
    vocab = np.array(vec.get_feature_names_out())
    frame = pl.from_numpy(M, schema=[f"t{i}" for i in range(M.shape[1])])
    res = DimReductionEngine().reduce(frame, algorithm="nmf", n_components=n_topics, seed=0, init="nndsvd", max_iter=500)
    W = np.asarray(res.transformed, dtype=float)
    H = np.array([nnls(W, M[:, j])[0] for j in range(M.shape[1])]).T  # (k, vocab)
    top_words = [vocab[np.argsort(-H[k])[:10]].tolist() for k in range(n_topics)]
    return {"doc_topics": [int(v) for v in W.argmax(axis=1)], "top_words": top_words}


if __name__ == "__main__":
    out = discover_topics(load_news(), 4)
    for k, words in enumerate(out["top_words"]):
        print(k, words)
