# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP04 — Assessment Task 4: Topics from News Text

Implement discover_topics(); problem.md defines its output and how it is
accepted. Submit this file to the portal for grading; the grader runs it on
news batches you have not seen.
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


def load_news(n: int = 1200, seed: int = 0) -> pl.DataFrame:
    """A development batch of raw articles with their section label (0-3).
    The grader never passes labels."""
    return MLFPDataLoader().load("mlfp05", "ag_news.parquet").sample(n, seed=seed)


def discover_topics(docs: list[str], n_topics: int) -> dict:
    raise NotImplementedError


if __name__ == "__main__":
    news = load_news()
    print(news.group_by("label").len().sort("label"))
    print(news["text"].head(3).to_list())
