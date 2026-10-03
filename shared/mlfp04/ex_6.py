# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP04 Exercise 6 — NLP & Topic Modelling.

Contains: news-corpus loading (AG News, real human-labelled news), a
human-labelled sentiment dataset loader (SST-2), NPMI coherence scoring,
a naive sentiment lexicon, the env-configured topic embedding model,
APPLY-phase scenario text, and plot output directory management.

Technique-specific code (TF-IDF/BM25 scoring, NMF decomposition, LDA
fitting, BERTopic pipeline, Word2Vec sentiment classifier) does NOT
belong here — it lives in the per-technique files under
modules/mlfp04/solutions/ex_6/.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from shared.data_loader import MLFPDataLoader
from shared.kailash_helpers import setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()

OUTPUT_DIR = Path("outputs") / "ex6_nlp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# TOY CORPUS — small Singapore-flavoured corpus for teaching derivations
# ════════════════════════════════════════════════════════════════════════

TOY_CORPUS: list[str] = [
    "Singapore economy grew strongly in 2024",
    "Singapore property market shows resilience",
    "MAS tightens monetary policy amid global uncertainty",
    "Property developers report strong demand",
    "Singapore government announces new housing measures",
    "Global markets react to central bank decisions",
    "Technology sector leads Singapore stock exchange",
    "Housing prices continue upward trend in Singapore",
]


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — AG News: real news articles with human topic labels
# ════════════════════════════════════════════════════════════════════════
# 5,000 news articles (title + lead) from the AG News benchmark, each
# labelled by humans with one of four sections. The labels are NEVER used
# to fit a topic model — only afterwards, to check what the unsupervised
# topics line up with.

NEWS_LABELS: dict[int, str] = {
    0: "world",
    1: "sports",
    2: "business",
    3: "sci_tech",
}

# Wire-service boilerplate and weekday words that appear in every section
# and would otherwise form a meaningless "catch-all" topic.
NEWS_STOP_WORDS: list[str] = sorted(
    ENGLISH_STOP_WORDS
    | {
        "said", "says", "reuters", "ap", "afp", "new", "york", "year",
        "years", "today", "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday", "week",
    }
)


def clean_news_text(text: str) -> str:
    """Strip AG News encoding artefacts and the '(Source) Source -' prefix."""
    text = text.replace("\\", " ").replace("#36;", "$").replace("#39;", "'")
    text = text.replace("quot;", '"')
    text = re.sub(r"&lt;.*?&gt;", " ", text)
    text = re.sub(r"&?#?[a-z0-9]{1,6};", " ", text)
    text = re.sub(r"\([^()]{1,40}\)\s+[^-]{1,40}?\s+-{1,2}\s", " ", text, count=1)
    return re.sub(r"\s+", " ", text).strip()


def load_corpus(max_docs: int | None = None, seed: int = 42) -> pl.DataFrame:
    """Load the AG News corpus as a polars DataFrame.

    Columns: ``text`` (cleaned title + lead), ``label`` (0-3) and
    ``category`` (world / sports / business / sci_tech). Duplicate texts
    are removed so every document is counted once. ``max_docs`` draws a
    reproducible random subset (useful for the slower neural pipeline).
    """
    loader = MLFPDataLoader()
    raw = loader.load("mlfp05", "ag_news.parquet")
    df = (
        raw.with_columns(
            pl.col("text").map_elements(clean_news_text, return_dtype=pl.String),
            pl.col("label")
            .replace_strict(NEWS_LABELS, return_dtype=pl.String)
            .alias("category"),
        )
        .filter(pl.col("text").str.len_chars() > 0)
        .unique(subset="text", keep="first", maintain_order=True)
    )
    if max_docs is not None and max_docs < df.height:
        df = df.sample(n=max_docs, seed=seed, shuffle=False)
    return df


def corpus_as_lists(
    df: pl.DataFrame,
) -> tuple[list[str], list[str]]:
    """Split a corpus frame into parallel (documents, categories) lists."""
    return df["text"].to_list(), df["category"].to_list()


# ════════════════════════════════════════════════════════════════════════
# NPMI TOPIC COHERENCE
# ════════════════════════════════════════════════════════════════════════


def compute_npmi(
    documents: list[str],
    topic_words: list[list[str]],
    analyzer: Callable[[str], list[str]] | None = None,
) -> list[float]:
    """Normalised Pointwise Mutual Information coherence, one value per topic.

    NPMI(w_i, w_j) = log(P(w_i, w_j) / (P(w_i) P(w_j))) / (-log P(w_i, w_j))

    Probabilities are document frequencies. Range is [-1, 1]: 0 means the
    two words co-occur exactly as often as chance predicts, > 0 above
    chance, and -1 means they never appear in the same document (such
    pairs are scored -1, not skipped). Pass the SAME ``analyzer`` your
    vectoriser used (``vectorizer.build_analyzer()``) so documents are
    tokenised exactly like the topic words were; the default is sklearn's
    standard lowercase word tokeniser.
    """
    if analyzer is None:
        analyzer = CountVectorizer().build_analyzer()
    topic_vocab = {w for topic in topic_words for w in topic}

    n_docs = len(documents)
    word_doc_count: Counter[str] = Counter()
    pair_doc_count: Counter[tuple[str, str]] = Counter()
    for doc in documents:
        present = sorted(set(analyzer(doc)) & topic_vocab)
        word_doc_count.update(present)
        pair_doc_count.update(combinations(present, 2))

    coherences: list[float] = []
    for topic in topic_words:
        scores: list[float] = []
        for w_i, w_j in combinations(topic, 2):
            pair = tuple(sorted((w_i, w_j)))
            p_i = word_doc_count[w_i] / n_docs
            p_j = word_doc_count[w_j] / n_docs
            p_ij = pair_doc_count[pair] / n_docs
            if p_ij == 0.0:
                scores.append(-1.0)
            elif p_ij == 1.0:
                scores.append(1.0)
            else:
                scores.append(float(np.log(p_ij / (p_i * p_j)) / -np.log(p_ij)))
        coherences.append(float(np.mean(scores)) if scores else 0.0)
    return coherences


# ════════════════════════════════════════════════════════════════════════
# SENTIMENT — human-labelled data + a naive lexicon baseline
# ════════════════════════════════════════════════════════════════════════
# SST-2 (Stanford Sentiment Treebank): English movie-review text labelled
# positive (1) / negative (0) by human annotators. The train split holds
# ~67K labelled phrases and sentences; the validation split holds 872
# complete sentences from different reviews, used here as the test set.

SENTIMENT_DATASET = "stanfordnlp/sst2"

# Lowercase words, keeping contractions ("it's") and the negation token
# "n't" that SST-2 splits off ("does n't") — negation matters for sentiment.
_REVIEW_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|n't")


def tokenize_review(text: str) -> list[str]:
    """Tokenise a review into lowercase words (keeps "n't")."""
    return _REVIEW_TOKEN_RE.findall(text.lower())


def load_sentiment_reviews() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return ``(train_df, test_df)`` with columns ``text`` and ``label``.

    Train texts are de-duplicated, and any test sentence whose exact text
    also appears in train is dropped, so test accuracy is measured on
    unseen text.
    """
    def _to_polars(split: str) -> pl.DataFrame:
        ds = MLFPDataLoader.load_hf(SENTIMENT_DATASET, split=split)
        return pl.DataFrame(
            {"text": ds["sentence"], "label": ds["label"]}
        ).with_columns(pl.col("text").str.strip_chars())

    train_df = _to_polars("train").unique(
        subset="text", keep="first", maintain_order=True
    )
    test_df = _to_polars("validation").join(
        train_df.select("text"), on="text", how="anti"
    )
    return train_df, test_df


POSITIVE_WORDS: frozenset[str] = frozenset(
    {
        "good", "great", "excellent", "best", "wonderful", "love", "loved",
        "enjoyable", "funny", "beautiful", "brilliant", "fine", "fun",
        "charming", "moving", "delightful", "entertaining", "perfect",
        "strong", "smart",
    }
)

NEGATIVE_WORDS: frozenset[str] = frozenset(
    {
        "bad", "worst", "boring", "dull", "awful", "terrible", "poor",
        "waste", "stupid", "mess", "flat", "lame", "tedious", "weak",
        "fails", "pointless", "unfunny", "disappointing", "mediocre",
        "predictable",
    }
)


def lexicon_sentiment(docs: Iterable[str]) -> np.ndarray:
    """Score each document in ``docs`` as a scalar in [-1, 1] via a naive lexicon.

    (pos - neg) / (pos + neg), zero when neither list matches. Negation is
    ignored, so "not good" scores as positive.
    """
    scores: list[float] = []
    for doc in docs:
        words = set(tokenize_review(doc))
        pos = len(words & POSITIVE_WORDS)
        neg = len(words & NEGATIVE_WORDS)
        total = pos + neg
        scores.append((pos - neg) / total if total > 0 else 0.0)
    return np.asarray(scores, dtype=np.float64)


# ════════════════════════════════════════════════════════════════════════
# TOPIC EMBEDDING MODEL — read from .env, never hardcoded
# ════════════════════════════════════════════════════════════════════════


def topic_embedding_model() -> str:
    """Return the sentence-transformers model name from ``TOPIC_EMBED_MODEL``.

    Set it in ``.env``. An English-only model suits this English news
    corpus; a multilingual sentence-transformers model is needed if your
    documents mix languages.
    """
    name = os.environ.get("TOPIC_EMBED_MODEL", "").strip()
    if not name:
        raise RuntimeError(
            "TOPIC_EMBED_MODEL is not set. Add a sentence-transformers model "
            "name to your .env (see .env.example) and re-run."
        )
    return name


# ════════════════════════════════════════════════════════════════════════
# SCENARIO HELPERS — APPLY-phase business context
# ════════════════════════════════════════════════════════════════════════
# Organisations are generic and every figure is an illustrative
# assumption for the arithmetic, not a measured or published number.

SCENARIOS: dict[str, str] = {
    "tfidf_bm25": (
        "CASE (illustrative): a Singapore engineering group's internal "
        "document search over ~180K reports, from 2-page memos to 80-page "
        "manuals. Ranking decides which report a manager sees first for "
        "'turbine blade fatigue'. Assume a missed relevant report causes "
        "duplicated R&D costing ~S$450K. BM25's length normalisation "
        "(b=0.75) lets short memos compete fairly with long manuals."
    ),
    "nmf_topics": (
        "CASE (illustrative): a news publisher tagging its daily wire "
        "intake (assume ~2,400 articles/day). NMF runs nightly on the "
        "day's TF-IDF matrix and proposes interpretable topics (markets, "
        "elections, sport, technology) for the website's section pages "
        "and recommendation engine. Non-negativity means the topic-keyword "
        "report can be audited directly by the editorial desk."
    ),
    "lda_topics": (
        "CASE (illustrative): a media-monitoring service routes news "
        "stories to client analysts (assume ~60K stories/year). Many "
        "stories genuinely span sections — a tech company's earnings "
        "report is both 'business' and 'technology'. LDA's per-document "
        "topic proportions let one story be routed to BOTH desks instead "
        "of forcing a single label."
    ),
    "bertopic": (
        "CASE (illustrative): a regional platform clusters customer-support "
        "tickets (assume ~35K/week). Sentence-embedding topics group "
        "paraphrases that share few words ('refund not received' vs "
        "'still waiting for my money back'). Tickets in several languages "
        "need a MULTILINGUAL sentence-transformers model — set "
        "TOPIC_EMBED_MODEL accordingly; an English-only model will not "
        "align languages."
    ),
    "sentiment_word2vec": (
        "CASE (illustrative): a Singapore bank triages app-store reviews "
        "(assume ~40K/month) so that negative reviews reach the customer "
        "experience team quickly. A word-embedding + logistic-regression "
        "classifier trained on labelled English reviews is cheap to run. "
        "Word2Vec has no vectors for unseen words, so other languages need "
        "their own (or cross-lingually aligned) embeddings and labelled "
        "data; subword models such as fastText help with unseen words, "
        "not with crossing languages."
    ),
}


def print_scenario(name: str) -> None:
    """Print a named Singapore/APAC scenario block for the APPLY phase."""
    body = SCENARIOS.get(name, "")
    if not body:
        return
    print("\n" + "=" * 70)
    print(f"  APPLY — {name}")
    print("=" * 70)
    print(body)
    print("=" * 70 + "\n")
