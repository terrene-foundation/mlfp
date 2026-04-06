# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT08 — Exercise 2: Bag of Words and TF-IDF
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Implement BoW, TF-IDF, and BM25 from scratch, then use them
#   for document classification on Singapore Parliament speeches via
#   TrainingPipeline.
#
# TASKS:
#   1. Build Bag-of-Words representation
#   2. Implement TF-IDF from formula
#   3. Implement BM25 scoring
#   4. Compare retrieval quality across methods
#   5. Use TF-IDF features for text classification via TrainingPipeline
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
from collections import Counter

import polars as pl

from kailash_ml import DataExplorer, TrainingPipeline, ModelVisualizer

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
df = loader.load("ascent08", "sg_parliament_speeches.parquet")

explorer = DataExplorer()
summary = explorer.analyze(df)
print(f"=== Dataset: {df.height} speeches, columns: {df.columns} ===")
print(summary)


# ── Helpers ───────────────────────────────────────────────────────────


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer with lowercasing."""
    import re

    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return text.split()


corpus = df.select("text").to_series().to_list()
tokenized_corpus = [tokenize(doc) for doc in corpus]

print(f"Corpus: {len(corpus)} documents")
print(
    f"Avg tokens/doc: {sum(len(d) for d in tokenized_corpus) / len(tokenized_corpus):.0f}"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Bag-of-Words representation
# ══════════════════════════════════════════════════════════════════════


def build_vocabulary(
    tokenized_docs: list[list[str]], max_vocab: int = 2000
) -> list[str]:
    """Build vocabulary from most common tokens across corpus."""
    word_counts = Counter()
    for doc in tokenized_docs:
        word_counts.update(doc)
    return [word for word, _ in word_counts.most_common(max_vocab)]


def bow_vectorize(tokens: list[str], vocab: list[str]) -> list[int]:
    """Convert token list to BoW vector using vocabulary."""
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    vector = [0] * len(vocab)
    for token in tokens:
        if token in word_to_idx:
            vector[word_to_idx[token]] += 1
    return vector


vocab = build_vocabulary(tokenized_corpus, max_vocab=2000)
print(f"\nVocabulary size: {len(vocab)}")
print(f"Top 10 terms: {vocab[:10]}")

bow_vectors = [bow_vectorize(doc, vocab) for doc in tokenized_corpus]
print(f"BoW matrix shape: ({len(bow_vectors)}, {len(bow_vectors[0])})")

# Sparsity check
total_entries = len(bow_vectors) * len(bow_vectors[0])
nonzero = sum(1 for row in bow_vectors for v in row if v > 0)
print(f"Sparsity: {1 - nonzero / total_entries:.2%} zeros")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: TF-IDF from formula
# ══════════════════════════════════════════════════════════════════════


def compute_tf(tokens: list[str], vocab: list[str]) -> list[float]:
    """Term frequency: count(t, d) / len(d)."""
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    tf = [0.0] * len(vocab)
    doc_len = len(tokens) if tokens else 1
    for token in tokens:
        if token in word_to_idx:
            tf[word_to_idx[token]] += 1.0 / doc_len
    return tf


def compute_idf(tokenized_docs: list[list[str]], vocab: list[str]) -> list[float]:
    """Inverse document frequency: log(N / df(t))."""
    n_docs = len(tokenized_docs)
    doc_freq = Counter()
    for doc in tokenized_docs:
        unique_tokens = set(doc)
        for token in unique_tokens:
            if token in set(vocab):
                doc_freq[token] += 1

    idf = []
    for word in vocab:
        df = doc_freq.get(word, 0)
        idf.append(math.log((n_docs + 1) / (df + 1)) + 1)  # smoothed IDF
    return idf


def tfidf_vectorize(
    tokens: list[str], vocab: list[str], idf: list[float]
) -> list[float]:
    """TF-IDF = TF * IDF."""
    tf = compute_tf(tokens, vocab)
    return [t * i for t, i in zip(tf, idf)]


idf_values = compute_idf(tokenized_corpus, vocab)
tfidf_vectors = [tfidf_vectorize(doc, vocab, idf_values) for doc in tokenized_corpus]

print(f"\nTF-IDF matrix shape: ({len(tfidf_vectors)}, {len(tfidf_vectors[0])})")
print(f"IDF range: [{min(idf_values):.2f}, {max(idf_values):.2f}]")

# Show highest TF-IDF terms for first document
first_tfidf = tfidf_vectors[0]
top_indices = sorted(
    range(len(first_tfidf)), key=lambda i: first_tfidf[i], reverse=True
)[:10]
print(
    f"\nTop TF-IDF terms (doc 0): {[(vocab[i], f'{first_tfidf[i]:.4f}') for i in top_indices]}"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 3: BM25 scoring
# ══════════════════════════════════════════════════════════════════════


def bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    tokenized_docs: list[list[str]],
    vocab: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """BM25 score for a query against a single document."""
    n_docs = len(tokenized_docs)
    avg_dl = sum(len(d) for d in tokenized_docs) / n_docs
    dl = len(doc_tokens)

    doc_freq = Counter()
    for doc in tokenized_docs:
        for token in set(doc):
            doc_freq[token] += 1

    score = 0.0
    doc_counts = Counter(doc_tokens)
    for qt in query_tokens:
        if qt not in set(vocab):
            continue
        df = doc_freq.get(qt, 0)
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        tf = doc_counts.get(qt, 0)
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
        score += idf * tf_norm

    return score


query = tokenize("economic policy budget Singapore")
print(f"\n--- BM25 Retrieval for: '{' '.join(query)}' ---")

scores = []
for i, doc_tokens in enumerate(tokenized_corpus):
    s = bm25_score(query, doc_tokens, tokenized_corpus, vocab)
    scores.append((i, s))

scores.sort(key=lambda x: x[1], reverse=True)
for rank, (idx, score) in enumerate(scores[:5]):
    snippet = corpus[idx][:80].replace("\n", " ")
    print(f"  Rank {rank+1}: doc[{idx}] score={score:.3f} — {snippet}...")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Compare retrieval quality
# ══════════════════════════════════════════════════════════════════════


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


query_bow = bow_vectorize(query, vocab)
query_tfidf = tfidf_vectorize(query, vocab, idf_values)

print(f"\n--- Retrieval Comparison ---")
print(f"{'Rank':<6} {'BoW cos':<12} {'TF-IDF cos':<12} {'BM25':<10}")
print("-" * 40)

bow_scores = [
    (
        i,
        cosine_similarity(
            [float(v) for v in query_bow], [float(v) for v in bow_vectors[i]]
        ),
    )
    for i in range(len(corpus))
]
tfidf_scores = [
    (i, cosine_similarity(query_tfidf, tfidf_vectors[i])) for i in range(len(corpus))
]
bm25_scores = scores[:5]

bow_scores.sort(key=lambda x: x[1], reverse=True)
tfidf_scores.sort(key=lambda x: x[1], reverse=True)

for rank in range(min(5, len(corpus))):
    b_idx, b_s = bow_scores[rank]
    t_idx, t_s = tfidf_scores[rank]
    bm_idx, bm_s = bm25_scores[rank] if rank < len(bm25_scores) else (-1, 0)
    print(
        f"  {rank+1:<6} doc[{b_idx}]={b_s:.3f}  doc[{t_idx}]={t_s:.3f}  doc[{bm_idx}]={bm_s:.3f}"
    )

print("\nBoW: raw frequency, no term importance weighting")
print("TF-IDF: down-weights common terms, highlights distinctive terms")
print("BM25: TF saturation + length normalization, best for retrieval")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: TF-IDF features for classification via TrainingPipeline
# ══════════════════════════════════════════════════════════════════════

pipeline = TrainingPipeline(
    model_type="text_classifier",
    target="topic",
    features=["text"],
)

result = pipeline.fit(df)
predictions = pipeline.predict(df)

print(f"\n=== TrainingPipeline Text Classification ===")
print(f"Model type: text_classifier")
print(f"Training result: {result}")

viz = ModelVisualizer()
if hasattr(result, "metrics"):
    print(f"Metrics: {result.metrics}")

print(
    "\n✓ Exercise 2 complete — BoW, TF-IDF, BM25 from scratch + TrainingPipeline classification"
)
