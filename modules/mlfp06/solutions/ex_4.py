# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 4: RAG Systems — Chunking, Retrieval, and Evaluation
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Implement 4 chunking strategies (fixed, sentence, paragraph, semantic)
#     and explain the trade-offs of each
#   - Build dense retrieval (cosine similarity on embeddings), sparse
#     retrieval (BM25), and hybrid retrieval (RRF fusion)
#   - Implement a cross-encoder re-ranker for precision improvement
#   - Evaluate RAG quality using RAGAS metrics (faithfulness, answer
#     relevance, context relevance, context recall)
#   - Implement HyDE (Hypothetical Document Embeddings) and measure its
#     retrieval improvement
#   - Build a complete end-to-end RAG pipeline with Kaizen Delegate
#
# PREREQUISITES:
#   Exercise 1 (Delegate, prompt engineering).  M4.6 (NLP, embeddings,
#   BM25 keyword search).  Understanding that LLMs have a knowledge
#   cutoff and cannot access documents unless injected into the prompt.
#
# ESTIMATED TIME: ~180 min
#
# TASKS:
#    1. Load RAG corpus (real Q&A dataset from HuggingFace)
#    2. Implement 4 chunking strategies
#    3. Generate dense embeddings via Delegate
#    4. Implement BM25 sparse retrieval from scratch
#    5. Build hybrid retrieval (dense + sparse + RRF)
#    6. Implement cross-encoder re-ranking
#    7. RAGAS evaluation framework
#    8. Implement HyDE (Hypothetical Document Embeddings)
#    9. Full RAG pipeline: retrieve -> rerank -> generate
#   10. Compare retrieval strategies quantitatively
#
# DATASET: neural-bridge/rag-dataset-12000 (HuggingFace)
#   12,000 real RAG question-answer-context triples.  Each row provides
#   the source context, a real question, and a ground-truth answer.
#   We use contexts as the retrieval corpus and questions as test queries.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math
import os
import re
from collections import Counter
from pathlib import Path

import polars as pl

from kaizen_agents import Delegate

from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
if not model:
    raise EnvironmentError("Set DEFAULT_LLM_MODEL or OPENAI_PROD_MODEL in .env")
print(f"LLM Model: {model}")

# ── Data Loading ─────────────────────────────────────────────────────────

CACHE_DIR = Path("data/mlfp06/rag")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "rag_corpus_1k.parquet"

if CACHE_FILE.exists():
    print(f"Loading cached RAG corpus from {CACHE_FILE}")
    corpus = pl.read_parquet(CACHE_FILE)
else:
    print("Downloading neural-bridge/rag-dataset-12000 from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset("neural-bridge/rag-dataset-12000", split="train")
    ds = ds.shuffle(seed=42).select(range(min(1000, len(ds))))
    rows = [
        {
            "section": f"doc_{i:04d}",
            "text": row["context"],
            "question": row["question"],
            "answer": row["answer"],
        }
        for i, row in enumerate(ds)
    ]
    corpus = pl.DataFrame(rows)
    corpus.write_parquet(CACHE_FILE)
    print(f"Cached {corpus.height} documents to {CACHE_FILE}")

print(f"Loaded {corpus.height:,} documents")
print(f"Columns: {corpus.columns}")

# Separate corpus texts and evaluation questions
doc_texts = corpus["text"].to_list()
eval_questions = corpus["question"].to_list()[:20]
eval_answers = corpus["answer"].to_list()[:20]


# ══════════════════════════════════════════════════════════════════════════
# TASK 1: Document Corpus Analysis
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 1: Document Corpus Analysis")
print("=" * 70)

doc_lengths = [len(t) for t in doc_texts]
print(f"Documents: {len(doc_texts)}")
print(
    f"Char lengths: min={min(doc_lengths)}, max={max(doc_lengths)}, "
    f"mean={sum(doc_lengths)/len(doc_lengths):.0f}"
)
print(f"Total characters: {sum(doc_lengths):,}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert len(doc_texts) > 0, "Task 1: corpus should not be empty"
assert len(eval_questions) > 0, "Task 1: should have evaluation questions"
print("✓ Checkpoint 1 passed — corpus loaded and analysed\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 2: Implement 4 Chunking Strategies
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: Chunking Strategies")
print("=" * 70)


def chunk_fixed(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Fixed-size chunking with character overlap."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


def chunk_sentence(text: str, max_chunk_chars: int = 500) -> list[str]:
    """Sentence-boundary chunking: group sentences up to max_chunk_chars."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > max_chunk_chars and current:
            chunks.append(current.strip())
            current = sent
        else:
            current = current + " " + sent if current else sent
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


def chunk_paragraph(text: str) -> list[str]:
    """Paragraph-boundary chunking: split on double newlines."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = [p.strip() for p in paragraphs if p.strip()]
    # Merge very short paragraphs
    merged = []
    current = ""
    for p in chunks:
        if len(current) + len(p) < 300:
            current = current + "\n\n" + p if current else p
        else:
            if current:
                merged.append(current)
            current = p
    if current:
        merged.append(current)
    return merged


def chunk_semantic(text: str, max_chunk_chars: int = 500) -> list[str]:
    """Semantic chunking: split on topic shifts (heading patterns, transitions)."""
    # Split on common topic markers
    markers = re.split(
        r"(?:^|\n)(?=#{1,3}\s|(?:However|Furthermore|In addition|Moreover|"
        r"On the other hand|In contrast|Finally|In conclusion))",
        text,
    )
    chunks = []
    current = ""
    for segment in markers:
        segment = segment.strip()
        if not segment:
            continue
        if len(current) + len(segment) > max_chunk_chars and current:
            chunks.append(current)
            current = segment
        else:
            current = current + "\n" + segment if current else segment
    if current:
        chunks.append(current)
    return [c for c in chunks if c]


# Apply all strategies and compare
sample_text = doc_texts[0]
strategies = {
    "Fixed (500, overlap 100)": chunk_fixed(sample_text, 500, 100),
    "Sentence (max 500)": chunk_sentence(sample_text, 500),
    "Paragraph": chunk_paragraph(sample_text),
    "Semantic (max 500)": chunk_semantic(sample_text, 500),
}

print(f"\nChunking comparison on sample document ({len(sample_text)} chars):")
for name, chunks in strategies.items():
    avg_len = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
    print(f"  {name}: {len(chunks)} chunks, avg {avg_len:.0f} chars")

# Use sentence chunking for the rest of the exercise (best general-purpose)
all_chunks = []
for i, text in enumerate(doc_texts):
    doc_chunks = chunk_sentence(text, max_chunk_chars=500)
    for j, chunk in enumerate(doc_chunks):
        all_chunks.append(
            {"doc_idx": i, "chunk_idx": j, "section": f"doc_{i:04d}", "text": chunk}
        )

chunks_df = pl.DataFrame(all_chunks)
print(f"\nCorpus chunked: {len(doc_texts)} docs -> {chunks_df.height} chunks")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert chunks_df.height > len(doc_texts), "Task 2: should produce more chunks than docs"
assert all(
    name in strategies
    for name in [
        "Fixed (500, overlap 100)",
        "Sentence (max 500)",
        "Paragraph",
        "Semantic (max 500)",
    ]
), "All 4 strategies should be implemented"
print("✓ Checkpoint 2 passed — 4 chunking strategies implemented\n")

# INTERPRETATION: Chunk size trade-offs:
#   Too small (<100 chars): loses context, fragments information
#   Too large (>1000 chars): retrieval less precise, injects irrelevant content
#   Sentence-boundary: avoids mid-sentence cuts, natural reading units
#   Overlap: prevents information loss at chunk boundaries


# ══════════════════════════════════════════════════════════════════════════
# TASK 3: Dense Retrieval (Embedding + Cosine Similarity)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Dense Retrieval (Embeddings)")
print("=" * 70)

EMBED_DIM = 8  # Simplified for demonstration; production uses 768-1536


async def generate_embedding(text: str, delegate: Delegate) -> list[float]:
    """Generate a pseudo-embedding via Delegate."""
    prompt = f"""Convert this text into a numeric vector of exactly {EMBED_DIM} numbers between -1 and 1.
Each number represents: [topic_relevance, factual_density, specificity,
formality, complexity, temporal_recency, sentiment, domain_expertise].

Text: "{text[:300]}"

Return ONLY {EMBED_DIM} comma-separated numbers."""

    response = ""
    async for event in delegate.run(prompt):
        if hasattr(event, "text"):
            response += event.text
    try:
        numbers = [float(x.strip()) for x in response.strip().split(",")[:EMBED_DIM]]
        while len(numbers) < EMBED_DIM:
            numbers.append(0.0)
        return numbers
    except (ValueError, IndexError):
        return [0.0] * EMBED_DIM


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class DenseVectorStore:
    """Vector store using cosine similarity for dense retrieval."""

    def __init__(self):
        self.documents: list[str] = []
        self.embeddings: list[list[float]] = []
        self.metadata: list[dict] = []

    def add(self, text: str, embedding: list[float], meta: dict | None = None):
        self.documents.append(text)
        self.embeddings.append(embedding)
        self.metadata.append(meta or {})

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        scores = [
            (i, cosine_similarity(query_embedding, emb))
            for i, emb in enumerate(self.embeddings)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {
                "text": self.documents[idx],
                "score": score,
                "metadata": self.metadata[idx],
            }
            for idx, score in scores[:top_k]
        ]


# Embed a subset of chunks
chunk_subset = [c["text"] for c in all_chunks[:30]]


async def embed_chunks(texts: list[str]) -> list[list[float]]:
    delegate = Delegate(model=model, max_llm_cost_usd=3.0)
    return [await generate_embedding(t, delegate) for t in texts]


embeddings = asyncio.run(embed_chunks(chunk_subset))

dense_store = DenseVectorStore()
for i, (text, emb) in enumerate(zip(chunk_subset, embeddings)):
    dense_store.add(
        text, emb, {"chunk_idx": i, "section": all_chunks[i].get("section", "")}
    )

print(f"Dense store: {len(dense_store.documents)} chunks, dim={EMBED_DIM}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert len(dense_store.documents) == len(
    chunk_subset
), "Task 3: store should have all chunks"
assert len(embeddings[0]) == EMBED_DIM, f"Embeddings should be {EMBED_DIM}-dim"
print("✓ Checkpoint 3 passed — dense retrieval store built\n")

# INTERPRETATION: Production uses dedicated embedding models (768-1536 dim).
# Cosine similarity measures angular distance: only direction matters, not
# magnitude.  Higher dimension = finer semantic distinctions.


# ══════════════════════════════════════════════════════════════════════════
# TASK 4: BM25 Sparse Retrieval (From Scratch)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 4: BM25 Sparse Retrieval")
print("=" * 70)


class BM25:
    """BM25 sparse retrieval — from-scratch implementation.

    BM25 score(q, d) = sum over terms t in q:
        IDF(t) * (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * |d|/avgdl))

    where:
        IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        tf(t,d) = term frequency of t in document d
        |d| = document length
        avgdl = average document length
        k1, b = tuning parameters
    """

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.N = len(documents)

        # Tokenise
        self.doc_tokens = [self._tokenise(d) for d in documents]
        self.doc_lengths = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / max(self.N, 1)

        # Document frequency: how many docs contain each term
        self.df: dict[str, int] = Counter()
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.df[token] += 1

        # Term frequencies per document
        self.tf: list[Counter] = [Counter(tokens) for tokens in self.doc_tokens]

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        """Simple whitespace + lowercasing tokeniser."""
        return re.findall(r"\w+", text.lower())

    def _idf(self, term: str) -> float:
        """Inverse document frequency with smoothing."""
        df_t = self.df.get(term, 0)
        return math.log((self.N - df_t + 0.5) / (df_t + 0.5) + 1)

    def score(self, query: str, doc_idx: int) -> float:
        """BM25 score for a single query-document pair."""
        query_tokens = self._tokenise(query)
        doc_len = self.doc_lengths[doc_idx]
        tf_doc = self.tf[doc_idx]
        total = 0.0
        for term in query_tokens:
            tf_val = tf_doc.get(term, 0)
            idf = self._idf(term)
            numerator = tf_val * (self.k1 + 1)
            denominator = tf_val + self.k1 * (
                1 - self.b + self.b * doc_len / self.avgdl
            )
            total += idf * numerator / denominator
        return total

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top-k documents by BM25 score."""
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {"text": self.documents[idx], "score": score, "doc_idx": idx}
            for idx, score in scores[:top_k]
        ]


bm25 = BM25(chunk_subset)

# Test BM25 on a real question
test_query = eval_questions[0]
bm25_results = bm25.search(test_query, top_k=5)
print(f"BM25 query: {test_query[:80]}...")
for i, r in enumerate(bm25_results[:3]):
    print(f"  {i+1}. score={r['score']:.3f}: {r['text'][:100]}...")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert len(bm25_results) == 5, "Task 4: BM25 should return top-5"
assert all(r["score"] >= 0 for r in bm25_results), "BM25 scores should be non-negative"
print("✓ Checkpoint 4 passed — BM25 sparse retrieval from scratch\n")

# INTERPRETATION: BM25 is keyword-based: exact term matching with IDF
# weighting.  Strengths: fast, no embeddings, works well for specific
# entity names and technical terms.  Weakness: misses semantic similarity
# (synonyms, paraphrases).  That's why hybrid retrieval helps.


# ══════════════════════════════════════════════════════════════════════════
# TASK 5: Hybrid Retrieval (Dense + Sparse + RRF Fusion)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 5: Hybrid Retrieval (RRF Fusion)")
print("=" * 70)

print(
    """
Reciprocal Rank Fusion (RRF):
  Given ranked lists from multiple retrievers, combine them:
    RRF_score(d) = sum over retrievers r: 1 / (k + rank_r(d))
  where k is a constant (typically 60) and rank_r(d) is the rank of
  document d in retriever r's results (1-indexed).

  RRF does NOT require score normalisation between retrievers.
  This is its key advantage over score-based fusion.
"""
)


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion."""
    doc_scores: dict[int, float] = {}
    doc_texts: dict[int, str] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            # Use text hash as identifier
            doc_id = hash(item["text"])
            doc_texts[doc_id] = item["text"]
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0.0
            doc_scores[doc_id] += 1.0 / (k + rank)

    # Sort by RRF score
    fused = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"text": doc_texts[doc_id], "rrf_score": score} for doc_id, score in fused]


async def hybrid_search(query: str, top_k: int = 5) -> list[dict]:
    """Run both dense and sparse retrieval, fuse with RRF."""
    delegate = Delegate(model=model, max_llm_cost_usd=0.5)

    # Dense retrieval
    query_emb = await generate_embedding(query, delegate)
    dense_results = dense_store.search(query_emb, top_k=top_k * 2)

    # Sparse retrieval (BM25)
    sparse_results = bm25.search(query, top_k=top_k * 2)

    # RRF fusion
    fused = reciprocal_rank_fusion([dense_results, sparse_results])
    return fused[:top_k]


hybrid_results = asyncio.run(hybrid_search(test_query, top_k=5))
print(f"\nHybrid (RRF) results for: {test_query[:80]}...")
for i, r in enumerate(hybrid_results[:3]):
    print(f"  {i+1}. rrf={r['rrf_score']:.4f}: {r['text'][:100]}...")

# ── Checkpoint 5 ─────────────────────────────────────────────────────────
assert len(hybrid_results) > 0, "Task 5: hybrid retrieval should return results"
assert all("rrf_score" in r for r in hybrid_results), "Results should have RRF scores"
print("✓ Checkpoint 5 passed — hybrid retrieval with RRF fusion\n")

# INTERPRETATION: Hybrid retrieval combines semantic understanding (dense)
# with exact keyword matching (sparse).  RRF is robust because it uses
# ranks rather than raw scores — no normalisation needed.


# ══════════════════════════════════════════════════════════════════════════
# TASK 6: Cross-Encoder Re-Ranking
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 6: Cross-Encoder Re-Ranking")
print("=" * 70)

print(
    """
Cross-encoder re-ranking:
  Bi-encoder (embedding): encode query and document SEPARATELY, then
  compare embeddings.  Fast (O(1) per query after indexing) but lossy.

  Cross-encoder: encode query AND document TOGETHER through the same
  transformer.  The model sees both and can attend across them.
  Much more accurate but O(N) per query — expensive for large corpora.

  Production pattern: bi-encoder retrieves top-100, cross-encoder
  re-ranks to top-10.  Best of both worlds.
"""
)


async def cross_encoder_rerank(
    query: str, candidates: list[dict], top_k: int = 3
) -> list[dict]:
    """Re-rank candidates using LLM-based cross-encoding."""
    delegate = Delegate(model=model, max_llm_cost_usd=1.0)

    scored = []
    for candidate in candidates[:10]:  # only re-rank top-10
        prompt = f"""Rate the relevance of this passage to the query on a scale of 0-10.

Query: {query[:300]}
Passage: {candidate['text'][:500]}

Output ONLY a single number (0-10):"""

        response = ""
        async for event in delegate.run(prompt):
            if hasattr(event, "text"):
                response += event.text
        try:
            score = float(re.search(r"[\d.]+", response.strip()).group())
            score = min(max(score, 0), 10)
        except (AttributeError, ValueError):
            score = 5.0

        scored.append({**candidate, "rerank_score": score})

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]


reranked = asyncio.run(cross_encoder_rerank(test_query, hybrid_results, top_k=3))
print(f"Re-ranked results for: {test_query[:80]}...")
for i, r in enumerate(reranked):
    print(f"  {i+1}. rerank={r['rerank_score']:.1f}: {r['text'][:100]}...")

# ── Checkpoint 6 ─────────────────────────────────────────────────────────
assert len(reranked) > 0, "Task 6: re-ranking should return results"
assert all("rerank_score" in r for r in reranked), "Results should have rerank scores"
print("✓ Checkpoint 6 passed — cross-encoder re-ranking complete\n")

# INTERPRETATION: Re-ranking improves precision at the cost of latency.
# The cross-encoder sees query+document together, catching semantic
# relationships that bi-encoders miss.  In production, re-rank top-20-50
# from the retriever to get top-5-10 high-quality results.


# ══════════════════════════════════════════════════════════════════════════
# TASK 7: RAGAS Evaluation Framework
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 7: RAGAS Evaluation Metrics")
print("=" * 70)

print(
    """
RAGAS (Retrieval-Augmented Generation Assessment):
  4 metrics that decompose RAG quality:

  1. Faithfulness: Is the answer supported by the retrieved context?
     Score: fraction of answer claims that can be verified from context.

  2. Answer Relevance: Does the answer address the question?
     Score: cosine similarity between question and answer embeddings.

  3. Context Relevance: Are the retrieved chunks relevant to the question?
     Score: fraction of retrieved sentences that are relevant.

  4. Context Recall: Does the retrieved context cover the ground-truth?
     Score: fraction of ground-truth sentences found in retrieved context.

  Combined: all 4 metrics should be high for a good RAG system.
  Low faithfulness + high relevance = hallucination.
  High faithfulness + low relevance = answering the wrong question.
"""
)


async def compute_ragas_metrics(
    question: str, answer: str, context: str, ground_truth: str
) -> dict:
    """Compute RAGAS-style metrics using LLM-as-judge."""
    delegate = Delegate(model=model, max_llm_cost_usd=1.0)

    # Faithfulness: is the answer supported by context?
    faith_prompt = f"""Given the context and answer, rate how well the answer
is supported by the context (0.0 = not supported, 1.0 = fully supported).

Context: {context[:500]}
Answer: {answer[:300]}

Output ONLY a number between 0.0 and 1.0:"""

    response = ""
    async for event in delegate.run(faith_prompt):
        if hasattr(event, "text"):
            response += event.text
    try:
        faithfulness = float(re.search(r"[\d.]+", response.strip()).group())
        faithfulness = min(max(faithfulness, 0), 1)
    except (AttributeError, ValueError):
        faithfulness = 0.5

    # Answer relevance: does the answer address the question?
    rel_prompt = f"""Rate how well this answer addresses the question
(0.0 = irrelevant, 1.0 = perfectly relevant).

Question: {question[:300]}
Answer: {answer[:300]}

Output ONLY a number between 0.0 and 1.0:"""

    response = ""
    async for event in delegate.run(rel_prompt):
        if hasattr(event, "text"):
            response += event.text
    try:
        answer_relevance = float(re.search(r"[\d.]+", response.strip()).group())
        answer_relevance = min(max(answer_relevance, 0), 1)
    except (AttributeError, ValueError):
        answer_relevance = 0.5

    # Context relevance: are retrieved chunks relevant?
    ctx_rel_prompt = f"""Rate how relevant this context is to the question
(0.0 = irrelevant, 1.0 = highly relevant).

Question: {question[:300]}
Context: {context[:500]}

Output ONLY a number between 0.0 and 1.0:"""

    response = ""
    async for event in delegate.run(ctx_rel_prompt):
        if hasattr(event, "text"):
            response += event.text
    try:
        context_relevance = float(re.search(r"[\d.]+", response.strip()).group())
        context_relevance = min(max(context_relevance, 0), 1)
    except (AttributeError, ValueError):
        context_relevance = 0.5

    # Context recall: does context cover ground truth?
    recall_prompt = f"""Rate how much of the ground-truth answer can be found
in the retrieved context (0.0 = none, 1.0 = all information present).

Ground truth: {ground_truth[:300]}
Context: {context[:500]}

Output ONLY a number between 0.0 and 1.0:"""

    response = ""
    async for event in delegate.run(recall_prompt):
        if hasattr(event, "text"):
            response += event.text
    try:
        context_recall = float(re.search(r"[\d.]+", response.strip()).group())
        context_recall = min(max(context_recall, 0), 1)
    except (AttributeError, ValueError):
        context_recall = 0.5

    return {
        "faithfulness": faithfulness,
        "answer_relevance": answer_relevance,
        "context_relevance": context_relevance,
        "context_recall": context_recall,
    }


# Evaluate on 3 questions (cost-limited)
async def run_ragas_eval():
    results = []
    for i in range(3):
        q = eval_questions[i]
        gt = eval_answers[i]
        # Retrieve context
        retrieved = await hybrid_search(q, top_k=3)
        context = "\n\n".join(r["text"] for r in retrieved)
        # Generate answer
        answer = await rag_answer_simple(q, context)
        # Compute metrics
        metrics = await compute_ragas_metrics(q, answer, context, gt)
        results.append(metrics)
        print(f"\n  Q{i+1}: {q[:60]}...")
        print(f"    Faithfulness:      {metrics['faithfulness']:.2f}")
        print(f"    Answer relevance:  {metrics['answer_relevance']:.2f}")
        print(f"    Context relevance: {metrics['context_relevance']:.2f}")
        print(f"    Context recall:    {metrics['context_recall']:.2f}")
    return results


async def rag_answer_simple(query: str, context: str) -> str:
    """Generate an answer from retrieved context."""
    delegate = Delegate(model=model, max_llm_cost_usd=0.5)
    prompt = f"""Answer the question using ONLY the provided context.
If the context doesn't contain enough information, say so.

Context:
{context[:2000]}

Question: {query}

Answer:"""
    response = ""
    async for event in delegate.run(prompt):
        if hasattr(event, "text"):
            response += event.text
    return response.strip()


ragas_results = asyncio.run(run_ragas_eval())

# Aggregate RAGAS metrics
avg_metrics = {}
for metric in [
    "faithfulness",
    "answer_relevance",
    "context_relevance",
    "context_recall",
]:
    avg_metrics[metric] = sum(r[metric] for r in ragas_results) / len(ragas_results)

print(f"\nAverage RAGAS metrics:")
for k, v in avg_metrics.items():
    print(f"  {k}: {v:.2f}")

# ── Checkpoint 7 ─────────────────────────────────────────────────────────
assert len(ragas_results) >= 3, "Task 7: should evaluate at least 3 questions"
assert all(
    0 <= v <= 1 for v in avg_metrics.values()
), "RAGAS metrics should be in [0, 1]"
print("✓ Checkpoint 7 passed — RAGAS evaluation complete\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 8: HyDE (Hypothetical Document Embeddings)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 8: HyDE — Hypothetical Document Embeddings")
print("=" * 70)

print(
    """
HyDE (Gao et al., 2022):
  Problem: query embeddings and document embeddings live in different
  semantic spaces.  A question like "What causes inflation?" is
  semantically different from a paragraph explaining inflation causes.

  Solution:
    1. Given query q, generate a HYPOTHETICAL document h that would
       answer q (even if h is factually wrong — that's fine)
    2. Embed h (not q)
    3. Retrieve documents similar to h
    4. h is closer to the answer documents in embedding space than q was

  The hypothetical document bridges the query-document semantic gap.
"""
)


async def hyde_retrieve(query: str, top_k: int = 5) -> list[dict]:
    """HyDE: generate hypothetical answer, embed it, retrieve similar docs."""
    delegate = Delegate(model=model, max_llm_cost_usd=1.0)

    # Step 1: Generate hypothetical document
    hyde_prompt = f"""Write a short paragraph (3-5 sentences) that would be the
ideal answer to this question. It does not need to be factually correct —
it should contain the key concepts and vocabulary that a real answer would use.

Question: {query}

Hypothetical answer:"""

    hypo_doc = ""
    async for event in delegate.run(hyde_prompt):
        if hasattr(event, "text"):
            hypo_doc += event.text

    print(f"  HyDE hypothetical: {hypo_doc.strip()[:150]}...")

    # Step 2: Embed the hypothetical document (not the query)
    hypo_emb = await generate_embedding(hypo_doc.strip(), delegate)

    # Step 3: Retrieve using the hypothetical embedding
    results = dense_store.search(hypo_emb, top_k=top_k)
    return results


# Compare HyDE vs direct query retrieval
async def compare_hyde():
    delegate = Delegate(model=model, max_llm_cost_usd=0.5)
    q = eval_questions[0]
    print(f"\nQuery: {q[:80]}...")

    # Direct retrieval
    query_emb = await generate_embedding(q, delegate)
    direct_results = dense_store.search(query_emb, top_k=3)

    # HyDE retrieval
    hyde_results = await hyde_retrieve(q, top_k=3)

    print(f"\n  Direct retrieval (top-1): {direct_results[0]['text'][:100]}...")
    print(f"    Score: {direct_results[0]['score']:.3f}")
    print(f"\n  HyDE retrieval (top-1):   {hyde_results[0]['text'][:100]}...")
    print(f"    Score: {hyde_results[0]['score']:.3f}")

    return direct_results, hyde_results


direct_results, hyde_results = asyncio.run(compare_hyde())

# ── Checkpoint 8 ─────────────────────────────────────────────────────────
assert len(hyde_results) > 0, "Task 8: HyDE should return results"
print("✓ Checkpoint 8 passed — HyDE retrieval implemented and compared\n")

# INTERPRETATION: HyDE typically improves retrieval for abstract queries
# where the question vocabulary differs from the answer vocabulary.
# It costs one extra LLM call per query (to generate the hypothetical).
# Best for: open-ended questions.  Less useful for: specific entity lookups.


# ══════════════════════════════════════════════════════════════════════════
# TASK 9: Full RAG Pipeline
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 9: Full RAG Pipeline (Retrieve -> Rerank -> Generate)")
print("=" * 70)


async def full_rag_pipeline(query: str) -> dict:
    """Complete RAG: hybrid retrieve -> rerank -> generate."""
    # Step 1: Hybrid retrieval (dense + BM25 + RRF)
    candidates = await hybrid_search(query, top_k=10)

    # Step 2: Cross-encoder re-ranking
    reranked = await cross_encoder_rerank(query, candidates, top_k=3)

    # Step 3: Build context from top reranked chunks
    context = "\n\n---\n\n".join(r["text"] for r in reranked)

    # Step 4: Generate answer
    answer = await rag_answer_simple(query, context)

    return {
        "query": query,
        "answer": answer,
        "n_retrieved": len(candidates),
        "n_reranked": len(reranked),
        "context_chars": len(context),
    }


# Run full pipeline on 3 real questions
async def run_full_pipeline():
    results = []
    for i, q in enumerate(eval_questions[:3]):
        print(f"\n  Q{i+1}: {q}")
        result = await full_rag_pipeline(q)
        print(f"  A: {result['answer'][:200]}...")
        print(
            f"  (retrieved={result['n_retrieved']}, "
            f"reranked={result['n_reranked']}, "
            f"context={result['context_chars']} chars)"
        )
        results.append(result)
    return results


pipeline_results = asyncio.run(run_full_pipeline())

# ── Checkpoint 9 ─────────────────────────────────────────────────────────
assert len(pipeline_results) >= 3, "Task 9: should process at least 3 questions"
assert all("answer" in r for r in pipeline_results), "Each result should have an answer"
print("✓ Checkpoint 9 passed — full RAG pipeline complete\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 10: Retrieval Strategy Comparison
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 10: Retrieval Strategy Comparison")
print("=" * 70)

print(
    """
Strategy comparison summary:

| Strategy      | Strengths                        | Weaknesses                    |
|---------------|----------------------------------|-------------------------------|
| Dense (cosine)| Semantic matching, synonyms      | Misses exact terms            |
| BM25 (sparse) | Exact term matching, fast        | No semantic understanding     |
| Hybrid (RRF)  | Best of both, no normalisation   | Slightly slower               |
| + Re-ranking  | Highest precision                | O(N) per query, costly        |
| HyDE          | Bridges query-document gap       | Extra LLM call per query      |

Production recommendation:
  Hybrid (dense + BM25 + RRF) -> cross-encoder re-rank top-20 -> top-5

RAG vs fine-tuning decision:
  Use RAG when: documents change frequently, need citations, audit trail
  Use fine-tuning when: style adaptation, domain vocabulary, latency
  Combine: fine-tune on domain + RAG for up-to-date facts
"""
)

comparison = pl.DataFrame(
    {
        "Strategy": [
            "Dense only",
            "BM25 only",
            "Hybrid (RRF)",
            "Hybrid + Rerank",
            "HyDE + Dense",
        ],
        "Semantic": ["High", "None", "High", "Highest", "High"],
        "Keyword": ["None", "High", "High", "High", "None"],
        "Cost": ["Low", "Zero", "Low", "Medium", "Medium"],
        "Latency": ["Low", "Very low", "Low", "Medium", "Medium"],
        "Best_For": [
            "General QA",
            "Entity lookup",
            "Production default",
            "High-stakes QA",
            "Abstract queries",
        ],
    }
)
print(comparison)

# ── Checkpoint 10 ────────────────────────────────────────────────────────
assert comparison.height >= 5, "Task 10: comparison should cover all strategies"
print("\n✓ Checkpoint 10 passed — retrieval strategy comparison complete\n")


# ══════════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  ✓ 4 chunking strategies: fixed, sentence, paragraph, semantic
    Trade-off: chunk size vs retrieval precision vs context completeness
  ✓ Dense retrieval: embeddings + cosine similarity (semantic matching)
  ✓ BM25 from scratch: IDF-weighted term frequency (keyword matching)
  ✓ Hybrid retrieval: dense + sparse + RRF fusion (best of both)
  ✓ Cross-encoder re-ranking: query+doc through same transformer
    Production pattern: retrieve top-100, rerank to top-10
  ✓ RAGAS evaluation: faithfulness, answer relevance, context relevance,
    context recall — the 4 metrics that decompose RAG quality
  ✓ HyDE: generate hypothetical answer, embed it for retrieval
    Bridges the query-document semantic gap
  ✓ Full RAG pipeline: retrieve -> rerank -> generate with grounding

  NEXT: Exercise 5 (AI Agents) gives the LLM the ability to ACT.
  A ReActAgent calls tools, observes results, reasons about what to
  do next, and iterates — autonomous multi-step problem solving.
"""
)
