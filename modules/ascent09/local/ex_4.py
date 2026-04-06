# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT09 — Exercise 4: Advanced RAG and Evaluation
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Implement hybrid search (BM25 + vector), re-ranking, and
#   evaluate RAG quality with RAGAS-style metrics.
#
# TASKS:
#   1. Implement BM25 retrieval
#   2. Combine BM25 + vector search (hybrid)
#   3. Implement cross-encoder re-ranking
#   4. Evaluate with faithfulness, relevance, answer correctness
#   5. Compare basic vs hybrid vs re-ranked RAG
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math
import os
import re
from collections import Counter

import polars as pl

from kaizen_agents import Delegate
from kaizen import Signature, InputField, OutputField
from kaizen_agents.agents.specialized.simple_qa import SimpleQAAgent

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
print(f"LLM Model: {model}")

# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
regulations = loader.load("ascent09", "sg_regulations.parquet")


# Chunk documents (reuse pattern from Ex 3)
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            bp = max(chunk.rfind("."), chunk.rfind("\n"))
            if bp > chunk_size // 2:
                chunk = text[start : start + bp + 1]
                end = start + bp + 1
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if c]


texts = regulations.select("text").to_series().to_list()
all_chunks = []
for i, text in enumerate(texts):
    for j, chunk in enumerate(chunk_text(text)):
        all_chunks.append({"doc_idx": i, "chunk_idx": j, "text": chunk})

print(f"Total chunks: {len(all_chunks)}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Implement BM25 retrieval
# ══════════════════════════════════════════════════════════════════════


def tokenize_simple(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return re.findall(r"\w+", text.lower())


class BM25:
    """BM25 ranking algorithm for text retrieval."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = documents
        self.n_docs = len(documents)
        self.doc_tokens = [tokenize_simple(d) for d in documents]
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.avg_dl = sum(self.doc_lens) / max(self.n_docs, 1)

        # Build document frequency table
        self.df = Counter()
        for tokens in self.doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.df[token] += 1

    def _idf(self, term: str) -> float:
        """Compute inverse document frequency for a term."""
        df = self.df.get(term, 0)
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str) -> list[float]:
        """Score all documents against a query."""
        query_tokens = tokenize_simple(query)
        scores = []
        for i, doc_tokens in enumerate(self.doc_tokens):
            tf_map = Counter(doc_tokens)
            doc_score = 0.0
            for term in query_tokens:
                tf = tf_map.get(term, 0)
                idf = self._idf(term)
                # TODO: Calculate BM25 numerator: tf * (k1 + 1)
                numerator = ____
                # TODO: Calculate BM25 denominator: tf + k1 * (1 - b + b * doc_len / avg_dl)
                denominator = ____
                doc_score += idf * (numerator / denominator)
            scores.append(doc_score)
        return scores

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top-k documents by BM25 score."""
        scores = self.score(query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [
            {"idx": idx, "score": score, "text": self.docs[idx]}
            for idx, score in ranked[:top_k]
        ]


chunk_texts = [c["text"] for c in all_chunks]
bm25 = BM25(chunk_texts)

test_query = "capital adequacy requirements for banks"
bm25_results = bm25.search(test_query, top_k=5)

print(f"\n=== BM25 Retrieval ===")
print(f"Query: {test_query}")
for i, r in enumerate(bm25_results[:3]):
    print(f"  [{i+1}] score={r['score']:.3f}: {r['text'][:150]}...")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Combine BM25 + vector search (hybrid)
# ══════════════════════════════════════════════════════════════════════


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


async def get_embedding(text: str, delegate: Delegate) -> list[float]:
    """Generate a pseudo-embedding via Delegate."""
    prompt = f"""Convert to 8 numbers between -1 and 1 representing:
[finance, legal, tech, compliance, sentiment, formality, specificity, complexity].
Text: "{text[:300]}"
Return ONLY 8 comma-separated numbers."""

    response = ""
    async for event in delegate.run(prompt):
        if hasattr(event, "text"):
            response += event.text
    try:
        nums = [float(x.strip()) for x in response.strip().split(",")[:8]]
        while len(nums) < 8:
            nums.append(0.0)
        return nums
    except (ValueError, IndexError):
        return [0.0] * 8


def normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize scores to [0, 1]."""
    mn, mx = min(scores), max(scores)
    rng = mx - mn
    if rng == 0:
        return [0.5] * len(scores)
    return [(s - mn) / rng for s in scores]


async def hybrid_search(query: str, top_k: int = 5, alpha: float = 0.5) -> list[dict]:
    """Combine BM25 and vector search with weighted fusion.

    alpha=1.0 means all BM25, alpha=0.0 means all vector.
    """
    # BM25 scores
    bm25_scores = bm25.score(query)
    bm25_norm = normalize_scores(bm25_scores)

    # Vector scores (embed a subset for cost efficiency)
    delegate = Delegate(model=model, max_llm_cost_usd=1.0)
    query_emb = await get_embedding(query, delegate)

    # Embed top BM25 candidates only (efficiency)
    bm25_top_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:20]

    vector_scores = [0.0] * len(chunk_texts)
    for idx in bm25_top_indices:
        emb = await get_embedding(chunk_texts[idx], delegate)
        vector_scores[idx] = cosine_similarity(query_emb, emb)

    vec_norm = normalize_scores(vector_scores)

    # TODO: Calculate hybrid scores using weighted fusion: alpha * bm25 + (1-alpha) * vector
    hybrid_scores = ____

    ranked = sorted(enumerate(hybrid_scores), key=lambda x: x[1], reverse=True)
    return [
        {
            "idx": idx,
            "score": score,
            "text": chunk_texts[idx],
            "bm25": bm25_norm[idx],
            "vector": vec_norm[idx],
        }
        for idx, score in ranked[:top_k]
    ]


hybrid_results = asyncio.run(hybrid_search(test_query, top_k=5, alpha=0.6))

print(f"\n=== Hybrid Search (alpha=0.6) ===")
for i, r in enumerate(hybrid_results[:3]):
    print(
        f"  [{i+1}] hybrid={r['score']:.3f} (bm25={r['bm25']:.3f}, vec={r['vector']:.3f})"
    )
    print(f"       {r['text'][:150]}...")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Implement cross-encoder re-ranking
# ══════════════════════════════════════════════════════════════════════


class RelevanceScore(Signature):
    """Score the relevance of a passage to a query."""

    query: str = InputField(description="The search query")
    passage: str = InputField(description="The candidate passage")
    # TODO: Define OutputField for relevance_score (float 0.0 to 1.0)
    relevance_score: float = ____
    # TODO: Define OutputField for reasoning (brief justification)
    reasoning: str = ____


async def rerank(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    """Re-rank candidates using LLM-based cross-encoder scoring."""
    # TODO: Create SimpleQAAgent with RelevanceScore signature
    reranker = SimpleQAAgent(____)

    scored = []
    for candidate in candidates:
        result = await reranker.run(
            query=query,
            passage=candidate["text"][:500],
        )
        scored.append(
            {
                **candidate,
                "rerank_score": result.relevance_score,
                "rerank_reason": result.reasoning,
            }
        )

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]


reranked = asyncio.run(rerank(test_query, hybrid_results, top_k=3))

print(f"\n=== Re-Ranked Results ===")
for i, r in enumerate(reranked):
    print(f"  [{i+1}] rerank={r['rerank_score']:.3f}: {r['rerank_reason'][:100]}")
    print(f"       {r['text'][:150]}...")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Evaluate with faithfulness, relevance, answer correctness
# ══════════════════════════════════════════════════════════════════════


class RAGEvaluation(Signature):
    """Evaluate RAG output quality across multiple dimensions."""

    question: str = InputField(description="The original question")
    context: str = InputField(description="Retrieved context used for answering")
    answer: str = InputField(description="The generated answer")

    # TODO: Define OutputField for faithfulness (0-1: supported by context?)
    faithfulness: float = ____
    # TODO: Define OutputField for relevance (0-1: addresses the question?)
    relevance: float = ____
    # TODO: Define OutputField for completeness (0-1: covers all aspects?)
    completeness: float = ____
    evaluation_notes: str = OutputField(description="Brief evaluation summary")


async def evaluate_rag(question: str, context: str, answer: str) -> dict:
    """Evaluate a RAG response using RAGAS-style metrics."""
    evaluator = SimpleQAAgent(
        signature=RAGEvaluation,
        model=model,
        max_llm_cost_usd=0.5,
    )

    result = await evaluator.run(
        question=question,
        context=context,
        answer=answer,
    )

    return {
        "faithfulness": result.faithfulness,
        "relevance": result.relevance,
        "completeness": result.completeness,
        "notes": result.evaluation_notes,
    }


async def rag_answer(query: str, results: list[dict]) -> tuple[str, str]:
    """Generate an answer from retrieved context."""
    delegate = Delegate(model=model, max_llm_cost_usd=0.5)
    context = "\n\n---\n\n".join(r["text"] for r in results)
    prompt = f"""Answer using ONLY the provided context.

Context:
{context}

Question: {query}

Answer:"""

    response = ""
    async for event in delegate.run(prompt):
        if hasattr(event, "text"):
            response += event.text
    return response.strip(), context


answer, context = asyncio.run(rag_answer(test_query, reranked))
eval_result = asyncio.run(evaluate_rag(test_query, context, answer))

print(f"\n=== RAG Evaluation ===")
print(f"Question: {test_query}")
print(f"Answer: {answer[:200]}...")
print(f"Faithfulness: {eval_result['faithfulness']:.2f}")
print(f"Relevance:    {eval_result['relevance']:.2f}")
print(f"Completeness: {eval_result['completeness']:.2f}")
print(f"Notes: {eval_result['notes']}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare basic vs hybrid vs re-ranked RAG
# ══════════════════════════════════════════════════════════════════════

comparison = pl.DataFrame(
    {
        "method": ["BM25-only", "Hybrid (BM25+Vector)", "Hybrid + Re-ranked"],
        "retrieval_quality": [
            "Keyword match only",
            "Semantic + keyword",
            "LLM-judged relevance",
        ],
        "latency": ["Fast (no LLM)", "Medium (embeddings)", "Slow (per-candidate LLM)"],
        "best_for": ["Exact term queries", "General questions", "High-stakes answers"],
    }
)

print(f"\n=== Method Comparison ===")
print(comparison)
print(f"\nProduction recommendation:")
print(f"  1. BM25 for fast first-pass retrieval")
print(f"  2. Vector search for semantic expansion")
print(f"  3. Re-ranking only on top-N candidates (cost control)")
print(f"  4. Always evaluate with faithfulness + relevance metrics")

print("\n✓ Exercise 4 complete — hybrid RAG with BM25, re-ranking, and evaluation")
