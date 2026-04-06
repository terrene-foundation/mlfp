# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT09 — Exercise 3: RAG Fundamentals
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build a RAG pipeline from scratch — document chunking,
#   embedding generation, vector similarity search — over Singapore
#   regulatory documents.
#
# TASKS:
#   1. Chunk documents with overlap strategy
#   2. Generate embeddings via Delegate
#   3. Build simple vector store (cosine similarity)
#   4. Implement retrieval with top-k
#   5. Generate answers with retrieved context via Delegate
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math
import os

import polars as pl

from kaizen_agents import Delegate

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
print(f"LLM Model: {model}")

# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
regulations = loader.load("ascent09", "sg_regulations.parquet")

print(f"Loaded {regulations.height:,} regulation sections")
print(f"Columns: {regulations.columns}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Chunk documents with overlap strategy
# ══════════════════════════════════════════════════════════════════════


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: Source text to chunk.
        chunk_size: Target characters per chunk.
        overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of text chunks with overlap.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at a sentence boundary
        if end < len(text):
            last_period = chunk.rfind(".")
            last_newline = chunk.rfind("\n")
            break_point = max(last_period, last_newline)
            if break_point > chunk_size // 2:
                chunk = text[start : start + break_point + 1]
                end = start + break_point + 1

        chunks.append(chunk.strip())
        # TODO: Advance start position with overlap (end minus overlap)
        start = ____

    return [c for c in chunks if c]


# Chunk all regulation documents
all_chunks = []
texts = regulations.select("text").to_series().to_list()
sections = (
    regulations.select("section").to_series().to_list()
    if "section" in regulations.columns
    else ["unknown"] * len(texts)
)

for i, (text, section) in enumerate(zip(texts, sections)):
    doc_chunks = chunk_text(text, chunk_size=500, overlap=100)
    for j, chunk in enumerate(doc_chunks):
        all_chunks.append(
            {
                "doc_idx": i,
                "chunk_idx": j,
                "section": section,
                "text": chunk,
            }
        )

chunks_df = pl.DataFrame(all_chunks)
print(f"\n=== Document Chunking ===")
print(f"Total documents: {len(texts)}")
print(f"Total chunks: {chunks_df.height}")
print(f"Avg chunks per doc: {chunks_df.height / max(len(texts), 1):.1f}")
print(f"Sample chunk: {all_chunks[0]['text'][:200]}...")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Generate embeddings via Delegate
# ══════════════════════════════════════════════════════════════════════


async def generate_embedding(text: str, delegate: Delegate) -> list[float]:
    """Generate a pseudo-embedding by asking the Delegate for numeric features.

    In production, use a dedicated embedding model. Here we demonstrate
    the concept using the LLM to produce feature vectors.
    """
    prompt = f"""Convert this text into a numeric feature vector of exactly 8 numbers between -1 and 1.
Each number represents: [topic_finance, topic_legal, topic_tech, topic_compliance,
                          sentiment, formality, specificity, complexity].

Text: "{text[:300]}"

Return ONLY 8 comma-separated numbers, nothing else. Example: 0.8,-0.2,0.1,0.9,0.3,0.7,0.6,0.4"""

    response = ""
    # TODO: Stream events from delegate.run(prompt), collecting text
    async for event in ____:
        if hasattr(event, "text"):
            response += event.text

    # Parse the numbers
    try:
        numbers = [float(x.strip()) for x in response.strip().split(",")[:8]]
        while len(numbers) < 8:
            numbers.append(0.0)
        return numbers
    except (ValueError, IndexError):
        return [0.0] * 8


async def embed_chunks(chunk_texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of chunks."""
    delegate = Delegate(model=model, max_llm_cost_usd=2.0)
    embeddings = []
    for text in chunk_texts:
        emb = await generate_embedding(text, delegate)
        embeddings.append(emb)
    return embeddings


# Embed a subset of chunks (limit for cost)
chunk_subset = [c["text"] for c in all_chunks[:20]]
embeddings = asyncio.run(embed_chunks(chunk_subset))

print(f"\n=== Embeddings ===")
print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding dimension: {len(embeddings[0])}")
print(f"Sample embedding: {embeddings[0]}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Build simple vector store (cosine similarity)
# ══════════════════════════════════════════════════════════════════════


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SimpleVectorStore:
    """Minimal vector store using cosine similarity for retrieval."""

    def __init__(self):
        self.documents: list[str] = []
        self.embeddings: list[list[float]] = []
        self.metadata: list[dict] = []

    def add(self, text: str, embedding: list[float], meta: dict | None = None):
        """Add a document with its embedding to the store."""
        self.documents.append(text)
        self.embeddings.append(embedding)
        self.metadata.append(meta or {})

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[dict]:
        """Find the top-k most similar documents."""
        scores = []
        for i, emb in enumerate(self.embeddings):
            sim = cosine_similarity(query_embedding, emb)
            scores.append((i, sim))

        # TODO: Sort scores by similarity in descending order
        ____
        results = []
        for idx, score in scores[:top_k]:
            results.append(
                {
                    "text": self.documents[idx],
                    "score": score,
                    "metadata": self.metadata[idx],
                }
            )
        return results


# Populate the vector store
store = SimpleVectorStore()
for i, (text, emb) in enumerate(zip(chunk_subset, embeddings)):
    store.add(text, emb, {"chunk_idx": i, "section": all_chunks[i].get("section", "")})

print(f"\n=== Vector Store ===")
print(f"Documents indexed: {len(store.documents)}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Implement retrieval with top-k
# ══════════════════════════════════════════════════════════════════════


async def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """Embed a query and retrieve the most relevant chunks."""
    delegate = Delegate(model=model, max_llm_cost_usd=0.5)
    query_emb = await generate_embedding(query, delegate)
    results = store.search(query_emb, top_k=top_k)
    return results


test_query = (
    "What are the compliance requirements for financial institutions in Singapore?"
)
retrieved = asyncio.run(retrieve(test_query, top_k=3))

print(f"\n=== Retrieval (top-3) ===")
print(f"Query: {test_query}")
for i, result in enumerate(retrieved):
    print(f"\n  Result {i+1} (score: {result['score']:.3f}):")
    print(f"    {result['text'][:200]}...")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Generate answers with retrieved context via Delegate
# ══════════════════════════════════════════════════════════════════════


async def rag_answer(query: str) -> str:
    """Full RAG pipeline: retrieve context, then generate answer."""
    # Retrieve relevant chunks
    delegate = Delegate(model=model, max_llm_cost_usd=1.0)
    query_emb = await generate_embedding(query, delegate)
    results = store.search(query_emb, top_k=3)

    # Build context from retrieved chunks
    context = "\n\n---\n\n".join(r["text"] for r in results)

    # TODO: Write the RAG prompt that instructs the model to answer using ONLY the context
    prompt = ____

    response = ""
    async for event in delegate.run(prompt):
        if hasattr(event, "text"):
            response += event.text

    return response.strip()


queries = [
    "What are the compliance requirements for financial institutions in Singapore?",
    "What penalties apply for regulatory violations?",
    "How should companies handle data protection under Singapore law?",
]

print(f"\n=== RAG Q&A ===")
for query in queries:
    answer = asyncio.run(rag_answer(query))
    print(f"\nQ: {query}")
    print(f"A: {answer[:300]}...")

print("\n✓ Exercise 3 complete — RAG pipeline with chunking, embeddings, and retrieval")
