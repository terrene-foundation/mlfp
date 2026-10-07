# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 4.2: Dense Retrieval with Embeddings + Cosine
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build a dense vector store with cosine similarity ranking
#   - Embed a corpus and a query into the same semantic space
#   - Inspect retrieved chunks and their similarity scores
#   - Apply dense retrieval to a Singapore healthcare FAQ
#
# PREREQUISITES: Exercise 4.1 (chunking), Exercise 1 (Delegate)
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Load and chunk the corpus (sentence chunking from 4.1)
#   2. Build a DenseVectorStore with 30 embedded chunks
#   3. Embed a real eval question and search the store
#   4. Visualise the similarity score distribution
#   5. Apply: Singapore polyclinic FAQ bot
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import re

from shared.mlfp06.ex_4 import (
    DenseVectorStore,
    EMBED_DIM,
    embed_many,
    generate_embedding,
    load_rag_corpus,
    plot_score_distribution,
    run_async,
    split_corpus,
)

# The sentence chunker is re-implemented inline here so this file is
# independently runnable — technique files must not chain across each
# other at runtime (see rules/exercise-standards.md R10).


def chunk_sentence(text: str, max_chunk_chars: int = 500) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > max_chunk_chars and current:
            chunks.append(current.strip())
            current = sent
        else:
            current = current + " " + sent if current else sent
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load corpus and build a chunk subset
# ════════════════════════════════════════════════════════════════════════

corpus = load_rag_corpus(sample_size=1000)
doc_texts, eval_questions, eval_answers = split_corpus(corpus, n_eval=20)

all_chunks: list[dict] = []
for i, text in enumerate(doc_texts):
    for j, chunk in enumerate(chunk_sentence(text, 500)):
        all_chunks.append({"doc_idx": i, "chunk_idx": j, "text": chunk})
print(f"Corpus chunked: {len(doc_texts)} docs -> {len(all_chunks)} chunks")

# Embed a 30-chunk subset so the first run stays quick on a laptop CPU.
# The subset covers the first few documents, including document 0 — the
# source of the eval question we search with in Task 3.
chunk_subset = [c["text"] for c in all_chunks[:30]]
print(f"Embedding {len(chunk_subset)} chunks ({EMBED_DIM}-dim each)...")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Dense retrieval as semantic search
# ════════════════════════════════════════════════════════════════════════
# A dense retriever represents every document and every query as a point
# in a high-dimensional vector space. "Similar meaning" becomes "nearby
# points". The similarity metric is usually cosine similarity — the
# angle between two vectors, ignoring their magnitudes.
#
# Analogy: Imagine every book in a library plotted on a map of concepts.
# Books about dogs cluster near books about wolves; books about bread
# cluster near books about sourdough. Searching is walking to the query
# point and looking around for the nearest books.
#
# WHAT WE USE: a dedicated embedding model served by Ollama —
# nomic-embed-text by default (set OLLAMA_EMBED_MODEL to change it) —
# which turns each chunk into a 768-dimensional vector. Individual
# components have no human-readable meaning; only the ANGLE between two
# vectors matters, which is exactly what cosine similarity measures.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build the DenseVectorStore
# ════════════════════════════════════════════════════════════════════════

embeddings = run_async(embed_many(chunk_subset))

dense_store = DenseVectorStore()
for i, (text, emb) in enumerate(zip(chunk_subset, embeddings)):
    dense_store.add(text, emb, {"chunk_idx": i, "doc_idx": all_chunks[i]["doc_idx"]})

print(f"Dense store: {len(dense_store.documents)} chunks, dim={EMBED_DIM}")
print(
    f"Chunk 0 embedding — first 8 of {len(embeddings[0])} components: "
    f"{[round(x, 4) for x in embeddings[0][:8]]}"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Embed a query and search
# ════════════════════════════════════════════════════════════════════════


async def dense_search(query: str, top_k: int = 5) -> list[dict]:
    q_emb = await generate_embedding(query)
    return dense_store.search(q_emb, top_k=top_k)


test_query = eval_questions[0]
print(f"\nQuery: {test_query[:100]}...")

dense_results = run_async(dense_search(test_query, top_k=5))
print("Top-5 dense results:")
for i, r in enumerate(dense_results):
    print(
        f"  {i+1}. score={r['score']:.3f} (doc {r['metadata']['doc_idx']}): "
        f"{r['text'][:80]}..."
    )
# The eval question was written about document 0, so a good retriever
# ranks a chunk from doc 0 near the top.
source_rank = next(
    (i + 1 for i, r in enumerate(dense_results) if r["metadata"]["doc_idx"] == 0),
    None,
)
print(
    f"Source document (doc 0) first appears at rank: "
    f"{source_rank if source_rank else 'not in top-5'}"
)

# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(dense_store.documents) == len(chunk_subset), "Store should have all chunks"
assert len(embeddings[0]) == EMBED_DIM, f"Embeddings should be {EMBED_DIM}-dim"
assert len(dense_results) == 5, "Dense retrieval should return top-5"
print("\n--- Checkpoint passed --- dense retrieval store built\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise the similarity distribution
# ════════════════════════════════════════════════════════════════════════

all_scores = dense_store.search(embeddings[0], top_k=len(chunk_subset))
score_values = [r["score"] for r in all_scores]
plot_score_distribution(
    score_values,
    title="Dense Retrieval — Cosine Similarity Distribution",
    xlabel="Cosine similarity to chunk 0",
    filename="ex4_02_dense_score_dist.png",
)

# R9A: query-based score distribution — shows how sharply the retriever
# separates relevant from irrelevant chunks for a REAL eval question.
query_all_scores = dense_store.search(
    run_async(generate_embedding(test_query)),
    top_k=len(chunk_subset),
)
query_score_values = [r["score"] for r in query_all_scores]
plot_score_distribution(
    query_score_values,
    title=f"Dense Retrieval — Query Score Distribution",
    xlabel="Cosine similarity to eval query",
    filename="ex4_02_dense_query_dist.png",
)

# INTERPRETATION: A sharp distribution (few high scores, long tail of
# low scores) means the retriever discriminates well — the top-k chunks
# are clearly more relevant than the rest. A flat distribution means
# everything looks equally similar and the top-k is nearly random.


# ════════════════════════════════════════════════════════════════════════
# APPLY — Singapore polyclinic FAQ bot
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore polyclinic cluster handles
# ~30,000 phone calls per month asking variations of the same
# 200 questions: "How early should I arrive for my appointment?",
# "Do I need a referral for a specialist?", "What do I bring for a
# diabetes check-up?".
#
# A dense retriever over a curated FAQ corpus (200 canonical Q&A) with
# cosine similarity is the right tool:
#   - Low corpus size — embeddings cheap to compute once and cache
#   - Semantic matching handles paraphrases ("come early" ~ "arrive
#     before my slot") that keyword search (BM25) would miss
#   - Top-1 retrieval is enough — no need for reranking or hybrid
#   - A compact local embedding model (like the 768-dim
#     nomic-embed-text used here) is plenty for 200 entries
#
# BUSINESS IMPACT (illustrative figures): at 30,000 calls/month and
# S$8/call (call-centre handling cost including agent time + telco), the
# cluster spends S$240,000/month on phone triage. If a dense-retrieval
# chatbot in the patient app resolves 60% of these calls before they
# reach an agent, the saving is S$144,000/month = S$1.7M/year. The
# dense retriever is what makes "my question in my words" find the
# canonical FAQ entry that would otherwise only match word-for-word
# against Ctrl-F.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built a dense vector store with cosine similarity
  [x] Embedded a real corpus subset with a local embedding model
  [x] Embedded a query and retrieved the top-k chunks
  [x] Inspected the similarity distribution to gauge retrieval sharpness
  [x] Mapped dense retrieval to a Singapore polyclinic FAQ use case

  KEY INSIGHT: Dense retrieval turns "find documents containing these
  words" into "find documents meaning roughly this". That's the leap
  BM25 cannot make — and it's why dense vectors became the first stage
  of most production RAG pipelines.

  Next: 03_sparse_bm25.py shows why BM25 is still in the mix...
"""
)
