# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 4.1: Chunking Strategies for RAG Corpora
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Implement 4 chunking strategies (fixed, sentence, paragraph, semantic)
#   - Understand the chunk size / retrieval precision trade-off
#   - Visualise how each strategy partitions a real document
#   - Choose chunking for a Singapore legal-search use case
#
# PREREQUISITES:
#   Exercise 1 (Delegate, prompt engineering). Understanding that LLMs
#   have a context window.
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Load the RAG corpus via the shared loader
#   2. Implement 4 chunking strategies
#   3. Apply all strategies to the full corpus
#   4. Visualise chunk counts and average sizes
#   5. Apply: pick a chunker for a Singapore legal-search SaaS
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import re

import polars as pl

from shared.mlfp06.ex_4 import (
    load_rag_corpus,
    plot_chunking_comparison,
    split_corpus,
)

# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load corpus
# ════════════════════════════════════════════════════════════════════════

# TODO: Call load_rag_corpus(sample_size=1000) and split_corpus(corpus, n_eval=20)
# Hint: split_corpus returns (doc_texts, eval_questions, eval_answers)
corpus = ____
doc_texts, eval_questions, eval_answers = ____

print(f"Loaded {corpus.height:,} documents")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why chunking exists
# ════════════════════════════════════════════════════════════════════════
# An LLM can only see what fits in its context window. A 500-page employee
# handbook cannot be dropped into a prompt. RAG splits documents into
# chunks, embeds each chunk, and injects only the top-k most relevant
# chunks at query time. The chunking step decides how the document is
# split — and bad chunking is the single biggest source of RAG failures.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Implement 4 chunking strategies
# ════════════════════════════════════════════════════════════════════════


def chunk_fixed(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Fixed-size chunking with character overlap."""
    # TODO: Slide a window of chunk_size across text, stepping by
    # (chunk_size - overlap). Return a list of stripped non-empty chunks.
    ____


def chunk_sentence(text: str, max_chunk_chars: int = 500) -> list[str]:
    """Sentence-boundary chunking: group sentences up to max_chunk_chars."""
    # TODO: Split on sentence-ending punctuation, then group sentences
    # until adding the next one would exceed max_chunk_chars.
    # Hint: use re.split(r"(?<=[.!?])\s+", text)
    ____


def chunk_paragraph(text: str) -> list[str]:
    """Paragraph-boundary chunking with merging of very short paragraphs."""
    # TODO: Split on double newlines; merge adjacent paragraphs until
    # the combined length reaches ~300 chars.
    ____


def chunk_semantic(text: str, max_chunk_chars: int = 500) -> list[str]:
    """Heuristic semantic chunking — split on heading / transition markers."""
    # TODO: Split on markdown headings (#, ##, ###) or transition phrases
    # (However, Furthermore, In conclusion, ...). Group segments until
    # the combined length exceeds max_chunk_chars.
    ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Apply strategies to a sample document
# ════════════════════════════════════════════════════════════════════════

sample_text = doc_texts[0]
# TODO: Build a dict mapping strategy name -> list of chunks for each
# of the 4 strategies above.
strategies = ____

for name, chunks in strategies.items():
    avg_len = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
    print(f"  {name}: {len(chunks)} chunks, avg {avg_len:.0f} chars")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Chunk the full corpus with the sentence strategy
# ════════════════════════════════════════════════════════════════════════

all_chunks = []
# TODO: Loop over doc_texts, apply chunk_sentence, and append each chunk
# as a dict {"doc_idx", "chunk_idx", "section", "text"} to all_chunks.
____

chunks_df = pl.DataFrame(all_chunks)
print(f"Corpus chunked: {len(doc_texts)} docs -> {chunks_df.height} chunks")

# ── Checkpoint ──────────────────────────────────────────────────────────
assert chunks_df.height > len(doc_texts), "Sentence chunking should expand docs"
assert all(
    len(v) > 0 for v in strategies.values()
), "All strategies should produce chunks"
print("\n--- Checkpoint passed ---\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise chunk counts and sizes
# ════════════════════════════════════════════════════════════════════════

# TODO: Call plot_chunking_comparison(strategies, title=..., filename=...)
____


# ════════════════════════════════════════════════════════════════════════
# APPLY — Singapore legal-search SaaS
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore legal-tech startup builds a search tool over the
# Statutes of Singapore. Lawyers query with questions like "What notice
# period is required for dismissing a probationary employee?" and expect
# exact citations.
#
# Which chunker would you pick, and why? Write your answer below, then
# compare with the solution.
#
# YOUR ANSWER:
# ____
#
# BUSINESS IMPACT: A lawyer charges S$500/hr. 20 lawyers × 4 lookups/day
# at 8 min each. Moving to RAG at 45 sec/lookup saves S$250 per lawyer
# per day — S$1.3M/year in reclaimed billable hours.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print(
    """
  [x] Implemented 4 chunking strategies
  [x] Visualised chunk counts and sizes
  [x] Applied chunking reasoning to a Singapore legal-search use case

  Next: 02_dense_retrieval.py embeds these chunks and searches them...
"""
)
