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
#   have a context window and cannot see documents longer than that.
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

import matplotlib.pyplot as plt

from shared.mlfp06.ex_4 import (
    OUTPUT_DIR,
    load_rag_corpus,
    plot_chunking_comparison,
    split_corpus,
)

# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load corpus
# ════════════════════════════════════════════════════════════════════════

corpus = load_rag_corpus(sample_size=1000)
doc_texts, eval_questions, eval_answers = split_corpus(corpus, n_eval=20)

print(f"Loaded {corpus.height:,} documents ({len(eval_questions)} eval questions)")
doc_lengths = [len(t) for t in doc_texts]
print(
    f"Char lengths: min={min(doc_lengths)}, max={max(doc_lengths)}, "
    f"mean={sum(doc_lengths) / len(doc_lengths):.0f}"
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why chunking exists
# ════════════════════════════════════════════════════════════════════════
# An LLM can only see what fits in its context window. A 500-page employee
# handbook cannot be dropped into a prompt. Retrieval-augmented generation
# solves this by splitting documents into chunks, embedding each chunk,
# and injecting only the top-k most relevant chunks at query time.
#
# The chunking step decides how the document is split — and bad chunking
# is the single biggest source of RAG failures.
#
# Analogy: Imagine handing a researcher a 500-page book and asking
# "Where does it discuss X?". If you tear pages randomly every 200 words
# (fixed chunking), you'll cut sentences mid-thought. If you split on
# section headers (semantic chunking), each chunk is a coherent unit the
# researcher can quote directly.
#
# Four strategies, four trade-offs:
#   Fixed         — simple, fast, breaks sentences. Baseline.
#   Sentence      — respects sentence boundaries. Good default.
#   Paragraph     — coherent topics, but very size-variable.
#   Semantic      — topic-shift detection, needs heuristics or an LLM.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Implement 4 chunking strategies
# ════════════════════════════════════════════════════════════════════════


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
    """Paragraph-boundary chunking with merging of very short paragraphs."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = [p.strip() for p in paragraphs if p.strip()]
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
    """Heuristic semantic chunking — split on heading / transition markers."""
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


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Apply strategies to a sample document
# ════════════════════════════════════════════════════════════════════════

sample_text = doc_texts[0]
strategies = {
    "Fixed(500,100)": chunk_fixed(sample_text, 500, 100),
    "Sentence(500)": chunk_sentence(sample_text, 500),
    "Paragraph": chunk_paragraph(sample_text),
    "Semantic(500)": chunk_semantic(sample_text, 500),
}

print(f"\nChunking comparison on sample document ({len(sample_text)} chars):")
for name, chunks in strategies.items():
    avg_len = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
    print(f"  {name}: {len(chunks)} chunks, avg {avg_len:.0f} chars")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Chunk the full corpus with the sentence strategy and save
# ════════════════════════════════════════════════════════════════════════

all_chunks = []
for i, text in enumerate(doc_texts):
    doc_chunks = chunk_sentence(text, max_chunk_chars=500)
    for j, chunk in enumerate(doc_chunks):
        all_chunks.append(
            {"doc_idx": i, "chunk_idx": j, "section": f"doc_{i:04d}", "text": chunk}
        )

chunks_df = pl.DataFrame(all_chunks)
print(f"\nCorpus chunked: {len(doc_texts)} docs -> {chunks_df.height} chunks")
print(f"Chunks per doc: avg {chunks_df.height / len(doc_texts):.1f}")

# ── Checkpoint ──────────────────────────────────────────────────────────
assert chunks_df.height > len(doc_texts), "Sentence chunking should expand docs"
assert all(
    len(v) > 0 for v in strategies.values()
), "All strategies should produce chunks"
print("\n--- Checkpoint passed --- 4 chunking strategies implemented\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise chunk counts and sizes
# ════════════════════════════════════════════════════════════════════════

plot_chunking_comparison(
    strategies,
    title="Chunking Strategy Comparison — Sample Document",
    filename="ex4_01_chunking_comparison.png",
)

# R9A: chunk size distribution histogram per strategy — shows the SPREAD
# of chunk lengths, not just the average. Wide spread = unpredictable
# embedding quality; tight spread = consistent retrieval granularity.
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Chunk Size Distribution per Strategy", fontsize=13, fontweight="bold")
colors = ["steelblue", "darkorange", "seagreen", "mediumpurple"]
for ax, (name, chunks), color in zip(axes.flat, strategies.items(), colors):
    lengths = [len(c) for c in chunks]
    ax.hist(lengths, bins=15, color=color, edgecolor="white", alpha=0.85)
    ax.set_title(name)
    ax.set_xlabel("Chunk length (chars)")
    ax.set_ylabel("Count")
    ax.axvline(
        sum(lengths) / len(lengths) if lengths else 0,
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label=f"mean={sum(lengths) / len(lengths):.0f}" if lengths else "",
    )
    ax.legend(fontsize=8)
plt.tight_layout()
fname = OUTPUT_DIR / "ex4_01_chunk_size_distributions.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {fname}")

# INTERPRETATION: Fixed chunking produces a tight cluster near the target
# size. Paragraph chunking has the widest spread — some paragraphs are
# 50 chars, others 2000. This matters for retrieval: very short chunks
# embed poorly (too little context), very long chunks dilute the signal.


# ════════════════════════════════════════════════════════════════════════
# APPLY — Singapore legal-search SaaS
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore legal-tech startup builds a search tool over the
# Statutes of Singapore (the Companies Act, Employment Act, Personal
# Data Protection Act, and 200 other laws). Lawyers query the tool with
# questions like "What notice period is required for dismissing a
# probationary employee?" and expect exact citations.
#
# CHUNKING DECISION:
#   - Fixed(500) — breaks mid-section; a citation could span two chunks.
#     A lawyer reading "...shall give notice of at least (a)" in one chunk
#     and "...one week during probation" in the next cannot cite either.
#   - Sentence(500) — respects sentence boundaries; each chunk is
#     self-contained legal prose. Good default.
#   - Paragraph — each statute sub-section is a paragraph. Preserves the
#     legal unit of citation (a "clause") exactly.  Highest precision.
#   - Semantic — the statute's own section headings are the best
#     topic-shift markers. Works if the PDF extraction preserves them.
#
# RECOMMENDATION: Paragraph chunking for statutes, because each paragraph
# is a legally-citable clause. For case law (longer, narrative judgments),
# use sentence chunking with 800-char chunks.
#
# BUSINESS IMPACT: A lawyer charges S$500/hr. The firm has 20 lawyers
# doing 4 statute lookups per day each. Moving from Ctrl-F on PDFs (avg
# 8 min/lookup) to RAG (avg 45 sec/lookup) saves each lawyer ~30 minutes
# per day, or S$250 per lawyer per day. Across the firm: S$5,000/day =
# S$1.3M/year in billable hours reclaimed.  The chunking decision is the
# difference between a lawyer citing the right clause and citing the
# wrong half of a clause — and in law, the wrong citation is malpractice.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Implemented 4 chunking strategies: fixed, sentence, paragraph, semantic
  [x] Understood the size-vs-precision trade-off:
        too small -> loses context
        too large -> retrieval less precise
  [x] Visualised chunk counts and sizes across strategies
  [x] Applied chunking reasoning to a Singapore legal-search use case
  [x] Produced a full corpus of sentence-chunks for downstream retrieval

  KEY INSIGHT: Chunking is not preprocessing — it is the first retrieval
  decision. A poor chunker with a great embedding model will still miss
  the answer. A great chunker with a middling embedding model still wins.

  Next: 02_dense_retrieval.py embeds these chunks and searches them...
"""
)
