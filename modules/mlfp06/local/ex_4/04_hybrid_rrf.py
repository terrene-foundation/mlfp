# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 4.4: Hybrid Retrieval with Reciprocal Rank Fusion
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Combine dense and sparse retrieval with Reciprocal Rank Fusion (RRF)
#   - Understand why rank-based fusion is more robust than score fusion
#   - Implement RRF with k=60 (the Cormack et al. default)
#   - Apply hybrid retrieval to a Singapore fintech compliance search
#
# PREREQUISITES: Exercises 4.2 (dense), 4.3 (sparse)
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Build a dense store and a BM25 index over the same chunks
#   2. Implement reciprocal_rank_fusion
#   3. Run hybrid retrieval on a real query
#   4. Visualise how dense, sparse, and fused rankings compare
#   5. Apply: fintech MAS compliance search
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
import re
from collections import Counter

import matplotlib.pyplot as plt

from shared.mlfp06.ex_4 import (
    DenseVectorStore,
    EMBED_DIM,
    OUTPUT_DIR,
    embed_many,
    generate_embedding,
    load_rag_corpus,
    plot_strategy_comparison,
    run_async,
    split_corpus,
)


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


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.N = len(documents)
        self.doc_tokens = [re.findall(r"\w+", d.lower()) for d in documents]
        self.doc_lengths = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / max(self.N, 1)
        self.df: dict[str, int] = Counter()
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.df[token] += 1
        self.tf = [Counter(tokens) for tokens in self.doc_tokens]

    def _idf(self, term: str) -> float:
        df_t = self.df.get(term, 0)
        return math.log((self.N - df_t + 0.5) / (df_t + 0.5) + 1)

    def score(self, query: str, doc_idx: int) -> float:
        query_tokens = re.findall(r"\w+", query.lower())
        doc_len = self.doc_lengths[doc_idx]
        tf_doc = self.tf[doc_idx]
        total = 0.0
        for term in query_tokens:
            tf_val = tf_doc.get(term, 0)
            idf = self._idf(term)
            num = tf_val * (self.k1 + 1)
            den = tf_val + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            total += idf * num / den
        return total

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {"text": self.documents[idx], "score": score, "doc_idx": idx}
            for idx, score in scores[:top_k]
        ]


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Build a dense store AND a BM25 index over the same chunks
# ════════════════════════════════════════════════════════════════════════

corpus = load_rag_corpus(sample_size=1000)
doc_texts, eval_questions, eval_answers = split_corpus(corpus, n_eval=20)

all_chunks = []
for text in doc_texts:
    all_chunks.extend(chunk_sentence(text, 500))

chunk_subset = all_chunks[:30]  # small enough to embed in a run
print(f"Indexing {len(chunk_subset)} chunks (both dense + BM25)...")

embeddings = run_async(embed_many(chunk_subset))
dense_store = DenseVectorStore()
for i, (text, emb) in enumerate(zip(chunk_subset, embeddings)):
    dense_store.add(text, emb, {"chunk_idx": i})

bm25 = BM25(chunk_subset)

print(f"Dense store: {len(dense_store.documents)} ({EMBED_DIM}-dim)")
print(f"BM25 index: {bm25.N} documents, avgdl={bm25.avgdl:.1f}")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Reciprocal Rank Fusion
# ════════════════════════════════════════════════════════════════════════
# Fusing two rankers has a problem: their scores live on different scales.
# BM25 scores can be 0–30, cosine similarity is bounded to [-1, 1]. A
# score-based fusion either needs normalisation (fragile) or overweights
# whichever ranker produces larger numbers.
#
# RRF (Cormack, Clarke, Büttcher 2009) sidesteps this by throwing scores
# away entirely and keeping only the RANK:
#
#     RRF_score(d) = sum over rankers r:  1 / (k + rank_r(d))
#
# where k is a constant (60 by convention) and rank_r(d) is the 1-indexed
# position of d in ranker r's result list. A document that shows up at
# rank 1 in both rankers scores 1/61 + 1/61 ≈ 0.033. A document at rank
# 5 in both scores 1/65 + 1/65 ≈ 0.031. The ranking is stable regardless
# of whether the rankers' underlying scores are in the [0, 1] range or
# the [0, 100] range.
#
# Analogy: Two food critics rank restaurants, one using Michelin stars
# (0–3) and one using a 100-point scale. RRF ignores both scoring scales
# and looks only at "whose top-5 restaurants overlap?". A restaurant in
# both top-5 lists wins, regardless of whether it got 2 stars or 96
# points.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Implement reciprocal_rank_fusion
# ════════════════════════════════════════════════════════════════════════


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Combine multiple ranked lists using RRF."""
    # TODO: For each ranked list, for each (rank, item) enumerate from 1,
    #       accumulate 1 / (k + rank) into a dict keyed by hash(item["text"]).
    #       Sort the resulting dict by score descending and return
    #       [{"text", "rrf_score"}, ...].
    ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Run hybrid retrieval on a real query
# ════════════════════════════════════════════════════════════════════════


async def hybrid_search(query: str, top_k: int = 5) -> dict:
    # TODO: Embed the query with generate_embedding, call dense_store.search
    #       and bm25.search for top_k*2 each, then fuse via
    #       reciprocal_rank_fusion. Return {"dense": ..., "sparse": ...,
    #       "fused": ...}, each cut to top_k.
    ____


test_query = eval_questions[0]
print(f"\nQuery: {test_query[:100]}...")
results = run_async(hybrid_search(test_query, top_k=5))

print("\nDense-only top-3:")
for i, r in enumerate(results["dense"][:3]):
    print(f"  {i+1}. score={r['score']:.3f}: {r['text'][:80]}...")

print("\nBM25-only top-3:")
for i, r in enumerate(results["sparse"][:3]):
    print(f"  {i+1}. score={r['score']:.3f}: {r['text'][:80]}...")

print("\nHybrid (RRF) top-3:")
for i, r in enumerate(results["fused"][:3]):
    print(f"  {i+1}. rrf={r['rrf_score']:.4f}: {r['text'][:80]}...")

# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(results["fused"]) > 0, "Hybrid retrieval should return results"
assert all("rrf_score" in r for r in results["fused"]), "RRF scores required"
print("\n--- Checkpoint passed --- hybrid retrieval with RRF fusion\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise how the three rankers rank different chunks
# ════════════════════════════════════════════════════════════════════════

# Normalise each strategy's top-k scores to comparable ranges for plotting
dense_scores = [r["score"] for r in results["dense"]]
sparse_scores = [
    r["score"] / max(r["score"] for r in results["sparse"]) for r in results["sparse"]
]  # bm25 into [0,1]
fused_scores = [r["rrf_score"] * 100 for r in results["fused"]]  # into comparable range

# TODO: Call plot_strategy_comparison with a dict of the 3 strategies'
#       score lists, a title, and filename="ex4_04_hybrid_comparison.png".
____

# R9A: rank comparison — how the same chunks rank across dense, sparse,
# and fused retrievers. Shows where the strategies agree (robust signal)
# and disagree (each catches different content).
fig, ax = plt.subplots(figsize=(10, 5))
top_k = min(5, len(results["fused"]))
chunk_labels = [f"...{r['text'][:40]}..." for r in results["fused"][:top_k]]

# Build rank lookup for dense and sparse
dense_text_rank = {r["text"]: i + 1 for i, r in enumerate(results["dense"])}
sparse_text_rank = {r["text"]: i + 1 for i, r in enumerate(results["sparse"])}
fused_text_rank = {r["text"]: i + 1 for i, r in enumerate(results["fused"])}

x = range(top_k)
width = 0.25
dense_ranks = [
    dense_text_rank.get(results["fused"][i]["text"], top_k + 1) for i in range(top_k)
]
sparse_ranks = [
    sparse_text_rank.get(results["fused"][i]["text"], top_k + 1) for i in range(top_k)
]
fused_ranks = [
    fused_text_rank.get(results["fused"][i]["text"], top_k + 1) for i in range(top_k)
]

ax.barh(
    [i - width for i in x],
    dense_ranks,
    height=width,
    label="Dense",
    color="steelblue",
    edgecolor="white",
)
ax.barh(
    list(x),
    sparse_ranks,
    height=width,
    label="BM25",
    color="darkorange",
    edgecolor="white",
)
ax.barh(
    [i + width for i in x],
    fused_ranks,
    height=width,
    label="Hybrid (RRF)",
    color="seagreen",
    edgecolor="white",
)
ax.set_yticks(list(x))
ax.set_yticklabels(chunk_labels, fontsize=8)
ax.set_xlabel("Rank (lower = better)")
ax.set_title(
    "Rank Comparison — Same Chunks Across 3 Strategies", fontsize=13, fontweight="bold"
)
ax.legend(loc="lower right")
ax.invert_xaxis()
plt.tight_layout()
fname = OUTPUT_DIR / "ex4_04_rank_comparison.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {fname}")

# INTERPRETATION: When dense and sparse agree on rank, the chunk is a
# strong match. When they disagree, RRF mediates — a chunk ranked #1 by
# BM25 but #8 by dense still appears in the fused top-5 because exact
# keyword matches carry weight even when the embedding distance is far.


# ════════════════════════════════════════════════════════════════════════
# APPLY — Singapore fintech compliance search
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore fintech (licensed by MAS under the Payment
# Services Act) runs a compliance search tool over ~5,000 regulatory
# documents: MAS notices, FATF guidance, travel rules, AML/CFT
# advisories. The compliance officer asks questions like "What is the
# reporting threshold for a cross-border wire transfer involving a PEP?".
#
# WHY HYBRID BEATS EITHER:
#   - "PEP" (Politically Exposed Person) is a regulatory acronym —
#     BM25 finds the exact-match documents, dense retrieval maps "PEP"
#     to unrelated content.
#   - "cross-border wire transfer" is semantic — BM25 misses documents
#     that say "remittance" or "SWIFT message"; dense retrieval catches
#     them.
#   - The officer needs BOTH: the exact clause citing the threshold
#     (BM25) AND the adjacent guidance on who qualifies as PEP (dense).
#
# DEPLOYMENT: Index dense embeddings in a vector database, BM25 in
# SQLite FTS5, run both in parallel, fuse with RRF k=60, top-10 to the
# reranker (next exercise).
#
# BUSINESS IMPACT (illustrative planning figures — measure your own
# recall with a hit@k eval set, as in 05_rerank_rag_pipeline.py): a
# fintech with 50 compliance staff spends ~S$8M/year on compliance
# labour, ~30% of it (S$2.4M) on regulatory lookups. If hybrid retrieval
# lets officers find the right clause on the first query instead of the
# third, cutting lookup time by ~40%, the saving is ~S$960K/year. The
# same system drives AML alert triage, where a missed clause is a
# compliance breach — so hybrid retrieval is risk management, not only
# an optimisation.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built a dense store and a BM25 index over the same corpus
  [x] Implemented Reciprocal Rank Fusion with k=60
  [x] Ran all three retrievers on a real query and compared top-k
  [x] Visualised how the rankings diverge
  [x] Mapped hybrid retrieval to a MAS fintech compliance use case

  KEY INSIGHT: RRF is boring and effective. The interesting part is not
  the algorithm (4 lines of Python) but the fact that ignoring scores
  and using only ranks produces more stable fusion than any score-based
  scheme. Engineering elegance beats tuning.

  Next: 05_rerank_rag_pipeline.py adds a cross-encoder reranker, RAGAS
  evaluation, HyDE query expansion, and the full retrieve-rerank-generate
  pipeline...
"""
)
