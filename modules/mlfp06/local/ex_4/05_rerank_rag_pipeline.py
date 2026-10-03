# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 4.5: Cross-Encoder Reranking + RAGAS + HyDE + Pipeline
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Implement an LLM-based cross-encoder reranker
#   - Evaluate RAG quality with RAGAS (4 metrics as LLM-as-judge)
#   - Implement HyDE (Hypothetical Document Embeddings) query expansion
#   - Measure BM25, dense, hybrid and HyDE retrieval with hit@k on an eval set
#   - Wire a full retrieve -> rerank -> generate RAG pipeline
#   - Apply end-to-end RAG to a Singapore insurance claims assistant
#
# PREREQUISITES: Exercises 4.2, 4.3, 4.4
# ESTIMATED TIME: ~60 min
#
# TASKS:
#   1. Build the retrieval substrate (dense + BM25 + hybrid), with every
#      chunk tagged by the document it came from
#   2. Implement cross_encoder_rerank
#   3. Implement compute_ragas_metrics (faithfulness, answer relevance,
#      context relevance, context recall)
#   4. Implement hyde_retrieve
#   5. Assemble the full RAG pipeline
#   6. Retrieval leaderboard: hit@k for BM25 / dense / hybrid / HyDE
#   7. Visualise RAGAS scores and the hit@k leaderboard
#   8. Apply: Singapore insurance claims assistant
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from shared.mlfp06.diagnostics import LLMObservatory
from shared.mlfp06.ex_4 import (
    DenseVectorStore,
    EMBED_DIM,
    OUTPUT_DIR,
    delegate_text,
    embed_many,
    generate_embedding,
    load_rag_corpus,
    make_delegate,
    plot_ragas_metrics,
    rag_answer,
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
        self.k1, self.b, self.documents, self.N = k1, b, documents, len(documents)
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
            num = tf_val * (self.k1 + 1)
            den = tf_val + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            total += self._idf(term) * num / den
        return total

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {"text": self.documents[idx], "score": s, "chunk_idx": idx}
            for idx, s in scores[:top_k]
        ]


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Fuse ranked chunk lists by chunk_idx: score = sum 1 / (k + rank)."""
    rrf_scores: dict[int, float] = {}
    texts: dict[int, str] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            idx = item["chunk_idx"]
            texts[idx] = item["text"]
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"text": texts[i], "score": s, "chunk_idx": i} for i, s in fused]


class JudgeParseError(ValueError):
    """The LLM replied, but not with the number the prompt asked for."""


def parse_score(response: str, low: float, high: float) -> float:
    """Pull the first number out of a judge reply; raise if there is none.

    A reply with no number is a FAILED judgement — it is never replaced by
    a neutral default, because a default would silently pass as a result.
    """
    match = re.search(r"\d+(?:\.\d+)?", response)
    if match is None:
        raise JudgeParseError(f"no score in judge reply: {response[:120]!r}")
    return min(max(float(match.group()), low), high)


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Build the retrieval substrate
# ════════════════════════════════════════════════════════════════════════
# Each corpus row is one source document with ONE question about it, so
# question i's relevant document is document i. We index the chunks of
# N_INDEX_DOCS documents (the N_EVAL documents behind the eval questions
# plus distractors) and tag every chunk with its source document id —
# that tag is what lets us MEASURE retrieval in Task 6.

N_EVAL = 10
N_INDEX_DOCS = 40

corpus = load_rag_corpus(sample_size=1000)
doc_texts, eval_questions, eval_answers = split_corpus(corpus, n_eval=N_EVAL)
doc_ids = corpus["section"].to_list()
eval_relevant = doc_ids[:N_EVAL]

chunk_texts: list[str] = []
chunk_doc_ids: list[str] = []
for doc_id, text in zip(doc_ids[:N_INDEX_DOCS], doc_texts[:N_INDEX_DOCS]):
    for chunk in chunk_sentence(text, 500):
        chunk_texts.append(chunk)
        chunk_doc_ids.append(doc_id)
print(f"Indexing {len(chunk_texts)} chunks from {N_INDEX_DOCS} documents...")

embeddings = run_async(embed_many(chunk_texts))
dense_store = DenseVectorStore()
for i, (text, emb) in enumerate(zip(chunk_texts, embeddings)):
    dense_store.add(text, emb, {"chunk_idx": i, "doc_id": chunk_doc_ids[i]})
bm25 = BM25(chunk_texts)

# Embed every eval question once; all retrievers below reuse these vectors.
query_embeddings = run_async(embed_many(eval_questions))


def dense_search(q_emb: list[float], top_k: int = 5) -> list[dict]:
    return [
        {"text": r["text"], "score": r["score"], "chunk_idx": r["metadata"]["chunk_idx"]}
        for r in dense_store.search(q_emb, top_k=top_k)
    ]


def hybrid_search(query: str, q_emb: list[float], top_k: int = 5) -> list[dict]:
    dense_results = dense_search(q_emb, top_k=top_k * 2)
    sparse_results = bm25.search(query, top_k=top_k * 2)
    # TODO: Fuse the two ranked lists with reciprocal_rank_fusion and keep top_k.
    return ____


# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert len(embeddings) == len(chunk_texts) and len(embeddings[0]) == EMBED_DIM
assert set(eval_relevant) <= set(chunk_doc_ids), "every eval doc must be indexed"
print("✓ Checkpoint 1 passed — retrieval substrate indexed\n")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why reranking, RAGAS, and HyDE all exist
# ════════════════════════════════════════════════════════════════════════
# Retrieval gets you CANDIDATE documents. Everything after retrieval is
# about precision: "of the top-20 candidates, which 3 should I actually
# inject into the prompt?".
#
# RERANKING: bi-encoders (the dense embedding model) compare query and
# document INDEPENDENTLY, which is fast but lossy. A cross-encoder feeds
# query AND document together through the same transformer, which is
# accurate but O(N) per query. The production pattern: bi-encoder
# retrieves top-100, cross-encoder reranks to top-5.
#
# RAGAS: four decomposed metrics that explain RAG failure modes.
#   - Low faithfulness + high answer relevance = hallucination (the
#     model made up a plausible answer unsupported by context)
#   - High faithfulness + low answer relevance = wrong question
#     (the answer is grounded but doesn't address the query)
#   - Low context relevance = retrieval failed
#   - Low context recall = the corpus doesn't contain the answer
#
# HyDE: query embeddings and document embeddings live in DIFFERENT
# semantic regions. "What causes inflation?" is short and
# interrogative; a paragraph about inflation causes is long and
# declarative. HyDE generates a hypothetical answer first, embeds the
# hypothetical (which is close to real answers in embedding space), and
# retrieves against that. One extra LLM call per query. Whether it helps
# on YOUR corpus is an empirical question — Task 6 measures it.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Cross-encoder reranker
# ════════════════════════════════════════════════════════════════════════


async def cross_encoder_rerank(
    query: str, candidates: list[dict], top_k: int = 3
) -> list[dict]:
    """Re-rank candidates using LLM-based cross-encoding.

    Production would use a dedicated cross-encoder model
    (ms-marco-MiniLM-L-6-v2) — here we use Delegate as a pedagogical stand-in.
    """
    delegate = make_delegate()
    scored = []
    for candidate in candidates[:10]:
        # TODO: Build a prompt asking for a 0-10 relevance score between
        #       query and candidate["text"], call delegate_text, and turn
        #       the reply into a number with parse_score (it raises on a
        #       reply with no number — do not catch that).
        #       Attach it as "rerank_score" on a copy of the candidate dict.
        ____
    # TODO: Sort by rerank_score descending and return top_k.
    ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — RAGAS metrics (LLM-as-judge)
# ════════════════════════════════════════════════════════════════════════


async def _judge_score(delegate, prompt: str) -> float:
    response = await delegate_text(delegate, prompt)
    return parse_score(response, 0.0, 1.0)


async def compute_ragas_metrics(
    question: str, answer: str, context: str, ground_truth: str
) -> dict:
    """RAGAS-style decomposition via LLM-as-judge."""
    delegate = make_delegate()
    # TODO: For each of (faithfulness, answer_relevance, context_relevance,
    #       context_recall), build a judge prompt that asks for ONE number
    #       between 0.0 and 1.0 and call _judge_score. Return them in a dict
    #       with those four keys.
    ____


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — HyDE
# ════════════════════════════════════════════════════════════════════════


async def hyde_retrieve(query: str, top_k: int = 5) -> list[dict]:
    """HyDE: generate a hypothetical answer, embed it, retrieve similar docs."""
    delegate = make_delegate()
    hyde_prompt = (
        "Write a short paragraph (3-5 sentences) that would be the ideal "
        "answer to this question. It does not need to be factually correct "
        "— it should contain the key concepts and vocabulary that a real "
        "answer would use.\n\n"
        f"Question: {query}\n\nHypothetical answer:"
    )
    # TODO: Generate the hypothetical answer with delegate_text, embed IT
    #       with generate_embedding, and search with dense_search.
    hypo_doc = ____
    hypo_emb = ____
    return ____


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Full RAG pipeline
# ════════════════════════════════════════════════════════════════════════


async def full_rag_pipeline(query: str, q_emb: list[float]) -> dict:
    """Retrieve (hybrid) -> rerank (cross-encoder) -> generate."""
    # TODO: 1. candidates = hybrid_search(query, q_emb, top_k=10)
    #       2. reranked = await cross_encoder_rerank(query, candidates, top_k=3)
    #       3. context = "\n\n---\n\n".join(r["text"] for r in reranked)
    #       4. answer = await rag_answer(query, context)
    #       5. Return {"query", "answer", "context", "n_retrieved", "n_reranked"}
    ____


async def run_pipeline_and_eval() -> tuple[list[dict], dict]:
    pipeline_results = []
    ragas_accum = {
        "faithfulness": [],
        "answer_relevance": [],
        "context_relevance": [],
        "context_recall": [],
    }
    for i in range(3):
        q = eval_questions[i]
        gt = eval_answers[i]
        print(f"\n  Q{i+1}: {q[:80]}...")
        result = await full_rag_pipeline(q, query_embeddings[i])
        print(f"  A: {result['answer'][:180]}...")
        pipeline_results.append(result)

        metrics = await compute_ragas_metrics(
            q, result["answer"], result["context"], gt
        )
        for k, v in metrics.items():
            ragas_accum[k].append(v)
        print(
            f"    RAGAS: faith={metrics['faithfulness']:.2f} "
            f"rel={metrics['answer_relevance']:.2f} "
            f"ctx={metrics['context_relevance']:.2f} "
            f"recall={metrics['context_recall']:.2f}"
        )

    avg_metrics = {k: sum(v) / len(v) for k, v in ragas_accum.items()}
    return pipeline_results, avg_metrics


pipeline_results, avg_ragas = run_async(run_pipeline_and_eval())

# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(pipeline_results) >= 3, "Pipeline should process at least 3 questions"
assert all("answer" in r for r in pipeline_results), "Each result should have an answer"
assert all(0 <= v <= 1 for v in avg_ragas.values()), "RAGAS metrics must be in [0,1]"
print("\n--- Checkpoint passed --- full RAG pipeline with RAGAS evaluation\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Retrieval leaderboard: hit@k for BM25 / dense / hybrid / HyDE
# ════════════════════════════════════════════════════════════════════════
# A retriever is good if the document that actually answers the question
# shows up near the top. With one relevant document per question,
# recall@k IS hit@k: the fraction of questions whose source document is
# in the top-k. Chunks are collapsed to their source document (first
# occurrence wins) so a document is never counted twice.

print("=" * 70)
print("TASK 6: Retrieval leaderboard on the eval set")
print("=" * 70)


def to_doc_ranking(chunk_results: list[dict]) -> list[tuple[str, str, float]]:
    """Collapse ranked chunks into ranked (doc_id, text, score), deduplicated."""
    seen: set[str] = set()
    ranking = []
    for r in chunk_results:
        doc_id = chunk_doc_ids[r["chunk_idx"]]
        if doc_id not in seen:
            seen.add(doc_id)
            ranking.append((doc_id, r["text"], float(r["score"])))
    return ranking


async def collect_rankings(depth: int = 20) -> dict[str, dict[str, list]]:
    rankings: dict[str, dict[str, list]] = {
        "bm25": {},
        "dense": {},
        "hybrid": {},
        "hyde": {},
    }
    for q, q_emb in zip(eval_questions, query_embeddings):
        rankings["bm25"][q] = to_doc_ranking(bm25.search(q, top_k=depth))
        rankings["dense"][q] = to_doc_ranking(dense_search(q_emb, top_k=depth))
        rankings["hybrid"][q] = to_doc_ranking(hybrid_search(q, q_emb, top_k=depth))
        # TODO: HyDE ranking — await hyde_retrieve(q, top_k=depth), then
        #       collapse it with to_doc_ranking like the three lines above.
        rankings["hyde"][q] = ____
    return rankings


rankings = run_async(collect_rankings())
hyde_results = rankings["hyde"][eval_questions[0]]

# The Observatory's Retrieval lens scores every retriever on the same
# eval set. Each retriever is a (query, k) -> [(doc_id, text, score)]
# lookup into the rankings computed above — no extra LLM calls.
obs = LLMObservatory(run_id="ex_4_5_retrieval")
eval_set = [
    {"query": q, "relevant_ids": [rel]} for q, rel in zip(eval_questions, eval_relevant)
]
retrievers = {
    name: (lambda q, k, _r=per_query: _r[q][:k]) for name, per_query in rankings.items()
}
# TODO: For k in (1, 3, 5), score every retriever with the Observatory's
#       retrieval lens (compare_retrievers on retrievers + eval_set).
#       Store the DataFrames in a dict keyed by k.
leaderboards = ____
hit_at_k = (
    pl.concat(
        [
            lb.select("retriever", pl.col("recall_at_k").alias(f"hit@{k}"))
            for k, lb in leaderboards.items()
        ],
        how="align",
    )
    .join(leaderboards[5].select("retriever", "mrr"), on="retriever")
    .sort("hit@5", descending=True)
)
print(hit_at_k)

best = hit_at_k.row(0, named=True)
hyde_vs_dense = (
    hit_at_k.filter(pl.col("retriever") == "hyde")["hit@5"][0]
    - hit_at_k.filter(pl.col("retriever") == "dense")["hit@5"][0]
)
print(
    f"\n  Best retriever on this eval set: {best['retriever']} "
    f"(hit@5={best['hit@5']:.0%}, MRR={best['mrr']:.2f}) over {N_EVAL} questions"
)
print(f"  HyDE vs plain dense, hit@5: {hyde_vs_dense:+.0%}")
print(
    f"  With only {N_EVAL} questions, one question moves hit@k by "
    f"{1 / N_EVAL:.0%} — treat small gaps as noise."
)

# ── Checkpoint 6 ─────────────────────────────────────────────────────────
assert len(hyde_results) > 0, "HyDE should return results"
assert hit_at_k.height == 4, "Leaderboard should rank all 4 retrievers"
assert all(
    0.0 <= v <= 1.0 for c in ("hit@1", "hit@3", "hit@5") for v in hit_at_k[c].to_list()
)
assert (hit_at_k["hit@1"] <= hit_at_k["hit@5"]).all(), "hit@k must grow with k"
print("✓ Checkpoint 6 passed — retrieval leaderboard measured\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 7 — Visualise RAGAS scores and the retrieval leaderboard
# ════════════════════════════════════════════════════════════════════════

print("\nAverage RAGAS metrics across 3 questions:")
for k, v in avg_ragas.items():
    print(f"  {k}: {v:.2f}")

# TODO: Call plot_ragas_metrics(avg_ragas, title=..., filename="ex4_05_ragas_metrics.png")
____

# R9A: RAGAS radar chart — shows the "shape" of RAG quality at a glance.
# A perfect system is a full diamond; a hallucinating system has high
# answer_relevance but low faithfulness (top-left dip).
labels = list(avg_ragas.keys())
values = [avg_ragas[l] for l in labels]
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
values_closed = values + [values[0]]
angles_closed = angles + [angles[0]]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(angles_closed, values_closed, "o-", linewidth=2, color="steelblue")
ax.fill(angles_closed, values_closed, alpha=0.25, color="steelblue")
ax.set_xticks(angles)
ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=9)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
ax.set_title(
    "RAGAS Radar — Pipeline Quality Shape",
    fontsize=13,
    fontweight="bold",
    pad=20,
)
# Draw the 0.70 target ring
target_ring = [0.70] * (len(angles) + 1)
ax.plot(
    angles_closed,
    target_ring,
    "--",
    color="grey",
    alpha=0.5,
    linewidth=1,
    label="target=0.70",
)
ax.legend(loc="lower right", bbox_to_anchor=(1.15, -0.05), fontsize=8)
plt.tight_layout()
fname = OUTPUT_DIR / "ex4_05_ragas_radar.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {fname}")

# R9A: measured hit@k per retriever — grouped bars at k = 1, 3, 5.
fig, ax = plt.subplots(figsize=(8, 4.5))
names = hit_at_k["retriever"].to_list()
x = np.arange(len(names))
for offset, k in zip((-0.27, 0.0, 0.27), (1, 3, 5)):
    ax.bar(x + offset, hit_at_k[f"hit@{k}"].to_list(), width=0.25, label=f"hit@{k}")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Fraction of questions whose source doc is in the top-k")
ax.set_title(
    f"Retrieval leaderboard — {N_EVAL} eval questions, {N_INDEX_DOCS} indexed docs",
    fontsize=12,
    fontweight="bold",
)
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
fname = OUTPUT_DIR / "ex4_05_hit_at_k.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {fname}")

# INTERPRETATION: The radar chart reveals your pipeline's "personality":
# - Balanced diamond = solid RAG system
# - Low faithfulness + high relevance = hallucination (the model answers
#   the right question but invents facts)
# - Low context_recall = your corpus doesn't contain the answer
# The hit@k bars show retrieval quality BEFORE the reranker and the LLM
# touch anything: if the source document is not in the top-k, no amount
# of reranking or prompting can recover it.


# ════════════════════════════════════════════════════════════════════════
# APPLY — Singapore insurance claims assistant
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore insurer runs a claims-processing
# assistant for its agents. When a customer files an accident claim, the
# agent needs to answer questions like "Is physiotherapy covered under
# the customer's plan after a motor vehicle accident if the policy was
# issued in 2021 and the accident happened in Malaysia?".
#
# Answering this requires pulling from:
#   - The customer's policy document (coverage, exclusions, effective dates)
#   - The claims manual (territorial scope, third-party rules)
#   - The medical schedule (what counts as physiotherapy, max visits)
#
# WHY THE FULL PIPELINE MATTERS:
#   - Hybrid retrieval finds both the exact plan name via BM25 AND the
#     medical-schedule clause ("physiotherapy covered up to 30 sessions
#     post-accident") via dense embeddings.
#   - A hit@k leaderboard like Task 6, built on the insurer's own
#     question set, decides which retriever goes to production.
#   - Cross-encoder reranking pushes the MOST relevant 3 clauses to the
#     top — critical because the wrong clause means a wrong payout
#     decision.
#   - RAGAS metrics run in shadow mode: every claim decision is scored,
#     and any answer with faithfulness < 0.8 is flagged for human review
#     BEFORE it goes to the customer.
#   - HyDE helps when the agent types a conversational query instead of
#     insurance jargon ("can we pay for his back treatment after the
#     car crash?") — keep it only if it wins on the leaderboard.

print("=" * 70)
print("APPLICATION — Insurance claims assistant (illustrative figures)")
print("=" * 70)

CLAIMS_PER_YEAR = 200_000
LOOKUP_MINUTES_BEFORE = 15
LOOKUP_MINUTES_AFTER = 3
AGENT_COST_SGD_PER_HOUR = 40

# TODO: minutes saved per claim x claims per year, converted to hours,
#       x the agent's hourly cost.
annual_saving_sgd = ____
print(f"  Claims per year:            {CLAIMS_PER_YEAR:,}")
print(f"  Lookup time per claim:      {LOOKUP_MINUTES_BEFORE} -> {LOOKUP_MINUTES_AFTER} min")
print(f"  Agent time saved per year:  S${annual_saving_sgd:,.0f}")
print(
    "  The RAGAS faithfulness gate is the compliance control: inconsistent\n"
    "  claims handling is a regulatory risk, so reranking + evaluation is\n"
    "  not polish — it is what makes the time saving safe to bank."
)

# ── Checkpoint Application ──────────────────────────────────────────────
assert annual_saving_sgd > 0
print("\n✓ Application checkpoint passed — claims assistant business case\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built an LLM-based cross-encoder reranker
  [x] Implemented RAGAS (faithfulness, answer relevance, context
      relevance, context recall) as LLM-as-judge — a judge reply with
      no score is an error, never a silent default
  [x] Implemented HyDE query expansion (generate hypothetical, embed it)
  [x] Measured BM25 / dense / hybrid / HyDE with hit@k and MRR on an
      eval set where the right document is known
  [x] Wired a full retrieve -> rerank -> generate pipeline
  [x] Visualised RAGAS metrics against a 0.70 target
  [x] Mapped the full pipeline to a Singapore insurance claims use case

  PRODUCTION RECIPE:
    1. Chunk documents (sentence chunking is a good default)
    2. Dense index in a vector DB + BM25 index in SQLite FTS5 / Elasticsearch
    3. Hybrid retrieve top-20 with RRF(k=60)
    4. Cross-encoder rerank to top-3
    5. Inject top-3 into the answer prompt
    6. Run RAGAS in shadow mode on every production query
    7. Alert on faithfulness < 0.8 (hallucination signal)
    8. Add HyDE only where it measurably lifts hit@k

  RAG VS FINE-TUNING:
    Use RAG when documents change frequently, need citations, audit trail
    Use fine-tuning when: style adaptation, domain vocabulary, latency
    Combine: fine-tune on domain + RAG for up-to-date facts

  NEXT: Exercise 5 moves from RAG (the LLM READS documents) to AGENTS
  (the LLM takes ACTIONS — calls tools, observes results, iterates).
"""
)
