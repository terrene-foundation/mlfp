# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 6.6: TF-IDF From Scratch — Open the sklearn Box
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Implement TF-IDF from first principles with a regex tokeniser and
#     numpy — no TfidfVectorizer
#   - Reproduce sklearn's smoothed IDF exactly, and explain why it differs
#     from the textbook formula
#   - Verify a from-scratch implementation against a trusted reference
#     (the discipline that makes hand-rolled code shippable)
#   - Rank documents for a query with from-scratch TF-IDF and from-scratch
#     BM25, and compare the two orderings
#   - Decide when writing it yourself is worth it (auditability, zero deps)
#
# PREREQUISITES: 01_tfidf_bm25.py (you used sklearn's TfidfVectorizer and
# saw the BM25 formula).
#
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — the formula sklearn hides, and its two smoothing choices
#   2. Build — tokeniser, vocabulary, counts, IDF, L2-normalised TF-IDF
#   3. Train — run on AG News; verify against sklearn to 1e-9
#   4. Visualise — IDF identity scatter; query ranking TF-IDF vs BM25
#   5. Apply — auditable retrieval for a compliance document store
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer

from kailash_ml import ModelVisualizer

from shared.mlfp04.ex_6 import (
    NEWS_STOP_WORDS,
    OUTPUT_DIR,
    corpus_as_lists,
    load_corpus,
    print_scenario,
)
from shared.mlfp004 import create_visualizer


# ════════════════════════════════════════════════════════════════════════
# THEORY — What TfidfVectorizer Actually Computes
# ════════════════════════════════════════════════════════════════════════
# In 01_tfidf_bm25.py you called TfidfVectorizer and trusted the numbers.
# Here we rebuild every number from raw text. The pipeline has FOUR steps:
#
#   1. TOKENISE: split each document into terms. sklearn's default pattern
#      is (?u)\b\w\w+\b — words of 2+ characters, lowercased.
#   2. COUNT: term counts per document -> a sparse (docs x vocab) matrix C.
#      Vocabulary pruning: min_df (term must appear in >= k docs) and
#      max_df (term must appear in <= x% of docs).
#   3. WEIGHT: TF-IDF = C * IDF, where sklearn's smoothed IDF is
#         IDF(t) = ln((1 + N) / (1 + df(t))) + 1
#      The +1 smoothing keeps every IDF >= 1, so a term in EVERY document
#      is shrunk to its floor but never zeroed.
#   4. NORMALISE: divide each document row by its L2 norm, so long and
#      short documents are comparable (cosine similarity becomes a dot
#      product).
#
# The TEXTBOOK formula you see in lectures is different:
#         TF(t, d) = count(t in d) / length(d);  IDF(t) = ln(N / df(t))
# It divides by zero on a term in every document and never normalises
# rows. sklearn's variant is the textbook idea with production guard-rails.
#
# WHY REBUILD IT? Three reasons: (1) auditability — every weight traces to
# a document frequency an auditor can count by hand; (2) zero dependencies
# — the whole ranker is ~60 lines of numpy; (3) understanding — you cannot
# debug what you cannot recompute. The verification discipline below
# (match a trusted reference to 1e-9) is what turns "I think it's right"
# into "I can prove it's right".


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: tokenise -> count -> weight -> normalise
# ════════════════════════════════════════════════════════════════════════

# sklearn's default token pattern: 2+ word characters, applied to lowercase.
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")

corpus_df = load_corpus(max_docs=1500)
documents, categories = corpus_as_lists(corpus_df)
N_DOCS = len(documents)
STOP_SET = frozenset(NEWS_STOP_WORDS)
print(f"Corpus: {N_DOCS:,} AG News documents (deduplicated subset)")


def tokenize(doc: str) -> list[str]:
    """Lowercase tokens of 2+ word characters, stop words removed."""
    return [t for t in TOKEN_RE.findall(doc.lower()) if t not in STOP_SET]


def build_count_matrix(
    documents: list[str], min_df: int = 3, max_df: float = 0.95
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (vocab, counts, df) — the from-scratch CountVectorizer.

    vocab: sorted terms with df >= min_df and df / N <= max_df (sklearn's
    pruning rules). counts: (N, V) term-count matrix. df: (V,) document
    frequency per term.
    """
    per_doc = [Counter(tokenize(doc)) for doc in documents]
    df_counter: Counter[str] = Counter()
    for counts in per_doc:
        df_counter.update(counts.keys())  # each doc contributes 1 per term
    vocab = sorted(
        t
        for t, df_t in df_counter.items()
        if df_t >= min_df and df_t / len(documents) <= max_df
    )
    index = {t: j for j, t in enumerate(vocab)}
    counts_matrix = np.zeros((len(documents), len(vocab)), dtype=np.float64)
    for row, counts in enumerate(per_doc):
        for term, n in counts.items():
            col = index.get(term)
            if col is not None:
                counts_matrix[row, col] = n
    df = (counts_matrix > 0).sum(axis=0)
    return vocab, counts_matrix, df


vocab, C, df = build_count_matrix(documents)
print(f"Vocabulary: {len(vocab):,} terms (min_df=3, max_df=0.95)")
print(f"Count matrix: {C.shape} — {int((C > 0).sum()):,} non-zero entries")

# Step 3 — sklearn's smoothed IDF and the TF-IDF weighting
idf_scratch = np.log((1 + N_DOCS) / (1 + df)) + 1
X_scratch = C * idf_scratch[np.newaxis, :]

# Step 4 — L2 normalise each document row (empty rows would divide by zero;
# guard them even though this corpus has none after dedup + stop-wording)
row_norms = np.linalg.norm(X_scratch, axis=1, keepdims=True)
row_norms[row_norms == 0.0] = 1.0
X_scratch = X_scratch / row_norms


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN-FREE VERIFICATION: match sklearn to 1e-9
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Verification Against sklearn (identical settings)")
print("=" * 70)

reference = TfidfVectorizer(stop_words=NEWS_STOP_WORDS, min_df=3, max_df=0.95, norm="l2")
X_ref = reference.fit_transform(documents).toarray()
ref_vocab = reference.get_feature_names_out().tolist()

vocab_matches = vocab == ref_vocab
max_diff = float(np.abs(X_scratch - X_ref).max()) if vocab_matches else float("nan")
print(f"Vocabulary identical to sklearn's: {vocab_matches}")
print(f"Max |from_scratch - sklearn| over {X_ref.size:,} weights: {max_diff:.3e}")

# Textbook variant on the SAME counts — see what smoothing changes
tf_textbook = C / C.sum(axis=1, keepdims=True).clip(min=1)
idf_textbook = np.log(N_DOCS / df)
for term in ("market", "olympics"):
    j = vocab.index(term)
    print(
        f"  '{term}': df={int(df[j])}/{N_DOCS} -> "
        f"sklearn IDF {idf_scratch[j]:.4f} vs textbook IDF {idf_textbook[j]:.4f}"
    )


# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert vocab_matches, "Task 3: from-scratch vocabulary must equal sklearn's exactly"
assert max_diff < 1e-9, f"Task 3: TF-IDF weights must match sklearn to 1e-9 (got {max_diff:.2e})"
nonzero_norms = np.linalg.norm(X_scratch, axis=1)
assert np.allclose(nonzero_norms[nonzero_norms > 0], 1.0), "Task 3: rows must be unit L2 norm"
print("\n[ok] Checkpoint 1 passed — from-scratch TF-IDF == sklearn to 1e-9\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE + QUERY: TF-IDF ranking vs BM25 ranking
# ════════════════════════════════════════════════════════════════════════

QUERY = ["space", "shuttle"]
query_set = [t for t in QUERY if t in vocab]
print(f"Query: {QUERY} (in vocab: {query_set})")

# TF-IDF ranking: cosine between the query vector and every document row
# (rows are already unit-norm; normalise the query the same way)
query_vec = np.zeros(len(vocab))
for term in query_set:
    query_vec[vocab.index(term)] = 1.0
query_vec = query_vec * idf_scratch
query_vec = query_vec / np.linalg.norm(query_vec)
tfidf_scores = X_scratch @ query_vec


def bm25_document_scores(query_terms: list[str], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """Score every document with the Robertson/Sparck-Jones BM25 from 6.1.

    IDF = log((N - df + 0.5) / (df + 0.5) + 1); the TF component saturates
    (k1) and normalises for document length (b).
    """
    doc_lengths = C.sum(axis=1)
    avgdl = float(doc_lengths.mean())
    scores = np.zeros(len(documents))
    for term in query_terms:
        j = vocab.index(term)
        idf = np.log((len(documents) - df[j] + 0.5) / (df[j] + 0.5) + 1)
        tf = C[:, j]
        scores += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_lengths / avgdl))
    return scores


bm25_scores = bm25_document_scores(query_set)

top_tfidf = tfidf_scores.argsort()[-5:][::-1]
top_bm25 = bm25_scores.argsort()[-5:][::-1]
overlap = set(top_tfidf.tolist()) & set(top_bm25.tolist())

print(f"\n{'Rank':<5} {'TF-IDF doc (score)':<22} {'BM25 doc (score)':<22}")
for rank in range(5):
    i, j_ = top_tfidf[rank], top_bm25[rank]
    print(f"{rank + 1:<5} doc {i:<6} ({tfidf_scores[i]:.4f})    doc {j_:<6} ({bm25_scores[j_]:.4f})")
print(f"\nTop-5 overlap: {len(overlap)}/5 documents")
print("\nBest TF-IDF match:", documents[top_tfidf[0]][:110], "...")

viz = create_visualizer()

# (A) IDF identity scatter — every term's IDF, from scratch vs sklearn.
# All points must sit on the diagonal; this is the verification made visible.
fig_idf = go.Figure()
fig_idf.add_trace(
    go.Scatter(
        x=reference.idf_,
        y=idf_scratch,
        mode="markers",
        marker=dict(size=3, opacity=0.35, color="#636EFA"),
        name="terms",
    )
)
lim = [0.0, float(max(reference.idf_.max(), idf_scratch.max())) * 1.05]
fig_idf.add_trace(
    go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="#EF553B", dash="dash"), name="identity")
)
fig_idf.update_layout(
    title=f"From-Scratch IDF vs sklearn IDF — {len(vocab):,} terms on the diagonal",
    xaxis_title="sklearn IDF",
    yaxis_title="from-scratch IDF",
)
idf_path = OUTPUT_DIR / "ex6_6_idf_identity.html"
fig_idf.write_html(str(idf_path))
print(f"[viz] IDF identity scatter: {idf_path}")

# (B) Query ranking comparison — top-5 documents per method, side by side
fig_rank = go.Figure()
fig_rank.add_trace(
    go.Bar(
        x=[f"doc {i}" for i in top_tfidf],
        y=tfidf_scores[top_tfidf],
        name="TF-IDF (cosine)",
        marker_color="#636EFA",
    )
)
fig_rank.add_trace(
    go.Bar(
        x=[f"doc {i}" for i in top_bm25],
        y=bm25_scores[top_bm25] / bm25_scores[top_bm25].max() * tfidf_scores[top_tfidf].max(),
        name="BM25 (rescaled for display)",
        marker_color="#00CC96",
    )
)
fig_rank.update_layout(
    title=f"Query '{' '.join(QUERY)}' — top-5 documents per ranker (scores rescaled for display)",
    xaxis_title="document",
    yaxis_title="score (display scale)",
    barmode="group",
)
rank_path = OUTPUT_DIR / "ex6_6_query_ranking.html"
fig_rank.write_html(str(rank_path))
print(f"[viz] Query ranking: {rank_path}")

# (C) Top terms by mean TF-IDF — the corpus fingerprint, from scratch
mean_tfidf = X_scratch.mean(axis=0)
top_idx = mean_tfidf.argsort()[-12:][::-1]
top_terms = {vocab[i]: float(mean_tfidf[i]) for i in top_idx}
term_data = {t: {"mean_tfidf": w} for t, w in top_terms.items()}
fig_terms = viz.metric_comparison(term_data)
fig_terms.update_layout(title="Top Terms by Mean TF-IDF (from scratch)")
terms_path = OUTPUT_DIR / "ex6_6_top_terms.html"
fig_terms.write_html(str(terms_path))
print(f"[viz] Top terms: {terms_path}")


# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert tfidf_scores.shape == (N_DOCS,) and bm25_scores.shape == (N_DOCS,), "Task 4: one score per document"
assert np.isfinite(bm25_scores).all(), "Task 4: BM25 scores must be finite"
best_tfidf_doc = documents[int(tfidf_scores.argmax())].lower()
best_bm25_doc = documents[int(bm25_scores.argmax())].lower()
assert any(t in best_tfidf_doc for t in query_set), "Task 4: top TF-IDF doc must contain a query term"
assert any(t in best_bm25_doc for t in query_set), "Task 4: top BM25 doc must contain a query term"
assert len(top_terms) == 12, "Task 4: top-12 term fingerprint"
print("\n[ok] Checkpoint 2 passed — both rankers score and order documents sensibly\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Auditable Retrieval for a Compliance Document Store
# ════════════════════════════════════════════════════════════════════════

print_scenario("tfidf_from_scratch")
print(
    """
WHY FROM SCRATCH WINS HERE:
  - Every weight is ln((1+N)/(1+df)) + 1 over a document frequency the
    auditor can recount from the raw files. There is no library version,
    no hidden default, no upstream behaviour change to defend.
  - The ranker you verified above (vocab + IDF + L2 + cosine, plus BM25
    as the alternative) is small enough to paste into the audit working
    papers. "Reproduce our top-5 ranking" becomes a 30-minute exercise
    for the auditor, not a software archaeology project.
  - Zero third-party dependencies means the retrieval logic survives
    every future environment rebuild unchanged.

ILLUSTRATIVE ARITHMETIC (assumptions, not measured figures):
  - Assume a compliance team answers ~600 retrieval-based audit queries
    a year, and that an unexplained ranking costs 3 analyst-hours to
    re-derive manually: an auditable ranker avoids ~1,800 hours/year
    (600 x 3h).
  - The honest cost: you own the maintenance. When sklearn fixes a bug,
    you get it free; when YOUR 60 lines have a bug, only your
    verification harness (Task 3) stands between you and a wrong ranking.

LIMITATION: from scratch does not mean better. BM25 still beats vanilla
TF-IDF on most retrieval benchmarks, and neither understands synonyms —
that is what the embedding rankers in 6.4/6.5 buy you, at the price of
explainability.
"""
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Tokenised, counted, pruned, and weighted a real corpus with nothing
      but a regex, Counter, and numpy
  [x] Reproduced sklearn's smoothed IDF formula exactly (1e-9 over every
      weight in the matrix)
  [x] Saw where the textbook formula and the production formula diverge
  [x] Ranked documents with from-scratch TF-IDF cosine and from-scratch
      BM25, and compared the two orderings
  [x] Argued both sides: when hand-rolled is the right call (audit,
      zero-dep) and when it is not (maintenance, benchmarks)

  KEY INSIGHT: "From scratch" is not about avoiding libraries — it is
  about owning the arithmetic. The verification harness (match a trusted
  reference to 1e-9) is what makes the hand-rolled version safe to ship.

  Next: 07_umass_coherence.py — from scratch again, this time for the
  metric that judges topic quality without human labels.
"""
)
