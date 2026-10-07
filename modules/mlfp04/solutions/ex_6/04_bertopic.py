# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 6.4: BERTopic — Neural Topic Modelling
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build a BERTopic pipeline: sentence embeddings -> UMAP -> HDBSCAN
#     -> c-TF-IDF topic extraction
#   - Explain why neural embeddings find topics that TF-IDF misses
#     (polysemy, paraphrase-robustness)
#   - Measure NPMI coherence and compare it with NMF on the same corpus
#   - Read the embedding model name from .env instead of hardcoding it
#   - Apply BERTopic to customer-support ticket clustering
#
# PREREQUISITES: Ex 6.1 (TF-IDF), Ex 6.2 (NMF), Ex 6.3 (LDA),
# Ex 3 (UMAP), Ex 1 (HDBSCAN).
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — BERTopic's four-stage pipeline
#   2. Build — assemble the pipeline
#   3. Train — fit on the corpus
#   4. Visualise — topic coherence vs NMF and topic sizes
#   5. Apply — customer-support ticket clustering
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from bertopic import BERTopic
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from umap import UMAP

from kailash_ml import ModelVisualizer

from shared.mlfp04.ex_6 import (
    NEWS_STOP_WORDS,
    OUTPUT_DIR,
    compute_npmi,
    corpus_as_lists,
    load_corpus,
    print_scenario,
    topic_embedding_model,
)
from shared.mlfp04 import create_visualizer


# ════════════════════════════════════════════════════════════════════════
# THEORY — The BERTopic Pipeline
# ════════════════════════════════════════════════════════════════════════
# BERTopic (Grootendorst 2022) replaces TF-IDF with a four-stage
# neural-assisted pipeline:
#
#   1. EMBED   — a sentence-transformers model encodes each document into
#                a dense vector (e.g. 384 dimensions for a small MiniLM
#                model). Semantically similar documents end up close in
#                this space, regardless of surface wording.
#   2. REDUCE  — UMAP projects the embeddings down to ~5D, preserving
#                local topology for clustering.
#   3. CLUSTER — HDBSCAN finds density-based clusters in the reduced
#                space. Noise points become the "outlier" topic (-1).
#   4. DESCRIBE— For each cluster, c-TF-IDF treats the cluster as a
#                single pseudo-document and computes TF-IDF weights
#                against other clusters — this yields interpretable
#                topic keywords that are genuinely differentiating.
#
# WHY THIS CAN BEAT LDA/NMF:
#   - Embeddings capture meaning, not just surface words. "hut" and
#     "shelter" cluster together even if they never co-occur.
#   - HDBSCAN finds the number of clusters from the data via
#     min_cluster_size — no manual K sweep required.
#   - With a MULTILINGUAL embedding model, documents in different
#     languages land in one space; an English-only model (fine for this
#     English news corpus) does not align languages.
#
# COST: much slower than NMF (a neural forward pass per document); a GPU
# helps on large corpora. The model name comes from TOPIC_EMBED_MODEL in
# .env — never hardcode model names.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 + 3 — BUILD and TRAIN
# ════════════════════════════════════════════════════════════════════════

corpus_df = load_corpus()
documents, categories = corpus_as_lists(corpus_df)
print(f"Corpus: {len(documents):,} documents")

embedding_model_name = topic_embedding_model()
print(f"Embedding model (from TOPIC_EMBED_MODEL): {embedding_model_name}")

print("\n" + "=" * 70)
print("  BERTopic pipeline (embeddings -> UMAP -> HDBSCAN -> c-TF-IDF)")
print("=" * 70)

# c-TF-IDF keywords come from this vectoriser; dropping stop words keeps
# "the", "of", "to" out of the topic descriptions.
topic_vectorizer = CountVectorizer(stop_words=NEWS_STOP_WORDS, min_df=2)
# Fixed random_state makes the UMAP step (and so the topics) repeatable.
umap_model = UMAP(
    n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42
)

topic_model = BERTopic(
    embedding_model=embedding_model_name,
    umap_model=umap_model,
    vectorizer_model=topic_vectorizer,
    min_topic_size=15,
    nr_topics="auto",
    verbose=False,
)
topics, probs = topic_model.fit_transform(documents)
topic_info = topic_model.get_topic_info()
n_topics = int((topic_info["Topic"] >= 0).sum())
doc_topics_vec = np.asarray(topics)
outliers = int((doc_topics_vec == -1).sum())

print(f"Topics discovered: {n_topics}")
print(f"Outlier documents: {outliers:,} / {len(documents):,}")

topic_words: list[list[str]] = []
for topic_id in range(min(n_topics, 10)):
    words = [w for w, _ in topic_model.get_topic(topic_id)[:10]]
    topic_words.append(words)

print("\nTop 10 BERTopic topics:")
for _, row in topic_info[topic_info["Topic"] >= 0].head(10).iterrows():
    name = str(row["Name"])[:60]
    print(f"  Topic {row['Topic']}: {name} (n={row['Count']})")


# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert len(topic_words) > 0, "Task 3: should discover at least one topic"
assert all(
    len(words) >= 5 for words in topic_words
), "Task 3: each topic should have at least 5 words"
assert len(doc_topics_vec) == len(documents), "Task 3: one topic id per document"
print(f"\n[ok] Checkpoint 1 passed — BERTopic produced {len(topic_words)} topics\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: NPMI coherence vs NMF, and topic sizes
# ════════════════════════════════════════════════════════════════════════
# Same corpus, same tokeniser, same NPMI function: an NMF baseline with
# the same number of topics makes the coherence comparison fair.

analyzer = topic_vectorizer.build_analyzer()
coherences = compute_npmi(documents, topic_words, analyzer=analyzer)
mean_npmi = float(np.mean(coherences))

tfidf_vectorizer = TfidfVectorizer(
    max_features=3000, stop_words=NEWS_STOP_WORDS, max_df=0.95, min_df=3
)
X_tfidf = tfidf_vectorizer.fit_transform(documents)
nmf_vocab = tfidf_vectorizer.get_feature_names_out()
nmf = NMF(n_components=len(topic_words), random_state=42, max_iter=400, init="nndsvd")
nmf.fit(X_tfidf)
nmf_topic_words = [
    [nmf_vocab[i] for i in nmf.components_[t].argsort()[-10:][::-1]]
    for t in range(len(topic_words))
]
nmf_coherences = compute_npmi(documents, nmf_topic_words, analyzer=analyzer)
nmf_mean_npmi = float(np.mean(nmf_coherences))

print(f"BERTopic mean NPMI coherence: {mean_npmi:+.4f}")
print(f"NMF      mean NPMI coherence: {nmf_mean_npmi:+.4f} (same K, same corpus)")
for i, c in enumerate(coherences):
    bar = "#" * max(0, int((c + 0.3) * 30))
    print(f"  Topic {i}: {c:+.4f} {bar}")
if mean_npmi > nmf_mean_npmi:
    print("  -> BERTopic's keywords co-occur more consistently than NMF's here.")
else:
    print(
        "  -> NMF's keywords co-occur at least as consistently here. NPMI only\n"
        "     rewards words that appear together in documents; topics built\n"
        "     from meaning (paraphrases) are not guaranteed to score higher."
    )

viz = create_visualizer()

coherence_data = {
    f"Topic_{i}": {"BERTopic NPMI": float(c), "NMF NPMI": float(n)}
    for i, (c, n) in enumerate(zip(coherences, nmf_coherences))
}
fig_coh = viz.metric_comparison(coherence_data)
fig_coh.update_layout(title="Topic Coherence (NPMI): BERTopic vs NMF")
fig_coh.write_html(str(OUTPUT_DIR / "ex6_4_bertopic_coherence.html"))

size_data = {
    f"Topic_{t}": {"docs": int((doc_topics_vec == t).sum())}
    for t in range(len(topic_words))
}
size_data["Outliers (-1)"] = {"docs": outliers}
fig_size = viz.metric_comparison(size_data)
fig_size.update_layout(title="BERTopic Topic Size Distribution")
fig_size.write_html(str(OUTPUT_DIR / "ex6_4_bertopic_sizes.html"))

print(f"\nSaved: {OUTPUT_DIR}/ex6_4_bertopic_coherence.html")
print(f"Saved: {OUTPUT_DIR}/ex6_4_bertopic_sizes.html")


# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(coherences) == len(topic_words), "Task 4: one NPMI per topic"
assert len(nmf_coherences) == len(topic_words), "Task 4: NMF baseline, same K"
assert -1.0 <= mean_npmi <= 1.0, f"Task 4: NPMI must lie in [-1, 1], got {mean_npmi:.4f}"
print("\n[ok] Checkpoint 2 passed — coherence computed and visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Customer-Support Ticket Clustering
# ════════════════════════════════════════════════════════════════════════

print_scenario("bertopic")
print(
    """
WHY BERTOPIC FOR SUPPORT TICKETS:
  - Customers describe the same problem in many ways ("refund not
    received", "still waiting for my money back"). Word-count topics
    split these; sentence embeddings put them together.
  - If tickets arrive in several languages, use a MULTILINGUAL
    sentence-transformers model (set TOPIC_EMBED_MODEL) so one model
    covers them all — an English-only model will not.
  - HDBSCAN's outlier bucket is operationally useful — it holds tickets
    that don't fit any known topic, so the ops team sees emerging
    issues (new scam patterns, new app bugs) faster.
  - c-TF-IDF's topic keywords are directly auditable by the support
    QA team, unlike raw embedding clusters.

ILLUSTRATIVE ARITHMETIC (assumptions, not measured figures):
  - Assume ~35K tickets/week, 85% routed correctly by topic, and
    ~S$3.20 of agent time saved per correctly routed ticket:
    35,000 x 0.85 x S$3.20 = S$95,200/week.
  - Minus an assumed ~S$180/week of GPU compute: ~S$95,020/week net.
  - Measure routing accuracy on a labelled sample before relying on it.
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
  [x] Assembled a four-stage BERTopic pipeline (embed/reduce/cluster/describe)
  [x] Explained why neural embeddings beat TF-IDF on polysemy and
      paraphrase
  [x] Read the embedding model from .env (TOPIC_EMBED_MODEL) instead of
      hardcoding it
  [x] Measured NPMI coherence and compared it with NMF on the same corpus
  [x] Mapped the technique to customer-support ticket clustering

  KEY INSIGHT: BERTopic is slower and benefits from a GPU, but it finds
  topics that survive paraphrase. When your documents describe the same
  thing in different words, no amount of TF-IDF tuning will catch up —
  and NPMI, which only counts word co-occurrence, may not show the gain.

  Next: 05_sentiment_word2vec.py — use word embeddings (not sentence
  embeddings) as features for a lightweight sentiment classifier.
"""
)
