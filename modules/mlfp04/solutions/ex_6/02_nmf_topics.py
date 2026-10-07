# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 6.2: NMF Topic Modelling — Non-Negative Factorisation
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Factorise a TF-IDF matrix X ≈ W @ H with NMF
#   - Read W as document-topic weights and H as topic-word weights
#   - Explain why non-negativity makes topics interpretable
#   - Measure NPMI topic coherence on a real news corpus
#   - Check discovered topics against human section labels
#   - Apply NMF to a newsroom's content tagging at scale
#
# PREREQUISITES: Exercise 6.1 (TF-IDF), linear algebra (matrix factorisation).
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — non-negativity and additive parts
#   2. Build — NMF on a TF-IDF matrix
#   3. Train — fit NMF and inspect reconstruction quality
#   4. Visualise — NPMI coherence per topic, topic-vs-section heatmap
#   5. Apply — newsroom auto-tagging scenario
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from kailash_ml import ModelVisualizer

from shared.mlfp04.ex_6 import (
    NEWS_STOP_WORDS,
    OUTPUT_DIR,
    compute_npmi,
    corpus_as_lists,
    load_corpus,
    print_scenario,
)
from shared.mlfp04 import create_visualizer


# ════════════════════════════════════════════════════════════════════════
# THEORY — Non-Negative Matrix Factorisation
# ════════════════════════════════════════════════════════════════════════
# NMF decomposes the TF-IDF matrix X (n_docs x n_vocab) as:
#
#     X ≈ W @ H     with W >= 0 and H >= 0
#
#     W[doc, topic]  = document's weight for each topic
#     H[topic, word] = topic's weight for each word
#
# Compare with PCA: PCA allows NEGATIVE loadings, so a topic might be
# "+housing -technology", which is hard to read. NMF forbids negatives,
# so every topic only ADDS word mass — you can write each document as a
# literal sum of topic contributions. This is called "parts-based
# representation" and it's the reason NMF topics read like human topics.
#
# Algorithm: alternating updates (coordinate descent in sklearn) — fix
# H and solve for W, fix W and solve for H. Each half-step is a convex
# problem, but the JOINT problem in (W, H) is NON-convex: different
# starts can land in different local optima. init="nndsvd" gives a
# deterministic SVD-based start, so reruns produce the same topics. No
# probabilistic interpretation and no Dirichlet priors — just
# least-squares reconstruction under non-negativity constraints.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: TF-IDF -> NMF pipeline
# ════════════════════════════════════════════════════════════════════════

corpus_df = load_corpus()
documents, categories = corpus_as_lists(corpus_df)
print(f"Corpus: {len(documents):,} documents across {len(set(categories))} categories")

vectorizer = TfidfVectorizer(
    max_features=3000,
    stop_words=NEWS_STOP_WORDS,
    max_df=0.95,
    min_df=3,
)
X = vectorizer.fit_transform(documents)
vocab = vectorizer.get_feature_names_out()
print(f"TF-IDF matrix: {X.shape} (docs x vocab)")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: fit NMF with a few topic counts
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  NMF Topic Discovery")
print("=" * 70)

n_topics = 10
nmf = NMF(n_components=n_topics, random_state=42, max_iter=400, init="nndsvd")
W = nmf.fit_transform(X)
H = nmf.components_

recon_error = np.linalg.norm(X.toarray() - W @ H) / np.linalg.norm(X.toarray())
print(f"Relative reconstruction error: {recon_error:.4f}")
print(f"Lower is better — 0 = perfect reconstruction, 1 = useless")

print(f"\nTop words per topic (K={n_topics}):")
topic_words: list[list[str]] = []
for t in range(n_topics):
    top_idx = H[t].argsort()[-8:][::-1]
    words = [vocab[i] for i in top_idx]
    topic_words.append(words)
    print(f"  Topic {t}: {', '.join(words)}")

# Hard-assign each document to its top topic
doc_topic = W.argmax(axis=1)
print(f"\nDocuments per topic:")
for t in range(n_topics):
    count = int((doc_topic == t).sum())
    print(f"  Topic {t}: {count} docs")


# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert W.shape == (len(documents), n_topics), "Task 3: W should be (n_docs, n_topics)"
assert H.shape == (n_topics, len(vocab)), "Task 3: H should be (n_topics, n_vocab)"
assert W.min() >= -1e-10, "Task 3: NMF W must be non-negative"
assert H.min() >= -1e-10, "Task 3: NMF H must be non-negative"
assert recon_error < 1.0, "Task 3: reconstruction should be better than zero matrix"
print("\n[ok] Checkpoint 1 passed — NMF factorisation valid and non-negative\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: topic coherence via NPMI
# ════════════════════════════════════════════════════════════════════════

coherences = compute_npmi(documents, topic_words, analyzer=vectorizer.build_analyzer())
mean_npmi = float(np.mean(coherences))
print(f"NPMI coherence — mean: {mean_npmi:+.4f}")
print("(Range -1..1. 0 = top words co-occur exactly as often as chance;")
print(" > 0 = above chance; -1 = they never appear in the same document.)")
for i, c in enumerate(coherences):
    bar = "#" * max(0, int((c + 0.3) * 30))
    print(f"  Topic {i}: {c:+.4f} {bar}")

viz = create_visualizer()

# Coherence bar chart
coherence_data = {f"Topic_{i}": {"NPMI": float(c)} for i, c in enumerate(coherences)}
fig_coh = viz.metric_comparison(coherence_data)
fig_coh.update_layout(title="NMF Topic Coherence (NPMI)")
fig_coh.write_html(str(OUTPUT_DIR / "ex6_2_nmf_coherence.html"))

# Topic size distribution
size_data = {
    f"Topic_{t}": {"docs": int((doc_topic == t).sum())} for t in range(n_topics)
}
fig_size = viz.metric_comparison(size_data)
fig_size.update_layout(title="NMF Topic Size Distribution")
fig_size.write_html(str(OUTPUT_DIR / "ex6_2_nmf_topic_sizes.html"))

# Topic-vs-section heatmap: the human section labels were NEVER used to
# fit NMF. If a topic's documents fall mostly in one section, NMF has
# rediscovered that section from word co-occurrence alone.
sections = sorted(set(categories))
category_arr = np.asarray(categories)
crosstab = np.array(
    [
        [int(((doc_topic == t) & (category_arr == sec)).sum()) for sec in sections]
        for t in range(n_topics)
    ]
)
topic_labels = [f"T{t}: {', '.join(topic_words[t][:3])}" for t in range(n_topics)]
fig_heat = go.Figure(
    go.Heatmap(
        z=crosstab,
        x=sections,
        y=topic_labels,
        colorscale="Blues",
        text=crosstab,
        texttemplate="%{text}",
    )
)
fig_heat.update_layout(
    title="NMF topics vs human news sections (documents per cell)",
    xaxis_title="Human section label (not used in fitting)",
    yaxis_title="NMF topic (top 3 words)",
    height=550,
)
fig_heat.write_html(str(OUTPUT_DIR / "ex6_2_nmf_topic_vs_section.html"))

purity = crosstab.max(axis=1) / np.maximum(crosstab.sum(axis=1), 1)
print("\nTopic purity (share of a topic's docs in its most common section):")
for t in range(n_topics):
    print(f"  Topic {t}: {purity[t]:.0%} {sections[int(crosstab[t].argmax())]}")

print(f"\nSaved: {OUTPUT_DIR}/ex6_2_nmf_coherence.html")
print(f"Saved: {OUTPUT_DIR}/ex6_2_nmf_topic_sizes.html")
print(f"Saved: {OUTPUT_DIR}/ex6_2_nmf_topic_vs_section.html")


# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(coherences) == n_topics, "Task 4: one NPMI per topic"
assert mean_npmi > -0.5, f"Task 4: mean NPMI should be > -0.5, got {mean_npmi:.4f}"
assert crosstab.sum() == len(documents), "Task 4: every document lands in one cell"
print("\n[ok] Checkpoint 2 passed — NPMI computed and visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Newsroom Content Tagging
# ════════════════════════════════════════════════════════════════════════

print_scenario("nmf_topics")
print(
    """
WHY NMF IS THE RIGHT TOOL FOR A NEWSROOM:
  - The editorial desk needs INTERPRETABLE topics, not black-box
    embeddings. A journalist must be able to read a topic such as
    "oil, prices, crude, barrel" and immediately name the beat.
  - NMF with a fixed init is deterministic and fast (this run fitted
    ~5,000 articles in seconds on a laptop), so a nightly tagging job
    fits easily in the pipeline window.
  - Non-negativity makes the topic-keyword report AUDITABLE. The
    editorial standards team can review the top-20 words per topic
    and flag any that mix semantically unrelated terms.

ILLUSTRATIVE ARITHMETIC (assumptions, not measured figures):
  - Assume ~2,400 articles/day and that manual tagging takes an editor
    ~1 minute per article: auto-suggested tags that are accepted 70% of
    the time save ~28 editor-hours a day (2,400 x 70% x 1 min).
  - Check acceptance on a sample before relying on such a number — the
    topic-vs-section heatmap above shows how clean the topics are.
  - Compare with BERTopic (04_bertopic.py), which embeds every document
    with a neural network: slower, but groups paraphrases that share
    few words.
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
  [x] Factorised a TF-IDF matrix with NMF (X ≈ W @ H)
  [x] Read W as document-topic weights, H as topic-word weights
  [x] Explained why non-negativity makes topics interpretable
  [x] Measured NPMI topic coherence without human annotation
  [x] Checked topics against human section labels held out of fitting
  [x] Mapped the technique to newsroom auto-tagging

  KEY INSIGHT: NMF is a strong default when you need interpretable
  topics FAST. It is not the most semantic method, but its topics are
  readable, a fixed init makes runs repeatable, and fits finish in
  seconds.

  Next: 03_lda_topics.py — probabilistic topic modelling with
  mixed-membership via Latent Dirichlet Allocation.
"""
)
