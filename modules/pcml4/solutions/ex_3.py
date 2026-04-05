# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT4 — Exercise 3: Topic Modeling with BERTopic
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: BERTopic pipeline on Singapore news corpus — discover topics,
#   evaluate coherence, analyse temporal evolution.
#
# TASKS:
#   1. Load and preprocess Singapore news corpus
#   2. Build BERTopic model (UMAP + HDBSCAN + c-TF-IDF)
#   3. Evaluate topic coherence (NPMI)
#   4. Analyse temporal topic evolution
#   5. Visualise topic distributions
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
from collections import Counter

from kailash_ml import ModelVisualizer

from shared import ASCENTDataLoader

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
except ImportError:
    BERTopic = None
    SentenceTransformer = None


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
news = loader.load("ascent04", "sg_news_corpus.parquet")

print(f"=== Singapore News Corpus ===")
print(f"Shape: {news.shape}")
print(f"Columns: {news.columns}")
print(f"Date range: {news['published_date'].min()} to {news['published_date'].max()}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Text preprocessing with Polars
# ══════════════════════════════════════════════════════════════════════

# Basic cleaning with Polars string expressions
news_clean = (
    news.with_columns(
        # Combine title and body for richer topic signals
        (pl.col("title") + ". " + pl.col("body")).alias("text"),
        # Parse date
        pl.col("published_date").str.to_date("%Y-%m-%d").alias("date"),
    )
    .filter(
        # Remove very short articles (likely stubs or metadata)
        pl.col("body").str.len_chars()
        > 100
    )
    .with_columns(
        # Extract year-month for temporal analysis
        pl.col("date")
        .dt.strftime("%Y-%m")
        .alias("year_month"),
    )
)

documents = news_clean["text"].to_list()
dates = news_clean["date"].to_list()
print(f"\nCleaned corpus: {len(documents):,} articles")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: BERTopic model
# ══════════════════════════════════════════════════════════════════════

if BERTopic is not None:
    # BERTopic pipeline:
    # 1. Sentence embeddings (SBERT)
    # 2. Dimensionality reduction (UMAP)
    # 3. Clustering (HDBSCAN)
    # 4. Topic representation (c-TF-IDF)

    topic_model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",
        umap_model=None,  # Use defaults
        hdbscan_model=None,
        min_topic_size=20,
        nr_topics="auto",
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(documents)

    # Topic summary
    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info) - 1  # Exclude outlier topic -1
    print(f"\n=== BERTopic Results ===")
    print(f"Topics found: {n_topics}")
    print(f"Outlier documents: {(np.array(topics) == -1).sum():,}")

    # Display top topics
    print(f"\nTop 10 Topics:")
    for _, row in topic_info.head(11).iterrows():
        if row["Topic"] == -1:
            continue
        print(f"  Topic {row['Topic']}: {row['Name'][:60]} (n={row['Count']})")

else:
    # Fallback: TF-IDF + NMF for environments without BERTopic
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF

    print("\nBERTopic not installed, using TF-IDF + NMF fallback")

    vectorizer = TfidfVectorizer(
        max_features=5000, stop_words="english", max_df=0.95, min_df=5
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    n_topics = 15
    nmf = NMF(n_components=n_topics, random_state=42)
    W = nmf.fit_transform(tfidf_matrix)  # Document-topic matrix
    H = nmf.components_  # Topic-word matrix

    topics = W.argmax(axis=1).tolist()
    probs = W / (W.sum(axis=1, keepdims=True) + 1e-10)

    print(f"\nNMF Topics ({n_topics}):")
    for topic_idx in range(n_topics):
        top_words = [feature_names[i] for i in H[topic_idx].argsort()[-8:][::-1]]
        count = sum(1 for t in topics if t == topic_idx)
        print(f"  Topic {topic_idx}: {', '.join(top_words)} (n={count})")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Topic coherence evaluation (NPMI)
# ══════════════════════════════════════════════════════════════════════
# NPMI (Normalised Pointwise Mutual Information):
# NPMI(w_i, w_j) = (log P(w_i,w_j)/(P(w_i)P(w_j))) / (-log P(w_i,w_j))
# Range: [-1, 1]. Higher = more coherent topic.


def compute_npmi(
    documents: list[str], topic_words: list[list[str]], window_size: int = 10
) -> list[float]:
    """Compute NPMI coherence for each topic."""
    # Build word co-occurrence counts
    word_doc_count = Counter()
    pair_doc_count = Counter()
    n_docs = len(documents)

    for doc in documents:
        words = set(doc.lower().split())
        for w in words:
            word_doc_count[w] += 1
        word_list = list(words)
        for i in range(len(word_list)):
            for j in range(i + 1, len(word_list)):
                pair = tuple(sorted([word_list[i], word_list[j]]))
                pair_doc_count[pair] += 1

    coherences = []
    for topic in topic_words:
        npmi_sum = 0
        n_pairs = 0
        for i in range(len(topic)):
            for j in range(i + 1, len(topic)):
                w_i, w_j = topic[i].lower(), topic[j].lower()
                pair = tuple(sorted([w_i, w_j]))
                p_i = word_doc_count.get(w_i, 0) / n_docs
                p_j = word_doc_count.get(w_j, 0) / n_docs
                p_ij = pair_doc_count.get(pair, 0) / n_docs

                if p_ij > 0 and p_i > 0 and p_j > 0:
                    pmi = np.log(p_ij / (p_i * p_j))
                    npmi = pmi / (-np.log(p_ij))
                    npmi_sum += npmi
                    n_pairs += 1

        coherences.append(npmi_sum / max(n_pairs, 1))
    return coherences


# Get topic words
if BERTopic is not None:
    topic_words = []
    for topic_id in range(n_topics):
        words = [w for w, _ in topic_model.get_topic(topic_id)[:10]]
        topic_words.append(words)
else:
    topic_words = []
    for topic_idx in range(n_topics):
        words = [feature_names[i] for i in H[topic_idx].argsort()[-10:][::-1]]
        topic_words.append(words)

coherences = compute_npmi(documents[:5000], topic_words)  # Sample for speed

print(f"\n=== Topic Coherence (NPMI) ===")
print(f"Mean NPMI: {np.mean(coherences):.4f}")
print(f"Best topic: {np.argmax(coherences)} (NPMI={max(coherences):.4f})")
print(f"Worst topic: {np.argmin(coherences)} (NPMI={min(coherences):.4f})")
for i, c in enumerate(coherences[:10]):
    bar = "█" * max(0, int((c + 0.5) * 20))
    print(f"  Topic {i}: {c:+.4f} {bar}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Temporal topic evolution
# ══════════════════════════════════════════════════════════════════════

# Add topics to dataframe
news_with_topics = news_clean.with_columns(
    pl.Series("topic", topics[: news_clean.height])
)

# Topic distribution over time
temporal = (
    news_with_topics.filter(pl.col("topic") >= 0)  # Exclude outliers
    .group_by("year_month", "topic")
    .agg(pl.col("topic").count().alias("count"))
    .sort("year_month", "topic")
)

# Compute topic proportion per month
monthly_totals = temporal.group_by("year_month").agg(
    pl.col("count").sum().alias("total")
)
temporal = temporal.join(monthly_totals, on="year_month").with_columns(
    (pl.col("count") / pl.col("total")).alias("proportion")
)

print(f"\n=== Temporal Evolution ===")
print(f"Months covered: {temporal['year_month'].n_unique()}")

# Show trending topics (biggest increase in recent months)
months = sorted(temporal["year_month"].unique().to_list())
if len(months) >= 6:
    early = months[:3]
    late = months[-3:]

    print("\nTrending topics (last 3 months vs first 3 months):")
    for topic_id in range(min(n_topics, 10)):
        early_prop = temporal.filter(
            (pl.col("year_month").is_in(early)) & (pl.col("topic") == topic_id)
        )["proportion"].mean()
        late_prop = temporal.filter(
            (pl.col("year_month").is_in(late)) & (pl.col("topic") == topic_id)
        )["proportion"].mean()

        if early_prop is not None and late_prop is not None:
            change = (late_prop - early_prop) * 100
            arrow = "↑" if change > 1 else "↓" if change < -1 else "→"
            print(f"  Topic {topic_id}: {change:+.1f}pp {arrow}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Visualise
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# Topic coherence comparison
coherence_data = {f"Topic_{i}": {"NPMI": c} for i, c in enumerate(coherences[:10])}
fig = viz.metric_comparison(coherence_data)
fig.update_layout(title="Topic Coherence (NPMI)")
fig.write_html("ex3_topic_coherence.html")
print("\nSaved: ex3_topic_coherence.html")

# Topic size distribution
topic_counts = Counter(t for t in topics if t >= 0)
size_data = {"Topic Size": [topic_counts.get(i, 0) for i in range(min(n_topics, 15))]}
fig_size = viz.training_history(size_data, x_label="Topic ID")
fig_size.update_layout(title="Topic Size Distribution")
fig_size.write_html("ex3_topic_sizes.html")
print("Saved: ex3_topic_sizes.html")

print("\n✓ Exercise 3 complete — topic modeling with BERTopic / NMF")
