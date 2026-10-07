# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 6.5: Word Embeddings (Word2Vec-style) + Sentiment
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Explain how Word2Vec learns dense word vectors (CBOW vs skip-gram)
#     and why it is implicitly a factorisation of a word-context PMI matrix
#   - Learn real word embeddings from unlabelled text with PPMI + SVD
#   - Average word vectors into a document vector
#   - Train a sentiment classifier on human-labelled reviews and test it
#     on unseen sentences against a lexicon and a TF-IDF baseline
#   - Apply the technique to bank app-review triage
#
# PREREQUISITES: Ex 6.1 (TF-IDF), Ex 3 (SVD / dimensionality reduction),
# basic classification (logistic regression).
#
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — Word2Vec and the PMI-factorisation view
#   2. Build — co-occurrence counts -> PPMI -> SVD word vectors
#   3. Train — document vectors + logistic regression vs baselines
#   4. Visualise — word-vector map, classifier comparison
#   5. Apply — bank app-store review triage
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import plotly.graph_objects as go
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from kailash_ml import ModelVisualizer

from shared.mlfp04.ex_6 import (
    NEGATIVE_WORDS,
    OUTPUT_DIR,
    POSITIVE_WORDS,
    lexicon_sentiment,
    load_sentiment_reviews,
    print_scenario,
    tokenize_review,
)
from shared.mlfp04 import create_visualizer


# ════════════════════════════════════════════════════════════════════════
# THEORY — Word2Vec, and Word2Vec as Matrix Factorisation
# ════════════════════════════════════════════════════════════════════════
# Word2Vec (Mikolov 2013) trains a shallow network to predict a target
# word from its context (CBOW) or the context from the target
# (skip-gram). The network is discarded after training — what you keep
# is the weight matrix: one dense vector per vocabulary word.
#
# Emergent properties of that objective:
#   - Words used in similar contexts end up near each other
#   - Dense 100-300D vectors replace 10K+ sparse bag-of-words columns
#
# THE FACTORISATION VIEW (Levy & Goldberg, 2014): skip-gram with negative
# sampling implicitly factorises a word-by-context matrix whose cells are
# (shifted) pointwise mutual information,
#
#     PMI(w, c) = log( P(w, c) / (P(w) P(c)) )
#
# So we can build the same family of embeddings EXPLICITLY, with tools
# from this module:
#   1. count how often each word appears within a few positions of each
#      context word (a sliding window over unlabelled text)
#   2. turn counts into Positive PMI (negative values clipped to 0)
#   3. factorise the PPMI matrix with truncated SVD — each word's row of
#      U * sqrt(S) is its embedding
# The course environment does not ship the gensim package, so this is
# how we get genuine distributional word vectors here; in production you
# would train skip-gram with gensim or load pretrained vectors.
#
# A WARNING TO CHECK IN THE DATA: "good" and "bad" appear in the same
# contexts ("a ___ movie"), so distributional vectors can put antonyms
# close together. Similar context != similar sentiment — inspect the
# nearest neighbours and the word map below.
#
# DOCUMENT VECTORS: a sentence is a bag of words, so a simple sentence
# vector is the mean of its word vectors. It loses word order (and so
# most negation) but is cheap and dense.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: co-occurrence -> PPMI -> SVD word vectors
# ════════════════════════════════════════════════════════════════════════

train_df, test_df = load_sentiment_reviews()
train_texts = train_df["text"].to_list()
test_texts = test_df["text"].to_list()
y_train = train_df["label"].to_numpy()
y_test = test_df["label"].to_numpy()
print(f"Human-labelled movie-review text (SST-2):")
print(f"  train: {len(train_texts):,} phrases/sentences (de-duplicated)")
print(f"  test:  {len(test_texts):,} unseen sentences")
print(f"  positive share — train {y_train.mean():.1%}, test {y_test.mean():.1%}")

MIN_COUNT = 10  # ignore words seen fewer than 10 times
WINDOW = 4  # context = up to 4 words either side
EMBED_DIM = 100

# Vocabulary from the TRAIN text only (labels are not used here)
train_tokens = [tokenize_review(t) for t in train_texts]
word_counts = Counter(w for toks in train_tokens for w in toks)
vocab = [w for w, c in word_counts.most_common() if c >= MIN_COUNT]
word_index = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
print(f"\nVocabulary: {V:,} words (count >= {MIN_COUNT})")


def cooccurrence_matrix(
    token_lists: list[list[str]], window: int
) -> sp.csr_matrix:
    """Symmetric word-context counts within `window` positions.

    A context word d positions away contributes 1/d (closer = stronger).
    Built with array shifts instead of a Python double loop.
    """
    ids = [
        np.array([word_index[w] for w in toks if w in word_index], dtype=np.int64)
        for toks in token_lists
    ]
    flat = np.concatenate(ids)
    sentence_id = np.concatenate([np.full(len(a), i) for i, a in enumerate(ids)])
    rows, cols, vals = [], [], []
    for d in range(1, window + 1):
        same_sentence = sentence_id[d:] == sentence_id[:-d]
        left, right = flat[:-d][same_sentence], flat[d:][same_sentence]
        weight = np.full(left.shape, 1.0 / d)
        rows += [left, right]
        cols += [right, left]
        vals += [weight, weight]
    return sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(V, V),
    ).tocsr()


def ppmi(counts: sp.csr_matrix, context_alpha: float = 0.75) -> sp.csr_matrix:
    """Positive PMI with context-distribution smoothing (alpha = 0.75)."""
    total = counts.sum()
    p_word = np.asarray(counts.sum(axis=1)).ravel() / total
    context = np.asarray(counts.sum(axis=0)).ravel() ** context_alpha
    p_context = context / context.sum()
    coo = counts.tocoo()
    p_joint = coo.data / total
    pmi = np.log(p_joint / (p_word[coo.row] * p_context[coo.col]))
    keep = pmi > 0
    return sp.csr_matrix(
        (pmi[keep], (coo.row[keep], coo.col[keep])), shape=counts.shape
    )


C = cooccurrence_matrix(train_tokens, WINDOW)
P = ppmi(C)
svd = TruncatedSVD(n_components=EMBED_DIM, random_state=42)
U_S = svd.fit_transform(P)  # = U * S
embeddings = U_S / np.sqrt(svd.singular_values_)  # = U * sqrt(S)
embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
print(f"Co-occurrence non-zeros: {C.nnz:,}; PPMI non-zeros: {P.nnz:,}")
print(f"Word embeddings: {embeddings.shape} (words x dimensions)")


def nearest_words(word: str, k: int = 6) -> list[str]:
    """Cosine nearest neighbours (rows are unit length, so dot = cosine)."""
    sims = embeddings @ embeddings[word_index[word]]
    return [vocab[i] for i in np.argsort(-sims)[1 : k + 1]]


print("\nNearest neighbours (cosine):")
for probe in ("good", "bad", "funny", "boring"):
    if probe in word_index:
        print(f"  {probe:<8} -> {', '.join(nearest_words(probe))}")


def document_vector(text: str) -> np.ndarray:
    """Average the embeddings of the in-vocabulary tokens of a text."""
    rows = [word_index[w] for w in tokenize_review(text) if w in word_index]
    if not rows:
        return np.zeros(EMBED_DIM, dtype=np.float64)
    return embeddings[rows].mean(axis=0)


X_train = np.stack([document_vector(t) for t in train_texts])
X_test = np.stack([document_vector(t) for t in test_texts])


# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert embeddings.shape == (V, EMBED_DIM), "Task 2: one EMBED_DIM vector per word"
assert np.allclose(
    np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-6
), "Task 2: word vectors should be unit length"
assert P.data.min() > 0, "Task 2: PPMI keeps only positive values"
assert not set(test_texts) & set(train_texts), "Task 2: test text must be unseen"
assert X_train.shape == (len(train_texts), EMBED_DIM), "Task 2: document vectors"
print("\n[ok] Checkpoint 1 passed — PPMI + SVD word vectors and document vectors\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: embedding classifier vs lexicon and TF-IDF baselines
# ════════════════════════════════════════════════════════════════════════

majority_class = int(y_train.mean() >= 0.5)
majority_acc = float((y_test == majority_class).mean())

# Lexicon baseline: positive if score > 0, negative if < 0; when no
# lexicon word appears (score 0) it falls back to the majority class.
lex_scores = lexicon_sentiment(test_texts)
lex_pred = np.where(lex_scores > 0, 1, np.where(lex_scores < 0, 0, majority_class))
lex_acc = float(accuracy_score(y_test, lex_pred))
lex_coverage = float((lex_scores != 0).mean())

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)
emb_train_acc = float(accuracy_score(y_train, clf.predict(X_train)))
emb_test_acc = float(accuracy_score(y_test, clf.predict(X_test)))

# Reference: sparse TF-IDF + logistic regression on the same split
tfidf = TfidfVectorizer(tokenizer=tokenize_review, token_pattern=None, min_df=2)
T_train = tfidf.fit_transform(train_texts)
T_test = tfidf.transform(test_texts)
tfidf_clf = LogisticRegression(max_iter=1000, random_state=42)
tfidf_clf.fit(T_train, y_train)
tfidf_test_acc = float(accuracy_score(y_test, tfidf_clf.predict(T_test)))

print("=" * 70)
print("  Test accuracy on unseen, human-labelled sentences")
print("=" * 70)
print(f"  Majority class ('always {majority_class}')     {majority_acc:.3f}")
print(f"  Lexicon ({lex_coverage:.0%} of sentences matched)  {lex_acc:.3f}")
print(f"  Averaged word vectors + LR ({EMBED_DIM}D)   {emb_test_acc:.3f}"
      f"  (train {emb_train_acc:.3f})")
print(f"  TF-IDF + LR ({T_train.shape[1]:,} sparse cols)  {tfidf_test_acc:.3f}")

best_name = max(
    [("lexicon", lex_acc), ("word vectors", emb_test_acc), ("TF-IDF", tfidf_test_acc)],
    key=lambda kv: kv[1],
)[0]
print(f"\n  Best on this test set: {best_name}.")
if tfidf_test_acc > emb_test_acc:
    print(
        "  Averaging 100-D vectors learned from this small corpus throws away\n"
        "  word identity and order; the sparse model keeps every word as its\n"
        "  own feature. Embeddings pay off when they come from a much larger\n"
        "  corpus than your labelled set, or when features must be compact."
    )


# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert 0.0 <= emb_test_acc <= 1.0, "Task 3: accuracy must be a probability"
assert emb_test_acc > majority_acc, "Task 3: embedding classifier must beat the majority class"
assert len(POSITIVE_WORDS) > 5 and len(NEGATIVE_WORDS) > 5, "Task 3: lexicons non-empty"
print("\n[ok] Checkpoint 2 passed — classifier trained and compared on unseen text\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: word-vector map + classifier comparison
# ════════════════════════════════════════════════════════════════════════
# Project the lexicon words' 100-D vectors to 2-D with PCA. If sentiment
# were what the vectors encode, blue and red would separate cleanly —
# look at where "good" and "bad" land.

map_words = sorted(w for w in POSITIVE_WORDS | NEGATIVE_WORDS if w in word_index)
coords = PCA(n_components=2, random_state=42).fit_transform(
    embeddings[[word_index[w] for w in map_words]]
)
colours = ["#1f77b4" if w in POSITIVE_WORDS else "#d62728" for w in map_words]
fig_map = go.Figure(
    go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode="markers+text",
        text=map_words,
        textposition="top center",
        marker=dict(color=colours, size=9),
    )
)
fig_map.update_layout(
    title="Sentiment-lexicon words in embedding space (PCA, blue=positive, red=negative)",
    xaxis_title="PC1",
    yaxis_title="PC2",
    height=600,
)
fig_map.write_html(str(OUTPUT_DIR / "ex6_5_word_vector_map.html"))

viz = create_visualizer()
acc_data = {
    "Majority class": {"test_accuracy": majority_acc},
    "Lexicon": {"test_accuracy": lex_acc},
    "Word vectors + LR": {"test_accuracy": emb_test_acc},
    "TF-IDF + LR": {"test_accuracy": tfidf_test_acc},
}
fig_acc = viz.metric_comparison(acc_data)
fig_acc.update_layout(title="Sentiment Classifiers — Test Accuracy on Unseen Sentences")
fig_acc.write_html(str(OUTPUT_DIR / "ex6_5_classifier_comparison.html"))

print(f"Saved: {OUTPUT_DIR}/ex6_5_word_vector_map.html")
print(f"Saved: {OUTPUT_DIR}/ex6_5_classifier_comparison.html")


# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert coords.shape == (len(map_words), 2), "Task 4: one 2-D point per mapped word"
assert len(acc_data) == 4, "Task 4: four classifiers compared"
print("\n[ok] Checkpoint 3 passed — visualisations written\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Bank App-Store Review Triage
# ════════════════════════════════════════════════════════════════════════

print_scenario("sentiment_word2vec")
print(
    """
WHY WORD VECTORS + LR FOR REVIEW TRIAGE:
  - Lexicon sentiment is free but brittle: it only fires on listed
    words, and "not good" still counts as positive because negation is
    ignored. Your run above shows how often it matched at all.
  - A learned classifier uses every word it saw in labelled training
    reviews. Averaged word vectors keep the model tiny (100 numbers per
    review) — but on this data a sparse TF-IDF model was compared too,
    so pick the one that wins on YOUR labelled sample.
  - Domain shift is real: these models learned from movie reviews. Bank
    app reviews use different words ("login", "OTP", "transfer"), so
    collect and label a few thousand in-domain reviews before deploying.
  - Other languages need their own labelled data and embeddings (or
    cross-lingually aligned ones); English vectors do not transfer.

ILLUSTRATIVE ARITHMETIC (assumptions, not measured figures):
  - Assume ~40K reviews/month, 15% negative, and that routing a negative
    review to the CX team within 10 minutes (instead of a daily batch)
    avoids escalation worth ~S$50 on average. Catching 1,000 more
    negatives a month is then ~S$50K/month, for a few dollars of CPU.

WHEN TO GO FURTHER:
  - Exercise 8 builds neural networks from these ideas; Module 5 covers
    transformers, which read word order and handle negation far better
    than averaged vectors.
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
  [x] Explained Word2Vec and why it is implicitly a PMI factorisation
  [x] Learned word embeddings from unlabelled text with PPMI + SVD
  [x] Inspected nearest neighbours: similar context != similar sentiment
  [x] Averaged word vectors into document vectors and trained a
      classifier on human-labelled reviews
  [x] Measured it on unseen sentences against lexicon, TF-IDF and
      majority-class baselines
  [x] Mapped the technique to bank app-review triage, with its
      domain-shift and language caveats

  KEY INSIGHT: Embeddings are learned representations — the training
  objective ("predict context", or here "factorise co-occurrence") is a
  proxy, and what you keep is the vector space it produces. Whether that
  space helps a downstream task is an empirical question you answer with
  a held-out test set, not an assumption.

  EXERCISE 6 COMPLETE. You now hold five distinct tools for text:
    - TF-IDF / BM25       — classic retrieval, no training
    - NMF                 — fast, interpretable topics
    - LDA                 — probabilistic, mixed-membership topics
    - BERTopic            — semantic topics from sentence embeddings
    - Word embeddings     — dense learned features for classification

  Next: Exercise 7 — matrix factorisation for recommender systems. The
  same factorise-a-co-occurrence-matrix idea you used here powers
  user-item embeddings.
"""
)
