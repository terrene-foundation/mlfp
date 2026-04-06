# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT08 — Exercise 3: Word Embeddings
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Train Word2Vec skip-gram from scratch, explore word analogies,
#   and visualize embedding space with ModelVisualizer (t-SNE).
#
# TASKS:
#   1. Build skip-gram training pairs from corpus
#   2. Implement Word2Vec training loop
#   3. Test word similarity and analogies
#   4. Visualize embeddings with ModelVisualizer (t-SNE)
#   5. Compare with pre-trained GloVe vectors
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter

import polars as pl

from kailash_ml import DataExplorer, ModelVisualizer

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
df = loader.load("ascent08", "sg_news_articles.parquet")

explorer = DataExplorer()
summary = explorer.analyze(df)
print(f"=== Dataset: {df.height} articles ===")
print(summary)


# ── Helpers ───────────────────────────────────────────────────────────


def tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric."""
    # TODO: Clean text with regex (remove non-alphanumeric) and split into tokens.
    # Hint: re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    return ____


corpus_texts = df.select("text").to_series().to_list()
# TODO: Tokenize each document in the corpus.
# Hint: [tokenize(t) for t in corpus_texts]
tokenized_corpus = ____
# TODO: Flatten all tokens from all documents into a single list.
# Hint: [tok for doc in tokenized_corpus for tok in doc]
all_tokens = ____

word_counts = Counter(all_tokens)
# TODO: Build vocabulary from top 3000 words with min frequency 2.
# Hint: [w for w, c in word_counts.most_common(3000) if c >= 2]
vocab = ____
# TODO: Create word-to-index mapping dictionary.
# Hint: {w: i for i, w in enumerate(vocab)}
word_to_idx = ____
vocab_size = len(vocab)

print(f"Vocabulary: {vocab_size} words (min freq=2)")
print(f"Total tokens: {len(all_tokens):,}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Build skip-gram training pairs
# ══════════════════════════════════════════════════════════════════════


def build_skipgram_pairs(
    tokens: list[str], word_to_idx: dict[str, int], window: int = 2
) -> list[tuple[int, int]]:
    """Generate (center, context) index pairs within a window."""
    pairs = []
    for i, token in enumerate(tokens):
        if token not in word_to_idx:
            continue
        center_idx = word_to_idx[token]
        # TODO: Compute the context window start index (clamped to 0).
        # Hint: max(0, i - window)
        start = ____
        # TODO: Compute the context window end index (clamped to len).
        # Hint: min(len(tokens), i + window + 1)
        end = ____
        for j in range(start, end):
            if j == i or tokens[j] not in word_to_idx:
                continue
            # TODO: Append the (center_idx, context_idx) pair.
            # Hint: pairs.append((center_idx, word_to_idx[tokens[j]]))
            ____
    return pairs


# Build pairs from first 50 articles for speed
# TODO: Flatten tokens from first 50 documents.
# Hint: [tok for doc in tokenized_corpus[:50] for tok in doc]
sample_tokens = ____
pairs = build_skipgram_pairs(sample_tokens, word_to_idx, window=2)
print(f"\nSkip-gram pairs: {len(pairs):,} from {len(sample_tokens):,} tokens")
print(f"Sample pairs: {[(vocab[c], vocab[t]) for c, t in pairs[:5]]}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Word2Vec training loop
# ══════════════════════════════════════════════════════════════════════

embedding_dim = 50

# Initialize embeddings randomly
W_center = [
    [random.gauss(0, 0.1) for _ in range(embedding_dim)] for _ in range(vocab_size)
]
W_context = [
    [random.gauss(0, 0.1) for _ in range(embedding_dim)] for _ in range(vocab_size)
]


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        # TODO: Return sigmoid for positive x.
        # Hint: 1.0 / (1.0 + math.exp(-x))
        return ____
    exp_x = math.exp(x)
    # TODO: Return sigmoid for negative x (using exp_x).
    # Hint: exp_x / (1.0 + exp_x)
    return ____


def dot_product(a: list[float], b: list[float]) -> float:
    """Dot product of two vectors."""
    # TODO: Compute dot product as sum of element-wise products.
    # Hint: sum(x * y for x, y in zip(a, b))
    return ____


def train_skipgram(
    pairs: list[tuple[int, int]],
    W_center: list[list[float]],
    W_context: list[list[float]],
    epochs: int = 3,
    lr: float = 0.01,
    n_negative: int = 5,
) -> list[float]:
    """Train skip-gram with negative sampling."""
    losses = []
    # Build unigram distribution for negative sampling
    token_freq = [0] * vocab_size
    for c, t in pairs:
        token_freq[c] += 1
    # TODO: Compute smoothed frequency sum (f^0.75 smoothing).
    # Hint: sum(f**0.75 for f in token_freq)
    freq_sum = ____
    # TODO: Build negative sampling probability distribution.
    # Hint: [(f**0.75) / freq_sum for f in token_freq]
    neg_probs = ____

    for epoch in range(epochs):
        epoch_loss = 0.0
        random.shuffle(pairs)
        for center, context in pairs[:5000]:  # Limit for speed
            # Positive pair
            # TODO: Compute dot product score between center and context embeddings.
            # Hint: dot_product(W_center[center], W_context[context])
            score = ____
            prob = sigmoid(score)
            # TODO: Compute negative log-likelihood loss for positive pair.
            # Hint: -math.log(prob + 1e-10)
            loss = ____
            # TODO: Compute gradient scaled by learning rate.
            # Hint: (prob - 1) * lr
            grad = ____

            for d in range(embedding_dim):
                W_center[center][d] -= grad * W_context[context][d]
                W_context[context][d] -= grad * W_center[center][d]

            # Negative samples
            for _ in range(n_negative):
                neg = random.choices(range(vocab_size), weights=neg_probs, k=1)[0]
                if neg == context:
                    continue
                score = dot_product(W_center[center], W_context[neg])
                prob = sigmoid(score)
                # TODO: Add negative sample loss (push apart).
                # Hint: loss += -math.log(1 - prob + 1e-10)
                loss += ____
                # TODO: Compute negative sample gradient.
                # Hint: prob * lr
                grad = ____
                for d in range(embedding_dim):
                    W_center[center][d] -= grad * W_context[neg][d]
                    W_context[neg][d] -= grad * W_center[center][d]

            epoch_loss += loss

        # TODO: Compute average loss for the epoch.
        # Hint: epoch_loss / min(len(pairs), 5000)
        avg_loss = ____
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    return losses


print(f"\nTraining skip-gram (dim={embedding_dim})...")
losses = train_skipgram(pairs, W_center, W_context, epochs=3, lr=0.01)


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Word similarity and analogies
# ══════════════════════════════════════════════════════════════════════


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors."""
    # TODO: Compute dot product of vectors a and b.
    # Hint: sum(x * y for x, y in zip(a, b))
    dot = ____
    # TODO: Compute L2 norm of vector a.
    # Hint: math.sqrt(sum(x * x for x in a))
    na = ____
    # TODO: Compute L2 norm of vector b.
    # Hint: math.sqrt(sum(y * y for y in b))
    nb = ____
    if na == 0 or nb == 0:
        return 0.0
    # TODO: Return cosine similarity (dot product / product of norms).
    # Hint: dot / (na * nb)
    return ____


def most_similar(word: str, top_k: int = 5) -> list[tuple[str, float]]:
    """Find the most similar words by cosine similarity."""
    if word not in word_to_idx:
        return []
    idx = word_to_idx[word]
    vec = W_center[idx]
    sims = []
    for i in range(vocab_size):
        if i == idx:
            continue
        # TODO: Append (word, similarity) tuple for each candidate.
        # Hint: sims.append((vocab[i], cosine_sim(vec, W_center[i])))
        ____
    # TODO: Sort similarities in descending order.
    # Hint: sims.sort(key=lambda x: x[1], reverse=True)
    ____
    return sims[:top_k]


def analogy(a: str, b: str, c: str, top_k: int = 3) -> list[tuple[str, float]]:
    """Solve a:b :: c:? via vector arithmetic (b - a + c)."""
    if any(w not in word_to_idx for w in [a, b, c]):
        return []
    # TODO: Compute the analogy vector: b - a + c (element-wise).
    # Hint: [W_center[word_to_idx[b]][d] - W_center[word_to_idx[a]][d] + W_center[word_to_idx[c]][d] for d in range(embedding_dim)]
    vec = ____
    exclude = {a, b, c}
    sims = []
    for i in range(vocab_size):
        if vocab[i] in exclude:
            continue
        sims.append((vocab[i], cosine_sim(vec, W_center[i])))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]


test_words = ["singapore", "economy", "government", "policy"]
for word in test_words:
    similar = most_similar(word, top_k=5)
    if similar:
        print(f"\nMost similar to '{word}': {similar}")

# Try analogies
print(f"\n--- Word Analogies ---")
analogy_tests = [("man", "woman", "king"), ("singapore", "asia", "london")]
for a, b, c in analogy_tests:
    result = analogy(a, b, c)
    if result:
        print(f"  {a}:{b} :: {c}:? → {result}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Visualize embeddings with ModelVisualizer (t-SNE)
# ══════════════════════════════════════════════════════════════════════

# Select top words for visualization
top_words = vocab[:100]
# TODO: Extract embedding vectors for top words.
# Hint: [W_center[word_to_idx[w]] for w in top_words]
top_embeddings = ____

viz = ModelVisualizer()
# TODO: Plot embeddings using t-SNE via ModelVisualizer.
# Hint: viz.plot_embeddings(embeddings=top_embeddings, labels=top_words, method="tsne", title="Word2Vec Skip-gram Embeddings (t-SNE)")
fig = ____
print(f"\n=== Embedding visualization generated ===")
print(f"Plotted {len(top_words)} words in 2D via t-SNE")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare with pre-trained GloVe vectors
# ══════════════════════════════════════════════════════════════════════

print(f"\n--- Trained vs Pre-trained Comparison ---")
print(
    f"Our model: {vocab_size} words, {embedding_dim}D, trained on {len(pairs):,} pairs"
)
print(f"GloVe-6B:  400K words, 50/100/200/300D, trained on 6B tokens")
print(f"\nKey differences:")
print(f"  1. Corpus size: ours ~small vs GloVe ~6B tokens (Wikipedia + news)")
print(f"  2. Vocabulary: ours {vocab_size} vs GloVe 400K")
print(f"  3. Training: our skip-gram vs GloVe co-occurrence matrix factorization")
print(f"  4. Quality: pre-trained captures more semantic relationships")
print(f"\nWhen to use each:")
print(f"  Pre-trained: general NLP, limited domain data")
print(f"  Custom-trained: domain-specific terminology (e.g. Singapore policy)")

print(
    "\n✓ Exercise 3 complete — Word2Vec skip-gram training, analogies, t-SNE visualization"
)
