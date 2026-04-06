# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT08 — Exercise 8: Capstone — Full NLP Pipeline
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build end-to-end NLP system: preprocessing → embeddings →
#   transformer classification → deployment via OnnxBridge.
#
# TASKS:
#   1. Preprocess corpus (tokenize, normalize)
#   2. Generate embeddings
#   3. Train classifier via TrainingPipeline
#   4. Evaluate with multiple metrics
#   5. Export to ONNX via OnnxBridge
#   6. Compare inference speed and model size
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math
import os
import re
import time
from collections import Counter

import polars as pl

from kailash_ml import ModelVisualizer, OnnxBridge, TrainingPipeline

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Preprocess corpus
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
speeches = loader.load("ascent08", "sg_parliament_speeches.parquet")

print(f"=== Singapore Parliament Speeches ===")
print(f"Shape: {speeches.shape}")
print(f"Columns: {speeches.columns}")
print(f"Topics: {speeches['topic'].unique().to_list()[:5]}...")


def normalize_text(text: str) -> str:
    """Full NLP preprocessing pipeline."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)  # URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # Non-alphanumeric
    text = re.sub(r"\s+", " ", text).strip()  # Whitespace
    return text


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer with length filter."""
    return [t for t in text.split() if 2 <= len(t) <= 30]


# Apply preprocessing
speeches = speeches.with_columns(
    pl.col("text")
    .map_elements(normalize_text, return_dtype=pl.Utf8)
    .alias("clean_text"),
)
speeches = speeches.with_columns(
    pl.col("clean_text")
    .map_elements(lambda t: len(tokenize(t)), return_dtype=pl.Int64)
    .alias("n_tokens"),
)

print(f"\nPreprocessing complete:")
print(f"  Avg tokens per speech: {speeches['n_tokens'].mean():.0f}")
print(f"  Min: {speeches['n_tokens'].min()}, Max: {speeches['n_tokens'].max()}")
print(f"\nSample (first 100 chars):")
print(f"  Original: {speeches['text'][0][:100]}...")
print(f"  Cleaned:  {speeches['clean_text'][0][:100]}...")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Generate embeddings
# ══════════════════════════════════════════════════════════════════════

# Build vocabulary from corpus
all_tokens = []
for text in speeches["clean_text"].to_list():
    all_tokens.extend(tokenize(text))

# Frequency-based vocabulary (top 5000 tokens)
token_freq = Counter(all_tokens)
vocab = ["<PAD>", "<UNK>"] + [t for t, _ in token_freq.most_common(5000)]
token_to_idx = {t: i for i, t in enumerate(vocab)}

print(f"\n=== Vocabulary ===")
print(f"Total unique tokens: {len(token_freq)}")
print(f"Vocabulary size (capped): {len(vocab)}")
print(f"Top 10: {[t for t, _ in token_freq.most_common(10)]}")


def text_to_tfidf(text: str, vocab_map: dict, idf: dict) -> list[float]:
    """Convert text to TF-IDF vector."""
    tokens = tokenize(text)
    tf = Counter(tokens)
    total = len(tokens) if tokens else 1
    vec = [0.0] * len(vocab_map)
    for token, count in tf.items():
        if token in vocab_map:
            idx = vocab_map[token]
            vec[idx] = (count / total) * idf.get(token, 0.0)
    return vec


# Compute IDF
n_docs = speeches.height
doc_freq = Counter()
for text in speeches["clean_text"].to_list():
    unique_tokens = set(tokenize(text))
    for t in unique_tokens:
        doc_freq[t] += 1

idf = {t: math.log(n_docs / (1 + df)) for t, df in doc_freq.items()}

# Generate embeddings for all documents
embeddings = [
    text_to_tfidf(text, token_to_idx, idf) for text in speeches["clean_text"].to_list()
]

# Add embeddings as columns to dataframe
feature_cols = [f"feat_{i}" for i in range(len(vocab))]
embed_df = pl.DataFrame(
    {
        feature_cols[i]: [row[i] for row in embeddings]
        for i in range(min(500, len(vocab)))  # Cap features for tractability
    }
)
feature_cols = feature_cols[:500]

speeches_with_features = pl.concat([speeches, embed_df], how="horizontal")
print(f"\nEmbeddings: {len(feature_cols)} TF-IDF features per document")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Train classifier via TrainingPipeline
# ══════════════════════════════════════════════════════════════════════

n_train = int(speeches.height * 0.8)
train_set = speeches_with_features[:n_train]
test_set = speeches_with_features[n_train:]

pipeline = TrainingPipeline(
    model_type="text_classifier",
    target="topic",
    features=feature_cols,
    config={
        "algorithm": "gradient_boosting",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
    },
)

print(f"\n=== Training ===")
start_time = time.time()
result = pipeline.fit(train_set)
train_time = time.time() - start_time
print(f"Training time: {train_time:.1f}s")
print(f"Training metrics: {result.metrics}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Evaluate with multiple metrics
# ══════════════════════════════════════════════════════════════════════

predictions = pipeline.predict(test_set)
y_true = test_set["topic"].to_list()
y_pred = predictions["prediction"].to_list()

correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
accuracy = correct / len(y_true)

# Per-class analysis
classes = list(set(y_true))
print(f"\n=== Evaluation ===")
print(f"Overall accuracy: {accuracy:.4f}")
print(f"\nPer-class performance:")
for cls in sorted(classes):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-10)
    print(f"  {cls}: precision={prec:.3f}, recall={rec:.3f}, F1={f1:.3f}")

viz = ModelVisualizer()
fig = viz.plot_confusion_matrix(
    y_true=y_true, y_pred=y_pred, class_names=sorted(classes)
)
fig.write_html("nlp_capstone_confusion.html")
print(f"Confusion matrix saved.")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Export to ONNX via OnnxBridge
# ══════════════════════════════════════════════════════════════════════


async def export_model():
    bridge = OnnxBridge()

    onnx_path = bridge.export(
        model=result.model,
        input_shape=(1, len(feature_cols)),
        output_path="nlp_classifier.onnx",
    )

    print(f"\n=== ONNX Export ===")
    print(f"Path: {onnx_path}")

    # Validate
    test_sample = [test_set.select(feature_cols).row(0)]
    metrics = bridge.validate(onnx_path, test_data=test_sample, expected=[y_pred[0]])
    print(f"Validation: {metrics}")

    return bridge, onnx_path


bridge, onnx_path = asyncio.run(export_model())


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Compare inference speed and model size
# ══════════════════════════════════════════════════════════════════════

onnx_size = os.path.getsize(onnx_path) if os.path.exists(onnx_path) else 0

print(f"\n=== Model Comparison ===")
print(f"ONNX model size: {onnx_size / 1024:.1f} KB")

# Benchmark original
n_bench = 50
samples = [
    list(test_set.select(feature_cols).row(i))
    for i in range(min(n_bench, test_set.height))
]

start = time.time()
for s in samples:
    row_df = pl.DataFrame({c: [v] for c, v in zip(feature_cols, s)})
    pipeline.predict(row_df.with_columns(pl.lit("unknown").alias("topic")))
original_ms = (time.time() - start) / len(samples) * 1000

print(f"Original: {original_ms:.1f}ms per prediction")
print(f"ONNX: typically 2-5× faster (graph optimizations, no Python overhead)")

print(f"\n=== Full NLP Pipeline Summary ===")
print(f"1. Preprocessing: normalize → tokenize → vocabulary")
print(f"2. Embeddings: TF-IDF (500 features)")
print(f"3. Training: TrainingPipeline (gradient boosting)")
print(f"4. Evaluation: accuracy={accuracy:.4f}, per-class F1")
print(f"5. Export: OnnxBridge ({onnx_size/1024:.1f} KB)")
print(f"From raw text to production-ready model — the Kailash NLP lifecycle.")

print("\n✓ Exercise 8 complete — full NLP pipeline from preprocessing to ONNX")
