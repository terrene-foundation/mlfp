# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT08 — Exercise 7: Transfer Learning with Transformers
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Fine-tune a pre-trained transformer for Singapore-specific
#   text classification using AutoMLEngine's text mode.
#
# TASKS:
#   1. Load pre-trained transformer embeddings
#   2. Configure AutoMLEngine for text classification
#   3. Fine-tune on domain-specific data
#   4. Evaluate with confusion matrix via ModelVisualizer
#   5. Register best model in ModelRegistry
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import pickle

import polars as pl

from kailash.infrastructure import ConnectionManager
from kailash_ml import AutoMLEngine, ModelRegistry, ModelVisualizer
from kailash_ml.types import MetricSpec

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Load pre-trained transformer embeddings
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
reviews = loader.load("ascent08", "sg_product_reviews.parquet")

print(f"=== Singapore Product Reviews ===")
print(f"Shape: {reviews.shape}")
print(f"Columns: {reviews.columns}")

# Label distribution
label_counts = reviews.group_by("rating").agg(pl.len().alias("count")).sort("rating")
print(f"\nRating distribution:")
for row in label_counts.iter_rows():
    print(f"  Rating {row[0]}: {row[1]} reviews")

# Binary classification: positive (4-5) vs negative (1-2)
reviews = reviews.with_columns(
    pl.when(pl.col("rating") >= 4)
    .then(pl.lit("positive"))
    .otherwise(pl.lit("negative"))
    .alias("sentiment")
)

# Train/test split
n_train = int(reviews.height * 0.8)
train_reviews = reviews[:n_train]
test_reviews = reviews[n_train:]

print(f"\nBinary sentiment: positive (rating >= 4) vs negative (rating <= 2)")
print(f"Train: {train_reviews.height}, Test: {test_reviews.height}")

# Transfer learning insight
print(f"\n=== Transfer Learning ===")
print(f"Pre-trained transformers already understand language structure,")
print(f"grammar, and general semantics from training on billions of tokens.")
print(f"Fine-tuning adapts this knowledge to our specific domain (Singapore")
print(f"product reviews) with relatively few examples.")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Configure AutoMLEngine for text classification
# ══════════════════════════════════════════════════════════════════════

# TODO: Configure AutoMLEngine for text classification on sentiment.
# Hint: AutoMLEngine(task="text_classification", target="sentiment", text_column="text", max_trials=10, optimization_metric="f1", time_budget_seconds=300)
engine = ____

print(f"\n=== AutoMLEngine Configuration ===")
print(f"Task: text_classification")
print(f"Target: sentiment (positive/negative)")
print(f"Text column: text")
print(f"Max trials: 10")
print(f"Optimization: F1 score")
print(f"AutoMLEngine automatically tries:")
print(f"  - TF-IDF + LogisticRegression")
print(f"  - TF-IDF + SVM")
print(f"  - Transformer embeddings + classifier")
print(f"  - Various hyperparameter combinations")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Fine-tune on domain-specific data
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Fine-tuning ===")
# TODO: Fit the engine on training data.
# Hint: engine.fit(train_reviews)
result = ____

print(f"Best model: {result.best_model_name}")
print(f"Best F1: {result.best_score:.4f}")
print(f"Trials completed: {result.n_trials}")

# Leaderboard
print(f"\nLeaderboard:")
for i, trial in enumerate(result.leaderboard[:5]):
    print(f"  {i+1}. {trial['model']}: F1={trial['score']:.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Evaluate with confusion matrix
# ══════════════════════════════════════════════════════════════════════

# TODO: Generate predictions on test data.
# Hint: engine.predict(test_reviews)
predictions = ____
y_true = test_reviews["sentiment"].to_list()
y_pred = predictions["prediction"].to_list()

# Accuracy
correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
accuracy = correct / len(y_true)

# Per-class metrics
tp = sum(1 for t, p in zip(y_true, y_pred) if t == "positive" and p == "positive")
fp = sum(1 for t, p in zip(y_true, y_pred) if t == "negative" and p == "positive")
fn = sum(1 for t, p in zip(y_true, y_pred) if t == "positive" and p == "negative")
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-10)

print(f"\n=== Test Evaluation ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")

viz = ModelVisualizer()
# TODO: Plot confusion matrix with ModelVisualizer.
# Hint: viz.plot_confusion_matrix(y_true=y_true, y_pred=y_pred, class_names=["negative", "positive"])
fig = ____
fig.write_html("sentiment_confusion_matrix.html")
print(f"Confusion matrix saved to sentiment_confusion_matrix.html")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Register best model in ModelRegistry
# ══════════════════════════════════════════════════════════════════════


async def register_best():
    conn = ConnectionManager("sqlite:///nlp_models.db")
    await conn.initialize()

    registry = ModelRegistry(conn)
    await registry.initialize()

    # TODO: Register the model with name, artifact, and metrics.
    # Hint: registry.register_model(name="sg_sentiment_classifier", artifact=pickle.dumps(result.best_model), metrics=[MetricSpec(name="f1", value=f1), MetricSpec(name="accuracy", value=accuracy), MetricSpec(name="precision", value=precision), MetricSpec(name="recall", value=recall)])
    version = await ____

    # TODO: Promote the model to production stage.
    # Hint: registry.promote_model(name="sg_sentiment_classifier", version=version.version, target_stage="production")
    await ____

    print(f"\n=== ModelRegistry ===")
    print(f"Registered: sg_sentiment_classifier v{version.version}")
    print(f"Stage: production")
    print(f"Metrics: F1={f1:.4f}, accuracy={accuracy:.4f}")

    # List all models
    models = await registry.list_models()
    print(f"Total registered models: {len(models)}")

    return registry


registry = asyncio.run(register_best())

print(f"\n=== Transfer Learning Summary ===")
print(f"Pre-trained transformers provide powerful text representations")
print(f"that can be fine-tuned with relatively few domain-specific examples.")
print(f"AutoMLEngine automates the search across model architectures")
print(f"and hyperparameters, finding the best approach for your data.")

print("\n✓ Exercise 7 complete — transformer transfer learning with AutoMLEngine")
