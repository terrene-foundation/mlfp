# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP04 — Assessment Task 4: Neural Network Foundations (Reference)

Reference implementation. Withheld from students. Verified to pass grader.py.
Framework-first: the network is trained through the kailash-ml SklearnTrainable
adapter (a multi-layer perceptron), NOT a raw torch training loop.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.neural_network import MLPClassifier

from kailash_ml import SklearnTrainable

SEED = 20260404
N = 800
SPLIT = 600  # first 600 rows train, last 200 test
FEATURES = ["x1", "x2"]
TARGET = "label"


def make_circles() -> pl.DataFrame:
    """Two concentric rings — class 0 inside, class 1 outside.

    The classes share a centre (the origin), so NO straight line separates
    them: a linear model is stuck near chance. Only a model with a hidden
    layer (a non-linear decision boundary) can solve it.
    """
    rng = np.random.default_rng(SEED)
    m = N // 2

    def ring(radius: float, noise: float) -> np.ndarray:
        theta = rng.uniform(0, 2 * np.pi, m)
        r = radius + rng.normal(0, noise, m)
        return np.c_[r * np.cos(theta), r * np.sin(theta)]

    X = np.vstack([ring(1.0, 0.18), ring(3.0, 0.30)])
    y = np.r_[np.zeros(m, dtype=int), np.ones(m, dtype=int)]
    perm = rng.permutation(N)
    X, y = X[perm], y[perm]
    return pl.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "label": y})


def _predict_labels(trainable: SklearnTrainable, frame: pl.DataFrame) -> np.ndarray:
    preds = trainable.predict(frame.select(FEATURES))
    return preds.to_polars()[preds.column].to_numpy().ravel().astype(int)


def solve() -> dict:
    """Train an MLP through kailash-ml and beat the linear ceiling on circles."""
    df = make_circles()
    train_df = df.head(SPLIT)
    test_df = df.tail(N - SPLIT)

    # Multi-layer perceptron driven through the kailash-ml Trainable adapter.
    mlp = SklearnTrainable(
        estimator=MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            max_iter=2000,
            random_state=SEED,
        ),
        target=TARGET,
        metric="accuracy",
    )
    mlp.fit(train_df)

    train_pred = _predict_labels(mlp, train_df)
    test_pred = _predict_labels(mlp, test_df)

    y_train = train_df[TARGET].to_numpy()
    y_test = test_df[TARGET].to_numpy()

    return {
        "test_predictions": [int(v) for v in test_pred],
        "test_accuracy": float((test_pred == y_test).mean()),
        "train_accuracy": float((train_pred == y_train).mean()),
    }


if __name__ == "__main__":
    out = solve()
    print(f"train_accuracy : {out['train_accuracy']:.4f}")
    print(f"test_accuracy  : {out['test_accuracy']:.4f}")
    print(f"n test preds   : {len(out['test_predictions'])}")
