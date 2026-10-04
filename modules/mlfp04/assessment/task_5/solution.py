# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP04 — Assessment Task 5: From Discovered Segments to a Neural Network (Reference)

Instructor-only reference. Withheld from students.

Part A: hand-written forward and backward pass of a 3-layer network with a
numerically stable cross-entropy (computed from the logit with logaddexp,
never log(sigmoid)).
Part B: unsupervised features — z-scored fields, PCA scores, and one-hot
segment membership from ClusteringEngine with K chosen by silhouette.
Part C: a multi-layer perceptron trained through kailash-ml SklearnTrainable
on raw + discovered features, with L2 penalty and early stopping on a
validation split. Discovery uses train AND test fields together (no labels
involved), so both get the same segments.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.neural_network import MLPClassifier

from kailash_ml import SklearnTrainable
from kailash_ml.engines.clustering import ClusteringEngine
from kailash_ml.engines.dim_reduction import DimReductionEngine

TARGET = "churned"


def load_dev_churn() -> pl.DataFrame:
    return pl.read_parquet(Path(__file__).with_name("dev_churn.parquet"))


# ── Part A ────────────────────────────────────────────────────────────────


def loss_and_gradients(params: dict, X: np.ndarray, y: np.ndarray, l2: float = 0.0) -> tuple[float, dict]:
    W1, b1, W2, b2, W3, b3 = (params[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3"))
    n = X.shape[0]
    y = y.reshape(-1).astype(float)
    Z1 = X @ W1 + b1
    H1 = np.maximum(Z1, 0.0)
    Z2 = H1 @ W2 + b2
    H2 = np.maximum(Z2, 0.0)
    z = (H2 @ W3 + b3).reshape(-1)
    # BCE from logits: log(1+e^z) - y z, stable for any |z|
    data_loss = float(np.mean(np.logaddexp(0.0, z) - y * z))
    loss = data_loss + 0.5 * l2 * float((W1**2).sum() + (W2**2).sum() + (W3**2).sum())

    p = 0.5 * (1.0 + np.tanh(0.5 * z))  # stable sigmoid
    dz = ((p - y) / n).reshape(-1, 1)
    gW3 = H2.T @ dz + l2 * W3
    gb3 = dz.sum(axis=0)
    dH2 = dz @ W3.T
    dZ2 = dH2 * (Z2 > 0)
    gW2 = H1.T @ dZ2 + l2 * W2
    gb2 = dZ2.sum(axis=0)
    dH1 = dZ2 @ W2.T
    dZ1 = dH1 * (Z1 > 0)
    gW1 = X.T @ dZ1 + l2 * W1
    gb1 = dZ1.sum(axis=0)
    return loss, {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2, "W3": gW3, "b3": gb3}


# ── Part B ────────────────────────────────────────────────────────────────


def _fields(frame: pl.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in ("customer_id", TARGET) and frame[c].dtype.is_numeric()]


def discover_features(customers: pl.DataFrame) -> pl.DataFrame:
    cols = _fields(customers)
    z = customers.select([((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(c) for c in cols])
    pcs = np.asarray(DimReductionEngine().reduce(z, algorithm="pca", n_components=3).transformed)
    engine = ClusteringEngine()
    k = engine.sweep_k(z, range(2, 9), algorithm="kmeans", criterion="silhouette").optimal_k
    # silhouette tends to merge the centre segment; check one K above it too
    labels = np.asarray(engine.fit(z, algorithm="kmeans", n_clusters=max(k, 5)).labels)
    onehot = {f"segment_{j}": (labels == j).astype(float) for j in range(int(labels.max()) + 1)}
    return z.with_columns(
        *[pl.Series(f"pc{j + 1}", pcs[:, j]) for j in range(pcs.shape[1])],
        *[pl.Series(name, v) for name, v in onehot.items()],
    )


# ── Part C ────────────────────────────────────────────────────────────────


def fit_and_predict(train: pl.DataFrame, test: pl.DataFrame) -> np.ndarray:
    fields = _fields(test)
    both = pl.concat([train.select(fields), test.select(fields)])
    feats = discover_features(both)
    tr = feats.head(train.height).with_columns(train[TARGET].alias(TARGET))
    te = feats.tail(test.height)
    net = SklearnTrainable(
        estimator=MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3, early_stopping=True,
                                validation_fraction=0.2, max_iter=500, random_state=0),
        target=TARGET,
        metric="accuracy",
    )
    net.fit(tr)
    return net.model.predict_proba(te.select(feats.columns).to_numpy())[:, 1]


if __name__ == "__main__":
    df = load_dev_churn()
    train, test = df.head(2200), df.tail(800)
    from sklearn.metrics import roc_auc_score

    p = fit_and_predict(train, test.drop(TARGET))
    print("dev AUC", round(roc_auc_score(test[TARGET].to_numpy(), p), 3))
