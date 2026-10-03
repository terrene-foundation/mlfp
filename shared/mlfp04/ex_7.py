# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP04 Exercise 7 — Recommender Systems.

Scenario: a fictional Singapore electronics marketplace.
  - 300 users x 120 SKUs, explicit 1-5 star ratings
  - Ratings follow a known generative model so every technique can be
    checked against the truth:
        rating = global mean + user bias + item bias
                 + (user taste vector . item trait vector) + noise
    with N_LATENT_TRUE = 5 hidden taste dimensions.
  - 30% of (user, item) pairs are observed
  - 30% of observed ratings are held out for evaluation (~10 per user, so
    precision@5 and MAP actually have to RANK items)
  - N_COLD_ITEMS brand-new SKUs have ALL their ratings in the holdout set:
    they are cold-start items that no collaborative method has seen
  - A validation split (VAL_FRAC of training ratings) is carved out for
    tuning hybrid blend weights — the holdout set is never used for tuning
  - Item content features = a noisy linear view of each SKU's hidden traits
    and quality (think price tier, category, spec sheet)

Contains: synthetic rating matrix generation, baseline predictors, evaluation
metrics (RMSE, precision@k, MAP), shared constants, and output-dir setup.

Technique-specific code (content-based profile, user/item CF, ALS, hybrid
blending) lives in the per-technique files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from shared.kailash_helpers import setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ════════════════════════════════════════════════════════════════════════

setup_environment()

OUTPUT_DIR = Path("outputs") / "ex7_recommenders"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# CONSTANTS — synthetic SG e-commerce dataset
# ════════════════════════════════════════════════════════════════════════

N_USERS = 300
N_ITEMS = 120
N_LATENT_TRUE = 5
N_ITEM_FEATURES = 8
SPARSITY = 0.30  # fraction of (user, item) pairs that carry a rating
HOLDOUT_FRAC = 0.30  # fraction of observed ratings held out for evaluation
VAL_FRAC = 0.20  # fraction of TRAINING ratings reserved for blend tuning
N_COLD_ITEMS = 10  # brand-new SKUs whose every rating is in the holdout
RNG_SEED = 42

RATING_MIN = 1.0
RATING_MAX = 5.0
RELEVANCE_THRESHOLD = 3.5  # ratings >= 3.5 are "relevant" for ranking metrics

# Generative-model scales (documented so students can reason about them)
_GLOBAL_MEAN = 3.2
_USER_BIAS_SD = 0.35
_ITEM_BIAS_SD = 0.45
_TASTE_SCALE = 0.9
_NOISE_SD = 0.35
_FEATURE_NOISE_SD = 0.5


# ════════════════════════════════════════════════════════════════════════
# SYNTHETIC RATING MATRIX — shared across every technique
# ════════════════════════════════════════════════════════════════════════


def build_rating_dataset(
    seed: int = RNG_SEED,
) -> dict:
    """Generate the shared synthetic e-commerce rating matrix.

    Returns a dict with:
      R_observed       — (N_USERS, N_ITEMS) with NaN where not rated
      R_train          — R_observed with holdout entries set to NaN
      mask             — boolean observed mask
      train_mask       — boolean training-only mask (holdout removed)
      holdout_mask     — boolean holdout (test) mask
      fit_mask         — train_mask minus the validation ratings
      val_mask         — validation ratings (subset of train_mask), used ONLY
                         for tuning blend weights in 05_hybrid_evaluation.py
      cold_items       — boolean (N_ITEMS,) — SKUs with zero training ratings
      U_true, V_true   — ground-truth latent taste / trait factors
      item_features    — (N_ITEMS, N_ITEM_FEATURES) content-feature matrix
      user_ids, item_ids — stable string IDs
      ratings_df       — long-format polars DataFrame of observed ratings
      rng              — the generator, for downstream reproducible draws
    """
    rng = np.random.default_rng(seed=seed)

    U_true = rng.normal(0, 1, size=(N_USERS, N_LATENT_TRUE))
    V_true = rng.normal(0, 1, size=(N_ITEMS, N_LATENT_TRUE))
    user_bias = rng.normal(0, _USER_BIAS_SD, size=N_USERS)
    item_bias = rng.normal(0, _ITEM_BIAS_SD, size=N_ITEMS)

    taste = U_true @ V_true.T / np.sqrt(N_LATENT_TRUE)  # ~N(0, 1) per pair
    R_full = (
        _GLOBAL_MEAN
        + user_bias[:, None]
        + item_bias[None, :]
        + _TASTE_SCALE * taste
        + rng.normal(0, _NOISE_SD, size=(N_USERS, N_ITEMS))
    )
    R_full = np.clip(R_full, RATING_MIN, RATING_MAX)

    mask = rng.random(size=(N_USERS, N_ITEMS)) < SPARSITY
    R_observed = np.where(mask, R_full, np.nan)

    cold_items = np.zeros(N_ITEMS, dtype=bool)
    cold_items[rng.choice(N_ITEMS, size=N_COLD_ITEMS, replace=False)] = True

    holdout_mask = mask & (rng.random(size=(N_USERS, N_ITEMS)) < HOLDOUT_FRAC)
    holdout_mask |= mask & cold_items[None, :]  # cold SKUs: every rating is test
    train_mask = mask & ~holdout_mask
    R_train = np.where(train_mask, R_observed, np.nan)

    val_mask = train_mask & (rng.random(size=(N_USERS, N_ITEMS)) < VAL_FRAC)
    fit_mask = train_mask & ~val_mask

    # Content features: a noisy linear view of each item's hidden traits
    # AND its quality (item bias). Real catalogues behave the same way —
    # spec sheets and price tiers correlate with, but do not equal, what
    # drives ratings.
    hidden = np.hstack(
        [V_true / np.sqrt(N_LATENT_TRUE), (item_bias / _ITEM_BIAS_SD)[:, None]]
    )
    W_feat = rng.normal(0, 1, size=(N_LATENT_TRUE + 1, N_ITEM_FEATURES))
    item_features = hidden @ W_feat / np.sqrt(N_LATENT_TRUE + 1) + rng.normal(
        0, _FEATURE_NOISE_SD, size=(N_ITEMS, N_ITEM_FEATURES)
    )

    user_ids = [f"sg_user_{i:03d}" for i in range(N_USERS)]
    item_ids = [f"sku_{j:03d}" for j in range(N_ITEMS)]

    rows = []
    for i in range(N_USERS):
        for j in range(N_ITEMS):
            if mask[i, j]:
                rows.append(
                    {
                        "user_id": user_ids[i],
                        "item_id": item_ids[j],
                        "rating": round(float(R_observed[i, j]), 1),
                        "in_holdout": bool(holdout_mask[i, j]),
                    }
                )
    ratings_df = pl.DataFrame(rows)

    return {
        "R_observed": R_observed,
        "R_train": R_train,
        "mask": mask,
        "train_mask": train_mask,
        "holdout_mask": holdout_mask,
        "fit_mask": fit_mask,
        "val_mask": val_mask,
        "cold_items": cold_items,
        "U_true": U_true,
        "V_true": V_true,
        "item_features": item_features,
        "user_ids": user_ids,
        "item_ids": item_ids,
        "ratings_df": ratings_df,
        "rng": rng,
    }


# ════════════════════════════════════════════════════════════════════════
# BASELINES — every recommender must beat these to be worth deploying
# ════════════════════════════════════════════════════════════════════════


def baseline_predictions(
    R: np.ndarray, obs_mask: np.ndarray, seed: int = RNG_SEED
) -> dict[str, np.ndarray]:
    """Return three no-skill reference predictors fitted on ``R[obs_mask]``.

    - "Global mean":  every pair gets the mean training rating (RMSE floor)
    - "Item mean":    each SKU's mean training rating (popularity), falling
                      back to the global mean for SKUs with no training data
    - "Random":       uniform scores in [1, 5] — the chance line for P@k/MAP
    """
    global_mean = float(R[obs_mask].mean())
    n_users, n_items = R.shape
    counts = obs_mask.sum(axis=0)
    sums = np.where(obs_mask, R, 0.0).sum(axis=0)
    item_mean = np.where(counts > 0, sums / np.maximum(counts, 1), global_mean)
    rng = np.random.default_rng(seed)
    return {
        "Global mean": np.full((n_users, n_items), global_mean),
        "Item mean": np.tile(item_mean, (n_users, 1)),
        "Random": rng.uniform(RATING_MIN, RATING_MAX, size=(n_users, n_items)),
    }


# ════════════════════════════════════════════════════════════════════════
# EVALUATION METRICS — shared across every technique
# ════════════════════════════════════════════════════════════════════════


def holdout_rmse(
    predictions: np.ndarray,
    R_true: np.ndarray,
    holdout_mask: np.ndarray,
) -> tuple[float, float]:
    """Return (RMSE, coverage) on the holdout mask.

    Coverage = fraction of holdout pairs where the method produced a
    non-NaN prediction. Methods that can't predict cold pairs have
    coverage < 1.0; their RMSE is computed only on the pairs they cover.
    """
    n_holdout = int(holdout_mask.sum())
    covered = holdout_mask & ~np.isnan(predictions)
    if not covered.any():
        return float("inf"), 0.0
    errors = (R_true[covered] - predictions[covered]) ** 2
    rmse = float(np.sqrt(np.mean(errors)))
    coverage = float(covered.sum()) / max(n_holdout, 1)
    return rmse, coverage


def precision_at_k(
    predictions: np.ndarray,
    R_true: np.ndarray,
    holdout_mask: np.ndarray,
    k: int = 5,
    threshold: float = RELEVANCE_THRESHOLD,
) -> float:
    """Precision@k averaged across users.

    For each user, rank holdout items by predicted score, take the top-k,
    and compute what fraction of those have true rating >= threshold.
    Items a method cannot score (NaN) are never recommended.
    """
    precisions = []
    for u in range(predictions.shape[0]):
        holdout_items = np.where(holdout_mask[u])[0]
        if len(holdout_items) == 0:
            continue
        relevant = {j for j in holdout_items if R_true[u, j] >= threshold}
        if not relevant:
            continue
        scored = [
            (j, predictions[u, j])
            for j in holdout_items
            if not np.isnan(predictions[u, j])
        ]
        scored.sort(key=lambda x: -x[1])
        top = [j for j, _ in scored[:k]]
        if not top:
            precisions.append(0.0)
            continue
        precisions.append(len(set(top) & relevant) / len(top))
    return float(np.mean(precisions)) if precisions else 0.0


def mean_average_precision(
    predictions: np.ndarray,
    R_true: np.ndarray,
    holdout_mask: np.ndarray,
    threshold: float = RELEVANCE_THRESHOLD,
) -> float:
    """Mean Average Precision across users on the holdout set.

    A relevant item the method cannot score counts as never retrieved, so
    low coverage lowers MAP.
    """
    aps = []
    for u in range(predictions.shape[0]):
        holdout_items = np.where(holdout_mask[u])[0]
        if len(holdout_items) == 0:
            continue
        relevant = {j for j in holdout_items if R_true[u, j] >= threshold}
        if not relevant:
            continue
        scored = [
            (j, predictions[u, j])
            for j in holdout_items
            if not np.isnan(predictions[u, j])
        ]
        scored.sort(key=lambda x: -x[1])

        hits = 0
        sum_precision = 0.0
        for rank, (j, _) in enumerate(scored, 1):
            if j in relevant:
                hits += 1
                sum_precision += hits / rank
        aps.append(sum_precision / len(relevant))
    return float(np.mean(aps)) if aps else 0.0


def evaluate_method(
    preds: np.ndarray, R_true: np.ndarray, holdout_mask: np.ndarray
) -> dict:
    """Return {"RMSE", "Coverage", "P@5", "MAP"} for one prediction matrix."""
    rmse, cov = holdout_rmse(preds, R_true, holdout_mask)
    return {
        "RMSE": rmse,
        "Coverage": cov,
        "P@5": precision_at_k(preds, R_true, holdout_mask, k=5),
        "MAP": mean_average_precision(preds, R_true, holdout_mask),
    }


def print_method_scores(
    name: str, preds: np.ndarray, R_true: np.ndarray, holdout_mask: np.ndarray
) -> dict:
    """Print a single-line scorecard and return the metrics dict."""
    m = evaluate_method(preds, R_true, holdout_mask)
    print(
        f"  {name:<22} RMSE={m['RMSE']:6.4f}  coverage={m['Coverage']:6.1%}  "
        f"P@5={m['P@5']:6.4f}  MAP={m['MAP']:6.4f}"
    )
    return m


def print_baselines(
    R_train: np.ndarray,
    train_mask: np.ndarray,
    R_true: np.ndarray,
    holdout_mask: np.ndarray,
) -> dict[str, dict]:
    """Score the three no-skill baselines on the holdout and print them."""
    print("  Baselines (a recommender must beat these):")
    return {
        name: print_method_scores(name, preds, R_true, holdout_mask)
        for name, preds in baseline_predictions(R_train, train_mask).items()
    }


def print_warm_comparison(
    name: str,
    preds: np.ndarray,
    R_train: np.ndarray,
    train_mask: np.ndarray,
    R_true: np.ndarray,
    holdout_mask: np.ndarray,
    cold_items: np.ndarray,
) -> tuple[float, float]:
    """Compare a CF-style method with the item-mean baseline on WARM SKUs.

    Collaborative methods cannot score cold-start SKUs, and MAP counts an
    unscored relevant item as a miss — so on the full holdout they can lose
    to a baseline that covers everything. Restricting to warm SKUs isolates
    ranking skill from coverage. Returns (method MAP, item-mean MAP).
    """
    warm = holdout_mask & ~cold_items[None, :]
    method_map = mean_average_precision(preds, R_true, warm)
    item_mean = baseline_predictions(R_train, train_mask)["Item mean"]
    base_map = mean_average_precision(item_mean, R_true, warm)
    verdict = "beats" if method_map > base_map else "does NOT beat"
    print(
        f"  Warm SKUs only: {name} MAP={method_map:.4f} vs item-mean "
        f"MAP={base_map:.4f} -> {name} {verdict} the popularity baseline "
        "on items it can see."
    )
    return method_map, base_map


# ════════════════════════════════════════════════════════════════════════
# VISUAL HELPERS
# ════════════════════════════════════════════════════════════════════════


def save_html(fig, filename: str) -> Path:
    """Write a plotly fig into OUTPUT_DIR and return the path."""
    path = OUTPUT_DIR / filename
    fig.write_html(str(path))
    print(f"  saved: {path}")
    return path
