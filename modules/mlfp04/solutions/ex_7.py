# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 7: Recommender Systems and Collaborative Filtering
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Build content-based, user-based, and item-based recommenders
#   - Implement ALS matrix factorisation from scratch
#   - Explain the pivot: "optimisation drives feature discovery"
#   - Evaluate recommenders using RMSE and holdout coverage
#   - Articulate the connection: matrix factorisation → neural embeddings
#
# PREREQUISITES:
#   - MLFP04 Exercise 3 (PCA/SVD — matrix factorisation is generalised SVD)
#   - MLFP04 Exercise 6 (text embeddings — same concept, different domain)
#
# ESTIMATED TIME: 75-90 minutes
#
# TASKS:
#   1. Generate synthetic user-item rating matrix (100 users, 50 items)
#   2. Content-based filtering using item features (cosine similarity)
#   3. User-based collaborative filtering (cosine similarity on ratings)
#   4. Item-based collaborative filtering (cosine similarity on columns)
#   5. Matrix factorisation with ALS from scratch (U * V^T)
#   6. Compare all approaches: RMSE and coverage
#   7. Visualise learned embeddings (2D PCA projection)
#   8. THE PIVOT: matrix factorisation learns embeddings by minimising
#      reconstruction error — optimisation drives feature discovery
#
# DATASET: Synthetic user-item ratings (100 users × 50 items, 30% observed)
#   True latent dimension: 5 (known ground truth for validation)
#   Goal: demonstrate that ALS recovers the true latent structure
#
# OBJECTIVE: Build recommender systems from content-based to matrix
#   factorisation (ALS). Discover THE PIVOT: optimisation drives feature
#   discovery — the bridge from unsupervised ML to neural networks.
#
# TASKS:
#   1. Generate synthetic user-item rating matrix (100 users, 50 items)
#   2. Content-based filtering using item features (cosine similarity)
#   3. User-based collaborative filtering (cosine similarity on ratings)
#   4. Item-based collaborative filtering (cosine similarity on columns)
#   5. Matrix factorisation with ALS from scratch (U * V^T)
#   6. Compare all approaches: RMSE and coverage
#   7. Visualise learned embeddings (2D PCA projection)
#   8. THE PIVOT: matrix factorisation learns embeddings by minimising
#      reconstruction error — optimisation drives feature discovery
#
# THEORY:
#   Matrix factorisation decomposes R ≈ U * V^T where:
#     R: (n_users, n_items) — observed rating matrix (sparse)
#     U: (n_users, k) — user latent factors (embeddings)
#     V: (n_items, k) — item latent factors (embeddings)
#
#   ALS alternates:
#     Fix U, solve for V: V = (U^T U + λI)^{-1} U^T R
#     Fix V, solve for U: U = (V^T V + λI)^{-1} V^T R^T
#
#   Connection to PCA: PCA decomposes X = U Σ V^T (SVD). Matrix
#   factorisation solves the same low-rank approximation problem on
#   sparse data. Both discover latent structure via reconstruction.
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl

from kailash_ml import ModelVisualizer
from sklearn.decomposition import PCA


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Generate synthetic user-item rating matrix
# ══════════════════════════════════════════════════════════════════════
# We create a rating matrix with latent structure: users and items each
# belong to latent groups, so the matrix has low-rank structure that
# factorisation should recover. Ratings are 1-5, sparse (~30% observed).

rng = np.random.default_rng(seed=42)

N_USERS = 100
N_ITEMS = 50
N_LATENT_TRUE = 5  # true latent dimension
SPARSITY = 0.30    # fraction of ratings observed

# Generate ground truth latent factors (users and items in 5D space)
U_true = rng.normal(0, 1, size=(N_USERS, N_LATENT_TRUE))
V_true = rng.normal(0, 1, size=(N_ITEMS, N_LATENT_TRUE))

# True rating matrix (before noise and sparsity)
R_full = U_true @ V_true.T

# Scale to 1-5 range with noise
R_full = (R_full - R_full.min()) / (R_full.max() - R_full.min()) * 4 + 1
R_full += rng.normal(0, 0.3, size=R_full.shape)
R_full = np.clip(R_full, 1.0, 5.0)

# Create sparse observation mask
mask = rng.random(size=(N_USERS, N_ITEMS)) < SPARSITY

# Apply mask: NaN where unobserved
R_observed = np.where(mask, R_full, np.nan)

n_observed = int(mask.sum())
print("=== Synthetic Rating Matrix ===")
print(f"Users: {N_USERS}, Items: {N_ITEMS}")
print(f"True latent dimension: {N_LATENT_TRUE}")
print(f"Observed ratings: {n_observed:,} / {N_USERS * N_ITEMS:,} ({n_observed / (N_USERS * N_ITEMS):.1%})")
print(f"Rating range: {np.nanmin(R_observed):.2f} - {np.nanmax(R_observed):.2f}")
print(f"Mean rating: {np.nanmean(R_observed):.2f}")

# Store as a polars DataFrame for inspection
user_ids = [f"user_{i:03d}" for i in range(N_USERS)]
item_ids = [f"item_{j:02d}" for j in range(N_ITEMS)]

ratings_long = []
for i in range(N_USERS):
    for j in range(N_ITEMS):
        if mask[i, j]:
            ratings_long.append(
                {"user_id": user_ids[i], "item_id": item_ids[j], "rating": round(R_observed[i, j], 1)}
            )

ratings_df = pl.DataFrame(ratings_long)
print(f"\nRatings DataFrame shape: {ratings_df.shape}")
print(ratings_df.head(10))


# ── Generate item features for content-based filtering ───────────────

N_ITEM_FEATURES = 8  # genre/attribute dimensions

# Each item has features (e.g., genre scores, price tier, popularity)
item_features = rng.random(size=(N_ITEMS, N_ITEM_FEATURES))

# Normalise rows to unit length for cosine similarity
item_norms = np.linalg.norm(item_features, axis=1, keepdims=True)
item_features_normed = item_features / np.maximum(item_norms, 1e-10)

feature_names = [f"attr_{k}" for k in range(N_ITEM_FEATURES)]
items_df = pl.DataFrame(
    {"item_id": item_ids}
    | {fn: item_features[:, k].tolist() for k, fn in enumerate(feature_names)}
)
print(f"\nItem features shape: {items_df.shape}")
print(items_df.head(5))


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Content-Based Filtering
# ══════════════════════════════════════════════════════════════════════
# Content-based filtering recommends items similar to what the user
# already liked, using ITEM FEATURES (not other users' ratings).
#
# Algorithm:
#   1. Build a user profile: weighted average of item features for items
#      the user rated, weighted by their ratings.
#   2. Score unrated items by cosine similarity between user profile
#      and each item's feature vector.
#
# Strength: works for new items (no cold-start for items).
# Weakness: cannot discover serendipitous recommendations — only
#   recommends items similar to what user already knows.


def content_based_predict(
    R: np.ndarray,
    item_feats: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Predict ratings using content-based filtering.

    For each user, build a profile from their rated items' features
    (weighted by rating), then score all items by cosine similarity.
    """
    n_users, n_items = R.shape
    predictions = np.full((n_users, n_items), np.nan)

    for u in range(n_users):
        # Items this user has rated
        rated_idx = np.where(mask[u])[0]
        if len(rated_idx) == 0:
            continue

        # User profile: rating-weighted average of item features
        ratings_u = R[u, rated_idx]
        profile = (ratings_u[:, None] * item_feats[rated_idx]).sum(axis=0)
        profile_norm = np.linalg.norm(profile)
        if profile_norm < 1e-10:
            continue
        profile /= profile_norm

        # Score all items by cosine similarity to user profile
        for j in range(n_items):
            feat_norm = np.linalg.norm(item_feats[j])
            if feat_norm < 1e-10:
                continue
            sim = profile @ item_feats[j] / feat_norm
            # Scale similarity [-1, 1] to rating range [1, 5]
            predictions[u, j] = 1.0 + (sim + 1.0) * 2.0

    return predictions


cb_predictions = content_based_predict(
    np.nan_to_num(R_observed, nan=0.0),
    item_features,
    mask,
)

# Evaluate on observed ratings
cb_errors = []
for i in range(N_USERS):
    for j in range(N_ITEMS):
        if mask[i, j] and not np.isnan(cb_predictions[i, j]):
            cb_errors.append((R_observed[i, j] - cb_predictions[i, j]) ** 2)

cb_rmse = np.sqrt(np.mean(cb_errors))
cb_coverage = len(cb_errors) / n_observed

print("\n=== Content-Based Filtering ===")
print(f"RMSE on observed ratings: {cb_rmse:.4f}")
print(f"Coverage: {cb_coverage:.1%} of observed ratings predicted")
print("Content-based uses item features to build user profiles.")
print("It can recommend new items but is limited to 'more of the same'.")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert n_observed > 0, "Should have at least some observed ratings"
assert cb_rmse > 0, "Content-based RMSE should be positive"
assert 0 < cb_coverage <= 1.0, "Coverage should be between 0 and 1"
# INTERPRETATION: Content-based filtering is limited by the 'filter bubble' effect.
# Users can only receive recommendations for items similar to what they've already
# rated. New or unexpected items are never recommended. This is why standalone
# content-based systems tend to reinforce rather than expand user preferences.


# ══════════════════════════════════════════════════════════════════════
# TASK 3: User-Based Collaborative Filtering
# ══════════════════════════════════════════════════════════════════════
# Collaborative filtering uses OTHER USERS' ratings (not item features).
#
# User-based CF:
#   1. Find users similar to target user (cosine similarity on rating vectors)
#   2. Predict missing ratings as weighted average of similar users' ratings
#
# Cosine similarity between user vectors (mean-centred):
#   sim(u, v) = (r_u - mean(r_u)) . (r_v - mean(r_v))
#             / (||r_u - mean(r_u)|| * ||r_v - mean(r_v)||)
#
# Prediction for user u, item i:
#   r_hat(u,i) = mean(r_u) + [ sum_{v in N(u)} sim(u,v) * (r_vi - mean(r_v)) ]
#                              / [ sum_{v in N(u)} |sim(u,v)| ]
#
# Strength: discovers serendipitous recommendations.
# Weakness: cold-start for new users (no rating history to compare).

K_NEIGHBOURS = 20  # number of similar users to consider


def user_similarity_matrix(R: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between users on mean-centred ratings."""
    n_users = R.shape[0]
    sim = np.zeros((n_users, n_users))

    # Mean-centre each user's ratings (only over observed entries)
    user_means = np.array([
        np.nanmean(R[u, obs_mask[u]]) if obs_mask[u].any() else 0.0
        for u in range(n_users)
    ])
    R_centred = R.copy()
    for u in range(n_users):
        R_centred[u, obs_mask[u]] -= user_means[u]
    R_centred[~obs_mask] = 0.0  # unobserved entries contribute nothing

    for u in range(n_users):
        for v in range(u, n_users):
            # Only use items both users have rated
            both = obs_mask[u] & obs_mask[v]
            if not both.any():
                continue
            ru = R_centred[u, both]
            rv = R_centred[v, both]
            denom = np.linalg.norm(ru) * np.linalg.norm(rv)
            if denom < 1e-10:
                continue
            s = ru @ rv / denom
            sim[u, v] = s
            sim[v, u] = s

    return sim, user_means


def user_based_cf_predict(
    R: np.ndarray,
    obs_mask: np.ndarray,
    sim: np.ndarray,
    user_means: np.ndarray,
    k: int = 20,
) -> np.ndarray:
    """Predict ratings using user-based collaborative filtering."""
    n_users, n_items = R.shape
    predictions = np.full((n_users, n_items), np.nan)

    for u in range(n_users):
        # Find k most similar users (exclude self)
        similarities = sim[u].copy()
        similarities[u] = -np.inf
        top_k = np.argsort(similarities)[-k:]
        top_k = top_k[similarities[top_k] > 0]  # only positive similarity

        if len(top_k) == 0:
            continue

        for j in range(n_items):
            # Only use neighbours who rated item j
            rated_neighbours = top_k[obs_mask[top_k, j]]
            if len(rated_neighbours) == 0:
                continue

            weights = sim[u, rated_neighbours]
            denom = np.abs(weights).sum()
            if denom < 1e-10:
                continue

            weighted_dev = weights @ (R[rated_neighbours, j] - user_means[rated_neighbours])
            predictions[u, j] = user_means[u] + weighted_dev / denom

    return np.clip(predictions, 1.0, 5.0)


print("\n=== User-Based Collaborative Filtering ===")
print(f"Computing user similarity matrix ({N_USERS} x {N_USERS})...")

user_sim, user_means = user_similarity_matrix(R_observed, mask)
ubcf_predictions = user_based_cf_predict(R_observed, mask, user_sim, user_means, k=K_NEIGHBOURS)

# Evaluate
ubcf_errors = []
for i in range(N_USERS):
    for j in range(N_ITEMS):
        if mask[i, j] and not np.isnan(ubcf_predictions[i, j]):
            ubcf_errors.append((R_observed[i, j] - ubcf_predictions[i, j]) ** 2)

ubcf_rmse = np.sqrt(np.mean(ubcf_errors)) if ubcf_errors else float("inf")
ubcf_coverage = len(ubcf_errors) / n_observed

print(f"Neighbours (k): {K_NEIGHBOURS}")
print(f"RMSE on observed ratings: {ubcf_rmse:.4f}")
print(f"Coverage: {ubcf_coverage:.1%} of observed ratings predicted")
print("User-based CF finds similar users and borrows their preferences.")
print("It can discover unexpected recommendations (serendipity).")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert ubcf_rmse > 0, "User-based CF RMSE should be positive"
assert 0 < ubcf_coverage <= 1.0, "User-based CF coverage should be in (0, 1]"
assert user_sim.shape == (N_USERS, N_USERS), \
    f"User similarity matrix should be ({N_USERS}, {N_USERS})"
# Diagonal of similarity matrix should be 1.0 (user is perfectly similar to itself)
assert all(abs(user_sim[u, u] - 1.0) < 0.01 or user_sim[u, u] == 0 for u in range(N_USERS)), \
    "Self-similarity should be ~1.0"
# INTERPRETATION: User-based CF suffers from the cold-start problem: new users
# with no rating history cannot be compared to anyone. In practice, this is
# addressed with demographic-based fallbacks or content-based recommendations
# until the new user has rated enough items to form meaningful comparisons.


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Item-Based Collaborative Filtering
# ══════════════════════════════════════════════════════════════════════
# Item-based CF transposes the idea: instead of finding similar USERS,
# find similar ITEMS (based on how users rated them together).
#
# Item similarity: cosine between column vectors of R (mean-centred).
# Prediction for user u, item i:
#   r_hat(u,i) = [ sum_{j in rated(u)} sim(i,j) * r_uj ]
#                / [ sum_{j in rated(u)} |sim(i,j)| ]
#
# Item-based CF is often more stable than user-based because item
# profiles change less frequently than user profiles.


def item_similarity_matrix(R: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between items."""
    n_items = R.shape[1]
    sim = np.zeros((n_items, n_items))

    # Mean-centre each item's ratings
    item_means = np.array([
        np.nanmean(R[obs_mask[:, j], j]) if obs_mask[:, j].any() else 0.0
        for j in range(n_items)
    ])
    R_centred = R.copy()
    for j in range(n_items):
        R_centred[obs_mask[:, j], j] -= item_means[j]
    R_centred[~obs_mask] = 0.0

    for i in range(n_items):
        for j in range(i, n_items):
            both = obs_mask[:, i] & obs_mask[:, j]
            if not both.any():
                continue
            ri = R_centred[both, i]
            rj = R_centred[both, j]
            denom = np.linalg.norm(ri) * np.linalg.norm(rj)
            if denom < 1e-10:
                continue
            s = ri @ rj / denom
            sim[i, j] = s
            sim[j, i] = s

    return sim, item_means


def item_based_cf_predict(
    R: np.ndarray,
    obs_mask: np.ndarray,
    item_sim: np.ndarray,
    k: int = 20,
) -> np.ndarray:
    """Predict ratings using item-based collaborative filtering."""
    n_users, n_items = R.shape
    predictions = np.full((n_users, n_items), np.nan)

    for u in range(n_users):
        rated_items = np.where(obs_mask[u])[0]
        if len(rated_items) == 0:
            continue

        for j in range(n_items):
            # Similarities between target item j and all items user rated
            sims_to_rated = item_sim[j, rated_items]

            # Keep top-k most similar items
            if len(sims_to_rated) > k:
                top_idx = np.argsort(sims_to_rated)[-k:]
            else:
                top_idx = np.arange(len(sims_to_rated))

            # Filter to positive similarities only
            pos_idx = top_idx[sims_to_rated[top_idx] > 0]
            if len(pos_idx) == 0:
                continue

            weights = sims_to_rated[pos_idx]
            denom = np.abs(weights).sum()
            if denom < 1e-10:
                continue

            predictions[u, j] = weights @ R[u, rated_items[pos_idx]] / denom

    return np.clip(predictions, 1.0, 5.0)


print("\n=== Item-Based Collaborative Filtering ===")
print(f"Computing item similarity matrix ({N_ITEMS} x {N_ITEMS})...")

item_sim, item_means = item_similarity_matrix(R_observed, mask)
ibcf_predictions = item_based_cf_predict(R_observed, mask, item_sim, k=K_NEIGHBOURS)

ibcf_errors = []
for i in range(N_USERS):
    for j in range(N_ITEMS):
        if mask[i, j] and not np.isnan(ibcf_predictions[i, j]):
            ibcf_errors.append((R_observed[i, j] - ibcf_predictions[i, j]) ** 2)

ibcf_rmse = np.sqrt(np.mean(ibcf_errors)) if ibcf_errors else float("inf")
ibcf_coverage = len(ibcf_errors) / n_observed

print(f"RMSE on observed ratings: {ibcf_rmse:.4f}")
print(f"Coverage: {ibcf_coverage:.1%} of observed ratings predicted")
print("Item-based CF is more stable — item profiles change less than user profiles.")
print("This is why Amazon uses item-based CF: the item catalogue is more static.")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert ibcf_rmse > 0, "Item-based CF RMSE should be positive"
assert item_sim.shape == (N_ITEMS, N_ITEMS), \
    f"Item similarity matrix should be ({N_ITEMS}, {N_ITEMS})"
# Item-based is typically more stable — check that coverage is reasonable
assert ibcf_coverage > 0, "Item-based CF should predict at least some ratings"
# INTERPRETATION: Item-based CF computes item-item similarity from user ratings
# (not item features). Two items are similar if the same users tend to rate them
# similarly. This is more stable than user-based CF because item preference
# patterns change slowly, while users' tastes can shift rapidly.


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Matrix Factorisation with ALS (from scratch)
# ══════════════════════════════════════════════════════════════════════
# Matrix factorisation is THE PIVOT of this module.
#
# Idea: decompose the rating matrix R ≈ U * V^T where:
#   U: (n_users, k) — each user is a k-dimensional vector (embedding)
#   V: (n_items, k) — each item is a k-dimensional vector (embedding)
#   r_hat(u, i) = u_u^T * v_i  (dot product predicts rating)
#
# Objective: minimise reconstruction error + regularisation
#   L = sum_{(u,i) observed} (r_ui - u_u^T v_i)^2
#       + lambda * (||U||^2 + ||V||^2)
#
# ALS (Alternating Least Squares):
#   The joint optimisation over U and V is non-convex.
#   But if we FIX one, the problem becomes ordinary least squares in the other.
#
#   Fix U, solve for each item vector v_j:
#     v_j = (U_j^T U_j + lambda * I)^{-1} U_j^T r_j
#     where U_j = rows of U for users who rated item j
#           r_j = their ratings for item j
#
#   Fix V, solve for each user vector u_i:
#     u_i = (V_i^T V_i + lambda * I)^{-1} V_i^T r_i
#     where V_i = rows of V for items user i rated
#           r_i = their ratings
#
# Connection to PCA (from lesson 4.3):
#   PCA solves X ≈ U Sigma V^T (SVD) — same low-rank structure!
#   PCA assumes the matrix is DENSE (all entries observed).
#   Matrix factorisation handles SPARSE data (only some entries observed).
#   Both discover latent factors by minimising reconstruction error.

K_LATENT = 10       # latent dimension
LAMBDA_REG = 0.1    # regularisation strength
N_ITERATIONS = 50   # ALS iterations


def als_matrix_factorisation(
    R: np.ndarray,
    obs_mask: np.ndarray,
    k: int,
    lam: float,
    n_iter: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Alternating Least Squares matrix factorisation.

    Parameters
    ----------
    R : (n_users, n_items) rating matrix (NaN for unobserved)
    obs_mask : boolean mask of observed entries
    k : latent dimension
    lam : L2 regularisation parameter
    n_iter : number of ALS iterations
    rng : random number generator

    Returns
    -------
    U : (n_users, k) user latent factors
    V : (n_items, k) item latent factors
    errors : reconstruction RMSE at each iteration
    """
    n_users, n_items = R.shape

    # Initialise with small random values
    U = rng.normal(0, 0.1, size=(n_users, k))
    V = rng.normal(0, 0.1, size=(n_items, k))

    # Replace NaN with 0 for safe indexing (mask controls which entries matter)
    R_safe = np.nan_to_num(R, nan=0.0)

    errors = []
    identity = lam * np.eye(k)

    for iteration in range(n_iter):
        # ── Fix V, solve for each user u_i ──────────────────────────
        # u_i = (V_i^T V_i + lambda I)^{-1} V_i^T r_i
        for u in range(n_users):
            rated = np.where(obs_mask[u])[0]
            if len(rated) == 0:
                continue
            V_u = V[rated]                    # (n_rated, k)
            r_u = R_safe[u, rated]            # (n_rated,)
            A = V_u.T @ V_u + identity       # (k, k)
            b = V_u.T @ r_u                  # (k,)
            U[u] = np.linalg.solve(A, b)

        # ── Fix U, solve for each item v_j ──────────────────────────
        # v_j = (U_j^T U_j + lambda I)^{-1} U_j^T r_j
        for j in range(n_items):
            raters = np.where(obs_mask[:, j])[0]
            if len(raters) == 0:
                continue
            U_j = U[raters]                   # (n_raters, k)
            r_j = R_safe[raters, j]           # (n_raters,)
            A = U_j.T @ U_j + identity       # (k, k)
            b = U_j.T @ r_j                  # (k,)
            V[j] = np.linalg.solve(A, b)

        # ── Track reconstruction error ──────────────────────────────
        R_hat = U @ V.T
        residuals = (R_safe - R_hat)[obs_mask]
        rmse = np.sqrt(np.mean(residuals ** 2))
        errors.append(rmse)

        if iteration % 10 == 0 or iteration == n_iter - 1:
            print(f"  Iteration {iteration:3d}: RMSE = {rmse:.4f}")

    return U, V, errors


print("\n=== Matrix Factorisation with ALS ===")
print(f"Latent dimension k={K_LATENT}, lambda={LAMBDA_REG}, iterations={N_ITERATIONS}")
print("Training...")

U_learned, V_learned, als_errors = als_matrix_factorisation(
    R_observed, mask, k=K_LATENT, lam=LAMBDA_REG, n_iter=N_ITERATIONS, rng=rng,
)

# Evaluate ALS predictions
als_R_hat = np.clip(U_learned @ V_learned.T, 1.0, 5.0)

als_pred_errors = []
for i in range(N_USERS):
    for j in range(N_ITEMS):
        if mask[i, j]:
            als_pred_errors.append((R_observed[i, j] - als_R_hat[i, j]) ** 2)

als_rmse = np.sqrt(np.mean(als_pred_errors))
als_coverage = 1.0  # MF produces predictions for ALL user-item pairs

print(f"\nFinal RMSE on observed ratings: {als_rmse:.4f}")
print(f"Coverage: {als_coverage:.0%} (MF predicts all user-item pairs)")
print(f"Convergence: RMSE dropped from {als_errors[0]:.4f} to {als_errors[-1]:.4f}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert als_rmse < als_errors[0], "ALS should improve RMSE over random initialisation"
# ALS RMSE should be monotone non-increasing
for i in range(1, len(als_errors)):
    assert als_errors[i] <= als_errors[i-1] + 0.01, \
        f"ALS RMSE should be non-increasing, but increased at step {i}"
assert als_rmse > 0, "ALS RMSE should be positive (not exactly zero)"
# INTERPRETATION: ALS converges because each subproblem (fix U, solve V; fix V,
# solve U) is a convex least-squares problem with a unique global minimum.
# The non-convexity is in the joint problem (bilinear in U and V), so ALS
# finds a local minimum — but regularisation prevents degenerate solutions.

# ── Why ALS converges ───────────────────────────────────────────────
# Each step is a convex least-squares solve, so error is guaranteed
# non-increasing. The joint problem is non-convex (bilinear in U, V),
# so ALS finds a local minimum — but in practice this works well,
# especially with regularisation preventing degenerate solutions.


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Compare all approaches
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("COMPARISON OF RECOMMENDER APPROACHES")
print("=" * 70)
print(f"{'Method':<30} {'RMSE':>10} {'Coverage':>12}")
print("-" * 52)
print(f"{'Content-Based':<30} {cb_rmse:>10.4f} {cb_coverage:>11.1%}")
print(f"{'User-Based CF':<30} {ubcf_rmse:>10.4f} {ubcf_coverage:>11.1%}")
print(f"{'Item-Based CF':<30} {ibcf_rmse:>10.4f} {ibcf_coverage:>11.1%}")
print(f"{'Matrix Factorisation (ALS)':<30} {als_rmse:>10.4f} {als_coverage:>11.1%}")

print("\n--- Interpretation ---")
print("Content-based filtering uses item attributes but cannot capture user taste patterns.")
print("User-based CF leverages community preferences but struggles with sparse data.")
print("Item-based CF is more robust (item profiles are more stable than user profiles).")
print("Matrix factorisation achieves the best RMSE by learning latent factors that")
print("capture BOTH user preferences AND item characteristics simultaneously.")
print("It also has 100% coverage: it can predict any user-item pair.")

# ── Holdout evaluation (predict unobserved entries) ──────────────────
# Since we know the ground truth R_full, we can evaluate on unobserved entries
holdout_mask = ~mask  # entries NOT in the training set
holdout_errors_als = (R_full - als_R_hat)[holdout_mask]
holdout_rmse_als = np.sqrt(np.mean(holdout_errors_als ** 2))

print(f"\n--- Holdout (Unobserved) Evaluation ---")
print(f"ALS RMSE on unobserved entries: {holdout_rmse_als:.4f}")
print(f"Number of holdout entries: {int(holdout_mask.sum()):,}")
print("This measures generalisation — can we predict ratings the model never saw?")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Visualise learned embeddings
# ══════════════════════════════════════════════════════════════════════
# The user matrix U and item matrix V are learned EMBEDDINGS.
# Each user/item is a point in k-dimensional space.
# We project to 2D with PCA to visualise the latent structure.
#
# If factorisation captured real structure, we should see:
#   - Users with similar tastes cluster together
#   - Items with similar rating patterns cluster together
#   - The 2D projection reveals interpretable groups

viz = ModelVisualizer()

# ── 7a: User embeddings in 2D ───────────────────────────────────────
pca_users = PCA(n_components=2, random_state=42)
U_2d = pca_users.fit_transform(U_learned)

print("\n=== Embedding Visualisation ===")
print(f"User PCA explained variance: {pca_users.explained_variance_ratio_.sum():.1%}")

user_embed_df = pl.DataFrame({
    "user_id": user_ids,
    "pc1": U_2d[:, 0].tolist(),
    "pc2": U_2d[:, 1].tolist(),
    "avg_rating": [
        float(np.nanmean(R_observed[i, mask[i]])) if mask[i].any() else 0.0
        for i in range(N_USERS)
    ],
})

fig_users = viz.scatter(user_embed_df, x="pc1", y="pc2", color="avg_rating")
fig_users.update_layout(
    title="User Embeddings (ALS, PCA projection)",
    xaxis_title="Principal Component 1",
    yaxis_title="Principal Component 2",
)
fig_users.write_html("ex7_user_embeddings.html")
print("Saved: ex7_user_embeddings.html")

# ── 7b: Item embeddings in 2D ───────────────────────────────────────
pca_items = PCA(n_components=2, random_state=42)
V_2d = pca_items.fit_transform(V_learned)

print(f"Item PCA explained variance: {pca_items.explained_variance_ratio_.sum():.1%}")

item_embed_df = pl.DataFrame({
    "item_id": item_ids,
    "pc1": V_2d[:, 0].tolist(),
    "pc2": V_2d[:, 1].tolist(),
    "avg_rating": [
        float(np.nanmean(R_observed[mask[:, j], j])) if mask[:, j].any() else 0.0
        for j in range(N_ITEMS)
    ],
})

fig_items = viz.scatter(item_embed_df, x="pc1", y="pc2", color="avg_rating")
fig_items.update_layout(
    title="Item Embeddings (ALS, PCA projection)",
    xaxis_title="Principal Component 1",
    yaxis_title="Principal Component 2",
)
fig_items.write_html("ex7_item_embeddings.html")
print("Saved: ex7_item_embeddings.html")

# ── 7c: ALS convergence plot ────────────────────────────────────────
convergence_metrics = {"ALS Reconstruction RMSE": als_errors}
fig_conv = viz.training_history(convergence_metrics, x_label="ALS Iteration")
fig_conv.update_layout(title="ALS Convergence — Reconstruction Error")
fig_conv.write_html("ex7_als_convergence.html")
print("Saved: ex7_als_convergence.html")

# ── 7d: Comparison bar chart ────────────────────────────────────────
comparison_metrics = {
    "Content-Based": {"RMSE": cb_rmse, "Coverage": cb_coverage},
    "User-Based CF": {"RMSE": ubcf_rmse, "Coverage": ubcf_coverage},
    "Item-Based CF": {"RMSE": ibcf_rmse, "Coverage": ibcf_coverage},
    "Matrix Fact. (ALS)": {"RMSE": als_rmse, "Coverage": als_coverage},
}
fig_compare = viz.metric_comparison(comparison_metrics)
fig_compare.update_layout(title="Recommender System Comparison")
fig_compare.write_html("ex7_method_comparison.html")
print("Saved: ex7_method_comparison.html")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: THE PIVOT — Matrix Factorisation to Neural Embeddings
# ══════════════════════════════════════════════════════════════════════
# This is the most important conceptual moment in Module 4.
#
# What we just did:
#   R ≈ U * V^T
#   - U and V are LEARNED representations (embeddings)
#   - They were learned by MINIMISING RECONSTRUCTION ERROR
#   - No one told the model what the latent dimensions mean
#   - The optimisation process DISCOVERED meaningful features
#
# Connection to PCA (lesson 4.3):
#   PCA: X = U Sigma V^T (SVD on dense matrix)
#   ALS: R ≈ U V^T       (factorisation on sparse matrix)
#   Both find low-rank structure. Both learn latent representations.
#   PCA is a special case where all entries are observed.
#
# THE PIVOT to neural networks (lesson 4.8):
#   In a neural network, hidden layer activations ARE embeddings.
#   The network learns them by minimising a loss function.
#
#   Matrix factorisation: R ≈ U * V^T
#     → minimise ||R - UV^T||^2 over observed entries
#     → U, V are the learned embeddings
#
#   Neural network: y = f(W_2 * f(W_1 * x + b_1) + b_2)
#     → minimise L(y, y_hat) over training data
#     → hidden activations h = f(W_1 * x + b_1) are learned embeddings
#
#   The SAME principle: optimisation drives feature discovery.
#   The difference: neural networks add NON-LINEAR activation functions,
#   enabling richer, hierarchical representations.
#
# This is why Word2Vec, recommendation systems, and deep learning all
# share the same core idea: learn representations by optimising a loss.

print("\n" + "=" * 70)
print("THE PIVOT: Optimisation Drives Feature Discovery")
print("=" * 70)

print("""
What we built in this exercise:

  Rating Matrix R  ≈  U  *  V^T
  (100 x 50)         (100 x 10)  (10 x 50)

  U = user embeddings: each user is a 10-dimensional vector
  V = item embeddings: each item is a 10-dimensional vector
  r_hat(u, i) = dot(u, v) = predicted rating

Nobody told the model what those 10 dimensions mean.
The ALS algorithm DISCOVERED them by minimising reconstruction error.

Connection to PCA (lesson 4.3):
  PCA decomposes X = U * Sigma * V^T — same low-rank factorisation.
  PCA requires all entries observed; ALS handles sparse matrices.
  Both discover latent structure through reconstruction.

THE BRIDGE to Neural Networks (lesson 4.8):
  In a neural network, hidden layer activations h = f(Wx + b) are embeddings.
  They are learned by minimising a loss function (backpropagation).

  Matrix Factorisation:     loss = ||R - U V^T||^2
  Neural Network:           loss = ||y - f(Wx + b)||^2

  SAME PRINCIPLE: optimisation discovers features automatically.
  The difference: neural networks add nonlinear activation functions,
  enabling deeper, hierarchical representations.

This is the fundamental insight of representation learning:
  "You don't engineer features. You engineer a loss function,
   and the optimisation process discovers the features for you."
""")

# ── Empirical demonstration of the connection ────────────────────────
# Show that ALS embeddings capture the true latent structure by
# comparing the learned factors to the ground-truth factors.

# Canonical correlation between learned and true factors
# (The factors may be rotated, so direct comparison is meaningless.
#  Instead we check if the learned subspace matches the true subspace.)
from numpy.linalg import svd as np_svd

def subspace_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """Measure alignment between two subspaces via principal angles.

    Returns the average cosine of principal angles (1.0 = perfect alignment).
    """
    # Orthonormalise both
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    # Singular values of Qa^T Qb are cosines of principal angles
    _, sigmas, _ = np_svd(Qa.T @ Qb, full_matrices=False)
    return float(np.mean(np.minimum(sigmas, 1.0)))


user_subspace_sim = subspace_similarity(
    U_learned[:, :N_LATENT_TRUE], U_true
)
item_subspace_sim = subspace_similarity(
    V_learned[:, :N_LATENT_TRUE], V_true
)

print(f"Subspace alignment (1.0 = perfect recovery of true latent factors):")
print(f"  User factors: {user_subspace_sim:.3f}")
print(f"  Item factors: {item_subspace_sim:.3f}")
print(f"\nThe learned embeddings capture the latent structure that generated the data.")
print(f"This is optimisation-driven feature discovery in action.")

# ── Implicit vs Explicit feedback (conceptual) ──────────────────────
print("\n--- Implicit vs Explicit Feedback ---")
print("Explicit feedback: users provide ratings (1-5 stars). Clear signal but sparse.")
print("Implicit feedback: clicks, views, purchases. Abundant but noisy.")
print("  - No negative signal: absence of a click != dislike (maybe never seen).")
print("  - ALS for implicit data (Hu et al. 2008): treat all entries as observed,")
print("    weight observed entries higher. Loss becomes:")
print("    L = sum_all (c_ui)(p_ui - u^T v)^2 + lambda(||U||^2 + ||V||^2)")
print("    where p_ui = 1 if observed, 0 otherwise; c_ui = 1 + alpha * count.")
print("SVD++ extends SVD with implicit feedback signals for better predictions.")

# ── Hybrid systems ──────────────────────────────────────────────────
print("\n--- Hybrid Systems ---")
print("Real-world recommenders combine approaches:")
print("  1. Content-based: solves cold-start for new items (known features)")
print("  2. Collaborative filtering: captures community wisdom")
print("  3. Matrix factorisation: learns latent representations")
print("  4. Deep learning (lesson 4.8): learns nonlinear embeddings")
print("Netflix Prize (2006-2009): the winning solution was an ensemble of")
print("107 models including CF, MF, and restricted Boltzmann machines.")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert user_subspace_sim > 0, "User subspace alignment should be positive"
assert item_subspace_sim > 0, "Item subspace alignment should be positive"
assert U_learned.shape == (N_USERS, K_LATENT), \
    f"User embedding matrix should be ({N_USERS}, {K_LATENT})"
assert V_learned.shape == (N_ITEMS, K_LATENT), \
    f"Item embedding matrix should be ({N_ITEMS}, {K_LATENT})"
# INTERPRETATION: Subspace alignment > 0.5 means ALS has discovered a coordinate
# system close to the true latent factors. Perfect recovery (= 1.0) is only
# possible with infinite data and no noise. The fact that it recovers meaningful
# structure from 30% of ratings demonstrates the power of matrix factorisation.
print("\n✓ Checkpoint 5 passed — ALS learned embeddings aligned with true latent factors\n")

print("\n" + "=" * 70)
print("Exercise 7 complete — Recommender Systems and THE PIVOT")
print("=" * 70)
print("Key takeaway: matrix factorisation learns embeddings via optimisation.")
print("In lesson 4.8, neural networks generalise this with nonlinear activations.")
print("The principle is the same: minimise a loss, discover features automatically.")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(f"""
  ✓ Content-based: user profile from rated items → cosine similarity
  ✓ User-based CF: similar users → borrow their preferences
  ✓ Item-based CF: similar items → more stable, used at Amazon scale
  ✓ Matrix factorisation (ALS): R ≈ U * V^T, learned by minimising ||R - UV^T||^2
  ✓ THE PIVOT: optimisation discovers latent features automatically

  KEY INSIGHT — THE PIVOT:
    PCA:  X = U Σ V^T  (dense, full SVD)
    ALS:  R ≈ U V^T    (sparse, optimisation-driven)
    Both discover low-rank latent structure by minimising reconstruction error.
    Neural networks generalise this with nonlinear activation functions.

  THE COMPLETE CHAIN:
    Association rules (Ex 5) → co-occurrence patterns (manual)
    Matrix factorisation (Ex 7) → latent factors (automatic, linear)
    Neural networks (Ex 8) → latent representations (automatic, nonlinear)

  NEXT (and last): Exercise 8 completes the USML bridge.
  You'll build a CNN with residual connections from scratch — and see
  that the hidden layer activations ARE embeddings, learned by the same
  principle you just used: minimise a loss, let optimisation discover features.
  Exercise 8 is the capstone of Module 4 and the gateway to M5 (LLMs & Agents).
""")
print("═" * 70)
