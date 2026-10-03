# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 7.1: Content-Based Filtering
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build a user profile from the items they already rated
#   - Score unseen items by cosine similarity to that profile
#   - Understand why content-based filtering handles cold-start items
#   - See where it FAILS: cold-start users, narrow interest tunnels
#
# PREREQUISITES: MLFP04 Exercise 3 (cosine similarity, vector embeddings)
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — item features + user profile intuition
#   2. Build — content-based prediction function
#   3. Train — no training loop; profiles are computed at inference
#   4. Visualise — predicted vs observed ratings scatter
#   5. Apply — Singapore e-commerce new-SKU launch
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
from kailash_ml import ModelVisualizer

from shared.mlfp04.ex_7 import (
    N_ITEMS,
    N_USERS,
    build_rating_dataset,
    holdout_rmse,
    print_baselines,
    print_method_scores,
    save_html,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — Why content-based works
# ════════════════════════════════════════════════════════════════════════
# Every item has a feature vector (price tier, category, brand, attributes).
# Every user has a history of ratings. The user's "taste profile" is a
# weighted sum of the (centred) features of the items they rated, where the
# weight is how far each rating sits from the user's OWN average:
#
#   profile_u = sum_i (r_ui - mean_u) * features_i
#
# Items rated above the user's average pull the profile towards their
# features; items rated below it push the profile AWAY (negative weight).
# Using raw 1-5 ratings as weights would not do this — every weight would
# be positive, so even a 1-star item would pull the profile towards itself.
#
# To score a new item, compute cosine similarity between the user profile
# and the item's feature vector. Items that "look like" what the user
# liked more than usual get scores above the user's mean; items that look
# like their disappointments get scores below it.
#
# STRENGTHS:
#   + works for cold-start items (new SKU, zero ratings) — features exist
#   + explainable: "we recommended this because you liked similar items"
#   + no cross-user leakage (each user is predicted independently)
#
# WEAKNESSES:
#   - cold-start user: no ratings = no profile = no recommendations
#   - filter bubble: only ever recommends what the user already likes
#   - feature quality caps accuracy: bad features = bad recommendations


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the content-based predictor
# ════════════════════════════════════════════════════════════════════════


def content_based_predict(
    R: np.ndarray,
    item_feats: np.ndarray,
    obs_mask: np.ndarray,
) -> np.ndarray:
    """Predict ratings via cosine similarity to the user's taste profile.

    For each user:
      1. Centre the ratings on the user's own mean (deviations)
      2. Build profile = sum_i(deviation_i * centred_features_i)
      3. Normalise profile to unit length
      4. Score each item as mean_u + 2 * std_u * cos(profile, feats_j)
         (cosine in [-1, 1] -> up to +/- two of the user's own std devs)
    """
    n_users, n_items = R.shape
    predictions = np.full((n_users, n_items), np.nan)
    # Centre each feature so "average-looking" items sit at the origin
    feats_c = item_feats - item_feats.mean(axis=0)

    for u in range(n_users):
        rated_idx = np.where(obs_mask[u])[0]
        if len(rated_idx) == 0:
            continue

        ratings_u = R[u, rated_idx]
        mean_u = float(ratings_u.mean())
        std_u = float(ratings_u.std()) + 1e-9
        deviations = ratings_u - mean_u
        profile = (deviations[:, None] * feats_c[rated_idx]).sum(axis=0)
        profile_norm = np.linalg.norm(profile)
        if profile_norm < 1e-10:
            continue
        profile /= profile_norm

        for j in range(n_items):
            feat_norm = np.linalg.norm(feats_c[j])
            if feat_norm < 1e-10:
                continue
            sim = profile @ feats_c[j] / feat_norm
            predictions[u, j] = mean_u + 2.0 * std_u * sim

    return np.clip(predictions, 1.0, 5.0)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — "TRAIN" (there is no training loop)
# ════════════════════════════════════════════════════════════════════════
# Content-based filtering is non-parametric at the user level — the user
# profile is recomputed at inference from whatever ratings they have. There
# is nothing to train. This is both the strength (instant adaptation to new
# ratings) and the weakness (no cross-user generalisation).

print("\n" + "=" * 70)
print("  Content-Based Filtering on SG E-commerce Ratings")
print("=" * 70)

data = build_rating_dataset()
R_train = data["R_train"]
R_observed = data["R_observed"]
train_mask = data["train_mask"]
holdout_mask = data["holdout_mask"]
item_features = data["item_features"]
cold_items = data["cold_items"]  # brand-new SKUs with zero training ratings

cb_predictions = content_based_predict(R_train, item_features, train_mask)


# ── Checkpoint ──────────────────────────────────────────────────────────
cb_rmse, cb_coverage = holdout_rmse(cb_predictions, R_observed, holdout_mask)
assert cb_predictions.shape == (N_USERS, N_ITEMS), "Prediction matrix shape"
assert cb_rmse > 0, "Content-based RMSE should be positive"
print(
    "\n[ok] Checkpoint passed — content-based predictor produced holdout "
    f"RMSE={cb_rmse:.4f}, coverage={cb_coverage:.1%}\n"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE predicted vs observed
# ════════════════════════════════════════════════════════════════════════
# A calibration scatter plot tells you whether the model is systematically
# biased (all predictions hug the mean?) or well-calibrated (points lie
# along the y=x diagonal).

viz = ModelVisualizer()

pairs = []
for i in range(N_USERS):
    for j in range(N_ITEMS):
        if holdout_mask[i, j] and not np.isnan(cb_predictions[i, j]):
            pairs.append(
                {
                    "sku_type": "cold-start SKU" if cold_items[j] else "warm SKU",
                    "true": float(R_observed[i, j]),
                    "pred": float(cb_predictions[i, j]),
                }
            )

pair_df = pl.DataFrame(pairs)
fig = viz.scatter(pair_df, x="true", y="pred", color="sku_type")
fig.update_layout(
    title="Content-Based: Predicted vs Observed Rating (holdout)",
    xaxis_title="Observed rating (1-5)",
    yaxis_title="Predicted rating (1-5)",
)
save_html(fig, "01_content_based_scatter.html")

print_baselines(R_train, train_mask, R_observed, holdout_mask)
print_method_scores("Content-Based", cb_predictions, R_observed, holdout_mask)

# Cold-start check: the brand-new SKUs have NO training ratings at all.
cold_holdout = holdout_mask & cold_items[None, :]
cold_rmse, cold_cov = holdout_rmse(cb_predictions, R_observed, cold_holdout)
cold_base = np.full_like(R_observed, float(R_train[train_mask].mean()))
cold_base_rmse, _ = holdout_rmse(cold_base, R_observed, cold_holdout)
print(
    f"\nCold-start SKUs ({int(cold_items.sum())} items, {int(cold_holdout.sum())} "
    f"held-out ratings): content-based RMSE={cold_rmse:.4f} "
    f"(coverage {cold_cov:.0%}) vs global-mean RMSE={cold_base_rmse:.4f}"
)
if cold_rmse < cold_base_rmse:
    print(
        "  -> Content features let us score SKUs nobody has rated yet, and "
        "the scores are better than guessing the average."
    )
else:
    print(
        "  -> On this draw the content features did NOT beat the global mean "
        "on cold SKUs — features this noisy carry little taste signal."
    )


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore E-commerce New-SKU Launch
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore consumer-electronics retailer launches 200 new
# SKUs per week. None have any ratings yet —
# they're pure cold-start. The CF methods below (user-CF, item-CF, ALS) all
# need co-rating history, which doesn't exist for a brand-new SKU.
#
# Content-based filtering is the ONLY method in this exercise that can
# recommend a SKU on day 0. As long as the new SKU has a feature vector
# (price tier, category, brand, screen size, etc.) the user profile can
# score it.
#
# BUSINESS IMPACT (illustrative assumptions, not measured figures): suppose
# a new SKU earns most of its lifetime revenue in its first 30 days, and
# that being unable to recommend new SKUs in that window forgoes about
# S$120K/month of cross-sell on a 200-SKU-per-week cadence. If content-based
# scoring recovers ~60% of that gap, that is ~S$72K/month preserved. The
# cold-start RMSE printed above is the evidence you would take to the
# business: it tells you whether the features are informative enough to
# justify the assumption.
#
# LIMITATIONS:
#   - cold-start USERS (first-time visitors) still get zero recommendations
#   - filter bubble: users who only rated laptops see only laptop recs
#   - feature quality: if "brand" isn't tagged, recs miss brand loyalty
#
# The next technique (02_user_cf.py) solves the filter-bubble problem by
# bringing in the opinions of similar users.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built a user taste profile from mean-centred ratings + item features
  [x] Scored unseen items via cosine similarity to the profile
  [x] Measured holdout RMSE and ranking metrics (P@5, MAP) against baselines
  [x] Understood WHEN to use content-based: cold-start items + feature-rich catalogue
  [x] Understood WHEN it fails: cold-start users, narrow interest graphs

  KEY INSIGHT: Content-based filtering needs NO cross-user data. It is the
  only cold-start-item recommender that works on day 0 for a brand-new SKU.

  Next: 02_user_cf.py — bring in community opinions to escape the filter
  bubble and borrow preferences from users who share your taste.
"""
)
