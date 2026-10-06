# MLFP04 — Task 3: Baskets and Recommendations

**Weight**: 20 marks · **Outcomes**: 4.5 (frequent itemsets, support / confidence / lift, association rules), 4.7 (collaborative filtering, matrix factorisation, cold start)
**Data**: till exports and rating histories in the formats below. `dev_baskets.parquet` and `dev_ratings.parquet` (shipped with the task) are one of each for development; the grader uses its own.

## Scenario

**Market baskets.** A chain of neighbourhood mini-marts wants the product
associations behind its baskets so it can plan shelf adjacency and bundles.
A till export has one row per scanned line:

| Column      | Meaning                          |
| ----------- | -------------------------------- |
| `basket_id` | one checkout                     |
| `item`      | product name as keyed at the till |

Some tills key names by hand: the same product appears with different
capitalisation and stray spaces, and a product scanned twice appears twice
in the same basket. Report products by their canonical name — lowercase,
with surrounding spaces removed.

The category manager wants **every** rule `A → C` (A and C non-empty,
disjoint sets of products, at most `max_len` products in A ∪ C) whose
itemset A ∪ C is in at least `min_support` of baskets and whose confidence
is at least `min_confidence`, with its support (share of baskets containing
A ∪ C), confidence and lift. Thresholds change from export to export.

**Recommendations.** The same chain's app lets members rate products from 1
to 5 in half steps. A rating history has `user_id`, `item_id`, `rating` and
`rated_at`. Members sometimes re-rate a product; only their latest rating
reflects their opinion. The app team wants a model that predicts how a member
would rate products they have not rated, and that ranks those products by
the member's own taste, not just by overall popularity. Some members joined
after the history was exported and have no ratings at all; the app still has
to show them something sensible.

## What to submit

`starter.py` with two functions (signatures fixed):

| Function                                                      | Returns                                                                                                                                                            |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `mine_rules(baskets, min_support, min_confidence, max_len=3)` | polars DataFrame, one row per rule, sorted by `lift` descending: `antecedent` (list of str), `consequent` (list of str), `support`, `confidence`, `lift`            |
| `fit_recommender(history)`                                    | a function `predict(pairs)` taking a polars DataFrame with `user_id`, `item_id` and returning a numpy array with one predicted rating per row, in row order |

## Acceptance criteria

- Rules are checked on two secret till exports with secret thresholds
  (support 0.02–0.05, confidence 0.35–0.60, `max_len` 3) against the complete
  rule set computed by brute force: no rule missing, no extra rule, every
  support / confidence / lift exact (relative tolerance 1e-6), strongest rule
  first.
- The recommender is checked on two secret histories. The grader has held
  back a quarter of every member's latest ratings and all ratings of about a
  dozen new members, and asks your `predict` for those pairs (in shuffled
  order):
  - for existing members, RMSE must be at most 85% of that of a
    bias-only model (overall mean + member offset + product offset) fitted by
    the grader on the same history;
  - ranking each member's held-back products by your prediction must beat the
    bias-only model's ranking by at least 0.08 in mean NDCG@5;
  - for new members, predictions must be finite and no worse (RMSE) than
    predicting the overall mean rating.
- The no-extra-rules check earns marks only when the rule set is complete or
  its metrics are exact; the new-member check only when an existing-member
  check passes.

## Rules

- kailash-ml has no rule-mining or recommender engine; use polars and numpy
  (no pandas). Library rule miners are allowed if their output meets the
  acceptance criteria.
- Your functions must work from their arguments alone: the grader never
  passes the same data twice.
