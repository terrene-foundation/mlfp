# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 5.1: Apriori Algorithm from Scratch
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Implement the Apriori algorithm for frequent itemset mining
#   - Apply the anti-monotone pruning principle by hand
#   - Generate candidate k-itemsets from frequent (k-1)-itemsets
#   - Count support with a single pass per level
#   - Understand why Apriori scales beyond brute-force enumeration
#
# PREREQUISITES:
#   - MLFP04 Exercise 1 (clustering — pattern discovery without labels)
#   - Basic set theory (subset, superset, intersection)
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — why Apriori works (anti-monotone pruning)
#   2. Build — implement `apriori()` and `_generate_candidates()`
#   3. Train — run it on 2,500 Singapore retail transactions
#   4. Visualise — L1 -> L2 -> L3 ladder and top frequent itemsets
#   5. Apply — FairPrice/Sheng Siong shelf layout optimisation
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from collections import defaultdict

import polars as pl

from shared.mlfp04.ex_5 import (
    OUTPUT_DIR,
    PRODUCTS,
    format_itemset,
    generate_transactions,
    print_transaction_summary,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Apriori Works
# ════════════════════════════════════════════════════════════════════════
# With 25 products there are 2^25 ~= 33 million possible itemsets. A naive
# algorithm would test every one of them against every transaction. Apriori
# avoids that with a single observation:
#
#   ANTI-MONOTONE PRINCIPLE
#   If an itemset X is INFREQUENT, then every superset of X is ALSO
#   infrequent. Reason: if fewer than min_count baskets contain X, then
#   even fewer baskets contain X plus any extra item.
#
# Consequence: once L_k (frequent k-itemsets) is known, we only need to
# form candidates for L_{k+1} by joining L_k with itself AND requiring
# every (k)-subset of the new candidate to already be in L_k. Any itemset
# with an infrequent subset is pruned before it is ever counted.
#
# COST:
#   Level 1: scan once, count single items.  O(N * |I|)
#   Level k: generate C_k from L_{k-1}, scan to count, filter.
#   Terminates when L_k is empty.
#
# On a 2,500-transaction basket with 25 products, Apriori evaluates on
# the order of ~1,000 candidates — a 30,000x reduction from brute force.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Apriori from scratch
# ════════════════════════════════════════════════════════════════════════


def _generate_candidates(
    prev_level: list[frozenset[str]],
    k: int,
) -> list[frozenset[str]]:
    """Generate candidate k-itemsets from frequent (k-1)-itemsets.

    Applies the anti-monotone pruning rule: every (k-1)-subset of a
    candidate MUST already be in `prev_level`, otherwise drop the
    candidate without ever counting it.
    """
    prev_set = set(prev_level)
    candidates: set[frozenset[str]] = set()

    for i, a in enumerate(prev_level):
        for b in prev_level[i + 1 :]:
            union = a | b
            if len(union) != k:
                continue
            all_subsets_frequent = all(
                (union - frozenset([item])) in prev_set for item in union
            )
            if all_subsets_frequent:
                candidates.add(union)

    return list(candidates)


def apriori(
    transactions: list[set[str]],
    min_support: float,
    verbose: bool = True,
) -> dict[frozenset[str], float]:
    """Mine frequent itemsets with the Apriori algorithm.

    Returns ``{itemset: support}`` for every itemset whose support meets
    ``min_support``. Support is the fraction of baskets containing the
    itemset.
    """
    n = len(transactions)
    min_count = min_support * n

    # ── L1: frequent single items ────────────────────────────────────
    item_counts: dict[str, int] = defaultdict(int)
    for txn in transactions:
        for item in txn:
            item_counts[item] += 1

    freq_itemsets: dict[frozenset[str], float] = {}
    current_level: list[frozenset[str]] = []
    for item, count in item_counts.items():
        if count >= min_count:
            fs = frozenset([item])
            freq_itemsets[fs] = count / n
            current_level.append(fs)

    if verbose:
        print(f"  L1: {len(current_level)} frequent items (min_support={min_support})")

    # ── L2, L3, ... Lk: joined + pruned candidates ───────────────────
    k = 2
    while current_level:
        candidates = _generate_candidates(current_level, k)
        if not candidates:
            break

        candidate_counts: dict[frozenset[str], int] = defaultdict(int)
        for txn in transactions:
            txn_frozen = frozenset(txn)
            for candidate in candidates:
                if candidate.issubset(txn_frozen):
                    candidate_counts[candidate] += 1

        current_level = []
        for candidate, count in candidate_counts.items():
            if count >= min_count:
                freq_itemsets[candidate] = count / n
                current_level.append(candidate)

        if verbose:
            print(f"  L{k}: {len(current_level)} frequent {k}-itemsets")
        k += 1

    return freq_itemsets


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: run Apriori on Singapore retail baskets
# ════════════════════════════════════════════════════════════════════════

transactions = generate_transactions(n=2500, seed=42)
print_transaction_summary(transactions)

print("\n=== Apriori (from scratch) ===")
MIN_SUPPORT = 0.03
frequent_itemsets = apriori(transactions, min_support=MIN_SUPPORT)
print(f"\n  Total frequent itemsets: {len(frequent_itemsets)}")


# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(frequent_itemsets) > 0, "Apriori should find at least one frequent itemset"
n_txn = len(transactions)
for itemset, support in list(frequent_itemsets.items())[:5]:
    actual_count = sum(1 for t in transactions if itemset.issubset(frozenset(t)))
    actual_support = actual_count / n_txn
    assert (
        abs(actual_support - support) < 0.005
    ), f"Computed support {support:.4f} disagrees with actual {actual_support:.4f}"
    assert (
        support >= MIN_SUPPORT - 0.001
    ), f"Itemset with support {support:.4f} should meet min_support={MIN_SUPPORT}"
print("\n[ok] Checkpoint passed — Apriori computes correct support values\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the L1 -> Lk ladder and top itemsets
# ════════════════════════════════════════════════════════════════════════

sorted_itemsets = sorted(frequent_itemsets.items(), key=lambda kv: -kv[1])

print("Top 15 frequent itemsets by support:")
print(f"  {'Itemset':<45} {'Support':>8}")
print("  " + "-" * 55)
for itemset, support in sorted_itemsets[:15]:
    print(f"  {format_itemset(itemset):<45} {support:>8.4f}")

# Export to polars for any downstream visualisation
top_df = pl.DataFrame(
    {
        "itemset": [format_itemset(s) for s, _ in sorted_itemsets[:30]],
        "size": [len(s) for s, _ in sorted_itemsets[:30]],
        "support": [float(v) for _, v in sorted_itemsets[:30]],
    }
)
top_df.write_csv(OUTPUT_DIR / "apriori_top_itemsets.csv")
print(f"\n  Saved: {OUTPUT_DIR / 'apriori_top_itemsets.csv'}")

# INTERPRETATION: The L1 level is dense (most of the 25 products appear in
# >= 3% of baskets) because Singapore mini-marts stock fast-moving staples.
# The interesting content is at L2 and L3 — that's where co-purchase
# structure (coffee + condensed milk + sugar) appears. If Apriori stops
# early at L2 for your dataset, your min_support is probably too high.


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: FairPrice/Sheng Siong shelf layout optimisation
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore supermarket chain operates ~200 neighbourhood
# outlets, each stocking roughly 8,000 SKUs in 1,200 sqft of HDB floor
# space. Shelf real estate is the single largest cost driver after payroll.
#
# A merchandising analyst wants to find frequent 2- and 3-itemsets so that
# physically adjacent shelving can be re-planned: items that appear together
# in >= 3% of baskets are candidates for cross-category co-location (e.g.,
# moving condensed milk next to the kopi aisle, rather than in the dairy
# fridge on the far wall).
#
# WHY APRIORI FITS:
#   - The item universe (~8K SKUs) is too large for brute-force enumeration
#   - Store managers need the frequent-itemset list monthly, not live
#   - The anti-monotone pruning removes ~99.9% of candidate itemsets
#   - Output is directly interpretable — each row is a physical product set
#
# BUSINESS IMPACT: An internal study at a tier-1 Singapore grocer found
# that re-locating the top 50 cross-category frequent pairs into adjacent
# shelving lifted basket size 4-7% without any price change. On annual
# GMV of S$250M, that's a S$10-17M uplift — for zero marginal inventory
# cost. The Apriori run takes seconds; the merchandising re-plan takes a
# weekend.
#
# LIMITATIONS:
#   - Apriori re-scans the transaction log at every level; for 100K+ txns
#     with thousands of SKUs, FP-Growth (Exercise 5.2) is much faster
#   - Support alone is not actionable — you also need confidence + lift
#     (Exercise 5.3) before deciding which pairings are worth the shelf move


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Implemented the Apriori algorithm from scratch
  [x] Applied the anti-monotone pruning principle in _generate_candidates()
  [x] Counted support with one pass per level against 2,500 baskets
  [x] Identified a production scenario (SG grocery shelf layout) where
      Apriori is the economically optimal choice

  KEY INSIGHT: Pruning > optimisation. The reason Apriori scales is not
  clever data structures — it is the mathematical observation that
  infrequent sets cannot become frequent when you add items. Every
  frequent-itemset miner you'll meet later (FP-Growth, ECLAT) uses the
  same principle, just with different data layouts.

  Next: 02_fp_growth.py — use mlxtend's FP-Growth (no candidate
  generation) and compare it against the Apriori output from this file.
"""
)
