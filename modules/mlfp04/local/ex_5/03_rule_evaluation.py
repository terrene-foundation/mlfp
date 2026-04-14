# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 5.3: Rule Evaluation — Support, Confidence, Lift, Conviction
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Turn frequent itemsets into directional association rules (X -> Y)
#   - Compute the four standard rule-quality metrics
#   - Apply a three-threshold filter (support + confidence + lift)
#   - Separate cross-category rules from within-category rules
#
# PREREQUISITES:
#   - 01_apriori_from_scratch.py
#   - Basic probability (conditional probability, independence)
#
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — rule metric definitions
#   2. Build — implement `generate_rules()` + three-threshold filter
#   3. Train — mine + score rules on Singapore retail baskets
#   4. Visualise — top rules + category breakdown + polars export
#   5. Apply — Watsons cart-page recommender
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from shared.mlfp04.ex_5 import (
    OUTPUT_DIR,
    categorise_rule,
    format_itemset,
    generate_transactions,
    print_transaction_summary,
    rules_to_polars,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — The Four Rule Quality Metrics
# ════════════════════════════════════════════════════════════════════════
# support     = supp(X ∪ Y) = count(X ∪ Y) / N
# confidence  = conf(X -> Y) = supp(X ∪ Y) / supp(X)   # = P(Y | X)
# lift        = conf(X -> Y) / supp(Y)                 # = P(Y|X) / P(Y)
# conviction  = (1 - supp(Y)) / (1 - conf(X -> Y))
#
# Three-threshold filter (the only one that matters in practice):
#   keep if support >= s_min and confidence >= c_min and lift > 1


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: rule generator + three-threshold filter
# ════════════════════════════════════════════════════════════════════════


def _apriori(
    transactions: list[set[str]], min_support: float
) -> dict[frozenset[str], float]:
    """Small self-contained Apriori — same contract as 5.1."""
    n = len(transactions)
    min_count = min_support * n
    item_counts: dict[str, int] = defaultdict(int)
    for txn in transactions:
        for item in txn:
            item_counts[item] += 1
    freq: dict[frozenset[str], float] = {}
    level: list[frozenset[str]] = []
    for item, count in item_counts.items():
        if count >= min_count:
            fs = frozenset([item])
            freq[fs] = count / n
            level.append(fs)
    k = 2
    while level:
        prev_set = set(level)
        candidates: set[frozenset[str]] = set()
        for i, a in enumerate(level):
            for b in level[i + 1 :]:
                u = a | b
                if len(u) == k and all((u - frozenset([it])) in prev_set for it in u):
                    candidates.add(u)
        if not candidates:
            break
        counts: dict[frozenset[str], int] = defaultdict(int)
        for txn in transactions:
            tf = frozenset(txn)
            for c in candidates:
                if c.issubset(tf):
                    counts[c] += 1
        level = []
        for c, ct in counts.items():
            if ct >= min_count:
                freq[c] = ct / n
                level.append(c)
        k += 1
    return freq


def generate_rules(
    freq_itemsets: dict[frozenset[str], float],
    min_confidence: float,
) -> list[dict]:
    """Generate all association rules that clear ``min_confidence``."""
    rules: list[dict] = []
    for itemset, support in freq_itemsets.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        for r in range(1, len(items)):
            for ant_tuple in combinations(items, r):
                antecedent = frozenset(ant_tuple)
                consequent = itemset - antecedent

                supp_ant = freq_itemsets.get(antecedent)
                supp_con = freq_itemsets.get(consequent)
                if supp_ant is None or supp_con is None:
                    continue

                # TODO: compute confidence = support / supp_ant
                # Hint: support is P(X and Y); supp_ant is P(X)
                confidence = ____
                if confidence < min_confidence:
                    continue

                # TODO: compute lift = confidence / supp_con
                lift = ____

                # TODO: compute conviction. When confidence == 1.0 the
                # denominator would be zero — return float("inf") instead.
                # Formula: (1 - supp_con) / (1 - confidence)
                conviction = ____

                rules.append(
                    {
                        "antecedent": antecedent,
                        "consequent": consequent,
                        "support": support,
                        "confidence": confidence,
                        "lift": lift,
                        "conviction": conviction,
                    }
                )
    return rules


def filter_actionable(
    rules: list[dict],
    min_support: float,
    min_confidence: float,
    min_lift: float,
) -> list[dict]:
    """Apply the three-threshold filter and sort by descending lift."""
    # TODO: keep a rule only if it clears ALL three thresholds:
    #   support >= min_support
    #   confidence >= min_confidence
    #   lift > min_lift
    kept = ____

    kept.sort(key=lambda r: -r["lift"])
    return kept


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: mine + score rules on the SG retail baskets
# ════════════════════════════════════════════════════════════════════════

transactions = generate_transactions(n=2500, seed=42)
print_transaction_summary(transactions)

MIN_SUPPORT = 0.03
MIN_CONFIDENCE = 0.3

print("\n=== Mining frequent itemsets ===")
frequent = _apriori(transactions, min_support=MIN_SUPPORT)
print(f"  Frequent itemsets: {len(frequent)}")

print("\n=== Generating association rules ===")
rules = generate_rules(frequent, min_confidence=MIN_CONFIDENCE)
print(f"  Rules at min_confidence={MIN_CONFIDENCE}: {len(rules)}")

actionable = filter_actionable(
    rules, min_support=0.03, min_confidence=0.4, min_lift=1.5
)
print(f"  Actionable (supp>=0.03, conf>=0.4, lift>1.5): {len(actionable)}")


# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(rules) > 0, "At least one rule should clear min_confidence"
for rule in rules[:10]:
    assert 0 <= rule["confidence"] <= 1.0, "confidence must be a probability"
    assert rule["lift"] > 0, "lift must be positive"
assert len(actionable) > 0, "At least one rule should clear the three thresholds"
assert actionable[0]["lift"] > 1.5, "Top actionable rule should have lift > 1.5"
print("\n[ok] Checkpoint passed — rule metrics valid and actionable set non-empty\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: top rules + category breakdown
# ════════════════════════════════════════════════════════════════════════

print("Top 15 actionable rules by lift:")
header = (
    f"  {'Antecedent':<28} {'->':>3} {'Consequent':<20} "
    f"{'Supp':>6} {'Conf':>6} {'Lift':>6} {'Conv':>7}"
)
print(header)
print("  " + "-" * 88)
for rule in actionable[:15]:
    ant = format_itemset(rule["antecedent"])
    con = format_itemset(rule["consequent"])
    conv = (
        f"{rule['conviction']:.2f}" if rule["conviction"] != float("inf") else "   inf"
    )
    print(
        f"  {ant:<28} {'->':>3} {con:<20} "
        f"{rule['support']:>6.3f} {rule['confidence']:>6.3f} "
        f"{rule['lift']:>6.2f} {conv:>7}"
    )

# Category breakdown
cross = 0
within = 0
for rule in actionable:
    _, _, rel = categorise_rule(rule["antecedent"], rule["consequent"])
    if rel.startswith("within-category"):
        within += 1
    else:
        cross += 1

print("\n=== Category Breakdown ===")
print(f"  Cross-category rules: {cross}")
print(f"  Within-category rules: {within}")

scatter_df = rules_to_polars(rules).sort("lift", descending=True).head(100)
scatter_df.write_csv(OUTPUT_DIR / "top_rules_scatter.csv")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Watsons cart-page recommender
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Watsons SG (~110 stores + online) wants a cart-page
# recommender that surfaces ONE high-lift next-product suggestion.
# Product spec:
#   reliable  -> high confidence (doesn't annoy shoppers)
#   surprising -> high lift (not just "shampoo -> soap")
#   popular    -> high support (enough stock to fulfil)
#
# These ARE the three-threshold filter inputs.
#
# BUSINESS IMPACT: Industry A/B tests show 2-4% conversion lift and 5-9%
# AOV lift on recommended items. Watsons SG online GMV ~S$200M/year;
# a 6% AOV lift on recommendations is ~S$3-6M/year in pure margin.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Generated directional association rules from frequent itemsets
  [x] Computed support, confidence, lift, and conviction
  [x] Applied the three-threshold filter
  [x] Separated cross-category rules from within-category rules

  Next: 04_rule_features.py — use the rules as features for a supervised
  classifier and compare against a raw product-presence baseline.
"""
)
