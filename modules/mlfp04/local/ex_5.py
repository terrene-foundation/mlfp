# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 5: Association Rules and Market Basket Analysis
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Implement the Apriori algorithm from scratch with pruning
#   - Compute support, confidence, and lift for association rules
#   - Compare Apriori and FP-Growth implementations for consistency
#   - Engineer rule-based features that improve a supervised classifier
#   - Explain the connection: co-occurrence → matrix factorisation → neural networks
#
# PREREQUISITES:
#   - MLFP04 Exercise 1 (clustering — pattern discovery without labels)
#   - MLFP03 Exercise 1 (feature engineering — rules as domain features)
#
# ESTIMATED TIME: 75-90 minutes
#
# TASKS:
#   1. Generate synthetic Singapore retail transaction data (2000+ txns)
#   2. Implement Apriori from scratch: frequent itemsets, candidate gen, pruning
#   3. Compute support, confidence, lift for discovered rules
#   4. Compare with mlxtend FP-Growth
#   5. Filter and rank top association rules by lift
#   6. Interpret rules with business meaning
#   7. Engineer features from discovered rules for classification
#   8. Demonstrate rule-based features improve a supervised model
#
# THEORY:
#   Support:    supp(X) = count(X) / N
#   Confidence: conf(X -> Y) = supp(X ∪ Y) / supp(X)
#   Lift:       lift(X -> Y) = conf(X -> Y) / supp(Y)
#               Lift > 1 = positive association (surprise)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from kailash_ml import ModelVisualizer

try:
    from mlxtend.frequent_patterns import apriori as mlx_apriori
    from mlxtend.frequent_patterns import association_rules as mlx_association_rules
    from mlxtend.frequent_patterns import fpgrowth as mlx_fpgrowth
except ImportError:
    mlx_apriori = None
    mlx_fpgrowth = None
    mlx_association_rules = None


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Generate synthetic Singapore retail transaction data
# ══════════════════════════════════════════════════════════════════════

rng = np.random.default_rng(42)

PRODUCTS = [
    "bread", "butter", "milk", "eggs", "rice",
    "noodles", "soy_sauce", "cooking_oil", "chicken", "fish",
    "coffee", "tea", "sugar", "condensed_milk", "biscuits",
    "chips", "soft_drink", "beer", "wine", "tissue",
    "shampoo", "soap", "detergent", "toothpaste", "bananas",
]

N_TRANSACTIONS = 2500

# Define realistic co-purchase patterns (bundles that appear together)
BUNDLES = [
    (["bread", "butter", "eggs"], 0.15),           # Breakfast bundle
    (["coffee", "condensed_milk", "sugar"], 0.12),  # Kopi bundle
    (["rice", "chicken", "soy_sauce"], 0.10),       # Home cooking
    (["noodles", "eggs", "soy_sauce"], 0.08),       # Quick meal
    (["beer", "chips"], 0.09),                      # Snack pairing
    (["milk", "biscuits"], 0.07),                    # Tea-time
    (["shampoo", "soap", "toothpaste"], 0.06),      # Personal care
    (["tea", "sugar", "biscuits"], 0.05),            # Afternoon tea
    (["wine", "chips", "biscuits"], 0.04),           # Entertaining
    (["cooking_oil", "rice", "fish"], 0.06),         # Fish rice
    (["detergent", "tissue", "soap"], 0.05),         # Household
    (["bananas", "milk", "eggs"], 0.05),             # Smoothie
]

transactions: list[set[str]] = []

for _ in range(N_TRANSACTIONS):
    basket: set[str] = set()

    for bundle_items, prob in BUNDLES:
        if rng.random() < prob:
            for item in bundle_items:
                if rng.random() < 0.85:
                    basket.add(item)

    n_random = rng.poisson(2)
    random_items = rng.choice(PRODUCTS, size=min(n_random, 5), replace=False)
    basket.update(random_items)

    if len(basket) > 0:
        transactions.append(basket)

print(f"=== Synthetic Retail Transactions ===")
print(f"Transactions: {len(transactions):,}")
print(f"Products: {len(PRODUCTS)}")
avg_basket = np.mean([len(t) for t in transactions])
print(f"Avg basket size: {avg_basket:.1f} items")

print(f"\nSample transactions:")
for i in range(5):
    print(f"  Txn {i}: {sorted(transactions[i])}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Implement Apriori from scratch
# ══════════════════════════════════════════════════════════════════════
# Apriori principle: if an itemset is infrequent, all its supersets
# must also be infrequent. This allows aggressive pruning.
#
# Algorithm:
#   1. Scan database, count support for all single items
#   2. Prune items below min_support → L1 (frequent 1-itemsets)
#   3. Generate candidate k-itemsets from L(k-1)
#   4. Scan database, count support for candidates
#   5. Prune below min_support → Lk
#   6. Repeat until no new frequent itemsets found


def apriori(
    transactions: list[set[str]],
    min_support: float,
) -> dict[frozenset[str], float]:
    """
    Apriori algorithm for frequent itemset mining.

    Parameters
    ----------
    transactions : list of sets
        Each set contains items in one transaction.
    min_support : float
        Minimum support threshold (0 to 1).

    Returns
    -------
    dict mapping frozenset -> support value
        All frequent itemsets and their support.
    """
    n = len(transactions)
    min_count = min_support * n

    # Step 1: Count single-item support
    # TODO: Count how many transactions contain each item
    item_counts: dict[str, int] = defaultdict(int)
    for txn in transactions:
        for item in txn:
            ____  # Hint: item_counts[item] += 1

    # Step 2: L1 — frequent 1-itemsets
    freq_itemsets: dict[frozenset[str], float] = {}
    current_level: list[frozenset[str]] = []
    for item, count in item_counts.items():
        # TODO: Check if count >= min_count; if so, add frozenset([item]) with support count/n
        if ____:  # Hint: count >= min_count
            fs = frozenset([item])
            freq_itemsets[fs] = count / n
            current_level.append(fs)

    print(f"\n  L1: {len(current_level)} frequent items (min_support={min_support})")

    k = 2
    while current_level:
        candidates = _generate_candidates(current_level, k)
        if not candidates:
            break

        # Step 4: Count support for candidates
        candidate_counts: dict[frozenset[str], int] = defaultdict(int)
        for txn in transactions:
            txn_frozen = frozenset(txn)
            for candidate in candidates:
                # TODO: Check if candidate is a subset of txn_frozen; if so, increment count
                if ____:  # Hint: candidate.issubset(txn_frozen)
                    candidate_counts[candidate] += 1

        # Step 5: Prune — keep only frequent
        current_level = []
        for candidate, count in candidate_counts.items():
            if count >= min_count:
                freq_itemsets[candidate] = count / n
                current_level.append(candidate)

        print(f"  L{k}: {len(current_level)} frequent {k}-itemsets")
        k += 1

    return freq_itemsets


def _generate_candidates(
    prev_level: list[frozenset[str]],
    k: int,
) -> list[frozenset[str]]:
    """
    Generate candidate k-itemsets from frequent (k-1)-itemsets.

    Uses the Apriori join step: merge two (k-1)-itemsets that share
    (k-2) items. Then prune any candidate whose (k-1)-subset is
    not in the previous level (Apriori property).
    """
    prev_set = set(prev_level)
    candidates: set[frozenset[str]] = set()

    for i, a in enumerate(prev_level):
        for b in prev_level[i + 1:]:
            union = a | b
            if len(union) == k:
                # Apriori pruning: every (k-1)-subset must be frequent
                all_subsets_frequent = all(
                    union - frozenset([item]) in prev_set for item in union
                )
                if all_subsets_frequent:
                    candidates.add(union)

    return list(candidates)


# Run Apriori
print(f"\n=== Apriori Algorithm (from scratch) ===")
min_sup = 0.03
frequent_itemsets = apriori(transactions, min_support=min_sup)

print(f"\nTotal frequent itemsets: {len(frequent_itemsets)}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert len(frequent_itemsets) > 0, "Apriori should find at least one frequent itemset"
n = len(transactions)
for itemset, support in list(frequent_itemsets.items())[:5]:
    actual_count = sum(1 for t in transactions if itemset.issubset(frozenset(t)))
    actual_support = actual_count / n
    assert abs(actual_support - support) < 0.005, \
        f"Reported support {support:.4f} doesn't match actual {actual_support:.4f}"
    assert support >= min_sup - 0.001, \
        f"Itemset with support {support:.4f} should not appear below min_support={min_sup}"
# INTERPRETATION: The Apriori principle is anti-monotone: if {bread, butter} is
# infrequent, no superset {bread, butter, eggs} can be frequent. This allows
# aggressive pruning — we never need to count supersets of infrequent sets.
print("\n✓ Checkpoint 1 passed — Apriori found frequent itemsets with correct support\n")

sorted_itemsets = sorted(frequent_itemsets.items(), key=lambda x: -x[1])
print(f"\nTop 15 frequent itemsets:")
print(f"  {'Itemset':<45} {'Support':>8}")
print("  " + "-" * 55)
for itemset, support in sorted_itemsets[:15]:
    items_str = ", ".join(sorted(itemset))
    print(f"  {items_str:<45} {support:>8.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Compute association rules — support, confidence, lift
# ══════════════════════════════════════════════════════════════════════
# For each frequent itemset of size >= 2, generate rules X -> Y
# Confidence: conf(X -> Y) = supp(X ∪ Y) / supp(X)
# Lift:       lift(X -> Y) = conf(X -> Y) / supp(Y)


def generate_rules(
    freq_itemsets: dict[frozenset[str], float],
    min_confidence: float = 0.5,
) -> list[dict]:
    """
    Generate association rules from frequent itemsets.
    """
    rules = []

    for itemset, support in freq_itemsets.items():
        if len(itemset) < 2:
            continue

        items = list(itemset)
        for r in range(1, len(items)):
            for antecedent_tuple in combinations(items, r):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent

                supp_antecedent = freq_itemsets.get(antecedent)
                supp_consequent = freq_itemsets.get(consequent)

                if supp_antecedent is None or supp_consequent is None:
                    continue

                # TODO: Compute confidence = support / supp_antecedent
                confidence = ____  # Hint: support / supp_antecedent
                if confidence < min_confidence:
                    continue

                # TODO: Compute lift = confidence / supp_consequent
                lift = ____  # Hint: confidence / supp_consequent
                if confidence < 1.0:
                    conviction = (1 - supp_consequent) / (1 - confidence)
                else:
                    conviction = float("inf")

                rules.append({
                    "antecedent": antecedent,
                    "consequent": consequent,
                    "support": support,
                    "confidence": confidence,
                    "lift": lift,
                    "conviction": conviction,
                })

    return rules


min_conf = 0.3
rules = generate_rules(frequent_itemsets, min_confidence=min_conf)
rules_by_lift = sorted(rules, key=lambda r: -r["lift"])

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(rules) > 0, "Should generate at least one association rule"
for rule in rules[:5]:
    assert 0 <= rule["confidence"] <= 1.0, \
        f"Confidence must be in [0,1], got {rule['confidence']:.4f}"
    assert rule["lift"] > 0, f"Lift must be positive, got {rule['lift']:.4f}"
    ant_support = frequent_itemsets.get(rule["antecedent"], 0)
    assert rule["support"] <= ant_support + 0.001, \
        "Rule support should not exceed antecedent support"
# INTERPRETATION: Confidence is the conditional probability P(Y|X): given the
# customer bought X, how likely are they to buy Y? Lift > 1 means X and Y are
# positively associated (appearing together more than by chance).
print("\n✓ Checkpoint 2 passed — association rules computed with valid metrics\n")

print(f"\n=== Association Rules ===")
print(f"Rules found: {len(rules)} (min_confidence={min_conf})")
print(f"\nTop 20 rules by lift:")
print(f"  {'Antecedent':<25} {'->':>3} {'Consequent':<20} {'Supp':>6} {'Conf':>6} {'Lift':>6}")
print("  " + "-" * 85)
for rule in rules_by_lift[:20]:
    ant = ", ".join(sorted(rule["antecedent"]))
    con = ", ".join(sorted(rule["consequent"]))
    print(
        f"  {ant:<25} {'->':>3} {con:<20} "
        f"{rule['support']:>6.3f} {rule['confidence']:>6.3f} {rule['lift']:>6.2f}"
    )


# ══════════════════════════════════════════════════════════════════════
# TASK 4: FP-Growth comparison (mlxtend)
# ══════════════════════════════════════════════════════════════════════

all_items = sorted(PRODUCTS)
rows = []
for txn in transactions:
    row = {item: item in txn for item in all_items}
    rows.append(row)

txn_df = pl.DataFrame(rows)
print(f"\n=== Transaction Matrix ===")
print(f"Shape: {txn_df.shape} (transactions x products)")
print(f"Density: {txn_df.select(pl.all().mean()).to_numpy().mean():.2%}")

if mlx_fpgrowth is not None:
    txn_pd = txn_df.to_pandas()

    fp_frequent = mlx_fpgrowth(txn_pd, min_support=min_sup, use_colnames=True)
    fp_rules = mlx_association_rules(
        fp_frequent, metric="confidence", min_threshold=min_conf
    )

    print(f"\n=== FP-Growth (mlxtend) ===")
    print(f"Frequent itemsets: {len(fp_frequent)}")
    print(f"Association rules: {len(fp_rules)}")

    print(f"\nComparison:")
    print(f"  Apriori (scratch): {len(frequent_itemsets)} itemsets, {len(rules)} rules")
    print(f"  FP-Growth (mlxtend): {len(fp_frequent)} itemsets, {len(fp_rules)} rules")

    fp_rules_sorted = fp_rules.sort_values("lift", ascending=False)
    print(f"\nFP-Growth top 10 rules by lift:")
    print(f"  {'Antecedent':<25} {'->':>3} {'Consequent':<20} {'Supp':>6} {'Conf':>6} {'Lift':>6}")
    print("  " + "-" * 85)
    for _, row in fp_rules_sorted.head(10).iterrows():
        ant = ", ".join(sorted(row["antecedents"]))
        con = ", ".join(sorted(row["consequents"]))
        print(
            f"  {ant:<25} {'->':>3} {con:<20} "
            f"{row['support']:>6.3f} {row['confidence']:>6.3f} {row['lift']:>6.2f}"
        )
else:
    print("\nmlxtend not installed — skipping FP-Growth comparison")
    print("Install with: pip install mlxtend")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Filter and rank top rules
# ══════════════════════════════════════════════════════════════════════

# TODO: Filter rules to those where support >= 0.03, confidence >= 0.4, lift > 1.5
actionable_rules = [
    r for r in rules
    if ____  # Hint: r["support"] >= 0.03 and r["confidence"] >= 0.4 and r["lift"] > 1.5
]
actionable_rules.sort(key=lambda r: -r["lift"])

print(f"\n=== Actionable Rules (supp>=0.03, conf>=0.4, lift>1.5) ===")
# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(actionable_rules) > 0, \
    "At least one rule should pass the actionable threshold (supp>=0.03, conf>=0.4, lift>1.5)"
top_lift = actionable_rules[0]["lift"]
assert top_lift > 1.5, f"Top rule by lift should exceed 1.5, got {top_lift:.2f}"
# INTERPRETATION: The three-threshold filter implements actionability. Support
# ensures we have enough data to trust the rule. Confidence ensures the pattern
# is reliable. Lift ensures the association is meaningfully above chance.
print("\n✓ Checkpoint 3 passed — actionable rules filtered by three-threshold criteria\n")

print(f"Rules passing all thresholds: {len(actionable_rules)}")
print(f"\n  {'Antecedent':<25} {'->':>3} {'Consequent':<20} {'Supp':>6} {'Conf':>6} {'Lift':>6}")
print("  " + "-" * 85)
for rule in actionable_rules[:15]:
    ant = ", ".join(sorted(rule["antecedent"]))
    con = ", ".join(sorted(rule["consequent"]))
    print(
        f"  {ant:<25} {'->':>3} {con:<20} "
        f"{rule['support']:>6.3f} {rule['confidence']:>6.3f} {rule['lift']:>6.2f}"
    )


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Interpret rules with business meaning
# ══════════════════════════════════════════════════════════════════════

CATEGORY_MAP = {
    "bread": "breakfast", "butter": "breakfast", "eggs": "breakfast",
    "milk": "dairy", "condensed_milk": "dairy",
    "coffee": "beverage", "tea": "beverage", "soft_drink": "beverage",
    "sugar": "pantry", "rice": "pantry", "cooking_oil": "pantry",
    "soy_sauce": "pantry", "noodles": "pantry",
    "chicken": "protein", "fish": "protein",
    "beer": "alcohol", "wine": "alcohol",
    "chips": "snack", "biscuits": "snack", "bananas": "fruit",
    "shampoo": "personal_care", "soap": "personal_care",
    "toothpaste": "personal_care",
    "tissue": "household", "detergent": "household",
}

print(f"\n=== Business Interpretation ===")
for i, rule in enumerate(actionable_rules[:10]):
    ant_items = sorted(rule["antecedent"])
    con_items = sorted(rule["consequent"])
    ant_cats = set(CATEGORY_MAP.get(item, "other") for item in ant_items)
    con_cats = set(CATEGORY_MAP.get(item, "other") for item in con_items)

    if ant_cats == con_cats:
        rel_type = "within-category complement"
    elif ant_cats & con_cats:
        rel_type = "cross-category with overlap"
    else:
        rel_type = "cross-category association"

    ant_str = " + ".join(ant_items)
    con_str = " + ".join(con_items)

    print(f"\n  Rule {i+1}: {ant_str} -> {con_str}")
    print(f"    Lift={rule['lift']:.2f}, Conf={rule['confidence']:.1%}")
    print(f"    Type: {rel_type}")
    if rule["lift"] > 3.0:
        print(f"    Action: STRONG pairing — co-locate on shelf, bundle discount")
    elif rule["lift"] > 2.0:
        print(f"    Action: Moderate pairing — cross-sell recommendation")
    else:
        print(f"    Action: Mild pairing — use in personalised suggestions")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Engineer features from association rules for classification
# ══════════════════════════════════════════════════════════════════════

basket_sizes = np.array([len(t) for t in transactions])
high_value_threshold = 6
labels = (basket_sizes >= high_value_threshold).astype(int)

print(f"\n=== Feature Engineering from Rules ===")
print(f"Target: high-value shopper (basket >= {high_value_threshold} items)")
print(f"Positive rate: {labels.mean():.1%}")

X_baseline = txn_df.to_numpy().astype(np.float64)

top_rules_for_features = actionable_rules[:20]

rule_features: list[dict[str, int]] = []
for txn in transactions:
    txn_set = frozenset(txn)
    row: dict[str, int] = {}

    rules_triggered = 0
    total_lift = 0.0

    for idx, rule in enumerate(top_rules_for_features):
        ant_present = int(rule["antecedent"].issubset(txn_set))
        full_present = int(
            (rule["antecedent"] | rule["consequent"]).issubset(txn_set)
        )

        ant_name = "_".join(sorted(rule["antecedent"]))
        con_name = "_".join(sorted(rule["consequent"]))
        row[f"rule_{idx}_ant_{ant_name}"] = ant_present
        row[f"rule_{idx}_full_{ant_name}_to_{con_name}"] = full_present

        if full_present:
            rules_triggered += 1
            total_lift += rule["lift"]

    row["n_rules_triggered"] = rules_triggered
    row["total_rule_lift"] = int(total_lift * 100)
    rule_features.append(row)

rule_df = pl.DataFrame(rule_features).fill_null(0)
X_rules = rule_df.to_numpy().astype(np.float64)
X_combined = np.hstack([X_baseline, X_rules])

print(f"Baseline features: {X_baseline.shape[1]} (product presence)")
print(f"Rule features: {X_rules.shape[1]} (from {len(top_rules_for_features)} rules)")
print(f"Combined features: {X_combined.shape[1]}")

rule_feature_names = rule_df.columns
print(f"\nSample rule features:")
for name in rule_feature_names[:8]:
    print(f"  {name}")
print(f"  ... and {len(rule_feature_names) - 8} more")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Demonstrate rule features improve supervised model
# ══════════════════════════════════════════════════════════════════════

X_base_train, X_base_test, y_train, y_test = train_test_split(
    X_baseline, labels, test_size=0.3, random_state=42, stratify=labels
)
X_rules_train, X_rules_test, _, _ = train_test_split(
    X_rules, labels, test_size=0.3, random_state=42, stratify=labels
)
X_comb_train, X_comb_test, _, _ = train_test_split(
    X_combined, labels, test_size=0.3, random_state=42, stratify=labels
)

scaler_base = StandardScaler()
scaler_rules = StandardScaler()
scaler_comb = StandardScaler()

X_base_train_s = scaler_base.fit_transform(X_base_train)
X_base_test_s = scaler_base.transform(X_base_test)
X_rules_train_s = scaler_rules.fit_transform(X_rules_train)
X_rules_test_s = scaler_rules.transform(X_rules_test)
X_comb_train_s = scaler_comb.fit_transform(X_comb_train)
X_comb_test_s = scaler_comb.transform(X_comb_test)

print(f"\n=== Model Comparison: Logistic Regression ===")
results: dict[str, dict[str, float]] = {}

for name, X_tr, X_te in [
    ("Baseline (products)", X_base_train_s, X_base_test_s),
    ("Rules only", X_rules_train_s, X_rules_test_s),
    ("Combined", X_comb_train_s, X_comb_test_s),
]:
    # TODO: Train LogisticRegression(max_iter=1000, random_state=42) and compute acc, f1, auc
    lr = ____  # Hint: LogisticRegression(max_iter=1000, random_state=42)
    # TODO: Fit lr on X_tr, y_train
    ____  # Hint: lr.fit(X_tr, y_train)
    y_pred = lr.predict(X_te)
    y_proba = lr.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results[f"LR: {name}"] = {"Accuracy": acc, "F1": f1, "AUC_ROC": auc}
    print(f"  {name:<25} Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

print(f"\n=== Model Comparison: Random Forest ===")
for name, X_tr, X_te in [
    ("Baseline (products)", X_base_train, X_base_test),
    ("Rules only", X_rules_train, X_rules_test),
    ("Combined", X_comb_train, X_comb_test),
]:
    # TODO: Train RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf = ____  # Hint: RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    # TODO: Fit rf on X_tr, y_train
    ____  # Hint: rf.fit(X_tr, y_train)
    y_pred = rf.predict(X_te)
    y_proba = rf.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results[f"RF: {name}"] = {"Accuracy": acc, "F1": f1, "AUC_ROC": auc}
    print(f"  {name:<25} Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

rf_combined = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
)
rf_combined.fit(X_comb_train, y_train)
importances = rf_combined.feature_importances_

product_importance = importances[: X_baseline.shape[1]].sum()
rule_importance = importances[X_baseline.shape[1]:].sum()
total_importance = product_importance + rule_importance

print(f"\n=== Feature Importance Attribution ===")
print(f"Product features contribute: {product_importance / total_importance:.1%}")
print(f"Rule features contribute:    {rule_importance / total_importance:.1%}")

all_feature_names = all_items + list(rule_feature_names)
top_features_idx = np.argsort(importances)[::-1][:15]
print(f"\nTop 15 features (combined model):")
for idx in top_features_idx:
    fname = all_feature_names[idx] if idx < len(all_feature_names) else f"feature_{idx}"
    print(f"  {fname:<50} {importances[idx]:.4f}")


# ══════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

fig = viz.metric_comparison(results)
fig.update_layout(title="Baseline vs Rule-Enhanced Model Comparison")
fig.write_html("ex5_model_comparison.html")
print(f"\nSaved: ex5_model_comparison.html")

rules_for_plot = pl.DataFrame({
    "support": [r["support"] for r in rules_by_lift[:100]],
    "confidence": [r["confidence"] for r in rules_by_lift[:100]],
    "lift": [r["lift"] for r in rules_by_lift[:100]],
    "rule": [
        f"{', '.join(sorted(r['antecedent']))} -> {', '.join(sorted(r['consequent']))}"
        for r in rules_by_lift[:100]
    ],
})

fig_scatter = viz.scatter(rules_for_plot, x="support", y="confidence", color="lift")
fig_scatter.update_layout(title="Association Rules: Support vs Confidence (colour=Lift)")
fig_scatter.write_html("ex5_rules_scatter.html")
print("Saved: ex5_rules_scatter.html")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
lr_base_auc = results.get("LR: Baseline (products)", {}).get("AUC_ROC", 0)
lr_comb_auc = results.get("LR: Combined", {}).get("AUC_ROC", 0)
rf_base_auc = results.get("RF: Baseline (products)", {}).get("AUC_ROC", 0)
rf_comb_auc = results.get("RF: Combined", {}).get("AUC_ROC", 0)
assert lr_base_auc > 0.5, "Baseline LR should beat random"
assert rf_base_auc > 0.5, "Baseline RF should beat random"
assert lr_comb_auc >= lr_base_auc - 0.05, \
    "Rule-enhanced LR should not significantly regress vs baseline"
# INTERPRETATION: Rule-based features encode co-purchase patterns as explicit
# signals. For simpler models (LR), the rule features provide structure the
# model cannot learn from raw product presence alone.
print("\n✓ Checkpoint 4 passed — rule-enhanced model trained and compared\n")

print("\n--- Exercise 5 complete --- association rules and market basket analysis")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(f"""
  ✓ Apriori: anti-monotone pruning eliminates supersets of infrequent sets
  ✓ Support, confidence, lift: the three axes of association rule quality
  ✓ FP-Growth: compressed FP-tree, faster than Apriori for large datasets
  ✓ Actionable rules: three-threshold filter (support, confidence, lift)
  ✓ Feature engineering from rules: co-purchase patterns → supervised features

  METRICS REMINDER:
    Support    = how common (need enough data to trust)
    Confidence = conditional probability (how reliable is the rule?)
    Lift       = surprise factor (is this above chance?)

  FORWARD CONNECTION:
    Rules discover co-occurrence. Matrix factorisation (next exercise)
    extends this by learning latent factors from co-occurrence matrices.
    Neural networks generalise further — hidden layers learn arbitrary
    nonlinear feature combinations. Same core idea, increasing power.
""")
print("═" * 70)
