# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 5.4: Rule-Based Features for Supervised Classification
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Engineer features from discovered association rules
#   - Compare a product-presence baseline against a rule-enhanced model
#   - Measure whether explicit rules add signal over raw one-hot features
#   - Build a prediction target that is NOT a function of the features
#   - Attribute model importance across product vs rule feature groups
#   - See the forward connection to matrix factorisation and neural nets
#
# PREREQUISITES:
#   - 01_apriori_from_scratch.py
#   - 03_rule_evaluation.py
#   - MLFP03 Exercise 1 (feature engineering)
#
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — rules as handcrafted co-occurrence features
#   2. Build — turn actionable rules into numeric feature columns
#   3. Train — predict next-trip breakfast-bundle purchase, baseline vs combined
#   4. Visualise — metric comparison + feature importance attribution
#   5. Apply — loyalty-programme next-trip offer targeting
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

from shared.mlfp04.ex_5 import (
    OUTPUT_DIR,
    generate_shopper_trips,
    print_transaction_summary,
    setup_engines,
    teardown_engines,
    track_run,
    transactions_to_onehot,
)

# ── Kailash-ML ExperimentTracker — every association-rules run logs here ─
tracker, exp_name = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Rules as Handcrafted Co-Occurrence Features
# ════════════════════════════════════════════════════════════════════════
# A linear model trained only on product-presence one-hot features sees
# each product as independent. It cannot represent "this basket contains
# the breakfast bundle" as a single signal — at best it learns a weight
# per product and adds them up.
#
# Association rules give you that missing structure for free. Every
# actionable rule X -> Y becomes one or two new binary features:
#
#   ant_present[r]   = 1 if the basket contains X
#   full_present[r]  = 1 if the basket contains X AND Y
#
# Plus a handful of aggregates:
#
#   n_rules_triggered  = how many rules fire on this basket
#   total_rule_lift    = sum of lifts of the rules that fire
#   max_rule_lift      = strongest rule that fires
#
# This is the MANUAL version of what matrix factorisation (Ex 7) and
# neural networks (Ex 8) will do AUTOMATICALLY. The progression is:
#
#   Manual rules  ->  Linear factorisation  ->  Nonlinear neural nets
#   explicit         compressed latent         learned non-linearity
#   interpretable    partially interpretable   least interpretable
#   cheap            medium                    expensive
#
# Ex 5.4 is the bottom rung. Every step above this file is a different
# way to discover co-occurrence structure without hand-writing rules.
#
# THE TARGET MUST NOT BE A FUNCTION OF THE FEATURES
# A tempting target is "big basket" (>= 6 items). But basket size is the
# row-sum of the one-hot product columns, so a linear model recovers it
# exactly (AUC = 1.0) and NO feature can add anything — a leak, not a
# finding. Here every shopper has TWO trips. Features come from THIS
# trip; the target is whether the NEXT trip contains the breakfast
# bundle (at least 2 of bread / butter / eggs). Both trips share the
# shopper's habits, so this trip is real but noisy evidence — exactly
# the setting where you can honestly ask whether rules add signal.


# ════════════════════════════════════════════════════════════════════════
# MINI-APRIORI + RULE GENERATION (inline, so this file stands alone)
# ════════════════════════════════════════════════════════════════════════


def _apriori(
    transactions: list[set[str]], min_support: float
) -> dict[frozenset[str], float]:
    """Inline Apriori — same contract as 01_apriori_from_scratch.apriori()."""
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


def _rules_from_itemsets(
    freq: dict[frozenset[str], float],
    min_confidence: float = 0.4,
    min_lift: float = 1.5,
) -> list[dict]:
    rules: list[dict] = []
    for itemset, support in freq.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        for r in range(1, len(items)):
            for ant_tuple in combinations(items, r):
                antecedent = frozenset(ant_tuple)
                consequent = itemset - antecedent
                supp_ant = freq.get(antecedent)
                supp_con = freq.get(consequent)
                if supp_ant is None or supp_con is None:
                    continue
                confidence = support / supp_ant
                lift = confidence / supp_con
                if confidence >= min_confidence and lift > min_lift:
                    rules.append(
                        {
                            "antecedent": antecedent,
                            "consequent": consequent,
                            "support": support,
                            "confidence": confidence,
                            "lift": lift,
                        }
                    )
    rules.sort(key=lambda r: -r["lift"])
    return rules


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: rule-based feature engineer
# ════════════════════════════════════════════════════════════════════════


def engineer_rule_features(
    transactions: list[set[str]],
    rules: list[dict],
) -> pl.DataFrame:
    """Turn each transaction into a row of rule-derived numeric features.

    Columns produced:
      - ``rule{i}_antecedent`` — 1 if the antecedent is present
      - ``rule{i}_full``       — 1 if antecedent AND consequent are present
      - ``n_rules_triggered``  — count of fully-matched rules
      - ``total_rule_lift_x100`` — sum of matched rule lifts (scaled int)
      - ``max_rule_lift_x100``   — max matched rule lift (scaled int)
    """
    rows: list[dict[str, int]] = []
    for txn in transactions:
        txn_set = frozenset(txn)
        row: dict[str, int] = {}
        total_lift = 0.0
        n_triggered = 0
        max_lift = 0.0

        for idx, rule in enumerate(rules):
            ant_present = int(rule["antecedent"].issubset(txn_set))
            full_present = int(
                (rule["antecedent"] | rule["consequent"]).issubset(txn_set)
            )
            row[f"rule{idx}_antecedent"] = ant_present
            row[f"rule{idx}_full"] = full_present
            if full_present:
                n_triggered += 1
                total_lift += float(rule["lift"])
                if float(rule["lift"]) > max_lift:
                    max_lift = float(rule["lift"])

        row["n_rules_triggered"] = n_triggered
        row["total_rule_lift_x100"] = int(total_lift * 100)
        row["max_rule_lift_x100"] = int(max_lift * 100)
        rows.append(row)

    return pl.DataFrame(rows).fill_null(0)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: baseline vs combined classifiers
# ════════════════════════════════════════════════════════════════════════

# Features come from this trip; the label comes from the shopper's NEXT
# trip, which is never used to build a feature.
transactions, next_trips = generate_shopper_trips(n_shoppers=2500, seed=42)
print_transaction_summary(transactions)

TARGET_BUNDLE = frozenset({"bread", "butter", "eggs"})
y = np.array([int(len(nxt & TARGET_BUNDLE) >= 2) for nxt in next_trips])
print(
    f"\n  Target: next trip contains >= 2 of {sorted(TARGET_BUNDLE)} "
    "(breakfast bundle)"
)
print(f"  Positive rate: {y.mean():.1%}")

# --- Baseline features: raw product presence ---
onehot = transactions_to_onehot(transactions)
X_baseline = onehot.to_numpy().astype(np.float64)
print(f"\n  Baseline features (product presence): {X_baseline.shape[1]}")

# --- Mine rules and engineer features ---
print("\n=== Mining rules for feature engineering ===")
freq = _apriori(transactions, min_support=0.03)
rules = _rules_from_itemsets(freq, min_confidence=0.4, min_lift=1.5)
top_rules = rules[:20]
print(f"  Actionable rules found: {len(rules)}")
print(f"  Using top {len(top_rules)} rules as features")

rule_df = engineer_rule_features(transactions, top_rules)
X_rules = rule_df.to_numpy().astype(np.float64)
X_combined = np.hstack([X_baseline, X_rules])
print(f"  Rule features:     {X_rules.shape[1]}")
print(f"  Combined features: {X_combined.shape[1]}")


def _split_and_scale(X: np.ndarray, y: np.ndarray, scale: bool):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    if scale:
        s = StandardScaler()
        X_tr = s.fit_transform(X_tr)
        X_te = s.transform(X_te)
    return X_tr, X_te, y_tr, y_te


results: dict[str, dict[str, float]] = {}

print("\n=== Model: Logistic Regression ===")
for name, X in [
    ("Baseline", X_baseline),
    ("Rules only", X_rules),
    ("Combined", X_combined),
]:
    X_tr, X_te, y_tr, y_te = _split_and_scale(X, y, scale=True)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_tr, y_tr)
    y_pred = lr.predict(X_te)
    y_proba = lr.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_proba)
    results[f"LR: {name}"] = {"accuracy": acc, "f1": f1, "auc_roc": auc}
    print(f"  {name:<12} acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")

print("\n=== Model: Random Forest ===")
for name, X in [
    ("Baseline", X_baseline),
    ("Rules only", X_rules),
    ("Combined", X_combined),
]:
    X_tr, X_te, y_tr, y_te = _split_and_scale(X, y, scale=False)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    y_proba = rf.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_proba)
    results[f"RF: {name}"] = {"accuracy": acc, "f1": f1, "auc_roc": auc}
    print(f"  {name:<12} acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")


# ── Checkpoint ──────────────────────────────────────────────────────────
assert (
    X_combined.shape[1] > X_baseline.shape[1]
), "Combined should add columns to the baseline, not replace them"
assert X_rules.shape[1] > 0, "Should have produced at least one rule feature"
lr_baseline = results["LR: Baseline"]["auc_roc"]
lr_combined = results["LR: Combined"]["auc_roc"]
rf_baseline = results["RF: Baseline"]["auc_roc"]
assert lr_baseline > 0.5, "Baseline LR should beat random"
assert rf_baseline > 0.5, "Baseline RF should beat random"
assert lr_baseline < 0.95, (
    "Baseline AUC is suspiciously close to 1.0 — check that the target is "
    "not computed from the feature columns (target leakage)"
)
assert (
    lr_combined >= lr_baseline - 0.05
), "Adding rule features should not significantly regress LR"
print("\n[ok] Checkpoint passed — rule-enhanced model trained + compared\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: feature importance attribution
# ════════════════════════════════════════════════════════════════════════

X_tr, X_te, y_tr, y_te = _split_and_scale(X_combined, y, scale=False)
rf_combined = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
)
rf_combined.fit(X_tr, y_tr)
importances = rf_combined.feature_importances_

product_importance = float(importances[: X_baseline.shape[1]].sum())
rule_importance = float(importances[X_baseline.shape[1] :].sum())
total = product_importance + rule_importance

print("=== Feature Importance Attribution (RF combined) ===")
print(f"  Product features contribute: {product_importance / total:.1%}")
print(f"  Rule features contribute:    {rule_importance / total:.1%}")

all_feature_names = list(onehot.columns) + list(rule_df.columns)
top_idx = np.argsort(importances)[::-1][:15]
print("\n  Top 15 features in the combined model:")
for idx in top_idx:
    fname = all_feature_names[idx]
    ftype = "product" if idx < X_baseline.shape[1] else "rule"
    print(f"    [{ftype:>7}] {fname:<40} {importances[idx]:.4f}")

# Metric comparison frame (for notebooks to chart)
metric_rows = [{"model": k, **v} for k, v in results.items()]
metric_df = pl.DataFrame(metric_rows)
metric_df.write_csv(OUTPUT_DIR / "rule_features_metrics.csv")
print(f"\n  Saved: {OUTPUT_DIR / 'rule_features_metrics.csv'}")

# ── Visualisation ─────────────────────────────────────────────────────
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# (A) Feature importance: product vs rule groups (pie + top-15 bar)
fig_imp = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "pie"}, {"type": "xy"}]],
    subplot_titles=["Importance by Group", "Top 15 Features"],
)
fig_imp.add_trace(
    go.Pie(
        labels=["Product Features", "Rule Features"],
        values=[product_importance, rule_importance],
        marker_colors=["#636EFA", "#EF553B"],
        textinfo="label+percent",
    ),
    row=1,
    col=1,
)
top_names = [all_feature_names[i] for i in top_idx]
top_vals = [float(importances[i]) for i in top_idx]
top_colors = ["#636EFA" if idx < X_baseline.shape[1] else "#EF553B" for idx in top_idx]
fig_imp.add_trace(
    go.Bar(
        x=top_vals,
        y=top_names,
        orientation="h",
        marker_color=top_colors,
        showlegend=False,
    ),
    row=1,
    col=2,
)
fig_imp.update_layout(
    title="Feature Importance: Product (blue) vs Rule (red) Features",
    height=500,
    width=1000,
)
fig_imp.update_yaxes(autorange="reversed", row=1, col=2)
imp_path = OUTPUT_DIR / "04_feature_importance.html"
fig_imp.write_html(str(imp_path))
print(f"[viz] Feature importance: {imp_path}")

# (B) Accuracy improvement: baseline vs combined across models
model_names = list(results.keys())
acc_vals = [results[k]["accuracy"] for k in model_names]
auc_vals = [results[k]["auc_roc"] for k in model_names]
fig_acc = go.Figure()
fig_acc.add_trace(
    go.Bar(
        x=model_names,
        y=acc_vals,
        name="Accuracy",
        marker_color="#636EFA",
        text=[f"{v:.3f}" for v in acc_vals],
        textposition="outside",
    )
)
fig_acc.add_trace(
    go.Bar(
        x=model_names,
        y=auc_vals,
        name="AUC-ROC",
        marker_color="#EF553B",
        text=[f"{v:.3f}" for v in auc_vals],
        textposition="outside",
    )
)
fig_acc.update_layout(
    title="Model Comparison: Baseline vs Rules-Only vs Combined",
    xaxis_title="Model Variant",
    yaxis_title="Score",
    barmode="group",
    yaxis_range=[0, 1.1],
)
acc_path = OUTPUT_DIR / "04_accuracy_comparison.html"
fig_acc.write_html(str(acc_path))
print(f"[viz] Accuracy comparison: {acc_path}")

# INTERPRETATION — computed from this run, not assumed in advance.
lr_lift = results["LR: Combined"]["auc_roc"] - results["LR: Baseline"]["auc_roc"]
rf_lift = results["RF: Combined"]["auc_roc"] - results["RF: Baseline"]["auc_roc"]
print("\n=== Did rule features add signal? ===")
for model_name, lift in [("Logistic regression", lr_lift), ("Random forest", rf_lift)]:
    if lift > 0.01:
        verdict = "rules ADDED signal the product columns did not carry"
    elif lift < -0.01:
        verdict = "rules HURT — extra correlated columns added noise"
    else:
        verdict = "no measurable gain (within +/-0.01 AUC)"
    print(f"  {model_name:<20} AUC change {lift:+.3f} -> {verdict}")
print(
    "  Read this honestly. When each item is independent evidence of a\n"
    "  shopper's habit, a model that adds up per-product weights already\n"
    "  uses that evidence, so 'bread AND butter' columns add little AUC.\n"
    "  Rule features earn their place when the outcome depends on the\n"
    "  COMBINATION itself, and as named, auditable columns. Ex 7 (matrix\n"
    "  factorisation) and Ex 8 (neural nets) learn such structure without\n"
    "  pre-specified rules."
)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: loyalty-programme next-trip offer targeting
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore supermarket chain runs a grocery loyalty
# programme (assume ~1.5M members for illustration). The CRM team wants a
# model that scores each shopper right after checkout and predicts what
# they will buy on the NEXT trip, so the top 20% can receive a targeted
# next-trip offer (e.g. a breakfast-bundle voucher).
#
# Two constraints drive the design:
#   - The model must be AUDITABLE (regulator + internal governance)
#   - The feature engineering must be REPRODUCIBLE (monthly re-training)
#
# Rule-based features meet both. Every feature column has a business
# name you can trace to a specific association rule, and the mining
# pipeline reruns monthly against the previous 90 days of baskets. Your
# run above shows whether they also add accuracy on top of the product
# columns — if they do not, they can still be kept for explainability,
# but the decision should rest on the measured AUC change, not on hope.
#
# BUSINESS IMPACT (illustrative assumptions, not measured figures): if
# better targeting reaches an extra 1% of 1.5M members per month (15,000
# shoppers) and each activated offer contributes S$4 of margin, that is
# ~S$60K/month. Plug in YOUR measured AUC change and the programme's
# real offer economics before quoting a number. The rule-based column
# names also feed a "why did this shopper qualify?" explanation panel.
#
# LIMITATIONS:
#   - Rules captured here are support >= 3%; rarer categories (baby
#     care, pet food) need their own mining run with lower thresholds
#   - Two trips per shopper is a minimal history; production models
#     aggregate many past trips per shopper


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log baseline-vs-rule-enhanced metrics to ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# Logs all four model variants side by side (LR Baseline / LR Combined /
# RF Baseline / RF Combined) plus the product-vs-rule importance split.
# Series = top-15 RF importances so the M4 dashboard can visualise the
# combined-model ranking alongside the rule-quality distribution from
# lesson 03.

lr_combined_lift = (
    results["LR: Combined"]["auc_roc"] - results["LR: Baseline"]["auc_roc"]
)
rf_combined_lift = (
    results["RF: Combined"]["auc_roc"] - results["RF: Baseline"]["auc_roc"]
)

track_run(
    tracker,
    exp_name,
    run_name="rule_features_vs_baseline",
    params={
        "algorithm": "rule_features",
        "implementation": "lr_plus_rf_baseline_vs_combined",
        "n_transactions": len(transactions),
        "n_baseline_features": int(X_baseline.shape[1]),
        "n_rule_features": int(X_rules.shape[1]),
        "n_combined_features": int(X_combined.shape[1]),
        "rf_n_estimators": 200,
        "rf_max_depth": 10,
    },
    scalar_metrics={
        "lr_baseline_auc": float(results["LR: Baseline"]["auc_roc"]),
        "lr_combined_auc": float(results["LR: Combined"]["auc_roc"]),
        "lr_auc_lift_combined_vs_baseline": float(lr_combined_lift),
        "rf_baseline_auc": float(results["RF: Baseline"]["auc_roc"]),
        "rf_combined_auc": float(results["RF: Combined"]["auc_roc"]),
        "rf_auc_lift_combined_vs_baseline": float(rf_combined_lift),
        "lr_baseline_f1": float(results["LR: Baseline"]["f1"]),
        "lr_combined_f1": float(results["LR: Combined"]["f1"]),
        "rf_baseline_f1": float(results["RF: Baseline"]["f1"]),
        "rf_combined_f1": float(results["RF: Combined"]["f1"]),
        "product_importance_share": float(product_importance / total),
        "rule_importance_share": float(rule_importance / total),
    },
    series_metrics={
        "rf_top15_importances": [float(importances[i]) for i in top_idx],
    },
)
print(f"  [tracked] Baseline-vs-combined classifier metrics logged to {exp_name}\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — kailash-ml FeatureEngineer + TrainingPipeline
# ════════════════════════════════════════════════════════════════════════
# This lesson hand-rolled the rule-feature builder, the per-rule binary
# columns, the aggregate (n_rules_triggered / total_lift / max_lift) +
# four full sklearn classifier runs — ~340 lines to internalise rules-as-
# features end-to-end. The destination-first surface in Kailash is two
# engines stacked:
#
#   FeatureEngineer().generate(data, schema, strategies=["interactions"])
#       -> pairwise interaction columns (on 0/1 product columns, these
#          are exactly the co-occurrence features you hand-built)
#   TrainingPipeline(feature_store, registry).train(
#       data, schema, model_spec, eval_spec, experiment_name)
#       -> trained, evaluated, registered model
#
# In production you ship the rule-feature columns (kept verbatim for the
# auditability story above) PLUS the FeatureEngineer-generated set, then
# let TrainingPipeline train + rank LR vs RF vs gradient-boosted variants
# in one call. Manual rule features become one feature group among many,
# which is exactly the spectrum-of-discovery story this lesson opened on.

print("  Destination contract:")
print(
    "    FeatureEngineer().generate(data, schema, strategies=['interactions'])"
    "  -> co-occurrence cols"
)
print(
    "    TrainingPipeline(feature_store, registry).train("
    "data, schema, model_spec, eval_spec, experiment_name)  -> trained model"
)
print(
    f"  Today's run: hand-built {X_rules.shape[1]} rule features, "
    f"trained 4 sklearn variants — top AUC = {max(auc_vals):.4f}"
)
print()
print("  Manual rules are the bottom rung. The engine ships every rung above.\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Engineered numeric features from discovered association rules
  [x] Compared baseline product-presence vs rule-enhanced models
  [x] Built a next-trip target that is not a function of the features,
      and guarded the baseline AUC against target leakage
  [x] Attributed feature importance across product and rule groups
  [x] Identified a production scenario (loyalty next-trip offers) where
      explicit rule features are valued for auditability
  [x] Pointed at the Kailash destination — FeatureEngineer +
      TrainingPipeline — that ships rule features alongside auto-discovered
      co-occurrence structure in one call

  KEY INSIGHT: Association rules are the MANUAL end of a spectrum that
  runs all the way to deep learning.

    Manual rules  ->  Linear factorisation  ->  Nonlinear neural nets
    (this file)      (Ex 7)                    (Ex 8)

  Every technique to the right of this one discovers the same
  co-occurrence structure, just with less human steering and less
  interpretability.

  Next: Exercise 6 moves to UNSTRUCTURED text — TF-IDF and BM25,
  NMF topic modelling, and topic quality evaluation via NPMI coherence.
"""
)

# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
