# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 5.1: Metrics Taxonomy & The Imbalanced Baseline
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Why accuracy is a lie on an imbalanced dataset
#   - The complete classification metrics taxonomy: precision, recall,
#     specificity, F1, AUC-ROC, AUC-PR, Brier
#   - How to pick the metric that matches the business cost structure
#   - How to visualise the confusion matrix so non-technical stakeholders
#     can see the imbalance problem at a glance
#
# PREREQUISITES: MLFP03 Exercise 4 (gradient boosting, AUC-PR)
# ESTIMATED TIME: ~30 min
#
# 5-PHASE STRUCTURE:
#   Theory   — why accuracy fails and which metric to use when
#   Build    — train a "do-nothing" LightGBM baseline
#   Train    — fit on the imbalanced training split
#   Visualise — confusion matrix + per-metric bar chart
#   Apply    — a Singapore retail bank's consumer-credit scorecard triage
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import lightgbm as lgb
import plotly.graph_objects as go
import polars as pl
from dotenv import load_dotenv

from shared.mlfp03.ex_5 import (
    DEFAULT_COSTS,
    OUTPUT_DIR,
    load_credit_splits,
    metrics_row,
    print_metrics_table,
    save_strategy_proba,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Accuracy Lies
# ════════════════════════════════════════════════════════════════════════
# Imagine you are the Chief Risk Officer of a Singapore retail bank. Every
# day, 300 consumer loan applications arrive. ~13% of approved applicants
# will eventually default. If you build a model that says "no default" for
# every single applicant, you get ~87% accuracy and zero defaults caught.
# Your CEO would fire you — but an accuracy dashboard would congratulate
# you. (Its F1 is 0: no true positives, so precision and recall are 0.)
#
# This is why we need a complete metrics taxonomy BEFORE we even pick a
# model. Each metric answers a different business question:
#
#   Precision  — "Of the applicants I flagged, how many actually defaulted?"
#                (High precision = few false declines = happy salespeople)
#
#   Recall     — "Of the actual defaulters, how many did I catch?"
#                (High recall = few missed defaults = happy CRO)
#
#   Specificity— "Of the good customers, how many did I correctly clear?"
#                (High specificity = low false-alarm rate on good applicants)
#
#   F1         — Harmonic mean of precision + recall. Balances both.
#
#   AUC-ROC    — Ranking quality across ALL thresholds. Insensitive to
#                class imbalance but can be misleadingly optimistic when
#                the majority class dominates.
#
#   AUC-PR     — Ranking quality focused on the positive class.
#                THIS IS THE METRIC TO REPORT for rare-event problems.
#
#   Brier score— Proper scoring rule for calibrated probabilities.
#                (p_predicted = 0.2 should mean ~20% default in reality)
#
# Rule of thumb for imbalanced credit data: report AUC-PR + Brier to the
# risk committee, and quote accuracy only next to the majority-class
# baseline it has to beat.
#
# WHY RAW LIGHTGBM HERE: every technique in Exercise 5 changes the training
# LOSS (class weights, per-row sample weights, a custom focal objective) or
# post-processes the probabilities. kailash-ml's TrainingPipeline.train()
# takes a ModelSpec(model_class, hyperparameters) and exposes no per-row
# sample_weight / init_score hook, so these loss-level experiments call
# LightGBM directly; 05_calibration.py closes with the kailash-ml
# km.diagnose engine on the final model.


# ════════════════════════════════════════════════════════════════════════
# BUILD — the baseline classifier
# ════════════════════════════════════════════════════════════════════════

X_train, y_train, X_test, y_test, pos_rate = load_credit_splits()
imbalance_ratio = (1 - pos_rate) / pos_rate

print("\n" + "=" * 70)
print("  Exercise 5.1 — Metrics Taxonomy & Baseline")
print("=" * 70)
print(f"  Default rate:     {pos_rate:.2%}")
print(f"  Imbalance ratio:  {imbalance_ratio:.0f}:1 (non-default : default)")
print(f"  Train rows:       {X_train.shape[0]:,}")
print(f"  Test rows:        {X_test.shape[0]:,}")
print(f"  Cost matrix:      FP=S${DEFAULT_COSTS.fp:,.0f}, FN=S${DEFAULT_COSTS.fn:,.0f}")


# ════════════════════════════════════════════════════════════════════════
# TRAIN — LightGBM with zero imbalance handling
# ════════════════════════════════════════════════════════════════════════
# This is the "null strategy": use a strong off-the-shelf learner and
# change nothing. Its failure mode on the imbalanced test set tells you
# exactly how much the later techniques have to claw back.

# TODO: Build a default LGBMClassifier with n_estimators=300, random_state=42
# Hint: verbose=-1 suppresses LightGBM's chatter
baseline = lgb.LGBMClassifier(____)
# TODO: Fit the baseline on X_train, y_train
____

# TODO: Predict POSITIVE-CLASS probabilities on X_test
# Hint: predict_proba returns shape (n, 2); you want column index 1
y_proba_base = ____
save_strategy_proba("baseline", y_proba_base)


# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert 0 < pos_rate < 0.5, "Positive class must be the minority class"
assert y_proba_base.shape[0] == X_test.shape[0], "Proba vector must match test rows"
assert y_proba_base.min() >= 0 and y_proba_base.max() <= 1, "Probabilities in [0,1]"
print("\n[ok] Checkpoint 1 — baseline trained, probabilities saved\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — full metrics taxonomy + confusion matrix
# ════════════════════════════════════════════════════════════════════════

# TODO: Call metrics_row() with name="Baseline (no correction)" and threshold=0.5
row = ____
print_metrics_table([row], "Baseline metrics at threshold=0.5")

print(
    f"\n  Confusion matrix (threshold=0.5):\n"
    f"                    Predicted 0    Predicted 1\n"
    f"       Actual 0   {row['tn']:>12,}   {row['fp']:>12,}\n"
    f"       Actual 1   {row['fn']:>12,}   {row['tp']:>12,}"
)

# TODO: Accuracy of the trivial "approve everyone" (predict 0) policy
# Hint: it equals the share of NON-defaulters in y_test
majority_accuracy = ____
print(f"\n  Majority-class ('approve everyone') accuracy: {majority_accuracy:.4f}")
print(f"  Baseline LightGBM accuracy @0.5:             {row['accuracy']:.4f}")
print(f"  Baseline recall @0.5:                        {row['recall']:.4f}")
print(
    f"  -> accuracy beats 'approve everyone' by only "
    f"{row['accuracy'] - majority_accuracy:+.4f}, while catching "
    f"{row['recall']:.0%} of the defaulters."
)
# INTERPRETATION: Compare the accuracy with the majority-class line above
# it — that is the number accuracy has to beat. Recall tells you how many
# defaulters the 0.5 threshold actually catches. Each missed default costs
# DEFAULT_COSTS.fn (illustratively S$10,000).

# ── Visual: confusion matrix heatmap (the picture for non-technical readers)
cm_fig = go.Figure(
    data=go.Heatmap(
        # TODO: 2x2 grid [[TN, FP], [FN, TP]] from the metrics row
        z=____,
        x=["Predicted: repay", "Predicted: default"],
        y=["Actual: repaid", "Actual: defaulted"],
        text=[[f"TN {row['tn']:,}", f"FP {row['fp']:,}"], [f"FN {row['fn']:,}", f"TP {row['tp']:,}"]],
        texttemplate="%{text}",
        colorscale="Blues",
    )
)
cm_fig.update_layout(title="Baseline confusion matrix @ threshold 0.5", height=420)
cm_path = OUTPUT_DIR / "ex5_01_confusion_matrix.html"
cm_fig.write_html(str(cm_path))
print(f"\n  Saved: {cm_path}")

# ── Visual: the whole metrics taxonomy as one bar chart
metric_names = ["accuracy", "precision", "recall", "specificity", "f1", "auc_roc", "auc_pr", "brier"]
bar_fig = go.Figure(go.Bar(x=metric_names, y=[row[m] for m in metric_names], marker_color="#6366f1"))
bar_fig.add_hline(y=majority_accuracy, line_dash="dot", annotation_text="majority-class accuracy")
bar_fig.update_layout(title="Baseline: one model, eight different stories", yaxis_title="Score", height=420)
bar_path = OUTPUT_DIR / "ex5_01_metric_taxonomy.html"
bar_fig.write_html(str(bar_path))
print(f"  Saved: {bar_path}")

print("\n  When to use which metric:")
print("    Accuracy     — NEVER for imbalanced data")
print("    Precision    — when FP is expensive (spam, fraud investigation)")
print("    Recall       — when FN is expensive (cancer, credit default)")
print("    F1           — when you need to balance precision + recall")
print("    AUC-ROC      — ranking quality, imbalance-insensitive")
print("    AUC-PR       — ranking quality for RARE events (use this)")
print("    Brier        — probability calibration (proper scoring rule)")

# Save the per-metric table to OUTPUT_DIR so later files can read it back
metrics_df = pl.DataFrame([row])
metrics_df.write_parquet(OUTPUT_DIR / "baseline_metrics.parquet")
print(f"\n  Saved: {OUTPUT_DIR / 'baseline_metrics.parquet'}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — A Singapore retail bank's consumer-credit scorecard triage
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore retail bank processes ~100,000
# unsecured personal-loan applications per year. The underwriting team
# uses a scorecard model as the first-pass filter.
#
# Illustrative cost structure (see shared.mlfp03.ex_5.DEFAULT_COSTS):
#   - Missed default (FN):  ~S$10,000 charged-off principal
#   - False decline (FP):   ~S$1,500 forgone interest margin
#
# Why this matters: the CRO needs ONE number to show the board. "Our
# scorecard has F1=0.2" loses budget; "our scorecard misses S$X of
# defaults per year at the current threshold" moves the needle. The
# code below computes X for THIS model — the rest of the exercise tries
# to move it by changing the loss, the threshold and the calibration.

n_def_test = int(y_test.sum())
n_missed_test = int(((y_test == 1) & (y_proba_base < 0.5)).sum())
miss_rate = n_missed_test / max(n_def_test, 1)
print("\n  Singapore retail-bank implication (illustrative volumes):")
print(f"    Defaults in test set:       {n_def_test:,}")
print(f"    Missed by baseline @0.5:    {n_missed_test:,} ({miss_rate:.0%})")
print(
    f"    Scaled to 100K apps/year:   ~S${DEFAULT_COSTS.fn * n_def_test * miss_rate * (100_000 / len(y_test)):,.0f} lost"
)
print("    Next file (02_sampling_strategies.py) adds SMOTE and cost-sensitive")
print("    learning and compares them on ranking AND calibration.")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — 5.1")
print("=" * 70)
print(
    """
  [x] Loaded the Singapore credit scoring dataset through MLFPDataLoader
  [x] Trained a baseline LightGBM with zero imbalance handling
  [x] Built the complete metrics taxonomy (precision/recall/specificity/
      F1/AUC-ROC/AUC-PR/Brier)
  [x] Saved the baseline probability vector for later technique files
  [x] Translated the baseline's failure into S$ lost per year for a bank

  KEY INSIGHT: Accuracy is the wrong metric for rare events. AUC-PR +
  Brier is the right pair to report. Everything in this exercise after
  this file is a different way of MOVING those two numbers.

  Next: 02_sampling_strategies.py — SMOTE vs cost-sensitive learning.
"""
)
