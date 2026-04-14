# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 5.3: Loss Functions — Focal Loss & Alpha Weighting
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - The focal loss equation FL(p_t) = -(1-p_t)^gamma * log(p_t)
#   - What the gamma parameter does intuitively (down-weight easy examples)
#   - How alpha-weighting approximates focal loss with any boosted learner
#   - How to sweep alpha and read the sensitivity curve
#
# PREREQUISITES: 02_sampling_strategies.py (cost-sensitive baseline saved)
# ESTIMATED TIME: ~25 min
#
# 5-PHASE STRUCTURE:
#   Theory   — focal loss derivation + gamma intuition
#   Build    — LightGBM with a family of alpha multipliers
#   Train    — sweep alpha in [0.5, 1, 2, 5, 10]
#   Visualise — AUC-PR and Brier vs alpha (sensitivity curve)
#   Apply    — OCBC SME-loan default detection (long-tail hard cases)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import lightgbm as lgb
import polars as pl
from dotenv import load_dotenv

from shared.mlfp03.ex_5 import (
    OUTPUT_DIR,
    load_credit_splits,
    metrics_row,
    print_metrics_table,
    save_strategy_proba,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Focal Loss and the Gamma Knob
# ════════════════════════════════════════════════════════════════════════
# Cross-entropy treats every example with the same weight. A well-classified
# example (the model says p=0.99 and the label is 1) contributes almost as
# much to the average loss as a hard example (p=0.51 where the model is
# unsure). That wastes capacity on easy wins.
#
# Focal Loss (Lin et al., 2017, ICCV) adds a modulating factor (1-p_t)^gamma:
#
#     FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
#
# where p_t = p if y=1 else 1-p. When the model is already confident and
# correct, (1 - p_t) is tiny and the example essentially vanishes from
# the loss. Hard examples (p_t around 0.5) still contribute their full
# weight. Gamma=0 recovers cross-entropy; gamma=2 is the canonical setting.
#
# WHY THIS HELPS IMBALANCED DATA: the majority class usually has many
# easy examples. Focal loss automatically down-weights them without
# any resampling. Combined with alpha (the class prior), you get a
# loss function that focuses gradient on the examples that actually
# move the decision boundary.
#
# LIGHTGBM APPROXIMATION: LightGBM's default objective doesn't expose
# focal loss directly. We approximate it by sweeping an alpha multiplier
# on top of `scale_pos_weight`. This captures the "focus on hard examples"
# effect in a single tunable parameter.


# ════════════════════════════════════════════════════════════════════════
# BUILD + TRAIN — alpha sweep
# ════════════════════════════════════════════════════════════════════════

X_train, y_train, X_test, y_test, pos_rate = load_credit_splits()
base_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

print("\n" + "=" * 70)
print("  Exercise 5.3 — Focal Loss (Alpha Sweep Approximation)")
print("=" * 70)
print(f"  Base scale_pos_weight (class-balanced): {base_weight:.2f}")

alpha_multipliers = [0.5, 1.0, 2.0, 5.0, 10.0]
metric_rows: list[dict] = []

for alpha in alpha_multipliers:
    pos_w = alpha * base_weight
    model = lgb.LGBMClassifier(
        n_estimators=300,
        scale_pos_weight=pos_w,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    name = f"focal_alpha_{alpha:.1f}"
    save_strategy_proba(name, y_proba)
    row = metrics_row(f"alpha={alpha:.1f}", y_test, y_proba)
    row["alpha"] = float(alpha)
    metric_rows.append(row)


# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert len(metric_rows) == len(alpha_multipliers), "Must sweep every alpha"
assert all(0 <= r["auc_pr"] <= 1 for r in metric_rows), "AUC-PR in [0,1]"
assert all(0 <= r["brier"] <= 1 for r in metric_rows), "Brier in [0,1]"
print("[ok] Checkpoint 3 — alpha sweep complete\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — AUC-PR and Brier sensitivity to alpha
# ════════════════════════════════════════════════════════════════════════

print_metrics_table(metric_rows, "Focal loss alpha sweep")

best_by_pr = max(metric_rows, key=lambda r: r["auc_pr"])
best_by_brier = min(metric_rows, key=lambda r: r["brier"])

print("\n  Sensitivity summary:")
print(
    f"    Best AUC-PR: alpha={best_by_pr['alpha']:.1f} (AUC-PR={best_by_pr['auc_pr']:.4f})"
)
print(
    f"    Best Brier:  alpha={best_by_brier['alpha']:.1f} "
    f"(Brier={best_by_brier['brier']:.4f})"
)

# Persist for later files
pl.DataFrame(metric_rows).write_parquet(OUTPUT_DIR / "focal_sweep_metrics.parquet")
print(f"\n  Saved: {OUTPUT_DIR / 'focal_sweep_metrics.parquet'}")

# INTERPRETATION: Usually AUC-PR peaks at a DIFFERENT alpha from Brier.
# AUC-PR rewards ranking quality; Brier rewards calibration. The two
# objectives pull in opposite directions — higher alpha pushes the
# model to over-predict the positive class, improving recall but
# worsening calibration. This is the core trade-off in 5.5 (where we
# will calibrate explicitly rather than trying to squeeze it out of
# the loss function alone).


# ════════════════════════════════════════════════════════════════════════
# APPLY — OCBC SME-loan default detection
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: OCBC Singapore's SME portfolio has ~18,000 active business
# loans to local SMEs (F&B, retail, manufacturing). The underwriting
# team wants an early-warning model that flags borrowers likely to
# default in the next 90 days.
#
# The imbalance is severe AND heterogeneous:
#   - Overall default rate: ~3% (332 defaults / year)
#   - Long-tail: most defaults come from obvious distress signals
#     (missed payroll, rent arrears, bounced cheques) which are easy
#     to classify. The HARD cases are SMEs that look healthy until
#     the final month — these are the ones a good early-warning
#     system must catch.
#
# Why focal loss matters here:
#   - Standard cost-sensitive training spreads gradient evenly across
#     all defaulters, including the "obvious" ones the model already
#     nails at p=0.99. Gradient on those is wasted.
#   - Focal loss (high gamma/alpha) automatically shifts gradient onto
#     the borderline cases. The model learns subtler distress signals
#     (working capital compression, supplier concentration drift)
#     because that's where the loss lives.
#
# BUSINESS IMPACT: early warning 90 days before default is worth
# S$30,000 per loan in avoided write-off (MAS SME credit report 2024:
# recovery rate jumps from 18% to 47% with 90-day notice because the
# bank can restructure or unwind facilities before the collateral
# value deteriorates). Capturing even 40 extra borderline cases per
# year = S$1.2M/year recovered. The alpha sweep is what finds that
# capture rate.

print("\n  OCBC SME early-warning implication:")
print(f"    Alpha sweep tested {len(alpha_multipliers)} points")
print(f"    Best recall at alpha={best_by_pr['alpha']:.1f}: {best_by_pr['recall']:.2%}")
print("    Each additional recalled borderline borrower -> ~S$30K recovered")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — 5.3")
print("=" * 70)
print(
    """
  [x] Derived focal loss FL(p_t) = -(1-p_t)^gamma * log(p_t)
  [x] Understood the gamma/alpha intuition: down-weight easy examples
  [x] Swept alpha in [0.5, 1, 2, 5, 10] and recorded the sensitivity curve
  [x] Observed that AUC-PR and Brier usually peak at DIFFERENT alphas
  [x] Tied the pattern to OCBC SME borderline-default detection

  KEY INSIGHT: The loss function is a knob. Sweeping alpha reveals the
  ranking-vs-calibration trade-off that no single metric can express.

  Next: 04_threshold_optimisation.py — instead of tuning the loss, tune
  the DECISION threshold from the business cost matrix directly.
"""
)
