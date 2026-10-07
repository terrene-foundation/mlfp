# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 4.6: CatBoost on Native Categoricals — No Invented
#                         Order
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Why ordinal encoding is a lie for unordered categories ("divorced < 3"
#     is not a fact)
#   - CatBoost's native handling: ordered target statistics + ordered
#     boosting — the leakage guard is the point
#   - The honest comparison: same split, same budget, ordinal vs native
#   - When native categoricals actually pay (cardinality, rare levels)
#
# PREREQUISITES: 03_lightgbm_catboost.py
# ESTIMATED TIME: ~25 min
#
# 5-PHASE STRUCTURE:
#   Theory   — the invented-order problem; ordered target statistics
#   Build    — CatBoost with cat_features on the SAME leak-free split
#   Train    — native vs ordinal, identical budgets
#   Visualise — metrics + feature-importance contrast
#   Apply    — a bureau feature review: which columns deserve categories
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from shared.mlfp03.ex_4 import (
    OUTPUT_DIR,
    as_catboost_categoricals,
    categorical_feature_indices,
    evaluate_classifier,
    make_catboost,
    prepare_credit_split,
    print_metrics,
)

# ── THEORY — ordinal encoding invents an order ───────────────────────────
# 02–03 ordinal-encode the nine string columns (gender, race, nationality,
# region, loan_purpose, marital_status, education, housing_type,
# application_channel) because XGBoost and LightGBM need numbers. That
# encoding says marital_status ∈ {0,1,2,3} — and a tree is now allowed to
# split on "marital_status ≤ 1.5", an order NOBODY asserted. On genuinely
# unordered categories the invented order adds noise the model must untangle.
#
# CatBoost's native path keeps them as CATEGORIES and encodes each level by
# its target statistic — but computed in a random row ORDER, where each
# row's encoding sees only rows BEFORE it (ordered target statistics). That
# ordering is the leakage guard: a naive target encoding uses the row's own
# label, which leaks; CatBoost's never does. Ordered boosting applies the
# same discipline to the gradient computation.

split = prepare_credit_split()
X_train, y_train = split["X_train"], split["y_train"]
X_test, y_test = split["X_test"], split["y_test"]
feature_names = split["feature_names"]

cat_idx = categorical_feature_indices(feature_names)
cat_names = [feature_names[i] for i in cat_idx]
print(f"  Native categorical columns ({len(cat_idx)}): {cat_names}")

# ── BUILD + TRAIN — same split, same budget, two encodings ───────────────
ITERS, LR, DEPTH = 500, 0.1, 6

print("\n  CatBoost on ORDINAL-encoded numerics (the 02–03 pipeline output)...")
ord_model = make_catboost(iterations=ITERS, learning_rate=LR, depth=DEPTH)
ord_model.fit(X_train, y_train)
m_ord = evaluate_classifier(y_test, ord_model.predict_proba(X_test)[:, 1])
print_metrics("CatBoost ordinal", m_ord)

print("  CatBoost with NATIVE categories (cat_features, codes as unordered)...")
X_train_cat = as_catboost_categoricals(X_train, cat_idx)
X_test_cat = as_catboost_categoricals(X_test, cat_idx)
nat_model = make_catboost(iterations=ITERS, learning_rate=LR, depth=DEPTH)
nat_model.fit(X_train_cat, y_train, cat_features=cat_idx)
m_nat = evaluate_classifier(y_test, nat_model.predict_proba(X_test_cat)[:, 1])
print_metrics("CatBoost native", m_nat)

# ── Checkpoint 1 ──────────────────────────────────────────────────────────
assert m_ord["auc_roc"] > 0.7 and m_nat["auc_roc"] > 0.7, "both variants must learn"
delta = m_nat["auc_pr"] - m_ord["auc_pr"]
print(f"\n  Native minus ordinal: AUC-PR {delta:+.4f}, AUC-ROC {m_nat['auc_roc'] - m_ord['auc_roc']:+.4f}")
print("[ok] Checkpoint 1 — both variants trained on identical splits and budgets")

# ── VISUALISE — metrics + what native mode does to importances ────────────
imp_ord = ord_model.get_feature_importance()
imp_nat = nat_model.get_feature_importance()
top = np.argsort(imp_nat)[::-1][:10]

fig = go.Figure()
fig.add_trace(go.Bar(
    y=[feature_names[i] for i in top], x=imp_ord[top],
    name="ordinal-encoded", orientation="h", marker_color="#64748B",
))
fig.add_trace(go.Bar(
    y=[feature_names[i] for i in top], x=imp_nat[top],
    name="native categories", orientation="h", marker_color="#0D9488",
))
fig.update_layout(
    title="Feature importance: native categoricals stop splitting on invented order",
    xaxis_title="CatBoost importance", barmode="group", height=480,
)
fig.write_html(str(OUTPUT_DIR / "ex4_06_catboost_native.html"))
print(f"  Saved: {OUTPUT_DIR / 'ex4_06_catboost_native.html'}")
print("[ok] Checkpoint 2 — importance contrast saved")

# ── APPLY — a bureau feature review ──────────────────────────────────────
worse_or_better = "paid" if delta > 0.002 else ("cost" if delta < -0.002 else "washed out")
print(
    f"\n  APPLY: on this dataset, native categoricals {worse_or_better} "
    f"(AUC-PR {delta:+.4f}). The review rule of thumb: keep native handling "
    f"for HIGH-cardinality unordered columns (nationality, loan_purpose, "
    f"application_channel — invented order is pure noise there); true "
    f"ordinals like education levels are honestly ordered and can stay "
    f"numeric either way. And whatever you choose, the comparison must be "
    f"this one: same split, same budget, both encodings — never a vendor "
    f"benchmark with different data."
)

# REFLECTION
print(
    """
  What you've mastered:
    ✓ Ordinal encoding asserts an order nobody asserted — trees then split
      on the invention
    ✓ CatBoost native: ordered target statistics + ordered boosting; the
      row-ordering IS the leakage guard
    ✓ The fair fight: same leak-free split, same budget, both encodings
    ✓ When native pays: high-cardinality unordered columns

  Exercise 4 complete: theory (01), XGBoost (02), LightGBM/CatBoost (03),
  tuning (04), AdaBoost warm-up (05), native categoricals (06).
"""
)
