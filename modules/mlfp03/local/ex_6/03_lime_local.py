# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 6.3: LIME Local Explanations + SHAP Waterfalls
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Use LIME to fit a local linear surrogate around a single prediction
#   - Compare LIME's local weights against SHAP for the same sample
#   - Explain the HIGHEST-risk and LOWEST-risk applications in the test set
#   - Check whether LIME and SHAP agree on each applicant's top-3 drivers
#   - Apply: reason codes for declined loan applicants
#
# PREREQUISITES: 01_shap_global.py (same model, same SHAP bundle).
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — how LIME's local linear surrogate works
#   2. Build — LimeTabularExplainer on the training distribution
#   3. Train — no training; EXPLAIN extreme-risk individual cases
#   4. Visualise — LIME weights and SHAP waterfall side-by-side
#   5. Apply — reason codes for declined applicants (illustrative bank)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
from lime.lime_tabular import LimeTabularExplainer

from shared.mlfp03.ex_6 import (
    OUTPUT_DIR,
    build_shap_explainer,
    print_section,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — LIME's Local Linear Surrogate
# ════════════════════════════════════════════════════════════════════════
# LIME (Ribeiro, Singh, Guestrin, 2016) — "Local Interpretable
# Model-agnostic Explanations":
#
# For a single prediction f(x):
#   1. Sample perturbed rows around x (Gaussian for continuous,
#      discrete bin flips for categorical)
#   2. Score each perturbation with the original model f
#   3. Weight perturbations by proximity to x (exponential kernel)
#   4. Fit a SPARSE LINEAR model on the weighted samples
#   5. The linear coefficients ARE the local feature importances
#
# SHAP vs LIME at a glance:
#   SHAP: axioms + exact for trees + consistent across samples
#   LIME: fast + model-agnostic + easier to explain to non-statisticians
#
# Production rule:
#   Tree model   -> TreeSHAP (exact)
#   Black-box    -> LIME or KernelSHAP (approximate)
#   Mixed stack  -> Both; SHAP as ground truth, LIME as sanity check


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the LIME explainer
# ════════════════════════════════════════════════════════════════════════

bundle = build_shap_explainer()
model = bundle["model"]
X_train = bundle["X_train"]
X_test = bundle["X_test"]
y_proba = bundle["y_proba"]
feature_names: list[str] = bundle["feature_names"]
shap_vals: np.ndarray = bundle["shap_vals"]

# TODO: build a LimeTabularExplainer on the TRAINING data (classification mode)
# Hint: pass training_data, feature_names, class_names, mode, discretize_continuous, random_state
lime_explainer = LimeTabularExplainer(
    training_data=____,
    feature_names=feature_names,
    class_names=["no_default", "default"],
    mode="classification",
    discretize_continuous=True,
    random_state=42,
)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — "TRAIN" = pick the most interesting individuals to explain
# ════════════════════════════════════════════════════════════════════════
# The model declines when P(default) >= 0.5 (bundle["y_pred"]). The
# BORDERLINE applicant is the one whose probability is closest to that
# decision threshold — where an explanation matters most.

DECISION_THRESHOLD = 0.5
risk_order = np.argsort(y_proba)
high_risk_idx = int(risk_order[-1])
low_risk_idx = int(risk_order[0])
# TODO: index of the applicant whose P(default) is CLOSEST to DECISION_THRESHOLD
# Hint: np.argmin of an absolute difference
borderline_idx = ____

print_section("Local Explanation Targets")
for label, idx in [("Highest risk", high_risk_idx), ("Lowest risk", low_risk_idx), ("Borderline", borderline_idx)]:
    print(f"  {label:<13} idx={idx:<6} P(default)={y_proba[idx]:.4f}")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE local explanations (LIME + SHAP side-by-side)
# ════════════════════════════════════════════════════════════════════════


def shap_top(idx: int, k: int = 3) -> list[str]:
    """Names of the k features with the largest |SHAP| for one row."""
    # TODO: indices of the k largest |SHAP| values in row idx
    order = ____
    return [feature_names[j] for j in order]


def lime_top(idx: int, k: int = 3) -> tuple[list[str], list[tuple[str, float]]]:
    """LIME's k strongest features for P(default) + its readable rules."""
    # TODO: explain row idx of X_test for label 1 (default) with 10 features
    # Hint: lime_explainer.explain_instance(row, predict_fn, labels=(1,), num_features=...)
    exp = ____
    ranked = sorted(exp.as_map()[1], key=lambda t: -abs(t[1]))[:k]
    return [feature_names[j] for j, _ in ranked], exp.as_list(label=1)


agreement: dict[str, int] = {}
for label, idx in [("HIGHEST-risk", high_risk_idx), ("LOWEST-risk", low_risk_idx), ("BORDERLINE", borderline_idx)]:
    lime_names, lime_rules = lime_top(idx)
    shap_names = shap_top(idx)
    # TODO: how many of the top-3 names do LIME and SHAP share?
    agreement[label] = ____
    print_section(f"{label} applicant — P(default)={y_proba[idx]:.4f}", char="─")
    print("  LIME rules (weight on P(default)):")
    for rule, weight in lime_rules[:5]:
        print(f"    {rule:<45} {weight:+.4f}")
    print("  SHAP top contributions (log-odds):")
    for j in np.argsort(-np.abs(shap_vals[idx]))[:5]:
        print(f"    {feature_names[j]:<30} value={X_test[idx, j]:>10.2f}  SHAP={shap_vals[idx, j]:+.4f}")
    print(f"  Top-3 agreement LIME vs SHAP: {agreement[label]}/3")

# ── Visual: SHAP bar chart for the borderline applicant ─────────────────
top10 = np.argsort(-np.abs(shap_vals[borderline_idx]))[:10][::-1]
fig = go.Figure(
    go.Bar(
        y=[feature_names[j] for j in top10],
        x=[shap_vals[borderline_idx, j] for j in top10],
        orientation="h",
        marker_color=["#ef4444" if shap_vals[borderline_idx, j] > 0 else "#3b82f6" for j in top10],
    )
)
fig.update_layout(
    title=f"SHAP contributions — borderline applicant (P(default)={y_proba[borderline_idx]:.3f})",
    xaxis_title="SHAP value in log-odds (red = raises risk, blue = lowers risk)",
    height=450,
)
viz_path = OUTPUT_DIR / "ex6_03_lime_shap_borderline.html"
fig.write_html(str(viz_path))
print(f"\n  Saved: {viz_path}")

# ── Checkpoint ──────────────────────────────────────────────────────────
assert y_proba[high_risk_idx] > y_proba[low_risk_idx], "Task 4: high risk must exceed low risk"
assert abs(y_proba[borderline_idx] - DECISION_THRESHOLD) <= abs(
    y_proba[risk_order[len(risk_order) // 2]] - DECISION_THRESHOLD
), "Task 4: borderline must be at least as close to the threshold as the median applicant"
# INTERPRETATION: The agreement counts are the useful number. Where LIME
# and SHAP share most of the top-3, a plain-language reason is safe to
# send; where they disagree, the explanation itself is uncertain and the
# case deserves human review.
print("\n[ok] Checkpoint — local explanations rendered for three risk tiers\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Reason Codes for Declined Applicants (illustrative bank)
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore retail bank wants every declined
# applicant to receive SPECIFIC reasons, not "did not meet our criteria".
# (In the US, "adverse-action notices" are a legal requirement under
# ECOA / Regulation B. Singapore has no equivalent statutory notice rule;
# here it is good practice and in the spirit of the MAS FEAT
# transparency principles.)
#
#     "Your application was declined. The main factors were:
#        1. [feature_1] — your value [v1]
#        2. [feature_2] — ...
#        3. [feature_3] — ..."
#
# Why LIME + SHAP together:
#   - SHAP gives the exact, additive decomposition of the score
#   - LIME gives readable rules ("months_employed <= 40") that translate
#     easily into customer language
#   - Send the notice automatically only when the two agree on the top
#     drivers; route disagreements to an analyst
#
# BUSINESS VALUE (illustrative assumptions): with ~15,000 declines a
# month and ~S$45 of analyst time per manual review, automating the
# agreeing cases saves S$45 × (number of agreeing cases). The agreement
# rate is the number to measure on your own model — computed below for
# the three applicants explained above.

DECLINES_PER_MONTH = 15_000  # illustrative
REVIEW_COST_SGD = 45  # illustrative analyst cost per manual review
full_agreement = sum(1 for v in agreement.values() if v == 3)
print_section("Reason-code feasibility (computed on 3 applicants)", char="─")
for label, n in agreement.items():
    print(f"  {label:<13} LIME/SHAP top-3 overlap: {n}/3")
print(f"  Full agreement on {full_agreement} of {len(agreement)} explained applicants")
print(
    f"  If that rate held, monthly analyst cost avoided ≈ "
    f"S${DECLINES_PER_MONTH * REVIEW_COST_SGD * full_agreement / len(agreement):,.0f} "
    f"(3 applicants is far too few to estimate it — run it on a sample)"
)
#
# LIMITATION: LIME's random perturbation sampling makes it unstable —
# running it with another seed can change the top-3. SHAP is the
# reference; LIME is the plain-language layer.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print_section("WHAT YOU'VE MASTERED")
print(
    """
  [x] Built a LIME explainer on the training distribution
  [x] Explained the highest-, lowest-, and borderline-risk applicants
  [x] Measured LIME/SHAP top-3 agreement per applicant
  [x] Designed a reason-code notice template backed by both methods
  [x] Turned the agreement rate into an (illustrative) S$ estimate

  KEY INSIGHT: Global explanations tell regulators how the MODEL works;
  local explanations tell individual customers why THEIR application
  was declined — one explanation per decision.

  Next: 04_shap_interactions.py — move beyond single-feature effects and
  uncover which FEATURE PAIRS the model uses together.
"""
)
