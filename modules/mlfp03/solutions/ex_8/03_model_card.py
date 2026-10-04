# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 8.3: Mitchell et al. Model Cards for Regulated ML
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Structure of the Mitchell et al. Model Card (9 sections)
#   - Generating every number in a card from measurements, not prose
#   - Measuring disaggregated fairness (race, gender, age band) for the card
#   - Distinguishing "intended use" from "out of scope"
#   - Rendering the card as a visual summary for non-technical reviewers
#
# PREREQUISITES: 01_conformal_prediction.py (this file re-trains the same
# model so it runs on its own).
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory     — why model cards exist and what goes in them
#   2. Build      — measure performance, coverage and per-group fairness
#   3. Train      — (no new training) render the card + its evidence file
#   4. Visualise  — a one-page model-card summary
#   5. Apply      — the questions the card hands to a risk committee
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
from datetime import datetime

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from shared.mlfp03.ex_8 import (
    BASELINE_PARAMS,
    CARD_EVIDENCE_PATH,
    CARD_PATH,
    COST_FN_SGD,
    COST_FP_SGD,
    DECISION_THRESHOLD,
    OUTPUT_DIR,
    conformal_on_test,
    evaluate_classification,
    fairness_report,
    fairness_summary,
    load_credit_split,
    train_calibrated_model,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Model Cards Exist
# ════════════════════════════════════════════════════════════════════════
# Mitchell et al. (2019) proposed Model Cards so that a trained model
# ships with a short document saying what it is for, how it was
# evaluated, and how well it works FOR DIFFERENT GROUPS of people. Their
# motivation included documented cases such as commercial face-analysis
# systems whose error rates differed sharply by skin type and gender
# (Buolamwini & Gebru, 2018) — differences an aggregate accuracy hides.
#
# The card is a contract between the ML team and everyone downstream:
# what the model does, where it was validated, where it MUST NOT be
# used, and how to tell when it stops working.
#
# Regulatory context (check the current texts before relying on them):
#   - EU AI Act: credit scoring of individuals is a high-risk use; such
#     systems need technical documentation and transparency information.
#   - Singapore's FEAT principles (MAS, 2018) ask firms using AI in
#     financial decisions to be able to justify, explain and account for
#     them. They do not prescribe a model card — a card is one practical
#     way to evidence those principles.
#   - NIST AI RMF: documentation of this kind supports its MAP and
#     MEASURE functions.
#
# THE GOLDEN RULE: every number in a card must come from a measurement
# of THIS model. A card that says "fairness within band" without the
# measurement is worse than no card — it is a false assurance.
#
# The 9 SECTIONS (Mitchell et al. 2019):
#   1. Model details        — type, version, date, contact
#   2. Intended use         — primary users, primary purpose, scope
#   3. Factors              — groups, instruments, environments
#   4. Metrics              — evaluation measures + decision thresholds
#   5. Evaluation data      — source, preprocessing, motivation
#   6. Training data        — same categories as evaluation
#   7. Quantitative analyses— aggregate AND disaggregated results
#   8. Ethical considerations— risks, mitigations, dual-use concerns
#   9. Caveats & recommendations — out-of-scope uses, future work


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: measure everything the card will state
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP03 Exercise 8.3 — Model Card Generation")
print("=" * 70)

split = load_credit_split()
X_train, y_train = split["X_train"], split["y_train"]
X_test, y_test = split["X_test"], split["y_test"]
feature_names = split["feature_names"]

calibrated_model = train_calibrated_model(X_train, y_train, feature_names)
y_proba = calibrated_model.predict_proba(X_test)[:, 1]

# Threshold-free metrics (AUC, Brier) plus the confusion-based ones at
# the cost-derived decision threshold the lender would actually use.
metrics = evaluate_classification(y_test, y_proba, threshold=DECISION_THRESHOLD)
conformal = conformal_on_test(y_test, y_proba, alpha=0.10)

# Disaggregated fairness: per-group rates, then one summary row per
# protected attribute (race, gender, age band).
group_table = fairness_report(y_test, y_proba, split["test_groups"], DECISION_THRESHOLD)
fair = fairness_summary(group_table)

print(
    f"\nAUC-ROC={metrics['auc_roc']:.4f}  Brier={metrics['brier']:.4f}  "
    f"Coverage={conformal['coverage']:.3f} at α={conformal['alpha']}"
)
print(f"Decision threshold p(default) >= {DECISION_THRESHOLD:.3f}")
print("\n=== Per-group rates on the test set ===")
print(group_table)
print("\n=== Fairness summary per attribute ===")
print(fair)


# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert metrics["auc_roc"] > 0.5, "Task 2: Model should beat random"
assert set(fair["attribute"].to_list()) == {"race", "gender", "age_band"}, (
    "Task 2: fairness must be measured for race, gender and age band"
)
assert fair["disparate_impact"].is_between(0, 1).all(), "Task 2: DI is a ratio in [0, 1]"
print("\n[ok] Checkpoint 1 — performance, coverage and fairness measured\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — render the card from the measurements (no new training)
# ════════════════════════════════════════════════════════════════════════
# Every number below is an f-string over a measured value. The fairness
# verdicts are computed too: the four-fifths rule of thumb (DI >= 0.8).

fairness_lines = []
for row in fair.iter_rows(named=True):
    verdict = "meets" if row["disparate_impact"] >= 0.8 else "FAILS"
    fairness_lines.append(
        f"- **{row['attribute']}**: disparate impact {row['disparate_impact']:.2f} "
        f"({verdict} the four-fifths rule of thumb); TPR gap {row['tpr_gap']:.3f}; "
        f"FPR gap {row['fpr_gap']:.3f}; max |mean prediction − base rate| "
        f"{row['max_calibration_gap']:.3f}"
    )
group_lines = [
    f"| {r['attribute']} | {r['group']} | {r['n']:,} | {r['base_rate']:.3f} | "
    f"{r['flag_rate']:.3f} | {r['tpr']:.3f} | {r['fpr']:.3f} | {r['mean_pred']:.3f} |"
    for r in group_table.iter_rows(named=True)
]
failing = fair.filter(pl.col("disparate_impact") < 0.8)["attribute"].to_list()

model_card = f"""
# Model Card: Singapore Credit Default Prediction

## 1. Model Details
- **Model type**: LightGBM classifier ({BASELINE_PARAMS['n_estimators']} trees, max depth
  {BASELINE_PARAMS['max_depth']}) trained with kailash-ml TrainingPipeline, isotonic-calibrated
  with TrainingPipeline.calibrate on a held-out 20% of the training rows
- **Version**: registered as a new version in the exercise registry each run (see 8.4)
- **Date**: {datetime.now().strftime("%Y-%m-%d")}
- **Framework**: kailash-ml (Terrene Foundation), Apache 2.0
- **Contact**: model-risk@example.org

## 2. Intended Use
- **Primary use**: teaching example of default-risk scoring for unsecured
  consumer credit applications in Singapore.
- **Primary users**: credit-risk analysts; an underwriting workflow that
  routes ambiguous applications (conformal set {{0, 1}}) to a human.
- **Out of scope**:
  - Corporate/commercial credit decisions
  - Applicants outside the population the data describes
  - Regulatory capital calculation (needs dedicated PD/LGD/EAD models)
  - Any decision without a human-review path

## 3. Factors
- **Groups evaluated**: race, gender, age band (<30, 30-44, 45-59, 60+)
- **Note**: race, gender and age are MODEL INPUTS in this version.

## 4. Metrics
- **Evaluation measures**: AUC-ROC, AUC-PR, Brier score, log loss; precision,
  recall and F1 at the decision threshold
- **Decision threshold**: flag as likely default when p >= {DECISION_THRESHOLD:.3f}
  (= c_FP / (c_FP + c_FN) with illustrative costs S${COST_FP_SGD:,.0f} / S${COST_FN_SGD:,.0f})
- **Uncertainty**: split-conformal prediction sets at α={conformal['alpha']}
- **Fairness measures**: disparate impact (lowest / highest group flag rate),
  TPR and FPR gaps (equalised odds), per-group calibration gap

## 5. Evaluation Data
- **Source**: the 20% test split of `sg_credit_scoring.parquet` (course dataset)
- **Size**: {X_test.shape[0]:,} applications
- **Preprocessing**: kailash-ml PreprocessingPipeline, ordinal encoding,
  fitted on the training split only (the test split was held out first);
  `customer_id` and the post-outcome `future_default_indicator` removed
- **Motivation / limit**: a seeded RANDOM split, stratified on `default` — it does not test how the
  model performs on future applicants (no time-ordered evaluation)

## 6. Training Data
- **Source**: the 80% training split of the same file
- **Size**: {X_train.shape[0]:,} applications, {X_train.shape[1]} features
- **Target**: binary default ({y_train.mean():.1%} positive rate)

## 7. Quantitative Analyses
### Aggregate (test split)
- **AUC-ROC**: {metrics['auc_roc']:.4f}
- **AUC-PR**: {metrics['auc_pr']:.4f} (a random ranking scores ≈ {y_test.mean():.3f})
- **Brier Score**: {metrics['brier']:.4f} (base-rate forecast: {y_test.mean() * (1 - y_test.mean()):.4f})
- **At threshold {DECISION_THRESHOLD:.3f}**: precision {metrics['precision']:.3f}, recall
  {metrics['recall']:.3f}, F1 {metrics['f1']:.3f}

### Uncertainty Quantification
- **Method**: split conformal prediction (q̂ from half the test split)
- **Coverage**: {conformal['coverage']:.1%} at α={conformal['alpha']} on the other half
- **Guarantee**: P(Y ∈ C(X)) ≥ 1-α on average over applicants (marginal),
  assuming future applicants are exchangeable with the calibration set

### Disaggregated fairness (measured, test split)
{chr(10).join(fairness_lines)}

| attribute | group | n | base rate | flag rate | TPR | FPR | mean p |
|---|---|---|---|---|---|---|---|
{chr(10).join(group_lines)}

## 8. Ethical Considerations
- **Protected attributes as inputs**: race and gender are used by the model;
  that use must be justified or removed (removing them alone does not
  guarantee fairness — proxies remain).
- **Measured disparities**: attributes failing the four-fifths rule of
  thumb: {', '.join(failing) if failing else 'none'}. Compare the base-rate
  column: when groups default at different rates, a calibrated model
  cannot also equalise flag rates and error rates (the impossibility
  result from Exercise 6). Which criterion to prioritise is a policy
  decision for the lender, not a modelling default.
- **Dual use**: outputs MUST NOT be used for marketing segmentation.
- **Contestability**: adverse decisions need reasons an applicant can act
  on (Exercise 6 shows SHAP / LIME explanations) and a human-appeal path.

## 9. Caveats and Recommendations
- **Drift**: re-check inputs with DriftMonitor (8.2); retrain on severe
  input drift or when measured AUC-PR falls below {metrics['auc_pr'] * 0.9:.4f}
  (a 10% degradation floor chosen for this example)
- **Exchangeability**: conformal coverage is void once applicants shift;
  recalibrate q̂ on recent labelled data first
- **Fairness**: re-measure the table above on every retrain
"""

CARD_PATH.write_text(model_card)
evidence = {
    "created": datetime.now().isoformat(),
    "decision_threshold": DECISION_THRESHOLD,
    "metrics": metrics,
    "conformal": conformal,
    "fairness_summary": fair.to_dicts(),
    "fairness_groups": group_table.to_dicts(),
}
CARD_EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2))
print(f"Saved: {CARD_PATH}\nSaved: {CARD_EVIDENCE_PATH}")


# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert CARD_PATH.exists(), "Task 3: Model card should be written"
required_sections = [
    "Model Details",
    "Intended Use",
    "Factors",
    "Metrics",
    "Evaluation Data",
    "Training Data",
    "Quantitative Analyses",
    "Ethical Considerations",
    "Caveats and Recommendations",
]
for section in required_sections:
    assert section in model_card, f"Task 3: Missing section '{section}'"
assert "AUC-ROC" in model_card
assert "Coverage" in model_card
assert f"disparate impact {fair['disparate_impact'][0]:.2f}" in model_card, (
    "Task 3: the card must quote the measured disparate impact"
)
print("\n[ok] Checkpoint 2 — all 9 sections present, numbers from measurements\n")

print("=== Model Card (excerpt) ===")
for line in model_card.splitlines()[:25]:
    print(line)
print("  ... (full card written to disk)")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the card as a one-page summary
# ════════════════════════════════════════════════════════════════════════
# Reviewers skim. Top row: headline metrics. Bottom row: the per-group
# flag rate next to each group's actual default rate — the picture that
# explains every fairness number in the card.

fig = make_subplots(
    rows=2,
    cols=4,
    specs=[[{"type": "indicator"}] * 4, [{"type": "xy", "colspan": 4}, None, None, None]],
    row_heights=[0.4, 0.6],
    vertical_spacing=0.12,
)
gauges = [
    ("AUC-ROC", metrics["auc_roc"], [0.5, 1.0]),
    ("AUC-PR", metrics["auc_pr"], [0.0, 1.0]),
    ("Brier (lower=better)", metrics["brier"], [0.0, 0.25]),
    (f"Coverage (target {1 - conformal['alpha']:.0%})", conformal["coverage"], [0.0, 1.0]),
]
for col, (title, value, axis_range) in enumerate(gauges, start=1):
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={"axis": {"range": axis_range}},
            number={"valueformat": ".3f"},
        ),
        row=1,
        col=col,
    )
group_labels = [f"{a}: {g}" for a, g in zip(group_table["attribute"], group_table["group"])]
fig.add_trace(
    go.Bar(x=group_labels, y=group_table["base_rate"].to_list(), name="Actual default rate", marker_color="#94a3b8"),
    row=2,
    col=1,
)
fig.add_trace(
    go.Bar(x=group_labels, y=group_table["flag_rate"].to_list(), name="Flagged by model", marker_color="#ef4444"),
    row=2,
    col=1,
)
fig.update_layout(
    title="Model Card Summary — Singapore Credit Default",
    barmode="group",
    height=760,
    legend=dict(orientation="h", y=-0.15),
)
viz_path = OUTPUT_DIR / "ex8_03_model_card_summary.html"
fig.write_html(str(viz_path))
print(f"\nSaved: {viz_path}")


# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert viz_path.exists(), "Task 4: Summary visual should be written"
print("\n[ok] Checkpoint 3 — one-page visual summary rendered\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: the questions the card hands to a risk committee
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore lender's model-risk committee must
# approve this model before 8.4 promotes it. The card's job is to put the
# decisions in front of them with numbers attached — not to make them.

questions = []
for row in fair.iter_rows(named=True):
    if row["disparate_impact"] < 0.8:
        groups = group_table.filter(pl.col("attribute") == row["attribute"])
        lo = groups.sort("flag_rate").row(0, named=True)
        hi = groups.sort("flag_rate").row(-1, named=True)
        questions.append(
            f"{row['attribute']}: '{hi['group']}' is flagged {hi['flag_rate']:.1%} vs "
            f"'{lo['group']}' {lo['flag_rate']:.1%} (actual default {hi['base_rate']:.1%} vs "
            f"{lo['base_rate']:.1%}). Is this attribute a permitted credit factor, and "
            f"is a calibrated-but-unequal outcome acceptable?"
        )
if conformal["coverage"] < 1 - conformal["alpha"]:
    questions.append(
        f"Conformal coverage {conformal['coverage']:.1%} is slightly below the "
        f"{1 - conformal['alpha']:.0%} target on this sample — accept as sampling noise "
        f"or recalibrate?"
    )
questions.append(
    "Race and gender are model inputs. Keep them (with a written justification) "
    "or retrain without them and re-measure the fairness table?"
)
print("=== Decisions for the model-risk committee (generated from the card) ===")
for i, q in enumerate(questions, start=1):
    print(f"  {i}. {q}")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] The 9 Mitchell et al. model card sections and when to use each
  [x] Generated every number in the card from measurements of this model
  [x] Measured disaggregated fairness for {fair.height} protected attributes
      ({len(failing)} below the four-fifths rule of thumb)
  [x] Rendered a one-page visual summary for non-technical reviewers
  [x] Turned the card into {len(questions)} concrete decisions for a risk committee

  KEY INSIGHT: A model card is not paperwork. It is the contract that
  says where the model was validated and for whom — and a card with an
  unmeasured claim is a false assurance.

  Next: 04_deployment_pipeline.py — register, version, promote and
  roll back the model with a full audit trail.
"""
)
