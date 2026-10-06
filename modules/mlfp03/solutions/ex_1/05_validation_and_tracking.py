# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 1.5: FeatureSchema Validation, Multi-Method
#                         Consensus, and Leakage Audit
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Declare a FeatureSchema contract (types, nullability, documentation)
#   - Validate the engineered matrix against the schema at runtime
#   - Vote across filter/wrapper/embedded selections to build a ROBUST
#     consensus feature set
#   - Log the final feature set + metrics to ExperimentTracker
#   - Run a leakage audit that every selection MUST pass before training
#
# PREREQUISITES: 02_filter_selection.py, 03_wrapper_selection.py,
#                04_embedded_selection.py
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — why schemas + audits are the last line of defence
#   2. Build — declare FeatureSchema + replay the three selections
#   3. Train — vote consensus across methods
#   4. Visualise — consensus table + leakage audit report
#   5. Apply — governance gates for a national health-data platform
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from collections import Counter

import numpy as np
import plotly.graph_objects as go
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from kailash_ml import DataExplorer
from kailash_ml.types import FeatureField, FeatureSchema

from shared.mlfp03.ex_1 import (
    OUTPUT_DIR,
    PREDICTION_HOURS,
    audit_feature_list,
    build_full_feature_frame,
    load_icu_tables,
    log_selection_run,
    prediction_window_report,
    prepare_selection_inputs,
    setup_tracking,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Schemas + Audits Are The Last Line Of Defence
# ════════════════════════════════════════════════════════════════════════
# Feature engineering is where the highest-stakes bugs live. A single
# leaky feature can pass every unit test, every code review, and every
# cross-validation fold — and then fail catastrophically in production
# because the validation set and the train set were drawn from the
# same leaky joint distribution.
#
# Two complementary defences catch these bugs:
#
#   1. FeatureSchema — a type + nullability contract checked at
#      runtime (by us, below: the schema object only DECLARES it). If a downstream refactor renames a column or changes
#      its dtype, the schema check fires before the model trains on
#      bad data. Think of it as a type system for ML features.
#
#   2. Leakage audit — a mechanical gate that FAILS when a feature is
#      a target source (los_days, discharge time), uses events after
#      the prediction cutoff, or correlates almost perfectly with the
#      target. A flagged feature MUST be removed before training.
#
# Neither defence is optional. Schemas catch "we renamed a column"
# bugs; the leakage audit catches "we accidentally used the future"
# bugs. Different failure modes, both fatal.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: feature matrix + FeatureSchema + re-run selections
# ════════════════════════════════════════════════════════════════════════

tables = load_icu_tables()
features = build_full_feature_frame(tables)
feature_cols, X_sel, y_binary = prepare_selection_inputs(features)

print("\n" + "=" * 70)
print("  Validation, Consensus & Leakage Audit")
print("=" * 70)
print(f"  Features: {len(feature_cols)}")
print(f"  Samples:  {X_sel.shape[0]}")

# FeatureSchema — declare expected types on the core clinical contract.
# Rows are ADMISSIONS (one patient can have several), so the entity key is
# admission_id. los_days is deliberately absent: it is the target source.
icu_schema = FeatureSchema(
    name="icu_clinical_features_v1",
    features=[
        FeatureField(
            name="age",
            dtype="int64",
            nullable=False,
            description="Patient age at admission",
        ),
        FeatureField(
            name="bmi",
            dtype="float64",
            nullable=True,
            description="Body-mass index (missing when weight not recorded)",
        ),
        FeatureField(
            name="n_unique_medications",
            dtype="uint32",
            nullable=False,
            description="Distinct medications started in the first 24h",
        ),
        FeatureField(
            name="received_vasopressors",
            dtype="bool",
            nullable=False,
            description="Whether patient received vasopressor drugs",
        ),
        FeatureField(
            name="n_abnormal_labs",
            dtype="uint32",
            nullable=False,
            description="Abnormal lab results in the first 24h",
        ),
        FeatureField(
            name="abnormal_lab_ratio",
            dtype="float64",
            nullable=False,
            description="Proportion of lab results flagged abnormal",
        ),
        FeatureField(
            name="medication_doses_per_hour",
            dtype="float64",
            nullable=False,
            description="Medication doses per hour in the first 24h",
        ),
    ],
    entity_id_column="admission_id",
    timestamp_column="admit_time",
    version=1,
)

print(f"\n--- FeatureSchema: {icu_schema.name} (v{icu_schema.version}) ---")
for f in icu_schema.features:
    nullable = "nullable" if f.nullable else "required"
    print(f"  {f.name:<25} {f.dtype:<10} {nullable}  -- {f.description}")

# Validate the schema against the built feature matrix: presence, dtype
# AND nullability. Collect every violation, then fail loudly.
DTYPE_NAMES = {
    "float64": pl.Float64,
    "int64": pl.Int64,
    "uint32": pl.UInt32,
    "bool": pl.Boolean,
}
violations: list[str] = []
for field_def in icu_schema.features:
    if field_def.name not in features.columns:
        violations.append(f"{field_def.name}: missing")
        continue
    actual = features[field_def.name].dtype
    if actual != DTYPE_NAMES[field_def.dtype]:
        violations.append(f"{field_def.name}: declared {field_def.dtype}, got {actual}")
    if not field_def.nullable and features[field_def.name].null_count() > 0:
        violations.append(f"{field_def.name}: declared non-null, has nulls")
for key in (icu_schema.entity_id_column, icu_schema.timestamp_column):
    if key not in features.columns:
        violations.append(f"key column {key}: missing")
# A schema must never declare a target source as a feature
audit_feature_list([f.name for f in icu_schema.features])
print(f"\n  Schema violations: {violations if violations else 'none'}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert icu_schema.name == "icu_clinical_features_v1", "Task 2: schema name mismatch"
assert len(icu_schema.features) == 7, "Task 2: schema should declare 7 fields"
assert not violations, f"Task 2: schema violations: {violations}"
assert features["admission_id"].n_unique() == features.height, (
    "Task 2: entity key admission_id must be unique per row"
)
print("\n[ok] Checkpoint 1 passed — FeatureSchema (names, dtypes, nulls) validated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: re-run the three selections for the consensus vote
# ════════════════════════════════════════════════════════════════════════

# (a) Filter — mutual information
mi_scores = mutual_info_classif(X_sel, y_binary, random_state=42)
mi_top = {
    name
    for name, _ in sorted(
        zip(feature_cols, mi_scores), key=lambda x: x[1], reverse=True
    )[:15]
}

# (b) Filter — chi-squared
X_chi2 = MinMaxScaler().fit_transform(X_sel)
chi2_scores, _ = chi2(X_chi2, y_binary)
chi2_top = {
    name
    for name, _ in sorted(
        zip(feature_cols, chi2_scores), key=lambda x: x[1], reverse=True
    )[:15]
}

# (c) Wrapper — RFE + Random Forest
rfe = RFE(
    estimator=RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    n_features_to_select=15,
    step=5,
)
rfe.fit(X_sel, y_binary)
rfe_top = {name for name, selected in zip(feature_cols, rfe.support_) if selected}

# (d) Embedded — L1 Lasso
X_scaled = StandardScaler().fit_transform(X_sel)
lasso = LogisticRegression(
    l1_ratio=1.0, C=0.1, solver="saga", max_iter=5000, random_state=42
)
lasso.fit(X_scaled, y_binary)
lasso_top = {
    name for name, coef in zip(feature_cols, lasso.coef_[0]) if abs(coef) > 1e-6
}

all_methods: dict[str, set[str]] = {
    "MI (top 15)": mi_top,
    "Chi2 (top 15)": chi2_top,
    "RFE (RF, 15)": rfe_top,
    "Lasso (C=0.1)": lasso_top,
}

votes: Counter = Counter()
for feats in all_methods.values():
    for f in feats:
        votes[f] += 1

consensus_3plus = [f for f, v in votes.most_common() if v >= 3]
consensus_2plus = [f for f, v in votes.most_common() if v >= 2]
final_features = consensus_3plus if len(consensus_3plus) >= 8 else consensus_2plus[:15]

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert len(final_features) > 0, "Task 3: consensus must select at least one feature"
assert len(final_features) <= len(
    feature_cols
), "Task 3: cannot select more features than exist"
print("\n[ok] Checkpoint 2 passed — multi-method consensus complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the consensus and run the leakage audit
# ════════════════════════════════════════════════════════════════════════

print("\n--- Feature Selection Method Comparison ---")
print(f"{'Method':<20} {'Selected':>10}")
print("-" * 32)
for method, feats in all_methods.items():
    print(f"  {method:<18} {len(feats):>10}")

print(f"\n  Features with >=3 method votes: {len(consensus_3plus)}")
print(f"  Features with >=2 method votes: {len(consensus_2plus)}")
print(f"\n  Final feature set ({len(final_features)} features):")
for f in final_features:
    picking_methods = [m for m, s in all_methods.items() if f in s]
    print(f"    {f:<35}  [{', '.join(picking_methods)}]")

# Visual: how many of the four methods picked each feature
vote_rows = votes.most_common(25)
fig_votes = go.Figure(
    go.Bar(
        x=[v for _, v in vote_rows][::-1],
        y=[f for f, _ in vote_rows][::-1],
        orientation="h",
    )
)
fig_votes.update_layout(
    title="Feature-selection consensus — votes out of 4 methods (top 25)",
    xaxis_title="Number of methods selecting the feature",
    height=650,
)
votes_path = OUTPUT_DIR / "ex1_05_consensus_votes.html"
fig_votes.write_html(str(votes_path))
print(f"\n  Saved: {votes_path}")


# --- Leakage Audit ---
print("\n--- Leakage Detection Audit ---")

# (1) Name / source gate: raises ValueError on any ID, target-source or
#     discharge column. Recorded as a list so it can be logged below.
leakage_suspects: list[str] = []
try:
    audit_feature_list(feature_cols)
except ValueError as exc:
    leakage_suspects = [c for c in feature_cols if c in str(exc)]
    print(f"  [FAIL] {exc}")
else:
    print(f"  [ok] none of the {len(feature_cols)} features is an ID / target source")

# (2) Point-in-time gate: the latest event that reached ANY feature,
#     measured from the data the builders actually used.
window = prediction_window_report(tables)
latest_event_hours = float(window["max_offset_hours"].max())
print(
    f"  Latest event used: {latest_event_hours:.1f}h after admission "
    f"(cutoff {PREDICTION_HOURS}h)"
)

# (3) Correlation sniff test — a feature with |r| > 0.95 against the
#     target is almost certainly leaked. Constant columns are skipped
#     (their correlation is undefined).
target_high_corr: list[tuple[str, float]] = []
for j, col in enumerate(feature_cols):
    column = X_sel[:, j]
    if column.std() == 0:
        continue
    r = float(np.corrcoef(column, y_binary)[0, 1])
    if abs(r) > 0.95:
        target_high_corr.append((col, r))

if target_high_corr:
    print("  [FAIL] near-perfect target correlation (|r|>0.95):")
    for name, r in target_high_corr:
        print(f"    {name:<33} r={r:.4f}")
else:
    print("  [ok] no feature has |r| > 0.95 with the target")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert (
    len(leakage_suspects) == 0
), f"Task 4: leakage suspects detected: {leakage_suspects}"
assert latest_event_hours <= PREDICTION_HOURS, "Task 4: event after the cutoff used"
assert (
    len(target_high_corr) == 0
), f"Task 4: features with r>0.95 to target: {target_high_corr}"
print("\n[ok] Checkpoint 3 passed — leakage audit clean\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4b — LOG the consensus run + profile to ExperimentTracker
# ════════════════════════════════════════════════════════════════════════


async def log_consensus() -> str:
    conn, tracker, exp_id = await setup_tracking()
    explorer = DataExplorer()
    profile = await explorer.profile(features)
    null_rate = sum(features[c].null_count() for c in features.columns) / (
        features.height * len(features.columns)
    )
    run_id = await log_selection_run(
        tracker,
        exp_id,
        run_name="consensus_multi_method",
        method="consensus",
        selected_features=final_features,
        total_features=len(feature_cols),
        extra_params={
            "schema": icu_schema.name,
            "schema_version": str(icu_schema.version),
            "voting_methods": ",".join(all_methods.keys()),
            "vote_threshold": "3",
        },
        extra_metrics={
            "n_features_engineered": float(len(feature_cols)),
            "n_features_consensus": float(len(final_features)),
            "null_rate": float(null_rate),
            "n_alerts": float(len(profile.alerts)),
            "leakage_suspects": float(len(leakage_suspects)),
            "high_target_corr": float(len(target_high_corr)),
        },
    )
    await conn.close()
    return run_id


run_id = asyncio.run(log_consensus())
print(f"\n  ExperimentTracker run: {run_id}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert run_id is not None, "Task 4: ExperimentTracker should return a run id"
print("\n[ok] Checkpoint 4 passed — consensus run logged\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: governance gates for a national health-data platform
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a national health-data platform ingests
# anonymised records from many hospitals and trains predictive models
# (readmission, length-of-stay) for capacity planning. Every model MUST
# pass:
#   1. A FeatureSchema contract check so schemas cannot silently drift
#      between contributing hospitals (names, dtypes, nullability)
#   2. A leakage audit so no post-outcome column (discharge diagnosis,
#      billed charges, death certificate) or post-cutoff event reaches
#      the training set
#   3. An ExperimentTracker record so any published statistic can be
#      reproduced later
#
# Why this exercise's pattern is the right tool:
#   - A per-hospital schema catches a contributor who renames a column
#     (e.g. "HR" vs "heart_rate") or changes its type before it poisons
#     the national run
#   - The leakage audit is mechanical and can run as a pipeline gate
#   - The multi-method consensus gives a feature shortlist that survives
#     methodological review
#
# COST OF A MISS (illustrative): a leaked feature makes validation look
# excellent and production fail; the cost is the bad decisions taken on
# the model's output plus the investigation and re-validation. A gate
# that runs on every pipeline execution costs a few seconds.
#
# LIMITATIONS:
#   - Schema checks catch type drift, not semantic drift (a column
#     still called "heart_rate" but suddenly measured differently)
#   - The leakage audit is HEURISTIC — it catches known patterns, but a
#     novel leakage source needs a clinician's eye
#   - In THIS dataset the audit passes, yet the window report shows
#     almost no event data before the cutoff — passing a leakage audit
#     does not mean the features are informative


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Declared a FeatureSchema contract for ICU clinical features
  [x] Validated names, dtypes and nullability against the feature matrix
  [x] Voted across filter + wrapper + embedded methods for a robust
      consensus feature set
  [x] Ran a leakage gate (target sources, prediction cutoff, correlation)
  [x] Logged the final consensus + audit metrics to ExperimentTracker
  [x] Applied the pattern to a national health-data platform where
      governance dominates

  KEY INSIGHT: Data quality beats model complexity. The FeatureSchema +
  leakage audit IS the model's warranty card — without it, you are
  shipping predictions on trust alone. And an audit that passes is not
  the same as features that carry signal: check both.

  Next: Exercise 2 — bias/variance trade-off, nested cross-validation,
  and how regularisation controls model complexity without touching
  the feature set built here.
"""
)
