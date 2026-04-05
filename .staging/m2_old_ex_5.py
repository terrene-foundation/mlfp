# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT2 — Exercise 5: FeatureEngineer + ExperimentTracker Review
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Use FeatureEngineer for automated feature generation
#   (interactions, polynomial, binning), then review the full experiment
#   history from all M2 exercises.
#
# TASKS:
#   1. Load Singapore credit data and engineer baseline features
#   2. Use FeatureEngineer for automated feature generation
#   3. Compare manual vs automated features with DataExplorer
#   4. Evaluate feature importance with mutual information
#   5. Review full M2 experiment history in ExperimentTracker
#   6. Generate experiment comparison report
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import polars as pl
from kailash.db.connection import ConnectionManager
from kailash_ml import FeatureEngineer, DataExplorer
from kailash_ml.engines.experiment_tracker import ExperimentTracker
from kailash_ml import ModelVisualizer

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
credit = loader.load("ascent03", "sg_credit_scoring.parquet")

print("=== Singapore Credit Scoring Data ===")
print(f"Shape: {credit.shape}")
print(f"Columns: {credit.columns}")
print(f"Default rate: {credit['default'].mean():.2%}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Manual feature engineering (baseline)
# ══════════════════════════════════════════════════════════════════════

credit_manual = credit.with_columns(
    # Debt-to-income ratio
    (pl.col("total_debt") / pl.col("annual_income").clip(lower_bound=1)).alias(
        "debt_to_income"
    ),
    # Credit utilisation
    (pl.col("credit_used") / pl.col("credit_limit").clip(lower_bound=1)).alias(
        "credit_utilisation"
    ),
    # Payment behaviour
    (
        pl.col("late_payments_12m") / pl.col("total_payments_12m").clip(lower_bound=1)
    ).alias("late_payment_ratio"),
    # Account age features
    (pl.col("account_age_months") / 12).alias("account_age_years"),
    # Income stability (std of monthly income / mean)
    (pl.col("income_std") / pl.col("annual_income").clip(lower_bound=1) * 12).alias(
        "income_cv"
    ),
    # Log-transformed skewed features
    pl.col("annual_income").log1p().alias("log_income"),
    pl.col("total_debt").log1p().alias("log_debt"),
)

manual_features = [c for c in credit_manual.columns if c not in credit.columns]
print(f"\nManual features ({len(manual_features)}): {manual_features}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Automated feature engineering with FeatureEngineer
# ══════════════════════════════════════════════════════════════════════


async def automated_features():
    """Use FeatureEngineer for automated feature generation."""

    engineer = FeatureEngineer(max_features=30)

    # Define schema for the features to engineer
    from kailash_ml.types import FeatureSchema, FeatureField

    numeric_cols = [
        "annual_income",
        "total_debt",
        "credit_limit",
        "credit_used",
        "account_age_months",
        "late_payments_12m",
    ]
    eng_schema = FeatureSchema(
        name="credit_features",
        features=[FeatureField(name=c, dtype="float64") for c in numeric_cols],
        entity_id_column="application_id",
    )

    # Automated feature generation using FeatureSchema
    generated = engineer.generate(
        data=credit,
        schema=eng_schema,
        strategies=["interactions", "polynomial", "binning", "ratios"],
    )

    # Select best features by importance
    result = engineer.select(
        data=generated.data,
        candidates=generated,
        target="default",
        method="importance",
        top_k=30,
    )

    print(f"\n=== FeatureEngineer Results ===")
    print(f"Original features: {len(numeric_cols)}")
    print(f"Generated candidates: {generated.total_candidates}")
    print(f"Selected features: {result.n_selected}")
    print(f"\nTop selected features:")
    for rank in result.rankings[:15]:
        print(f"  {rank.column_name}: score={rank.score:.4f}")
    if result.n_selected > 15:
        print(f"  ... and {result.n_selected - 15} more")

    # Feature importance from rankings
    if result.rankings:
        print(f"\nTop 10 by importance:")
        for rank in result.rankings[:10]:
            print(f"  {rank.column_name}: {rank.score:.4f}")

    return result


engineer_result = asyncio.run(automated_features())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Compare manual vs automated features
# ══════════════════════════════════════════════════════════════════════


async def compare_features():
    """Compare manual and automated feature sets with DataExplorer."""

    explorer = DataExplorer()

    # Profile manual features
    manual_df = credit_manual.select(manual_features + ["default"])
    profile_manual = await explorer.profile(manual_df)

    # Profile automated features
    auto_df = engineer_result.data.select(
        engineer_result.selected_columns + ["default"]
    )
    profile_auto = await explorer.profile(auto_df)

    print(f"\n=== Feature Set Comparison ===")
    print(f"{'Metric':<25} {'Manual':>10} {'Automated':>12}")
    print("─" * 50)
    print(
        f"{'Feature count':<25} {len(manual_features):>10} {len(engineer_result.selected_columns):>12}"
    )
    print(
        f"{'Alerts':<25} {len(profile_manual.alerts):>10} {len(profile_auto.alerts):>12}"
    )
    print(
        f"{'Avg null %':<25} "
        f"{np.mean([c.null_pct for c in profile_manual.columns]):>10.2%} "
        f"{np.mean([c.null_pct for c in profile_auto.columns]):>12.2%}"
    )

    return profile_manual, profile_auto


profile_manual, profile_auto = asyncio.run(compare_features())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Feature importance with mutual information
# ══════════════════════════════════════════════════════════════════════

from sklearn.feature_selection import mutual_info_classif
from kailash_ml.interop import to_sklearn_input

# Combine all features
all_feature_cols = manual_features + [
    f for f in engineer_result.selected_columns if f not in manual_features
]

# Prepare combined dataset
combined = credit_manual.join(
    engineer_result.data.select(
        ["default"]
        + [
            f
            for f in engineer_result.selected_columns
            if f not in credit_manual.columns
        ]
    ),
    on="default",
    how="cross",
)

# Use manual features for MI analysis (cleaner)
mi_df = credit_manual.select(manual_features + ["default"]).drop_nulls()
X, y, col_info = to_sklearn_input(
    mi_df,
    feature_columns=manual_features,
    target_column="default",
)

# Mutual information
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_ranking = sorted(zip(manual_features, mi_scores), key=lambda x: x[1], reverse=True)

print(f"\n=== Mutual Information Ranking ===")
for name, score in mi_ranking:
    bar = "█" * int(score * 100)
    print(f"  {name:<25} {score:.4f} {bar}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Review full M2 experiment history
# ══════════════════════════════════════════════════════════════════════


async def review_experiments():
    """Review all experiments from Module 2 exercises."""

    conn = ConnectionManager("sqlite:///ascent02_experiments.db")
    await conn.initialize()
    tracker = ExperimentTracker(conn)
    await tracker.initialize()

    # Log this exercise's feature engineering run
    exp_id = await tracker.create_experiment(
        name="ascent02_feature_engineering",
        description="Automated feature engineering comparison",
        tags=["ascent02", "feature-engineering", "automated"],
    )

    async with tracker.run(exp_id, run_name="manual_vs_automated_features") as run:
        await run.log_params(
            {
                "manual_features": ",".join(manual_features),
                "auto_strategies": "interactions,polynomial,binning,ratios",
                "max_features": "30",
            }
        )
        await run.log_metrics(
            {
                "n_manual_features": float(len(manual_features)),
                "n_auto_features": float(len(engineer_result.selected_columns)),
                "top_mi_score": float(mi_ranking[0][1]),
            }
        )
        await run.set_tag("type", "feature-engineering")

    # List ALL experiments from M2
    experiments = await tracker.list_experiments()
    print(f"\n=== Module 2 Experiment History ===")
    print(f"Total experiments: {len(experiments)}")
    for exp in experiments:
        print(f"\n  Experiment: {exp.get('name', 'unnamed')}")
        print(f"  Tags: {exp.get('tags', [])}")
        runs = await tracker.list_runs(exp["id"])
        print(f"  Runs: {len(runs)}")
        for run in runs:
            print(
                f"    - {run.get('name', 'unnamed')}: "
                f"metrics={list(run.get('metrics', {}).keys())}"
            )

    # Compare experiments
    print(f"\n=== Cross-Experiment Summary ===")
    print("This module tracked:")
    print("  Ex 1: Healthcare feature engineering (clinical features)")
    print("  Ex 2: FeatureStore lifecycle (schema versioning, lineage)")
    print("  Ex 3: A/B test with CUPED + Bayesian analysis")
    print("  Ex 4: Causal inference (DiD on cooling measures)")
    print("  Ex 5: Automated vs manual feature engineering")
    print("\nEvery analysis is auditable. Every feature is versioned.")
    print("This is production-grade experiment management.")

    await conn.close()
    return experiments


experiments = asyncio.run(review_experiments())


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Visualise experiment comparison
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# Compare manual feature importance
mi_data = {f: {"MI_Score": score} for f, score in mi_ranking[:8]}
fig = viz.metric_comparison(mi_data)
fig.update_layout(title="Feature Importance (Mutual Information)")
fig.write_html("ex5_feature_importance.html")
print("\nSaved: ex5_feature_importance.html")

print("\n✓ Exercise 5 complete — automated feature engineering + experiment review")
print("  Module 2 complete: 5 exercises tracked in ExperimentTracker")
