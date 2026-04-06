# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT05 — Exercise 6: ML Agent Pipeline
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Chain all 6 kailash-ml agents in a full ML pipeline:
#   DataScientist → FeatureEngineer → ModelSelector → ExperimentInterpreter
#   → DriftAnalyst → RetrainingDecision. Demonstrate the double opt-in
#   pattern (AgentInfusionProtocol).
#
# TASKS:
#   1. Set up the 6 ML agents with cost budgets
#   2. DataScientistAgent: initial data analysis
#   3. FeatureEngineerAgent: suggest features
#   4. ModelSelectorAgent: recommend model architecture
#   5. ExperimentInterpreterAgent: interpret results
#   6. DriftAnalyst + RetrainingDecision: production monitoring
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl
from dotenv import load_dotenv

from kailash_ml.agents.data_scientist import DataScientistAgent
from kailash_ml.agents.model_selector import ModelSelectorAgent
from kailash_ml.agents.feature_engineer import FeatureEngineerAgent
from kailash_ml.agents.experiment_interpreter import ExperimentInterpreterAgent
from kailash_ml.agents.drift_analyst import DriftAnalystAgent
from kailash_ml.agents.retraining_decision import RetrainingDecisionAgent

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
credit = loader.load("ascent03", "sg_credit_scoring.parquet")

data_context = {
    "dataset": "Singapore Credit Scoring",
    "rows": credit.height,
    "columns": credit.columns,
    "target": "default",
    "default_rate": float(credit["default"].mean()),
    "features": len(credit.columns) - 1,
}

print(f"=== ML Agent Pipeline ===")
print(f"Dataset: {data_context['dataset']}")
print(f"Shape: {data_context['rows']:,} × {data_context['features']} features")
print(f"Default rate: {data_context['default_rate']:.2%}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: DataScientistAgent — initial analysis
# ══════════════════════════════════════════════════════════════════════


async def step_1_data_scientist():
    agent = DataScientistAgent(
        model=model,
        max_llm_cost_usd=1.0,
    )

    result = await agent.analyze(
        data=credit,
        target="default",
        context="Singapore credit scoring for a bank. 12% default rate. Need production model.",
    )

    print(f"\n=== Step 1: DataScientistAgent ===")
    print(f"Data quality assessment: {result.quality_summary}")
    print(f"Key issues: {result.issues}")
    print(f"Recommendations: {result.recommendations}")

    return result


ds_result = asyncio.run(step_1_data_scientist())


# ══════════════════════════════════════════════════════════════════════
# TASK 2: FeatureEngineerAgent — suggest features
# ══════════════════════════════════════════════════════════════════════


async def step_2_feature_engineer():
    agent = FeatureEngineerAgent(
        model=model,
        max_llm_cost_usd=1.0,
    )

    result = await agent.suggest(
        data=credit,
        target="default",
        existing_features=credit.columns,
        domain="credit_scoring",
    )

    print(f"\n=== Step 2: FeatureEngineerAgent ===")
    print(f"Suggested features: {result.suggested_features}")
    print(f"Transformation recommendations: {result.transformations}")
    print(f"Features to drop: {result.drop_candidates}")

    return result


fe_result = asyncio.run(step_2_feature_engineer())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: ModelSelectorAgent — recommend model
# ══════════════════════════════════════════════════════════════════════


async def step_3_model_selector():
    agent = ModelSelectorAgent(
        model=model,
        max_llm_cost_usd=1.0,
    )

    result = await agent.recommend(
        task_type="binary_classification",
        dataset_size=credit.height,
        n_features=len(credit.columns) - 1,
        target_metric="auc_pr",
        constraints={"interpretability": "high", "latency_ms": 50},
    )

    print(f"\n=== Step 3: ModelSelectorAgent ===")
    print(f"Recommended model: {result.recommended_model}")
    print(f"Alternatives: {result.alternatives}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Hyperparameter suggestions: {result.hyperparameters}")

    return result


ms_result = asyncio.run(step_3_model_selector())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: ExperimentInterpreterAgent — interpret M3 results
# ══════════════════════════════════════════════════════════════════════

# Simulated experiment results from Module 3
experiment_results = {
    "LightGBM": {"auc_roc": 0.89, "auc_pr": 0.62, "brier": 0.08, "log_loss": 0.31},
    "XGBoost": {"auc_roc": 0.88, "auc_pr": 0.60, "brier": 0.09, "log_loss": 0.33},
    "CatBoost": {"auc_roc": 0.87, "auc_pr": 0.58, "brier": 0.10, "log_loss": 0.35},
}


async def step_4_experiment_interpreter():
    agent = ExperimentInterpreterAgent(
        model=model,
        max_llm_cost_usd=1.0,
    )

    result = await agent.interpret(
        experiment_results=experiment_results,
        target_metric="auc_pr",
        context="Credit scoring model comparison for Singapore bank",
    )

    print(f"\n=== Step 4: ExperimentInterpreterAgent ===")
    print(f"Winner: {result.best_model}")
    print(f"Interpretation: {result.interpretation}")
    print(f"Production readiness: {result.production_ready}")
    print(f"Remaining concerns: {result.concerns}")

    return result


ei_result = asyncio.run(step_4_experiment_interpreter())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: DriftAnalyst + RetrainingDecision
# ══════════════════════════════════════════════════════════════════════

# Simulated drift report from Module 4
drift_report = {
    "model": "credit_default_lgbm",
    "overall_severity": "moderate",
    "features_drifted": 3,
    "max_psi": 0.18,
    "drifted_features": ["annual_income", "total_debt", "credit_utilisation"],
    "current_auc_pr": 0.55,
    "baseline_auc_pr": 0.62,
}


async def step_5_drift_and_retrain():
    drift_agent = DriftAnalystAgent(
        model=model,
        max_llm_cost_usd=1.0,
    )

    drift_analysis = await drift_agent.analyze(
        drift_report=drift_report,
        context="Credit model in production for 6 months, economic downturn in Singapore",
    )

    print(f"\n=== Step 5a: DriftAnalystAgent ===")
    print(f"Root cause: {drift_analysis.root_cause}")
    print(f"Severity assessment: {drift_analysis.severity}")
    print(f"Affected segments: {drift_analysis.affected_segments}")

    retrain_agent = RetrainingDecisionAgent(
        model=model,
        max_llm_cost_usd=1.0,
    )

    retrain_decision = await retrain_agent.decide(
        drift_analysis=drift_analysis,
        performance_degradation=drift_report["baseline_auc_pr"]
        - drift_report["current_auc_pr"],
        model_age_days=180,
        retraining_cost_estimate=500.0,
    )

    print(f"\n=== Step 5b: RetrainingDecisionAgent ===")
    print(f"Decision: {retrain_decision.decision}")
    print(f"Urgency: {retrain_decision.urgency}")
    print(f"Reasoning: {retrain_decision.reasoning}")
    print(f"Recommended actions: {retrain_decision.actions}")

    return drift_analysis, retrain_decision


drift_analysis, retrain_decision = asyncio.run(step_5_drift_and_retrain())


# ══════════════════════════════════════════════════════════════════════
# Summary: Full ML Agent Pipeline
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Full Pipeline Summary ===")
print(
    """
DataScientistAgent → FeatureEngineerAgent → ModelSelectorAgent
       ↓                    ↓                      ↓
  Data quality         Feature ideas          Model choice
       ↓                    ↓                      ↓
       └──────────→ Train + Evaluate ←─────────────┘
                         ↓
              ExperimentInterpreterAgent
                         ↓
                   Deploy to production
                         ↓
                 DriftAnalystAgent (monitor)
                         ↓
              RetrainingDecisionAgent (trigger)

Each agent has max_llm_cost_usd — governance at every step.
The double opt-in pattern: agent=True in config + kailash-ml[agents] installed.
"""
)

print("✓ Exercise 6 complete — full ML agent pipeline with 6 agents")
