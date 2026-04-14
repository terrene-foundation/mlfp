# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP06 Exercise 7 — AI Governance with PACT.

Contains: adversarial-prompt loading, YAML org definition, clearance hierarchy,
BudgetTracker, GovernanceEngine compile helper. Technique-specific code does
NOT belong here — each technique file builds its own scenario on top.

Import from any cwd after `uv sync`:

    from shared.mlfp06.ex_7 import (
        CLEARANCE_LEVELS, ORG_YAML, load_adversarial_prompts,
        write_org_yaml, compile_governance, BudgetTracker,
    )
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import polars as pl

from shared.kailash_helpers import setup_environment

setup_environment()

# ════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════

# Clearance hierarchy used by every technique file for monotonic-tightening
# checks. restricted > confidential > internal > public.
CLEARANCE_LEVELS: dict[str, int] = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
    "restricted": 3,
}


# Default LLM (lazy-resolved; agents read at construction time)
def default_model_name() -> str | None:
    return os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))


# ════════════════════════════════════════════════════════════════════════
# ADVERSARIAL PROMPT DATASET
# ════════════════════════════════════════════════════════════════════════

CACHE_DIR = Path("data/mlfp06/toxicity")
CACHE_FILE = CACHE_DIR / "real_toxicity_50.parquet"


def load_adversarial_prompts(n: int = 50) -> pl.DataFrame:
    """Load (and cache) the allenai/real-toxicity-prompts adversarial slice.

    Filters to prompts with toxicity > 0.5, shuffles with a fixed seed,
    and returns the first `n` rows as a polars DataFrame with columns
    `prompt_text` and `toxicity_score`.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists():
        return pl.read_parquet(CACHE_FILE)

    from datasets import load_dataset

    ds = load_dataset("allenai/real-toxicity-prompts", split="train")
    ds = ds.filter(
        lambda r: r["prompt"]["toxicity"] is not None and r["prompt"]["toxicity"] > 0.5
    )
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    rows = [
        {
            "prompt_text": row["prompt"]["text"],
            "toxicity_score": row["prompt"]["toxicity"],
        }
        for row in ds
    ]
    df = pl.DataFrame(rows)
    df.write_parquet(CACHE_FILE)
    return df


# ════════════════════════════════════════════════════════════════════════
# CANONICAL SINGAPORE FINTECH ORG YAML (D/T/R GRAMMAR)
# ════════════════════════════════════════════════════════════════════════
#
# Every technique file uses the same organisation so students can track
# how envelopes, budgets, and access decisions evolve as they add more
# governance structure. The D/T/R grammar is:
#   D (Delegator):   Human authority who authorises the task
#   T (Task):        Bounded scope of work
#   R (Responsible): The agent that executes within the envelope

ORG_YAML: str = """
# Singapore FinTech AI Organisation — PACT Governance Definition
# D/T/R: every agent action traces to a human Delegator

organization:
  name: "SG FinTech AI Division"
  jurisdiction: "Singapore"
  regulatory_framework: "MAS TRM, AI Verify, PDPA"

departments:
  - name: "ML Engineering"
    head: "chief_ml_officer"
    agents:
      - id: "data_analyst"
        role: "analyst"
        clearance: "internal"
        description: "Analyses datasets, generates exploratory reports"
      - id: "model_trainer"
        role: "engineer"
        clearance: "confidential"
        description: "Trains and evaluates ML models"
      - id: "model_deployer"
        role: "operator"
        clearance: "confidential"
        description: "Deploys models to production infrastructure"

  - name: "Risk & Compliance"
    head: "chief_risk_officer"
    agents:
      - id: "risk_assessor"
        role: "auditor"
        clearance: "restricted"
        description: "Assesses model risk, bias, and regulatory compliance"
      - id: "bias_checker"
        role: "auditor"
        clearance: "confidential"
        description: "Checks models for bias and fairness violations"

  - name: "Customer Intelligence"
    head: "vp_customer"
    agents:
      - id: "customer_agent"
        role: "analyst"
        clearance: "public"
        description: "Handles customer-facing AI interactions"

delegations:
  - delegator: "chief_ml_officer"
    task: "data_analysis"
    responsible: "data_analyst"
    envelope:
      max_budget_usd: 20.0
      allowed_tools: ["read_data", "summarise_data", "generate_report"]
      allowed_data_clearance: "internal"
      max_data_rows: 500000

  - delegator: "chief_ml_officer"
    task: "model_training"
    responsible: "model_trainer"
    envelope:
      max_budget_usd: 100.0
      allowed_tools: ["train_model", "evaluate_model", "read_data"]
      allowed_data_clearance: "confidential"
      max_data_rows: 1000000

  - delegator: "chief_ml_officer"
    task: "model_deployment"
    responsible: "model_deployer"
    envelope:
      max_budget_usd: 50.0
      allowed_tools: ["deploy_model", "monitor_model", "rollback_model"]
      allowed_data_clearance: "confidential"

  - delegator: "chief_risk_officer"
    task: "risk_assessment"
    responsible: "risk_assessor"
    envelope:
      max_budget_usd: 200.0
      allowed_tools: ["read_data", "audit_model", "generate_report", "access_audit_log"]
      allowed_data_clearance: "restricted"

  - delegator: "chief_risk_officer"
    task: "bias_audit"
    responsible: "bias_checker"
    envelope:
      max_budget_usd: 75.0
      allowed_tools: ["read_data", "audit_model", "run_fairness_check"]
      allowed_data_clearance: "confidential"

  - delegator: "vp_customer"
    task: "customer_interaction"
    responsible: "customer_agent"
    envelope:
      max_budget_usd: 5.0
      allowed_tools: ["answer_question", "search_faq"]
      allowed_data_clearance: "public"
      max_response_length: 500

operating_envelopes:
  global:
    max_llm_cost_per_request_usd: 0.50
    require_audit_trail: true
    pii_handling: "mask"
    log_retention_days: 90
    fail_mode: "closed"
"""


def write_org_yaml(path: str | Path | None = None) -> str:
    """Write the canonical org YAML to a temp file and return the path."""
    if path is None:
        path = os.path.join(tempfile.gettempdir(), "sg_fintech_org.yaml")
    with open(path, "w") as f:
        f.write(ORG_YAML)
    return str(path)


# ════════════════════════════════════════════════════════════════════════
# GOVERNANCE ENGINE COMPILATION
# ════════════════════════════════════════════════════════════════════════


async def _compile_async(yaml_path: str):
    """Async compile helper — GovernanceEngine is async-first."""
    from kailash_pact import GovernanceEngine

    engine = GovernanceEngine()
    org = engine.compile_org(yaml_path)
    return engine, org


def compile_governance(yaml_path: str | None = None):
    """Compile the canonical org YAML. Returns (engine, org).

    Validates: every agent has a delegation chain, no circular delegations,
    clearance decreases down each chain, budgets don't exceed parent limits.
    """
    if yaml_path is None:
        yaml_path = write_org_yaml()
    return asyncio.run(_compile_async(yaml_path))


# ════════════════════════════════════════════════════════════════════════
# BUDGET TRACKER
# ════════════════════════════════════════════════════════════════════════


class BudgetTracker:
    """Track budget allocation and consumption across an agent hierarchy.

    Parent allocates to children; children cannot spend more than their
    allocation. Used to demonstrate budget cascading in technique 03.
    """

    def __init__(self, total_budget: float) -> None:
        self.total_budget = total_budget
        self.consumed: dict[str, float] = {}
        self.allocations: dict[str, float] = {}

    def allocate(self, agent_id: str, amount: float) -> bool:
        """Allocate budget to an agent. Returns False if insufficient."""
        total_allocated = sum(self.allocations.values())
        if total_allocated + amount > self.total_budget:
            return False
        self.allocations[agent_id] = self.allocations.get(agent_id, 0) + amount
        return True

    def spend(self, agent_id: str, amount: float) -> bool:
        """Record spending. Returns False if exceeds allocation."""
        allocation = self.allocations.get(agent_id, 0)
        current = self.consumed.get(agent_id, 0)
        if current + amount > allocation:
            return False
        self.consumed[agent_id] = current + amount
        return True

    def remaining(self, agent_id: str) -> float:
        return self.allocations.get(agent_id, 0) - self.consumed.get(agent_id, 0)

    def summary(self) -> pl.DataFrame:
        agents = set(self.allocations.keys()) | set(self.consumed.keys())
        rows = []
        for a in sorted(agents):
            rows.append(
                {
                    "agent": a,
                    "allocated": self.allocations.get(a, 0),
                    "consumed": self.consumed.get(a, 0),
                    "remaining": self.remaining(a),
                }
            )
        return pl.DataFrame(rows)
