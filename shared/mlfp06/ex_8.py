# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP06 Exercise 8 — Capstone: Full Production Platform.

Contains: LLM model resolution, MMLU evaluation data loader, shared PACT
governance YAML, the base Signature/Agent classes, the governed-agent QA
handler used by every technique file, and a small rate limiter and JWT
stub used by the serving/monitoring modules.

Technique-specific code (adapter loading, governance wiring, nexus
registration, drift analysis, compliance reporting) does NOT belong here —
it lives in the per-technique files under
``modules/mlfp06/solutions/ex_8/``.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import polars as pl
from dotenv import load_dotenv

from kaizen import InputField, OutputField, Signature
from kaizen.core import BaseAgent

from shared.kailash_helpers import setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()
load_dotenv()

MODEL = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
if not MODEL:
    raise EnvironmentError(
        "Set DEFAULT_LLM_MODEL or OPENAI_PROD_MODEL in .env before running"
    )

# Output + cache directories
OUTPUT_DIR = Path("outputs") / "ex8_capstone"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_CACHE_DIR = Path("data/mlfp06/mmlu")
EVAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
EVAL_CACHE_FILE = EVAL_CACHE_DIR / "mmlu_100.parquet"


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — MMLU (Massive Multitask Language Understanding)
# ════════════════════════════════════════════════════════════════════════


def load_mmlu_eval(n_rows: int = 100) -> pl.DataFrame:
    """Load MMLU evaluation data as a polars DataFrame, cached to parquet.

    Schema:
        instruction (str) — question + A/B/C/D choices as plain text
        response    (str) — correct letter (A/B/C/D)
        subject     (str) — MMLU subject area

    Returns:
        A polars DataFrame with at most ``n_rows`` shuffled MMLU questions.
    """
    if EVAL_CACHE_FILE.exists():
        print(f"Loading cached MMLU from {EVAL_CACHE_FILE}")
        return pl.read_parquet(EVAL_CACHE_FILE)

    print("Downloading cais/mmlu from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="validation")
    ds = ds.shuffle(seed=42).select(range(min(n_rows, len(ds))))
    rows: list[dict[str, Any]] = []
    for row in ds:
        choices = row["choices"]
        answer_idx = row["answer"]
        rows.append(
            {
                "instruction": (
                    f"{row['question']}\n\n"
                    f"A) {choices[0]}\nB) {choices[1]}\n"
                    f"C) {choices[2]}\nD) {choices[3]}"
                ),
                "response": ["A", "B", "C", "D"][answer_idx],
                "subject": row["subject"],
            }
        )
    eval_data = pl.DataFrame(rows)
    eval_data.write_parquet(EVAL_CACHE_FILE)
    print(f"Cached {eval_data.height} MMLU rows to {EVAL_CACHE_FILE}")
    return eval_data


# ════════════════════════════════════════════════════════════════════════
# SHARED SIGNATURE & BASE AGENT
# ════════════════════════════════════════════════════════════════════════


class CapstoneQASignature(Signature):
    """Answer questions with a governed, audited, confidence-scored response."""

    question: str = InputField(description="User's question")
    answer: str = OutputField(description="Detailed, grounded answer")
    confidence: float = OutputField(description="Confidence score 0-1")
    sources: list[str] = OutputField(description="Knowledge sources referenced")
    reasoning_steps: list[str] = OutputField(description="Step-by-step reasoning")


class CapstoneQAAgent(BaseAgent):
    """Capstone QA agent: wraps the fine-tuned model behind a typed signature."""

    signature = CapstoneQASignature
    model = MODEL
    max_llm_cost_usd = 5.0


# ════════════════════════════════════════════════════════════════════════
# PACT GOVERNANCE — shared org yaml
# ════════════════════════════════════════════════════════════════════════

ORG_YAML = """
organization:
  name: "MLFP Capstone ML Platform"
  jurisdiction: "Singapore"
  regulatory_framework: "MAS TRM, AI Verify, PDPA"

departments:
  - name: "AI Services"
    head: "ml_director"
    agents:
      - id: "qa_agent"
        role: "responder"
        clearance: "internal"
        description: "Answers domain questions using fine-tuned model"
      - id: "admin_agent"
        role: "operator"
        clearance: "confidential"
        description: "Manages model lifecycle, monitoring, and metrics"
      - id: "audit_agent"
        role: "auditor"
        clearance: "restricted"
        description: "Full audit access for compliance reporting"

delegations:
  - delegator: "ml_director"
    task: "question_answering"
    responsible: "qa_agent"
    envelope:
      max_budget_usd: 1.0
      allowed_tools: ["generate_answer", "search_context"]
      allowed_data_clearance: "internal"

  - delegator: "ml_director"
    task: "model_management"
    responsible: "admin_agent"
    envelope:
      max_budget_usd: 10.0
      allowed_tools: ["generate_answer", "search_context", "update_model",
                       "view_metrics", "monitor_drift"]
      allowed_data_clearance: "confidential"

  - delegator: "ml_director"
    task: "compliance_audit"
    responsible: "audit_agent"
    envelope:
      max_budget_usd: 50.0
      allowed_tools: ["generate_answer", "search_context", "view_metrics",
                       "access_audit_log", "generate_report"]
      allowed_data_clearance: "restricted"

operating_envelopes:
  global:
    max_llm_cost_per_request_usd: 0.10
    require_audit_trail: true
    pii_handling: "mask"
    fail_mode: "closed"
"""


def write_org_yaml() -> str:
    """Write the shared org YAML to a temp file and return the path."""
    org_path = os.path.join(tempfile.gettempdir(), "capstone_org.yaml")
    with open(org_path, "w") as f:
        f.write(ORG_YAML)
    return org_path


# ════════════════════════════════════════════════════════════════════════
# SHARED HANDLER — used by Nexus deployment AND monitoring/test files
# ════════════════════════════════════════════════════════════════════════


async def handle_qa(
    question: str,
    role: str,
    agents_by_role: dict[str, Any],
) -> dict[str, Any]:
    """Route a question to the governed agent matching the role.

    Args:
        question: The user's question.
        role: Access role — keys of ``agents_by_role`` (e.g. 'qa', 'admin').
        agents_by_role: Mapping of role name to a PactGovernedAgent instance.

    Returns:
        Response dict with answer + governance metadata, or an error dict
        when governance blocks the call.
    """
    agent = agents_by_role.get(role) or next(iter(agents_by_role.values()))
    start = time.time()

    try:
        result = await agent.run(question=question)
        latency = time.time() - start
        return {
            "answer": result.answer,
            "confidence": result.confidence,
            "sources": result.sources,
            "reasoning_steps": result.reasoning_steps,
            "latency_ms": latency * 1000,
            "governed": True,
            "role": role,
        }
    except Exception as e:  # governance / budget / clearance denial
        return {
            "error": str(e),
            "governed": True,
            "blocked": True,
            "role": role,
        }


# ════════════════════════════════════════════════════════════════════════
# SHARED MIDDLEWARE UTILITIES
# ════════════════════════════════════════════════════════════════════════


class SimpleJWTAuth:
    """Stub JWT validator — production uses RS256 signed tokens.

    This exists so the capstone can demonstrate the auth surface without
    requiring a real JWKS endpoint. Every token maps to a role claim.
    """

    VALID_TOKENS: dict[str, dict[str, str]] = {
        "token_viewer_001": {"sub": "alice", "role": "qa"},
        "token_operator_001": {"sub": "bob", "role": "admin"},
        "token_auditor_001": {"sub": "carol", "role": "audit"},
    }

    @classmethod
    def validate(cls, token: str) -> dict[str, str] | None:
        """Return token claims, or None if the token is invalid."""
        return cls.VALID_TOKENS.get(token)


class RateLimiter:
    """Sliding-window rate limiter: ``max_requests`` per ``window_seconds``."""

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests: dict[str, list[float]] = {}

    def allow(self, client_id: str) -> bool:
        now = time.time()
        bucket = self.requests.setdefault(client_id, [])
        bucket[:] = [t for t in bucket if now - t < self.window]
        if len(bucket) >= self.max_requests:
            return False
        bucket.append(now)
        return True


# ════════════════════════════════════════════════════════════════════════
# RUN HELPER — for technique files that use asyncio.run() at module scope
# ════════════════════════════════════════════════════════════════════════


def run_async(coro):  # noqa: ANN001 — coroutine
    """Run an async coroutine, tolerating already-running event loops."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
