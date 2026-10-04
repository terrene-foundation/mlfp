# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP06 Exercise 8 — Capstone: Full Production Platform.

Contains: LLM model resolution (Ollama via the course bootstrap), MMLU
evaluation data loader, the PACT governance YAML (D/T/R = Department / Team /
Role) for the MLFP Capstone org and ``compile_capstone_governance()`` which
applies its clearances + envelopes, canonical Signature/Agent classes
(dataclass config + instance signature), ``build_capstone_stack(engine)``,
and the ``handle_qa`` router used by every technique file. ``handle_qa``
makes a REAL call to the local Ollama model — there is no offline stub; a
missing daemon raises with "start Ollama: ollama serve".

Technique-specific code (adapter loading, nexus registration, drift analysis,
compliance reporting) does NOT belong here — it lives in the per-technique
files under ``modules/mlfp06/solutions/ex_8/``.

Import from any cwd after ``uv sync``:

    from shared.mlfp06.ex_8 import (
        MODEL, OUTPUT_DIR, load_mmlu_eval, write_org_yaml,
        compile_capstone_governance, CapstoneQASignature, CapstoneQAConfig,
        CapstoneQAAgent, build_capstone_stack, handle_qa, run_async,
    )
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from dotenv import load_dotenv

from kaizen import InputField, OutputField, Signature
from kaizen.core.base_agent import BaseAgent
from kaizen_agents import GovernedSupervisor

from shared.kailash_helpers import setup_environment

if TYPE_CHECKING:  # pragma: no cover — type-only imports
    from pact import GovernanceEngine

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()
load_dotenv()

from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL, OLLAMA_BASE_URL

MODEL = DEFAULT_CHAT_MODEL  # resolved from OLLAMA_CHAT_MODEL by the bootstrap

if not MODEL:  # pragma: no cover — bootstrap default never returns empty
    raise EnvironmentError("OLLAMA_CHAT_MODEL must be set")

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
# TRAINED-ADAPTER DISCOVERY — what Ex 2.6 (SFT) and Ex 3.3 (DPO) wrote
# ════════════════════════════════════════════════════════════════════════
#
# kailash-align's AlignmentPipeline saves every trained adapter to
# ``<experiment_dir>/<adapter_name>/<method>/adapter/`` (PEFT format:
# adapter_config.json + adapter_model.safetensors). An ``AdapterRegistry()``
# built without a backing model registry lives in memory only, so a new
# process cannot see adapters registered by an earlier exercise. The
# capstone therefore re-discovers the adapters on disk and registers them.

ADAPTER_SEARCH_ROOTS: tuple[Path, ...] = (
    Path("outputs") / "ex2_finetuning",  # Ex 2.6 SFT experiment_dir
    Path("dpo_output"),  # Ex 3.3 DPO experiment_dir
)


def discover_trained_adapters(
    roots: tuple[Path, ...] = ADAPTER_SEARCH_ROOTS,
) -> list[dict[str, Any]]:
    """Return one dict per PEFT adapter found under ``roots``.

    Keys: adapter_name, method, adapter_path, base_model_id, rank, alpha,
    target_modules, trainable_params (counted from the safetensors file).
    """
    import json

    found: list[dict[str, Any]] = []
    for root in roots:
        for cfg_path in sorted(root.glob("**/adapter/adapter_config.json")):
            adapter_dir = cfg_path.parent
            meta = json.loads(cfg_path.read_text())
            weights = adapter_dir / "adapter_model.safetensors"
            found.append(
                {
                    "adapter_name": adapter_dir.parent.parent.name,
                    "method": adapter_dir.parent.name,
                    "adapter_path": str(adapter_dir),
                    "base_model_id": meta.get("base_model_name_or_path") or "",
                    "rank": int(meta.get("r") or 0),
                    "alpha": int(meta.get("lora_alpha") or 0),
                    "target_modules": tuple(sorted(meta.get("target_modules") or ())),
                    "trainable_params": (
                        count_safetensors_params(weights) if weights.exists() else 0
                    ),
                }
            )
    return found


def count_safetensors_params(path: str | Path) -> int:
    """Count parameters in one ``.safetensors`` file, or every shard in a dir.

    Reads tensor shapes from the file header only — no weights are loaded.
    """
    from math import prod

    from safetensors import safe_open

    path = Path(path)
    files = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
    total = 0
    for f in files:
        with safe_open(str(f), framework="pt") as handle:
            for key in handle.keys():
                total += prod(handle.get_slice(key).get_shape())
    return total


# ════════════════════════════════════════════════════════════════════════
# SHARED SIGNATURE & BASE AGENT
# ════════════════════════════════════════════════════════════════════════
#
# Canonical pattern:
#   1. `@dataclass` config carries the Ollama provider, model, base_url and
#      `use_async_llm=True` (run_async raises without it)
#   2. Signature is PASSED as an instance to `super().__init__(signature=...)`
#      — omitting the `signature=` keyword silently falls back to
#      `DefaultSignature()` and the declared output schema is ignored.


class CapstoneQASignature(Signature):
    """Answer questions with a governed, audited, confidence-scored response."""

    question: str = InputField(description="User's question")
    answer: str = OutputField(description="Detailed, grounded answer")
    confidence: float = OutputField(description="Confidence score 0-1")
    sources: list[str] = OutputField(description="Knowledge sources referenced")
    reasoning_steps: list[str] = OutputField(description="Step-by-step reasoning")


def _json_object_format() -> dict[str, str]:
    return {"type": "json_object"}


@dataclass
class CapstoneQAConfig:
    """Domain config — BaseAgent auto-converts to BaseAgentConfig.

    ``budget_limit_usd`` is the agent-level cost cap. On free local Ollama it
    never trips (there is no dollar cost); it matters with a paid provider.
    """

    llm_provider: str = "ollama"
    model: str = MODEL
    base_url: str = OLLAMA_BASE_URL
    temperature: float = 0.2
    budget_limit_usd: float = 5.0
    use_async_llm: bool = True
    response_format: dict = field(default_factory=_json_object_format)
    structured_output_mode: str = "explicit"


class CapstoneQAAgent(BaseAgent):
    """Capstone QA agent: wraps the fine-tuned model behind a typed signature."""

    def __init__(self, config: CapstoneQAConfig | None = None) -> None:
        super().__init__(
            config=config or CapstoneQAConfig(),
            signature=CapstoneQASignature(),
        )


# ════════════════════════════════════════════════════════════════════════
# PACT GOVERNANCE — shared org yaml (D/T/R = Department / Team / Role)
# ════════════════════════════════════════════════════════════════════════
#
# The MLFP Capstone org has one department (AI Services, D1) headed by the
# ML Director role (D1-R1), and three teams, each headed by an agent role:
#   qa_agent    D1-R1-T1-R1  public clearance        — customer-facing answers
#   admin_agent D1-R1-T2-R1  confidential clearance  — model lifecycle / metrics
#   audit_agent D1-R1-T3-R1  secret clearance        — compliance access
# pact's ladder: public < restricted < confidential < secret < top_secret.
# Each agent role gets an envelope defined by the ML Director.

ORG_YAML: str = """
# MLFP Capstone ML Platform — PACT Governance Definition
# D/T/R = Department / Team / Role; every agent role reports to a human head

org_id: "mlfp_capstone"
name: "MLFP Capstone ML Platform"

# One department, headed by the ML Director (a human role).
departments:
  - id: "ai_services"
    name: "AI Services"

# Three teams — one per agent workstream.
teams:
  - id: "qa_team"
    name: "Question Answering"
  - id: "ops_team"
    name: "Model Operations"
  - id: "audit_team"
    name: "Compliance Audit"

roles:
  # ── Department head (human) ──
  - id: "ml_director"
    name: "ML Director"
    heads: "ai_services"

  # ── Team heads (agents) ──
  - id: "qa_agent"
    name: "QA Agent"
    reports_to: "ml_director"
    heads: "qa_team"
  - id: "admin_agent"
    name: "Admin Agent"
    reports_to: "ml_director"
    heads: "ops_team"
  - id: "audit_agent"
    name: "Audit Agent"
    reports_to: "ml_director"
    heads: "audit_team"

# Clearances — pact levels; every agent at or below its head.
clearances:
  - role: "ml_director"
    level: "secret"
  - role: "qa_agent"
    level: "public"
  - role: "admin_agent"
    level: "confidential"
  - role: "audit_agent"
    level: "secret"

# Envelopes = delegations (defined_by -> target). Applied to the engine by
# compile_capstone_governance().
envelopes:
  - target: "qa_agent"
    defined_by: "ml_director"
    financial:
      max_spend_usd: 1.0
    operational:
      allowed_actions: ["generate_answer", "search_context"]
    communication:
      max_response_length: 2000

  - target: "admin_agent"
    defined_by: "ml_director"
    financial:
      max_spend_usd: 10.0
    operational:
      allowed_actions:
        - "generate_answer"
        - "search_context"
        - "update_model"
        - "view_metrics"
        - "monitor_drift"

  - target: "audit_agent"
    defined_by: "ml_director"
    financial:
      max_spend_usd: 50.0
    operational:
      allowed_actions:
        - "generate_answer"
        - "search_context"
        - "view_metrics"
        - "access_audit_log"
        - "generate_report"
"""


def write_org_yaml(path: str | Path | None = None) -> str:
    """Write the shared capstone org YAML to a temp file and return the path."""
    if path is None:
        path = os.path.join(tempfile.gettempdir(), "capstone_org.yaml")
    with open(path, "w") as f:
        f.write(ORG_YAML)
    return str(path)


def compile_capstone_governance(
    yaml_path: str | None = None,
) -> tuple["GovernanceEngine", Any]:
    """Load the capstone YAML, build the engine, APPLY clearances + envelopes.

    ``GovernanceEngine(loaded.org_definition)`` alone compiles only the
    structure; until ``apply_governance_specs`` runs, no envelope is attached
    and ``verify_action`` auto-approves every role (the installed default).

    Returns ``(engine, loaded)`` where ``loaded`` is the ``LoadedOrg``.
    """
    from kailash.trust.pact.yaml_resolvers import apply_governance_specs
    from pact import GovernanceEngine, load_org_yaml

    loaded = load_org_yaml(yaml_path or write_org_yaml())
    engine = GovernanceEngine(loaded.org_definition)
    apply_governance_specs(engine, loaded)
    return engine, loaded


# ════════════════════════════════════════════════════════════════════════
# BUILD_CAPSTONE_STACK — shared 3-tier GovernedSupervisor builder
# ════════════════════════════════════════════════════════════════════════
#
# Role -> tier mapping (clearance on pact's ladder):
#   qa    -> public        (low budget, narrow tools)
#   admin -> confidential  (mid budget, ops tools)
#   audit -> secret        (high budget, audit tools)
# The three tiers are SIBLING envelopes, each defined by (and no wider than)
# the ML Director — they are not a superset chain of one another.
#
# The helper also attaches a full 5-dimension `ConstraintEnvelopeConfig` to
# each agent role address via `engine.set_role_envelope(...)` (replacing the
# thinner YAML envelope), so `engine.verify_action()` enforces the tier.

_QA_ADDR = "D1-R1-T1-R1"
_ADMIN_ADDR = "D1-R1-T2-R1"
_AUDIT_ADDR = "D1-R1-T3-R1"
_DIRECTOR_ADDR = "D1-R1"


@dataclass
class CapstoneTier:
    """Metadata for one GovernedSupervisor tier in the capstone stack."""

    role: str
    address: str
    budget_usd: float
    tools: list[str]
    clearance: str  # pact clearance string (public | confidential | secret ...)
    description: str


CAPSTONE_TIERS: list[CapstoneTier] = [
    CapstoneTier(
        role="qa",
        address=_QA_ADDR,
        budget_usd=1.0,
        tools=["generate_answer", "search_context"],
        clearance="public",
        description="Retail-facing QA — narrow tools, low budget",
    ),
    CapstoneTier(
        role="admin",
        address=_ADMIN_ADDR,
        budget_usd=10.0,
        tools=[
            "generate_answer",
            "search_context",
            "update_model",
            "view_metrics",
            "monitor_drift",
        ],
        clearance="confidential",
        description="Model ops — update + metrics + drift",
    ),
    CapstoneTier(
        role="audit",
        address=_AUDIT_ADDR,
        budget_usd=50.0,
        tools=[
            "generate_answer",
            "search_context",
            "view_metrics",
            "access_audit_log",
            "generate_report",
        ],
        clearance="secret",
        description="Compliance audit — full audit log + reports",
    ),
]
_TIER_BY_ROLE: dict[str, CapstoneTier] = {t.role: t for t in CAPSTONE_TIERS}


def _attach_envelopes(engine: "GovernanceEngine") -> None:
    """Attach a full ConstraintEnvelopeConfig to every agent role address.

    Without an envelope, `engine.verify_action()` on a role auto-approves
    ("No envelope constraints -- action permitted"). The envelope is the
    source of restriction.
    """
    from pact import (
        CommunicationConstraintConfig,
        ConfidentialityLevel,
        ConstraintEnvelopeConfig,
        DataAccessConstraintConfig,
        FinancialConstraintConfig,
        OperationalConstraintConfig,
        RoleEnvelope,
        TemporalConstraintConfig,
    )

    for tier in CAPSTONE_TIERS:
        envelope = ConstraintEnvelopeConfig(
            id=f"{tier.role}_envelope",
            description=tier.description,
            confidentiality_clearance=ConfidentialityLevel(tier.clearance),
            financial=FinancialConstraintConfig(max_spend_usd=tier.budget_usd),
            operational=OperationalConstraintConfig(
                allowed_actions=list(tier.tools),
                blocked_actions=[],
            ),
            temporal=TemporalConstraintConfig(blackout_periods=[]),
            data_access=DataAccessConstraintConfig(
                read_paths=["/public/*"],
                write_paths=[],
                blocked_data_types=[],
            ),
            communication=CommunicationConstraintConfig(allowed_channels=["internal"]),
            max_delegation_depth=3,
        )
        engine.set_role_envelope(
            RoleEnvelope(
                id=f"{tier.role}_role_envelope",
                defining_role_address=_DIRECTOR_ADDR,
                target_role_address=tier.address,
                envelope=envelope,
            )
        )


def build_capstone_stack(
    engine: "GovernanceEngine",
) -> tuple[dict[str, GovernedSupervisor], list[CapstoneTier]]:
    """Build the shared 3-tier GovernedSupervisor stack.

    Returns:
        ``(agents_by_role, tiers)`` — ``agents_by_role`` maps
        ``"qa" | "admin" | "audit"`` to its ``GovernedSupervisor`` (the shape
        ``handle_qa`` expects); ``tiers`` is the ``CapstoneTier`` metadata.

    Side effect: attaches a ``ConstraintEnvelopeConfig`` to every agent role
    address on ``engine``.
    """
    _attach_envelopes(engine)

    agents_by_role: dict[str, GovernedSupervisor] = {}
    for tier in CAPSTONE_TIERS:
        agents_by_role[tier.role] = GovernedSupervisor(
            model=MODEL,
            budget_usd=tier.budget_usd,
            tools=list(tier.tools),
            data_clearance=tier.clearance,
        )
    return agents_by_role, list(CAPSTONE_TIERS)


# ════════════════════════════════════════════════════════════════════════
# SHARED QA HANDLER — used by Nexus deployment AND monitoring/test files
# ════════════════════════════════════════════════════════════════════════
#
# `handle_qa()`:
#   1. Refuses an unknown role (never falls back to another tier).
#   2. If an engine is passed, asks `engine.verify_action(tier address,
#      action)` first — a blocked verdict is returned as blocked.
#   3. Runs the tier's `GovernedSupervisor.run(objective, execute_node=...)`
#      whose executor calls the shared `CapstoneQAAgent` on local Ollama.
#   4. A node HELD by the budget envelope is returned as blocked; a FAILED
#      node (LLM error) RAISES — there is no offline stub.

_shared_qa_agent: CapstoneQAAgent | None = None


def _get_shared_agent() -> CapstoneQAAgent:
    global _shared_qa_agent
    if _shared_qa_agent is None:
        _shared_qa_agent = CapstoneQAAgent(CapstoneQAConfig())
    return _shared_qa_agent


async def _capstone_execute_node(_spec: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    """Executor callback for GovernedSupervisor.run() — a real LLM call.

    Returns ``{"result": <agent output dict>, "cost": 0.0}``. Cost is zero
    because local Ollama is free. Raises if the agent reports an error, so
    the supervisor marks the node FAILED and ``handle_qa`` surfaces it.
    """
    # GovernedSupervisor puts the objective in the plan node's AgentSpec
    # (``spec.description``); ``inputs`` only carries resolved upstream
    # outputs and is EMPTY for the root task. Reading only ``inputs`` sent the
    # model the literal string "{}".
    objective = (
        getattr(_spec, "description", None)
        or inputs.get("objective")
        or inputs.get("question")
        or inputs.get("prompt")
    )
    if not objective:
        raise ValueError("executor received no objective (spec.description and inputs empty)")
    out = await _get_shared_agent().run_async(question=str(objective))
    if not isinstance(out, dict) or out.get("error"):
        detail = out.get("error") if isinstance(out, dict) else repr(out)
        raise RuntimeError(f"CapstoneQAAgent failed: {detail}")
    if not str(out.get("answer", "")).strip():
        raise RuntimeError(f"CapstoneQAAgent returned no answer: {out!r}")
    return {"result": out, "cost": 0.0}


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


async def handle_qa(
    question: str,
    role: str,
    agents_by_role: dict[str, GovernedSupervisor],
    *,
    engine: "GovernanceEngine | None" = None,
    action: str = "generate_answer",
) -> dict[str, Any]:
    """Route a question to the governed supervisor for ``role``.

    Returns on success::

        {"answer": str, "confidence": float | None, "sources": list[str],
         "reasoning_steps": list[str], "latency_ms": float,
         "budget_consumed": float, "governed": True, "blocked": False,
         "verdict": "served", "role": str}

    ``confidence`` is the agent's own self-reported value (None if it did
    not return a number). On a governance refusal (unknown role, blocked
    verdict, budget HELD) returns ``{"error", "blocked": True, "verdict",
    "governed": True, "role"}``. An LLM failure raises ``RuntimeError``.
    """
    if role not in agents_by_role or role not in _TIER_BY_ROLE:
        return {
            "error": f"unknown role {role!r} — refused",
            "blocked": True,
            "verdict": "unknown_role",
            "governed": True,
            "role": role,
        }
    tier = _TIER_BY_ROLE[role]
    if engine is not None:
        verdict = engine.verify_action(tier.address, action, {"cost": 0.0})
        if not verdict.allowed:
            return {
                "error": verdict.reason,
                "blocked": True,
                "verdict": verdict.level,
                "governed": True,
                "role": role,
            }

    gs = agents_by_role[role]
    start = time.perf_counter()
    result = await gs.run(objective=question, execute_node=_capstone_execute_node)
    latency_ms = (time.perf_counter() - start) * 1000

    if not result.success:
        nodes = list(result.plan.nodes.values()) if result.plan else []
        if any(n.state.name == "HELD" for n in nodes):
            return {
                "error": "budget envelope exhausted — request held",
                "blocked": True,
                "verdict": "held",
                "governed": True,
                "role": role,
            }
        errors = [n.error for n in nodes if n.error]
        raise RuntimeError(
            f"capstone '{role}' tier: LLM call failed: {errors}. "
            "Start Ollama: ollama serve"
        )

    out = next(iter(result.results.values()))
    return {
        "answer": str(out.get("answer", "")),
        "confidence": _as_float(out.get("confidence")),
        "sources": list(out.get("sources") or []),
        "reasoning_steps": list(out.get("reasoning_steps") or []),
        "latency_ms": latency_ms,
        "budget_consumed": result.budget_consumed,
        "governed": True,
        "blocked": False,
        "verdict": "served",
        "role": role,
    }


# ════════════════════════════════════════════════════════════════════════
# RUN HELPER — for technique files that use asyncio at module scope
# ════════════════════════════════════════════════════════════════════════


def run_async(coro):  # noqa: ANN001 — coroutine
    """Run a coroutine to completion from synchronous code.

    In a plain script there is no running loop, so ``asyncio.run`` is used.
    Inside Jupyter/Colab a loop is already running, so the coroutine runs on
    a fresh loop in a worker thread. Exceptions from the coroutine propagate
    unchanged.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
