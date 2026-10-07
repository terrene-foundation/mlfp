# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP06 Exercise 7 — AI Governance with PACT.

Contains: adversarial-prompt loading, canonical Singapore FinTech org YAML,
pact's clearance ladder, teaching budget tracker, GovernanceEngine compile
helper (which APPLIES the YAML clearances + envelopes to the engine),
CompiledOrgAdapter (preserves the `.n_agents / .n_delegations / .n_departments`
caller contract for technique files), a clearance-chain checker, and
`make_llm_executor()` — the `GovernedSupervisor.run(execute_node=...)`
callback that makes a REAL call to the local Ollama model.

Technique-specific code does NOT belong here — each technique file builds
its own scenario on top.

Import from any cwd after `uv sync`:

    from shared.mlfp06.ex_7 import (
        CLEARANCE_LEVELS, ORG_YAML, load_adversarial_prompts,
        write_org_yaml, compile_governance, TeachingBudgetTracker,
        CompiledOrgAdapter, clearance_chain_violations, default_model_name,
        make_llm_executor,
    )
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import polars as pl

from shared.kailash_helpers import setup_environment

setup_environment()

if TYPE_CHECKING:  # pragma: no cover — type-only imports
    from kailash.trust.pact.yaml_loader import LoadedOrg
    from pact import CompiledOrg, GovernanceEngine

# ════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════

# pact's clearance ladder, lowest to highest — exactly the order of
# `pact.ConfidentialityLevel`:
#   PUBLIC < RESTRICTED < CONFIDENTIAL < SECRET < TOP_SECRET
# "restricted" is the SECOND-LOWEST rung (just above public), NOT the top.
# kaizen_agents also accepts "internal" as an alias of RESTRICTED.
CLEARANCE_LEVELS: dict[str, int] = {
    "public": 0,
    "restricted": 1,
    "confidential": 2,
    "secret": 3,
    "top_secret": 4,
}


# Default LLM (lazy-resolved; agents read at construction time)
def default_model_name() -> str | None:
    from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL

    return DEFAULT_CHAT_MODEL


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
# governance structure. The D/T/R grammar is pact's addressing grammar:
#   D (Department): an organisational unit        e.g. D1 = ML Engineering
#   T (Team):       a unit inside a department    e.g. D1-R1-T1 = Data Analysis
#   R (Role):       a position (human or agent)   e.g. D1-R1-T1-R1 = data_analyst
# Every D or T is immediately followed by its head R, so an address such
# as D1-R1-T1-R1 reads "the role heading Team 1, under the role heading
# Department 1". Delegation is a SEPARATE envelope concept: an envelope
# is defined by one role (`defined_by`) for another (`target`).
#
# Modern pact's load_org_yaml expects a flat top-level schema:
#   org_id, name
#   departments[] teams[] roles[] clearances[] envelopes[]
#
# Roles attach to units via `heads: <dept_or_team_id>` and chain via
# `reports_to: <role_id>`. Envelopes carry the delegation contract
# (financial + operational + ...); each `envelopes[]` entry is one
# D/T/R delegation in the compiled org.

ORG_YAML: str = """
# Singapore FinTech AI Organisation — PACT Governance Definition
# D/T/R = Department / Team / Role; every agent role reports to a human head

org_id: "sg_fintech_ai"
name: "SG FinTech AI Division"

# Three departments — each headed by a named human role.
departments:
  - id: "ml_eng"
    name: "ML Engineering"
  - id: "risk_compliance"
    name: "Risk and Compliance"
  - id: "customer_intel"
    name: "Customer Intelligence"

# One team per agent workstream.
teams:
  - id: "data_team"
    name: "Data Analysis"
  - id: "training_team"
    name: "Model Training"
  - id: "deploy_team"
    name: "Model Deployment"
  - id: "risk_team"
    name: "Risk Assessment"
  - id: "bias_team"
    name: "Bias Audit"
  - id: "customer_team"
    name: "Customer Interaction"

# Department heads (3 humans) + team heads (6 agents).
roles:
  # ── Department heads (humans) ──
  - id: "chief_ml_officer"
    name: "Chief ML Officer"
    heads: "ml_eng"
  - id: "chief_risk_officer"
    name: "Chief Risk Officer"
    heads: "risk_compliance"
  - id: "vp_customer"
    name: "VP Customer"
    heads: "customer_intel"

  # ── Team heads (agents), each reporting to a department head ──
  - id: "data_analyst"
    name: "Data Analyst"
    reports_to: "chief_ml_officer"
    heads: "data_team"
  - id: "model_trainer"
    name: "Model Trainer"
    reports_to: "chief_ml_officer"
    heads: "training_team"
  - id: "model_deployer"
    name: "Model Deployer"
    reports_to: "chief_ml_officer"
    heads: "deploy_team"
  - id: "risk_assessor"
    name: "Risk Assessor"
    reports_to: "chief_risk_officer"
    heads: "risk_team"
  - id: "bias_checker"
    name: "Bias Checker"
    reports_to: "chief_risk_officer"
    heads: "bias_team"
  - id: "customer_agent"
    name: "Customer Agent"
    reports_to: "vp_customer"
    heads: "customer_team"

# Clearances — pact levels, lowest to highest:
#   public < restricted < confidential < secret < top_secret
# Every agent's clearance is at or below the head it reports to.
clearances:
  - role: "chief_ml_officer"
    level: "secret"
  - role: "chief_risk_officer"
    level: "secret"
  - role: "vp_customer"
    level: "confidential"
  - role: "data_analyst"
    level: "restricted"
  - role: "model_trainer"
    level: "confidential"
  - role: "model_deployer"
    level: "confidential"
  - role: "risk_assessor"
    level: "secret"
  - role: "bias_checker"
    level: "confidential"
  - role: "customer_agent"
    level: "public"

# Envelopes = delegations. Each entry is defined by one role
# (`defined_by`, the human head) for another (`target`, the agent) and
# carries the constraint set the agent runs within. compile_governance()
# APPLIES these to the engine, so verify_action() enforces them.
envelopes:
  - target: "data_analyst"
    defined_by: "chief_ml_officer"
    financial:
      max_spend_usd: 20.0
    operational:
      allowed_actions: ["read_data", "summarise_data", "generate_report"]
    data_access:
      max_rows: 500000

  - target: "model_trainer"
    defined_by: "chief_ml_officer"
    financial:
      max_spend_usd: 100.0
    operational:
      allowed_actions: ["train_model", "evaluate_model", "read_data"]
    data_access:
      max_rows: 1000000

  - target: "model_deployer"
    defined_by: "chief_ml_officer"
    financial:
      max_spend_usd: 50.0
    operational:
      allowed_actions: ["deploy_model", "monitor_model", "rollback_model"]

  - target: "risk_assessor"
    defined_by: "chief_risk_officer"
    financial:
      max_spend_usd: 200.0
    operational:
      allowed_actions:
        - "read_data"
        - "audit_model"
        - "generate_report"
        - "access_audit_log"

  - target: "bias_checker"
    defined_by: "chief_risk_officer"
    financial:
      max_spend_usd: 75.0
    operational:
      allowed_actions: ["read_data", "audit_model", "run_fairness_check"]

  - target: "customer_agent"
    defined_by: "vp_customer"
    financial:
      max_spend_usd: 5.0
    operational:
      allowed_actions: ["answer_question", "search_faq"]
    communication:
      max_response_length: 500
"""


def write_org_yaml(path: str | Path | None = None) -> str:
    """Write the canonical org YAML to a temp file and return the path."""
    if path is None:
        path = os.path.join(tempfile.gettempdir(), "sg_fintech_org.yaml")
    with open(path, "w") as f:
        f.write(ORG_YAML)
    return str(path)


# ════════════════════════════════════════════════════════════════════════
# COMPILED ORG ADAPTER — caller-contract shim
# ════════════════════════════════════════════════════════════════════════
#
# Modern pact's `CompiledOrg` exposes `org_id` and `nodes` — a flat
# dict of addresses (e.g. "D1-R1-T1-R1") to `OrgNode` objects with a
# `node_type` enum (DEPARTMENT | TEAM | ROLE). The MLFP06 course code
# has always used friendlier counters: `org.n_agents`,
# `org.n_delegations`, `org.n_departments`. Rather than rewrite every
# technique file, the adapter computes those counters from the flat
# nodes dict + the original envelopes list.


@dataclass
class CompiledOrgAdapter:
    """Thin facade over `pact.CompiledOrg` preserving the course's counter API.

    Agents in MLFP06 are the non-vacant ROLE nodes that head a TEAM — the
    6 team-head roles in the SG FinTech org. The 3 department-head roles
    are humans, not agents. Delegations are envelopes; there is one
    envelope per delegation.
    """

    _compiled: "CompiledOrg"
    _n_envelopes: int
    _loaded: "LoadedOrg | None" = None

    @property
    def clearances(self) -> dict[str, str]:
        """role_id -> clearance string, read from the loaded YAML."""
        if self._loaded is None:
            return {}
        return {c.role_id: c.level for c in self._loaded.clearances}

    @property
    def envelope_specs(self) -> list[Any]:
        """The YAML envelope specs (one per delegation)."""
        return [] if self._loaded is None else list(self._loaded.envelopes)

    def address_of(self, role_id: str) -> str:
        """Positional D/T/R address of a role id (e.g. 'D1-R1-T1-R1')."""
        for addr, node in self._compiled.nodes.items():
            if node.role_definition is not None and node.role_definition.role_id == role_id:
                return addr
        raise KeyError(f"role {role_id!r} is not in the compiled org")

    @property
    def n_departments(self) -> int:
        from pact import NodeType

        return sum(
            1
            for n in self._compiled.nodes.values()
            if n.node_type == NodeType.DEPARTMENT
        )

    @property
    def n_teams(self) -> int:
        from pact import NodeType

        return sum(
            1 for n in self._compiled.nodes.values() if n.node_type == NodeType.TEAM
        )

    @property
    def n_agents(self) -> int:
        """Count agent roles (team-head ROLE nodes, non-vacant).

        An agent role is a ROLE node whose address sits under a TEAM (has
        a `-T<n>-R<n>` suffix) — distinguishing it from the department-head
        ROLE nodes that sit directly under a DEPARTMENT address (e.g.
        "D1-R1"). Vacant placeholders are excluded.
        """
        from pact import NodeType

        count = 0
        for addr, node in self._compiled.nodes.items():
            if node.node_type != NodeType.ROLE:
                continue
            if node.is_vacant:
                continue
            # Department-head addresses are "D<n>-R<n>" — two segments.
            # Agent-role addresses sit under a team and have
            # the "-T<n>-R<n>" suffix, giving four or more segments.
            if "-T" in addr:
                count += 1
        return count

    @property
    def n_delegations(self) -> int:
        """One envelope == one delegation (defined_by -> target)."""
        return self._n_envelopes

    @property
    def org_id(self) -> str:
        return self._compiled.org_id


# ════════════════════════════════════════════════════════════════════════
# GOVERNANCE ENGINE COMPILATION
# ════════════════════════════════════════════════════════════════════════


def compile_governance(
    yaml_path: str | None = None,
    *,
    apply_specs: bool = True,
) -> tuple["GovernanceEngine", CompiledOrgAdapter]:
    """Compile the canonical org YAML. Returns (engine, adapter).

    Flow:
        loaded   <- load_org_yaml(path)
        engine   <- GovernanceEngine(loaded.org_definition)
        apply_governance_specs(engine, loaded)   # clearances + envelopes
        adapter  <- CompiledOrgAdapter(engine.get_org(), ...)

    ``GovernanceEngine(loaded.org_definition)`` on its own compiles ONLY
    the structure (departments, teams, roles). The YAML ``clearances`` and
    ``envelopes`` are separate specs; until they are applied the engine
    has no envelopes and ``verify_action`` auto-approves every role.
    ``apply_specs=True`` (the default) applies them so the YAML is
    actually enforced. Pass ``apply_specs=False`` to see the bare
    structural compile.

    What loading + compiling checks (installed kailash-pact):
      - every role references a known unit via ``heads``
      - ``reports_to`` chains resolve to declared roles
      - clearance strings are valid pact levels
      - envelope ``target`` / ``defined_by`` resolve to real roles
      - applied envelopes do not widen the defining role's own envelope
    What it does NOT check:
      - that a role's clearance is at or below its head's clearance
        (use ``clearance_chain_violations()`` below)
      - content safety of LLM outputs (needs adversarial testing)
      - roles WITHOUT an envelope, or unknown addresses: the installed
        default auto-approves them ("No envelope constraints -- action
        permitted"). Deny paths only exist where an envelope is attached.
    """
    from kailash.trust.pact.yaml_resolvers import apply_governance_specs
    from pact import GovernanceEngine, load_org_yaml

    if yaml_path is None:
        yaml_path = write_org_yaml()
    loaded = load_org_yaml(yaml_path)
    engine = GovernanceEngine(loaded.org_definition)
    if apply_specs:
        apply_governance_specs(engine, loaded)
    compiled = engine.get_org()
    adapter = CompiledOrgAdapter(
        _compiled=compiled,
        _n_envelopes=len(loaded.envelopes),
        _loaded=loaded,
    )
    return engine, adapter


def clearance_chain_violations(yaml_path: str | None = None) -> pl.DataFrame:
    """Return every role whose clearance is ABOVE the role it reports to.

    pact does not reject such an org at load time, so this check is ours.
    Uses pact's ladder (``CLEARANCE_LEVELS``). An empty DataFrame means
    every reporting chain is monotonic (child <= parent).
    """
    from pact import load_org_yaml

    loaded = load_org_yaml(yaml_path or write_org_yaml())
    level_of = {c.role_id: c.level for c in loaded.clearances}
    rows = []
    for role in loaded.org_definition.roles:
        parent = role.reports_to_role_id
        if not parent or role.role_id not in level_of or parent not in level_of:
            continue
        child_level, parent_level = level_of[role.role_id], level_of[parent]
        if CLEARANCE_LEVELS[child_level] > CLEARANCE_LEVELS[parent_level]:
            rows.append(
                {
                    "role": role.role_id,
                    "role_clearance": child_level,
                    "reports_to": parent,
                    "parent_clearance": parent_level,
                }
            )
    schema = {
        "role": pl.Utf8,
        "role_clearance": pl.Utf8,
        "reports_to": pl.Utf8,
        "parent_clearance": pl.Utf8,
    }
    return pl.DataFrame(rows, schema=schema)


# ════════════════════════════════════════════════════════════════════════
# TEACHING BUDGET TRACKER
# ════════════════════════════════════════════════════════════════════════
#
# Named `TeachingBudgetTracker` to avoid collision with internal
# `pact.BudgetTracker` / `pact.CostTracker` primitives. This is a
# simple pedagogical object for illustrating parent-to-child budget
# cascading in ex_7/03_budget_access — NOT a production substitute
# for the governance engine's own financial envelope enforcement.


class TeachingBudgetTracker:
    """Track budget allocation and consumption across an agent hierarchy.

    Parent allocates to children; children cannot spend more than their
    allocation. Used by ex_7/03_budget_access to demonstrate the
    monotonic-tightening property of financial envelopes in a form
    students can step through by hand.
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


# ════════════════════════════════════════════════════════════════════════
# LLM EXECUTOR — real `GovernedSupervisor.run(execute_node=...)` callback
# ════════════════════════════════════════════════════════════════════════
#
# `GovernedSupervisor.run(objective, execute_node=...)` builds a plan and
# invokes `execute_node(spec, inputs)` for each node. The executor is
# where the LLM call lives. The supervisor tracks the cost the executor
# reports against its financial envelope and appends audit records.
#
# The executor below makes a REAL call to the local Ollama model through
# the course's `make_delegate()` factory. There is no offline stub: if
# Ollama is not running the call raises, the supervisor records the node
# as FAILED, and the technique file reports it. Run `ollama serve` first.
#
# Ollama is free, so there is no real dollar cost. To make the financial
# dimension observable, the executor charges a NOTIONAL price per 1,000
# tokens (`notional_usd_per_1k_tokens`). It is a teaching device, not a
# bill — say so whenever you print it.


def make_llm_executor(
    *,
    notional_usd_per_1k_tokens: float = 0.0,
    system_prompt: str | None = None,
) -> Callable[[Any, dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Build an async executor that calls the local Ollama model.

    The returned callable accepts ``(spec, inputs)`` and returns the four
    keys GovernedSupervisor reads: ``result``, ``cost``, ``prompt_tokens``,
    ``completion_tokens``. ``cost`` is ``total_tokens / 1000 *
    notional_usd_per_1k_tokens`` (0.0 by default — Ollama is free).
    """
    from shared.mlfp06._ollama_bootstrap import make_delegate, run_delegate_text

    async def _execute(spec: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        # The objective lives in the plan node's AgentSpec.description; inputs
        # is empty for the root task (reading only inputs sent the model "{}").
        objective = (
            getattr(spec, "description", None)
            or inputs.get("objective")
            or inputs.get("prompt")
        )
        if not objective:
            raise ValueError(
                "executor received no objective (spec.description and inputs empty)"
            )
        delegate = make_delegate(system_prompt=system_prompt)
        text, usage, _latency = await run_delegate_text(delegate, str(objective))
        if not text.strip():
            raise RuntimeError("LLM returned an empty response")
        total = usage.get("total_tokens", 0)
        return {
            "result": text,
            "cost": total / 1000.0 * float(notional_usd_per_1k_tokens),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }

    return _execute
