# MLFP06 Migration — Scoping Document

**Created**: 2026-04-14 from parallel pact-specialist + kaizen-specialist investigations.
**Target versions**: `kailash-pact==0.8.1`, `kailash-kaizen==2.7.3`, `kailash-nexus==2.0.1`, `kaizen-agents==0.9.2`.
**Source of truth for**: drift inventory, file inventory, pedagogical pivots, specialist findings.

## Headline Finding

`kaizen_agents.GovernedSupervisor(model, budget_usd, tools, data_clearance)` is a near-drop-in replacement for the legacy `PactGovernedAgent(agent, governance_engine, role, max_budget_usd, allowed_tools, clearance_level)` wrapper. Same three knobs, same `.audit.to_list()` readout, PLUS hash-chain tamper-evidence via `.audit.verify_chain()`. The teaching narrative survives with one correction (Task 4 reframe — see `decisions.md` § 1).

Migration shape is therefore **smaller than the initial drift investigation suggested**. The hard work concentrates in 4 places:

1. Rewriting ORG_YAML schemas in 2 shared helpers to modern pact format
2. Adapting `compile_governance()` callers to the new `(engine, CompiledOrg)` return via a compatibility adapter
3. Replacing `class XxxAgent(BaseAgent)` class-level-attribute pattern with dataclass-config + instance-signature pattern
4. Rewriting exam.py PACT block (lines 1107–1200) from `Address(domain=, team=, role=)` kwargs to `Address.parse("D1-R1-...")` dash grammar

## Pre-Existing Bug Bombs Discovered

These are silent failures that M6 has been shipping for an unknown period. The migration fixes real broken runtime behavior, not just cosmetic drift.

### Bug 1 — DefaultSignature Silent Fallback (kaizen-specialist finding)

Every `class XxxAgent(BaseAgent): signature = XxxSignature` in M6 **silently falls back to `DefaultSignature`**. `BaseAgent.__init__` only honors `signature` passed via the constructor argument, not a class-level attribute. The class attribute creates a no-op. Students have been getting generic default output regardless of what Signature they declared.

**Affected files**: 9+ (every file that defines `class XxxAgent(BaseAgent)` in `shared/mlfp06/ex_6.py`, `ex_8.py`, `modules/mlfp06/solutions/ex_5/{03,04}`, `ex_7/04`, `assessment/exam.py`).

**Fix** (applied uniformly in every shard): pass signature as instance via `super().__init__(config=QAConfig(), signature=QASignature())`.

### Bug 2 — `await agent.run(...)` Is A Type Error

Every M6 file does `result = await agent.run(question=...)`. But `agent.run(...)` is **synchronous** and returns a `dict`. Awaiting a dict raises `TypeError: object dict can't be used in 'await' expression`. The async entry point is `agent.run_async(...)`.

The exercises have been wrapped in `try/except` for offline fallback, which silently swallowed this error — so the exercises appeared to "work" (printed the exception and moved on) but no agent call ever succeeded.

**Fix**: `await agent.run_async(**inputs)` + dict-indexed result access (`result["answer"]`, not `result.answer`).

### Bug 3 — `max_llm_cost_usd` Is A No-Op

`BaseAgent` has no `max_llm_cost_usd` attribute. Class-level assignment creates a no-op attribute that nothing reads. Budget enforcement moved to `BaseAgentConfig.budget_limit_usd`.

**Fix**: move to `QAConfig(budget_limit_usd=5.0)` dataclass field.

### Bug 4 — exam.py Uses An Even Older API

`OperatingEnvelope`, `governance.register_envelope()`, `governance.can_access(addr, action=...)`, `governance.explain_access(...)` — none of these exist in pact 0.8.1. exam.py's PACT section has been untestable at import time for longer than the rest of M6.

**Fix**: full rewrite of lines 1107–1200 using `pact.Address.parse()`, `engine.verify_action()`, `engine.set_role_envelope(RoleEnvelope(...))`. See `api-cheatsheet.md` for canonical patterns.

### Bug 5 — `LLMCostTracker` Removed

`from kaizen_agents import LLMCostTracker` raises `ImportError`. 5 call sites fail at module load.

**Affected files**: `exam.py:1011`, `lessons/05/textbook.html:504,609`, `lessons/06/textbook.html:321,688`.

**Fix**: replace with `BaseAgentConfig.budget_limit_usd` narrative.

### Bug 6 — Textbook HTML Stale Imports

5+ stale import paths in HTML textbooks: `kaizen.protocol`, `kaizen.memory` three-tier taxonomy (`ShortTermMemory`/`LongTermMemory`/`EntityMemory`), top-level `ReActAgent`/`BaseAgent`/`LLMCostTracker`. These don't execute but teach students wrong paths.

**Fix**: corrections in Shard 9.

## File Inventory (52 Files)

### Shared helpers (3 files — foundation of the whole migration)

| File                    | What's broken                                                                                                                    |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `shared/mlfp06/ex_6.py` | `from kaizen.core import BaseAgent` + `class XxxAgent(BaseAgent): signature = ...` pattern for 5 specialist agents               |
| `shared/mlfp06/ex_7.py` | `from kailash_pact import GovernanceEngine` + `engine.compile_org(yaml_path)` + `ORG_YAML` uses legacy schema                    |
| `shared/mlfp06/ex_8.py` | Same kaizen imports + `CapstoneQAAgent` pattern + `ORG_YAML` legacy schema + `handle_qa()` calls `await agent.run(question=...)` |

### ex_5 — self-contained kaizen agents (4 exercise files + 2 Colabs)

| File                                                   | What's broken                                                         |
| ------------------------------------------------------ | --------------------------------------------------------------------- |
| `modules/mlfp06/solutions/ex_5/03_structured_agent.py` | kaizen import + `class DataAnalysisAgent(BaseAgent): signature = ...` |
| `modules/mlfp06/solutions/ex_5/04_critic_agent.py`     | Same                                                                  |
| `modules/mlfp06/local/ex_5/03_structured_agent.py`     | Same (mirror)                                                         |
| `modules/mlfp06/local/ex_5/04_critic_agent.py`         | Same (mirror)                                                         |
| `modules/mlfp06/colab/ex_5/03_structured_agent.ipynb`  | Regenerated from solution                                             |
| `modules/mlfp06/colab/ex_5/04_critic_agent.ipynb`      | Regenerated from solution                                             |

**Note**: ex_5 technique files define their own `BaseAgent` subclasses independent of `shared/mlfp06/ex_5.py`. They are NOT blocked by the shared helper.

### ex_6 — multi-agent orchestration (10 exercise files + 5 Colabs)

All 5 technique files (`01_supervisor_worker`, `02_sequential_pipeline`, `03_parallel_router`, `04_mcp_server`, `05_memory_and_security`) import specialist agents from `shared/mlfp06/ex_6.py`. Migration of the shared helper unblocks all 5 files.

### ex_7 — PACT governance (8 exercise files + 4 Colabs)

All 4 technique files (`01_org_compile`, `02_envelopes`, `03_budget_access`, `04_runtime_audit`) import from `shared/mlfp06/ex_7.py`. Migration of the shared helper unblocks all 4.

### ex_8 — Capstone (8 exercise files + 4 Colabs)

4 technique files (`02_governance_pipeline`, `03_multichannel_serving`, `04_drift_monitoring`, `05_compliance_audit`) import from `shared/mlfp06/ex_8.py`. `ex_8/01_adapter_loading.py` is not PACT-dependent but must be verified.

### exam.py (1 file, 1,531 LOC)

1,531-LOC capstone assessment. Uses an older API variant than the exercises:

- Line 104: wrong `from kaizen.core import Signature, InputField, OutputField` (should be `from kaizen import ...`)
- Line 751: wrong `from kaizen.core import BaseAgent` (should be `from kaizen.core.base_agent import BaseAgent`)
- Line 1011: `LLMCostTracker` ImportError
- Lines 1107–1200: entire PACT block uses `OperatingEnvelope`, `register_envelope`, `can_access`, `explain_access` — all non-existent in pact 0.8.1
- Line 1334: `from kailash_nexus import Nexus` (should be `from nexus import Nexus`)

Load-bearing changes are ~100 LOC; the other ~1,400 lines of business logic (budget cascading, pipeline processing, MMLU eval) are unchanged.

### Textbook HTML (2 files)

| File                                      | Stale imports                                                                           |
| ----------------------------------------- | --------------------------------------------------------------------------------------- |
| `modules/mlfp06/lessons/05/textbook.html` | `LLMCostTracker` at lines 504, 609                                                      |
| `modules/mlfp06/lessons/06/textbook.html` | `kaizen.protocol`, `kaizen.memory` three-tier, top-level `ReActAgent`, `LLMCostTracker` |

## Kaizen Findings (from kaizen-specialist)

### Canonical import paths

| Symbol                                   | Canonical path                                                                             |
| ---------------------------------------- | ------------------------------------------------------------------------------------------ |
| `BaseAgent`                              | `from kaizen.core.base_agent import BaseAgent` (submodule, NOT top-level or `kaizen.core`) |
| `Signature`, `InputField`, `OutputField` | `from kaizen import Signature, InputField, OutputField` (top-level, NOT `kaizen.core`)     |
| `Delegate`                               | `from kaizen_agents import Delegate` (unchanged)                                           |
| `GovernedSupervisor`                     | `from kaizen_agents import GovernedSupervisor` (NEW — replaces PactGovernedAgent)          |

### Canonical BaseAgent pattern

**OLD (broken — silent DefaultSignature fallback)**:

```python
from kaizen.core import BaseAgent  # ImportError

class QAAgent(BaseAgent):
    signature = QASignature   # class attribute — IGNORED
    model = default_model_name()   # no-op
    max_llm_cost_usd = 5.0   # no-op
```

**NEW (canonical, verified hands-on)**:

```python
import os
from dataclasses import dataclass
from kaizen.core.base_agent import BaseAgent
from kaizen import Signature, InputField, OutputField

@dataclass
class QAConfig:
    llm_provider: str = os.environ.get("LLM_PROVIDER", "openai")
    model: str = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    temperature: float = 0.2
    budget_limit_usd: float = 5.0  # replaces max_llm_cost_usd

class QASignature(Signature):
    question: str = InputField(description="User's question")
    answer: str = OutputField(description="Governed response")
    confidence: float = OutputField(description="0-1")

class QAAgent(BaseAgent):
    def __init__(self, config: QAConfig):
        super().__init__(config=config, signature=QASignature())
```

Key deltas: (1) signature is an **instance** passed to `super().__init__`, (2) config is a dataclass with `model` and `budget_limit_usd` fields, (3) no class attributes — everything through the constructor.

### Result shape

`agent.run_async(**inputs)` returns a **dict** keyed by OutputField names. NOT an attribute-access object.

```python
# DO
result = await agent.run_async(question="What is ML?")
print(result["answer"])
print(result["confidence"])

# DO NOT
result = await agent.run(question="What is ML?")  # sync — returns dict, can't await
print(result.answer)  # dict, no attribute access
```

### Pyright cosmetic issue

`Signature` inherits from `object` (not Pydantic). `InputField`/`OutputField` are plain descriptors. Pyright complains: `Expression of type 'InputField' is incompatible with declared type 'str'`. This is **cosmetic** — the signature works at runtime.

**Fix** (Shard 0): add `pyrightconfig.json` with `reportAssignmentType: none` scoped to `modules/mlfp06/**` + `shared/mlfp06/**`.

## Pact Findings (from pact-specialist)

### Canonical imports

| Symbol                                                                          | Canonical path                                                                     |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `GovernedSupervisor`                                                            | `from kaizen_agents import GovernedSupervisor` (the PactGovernedAgent replacement) |
| `GovernanceEngine`, `Address`, `RoleEnvelope`, `ConstraintEnvelopeConfig`       | `from pact import ...`                                                             |
| `load_org_yaml`, `compile_org`, `can_access`, `explain_access`, `governed_tool` | `from pact import ...` (module-level functions)                                    |
| `TrustPostureLevel`, `ConfidentialityLevel`, `EnforcementMode`                  | `from pact import ...`                                                             |

### Canonical GovernedSupervisor pattern

**OLD (broken — wrapper API no longer exists)**:

```python
from kailash_pact import PactGovernedAgent

governed = PactGovernedAgent(
    agent=base_qa,
    governance_engine=engine,
    role="analyst",
    max_budget_usd=5.0,
    allowed_tools=["answer_question", "search_faq"],
    clearance_level="public",
)

result = await governed.run(question="...")
trail = governed.get_audit_trail()
```

**NEW (canonical)**:

```python
from kaizen_agents import GovernedSupervisor

governed = GovernedSupervisor(
    model="gpt-4o-mini",
    budget_usd=5.0,
    tools=["answer_question", "search_faq"],
    data_clearance="public",   # "public" | "internal" | "restricted" | "confidential" | "secret" | "top_secret"
)

# Two-layer run: supervisor plans, execute_node executes
async def executor(spec, inputs):
    # Real LLM call OR deterministic stub
    return {"result": "...", "cost": 0.01, "prompt_tokens": 100, "completion_tokens": 50}

result = await governed.run(objective="...", execute_node=executor)
# result.success, result.budget_consumed, result.audit_trail

trail = governed.audit.to_list()
valid = governed.audit.verify_chain()   # hash-chain tamper-evidence — NEW
```

### Canonical GovernanceEngine construction

**OLD**:

```python
engine = GovernanceEngine()
org = engine.compile_org(yaml_path)   # instance method
print(org.n_agents, org.n_delegations, org.n_departments)
```

**NEW**:

```python
from pact import GovernanceEngine, load_org_yaml

loaded = load_org_yaml(yaml_path)   # returns LoadedOrg(org_definition, clearances, envelopes, bridges, ksps)
engine = GovernanceEngine(loaded.org_definition)   # compiled at construction
compiled = engine.get_org()   # returns CompiledOrg(org_id, nodes)

# .n_agents / .n_delegations / .n_departments DO NOT EXIST on CompiledOrg
# Compute from nodes, or wrap in an adapter (see shard 3 — CompiledOrgAdapter)
```

### Canonical access check

**OLD**:

```python
decision = engine.check_access(
    agent_id="model_trainer",
    resource="training_data",
    action="read",
)
print(decision.allowed)
```

**NEW**:

```python
verdict = engine.verify_action(
    role_address="D1-R1-D2-R1-T2-R1",   # dash-delimited D/T/R address
    action="read",
    context={"cost": 0.10, "data_classification": "confidential"},
)
print(verdict.allowed, verdict.level, verdict.reason)
```

`verdict.level` is one of `"allowed"`, `"blocked"`, `"warn"`, `"audit"` — the 4 enforcement modes.

### Canonical envelope construction

**OLD (`exam.py` only)**:

```python
from kailash_pact import OperatingEnvelope
env = OperatingEnvelope(
    role="officer",
    allowed_actions=["classify", "respond"],
    denied_actions=["escalate"],
    max_cost_per_action=0.05,
)
governance.register_envelope("officer", env)
```

**NEW**:

```python
from pact import (
    RoleEnvelope, ConstraintEnvelopeConfig,
    FinancialConstraintConfig, OperationalConstraintConfig,
    TemporalConstraintConfig, DataAccessConstraintConfig,
    CommunicationConstraintConfig, ConfidentialityLevel,
)

envelope = ConstraintEnvelopeConfig(
    id="officer_envelope",
    description="Citizen service officer envelope",
    confidentiality_clearance=ConfidentialityLevel.RESTRICTED,
    financial=FinancialConstraintConfig(max_spend_usd=0.05),
    operational=OperationalConstraintConfig(allowed_actions=["classify", "respond"]),
    temporal=TemporalConstraintConfig(),
    data_access=DataAccessConstraintConfig(),
    communication=CommunicationConstraintConfig(),
    max_delegation_depth=3,
)

role_env = RoleEnvelope(
    id="officer_role_envelope",
    defining_role_address="D1-R1",  # department head
    target_role_address="D1-R1-T1-R1",  # officer
    envelope=envelope,
)
engine.set_role_envelope(role_env)

# Monotonic tightening is enforced structurally — this replaces the manual
# CLEARANCE_LEVELS integer comparison in the legacy code
RoleEnvelope.validate_tightening(parent_envelope, child_envelope)  # raises MonotonicTighteningError
```

The 5 dimensions (Financial, Operational, Temporal, Data Access, Communication) are canonical — the course's current materials **describe** these 5 in narrative but only **implement** 3 (budget, tools, clearance). The migration gets the remaining 2 (temporal, communication) for free from the modern config classes.

### Canonical Address grammar

**OLD (`exam.py` only)**:

```python
addr = Address(domain="feedback", team="operations", role="officer")
```

**NEW**:

```python
from pact import Address
addr = Address.parse("D1-R1-T1-R1")   # dash-delimited positional grammar
# Or: addr = Address.from_segments((AddressSegment(kind="D", index=1), ...))
```

Grammar rule: every `D` or `T` must be immediately followed by exactly one `R`. The address carries the full delegation chain in its prefix — "D1-R1-T1-R1" means "department 1 head, who owns task 1, delegated to responsible 1".

### Canonical audit trail

`supervisor.audit.to_list()` returns `list[dict]` where each entry has keys: `record_id`, `record_type`, `timestamp`, `agent_id`, `parent_id`, `action`, `details`, `prev_hash`, `record_hash`. The hash chain is tamper-evident — `supervisor.audit.verify_chain() -> bool`. Also `supervisor.audit.query_by_agent(agent_id)` for filtered reads.

This is a strict pedagogical upgrade for Exercise 8.5's compliance audit narrative: "audit trails are tamper-evident by construction" is a legitimate teaching point that the old API did not support.

### ORG_YAML schema change

The legacy `organization: / departments: / delegations: / operating_envelopes:` schema is NOT compatible with `pact.load_org_yaml()`. The modern schema uses `org: / departments[] / teams[] / agents[] / envelopes[] / workspaces[]` at the top level, with agents carrying `role:` + `constraint_envelope:` ID references.

The SG FinTech narrative (ML Engineering, Risk & Compliance, Customer Intelligence departments with 6 delegations) survives unchanged — only the YAML keys change. Similarly for the MLFP Capstone org. Rewrites happen in Shards 3 (ex_7 YAML) and 6 (ex_8 YAML). See `api-cheatsheet.md` § ORG_YAML for side-by-side examples.

## Unresolved Verification

One lightweight check deferred to Shard 7: verify `Nexus().register(serve_qa)` still accepts a bare async function via `inspect.signature(Nexus.register)`. Fallback if broken: wrap `serve_qa` in a single-node `WorkflowBuilder` (a pattern the course already teaches in earlier modules). No approval needed; see `decisions.md` § 5.

## What Wasn't A Blocker

The pact-specialist initially flagged shards 4–5 as blocked on kaizen-specialist confirmation of the structured-output path. The kaizen-specialist answered in parallel: `agent.run_async()` returns a dict keyed by OutputField names, so `result["answer"]` replaces `result.answer` everywhere. BaseAgent is alive at `kaizen.core.base_agent.BaseAgent`, the subclassing pattern survives — just with instance-signature in `__init__` instead of class-level attribute. No unblocking needed.
