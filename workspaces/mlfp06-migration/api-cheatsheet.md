# MLFP06 API Cheatsheet — Old → New

**Purpose**: Copy-paste reference for every migration edit. Use this as the source of truth during shard execution.
**Verified against**: `kailash-pact==0.8.1`, `kailash-kaizen==2.7.3`, `kailash-nexus==2.0.1`, `kaizen-agents==0.9.2`.

## Import Renames

| Old                                                                       | New                                                                                  |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `from kailash_pact import ...`                                            | `from pact import ...`                                                               |
| `from kailash_nexus import ...`                                           | `from nexus import ...`                                                              |
| `from kaizen.core import BaseAgent`                                       | `from kaizen.core.base_agent import BaseAgent`                                       |
| `from kaizen.core import Signature, InputField, OutputField`              | `from kaizen import Signature, InputField, OutputField`                              |
| `from kaizen_agents import LLMCostTracker`                                | **REMOVED** — use `BaseAgentConfig.budget_limit_usd`                                 |
| `from kaizen import ReActAgent`                                           | `from kaizen_agents.agents.specialized.react import ReActAgent`                      |
| `from kaizen.protocol import A2AMessage`                                  | `from kaizen import A2ATask, A2AAgentCard` (top-level)                               |
| `from kaizen.memory import ShortTermMemory, LongTermMemory, EntityMemory` | Use `kaizen.memory.BufferMemory` or `SharedMemoryPool` — three-tier taxonomy is gone |

## Kaizen — BaseAgent Pattern

### Old (broken — silent DefaultSignature fallback)

```python
from kaizen import InputField, OutputField, Signature
from kaizen.core import BaseAgent

class QASignature(Signature):
    question: str = InputField(description="User's question")
    answer: str = OutputField(description="Governed response")
    confidence: float = OutputField(description="Answer confidence 0-1")

class QAAgent(BaseAgent):
    signature = QASignature           # class attribute — IGNORED by __init__
    model = default_model_name()      # no-op — model lives on config
    max_llm_cost_usd = 5.0            # no-op — budget moved to config

base_qa = QAAgent()
result = await base_qa.run(question="What is ML?")   # TypeError — sync method, returns dict
print(result.answer)                                  # AttributeError — result is dict
```

### New (canonical)

```python
import os
from dataclasses import dataclass
from kaizen.core.base_agent import BaseAgent
from kaizen import Signature, InputField, OutputField

@dataclass
class QAConfig:
    """Domain config — BaseAgent auto-converts to BaseAgentConfig."""
    llm_provider: str = os.environ.get("LLM_PROVIDER", "openai")
    model: str = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    temperature: float = 0.2
    budget_limit_usd: float = 5.0   # replaces max_llm_cost_usd

class QASignature(Signature):
    question: str = InputField(description="User's question")
    answer: str = OutputField(description="Governed response")
    confidence: float = OutputField(description="Answer confidence 0-1")

class QAAgent(BaseAgent):
    def __init__(self, config: QAConfig):
        super().__init__(config=config, signature=QASignature())   # instance, not class

# Call site
base_qa = QAAgent(QAConfig())
try:
    result = await base_qa.run_async(question="What is ML?")      # async entry point
    print(result["answer"])                                       # dict, not attribute
    print(result["confidence"])
except Exception as exc:
    # Offline / missing-key graceful fallback
    print(f"[offline] agent run skipped: {type(exc).__name__}: {exc}")
```

**Critical deltas**:

1. `from kaizen.core.base_agent import BaseAgent` — submodule path, NOT `kaizen.core`
2. `from kaizen import Signature, InputField, OutputField` — top-level, NOT `kaizen.core`
3. Config is a **dataclass** with `model` and `budget_limit_usd` fields
4. Signature is passed as **instance** to `super().__init__(signature=QASignature())` — NOT class-level attribute
5. Async entry is `agent.run_async(**inputs)`, NOT `await agent.run(...)` (sync method returns dict, can't await)
6. Result is a **dict** keyed by OutputField names — `result["answer"]`, NOT `result.answer`

## Pact — GovernedSupervisor (replaces PactGovernedAgent)

### Old (broken — PactGovernedAgent no longer a wrapper)

```python
from kailash_pact import PactGovernedAgent

governed_public = PactGovernedAgent(
    agent=base_qa,                              # wrapped Kaizen agent
    governance_engine=engine,
    role="analyst",
    max_budget_usd=5.0,
    allowed_tools=["answer_question", "search_faq"],
    clearance_level="public",
)

result = await governed_public.run(question="What is ML?")
trail = governed_public.get_audit_trail()
```

### New (canonical)

```python
from kaizen_agents import GovernedSupervisor

governed_public = GovernedSupervisor(
    model="gpt-4o-mini",
    budget_usd=5.0,
    tools=["answer_question", "search_faq"],
    data_clearance="public",    # canonical strings: public, internal, restricted, confidential, secret, top_secret
)

# Two-layer run: supervisor plans, execute_node is YOUR callback that runs the LLM or stub
async def executor(spec, inputs):
    """This is where your real LLM call (or offline stub) lives."""
    return {
        "result": "Machine learning is...",
        "cost": 0.01,
        "prompt_tokens": 100,
        "completion_tokens": 50,
    }

result = await governed_public.run(objective="What is ML?", execute_node=executor)
# result.success                  → bool
# result.budget_consumed          → float
# result.audit_trail              → list[dict]

# Audit readout — same shape as old .get_audit_trail()
trail = governed_public.audit.to_list()

# NEW: hash-chain tamper-evidence (structural compliance upgrade for Ex 8.5)
chain_valid = governed_public.audit.verify_chain()

# Envelope introspection (new capability)
print(governed_public.envelope.financial.max_spend_usd)             # 5.0
print(governed_public.envelope.operational.allowed_actions)          # ["answer_question", ...]
print(governed_public.envelope.confidentiality_clearance.name)       # "PUBLIC"
```

### data_clearance string alias gotcha

`kaizen_agents._CLEARANCE_MAP` has a historical alias: `"internal"` → `ConfidentialityLevel.RESTRICTED`. The canonical 5-level hierarchy is:

```
PUBLIC < RESTRICTED < CONFIDENTIAL < SECRET < TOP_SECRET
```

The course teaches a 4-level hierarchy `public < internal < confidential < restricted` where "restricted" is the max. Per `decisions.md` § 4, keep the course's 4-level teaching with a footnote — `"restricted"` as string still maps to `ConfidentialityLevel.RESTRICTED`, and `"internal"` is a historical alias at the same level (technically equivalent to `"restricted"` in the course's mental model).

## Pact — GovernanceEngine Construction

### Old (broken — instance method doesn't exist)

```python
from kailash_pact import GovernanceEngine

engine = GovernanceEngine()
org = engine.compile_org("/path/to/org.yaml")
print(org.n_agents, org.n_delegations, org.n_departments)
```

### New (canonical)

```python
from pact import GovernanceEngine, load_org_yaml

# Two-step: load YAML, then construct engine
loaded = load_org_yaml("/path/to/org.yaml")
# loaded: LoadedOrg(org_definition, clearances, envelopes, bridges, ksps)

engine = GovernanceEngine(loaded.org_definition)

# Access the compiled org
compiled = engine.get_org()
# compiled: CompiledOrg(org_id, nodes)
# NO .n_agents / .n_delegations / .n_departments — compute from nodes dict
```

### CompiledOrgAdapter (Shard 3 — zero caller changes)

```python
# Add to shared/mlfp06/ex_7.py
from dataclasses import dataclass

@dataclass
class CompiledOrgAdapter:
    """Preserves .n_agents / .n_delegations / .n_departments for course callers."""
    _compiled: "CompiledOrg"

    @property
    def n_agents(self) -> int:
        return sum(1 for n in self._compiled.nodes.values() if n.kind == "agent")

    @property
    def n_delegations(self) -> int:
        return sum(1 for n in self._compiled.nodes.values() if n.kind == "task")

    @property
    def n_departments(self) -> int:
        return sum(1 for n in self._compiled.nodes.values() if n.kind == "department")

def compile_governance(yaml_path=None):
    """Preserved call signature. Returns (engine, adapter)."""
    if yaml_path is None:
        yaml_path = write_org_yaml()
    loaded = load_org_yaml(yaml_path)
    engine = GovernanceEngine(loaded.org_definition)
    return engine, CompiledOrgAdapter(_compiled=engine.get_org())
```

This preserves `org.n_agents / .n_delegations / .n_departments` for all 4 ex_7 technique files without requiring per-file edits.

## Pact — Access Check (verify_action)

### Old

```python
decision = engine.check_access(
    agent_id="model_trainer",
    resource="training_data",
    action="read",
)
# decision.allowed → bool
```

### New

```python
verdict = engine.verify_action(
    role_address="D1-R1-D2-R1-T2-R1",   # dash-delimited D/T/R address
    action="read",
    context={"cost": 0.10, "data_classification": "confidential"},
)
# verdict.allowed  → bool
# verdict.level    → "allowed" | "blocked" | "warn" | "audit"
# verdict.reason   → str (explanation)
```

### Fail-closed pattern (unknown role)

```python
# Passing a non-existent address triggers the 5-step algorithm's Step 1:
# "Resolve clearance — missing → DENY"
verdict = engine.verify_action(
    role_address="D99-R99-T99-R99",   # does not exist in compiled org
    action="any_action",
    context={},
)
assert not verdict.allowed
assert verdict.level == "blocked"
```

## Pact — Envelope Construction (RoleEnvelope)

### Old (`exam.py` only — `OperatingEnvelope` never existed in 0.8.1)

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

### New (5 canonical dimensions)

```python
from pact import (
    GovernanceEngine, RoleEnvelope, ConstraintEnvelopeConfig,
    FinancialConstraintConfig, OperationalConstraintConfig,
    TemporalConstraintConfig, DataAccessConstraintConfig,
    CommunicationConstraintConfig, ConfidentialityLevel,
)

envelope = ConstraintEnvelopeConfig(
    id="officer_envelope",
    description="Citizen service officer — bounded action surface",
    confidentiality_clearance=ConfidentialityLevel.RESTRICTED,
    financial=FinancialConstraintConfig(max_spend_usd=0.05),
    operational=OperationalConstraintConfig(
        allowed_actions=["classify", "respond"],
    ),
    temporal=TemporalConstraintConfig(),
    data_access=DataAccessConstraintConfig(),
    communication=CommunicationConstraintConfig(),
    max_delegation_depth=3,
)

role_env = RoleEnvelope(
    id="officer_role_envelope",
    defining_role_address="D1-R1",       # supervisor
    target_role_address="D1-R1-T1-R1",   # the officer
    envelope=envelope,
)

engine.set_role_envelope(role_env)
```

### Monotonic tightening (structural upgrade)

```python
# Old: manual integer comparison against CLEARANCE_LEVELS dict
# New: framework catches the violation structurally
try:
    RoleEnvelope.validate_tightening(parent_envelope, child_envelope)
except MonotonicTighteningError as exc:
    print(f"Privilege escalation caught: {exc}")
```

This is a strict pedagogical upgrade for `ex_7/02_envelopes.py`: the "privilege escalation caught at compile time" teaching beat becomes structurally real.

## Pact — Address Grammar

### Old (`exam.py` only)

```python
from kailash_pact import Address
addr = Address(domain="feedback", team="operations", role="officer")
```

### New

```python
from pact import Address

# Parse from dash-delimited grammar
addr = Address.parse("D1-R1-T1-R1")

# Grammar rule: every D or T must be immediately followed by exactly one R
#   "D1-R1"         → department 1 head
#   "D1-R1-T1-R1"   → dept 1 head owns task 1, delegated to responsible 1
#   "D1-R1-D2-R1"   → dept 1 head delegated dept 2 head authority (bridge)
```

## Pact — Access Check + Explanation (module-level fns)

### Old

```python
decision = governance.can_access(addr, action="classify")
explanation = governance.explain_access(addr, action="...")
```

### New

```python
from pact import can_access, explain_access

decision = can_access(
    role_address="D1-R1-T1-R1",
    knowledge_item=KnowledgeItem(...),
    posture=TrustPostureLevel.SUPERVISED,
    compiled_org=compiled,
    clearances=clearances_dict,
    ksps=ksps_list,
    bridges=bridges_list,
)

explanation = explain_access(
    role_address="D1-R1-T1-R1",
    knowledge_item=KnowledgeItem(...),
    posture=TrustPostureLevel.SUPERVISED,
    compiled_org=compiled,
    clearances=clearances_dict,
    ksps=ksps_list,
    bridges=bridges_list,
)
```

For the exam's simple "can officer classify?" pattern, prefer `engine.verify_action(role_address, action, context)` — it returns a verdict with `.allowed` + `.reason` in one call and doesn't require assembling all 7 `can_access()` arguments.

## Pact — ORG_YAML Schema

### Old schema (MLFP06 current — will not load)

```yaml
organization:
  name: "SG FinTech AI Division"
  jurisdiction: "Singapore"

departments:
  - name: "ML Engineering"
    head: "chief_ml_officer"
    agents:
      - id: "data_analyst"
        role: "analyst"
        clearance: "internal"

delegations:
  - delegator: "chief_ml_officer"
    task: "data_analysis"
    responsible: "data_analyst"
    envelope:
      max_budget_usd: 20.0
      allowed_tools: ["read_data", "summarise_data"]
      allowed_data_clearance: "internal"

operating_envelopes:
  global:
    max_llm_cost_per_request_usd: 0.50
    fail_mode: "closed"
```

### New schema (modern pact — loads via `pact.load_org_yaml`)

```yaml
org:
  org_id: "sg_fintech_ai"
  name: "SG FinTech AI Division"
  description: "Singapore FinTech AI Organisation — PACT Governance Definition"

departments:
  - id: "D1"
    name: "ML Engineering"
    head_role: "D1-R1"

teams:
  - id: "D1-T1"
    name: "Data Analysis"
    department: "D1"

agents:
  - id: "D1-R1-T1-R1"
    name: "data_analyst"
    role: "analyst"
    clearance: "internal"
    constraint_envelope: "data_analysis_envelope"

envelopes:
  - id: "data_analysis_envelope"
    confidentiality_clearance: "restricted"
    financial:
      max_spend_usd: 20.0
    operational:
      allowed_actions: ["read_data", "summarise_data", "generate_report"]
    max_delegation_depth: 3

workspaces:
  - id: "ml_eng_workspace"
    name: "ML Engineering Workspace"
    departments: ["D1"]
```

**Key mapping rules**:

- `organization` → `org`
- Top-level `departments` keeps name but gains explicit IDs
- Top-level `delegations` list disappears — each delegation becomes (one agent entry with a role address + one envelope entry referenced by ID)
- `operating_envelopes.global` fields distribute into the relevant `envelopes[]` entries
- `fail_mode: closed` is implicit (pact is fail-closed by construction — no need to declare)
- `pii_handling: mask` moves to `DataAccessConstraintConfig` at the envelope level

Full rewrites of both ORG_YAMLs (`shared/mlfp06/ex_7.py` — SG FinTech, 6 delegations; `shared/mlfp06/ex_8.py` — MLFP Capstone, 3 delegations) land in Shards 3 and 6 respectively.

## Nexus — Unchanged (verify at Shard 7 start)

### Canonical pattern (still valid per `rules/patterns.md`)

```python
from nexus import Nexus

async def serve_qa(question: str, role: str = "qa") -> dict:
    """Shared handler for API + CLI + MCP channels."""
    return {"answer": f"[stub] {question}", "role": role}

app = Nexus()
app.register(serve_qa)
# app.start() — not called in the exercise, just the registration is demonstrated
```

### New 3.4.7 patterns (available but not required)

Template 3.4.7 brought in `nexus-specialist` updates for middleware/mount/websocket. These are optional enhancements; M6 exercises do not need them. See `.claude/agents/frameworks/nexus-specialist.md` for the new patterns if a pedagogical enhancement is wanted in Shard 7.

## kaizen_agents — Delegate (unchanged, kept for reference)

```python
import os
from kaizen_agents import Delegate

delegate = Delegate(
    model=os.environ["OPENAI_PROD_MODEL"],
    budget_usd=10.0,
    tools="all",            # or a list of tool names
)

# Delegate.run(prompt) is an async generator of events
async for event in delegate.run("Analyze this data"):
    print(event)

# Delegate.run_sync(prompt) returns a string
result_str = delegate.run_sync("Analyze this data")
```

Used by `shared/mlfp06/ex_1.py` and `shared/mlfp06/ex_4.py`. No migration needed for those helpers — `kaizen_agents.Delegate` is stable.

## Runtime Execution (kailash Core SDK — unchanged)

```python
from kailash.workflow.builder import WorkflowBuilder
from kailash.runtime.local import LocalRuntime

workflow = WorkflowBuilder()
workflow.add_node("PythonCodeNode", "hello", {"code": "print('hi')"})

runtime = LocalRuntime()
results, run_id = runtime.execute(workflow.build())   # MUST call .build()
```

Only relevant for the Nexus fallback path in Shard 7 (if `app.register(serve_qa)` turns out to require a workflow wrapper).
