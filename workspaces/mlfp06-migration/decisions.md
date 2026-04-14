# MLFP06 Migration — Pedagogical Decisions

**All decisions below were approved by Jack Hong on 2026-04-14 with the directive "all defaults, go".**

These are the 6 pedagogical pivots the migration requires. Each has a rationale for the recommendation and the scope of files it affects. If Jack changes his mind, these are the only decisions that need to be re-litigated — everything else in the migration is mechanical.

---

## Decision 1 — Task 4 (ex_7/04) Reframe

**APPROVED 2026-04-14.**

**Problem**: MLFP06 Task 4 in `ex_7/04_runtime_audit.py` currently asserts that `engine.check_access(agent_id, resource, action, payload=prompt_text)` returns `allowed=False` for toxic prompts — "PACT governance blocks adversarial inputs". The old code had a `try/except TypeError` fallback admitting the `payload` kwarg was already non-standard.

**Reality in pact 0.8.1**: The governance engine does not classify content. Content classification is a separate control (moderation API, classifier head, or `pact.KnowledgeFilter` hook). The 5-step access algorithm compares a `KnowledgeItem`'s classification to a role's clearance — it does not analyze text.

**Resolution**: Reframe Task 4 as "**blast-radius containment**". Governance does not stop the injection from reaching the model; it caps the damage a successful injection can do via envelope constraints:

- Budget cap (looped injection cannot drain more than $5 on public tier)
- Tool allowlist (refusal when prompt convinces model to invoke `train_model`)
- Clearance gate (refusal when prompt convinces model to read restricted data)

**Why this is strictly better teaching**:

1. It is honest about what PACT actually does (aligns with `pact-access-enforcement.md`)
2. It gives students a more nuanced view: "governance is a runtime limit, not a content filter"
3. It matches the canonical pact 5-step algorithm documentation
4. "Prompt injection is contained, not prevented" is a better security heuristic for production

**Scope**: `modules/mlfp06/solutions/ex_7/04_runtime_audit.py` — Task 4 narrative block (lines ~207–260) + the reflection block. The RealToxicityPrompts dataset load and the iteration loop remain; only the framing of what the loop asserts changes. Applied in Shard 5 using the reference implementation in `reference/ex_7_04_runtime_audit.py`.

---

## Decision 2 — Extract `build_capstone_stack()` Helper

**APPROVED 2026-04-14.**

**Problem**: 4 ex_8 technique files (`02_governance_pipeline`, `03_multichannel_serving`, `04_drift_monitoring`, `05_compliance_audit`) each start with a near-identical 40-LOC 3-tier construction block:

```python
governance_engine = GovernanceEngine(...)
# ...
base_qa = CapstoneQAAgent(CapstoneQAConfig())
governed_qa = GovernedSupervisor(model=..., budget_usd=1.0, tools=[...], data_clearance="internal")
governed_admin = GovernedSupervisor(model=..., budget_usd=10.0, tools=[...], data_clearance="confidential")
governed_audit = GovernedSupervisor(model=..., budget_usd=50.0, tools=[...], data_clearance="restricted")
agents_by_role = {"qa": governed_qa, "admin": governed_admin, "audit": governed_audit}
```

4 files × 40 LOC = 160 LOC of duplication. After migration, the pattern moves from `PactGovernedAgent(...)` to `GovernedSupervisor(...)` but the duplication stays.

**Resolution**: Extract `build_capstone_stack(engine)` into `shared/mlfp06/ex_8.py`:

```python
def build_capstone_stack(engine: GovernanceEngine) -> dict[str, GovernedSupervisor]:
    """Shared 3-tier stack builder for all ex_8 technique files."""
    return {
        "qa": GovernedSupervisor(model=MODEL, budget_usd=1.0,
                                   tools=["generate_answer", "search_context"],
                                   data_clearance="internal"),
        "admin": GovernedSupervisor(model=MODEL, budget_usd=10.0,
                                      tools=["generate_answer", "search_context", "update_model",
                                             "view_metrics", "monitor_drift"],
                                      data_clearance="confidential"),
        "audit": GovernedSupervisor(model=MODEL, budget_usd=50.0,
                                      tools=["generate_answer", "search_context", "view_metrics",
                                             "access_audit_log", "generate_report"],
                                      data_clearance="restricted"),
    }
```

Each ex_8 technique file calls `agents_by_role = build_capstone_stack(governance_engine)` at the top and runs its per-file narrative unchanged.

**Why this is OK under R10** (`rules/exercise-standards.md` § "R10 Directory Structure"): "Each file is independently runnable" is satisfied because each file imports the helper and calls it locally — no hidden orchestration, no cross-file state. The student reads the file top-to-bottom and sees exactly what the 3 tiers are via the import + call site.

**Saves**: ~120 LOC of duplication across the 4 technique files. Each file becomes ~30 LOC shorter.

**Scope**: `shared/mlfp06/ex_8.py` gains one function; 4 technique files each lose ~30 LOC and gain one import line. Applied in Shard 6 (helper extraction) and Shard 7 (technique file migration).

---

## Decision 3 — exam.py Task 4a Address Grammar Shift

**APPROVED 2026-04-14.**

**Problem**: exam.py Task 4a currently teaches `Address(domain="feedback", team="operations", role="officer")` — a kwarg-based constructor. This constructor does not exist in pact 0.8.1.

**Modern pact**: `pact.Address.parse("D1-R1-T1-R1")` — dash-delimited positional grammar. Rule: every `D` or `T` must be immediately followed by exactly one `R`.

**Resolution**: The exam question shape changes. Students now write dash-delimited D/T/R addresses instead of kwargs. The learning outcome (students demonstrate understanding of the D/T/R accountability grammar) is preserved — arguably strengthened, because the dash grammar is closer to how PACT actually models accountability.

**Old exam question**:

> Q: Write the `Address` for a citizen service officer in the feedback department's operations team.
> A: `Address(domain="feedback", team="operations", role="citizen_service_officer")`

**New exam question**:

> Q: Write the `Address.parse()` call for a citizen service officer under the feedback department's operations team, using the D/T/R grammar (D1 = feedback dept, T1 = operations team, R1 = officer role).
> A: `Address.parse("D1-R1-T1-R1")`

**Why this is OK under `rules/domain-integrity.md`**: The question still tests context-specific application (student must map the business scenario to the grammar) and cannot be pattern-matched from generic examples. The grammar is strict and unambiguous, so grading is deterministic.

**Scope**: `modules/mlfp06/assessment/exam.py` lines 1107–1200 (Task 4a PACT block). Applied in Shard 8.

---

## Decision 4 — Four-Level Clearance Hierarchy Preserved With Footnote

**APPROVED 2026-04-14.**

**Problem**: MLFP06 teaches `public < internal < confidential < restricted` (4 levels, `restricted` is the max). Canonical pact is 5 levels: `PUBLIC < RESTRICTED < CONFIDENTIAL < SECRET < TOP_SECRET`.

`kaizen_agents._CLEARANCE_MAP` has a gotcha: `"internal"` → `ConfidentialityLevel.RESTRICTED`, so the course's `"internal"` and `"restricted"` strings collide at one canonical level. The course's mental model works but is technically less granular than canonical.

**Resolution**: Keep the course's 4-level teaching. Add a one-paragraph footnote sidebar in `ex_7/02_envelopes.py` explaining:

> **Sidebar — Canonical PACT clearance hierarchy**
>
> MLFP06 teaches a 4-level clearance hierarchy (`public < internal < confidential < restricted`) because it is easier to hold in your head. Canonical pact uses 5 levels: `PUBLIC < RESTRICTED < CONFIDENTIAL < SECRET < TOP_SECRET`. The mapping at the string interface:
>
> - `"public"` → `PUBLIC`
> - `"internal"` → `RESTRICTED` (historical alias)
> - `"confidential"` → `CONFIDENTIAL`
> - `"restricted"` → `RESTRICTED` (see historical alias above — collides with `"internal"` at one canonical level)
>
> This matters when you cross the boundary from course code into production pact systems — you may see `SECRET` and `TOP_SECRET` in real governance configs for high-sensitivity data (e.g., trading algorithms, patient records).

**Why this is OK**: The course's learning outcome ("clearance hierarchy prevents read-up attacks") is achievable at 4 levels. Rewriting to 5 levels would require renaming the course's mental model throughout M6 for marginal pedagogical gain. The footnote gives students the bridge to canonical terminology when they need it.

**Scope**: `modules/mlfp06/solutions/ex_7/02_envelopes.py` (and local mirror) — add the sidebar block. Applied in Shard 4.

---

## Decision 5 — Nexus Registration Verified Pre-Shard-7

**APPROVED 2026-04-14** (with deferred verification step).

**Problem**: `modules/mlfp06/solutions/ex_8/03_multichannel_serving.py` uses `app = Nexus(); app.register(serve_qa)` where `serve_qa` is a bare `async def`. The pattern is documented in `rules/patterns.md` § Nexus, but template 3.4.7 brought in new Nexus patterns (middleware/mount/websocket) that may have changed `register()` semantics.

**Resolution**: At the start of Shard 7, run a one-line verification:

```python
import inspect
from nexus import Nexus
print(inspect.signature(Nexus.register))
```

If the signature accepts a callable (bare async function), the current pattern stays and no narrative change is needed. If it requires a `WorkflowBuilder`-constructed workflow, wrap `serve_qa` in a single-node workflow:

```python
from kailash.workflow.builder import WorkflowBuilder
from kailash.runtime.local import LocalRuntime

workflow = WorkflowBuilder()
workflow.add_node("PythonCodeNode", "serve_qa", {
    "code": "import asyncio; result = asyncio.run(serve_qa(question, role))",
})
app = Nexus()
app.register(workflow.build())
```

This fallback is the pattern the course already teaches in earlier modules (`WorkflowBuilder` + `runtime.execute(workflow.build())` is the Core SDK runtime pattern) — no pedagogical regression.

**Why this is OK**: The verification is one line of code. The fallback is a known pattern. Either way the exercise runs.

**Scope**: `modules/mlfp06/solutions/ex_8/03_multichannel_serving.py` — potentially the top of Task 2 (where the `app.register(serve_qa)` call lives). Applied in Shard 7.

---

## Decision 6 — `LLMCostTracker` Replaced With `budget_limit_usd` Narrative

**APPROVED 2026-04-14.**

**Problem**: `from kaizen_agents import LLMCostTracker` raises `ImportError` in the current pin. The class was removed and its responsibility moved to `BaseAgentConfig.budget_limit_usd`. 5 call sites fail at module load:

- `modules/mlfp06/assessment/exam.py:1011` (code call)
- `modules/mlfp06/lessons/05/textbook.html:504,609` (teaching text)
- `modules/mlfp06/lessons/06/textbook.html:321,688` (teaching text)

**Resolution**: Replace every `LLMCostTracker()` call with the `budget_limit_usd` pattern, and update the teaching text to explain that budget is now a per-agent config field (not a standalone tracker).

**Replacement in code** (`exam.py:1011`):

```python
# OLD
from kaizen_agents import LLMCostTracker
tracker = LLMCostTracker()
# ... code that reads tracker.total_cost ...

# NEW
# Budget is per-agent via BaseAgentConfig — no separate tracker needed.
# Each agent's consumption is tracked internally; read via agent.config.budget_limit_usd
# and check remaining via the supervisor's audit trail.
```

**Replacement in teaching text** (textbook HTMLs): One-paragraph update explaining the API shift:

> Earlier versions of kaizen shipped a standalone `LLMCostTracker` class for monitoring cumulative LLM spend across agents. In kaizen 2.7.3, cost tracking moved into `BaseAgentConfig.budget_limit_usd` — every agent carries its own budget cap as a config field, and the supervisor's audit trail records per-call consumption. This is a strictly better model because the budget is bound to the agent's envelope (not a separate global tracker) and the hash-chained audit trail makes spend verifiable in compliance audits.

**Why this is OK**: The teaching point (agents have budget caps, cost tracking is auditable) is preserved. The API shift is a net improvement — bound budgets are more local than global trackers.

**Scope**: 5 spots across 3 files. Applied in Shards 8 (exam.py) and 9 (textbook HTMLs).

---

## Re-Approval Process

If Jack wants to revise any of these decisions in a future session:

1. Update this file — change `APPROVED 2026-04-14` to `SUPERSEDED 2026-04-XX — see new direction below`.
2. Add the new direction as a new paragraph at the bottom of the affected decision.
3. Update the affected shard in `shard-plan.md` § invariants.
4. The next session picks up the revised plan from the updated `shard-plan.md`.

Do NOT re-litigate decisions that have already been applied in earlier shards — reverting a migrated shard is strictly more expensive than making the fix forward in a later shard.
