# MLFP06 Migration — 9-Shard Execution Plan

**Per**: `.claude/rules/autonomous-execution.md` § Per-Session Capacity Budget. Each shard stays under 500 LOC load-bearing logic, ≤5–10 invariants, ≤3–4 call-graph hops, describable in 3 sentences.

**Execution order**: Shard 0 MUST run first. After that: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9, OR Shards 1+2 in parallel worktrees (kaizen-only, no cross-dependency), then 3→4→5, then 6→7, then 8, then 9.

## Shard 0 — Pre-Flight

**Goal**: Clean slate + infrastructure for the migration.

**Files**:

- REVERT: `modules/mlfp06/solutions/ex_7/04_runtime_audit.py`, `ex_8/02_governance_pipeline.py`, `ex_8/03_multichannel_serving.py`, `ex_8/04_drift_monitoring.py`, `ex_8/05_compliance_audit.py` (5 files with half-applied import renames — restore via `git checkout`)
- CREATE: `pyrightconfig.json` at repo root
- VERIFY: `uv run python scripts/py_to_notebook.py --help` returns cleanly

**Invariants** (4):

1. The 5 half-edited files are restored to their pre-session state (`git status` shows them as unmodified).
2. `pyrightconfig.json` exists at repo root with `reportAssignmentType: none` scoped to `modules/mlfp06/**` and `shared/mlfp06/**`.
3. `pyproject.toml` pin bumps REMAIN (kailash 2.8.5, kailash-dataflow 2.0.7, kailash-mcp 0.2.3). Do NOT revert.
4. `scripts/py_to_notebook.py` runs without error and accepts the `--module mlfp06` argument shape.

**Description**: Restore the 5 partial solutions files via `git checkout`, create the pyrightconfig for M6 to silence Signature field cosmetic errors, and verify the Colab regeneration script works. Do not touch any M6 content.

**LOC estimate**: ~30 LOC (pyrightconfig.json only — everything else is git reverts).

**Feedback loop**:

```bash
# Verify reverts
git status modules/mlfp06/solutions/ex_7/04_runtime_audit.py \
           modules/mlfp06/solutions/ex_8/*.py
# Should show no modifications

# Verify pyrightconfig
cat pyrightconfig.json

# Verify notebook script
uv run python scripts/py_to_notebook.py --help
```

---

## Shard 1 — ex_5 Kaizen Migration (Independent, Smallest)

**Goal**: Migrate the 2 structured-agent exercises from `ex_5` to the canonical `BaseAgent` pattern. These files are self-contained — no PACT, no Nexus, no shared-helper dependency.

**Files**:

- `modules/mlfp06/solutions/ex_5/03_structured_agent.py`
- `modules/mlfp06/solutions/ex_5/04_critic_agent.py`
- `modules/mlfp06/local/ex_5/03_structured_agent.py`
- `modules/mlfp06/local/ex_5/04_critic_agent.py`
- REGENERATE: `modules/mlfp06/colab/ex_5/03_structured_agent.ipynb`, `04_critic_agent.ipynb`

**Invariants** (5):

1. Imports use canonical paths: `from kaizen.core.base_agent import BaseAgent` and `from kaizen import Signature, InputField, OutputField`.
2. Every `class XxxAgent(BaseAgent)` uses the new pattern: `def __init__(self, config: XxxConfig): super().__init__(config=config, signature=XxxSignature())`.
3. Every agent config is a `@dataclass` with `model: str` and `budget_limit_usd: float` fields.
4. Every `await agent.run(...)` becomes `await agent.run_async(...)`, and every `result.field` becomes `result["field"]`.
5. Each of the 4 .py files passes `uv run python -m py_compile`, and each maintains its checkpoint assertions unchanged.

**Description**: Rewrite the structured-agent and critic-agent exercises in ex_5 to use the canonical kaizen 2.7.3 BaseAgent pattern (dataclass config + instance signature in `super().__init__`). Update call sites to `run_async` and dict-indexed result access. Preserve every teaching block, checkpoint, and reflection verbatim.

**LOC estimate**: ~400 LOC load-bearing edits across 4 files.

**Feedback loop**:

```bash
# Byte-compile both files
uv run python -m py_compile modules/mlfp06/solutions/ex_5/03_structured_agent.py
uv run python -m py_compile modules/mlfp06/solutions/ex_5/04_critic_agent.py
uv run python -m py_compile modules/mlfp06/local/ex_5/03_structured_agent.py
uv run python -m py_compile modules/mlfp06/local/ex_5/04_critic_agent.py

# Import smoke test — construct one agent of each type
uv run python -c "
import sys; sys.path.insert(0, 'modules/mlfp06/solutions/ex_5')
import importlib
for m in ['03_structured_agent', '04_critic_agent']:
    mod = importlib.import_module(m.replace('-', '_'))
    print(f'{m}: import OK')
"

# Regenerate Colab notebooks
uv run python scripts/py_to_notebook.py modules/mlfp06/local/ex_5/
```

---

## Shard 2 — ex_6 Multi-Agent Migration

**Goal**: Migrate `shared/mlfp06/ex_6.py` + all 5 ex_6 technique files to canonical kaizen BaseAgent pattern. Still kaizen-only — no PACT.

**Files**:

- `shared/mlfp06/ex_6.py` (foundation — must land first)
- `modules/mlfp06/solutions/ex_6/{01_supervisor_worker, 02_sequential_pipeline, 03_parallel_router, 04_mcp_server, 05_memory_and_security}.py` (5 files)
- `modules/mlfp06/local/ex_6/{01..05}.py` (5 mirrors)
- REGENERATE: `modules/mlfp06/colab/ex_6/*.ipynb` (5 notebooks)

**Invariants** (6):

1. `shared/mlfp06/ex_6.py` uses canonical imports and migrates all specialist `BaseAgent` subclasses (Researcher, Reviewer, Writer, etc.) to the dataclass-config + instance-signature pattern.
2. The SQuAD 2.0 corpus loading, output directory setup, and shared tool definitions are preserved byte-for-byte.
3. All 5 technique files import from `shared.mlfp06.ex_6` with the same public surface (function/class names unchanged).
4. All `await agent.run(...)` call sites become `await agent.run_async(...)` + dict result access.
5. `kaizen_agents.Delegate` (used in `01_supervisor_worker`) remains unchanged — its API is stable.
6. Every file byte-compiles cleanly and preserves its checkpoint assertions.

**Description**: Rewrite the ex_6 shared helpers and 5 technique files to the canonical kaizen pattern. Preserve the public surface of `shared/mlfp06/ex_6.py` so call sites don't change. Regenerate the 5 Colab notebooks at the end.

**LOC estimate**: ~500 LOC load-bearing.

**Feedback loop**:

```bash
uv run python -m py_compile shared/mlfp06/ex_6.py
uv run python -c "from shared.mlfp06 import ex_6; print('shared.mlfp06.ex_6: OK')"

for f in modules/mlfp06/solutions/ex_6/*.py modules/mlfp06/local/ex_6/*.py; do
  uv run python -m py_compile "$f" || exit 1
done

uv run python scripts/py_to_notebook.py modules/mlfp06/local/ex_6/
```

---

## Shard 3 — ex_7 Shared Foundation

**Goal**: Migrate `shared/mlfp06/ex_7.py` + `ex_7/01_org_compile.py` to modern pact. This is the first PACT shard and lays the foundation for Shards 4 and 5.

**Files**:

- `shared/mlfp06/ex_7.py` (foundation — rewrite ORG_YAML, compile_governance, add CompiledOrgAdapter + make_fake_executor, rename BudgetTracker)
- `modules/mlfp06/solutions/ex_7/01_org_compile.py`
- `modules/mlfp06/local/ex_7/01_org_compile.py`
- REGENERATE: `modules/mlfp06/colab/ex_7/01_org_compile.ipynb`

**Invariants** (7):

1. `ORG_YAML` is rewritten to the modern schema (`org:`, `departments[]`, `teams[]`, `agents[]`, `envelopes[]`, `workspaces[]`) and loads cleanly via `pact.load_org_yaml()`.
2. `compile_governance()` returns `(engine, CompiledOrgAdapter)` where the adapter exposes `.n_agents`, `.n_delegations`, `.n_departments` as computed properties (preserving caller contract).
3. `BudgetTracker` renamed to `TeachingBudgetTracker` — avoids collision with internal `pact.BudgetTracker`.
4. `make_fake_executor()` factory added to `shared/mlfp06/ex_7.py`, returns an async callable `(spec, inputs) -> {"result": str, "cost": float, "prompt_tokens": int, "completion_tokens": int}` suitable for `GovernedSupervisor.run(execute_node=...)`.
5. `load_adversarial_prompts()`, `default_model_name()`, `CLEARANCE_LEVELS` constants — unchanged (pure Python, no framework dependency).
6. `solutions/ex_7/01_org_compile.py` still shows the same SG FinTech output (same agent count, same department count, same delegation count) and passes its 3 checkpoint assertions.
7. The SG FinTech narrative in teaching comments is preserved verbatim.

**Description**: Rewrite the ex_7 shared helpers to modern pact's `load_org_yaml` / `GovernanceEngine` construction, add `CompiledOrgAdapter` to preserve `.n_agents / .n_delegations / .n_departments` for callers, rename `BudgetTracker` → `TeachingBudgetTracker`, and add `make_fake_executor()` for offline GovernedSupervisor runs. Migrate the first technique file to use the new helpers. The SG FinTech teaching narrative is preserved.

**LOC estimate**: ~450 LOC load-bearing (YAML rewrite + adapter class + 1 file migration + local mirror).

**Feedback loop**:

```bash
# Verify shared helper
uv run python -m py_compile shared/mlfp06/ex_7.py
uv run python -c "
from shared.mlfp06.ex_7 import compile_governance, make_fake_executor, TeachingBudgetTracker
engine, org = compile_governance()
print(f'agents={org.n_agents} delegations={org.n_delegations} departments={org.n_departments}')
"

# Verify ex_7/01
uv run python -m py_compile modules/mlfp06/solutions/ex_7/01_org_compile.py
uv run python -m py_compile modules/mlfp06/local/ex_7/01_org_compile.py

# Regenerate Colab
uv run python scripts/py_to_notebook.py modules/mlfp06/local/ex_7/01_org_compile.py
```

---

## Shard 4 — ex_7 Envelopes + Budget

**Goal**: Migrate `02_envelopes.py` and `03_budget_access.py` to modern pact envelope/access APIs.

**Files**:

- `modules/mlfp06/solutions/ex_7/02_envelopes.py` + local mirror
- `modules/mlfp06/solutions/ex_7/03_budget_access.py` + local mirror
- REGENERATE: 2 Colab notebooks

**Invariants** (6):

1. `ex_7/02_envelopes.py` constructs `ConstraintEnvelopeConfig` with all 5 dimensions (Financial, Operational, Temporal, Data Access, Communication) — the teaching beat "5 envelope dimensions" becomes structurally real, not just narrated.
2. The privilege-escalation demo uses `RoleEnvelope.validate_tightening()` to catch the violation — upgrade from manual integer comparison.
3. `ex_7/03_budget_access.py` imports `TeachingBudgetTracker` (renamed) from `shared.mlfp06.ex_7` and passes its overspend-denial checkpoint unchanged.
4. The 10-case access-control table uses `engine.verify_action(role_address, action, context)` instead of `check_access(agent_id, resource, action)`.
5. All existing visualisations (radar chart, clearance lattice, budget bar chart) are preserved byte-for-byte — they used mock data already.
6. Neither file imports `PactGovernedAgent`, `kailash_pact`, or `kaizen.core.BaseAgent`.

**Description**: Migrate ex_7/02 (envelopes) and ex_7/03 (budget access) to modern pact — envelopes via `ConstraintEnvelopeConfig` + `RoleEnvelope.validate_tightening`, access checks via `engine.verify_action` with dash-delimited role addresses. Use `TeachingBudgetTracker` from the updated shared helper. Preserve all checkpoints, visualisations, and narrative.

**LOC estimate**: ~400 LOC load-bearing.

**Feedback loop**:

```bash
uv run python -m py_compile modules/mlfp06/solutions/ex_7/02_envelopes.py
uv run python -m py_compile modules/mlfp06/solutions/ex_7/03_budget_access.py
uv run python -m py_compile modules/mlfp06/local/ex_7/02_envelopes.py
uv run python -m py_compile modules/mlfp06/local/ex_7/03_budget_access.py

uv run python scripts/py_to_notebook.py modules/mlfp06/local/ex_7/02_envelopes.py
uv run python scripts/py_to_notebook.py modules/mlfp06/local/ex_7/03_budget_access.py
```

---

## Shard 5 — ex_7/04 Runtime Wrapper (Reference Implementation)

**Goal**: Migrate `04_runtime_audit.py` — the single largest single-file rewrite. Template is `reference/ex_7_04_runtime_audit.py` in this workspace.

**Files**:

- `modules/mlfp06/solutions/ex_7/04_runtime_audit.py` + local mirror
- REGENERATE: 1 Colab notebook

**Invariants** (5):

1. Three `GovernedSupervisor` tiers wired with matching budget/tools/data_clearance (public/internal/restricted). Budgets: $5/$50/$200. Tools: answer_question+search_faq / +read_data+train_model / +audit_model+access_audit_log.
2. Task 3 fail-closed verified via `engine.verify_action("D99-R99-T99-R99", "answer_question", {"cost": 0.10})` — asserts `not verdict.allowed` and `verdict.level == "blocked"`.
3. Task 4 reframed to "blast-radius containment" per `decisions.md` § 1 — no longer claims governance blocks content; claims governance caps what a successful injection can do.
4. Audit readout uses `supervisor.audit.to_list()` + `.verify_chain()` for tamper-evidence. Regulatory-mapping DataFrame unchanged (6 rows).
5. All 5 task checkpoints pass; the visualisation block (audit timeline + enforcement outcome pie chart) is preserved byte-for-byte.

**Description**: Apply the pact-specialist's reference implementation from `reference/ex_7_04_runtime_audit.py` verbatim. Construct three `GovernedSupervisor` tiers, verify fail-closed via `engine.verify_action`, reframe Task 4 to blast-radius containment, and use the hash-chained `supervisor.audit` readout. Preserve all checkpoints, the regulatory mapping, the visualisation, and the PDPA reflection.

**LOC estimate**: ~450 LOC (the reference impl is ~450 LOC — near-verbatim copy with local-mirror stripping).

**Feedback loop**:

```bash
uv run python -m py_compile modules/mlfp06/solutions/ex_7/04_runtime_audit.py
uv run python -m py_compile modules/mlfp06/local/ex_7/04_runtime_audit.py

# Offline smoke test — runs the reference flow with make_fake_executor
uv run python modules/mlfp06/solutions/ex_7/04_runtime_audit.py

uv run python scripts/py_to_notebook.py modules/mlfp06/local/ex_7/04_runtime_audit.py
```

---

## Shard 6 — ex_8 Shared Helpers

**Goal**: Migrate `shared/mlfp06/ex_8.py` — the foundation for all 4 capstone technique files. Highest-risk shard because 4 files depend on its public contract.

**Files**:

- `shared/mlfp06/ex_8.py` (standalone — no technique files in this shard)

**Invariants** (7):

1. `ORG_YAML` rewritten to modern pact schema (MLFP Capstone org — 3 agents: qa/admin/audit).
2. `CapstoneQASignature` unchanged — Signature is still valid in 2.7.3.
3. `CapstoneQAAgent` rewritten to canonical pattern: `@dataclass CapstoneQAConfig` with `model` + `budget_limit_usd`, `class CapstoneQAAgent(BaseAgent): def __init__(self, config): super().__init__(config=config, signature=CapstoneQASignature())`.
4. `handle_qa(question, role, agents_by_role)` accepts `agents_by_role: dict[str, GovernedSupervisor]` (changed from `PactGovernedAgent`) and preserves its return dict shape: `{answer, confidence, sources, reasoning_steps, latency_ms, governed, role}`. The call becomes `await gs.run(objective=question, execute_node=<shared_executor>)` where the shared executor wraps a `CapstoneQAAgent` instance.
5. `build_capstone_stack(engine)` helper added per `decisions.md` § 2 — returns `agents_by_role: dict[str, GovernedSupervisor]` + the 3 tier metadata. Deduplicates the 4× 40-LOC tier-rebuild block in Shards 7's technique files.
6. `SimpleJWTAuth`, `RateLimiter`, `run_async`, `MODEL`, `load_mmlu_eval`, `write_org_yaml` — unchanged.
7. Module imports without errors.

**Description**: Rewrite `shared/mlfp06/ex_8.py` to use `GovernedSupervisor` + canonical `BaseAgent` pattern. Preserve the `handle_qa()` return dict contract so the 4 downstream technique files are unaffected at call sites. Add `build_capstone_stack()` helper to dedupe the 4-file boilerplate.

**LOC estimate**: ~350 LOC load-bearing.

**Feedback loop**:

```bash
uv run python -m py_compile shared/mlfp06/ex_8.py
uv run python -c "
from shared.mlfp06.ex_8 import (
    CapstoneQAAgent, CapstoneQAConfig, CapstoneQASignature,
    handle_qa, build_capstone_stack, write_org_yaml,
    MODEL, load_mmlu_eval, SimpleJWTAuth, RateLimiter, run_async,
)
print('shared.mlfp06.ex_8: all symbols importable')
# Construct one CapstoneQAAgent
agent = CapstoneQAAgent(CapstoneQAConfig())
print(f'signature={type(agent.signature).__name__}')
"
```

---

## Shard 7 — ex_8 Capstone Technique Files

**Goal**: Migrate the 4 capstone technique files. Each rebuilds the 3-tier governed stack (or delegates to `build_capstone_stack()`), then runs its existing narrative.

**Files**:

- `modules/mlfp06/solutions/ex_8/{02_governance_pipeline, 03_multichannel_serving, 04_drift_monitoring, 05_compliance_audit}.py` (4 files)
- `modules/mlfp06/local/ex_8/{02..05}.py` (4 mirrors)
- REGENERATE: 4 Colab notebooks

**Pre-shard verification**: Run `inspect.signature(nexus.Nexus.register)` to confirm `app.register(bare_async_fn)` still works per `decisions.md` § 5. If broken, wrap `serve_qa` in a single-node `WorkflowBuilder` (pattern taught in earlier modules — not a pedagogical regression).

**Invariants** (8):

1. All 4 files use `build_capstone_stack(engine)` from the updated shared helper (or the equivalent inline 3-tier construction if the helper wasn't extracted). No file imports `PactGovernedAgent`, `kailash_pact`, or `kaizen.core.BaseAgent`.
2. `02_governance_pipeline.py` visualises the envelope hierarchy (unchanged) and passes the 3-tier checkpoint.
3. `03_multichannel_serving.py` registers `serve_qa` with `Nexus` (verified pre-shard). The middleware demo (CORS/rate limit/JWT/log/handler/governance) is preserved.
4. `04_drift_monitoring.py` wires `kailash_ml.DriftMonitor` + runs the 5-test harness + produces the PSI timeline chart.
5. `05_compliance_audit.py` extracts `supervisor.audit.to_list()` for all 3 tiers + calls `verify_chain()` for tamper-evidence, builds the compliance report, maps to 6 regulations, produces the traffic-light chart.
6. The 3 teaching scenarios (MAS TRM wealth advisory, Shopee 11.11 fraud, MAS Section 27 production order) are preserved verbatim.
7. All checkpoints pass.
8. The dict return shape of `handle_qa()` is unchanged at every call site — `result["answer"]`, `result["confidence"]`, etc.

**Description**: Migrate the 4 ex_8 technique files to use `GovernedSupervisor` via `build_capstone_stack()`. Each file rebuilds the 3-tier stack (or delegates to the helper), then runs its existing teaching narrative unchanged. All checkpoints, visualisations, and business scenarios are preserved.

**LOC estimate**: ~500 LOC load-bearing (mostly boilerplate deduplication via `build_capstone_stack`).

**Feedback loop**:

```bash
# Pre-shard Nexus verification
uv run python -c "
import inspect
from nexus import Nexus
print('Nexus.register signature:', inspect.signature(Nexus.register))
"

# Byte-compile all 8 files
for f in modules/mlfp06/solutions/ex_8/0{2,3,4,5}_*.py modules/mlfp06/local/ex_8/0{2,3,4,5}_*.py; do
  uv run python -m py_compile "$f" || exit 1
done

# Regenerate Colab
uv run python scripts/py_to_notebook.py modules/mlfp06/local/ex_8/
```

---

## Shard 8 — exam.py PACT Rewrite

**Goal**: Rewrite the PACT block of the 1,531-LOC exam (lines 1107–1200) + 4 isolated import/removal fixes. Total load-bearing changes ~100 LOC in a 1,531-LOC file.

**Files**:

- `modules/mlfp06/assessment/exam.py` (single file, multiple patches)

**Invariants** (6):

1. Line 104: `from kaizen.core import Signature, InputField, OutputField` → `from kaizen import Signature, InputField, OutputField`.
2. Line 751: `from kaizen.core import BaseAgent` → `from kaizen.core.base_agent import BaseAgent`. Every `class XxxAgent(BaseAgent)` in exam.py also gets the dataclass-config + instance-signature pattern.
3. Line 1011: `from kaizen_agents import LLMCostTracker` removed. The `LLMCostTracker()` call site is replaced with `BaseAgentConfig.budget_limit_usd` narrative (~20 LOC of teaching text update).
4. Lines 1107–1200: Task 4a PACT block rewritten to use `pact.Address.parse("D1-R1-T1-R1")`, `engine.verify_action()`, `engine.set_role_envelope(RoleEnvelope(...))`. Per `decisions.md` § 3, the exam question shifts from `Address(domain=, team=, role=)` kwargs to dash-delimited parse — difficulty preserved, shape changes.
5. Line 1334: `from kailash_nexus import Nexus` → `from nexus import Nexus`.
6. Task 4b (budget cascade), Task 4c (audit trail — pure Python list), Task 4d (Nexus deployment) are unchanged. The other ~1,400 lines of the exam (prompt engineering, RAG, multi-agent) are unchanged.

**Description**: Rewrite the PACT block (lines 1107–1200) of the 1,531-LOC exam to use modern pact — `Address.parse`, `verify_action`, `set_role_envelope`. Apply the 4 isolated import/removal fixes (Signature import, BaseAgent import, LLMCostTracker removal, Nexus import). Preserve the budget-cascade and audit-trail teaching props (not PACT-specific). The exam question shape changes: students now write dash-delimited D/T/R addresses instead of kwargs.

**LOC estimate**: ~100 LOC load-bearing changes in a 1,531-LOC file.

**Feedback loop**:

```bash
uv run python -m py_compile modules/mlfp06/assessment/exam.py

# Smoke test: import should not fail
uv run python -c "
import sys; sys.path.insert(0, 'modules/mlfp06/assessment')
# Note: exam.py is not a module — it's a runnable script with module-level side effects
# Use py_compile only; full runtime execution is out of scope for shard verification
print('exam.py byte-compiles cleanly')
"
```

---

## Shard 9 — Textbook HTML + Final Verification

**Goal**: Fix stale imports in the textbook HTMLs, regenerate any missed Colab notebooks, and run the full M6 verification sweep.

**Files**:

- `modules/mlfp06/lessons/05/textbook.html` (fix `LLMCostTracker` at lines 504, 609)
- `modules/mlfp06/lessons/06/textbook.html` (fix `kaizen.protocol`, `kaizen.memory` three-tier, top-level `ReActAgent` / `BaseAgent` / `LLMCostTracker` at 5+ spots)
- REGENERATE: any Colab notebooks missed by earlier shards (sanity check all 15+)

**Invariants** (3):

1. The 2 textbook HTML files contain only canonical import paths (per `api-cheatsheet.md`).
2. Every `.py` file under `modules/mlfp06/**` and `shared/mlfp06/**` byte-compiles cleanly.
3. Every Colab notebook under `modules/mlfp06/colab/**` reflects the latest .py content (regenerated via `scripts/py_to_notebook.py`).

**Description**: Fix stale imports in the M5 and M6 textbook HTMLs — they don't execute but they teach students wrong paths. Regenerate any Colab notebooks missed by earlier shards. Run the full M6 byte-compile sweep as the final verification. Commit the migration complete.

**LOC estimate**: ~30 LOC load-bearing (HTML text replacements).

**Feedback loop**:

```bash
# Full M6 byte-compile sweep
find modules/mlfp06 shared/mlfp06 -name "*.py" -exec uv run python -m py_compile {} \;

# Verify no broken imports remain
grep -rE "from kailash_pact|from kailash_nexus|from kaizen.core import BaseAgent|from kaizen.core import.*Signature|LLMCostTracker" \
  modules/mlfp06 shared/mlfp06 --include="*.py" --include="*.html"
# Should return zero matches

# Regenerate all Colab notebooks
uv run python scripts/py_to_notebook.py --module mlfp06

# Final commit
git add -A modules/mlfp06 shared/mlfp06 workspaces/mlfp06-migration pyrightconfig.json
git commit -m "feat(mlfp06): complete migration to pact 0.8.1 + kaizen 2.7.3 + nexus 2.0.1

Migrates all 52 M6 files across 9 shards per workspaces/mlfp06-migration/
shard-plan.md. Headline: GovernedSupervisor replaces the legacy
PactGovernedAgent wrapper; canonical BaseAgent pattern (dataclass config +
instance signature) fixes a silent DefaultSignature fallback bug where
M6 students had been receiving generic default output regardless of their
declared Signatures.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Shard Dependency Graph

```
    0 (pre-flight)
    ├── 1 (ex_5)      ─┐
    ├── 2 (ex_6)      ─┤
    ├── 3 (ex_7 shared)── 4 (ex_7 env/budget) ── 5 (ex_7/04 runtime)
    ├── 6 (ex_8 shared)── 7 (ex_8 technique)
    ├── 8 (exam.py)   ─┤
    └──────────────────┴── 9 (final verify + textbooks)
```

Shards 1, 2, 8 are independent and can run in any order relative to each other (or in parallel worktrees). Shards 3→4→5 are strictly sequential on the ex_7 branch. Shards 6→7 are strictly sequential on the ex_8 branch. Shard 9 is the final gate and depends on all others.

## Estimated Throughput

- **Sequential**: 9 sessions (1 shard per session)
- **Parallel (2 worktrees)**: 6 sessions if Shards 1+2 run in parallel with 3+4+5, and 8 runs in parallel with 6+7
- **LOC**: ~4,000 load-bearing across 52 files
- **Feedback loop cost**: ~30 seconds per shard (byte-compile + smoke test)
