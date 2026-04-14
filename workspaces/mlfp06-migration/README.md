# MLFP06 Migration Workspace

**Status**: Planning complete. Execution deferred to next session.
**Created**: 2026-04-14
**Owner**: Jack Hong
**Scope**: Full Module 6 migration to `kailash-pact==0.8.1` + `kailash-kaizen==2.7.3` + `kailash-nexus==2.0.1`. 52 files across 9 shards.

## Why This Workspace Exists

M6 was never in the `pyproject.toml` "M1-M5 fresh-venv tests verified" baseline. It has been silently broken against the pinned framework versions for an unknown period. Discovery happened during a `/sync` follow-up on 2026-04-14 when Pyright diagnostics exposed the drift after a mechanical import rename.

Investigation revealed the drift was much deeper than imports — entire API surfaces had moved or been redesigned, plus a silent bug (`DefaultSignature` fallback) that meant students weren't getting the typed output they thought they had. Every `class XxxAgent(BaseAgent): signature = XxxSig` in M6 has been producing generic default output.

Parallel specialist investigations (pact-specialist + kaizen-specialist) produced the full scoping. **Headline finding**: `kaizen_agents.GovernedSupervisor(model, budget_usd, tools, data_clearance)` is a near-drop-in replacement for the legacy `PactGovernedAgent` wrapper. Same three knobs, plus hash-chain tamper-evidence as a free pedagogical upgrade. This reduced the migration from "rewrite teaching narrative" to "swap wrapper API + rewrite shared helpers + fix kaizen canonical path".

## Files in This Workspace

| File                                 | Purpose                                                                  |
| ------------------------------------ | ------------------------------------------------------------------------ |
| `README.md`                          | Entry point + resume instructions (this file)                            |
| `scoping.md`                         | Full investigation findings — drift inventory, file inventory, bug bombs |
| `api-cheatsheet.md`                  | Old → New API mapping with copy-paste code blocks                        |
| `shard-plan.md`                      | 9-shard execution plan per `rules/autonomous-execution.md` budgets       |
| `decisions.md`                       | 6 pedagogical decisions, all approved 2026-04-14                         |
| `reference/ex_7_04_runtime_audit.py` | Pact-specialist's 450-LOC reference implementation for Shard 5           |

## How to Resume in the Next Session

1. **Read in order**: `scoping.md` → `api-cheatsheet.md` → `decisions.md` → `shard-plan.md`. Together they are ~1500 lines; budget one context window for the read.
2. **Execute Shard 0** (pre-flight): revert the 5 partial import-rename edits, install `pyrightconfig.json`, verify `scripts/py_to_notebook.py`. See `shard-plan.md` § Shard 0.
3. **Execute shards in order** 1 → 9. Each shard ships a per-shard feedback loop (byte-compile + smoke test + regenerate Colab notebooks). Do NOT skip the loop — it is the only structural guard against re-introducing the same drift.
4. **After each shard**: commit with a conventional commit message (`feat(mlfp06): migrate <shard-name> to pact 0.8.1 / kaizen 2.7.3`). Each shard is a natural commit boundary.
5. **Parallel opportunities**: Shards 1 and 2 are kaizen-only and have no shared dependencies on Shards 3–9 — they can run in parallel worktrees if throughput matters. Shards 3→4→5 are strictly sequential (each depends on the prior's edits to `shared/mlfp06/ex_7.py`). Shard 6 is strictly sequential before Shard 7 (shared → technique files). Shard 8 (exam.py) is independent and can run anywhere in the sequence after Shard 0.

## Current State at End of Session 2026-04-14

### In-Progress / Partial

Five files have half-applied import renames that will be REVERTED in Shard 0 and re-migrated cleanly in Shards 4, 5, 7. They are NOT safe to leave in their current state — they contain the new imports but the old API calls, so they import-succeed then crash at runtime.

| File                                                       | Current state                                                                                                                   | Will be fixed in |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| `modules/mlfp06/solutions/ex_7/04_runtime_audit.py`        | `from pact import PactGovernedAgent` ✓ / `PactGovernedAgent(agent=..., governance_engine=..., role=..., max_budget_usd=...)` ✗  | Shard 5          |
| `modules/mlfp06/solutions/ex_8/02_governance_pipeline.py`  | `from pact import GovernanceEngine, PactGovernedAgent` ✓ / `engine.compile_org(path)` ✗ / `PactGovernedAgent(agent=..., ...)` ✗ | Shard 7          |
| `modules/mlfp06/solutions/ex_8/03_multichannel_serving.py` | `from nexus import Nexus` ✓ / `from pact import ...` ✓ / everything downstream ✗                                                | Shard 7          |
| `modules/mlfp06/solutions/ex_8/04_drift_monitoring.py`     | Same shape as `02`                                                                                                              | Shard 7          |
| `modules/mlfp06/solutions/ex_8/05_compliance_audit.py`     | Same shape as `02`                                                                                                              | Shard 7          |

### Clean and Committed-Ready

- `pyproject.toml` pin bumps (`kailash>=2.8.5`, `kailash-dataflow>=2.0.7`, `kailash-mcp>=0.2.3`) — valid, `uv sync` ran cleanly, M1-M5 byte-compiles 192/192. **DO NOT REVERT.** These pin bumps are part of the migration and land with it.
- `uv.lock` updated by `uv sync`.
- `.claude/*` sync artifacts from kailash-coc-claude-py template 3.4.8 (38 updated, 3 new files, 10 project-specific preserved).
- `.claude/.coc-sync-marker` updated to `template_version: 3.4.8`.

### Not Touched This Session

- All 15 `modules/mlfp06/local/` mirror files
- All 15 Colab notebooks under `modules/mlfp06/colab/`
- `modules/mlfp06/assessment/exam.py` (1,531 LOC)
- `shared/mlfp06/ex_6.py`, `ex_7.py`, `ex_8.py`
- `modules/mlfp06/lessons/05/textbook.html` and `lessons/06/textbook.html` (5+ stale imports)

## Decisions Baked Into This Plan

All 6 pedagogical decisions in `decisions.md` were approved by Jack on 2026-04-14 with "all defaults, go" directive. Summary:

1. Task 4 (ex_7/04) reframed from "adversarial prompt blocking" to "blast-radius containment"
2. `build_capstone_stack()` helper extracted to dedupe 4× 40-LOC blocks in Shard 7
3. exam.py Task 4a shifts from `Address(domain=, team=, role=)` kwargs to `Address.parse("D1-R1-...")` dash grammar
4. Four-level clearance hierarchy preserved with footnote explaining canonical 5-level alignment
5. Nexus `app.register(bare_async_fn)` verified at Shard 7 start; fallback is one-node `WorkflowBuilder` (already taught in course)
6. `LLMCostTracker` replaced by `BaseAgentConfig.budget_limit_usd` narrative in 5 spots

## Throughput Estimate

- **~4,000 LOC** load-bearing edits across 52 files
- **9 shards** at ~1 shard per autonomous session → **~8 sessions** (Shards 1+2 can parallelize)
- **No external blockers** — all specialist questions answered
- **Feedback loop per shard**: `uv run python -m py_compile <files>` + offline smoke test + regenerate Colabs

## Resume Integrity Warning

If a future session edits M6 files without reading this workspace first, it will discover drift symptoms (Pyright errors, broken imports, silent `DefaultSignature` fallback) and may make local fixes that drift from the shard plan. The next session's FIRST action MUST be to read `scoping.md` + `api-cheatsheet.md` before touching any M6 file.
