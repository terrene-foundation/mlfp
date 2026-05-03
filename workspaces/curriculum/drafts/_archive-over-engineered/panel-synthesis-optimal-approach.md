# Expert Panel Synthesis — Revised Optimal Approach

8 of 9 specialists returned findings (open-source-strategist hit a model-access error; flagged as gap, not fatal — strategic framing can be re-solicited or absorbed into the ADR's existing Foundation-independence section).

**Verdict**: the v1 drafts are content-strong but **should NOT be filed as written**. The panel surfaced two procedure-compliance blockers, one CRITICAL security gap, two placement errors, and one commercial-coupling removal required before upstream. With those fixed, the proposal lands.

## What the panel found (ranked by severity)

### 1. Artifact-flow violation — BLOCKING procedure issue

**claude-code-architect**: MLFP is two layers downstream of the BUILD repos. Per `rules/artifact-flow.md`, the canonical path is `MLFP → loom (proposal) → kailash-py/rs`. Filing direct from MLFP bypasses loom classification and `rules/nested-repos.md` Rule 1.

**Implication**: Drafts should be authored at MLFP (done), but filing mechanism is **not** `gh issue create` from MLFP. It's:

1. MLFP writes a `/codify` proposal at loom targeting SDK artifacts
2. Loom reviews, classifies (global vs. variant), routes
3. Loom or a BUILD-repo maintainer files the ADR PR + epic issues on kailash-py / kailash-rs

I was about to violate this. Correction: the final step is "open Claude Code in loom and run /codify", not "gh issue create from mlfp".

### 2. PII leakage in JSONL events — CRITICAL security gap

**security-reviewer**: Proposal's security/multi-tenancy section is 3 bullet lines. Event payloads (`TraceEvent.args`, `JudgeVerdict.rationale`, `AuditEntry.subject`) carry raw user content by default. First DSAR / subpoena → catastrophic disclosure. `rules/event-payload-classification.md` and `rules/tenant-isolation.md` must be hard acceptance criteria on every Phase 1 engine.

Also HIGH: path-traversal via `experiment_id` (no tenant dim on disk layout), file perms / umask / symlink (Python side silent on what Rust already enforces at 0o600), WebSocket multiplexing (unbounded buffers, cross-tenant subscription, channel-ID spoofing), supply-chain CVE floors (torch < 2.2 RCE, trust_remote_code defaults, ragas/trl chains), audit-chain truncation attack (no signing, only hashing).

### 3. LLMJudge placement error — cross-package ownership drift

**ml-specialist + kaizen-specialist** both independently flagged: `LLMJudgeEngine` in `kailash_ml.engines.llm_judge` is **wrong**. It forces kailash-ml to depend on Kaizen (Delegate), inverting the existing dep graph (kaizen consumes ML, not peer). Correct placement:

- `kaizen.judges.llm_judge_callable.LLMJudgeCallable` (depends on Delegate; Kaizen owns it)
- `kailash_ml.engines.llm_judge.LLMJudgeEngine` accepts a `kaizen.judges.Judge` Protocol — kailash-ml depends on the **interface**, not the implementation.

Also: drop `Engine` suffix (judges aren't pipelines); use `LLMJudge` bare class.

### 4. AgentTracer path is fantasy — real observability home is different

**kaizen-specialist**: `kaizen.observability.agent_tracer` doesn't exist. Real home: `kaizen.core.autonomy.observability.*`. Two fixes:

- Promote `kaizen.observability` to a stable public surface in the SAME PR (per `rules/orphan-detection.md` §6 — eager import + `__all__`)
- Compose, don't sit beside: AgentTracer routes through `ObservabilityManager`, writes cost via `CostTracker` (which uses **microdollars** — the MLFP helper sums floats, re-introducing the precision bug CostTracker already solved)

### 5. Commercial coupling in MLFP source — BLOCKING for upstream

**kaizen-specialist**: `shared/mlfp06/diagnostics/agent.py` lines 488-564 hard-code a **Langfuse exporter** with env-var autoload. Per `rules/independence.md` §"No Commercial References", this gets blocked at gold-standards-validator. Must strip to a `TraceExporter` Protocol with `LangfuseExporter` in a separate optional extra before any upstream PR.

### 6. Single-consumer risk — ADR rejection vector

**analyst**: MLFP is the sole evidence base. Standard maintainer pushback: "who else is asking for this?" Without a second co-signing consumer, the ADR reads as "upstream my course's helpers." Either:

- Before filing ADR, identify 2-3 Foundation projects that commit to adopting (candidates: Terrene internal ML systems, kailash-align needing its own diagnostics, a third active engagement)
- OR land engines as individual feature PRs without the convention ADR

This is the **single biggest structural risk** per analyst.

### 7. Convention without enforcement — Phase-4 gate too late

**analyst**: CI gate ("new engine PRs must touch diagnostics/") is in Phase 4. Without the gate, new ML engines land without diagnostics for 18 months. Conventions without enforcement = dead letter within 2 release cycles.

Fix: Ship CI gate in **Phase 1**, same PR as ADR. Warn-only initially, promote to blocking at Phase 2.

### 8. Schema governance — duplicate-source drift

**release-specialist + analyst**: Drafts say schemas live in `docs/schemas/` (py) AND `crates/kailash-observatory/schemas/` (rs). Two sources of truth → guaranteed drift under 4-month version lag. Fix: **single canonical repo** `terrene-foundation/kailash-schemas` (PyPI + crates.io published as `kailash-schemas` package). Both SDKs depend on it with version pin. Schema evolution = schema-package SemVer bump.

### 9. Canonical JSON drift — cross-SDK chain verification breaks silently

**pact-specialist**: `GovernanceAuditor.verify_chain()` must match bytes across py and rs. Python `json.dumps(sort_keys=True, default=str)` vs Rust `serde_json` canonical form differ on: float representation (`1.0` vs `1`), Unicode normalization (`\u00e9` vs `é`), timestamp serialization (ISO vs RFC3339). Without a shared RFC 8785 JCS spec, every row with non-ASCII breaks. Spec the canonical-JSON contract in ADR-0058, ship as `kailash-schemas` primitive, enforce via golden-file tests.

### 10. Release ordering — 5 coordinated publishes is a P0 incident vector

**release-specialist**: Phase 3 publishes kailash-ml 0.15, kailash-kaizen (minor), kailash-align (minor), kailash-pact (minor), and kailash-observatory 0.15.0 — five packages in one phase. One failure = user `pip install` breakage. Dep-ordered publish with clean-venv verification gate after each:

1. kailash-ml 0.15
2. kailash-kaizen (needs kailash-ml 0.15)
3. kailash-align (AlignmentMonitor submodule)
4. kailash-pact (GovernanceAuditor submodule)
5. kailash-observatory 0.15.0 (peer deps on all four, pins floor versions)

Also: **reserve names now** on PyPI + crates.io with `0.0.1` placeholder. `kailash-observatory` availability window is 4 months; name squatting is real.

### 11. Other drift findings (smaller, concrete)

- **ADR number collision** (ml-specialist) — verify 0058 isn't taken before filing
- **API async deviation** (ml-specialist) — kailash-ml engines are async; proposed sync context managers deviate. Wrap sync hooks in async outer API
- **dep extras collision** (ml-specialist) — `[training-diagnostics]` pulls torch, but `kailash-ml[dl]` already does. Use `[dl,diagnostics]` inheritance
- **AttentionExplainer split** (ml-specialist) — don't overload ModelExplainer with logit-lens/SAE; new bare class
- **AlignmentMonitor stateless utilities** (align-specialist) — `kl_divergence`/`win_rate` as module-level, not class methods
- **trl dep drop** (align-specialist) — closed-form KL is adequate; trl pulls 2GB for marginal gain
- **Schema version cross-SDK compat-window test** (ml-specialist, missed by red-team) — CI must replay JSONL fixtures from prior N minor versions and assert portal renders without loss
- **Rust portal decoupling** (release-specialist) — don't gate Phase 3 on Leptos SPA maturity; ship Rust observatory as pure JSONL lib, portal UI in Phase 4
- **MLFP import breakage timing** (ml-specialist) — M5 self-contained generator bug (still open) will re-break 84 notebooks if migration happens before generator is fixed

### 12. Artifact shape (claude-code-architect — meta)

- ADR draft is **127 lines** → house ADR convention is ~90 (ADR-0017 reference). Slim. Move rollout/governance/dep-policy to epics.
- Epics are **140-230 lines** → GitHub intake pattern is **≤60-line tracker + 7 child issues** at file time. Redistribute.
- Drafts in `workspaces/curriculum/drafts/` violates `rules/specs-authority.md` (workspaces = process artifacts). Move to `proposals/sdk/observability/` with an INDEX.md.
- Missing artifacts: CC skill for the pattern (`sdk-upstream-proposal/`), `observability-specialist` agent draft for loom, `rules/diagnostics-convention.md` for loom to distribute, `specs/diagnostics-portal.md` as the MLFP authoritative spec.

## Revised optimal approach

Incorporating panel findings, the path is:

### Phase 0 — Pre-file work (NEW, before anything touches GitHub)

1. **Re-route via loom**. Move drafts from `mlfp/workspaces/drafts/` → `mlfp/proposals/sdk/observability/` + `INDEX.md`. Then at loom, run `/codify` to create the BUILD-repo proposal that carries this content forward. Filing is loom's job, not MLFP's.

2. **Strip Langfuse** from `shared/mlfp06/diagnostics/agent.py`. Replace with `TraceExporter` Protocol + optional extra. MLFP keeps local Langfuse wiring if needed (that's user-side config), but the upstream surface ships exporter-agnostic.

3. **Secure the event contract**. Fold `rules/event-payload-classification.md` + `rules/tenant-isolation.md` into every engine's acceptance criteria. Single-point JSONL sink with `format_record_id_for_event`, path validation (`^[a-zA-Z0-9_-]{1,64}$` on experiment_id / run_id), `{ROOT}/{tenant_id}/{experiment_id}/{run_id}/{lens}.jsonl` disk layout.

4. **Fix placement errors**:
   - `LLMJudge` → kaizen.judges (interface) + kailash-ml consumer
   - AgentTracer → promote `kaizen.observability` public surface, compose through `ObservabilityManager` + `CostTracker` (microdollars)
   - AttentionExplainer → bare class, not `ModelExplainer` method

5. **Reserve package names**. `kailash-observatory` 0.0.1 placeholder to PyPI + crates.io. Owner: `terrene-foundation`.

6. **Identify co-signing consumers**. 2-3 Foundation projects committed to adopt post-Phase-1. Without this, ADR is high-risk to reject.

7. **Slim artifacts to house style**. ADR → ~90 lines per ADR-0017 template. Epics → ~60-line trackers with child-issue links. Move implementation detail (API conventions, JSONL schemas, extras tables) into child issues and the ADR's Architecture Components section.

8. **Verify ADR number**. Check `docs/adr/` for 0058 availability before claiming the number.

### Phase 1 — Ship with gate, not facade (REVISED)

- Ship CI convention gate **in Phase 1**, warn-only (blocks nothing, but logs every engine PR that doesn't touch diagnostics/). This is the enforcement spine.
- Ship 4 low-risk engines: TrainingDiagnostics (py), AgentTracer (py+rs), RetrievalEvaluator (py+rs), GovernanceAuditor (py+rs).
- **Do NOT ship Observatory facade yet**. Defer to Phase 3 when all 7 lenses exist. Facade orphan risk per analyst FM-B.
- Canonical-JSON spec lands with GovernanceAuditor. Shared `kailash-schemas` repo established.

### Phase 2 — LLM lenses

- LLMJudge (kaizen.judges + kailash-ml consumer), TextQualityMetrics, AttentionExplainer, AlignmentMonitor
- Convention gate promotes from warn-only to **blocking**
- MLFP deprecation warnings on local helpers (per-phase cohort cycle)

### Phase 3 — Facade + portal (py-first)

- `kailash-observatory` package with Observatory facade (composes all 7 available lenses, graceful degradation on missing extras)
- Portal extension to `kaizen.monitoring.PerformanceDashboard` — agent-panel addition for Kaizen, ML-lens composition in observatory
- MLFP hard-deletes local helpers, migrates to SDK imports
- **Dep-ordered release**: ml → kaizen → align → pact → observatory, with clean-venv verification after each
- Rust `kailash-observatory` crate ships as JSONL library only (no Leptos SPA yet)

### Phase 4 — 1.0 milestone (meaningful cut)

- Rust Leptos portal SPA
- Multi-run comparison UI
- OTel wiring completion (every `record_*` instrumented)
- **`kailash-ml 1.0` is the "observability-first" cut**, not just CI polish (release-specialist's rebrand)

### Phase 0.5 — Parallel: pitch the pattern as reusable

Build a CC skill at atelier / loom: `sdk-upstream-proposal/` — codifies audit → red-team → ADR → epic + child issues flow so the next cross-SDK proposal reuses the template instead of rediscovering it.

## What the user needs to decide

1. **Procedure correction** — Do you accept re-routing through loom (correct per artifact-flow.md) rather than direct MLFP → GitHub? The drafts stay MLFP-authored; loom handles the filing. If you prefer direct, we'd need an explicit exemption documented.

2. **Co-signers** — Shall I identify Foundation projects for co-signing before any filing? Without this, analyst's FM-C (single-consumer rejection) is the single biggest rejection risk.

3. **Scope re-cut** — Accept the 4 fixes above (strip Langfuse, slim artifacts, security fold-in, placement corrections, reserve names), plus the Phase-1-gate + Phase-4-as-1.0 changes? If yes, I revise the 5 drafts and route through loom.

4. **open-source-strategist re-run** — The strategic-framing agent errored out. Do you want me to retry with a fallback model, or is the Foundation-independence section in the ADR + analyst's FM-C adequate coverage?

5. **The biggest structural call** — Three delivery options surfaced:

   | Option                             | What it means                                                                                                                                                                |
   | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
   | **A. Full proposal** (recommended) | Fix blockers, route via loom, ADR-first (with co-signers), phased 18-month rollout                                                                                           |
   | **B. Smaller version**             | Land 2-3 least controversial engines (AgentTracer, RetrievalEvaluator, GovernanceAuditor) as individual feature PRs, no convention ADR. Saves 18 months but the gap reopens. |
   | **C. No upstream**                 | Keep diagnostics in MLFP, accept the maintenance tax, revisit when Foundation has more ML consumers                                                                          |

   My recommendation after panel: **A, but with Phase 0 pre-file work done first**. B is a retreat that cedes the convention. C is the status quo the user just rejected. But A requires co-signers — if no second consumer exists in the Foundation today, B becomes the honest move until one does.

## What's needed from you next

Pick: proceed with Phase 0 pre-file work (Option A), scope down to Option B, defer to Option C, or surface something else. The 5 drafts are on disk for reference; any option starts by addressing the panel's findings.
