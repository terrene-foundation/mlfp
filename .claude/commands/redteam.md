---
name: redteam
description: "Load phase 04 (validate) for the current workspace. Red team testing."
---

## Workspace Resolution

1. If `$ARGUMENTS` specifies a project name, use `workspaces/$ARGUMENTS/`
2. Otherwise, use the most recently modified directory under `workspaces/` (excluding `instructions/`)
3. If no workspace exists, ask the user to create one first
4. Read all files in `workspaces/<project>/briefs/` for user context

## Phase Check

- Verify `todos/active/` is empty (all implemented) or note remaining items
- Read `workspaces/<project>/03-user-flows/` for validation criteria
- Validation results go into `workspaces/<project>/04-validate/`
- If gaps are found, document them and feed back to implementation (use `/implement` to fix)

## Execution Model

Autonomous execution model (see `rules/autonomous-execution.md`). Red team converges through iterative rounds. Findings are fixed autonomously, not reported for human triage.

## Workflow

### 1. Spec compliance audit (MUST run first)

**File existence is NOT compliance.** Use the protocol in `skills/spec-compliance/SKILL.md` to verify each spec promise via AST parsing and targeted greps, NOT file existence or self-reports.

A "spec" is any documented promise about behavior, regardless of where it lives. Sources to audit:

- `specs/**` — domain specifications (PRIMARY source of truth)
- `workspaces/<project>/briefs/**` — user-supplied requirements
- `workspaces/<project>/01-analysis/**` — analyst findings, deep analyses, design notes
- `workspaces/<project>/02-plans/**` — implementation plans, ADRs, contracts
- `workspaces/<project>/todos/completed/**` — what each todo claimed to deliver
- Inline spec sections in README.md, CHANGELOG.md, or design docs the project references

For every spec promise found in these sources:

1. Extract literal acceptance assertions from the spec text (class signatures, field names, decorator call sites, MOVE shim semantics, security tests, migration completion).
2. Verify each assertion via grep or `ast.parse` against the actual code.
3. Re-derive every check from scratch — do NOT trust `.spec-coverage`, `.test-results`, `convergence-verify.py`, or any prior round's self-report. Self-reports are inputs to verify, not evidence to trust.
4. Save the assertion table to `workspaces/<project>/.spec-coverage-v2.md` (the `-v2` suffix prevents confusion with legacy file-existence reports).

**Critical patterns to flag (see `skills/spec-compliance/` for full list):**

- Class/method exists but constructor signature differs from spec
- Frozen dataclass missing spec-required fields (grep returns 0)
- `@deprecated` decorator defined in `deprecation.py` but never applied at call sites
- "MOVE A → B" tasks where source A still exists at full size (drift risk)
- New modules with zero importing tests (`grep -rln "from <new_module>" tests/` empty)
- `def run_stream / async def stream_*` methods with only one `yield` (fake stream)
- Consumer files still importing from OLD path after a "migrate to Y" task

**Specs-to-code verification** — for every file in `specs/`, extract assertions at FIELD level (not just endpoint/class level) and verify against code via grep/AST. Code diverging from spec without a logged deviation = HIGH. **Cross-spec consistency** — grep all specs for shared terms (TTLs, limits, field names, endpoint paths); contradictory values across specs = HIGH. **Brief-to-spec coverage** — for each requirement in `briefs/`, verify it maps to at least one spec section; unmapped requirements = HIGH.

### 2. End-to-end validation

Review implementation with red team agents using playwright mcp (web) and marionette mcp (flutter).

- Test all workflows end-to-end:
  - Using backend API endpoints only
  - Using frontend API endpoints only
  - Using browser via Playwright MCP only

### 3. User flow validation

Red team agents read `workspaces/<project>/03-user-flows/` and validate every detailed storyboard.

- Workflows include: what is seen, clicked, expected, value delivered
- Every transition between steps must be evaluated
- Focus on intent, vision, requirements — never naive technical assertions

### 4. Test verification — re-derive, do NOT trust .test-results

See `rules/testing.md` § Audit Mode Rules.

1. Do NOT read `.test-results` to verify test counts. The file is written by `/implement` and may report old-code coverage while new spec modules have zero tests.
2. Run `pytest --collect-only -q` (or your project's equivalent test enumeration command) on the test directories.
3. For each new module the spec created, grep the test directory for an import of that module. Zero importing tests = HIGH finding regardless of "tests pass".
4. Run any NEW tests that red team writes (E2E, regression tests for findings).
5. If a test is suspected wrong, re-run THAT test specifically.

### 5. Report results

Report all detailed steps and results in validation. Include the assertion tables from Step 1 verbatim — every row must show the literal verification command and its actual output, not "exists: yes".

### 6. Parity check (if required)

If parity required: test-run old system, record outputs. For natural-language output, use LLM evaluation (not keyword/regex). See `.env` for model.

### 7. Log triage gate

Per `rules/observability.md` MUST Rule 5: scan build/test output + `*.log` for WARN+ entries. Group identical entries, disposition each as Fixed (commit SHA) / Deferred (tracked todo) / Upstream (pinned version) / False positive. Unacknowledged WARN+ entries BLOCK convergence.

## Agent Teams

**Core red team (always):**

- **analyst** — Step 1 owner. Reads `skills/spec-compliance/SKILL.md`, derives assertion tables from each plan, runs AST/grep verification, produces `.spec-coverage-v2.md`.
- **testing-specialist** — Step 4 owner. Re-derives test coverage via `pytest --collect-only` (or the project's equivalent). Verifies new modules have new tests.
- **value-auditor** — Skeptical buyer perspective on every page/flow
- **security-reviewer** — Full security audit; verifies every spec § Security Threats subsection has tests

**Validation perspectives (selective):**

- `co-reference` skill — methodological compliance
- **gold-standards-validator** — naming/licensing compliance
- **reviewer** — code quality across changed files

**Frontend validation (if applicable):**

- **uiux-designer** — visual hierarchy, responsive, accessibility, AI interaction

## Convergence Criteria (D237)

D237: Review and check discipline (supersedes "2 consecutive clean rounds" and "full re-run after each fix")

REVIEW
1. Round 1 is a full review. Security-critical changes use a PAIRED round 1: a correctness reviewer + an adversarial security reviewer, in parallel. Security-critical includes auth, signing, revocation, tenant isolation, fail-closed gates, kill paths, destructive ops, disclosure, and ANY trust boundary. When unclear, treat it as security-critical. Where a repo rule requires a larger team (e.g. self-referential artifact changes), that team replaces the pair in every round.
2. Evidence gate, EVERY round: a reviewer counts only if it genuinely ran. An errored, empty or timed-out review is NO evidence: re-run it, never count it.
3. A round is "clean" when it has no CRITICAL/HIGH finding. Round 1 clean = review ends.
4. Round 1 not clean: fix, then Round 2 reviews ONLY the fixes plus their blast radius (callers of changed code, same-class sibling sites), with the SAME reviewer composition as Round 1. Then review ends.
4a. Exception, small security-critical fixes only: if the Round-2 fix for a security-critical finding is itself small, ONE extra Round 3 may run. It reviews ONLY that fix's delta, with the same composition. Anything still CRITICAL/HIGH after Round 3 goes to the owner. There is never a Round 4.
5. WHEN REVIEW ENDS, at any round, every remaining finding of ANY severity goes into the repo's durable work LEDGER (tracked todos or issue tracker). None is dropped:
   a) A finding that leaves NO residual risk on a shipped path goes to the ledger as deferred, with its reason.
   b) A small defect that does not affect landing goes to the ledger. It does not block the landing.
   c) A small KNOWN RISK on a shipped path may be PROVISIONALLY accepted by the agent, so the change can land. It goes to the ledger as OPEN, with the risk stated plainly, and STAYS OPEN until a human either accepts it (named acceptor) or asks for the fix. The agent's acceptance never closes it.
   d) Any other residual risk on a shipped path, whatever its severity, is fixed, or goes to the owner as an accepted risk with a named acceptor.
   Severity never decides fix-vs-defer.
6. Never deferred at any severity: stubs/placeholders, silent error-swallowing, failing tests, warnings. Fix them; no extra review round.
CHECKS
7. While fixing: re-run the failed check + the checks the fix's diff touches. Never the full set per fix.
8. The full set runs ONCE, on the final version, as sign-off. If it fails: fix, run rule 7, then ONE more full sign-off. A second failure stops the batch for re-planning.
9. The local sign-off set must match CI's required set. A red that only CI caught is a gap in the local set; close it.

Repo-specific gates (unchanged by D237):

1. **Spec compliance: 100% AST/grep verified** — every spec section has an assertion table where every row shows a literal verification command (`grep …`, `ast.parse(…)`, `wc -l …`) and its actual output. Rows saying "exists: yes" are BLOCKED.
2. **New code has new tests** — `pytest --collect-only` shows ≥1 test importing each new module. Zero new tests for a new module = HIGH, regardless of suite-level "tests pass".
3. **Frontend integration: 0 mock data** — no `MOCK_*/FAKE_*/DUMMY_*` constants, no `mock*()` / `generate*Data()` functions, no hardcoded display arrays.

D237's review rules are necessary but NOT sufficient. Without the repo-specific gates, convergence certifies code quality on incomplete software.

### Journal (MUST — phase-complete gate)

Before reporting `/redteam` complete, create journal entries for journal-worthy findings surfaced during validation:

- **RISK** — vulnerabilities, weaknesses, or failure modes discovered
- **GAP** — missing tests, docs, edge cases, or spec-compliance holes

Use `/journal new <TYPE> <slug>` (or write directly to `workspaces/<project>/journal/NNNN-TYPE-slug.md`). Skip only when validation genuinely produced nothing journal-worthy — use judgment, not formulas. Do not batch: create each entry as you recognize it.
