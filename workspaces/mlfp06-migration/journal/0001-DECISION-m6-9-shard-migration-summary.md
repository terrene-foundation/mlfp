---
type: DECISION
date: 2026-04-15
created_at: 2026-04-15T00:00:00Z
author: co-authored
project: mlfp06-migration
topic: M6 nine-shard migration to pact 0.8.1 + kaizen 2.7.3 + nexus 2.0.1
phase: codify
tags: [m6, pact-0-8-1, kaizen-2-7-3, nexus-2-0-1, governed-supervisor, migration-summary]
---

# DECISION: M6 migrated to pact 0.8.1 + kaizen 2.7.3 + nexus 2.0.1 across 9 shards

This entry consolidates the 9-shard migration record that previously lived as 30
auto-extract pending entries under `.pending/`. Each shard has a delivered commit;
the per-commit detail is recoverable via `git show <sha>` and `git log --oneline
--grep="Shard N"`. Promoting to a single curated summary preserves the migration
narrative without retaining the per-commit hook noise.

## Migration scope

Migrated MLFP Module 6 (LLMs and Agentic Workflows) shared foundation, exercises,
decks, and capstone to:

- **pact 0.8.1** — `GovernedSupervisor`, modern envelope + budget access, runtime audit
- **kaizen 2.7.3** — modern agent imports, structured/critic agent contracts, multi-agent composition
- **nexus 2.0.1** — capstone deployment surface

## Shard delivery (chronological)

| Shard | Commit       | Subject                                                                  |
| ----- | ------------ | ------------------------------------------------------------------------ |
| 0     | `9e2a47de`   | feat(mlfp06): add Pyright config scoped to M6 migration                  |
| 1     | `bf3c55b4`   | feat(mlfp06): migrate ex_5 structured/critic agents to kaizen 2.7.3      |
| 2     | `9b8bb6c8`   | feat(mlfp06): migrate ex_6 multi-agent exercises to kaizen 2.7.3         |
| 3     | `8d563521`   | feat(mlfp06): migrate ex_7 shared PACT foundation to pact 0.8.1          |
| 4     | `e6536b5d`   | feat(mlfp06): migrate ex_7 envelopes + budget access to pact 0.8.1       |
| 5     | `d1b03c58`   | feat(mlfp06): migrate ex_7 runtime audit to GovernedSupervisor           |
| 6     | `24765e76`   | feat(mlfp06): migrate ex_8 shared foundation to GovernedSupervisor       |
| 7     | `70c27e27`   | feat(mlfp06): migrate ex_8 capstone technique files to GovernedSupervisor|
| 8     | `e34ea873`   | feat(mlfp06): migrate exam.py PACT block to modern pact 0.8.1            |
| 9     | `6623c3f0`   | feat(mlfp06): complete migration to pact 0.8.1 + kaizen 2.7.3 + nexus 2.0.1 |

Plan + reference implementation: `24f9412a` — `docs(mlfp06): add M6 migration
workspace — 9-shard plan + reference impl`.

## Adjacent work captured separately

The following work landed during the migration window but lives in
`workspaces/curriculum/journal/`, not here, because it is curriculum-wide rather
than M6-specific:

- **0003-DECISION** — Codify R11–R13 (self-contained Colab + transitive inlining)
- **0004-DISCOVERY** — M6 holistic audit (six-lens library coherent with 8 lessons)
- **0005-DECISION** — M6 Ollama migration (Redline 14, no commercial LLM provider)
- **0006-DISCOVERY** — Kaizen `TextDelta.text` (not `.delta_text`) event-shape trap
- **0007-DISCOVERY** — kailash-ml 0.12 gap analysis (~4,800 LOC upstream candidates)

## Why a single curated summary instead of 30 auto-extracts

The 30 auto-extracts under `.pending/` were SessionEnd-hook outputs — each
fired multiple times per commit, producing 2–4 duplicates per shard. The unique
content was 19 commit subjects, all already in `git log`. A single curated entry
with the shard table is searchable, reviewable, and self-contained per
`rules/journal.md` § Requirements; the 30 hook outputs were neither.

## Consequences

- M6 ships on the modern Foundation stack; no kaizen 0.x / pact 0.7.x / nexus 1.x carry-over.
- `GovernedSupervisor` is the M6 reference pattern for governance integration.
- Migration workspace is at end-of-life — Shard 9 closed the scope. Workspace can be archived after this entry lands.

## For Discussion

1. The migration spent 9 shards across 4 SDK upgrades (pact, kaizen, nexus, plus
   Pyright tightening). Should the next equivalent migration (e.g. when kaizen
   3.0 lands) follow the same shard cadence, or is there evidence the work could
   have been consolidated into 4–5 shards (one per package + one integration)?
2. The SessionEnd hook produced 2–4 duplicates per commit, generating the noise
   that motivated this triage. Is the hook firing on every session OR every
   commit OR both? If it's commit-driven the dedup is impossible without
   teaching the hook to debounce — is that worth fixing upstream, or do future
   sessions just live with .pending/ noise that gets curated at /journal time?
3. The migration delivered without a single workspace journal entry being
   manually authored — every record was auto-extracted. That's a scale signal:
   either the work was routine enough not to need decision capture, or the
   right decisions weren't captured. Looking at the curriculum/.pending entries
   (the M6 ollama migration generated three rich manual entries), is the
   pattern that ad-hoc work captures decisions and planned migrations don't,
   and is that healthy?
