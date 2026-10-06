# MLFP06 — End-of-Module Assessment: Language Models & Agentic Workflows

Four practical coding tasks on the module's production skills: PACT operating
envelopes and deny-paths, a governed agent's tools and config, a served
governed endpoint behind real middleware, and a governance organisation
designed from a brief. There is no multiple choice.

The tasks state **goals, contracts and acceptance criteria**. They do not
give the implementation. Designing it is part of every task.

**Duration**: 3 hours · **Total**: 100 marks · **Open book**: documentation is
allowed; AI assistants are **not** allowed.

## No LLM required

Every task runs **in-process with no LLM calls** — governance engines, tool
registries, governed-supervisor config and Nexus middleware are all exercised
directly, the way the merged exercises probe them. You do not need Ollama (or
any provider) running, and no task downloads a dataset.

## Tasks

| Task | Marks | Framework                                  | What it assesses                                                                                                                                                           | Spec lessons |
| ---- | ----- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| 1    | 30    | PACT (`GovernanceEngine`, `RoleEnvelope`)  | Attach least-privilege envelopes; deny-paths blocked, within-envelope allowed; the installed fail-open default pinned; monotonic tightening with the real clearance ladder | 6.7          |
| 2    | 20    | Kaizen (`ToolRegistry`), kaizen-agents     | Register tools with JSON schemas and executors; build a `GovernedSupervisor` with budget + clearance set at construction                                                   | 6.5, 6.7     |
| 3    | 30    | Nexus (`NexusAuthPlugin`, JWT, rate limit) | Serve one governed endpoint: verified role claim routes the tier, body `role` is ignored, 401/429/CORS all exercised in-process                                            | 6.8, 6.7     |
| 4    | 20    | PACT (`load_org_yaml`)                     | Author a governance org from a brief (departments/teams/roles/clearances/envelopes); the grader rebuilds and probes your YAML                                              | 6.7          |

Each task directory contains:

- `problem.md`: the scenario, the interface, the acceptance criteria and the
  rules;
- `starter.py`: the contract as code (constants, signatures, a local runner).
  You complete it and submit it.

Instructors also hold `solution.py` (the reference), `grader.py`, and the
shared `grading_harness.py`. These are not given to students.

## How grading works

Every grader measures **the behaviour of the objects you return** on inputs
you cannot influence. No number or verdict a submission reports about itself
is trusted:

- Task 1 probes **your returned engine** with grader-drawn costs (fresh
  multipliers every run) on actions inside, outside and over budget — plus an
  unknown address and envelope-less roles, which the installed engine
  auto-approves (that default is pinned as a check). `validate_child` is fed
  grader-built envelope configs, including a `secret` child under a
  `confidential` parent (must be rejected) and a `restricted` child (legal —
  `restricted` is the second-lowest rung of the ladder).
- Task 2 calls your registered executors with fresh seeded inputs, reads your
  agent's envelope attributes directly (`data_clearance="internal"` maps to
  `RESTRICTED`), and runs one governed objective with a grader-supplied
  executor to check the audit chain grows and verifies.
- Task 3 mints its own JWTs with your returned issuer (per-run subjects and
  roles), drives your app's full middleware stack in-process
  (`httpx.ASGITransport` — no ports), and checks served/blocked decisions,
  401s for missing and forged tokens, a 429 burst, and CORS echo behaviour.
- Task 4 re-loads **your YAML**, rebuilds the engine itself, applies the
  specs, discovers role addresses from the rebuilt org, and probes that
  engine. Pre-cooked verdicts are impossible.

Marks are awarded per check: weight × (checks passed / checks). Each task has
a **gate** (the returned contract must exist and work); if a gate fails, the
task scores 0.

Run a grader (instructors):

```bash
python modules/mlfp06/assessment/task_1/grader.py path/to/submission.py
python modules/mlfp06/assessment/task_1/grader.py path/to/submission.py --seed 123  # replay
```

It prints a JSON report with each check, the marks, and diagnostic notes.

## How to work

1. Read `task_N/problem.md` in full.
2. Implement the functions in `task_N/starter.py` and run it — the runner
   exercises your code end to end in-process.
3. Submit your completed `starter.py` files. Do not rename the functions or
   change their contracts.

## Rules

- No LLM calls in any task; no network listeners (Task 3 is driven
  in-process).
- No hardcoded model names — where a model name is needed (Task 2's agent),
  read the course default from `shared.mlfp06._ollama_bootstrap`.
- Governance facts as installed: `verify_action` is the only decision call;
  `.level` is one of `auto_approved` / `flagged` / `held` / `blocked`; the
  clearance ladder is `public < restricted < confidential < secret <
top_secret`; `validate_tightening` is keyword-only.
- Task 3: the handler's module must not use `from __future__ import
annotations` (Nexus's extractor reads runtime annotations); JWT HS256
  secrets must be at least 32 characters (the framework enforces it).
- Deterministic: the grader owns all randomness; the same `--seed` replays
  the same grading run.
