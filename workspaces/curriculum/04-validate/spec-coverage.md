# Spec Coverage Audit — ASCENT Repository

**Audit date**: 2026-04-05
**Plan file**: `/Users/esperie/.claude/plans/hidden-exploring-eich.md`
**Repository**: `/Users/esperie/repos/training/ascent/`

---

## Summary

- **PASS**: 28
- **FAIL**: 14
- **MISSING**: 17
- **Total items**: 59

The repository skeleton (Phase 0) is partially built. Root files, shared utilities, agents, rules, workspace briefs, module READMEs, and docs are in place. The primary gaps are: (1) module content directories (local/, notebooks/, colab/, solutions/, quiz/) are entirely absent, (2) several COC artifacts are missing (commands, skills, settings, hooks), (3) several docs are missing, (4) tests/ and decks/ directories are absent, and (5) there are cross-document inconsistencies in module titles and assessment weights.

---

## 1. Repository Structure

### Root Files

| Item | Status | Notes |
|------|--------|-------|
| CLAUDE.md | PASS | Exists, well-structured |
| pyproject.toml | PASS | Exists with correct dependencies |
| LICENSE | PASS | Apache 2.0, Terrene Foundation copyright 2026 |
| .gitignore | PASS | Exists, includes .data_cache/, .env, Jupyter, ML artifacts |
| .env.example | PASS | Exists with OPENAI, ANTHROPIC, GROQ, HF_TOKEN keys |
| README.md | PASS | Exists with quick start, data loading, platform table |
| CHANGELOG.md | **MISSING** | Plan Section 1 specifies `CHANGELOG.md` at repo root. File does not exist. |

### Module Directories (modules/ascent01 through ascent06)

| Item | Status | Notes |
|------|--------|-------|
| modules/ascent01/README.md | PASS | Exists |
| modules/ascent02/README.md | PASS | Exists |
| modules/ascent03/README.md | PASS | Exists |
| modules/ascent04/README.md | PASS | Exists |
| modules/ascent05/README.md | PASS | Exists |
| modules/ascent06/README.md | PASS | Exists |
| modules/ascent01/local/ | **MISSING** | Plan specifies `local/` subdirectory for .py scripts. Directory does not exist. |
| modules/ascent01/notebooks/ | **MISSING** | Plan specifies `notebooks/` subdirectory for Jupyter. Directory does not exist. |
| modules/ascent01/colab/ | **MISSING** | Plan specifies `colab/` subdirectory for Colab. Directory does not exist. |
| modules/ascent01/solutions/ | **MISSING** | Plan specifies `solutions/` subdirectory. Directory does not exist. |
| modules/ascent01/quiz/ | **MISSING** | Plan specifies `quiz/` subdirectory. Directory does not exist. |
| modules/ascent02-6 (same 5 dirs) | **MISSING** | Same pattern missing for all 6 modules (30 directories total). Confirmed by globbing `modules/*/local/`, `modules/*/notebooks/`, etc. -- all returned empty. |

### Assessment Directory

| Item | Status | Notes |
|------|--------|-------|
| modules/assessment/individual/ | **MISSING** | Plan specifies `modules/assessment/individual/`. Directory does not exist. |
| modules/assessment/group/ | **MISSING** | Plan specifies `modules/assessment/group/`. Directory does not exist. |

### Shared Utilities

| Item | Status | Notes |
|------|--------|-------|
| shared/__init__.py | PASS | Exports `ASCENTDataLoader` |
| shared/data_loader.py | PASS | Full implementation: `ASCENTDataLoader` with Colab detection, gdown fallback, polars readers, module shortcuts |
| shared/kailash_helpers.py | PASS | `setup_environment()`, `get_connection_manager()`, `get_llm_model()` |
| shared/colab_setup.py | PASS | `COLAB_SETUP_CELL`, `COLAB_AGENT_SETUP_CELL`, `COLAB_ALIGN_SETUP_CELL` constants |

### Docs

| Item | Status | Notes |
|------|--------|-------|
| docs/course-outline.md | PASS | Exists, comprehensive curriculum overview |
| docs/setup-guide.md | PASS | Exists, covers all 3 formats + troubleshooting |
| docs/polars-cheatsheet.md | PASS | Exists, pandas-to-polars migration |
| docs/kailash-quick-reference.md | **MISSING** | Plan Section 1 specifies this file. Does not exist. |
| docs/instructor-guide.md | **MISSING** | Plan Section 1 specifies this file. Does not exist. |

### Tests

| Item | Status | Notes |
|------|--------|-------|
| tests/conftest.py | **MISSING** | Plan Section 1 specifies `tests/conftest.py`. No tests/ directory exists at all. |
| tests/test_data_loader.py | **MISSING** | Plan specifies. Does not exist. |
| tests/test_module1-6.py | **MISSING** | Plan specifies. Do not exist. `setup-guide.md` line 49 references `uv run pytest tests/test_module1.py -v` but the file does not exist. |

### Decks

| Item | Status | Notes |
|------|--------|-------|
| decks/README.md | **MISSING** | Plan Section 1 specifies `decks/README.md`. No decks/ directory exists at all. |

---

## 2. COC Artifacts

### Agents

| Item | Status | Notes |
|------|--------|-------|
| .claude/agents/education/exercise-designer.md | PASS | Exists with frontmatter, scaffolding table, exercise format template |
| .claude/agents/education/kailash-tutor.md | PASS | Exists with framework mapping table, hierarchy, communication style |
| .claude/agents/education/dataset-curator.md | PASS | Exists with quality criteria, validation checklist, data sources |
| .claude/agents/education/quiz-designer.md | PASS | Exists with 5 question types, AI-resilience section, rules |
| .claude/agents/quality/notebook-validator.md | PASS | Exists with format parity, code consistency, banned patterns |
| .claude/agents/frameworks/ | **MISSING** | Plan says "Inherited: dataflow, nexus, kaizen, ml, pact, align, mcp". CLAUDE.md line 103-105 references them. The directory does not exist. These should have been synced from kailash-coc-claude-py. |
| .claude/agents/management/ | **MISSING** | CLAUDE.md lines 115-116 reference todo-manager and gh-manager. The directory does not exist. Should be synced from kailash-coc-claude-py. |
| .claude/agents/quality/reviewer.md | **MISSING** | CLAUDE.md line 109 references "reviewer". Only notebook-validator.md exists in quality/. Should be synced from kailash-coc-claude-py. |
| .claude/agents/quality/security-reviewer.md | **MISSING** | CLAUDE.md line 111 references "security-reviewer". Not present. Should be synced from kailash-coc-claude-py. |

### Rules

| Item | Status | Notes |
|------|--------|-------|
| .claude/rules/exercise-standards.md | PASS | Exists with full scaffolding table and MUST NOT rules |
| .claude/rules/three-format.md | PASS | Exists with format matrix, data loading pattern, MUST NOT rules |
| .claude/rules/domain-integrity.md | PASS | Exists with AI-resilience requirements, quiz-to-module alignment |

### Commands

| Item | Status | Notes |
|------|--------|-------|
| .claude/commands/build-module.md | **MISSING** | Plan Section 6 specifies. CLAUDE.md line 82 references `/build-module`. File does not exist. No .claude/commands/ directory exists. |
| .claude/commands/build-exercise.md | **MISSING** | Plan Section 6 specifies. CLAUDE.md line 83 references `/build-exercise`. File does not exist. |
| .claude/commands/validate-notebooks.md | **MISSING** | Plan Section 6 specifies. CLAUDE.md line 84 references `/validate-notebooks`. File does not exist. |

### Skills

| Item | Status | Notes |
|------|--------|-------|
| .claude/skills/ (9 directories) | **MISSING** | Plan Section 1 specifies `01-core-sdk/` through `07-kailash-pact/` plus `08-exercise-patterns/` and `09-colab-patterns/`. No .claude/skills/ directory exists. Inherited skills not synced from kailash-coc-claude-py; custom skills not created. |

### Settings and Hooks

| Item | Status | Notes |
|------|--------|-------|
| .claude/settings.json | **MISSING** | Plan Section 1 specifies. Does not exist. |
| .claude/scripts/hooks/ | **MISSING** | Plan Section 1 specifies "Inherited hooks + notebook validator". Does not exist. |

---

## 3. Workspace Briefs

| Item | Status | Notes |
|------|--------|-------|
| workspaces/curriculum/briefs/course-brief.md | PASS | Exists, 404 lines, comprehensive |
| workspaces/ascent01/briefs/module-brief.md | PASS | Exists |
| workspaces/ascent02/briefs/module-brief.md | PASS | Exists |
| workspaces/ascent03/briefs/module-brief.md | PASS | Exists |
| workspaces/ascent04/briefs/module-brief.md | PASS | Exists |
| workspaces/ascent05/briefs/module-brief.md | PASS | Exists |
| workspaces/ascent06/briefs/module-brief.md | PASS | Exists |
| workspaces/assessment/briefs/module-brief.md | PASS | Exists |
| workspaces/datasets/briefs/dataset-brief.md | PASS | Exists with per-module dataset inventory, quality criteria |
| workspaces/decks/briefs/deck-brief.md | PASS | Exists with Reveal.js spec, color palette, slide structure |
| workspaces/_template/ | **MISSING** | Plan Section 7 specifies `workspaces/_template/`. Does not exist. |
| Workspace phase dirs (01-analysis/, 02-plans/, etc.) | **MISSING** | Plan Section 7 specifies phase directories per workspace. Only `briefs/` subdirectories exist; no `01-analysis/`, `02-plans/`, `03-user-flows/`, `04-validate/`, or `todos/` directories. |

---

## 4. Content Consistency

### 4.1 Module Titles — FAIL

Module titles are inconsistent across documents. The original plan used different titles than what was built, and even within the built files, titles vary.

| Module | CLAUDE.md (line 52) | README.md (lines 15-20) | course-outline.md (lines 27-32) | Module README | Module Brief | course-brief.md |
|--------|-------|---------|----------------|---------------|-------------|----------------|
| M1 | "Statistics, Probability & Data Fluency" | "Statistics & Data Fluency" | "Statistics, Probability & Data Fluency" | "Foundations -- Statistics, Probability & Data Fluency" | same as README | same as outline |
| M5 | "LLMs, AI Agents & RAG Systems" | "LLMs, AI Agents & RAG" | "LLMs, AI Agents & RAG Systems" | "LLMs, AI Agents & RAG Systems" | **"Unsupervised ML, Deep Learning & AI Agents"** | "LLMs, AI Agents & RAG Systems" |
| M6 | "Alignment, Governance, RL & Deployment" | "Alignment, Governance & Deployment" | "Alignment, Governance, RL & Deployment" | "Alignment, Governance, RL & **Production** Deployment" | "LLMs, Fine-Tuning, Governance & Deployment" | "Alignment, Governance, RL & **Production** Deployment" |

**Specific findings:**

- **M5 workspace brief** (`/Users/esperie/repos/training/ascent/workspaces/ascent05/briefs/module-brief.md`, line 1): Title is "Module 5: Unsupervised ML, Deep Learning & AI Agents" which matches the **original plan title**, not the revised built title "LLMs, AI Agents & RAG Systems" used in CLAUDE.md line 56, course-outline.md line 31, and module README line 1.

- **M6 workspace brief** (`/Users/esperie/repos/training/ascent/workspaces/ascent06/briefs/module-brief.md`, line 1): Title is "Module 6: LLMs, Fine-Tuning, Governance & Deployment" which matches the **original plan title**, not the revised title "Alignment, Governance, RL & Deployment" used in CLAUDE.md line 57.

- **M6 module README** (`/Users/esperie/repos/training/ascent/modules/ascent06/README.md`, line 1): Uses "Alignment, Governance, RL & Production Deployment" (with "Production") which differs from CLAUDE.md's "Alignment, Governance, RL & Deployment" (without "Production").

- **M1 README.md** (`/Users/esperie/repos/training/ascent/README.md`, line 15): Uses truncated "Statistics & Data Fluency" missing "Probability" which appears in CLAUDE.md and course-outline.md.

- **M6 README.md** (`/Users/esperie/repos/training/ascent/README.md`, line 20): Uses "Alignment, Governance & Deployment" missing "RL" which appears in CLAUDE.md, course-outline.md, and module README.

### 4.2 Scaffolding Percentages — PASS

Consistent across all documents that mention them.

| Module | CLAUDE.md | exercise-standards.md | exercise-designer.md | Module Brief | course-brief.md |
|--------|-----------|----------------------|---------------------|-------------|----------------|
| M1 | 70% | ~70% | ~70% | 70% | 70% |
| M2 | 60% | ~60% | ~60% | 60% | 60% |
| M3 | 50% | ~50% | ~50% | 50% | 50% |
| M4 | 40% | ~40% | ~40% | 40% | 40% |
| M5 | 30% | ~30% | ~30% | 30% | 30% |
| M6 | 20% | ~20% | ~20% | 20% | 20% |

### 4.3 Drive Folder ID — PASS

Consistent `16c3RkGmiwMWbjD7cJKbJx-JRZlgmQdws` across:
- `/Users/esperie/repos/training/ascent/shared/data_loader.py` line 16
- `/Users/esperie/repos/training/ascent/.claude/agents/education/dataset-curator.md` line 27
- `/Users/esperie/repos/training/ascent/workspaces/datasets/briefs/dataset-brief.md` line 4

### 4.4 Kailash Framework-to-Module Mapping — FAIL

The curriculum was significantly reorganized from the plan. This is a deliberate design decision (moving content for better pedagogy), but it creates inconsistencies with the M5 and M6 workspace briefs which still reference the plan's original mapping.

**Plan mapping vs Built mapping:**

| Module | Plan Title / Content | Built Title / Content |
|--------|---------------------|----------------------|
| M1 | "Python, Polars & Visualization" -- kailash-ml (3 engines) | "Statistics, Probability & Data Fluency" -- kailash-ml (same 3 engines) |
| M2 | "Statistical Foundations & Feature Engineering" -- kailash-ml (+3 engines) | "Feature Engineering & Experiment Design" -- kailash-ml (same +3) |
| M3 | "Inferential Statistics & Workflow Orchestration" -- Core SDK, DataFlow **only** | "Supervised ML -- Theory to Production" -- Core SDK, DataFlow **+ TrainingPipeline, HyperparameterSearch, ModelRegistry** |
| M4 | "Supervised ML & Production Lifecycle" -- 6 engines + Nexus | "Unsupervised ML, NLP & Deep Learning" -- AutoMLEngine, EnsembleEngine, DriftMonitor, InferenceServer + Nexus |
| M5 | "Unsupervised ML, Deep Learning & AI Agents" -- Kaizen + ML agents | "LLMs, AI Agents & RAG Systems" -- Kaizen + ML agents |
| M6 | "LLMs, Fine-Tuning, Governance & Deployment" -- Align + PACT + RL | "Alignment, Governance, RL & Deployment" -- Align + PACT + RL |

The M5 workspace brief (`/Users/esperie/repos/training/ascent/workspaces/ascent05/briefs/module-brief.md`) retains the plan's original title and description ("Unsupervised ML, Deep Learning & AI Agents") while the module README uses the revised title. Its exercises reference "Module 4" correctly for the revised structure but the title/overview are stale.

### 4.5 Assessment Weights — FAIL

Two documents specify assessment weights with different values.

**course-brief.md** (`/Users/esperie/repos/training/ascent/workspaces/curriculum/briefs/course-brief.md`, lines 287-292):
| Component | Weight |
|-----------|--------|
| Module Quizzes (6) | 20% |
| Individual Portfolio | 35% |
| Team Capstone | 35% |
| Peer Review | 10% |

**assessment module-brief.md** (`/Users/esperie/repos/training/ascent/workspaces/assessment/briefs/module-brief.md`, lines 5-9):
| Component | Weight |
|-----------|--------|
| Module Quizzes (6) | 30% |
| Individual Assignment | 30% |
| Group Project | 40% |

Discrepancies: Quizzes (20% vs 30%), Individual (35% vs 30%), Group (35% vs 40%), Peer Review (10% in course-brief, absent in assessment brief). These cannot both be correct.

Additionally, the quiz question count differs: course-brief line 289 says "15 questions each" while assessment brief line 13 says "10-15 questions".

### 4.6 Exercise Count Consistency — FAIL

Plan Section 3 specifies 5 exercises for M1, 5 for M2, 5 for M3, 6 for M4, 6 for M5, 6 for M6.

Built files show:
- M1: 5 exercises (module README, brief) -- matches plan
- M2: 5 exercises (module README, brief) -- matches plan
- M3: **6 exercises** (module README line 11, brief line 28) -- plan says 5, built as 6
- M4: 6 exercises (module README, brief) -- matches plan
- M5: 6 exercises (module README, brief) -- matches plan
- M6: 6 exercises (module README, brief) -- matches plan

M3 was expanded from 5 to 6 exercises when TrainingPipeline/HyperparameterSearch/ModelRegistry were moved into it. This is consistent within the built files but differs from the plan.

---

## 5. Dependencies

| Item | Status | Notes |
|------|--------|-------|
| kailash>=1.0 | PASS | Present in pyproject.toml line 30 |
| kailash-ml>=0.4.0 | PASS | line 31 (plan Section 11: >=0.4.0) |
| kailash-dataflow>=1.7.0 | PASS | line 32 (plan Section 11: >=1.7.0) |
| kailash-nexus>=1.9.0 | PASS | line 33 (plan Section 11: >=1.9.0) |
| kailash-kaizen>=2.5.0 | PASS | line 34 (plan Section 11: >=2.5.0) |
| kaizen-agents>=0.6.0 | PASS | line 35 (plan Section 11: >=0.6.0) |
| kailash-pact>=0.7.2 | PASS | line 36 (plan Section 11: >=0.7.2) |
| kailash-align>=0.2.1 | PASS | line 37 (plan Section 11: >=0.2.1) |
| polars>=1.0 | PASS | line 39 |
| plotly>=5.18 | PASS | line 41 |
| gdown>=5.0 | PASS | line 43 |
| python-dotenv>=1.0 | PASS | line 45 (plan says kailash-ml[full] in deps; built uses base kailash-ml with [full] as optional extra -- acceptable) |
| jupyter/nbformat | **FAIL** | Plan Section 6 has these as main dependencies; built pyproject.toml has them as optional `[notebooks]` extra (lines 50-54). This is an improvement but diverges from plan. Low severity. |

---

## 6. Cross-Reference Audit: Internal Consistency Issues

### 6.1 CLAUDE.md references non-existent commands

`/Users/esperie/repos/training/ascent/CLAUDE.md` lines 82-84 reference three custom commands (`/build-module`, `/build-exercise`, `/validate-notebooks`) but the `.claude/commands/` directory does not exist. Any invocation of these commands would fail.

### 6.2 CLAUDE.md references non-existent inherited agents

`/Users/esperie/repos/training/ascent/CLAUDE.md` lines 103-116 reference inherited agents (frameworks/, management/, reviewer, security-reviewer) that were never synced from kailash-coc-claude-py.

### 6.3 setup-guide.md references non-existent test files

`/Users/esperie/repos/training/ascent/docs/setup-guide.md` line 49: `uv run pytest tests/test_module1.py -v` references a file that does not exist.

### 6.4 M5 brief title/framework mismatch with module README

The M5 workspace brief uses the plan's original title ("Unsupervised ML, Deep Learning & AI Agents") and original framework focus. The M5 module README uses the revised title ("LLMs, AI Agents & RAG Systems") with revised content focusing on agents and RAG rather than unsupervised/deep learning. The brief's exercise descriptions (5.1-5.6) are internally consistent with the revised curriculum, but the overview text at lines 1-8 is stale.

### 6.5 M6 brief title mismatch

The M6 workspace brief title ("LLMs, Fine-Tuning, Governance & Deployment") differs from the M6 module README title ("Alignment, Governance, RL & Production Deployment") and CLAUDE.md ("Alignment, Governance, RL & Deployment").

---

## 7. Severity Classification

### Critical (must fix before proceeding to Phase 1)

1. **Module content directories missing** (local/, notebooks/, colab/, solutions/, quiz/) -- These are the primary deliverable of the entire repository. Without them, no exercises can be built.
2. **tests/ directory missing** -- No test infrastructure exists despite being specified in plan and referenced by setup-guide.md.
3. **Assessment weight inconsistency** -- Two authoritative documents contradict each other on grading weights.

### Major (should fix in Phase 0 completion)

4. **.claude/commands/ missing** -- CLAUDE.md promises 3 custom commands that don't exist. COC workflow is broken.
5. **.claude/skills/ missing** -- No skills directory exists. Framework specialist agents cannot access reference material.
6. **.claude/settings.json missing** -- COC configuration incomplete.
7. **.claude/scripts/hooks/ missing** -- Hooks not synced from kailash-coc-claude-py.
8. **Inherited agents not synced** -- frameworks/, management/, reviewer, security-reviewer all missing.
9. **Module title inconsistencies** -- 3 modules have title mismatches across documents.
10. **CHANGELOG.md missing** -- Required root file per plan.

### Significant (should fix before Phase 1 content authoring)

11. **docs/kailash-quick-reference.md missing** -- Student-facing SDK cheatsheet.
12. **docs/instructor-guide.md missing** -- Teaching notes for instructors.
13. **decks/ directory missing** -- Slide content directory.
14. **modules/assessment/ missing** -- Capstone assessment directory structure.
15. **workspaces/_template/ missing** -- Template workspace for new workspaces.
16. **Workspace phase directories missing** -- Only briefs/ exists per workspace; no 01-analysis/, 02-plans/, etc.

### Minor (acceptable for now)

17. **jupyter/nbformat as optional dep** -- Better engineering than plan spec but technically divergent.
18. **M3 expanded to 6 exercises** -- Consistent within built files, but differs from plan.

---

## 8. Remediation Checklist

```
Phase 0 Completion (in order of priority):

[ ] 1.  Create module content directories for all 6 modules:
        modules/ascent{1-6}/{local,notebooks,colab,solutions,quiz}/
[ ] 2.  Create modules/assessment/{individual,group}/
[ ] 3.  Create tests/{conftest.py,test_data_loader.py}
[ ] 4.  Create CHANGELOG.md at repo root
[ ] 5.  Create decks/README.md
[ ] 6.  Sync inherited COC artifacts from kailash-coc-claude-py:
        - .claude/agents/frameworks/ (7 specialists)
        - .claude/agents/management/ (todo-manager, gh-manager)
        - .claude/agents/quality/reviewer.md
        - .claude/agents/quality/security-reviewer.md
        - .claude/skills/ (01-core-sdk through 07-kailash-pact)
        - .claude/scripts/hooks/
        - .claude/settings.json
[ ] 7.  Create custom COC artifacts:
        - .claude/commands/build-module.md
        - .claude/commands/build-exercise.md
        - .claude/commands/validate-notebooks.md
        - .claude/skills/08-exercise-patterns/
        - .claude/skills/09-colab-patterns/
[ ] 8.  Create missing docs:
        - docs/kailash-quick-reference.md
        - docs/instructor-guide.md
[ ] 9.  Create workspace infrastructure:
        - workspaces/_template/
        - Phase directories (01-analysis/, 02-plans/, etc.) per workspace
[ ] 10. Fix module title inconsistencies — pick canonical title per module,
        update: CLAUDE.md, README.md, course-outline.md, module READMEs,
        module briefs, course-brief.md
[ ] 11. Fix assessment weight inconsistency — reconcile course-brief.md
        and assessment/briefs/module-brief.md
```
