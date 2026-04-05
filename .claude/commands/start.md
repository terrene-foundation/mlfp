# ASCENT — Getting Started

Welcome to the ASCENT training repository. This is both a professional ML course AND the definitive training material for the Kailash Python SDK.

## What This Repo Is

**Dual purpose**:
1. **Professional ML curriculum** — 6 modules covering statistics through governed AI deployment, benchmarked at Stanford CS229 depth
2. **Kailash SDK training** — every exercise uses Kailash engines, every concept bridges theory to SDK

**Three delivery formats**: Local Python, Jupyter notebooks, Google Colab

## Repository Layout

```
modules/ascent01-6/        — Exercise content (local/, notebooks/, colab/, solutions/, quiz/)
shared/                 — Data loader, Kailash helpers, Colab setup
docs/                   — Course outline, setup guide, polars cheatsheet
decks/                  — Reveal.js presentation slides
workspaces/             — COC workspaces for course development
tests/                  — Exercise validation tests
```

## Workflow Commands (for instructors and course developers)

### COC Phase Commands (standard workflow)

| Command | Phase | Purpose |
|---------|-------|---------|
| `/analyze` | 01 | Research: examine SDK capabilities, dataset suitability, pedagogical approach |
| `/todos` | 02 | Plan: create task list for module implementation, stops for approval |
| `/implement` | 03 | Build: write solutions, generate exercises, create notebooks |
| `/redteam` | 04 | Validate: test all exercises, verify 3-format consistency, SDK correctness |
| `/codify` | 05 | Capture: update agents/skills with lessons learned |
| `/release` | — | Publish: version, tag, prepare for distribution |

### Course-Specific Commands

| Command | Purpose |
|---------|---------|
| `/build-module` | Scaffold a complete module: solutions → exercises → 3 formats → quiz |
| `/build-exercise` | Create a single exercise across local/Jupyter/Colab from a solution |
| `/validate-notebooks` | Cross-validate all 3 formats for consistency |
| `/start` | This guide — explains the repo and all commands |

### Utility Commands

| Command | Purpose |
|---------|---------|
| `/ws` | Show workspace status dashboard (read-only) |
| `/wrapup` | Write session notes before ending |
| `/cc-audit` | Audit COC artifacts for quality and completeness |

## How to Build a Module (instructor workflow)

```
1. Choose workspace:     cd to workspaces/ascentN/
2. Read the brief:       workspaces/ascentN/briefs/module-brief.md
3. Write solutions:      modules/ascentN/solutions/ (complete, runnable code)
4. Generate exercises:   /build-exercise (strips solutions → fill-in-blank)
5. Create 3 formats:     /build-module (local .py + Jupyter .ipynb + Colab .ipynb)
6. Write quiz:           modules/ascentN/quiz/ (see quiz-designer agent)
7. Build deck:           decks/ascentN/ (Reveal.js slides from deck brief)
8. Validate:             /validate-notebooks (cross-check all formats)
9. Red team:             /redteam (SDK correctness, exercise quality)
```

## How to Add/Update Content (developer workflow)

```
1. /analyze              — Research what needs changing
2. /todos                — Plan the changes (get approval)
3. /implement            — Make the changes
4. /redteam              — Validate (SDK paths, consistency, quality)
5. /codify               — Update agents/skills if patterns changed
```

## Key Agents (available to all commands)

### Education Specialists
- **exercise-designer** — Generates exercises from solutions with progressive scaffolding
- **kailash-tutor** — Maps traditional ML → Kailash patterns
- **dataset-curator** — Validates dataset quality and availability
- **quiz-designer** — Creates Kailash-pattern assessment questions

### Quality
- **notebook-validator** — Cross-validates 3 delivery formats
- **reviewer** — Code review and content quality
- **security-reviewer** — No hardcoded keys, no exposed secrets

### Framework Specialists (inherited from COC)
- **ml-specialist**, **dataflow-specialist**, **nexus-specialist**
- **kaizen-specialist**, **pact-specialist**, **align-specialist**

## Key Rules

1. **Framework-first** — All ML content uses Kailash. Never raw sklearn/pandas.
2. **Polars-native** — No pandas anywhere. kailash-ml is polars-native.
3. **Three-format** — Every exercise in local + Jupyter + Colab.
4. **Solution-first** — Write solutions first, then strip to exercises.
5. **Progressive disclosure** — M1: 70% scaffolding → M6: 20%.
6. **Kailash from Lesson 1** — First exercise uses a Kailash engine.
7. **Governance throughout** — Trust, audit, governance concepts woven into every module, not just M6.

## Environment Setup

```bash
uv venv && uv sync          # Local development
cp .env.example .env         # Configure API keys (M5-M6 need LLM keys)
uv run pytest tests/ -v      # Validate exercises
```
