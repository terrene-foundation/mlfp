# RT6: Naming, Branding, and Cross-Reference Consistency

**Date**: 2026-04-13
**Scope**: All MLFP course files — 96 notebooks (.ipynb), 96 exercises (.py), 48 textbook.html, 6 deck.html, 6 specs, scripts, CSS
**Method**: Grep scan for forbidden branding (ASCENT/PCML/SMU/SSG), title comparison across spec/textbook/exercise, navigation link validation

---

## Executive Summary

| Issue Category | Critical | Important | Minor | Total |
|----------------|----------|-----------|-------|-------|
| Wrong ASCENT prefix in notebooks | 44 files | — | — | 44 |
| Wrong exercise numbers in notebooks | 16 files | — | — | 16 |
| CSS variable naming | — | 1 file (+ 5 consumers) | — | 6 |
| Script branding | — | 1 file | — | 1 |
| Code output branding | 2 files | — | — | 2 |
| Title mismatches (spec vs textbook vs exercise) | — | 31 lessons | — | 31 |
| Navigation link issues | 2 cross-module | 48 label mismatches | — | 50 |
| PCML in specs | — | — | acceptable | — |
| **Total** | **62** | **87** | **0** | **149** |

**Bottom line**: 60 notebook files have wrong titles — either wrong program prefix ("ASCENT" instead of "MLFP") or wrong exercise numbers. This is the single largest branding issue. All `.py` files (local + solutions) are clean.

---

## 1. ASCENT Branding in Notebooks (CRITICAL — 60 files)

Notebooks in M1, M3, M4, and M5 retain ASCENT program prefixes from the source ASCENT exercises they were derived from. Each issue appears in both the `notebooks/` and `colab/` copy.

### M1: All 8 notebooks wrong (16 files)

All say `# ASCENT1 — Exercise N` instead of `# MLFP01 — Exercise N`. Exercise numbers are correct.

| File | Current Title | Correct Title |
|------|--------------|---------------|
| ex_1.ipynb | `# ASCENT1 — Exercise 1: Your First Data Exploration` | `# MLFP01 — Exercise 1: ...` |
| ex_2.ipynb | `# ASCENT1 — Exercise 2: Filtering and Transforming Data` | `# MLFP01 — Exercise 2: ...` |
| ex_3.ipynb | `# ASCENT1 — Exercise 3: Functions and Aggregation` | `# MLFP01 — Exercise 3: ...` |
| ex_4.ipynb | `# ASCENT1 — Exercise 4: Joins and Multi-Table Data` | `# MLFP01 — Exercise 4: ...` |
| ex_5.ipynb | `# ASCENT1 — Exercise 5: Window Functions and Trends` | `# MLFP01 — Exercise 5: ...` |
| ex_6.ipynb | `# ASCENT1 — Exercise 6: Data Visualization` | `# MLFP01 — Exercise 6: ...` |
| ex_7.ipynb | `# ASCENT1 — Exercise 7: Automated Data Profiling` | `# MLFP01 — Exercise 7: ...` |
| ex_8.ipynb | `# ASCENT1 — Exercise 8: Data Cleaning and End-to-End Project` | `# MLFP01 — Exercise 8: ...` |

### M3: 7 of 8 notebooks wrong (14 files)

Wrong ASCENT prefix AND wrong exercise numbers (from ASCENT source mapping).

| File | Current Title | Correct Title |
|------|--------------|---------------|
| ex_1.ipynb | `# ASCENT2 — Exercise 7: Feature Engineering` | `# MLFP03 — Exercise 1: Feature Engineering` |
| ex_2.ipynb | `# ASCENT3 — Exercise 1: Bias-Variance and Regularisation` | `# MLFP03 — Exercise 2: Bias-Variance and Regularisation` |
| ex_3.ipynb | `# MLFP03 — Exercise 3: The Complete Supervised Model Zoo` | CORRECT |
| ex_4.ipynb | `# ASCENT3 — Exercise 2: Gradient Boosting Deep Dive` | `# MLFP03 — Exercise 4: Gradient Boosting Deep Dive` |
| ex_5.ipynb | `# ASCENT3 — Exercise 3: Class Imbalance and Calibration` | `# MLFP03 — Exercise 5: Class Imbalance and Calibration` |
| ex_6.ipynb | `# ASCENT3 — Exercise 4: SHAP, LIME, and Fairness` | `# MLFP03 — Exercise 6: SHAP, LIME, and Fairness` |
| ex_7.ipynb | `# ASCENT3 — Exercise 5: Workflow Orchestration and Custom Nodes` | `# MLFP03 — Exercise 7: Workflow Orchestration and Custom Nodes` |
| ex_8.ipynb | `# ASCENT3 — Exercise 8: Production Pipeline Project` | `# MLFP03 — Exercise 8: Production Pipeline Project` |

### M4: 6 of 8 notebooks wrong (12 files)

| File | Current Title | Correct Title |
|------|--------------|---------------|
| ex_1.ipynb | `# ASCENT4 — Exercise 1: Clustering Comparison` | `# MLFP04 — Exercise 1: ...` |
| ex_2.ipynb | `# ASCENT4 — Exercise 2: UMAP + Anomaly Detection` | `# MLFP04 — Exercise 2: ...` |
| ex_3.ipynb | `# ASCENT4 — Exercise 3: Topic Modeling with BERTopic` | `# MLFP04 — Exercise 3: ...` |
| ex_4.ipynb | `# ASCENT4 — Exercise 4: DriftMonitor` | `# MLFP04 — Exercise 4: ...` |
| ex_5.ipynb | `# MLFP04 — Exercise 5: Association Rules` | CORRECT |
| ex_6.ipynb | `# ASCENT4 — Exercise 5: Deep Learning with ONNX Export` | `# MLFP04 — Exercise 6: ...` (wrong ex#) |
| ex_7.ipynb | `# MLFP04 — Exercise 7: Recommender Systems` | CORRECT |
| ex_8.ipynb | `# ASCENT4 — Exercise 7: Deep Learning Foundations` | `# MLFP04 — Exercise 8: ...` (wrong ex#) |

### M5: 4 of 8 notebooks wrong (8 files)

| File | Current Title | Correct Title |
|------|--------------|---------------|
| ex_1.ipynb | `# MLFP05 — Exercise 1: Autoencoders` | CORRECT |
| ex_2.ipynb | `# MLFP05 — Exercise 7: CNNs for Image Classification` | `# MLFP05 — Exercise 2: ...` (wrong ex#) |
| ex_3.ipynb | `# MLFP05 — Exercise 4: Sequence Models` | `# MLFP05 — Exercise 3: ...` (wrong ex#) |
| ex_4.ipynb | `# MLFP05 — Exercise 6: Transformer Architecture` | `# MLFP05 — Exercise 4: ...` (wrong ex#) |
| ex_5.ipynb | `# MLFP05 — Exercise 5: Generative Models` | CORRECT |
| ex_6.ipynb | `# MLFP05 — Exercise 6: Graph Neural Networks` | CORRECT |
| ex_7.ipynb | `# MLFP05 — Exercise 7: Transfer Learning` | CORRECT |
| ex_8.ipynb | `# ASCENT6 — Exercise 3: PACT Governance Setup` | `# MLFP05 — Exercise 8: Reinforcement Learning` (WRONG MODULE + WRONG TOPIC) |

**M5 ex_8 is the worst single case**: notebook title says it's an ASCENT6 exercise about PACT Governance, but the file is actually MLFP05's reinforcement learning exercise.

### M6: 5 of 8 notebooks wrong (10 files)

Correct MLFP06 prefix but wrong exercise numbers.

| File | Current Title | Correct Title |
|------|--------------|---------------|
| ex_1.ipynb | `# MLFP06 — Exercise 2: Prompt Engineering` | `# MLFP06 — Exercise 1: ...` |
| ex_2.ipynb | `# MLFP06 — Exercise 1: LoRA Fine-Tuning` | `# MLFP06 — Exercise 2: ...` |
| ex_3.ipynb | `# MLFP06 — Exercise 2: DPO Preference Alignment` | `# MLFP06 — Exercise 3: ...` |
| ex_4.ipynb | `# MLFP06 — Exercise 3: RAG Fundamentals` | `# MLFP06 — Exercise 4: ...` |
| ex_5.ipynb | `# MLFP06 — Exercise 5: Building Agents` | CORRECT |
| ex_6.ipynb | `# MLFP06 — Exercise 6: Multi-Agent Orchestration` | CORRECT |
| ex_7.ipynb | `# MLFP06 — Exercise 6: AI Governance with PACT` | `# MLFP06 — Exercise 7: ...` |
| ex_8.ipynb | `# MLFP06 — Exercise 8: Capstone` | CORRECT |

### M2: All correct (0 issues)

All M2 notebooks use `# MLFP02 — Exercise N: Title` with correct numbers.

---

## 2. ASCENT in CSS Theme (IMPORTANT — 6 files)

**File**: `decks/assets/css/theme.css`
**Consumers**: All 6 `deck.html` files (mlfp01–mlfp06)

CSS custom properties use `--ascent-*` naming:
```css
--ascent-primary: #0D9488;
--ascent-depth: #4F46E5;
--ascent-bg: #FFFFFF;
--ascent-bg-alt: #F8FAFC;
--ascent-text: #1E293B;
--ascent-code-text: #334155;
--ascent-code-bg: #F8FAFC;
--ascent-alert: #F59E0B;
--ascent-error: #F43F5E;
--ascent-success: #10B981;
```

**Impact**: Not user-visible (CSS internals), but breaks naming consistency. Should be `--mlfp-*`.

**Scope**: theme.css defines ~10 variables; all 6 deck.html files reference them extensively.

---

## 3. ASCENT in Scripts (IMPORTANT — 1 file)

**File**: `scripts/deck-audit.mjs`

```javascript
// Line 7:  e.g.: node scripts/deck-audit.mjs ascent01
// Line 162: const targetModule = process.argv[2] || 'ascent01';
// Line 168: for (const mod of ['ascent01', 'ascent02', 'ascent03', 'ascent04', 'ascent05', 'ascent06']) {
```

Module names should be `mlfp01`–`mlfp06`.

---

## 4. ASCENT in Code Output (CRITICAL — 2 files)

**Files**: `modules/mlfp06/notebooks/ex_8.ipynb` and `modules/mlfp06/colab/ex_8.ipynb`

```python
print(f"  System: ASCENT Capstone Governed ML System")
```

Should say "MLFP Capstone Governed ML System".

---

## 5. PCML References (ACCEPTABLE)

PCML appears in specs as R5 provenance documentation (e.g., "R5 Source: PCML6-3", "R5 PCML_DIS_R5"). These are historical source references in internal spec files and are appropriate to keep.

**Locations**: `specs/module-{2-6}.md`, `specs/r5-mapping.md`, `specs/_index.md`, `specs/ascent-additions.md`

---

## 6. SMU Reference (ACCEPTABLE)

One occurrence in `scripts/generate_datasets.py:1385` — factual statement about Singapore's six autonomous universities. Not a course/institution branding reference.

---

## 7. SSG References (FALSE POSITIVE)

All SSG matches are in `.test_venv/` third-party packages (networkx, packaging). Not our code.

---

## 8. Title Mismatches: Spec vs Textbook vs Exercise (IMPORTANT — 31 lessons)

31 of 48 lessons have title inconsistencies across the three canonical sources.

### Mismatch Patterns

**Pattern A — Truncated exercise titles** (most common):
- Spec: "Feature Engineering, ML Pipeline, and Feature Selection" → Exercise: "Feature Engineering"
- Spec: "Model Evaluation, Imbalance, and Calibration" → Exercise: "Class Imbalance and Calibration"
- Spec: "Workflow Orchestration, Model Registry, and Hyperparameter Search" → Exercise: "Workflow Orchestration and Custom Nodes"

**Pattern B — Spelling inconsistency**:
- "Visualisation" (spec, UK) vs "Visualization" (exercise, US)

**Pattern C — Punctuation differences**:
- "Capstone — Statistical Analysis Project" (spec) vs "Capstone: Statistical Analysis Project" (textbook)
- "NLP — Text to Topics" vs "NLP: Text to Topics"

**Pattern D — Restructured scope**:
- Spec: "Interpretability and Fairness" → Exercise: "SHAP, LIME, and Fairness"
- Spec: "DL Foundations — Neural Networks, Backpropagation, and the Training Toolkit" → Exercise: "Deep Learning Foundations"
- Textbook: "Recommender Systems: THE PIVOT" (contains informal emphasis marker)

### Lessons with All Three Sources Matching (17)
1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.1, 2.4, 2.5, 2.6, 2.7, 3.3, 3.4, 4.1, 4.3, 5.2, 5.5

---

## 9. Navigation Link Issues (50 total)

### Cross-Module Errors (CRITICAL — 2)
- **mlfp04/lesson 08**: "Next" links to mlfp05/lesson 01 instead of module index
- **mlfp05/lesson 08**: "Next" links to mlfp06/lesson 01 instead of module index

### Label Mismatches (IMPORTANT — 48)
All 48 textbook.html prev/next navigation labels show lesson titles (e.g., "1.2 Filtering and Transforming Data"), but the target file's H2 heading is a content heading (e.g., "Why This Matters"), not the lesson title. This is a structural mismatch — the H2 in each textbook is the first section heading, not the lesson title.

**Root cause**: Navigation was generated from spec lesson titles, but textbook H2 headings are content headings. Either H2 should become the lesson title, or navigation should reference `<title>` tags instead.

---

## 10. ASCENT References in Workspace/Brief Files (LOW — context docs)

ASCENT appears in workspace briefs and curriculum documents as source program references. These are internal planning artifacts and acceptable as-is:
- `workspaces/curriculum/briefs/course-brief.md` — references ASCENT as source
- `workspaces/curriculum/briefs/mlfp-curriculum-v1.md` — maps MLFP to ASCENT
- `workspaces/ascent01/briefs/module-brief.md` — ASCENT module planning
- `workspaces/datasets/briefs/dataset-brief.md` — dataset sources
- `workspaces/assessment/briefs/module-brief.md` — quiz structure
- `workspaces/decks/briefs/deck-brief.md` — deck authoring instructions

---

## Prioritised Fix List

### Tier 1 — Critical (must fix before delivery)

| # | Issue | Files | Fix |
|---|-------|-------|-----|
| 1 | ASCENT prefix in M1 notebooks | 16 | Replace `ASCENT1` → `MLFP01` |
| 2 | ASCENT prefix + wrong ex# in M3 notebooks | 14 | Replace prefix + fix exercise numbers |
| 3 | ASCENT prefix in M4 notebooks | 12 | Replace `ASCENT4` → `MLFP04` + fix ex# in ex_6, ex_8 |
| 4 | Wrong ex# in M5 notebooks | 8 | Fix exercise numbers; fix ex_8 (says ASCENT6/PACT, should be MLFP05/RL) |
| 5 | Wrong ex# in M6 notebooks | 10 | Fix exercise numbers in ex_1–4, ex_7 |
| 6 | "ASCENT Capstone" in code output | 2 | Replace with "MLFP Capstone" |
| 7 | Cross-module nav links (4.8, 5.8) | 2 | Point "Next" to module index, not next module's 01 |

### Tier 2 — Important (should fix)

| # | Issue | Files | Fix |
|---|-------|-------|-----|
| 8 | CSS `--ascent-*` variables | 6 | Rename to `--mlfp-*` in theme.css + all deck.html |
| 9 | deck-audit.mjs module names | 1 | Replace `ascent01`–`06` with `mlfp01`–`06` |
| 10 | Title standardisation (31 lessons) | ~93 files | Decide canonical title per lesson; align all three |
| 11 | Nav label vs H2 mismatch | 48 | Structural: decide if H2 should be lesson title |
| 12 | UK/US spelling consistency | ~10 files | Standardise on one (UK per Terrene convention) |

### Tier 3 — Deferred

| # | Issue | Notes |
|---|-------|-------|
| 13 | PCML in specs | Acceptable provenance references |
| 14 | ASCENT in workspace briefs | Internal planning docs |
| 15 | "THE PIVOT" in textbook 4.7 | Informal emphasis; decide if appropriate |

---

*Audit conducted by: compliance-rt4 (reviewer agent)*
*Method: grep scan + 2 parallel subagent auditors (navigation + title comparison)*
