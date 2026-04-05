# ASCENT Curriculum Review

**Reviewer**: Quality Reviewer Agent  
**Date**: 2026-04-05  
**Scope**: Course brief, 6 module briefs, assessment brief, deck brief, dataset brief  
**Benchmark**: Georgia Tech OMSCS / Stanford CS229 depth, production-practice reality  
**SDK Validation Target**: Kailash Python SDK (`kailash-py` at `/Users/esperie/repos/loom/kailash-py/`)

---

## Summary

**Overall Status**: Issues Found

The curriculum is strong. Mathematical depth, production-reality philosophy, progressive framework introduction, and dataset difficulty all meet the Georgia Tech / Stanford CS229 benchmark. The theory-to-practice mapping through Kailash engines is well conceived and covers all 7 framework packages across the 6 modules.

However, there are structural inconsistencies between the course brief and individual module briefs, contradictory assessment weights, a significant content misalignment in Module 5, missing lecture content in several module briefs, and an engine count discrepancy. These need resolution before implementation begins.

**Critical Issues**: 4  
**High Issues**: 8  
**Medium Issues**: 7  
**Low Issues**: 5  

---

## CRITICAL (Must Fix Before Implementation)

### C1. Module 5 Brief Title and Content Mismatch Against Course Brief

**Location**: `workspaces/ascent05/briefs/module-brief.md` vs `workspaces/curriculum/briefs/course-brief.md`

The course brief defines Module 5 as **"LLMs, AI Agents & RAG Systems"** with three lecture sections:
- 5A: Transformer Architecture & LLMs (90 min) -- tokenization, pre-training objectives, scaling laws, inference optimization
- 5B: RAG Architecture & Evaluation (60 min)
- 5C: Agent Architecture & Multi-Agent Systems (60 min)

The Module 5 brief titles itself **"Unsupervised ML, Deep Learning & AI Agents"** and includes a section saying "Exercises 5.1-5.2 also cover unsupervised ML (clustering, dimensionality reduction)" -- topics that belong in Module 4.

The module brief has **no lecture topic sections at all** -- only exercises and import paths. The entire transformer deep dive, RAG evaluation theory, and agent architecture theory are absent from the module brief.

**Fix**: Rewrite the M5 module brief to match the course brief. Title should be "LLMs, AI Agents & RAG Systems". Add the three lecture sections (5A/5B/5C). Remove the "Unsupervised ML Integration" and "Deep Learning Integration" sections that reference Module 4 content. This brief appears to be from a previous curriculum revision where Modules 4 and 5 had different scope boundaries.

### C2. Assessment Weight Contradiction

**Location**: `workspaces/curriculum/briefs/course-brief.md` (lines 287-292) vs `workspaces/assessment/briefs/module-brief.md` (lines 3-9)

Course brief:
| Component | Weight |
|-----------|--------|
| Module Quizzes | 20% |
| Individual Portfolio | 35% |
| Team Capstone | 35% |
| Peer Review | 10% |

Assessment brief:
| Component | Weight |
|-----------|--------|
| Module Quizzes | 30% |
| Individual Assignment | 30% |
| Group Project | 40% |

These are fundamentally different schemes. The course brief includes a Peer Review component (10%) that the assessment brief does not mention at all. The weights differ by 10 percentage points on every component. Terminology also differs ("Individual Portfolio" vs "Individual Assignment", "Team Capstone" vs "Group Project").

**Fix**: Decide on one authoritative scheme and update both documents. The course brief's version is more detailed (includes peer review and model cards) and aligns better with the graduate-level standard. Recommend adopting the course brief version (20/35/35/10) as canonical and updating the assessment brief to match.

### C3. Exercise Ordering Mismatch Between Course Brief and Module 5 Brief

**Location**: `workspaces/curriculum/briefs/course-brief.md` M5 exercises vs `workspaces/ascent05/briefs/module-brief.md`

Course brief M5 exercise order:
1. Delegate + SimpleQAAgent
2. ChainOfThoughtAgent
3. ReActAgent with tools
4. RAGResearchAgent
5. ML Agent Pipeline
6. Multi-agent A2A

Module 5 brief exercise order:
1. Delegate + SimpleQAAgent
2. Chain-of-Thought
3. ML Agent Pipeline (exercise 3, not 5)
4. ReAct Agent (exercise 4, not 3)
5. RAG Research Agent (exercise 5, not 4)
6. Multi-Agent A2A

The reordering breaks the pedagogical progression. The course brief's order follows a sound logic: simple agent -> reasoning -> action -> retrieval -> ML-specific -> multi-agent coordination. The module brief's order jumps from reasoning to ML pipeline before covering the ReAct pattern that ML agents build on.

**Fix**: Adopt the course brief ordering. The module brief should be updated to match.

### C4. Module 5 References "Module 4" Where It Should Reference "Module 3"

**Location**: `workspaces/ascent05/briefs/module-brief.md` exercise 5.3

Exercise 5.3 says: "Compare agent recommendations to manual choices from Module 4". But supervised ML model training is Module 3, not Module 4. Module 4 covers unsupervised ML, NLP, and deep learning. Exercise 5.6 also says "Compare autonomous result to manual pipeline from Module 4" -- again, the comparison should be to Module 3's supervised ML work.

This error further confirms the M5 brief is from a previous curriculum revision where module numbering was different.

**Fix**: Update cross-references to point to the correct modules.

---

## HIGH (Should Fix Before Implementation)

### H1. Module Briefs Missing Lecture Content Structure

**Location**: `workspaces/ascent05/briefs/module-brief.md`, `workspaces/ascent06/briefs/module-brief.md`

The course brief contains detailed lecture sections (5A/5B/5C, 6A/6B/6C) with specific topic breakdowns. The module briefs for M5 and M6 have "Learning Outcomes" and "Key Import Paths" but omit the lecture topic sections entirely. Modules 1-4 briefs do include lecture topic sections, creating an inconsistency in document structure.

A module brief without lecture content prevents the deck-building agent and exercise-designer agent from operating correctly -- they need to know what theory is covered in each section to design exercises that bridge theory to practice.

**Fix**: Add lecture topic sections to M5 and M6 module briefs matching the course brief content. Follow the structure pattern of M1-M4 briefs.

### H2. Engine Count Discrepancy

**Location**: `README.md` (line 54), `CLAUDE.md` (line 68), `workspaces/curriculum/briefs/course-brief.md` (line 8)

README.md says "14 engines". CLAUDE.md says "13 engines". Actual engine files in `packages/kailash-ml/src/kailash_ml/engines/` total **13** (excluding internal helpers `_data_explorer_report.py`, `_feature_sql.py`, `_guardrails.py`, `_shared.py`):

1. AutoMLEngine
2. DataExplorer
3. DriftMonitor
4. EnsembleEngine
5. ExperimentTracker
6. FeatureEngineer
7. FeatureStore
8. HyperparameterSearch
9. InferenceServer
10. ModelRegistry
11. ModelVisualizer
12. PreprocessingPipeline
13. TrainingPipeline

**Fix**: Update README.md to say "13 engines" to match the actual codebase and CLAUDE.md.

### H3. Module 4 Engine Usage Gap

**Location**: `workspaces/ascent04/briefs/module-brief.md`

The M4 header lists "kailash-ml (AutoMLEngine, EnsembleEngine, DriftMonitor, InferenceServer)" but the lab exercises only explicitly use DriftMonitor (exercise 4) and InferenceServer (exercise 6). AutoMLEngine and EnsembleEngine appear in the header but are never used in any M4 exercise.

The course outline claims M4 introduces "+6 engines" but only 4 new engines appear in the framework header, and only 2 are used in exercises.

**Fix**: Either add exercises that demonstrate AutoMLEngine and EnsembleEngine (e.g., AutoML comparison for the clustering task, ensemble of unsupervised methods) or remove them from the M4 header and move them to M3 where TrainingPipeline and HyperparameterSearch are already covered. The most natural fit is to add AutoMLEngine to M3 (comparing manual model selection with AutoML) and EnsembleEngine to M3 (stacking/blending lab exercise).

### H4. Course Brief Capstone Assessment Discrepancies

**Location**: `workspaces/curriculum/briefs/course-brief.md` vs `workspaces/assessment/briefs/module-brief.md`

Beyond the weight issue (C2), the scope requirements differ:

Course brief Team Capstone requires: "3+ Kailash packages", "data pipeline (DataFlow), model lifecycle (ModelRegistry), deployment (Nexus), governance (PACT) OR agents (Kaizen)", "15-minute live demo + 10-minute Q&A", "governance specification".

Assessment brief Group Project requires: "3+ Kailash packages", "DataFlow AND at least one deployment channel (Nexus)", "PACT OR Kaizen", "10-minute demo + 5-minute Q&A".

The demo/Q&A times differ (25 min vs 15 min). The course brief is more prescriptive about required components.

**Fix**: Align on one specification. The course brief version is more appropriate for the graduate-level standard.

### H5. Individual Assignment Grading Rubric Discrepancy

**Location**: `workspaces/curriculum/briefs/course-brief.md` vs `workspaces/assessment/briefs/module-brief.md`

Course brief: "statistical rigor (25%), Kailash pattern mastery (25%), production readiness (25%), documentation quality (25%)"

Assessment brief: "Kailash pattern correctness (30%), production readiness (30%), documentation quality (20%), code quality (20%)"

The course brief emphasizes statistical rigor as a dimension; the assessment brief replaces it with code quality and shifts weights. The course brief version requires model card, calibration analysis, and drift monitoring -- the assessment brief does not.

**Fix**: Reconcile rubrics. The course brief version is more aligned with the stated "theory meets production" philosophy.

### H6. Missing Quiz Topics for Modules 5 and 6

**Location**: `workspaces/ascent05/briefs/module-brief.md`, `workspaces/ascent06/briefs/module-brief.md`

Module briefs for M1-M4 each include a "Quiz Topics" section listing specific question themes. M5 and M6 briefs omit this section entirely. Without quiz topic specifications, the quiz-designer agent cannot generate appropriately scoped assessment questions.

**Fix**: Add "Quiz Topics" sections to M5 and M6 briefs. Example M5 topics: Signature InputField/OutputField contracts, ReAct vs CoT agent selection, RAG evaluation metrics (faithfulness, relevance), multi-agent A2A coordination patterns, ML agent pipeline design. Example M6 topics: LoRA rank selection, DPO vs SFT trade-offs, GovernanceContext frozen semantics, monotonic tightening, Bellman equation interpretation, PPO advantage estimation.

### H7. Missing Deck Opening Cases for Modules 5 and 6 in Module Briefs

**Location**: `workspaces/ascent05/briefs/module-brief.md`, `workspaces/ascent06/briefs/module-brief.md`

M1-M4 module briefs each have a "Deck Opening Case" section. M5 and M6 module briefs omit this. The opening cases ARE defined in the deck brief and course brief (M5: BloombergGPT, M6: EU AI Act enforcement) but should also appear in the module briefs for consistency and to ensure the exercise-designer and deck-building agents have a single source of truth per module.

**Fix**: Add "Deck Opening Case" sections to M5 and M6 briefs.

### H8. Dataset Brief Module Numbering Stale

**Location**: `workspaces/datasets/briefs/dataset-brief.md`

The dataset brief uses old module titles:
- ascent03 is called "Workflows & Inferential Stats" (should be "Supervised ML -- Theory to Production")
- ascent04 is called "Supervised ML & Production" (should be "Unsupervised ML, NLP & Deep Learning")
- ascent05 is called "Agents & Unsupervised ML" (should be "LLMs, AI Agents & RAG Systems")
- ascent06 is called "LLMs, Fine-Tuning, Governance" (should be "Alignment, Governance, RL & Deployment")

The dataset recommendations also reflect the old module boundaries -- ascent04 datasets mention "credit scoring" and "supply chain" that now belong in ascent03.

**Fix**: Rewrite the dataset brief to match current module structure. Reassign datasets to their correct modules based on the course brief.

---

## MEDIUM (Fix During Implementation)

### M1. No Explicit Cross-Module Prerequisite Validation

**Location**: All module briefs

The course brief mentions progressive disclosure (70% -> 20% scaffolding) and modules building on each other, but no module brief specifies which specific skills or concepts from prior modules are prerequisites. For example, M4 DriftMonitor exercise says "Deploy Module 3 model" but does not specify which model artifact or exercise output is carried forward.

**Recommendation**: Add a "Prerequisites from Prior Modules" section to each module brief (M2-M6) specifying: which exercise outputs carry forward, which concepts are assumed known, which Kailash engines are assumed familiar.

### M2. Kailash SDK Import Path for Kaizen BaseAgent

**Location**: `workspaces/ascent05/briefs/module-brief.md` line 28

The brief specifies:
```python
from kaizen.core.base_agent import BaseAgent
```

This is a valid path but bypasses the public API. The canonical import should use the top-level kaizen package or kaizen_agents. Verify whether `BaseAgent` is exported from `kaizen.__init__` or if the internal path is the intended usage.

**Recommendation**: Verify the intended public API for BaseAgent and update the import path if a higher-level export exists.

### M3. Missing Explicit Mention of ModelSpec/EvalSpec Pattern

**Location**: `workspaces/ascent03/briefs/module-brief.md`

The M3 brief includes a "Key Patterns" code snippet showing `ModelSpec` and `EvalSpec` usage, but these classes are not mentioned in the lecture topics or exercises. Students need to understand these configuration objects to use TrainingPipeline effectively.

**Recommendation**: Add ModelSpec/EvalSpec to the 3C lecture section or to the lab exercise that introduces TrainingPipeline.

### M4. No Explicit Coverage of ModelVisualizer After Module 1

**Location**: All module briefs

ModelVisualizer is introduced in M1 but never explicitly appears in M2-M6 exercises despite being a natural companion to SHAP plots (M3), clustering visualization (M4), and agent output visualization (M5). Students may forget it exists.

**Recommendation**: Reference ModelVisualizer in at least one exercise each in M3 (SHAP visualization) and M4 (clustering visualization) to reinforce the engine across the curriculum.

### M5. Transformer Lecture (5A) Ambitious for 90 Minutes

**Location**: `workspaces/curriculum/briefs/course-brief.md` M5 section

Section 5A covers: encoder-decoder architecture, self vs cross attention, positional encodings (sinusoidal, RoPE, ALiBi), tokenization internals (BPE, WordPiece, Unigram, SentencePiece), pre-training objectives (MLM, CLM, span corruption), scaling laws (Chinchilla), and inference optimization (KV-cache, speculative decoding, quantization). This is an entire graduate course lecture compressed into 90 minutes. Students need time to absorb the attention mechanism derivation alone.

**Recommendation**: Prioritize. The core path is: attention mechanism derivation -> positional encoding intuition -> pre-training objectives overview -> scaling laws highlights. Move tokenization internals and inference optimization to recommended reading or an appendix slide. These are important but not essential for the lab exercises.

### M6. RL Section (6C) Covers Both Theory and Advanced Topics

**Location**: `workspaces/curriculum/briefs/course-brief.md` M6 section

Section 6C packs RL foundations (MDPs, Bellman, value/policy iteration, TD learning), deep RL (DQN, policy gradient, A2C, PPO, SAC), practical RL applications, AND emerging topics (multi-modal, federated learning, differential privacy, synthetic data). The RLTrainer lab only uses PPO/SAC on Gymnasium environments, so the practical RL applications section has no lab exercise to reinforce it.

**Recommendation**: Trim 6C to RL foundations + PPO/SAC theory (which map directly to the lab). Move the "Emerging" subsection to a "Further Reading" slide. Dynamic pricing and recommendation system RL are interesting but lack lab exercises and dilute the core RL message.

### M7. No Explicit Polars-to-Kailash Transition Guidance

**Location**: Course brief and M1 brief

M1 teaches Polars deeply (45 minutes on expression API, lazy frames, Arrow backend). The kailash-ml engines accept Polars DataFrames as input, but the curriculum never explicitly shows the hand-off pattern (e.g., "you loaded and cleaned with Polars, now pass the DataFrame to DataExplorer/TrainingPipeline"). Students may struggle with the seam between raw Polars work and engine input.

**Recommendation**: Add a brief bridge exercise or code pattern in M1 Lab 3 (DataExplorer profiling) that explicitly demonstrates: `df = pl.read_csv(...); explorer = DataExplorer(); profile = await explorer.profile(df)`.

---

## LOW (Track for Future Improvement)

### L1. No Explicit Error Handling Patterns in Exercises

None of the module briefs mention error handling, retry logic, or graceful degradation. In a production-reality course, students should see at least one exercise where an engine raises an error (bad data type, missing column, model not found in registry) and learn to handle it.

**Recommendation**: Add an error handling sub-exercise in M3 or M4 where students encounter and resolve a deliberate engine error.

### L2. No Accessibility Mention in Deck Brief

The deck brief specifies colors, canvas size, and Reveal.js but does not mention accessibility: alt text for images, color contrast ratios, screen reader compatibility for code blocks.

**Recommendation**: Add an accessibility section to the deck brief.

### L3. E-commerce Dataset Used Across 3 Modules

The "Singapore E-commerce" dataset appears in M4 (clustering), M5 (agent data analysis), and the assessment brief references it implicitly. This is good for continuity but creates a risk that students become overly familiar with one dataset's quirks rather than adapting to new data.

**Recommendation**: Acceptable as-is since the same data is used for fundamentally different tasks (clustering vs agent reasoning), but document the intentional reuse in the dataset brief.

### L4. No Explicit Mention of Version Pinning in Setup Guide

The README shows `uv sync` but does not discuss pinning Kailash SDK versions for reproducibility. As a production-reality course, students should see version pinning.

**Recommendation**: Add a note about version pinning in the setup guide or M1 lab setup slides.

### L5. Deck Brief Speaker Notes Format Underspecified

The deck brief mentions `speaker-notes.md` per module but does not specify format, depth, or timing annotations. Instructors need to know how much time to spend per slide block.

**Recommendation**: Add a speaker notes template with timing markers (e.g., "Slide 5-8: Bias-variance derivation, 15 min").

---

## Academic Rigor Assessment

### Mathematical Depth: STRONG

All required mathematical topics from the benchmark checklist are present:

| Topic | Module | Status |
|-------|--------|--------|
| Bayesian methods (priors, posteriors, conjugate) | M1 | Covered in 1A |
| MLE derivation, Fisher information | M1 | Covered in 1A |
| Causal inference (Rubin, Pearl, do-calculus) | M2 | Covered in 2B |
| Double ML | M2 | Covered in 2B |
| Bias-variance derivation | M3 | Covered in 3A |
| SHAP (Shapley theory, TreeSHAP, KernelSHAP) | M3 | Covered in 3B |
| Calibration (Platt, isotonic, ECE) | M3 | Covered in 3B |
| Spectral clustering (graph Laplacian) | M4 | Covered in 4A |
| EM algorithm derivation (E-step, M-step) | M4 | Covered in 4A |
| Attention mechanism (scaled dot-product) | M4 (theory), M5 (full) | Covered |
| Transformer architecture | M5 | Covered in 5A |
| LoRA theory (low-rank adaptation) | M6 | Covered in 6A |
| DPO derivation | M6 | Covered in 6A |
| Bellman equations | M6 | Covered in 6C |

### Theory Progression: STRONG

The module sequence builds logically:
- M1 (statistics) -> M2 (features + experiments) builds on statistical foundations
- M2 (features) -> M3 (supervised ML) uses engineered features as input
- M3 (supervised ML) -> M4 (unsupervised + NLP + DL) broadens the model family
- M4 (DL foundations) -> M5 (LLMs + agents) builds on attention mechanism
- M5 (agents) -> M6 (alignment + governance) wraps agents in governance

One concern: the jump from M4 (deep learning foundations) to M5 (full transformer architecture) may be steep for students who struggle with M4's attention mechanism introduction. The course addresses this by introducing attention in M4C and deepening it in M5A.

### Missing Topics: MINOR GAPS

| Topic | Status | Severity |
|-------|--------|----------|
| Information-theoretic feature selection (mutual information) | Covered in M2 | OK |
| Bayesian optimization theory | Used in M3 (HyperparameterSearch) but theory not lectured | Low gap |
| Multi-task learning | Not covered | Acceptable omission for scope |
| Graph neural networks | Not covered | Acceptable omission for scope |
| Conformal prediction / prediction sets | Not covered | Worth mentioning in M3 as modern calibration alternative |
| Online learning / streaming ML | Not covered | Worth mentioning in M4 drift monitoring context |

---

## Production Reality Assessment: STRONG

### Real-World Failure Cases

| Case | Module | Purpose |
|------|--------|---------|
| Singapore HDB flash crash | M1 | EDA catches what dashboards miss |
| Healthcare feature leakage | M2 | Point-in-time correctness |
| Zillow iBuyer $500M write-off | M3 | Calibration matters |
| Credit Suisse AML | M4 | Class imbalance in production |
| BloombergGPT | M5 | Domain agents vs general LLMs |
| EU AI Act enforcement | M6 | Governance as competitive advantage |

### Dataset Difficulty: STRONG

| Dataset | Challenge Factor |
|---------|-----------------|
| HDB Resale 15M+ rows | Scale, multi-table joins, geographic, 25-year span |
| Credit Card Fraud 0.17% | Extreme class imbalance, PCA-transformed features |
| Healthcare ICU 60K stays | Irregular time series, multi-table, clinical missing patterns |
| Singapore Credit 100K | Deliberate leakage trap, protected attributes, 30% missing income |
| Taxi/Ridehail LTA | Schema drift across years, GPS noise |
| E-commerce A/B Test | SRM issues, multiple metrics |

The datasets are genuinely messy and challenging. The deliberate leakage trap in M3 credit data is particularly good pedagogy.

---

## Kailash SDK Coverage Assessment

### Framework Coverage: COMPLETE (all 7 frameworks)

| Framework | Module(s) | Engines/Components Used |
|-----------|-----------|------------------------|
| **kailash-ml** | M1-M6 | All 13 engines + 6 ML agents + RLTrainer |
| **Core SDK** | M3 | WorkflowBuilder, LocalRuntime |
| **DataFlow** | M3 | @db.model, db.express |
| **Nexus** | M4, M6 | Multi-channel deployment (API + CLI + MCP) |
| **Kaizen** | M5 | Delegate, BaseAgent, Signature, specialized agents |
| **PACT** | M6 | GovernanceEngine, PactGovernedAgent, D/T/R, compile_org |
| **Align** | M6 | AlignmentPipeline, AlignmentConfig, AdapterRegistry |

### kailash-ml Engine Coverage

| Engine | Module | Exercise |
|--------|--------|----------|
| DataExplorer | M1 | Lab 3 (profiling), Lab 5 (challenge) |
| PreprocessingPipeline | M1 | Lab 5 |
| ModelVisualizer | M1 | Lab 5 |
| FeatureStore | M2 | Lab 2 |
| FeatureEngineer | M2 | Lab 5 |
| ExperimentTracker | M2 | Lab 5 |
| TrainingPipeline | M3 | Lab 1, Lab 6 |
| HyperparameterSearch | M3 | Lab 5 |
| ModelRegistry | M3 | Lab 5 |
| AutoMLEngine | M4 | **Listed in header but no exercise** |
| EnsembleEngine | M4 | **Listed in header but no exercise** |
| DriftMonitor | M4 | Lab 4 |
| InferenceServer | M4, M6 | M4 Lab 6, M6 Lab 6 |

**Gap**: AutoMLEngine and EnsembleEngine are listed in M4's framework header but have no dedicated exercise. See issue H3.

### Import Path Verification (against actual SDK source)

| Import | Brief Location | SDK Status |
|--------|---------------|------------|
| `from kaizen_agents import Delegate` | M5 | Verified: exported from `kaizen_agents.__init__` |
| `from kaizen import Signature, InputField, OutputField` | M5 | Verified: exported from `kaizen.__init__` |
| `from kaizen.core.base_agent import BaseAgent` | M5 | Verified: class exists at this path |
| `from kaizen_agents.agents.specialized.simple_qa import SimpleQAAgent` | M5 | Verified: file exists |
| `from kaizen_agents.agents.specialized.chain_of_thought import ChainOfThoughtAgent` | M5 | Verified: file exists |
| `from kaizen_agents.agents.specialized.react import ReActAgent` | M5 | Verified: file exists |
| `from kaizen_agents.agents.specialized.rag_research import RAGResearchAgent` | M5 | Verified: file exists |
| `from kailash_ml.agents.data_scientist import DataScientistAgent` | M5 | Verified: file exists |
| `from kailash_ml.agents.model_selector import ModelSelectorAgent` | M5 | Verified: file exists |
| `from kailash_ml.agents.feature_engineer import FeatureEngineerAgent` | M5 | Verified: file exists |
| `from kailash_ml.agents.experiment_interpreter import ExperimentInterpreterAgent` | M5 | Verified: file exists |
| `from kailash_ml.agents.drift_analyst import DriftAnalystAgent` | M5 | Verified: file exists |
| `from kailash_ml.agents.retraining_decision import RetrainingDecisionAgent` | M5 | Verified: file exists |
| `from kailash_align import AlignmentConfig, AlignmentPipeline, AdapterRegistry` | M6 | Verified: all exported from `kailash_align.__init__` |
| `from pact import GovernanceEngine, GovernanceContext, PactGovernedAgent` | M6 | Verified: all exported from `pact.__init__` |
| `from pact import Address, RoleEnvelope, TaskEnvelope` | M6 | Verified: all exported from `pact.__init__` |
| `from pact import compile_org, load_org_yaml` | M6 | Verified: both exported from `pact.__init__` |
| `from kailash_ml.rl.trainer import RLTrainer` | M6 | Verified: class exists at this path |
| `from nexus import Nexus` | M6 | Verified: class exists in `nexus.core` |

All 19 import paths verified against the actual SDK source. No broken imports.

---

## Assessment Quality

### AI-Resilience: MODERATE

The assessment structure (quizzes + individual assignment + group project) tests application rather than recall, which is inherently AI-resilient. The quiz topics in M1-M4 briefs include interpretation questions ("What does this output mean?", "Would you approve this loan?") that require contextual judgment.

However, the multiple-choice component (5-7 questions per quiz) is vulnerable to LLM-assisted answering. The short-code and open-interpretation questions are stronger.

**Recommendation**: Weight open-interpretation questions more heavily. Consider adding "debug this pipeline" questions where students must identify and fix errors in Kailash code -- these test understanding that LLMs struggle with for SDK-specific patterns.

### Rubric Specificity: NEEDS IMPROVEMENT

The individual assignment rubric has four dimensions but no sub-criteria or scoring examples. "Kailash pattern correctness (30%)" could mean anything from "used the right import" to "followed the 4-param connection pattern correctly and used runtime.execute(workflow.build())". Without sub-criteria, grading will be inconsistent across assessors.

### Peer Review: VALUE-ADD (if adopted)

The course brief's peer review component (10%) adds genuine value for a professional audience: code review is a daily practice for senior ML engineers. The assessment brief omits this entirely. If adopted, the peer review rubric should focus on: code review quality, SHAP interpretation critique, and governance audit of another team's deployment.

---

## Deck Specifications

### Completeness: STRONG

The deck brief is well structured with clear design principles, color palette, slide structure, file organization, and build process. The opening cases are compelling and real. The "one idea per slide" and "questions over statements" principles follow evidence-based pedagogy.

### Implementation Readiness: READY

The Reveal.js specification (5.1.0, 1280x720, custom theme CSS) is complete enough for implementation. The 50-60 slide target per module is appropriate for 3 hours of lecture content.

### Gap: No Visual Asset Pipeline

The brief mentions `decks/assets/img/` for shared images but does not specify how mathematical equations will be rendered (MathJax? LaTeX images? KaTeX?), how code syntax highlighting works (Reveal.js plugins?), or how diagrams are created (Mermaid? SVG? PNG?).

**Recommendation**: Add a "Technical Requirements" section specifying: MathJax/KaTeX for equations, highlight.js for code (bundled with Reveal.js), and preferred diagramming tool.

---

## Relevant File Paths

- Course brief: `/Users/esperie/repos/training/ascent/workspaces/curriculum/briefs/course-brief.md`
- Module briefs: `/Users/esperie/repos/training/ascent/workspaces/ascent{1-6}/briefs/module-brief.md`
- Assessment brief: `/Users/esperie/repos/training/ascent/workspaces/assessment/briefs/module-brief.md`
- Deck brief: `/Users/esperie/repos/training/ascent/workspaces/decks/briefs/deck-brief.md`
- Dataset brief: `/Users/esperie/repos/training/ascent/workspaces/datasets/briefs/dataset-brief.md`
- Course outline: `/Users/esperie/repos/training/ascent/docs/course-outline.md`
- README: `/Users/esperie/repos/training/ascent/README.md`
- CLAUDE.md: `/Users/esperie/repos/training/ascent/CLAUDE.md`
- SDK source (engines): `/Users/esperie/repos/loom/kailash-py/packages/kailash-ml/src/kailash_ml/engines/`
- SDK source (agents): `/Users/esperie/repos/loom/kailash-py/packages/kailash-ml/src/kailash_ml/agents/`
- SDK source (kaizen): `/Users/esperie/repos/loom/kailash-py/packages/kaizen-agents/src/kaizen_agents/`
- SDK source (pact): `/Users/esperie/repos/loom/kailash-py/packages/kailash-pact/src/pact/`
- SDK source (align): `/Users/esperie/repos/loom/kailash-py/packages/kailash-align/src/kailash_align/`
