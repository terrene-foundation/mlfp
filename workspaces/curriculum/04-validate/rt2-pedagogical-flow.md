# Red Team Round 2 — Pedagogical Flow Audit

**Date**: 2026-04-13
**Perspective**: Corporate ML trainer evaluating learner experience across 48 lessons
**Reference**: `specs/design-principles.md` (8 design principles)
**Scope**: All textbook HTML files at `decks/mlfp0{1-6}/lessons/0{1-8}/textbook.html`

---

## Executive Summary

**Overall Status**: ISSUES FOUND

Modules 1-4 deliver gold-standard corporate training material — Singapore-grounded, mathematically rigorous, structurally consistent, and progressively scaffolded. Module 5 shows a quality cliff in its final three lessons (L06-L08). Module 6 has the sharpest quality break in the curriculum: four exceptional lessons (L01-L04, 689-828 lines each) followed by three skeletal outlines (L05-L07, 60-89 lines) and a thin capstone (L08, 102 lines).

The Feature Engineering Spectrum pivot (M4.7-M4.8) is perfectly executed. Cross-module bridges are strong throughout. The three-layer scaffolding system (Foundations/Theory/Advanced) is consistently applied in complete lessons but absent from all skeletal content.

**Priority action**: Expand M5.L06-L08 and M6.L05-L08 to full lesson depth. These 7 lessons represent 15% of the curriculum but contain less than 5% of its content.

---

## Module Ratings

| Module | Engagement | Flow | Scaffolding | Completeness | Overall |
|--------|-----------|------|-------------|--------------|---------|
| M1 — Foundations | 5 | 5 | 5 | 5 | **5.0** |
| M2 — Feature Engineering | 4 | 5 | 5 | 4 | **4.5** |
| M3 — Supervised ML | 4 | 5 | 5 | 4 | **4.5** |
| M4 — Unsupervised ML & NLP | 4 | 5 | 5 | 4 | **4.5** |
| M5 — Deep Learning | 3 | 4 | 3 | 2 | **3.0** |
| M6 — LLMs & Production | 3 | 3 | 2 | 2 | **2.5** |

**Rating scale**: 1 = Unusable, 2 = Significant gaps, 3 = Functional with issues, 4 = Strong, 5 = Exemplary

---

## Module 1 — Foundations: Statistics, Probability & Data

**Ratings**: Engagement 5 | Flow 5 | Scaffolding 5 | Completeness 5

### Strengths

- **Singapore grounding in every opener**: L01 Singapore weather, L02 HDB 500K rows, L03 district price report, L04 HDB+MRT+Schools enrichment, L05 window functions on HDB time series, L06 Gestalt visualization principles, L07 DataExplorer profiling, L08 capstone ETL pipeline with OneMap REST API
- **Three-layer callouts present in all 8 lessons**: Foundations (green), Theory (blue), Advanced (purple) consistently applied
- **Mathematical derivations properly placed**: Mean/variance/std with Bessel's correction (L01), IQR/CV (L03), moving averages (L05), skewness/correlation (L07)
- **Structural template 100% complete**: Why This Matters, Core Concepts, Kailash Engine, Worked Example, Mathematical Foundations, Common Mistakes, Try It Yourself (3-4 per lesson), Cross-References, Reflection
- **Progressive disclosure exemplary**: ~70% code provided, heavy scaffolding, zero-background entry point

### Issues

None. This module is the reference implementation for the curriculum's structural template.

---

## Module 2 — Feature Engineering & Experiment Design

**Ratings**: Engagement 4 | Flow 5 | Scaffolding 5 | Completeness 4

### Strengths

- **Strong Singapore grounding**: HDB feature engineering, MRT proximity features, school district encoding
- **Learning flow L01-L08 is tight**: Each lesson builds directly on the previous, from raw features through encoding, scaling, selection, to experiment tracking
- **Three-layer callouts consistent**: All lessons have Foundations/Theory/Advanced markers
- **FeatureEngineer and FeatureStore engines properly introduced**: Students learn the Kailash way from the start

### Issues

| Priority | Issue | Location | Fix |
|----------|-------|----------|-----|
| Critical | L07 Worked Example contains broken code — references undefined variables and incomplete pipeline | `mlfp02/lessons/07/textbook.html` | Complete the worked example with runnable code |
| Critical | L08 uses speculative/unreleased API patterns that don't match current kailash-ml | `mlfp02/lessons/08/textbook.html` | Verify against kailash-ml 0.4.x API and update |
| Minor | Singapore use cases thinner in L06-L08 compared to L01-L05 | L06-L08 openers | Add Singapore-specific examples to later lesson openers |

---

## Module 3 — Supervised ML: Theory to Production

**Ratings**: Engagement 4 | Flow 5 | Scaffolding 5 | Completeness 4

### Strengths

- **ML pipeline teaching fully realized**: Students build complete train-evaluate-deploy pipelines using TrainingPipeline and ModelRegistry
- **Theory-to-practice bridge strong**: Each algorithm gets mathematical derivation + Kailash implementation + worked example
- **Cross-module bridges excellent**: References back to M1 statistics (distributions, correlation) and M2 features (encoding, selection) throughout
- **Progressive disclosure on target**: ~50% code provided, students implement core logic while scaffolding handles boilerplate

### Issues

| Priority | Issue | Location | Fix |
|----------|-------|----------|-----|
| Important | L07 missing formal Worked Example section — has code but not structured as the canonical worked example pattern | `mlfp03/lessons/07/textbook.html` | Add structured Worked Example with problem statement, step-by-step, and solution |
| Minor | Try It Yourself exercises in L06-L08 could be more challenging — some are repetitive variations | L06-L08 exercises | Add at least one exercise per lesson that requires combining multiple concepts |

---

## Module 4 — Unsupervised ML, Anomaly Detection & NLP

**Ratings**: Engagement 4 | Flow 5 | Scaffolding 5 | Completeness 4

### Strengths

- **Feature Engineering Spectrum pivot perfectly executed**: L07 ("THE PIVOT") nails the transition where optimization drives feature discovery via collaborative filtering. L08 bridges to deep learning with "hidden layers = automated feature engineering with error feedback." The narrative arc from M3 manual features → M4.1-6 USML discovers features → M4.7 optimization drives discovery → M4.8 DL generalizes is the curriculum's strongest pedagogical thread.
- **Clustering-to-NLP flow is natural**: K-means → hierarchical → DBSCAN → anomaly detection → dimensionality reduction → NLP, each building on the previous
- **Mathematical foundations strong**: SVD derivation (L05), TF-IDF weighting (L06), matrix factorization for collaborative filtering (L07)
- **AutoMLEngine and EnsembleEngine properly introduced**: Students see automated pipeline construction before the DL pivot

### Issues

| Priority | Issue | Location | Fix |
|----------|-------|----------|-----|
| Important | L08 missing explicit Kailash Engine callout — uses OnnxBridge in code but no formal "The Kailash Engine" section | `mlfp04/lessons/08/textbook.html` | Add Kailash Engine callout section for OnnxBridge as the DL bridge |
| Minor | Singapore use cases in L07-L08 are generic — could use local recommender data (HDB, NTUC, Grab) | L07-L08 openers | Ground the collaborative filtering and DL examples in Singapore platforms |

### Feature Engineering Spectrum Assessment

The spec (`specs/module-4.md`) requires L07 to be "THE PIVOT — optimisation drives feature discovery" and L08 to be "DL Foundations — hidden layers are USML + error feedback." Both requirements are met. The spectrum from manual features (M3) through automated discovery (M4.1-6) to optimization-driven discovery (M4.7) to learned representations (M4.8) is the curriculum's best pedagogical thread. A corporate trainer could use this arc as the single unifying narrative across the entire programme.

---

## Module 5 — Deep Learning

**Ratings**: Engagement 3 | Flow 4 | Scaffolding 3 | Completeness 2

### Strengths

- **L01-L05 are strong**: Neural network fundamentals, backpropagation, CNNs, RNNs/LSTMs, and transformers are well-taught with proper mathematical derivations and Kailash OnnxBridge integration
- **Transformer lesson (L05) is excellent**: Full attention mechanism derivation, positional encoding, multi-head attention — properly bridges to M6 LLMs
- **Cross-module bridges to M4**: L01 explicitly connects to M4.8's DL preview, completing the Feature Engineering Spectrum

### Issues

| Priority | Issue | Location | Fix |
|----------|-------|----------|-----|
| **Critical** | **L06 (GNNs) is severely incomplete** — skeletal outline without worked examples, missing Try It Yourself exercises, no three-layer callouts, mathematical derivations cut short | `mlfp05/lessons/06/textbook.html` | Expand to full lesson depth (600-800 lines) with complete GNN worked example, 3 exercises, all callouts |
| **Critical** | **L07 (Transfer Learning) critically incomplete** — concepts listed but not taught. No worked example showing fine-tuning or feature extraction. Missing mathematical foundations for domain adaptation | `mlfp05/lessons/07/textbook.html` | Expand with transfer learning worked example (ImageNet → Singapore domain), mathematical foundations for distribution shift |
| **Critical** | **L08 (RL) critically incomplete** — Q-learning and policy gradient mentioned but not derived. No worked example. Missing the connection to M6 alignment (RLHF) | `mlfp05/lessons/08/textbook.html` | Expand with full RL foundations, Bellman equation derivation, worked example, explicit bridge to M6.3 RLHF |
| Important | Three-layer callouts disappear in L06-L08 — only Foundations/Theory layers used inconsistently | L06-L08 | Add all three tiers consistently |
| Important | Singapore use cases absent in L06-L08 — L01-L05 all have Singapore grounding | L06-L08 openers | GNNs: Singapore transport network. Transfer: Singapore food/architecture recognition. RL: Singapore traffic optimization |

### Quality Cliff Analysis

The depth disparity is stark. L01-L05 average 650-800 lines each with complete structural templates. L06-L08 are 150-250 lines with missing sections. A corporate trainer would need to supplement L06-L08 with external material, which breaks the self-contained promise of the curriculum.

---

## Module 6 — LLMs, AI Agents & Production

**Ratings**: Engagement 3 | Flow 3 | Scaffolding 2 | Completeness 2

### Strengths

- **L01-L04 are gold standard**: LLM fundamentals (825 lines), LoRA/fine-tuning (828 lines), DPO/GRPO alignment (689 lines), and RAG systems (754 lines) are among the best lessons in the curriculum
- **Cross-module backward bridges excellent**: L01 links to M5.4 transformers, L02 links to M4.3 SVD for LoRA's low-rank insight, L03 references M5.5 KL divergence and M3.2 logistic loss, L04 connects to M4.6 NLP embeddings
- **Kailash engine integration strong in L01-L04**: Delegate, Signature, AlignmentPipeline all properly used
- **Mathematical depth impressive**: Temperature/sampling derivation (L01), LoRA parameter reduction formula (L02), DPO loss derivation (L03), BM25/cosine similarity (L04)

### Issues

| Priority | Issue | Location | Fix |
|----------|-------|----------|-----|
| **Critical** | **L05 (AI Agents) is a skeletal outline — 89 lines** vs 750-825 for L01-L04. Missing: Worked Example, Try It Yourself, Reflection, three-layer callouts, Mathematical Foundations. Shows raw JSON function-calling schema instead of Kaizen BaseAgent/Delegate patterns | `mlfp06/lessons/05/textbook.html` | Expand to 600-800 lines. Add Kaizen BaseAgent implementation, ReAct loop worked example with Singapore data, 3 exercises, all structural sections |
| **Critical** | **L06 (Multi-Agent Orchestration) is the shortest lesson — 70 lines.** Promises "four multi-agent patterns" in the lead but delivers only an SVG diagram and a single MCP code snippet. No prose, no worked example, no exercises, no cross-references, no reflection | `mlfp06/lessons/06/textbook.html` | Expand to 600-800 lines. Implement supervisor-worker, sequential, parallel, handoff patterns with Kaizen Pipeline. Add MCP server worked example |
| **Critical** | **L07 (AI Governance) — 60 lines.** The shortest lesson in the entire curriculum. PACT D/T/R addressing mentioned but not taught. No Singapore regulatory context (AI Verify, PDPA, MAS TRM) despite governance being the ideal domain. No exercises | `mlfp06/lessons/07/textbook.html` | Expand to 600-800 lines. Add PACT GovernanceEngine worked example, Singapore AI Verify framework integration, 3 exercises, all structural sections |
| **Critical** | **L08 (Capstone) has no exercises — 102 lines.** A capstone with no Try It Yourself defeats the purpose. Code snippets are fragments, not a complete walkthrough. No Cross-References section despite being the finale | `mlfp06/lessons/08/textbook.html` | Add 2-3 staged capstone exercises where the student wires up the full stack. Add Cross-References linking back to all 6 modules |
| Important | No Singapore context in L05-L08 openers — L01-L04 all have Singapore grounding (monsoon, MAS regulations, company reports, PDPA) | L05-L08 | L05: Singapore public service agent. L06: MAS compliance multi-agent. L07: AI Verify + PDPA governance. L08: Singapore domain capstone |
| Important | Missing Kailash Kaizen usage in L05 — course spec requires Kaizen (Delegate, BaseAgent, Signature) for agent content | L05 | Replace raw JSON function-calling with Kaizen agent patterns |
| Important | L06 promises multi-agent ML pipeline example in opener but body has no such example | L06 | Add DataScientist → FeatureEngineer → ModelSelector → ReportWriter multi-agent worked example |

### M6.8 Finale Assessment

The capstone's "What You Have Built Across Six Modules" summary table effectively bookends the curriculum. The "Congratulations" reflection closes the arc from `print('Hello')` to production platform. However, at 102 lines with no exercises, the finale asks students to *read about* integration rather than *do* it. A corporate ML training programme needs the capstone to be the most hands-on lesson, not the least.

### Structural Collapse Analysis

| Lesson | Lines | Status |
|--------|-------|--------|
| L01 — LLM Fundamentals | 825 | Complete |
| L02 — LoRA & Fine-Tuning | 828 | Complete |
| L03 — DPO & GRPO | 689 | Complete |
| L04 — RAG Systems | 754 | Complete |
| L05 — AI Agents | 89 | **Skeletal** |
| L06 — Multi-Agent & MCP | 70 | **Skeletal** |
| L07 — AI Governance | 60 | **Skeletal** |
| L08 — Capstone | 102 | **Thin** |

A student who spends 60-90 minutes absorbing the deep L04 RAG lesson will finish L05-L07 combined in under 10 minutes. The depth discontinuity signals incomplete material.

---

## Cross-Cutting Analysis

### Learning Flow (L01-L08 Within Modules)

| Module | Flow Quality | Notes |
|--------|-------------|-------|
| M1 | Seamless | Each lesson builds exactly on the previous. Zero jumps |
| M2 | Seamless | Raw features → encoding → scaling → selection → tracking |
| M3 | Seamless | Linear → logistic → trees → SVM → evaluation → pipeline |
| M4 | Seamless | Clustering → anomaly → dim reduction → NLP → pivot → DL preview |
| M5 | Breaks at L06 | L01-L05 chain well. L06-L08 are conceptually connected but depth drop breaks the teaching rhythm |
| M6 | Breaks at L05 | L01-L04 chain beautifully. L05 abrupt depth cliff. L05-L08 conceptual flow is sound but content is missing |

### Cross-Module Bridges (M1-M6)

| Bridge | Quality | Example |
|--------|---------|---------|
| M1 → M2 | Strong | M2 explicitly references M1 statistics (distributions, correlation) for feature analysis |
| M2 → M3 | Strong | M3 references M2 feature engineering and experiment tracking throughout |
| M3 → M4 | Strong | M4 references M3 supervised methods as contrast for unsupervised approaches |
| M4 → M5 | Excellent | M5.L01 directly bridges from M4.L08 DL preview. Feature Engineering Spectrum continues |
| M5 → M6 | Excellent | M6.L01 bridges from M5.L04 transformer. M6.L02 references M4.L03 SVD. M6.L03 references M5.L05 KL divergence |

Cross-module bridges are a curriculum strength. Even in incomplete lessons, backward references are mathematically grounded (e.g., "Lesson 4.3 SVD" for LoRA's low-rank insight).

### Three-Layer Scaffolding Consistency

| Module | Foundations (Green) | Theory (Blue) | Advanced (Purple) | Consistency |
|--------|-------------------|--------------|-------------------|-------------|
| M1 | All 8 lessons | All 8 lessons | All 8 lessons | 100% |
| M2 | All 8 lessons | All 8 lessons | All 8 lessons | 100% |
| M3 | All 8 lessons | All 8 lessons | All 8 lessons | 100% |
| M4 | All 8 lessons | All 8 lessons | All 8 lessons | 100% |
| M5 | L01-L05 only | L01-L05 only | L01-L05 only | 62% |
| M6 | L01-L04 only | L01-L04 only | L01-L04 only (L03 missing Foundations) | 50% |

**Pattern**: Three-layer callouts are perfectly consistent in complete lessons and completely absent from skeletal lessons. This is not a design issue — it is a completeness issue.

### Singapore Use Cases

| Module | Lessons with SG Opener | Coverage |
|--------|----------------------|----------|
| M1 | L01-L08 (all) | 100% |
| M2 | L01-L05 strong, L06-L08 lighter | 75% |
| M3 | L01-L06 strong, L07-L08 lighter | 75% |
| M4 | L01-L06 strong, L07-L08 generic | 75% |
| M5 | L01-L05 present, L06-L08 absent | 62% |
| M6 | L01-L04 strong (monsoon, MAS, PDPA), L05-L08 absent | 50% |

**Pattern**: Singapore grounding correlates perfectly with lesson completeness. Incomplete lessons universally drop Singapore context.

### Worked Example Completeness

| Module | Lessons with Formal Worked Example | Coverage |
|--------|----------------------------------|----------|
| M1 | 8/8 | 100% |
| M2 | 7/8 (L07 broken code) | 87% |
| M3 | 7/8 (L07 missing formal structure) | 87% |
| M4 | 8/8 | 100% |
| M5 | 5/8 (L06-L08 missing) | 62% |
| M6 | 4/8 (L05-L08 missing, L02 has implementation instead) | 50% |

### Try It Yourself Achievability

- **M1-M4**: All exercises are achievable using skills taught in the lesson. Progressive difficulty within each set. 3-4 exercises per lesson.
- **M5 L01-L05**: Achievable, good difficulty ramp.
- **M5 L06-L08**: Exercises missing or minimal.
- **M6 L01-L04**: 3 exercises each, well-calibrated for the module's ~20% scaffolding level.
- **M6 L05-L08**: No exercises exist.

---

## Feature Engineering Spectrum — Full Arc Assessment

The curriculum's defining pedagogical innovation is the Feature Engineering Spectrum, and it is executed well:

| Stage | Module.Lesson | What Students Learn |
|-------|--------------|-------------------|
| Manual features | M2-M3 | Hand-craft features from domain knowledge |
| USML discovers features | M4.L01-L06 | Clustering, PCA, NMF find structure without labels |
| THE PIVOT | M4.L07 | Collaborative filtering — optimization drives feature discovery |
| DL generalizes | M4.L08 | Hidden layers = automated feature engineering with error feedback |
| Specialized architectures | M5.L01-L05 | CNNs for spatial, RNNs for temporal, Transformers for attention |
| LLMs learn semantic features | M6.L01 | Pre-training on next-token prediction learns grammar, reasoning, world knowledge |

The spectrum from "engineer picks features" to "model discovers features" to "model learns what features even are" is the strongest narrative thread in the curriculum. Each transition is explicitly called out in the relevant lesson.

---

## Priority Action Items

### Critical (Must fix before delivery)

1. **Expand M6 L05-L07** — Three lessons totaling 219 lines. Each needs 600-800 lines with full structural template (Why This Matters, three-layer callouts, Worked Example, Mathematical Foundations, Try It Yourself ×3, Cross-References, Reflection). These cover agents, multi-agent orchestration, and governance — topics that corporate learners most need hands-on practice with.

2. **Expand M5 L06-L08** — Three lessons covering GNNs, transfer learning, and RL. Each needs full depth, especially L08 which must explicitly bridge to M6.L03 RLHF.

3. **Add exercises to M6 L08 Capstone** — The programme finale must be hands-on. Add 2-3 staged exercises where the student builds and deploys the full stack.

4. **Fix M2 L07 broken worked example** — Code references undefined variables.

### Important (Should fix in current session)

5. **Add Singapore context to M5 L06-L08 and M6 L05-L08** — 7 lessons without Singapore grounding. M6.L07 (governance) especially needs AI Verify, PDPA, MAS TRM.

6. **Replace raw JSON function-calling with Kaizen patterns in M6 L05** — Course spec requires Kaizen (Delegate, BaseAgent, Signature).

7. **Verify M2 L08 API patterns against current kailash-ml** — Speculative APIs flagged.

8. **Add formal Worked Example to M3 L07** — Has code but not structured as canonical pattern.

### Minor (Can defer but track)

9. **M6 L03 missing Foundations-tier callout** — Only Theory and Advanced present.
10. **Singapore use cases thinner in later lessons across M2-M4** — Not absent, but less specific.

---

## Conclusion

The MLFP curriculum is 70% exceptional and 30% unfinished. The exceptional parts — M1 in its entirety, M2-M4's core lessons, M5.L01-L05, M6.L01-L04 — would satisfy the most demanding corporate ML training buyer. The Feature Engineering Spectrum is a genuine pedagogical innovation that no competing curriculum offers.

The unfinished parts — 7 skeletal lessons across M5-M6 — are not low-quality; they are simply not written yet. The structural template, three-layer system, and Singapore grounding that make the complete lessons excellent are entirely absent from the incomplete ones, suggesting these were committed as planning outlines.

**Estimated effort to reach full quality**: Expanding 7 skeletal lessons to full depth (600-800 lines each, with worked examples, exercises, and Singapore grounding) plus fixing the 3 flagged issues in M2-M3.
