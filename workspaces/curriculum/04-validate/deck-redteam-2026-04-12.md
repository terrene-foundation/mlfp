# Deck Red Team Report — MLFP v2 Specification Audit

**Date**: 2026-04-12
**Scope**: All 6 MLFP module decks vs v2 module specifications
**Auditor**: Quality Reviewer Agent (Claude Opus 4.6)

---

## Summary Table

| Module | Deck Slides | Slide Range (78-100) | Viewport | Theme CSS | Lessons Present | Formulas (KaTeX) | Three Layers | Speaker Notes Aligned | Overall |
|--------|-------------|----------------------|----------|-----------|-----------------|-------------------|--------------|----------------------|---------|
| MLFP01 | 78          | PASS (boundary)      | PASS     | PASS      | 8/8             | N/A (no formulas in spec) | PASS | FAIL | FAIL |
| MLFP02 | 99          | PASS                 | PASS     | PASS      | 8/8             | PASS              | PASS         | FAIL                 | FAIL |
| MLFP03 | 90          | PASS                 | PASS     | PASS      | 8/8             | PASS              | PASS         | FAIL (CRITICAL)      | FAIL |
| MLFP04 | 88          | PASS                 | PASS     | PASS      | 8/8             | PASS              | PASS         | FAIL (CRITICAL)      | FAIL |
| MLFP05 | 96          | PASS                 | PASS     | PASS      | 8/8             | PASS              | PASS         | FAIL (CRITICAL)      | FAIL |
| MLFP06 | 90          | PASS                 | PASS     | PASS      | 8/8             | PASS              | PASS         | FAIL (CRITICAL)      | FAIL |

**Overall Status**: BLOCKED (4 critical issues, 5 high issues, 6 medium issues)

---

## CRITICAL Issues (Must Fix)

### C1. Speaker Notes MLFP03 Are From Wrong Module (ASCENT M4)

**Location**: `/Users/esperie/repos/lyceum/courses/mlfp/decks/mlfp03/speaker-notes.md`
**Evidence**: File header reads `# Module 4: Advanced ML -- Unsupervised Methods and Deep Learning -- Speaker Notes`. Content covers clustering, HDBSCAN, PCA, EM algorithm, t-SNE/UMAP, anomaly detection (ASCENT Module 4 topics). The MLFP03 deck covers supervised ML: bias-variance, regularisation, model zoo, gradient boosting, SHAP, workflow orchestration, DataFlow/drift (MLFP Module 3 topics).
**Impact**: Instructor following these notes would teach completely wrong content. Every slide reference, timing, and talking point is for unsupervised ML while the slides show supervised ML.
**Fix**: Regenerate speaker notes from MLFP03 deck content. Must cover lessons 3.1-3.8 as specified in `specs/module-3.md`.

### C2. Speaker Notes MLFP04 Are From Wrong Module (ASCENT M6)

**Location**: `/Users/esperie/repos/lyceum/courses/mlfp/decks/mlfp04/speaker-notes.md`
**Evidence**: File header reads `# Module 6: Alignment, Governance & Organisational Transformation -- Speaker Notes`. Content covers DPO derivation, PACT governance, operating envelopes, alignment (ASCENT Module 6 topics). The MLFP04 deck covers clustering, EM, PCA, anomaly detection, NLP topics, recommender systems, DL foundations (MLFP Module 4 topics).
**Impact**: Complete content mismatch. Instructor would discuss DPO loss functions while slides show K-means clustering.
**Fix**: Regenerate speaker notes from MLFP04 deck content. Must cover lessons 4.1-4.8 as specified in `specs/module-4.md`.

### C3. Speaker Notes MLFP05 Are From Wrong Module (ASCENT M7)

**Location**: `/Users/esperie/repos/lyceum/courses/mlfp/decks/mlfp05/speaker-notes.md`
**Evidence**: File header reads `# Module 7: Deep Learning -- Speaker Notes`. Content references "Module 7", "Preview: Module 8 -- NLP & Transformers", and covers only 89 slides focused on DL foundations through CNNs. The MLFP05 deck has 96 slides covering all 8 lessons (5.1-5.8) including autoencoders, CNNs, RNNs, Transformers, GANs, GNNs, Transfer Learning, and RL.
**Impact**: Speaker notes end at slide 89 covering only autoencoders through CNNs. Slides 5.4-5.8 (Transformers, GANs, GNNs, Transfer Learning, RL) have no speaker notes at all. Module numbering references are wrong throughout.
**Fix**: Regenerate speaker notes from MLFP05 deck content. Must cover all 96 slides across lessons 5.1-5.8.

### C4. Speaker Notes MLFP06 Are From Wrong Module (ASCENT M9)

**Location**: `/Users/esperie/repos/lyceum/courses/mlfp/decks/mlfp06/speaker-notes.md`
**Evidence**: File header reads `# Module 9: LLMs, AI Agents & RAG -- Speaker Notes`. Content references "M8 Recap", "Module 10", "The Full M9 Stack", and describes 200 slides (the deck has only 90). Speaker notes discuss ASCENT 10-module curriculum context while MLFP has only 6 modules.
**Impact**: 200 speaker note slides vs 90 deck slides. Content is from a 10-module programme (ASCENT) applied to a 6-module programme (MLFP). Slide numbering, cross-module references, and scope are all wrong.
**Fix**: Regenerate speaker notes from MLFP06 deck content. Must cover all 90 slides across lessons 6.1-6.8 with correct MLFP (not ASCENT) module references.

---

## HIGH Issues (Should Fix This Session)

### H1. Speaker Notes MLFP01 Contain Theory Slides Not Present in Deck

**Location**: `/Users/esperie/repos/lyceum/courses/mlfp/decks/mlfp01/speaker-notes.md`, slides 23-43
**Evidence**: Speaker notes describe 21 Theory/Advanced slides covering: histograms-to-probability, Normal distribution formula, Poisson/Exponential distributions, Exponential Family, Sufficient Statistics, MLE derivation (2 slides), Fisher Information, MLE Properties, Information Geometry, Bayesian thinking, Conjugate Priors, MAP estimation, Bayesian Predictive Distribution, Hypothesis Testing, Neyman-Pearson Framework, Power Analysis, Permutation Tests, Bootstrap, BCa intervals. None of these slides exist in the M1 deck HTML, which has 78 sections focused on Python/Polars/visualization.
**Impact**: Speaker notes reference slides that do not exist. Instructor attempting to follow notes would find no corresponding slides after slide ~22.
**Fix**: Either (a) strip M1 speaker notes of theory slides 23-43 that have no deck equivalent, or (b) add the theory slides to the deck. Option (a) is recommended since M1 spec does not list key formulas and the statistical theory belongs in M2.

### H2. Speaker Notes MLFP02 Missing Coverage for Lessons 2.5-2.8

**Location**: `/Users/esperie/repos/lyceum/courses/mlfp/decks/mlfp02/speaker-notes.md`
**Evidence**: Speaker notes cover 85 slides focused on Probability, Experiment Design, CUPED, DiD, FeatureStore. The M2 deck has 99 slides that also include Lesson 2.5 (Linear Regression, 17 slides), Lesson 2.6 (Logistic Regression + ANOVA, 17 slides), and Lesson 2.8 (Capstone). The speaker notes have no talking points for the regression and capstone lessons.
**Impact**: Lessons 2.5, 2.6, and 2.8 have deck slides but no speaker notes guidance. Instructor has no timing annotations, talking points, or transition cues for approximately 35 slides.
**Fix**: Extend M2 speaker notes to cover all 99 slides including lessons 2.5 (OLS, t-statistic, R-squared, F-statistic, categorical encoding), 2.6 (sigmoid, log-odds, odds ratios, ANOVA), and 2.8 (capstone statistical analysis project).

### H3. Speaker Notes MLFP01 Line 12 Uses ASCENT Subtitle

**Location**: `/Users/esperie/repos/lyceum/courses/mlfp/decks/mlfp01/speaker-notes.md`, line 12
**Evidence**: `"Welcome to Module 1 of the MLFP -- ML Engineering from Foundations to Mastery at Terrene Open Academy."` The correct MLFP subtitle is "ML Foundations for Professionals" (used in all deck HTML titles).
**Impact**: Instructor reads wrong programme name aloud. "ML Engineering from Foundations to Mastery" is the ASCENT programme subtitle.
**Fix**: Change to "ML Foundations for Professionals".

### H4. MLFP04 Lesson Headers Use Different Convention

**Location**: `/Users/esperie/repos/lyceum/courses/mlfp/decks/mlfp04/deck.html`
**Evidence**: MLFP04 uses `<h1>4.1 Clustering</h1>` while all other decks use `<h2>Lesson X.Y: Topic Name</h2>`. This means `grep 'Lesson'` returns zero matches for M4.
**Impact**: Inconsistent slide structure across modules. Automated tooling that searches for "Lesson X.Y" patterns will miss all M4 lesson boundaries. Presentation navigation relies on consistent heading patterns.
**Fix**: Standardise M4 lesson headers to `<h2>Lesson 4.1: Clustering</h2>` format matching other modules.

### H5. Speaker Notes Timing Sums Exceed 180 Minutes

**Location**: All speaker notes files
**Evidence**:
- MLFP01: 245 min (deck aside: 196 min)
- MLFP02: 234 min (deck aside: 245 min)
- MLFP03: 265 min (deck aside: 164 min)
- MLFP04: 238 min (deck aside: 179 min)
- MLFP05: 300 min (deck aside: 216 min)
- MLFP06: 244 min (deck aside: 209 min)

All header lines state "Total time: ~180 minutes (3 hours)" but actual sum of individual slide timings significantly exceeds this target.
**Impact**: Instructor cannot deliver all content in the stated 3-hour window. M5 notes sum to 300 min (67% over budget). Even accounting for skippable slides, most modules are 30-60% over target.
**Fix**: Audit each module's timing and either (a) mark more slides as "[CAN SKIP IF RUNNING SHORT]" to fit within 180 min, or (b) update the total time header to reflect realistic delivery time.

---

## MEDIUM Issues (Can Defer but Track)

### M1. CSS Variable Naming Uses `--ascent-*` Instead of `--mlfp-*`

**Location**: `/Users/esperie/repos/lyceum/courses/mlfp/decks/assets/css/theme.css` and inline styles in mlfp03, mlfp04, mlfp05, mlfp06 deck HTML
**Evidence**: 103 total occurrences of `--ascent-*` CSS variables across theme and deck files. Variables include `--ascent-primary`, `--ascent-depth`, `--ascent-bg`, `--ascent-text`, `--ascent-foundations`, `--ascent-theory`, `--ascent-advanced`.
**Impact**: Internal naming convention mismatch. CSS variables are not user-visible, but developer/instructor confusion when inspecting styles. Creates coupling to ASCENT naming when MLFP should be independent.
**Fix**: Rename all `--ascent-*` variables to `--mlfp-*` in theme.css and all deck HTML files that reference them inline.

### M2. MLFP01 Slide Count at Minimum Boundary (78)

**Location**: `/Users/esperie/repos/lyceum/courses/mlfp/decks/mlfp01/deck.html`
**Evidence**: 78 slide sections. Spec requires 78-100. M1 is at the exact minimum.
**Impact**: No room for additions. If any content from the theory slides (H1) or additional examples are needed, the deck has no margin.
**Fix**: Consider adding 2-4 slides to provide more buffer. Candidate additions: an ADVANCED slide on exponential families (connects M1 distributions to M2 MLE) and a Kailash bridge summary slide for lesson 1.5.

### M3. MLFP03 Missing KaTeX in Highlight.js Import Line

**Location**: `/Users/esperie/repos/lyceum/courses/mlfp/decks/mlfp03/deck.html`
**Evidence**: Head section imports KaTeX CSS (line 16) and KaTeX JS (at bottom), which is correct. However, the head structure differs from M1/M2 (separate Highlight.js line vs combined). This is cosmetic but inconsistent.
**Impact**: None functionally; Reveal.js loads the same plugins. Code maintenance is slightly harder.
**Fix**: Standardise head section structure across all 6 decks.

### M4. Deck-Notes Slide Count Mismatches (Non-Critical Modules)

**Location**: All modules
**Evidence**:
| Module | Notes Slides | Deck Sections | Delta |
|--------|-------------|---------------|-------|
| MLFP01 | 73          | 78            | -5    |
| MLFP02 | 85*         | 99            | -14   |
| MLFP03 | 86**        | 90            | -4    |
| MLFP04 | 90**        | 88            | +2    |
| MLFP05 | 89**        | 96            | -7    |
| MLFP06 | 200**       | 90            | +110  |

\* M2 notes missing regression lessons. \*\* Notes from wrong ASCENT modules (C1-C4).

**Impact**: Even after fixing C1-C4 (regenerating notes), the slide counts should be verified to match.
**Fix**: After regeneration, verify each speaker notes file has exactly one `## Slide N` entry per deck `<section>`.

### M5. M5 Deck Self-Closing Viewport Meta Tag

**Location**: `/Users/esperie/repos/lyceum/courses/mlfp/decks/mlfp05/deck.html`, line 5
**Evidence**: Uses `<meta name="viewport" content="width=device-width, initial-scale=1.0" />` (self-closing) while all other decks use `<meta name="viewport" content="width=device-width, initial-scale=1.0">` (no self-closing).
**Impact**: No functional difference (both valid HTML5). Minor inconsistency.
**Fix**: Remove trailing ` /` for consistency.

### M6. Theme CSS Uses Emoji in Comments

**Location**: `/Users/esperie/repos/lyceum/courses/mlfp/decks/assets/css/theme.css`, line 6
**Evidence**: Comment contains emoji characters for the three layers.
**Impact**: Minor style inconsistency with codebase conventions.
**Fix**: Replace emoji with text labels in CSS comments.

---

## Module-by-Module Findings

### MLFP01: Data Pipelines and Visualisation

**Deck Quality**: GOOD. All 8 lessons present (1.1-1.8) with clear section markers. Correct MLFP branding throughout. Three-layer callouts: 92 FOUNDATIONS, 21 THEORY, 6 ADVANCED. Kailash bridge callouts: 19. Code examples use `MLFPDataLoader` (correct). No ASCENT references.

**Spec Coverage**:
- All 8 lesson topic blocks present with matching content
- Kailash engines (DataExplorer, PreprocessingPipeline, ModelVisualizer) all present
- No Key Formulas required (spec says "None" for lesson 1.6)
- M1 spec does not list formulas for any lesson; deck correctly focuses on Python/Polars

**Issues**:
- (H1) Speaker notes slides 23-43 describe theory content not in deck
- (H3) Speaker notes line 12 uses ASCENT subtitle
- (M2) Slide count at minimum boundary (78)

**Speaker Notes**: Mostly aligned for the foundations content (slides 1-22, 44-73). The theory block (slides 23-43) is orphaned -- these slides reference Normal distribution, MLE, Bayesian inference that are M2 topics. Timing: 245 min in notes vs 180 min target.

### MLFP02: Statistical Mastery for ML and AI

**Deck Quality**: EXCELLENT. All 8 lessons present (2.1-2.8). 99 slides (highest count). Strong KaTeX usage (108 math expressions). Three-layer coverage: 96 FOUNDATIONS, 89 THEORY, 27 ADVANCED. Kailash bridge callouts: 13.

**Spec Coverage**:
- Bayes' theorem with full derivation: PRESENT
- Expected value formula: PRESENT
- Bootstrap CI, test statistic, Bonferroni: PRESENT
- OLS, T-statistic, R-squared, F-statistic: PRESENT
- Sigmoid, log-odds, odds ratio, ANOVA F-statistic: PRESENT
- CUPED variance reduction derivation: PRESENT
- DiD ATT formula: PRESENT
- FeatureEngineer, FeatureStore: PRESENT

**Issues**:
- (H2) Speaker notes missing coverage for lessons 2.5 (linear regression), 2.6 (logistic/ANOVA), 2.8 (capstone) -- approximately 35 slides without notes

**Speaker Notes**: Correct module identification and content for lessons 2.1-2.4, 2.7. Missing coverage for lessons 2.5-2.6 and 2.8. Timing: 234 min in notes vs 180 min target.

### MLFP03: Supervised Machine Learning

**Deck Quality**: GOOD. All 8 lessons present (3.1-3.8). 90 slides. KaTeX: 55 math expressions. Three layers: 61 FOUNDATIONS, 32 THEORY, 11 ADVANCED. Kailash bridge callouts: 26 (highest).

**Spec Coverage**:
- Bias-variance decomposition: PRESENT (slides 16-17)
- L1, L2, Elastic Net penalties: PRESENT
- SVM margin, Gini impurity, Information gain: PRESENT
- XGBoost 2nd-order Taylor, split gain: PRESENT
- LightGBM GOSS: PRESENT
- Precision, Recall, F1, AUC, Focal Loss, Brier Score: PRESENT
- Shapley values formula, Disparate impact: PRESENT
- WorkflowBuilder with `runtime.execute(wf.build())`: PRESENT (line 1713)
- DataFlow, DriftMonitor, PSI, KS: PRESENT

**Issues**:
- (C1) Speaker notes are from ASCENT M4 (unsupervised ML) -- completely wrong module

### MLFP04: Unsupervised ML and Advanced Techniques

**Deck Quality**: GOOD. All 8 lesson blocks present (4.1-4.8). 88 slides. KaTeX: 42 math expressions. Feature Engineering Spectrum present (line 189). THE PIVOT present (line 1595). Three layers: 69 FOUNDATIONS, 28 THEORY, 10 ADVANCED. Kailash bridge callouts: 17.

**Spec Coverage**:
- K-means objective, Silhouette, Davies-Bouldin: PRESENT
- EM E-step, M-step, log-likelihood: PRESENT
- PCA, SVD, variance explained, reconstruction error: PRESENT
- Isolation Forest anomaly score, LOF, Z-score, IQR: PRESENT
- Support, Confidence, Lift (association rules): PRESENT
- TF-IDF, LDA, NPMI: PRESENT
- Matrix factorisation, ALS update: PRESENT
- Forward pass, gradient descent, backprop chain rule, batch norm, Adam: PRESENT
- Feature Engineering Spectrum (THE organizing spine): PRESENT

**Issues**:
- (C2) Speaker notes are from ASCENT M6 (alignment/governance) -- completely wrong module
- (H4) Lesson headers use `<h1>4.N Topic</h1>` instead of `<h2>Lesson 4.N: Topic</h2>`

### MLFP05: Deep Learning and Vision

**Deck Quality**: EXCELLENT. All 8 lessons present (5.1-5.8). 96 slides. KaTeX: 152 math expressions (highest). Three layers: 61 FOUNDATIONS, 67 THEORY, 16 ADVANCED. Kailash bridge callouts: 11.

**Spec Coverage**:
- VAE ELBO, reparameterisation trick: PRESENT
- CNN output size, ResNet skip connection, SE block: PRESENT
- LSTM all 6 gate equations, Perplexity: PRESENT
- Scaled dot-product attention, multi-head attention, positional encoding: PRESENT
- GAN minimax, WGAN, FID: PRESENT
- GCN, GAT attention: PRESENT
- Transfer learning architecture guide: PRESENT
- Bellman equations, DQN loss, PPO clipped objective: PRESENT

**Issues**:
- (C3) Speaker notes are from ASCENT M7 -- only covers autoencoders through CNNs (89 slides for 96-slide deck), missing notes for Transformers, GANs, GNNs, Transfer Learning, RL
- (M5) Self-closing viewport meta tag

### MLFP06: LLMs, Agents and Governance

**Deck Quality**: GOOD. All 8 lessons present (6.1-6.8). 90 slides. KaTeX: 48 math expressions. Three layers: 54 FOUNDATIONS, 49 THEORY, 21 ADVANCED. Kailash bridge callouts: 12.

**Spec Coverage**:
- LLM pre-training, scaling laws, notable models: PRESENT
- Prompt engineering (zero-shot, few-shot, CoT, self-consistency): PRESENT
- Kaizen Delegate, Signature, structured output: PRESENT
- LoRA from scratch, adapter layers from scratch: PRESENT
- LoRA vs adapter comparison: PRESENT
- Fine-tuning landscape (10 techniques): PRESENT
- DPO loss, GRPO: PRESENT
- RAG pipeline, RAGAS evaluation, HyDE: PRESENT
- ReAct agents, function calling: PRESENT
- Multi-agent patterns, MCP protocol: PRESENT
- PACT D/T/R addressing, GovernanceEngine, operating envelopes: PRESENT
- Nexus multi-channel deployment: PRESENT

**Issues**:
- (C4) Speaker notes are from ASCENT M9 -- 200 slides vs 90 deck slides, references M8/M10 which don't exist in MLFP

---

## Feature Engineering Spectrum Check (Design Principles)

**Requirement**: The Feature Engineering Spectrum (from `specs/design-principles.md`, section 4) must be present in the M4 deck as THE organizing spine.

**Finding**: PRESENT. The Feature Engineering Spectrum appears at:
- Line 189: Full spectrum slide with `<h2>The Feature Engineering Spectrum</h2>`
- Line 1708: Referenced again in the 4.8 DL Foundations section (aside notes)
- Multiple forward/backward references in lesson transition slides

The spectrum correctly maps: M3 manual -> M4.1-6 USML -> M4.7 collaborative filtering pivot -> M4.8 DL generalisation.

**Status**: PASS

---

## Infrastructure Checks

| Check | MLFP01 | MLFP02 | MLFP03 | MLFP04 | MLFP05 | MLFP06 |
|-------|--------|--------|--------|--------|--------|--------|
| Viewport meta | PASS | PASS | PASS | PASS | PASS* | PASS |
| Theme CSS import | PASS | PASS | PASS | PASS | PASS | PASS |
| KaTeX CSS | PASS | PASS | PASS | PASS | PASS | PASS |
| KaTeX JS | PASS | PASS | PASS | PASS | PASS | PASS |
| Reveal.js 5.1.0 | PASS | PASS | PASS | PASS | PASS | PASS |
| Highlight.js | PASS | PASS | PASS | PASS | PASS | PASS |
| MLFP branding | PASS | PASS | PASS | PASS | PASS | PASS |
| No ASCENT refs (deck) | PASS | PASS | PASS | PASS | PASS | PASS |
| Slide footer | PASS | PASS | PASS | PASS | PASS | PASS |

\* M5 uses self-closing `/>` on viewport meta (M5 issue).

---

## Prioritised Fix Order

1. **C1-C4**: Regenerate all 4 mismatched speaker notes (MLFP03, 04, 05, 06) from deck content
2. **H1**: Strip orphaned theory slides (23-43) from MLFP01 speaker notes
3. **H2**: Extend MLFP02 speaker notes to cover lessons 2.5, 2.6, 2.8
4. **H3**: Fix ASCENT subtitle in MLFP01 speaker notes line 12
5. **H4**: Standardise MLFP04 lesson headers to `Lesson 4.N: Topic` format
6. **H5**: Audit timing annotations across all modules
7. **M1-M6**: CSS variable renaming, viewport tag, slide count padding (defer to next session)
