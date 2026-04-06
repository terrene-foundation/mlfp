# Presentation Deck Brief

## Audience Reality

The classroom contains two extremes sitting side by side:
- **Zero-background professionals**: Career-switchers, domain experts (doctors, lawyers, bankers) who have never written code. They need "What is a variable?" explained clearly.
- **Masters-and-above practitioners**: Working data scientists, ML engineers, PhD holders who want the EM derivation, the DPO proof, and the SHAP axioms. They will disengage if the content is shallow.

**The decks must serve BOTH simultaneously.** This is the core design challenge.

## Dual-Track Slide Architecture

Every concept is taught in three layers within the same deck:

### Layer 1: Intuition (everyone follows)
Plain English explanation with visual analogy. No math, no code. A banker or doctor can understand this slide.

**Example** (bias-variance):
> "Imagine you throw darts at a target. Bias is how far the center of your throws is from the bullseye. Variance is how spread out your throws are. A good model has low bias AND low variance — but improving one often worsens the other."

### Layer 2: Mathematical Foundation (intermediate follows, beginners see the shape)
Show the key derivation or formula. Explain every symbol. Walk through step by step. Beginners see "there's math behind this" and absorb the structure; intermediates follow the logic.

**Example** (bias-variance):
> Expected prediction error = Bias² + Variance + Irreducible Noise
> E[(y - ŷ)²] = (E[ŷ] - f(x))² + E[(ŷ - E[ŷ])²] + σ²
> [annotated: each term labeled, color-coded]

### Layer 3: Advanced Insight (experts engage)
The subtle point, the counter-intuition, the connection to cutting-edge research, the production gotcha. Often a single slide or a callout box within a Layer 2 slide.

**Example** (bias-variance):
> "In the interpolation regime (over-parameterized models), the classical bias-variance trade-off breaks down. Double descent means test error can DECREASE as you go past the interpolation threshold. (Belkin et al., 2019)"

### Slide Markers

Use visual markers so students can self-navigate:
- 🟢 **FOUNDATIONS** — everyone must understand this
- 🔵 **THEORY** — mathematical derivation, follow if you can
- 🟣 **ADVANCED** — cutting-edge insight, production nuance

These markers appear as colored dots in the top-right corner of each slide.

## Format

- **Framework**: Reveal.js 5.1.0
- **Canvas**: 1280×720 pixels
- **Export**: `?print-pdf` URL parameter for slide-by-slide PDF
- **Theme**: Custom CSS at `decks/assets/css/theme.css`
- **Nested slides**: Use vertical slides (Reveal.js `<section>` nesting) for depth layers — horizontal navigation for main flow, vertical for going deeper into a concept

## Color Palette

| Role | Color | Hex |
|------|-------|-----|
| Primary accent | Teal | `#0D9488` |
| Depth accent | Indigo | `#4F46E5` |
| Background | White/Slate | `#FFFFFF` / `#F8FAFC` |
| Text | Slate 800 | `#1E293B` |
| Code | Slate 700 on Slate 50 | `#334155` on `#F8FAFC` |
| Alert/Warning | Amber | `#F59E0B` |
| Error/Critical | Rose | `#F43F5E` |
| Success | Emerald | `#10B981` |
| Foundations marker | Green | `#22C55E` |
| Theory marker | Blue | `#3B82F6` |
| Advanced marker | Purple | `#8B5CF6` |

## Slide Design Principles

1. **One idea per slide** — if more than 3 bullet points, split
2. **Three-layer depth** — every concept gets intuition → math → advanced insight
3. **Questions over statements** — provoke thinking, not delivery
4. **Cases over theory** — every concept introduced with a real failure or success story
5. **Math with intuition** — never show a formula without first explaining what it means in plain English and WHY it matters
6. **Annotated equations** — color-code terms, label each component, show what happens when you change each variable
7. **Active language** — "What would you do?" not "Here is the answer"
8. **Kailash on every concept** — after showing the math, show how the Kailash engine implements it. The theory-to-engine bridge is on every section's final slide.
9. **Singapore context** — local data, local regulations, local market
10. **Code live, not on slides** — slides set up the problem, the lab solves it. Exception: Kailash import patterns and 3-line engine calls ARE shown on slides.
11. **Visual diagrams over text** — architecture diagrams, data flow diagrams, decision trees for "which engine to use"

## Deck Structure per Module

Each module deck is **self-contained** — a student who missed prior modules can still follow if they read the recap slides.

```
1. TITLE SLIDE
   Module N: [Title]
   [Provocative question that frames the session]

2. RECAP (3-5 slides)
   "Last time we learned..." — key concepts from prior modules
   Kailash engines we've used so far (cumulative diagram)
   What's new today (preview)

3. OPENING CASE (2-3 slides)
   Real-world story that motivates the module
   "This is why today's topic matters in production"

4. FOUNDATIONS BLOCK (10-15 slides) 🟢
   Start from basics. Assume zero knowledge of this topic.
   Build vocabulary, mental models, visual intuitions.
   Every slide has a plain-English explanation.
   End with: "Now let's see the math behind this."

5. THEORY BLOCK A (15-20 slides) 🔵
   First lecture section — math + intuition side by side
   Derivations shown step-by-step with annotated equations
   Connection to Kailash engine at the end of each sub-section
   Advanced callouts (🟣) on 2-3 slides for expert engagement

6. THEORY BLOCK B (15-20 slides) 🔵
   Second lecture section — same pattern

7. THEORY BLOCK C (8-12 slides) 🔵
   Third lecture section — typically the Kailash-specific content
   How the theory maps to engines, API patterns, production usage

8. KAILASH ENGINE DEEP DIVE (5-8 slides)
   Architecture diagram: what the engine does internally
   API surface: key methods, parameters, return types
   "Theory → Engine" mapping table
   Side-by-side: mathematical concept ↔ Kailash API call
   Governance integration: how this engine fits into the trust plane

9. LAB SETUP (3-5 slides)
   Problem statement, dataset description, expected outputs
   Exercise progression overview
   "Open modules/ascentNN/local/01_exercise.py"

10. DISCUSSION PROMPTS (2-3 slides)
    "Given this SHAP output, would you approve this loan?"
    "Your DriftMonitor fires at 3am. What's your runbook?"

11. SYNTHESIS (3-5 slides)
    Key takeaways by level:
    - 🟢 "Everyone should remember..."
    - 🔵 "If you followed the math..."
    - 🟣 "For the experts..."
    Kailash cumulative map (updated with this module's engines)
    Connection to next module

12. ASSESSMENT PREVIEW (1 slide)
    Quiz topics, assignment connection
```

**Target**: 80-100 slides per module (covering 3 hours of lecture). The increase from 50-60 is because every concept now has 3 layers of depth.

## Module-Specific Content Plan

### Module 1: Statistics, Probability & Data Fluency (80-90 slides)

**Foundations** 🟢 (must cover for zero-background):
- What is data? Types (numeric, categorical, ordinal, temporal)
- What is a distribution? (histogram → probability density, visual)
- Mean, median, mode — when each fails
- What is correlation? (scatter plots, not formulas first)
- What is Python? What is polars? Why not Excel?
- What is Kailash? Why use an SDK instead of jupyter + sklearn?

**Theory** 🔵:
- Probability: distributions (Normal, Poisson, Exponential), exponential family, sufficient statistics
- MLE: derivation for Gaussian, properties, Fisher information, Cramér-Rao bound
- Bayesian: prior → posterior → predictive, conjugate priors (Normal-Normal, Beta-Binomial), MAP estimation
- Hypothesis testing: Neyman-Pearson, power analysis, p-value interpretation pitfalls, FDR
- Bootstrap: Efron's insight, BCa intervals, when bootstrap fails

**Advanced** 🟣:
- Information geometry (Fisher metric as Riemannian metric on distribution manifold)
- Bayesian model comparison: BIC vs WAIC vs LOO-CV
- Permutation tests vs parametric tests
- Wild bootstrap for heteroscedastic data

**Kailash bridge**:
- DataExplorer: "This is what you just learned about distributions — DataExplorer computes all of it in one async call"
- PreprocessingPipeline: "This is what you learned about data types — PreprocessingPipeline auto-detects and transforms"
- ConnectionManager: first introduction — "This is where all your results will be stored"

### Module 2: Feature Engineering & Experiment Design (80-90 slides)

**Foundations** 🟢:
- What is a feature? (column in your data → input to your model)
- Why does feature engineering matter? (garbage in, garbage out)
- What is an experiment? What is causation vs correlation?
- What is an A/B test? (simple: show two groups different things, measure difference)

**Theory** 🔵:
- Feature selection: mutual information derivation, Boruta algorithm, stability selection
- Temporal features: lag, rolling, Fourier — point-in-time correctness (with visual timeline showing leakage)
- Target encoding: why naive encoding overfits, James-Stein shrinkage
- Causal inference: Rubin's potential outcomes (Y(1), Y(0)), ATE/ATT/CATE
- Pearl's DAGs: d-separation, backdoor criterion, front-door criterion
- A/B testing: power analysis formula, MDE, sample size calculation
- CUPED: regression adjustment derivation, variance reduction proof
- Diff-in-diff: parallel trends assumption, visualization

**Advanced** 🟣:
- Double ML / Debiased ML (Chernozhukov et al.)
- Heterogeneous treatment effects (CATE estimation with causal forests)
- Bayesian A/B testing (posterior probability of improvement)
- Interference and network effects in experiments

**Kailash bridge**:
- FeatureSchema: "This is how you declare feature contracts — typed, versioned, auditable"
- FeatureStore: "Point-in-time correctness built into the engine — impossible to leak"
- ExperimentTracker: "Every experiment logged, comparable, reproducible"

### Module 3: Supervised ML — Theory to Production (90-100 slides)

**Foundations** 🟢:
- What is prediction? (given inputs, guess the output)
- Regression vs classification (continuous vs discrete target)
- Training vs testing (why you can't grade your own homework)
- What is overfitting? (memorizing vs learning — visual with curves)
- What is a decision tree? (visual, split-by-split)

**Theory** 🔵:
- Bias-variance decomposition: full derivation for squared loss
- Regularization: L1/L2 geometry (diamond vs circle constraint), Bayesian interpretation
- Decision trees → random forests → gradient boosting (evolution)
- XGBoost: 2nd-order Taylor expansion of objective
- LightGBM: histogram-based splitting, GOSS
- Class imbalance: SMOTE (and failure modes), cost-sensitive learning, Focal Loss derivation
- Calibration: Platt scaling, isotonic regression, reliability diagram, ECE
- SHAP: Shapley axioms (efficiency, symmetry, dummy, linearity), TreeSHAP algorithm
- Model cards: Mitchell et al. template, filled example

**Advanced** 🟣:
- Double descent and benign overfitting
- NGBoost (natural gradient boosting for uncertainty)
- Proper scoring rules (why Brier score IS proper, accuracy IS NOT)
- Counterfactual explanations (Wachter et al.)

**Kailash bridge**:
- TrainingPipeline + ModelSpec + EvalSpec: "One config object describes your entire experiment"
- HyperparameterSearch: "Bayesian optimization finds the best model without exhaustive search"
- ModelRegistry: "Models have a lifecycle — staging → shadow → production → archived"
- WorkflowBuilder: "Connect steps into reproducible pipelines"
- DataFlow: "Persist everything — models, metrics, SHAP values"

### Module 4: Unsupervised ML, NLP & Deep Learning (90-100 slides)

**Foundations** 🟢:
- What is clustering? (grouping similar things — visual with customer segments)
- What is dimensionality reduction? (projecting 100 columns to 2 so you can see patterns)
- What is NLP? (teaching computers to read — from bag of words to transformers)
- What is a neural network? (layers of simple math that together learn complex patterns)
- What is deep learning? (neural networks with many layers)

**Theory** 🔵:
- K-means: Lloyd's algorithm, convergence, k-means++ initialization
- Spectral clustering: graph Laplacian construction, eigengap heuristic, normalized cuts
- HDBSCAN: density-based hierarchy, mutual reachability distance
- GMM/EM: full derivation (E-step posterior, M-step MLE, convergence guarantee)
- PCA: eigendecomposition, connection to SVD, scree plot
- t-SNE: KL divergence objective, perplexity tuning
- UMAP: topological data analysis intuition, cross-entropy loss
- Word2Vec: skip-gram objective with negative sampling
- BERTopic: UMAP + HDBSCAN + c-TF-IDF pipeline
- Neural networks: universal approximation theorem, backpropagation
- Attention: scaled dot-product derivation from first principles
- CNN: convolution as learned feature extraction, receptive field theory
- LSTM: gate mechanism (forget, input, output), gradient flow

**Advanced** 🟣:
- Information-theoretic clustering
- Kernel PCA, ICA
- Flash attention (IO-aware algorithm)
- Sharpness-aware minimization (SAM)
- Vision Transformers (ViT): patch embedding

**Kailash bridge**:
- AutoMLEngine: "Compare clustering algorithms programmatically"
- EnsembleEngine: "Combine multiple models — blending, stacking, bagging, boosting"
- DriftMonitor: "Your model in production — PSI and KS detect when data shifts"
- InferenceServer: "Serve predictions with caching and ONNX optimization"
- Nexus: "Deploy as API + CLI + MCP with one command"

### Module 5: LLMs, AI Agents & RAG Systems (90-100 slides)

**Foundations** 🟢:
- What is a language model? (predicts the next word — that's it)
- What is a token? (not a word — show BPE splitting visually)
- What is an API call to an LLM? (send text, get text back)
- What is an agent? (LLM + tools + memory — can take actions, not just talk)
- What is RAG? (look up relevant documents, then answer with context)

**Theory** 🔵:
- Transformer: encoder-decoder, self-attention step-by-step (Q, K, V matrices)
- Why scale by √d_k? (prevent softmax saturation — show numerically)
- Multi-head attention: parallel attention "heads" see different relationships
- Positional encoding: sinusoidal (derivation), RoPE (rotation matrix), ALiBi (linear bias)
- Pre-training: MLM (BERT), CLM (GPT), span corruption (T5)
- Scaling laws: Chinchilla optimal (compute = 6ND tokens), emergent abilities
- RAG: chunking strategies, embedding models, hybrid retrieval, re-ranking
- RAG evaluation: RAGAS framework (faithfulness, relevance, context precision/recall)
- Agent architecture: Signature contracts, tool use formalization, ReAct loop
- Multi-agent: A2A protocol, supervisor-worker, debate patterns
- Agent safety: prompt injection taxonomy, output validation, cost budgets

**Advanced** 🟣:
- Flash attention (IO-aware algorithm, tiling for GPU memory)
- Grouped-query attention (GQA), multi-query attention (MQA)
- Speculative decoding (draft model + verify)
- Self-RAG, corrective RAG (CRAG)
- Tree-of-thought, graph-of-thought planning
- Agent evaluation frameworks (AgentBench, WebArena)

**Kailash bridge**:
- Delegate: "The recommended way to build agents — engine-level"
- BaseAgent + Signature: "When you need custom agent logic — primitive-level"
- 6 ML agents: "LLMs augmenting your ML workflow — DataScientist suggests features, ModelSelector picks models"
- Cost budgets: "Every agent has a spending limit — enforced by Kailash, not by trust"

### Module 6: Alignment, Governance, RL & Deployment (90-100 slides)

**Foundations** 🟢:
- What is fine-tuning? (teaching a pre-trained model new things — like giving a new employee training)
- What is alignment? (making AI do what humans actually want, not just what they literally said)
- What is governance? (rules about who can do what, enforced automatically)
- What is RL? (learning by trial and error — reward for good actions, penalty for bad)
- What is deployment? (making your model available to users — API, CLI, dashboard)

**Theory** 🔵:
- LoRA: low-rank approximation theory, why rank r << d works (show matrix dimensions)
- QLoRA: NF4 quantization, double quantization, 4-bit training
- DPO: derive from RLHF — Bradley-Terry preference model → closed-form solution (full derivation)
- GRPO: group relative policy optimization (DeepSeek-R1 approach)
- Evaluation: perplexity, BLEU, ROUGE, BERTScore, LLM-as-judge (position bias, verbosity bias)
- EU AI Act: risk tiers (minimal/limited/high/unacceptable), Art. 6, 9, 13, 52
- Singapore AI Verify: ISAGO 2.0, testing toolkit components
- PACT D/T/R: Department/Team/Role addressing, monotonic tightening, fail-closed
- Operating envelopes: role envelope ∩ task envelope = effective envelope
- Knowledge clearance: 5 levels, classification independent of authority
- RL: Bellman expectation AND optimality equations (both derived)
- PPO: clipped objective derivation, advantage estimation (GAE)
- SAC: maximum entropy RL, temperature parameter

**Advanced** 🟣:
- ORPO, SimPO, KTO (latest alignment methods)
- Model merging (TIES, DARE, model soups) as alternative to fine-tuning
- Offline RL (Conservative Q-Learning, Decision Transformer)
- RLHF as RL application (connecting 6A alignment to 6C RL)
- Differential privacy (DP-SGD, privacy budgets)
- Algorithmic impact assessment methodology

**Kailash bridge**:
- AlignmentPipeline: "SFT, DPO, QLoRA — one unified API"
- AdapterRegistry: "Track adapters through their lifecycle — training → eval → merge → GGUF → deploy"
- GovernanceEngine: "Define your org, compile it, enforce it"
- PactGovernedAgent: "Wrap any agent with governance — cost limits, tool restrictions, data access"
- RLTrainer: "PPO, SAC, DQN — backed by Stable-Baselines3"
- Nexus: "Full platform — API + CLI + MCP, governed by PACT"
- TrustPosture + ConfidentialityLevel: "The trust plane underneath everything"

## Speaker Notes Format

Each module has `speaker-notes.md` with:

```markdown
## Slide N: [Title]
**Time**: ~2 min
**Talking points**:
- Key point to emphasize
- Common student question and answer
- If beginners look confused: [simplified re-explanation]
- If experts look bored: [advanced tangent to mention]
**Transition**: "Now that we understand X, let's see how Y builds on it..."
```

**Timing annotations**: Every slide block has a target time. Total must sum to ~180 minutes (3h lecture).

## Module-Specific Opening Cases

| Module | Opening Case | Point | Audience Hook |
|--------|-------------|-------|---------------|
| 1 | Singapore HDB flash crash (Q4 2023 price anomaly) | EDA catches what dashboards miss | "This affected YOUR property value" |
| 2 | Healthcare feature leakage (target in features) | Point-in-time correctness is non-negotiable | "This killed a clinical trial" |
| 3 | Zillow iBuyer $500M write-off | Accurate but uncalibrated models destroy value | "They were RIGHT but still lost half a billion" |
| 4 | Credit Suisse AML — 99.9% accuracy, 0% useful | Class imbalance requires principled approaches | "Would YOU trust a system that cries wolf 10,000 times a day?" |
| 5 | BloombergGPT — domain agents beat general models | Specialized agents with tools > monolithic LLMs | "Bloomberg's model was 10x smaller but beat GPT-3 on finance" |
| 6 | EU AI Act first fines expected 2025 | Governance is competitive advantage | "Your competitors are scrambling. You'll be ready." |

## File Structure

```
decks/
├── assets/
│   ├── css/theme.css          # Reveal.js custom theme (dual-track markers)
│   ├── img/                   # Shared images, diagrams, architecture visuals
│   └── diagrams/              # Kailash architecture diagrams, engine flow charts
├── ascent01/
│   ├── deck.html              # Reveal.js slides (80-100 slides)
│   └── speaker-notes.md       # Instructor notes with timing + dual-audience tips
├── ascent02/ through ascent06/     # Same structure
└── README.md                  # How to present, PDF export, timing guide
```

## Build Process

1. Session design in `workspaces/ascentNN/02-plans/` (content outline)
2. `/build-deck` generates deck from session plan + module brief
3. Review: does every concept have 3 layers (intuition → math → advanced)?
4. Review: is Kailash bridged on every theory section?
5. Speaker notes with timing + dual-audience tips
6. Test with `python -m http.server` for local Reveal.js rendering
7. Export PDF via `?print-pdf` for distribution

## Quality Checklist per Deck

- [ ] Every concept has a 🟢 foundations slide (beginner-accessible)
- [ ] Every concept has a 🔵 theory slide (math + intuition)
- [ ] At least 5 slides per module have 🟣 advanced callouts
- [ ] Every theory section ends with a "Kailash bridge" slide
- [ ] Recap slides reference all prior modules' engines
- [ ] Opening case is a real story with source citation
- [ ] Total slides: 80-100 per module
- [ ] Total timing: ~180 minutes per module
- [ ] No formula without plain-English explanation first
- [ ] No Kailash engine without its mathematical foundation
- [ ] Speaker notes include dual-audience tips on every slide block
