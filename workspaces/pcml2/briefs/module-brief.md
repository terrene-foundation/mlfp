# Module 2: Feature Engineering & Experiment Design

**Duration**: 7 hours  
**Kailash**: kailash-ml (FeatureStore, FeatureEngineer, ExperimentTracker)  
**Scaffolding**: 60%

## Lecture Topics

### 2A: Feature Engineering Theory (90 min)
- Feature selection: mutual information, Boruta, RFE, stability selection
- Collinearity: VIF, condition number, eigenvalue analysis
- Interactions: polynomial features, tree-based interaction detection
- Temporal features: lag, rolling stats, Fourier seasonality, point-in-time correctness (leakage prevention)
- Target encoding: James-Stein shrinkage, hierarchical encoding, CV-based encoding
- Domain-specific: RFM (retail), technical indicators (finance), clinical extraction (healthcare)

### 2B: Experiment Design & Causal Inference (90 min)
- **Metric design hierarchy** (P0): guardrail / driver / primary metrics — how Netflix, Airbnb, Bing structure experimentation
- A/B testing: power analysis, MDE, sample size calculation, multi-armed bandits (Thompson sampling, UCB), sequential testing (always-valid p-values)
- **Bayesian A/B testing** (P0): posterior probability of improvement, credible intervals for lift, expected loss — now standard at Spotify, Dynamic Yield, Statsig/Eppo
- **CUPED math** (P0): derive Var(Y_adj) = Var(Y)(1 - ρ²) — students need this formula to answer quiz question on CI width reduction
- Variance reduction: CUPED/CUPAC (regression adjustment derivation, pre-experiment covariate selection), stratification
- Causal inference: potential outcomes (Rubin), ATE/ATT/CATE, DAGs (Pearl), d-separation, backdoor criterion, do-calculus
- **Heterogeneous treatment effects** (P0): CATE estimation via meta-learners (S/T/X-learner), causal forests — "for WHOM is the effect largest?" — bridge between causal inference and personalization
- Quasi-experimental: diff-in-diff (parallel trends assumption, visualization), regression discontinuity, instrumental variables, propensity score matching
- Double ML: ML for nuisance parameter estimation (Chernozhukov et al.), orthogonal estimation

### 2C: Feature Management at Scale + Data Lineage (30 min)
- FeatureSchema: typed fields, entity IDs, timestamps
- FeatureStore: persist, version, retrieve with point-in-time correctness — **frame as data lineage**: "If a regulator asks 'what data did this model see?', you can answer."
- ExperimentTracker: log, compare, reproduce — **frame as governance artifact**: "Every experiment is auditable."
- **Model provenance** concept: where did this model come from, what data trained it, who approved it?
- EATP trust protocol: 1 slide on model provenance and audit chains (seeds Module 6)

## Lab Exercises (5)

All exercises use ExperimentTracker from Exercise 1 (cumulative experiment history):

1. **Healthcare feature engineering**: Engineer clinical features from messy ICU data. **Start with** `ExperimentTracker.create_experiment("healthcare_features")` — every subsequent exercise adds runs.
2. **FeatureStore lifecycle**: Define FeatureSchema → compute → version → retrieve at different timestamps. Demonstrate leakage prevention. **Explicitly discuss data lineage**.
3. **A/B test analysis**: Full experiment on e-commerce data — power analysis, SRM check, CUPED variance reduction, multiple metric correction. Track as experiment run.
4. **Causal inference**: Diff-in-diff on Singapore housing (before/after cooling measures). Track as experiment run.
5. **FeatureEngineer + ExperimentTracker**: Automated feature generation (interaction, polynomial, binning), review full experiment history from all M2 exercises.

## Datasets
- **Healthcare ICU** (MIMIC-style synthetic): 60K stays, irregular vitals, multi-table, clinical missing patterns
- **E-commerce Experiment**: 500K users, A/B test with SRM issues, sequential testing data
- **Singapore Housing + Policy**: HDB prices + cooling measure dates for causal analysis

## Quiz Topics
- Point-in-time correctness: "What's wrong with this feature pipeline?" (leakage identification)
- Power analysis calculation
- VIF interpretation
- CUPED: "By how much would this reduce your CI width?"
- FeatureStore API: correct versioning pattern

## Deck Opening Case
**Healthcare feature leakage disaster** — a published clinical ML model included the target variable (mortality) as an input feature through a chain of lookups. Model showed 99% AUC in development, 51% in production. Point-in-time correctness is non-negotiable.
