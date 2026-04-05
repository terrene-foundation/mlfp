# Expert Review: Kailash SDK Coverage in ASCENT Curriculum

**Reviewer**: Chief Architect, Kailash Python SDK  
**Date**: 2026-04-05  
**Scope**: All 8 Kailash packages, all 6 modules, all exercises  
**Verdict**: Curriculum is strong on ML engines but has critical gaps in governance-from-day-1, platform infrastructure, and several SDK capabilities that graduates will need in production.

---

## 1. SDK Coverage Matrix

### 1.1 kailash-ml (13 engines + bridge + compat + dashboard)

| Export | Taught in Exercise? | Mentioned in Lecture? | Module | Status |
|--------|--------------------|-----------------------|--------|--------|
| `DataExplorer` | M1-Ex3, M1-Ex5 | M1-1C | M1 | COVERED |
| `PreprocessingPipeline` | M1-Ex5 (implicit) | M1-1C | M1 | COVERED |
| `ModelVisualizer` | M1-Ex5, M5-Ex3 (tool) | M1-1C | M1,M5 | COVERED |
| `FeatureStore` | M2-Ex2 | M2-2C | M2 | COVERED |
| `FeatureEngineer` | M2-Ex5, M5-Ex3 (tool) | M2-2A (implicit) | M2,M5 | COVERED |
| `ExperimentTracker` | M2-Ex5, M6-Ex5 | M2-2C | M2,M6 | COVERED |
| `TrainingPipeline` | M3-Ex4, M3-Ex6 | M3-3C | M3 | COVERED |
| `ModelSpec` | M3-Ex4 | M3-3C | M3 | COVERED |
| `EvalSpec` | M3-Ex4 | M3-3C | M3 | COVERED |
| `HyperparameterSearch` | M3-Ex5 | M3 (implicit) | M3 | COVERED |
| `SearchSpace` | -- | -- | -- | ABSENT |
| `SearchConfig` | -- | -- | -- | ABSENT |
| `ParamDistribution` | -- | -- | -- | ABSENT |
| `AutoMLEngine` | M4-Ex1 | M4 header | M4 | COVERED |
| `AutoMLConfig` | -- | -- | -- | ABSENT (needed for agent opt-in) |
| `EnsembleEngine` | M4-Ex2 | M3-3A (theory) | M3-4 | PARTIALLY COVERED |
| `ModelRegistry` | M3-Ex5 | M3-3C | M3 | COVERED |
| `LocalFileArtifactStore` | -- | -- | -- | ABSENT |
| `InferenceServer` | M4-Ex6 | M4 header | M4 | COVERED |
| `DriftMonitor` | M4-Ex4 | M4 header | M4 | COVERED |
| `DriftSpec` | -- | -- | -- | ABSENT |
| `OnnxBridge` | -- | -- | -- | ABSENT |
| `MlflowFormatReader` | -- | -- | -- | ABSENT |
| `MlflowFormatWriter` | -- | -- | -- | ABSENT |
| `MLDashboard` | -- | -- | -- | ABSENT |
| `AlertConfig` | -- | M1-1C (8 alert types) | M1 | LECTURE ONLY |
| `ExperimentalWarning` | -- | -- | -- | N/A (internal) |

**kailash_ml.types:**

| Export | Taught? | Module | Status |
|--------|---------|--------|--------|
| `FeatureSchema` | M2-Ex2 | M2 | COVERED |
| `FeatureField` | M2-Ex2 | M2 | COVERED |
| `ModelSignature` | -- | -- | ABSENT |
| `MetricSpec` | -- | -- | ABSENT |
| `MLToolProtocol` | -- | -- | ABSENT |
| `AgentInfusionProtocol` | -- | -- | ABSENT |

**kailash_ml.interop:**

| Export | Taught? | Module | Status |
|--------|---------|--------|--------|
| `to_sklearn_input` | -- | -- | ABSENT |
| `from_sklearn_output` | -- | -- | ABSENT |
| `to_lgb_dataset` | -- | -- | ABSENT |
| `to_hf_dataset` | -- | -- | ABSENT |
| `polars_to_arrow` | -- | -- | ABSENT |
| `from_arrow` | -- | -- | ABSENT |
| `to_pandas` | -- | -- | ABSENT |
| `from_pandas` | -- | -- | ABSENT |
| `polars_to_dict_records` | -- | -- | ABSENT |
| `dict_records_to_polars` | -- | -- | ABSENT |

**kailash_ml.rl:**

| Export | Taught? | Module | Status |
|--------|---------|--------|--------|
| `RLTrainer` | M6-Ex5 | M6-6C | COVERED |
| `EnvironmentRegistry` | -- | -- | ABSENT |
| `PolicyRegistry` | -- | -- | ABSENT |
| `RLTrainingConfig` | -- | -- | ABSENT |

**kailash_ml.agents:**

| Export | Taught? | Module | Status |
|--------|---------|--------|--------|
| `DataScientistAgent` | M5-Ex5 | M5-5C | COVERED |
| `FeatureEngineerAgent` | M5-Ex5 | M5-5C | COVERED |
| `ModelSelectorAgent` | M5-Ex5 | M5-5C | COVERED |
| `ExperimentInterpreterAgent` | -- | M5-5C (mentioned) | M5 | LECTURE ONLY |
| `DriftAnalystAgent` | -- | M5-5C (mentioned) | M5 | LECTURE ONLY |
| `RetrainingDecisionAgent` | -- | M5-5C (mentioned) | M5 | LECTURE ONLY |
| Agent tools module | -- | -- | -- | ABSENT |

### 1.2 kailash (Core SDK)

| Export | Taught? | Module | Status |
|--------|---------|--------|--------|
| `WorkflowBuilder` | M3-Ex4 | M3-3C | COVERED |
| `LocalRuntime` | M3-Ex4 | M3-3C | COVERED |
| `Workflow` | -- | -- | ABSENT |
| `NodeInstance` | -- | -- | ABSENT |
| `Connection` | -- | -- | ABSENT |
| `Node` (base class) | -- | -- | ABSENT |
| `NodeParameter` | -- | -- | ABSENT |
| `NodeMetadata` | -- | -- | ABSENT |
| `WorkflowVisualizer` | -- | -- | ABSENT |
| `AsyncLocalRuntime` | -- | M3-3C (implied) | -- | ABSENT |
| `WorkflowServer` | -- | -- | ABSENT |
| Node categories (20+) | -- | -- | ABSENT |

**kailash.db.connection:**

| Export | Taught? | Module | Status |
|--------|---------|--------|--------|
| `ConnectionManager` | -- | -- | **CRITICALLY ABSENT** |

### 1.3 kailash-dataflow

| Export | Taught? | Module | Status |
|--------|---------|--------|--------|
| `DataFlow` | M3-Ex4 (implicit) | M3-3C | PARTIALLY COVERED |
| `@db.model` | M3-Ex4 (implied) | M3-3C | LECTURE ONLY |
| `db.express` | -- | M3-3C (mentioned) | M3 | LECTURE ONLY |
| `DataFlowEngine` | -- | -- | ABSENT |
| `QueryEngine` | -- | -- | ABSENT |
| `ValidationLayer` | -- | -- | ABSENT |
| `DataClassificationPolicy` | -- | -- | ABSENT |
| `ProgressiveConfiguration` | -- | -- | ABSENT |
| `zero_config` / `production_config` / `enterprise_config` | -- | -- | ABSENT |
| `field_validator` / `validate_model` | -- | -- | ABSENT |
| `DataClassification` / `classify` | -- | -- | ABSENT |
| `Provenance` / `ProvenanceMetadata` | -- | -- | ABSENT |
| `TenantContextSwitch` / `TenantInfo` | -- | -- | ABSENT |
| `SyncExpress` | -- | -- | ABSENT |
| `QueryBuilder` (MongoDB-style) | -- | -- | ABSENT |
| Bulk operations (BulkCreate, BulkUpdate, BulkDelete) | -- | -- | ABSENT |

### 1.4 kailash-nexus

| Export | Taught? | Module | Status |
|--------|---------|--------|--------|
| `Nexus` | M4-Ex6, M6-Ex6 | M4, M6 | COVERED |
| `NexusEngine` | -- | -- | ABSENT |
| `Preset` / `PRESETS` / `get_preset` | -- | -- | ABSENT |
| `NexusConfig` / `PresetConfig` | -- | -- | ABSENT |
| `NexusPluginProtocol` | -- | -- | ABSENT |
| `create_nexus()` | -- | -- | ABSENT |
| `Transport` / `HTTPTransport` / `MCPTransport` | -- | -- | ABSENT |
| `HandlerDef` / `HandlerParam` / `HandlerRegistry` | -- | -- | ABSENT |
| `EventBus` / `NexusEvent` / `NexusEventType` | -- | -- | ABSENT |
| `BackgroundService` | -- | -- | ABSENT |
| `NexusFile` | -- | -- | ABSENT |
| `ProbeManager` / `ProbeState` | -- | -- | ABSENT |
| `OpenApiGenerator` / `OpenApiInfo` | -- | -- | ABSENT |
| `register_metrics_endpoint` | -- | -- | ABSENT |
| `register_sse_endpoint` | -- | -- | ABSENT |
| `app.create_session()` | -- | -- | ABSENT |

### 1.5 kailash-kaizen

| Export | Taught? | Module | Status |
|--------|---------|--------|--------|
| `Kaizen` (framework) | -- | -- | ABSENT |
| `KaizenConfig` | -- | -- | ABSENT |
| `Agent` (unified API) | -- | -- | ABSENT |
| `AgentManager` | -- | -- | ABSENT |
| `Signature` | M5-Ex1 | M5-5C | COVERED |
| `InputField` | M5-Ex1 | M5-5C | COVERED |
| `OutputField` | M5-Ex1 | M5-5C | COVERED |
| `configure()` / `create_agent()` | -- | -- | ABSENT |
| A2A types (`A2AAgentCard`, `A2ATask`, etc.) | -- | M5-5C (mentioned) | M5 | LECTURE ONLY |
| A2A factory functions | -- | -- | ABSENT |

### 1.6 kaizen-agents

| Export | Taught? | Module | Status |
|--------|---------|--------|--------|
| `Delegate` | M5-Ex1 | M5-5C | COVERED |
| `GovernedSupervisor` | -- | -- | ABSENT |
| `Agent` (async API) | -- | -- | ABSENT |
| `ReActAgent` | M5-Ex3 | M5-5C | COVERED |
| `Pipeline` | -- | -- | ABSENT |
| `SimpleQAAgent` | M5-Ex1 | M5-5C | COVERED |
| `ChainOfThoughtAgent` | M5-Ex2 | M5-5C | COVERED |
| `RAGResearchAgent` | M5-Ex4 | M5-5C | COVERED |
| `TreeOfThoughtsAgent` | -- | -- | ABSENT |
| `PlanningAgent` | -- | -- | ABSENT |
| `SelfReflectionAgent` | -- | -- | ABSENT |
| `StreamingChatAgent` | -- | -- | ABSENT |
| `HumanApprovalAgent` | -- | -- | ABSENT |
| `MemoryAgent` | -- | -- | ABSENT |
| `BatchProcessingAgent` | -- | -- | ABSENT |
| `CodeGenerationAgent` | -- | -- | ABSENT |
| `ResilientAgent` | -- | -- | ABSENT |
| `PEVAgent` | -- | -- | ABSENT |

### 1.7 kailash-pact

| Export | Taught? | Module | Status |
|--------|---------|--------|--------|
| `GovernanceEngine` | M6-Ex3 | M6-6B | COVERED |
| `GovernanceContext` | M6-Ex4 | M6-6B | COVERED |
| `PactGovernedAgent` | M6-Ex4 | M6-6B | COVERED |
| `Address` | M6-Ex3 | M6-6B | COVERED |
| `RoleEnvelope` | M6-Ex4 | M6-6B | COVERED |
| `TaskEnvelope` | M6-Ex4 | M6-6B | COVERED |
| `compile_org` | M6-Ex3 | M6-6B | COVERED |
| `load_org_yaml` | M6-Ex3 | M6-6B | COVERED |
| `can_access` | M6-Ex3 | M6-6B | COVERED |
| `explain_access` | M6-Ex3 | M6-6B | COVERED |
| `intersect_envelopes` | M6-Ex4 | M6-6B | COVERED |
| `AuditChain` | -- | -- | ABSENT |
| `GradientEngine` | -- | -- | ABSENT |
| `McpGovernanceEnforcer` | -- | -- | ABSENT |
| `McpGovernanceMiddleware` | -- | -- | ABSENT |
| `McpToolPolicy` | -- | -- | ABSENT |
| `VacancyStatus` | -- | -- | ABSENT |
| `VettingStatus` | -- | -- | ABSENT |
| `PactEngine` (Dual Plane bridge) | -- | -- | ABSENT |
| `CostTracker` | -- | -- | ABSENT |
| `EventBus` | -- | -- | ABSENT |
| `EnforcementMode` | -- | -- | ABSENT |
| `PactGovernanceMiddleware` | -- | -- | ABSENT |
| `governed_tool` decorator | -- | -- | ABSENT |
| Knowledge clearance (5 levels) | -- | M6-6B (mentioned) | M6 | LECTURE ONLY |
| `POSTURE_CEILING` | -- | -- | ABSENT |
| `RoleClearance` | -- | -- | ABSENT |
| `effective_clearance` | -- | -- | ABSENT |
| Constraint dimension configs (Financial, Operational, Temporal, DataAccess, Communication) | -- | -- | ABSENT |
| Store protocols (OrgStore, EnvelopeStore, ClearanceStore, AccessPolicyStore) | -- | -- | ABSENT |

### 1.8 kailash-align

| Export | Taught? | Module | Status |
|--------|---------|--------|--------|
| `AlignmentPipeline` | M6-Ex1 | M6-6A | COVERED |
| `AlignmentConfig` | M6-Ex1 | M6-6A | COVERED |
| `AdapterRegistry` | M6-Ex1 | M6-6A | COVERED |
| `AlignmentResult` | -- | -- | ABSENT |
| `LoRAConfig` | -- | M6-6A (LoRA theory) | M6 | LECTURE ONLY |
| `SFTConfig` | -- | -- | ABSENT |
| `DPOConfig` | -- | -- | ABSENT |
| `KTOConfig` | -- | -- | ABSENT |
| `ORPOConfig` | -- | -- | ABSENT |
| `GRPOConfig` | -- | -- | ABSENT |
| `RLOOConfig` | -- | -- | ABSENT |
| `OnlineDPOConfig` | -- | -- | ABSENT |
| `AdapterSignature` | -- | -- | ABSENT |
| `AlignmentEvaluator` | -- | M6-6A (eval theory) | M6 | LECTURE ONLY |
| `EvalConfig` | -- | -- | ABSENT |
| `AlignmentServing` | -- | -- | ABSENT |
| `ServingConfig` | -- | -- | ABSENT |
| `AdapterMerger` | -- | -- | ABSENT |
| `KaizenModelBridge` | -- | -- | ABSENT |
| `OnPremModelCache` | -- | -- | ABSENT |
| `VLLMBackend` / `VLLMConfig` | -- | -- | ABSENT |
| `HFGenerationBackend` | -- | -- | ABSENT |
| `GPUMemoryEstimate` / `estimate_training_memory` | -- | -- | ABSENT |
| `RewardRegistry` | -- | -- | ABSENT |
| `METHOD_REGISTRY` | -- | -- | ABSENT |

### 1.9 kailash.trust (Core Trust Plane)

| Export | Taught? | Module | Status |
|--------|---------|--------|--------|
| `TrustPosture` | -- | -- | ABSENT |
| `ConfidentialityLevel` | -- | -- | ABSENT |
| `CapabilityAttestation` | -- | -- | ABSENT |
| `AuditAnchor` | -- | -- | ABSENT |
| `GenesisRecord` | -- | -- | ABSENT |
| `TrustLineageChain` | -- | -- | ABSENT |
| `TrustOperations` | -- | -- | ABSENT |
| `TrustKeyManager` | -- | -- | ABSENT |
| `PostureStateMachine` | -- | -- | ABSENT |
| `TrustRole` | -- | -- | ABSENT |
| `ReasoningTrace` | -- | -- | ABSENT |
| `EATPHook` / `HookRegistry` | -- | -- | ABSENT |
| Ed25519 signing functions | -- | -- | ABSENT |
| HMAC functions | -- | -- | ABSENT |

---

## 2. Kailash-From-Lesson-1 Audit

### 2.1 Module 1 -- CRITICAL ISSUES

**Issue 1: Exercise 1 is pure polars, not polars-through-Kailash.**

M1-Ex1 ("Polars deep dive") loads HDB data with pure polars and performs joins/window functions with no Kailash engine involvement. This directly violates the "Kailash from Lesson 1" directive. The very first exercise a student touches should use a Kailash engine.

**Recommendation**: M1-Ex1 should still teach polars (the course needs it), but wrap the exercise in `DataExplorer.profile()` at the end. Students learn polars expressions AND see that DataExplorer can consume their polars DataFrame to produce a professional profile. The exercise should end with:
```python
from kailash_ml import DataExplorer
explorer = DataExplorer()
profile = await explorer.profile(df)
```
This way the first thing students see is: polars creates the data, Kailash analyzes it.

**Issue 2: `ConnectionManager` is never introduced.**

`ConnectionManager` is the foundation for every persistent engine (FeatureStore, ModelRegistry, ExperimentTracker, DriftMonitor). It should be introduced in M1 when students first use DataExplorer, so that by M2 when they need FeatureStore, the pattern is already familiar.

**Recommendation**: Add a "Setting up your ML environment" section to M1-Ex3 (the DataExplorer exercise) that initializes `ConnectionManager("sqlite:///ml.db")`. Students see the pattern once; every subsequent module reuses it.

**Issue 3: `FeatureSchema` and `FeatureField` are not mentioned until M2.**

These types define the data contracts that every downstream engine uses. Students should see them in M1 when they first work with structured data, even if just as "here is how Kailash describes a dataset."

**Recommendation**: Add a brief introduction to `FeatureSchema` in M1-Ex3, where students define a schema for the economic indicators dataset they are profiling. This primes them for M2's full FeatureStore lifecycle.

**Issue 4: M1-Ex2 (Bayesian estimation) uses no Kailash engine.**

This is a pure statistics exercise with scipy/numpy. While the theory is valuable, the exercise should use `ModelVisualizer` for plotting posterior distributions rather than raw matplotlib/plotly.

**Recommendation**: Use `ModelVisualizer` for the visualization component. Students still compute posteriors manually (that is the learning objective), but Kailash handles the presentation.

### 2.2 Module 2 -- GOOD BUT DELAYED

**Issue: ExperimentTracker is used only in Ex5 (the last exercise), not from Ex1.**

The ExperimentTracker should be the first thing students set up in M2. Every feature engineering experiment should be tracked from the start. Ex1 (healthcare feature engineering) should log feature computations as experiment runs.

**Recommendation**: Restructure M2 so that Ex1 starts with `ExperimentTracker.create_experiment("healthcare_features")`, and every subsequent exercise adds runs to it. By Ex5, students have a full experiment history spanning all their M2 work.

### 2.3 Module 3 -- GOOD STRUCTURE

WorkflowBuilder is taught in Ex4, which is appropriate given the preceding theory exercises. However:

**Issue: `ConnectionManager` appears for the first time in M3 if not fixed in M1.**

If M1 does not introduce ConnectionManager, students encounter it cold in M3 when they need DataFlow persistence. This is too late.

**Issue: DataFlow patterns (`@db.model`, `db.express`) are lecture-only, not exercised.**

M3-3C mentions DataFlow but no exercise requires students to define a model or use `db.express`. Ex4 says "persist to DataFlow" but the module brief does not specify that students define their own model.

**Recommendation**: M3-Ex4 should explicitly require students to define a `@db.model` for storing model evaluation results, then use `db.express.create()` to persist them. This is their first hands-on DataFlow experience.

### 2.4 Module 4 -- ACCEPTABLE

All six exercises use at least one Kailash engine. The coverage is solid.

**Issue**: `AutoMLConfig` is not taught despite `AutoMLEngine` being used in Ex1. Students need to know about `agent=True`, `max_llm_cost_usd`, and `auto_approve` parameters to understand the double opt-in pattern.

### 2.5 Module 5 -- STRONG

All six exercises are deeply Kailash-native. The agent exercises properly use Delegate, Signature, and specialized agents.

**Issue**: Only 3 of 6 ML agents are exercised (DataScientist, FeatureEngineer, ModelSelector). The remaining 3 (ExperimentInterpreter, DriftAnalyst, RetrainingDecision) are mentioned in lecture but never used. At minimum, DriftAnalystAgent should be used in M5-Ex5 alongside the ML agent pipeline, since students already have drift data from M4.

### 2.6 Module 6 -- STRONG

Good coverage of Align, PACT, and RL. The capstone exercise properly integrates multiple packages.

---

## 3. Governance-From-Day-1 Audit

### 3.1 Current State: Governance is Module 6 Only

CARE, EATP, PACT, and TrustPlane are entirely absent from Modules 1-5. The course brief mentions "PACT framework" only in M6-6B. This means students spend 30+ hours without any governance awareness, then get a 60-minute lecture followed by two exercises.

This is the single largest structural gap in the curriculum. For a course that positions itself as production ML engineering, governance cannot be an afterthought.

### 3.2 Module-by-Module Governance Recommendations

**Module 1 -- Data Governance Foundations**

- M1-1C should introduce the concept of data classification levels (Public, Internal, Confidential, Restricted, Secret). DataFlow's `DataClassification` enum maps directly to these.
- M1-Ex3 (DataExplorer profiling) should include a sub-task: "Which columns in this dataset might contain PII? What classification level would you assign?" This is a discussion question, not a coding exercise, but it plants the seed.
- The `AlertConfig` with its 8 alert types is a natural fit for governance: "high cardinality in a name column might indicate PII."
- CARE principles (accountability, transparency) should be named in M1 lecture as the philosophical foundation. One slide is sufficient.

**Module 2 -- Data Lineage and Audit**

- FeatureStore inherently tracks data lineage (which features were computed from which source, at what time). M2-Ex2 should explicitly discuss this: "FeatureStore gives you point-in-time correctness AND an audit trail. If a regulator asks 'what data did this model see?', you can answer."
- ExperimentTracker provides experiment audit trails. M2-Ex5 should frame tracked experiments as governance artifacts: "Every experiment is auditable. You can prove which approaches you tried and why you chose this one."
- Introduce the concept of "model provenance" -- where did this model come from, what data trained it, who approved it?

**Module 3 -- Model Governance**

- ModelRegistry's lifecycle stages (staging, shadow, production, archived) are governance mechanisms. M3-Ex5 should frame promotion as a governance gate: "A model cannot go to production without explicit promotion. This is the simplest form of model governance."
- M3-Ex6 (end-to-end pipeline with model card) already includes model cards -- this is good. Explicitly connect model cards to the governance narrative.
- Introduce `ModelSignature` as a governance contract: "This signature defines what the model expects and produces. Any input that violates the signature is rejected."

**Module 4 -- Monitoring as Governance**

- DriftMonitor is already a governance tool. M4-Ex4 should frame drift detection as a governance obligation: "In production, you must prove your model is still performing as expected. DriftMonitor is how you meet that obligation."
- InferenceServer + Nexus deployment (M4-Ex6) should include a discussion of who can access the API. This primes students for PACT without introducing it formally.

**Module 5 -- Agent Safety as Governance**

- Agent cost budgets are already in the curriculum (M5-5C mentions them). Make them mandatory from M5-Ex1: every agent exercise must specify `max_llm_cost_usd`.
- The double opt-in pattern for ML agents is a governance pattern. Frame it as such.
- M5-Ex3 (ReActAgent with tools) should include a sub-task: "What happens if you remove the cost budget? What if the agent calls DataExplorer on a 100GB dataset?" This makes governance concrete.

**Module 6 -- Formal Governance (already covered)**

The current M6 content is strong. The gap is that it arrives too late. By the time students reach M6, they should already understand governance *intuitively* from Modules 1-5, and M6 formalizes what they already know.

### 3.3 TrustPlane / Trust Posture

The TrustPlane (`kailash.trust`) is completely absent from the curriculum. This is a substantial body of code (70+ exports) covering:
- Trust posture management (TrustPosture, PostureStateMachine)
- Cryptographic trust chains (GenesisRecord, CapabilityAttestation, DelegationRecord)
- Reasoning traces (ReasoningTrace, ConfidentialityLevel)
- Ed25519 signing for audit integrity
- EATP hooks and roles

**Recommendation**: TrustPlane is an advanced topic not suited for a 6-module course at this level. However, it SHOULD be mentioned in M6-6B lecture as "the cryptographic foundation that PACT builds on." A single slide showing the architecture diagram (Trust Plane sits beneath PACT) gives students awareness without requiring implementation. The capstone team project assessment could offer extra credit for TrustPlane integration.

### 3.4 CARE/EATP/CO References Before Module 6

Currently zero. The curriculum mentions CARE principles nowhere before M6.

**Recommendation**:
- M1 deck: One slide naming CARE (accountability, transparency) as the philosophical foundation
- M3 deck: One slide on EATP as the trust protocol (model provenance, audit chains)
- M5 deck: One slide on CO as the methodology (human-on-the-loop, not in-the-loop)
- M6: Full treatment (already planned)

This creates a narrative arc: philosophy (M1) -> protocol (M3) -> methodology (M5) -> implementation (M6).

---

## 4. Missing Kailash Capabilities -- Detailed Analysis

### 4.1 CRITICAL GAPS (MUST fix before launch)

**Gap 1: `ConnectionManager` is never taught.**

Every persistent engine requires `ConnectionManager`. The quick-start in the kailash-ml README starts with it. Students who complete this course will not know how to initialize a database connection for their ML engines.

- **Module**: M1 (first introduction), reinforced in M2, M3
- **Format**: New sub-task in M1-Ex3
- **Priority**: MUST

**Gap 2: `kailash_ml.interop` module is never mentioned.**

The interop module is the sole conversion point between polars and external frameworks. Students will inevitably need to convert between polars and pandas/numpy/sklearn/HuggingFace. Teaching them to use `interop` instead of ad-hoc conversion is essential.

- **Module**: M1 (mention in lecture), M3 (exercise use when training models)
- **Format**: Lecture mention in M1-1B, exercise use in M3-Ex1 (when preparing data for sklearn)
- **Priority**: MUST

**Gap 3: `ModelSignature` is never taught.**

ModelSignature captures input/output schema of trained models. It is the contract between training and serving. Students using InferenceServer without understanding ModelSignature will produce fragile deployments.

- **Module**: M3 (introduce with TrainingPipeline), M4 (use with InferenceServer)
- **Format**: Sub-task in M3-Ex6, reinforced in M4-Ex6
- **Priority**: MUST

**Gap 4: `SearchSpace`, `SearchConfig`, `ParamDistribution` are absent despite `HyperparameterSearch` being exercised.**

M3-Ex5 uses HyperparameterSearch but the brief does not mention these essential config types. Students cannot use HyperparameterSearch without them.

- **Module**: M3
- **Format**: Already part of M3-Ex5 (just needs explicit mention in brief)
- **Priority**: MUST

**Gap 5: `DriftSpec` is absent despite `DriftMonitor` being exercised.**

Same issue as Gap 4. DriftMonitor requires DriftSpec to configure thresholds.

- **Module**: M4
- **Format**: Already part of M4-Ex4 (needs explicit mention)
- **Priority**: MUST

**Gap 6: `AutoMLConfig` is absent despite `AutoMLEngine` being exercised.**

AutoMLConfig controls the double opt-in pattern for agents (`agent=True`, `max_llm_cost_usd`). This is a governance-relevant configuration.

- **Module**: M4
- **Format**: Already part of M4-Ex1 (needs explicit mention)
- **Priority**: MUST

### 4.2 IMPORTANT GAPS (SHOULD fix)

**Gap 7: `OnnxBridge` -- ONNX export/import.**

ONNX export is critical for production deployment ("train in Python, serve in Rust"). The InferenceServer exercise in M4-Ex6 should include an ONNX export step.

- **Module**: M4-Ex6
- **Format**: Sub-task in existing exercise
- **Priority**: SHOULD

**Gap 8: `MLDashboard` -- interactive ML dashboard.**

Students should see the dashboard at least once. It is the visual interface to everything they build with ExperimentTracker and ModelRegistry.

- **Module**: M3 (after ModelRegistry exercise)
- **Format**: 5-minute demo in lab, not a full exercise
- **Priority**: SHOULD

**Gap 9: `MlflowFormatReader`/`MlflowFormatWriter` -- MLflow compatibility.**

Many students will come from MLflow environments. Showing that Kailash can read/write MLflow format demonstrates interoperability and eases migration concerns.

- **Module**: M3 (brief mention in lecture)
- **Format**: Lecture mention in M3-3C
- **Priority**: SHOULD

**Gap 10: DataFlow `@db.model` and `db.express` as hands-on exercises.**

Currently lecture-only in M3-3C. Students need to define at least one model and perform CRUD operations.

- **Module**: M3-Ex4
- **Format**: Sub-task requiring students to define a model and use express API
- **Priority**: SHOULD

**Gap 11: Nexus presets (`lightweight`, `saas`, `enterprise`).**

Presets are the zero-config entry point for Nexus deployment. M4-Ex6 should use `Nexus(preset="lightweight")` instead of bare `Nexus()`.

- **Module**: M4-Ex6
- **Format**: Single line change in exercise
- **Priority**: SHOULD

**Gap 12: Nexus `app.create_session()` for unified state.**

Session management is how Nexus maintains state across API/CLI/MCP channels. The M4-Ex6 or M6-Ex6 deployment exercise should demonstrate this.

- **Module**: M4-Ex6 or M6-Ex6
- **Format**: Sub-task in deployment exercise
- **Priority**: SHOULD

**Gap 13: `AuditChain` (tamper-evident audit trail).**

AuditChain is how PACT provides tamper-evident governance records. M6-Ex3 (governance setup) should include audit chain verification.

- **Module**: M6-Ex3
- **Format**: Sub-task: "Verify that access decisions are recorded in the audit chain"
- **Priority**: SHOULD

**Gap 14: `GradientEngine` (verification gradient).**

The verification gradient determines how rigorously different actions are verified based on risk. M6-6B lectures about it but no exercise uses it.

- **Module**: M6-Ex3
- **Format**: Sub-task: configure verification rules for different risk levels
- **Priority**: SHOULD

**Gap 15: `McpGovernanceEnforcer` -- MCP tool governance.**

The M6-Ex6 capstone deploys via Nexus (which includes MCP). The MCP governance enforcer should be demonstrated to show that governance extends to tool invocations.

- **Module**: M6-Ex6
- **Format**: Sub-task in capstone exercise
- **Priority**: SHOULD

**Gap 16: Remaining 3 ML agents (ExperimentInterpreter, DriftAnalyst, RetrainingDecision).**

These are production-critical agents. At minimum, DriftAnalystAgent should be exercised since students already have drift data from M4.

- **Module**: M5-Ex5
- **Format**: Extended exercise: add DriftAnalyst to the ML agent chain
- **Priority**: SHOULD

**Gap 17: `LocalFileArtifactStore` for ModelRegistry.**

Students using ModelRegistry need to configure artifact storage. The README quick-start shows this but the curriculum does not mention it.

- **Module**: M3-Ex5
- **Format**: Already part of exercise (needs explicit mention)
- **Priority**: SHOULD

**Gap 18: Align method-specific configs (`LoRAConfig`, `DPOConfig`, `SFTConfig`).**

M6-Ex1 and M6-Ex2 teach SFT and DPO but do not mention the specific config classes. Students need to know these exist.

- **Module**: M6
- **Format**: Already part of exercises (needs explicit mention)
- **Priority**: SHOULD

**Gap 19: `AlignmentEvaluator` and `EvalConfig`.**

M6-6A lectures about evaluation metrics but does not use the Kailash evaluator engine. Students should use `AlignmentEvaluator` rather than computing metrics manually.

- **Module**: M6-Ex2
- **Format**: Sub-task: evaluate with AlignmentEvaluator
- **Priority**: SHOULD

**Gap 20: `AlignmentServing` and backend support (vLLM, llama-cpp, Ollama).**

The adapter lifecycle (train -> eval -> merge -> GGUF -> deploy) is not taught. Students learn to train adapters but not to serve them through different backends.

- **Module**: M6-Ex6 (capstone)
- **Format**: Lecture mention in M6-6A, optional capstone extension
- **Priority**: SHOULD

### 4.3 NICE-TO-HAVE GAPS

**Gap 21: DataFlow `QueryBuilder` (MongoDB-style queries).**

Advanced query patterns. Not essential for an ML course.

- **Module**: M3 (optional sidebar)
- **Priority**: NICE

**Gap 22: DataFlow bulk operations (`BulkCreate`, `BulkUpdate`, `BulkDelete`).**

Important for production data pipelines but not core ML.

- **Module**: M3 (optional sidebar)
- **Priority**: NICE

**Gap 23: DataFlow `DataClassification` and `classify()`.**

Relevant for governance-from-day-1 but could be introduced as a discussion point rather than an exercise.

- **Module**: M1 (discussion) or M3 (sidebar)
- **Priority**: NICE (upgrades to SHOULD if governance-from-day-1 is adopted)

**Gap 24: DataFlow validation (`field_validator`, `validate_model`).**

Input validation is security best practice. Could be a sidebar in M3.

- **Module**: M3
- **Priority**: NICE

**Gap 25: DataFlow `Provenance` and `ProvenanceMetadata`.**

Data provenance is a governance feature. Relevant for governance narrative.

- **Module**: M2 (governance narrative) or M3
- **Priority**: NICE

**Gap 26: Nexus `NexusEngine`, `Transport`, `EventBus`.**

Enterprise Nexus features. Beyond scope for this course level.

- **Priority**: NICE (mention in lecture only)

**Gap 27: Kaizen `GovernedSupervisor`.**

Advanced multi-agent governance pattern. Could be a capstone extension.

- **Module**: M6-Ex6 (optional extension)
- **Priority**: NICE

**Gap 28: Kaizen A2A factory functions (`create_research_agent_card`, etc.).**

M5-5C mentions A2A in lecture. A brief exercise creating agent cards would reinforce the concept.

- **Module**: M5-Ex6
- **Priority**: NICE

**Gap 29: PACT `VacancyStatus`, `VettingStatus`.**

Operational governance features (what happens when a role is unfilled). Lecture mention only.

- **Module**: M6-6B
- **Priority**: NICE

**Gap 30: PACT constraint dimension configs (Financial, Operational, Temporal, DataAccess, Communication).**

These are the five constraint dimensions. M6-Ex4 tests envelopes but does not require students to configure individual dimensions. Could enhance the exercise.

- **Module**: M6-Ex4
- **Priority**: NICE

**Gap 31: Align `AdapterMerger`.**

Merging LoRA adapters back into base model. Advanced topic.

- **Module**: M6-Ex2 (optional extension)
- **Priority**: NICE

**Gap 32: Align `GPUMemoryEstimate` / `estimate_training_memory()`.**

Practical production concern. Quick mention in M6-6A lecture.

- **Module**: M6
- **Priority**: NICE

**Gap 33: Core SDK node categories (20+ categories, 100+ nodes).**

The course teaches WorkflowBuilder but never surveys the available node types. A reference card listing categories (AI, API, Code, Data, Database, File, Logic, Monitoring, Admin, Transaction, Transform, etc.) would help students understand what is available.

- **Module**: M3 (reference handout)
- **Priority**: NICE

**Gap 34: `WorkflowVisualizer`.**

Visualizing workflow DAGs. Nice for understanding but not essential.

- **Module**: M3 (quick demo)
- **Priority**: NICE

**Gap 35: Specialized Kaizen agents (TreeOfThoughts, Planning, SelfReflection, Memory, Streaming, HumanApproval, etc.).**

15+ specialized agents exist beyond the 5 taught. A catalog slide in M5 showing what is available would help students choose for their capstone projects.

- **Module**: M5 (catalog slide)
- **Priority**: NICE

**Gap 36: `EnvironmentRegistry` and `PolicyRegistry` for RL.**

M6-Ex5 uses RLTrainer but does not mention the registry pattern. Should be implicit in the exercise.

- **Module**: M6-Ex5
- **Priority**: NICE

**Gap 37: `RLTrainingConfig`.**

Configuration type for RLTrainer. Students need it to use the trainer.

- **Module**: M6-Ex5 (already implicit, needs explicit mention)
- **Priority**: SHOULD (upgraded from NICE because students cannot use RLTrainer without it)

---

## 5. Summary of Recommendations by Priority

### MUST (6 items) -- Fix before course launch

| # | Gap | Module | Action |
|---|-----|--------|--------|
| 1 | `ConnectionManager` never taught | M1 | Add to M1-Ex3 as "ML environment setup" |
| 2 | M1-Ex1 is pure polars, no Kailash engine | M1 | Add DataExplorer.profile() at end of Ex1 |
| 3 | `kailash_ml.interop` never mentioned | M1, M3 | Lecture in M1-1B, exercise use in M3 |
| 4 | `ModelSignature` never taught | M3, M4 | Sub-task in M3-Ex6, used in M4-Ex6 |
| 5 | Config types missing from briefs (`SearchSpace`, `SearchConfig`, `DriftSpec`, `AutoMLConfig`) | M3, M4 | Update briefs to list required types |
| 6 | Governance-from-day-1 narrative absent from M1-M5 | All | Add governance thread per Section 3.2 |

### SHOULD (14 items) -- Strengthen production readiness

| # | Gap | Module | Action |
|---|-----|--------|--------|
| 7 | `OnnxBridge` absent | M4 | Sub-task in M4-Ex6 |
| 8 | `MLDashboard` absent | M3 | 5-min demo after ModelRegistry exercise |
| 9 | `MlflowFormatReader/Writer` absent | M3 | Lecture mention in M3-3C |
| 10 | DataFlow `@db.model`/`db.express` not exercised | M3 | Sub-task in M3-Ex4 |
| 11 | Nexus presets not used | M4 | Use `preset="lightweight"` in M4-Ex6 |
| 12 | Nexus `create_session()` absent | M4/M6 | Sub-task in deployment exercise |
| 13 | `AuditChain` absent | M6 | Sub-task in M6-Ex3 |
| 14 | `GradientEngine` absent | M6 | Sub-task in M6-Ex3 |
| 15 | `McpGovernanceEnforcer` absent | M6 | Sub-task in M6-Ex6 capstone |
| 16 | 3 ML agents unused | M5 | Extend M5-Ex5 with DriftAnalyst |
| 17 | `LocalFileArtifactStore` not mentioned | M3 | Explicit in M3-Ex5 |
| 18 | Align method configs not named | M6 | Explicit in M6-Ex1/Ex2 |
| 19 | `AlignmentEvaluator` not exercised | M6 | Sub-task in M6-Ex2 |
| 20 | `AlignmentServing` and backends absent | M6 | Lecture mention + optional capstone |

### NICE (17 items) -- Advanced / reference material

Gaps 21-37 from Section 4.3. These are valuable for advanced students and capstone projects but not required for the core curriculum. Recommend including them in a "Kailash SDK Reference" appendix that students can consult during capstone work.

---

## 6. Structural Recommendations

### 6.1 Add a Kailash SDK Quick Reference Card

Create a one-page reference card listing all 8 packages, their key exports, and when each is first introduced in the course. Students should carry this from M1 onward so they know what tools are available.

### 6.2 Establish a "Growing Stack" Visual

Each module deck should include a slide showing which Kailash packages the student has learned so far. By M6, the full stack is colored in. This reinforces the platform narrative.

### 6.3 M1-M5 Governance Thread

Add a recurring "Governance Corner" section to each module's lab (10-15 minutes). This is not a separate exercise but a discussion/reflection embedded in existing exercises:

- **M1**: "What data classification would you assign to each column?" (discussion)
- **M2**: "How does FeatureStore provide audit trail?" (observation)
- **M3**: "Who should approve a model promotion from staging to production?" (discussion)
- **M4**: "What drift threshold should trigger a governance review?" (discussion)
- **M5**: "What is the maximum cost budget you would give this agent?" (configuration)
- **M6**: Formal PACT implementation (already planned)

### 6.4 ExperimentTracker from M2-Ex1

Move ExperimentTracker introduction to M2-Ex1. Every feature engineering attempt in M2 should be tracked as an experiment run. This builds the habit of tracking experiments from the start.

### 6.5 ConnectionManager from M1-Ex3

Introduce ConnectionManager in M1-Ex3 as part of the DataExplorer exercise. The pattern is:
```python
from kailash.db.connection import ConnectionManager
conn = ConnectionManager("sqlite:///ml.db")
await conn.initialize()
```
Students learn it once, use it everywhere.

---

## 7. Coverage Score

| Package | Exports Covered | Exports Total | Coverage |
|---------|----------------|---------------|----------|
| kailash-ml (engines) | 13 | 16 | 81% |
| kailash-ml (types) | 2 | 6 | 33% |
| kailash-ml (interop) | 0 | 10 | 0% |
| kailash-ml (rl) | 1 | 4 | 25% |
| kailash-ml (agents) | 3 | 7 | 43% |
| kailash (core) | 2 | 10 | 20% |
| kailash.trust | 0 | 70+ | 0% |
| kailash-dataflow | 1 | 40+ | ~3% |
| kailash-nexus | 1 | 30+ | ~3% |
| kailash-kaizen | 3 | 15+ | 20% |
| kaizen-agents | 5 | 16+ | 31% |
| kailash-pact | 10 | 50+ | 20% |
| kailash-align | 3 | 30+ | 10% |

**Overall**: The course covers kailash-ml engines well (81%) but has significant gaps in all other packages. The platform packages (DataFlow, Nexus, PACT, Align, Trust) are barely touched beyond surface-level usage.

**Net assessment**: The course is excellent for teaching ML theory with Kailash-ML engines. It is NOT yet sufficient as "THE definitive training course for the Kailash platform." To reach that bar, it needs the MUST fixes (especially ConnectionManager, governance-from-day-1, and interop), plus the SHOULD fixes to give students real production platform experience.

After the MUST and SHOULD fixes, expected coverage of user-facing APIs (excluding internal types and store protocols) would rise to approximately 60-65% across the platform -- appropriate for a 6-module course that can point students to documentation for the remaining advanced features.
