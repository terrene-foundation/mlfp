# SDK Coverage Matrix

Kailash package-to-lesson mapping for MLFP. Every ML operation uses Kailash engines exclusively.

Source: mlfp-curriculum-v2.md, Part III.

| Package | Lessons | Key Classes |
|---|---|---|
| **kailash** (core) | M1, M3.6-7, M6.6 | WorkflowBuilder, LocalRuntime, Node, @register_node, PythonCodeNode, ConditionalNode, ConnectionManager, MCP server |
| **kailash-ml** | M1-M5 | DataExplorer, PreprocessingPipeline, FeatureEngineer, FeatureStore, TrainingPipeline, AutoMLEngine, HyperparameterSearch, EnsembleEngine, ModelRegistry, InferenceServer, DriftMonitor, ExperimentTracker, ModelVisualizer, OnnxBridge, RLTrainer |
| **kailash-dataflow** | M3.8 | @db.model, field(), db.express CRUD |
| **kailash-nexus** | M6.8 | Nexus, auth (RBAC/JWT), middleware, plugins |
| **kailash-kaizen** | M6.1, M6.4-6.6 | Signature, InputField/OutputField, Delegate, BaseAgent, RAGResearchAgent, ReActAgent, ChainOfThoughtAgent, MemoryAgent, coordination patterns |
| **kaizen-agents** | M6.5-6.6 | ML agents (DataScientist, FeatureEngineer, ModelSelector, etc.) |
| **kailash-pact** | M6.7 | GovernanceEngine, PactGovernedAgent, Address, CostTracker, operating envelopes |
| **kailash-align** | M6.2-6.3 | AlignmentPipeline, AlignmentConfig, AdapterRegistry, evaluator |
