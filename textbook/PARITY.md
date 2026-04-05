# Kailash SDK Parity Matrix — Python vs Rust

Cross-language parity reference built from tutorials and the 20 known divergences in `kailash-rs/tests/parity/divergences.json`.

## Parity Statuses

| Status         | Meaning                                                     |
| -------------- | ----------------------------------------------------------- |
| **Full**       | Same concept, same API shape, both languages                |
| **Equivalent** | Same concept, idiomatic differences (referenced by DIV-NNN) |
| **Py-only**    | Exists only in Python SDK                                   |
| **Rs-only**    | Exists only in Rust SDK                                     |
| **Different**  | Same domain, architecturally different approach             |

## Known Divergences (from divergences.json)

| ID      | Category         | Summary                                                           |
| ------- | ---------------- | ----------------------------------------------------------------- |
| DIV-001 | Value types      | Python float for all numbers; Rust distinguishes i64/f64          |
| DIV-002 | Strings          | Rust Arc\<str\> for zero-copy; transparent to users               |
| DIV-003 | Map ordering     | Python dict (insertion order); Rust BTreeMap (sorted)             |
| DIV-004 | Errors           | Python exceptions; Rust Result\<T, E\> typed errors               |
| DIV-005 | Runtime          | Python separate sync/async; Rust single Runtime with sync wrapper |
| DIV-006 | Bytes            | Rust Value::Bytes variant; Python bytes type                      |
| DIV-007 | Build            | Rust build() does pre-computation; Python build() is lighter      |
| DIV-008 | Execution result | Rust ExecutionResult has metadata; Python returns (dict, run_id)  |
| DIV-009 | Code nodes       | Python PythonCodeNode sandbox; Rust WASM + native cdylib          |
| DIV-010 | WASM             | WASM binding cannot use full kailash-core (tokio)                 |
| DIV-011 | Agent nodes      | Python ~18 specialized agent nodes; Rust uses Agent + AgentConfig |
| DIV-012 | Connectors       | SharePoint/Salesforce/SAP not in Rust                             |
| DIV-013 | Async            | Python has Async-prefixed variants; Rust nodes always async       |
| DIV-014 | Introspection    | Python from_function/from_class not applicable to Rust            |
| DIV-015 | Niche nodes      | Admin/notification nodes not implemented in Rust                  |
| DIV-016 | HandlerNode      | Absent in Rust; Python uses register_callback                     |
| DIV-017 | FilterNode       | Rust structured dict config; Python string expressions            |
| DIV-018 | SwitchNode       | Rust cases in config; Python in runtime inputs                    |
| DIV-019 | Parameters       | Python per-node params dict; Rust flat inputs dict                |
| DIV-020 | Connections      | Python add_connection(); Rust connect()                           |

---

## 00-core: Workflow Orchestration

| Concept        | Python API                                      | Rust API                                          | Status                   | Tutorial |
| -------------- | ----------------------------------------------- | ------------------------------------------------- | ------------------------ | -------- |
| Build workflow | `WorkflowBuilder()`                             | `WorkflowBuilder::new()`                          | Full (DIV-007)           | 00/01    |
| Add node       | `builder.add_node(type, id, params)`            | `builder.add_node(id, node)`                      | Equiv (DIV-019)          | 00/02    |
| Custom node    | `@register_node` decorator + `Node` subclass    | `impl Node for T` trait                           | Equiv (DIV-014)          | 00/03    |
| Connect        | `builder.connect(src, dst, mapping)`            | `builder.connect(src, dst)`                       | Equiv (DIV-020)          | 00/04    |
| Execute        | `LocalRuntime().execute(wf)` → `(dict, run_id)` | `Runtime::new().execute(&wf)` → `ExecutionResult` | Equiv (DIV-005, DIV-008) | 00/05    |
| Code node      | `PythonCodeNode`                                | WASM / native cdylib                              | Different (DIV-009)      | 00/06    |
| Conditional    | `ConditionalNode`                               | Structured config                                 | Equiv (DIV-017, DIV-018) | 00/07    |
| Value types    | Python native (int, float, str, bytes)          | `Value` enum (Integer, Float, String, Bytes)      | Equiv (DIV-001, DIV-006) | 00/08    |

## 01-dataflow: Database Operations

| Concept          | Python API                                 | Rust API                              | Status  | Tutorial |
| ---------------- | ------------------------------------------ | ------------------------------------- | ------- | -------- |
| Model definition | `@db.model` decorator, `DataFlowModel`     | `ModelDefinition::new()`, `FieldType` | Full    | 01/01    |
| Express CRUD     | `db.express.create/read/update/delete`     | `express` module                      | Full    | 01/02    |
| Filters          | `db.express.list(filters=...)`             | `FilterCondition` builder             | Full    | 01/03    |
| Validators       | `field_validator`, `email_validator`, etc. | —                                     | Py-only | 01/04    |
| Classification   | `DataClassification`, `MaskingStrategy`    | —                                     | Py-only | 01/05    |
| Transactions     | Transaction management                     | `DataFlowTransaction` RAII            | Full    | 01/06    |
| Bulk ops         | `BulkCreate`, `BulkUpdate`, etc.           | —                                     | Py-only | 01/07    |
| Multi-tenant     | `TenantContextSwitch`                      | `QueryInterceptor`                    | Full    | 01/08    |
| Provenance       | `Provenance`, `ProvenanceMetadata`         | —                                     | Py-only | 01/09    |

## 02-nexus: Multi-Channel Deployment

| Concept    | Python API               | Rust API                        | Status  | Tutorial |
| ---------- | ------------------------ | ------------------------------- | ------- | -------- |
| Setup      | `Nexus()`                | `Nexus::new()`                  | Full    | 02/01    |
| HTTP       | `HTTPTransport`          | `api::build_api_router` (axum)  | Full    | 02/02    |
| MCP        | `MCPTransport`           | `mcp::build_mcp_router`         | Full    | 02/03    |
| CLI        | —                        | `cli::build_cli_command` (clap) | Rs-only | 02/04    |
| SSE        | SSE endpoint             | —                               | Py-only | 02/05    |
| Middleware | Python middleware stack  | tower middleware                | Equiv   | 02/06    |
| Presets    | `Preset`, `NexusConfig`  | `middleware::Preset`            | Full    | 02/06    |
| Events     | `EventBus`, `NexusEvent` | Event system                    | Full    | 02/07    |

## 03-kaizen: AI Agent Framework

| Concept       | Python API                               | Rust API                     | Status | Tutorial |
| ------------- | ---------------------------------------- | ---------------------------- | ------ | -------- |
| Signature     | `Signature`, `InputField`, `OutputField` | Agent config                 | Equiv  | 03/01    |
| Agent         | `Agent` / `CoreAgent`                    | `BaseAgent` trait            | Equiv  | 03/02    |
| LLM providers | Via .env config                          | `llm` module, multi-provider | Full   | 03/03    |
| Memory        | `AgentManager`                           | `AgentMemory` trait          | Full   | 03/04    |
| Cost tracking | `max_llm_cost_usd`                       | Cost module                  | Full   | 03/05    |

## 04-agents: Specialized Agents

| Concept        | Python API            | Rust API              | Status  | Tutorial |
| -------------- | --------------------- | --------------------- | ------- | -------- |
| Delegate       | `Delegate`            | `DelegateEngine`      | Equiv   | 04/01    |
| SimpleQA       | `SimpleQAAgent`       | —                     | Py-only | 04/02    |
| ReAct          | `ReActAgent`          | —                     | Py-only | 04/03    |
| ChainOfThought | `ChainOfThoughtAgent` | —                     | Py-only | 04/04    |
| RAG            | `RAGResearchAgent`    | —                     | Py-only | 04/05    |
| Streaming      | `StreamingChatAgent`  | —                     | Py-only | 04/06    |
| Supervisor     | `GovernedSupervisor`  | Orchestration runtime | Equiv   | 04/07    |
| Pipeline       | `Pipeline`            | Protocols module      | Equiv   | 04/08    |

## 05-ml: Machine Learning

**Architecturally different**: Python provides 13 high-level engine wrappers around sklearn/torch. Rust provides 15+ crates with native algorithm implementations using type-state traits (Fit, Predict, Transform, Score).

| Python Engine           | Rust Equivalent                                                | Status    |
| ----------------------- | -------------------------------------------------------------- | --------- |
| `DataExplorer`          | `kailash-ml-explorer`                                          | Different |
| `PreprocessingPipeline` | `kailash-ml-preprocessing`                                     | Different |
| `TrainingPipeline`      | `kailash-ml-pipeline`                                          | Different |
| `AutoMLEngine`          | —                                                              | Py-only   |
| `EnsembleEngine`        | `kailash-ml-ensemble`                                          | Different |
| `ModelRegistry`         | —                                                              | Py-only   |
| `InferenceServer`       | —                                                              | Py-only   |
| `DriftMonitor`          | —                                                              | Py-only   |
| (sklearn wrappers)      | `kailash-ml-linear`, `kailash-ml-tree`, `kailash-ml-svm`, etc. | Rs-native |

## 06-pact: Governance

| Concept          | Python API                             | Rust API                              | Status | Tutorial |
| ---------------- | -------------------------------------- | ------------------------------------- | ------ | -------- |
| Address          | `Address` (from `kailash.trust.pact`)  | `Address` (from `kailash_governance`) | Full   | 06/01    |
| Compile org      | `compile_org()`                        | `compile_org()`                       | Full   | 06/02    |
| Envelopes        | `RoleEnvelope`, `TaskEnvelope`         | `envelopes` module                    | Full   | 06/03    |
| Clearance        | `RoleClearance`, `effective_clearance` | `clearance` module                    | Full   | 06/04    |
| Access           | `can_access()`, `explain_access()`     | `access` module                       | Full   | 06/05    |
| GovernanceEngine | `GovernanceEngine`                     | `GovernanceEngine`                    | Full   | 06/05    |
| Governed agent   | `PactGovernedAgent`                    | `agent` module                        | Equiv  | 06/09    |

## 07-align: LLM Fine-Tuning

| Concept          | Python API                        | Rust API                                      | Status  |
| ---------------- | --------------------------------- | --------------------------------------------- | ------- |
| SFT              | `AlignmentPipeline` + `SFTConfig` | —                                             | Py-only |
| LoRA             | `LoRAConfig`                      | —                                             | Py-only |
| DPO              | `DPOConfig`                       | —                                             | Py-only |
| KTO/ORPO/GRPO    | Method configs                    | —                                             | Py-only |
| Adapter registry | `AdapterRegistry`                 | —                                             | Py-only |
| Merge            | `AdapterMerger`                   | —                                             | Py-only |
| Evaluation       | `AlignmentEvaluator`              | —                                             | Py-only |
| Serving          | `AlignmentServing`                | `kailash-align-serving` (GGUF, LoRA hot-swap) | Partial |

---

_This matrix is updated as tutorials are written. Each tutorial validates its parity claims against the actual SDK source._
