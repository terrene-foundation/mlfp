# MLFP06 Audit — Language Models & Agentic Workflows

**Scope:**
- Repo: `/Users/esperie/repos/lyceum/courses/mlfp`, module `modules/mlfp06/` plus `shared/mlfp06/`. The audit was read-only.
- Installed stack used for every API check, via `.venv/bin/python`:

| Package | Version |
|---|---|
| kailash | 2.44.1 |
| kailash-kaizen | 2.28.0 |
| kaizen-agents | 0.9.11 |
| kailash-pact | 0.14.1 |
| kailash-align | 0.7.3 |
| kailash-ml | 2.2.2 |
| kailash-nexus | 2.11.0 |
| kailash-mcp | 0.2.15 |
| ragas | 0.4.3 |

- Excluded: everything listed as ALREADY KNOWN, and all `colab-selfcontained*` notebooks.
- Severity counts:

| Severity | Count |
|---|---|
| BLOCKING | 22 |
| MAJOR | 35 |
| MINOR | 26 |

Findings are grouped as follows:
1. Cross-cutting: the same defect appears in several artifacts and is reported once.
2. Master deck, textbook and speaker notes.
3. Assessment.
4. Exercises ex_1–ex_4.
5. Exercises ex_5–ex_8.
6. Lesson pages (`lessons/01..08`).

---

## 1. Cross-cutting findings

### [BLOCKING] X1 — Clearance ladder taught upside-down: course "restricted" is presented as the top tier, but pact RESTRICTED sits just above PUBLIC
**File:**
- **Master materials:**
  - `modules/mlfp06/textbook.md:1208`
  - `modules/mlfp06/deck.html:3576-3584`
  - `modules/mlfp06/speaker-notes.md:898`
- **Assessment:**
  - `modules/mlfp06/assessment/task_4/problem.md:65-70`
  - `modules/mlfp06/assessment/task_4/solution.py:150-154`
- **Shared helpers:**
  - `shared/mlfp06/ex_7.py:51-56` (`CLEARANCE_LEVELS` restricted=3 > confidential=2) and `:195-212` (org YAML)
  - `shared/mlfp06/ex_8.py:173-176, 230-245, 330-365`
- **Solutions:**
  - `solutions/ex_7/04_runtime_audit.py:105-118`
  - `solutions/ex_7/02_envelopes.py:456-466`
  - `solutions/ex_8/02_governance_pipeline.py:187`
- **Spec:** `specs/module-6.md` (Lesson 6.7, "Clearance levels" bullet)

**Evidence:**
- What the course teaches:
  - textbook.md:1208: "four-level clearance ladder — `public < internal < confidential < restricted` — where "restricted" is the maximum. That maps onto pact's canonical five-level ladder (`PUBLIC < RESTRICTED < CONFIDENTIAL < SECRET < TOP_SECRET`) with `"restricted"` (course) matching `RESTRICTED` (pact)".
  - deck.html:3579 has "Restricted — System admin, model deployment" as the top row.
  - task_4/problem.md:67: "a **rogue child** envelope that tries to escalate (clearance RESTRICTED …)" under a CONFIDENTIAL parent.
  - task_4/solution.py:151: `ConfidentialityLevel.RESTRICTED,  # escalated clearance`.
- What the installed stack does:
  - `[x.name for x in pact.ConfidentialityLevel]` gives `['PUBLIC','RESTRICTED','CONFIDENTIAL','SECRET','TOP_SECRET']`.
  - `kaizen_agents/supervisor.py:162` `_CLEARANCE_MAP` maps both "internal" and "restricted" to RESTRICTED.
  - `GovernedSupervisor(data_clearance="restricted").envelope.confidentiality_clearance` gives RESTRICTED; `"confidential"` gives CONFIDENTIAL.
  - `RoleEnvelope.validate_tightening` does not raise for a CONFIDENTIAL parent with a RESTRICTED child (everything else equal).
  - For a RESTRICTED parent with a TOP_SECRET child it does raise: `Confidentiality: child clearance (top_secret) exceeds parent (restricted)`.
  - Running the task 4 grader on the solution reports violations only for "Financial … Operational", not Confidentiality.
  - In the ex_7 YAML, `chief_ml_officer: restricted` heads `model_trainer: confidential`, so the child outranks its parent at runtime.
  - In the ex_8 YAML, `audit_agent: restricted` sits below `admin_agent: confidential`.

**Problem:**
- An order-preserving map cannot send the top of the course ladder to pact's second-lowest level. Students who give the most-privileged role "restricted" actually give it less access than "confidential".
- The org charts, the "monotonic escalation" visuals and the assessment all contradict the engine.
- Task 4 marks a legitimate tightening as "escalated clearance". `escalation_caught` is True only because budget and actions were widened.
- The spec itself carries the inverted 4-level ordering.

**Fix:**
- Teach pact's real order everywhere: `public < restricted (alias "internal") < confidential < secret < top_secret`.
- Remove "Restricted = max" from the deck, textbook, notes and spec.
- Re-tier the ex_7 and ex_8 YAMLs, and set `CLEARANCE_LEVELS` to pact's order (use "secret" for the top tier).
- In task 4, give the rogue child SECRET or TOP_SECRET so the clearance violation is part of what is caught, and fix the comment.

### [BLOCKING] X2 — DPO β explained backwards (textbook and ex_3 exercise code)
**File:**
- `modules/mlfp06/textbook.md:608`
- `solutions/ex_3/02_dpo_loss.py:171-176`
- `shared/mlfp06/ex_3.py:231`
- `solutions/ex_3/03_dpo_training.py:372`
- `solutions/ex_3/04_grpo_and_judge.py:304`
- `local/ex_3/02_dpo_loss.py:152-157`

**Evidence:**
- textbook.md:608: "Small β: the model stays close to the reference (conservative alignment). Large β: the model can deviate further".
- ex_3/02: "0.01 - 0.05 : Weak preference pressure. Stays close to SFT base"; "0.2 - 0.5 : Strong alignment… Risk: over-refusal". Chart title: "higher beta, stronger preference pressure".
- ex_3/04: "if MMLU drops > 3pp, beta is too high".

**Problem:**
- β is the KL-penalty coefficient: π* ∝ π_ref·exp(r/β), as shown at textbook.md:584 itself. A large β keeps the policy close to π_ref; a small β lets it drift.
- The deck notes are correct (deck.html:2157; speaker-notes.md:402: "Higher beta means stricter adherence to the reference"), so the textbook contradicts the deck.
- Students learn the wrong tuning rule, including the wrong fix for an MMLU regression. The correct fix is to raise β.

**Fix:**
- Small β = weak KL constraint and large drift. Large β = stays close to the reference.
- Correct the robo-advisor and IMDA decision text, the chart title and the MMLU remedy.

### [BLOCKING] X3 — "Fail-closed" governance taught, but the engine auto-approves unknown and envelope-less roles; shown deny-tests would fail
**File:**
- **Deck:** `deck.html:3320` ("Fail-closed… access is denied"), `3469-3474`, `3505-3519`
- **Textbook:** `textbook.md:1152-1166`, `1538` (glossary)
- **Lesson 6.7:** `lessons/07/textbook.html:607-616, 758-767`
- **Spec:** `specs/module-6.md` (6.7 "Fail-closed", "Governance testing … D99-R99-T99-R99 returns .allowed == False")
- **Solutions:**
  - `solutions/ex_8/02_governance_pipeline.py:51-53` ("The GovernanceEngine's default is fail-closed")
  - `solutions/ex_7/01_org_compile.py:128-136`
  - `shared/mlfp06/ex_7.py:367-394`

**Evidence:** Calls to `engine.verify_action(...)` on the course org (`compile_governance()` from shared.mlfp06.ex_7, with no envelopes attached):

| Role | Action | allowed | level | reason |
|---|---|---|---|---|
| D1-R1-T1-R1 | delete_all_records | True | auto_approved | No envelope constraints -- action permitted |
| D99-R99-T99-R99 | read_data | True | auto_approved | No envelope constraints -- action permitted |
| D1-R1-T1-R1 | delete_customer_data | True | auto_approved | No envelope constraints -- action permitted |

The audit of the ex_5–8 exercises also found:
- `GovernanceEngine(loaded.org_definition)` ignores `loaded.envelopes`, so YAML envelopes are never enforced.
- `load_org_from_dict` accepts a child clearance above its head's, and a $1 parent with a $5 child. Yet ex_7/01 prints "Compilation validates: … Clearance levels decrease monotonically … Budget envelopes don't exceed parent limits".
- The project's own solution already says this: `solutions/ex_7/04_runtime_audit.py:209-212`, "verify_action() on a role with NO attached envelope auto-approves".

**Problem:**
- Students are taught a security property the installed engine does not have.
- The deck's `assert not verdict.allowed` (3474), `test_analyst_cannot_delete` (3513-3519 / textbook 1152-1158) and the lesson 6.7 unknown-address tests would all fail.
- The spec's required negative test is absent from the exercises, and it would fail if added.

**Fix:**
- State the real behaviour: envelope-less and unknown roles are auto-approved in pact 0.14.1. Teach deny-paths by first attaching envelopes with `engine.set_role_envelope(...)`, as assessment task 4 does.
- Make every deny-test set an envelope first.
- Correct the compile-time validation claims, and attach the YAML envelopes or call them metadata only.
- Update the spec, or raise the fail-open default as an SDK issue.

### [BLOCKING] X4 — kailash-align code (deck, textbook, lessons 6.2/6.3) uses a config and API that do not exist
**File:**
- `deck.html:2028-2048, 2326-2344`
- `textbook.md:457-465, 643-647`
- `lessons/02/slides.html:333-364`
- `lessons/02/textbook.html:612-652, 730`
- `lessons/02/notes.html:355`
- `lessons/03/slides.html:288-302`
- `lessons/03/textbook.html:220-252, 481-497`
- `speaker-notes.md:473`

**Evidence:**
- What the materials show:
  - `AlignmentConfig(method="lora", model_name=..., rank=8, alpha=16, target_modules=[...], learning_rate=..., num_epochs=3)`
  - `pipeline.train(dataset="imdb_sentiment")`
  - `registry.register("sentiment-lora-v1", result.adapter)`
  - `result.trainable_params`
  - `AlignmentConfig(method="dpo", beta=0.1, lora_rank=8, …)` and `pipeline.train(dataset=…, eval_with="llm_judge")`
  - `result.win_rate`, `result.mmlu_delta`
  - Textbook: `AlignmentConfig(method="dpo", beta=0.1, epochs=3)`
  - Lessons: `base_model=`, `lora_r=`, `dpo_beta=`, `output_dir=`, `pipeline.train()`
- What is installed:
  - `AlignmentConfig(method='sft_then_dpo', base_model_id='', lora: LoRAConfig, sft: SFTConfig, dpo: DPOConfig, …)`.
  - `AlignmentConfig(method="lora", base_model_id="x")` raises `ValueError: Unknown training method 'lora'. Available: ['bco','cpo','dpo','grpo','kto','nash_md','ppo','rloo','sft','xpo','sft_then_dpo']`.
  - `AlignmentPipeline.train(self, dataset, adapter_name, preference_dataset=None, reward_funcs=None)`: `adapter_name` is required and there is no `eval_with`.
  - `AdapterRegistry` has no `register`, only async `register_adapter(name, adapter_path, signature, …)`, and it takes no path argument.
  - `AlignmentResult` fields are `adapter_name, adapter_path, adapter_version, training_metrics, experiment_dir, method`. There is no `trainable_params`, `win_rate`, `mmlu_delta`, `train_loss` or `eval_loss`.

**Problem:** Every "Kailash bridge" fine-tuning and DPO block raises. The slides also promise a win rate and an MMLU delta that the pipeline never computes. All of this contradicts `solutions/ex_2/06` and `solutions/ex_3/03`.

**Fix:** Rewrite against the solutions' pattern:
- `AlignmentConfig(method="sft"|"dpo", base_model_id=os.environ["SFT_BASE_MODEL"], lora=LoRAConfig(rank=8, alpha=16, target_modules=("q_proj","v_proj")), dpo=DPOConfig(beta=0.1))`
- `await pipeline.train(dataset=hf_ds, adapter_name=..., preference_dataset=...)`
- `await registry.register_adapter(...)`
- Read losses from `result.training_metrics`.

### [BLOCKING] X5 — Agent cost-budget API taught everywhere does not exist (`ReActAgent(tools=…, max_llm_cost_usd=…)`, `Delegate(max_llm_cost_usd=…)`)
**File:**
- `deck.html:2878-2885, 2911-2930, 921, 4287`
- `textbook.md:889-905, 937-941`
- `speaker-notes.md:671-673, 686`
- `lessons/01/slides.html:325-378`, `lessons/01/textbook.html:415-432`
- `lessons/05/slides.html:106-114`, `lessons/05/textbook.html:439-440, 447-536`
- `local/ex_5/02_cost_budget_agent.py:66-68, 98`
- `local/ex_3/04_grpo_and_judge.py:103-104`
- `solutions/ex_6/01_supervisor_worker.py:62`

**Evidence:**
- What is taught:
  - deck 2881: `max_llm_cost_usd=1.00,   # $1 budget cap; agent halts when exceeded`
  - textbook 891: "Kaizen 2.7 moved the cost cap onto the agent itself … `ReActAgent` accepts `max_llm_cost_usd`"
  - deck 4287: "Always set max_cost on every agent"
  - The notes describe a `max_cost` argument plus a Delegate argument.
- What happens:
  - `ReActAgent(model='llama3.2:3b', llm_provider='ollama', tools=[f], max_llm_cost_usd=1.0)` raises `TypeError: BaseAgent.__init__() got an unexpected keyword argument 'tools'`.
  - `Delegate.__init__` has no `max_llm_cost_usd`.
  - `grep -rn max_llm_cost_usd site-packages/kaizen*` finds only a kailash-ml AutoML comment.
  - `ReActAgent.run` is synchronous (`iscoroutinefunction` False), so `await agent.run(...)` also fails.
  - The real cap is `BaseAgentConfig.budget_limit_usd`, which is the name the spec uses.
  - The lessons contradict each other: 6.5 slides say `max_llm_cost_usd`, 6.5 textbook says `budget_limit_usd`.

**Problem:** The module's headline safety mechanism ("Non-negotiable in production: every agent must have a cost budget") is taught with a parameter that crashes, under three different names. The installed Kaizen is 2.28, not "2.7".

**Fix:**
- Standardise on `budget_limit_usd` on the agent config, passed at construction.
- Register tools the way the solutions and assessment task 3 do (`ToolRegistry.register(...)` plus `make_delegate(tools=reg)`).
- Use `run_async` / `run_sync` correctly.
- State that dollar budgets are a no-op on free local Ollama (`make_delegate` forces `budget_usd=None`).

### [MAJOR] X6 — Hardcoded or OpenAI-default model names in deck, textbook, lessons and assessment (check 8)
**File:**
- **deck.html:** 1611-1612, 1702, 2879, 2920, 3048, 3053, 3449
- **textbook.md:** 147-148, 190, 219-220, 898, 938, 1182, 1225, 1231, 1389
- **lessons, `gpt-4o-mini`:**
  - `01/slides.html:329, 372`
  - `01/textbook.html:417, 471`
  - `05/slides.html:110`
  - `05/textbook.html:501, 524`
  - `06/textbook.html:249, 681`
  - `07/slides.html:103`
  - `07/textbook.html:561, 797`
  - `08/textbook.html:437, 601`
- **lessons, `os.environ["DEFAULT_LLM_MODEL"]`:** `01/textbook.html:511`, `04/textbook.html:506`
- **lessons, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`:** `02/slides:338`, `02/textbook:620`, `03/slides:293`, `03/textbook:224, 485`
- **assessment:**
  - `task_2/solution.py:104, 124`
  - `task_2/starter.py:107`
  - `task_3/solution.py:109`
  - `task_3/starter.py:86` (hint)

**Evidence:**
- `os.environ.get("LLM_PROVIDER", "openai")` and `os.environ.get("LLM_MODEL", "gpt-4o-mini")`. Neither variable is in `.env.example`, so the OpenAI default always applies.
- `GovernedSupervisor(model="gpt-4o-mini", …)`.
- Assessment: `make_delegate(model="llama3.2:3b", …)` and `make_embedder(model="nomic-embed-text")`.
- textbook:190: `model="llama3.2:3b",  # default; override via OLLAMA_CHAT_MODEL`. This is false: an explicit `model=` bypasses `DEFAULT_CHAT_MODEL = _env("OLLAMA_CHAT_MODEL", …)` (`_ollama_bootstrap.py:72`).
- `DEFAULT_LLM_MODEL` is commented out in `.env.example`, so `os.environ["DEFAULT_LLM_MODEL"]` raises KeyError.

**Problem:**
- Breaks `.claude/rules/env-models.md` and Redline 14 (Ollama-first).
- Teaching code routes to a paid provider, or fails with no key.
- The assessment hardcodes ignore a student's `OLLAMA_CHAT_MODEL` and contradict `assessment/README.md:54` ("Read model names … from the course Ollama bootstrap").
- Task 1 does it right (`DEFAULT_CHAT_MODEL`, `OLLAMA_BASE_URL`).
- The ex_1–ex_8 solutions themselves are clean: everything resolves through `_ollama_bootstrap`.

**Fix:** Apply one pattern everywhere, matching the solutions.
1. `make_delegate(...)` and `make_embedder()` with no `model=`.
2. BaseAgent and ReAct configs use `{"llm_provider": "ollama", "model": DEFAULT_CHAT_MODEL, "base_url": OLLAMA_BASE_URL, "use_async_llm": True}`, imported from `shared.mlfp06._ollama_bootstrap`.
3. `GovernedSupervisor(model=DEFAULT_CHAT_MODEL, …)`.
4. Fine-tuning reads `SFT_BASE_MODEL`.
5. Delete every `LLM_MODEL`, `LLM_PROVIDER` and `DEFAULT_LLM_MODEL` fallback.

Prose may still say "the `OLLAMA_CHAT_MODEL` default (llama3.2:3b)".

### [MAJOR] X7 — Governance API details wrong: invented verdict levels and enforcement modes, positional `validate_tightening`, and a test that cannot parse
**File:**
- `deck.html:3292-3294, 3357-3361, 3521-3532`
- `textbook.md:1090, 1103, 1171, 1536`
- `lessons/07/slides.html:116, 197-214`
- `lessons/07/textbook.html:87, 255-277, 838-839, 857, 867`

**Evidence:**
- **Verdict levels and enforcement modes:**
  - The materials say "verdict.level ∈ {allowed, blocked, warn, audit}", and the glossary has "Enforcement mode: warn / block / audit".
  - Lesson 6.7 says "PACT's default is enforcement="block"" and uses `can_access`.
  - Installed: `GovernanceVerdict.allowed` returns `self.level in ("auto_approved", "flagged")`. The levels in source are `auto_approved, flagged, held, blocked`. `GovernanceEngine.__init__` has no `enforcement` parameter, and `hasattr(GovernanceEngine,'can_access')` is False.
- **`validate_tightening` is keyword-only:**
  - Signature: `RoleEnvelope.validate_tightening(*, parent_envelope, child_envelope, …)`.
  - The deck, textbook and lessons call `validate_tightening(parent, child)`, which raises TypeError. Inside `pytest.raises(MonotonicTighteningError)` that error escapes.
- **Budget test:**
  - deck 3528-3532 is `def test_zero_budget_stops_agent(): result = await governed.run(...)`. An `await` in a non-async `def` is a SyntaxError.
  - It only asserts `result.budget_consumed` is truthy, which proves nothing about stopping.

**Fix:**
- Teach the real four levels and `verify_action` as the only decision call. Delete the enforcement-mode table and `can_access`.
- Pass keyword arguments to `validate_tightening`.
- Write the budget test as an `async def` with a tiny `budget_usd`, asserting `not result.success` or `budget_consumed <= budget_allocated`.

### [BLOCKING] X8 — Capstone production code (Nexus, DriftMonitor, TrainingPipeline, ModelRegistry) uses constructors and methods that do not exist
**File:**
- `deck.html:3833-3848, 2872-2876`
- `textbook.md:930-935, 1002-1004, 1380-1382, 1396, 1416-1419`
- `lessons/08/slides.html:89-110`
- `lessons/08/textbook.html:323-327, 402-504`
- `lessons/08/notes.html:13`
- `lessons/05/textbook.html`, `lessons/06/textbook.html` (kailash_ml calls)

**Evidence:**
- **DriftMonitor:**
  - Shown: `DriftMonitor(reference_data=…, model_id=…, alert_threshold=0.05)`, `monitor.check(batch)`, `report.drift_detected/.p_value/.psi`.
  - Installed: `DriftMonitor(self, conn: ConnectionManager, *, tenant_id: str, psi_threshold=0.2, ks_threshold=0.05, …)`, with methods `check_drift`, `set_reference_data`, …. There is no `check`.
- **TrainingPipeline:**
  - Shown: `TrainingPipeline(task="classification").fit(...)` and `TrainingPipeline(model_type=…)`.
  - Installed: `TrainingPipeline(self, feature_store, registry)`, with methods `train/evaluate/retrain/calibrate`.
- **ModelRegistry:**
  - Shown: `ModelRegistry().load("hdb_predictor", stage="production")`.
  - Installed: `ModelRegistry(self, conn, artifact_store=None, …)`, with no `load`.
- **Nexus (lessons 6.8):**
  - Shown: `Nexus("ml-platform")`, `add_auth`, `register_service`, `serve`, `register_monitor`.
  - Installed: `Nexus.__init__(api_port=8000, …)` takes no name, so `"ml-platform"` becomes `api_port`. None of those four methods exist. The master deck and textbook already use `app.register(name, wf.build())`.
- **Lessons 6.8 inconsistency:** the slides use `alert_threshold=0.05`, the textbook `0.25`.

**Problem:** The capstone monitoring, deployment and agent-tool examples fail at the first constructor. They also contradict the M3/M4 usage and the ex_8 solutions.

**Fix:**
- `DriftMonitor(conn, tenant_id=…, psi_threshold=…)`, then `set_reference_data(...)`, then `check_drift(...)`.
- Build `TrainingPipeline` and `ModelRegistry` with a `ConnectionManager`, or call the earlier-module helpers.
- Port the master `app.register(name, workflow.build())` pattern into lessons/08.

### [MAJOR] X9 — Datasets named in index.html, deck, textbook and lessons do not exist; the RAG corpus is misdescribed as Singapore policy / MAS regulations
**File:**
- `index.html:80-86, 125`
- `deck.html:2632, 2647, 763-779, 827`
- `textbook.md:795, 828`
- `lessons/01/slides.html:420`, `01/textbook.html:516, 748` (`sg_company_reports.parquet`)
- `lessons/02/*` (`sg_domain_qa*`); `02/textbook.html:761` ("Run ex_2.py")
- `lessons/03/*` (`preference_pairs*`)
- `lessons/04/textbook.html:470, 485` (`sg_regulations.parquet`)
- `lessons/05`, `lessons/06` (`sg_telco_churn.csv`, `hdb_resale.csv`, `mrt_ridership.csv`)

**Evidence:**
- `ls data/mlfp06/` lists `ex8_drift.db hotpotqa imdb mmlu rag squad sst2 toxicity ultrafeedback`. None of the named files exist.
- `sg_domain_qa`/`preference_pairs` exist only under data/mlfp04.
- The 6.4 corpus is `rag/rag_corpus_1k.parquet` (neural-bridge rag-dataset-12000).
- `ex_2` is now a directory, so "Run ex_2.py" points nowhere.
- The deck Lens 3 shows a "Retriever leaderboard — 50 SG policy queries" with "dense (bge-m3)". The course embedder is nomic-embed-text, and no exercise computes this table.

**Fix:**
- Regenerate the index.html dataset table from `data/mlfp06/*`.
- Point each lesson example at the dataset and `ex_N/NN_*.py` file its exercise actually uses.
- Describe the 6.4 corpus accurately.
- Label the leaderboard as illustrative, or replace it with ex_4 output.

### [BLOCKING] X10 — D/T/R expanded three different ways; exercises teach "Delegator/Task/Responsible"
**File:**
- `shared/mlfp06/ex_7.py:110-113`
- `solutions/ex_7/01_org_compile.py:9, 45-55, 158-195` (columns Delegator/Task/Responsible)
- `solutions/ex_8/02_governance_pipeline.py:43-46`
- `solutions/ex_7/04_runtime_audit.py:432`
- `lessons/07/slides.html` ("PACT: who Decides, Trusts, Responds": Decision-maker / Trust-holder / Responder)
- `lessons/07/textbook.html:145, 403` ("task")

**Evidence:**
- `kailash/trust/pact/addressing.py:30-33`: `DEPARTMENT = "D"`.
- The compiled org prints `D1 DEPARTMENT`, `D1-R1-T1 TEAM Data Analysis`, `D1-R1-T1-R1 ROLE data_analyst`.
- The spec, `deck.html:3261-3265` and `textbook.md:1069` say Department/Team/Role.

**Problem:** The core vocabulary of lesson 6.7 is contradicted by its own exercises and lesson slides.

**Fix:** Use D = Department, T = Team, R = Role everywhere. Every D or T is immediately followed by its head R. Teach delegation (`defining_role_address` → `target_role_address`) as a separate envelope concept.

### [MAJOR] X11 — GRPO formula incomplete or inconsistent, and its origin mis-attributed
**File:**
- `deck.html:2208, 2216, 4471`
- `textbook.md:612-616`
- `speaker-notes.md:388, 1223`
- `solutions/ex_3/04_grpo_and_judge.py:52-76`
- `shared/mlfp06/ex_3.py:135-142`

**Evidence:**
- The deck divides by std: `Â_i = (r_i − mean(r))/std(r)`.
- textbook.md:616 and ex_3 use `A_i = r_i − mean(r)`.
- ex_3 adds `L_GRPO = −E[Σ A_i log π(y_i|x)]` and claims "Group-relative normalisation makes training stable regardless of the reward scale".
- deck 4471 cites "Shao et al. (2024) — DeepSeek-R1 (GRPO)". Shao et al. 2024 is DeepSeekMath, which textbook.md:1626 correctly cites as introducing GRPO.

**Problem:**
- Mean-subtraction alone is not scale-invariant.
- GRPO uses std-normalised advantages, a PPO-style clipped ratio and a KL term to π_ref.
- Students get two different formulas and a wrong citation.

**Fix:**
- Use `(r_i − mean)/(std+ε)` everywhere.
- State the clipped surrogate plus the KL term, or label the code a simplified REINFORCE-with-baseline.
- Cite DeepSeekMath (2024), "later used in DeepSeek-R1 (2025)".

### [MINOR] X12 — Commercial-product comparisons and real companies cast in invented scenarios (check 7)
**File:**
- `deck.html:1111, 1127` ("Langfuse self-hosted over LangSmith; DeepEval over proprietary evals")
- `lessons/01/notes.html:458` ("Can I just use ChatGPT instead? (you can, but … no budget control, no structured output, no audit trail)")
- Scenario protagonists, with invented operating figures:
  - `solutions/ex_1/01_zero_shot.py:161-177` (DBS: "DBS Bank (Singapore) receives ~40,000 app-store reviews per …")
  - `ex_1/06:191-193` (Grab)
  - `ex_1/04` (SingPost)
  - `ex_1/03` (SGH)
  - `ex_3/04` (NUH ×9)
  - `ex_4/05` (AIA)
  - `ex_4/02` (SingHealth)
  - `ex_8/04` (Shopee: "Shopee-scale 11.11")
  - the matching local files

**Problem:**
- `rules/independence.md` blocks positioning against commercial products.
- Attributing invented volumes and dollar impacts to named firms can read as endorsement or partnership, and puts unverifiable claims about real companies into a Foundation publication.
- Acceptable as context:
  - public regulators and public-domain references (MAS, IMDA, PDPA, GovTech, AI Verify);
  - Claude Desktop / VS Code as examples of MCP clients;
  - public dataset entity names (SQuAD and HotpotQA article subjects).

**Fix:**
- Describe tools on their own terms ("self-hosted, open-source").
- Answer the ChatGPT question without naming the product.
- Replace company names with generic descriptors, e.g. "a Singapore retail bank" or "a regional e-commerce marketplace".

---

## 2. Master deck, textbook and speaker notes

### [BLOCKING] D1 — Observatory "Code" slides call methods and kwargs that do not exist in `shared.mlfp06.diagnostics`
**File:** `deck.html:550-575` (Lens 1), `680-703` (Lens 2), `801-824` (Lens 3), `917-949` (Lens 4)
**Evidence:** Compared with `inspect.signature` on the shared package:
- **Lens 1:**
  - `obs.output.self_consistency(prompt=…, n=5)`, but the installed signature is `self_consistency(responses, *, prompt='', run_id=None) -> pl.DataFrame`. There is no `n`, and `responses` is required.
  - The promised `result.agreement_rate/.flagged_as_hallucination` do not exist; the DataFrame columns are `agreement`, `is_outlier`.
  - `refusal_rate(harmful_prompts)` and `over_refusal(benign_prompts)` take model responses, not prompts.
- **Lens 2:**
  - `attention_heatmap(…, model="google/gemma-2-2b", …)`, `logit_lens(…, model=…)` and `sae_features(…, model=…)` all fail: there is no `model` kwarg. The model is set via `LLMObservatory(attention_model=…)`.
  - `circuit_trace` does not exist.
- **Lens 3:**
  - `recall_at_k(queries=, relevant_ids=, retriever=, k=)`, but the installed signature is `(retrieved_ids, relevant_ids, *, k=5) -> float`.
  - `compare_retrievers(retrievers=, queries=, relevant_ids=, k=)`, but the installed signature is `(retrievers, eval_set, *, k=5, run_id=None)`.
  - `context_utilisation(response=, context=)`, but the installed signature is `(answer, contexts) -> float`.
  - `chunk_size_sweep` and `plot_retrieval_dashboard` do not exist; the real one is `plot_rag_dashboard`.
- **Lens 4:**
  - `obs.agent.capture(...)` as a context manager, `tool_usage_summary`, `loop_detection`, `budget_timeline`, `replay` and `export_to_langfuse` do not exist. The real methods are `capture_run` (async), `tool_usage`, `detect_loops`, `plot_trace` and `plot_cost_across_runs`.
  - `ReActAgent(tools=[…])` raises TypeError (see X5).

**Problem:** These slides are presented as the method surface used in every M6 exercise. Nearly every call raises, and the sample outputs describe objects that are never returned.
**Fix:** Regenerate the four slides from the real signatures, for example:
- `obs.output.self_consistency([r1,…,r5], prompt=…)`
- `obs.retrieval.recall_at_k(ids, gold, k=5)`
- `trace = await obs.agent.capture_run(delegate, prompt)`; `obs.agent.tool_usage(trace.run_id)`

Then remove the non-existent methods.

### [BLOCKING] D2 — Multi-agent and MCP slides: code crashes or silently registers zero tools
**File:** `deck.html:3044-3065, 3127-3150, 4155`; `textbook.md:990-1006, 1022-1031`
**Evidence:**
- **Router:** the deck uses `Pipeline.router(agents=[data_scientist, feature_engineer])` with Delegates.
  - The installed signature is `Pipeline.router(agents: list['BaseAgent'], …)`, and `issubclass(Delegate, BaseAgent)` is False.
  - `Pipeline.run(self, **inputs) -> dict` is synchronous, but the deck does `await supervisor.run(objective=…)`.
- **Callable tools are dropped:**
  - deck 3047: `Delegate(model=…, tools=[explore_data], budget_usd=2.00)`. Textbook 1022: `make_delegate(tools=[profile_data, visualise_data])`.
  - Delegate's `tools` is typed `ToolRegistry | list[str] | None`.
  - `make_delegate(tools=[profile_data]).tool_registry.tool_names` returns `[]`: the callable is silently dropped.
- **MCP:**
  - `from kailash_mcp import MCPServer, tool` raises `ImportError: cannot import name 'tool'`. The textbook's `Tool` is not exported either.
  - `@server.tool` (no parentheses) passes the function as `cache_key`, because `MCPServer.tool(self, cache_key=None, …)` is a decorator factory. `@server.tool(description=…)` is not a valid kwarg.
  - `server.run(transport="stdio")` fails: `MCPServer.run(self)` takes no arguments, and transport belongs in the constructor (solutions/ex_6/04 uses `MCPServer(name=…, transport="stdio")` with `@mcp_server.tool()`).

**Fix:**
- Build a `ToolRegistry` with `register(name=, description=, parameters=, executor=)`, as assessment task 3 does.
- Use BaseAgent instances with `Pipeline`, or show the sequential pattern from solutions/ex_6.
- MCP slide: `MCPServer("ml-tools", transport="stdio")`, then `@server.tool()`, then `server.run()`.

### [MAJOR] D3 — Textbook 6.1 "Solution" code calls `.get()` on the string returned by `Delegate.run_sync`
**File:** `textbook.md:269-270, 285-286`
**Evidence:** `result = delegate.run_sync(...)` then `float(result.get("answer", 0))`. The installed signature is `Delegate.run_sync(self, prompt: str) -> str` ("returns the complete response string").
**Problem:** Both drill solutions raise AttributeError.
**Fix:** Use the BaseAgent `run_async(problem=…)` dict path from the worked example, or parse the text.

### [MAJOR] D4 — Deck uses a non-existent `delegate.generate(...)` and builds `Delegate` directly with OpenAI defaults
**File:** `deck.html:1527-1534` (self-consistency), `2576-2578` (HyDE), `1701-1705` (Exercise 6.1 skeleton)
**Evidence:**
- `dir(Delegate)` lists only `budget_remaining, budget_usd, close, consumed_usd, core_agent, interrupt, loop, model, run, run_sync, signature, tool_registry, wrapper_stack`. There is no `generate`.
- The skeleton is `Delegate(model=os.environ.get("LLM_MODEL","gpt-4o-mini"), budget_usd=0.50, signature=…)`. Redline 14 says "Direct `Delegate(...)` construction in M6 code is BLOCKED".

**Fix:**
- Replace `generate` with `text, usage, _ = await run_delegate_text(make_delegate(temperature=0.7), prompt)`.
- Show the actual local/ex_1 skeleton.

### [MAJOR] D5 — Kojima et al. zero-shot-CoT result misquoted
**File:** `deck.html:1502`; `speaker-notes.md:161`
**Evidence:**
- "improves accuracy on GSM8K from 17.7% to 78.7%"; the notes add "a 4x gain from seven words".
- 17.7→78.7 is the paper's MultiArith result (text-davinci-002). GSM8K is 10.4→40.7.
- "Let's think step by step" is five words.

**Fix:** "MultiArith 17.7%→78.7%; GSM8K 10.4%→40.7%"; "five words".

### [MAJOR] D6 — End-of-module assessment described three different ways
**File:** `deck.html:4426-4447`; `speaker-notes.md:1204-1210`; `assessment/README.md:8-22`; `specs/module-6.md` (last line); `specs/redlines.md` R6
**Evidence:**
- Deck and notes: capstone presentation (15+5 min) plus a 60-minute open-book quiz "covering all 8 lessons".
- assessment/README.md: "Four auto-graded tasks … 100 marks · Duration: 3 hours … No AI assistants". The tasks cover prompting, RAG, a tool agent and PACT.
- Nothing assesses 6.2, 6.3, 6.6 or 6.8.

**Fix:** Make the deck and notes describe the shipped 3-hour, 4-task assessment, or add the missing components. Reconcile with spec R6.

### [MAJOR] D7 — Textbook LoRA worked example never freezes the base model, so it is full fine-tuning
**File:** `textbook.md:470-495` (LoRALayer at 369-382)
**Evidence:**
- The loop builds `LoRALayer(...)`, then sets `layer.attention.self.query.original = original_q`. `original_q` is the pretrained Linear, with `requires_grad=True` and a bias. The freeze in `__init__` applied only to the Linear that was just discarded.
- No other BERT parameter is frozen.
- The optimiser takes every parameter with `requires_grad`.

**Problem:**
- The printed "Trainable %" will be about 100%, and every BERT weight is updated.
- This is exactly the "Common mistake" deck.html:2083 warns about.

**Fix:**
- Run `for p in model.parameters(): p.requires_grad = False` before wrapping.
- Freeze `original_q` inside the wrapper.
- Leave only A, B and the classifier head trainable.

### [MAJOR] D8 — Placeholder "Solution:" blocks in the textbook
**File:** `textbook.md:504-507, 529-533, 539-543, 685-688, 694-699, 713`
**Evidence:**
- 6.2 Drill 1's solution is `# Apply LoRALayer to Q, K, V … / # Train for 3 epochs, evaluate on test set`.
- 6.2 Drill 3 is a loop containing `pass`.
- 6.2 Drill 4 and 6.3 Drills 1-3 are comment or `pass` stubs.
- 6.4-6.8 have no solutions at all; 6.1 has full ones.

**Fix:** Write real solutions (pointing at solutions/ex_N where possible), or drop the "Solution:" label. See the no-stubs rule.

### [MAJOR] D9 — Spec topics not taught in deck, textbook or notes
**File:** `deck.html`, `textbook.md`, `speaker-notes.md` (absence); `specs/module-6.md` 6.4, 6.5, 6.6 and 6.8
**Evidence:** `grep -ci` counts are 0 in all three for:
- "metadata filter" and "multi-hop" (6.4 Advanced RAG patterns)
- "RAGResearchAgent" and "MemoryAgent" (6.4 Kaizen RAG agents)
- "ChainOfThoughtAgent" (6.5)
- "load balancing" and "dynamic agent" (6.6 architectural considerations)
- "CORS" (6.8 middleware)

Nexus plugins appear nowhere; the deck's only "plugin" hits are the Reveal.js config.

**Fix:** Add slide and textbook coverage for each named spec topic.

### [MAJOR] D10 — Agent-debugging and testing slides use a fictional trace API, and deck and textbook disagree
**File:** `deck.html:3867-3880, 3921-3938`; `textbook.md:1364-1368`
**Evidence:**
- Deck: `agent.run(..., trace=True)`, then `result.trace` steps with `.tool/.cost/.decision`, plus `result.success/.output/.cost`.
- Textbook: `return_trace=True` with `.thought/.action/.observation`.
- Installed: `ReActAgent.run(self, task, context='', **kwargs) -> dict`.
- The course's real trace path is `LLMObservatory().agent.capture_run(...)` returning an `AgentTrace` (`shared/mlfp06/diagnostics/agent.py:135`).

**Fix:** Rewrite using `capture_run`, `tool_usage` and `detect_loops`.

### [MINOR] D11 — Scaling-law formula omits irreducible loss and is mis-attributed
**File:** `deck.html:1362-1367`
**Evidence:** `L(N,D) ≈ (N_c/N)^{α_N} + (D_c/D)^{α_D}` appears directly under "Kaplan et al. (2020)" and "Chinchilla (2022)".
**Problem:** Kaplan's joint law is `[(N_c/N)^{α_N/α_D} + D_c/D]^{α_D}`, and Chinchilla's is `E + A/N^α + B/D^β`. The shown form implies loss → 0.
**Fix:** Use the Chinchilla form including the irreducible-loss term `E`, and label it.

### [MINOR] D12 — Unsupported or incorrect numbers in 6.2
**File:** `deck.html:1375, 2048`; `speaker-notes.md:363`
**Evidence:**
- "GPT-4 ~1.8T (MoE)" is stated as fact.
- `# ~262K` trainable parameters for rank-8 LoRA on Llama-3-8B q_proj and v_proj; the notes add "less than 0.01%".

**Problem:**
- GPT-4's parameter count is undisclosed.
- q_proj is 4096→4096 and v_proj is 4096→1024 (GQA), over 32 layers: 32·(8·8192 + 8·5120) = 3,407,872 (~3.4M, ≈0.04%).

**Fix:** "undisclosed (reported ~1.8T MoE)"; "~3.4M (≈0.04%)".

### [MINOR] D13 — Stale framework version references
**File:** `deck.html:3584` ("In pact 0.8.1"), `2918` ("in 2.7"); `textbook.md:891` ("Kaizen 2.7")
**Evidence:** Installed versions are kailash-pact 0.14.1 and kailash-kaizen 2.28.0.
**Fix:** Drop or update the version numbers.

### [MINOR] D14 — Deck imports libraries that are not usable in the course environment
**File:** `deck.html:2518`, `2607-2612`
**Evidence:**
- `rank-bm25` is not installed.
- ragas 0.4.3 has no `context_relevancy` (grep of `ragas/metrics/__init__.py` finds `ContextRelevance` only).
- `from ragas.metrics import faithfulness` raises `ModuleNotFoundError: langchain_community.chat_models.vertexai` in .venv.

**Fix:** Show the course BM25 (solutions/ex_4/03) and `obs.retrieval.ragas_scores(...)`. If ragas is shown directly, use ragas ≥0.2 names.

### [MINOR] D15 — Small slide inconsistencies
**File:** `deck.html:1606` vs `1624`; `2071`/`2080` vs `2091`
**Evidence:**
- The OutputField allows "positive, negative, or neutral", but the sample output says `# "mixed" or "neutral"`.
- The Exercise 6.2 slide makes TIES merging a required task and criterion, while its notes call it "a stretch goal".

**Fix:** Use an allowed value in the sample output. Decide whether merging is required, and make the slide and notes agree.

### [MINOR] D16 — Unsourced anecdote presented as fact
**File:** `deck.html:386` (speaker notes; copied into `lessons/01/notes.html`)
**Evidence:** "In 2024 a major bank shipped a chatbot that cheerfully invented refund policies for three weeks before anyone noticed".
**Problem:** No source is given. The well-known 2024 case of a chatbot inventing a refund policy involved an airline (a tribunal ruling), not a bank.
**Fix:** Frame the story as an explicit hypothetical.

### [MINOR] D17 — Study-time totals disagree
**File:** `index.html:51` ("~28 hours"); `textbook.md:70` ("roughly 24 hours"); `deck.html:4558` ("~23.5 hours")
**Fix:** Use one figure. The textbook table sums to about 24h.

---

## 3. Assessment (`modules/mlfp06/assessment`)

Reasoning from the code, all four solutions satisfy their graders' checks:
- I ran the task 4 grader on `solution.py`: `passed: true, 10/10`, in 7 s.
- Tasks 1-3 need live Ollama and were not run. Their check counts (11, 9, 11) match each problem.md.
- `build_corpus_and_questions()` in task 2 reproduces the grader's gold indices (gold idx 1, 2, 3, 5, 6, 7 over 30 contexts).
- Task 3's SST-2 labels are `positive`/`negative` (107/93), matching the `count_by_label` arguments.

The clearance error in task 4 is covered by X1, and the hardcoded models in tasks 2 and 3 by X6.

### [MINOR] A1 — Prettier turned "+" into a bullet, breaking sentences in two problem statements
**File:** `assessment/task_3/problem.md:3-6`; `assessment/task_4/problem.md:10-14`
**Evidence:**
- task_3: "**Framework**: Kaizen `Delegate`", then the bullet "- `ToolRegistry` (Ollama, …) · **Dataset**: …".
- task_4: "(a dollar budget", then the bullet "- an allow-listed action set), every access decision …".

**Fix:** Write "Kaizen `Delegate` + `ToolRegistry`" and "a dollar budget plus an allow-listed action set" on one line.

### [MINOR] A2 — Task 3 grading description contradicts the hard non-empty-answer check
**File:** `assessment/task_3/grader.py:14-16, 157-160`; `task_3/problem.md:79-87`
**Evidence:**
- The grader docstring says "final wording is graded only as a soft floor".
- problem.md says final wording "is NOT graded … small local models often fail to populate" it.
- The grader nevertheless requires `answers_nonempty = all(...)`, and the task only passes if all 11 checks pass.

**Fix:** Make the check a floor (e.g. ≥4/5), or state in problem.md that an empty final answer fails the task.

---

## 4. Exercises ex_1 – ex_4 (solutions, local scaffolds, shared/mlfp06/ex_1..4)

All 21 local scaffolds have the header, WHAT YOU'LL LEARN, PREREQUISITES, ESTIMATED TIME and REFLECTION blocks. No hardcoded model names: everything resolves through `OLLAMA_CHAT_MODEL`, `OLLAMA_EMBED_MODEL` or `SFT_BASE_MODEL`.

### [BLOCKING] E1 — Ex 1.6 structured-output solution never calls an LLM; a silent fallback makes every checkpoint pass
**File:** `solutions/ex_1/06_structured_output.py:101-123`
**Evidence:**
- The agent is `BaseAgent(config={"model": MODEL, "budget_limit_usd": 1.0}, signature=ReviewExtraction())`. Calls are wrapped in `try … except Exception: result = {"sentiment": "unknown", "confidence": 0.0, "key_phrases": [], …}`.
- In .venv, `run_async` raises `ValueError: Agent not configured for async mode. Set use_async_llm=True` on every call, before any network access.
- The constructed config has `llm_provider=None` (falling back to the `mock` provider) and `budget_limit_usd=None`.

**Problem:**
- All 10 results are the fake dict, the checkpoints pass on it, and the chart shows 100% extraction.
- Breaks Redline 14 and zero-tolerance Rules 2 and 3.

**Fix:**
- Remove the try/except.
- Use `BaseAgentConfig(llm_provider="ollama", model=MODEL, base_url=OLLAMA_BASE_URL, use_async_llm=True)`, or `make_delegate(signature=ReviewExtraction)`.

### [BLOCKING] E2 — Ex 1.6 local scaffold points at an API that cannot work and diverges from the solution
**File:** `local/ex_1/06_structured_output.py:25, 67-76, 89-97`
**Evidence:**
- The hint is `SimpleQAAgent(signature=ReviewExtraction, model=MODEL, max_llm_cost_usd=1.0)` and `await agent.run(review_text=...)`, followed by `hasattr(sample,"sentiment")`.
- `SimpleQAAgent.__init__` has no signature or cost parameter.
- `SimpleQAAgent.run(self, question, …) -> dict` is synchronous.

**Fix:** Rewrite the TODO and checkpoints to mirror the corrected solution.

### [BLOCKING] E3 — Ex 3.4 local judge hint is an invalid Delegate call, and its streaming loop duplicates text
**File:** `local/ex_3/04_grpo_and_judge.py:35, 103-104, 121-129`
**Evidence:**
- The TODO says "Construct a Kaizen Delegate with … max_llm_cost_usd=0.5", which raises TypeError. Direct `Delegate(...)` is also blocked by Redline 14.
- The loop `if hasattr(event,"text"): response += event.text` also appends the final `TurnComplete.text`, which holds the whole response. The reply is doubled, `json.loads` fails, and the code silently returns `{"winner":"tie"}`.

**Fix:** Hint `make_delegate(model=MODEL_NAME)` plus `run_delegate_text`, as in the solution.

### [BLOCKING] E4 — Local DPO and SFT hints pass a polars DataFrame to `AlignmentPipeline.train`
**File:** `local/ex_3/03_dpo_training.py:104-106`; `local/ex_2/06_sft_alignment_pipeline.py:141-143`
**Evidence:**
- The hints are `pipeline.train(None, …, preference_dataset=train_pref)` and `pipeline.train(train_data, …)` with polars frames.
- kailash_align's `_validate_preference_columns` uses `dataset.column_names`, and `hasattr(pl.DataFrame(), 'column_names')` is False.
- The solutions convert with `Dataset.from_dict(...to_dict(as_series=False))` and say the conversion is required.

**Fix:** Add the conversion step to both hints.

### [MAJOR] E5 — INT8 memory saving stated as about 6× versus FP16
**File:** `solutions/ex_2/05_quantisation.py:50`
**Evidence:** "INT8: 256 levels (~6x memory savings vs FP16 for weights)". The file's own chart uses 2 bytes versus 1 byte.
**Fix:** "~2× vs FP16 (4× vs FP32)".

### [MAJOR] E6 — SFT "instruction tuning" actually trains on raw review text
**File:** `solutions/ex_2/06_sft_alignment_pipeline.py:168-175`; `shared/mlfp06/ex_2.py:73-85`
**Evidence:**
- `train_data` has columns `instruction, response, text, label`, and `text` is the raw review.
- `SFTConfig.dataset_text_field` defaults to `'text'`.

**Problem:** The adapter learns IMDB text, not the instruction → "Sentiment: X" mapping the header describes.
**Fix:** Build `text = instruction + "\n\n" + response`, or apply the chat template, before `Dataset.from_dict`.

### [MAJOR] E7 — Ex 1.2–1.5 plot hardcoded "expected" baselines beside real results
**File:** `solutions/ex_1/02_few_shot.py:163`; `03_chain_of_thought.py:153-166`; `04_zero_shot_cot.py:137-157`; `05_self_consistency.py:180-200`
**Evidence:**
- `zero_shot_expected = {"accuracy": 0.80, "total_tokens": 1500, …}` and `cot_expected = {"accuracy": 0.90, …}` are fed into "Prompting Ladder — All 4 Methods Compared".
- In 1.5, n=10 measured documents are compared against baselines that claim n=20.

**Fix:** Persist each technique's metrics to `OUTPUT_DIR` and load them, or plot only measured values.

### [MAJOR] E8 — Position-bias test counts judge parse failures and ties as "consistent"; no mitigation is implemented
**File:** `solutions/ex_3/04_grpo_and_judge.py:140-153, 186-190` (the local copy has the same pattern)
**Evidence:**
- `except (ValueError, Exception): return {"winner": "tie", …}`.
- The test sets `consistent_this = ab_picks_good == ba_picks_good`, so two failures (False == False) count as consistent.

**Problem:** A judge that always fails scores 100% consistency and "Bias: LOW". Spec 6.3 also requires mitigation (swap-and-average), which is absent.
**Fix:** Count failures separately. Count as consistent only when the good answer is picked in both orders. Add a swap-averaged judge.

### [MAJOR] E9 — Ex 4.5: HyDE "improvement" uses a non-comparable score, precision@k is meaningless, and judges fall back silently
**File:** `solutions/ex_4/05_rerank_rag_pipeline.py:200-205, 217-220, 352-361, 426-440`; `local/ex_4/05`
**Evidence:**
- HyDE is judged by comparing the top-1 cosine similarity of two different query vectors.
- "Precision@k" treats any chunk whose first three words appear in the answer as relevant.
- Parse failures become `score=5.0` / `0.5`.
- In the local file `hyde_retrieve` is never called and its assert was dropped.

**Fix:**
- Measure hit@k against each eval question's source document for BM25, dense, hybrid and HyDE.
- Raise on judge parse failures.
- Restore the HyDE call and assert in the local file.

### [MAJOR] E10 — Ex 4.2 theory still describes the old 8-dim "LLM-as-projector" embeddings
**File:** `solutions/ex_4/02_dense_retrieval.py:74, 92-94, 185-186`
**Evidence:**
- The text says "We use 8-dim vectors generated by Delegate…", but the code uses `EMBED_DIM = 768` (nomic-embed-text).
- Line 102 prints all 768 floats.

**Fix:** Rewrite the theory to describe 768-dim nomic-embed-text embeddings, and print only a few components.

### [MAJOR] E11 — Every ex_1–ex_8 technique file ends with dead `if False:` Observatory diagnostics and a fabricated "EXPECTED OUTPUT"
**File:**
- All 39 solution technique files and their local counterparts.
- Examples: `solutions/ex_1/01_zero_shot.py:206-243`, `ex_5/01_react_agent.py:305-337`, `ex_7/01_org_compile.py:337`, `ex_8/04_drift_monitoring.py:398`.

**Evidence:**
- Each block starts `if False:  # scaffold — requires OPENAI_API_KEY + judge budget` (or "a live Delegate + API key", "a trained base + adapter", …).
- It is followed by "EXPECTED OUTPUT (synthesised reference)" with invented numbers ("judge coherence 0.91", "5 TAOD steps … $0.017", "7 handoffs, 840ms").
- ex_5/04 references an undefined `react_agent`.

**Problem:**
- The Observatory, the module's diagnostic thread, never runs.
- The OpenAI/API-key premise contradicts Ollama-first.
- The stated prerequisites are produced in the same file.

**Fix:** Run the relevant lens on the in-file results, or delete the blocks and their invented outputs.

### [MAJOR] E12 — Ex 1 promises cost tracking and a cost budget; neither exists
**File:** `solutions/ex_1/01_zero_shot.py:12, 254-257` (the same text is in every ex_1 file and local)
**Evidence:**
- "Measure accuracy, cost, and latency"; "Invoked an LLM via Kaizen Delegate with a cost budget".
- `make_delegate` forces `budget_usd=None`, and the metrics are tokens only.

**Problem:** Spec 6.1 requires cost tracking.
**Fix:** Reword to tokens and latency, and add a tokens × reference-price estimate to meet the spec.

### [MAJOR] E13 — Spec 6.2 exercise gap: no LoRA or adapter training, and merging only on synthetic tensors
**File:** `solutions/ex_2/01_lora_from_scratch.py:183-190`; `02_adapter_from_scratch.py:131-136`; `04_model_merging.py:78-120`
**Evidence:**
- "TASK 3 — TRAIN: no-op — LoRA starts as identity"; the IMDB data is never used in 2.1 or 2.2.
- TIES runs on `torch.randn(128,128)*0.1`.
- DARE and task arithmetic appear only in comments.

**Problem:** The spec requires a performance, parameter-count and training-time comparison, plus a functional merged model.
**Fix:**
- Add a short training loop for LoRA and the adapter (accuracy and wall-clock).
- Merge two trained adapters and evaluate the result.
- Implement DARE.

### [MAJOR] E14 — TIES sign election deviates from the method
**File:** `solutions/ex_2/04_model_merging.py:93-99`
**Evidence:** `elected_sign = (sign_A + sign_B).sign()`. On a sign conflict this is 0, so the parameter is zeroed. Trimming uses a fixed absolute threshold.
**Fix:** `elected_sign = (δA_trim + δB_trim).sign()`, with a top-k% trim per task.

### [MAJOR] E15 — Spec 6.3 and 6.4 exercise coverage gaps
**File:** `solutions/ex_3/04_grpo_and_judge.py:288-301`; `solutions/ex_4/*`
**Evidence:**
- lm-eval-harness appears only as a printed usage string; there are no before/after benchmarks.
- Judge bias is not mitigated.
- RAG methods are compared on a single query with no recall metric.
- There are no Kaizen RAG agents.
- The corpus is neural-bridge, not the Singapore policy documents the spec names.

**Fix:**
- Run a small lm-eval (e.g. `hellaswag --limit 50`) before and after DPO.
- Add an eval-set hit@k table.
- Add a swap-averaged judge.

### [MAJOR] E16 — UltraFeedback described as human-labelled
**File:** `solutions/ex_3/01_preference_data.py:27, 88`; `local/ex_3/01_preference_data.py:173`
**Evidence:** "Real human-curated preference pairs"; "CHOSEN (human-preferred)".
**Problem:** UltraFeedback ratings are GPT-4-generated (AI feedback).
**Fix:** Change to "AI-feedback (GPT-4-rated) preference pairs".

### [MAJOR] E17 — Ex 1 local scaffolds drop every visualisation
**File:** `local/ex_1/01..06` ("TASK 4 — VISUALISE" contains only `print_summary`)
**Evidence:** The solutions call `plot_accuracy_bars`, `plot_comparison_bars`, `plot_tokens_vs_accuracy`, `plot_vote_agreement` and `plot_extraction_accuracy`. The locals have neither the calls nor blanks for them, and the imports were removed.
**Fix:** Restore the plot calls, as provided code or as TODOs (Redline R9A).

### [MAJOR] E18 — False claim that AlignmentConfig validates LoRA target modules against the model
**File:** `solutions/ex_2/06_sft_alignment_pipeline.py:56-58`
**Evidence:** "AlignmentConfig validates targets against the base model's module tree at construction time". The installed config only checks `if not self.target_modules: raise`.
**Fix:** Remove the claim.

### [MINOR] E19 — Checkpoint asserts stripped from local files
**File:**
- `local/ex_3/02_dpo_loss.py` (drops `assert cost_balanced < max(...)`)
- `local/ex_3/03_dpo_training.py` (drops `dpo_config.lora.rank == 16` and `0 <= aligned_rate <= 1`)
- `local/ex_4/05_rerank_rag_pipeline.py` (drops `len(hyde_results) > 0`)

**Fix:** Restore the asserts. exercise-standards.md says checkpoints are never stripped.

### [MINOR] E20 — Data loading blanked in local files
**File:** `local/ex_2/01:59`, `local/ex_2/06:66`, `local/ex_4/01:47-48`
**Evidence:** `sft_data, train_data, eval_data = ____`; `corpus = ____`.
**Fix:** Provide these lines. exercise-standards.md says "MUST NOT strip data loading code".

### [MINOR] E21 — Numeric and reference slips in ex_1–ex_4 prose
**File and evidence:**
- ex_1/03:217: going from 85% to 92% on 200 intakes per day is about 14 fewer errors in total, not "~14 FN + ~9 FP per day". Line 223 says "PACT (see Exercise 6)", but PACT is Exercise 7.
- ex_2/03:299: the prose says "~S$480,000/year", but the code prints S$973,440.
- ex_2/02:274,277: "~65K params at r=16 for a 7B base" — it is about 8.4M.
- ex_2/02:154: the identity gap is exactly 0, not "not exactly 0 due to LayerNorm". The parameter-count formula is wrong (it should be `2db+b+3d`).
- ex_2/05:184: "well under 1%" — recomputed with the same seed it is 1.14%.
- ex_4/03:77: "BM25 powered Google". It was Okapi, then Lucene/Elasticsearch.
- ex_3/04:295: says lm-eval runs MT-Bench; it does not.
- ex_2/01:189 and ex_2/02:135 reference `05_sft_alignment_pipeline.py`; the file is `06_…`.
- ex_3/01:165: cites PDPA "recurring-authorisation rules"; PDPA is a data-protection law.

**Fix:** Correct each item.

### [MINOR] E22 — Opt-in synthetic training metrics, and file-existence checkpoints that pass on stale outputs
**File:** `solutions/ex_2/06:122-133` (`MLFP_SKIP_SFT_TRAIN=1` reports `train_loss 0.742`); `assert fname.exists()` in `local/ex_2/03-06`
**Problem:** The skip path fakes metrics that pass the checkpoints. The existence asserts pass on a PNG left behind by an earlier solution run, even if the student never fills the plot blank.
**Fix:** Make the skip path raise, or label its output clearly. Delete the file before plotting, or assert on the figure object.

---

## 5. Exercises ex_5 – ex_8 (solutions, local scaffolds, shared/mlfp06/ex_5..8)

All 18 local scaffolds have the header and REFLECTION blocks. No hardcoded model names (everything resolves through `DEFAULT_CHAT_MODEL`); the one exception is the provider-key gate in ex_7/04 (F6).

### [BLOCKING] F1 — Capstone LLM never runs: every call quietly falls back to a canned stub, and the "answer" is the question echoed back
**File:** `shared/mlfp06/ex_8.py:143-155` (`CapstoneQAConfig`), `498-536` (`_capstone_execute_node`), `583-593` (`handle_qa`). Used by every ex_8/02–05.
**Evidence:**
- `CapstoneQAConfig` lacks `use_async_llm`, `response_format` and `structured_output_mode`. ex_6's configs set them.
- `CapstoneQAAgent(CapstoneQAConfig()).run_async(question="hi")` raises `ValueError Agent not configured for async mode`.
- `_capstone_execute_node(None, {"objective": "What is ML?"})` returns `{'result': '[offline-fallback (ValueError)] What is ML?', 'cost': 0.005, …}`.
- `handle_qa` reads `result.audit_trail[-1].get("result")`, but audit records have no `result` key; node output lives in `result.results`. So the answer is always `f"[capstone:{role}] {question}"`.
- `confidence` is hardcoded to 0.85.

**Problem:** This is the silent stub Redline 14 forbids. The capstone appears to work while no LLM fires.
**Fix:**
- Add the async and structured-output fields.
- Remove the fallback, or raise `OllamaUnreachableError`.
- Read `next(iter(result.results.values()))`.
- Return the agent's own confidence.

### [BLOCKING] F2 — Ex 5.1: the ReAct agent never calls its tools
**File:** `solutions/ex_5/01_react_agent.py:109-110, 144`; same wiring in `02_cost_budget_agent.py:84-90`
**Evidence:**
- The code uses `react_agent.available_tools.append(tool)` and then `run_async(task=...)`.
- `available_tools` is forwarded only by `ReActAgent.run()` (`react.py:412`); `run_async` is inherited from BaseAgent.
- `_check_convergence` always returns True.
- Tool execution resolves only MCP tools (`mcp__kaizen_builtin__<name>`), never Python callables.

**Problem:** The Thought/Action/Observation narrative and the tool-order interpretation (lines 162-165) describe behaviour that cannot occur.
**Fix:** Register tools through `make_delegate(tools=ToolRegistry(...))` or an MCP server, and print the real trace.

### [BLOCKING] F3 — Ex 5.2: budget demo is a false positive and the budget is never enforced
**File:** `solutions/ex_5/02_cost_budget_agent.py:82-90, 115-126, 235`; `local/ex_5/02_cost_budget_agent.py:66-68, 98`
**Evidence:**
- `ReActAgent(model=MODEL)` has no provider or async flag, so it raises "Agent not configured for async mode". The `except Exception` prints that as "Budget tripped" for both agents.
- The budget is set after construction, so it is never copied into `execution_context` (base_agent.py:172 copies it only in `__init__`). Output: `config: 0.1 enforced ctx: None`.
- `spent = [0.10 if low_tripped else 0.06, 0.45 …]` is invented.
- The prose relies on a non-existent `LLMCostTracker` (80% warning, refused calls).
- The local hint `max_llm_cost_usd` raises TypeError (see X5).

**Fix:**
- Wire the provider and async flag as in 01, and pass `budget_limit_usd` at construction.
- Report a trip only when the budget exception is the cause, and plot measured spend.
- Remove the `LLMCostTracker` text, and note that budgets are a no-op on Ollama.

### [BLOCKING] F4 — Capstone Nexus deployment serves a stub, auth is a hardcoded token dict, and no channel is ever exercised
**File:** `solutions/ex_8/03_multichannel_serving.py:52-61, 120-137, 160-175, 225-233`; `shared/mlfp06/ex_8.py:618-634`
**Evidence:**
- The registered workflow body is `result = {'answer': f'[nexus-stub] {question}', 'role': role}`, so governance (`serve_qa`/`handle_qa`) never runs on any channel.
- `SimpleJWTAuth.VALID_TOKENS = {"token_viewer_001": …}`: no JWT is parsed or signed.
- `RateLimiter` is hand-rolled, CORS is never set, and no API, CLI or MCP request is made.
- Latency histograms come from `rng.lognormal`.
- Installed Nexus provides `register_handler(name, handler_func, …)`, `rate_limit=`, `cors_origins=` and `NexusAuthPlugin(jwt=JWTConfig, rbac=…)`.

**Problem:** The spec's 6.8 assessment criteria (3 channels, 401 for unauthenticated calls, governance in production) are unmet, and the theory claim that governance runs on every channel is false.
**Fix:**
- `app.register_handler("capstone_serve_qa", serve_qa)` with `NexusAuthPlugin`, `rate_limit` and `cors_origins`.
- Make one real call per channel, plus a 401 test.
- Drop the simulated latency plot.

### [BLOCKING] F5 — Ex 8.1 adapter loading uses the wrong AdapterRegistry API and never loads an adapter
**File:** `solutions/ex_8/01_adapter_loading.py:80-102, 130-155, 178-186`; `local/ex_8/01_adapter_loading.py:32, 110-115`
**Evidence:**
- `AdapterRegistry()` is in-memory (`model_registry=None`), so a new process always sees an empty registry.
- `list_adapters`/`get_adapter` return `AdapterVersion` objects (`hasattr(AdapterVersion,'get')` is False), but the code calls `a.get('name')`. A bare `except Exception: continue` hides the failure.
- `AlignmentServing` is constructed but never deploys anything.
- The plot uses an invented 7B base model.
- The local hint `AlignmentConfig(method="inference", adapter_path=...)` raises TypeError.

**Fix:**
- Use a persistent registry shared with Ex 2/3, and the `av.adapter_name` / `av.base_model_id` attributes.
- Remove the bare except.
- Actually deploy or load the adapter, and plot real `lora_config` counts.
- Regenerate local Task 3 from the solution.

### [MAJOR] F6 — Ex 7.4 governed runtime always uses a fake executor, gated on OpenAI/Anthropic keys
**File:** `solutions/ex_7/04_runtime_audit.py:150-168`; `local/ex_7/04_runtime_audit.py:~120-130`; `shared/mlfp06/ex_7.py:460-517`
**Evidence:**
- `executor = make_fake_executor() if not live_mode else make_fake_executor()`.
- `live_mode = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))`.

**Problem:** The adversarial "blast radius" test only ever runs `[offline-fake]`. This breaks Redline 14 and Ollama-first.
**Fix:** Build the executor around `make_delegate()`, and remove the key gate and the fake executor.

### [MAJOR] F7 — Visualisations and "results" in ex_5 – ex_8 are hardcoded or simulated
**File:**
- **ex_5:** `01_react_agent.py:267-269` (`step_counts=[5,3,4]`, `latencies_s=[12.4,8.1,10.7]`)
- **ex_6:**
  - `03_parallel_router.py:261` (`route_counts=[1,1,1]`)
  - `04_mcp_server.py` (`simulated_calls=[45,15,120]`)
  - `05_memory_and_security.py:377, 402` (`mitigated=[1.0,1.0,0.8,0.9,0.85]`)
- **ex_7:**
  - `02_envelopes.py:488-493` (radar shows clearance 1/3, 2/3, 3/3 for envelopes that are all RESTRICTED)
  - `04_runtime_audit.py:411-440, 463-476` (all six regulations "COMPLIANT", `counts=[6,2,1]`)
- **ex_8:**
  - `04_drift_monitoring.py:357` (`psi_values=[…]`)
  - `05_compliance_audit.py:222` (checkpoint `(regulatory["Status"] == "COMPLIANT").all()`)

**Problem:** These break zero-tolerance Rule 2 and Redline 9A. Students read fabricated measurements and compliance attestations as outputs of their own run.
**Fix:** Plot values computed in the run, or delete the plots. Derive any compliance status from checks that actually execute.

### [MAJOR] F8 — Ex 8.5 compliance report asserts controls that do not exist
**File:** `solutions/ex_8/05_compliance_audit.py:145, 154, 160-190, 214-222`
**Evidence:**
- It prints "Auth method: JWT (RS256)" (actually a stub dict), "CORS: ACTIVE" (never set), "GovernedSupervisor on ALL channels" (Nexus serves a stub), "SFT adapter: imdb_sentiment_sft_v1" (the registry is empty), "PII masking" (none exists) and "Generated: 2026-04-14" (hardcoded).
- `qa_ok` counts `e.get("success")`, a key the audit records do not have, so it is always 0/0.

**Fix:** Generate every line from live objects (auth config, registry lookup, today's date), count outcomes from `record_type`/`details`, and drop unverifiable claims.

### [MAJOR] F9 — Ex 8.4 drift check cannot detect drift, and the tests cannot fail
**File:** `solutions/ex_8/04_drift_monitoring.py:128, 234, 273`, Tests 2-5
**Evidence:**
- `production_df = reference_df.head(50)`: a subset of the reference, so PSI ≈ 0 by construction.
- Tests 2 and 5 hardcode `"passed": True`.
- Tests 3 and 4 only check for no error, and the F1 fallback guarantees no error.

**Fix:** Build a genuinely shifted production sample, and assert concrete governance outcomes, including a deny case.

### [MAJOR] F10 — Ex 8.2 "monotonic tightening" interpretation is factually wrong
**File:** `solutions/ex_8/02_governance_pipeline.py:171-175, 187`
**Evidence:**
- The text says "admin is a superset of qa, audit is a superset of admin". The audit tool list lacks admin's `update_model` and `monitor_drift`.
- Monotonic tightening constrains parent → child, not sibling tiers.
- `clearance_numeric=[1,2,3]` contradicts the actual clearances (RESTRICTED, RESTRICTED).

**Fix:** Describe the tiers as sibling envelopes, each tighter than `ml_director`, and plot the real clearances.

### [MAJOR] F11 — Ex 6.3 LLM router is configured but never called
**File:** `solutions/ex_6/03_parallel_router.py:168-170, 203-208`; `local/ex_6/03_parallel_router.py:137-139`
**Evidence:** The Task 5 loop only prints `Expected specialist: …`; the comment says it avoids "burning LLM budget", even though the backend is free Ollama. The Reflection and pie chart claim routing happened.
**Fix:** Call the router for each query, and compare and plot its choice against `expected`.

### [MAJOR] F12 — Ex 6.4 MCP server: placeholder tool, schemas never attached, and no client ever calls it
**File:** `solutions/ex_6/04_mcp_server.py:145, 205-215`, Task 4
**Evidence:**
- `analyse_passage` returns `"[would run {analysis_type}_agent against the passage]"`.
- The `StructuredTool(...)` cards are built in a separate dict and never attached.
- Task 4 reads the private `mcp_server._tool_registry`.
- No MCP client or agent ever calls the server.

**Fix:** Run the real specialist inside the tool, register schemas with `@structured_tool(input_schema=…)`, and add an in-process client or `mcp_servers=` agent call.

### [MAJOR] F13 — Ex 6.5 prompt-injection "guard" is a keyword replace, and data isolation is a literal string
**File:** `solutions/ex_6/05_memory_and_security.py:236-252`
**Evidence:**
- `malicious_output.replace("IGNORE","[BLOCKED]").replace("INSTRUCTIONS",…)`.
- `sanitised_summary` is a hardcoded literal.
- The checkpoints assert on these literals.

**Problem:** A denylist is taught as an injection defence, and the checkpoints test nothing.
**Fix:**
- Mask the NRIC with a regex applied to the document.
- Teach injection mitigation as data/instruction separation plus envelope-limited tools.
- Test against a paraphrased attack.

### [MAJOR] F14 — Spec coverage gaps for lessons 6.5–6.8 at the exercise level
**File:** `solutions/ex_5/*`, `shared/mlfp06/ex_5.py:111-241`, `solutions/ex_6/*`, `solutions/ex_7/03_budget_access.py`, `solutions/ex_8/*`
**Evidence:**
- **6.5:** The tools are HotpotQA keyword lookups, not DataExplorer, TrainingPipeline or ModelVisualizer. `answer_question` returns the ground truth. `tool_choice` and parallel calls are printed text only. There is no human-in-the-loop and no ChainOfThoughtAgent.
- **6.6:** The required DataScientist → FeatureEngineer → ModelSelector → ReportWriter pipeline is replaced by SQuAD factual/semantic/structural analysis. There is no handoff pattern and no A2A (both appear only in `if False` blocks). The MCP tools do not expose Kailash engines.
- **6.7:** Budget cascading uses a hand-rolled `TeachingBudgetTracker`, not the `FinancialConstraintConfig` cascade plus `budget_limit_usd`, and there is no graceful degradation.
- **6.8:** No real JWT/RBAC, no API/CLI/MCP end-to-end queries, and no reasoning-chain debugging (see F4).

**Fix:** Implement these items, or record the deviations in specs/module-6.md.

### [MINOR] F15 — Local hints point at the synchronous `run`
**File:** `local/ex_5/01_react_agent.py:123` ("Await react_agent.run(task)"); `local/ex_5/02_cost_budget_agent.py:98`
**Fix:** `await ….run_async(task=…)`, as in the solutions.

### [MINOR] F16 — Ex 7.4 checkpoint asserts differ between solution and local
**File:** `solutions/ex_7/04_runtime_audit.py` vs `local/ex_7/04_runtime_audit.py` (15 vs 14 asserts)
**Evidence:**
- The local file drops the `.level == "blocked"` asserts.
- The solution's Checkpoint 4 accepts `== 10 or == 0`, which passes when the outer `except` skips the whole test.

**Fix:** Make both files identical, and remove the `== 0` escape and the catch-all.

### [MINOR] F17 — Ex 7.1 clearance table contradicts the YAML
**File:** `solutions/ex_7/01_org_compile.py:186`
**Evidence:** The table shows `data_analyst` as "internal"; the YAML says "restricted".
**Fix:** Build the table from `loaded.clearances`.

### [MINOR] F18 — Supervisor-worker "fan-out" runs sequentially while the theory promises max(stages) latency
**File:** `solutions/ex_6/01_supervisor_worker.py:114-119`; `ex_6/02` theory text
**Fix:** Use `asyncio.gather`, or correct the latency claim.

### [MINOR] F19 — Arithmetic slips in business-impact text
**File:** `solutions/ex_5/02_cost_budget_agent.py:191-202`; `solutions/ex_6/01_supervisor_worker.py:245-247`
**Evidence:**
- $2,000/day × 365 ≈ $730K, not "S$270,000/year".
- 10 attackers × $0.25 × 365 ≈ $912, not "S$350".
- USD and S$ are mixed.
- A 7-point lift in fraud catch rate is not "~550 extra fraud cases" (7% of all 8,000 claims).

**Fix:** Recompute in one currency.

### [MINOR] F20 — Unverifiable regulation, stale API names and a wrong cross-reference
**File:**
- `solutions/ex_5/02:204` ("MAS FSM-GL-04" — no such instrument found)
- `ex_6/01:62` ("Kaizen max_llm_cost_usd")
- `ex_8/01:239` (`$DEFAULT_LLM_MODEL`)
- `ex_5/04:73` ("Ex 1 Task 6"; self-consistency is Ex 1.5)

**Fix:** Remove the citation or replace it with a real one. Update the names to `budget_limit_usd` / `OLLAMA_CHAT_MODEL`, and correct the reference.

---

## 6. Lesson pages (`modules/mlfp06/lessons/01..08`)

The lesson pages are an older fork of the master deck and textbook. No relative links are broken. Problems shared with the master materials are reported under X1–X12.

### [BLOCKING] L1 — Lesson 6.1 Delegate / SimpleQAAgent code raises at construction
**File:** `lessons/01/slides.html:325-378`; `lessons/01/textbook.html:415-425, 448-476, 502-619`; `lessons/01/notes.html:344`
**Evidence:**
- `Delegate(model=…, max_llm_cost_usd=0.50)` raises TypeError.
- `SimpleQAAgent` already passes `signature=QASignature()` to super (`simple_qa.py:205-210`), so adding `signature=ReportExtraction` raises TypeError (multiple values).
- `SimpleQAAgent.run(question, …) -> dict` is synchronous, so `await agent.run(report_text=…)` and `result.category` fail.
- textbook.html:430-432 claims "the max_llm_cost_usd guard trips hard". The master uses `make_delegate` with `budget_usd=None`.

**Fix:** Port the master pattern: `make_delegate(signature=…)`, or BaseAgent + Signature with an Ollama config.

### [BLOCKING] L2 — Lesson 6.4 RAG embedding code calls a non-existent `Delegate.embed`
**File:** `lessons/04/textbook.html:503-516, 607`
**Evidence:**
- `hasattr(Delegate, 'embed')` is False.
- `Delegate(model=model, max_llm_cost_usd=2.0)` raises TypeError.
- `os.environ["DEFAULT_LLM_MODEL"]` raises KeyError, because the variable is commented out in `.env.example`.

**Fix:** Use `make_embedder()` / `make_delegate()` from `shared.mlfp06._ollama_bootstrap`.

### [BLOCKING] L3 — Lessons 6.5/6.6 agent code: wrong imports and signatures throughout
**File:**
- `lessons/05/slides.html:106-114`
- `lessons/05/textbook.html:447-512, 518-531, 627-634`
- `lessons/06/slides.html:205-230`
- `lessons/06/textbook.html:237-327, 378-401, 467-487, 556-602, 671-773`

**Evidence:**
- **Imports that fail:**
  - `from kaizen.tools import tool` raises ImportError.
  - `kaizen_agents` has no `BaseAgent`.
  - `from kailash_mcp import MCPServer, tool` raises ImportError.
- **MCP server:**
  - `MCPServer(..., version=…)` is not a valid argument.
  - `server.tool("explore")` binds "explore" to `cache_key`.
- **ReActAgent:**
  - `ReActAgent(tools=…, max_llm_cost_usd=…)` raises TypeError.
  - A subclass passing `signature=` collides with ReActAgent's own (`react.py:259-265`).
- **kailash_ml:**
  - `TrainingPipeline(algorithm=…).fit` does not exist.
  - `DataExplorer.profile` is async.
  - `ModelVisualizer.compute_importance` does not exist.
- **A2A and memory:**
  - `A2ATask(sender=, receiver=, payload=, metadata=)` uses fields that do not exist.
  - `A2AAgentCard` requires `agent_name`, `agent_type` and `version`, and has no `capabilities`.
  - `SharedMemoryPool(...)` takes no arguments and has no `.get` (it has `read_all`/`read_relevant`/`write_insight`).

**Fix:** Rebuild the samples from `solutions/ex_5/*` and `solutions/ex_6/*`, and the master textbook's `make_delegate(tools=…)` pattern (with a `ToolRegistry`, per D2).

### [BLOCKING] L4 — Lesson 6.7 PACT / GovernedSupervisor code fails or prints fabricated output
**File:** `lessons/07/slides.html:197-214`; `lessons/07/textbook.html:407-431, 466-491, 568-576, 605, 619-630, 791, 808`; `lessons/08/textbook.html:464, 561, 614, 619`
**Evidence:**
- **Org YAML:** fails to load with `ConfigurationError: Required field 'name' is missing`. The nested `head`/`tasks`/`responsible` schema is not the real flat schema, and the printed node list at line 431 is fabricated.
- **`denied_actions`:** `OperationalConstraintConfig` has `blocked_actions`, not `denied_actions`; the `denied_actions=` kwarg is silently dropped. The printed reasons "in envelope.operational.denied_actions" (605, 629) are fabricated.
- **Audit records:** the keys are `record_id, record_type, timestamp, agent_id, parent_id, action, details, prev_hash, record_hash`, so `entry['role_address'/'verdict'/'reason'/'phase']` raises KeyError.
- **Other missing attributes:** `AgentSpec.role` does not exist, and `SupervisorResult` has no `failure_reason` or `output`.
- **`validate_tightening`:** called positionally (see X7).

**Fix:**
- Use the flat YAML schema from `shared/mlfp06/ex_7.py`, and `blocked_actions`.
- Use the real audit keys, `AgentSpec.name`/`capabilities`, and `result.results`/`result.success`.

### [MINOR] L5 — Wrong lesson cross-references
**File:** `lessons/03/textbook.html:644, 648`; `lessons/04/textbook.html:712`; `lessons/02/textbook.html:786-787`
**Evidence:**
- KL for variational inference is cited as Lesson 5.5; it is 5.1.
- Logistic loss is cited as 3.2; it is 2.6.
- Cosine/clustering is cited as 4.4; it is 4.1.
- Optimisers are cited as 5.2/5.3; they are in 4.8.

**Fix:** Correct the lesson numbers.

### [MINOR] L6 — MCP described as having a REST `/tools` endpoint
**File:** `lessons/06/textbook.html:616-618, 866-868`
**Problem:** MCP discovery is the JSON-RPC `tools/list` method. With `transport="stdio"` there is no HTTP endpoint at all.
**Fix:** Say "call `tools/list`".

### [MINOR] L7 — Agent-memory model contradicts itself in lesson 6.6
**File:** `lessons/06/slides.html:220` and `06/notes.html:13` (short/long/entity tiers) vs `06/textbook.html:497-500` ("from earlier Kaizen releases") vs `:663, 841-844`
**Fix:** Use one framing, and map the spec's three tiers onto BufferMemory and SharedMemoryPool.

### [MINOR] L8 — lm-eval-harness claims
**File:** `lessons/03/slides.html:402-430, 495`; `lessons/03/textbook.html:415-424, 435-446`
**Problem:** MT-Bench is not an lm-eval task. `metrics['acc,none']` raises KeyError for humaneval, which reports `pass@1`.
**Fix:** Drop MT-Bench from the harness list, and key each task's metric correctly.

### [MINOR] L9 — GovernedSupervisor called a wrapper in 6.5 but "not a wrapper" in 6.7
**File:** `lessons/05/textbook.html:813-814` vs `lessons/07/textbook.html:552-556`
**Fix:** Align 6.5's wording with 6.7: it is a planner driven by an `execute_node` callback.

### [MINOR] L10 — Lesson 6.4–6.8 speaker notes are thin and mis-numbered against their slides
**File:** `lessons/04/notes.html:14` (says 12 slides; there are 11); `lessons/06/notes.html:8` (says 9; there are 12, and the "Slides 6–9" block is mis-numbered); `lessons/07/notes.html:8` (says 9; there are 11)
**Problem:** These notes have 3–6 grouped paragraphs, while the master `speaker-notes.md` has per-slide notes for the same lessons. The added diagram slides in 06 and 07 have no notes.
**Fix:** Regenerate the lesson notes from the master `speaker-notes.md`, numbered against the current slides.

---

## Coverage summary

**What was checked, by check number:**
1. **Spec coverage:** every topic and learning objective in `specs/module-6.md` was compared against the deck, textbook and notes (grep counts) and against the exercises.
2. **Technical correctness:** the full master deck (4,641 lines, 105 sections, every slide incl. notes), `textbook.md` (1,649 lines), the high-risk sections of `speaker-notes.md` (1,338 lines; grep-targeted on every number and API claim, plus a slide-count reconciliation), `README.md` and `index.html`.
3. **Code shown to students:** every API call in the deck, textbook and lesson pages was checked against the installed `.venv` with `inspect.signature`, `hasattr` or construction probes. No LLM calls, training or graders were run, apart from the task 4 grader.
4. **Exercises:** all 39 solution technique files under `solutions/ex_1..8` were read in full. Each was diffed against its local scaffold (39 files) for blanks, asserts, hints and headers. All 8 `shared/mlfp06/ex_N.py` files, `_ollama_bootstrap.py` and the 12 `shared/mlfp06/diagnostics/*` modules were checked for signatures and use.
5. **Assessment:** README, plus `problem.md`, `starter.py`, `solution.py` and `grader.py` for all 4 tasks (17 files). The task 4 grader was run on its solution and passed 10/10. Task 2's gold indices were re-derived from parquet.
6. **Internal consistency:** all 24 lesson pages (`lessons/01..08/{slides,textbook,notes}.html`) were compared with the master deck, textbook and notes. Every relative link and every referenced dataset and exercise path was checked against the filesystem (`data/mlfp06/`).
7. **Independence and naming:** grep for institutions, companies, course codes and commercial products across the module and shared package.
8. **Hardcoded models:** grep for model names and `LLM_MODEL` / `DEFAULT_LLM_MODEL` across the module and shared package, cross-checked against `.env.example` and `_ollama_bootstrap.py`.

**Files read:** about 145 non-generated files in `modules/mlfp06` (excluding PDFs) and 22 in `shared/mlfp06`.

**Biggest systemic issues:**
- The master deck, textbook and lesson pages teach many APIs that do not exist in the installed stack (X4, X5, X7, X8, D1, D2, L1–L4).
- The governance lesson's core facts are wrong: the clearance order (X1), fail-closed behaviour (X3) and the meaning of D/T/R (X10).
- Several exercises "pass" via silent fallbacks or fabricated numbers rather than real LLM or tool execution (E1, E7, E11, F1–F3, F7–F9).
