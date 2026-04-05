# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""ASCENT Module 6 — AI-Resilient Assessment Questions

Alignment, Governance, RL & Deployment
Covers: SFT, DPO, RL (PPO/SAC), model merging, PACT, governed agents,
        budget cascading, AuditChain, capstone deployment
"""

QUIZ = {
    "module": "ASCENT6",
    "title": "Alignment, Governance, RL & Deployment",
    "questions": [
        # ── Lesson 1: SFT fine-tuning ─────────────────────────────────────
        {
            "id": "6.1.1",
            "lesson": "6.1",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student configures AlignmentConfig for SFT fine-tuning but the training "
                "crashes with an OOM (out-of-memory) error on the first batch. "
                "They have 16GB VRAM. What is likely wrong and what config change fixes it?"
            ),
            "code": (
                "from kailash_align import AlignmentConfig, AlignmentPipeline\n"
                "\n"
                "config = AlignmentConfig(\n"
                "    method='sft',\n"
                "    base_model=os.environ['SFT_BASE_MODEL'],\n"
                "    dataset_format='instruction',\n"
                "    lora_r=16,\n"
                "    lora_alpha=32,\n"
                "    lora_dropout=0.05,\n"
                "    target_modules=['q_proj', 'v_proj'],\n"
                "    num_train_epochs=3,\n"
                "    per_device_train_batch_size=32,  # Bug: batch size too large\n"
                "    max_seq_length=2048,\n"
                "    gradient_checkpointing=False,  # Bug: not enabled\n"
                ")"
            ),
            "options": [
                "A) lora_r=16 is too large; reduce to lora_r=4",
                "B) Use method='sft_lora' instead of method='sft' to enable memory-efficient training",
                "C) AlignmentConfig does not support max_seq_length; remove the parameter",
                "D) per_device_train_batch_size=32 at max_seq_length=2048 fills 16GB VRAM. Reduce to per_device_train_batch_size=4 and set gradient_checkpointing=True — this trades compute for memory, enabling training on 16GB hardware at the cost of ~30% slower training"
            ],
            "answer": "D",
            "explanation": (
                "Batch size × sequence length × model parameters dominates VRAM. "
                "At batch=32 and seq_len=2048, activations alone exhaust 16GB for a 7B model. "
                "Reducing batch size to 4 cuts the activation footprint 8×. "
                "gradient_checkpointing=True recomputes activations during the backward pass "
                "instead of storing them — trades ~30% compute for ~50% memory reduction. "
                "Both changes together make the training fit on 16GB."
            ),
            "learning_outcome": "Tune AlignmentConfig batch size and gradient checkpointing for VRAM constraints",
        },
        {
            "id": "6.1.2",
            "lesson": "6.1",
            "type": "context_apply",
            "difficulty": "advanced",
            "question": (
                "In Exercise 1, you fine-tune a model on Singapore domain Q&A pairs "
                "and register the adapter in AdapterRegistry. "
                "A colleague asks: 'Why use LoRA instead of full fine-tuning?' "
                "Your LoRA config has lora_r=16, lora_alpha=32, target_modules=['q_proj', 'v_proj']. "
                "Explain the memory advantage mathematically for a 7B parameter model "
                "where q_proj has dimension 4096×4096."
            ),
            "options": [
                "A) LoRA is faster at inference; memory during training is the same as full fine-tuning",
                "B) LoRA reduces parameters by 16× because lora_r=16 divides the original dimension",
                "C) Full fine-tuning updates all 4096×4096 = 16.7M parameters in q_proj. LoRA decomposes the update as two matrices: A (4096×16) and B (16×4096) = 2 × 65,536 = 131,072 parameters — 128× fewer trainable parameters for this layer. The base model weights are frozen, so only the adapter is stored in optimizer memory",
                "D) LoRA only trains 2 layers (q_proj and v_proj); full fine-tuning trains all layers"
            ],
            "answer": "C",
            "explanation": (
                "For a weight matrix W of shape (d_out, d_in), LoRA replaces ΔW with A×B "
                "where A is (d_out, r) and B is (r, d_in). "
                "For q_proj (4096, 4096) with r=16: "
                "ΔW = 4096×4096 = 16.8M params. "
                "A+B = 4096×16 + 16×4096 = 65,536 + 65,536 = 131,072 params. "
                "Reduction: 16.8M / 131k ≈ 128×. "
                "Across all target_modules, total trainable params stay under 1% of the base model — "
                "the rest is frozen and does not require gradient storage."
            ),
            "learning_outcome": "Calculate LoRA parameter reduction relative to full fine-tuning for a given rank",
        },
        # ── Lesson 2: DPO alignment ───────────────────────────────────────
        {
            "id": "6.2.1",
            "lesson": "6.2",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "You have 8,000 preference pairs (chosen/rejected) for aligning a credit "
                "explanation model to be more helpful and less technical. "
                "Should you use SFT + RLHF (PPO) or DPO, and what practical advantage "
                "does DPO provide for a team without a dedicated RL engineer?"
            ),
            "options": [
                "A) RLHF is always better because it includes a reward model that can be reused",
                "B) SFT alone is sufficient; preference data is only needed for RLHF",
                "C) DPO is preferable: it optimises the preference objective directly from the (chosen, rejected) pairs without training a separate reward model or running PPO's complex actor-critic loop. For a team without RL expertise, DPO reduces the pipeline from 3 stages (SFT → reward model → PPO) to 2 stages (SFT → DPO), with fewer hyperparameters and no KL controller to tune",
                "D) DPO requires at least 100,000 preference pairs; 8,000 is insufficient"
            ],
            "answer": "C",
            "explanation": (
                "RLHF pipeline: (1) supervised fine-tune base model, (2) train reward model on preferences, "
                "(3) run PPO with reward model as the signal. Each stage has its own failure modes. "
                "DPO reframes the preference optimisation as a classification problem directly on the "
                "language model, bypassing the reward model and PPO entirely. "
                "The AlignmentConfig for DPO: method='dpo', beta=0.1 (KL regularisation strength). "
                "8,000 preference pairs is typically sufficient for domain adaptation with DPO."
            ),
            "learning_outcome": "Select DPO over RLHF for teams without RL infrastructure",
        },
        # ── Lesson 3: PACT governance basics ─────────────────────────────
        {
            "id": "6.3.1",
            "lesson": "6.3",
            "type": "output_interpret",
            "difficulty": "intermediate",
            "question": (
                "Running GovernanceEngine.check_permission() for agent 'ml_agent' in the "
                "'data_science/modeling' team returns:\n\n"
                "  PermissionResult(allowed=False, reason='max_cost_usd exceeded: 5.20 > 5.00')\n\n"
                "The agent has spent $5.20 against a $5.00 budget. "
                "What should happen to the agent's current task, and which PACT principle "
                "explains why the budget cannot be raised by the agent itself?"
            ),
            "options": [
                "A) The current task must halt immediately. PACT's monotonic tightening principle states that an operating envelope can only be tightened (never loosened) by the agent itself — the agent cannot raise its own budget. A human operator at a higher organisational level must authorise a budget increase via GovernanceEngine.update_envelope()",
                "B) The agent should retry after 1 minute; budgets reset automatically",
                "C) The agent can continue because $0.20 overage is within the 5% tolerance band",
                "D) The agent should switch to a cheaper model to continue under budget"
            ],
            "answer": "A",
            "explanation": (
                "PACT's monotonic tightening rule: agents can tighten their own constraints "
                "(e.g., reduce their own cost limit) but cannot loosen them. "
                "This prevents an agent from escalating its own permissions in a loop. "
                "When a budget is exceeded, the governance layer halts execution — "
                "it does not give the agent discretion to continue. "
                "Budget increases require explicit authorisation from a human operator "
                "at the parent organisational node."
            ),
            "learning_outcome": "Apply PACT monotonic tightening to explain why agents cannot self-authorise budget increases",
        },
        {
            "id": "6.3.2",
            "lesson": "6.3",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student sets up a PACT organisation but governance checks always return "
                "allowed=True regardless of the agent's permissions. What is wrong?"
            ),
            "code": (
                "from pact import GovernanceEngine, compile_org\n"
                "\n"
                "org_dict = {\n"
                "    'organization': {\n"
                "        'name': 'ASCENT Demo',\n"
                "        'departments': [{'name': 'data_science', 'teams': [...]}]\n"
                "    }\n"
                "}\n"
                "\n"
                "# Bug: org is defined but never compiled/loaded into the engine\n"
                "engine = GovernanceEngine()  # empty engine\n"
                "result = engine.check_permission(address, action)"
            ),
            "options": [
                "A) compile_org(org_dict) is never called and the result never passed to GovernanceEngine — an empty GovernanceEngine has no policy rules and defaults to allowing all actions; pass the compiled org: engine = GovernanceEngine(compile_org(org_dict))",
                "B) GovernanceEngine requires an async context manager",
                "C) org_dict must be a YAML file path, not a Python dict",
                "D) check_permission() requires await because GovernanceEngine is async"
            ],
            "answer": "A",
            "explanation": (
                "compile_org() parses the organisation dict, validates the permission structure, "
                "and produces a compiled policy object. "
                "GovernanceEngine(policy) uses this policy for all subsequent checks. "
                "An empty GovernanceEngine() has no policy to enforce — it is fail-open by default "
                "in development mode (always allowed=True). "
                "In production, PACT defaults to fail-closed but the exercise scaffolding "
                "uses the development default, which is why checks always pass."
            ),
            "learning_outcome": "Pass a compiled org policy to GovernanceEngine to activate permission enforcement",
        },
        # ── Lesson 4: Governed agents ─────────────────────────────────────
        {
            "id": "6.4.1",
            "lesson": "6.4",
            "type": "output_interpret",
            "difficulty": "advanced",
            "question": (
                "In Exercise 4, you wrap a ReActAgent with PactGovernedAgent and set "
                "operating envelope: tools=['profile_data', 'describe_column'], "
                "max_cost_usd=5.0, data_access=['credit_data']. "
                "The agent attempts to call 'visualize_feature' (not in its tools list) "
                "and access 'customer_pii_data' (not in its data_access list). "
                "What does PactGovernedAgent do and what does it log?"
            ),
            "options": [
                "A) The agent silently skips the disallowed calls and continues",
                "B) Governance only applies to cost; tool and data access violations are not intercepted",
                "C) The agent substitutes 'describe_column' for 'visualize_feature' automatically",
                "D) PactGovernedAgent intercepts both attempts before they reach the LLM or data layer. Each violation raises GovernanceViolation and is logged to AuditChain with: timestamp, agent_address, action_attempted, permission_required, result=DENIED. The agent's task halts on the first violation unless the envelope specifies continue_on_violation=True"
            ],
            "answer": "D",
            "explanation": (
                "PactGovernedAgent wraps every agent action with a permission check before execution. "
                "When an agent attempts to call a tool not in its envelope, GovernanceViolation is raised. "
                "AuditChain records every check — both allowed and denied — in a tamper-evident log. "
                "This is the key difference from a bare dict of allowed tools: "
                "the AuditChain provides an auditable record that the governance was enforced, "
                "not just declared, which regulators require."
            ),
            "learning_outcome": "Trace PactGovernedAgent's violation interception and AuditChain logging",
        },
        {
            "id": "6.4.2",
            "lesson": "6.4",
            "type": "context_apply",
            "difficulty": "advanced",
            "question": (
                "Exercise 4 demonstrates that a governed agent cannot modify its own "
                "GovernanceContext. A student asks: 'What prevents the agent from calling "
                "engine.update_envelope() to loosen its own constraints?' "
                "Explain the specific mechanism in PactGovernedAgent that enforces this."
            ),
            "options": [
                "A) The agent does not have access to the GovernanceEngine object",
                "B) Python's property decorator prevents attribute modification",
                "C) GovernanceContext is frozen (immutable) after creation — PactGovernedAgent passes a read-only view of the context to the agent. Even if the agent had access to GovernanceEngine, update_envelope() checks that the requester's address is a parent node of the target; an agent cannot be its own parent",
                "D) The engine runs in a separate process; the agent cannot call its methods"
            ],
            "answer": "C",
            "explanation": (
                "PACT's frozen context uses Python's dataclass(frozen=True) so attributes cannot be "
                "reassigned after creation. "
                "More importantly, update_envelope() validates the organisational hierarchy: "
                "only a node's parent (or higher) can modify its envelope. "
                "An agent's D/T/R address places it in the hierarchy — "
                "it cannot address itself as a parent. "
                "This structural rule is enforced by compile_org(), not just runtime checks."
            ),
            "learning_outcome": "Explain the dual mechanism (frozen context + hierarchy check) preventing agent self-escalation",
        },
        # ── Lesson 5: Reinforcement learning ─────────────────────────────
        {
            "id": "6.5.1",
            "lesson": "6.5",
            "type": "output_interpret",
            "difficulty": "advanced",
            "question": (
                "In Exercise 5, the RL agent (PPO) on the inventory management environment "
                "achieves average episode reward = 142 after 200,000 steps. "
                "A fixed reorder-point heuristic achieves 118. "
                "During evaluation, the RL policy sometimes orders 0 units when stock is at 12 "
                "(low stock), while the heuristic always reorders. "
                "Is the RL policy making a mistake, and how do you determine this from the "
                "ExperimentTracker-logged training curves?"
            ),
            "options": [
                "A) Yes — the RL policy is suboptimal; add a constraint preventing zero orders at low stock",
                "B) The heuristic should be used instead because RL behaviour is unpredictable",
                "C) The RL policy is always correct if its mean reward exceeds the heuristic",
                "D) Not necessarily — the RL policy may have learned that demand is low on certain days of the week, making a zero order rational when holding cost exceeds expected stockout cost. Check the ExperimentTracker curves for: episode length (longer = fewer stockouts), reward breakdown by component (holding vs stockout cost), and whether the zero-order states correlate with low-demand state features"
            ],
            "answer": "D",
            "explanation": (
                "An RL agent maximising total episode reward learns to balance holding costs, "
                "order costs, and stockout penalties. A zero order at low stock can be rational "
                "if the agent predicts low demand (e.g., weekend) and holding cost exceeds "
                "expected stockout cost. "
                "ExperimentTracker-logged metrics to check: per-step reward breakdown "
                "(was there a stockout cost spike after zero-order days?) and "
                "correlation of zero-order actions with the 'day_of_week' state feature. "
                "If stockouts follow zero-order days, the policy is suboptimal; "
                "if they do not, the policy is correctly anticipating low demand."
            ),
            "learning_outcome": "Interpret RL policy behaviour by correlating actions with state features in training logs",
        },
        # ── Lesson 6: Agent governance at scale ───────────────────────────
        {
            "id": "6.6.1",
            "lesson": "6.6",
            "type": "output_interpret",
            "difficulty": "advanced",
            "question": (
                "In Exercise 6, you configure a budget cascade:\n\n"
                "  Organisation: max_cost_usd = 50.00\n"
                "  Supervisor agent: max_cost_usd = 20.00\n"
                "  Worker agent A: max_cost_usd = 8.00\n"
                "  Worker agent B: max_cost_usd = 8.00\n"
                "  Worker agent C: max_cost_usd = 8.00\n\n"
                "Worker A spends $8.00 (hits its limit). Worker B and C have spent $3.00 each. "
                "Can the supervisor now reassign Worker A's remaining tasks to Worker B? "
                "Why or why not?"
            ),
            "options": [
                "A) Yes — the supervisor can transfer budget between workers freely",
                "B) No — once any worker hits its limit, the entire supervisor budget is frozen",
                "C) Yes — the supervisor has $20.00 total and can dynamically allocate to any worker",
                "D) No — each worker's envelope is independent and cannot receive budget from another worker's allocation. Worker A's $8.00 is exhausted; Worker B's individual budget is $8.00 and it has $5.00 remaining. The supervisor cannot 'transfer' the concept of A's budget to B, but it CAN reassign the tasks to B as long as B has sufficient remaining budget ($5.00) to complete them"
            ],
            "answer": "D",
            "explanation": (
                "Budget cascade means each node in the hierarchy has its own independent budget "
                "that is always <= its parent's budget. "
                "Budgets are not pooled — Worker A's exhausted $8.00 cannot be transferred to B. "
                "However, task reassignment is legitimate governance: "
                "the supervisor can give Worker B new tasks as long as B's remaining $5.00 covers them. "
                "The supervisor's own $20.00 budget is the ceiling for all workers combined — "
                "it is not distributable on-the-fly but enforces the aggregate cap."
            ),
            "learning_outcome": "Apply budget cascade rules to multi-agent task reassignment decisions",
        },
        {
            "id": "6.6.2",
            "lesson": "6.6",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 6 introduces AuditChain for tamper-evident governance logging. "
                "A compliance officer asks: 'How do we know the audit log has not been "
                "retroactively modified to remove a governance violation?' "
                "Describe the mechanism AuditChain uses and why deleting one record "
                "is detectable."
            ),
            "options": [
                "A) AuditChain uses cryptographic chaining: each record includes the hash of the previous record (H(record_n-1)). If record k is deleted, record k+1's stored H(record_k) no longer matches the recomputed hash of the now-absent record — the chain is broken at that point. Any chain break is immediately visible during verification",
                "B) AuditChain stores logs in a separate database that only admins can access",
                "C) AuditChain uses append-only storage; deletion operations are blocked at the OS level",
                "D) AuditChain replicates logs to three databases; a modification in one is overwritten by the majority"
            ],
            "answer": "A",
            "explanation": (
                "AuditChain's tamper evidence is based on the same principle as blockchain: "
                "each record contains H(previous_record) as a field. "
                "verify_chain() recomputes each hash and compares it to the stored value. "
                "Deleting record k breaks the chain: record k+1 has H(record_k) in its header, "
                "but when you try to verify, the predecessor is missing and the hashes diverge. "
                "This makes deletions, insertions, and modifications detectable without "
                "requiring append-only storage (though that can be layered on top)."
            ),
            "learning_outcome": "Explain AuditChain's cryptographic chaining mechanism for tamper detection",
        },
        # ── Lesson 7: Dereliction handling and clearance levels ───────────
        {
            "id": "6.7.1",
            "lesson": "6.7",
            "type": "process_doc",
            "difficulty": "advanced",
            "question": (
                "In Exercise 7, you configure DerelictionPolicy for a worker agent that "
                "exceeds its cost budget. The policy is SUSPEND_AND_ESCALATE. "
                "Trace the four events that occur from the moment the worker hits its budget "
                "to the moment the supervisor receives the escalation. "
                "Reference specific class names from the Exercise 7 solution."
            ),
            "options": [
                "A) Worker raises Exception → supervisor catches it → task is retried → escalation email is sent",
                "B) Worker pauses itself → governance log is written → supervisor polls for status → task is reassigned",
                "C) (1) Worker's action triggers GovernanceEngine.check_permission() which returns denied; (2) PactGovernedAgent raises GovernanceViolation; (3) DerelictionHandler.handle(violation, policy=SUSPEND_AND_ESCALATE) suspends the worker and writes to AuditChain; (4) DerelictionHandler notifies the supervisor agent via its registered escalation callback — the supervisor receives a DerelictionEvent with the worker's address and violation details",
                "D) BudgetCascade detects overage → resets the budget → logs a warning → continues execution"
            ],
            "answer": "C",
            "explanation": (
                "Step-by-step with Module 6 class names: "
                "(1) GovernanceEngine.check_permission() catches the budget overage. "
                "(2) PactGovernedAgent converts the denied permission into GovernanceViolation. "
                "(3) DerelictionHandler (configured with DerelictionPolicy.SUSPEND_AND_ESCALATE) "
                "writes the suspension record to AuditChain (tamper-evident). "
                "(4) The escalation path is the supervisor agent's callback registered via "
                "SupervisorWorkerPattern — the supervisor receives a DerelictionEvent and "
                "can apply fallback logic (reassign, abort, or alert a human)."
            ),
            "learning_outcome": "Trace the SUSPEND_AND_ESCALATE dereliction flow through PACT class hierarchy",
        },
        # ── Lesson 8: Capstone ────────────────────────────────────────────
        {
            "id": "6.8.1",
            "lesson": "6.8",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "For the Module 6 capstone, you deploy a governed multi-agent credit "
                "assessment system. A regulator audit finds that two agents accessed "
                "customer PII data that was outside their clearance level. "
                "The AuditChain log shows the accesses. What does the presence of these "
                "records in AuditChain tell the regulator, and what must you change in "
                "the GovernanceEngine configuration to prevent recurrence?"
            ),
            "options": [
                "A) The AuditChain records prove that the accesses occurred and were logged (tamper-evident) — but they also prove the governance failed to prevent the access. The fix: add a ClearanceLevel requirement to the PII data resource (minimum clearance=3) and ensure the agents' roles have clearance < 3, so GovernanceEngine.check_permission() blocks the access before it reaches the data layer",
                "B) The AuditChain records prove the agents are working correctly; no changes needed",
                "C) Delete the offending audit records and increase the clearance level retroactively",
                "D) The AuditChain records are sufficient evidence of compliance; clearance levels are optional"
            ],
            "answer": "A",
            "explanation": (
                "AuditChain provides evidence of what happened — both compliant and non-compliant events. "
                "Finding PII access records means governance was logging but not blocking. "
                "This indicates the data resource was not tagged with a minimum clearance requirement. "
                "The fix uses ClearanceLevel in the resource definition: "
                "{'resource': 'customer_pii', 'min_clearance': 3}. "
                "An agent with clearance level 1 or 2 will receive allowed=False from check_permission() "
                "before the data layer is ever reached. "
                "The AuditChain cannot be retroactively modified — that would break the hash chain."
            ),
            "learning_outcome": "Configure ClearanceLevel requirements to prevent unauthorised data access at the governance layer",
        },
        {
            "id": "6.8.2",
            "lesson": "6.8",
            "type": "process_doc",
            "difficulty": "advanced",
            "question": (
                "You are presenting the completed ASCENT capstone to the Terrene Foundation board. "
                "The system includes: SFT-tuned credit explanation model (kailash-align), "
                "governed multi-agent pipeline (kailash-pact), RL-optimised inventory agent "
                "(kailash-ml RLTrainer), and all served via Nexus. "
                "A board member asks: 'What prevents an agent from bypassing governance by "
                "calling the Nexus API directly?' "
                "Explain the architectural answer using two specific system components."
            ),
            "options": [
                "A) Agents cannot access the Nexus API because it runs on a different port",
                "B) (1) Nexus authentication middleware validates every request's session against the GovernanceEngine — a session without a valid governed context is rejected before reaching the workflow; (2) PactGovernedAgent wraps every tool call inside the governance check, so even if an agent somehow obtained a raw function reference, the call is intercepted by the GovernanceContext before execution",
                "C) Agents do not have network access in production; they can only use registered tools",
                "D) The Nexus API is internal and not reachable from agent code",
            ],
            "answer": "B",
            "explanation": (
                "Defence in depth: "
                "(1) Nexus's session middleware is the outer boundary — "
                "every inbound HTTP request must carry a valid session that was created by GovernanceEngine. "
                "Sessions are bound to a GovernanceContext; the middleware rejects contexts with expired "
                "or exceeded budgets before the workflow handler runs. "
                "(2) PactGovernedAgent is the inner boundary — it wraps the agent's execution loop, "
                "intercepting every tool call before dispatch. "
                "An agent that somehow bypasses Nexus auth still hits PactGovernedAgent's check_permission(). "
                "Two independent layers make bypass significantly harder."
            ),
            "learning_outcome": "Describe defence-in-depth governance using Nexus auth middleware and PactGovernedAgent",
        },
        # ── Additional questions covering lessons 1–8 breadth ─────────────
        {
            "id": "6.2.2",
            "lesson": "6.2",
            "type": "output_interpret",
            "difficulty": "advanced",
            "question": (
                "After DPO training in Exercise 2, you compare the base model and "
                "DPO-aligned model on 50 credit explanation prompts using a pairwise evaluator. "
                "The aligned model is preferred 78% of the time. "
                "The DPO training loss converged to 0.41. "
                "What does the 78% preference rate tell you, and what is the risk of "
                "continuing DPO training to further reduce the loss?"
            ),
            "options": [
                "A) 78% preference rate indicates meaningful alignment improvement over the base model (random would be 50%). Continuing DPO to further reduce loss risks over-alignment: the model may generate responses that score well on preference pairs but become formulaic and lose helpfulness on out-of-distribution prompts — a phenomenon called reward hacking or mode collapse on preferred templates",
                "B) 78% preference means DPO failed; target is always >90%",
                "C) DPO training loss should always reach 0.0; convergence at 0.41 is incomplete",
                "D) Preference rate and DPO loss are unrelated metrics; evaluate only by loss"
            ],
            "answer": "A",
            "explanation": (
                "78% preference is a strong signal — it means the aligned model clearly beats "
                "the base model on the preference distribution. "
                "DPO's beta parameter (KL regularisation) controls how far the aligned model "
                "drifts from the base model. Reducing beta too much or training too long "
                "causes the model to 'game' the implicit reward: it learns the surface patterns "
                "of preferred responses (formal language, specific phrases) without actually improving "
                "explanatory quality — the classic alignment tax."
            ),
            "learning_outcome": "Interpret DPO preference rate and identify over-alignment risk from excessive training",
        },
        {
            "id": "6.1.3",
            "lesson": "6.1",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 1 stores the SFT adapter in AdapterRegistry after training. "
                "A production pipeline needs to serve both the base model (for general queries) "
                "and the Singapore-domain adapter (for regulatory questions). "
                "Which AdapterRegistry pattern supports this without loading two full model copies, "
                "and what is the memory advantage?"
            ),
            "options": [
                "A) Load two separate model copies; there is no way to share the base model",
                "B) Base model sharing only works with lora_r=1; higher ranks require separate models",
                "C) AdapterRegistry can only serve one adapter at a time; use separate API endpoints",
                "D) LoRA adapters are small weight deltas — the base model weights are shared in memory. AdapterRegistry.load(base_model, adapter_name='sg_domain') applies the adapter on-the-fly by adding A×B to the relevant weight matrices. Swapping adapters at inference time requires only replacing ~1% of weights (the low-rank matrices), not reloading 7B parameters. Memory: base model (7B × 2 bytes = 14GB) + adapter (~50MB) vs two full copies (28GB)"
            ],
            "answer": "D",
            "explanation": (
                "LoRA adapters are the A and B matrices for specific layers — typically <100MB for a 7B model. "
                "The base model (frozen weights) is shared: loaded once into GPU memory. "
                "Switching between base and adapter requires adding or removing the low-rank delta "
                "from the weight matrices — a parameter-efficient operation. "
                "AdapterRegistry tracks available adapters by name and handles this swap. "
                "Memory saving: 14GB shared base + 50MB adapter vs 28GB for two full copies."
            ),
            "learning_outcome": "Explain LoRA adapter memory efficiency for multi-adapter serving via AdapterRegistry",
        },
        {
            "id": "6.3.3",
            "lesson": "6.3",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "You define a PACT organisation with a data_science department containing "
                "a modeling team. The modeling team's ml_agent role has "
                "data_access=['credit_data', 'feature_store']. "
                "An engineer wants to add 'customer_pii' to the access list for a new project. "
                "Write the Address that identifies the ml_agent role and explain "
                "why the engineer cannot add the access in their local code."
            ),
            "options": [
                "A) Address('ml_agent') — flat names identify roles; the engineer can modify permissions directly",
                "B) Address('data_science/modeling/ml_agent') — the full D/T/R path. The engineer cannot modify permissions in their code because: (1) the organisation is compiled via compile_org() which validates the hierarchy; (2) GovernanceEngine checks whether the requester's address has authority over the target; a peer engineer's address is not a parent of ml_agent in the hierarchy",
                "C) Address('modeling.ml_agent') — dot notation for nested roles",
                "D) Address must be created by the GovernanceEngine, not manually",
            ],
            "answer": "B",
            "explanation": (
                "PACT addresses follow Department/Team/Role (D/T/R) notation. "
                "data_science is the department, modeling is the team, ml_agent is the role. "
                "Full address: data_science/modeling/ml_agent. "
                "To update permissions, the requester must be the parent node "
                "(data_science/modeling team lead) or higher in the hierarchy. "
                "An engineer who is also a peer ml_agent cannot modify a sibling role's permissions — "
                "this prevents lateral privilege escalation."
            ),
            "learning_outcome": "Construct PACT D/T/R addresses and explain hierarchy-based permission update authority",
        },
        {
            "id": "6.5.2",
            "lesson": "6.5",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student's RL training loop with RLTrainer never improves beyond random policy. "
                "After 200,000 steps, average reward is identical to the first episode. "
                "What is the most likely configuration bug?"
            ),
            "code": (
                "from kailash_ml.rl.trainer import RLTrainer\n"
                "\n"
                "trainer = RLTrainer(\n"
                "    env=InventoryEnv(),\n"
                "    algorithm='ppo',\n"
                "    learning_rate=0.0003,\n"
                "    n_steps=2048,\n"
                "    gamma=0.99,\n"
                "    # Bug: reward is computed incorrectly in the environment\n"
                ")\n"
                "\n"
                "# InventoryEnv.step() always returns reward=0.0 due to a sign error:\n"
                "# reward = -(revenue - holding_cost - stockout)  # negated by mistake"
            ),
            "options": [
                "A) learning_rate=0.0003 is too high for PPO; reduce to 0.00001",
                "B) RLTrainer requires n_steps to be divisible by the episode length",
                "C) gamma=0.99 is incorrect for inventory problems; use gamma=0.95",
                "D) The environment's reward is always 0.0 due to a sign error (revenue minus costs is negative, negated becomes positive, but the holding cost subtraction makes it 0). PPO cannot learn from zero reward signal — it has no gradient to follow. Fix: reward = revenue - holding_cost - stockout_penalty (without negation)"
            ],
            "answer": "D",
            "explanation": (
                "PPO optimises the expected cumulative reward. If the reward is always 0.0, "
                "every policy has the same expected return — the gradient is zero and no learning occurs. "
                "RL bugs often hide in the reward function. "
                "The debug process: print(env.step(action)) for a few steps to verify reward is non-zero "
                "and has the correct sign. Reward should be positive for profitable operations "
                "and negative for stockouts and excess holding — a signed signal the agent can optimise."
            ),
            "learning_outcome": "Debug RL non-convergence by verifying reward signal is non-zero and correctly signed",
        },
        {
            "id": "6.4.3",
            "lesson": "6.4",
            "type": "process_doc",
            "difficulty": "advanced",
            "question": (
                "In Exercise 4, you test governance enforcement by asking the "
                "PactGovernedAgent to access 'customer_pii_data'. The test confirms "
                "GovernanceViolation is raised. A student asks: 'How do I write an automated "
                "test that verifies governance is working, not just that the code runs?' "
                "Describe the test pattern and which class provides the assertion evidence."
            ),
            "options": [
                "A) Run the agent and check that it does not crash — if no exception, governance passed",
                "B) Mock the GovernanceEngine to return allowed=False and verify the agent stops",
                "C) Use pytest.raises(GovernanceViolation) to assert the violation is raised; then query AuditChain to assert a DENIED record exists with the correct agent_address and resource. Two-part test: (1) exception assertion confirms the action was blocked; (2) AuditChain record assertion confirms it was logged — both must pass for the governance test to be valid",
                "D) Governance tests are not automatable; they require manual review of audit logs"
            ],
            "answer": "C",
            "explanation": (
                "A complete governance test has two assertions: "
                "(1) pytest.raises(GovernanceViolation) confirms the agent was stopped. "
                "This alone is insufficient — the exception might come from the wrong source. "
                "(2) AuditChain query confirms the event is permanently recorded: "
                "chain.get_events(action='data_access', result='DENIED') should return one record "
                "with agent_address='data_science/modeling/ml_agent'. "
                "Both assertions together prove the governance layer worked as designed, not just that code ran."
            ),
            "learning_outcome": "Write two-part governance tests using pytest.raises and AuditChain record assertions",
        },
        {
            "id": "6.7.2",
            "lesson": "6.7",
            "type": "output_interpret",
            "difficulty": "advanced",
            "question": (
                "Exercise 7 sets clearance levels: ml_agent has clearance=1, "
                "credit_data resource requires min_clearance=1, "
                "customer_pii requires min_clearance=3. "
                "A new project requires the ml_agent to access anonymised_pii (min_clearance=2). "
                "The GovernanceEngine returns allowed=False. "
                "Without changing the data resource's clearance requirement, "
                "how should the organisation administrator grant access?"
            ),
            "options": [
                "A) Add 'anonymised_pii' to the agent's data_access list",
                "B) The administrator must raise the ml_agent role's clearance from 1 to 2 via GovernanceEngine.update_clearance(address='data_science/modeling/ml_agent', clearance=2). The clearance update propagates through the hierarchy check: clearance(agent) >= min_clearance(resource). This is an authority delegation, not just a data permission — the administrator must have clearance >= 2 themselves to grant it",
                "C) Set min_clearance=1 on anonymised_pii to match the agent's existing clearance",
                "D) Clearance levels cannot be changed after the org is compiled",
            ],
            "answer": "B",
            "explanation": (
                "Clearance levels enforce a security principle separate from role-based permissions. "
                "Adding the resource to data_access[] is necessary but insufficient — "
                "the agent also needs clearance >= the resource's min_clearance. "
                "Raising the agent's clearance from 1 to 2 grants access to all clearance-2 resources "
                "in the agent's data_access list. "
                "The administrator must have clearance >= 2 to grant it (you cannot delegate authority you lack). "
                "Lowering the resource requirement (option C) would grant access to all clearance-1 agents, "
                "which may be too broad."
            ),
            "learning_outcome": "Update agent clearance level via GovernanceEngine.update_clearance() for graduated data access",
        },
        {
            "id": "6.6.3",
            "lesson": "6.6",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "In Exercise 6, you scale governance to 10 agents by applying the same "
                "RoleEnvelope to all agents in a team. One agent needs a higher cost budget "
                "than the other nine. Which PACT pattern handles individual exception without "
                "changing the team-wide policy?"
            ),
            "options": [
                "A) Create a separate team for the agent with the higher budget",
                "B) Apply a TaskEnvelope on top of the RoleEnvelope for that specific agent's task — TaskEnvelopes are ephemeral overrides that extend (within the parent's budget) the agent's permissions for a specific task duration. The RoleEnvelope (team policy) is unchanged; the TaskEnvelope adds a time-bounded extension for one agent",
                "C) Modify the RoleEnvelope to have the higher budget; all agents receive it",
                "D) Individual exceptions are not supported; all agents in a team must have identical envelopes",
            ],
            "answer": "B",
            "explanation": (
                "PACT's envelope hierarchy: RoleEnvelope defines the default for a role, "
                "TaskEnvelope provides a task-scoped override for a specific instance. "
                "TaskEnvelopes must be <= their parent (the role's budget is the ceiling), "
                "but they can be <= the supervisor's budget which may be higher than the role's default. "
                "The administrator creates: TaskEnvelope(agent_address=..., max_cost_usd=15.0, expires_at=...) "
                "which overrides the RoleEnvelope's 5.0 limit for this agent during the task."
            ),
            "learning_outcome": "Use TaskEnvelope for individual agent budget exceptions without modifying team RoleEnvelope",
        },
        {
            "id": "6.3.4",
            "lesson": "6.3",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student's PACT organisation YAML is valid but GovernanceEngine.check_permission() "
                "always returns allowed=True for the ml_agent even when the resource is not in "
                "data_access. What is the likely bug?"
            ),
            "code": (
                "from kailash_pact import GovernanceEngine, compile_org\n"
                "\n"
                "org = compile_org(org_dict)  # compiles correctly\n"
                "\n"
                "# Bug: engine created without the compiled org\n"
                "engine = GovernanceEngine()  # empty — no policy loaded\n"
                "\n"
                "# Alternatively: wrong mode\n"
                "engine2 = GovernanceEngine(org, mode='development')  # development mode = permissive"
            ),
            "options": [
                "A) compile_org() must be awaited; the result is a coroutine",
                "B) data_access permissions require a separate PermissionEngine, not GovernanceEngine",
                "C) Two possible bugs: (1) engine = GovernanceEngine() without passing the compiled org defaults to fail-open (development mode); (2) mode='development' explicitly sets permissive mode where all checks pass. Use GovernanceEngine(org) with no mode argument (defaults to 'production', fail-closed) for real enforcement",
                "D) check_permission() only enforces cost budgets; data_access is not checked"
            ],
            "answer": "C",
            "explanation": (
                "GovernanceEngine has two modes relevant here: "
                "(1) Empty constructor: GovernanceEngine() has no policy and defaults to permissive "
                "in the development scaffold used by the exercise — always allowed=True. "
                "(2) Explicit mode='development': also permissive, used for testing governance code "
                "without blocking access. "
                "Production enforcement requires: GovernanceEngine(compile_org(org_dict)) "
                "with default mode='production' (fail-closed). "
                "This is the most common setup error in the Module 6 exercises."
            ),
            "learning_outcome": "Distinguish GovernanceEngine production vs development mode and pass compiled org",
        },
        {
            "id": "6.2.3",
            "lesson": "6.2",
            "type": "context_apply",
            "difficulty": "advanced",
            "question": (
                "You have 8,000 (chosen, rejected) preference pairs where 'chosen' responses "
                "are concise and 'rejected' responses are verbose. After DPO training, "
                "the model produces very short responses that sometimes omit required information. "
                "The beta parameter was set to 0.05 (very low). "
                "Explain the trade-off and what beta value you would try next."
            ),
            "options": [
                "A) Reduce beta further to make responses even shorter",
                "B) beta=0.05 allows large divergence from the base model — the aligned model aggressively pursues conciseness (the preference signal) at the cost of completeness. Try beta=0.2 or 0.5 to increase KL regularisation, which pulls the aligned model closer to the base model's natural verbosity. Higher beta = more conservative alignment that preserves base model capabilities",
                "C) beta controls the learning rate; lower beta means slower training",
                "D) The preference data is biased; collect more verbose chosen examples",
            ],
            "answer": "B",
            "explanation": (
                "DPO's beta parameter sets the strength of the KL divergence penalty "
                "between the aligned model and the base model. "
                "Low beta (0.05) = weak regularisation = model can drift far from the base model "
                "to maximise the preference signal. "
                "High beta = strong regularisation = model stays close to the base distribution. "
                "With beta=0.05, the model overfits the preference signal (conciseness) "
                "and loses other base model qualities (completeness, accuracy). "
                "beta=0.2 is a standard starting point that balances alignment and preservation."
            ),
            "learning_outcome": "Tune DPO beta to balance alignment strength vs base model capability preservation",
        },
        {
            "id": "6.8.3",
            "lesson": "6.8",
            "type": "process_doc",
            "difficulty": "advanced",
            "question": (
                "For the capstone, you need to demonstrate 'full model lineage' to a regulator: "
                "they want to know exactly what data trained the credit model, "
                "what governance policy governed the ML agents that selected features, "
                "and which adapter was applied for domain fine-tuning. "
                "Name the three Kailash components that provide each piece of lineage, "
                "and where each lineage record is stored."
            ),
            "options": [
                "A) All lineage is in the ModelRegistry; check the model metadata field",
                "B) (1) Training data lineage: FeatureStore schema version record — links features to the dataset version and computation timestamp; (2) Agent governance lineage: AuditChain — tamper-evident log of every governance check, tool call, and permission decision made during feature selection; (3) Adapter lineage: AdapterRegistry — records which base model, LoRA config, and training dataset produced each adapter version",
                "C) ExperimentTracker contains all lineage; the other components are operational only",
                "D) Lineage must be manually documented in a spreadsheet; no Kailash component captures it automatically",
            ],
            "answer": "B",
            "explanation": (
                "Complete ML lineage requires three distinct records: "
                "(1) FeatureStore.retrieve(schema, version, as_of) gives the exact feature values "
                "and schema that went into training — data lineage. "
                "(2) AuditChain.verify_chain() provides a cryptographically verified log of "
                "every governance decision during the ML pipeline run — process lineage. "
                "(3) AdapterRegistry.get(adapter_name, version) provides the fine-tuning config, "
                "base model reference, and training dataset hash — model lineage. "
                "Together these three satisfy the regulatory requirement for full provenance."
            ),
            "learning_outcome": "Map full ML lineage to FeatureStore, AuditChain, and AdapterRegistry as the three source components",
        },
        {
            "id": "6.5.3",
            "lesson": "6.5",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 5 compares PPO and SAC on the inventory management environment. "
                "PPO achieves average reward 142 in 200k steps. "
                "SAC achieves 158 in 200k steps but requires 3× more memory. "
                "The production system runs on a 4GB RAM device. "
                "Which algorithm should you deploy, and what trade-off does this illustrate "
                "about RL algorithm selection in resource-constrained environments?"
            ),
            "options": [
                "A) Deploy PPO — its lower memory footprint fits the 4GB constraint. SAC's 3× memory (actor + two critics + two target networks) likely exceeds 4GB for this environment. A 16-point reward difference (11%) is meaningful but not if the system cannot run SAC at all. This illustrates that RL algorithm selection must consider inference memory, not just training performance",
                "B) Always deploy SAC — higher reward justifies the memory cost",
                "C) Neither — use the heuristic baseline which uses zero memory",
                "D) Compress SAC with quantisation to fit 4GB"
            ],
            "answer": "A",
            "explanation": (
                "SAC (Soft Actor-Critic) maintains more neural networks than PPO: "
                "a policy network, two Q-networks, and two target Q-networks for stability. "
                "This 5-network structure is why SAC typically uses 3× more memory. "
                "On a 4GB embedded device (e.g., edge inventory controller), "
                "SAC may not fit alongside the operating system and other processes. "
                "PPO's single policy network (actor-critic shared backbone) is memory-efficient. "
                "The 11% reward gap is a worthwhile trade-off when the alternative is a non-functional system."
            ),
            "learning_outcome": "Select RL algorithm based on inference memory constraints alongside performance metrics",
        },
    ],
}

if __name__ == "__main__":
    for q in QUIZ["questions"]:
        print(f"\n{'=' * 60}")
        print(f"[{q['id']}] ({q['type']}) — Lesson {q['lesson']}  [{q['difficulty']}]")
        print(f"{'=' * 60}")
        print(q["question"])
        if q.get("code"):
            print(f"\n```python\n{q['code']}\n```")
        for opt in q["options"]:
            print(f"  {opt}")
        print(f"\nAnswer: {q['answer']}")
