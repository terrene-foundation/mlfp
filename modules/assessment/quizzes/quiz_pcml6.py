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
                "B) per_device_train_batch_size=32 at max_seq_length=2048 fills 16GB VRAM. Reduce to per_device_train_batch_size=4 and set gradient_checkpointing=True — this trades compute for memory, enabling training on 16GB hardware at the cost of ~30% slower training",
                "C) AlignmentConfig does not support max_seq_length; remove the parameter",
                "D) Use method='sft_lora' instead of method='sft' to enable memory-efficient training",
            ],
            "answer": "B",
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
                "B) Full fine-tuning updates all 4096×4096 = 16.7M parameters in q_proj. LoRA decomposes the update as two matrices: A (4096×16) and B (16×4096) = 2 × 65,536 = 131,072 parameters — 128× fewer trainable parameters for this layer. The base model weights are frozen, so only the adapter is stored in optimizer memory",
                "C) LoRA reduces parameters by 16× because lora_r=16 divides the original dimension",
                "D) LoRA only trains 2 layers (q_proj and v_proj); full fine-tuning trains all layers",
            ],
            "answer": "B",
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
                "B) DPO is preferable: it optimises the preference objective directly from the (chosen, rejected) pairs without training a separate reward model or running PPO's complex actor-critic loop. For a team without RL expertise, DPO reduces the pipeline from 3 stages (SFT → reward model → PPO) to 2 stages (SFT → DPO), with fewer hyperparameters and no KL controller to tune",
                "C) SFT alone is sufficient; preference data is only needed for RLHF",
                "D) DPO requires at least 100,000 preference pairs; 8,000 is insufficient",
            ],
            "answer": "B",
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
                "A) The agent should retry after 1 minute; budgets reset automatically",
                "B) The current task must halt immediately. PACT's monotonic tightening principle states that an operating envelope can only be tightened (never loosened) by the agent itself — the agent cannot raise its own budget. A human operator at a higher organisational level must authorise a budget increase via GovernanceEngine.update_envelope()",
                "C) The agent can continue because $0.20 overage is within the 5% tolerance band",
                "D) The agent should switch to a cheaper model to continue under budget",
            ],
            "answer": "B",
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
                "A) GovernanceEngine requires an async context manager",
                "B) compile_org(org_dict) is never called and the result never passed to GovernanceEngine — an empty GovernanceEngine has no policy rules and defaults to allowing all actions; pass the compiled org: engine = GovernanceEngine(compile_org(org_dict))",
                "C) org_dict must be a YAML file path, not a Python dict",
                "D) check_permission() requires await because GovernanceEngine is async",
            ],
            "answer": "B",
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
                "B) PactGovernedAgent intercepts both attempts before they reach the LLM or data layer. Each violation raises GovernanceViolation and is logged to AuditChain with: timestamp, agent_address, action_attempted, permission_required, result=DENIED. The agent's task halts on the first violation unless the envelope specifies continue_on_violation=True",
                "C) The agent substitutes 'describe_column' for 'visualize_feature' automatically",
                "D) Governance only applies to cost; tool and data access violations are not intercepted",
            ],
            "answer": "B",
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
                "B) GovernanceContext is frozen (immutable) after creation — PactGovernedAgent passes a read-only view of the context to the agent. Even if the agent had access to GovernanceEngine, update_envelope() checks that the requester's address is a parent node of the target; an agent cannot be its own parent",
                "C) Python's property decorator prevents attribute modification",
                "D) The engine runs in a separate process; the agent cannot call its methods",
            ],
            "answer": "B",
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
                "B) Not necessarily — the RL policy may have learned that demand is low on certain days of the week, making a zero order rational when holding cost exceeds expected stockout cost. Check the ExperimentTracker curves for: episode length (longer = fewer stockouts), reward breakdown by component (holding vs stockout cost), and whether the zero-order states correlate with low-demand state features",
                "C) The RL policy is always correct if its mean reward exceeds the heuristic",
                "D) The heuristic should be used instead because RL behaviour is unpredictable",
            ],
            "answer": "B",
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
                "B) No — each worker's envelope is independent and cannot receive budget from another worker's allocation. Worker A's $8.00 is exhausted; Worker B's individual budget is $8.00 and it has $5.00 remaining. The supervisor cannot 'transfer' the concept of A's budget to B, but it CAN reassign the tasks to B as long as B has sufficient remaining budget ($5.00) to complete them",
                "C) Yes — the supervisor has $20.00 total and can dynamically allocate to any worker",
                "D) No — once any worker hits its limit, the entire supervisor budget is frozen",
            ],
            "answer": "B",
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
                "A) AuditChain stores logs in a separate database that only admins can access",
                "B) AuditChain uses cryptographic chaining: each record includes the hash of the previous record (H(record_n-1)). If record k is deleted, record k+1's stored H(record_k) no longer matches the recomputed hash of the now-absent record — the chain is broken at that point. Any chain break is immediately visible during verification",
                "C) AuditChain uses append-only storage; deletion operations are blocked at the OS level",
                "D) AuditChain replicates logs to three databases; a modification in one is overwritten by the majority",
            ],
            "answer": "B",
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
                "B) (1) Worker's action triggers GovernanceEngine.check_permission() which returns denied; (2) PactGovernedAgent raises GovernanceViolation; (3) DerelictionHandler.handle(violation, policy=SUSPEND_AND_ESCALATE) suspends the worker and writes to AuditChain; (4) DerelictionHandler notifies the supervisor agent via its registered escalation callback — the supervisor receives a DerelictionEvent with the worker's address and violation details",
                "C) Worker pauses itself → governance log is written → supervisor polls for status → task is reassigned",
                "D) BudgetCascade detects overage → resets the budget → logs a warning → continues execution",
            ],
            "answer": "B",
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
                "A) The AuditChain records prove the agents are working correctly; no changes needed",
                "B) The AuditChain records prove that the accesses occurred and were logged (tamper-evident) — but they also prove the governance failed to prevent the access. The fix: add a ClearanceLevel requirement to the PII data resource (minimum clearance=3) and ensure the agents' roles have clearance < 3, so GovernanceEngine.check_permission() blocks the access before it reaches the data layer",
                "C) Delete the offending audit records and increase the clearance level retroactively",
                "D) The AuditChain records are sufficient evidence of compliance; clearance levels are optional",
            ],
            "answer": "B",
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
