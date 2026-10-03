# mlfp06 — handoffs from the exercise shard (S1, merged b9fe7735)

## Content corrections the teaching material must follow (deck, textbook, lessons, notes, specs/module-6.md)
- X1 clearance order = pact's: public < restricted < confidential < secret < top_secret ("restricted" is NOT the top). Department heads are `secret` in the course org YAMLs. Escalation demos ask for SECRET/TOP_SECRET.
- X3 / D1: the installed GovernanceEngine (pact 0.14.1) AUTO-APPROVES envelope-less roles and addresses not in the org. Every deny-path attaches an envelope first (`engine.set_role_envelope(...)`). `compile_governance()` (shared/mlfp06/ex_7) now applies YAML clearances and envelopes. Spec 6.7 "fail-closed" + "D99… returns allowed == False" must be corrected.
- X10: D/T/R = Department / Team / Role (never Delegator/Task/Responsible, never Decides/Trusts/Responds).
- X7: use pact's real verdict levels; `validate_tightening` call shape as in ex_7.
- X2: DPO β — LARGE β keeps the policy close to the reference.
- X11: GRPO advantage = (r − mean_group) / std_group; clipped objective + KL term; cite DeepSeekMath (Shao et al., 2024).
- X5/F3: agent budget = `budget_limit_usd` at construction; it does nothing on free local Ollama.
- X6: build agents with `shared.mlfp06._ollama_bootstrap.make_delegate()` / `make_embedder()` with NO `model=`; model comes from OLLAMA_CHAT_MODEL. Delete every `os.environ.get("LLM_MODEL", "gpt-4o-mini")`.
- F4: Nexus deployment serves the governed handler with signed JWTs (role from token), rate limit, CORS; API channel tested incl. 401s; CLI/MCP registered but not exercised in-process.
- E5: INT8 vs FP16 = 2× memory, not 6×.  E16: UltraFeedback ratings are GPT-4-generated, not human.
- Named organisations anonymised; figures illustrative.

## assessment/task_4
- Escalation attempt must request SECRET or TOP_SECRET (not RESTRICTED); fix its comment (X1). Grader currently passes 10/10 on the solution after the shared governance change.

## Integration (S7)
- Regenerate ex_1–ex_8 notebooks; generator must handle ex_6.4 (launches its own file as the MCP server).
- Runs need network (RAG corpus, UltraFeedback) and Ollama. Order: ex_2.6 and ex_3.3 save adapters that ex_8.1 reads.
- Need Ollama: ex_1/01–06, ex_3/01,02,04, ex_4/01–05, ex_5/01–04, ex_6/01–05, ex_7/04, ex_8/03–05.
- Need training: ex_2/01–06, ex_3/03, ex_8/01. Already clean: ex_7/01–03, ex_8/02.

## Deferred spec gaps (S6)
- E13 LoRA + adapter training loops, merging two trained adapters, DARE.
- E15 lm-eval before/after DPO; Kaizen RAG agents; Singapore-policy corpus.
- F14 (6.5–6.6) DataExplorer/TrainingPipeline tools, HITL, ChainOfThoughtAgent, DataScientist→…→ReportWriter pipeline, handoff, A2A. (6.7–6.8) FinancialConstraintConfig budget cascade; reasoning-chain debugging.
- shared ex_5 `answer_question` still returns the ground-truth answer (fix in S6).

## Upstream notes (not course-fixable)
- kailash-nexus logs "Unknown parameter(s) for HandlerNode" on every request; CORS preflight (OPTIONS) gets 401 because JWT runs before CORS.
- @structured_tool(input_schema=…) combined with @server.tool() registers the tool with no parameters.
