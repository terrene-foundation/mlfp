# Module 6: Alignment, Governance, RL & Production Deployment

**Kailash**: Align, PACT, kailash-ml (RLTrainer), Nexus | **Scaffolding**: 20%

## Lecture (3h)
- **6A** Fine-Tuning & Alignment: LoRA/QLoRA/DoRA theory, RLHF pipeline, DPO/GRPO, alignment data quality, evaluation (perplexity, BERTScore, LLM-as-judge), catastrophic forgetting trade-offs
- **6B** AI Governance: EU AI Act (risk tiers, GPAI rules), Singapore AI Verify (ISAGO 2.0), MAS guidelines, PACT D/T/R grammar, operating envelopes, bias/fairness metrics, algorithmic auditing, model cards
- **6C** RL & Advanced Topics: MDPs, Bellman equations, DQN, PPO, SAC, practical RL (dynamic pricing, recommendations), multi-modal AI, federated learning, differential privacy

## Lab (3h) — 6 Exercises
1. AlignmentPipeline SFT on small model with AdapterRegistry tracking
2. DPO alignment with LLM-as-judge evaluation
3. GovernanceEngine: define org in YAML, compile, verify access, explain decisions
4. PactGovernedAgent: wrap ReActAgent with cost budgets, tool restrictions, envelope enforcement
5. RLTrainer (PPO) on inventory management environment vs heuristic baseline
6. Capstone: full governed ML platform — InferenceServer + Kaizen agent + PACT + Nexus

## Datasets
Domain Q&A (SFT, 1000 pairs), Preference pairs (DPO, 500), Singapore Hansard, Gymnasium environments
