# Module 6: Alignment, Governance, RL & Production Deployment

**Duration**: 7 hours  
**Kailash**: kailash-align, kailash-pact, kailash-ml (RLTrainer), kailash-nexus (full platform)  
**Prerequisites**: Modules 1-5 (full Kailash platform minus Align/PACT)  
**Scaffolding**: 20% code provided (imports and comments only)  
**API Keys Required**: OPENAI_API_KEY, HF_TOKEN, DEFAULT_LLM_MODEL

## Lecture Topics

### 6A: LLM Fine-Tuning & Alignment (90 min)
- Fine-tuning landscape: full fine-tuning, **LoRA derivation** (MUST: show low-rank approximation W = W₀ + BA where B∈ℝ^{d×r}, A∈ℝ^{r×k}, r << min(d,k) — why it works: weight updates during fine-tuning are low-rank), QLoRA (**NF4 quantization** + double quantization), DoRA, prefix tuning, adapter layers
- Alignment methods: RLHF full pipeline (reward model training → PPO optimization loop), **DPO full derivation** (MUST: start from Bradley-Terry preference model, derive closed-form solution that eliminates reward model — show π*(y|x) ∝ π_ref(y|x)·exp(β⁻¹r*(x,y))), **GRPO** (MUST: Group Relative Policy Optimization — DeepSeek-R1's training paradigm, replaces PPO's value function with group-relative advantage), **ORPO** (odds ratio preference optimization — single-stage alignment without reference model)
- **Model merging** as alternative to fine-tuning: TIES, DARE, model soups — when merging beats training
- Data for alignment: instruction datasets (quality > quantity), preference pair construction, synthetic data generation
- Evaluation: perplexity, BLEU, ROUGE, BERTScore, **LLM-as-judge** (MUST: explain position bias, verbosity bias, self-enhancement bias), **Chatbot Arena Elo** rating, contamination detection
- Practical trade-offs: cost, quality, catastrophic forgetting, quantization impact on fine-tuned models

### 6B: AI Governance & Responsible Deployment (60 min)
- Regulatory landscape: **EU AI Act** (Art. 6 high-risk classification, Art. 9 risk management, Art. 13 transparency, Art. 52 disclosure — GPAI rules effective Aug 2025), Singapore AI Verify (ISAGO 2.0, testing toolkit), MAS AI guidelines for finance, **US NIST AI RMF**, US executive orders (Oct 2023, Jan 2025)
- PACT framework: D/T/R accountability grammar, operating envelopes, knowledge clearance (5 levels), verification gradient, **AuditChain** (tamper-evident), **CostTracker**, **EnforcementMode**
- **TrustPlane overview** (1 slide): TrustPosture, ConfidentialityLevel, CapabilityAttestation — the cryptographic foundation beneath PACT. Capstone extra credit for TrustPlane integration.
- Bias & fairness: demographic parity, equalized odds, calibration across groups, **FairLearn / Aequitas** toolkits, mitigation strategies (pre/in/post-processing)
- Algorithmic auditing: model cards (filled example), datasheets for datasets, **red teaming methodology**, disparate impact testing, **algorithmic impact assessment** (Canada model)
- Governance as competitive advantage: human-in-the-loop, appeal mechanisms, incident response

### 6C: Reinforcement Learning (60 min)
- RL foundations: MDPs, **Bellman expectation AND optimality equations** (both derived — MUST), value iteration, policy iteration, temporal difference learning
- Deep RL: DQN (experience replay, target networks), policy gradient (REINFORCE), actor-critic (A2C), **PPO clipped objective derivation** (MUST: show L^{CLIP}(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)] — why clipping prevents catastrophic policy updates), SAC (maximum entropy RL, temperature parameter)
- **RLHF as RL application** (MUST: connects 6A to 6C — the reward model IS the environment, the policy IS the LLM, PPO updates the LLM to maximize human preference reward)
- Practical RL: dynamic pricing, recommendation systems (exploration-exploitation), inventory optimization, **real case studies** (AlphaFold, chip design)
- Limitations: sample efficiency, reward hacking (concrete examples), safety constraints

## Key Import Paths

```python
# Alignment
from kailash_align import AlignmentConfig, AlignmentPipeline, AdapterRegistry

# Governance
from pact import GovernanceEngine, GovernanceContext, PactGovernedAgent
from pact import Address, RoleEnvelope, TaskEnvelope
from pact import compile_org, load_org_yaml

# RL
from kailash_ml.rl.trainer import RLTrainer

# Full deployment
from nexus import Nexus
```

## Lab Exercises (6)

### 6.1: SFT Fine-Tuning
- Configure `AlignmentConfig` with base model + dataset
- Run `AlignmentPipeline` for supervised fine-tuning on a small model
- Track adapter in `AdapterRegistry`
- Dataset: Singapore domain Q&A pairs

### 6.2: DPO / QLoRA Alignment
- Compare DPO (preference optimization) vs QLoRA (quantized fine-tuning)
- Run both methods on the same base model
- Evaluate quality with LLM-as-judge and human evaluation rubric

### 6.3: Governance Setup
- Define a realistic organization in YAML (3 departments, 8 roles, 15 agents)
- `compile_org()` to validate the structure
- Create `GovernanceEngine` from the compiled org
- Verify access decisions: `can_access()`, `explain_access()`

### 6.4: Governed Agents
- Wrap Module 5's ReActAgent with `PactGovernedAgent`
- Define operating envelopes (cost budgets $5 max, tool restrictions, data access policies)
- Demonstrate that `GovernanceContext` is a frozen dataclass — agents receive it but cannot modify their own governance
- Test monotonic tightening: child envelopes cannot exceed parent

### 6.5: Reinforcement Learning
- Use `RLTrainer` with PPO/SAC on an inventory management Gymnasium environment
- Compare RL policy vs heuristic baseline
- Track with ExperimentTracker from Module 2

### 6.6: Capstone — Full Platform Deployment
- Deploy a governed ML system combining:
  - Trained model (from M3) served via InferenceServer
  - Kaizen agent (from M5) with PactGovernedAgent wrapper
  - Nexus multi-channel deployment (API + CLI + MCP)
  - PACT governance enforcing cost budgets and data access policies
- This is the capstone exercise demonstrating the full Kailash platform

## Datasets

- **Domain Q&A pairs**: Synthetic instruction dataset for SFT (1000 pairs, Kailash SDK domain)
- **Preference pairs**: For DPO exercise (500 pairs, good/bad response comparisons)
- **Singapore parliamentary Hansard**: Public record for governance text analysis
- **Gymnasium environments**: CartPole, LunarLander, custom inventory management

**Data source**: `ascent_data/ascent06/` and `ascent_data/ascent06-dl/` on shared Google Drive.

## PACT Patterns (Critical)

Exercises MUST follow PACT governance rules:
- Agents receive frozen `GovernanceContext`, never `GovernanceEngine` directly
- All envelopes enforce monotonic tightening
- Access decisions are fail-closed (deny on error)
- D/T/R addresses always terminate with a Role

## Quiz Topics
- LoRA rank selection: "How does rank affect quality vs training cost?"
- DPO vs SFT: "When would you choose DPO over SFT? What data does each need?"
- GovernanceContext: "Why is GovernanceContext frozen? What attack does this prevent?"
- Monotonic tightening: "A child agent requests a $100 budget but the parent has $50. What happens?"
- Bellman equation: "Interpret this value function for the inventory environment"
- PPO advantage: "What is the advantage function and why does PPO clip it?"

## Deck Opening Case
**EU AI Act enforcement** — first fines expected 2025. Singapore AI Verify already requiring self-assessment for high-risk systems. Companies without governance infrastructure are scrambling. PACT's D/T/R grammar is how you build governance that works — not as a compliance checkbox, but as competitive advantage.
