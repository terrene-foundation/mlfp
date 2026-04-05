---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 6.3: Reinforcement Learning

### Module 6: Alignment and Governance

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain the Bellman equation and value functions
- Describe how PPO (Proximal Policy Optimisation) trains policies
- Use `RLTrainer` for reward-based model improvement
- Connect RL to LLM alignment (RLHF)

---

## Recap: Lesson 6.2

- DPO learns from preference pairs without a reward model
- LLM-as-judge automates preference labelling
- SFT-then-DPO pipeline produces aligned domain models
- Win rate comparisons evaluate alignment quality

---

## What Is Reinforcement Learning?

```
Supervised learning:  "Here is the right answer. Learn it."
                      Data: (input, correct_output) pairs

Reinforcement learning: "Try things. I will tell you how well you did."
                        Data: (state, action, reward) sequences

  Agent ──→ Action ──→ Environment
    ↑                      │
    └──── Reward ◄─────────┘

The agent learns a POLICY: which action to take in each state.
```

---

## RL Terminology

| Term            | Meaning               | HDB Example                               |
| --------------- | --------------------- | ----------------------------------------- |
| **State (s)**   | Current situation     | Market conditions, flat details           |
| **Action (a)**  | Decision made         | Price recommendation, buy/sell            |
| **Reward (r)**  | Feedback signal       | Accuracy of prediction, user satisfaction |
| **Policy (pi)** | Strategy (s → a)      | How to make recommendations               |
| **Value (V)**   | Expected total reward | Long-term quality of a state              |
| **Episode**     | One complete sequence | One advisory session                      |

---

## The Bellman Equation

The value of a state = immediate reward + discounted future value.

```
V(s) = R(s, a) + γ · V(s')

Where:
  V(s)  = value of current state
  R(s,a) = immediate reward for action a in state s
  γ     = discount factor (0 to 1, typically 0.99)
  V(s') = value of next state

γ close to 1: patient (values future rewards)
γ close to 0: greedy (values immediate rewards)
```

---

## Bellman Equation Visual

```
State s₀ ──action──→ State s₁ ──action──→ State s₂ ──→ ...
   │                    │                    │
   ↓                    ↓                    ↓
  r₀ = 10             r₁ = 5               r₂ = 20

V(s₀) = 10 + 0.99 × V(s₁)
V(s₁) = 5  + 0.99 × V(s₂)
V(s₂) = 20 + 0.99 × V(s₃)
...

Total return from s₀ = 10 + 0.99(5) + 0.99²(20) + ...
```

---

## Policy Gradient Methods

Instead of learning values, learn the **policy** directly.

```
Policy gradient:
  1. Follow current policy → collect trajectories
  2. Compute returns for each action
  3. Increase probability of actions with HIGH returns
  4. Decrease probability of actions with LOW returns

Intuition:
  "Do more of what worked, less of what didn't."
```

---

## PPO: Proximal Policy Optimisation

The standard algorithm for RL in LLM alignment.

```
PPO adds two key ideas:

1. Clipped objective:
   Don't change the policy too much in one step.
   Prevents catastrophic updates.

2. KL penalty:
   Stay close to the reference policy.
   Prevents forgetting useful knowledge.

L_PPO = min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)

r(θ) = π_new(a|s) / π_old(a|s)  (probability ratio)
A = advantage (how much better than expected)
ε = clipping range (typically 0.1-0.2)
```

---

## PPO Visual Intuition

```
Policy update
  |
  |      ╱  Unconstrained update
  |    ╱     (could be too large)
  |  ╱
  |╱─────── Clipped update (PPO)
  |           (safe, bounded change)
  |
  └────────────→ Step

PPO says: "Improve, but not too fast."
This prevents the policy from suddenly collapsing.
```

---

## RLTrainer: Kailash RL

```python
from kailash_ml import RLTrainer

trainer = RLTrainer()
trainer.configure(
    environment="hdb_pricing",
    algorithm="ppo",

    # PPO hyperparameters
    learning_rate=3e-4,
    gamma=0.99,              # discount factor
    clip_range=0.2,          # PPO clipping
    n_epochs=10,             # update epochs per batch
    batch_size=64,

    # Training
    total_timesteps=100_000,
    eval_frequency=10_000,
)

result = trainer.run()
print(f"Final reward: {result.final_reward:.2f}")
```

---

## Custom Reward Functions

```python
def pricing_reward(state, action, next_state):
    """Reward function for HDB pricing agent."""
    predicted_price = action["price"]
    actual_price = next_state["actual_price"]

    # Accuracy reward
    error_pct = abs(predicted_price - actual_price) / actual_price
    accuracy_reward = max(0, 1 - error_pct * 5)  # 0-1 scale

    # Safety penalty (no wildly wrong predictions)
    if error_pct > 0.3:  # >30% error
        safety_penalty = -1.0
    else:
        safety_penalty = 0.0

    return accuracy_reward + safety_penalty

trainer.configure(
    reward_function=pricing_reward,
)
```

---

## RLHF: RL from Human Feedback

```
The full RLHF pipeline for LLM alignment:

1. SFT: Fine-tune on instruction data
   └→ AlignmentPipeline(method="sft")

2. Reward Model: Train a model to predict human preferences
   └→ AlignmentPipeline(method="reward_model")

3. RL (PPO): Optimise the LLM against the reward model
   └→ AlignmentPipeline(method="rlhf")

DPO (Lesson 6.2) skips steps 2 and 3 by directly
optimising from preferences. Simpler but less flexible.
```

---

## RLHF with AlignmentPipeline

```python
from kailash_align import AlignmentPipeline

# Step 1: Train reward model
reward_pipeline = AlignmentPipeline()
reward_pipeline.configure(
    base_model="meta-llama/Llama-3-8B",
    method="reward_model",
    preference_data=preference_data,
    epochs=2,
)
reward_model = reward_pipeline.run()

# Step 2: RL training against reward model
rl_pipeline = AlignmentPipeline()
rl_pipeline.configure(
    base_model="meta-llama/Llama-3-8B",
    adapter_path="./adapters/hdb_expert_v1",
    method="rlhf",
    reward_model=reward_model,
    ppo_config={"clip_range": 0.2, "kl_coeff": 0.05},
    total_steps=1000,
)
result = rl_pipeline.run()
```

---

## RL Beyond LLMs

```
RL applies wherever you have:
  Sequential decisions + delayed rewards

Examples in ML:
  - Portfolio optimisation (buy/sell/hold)
  - Dynamic pricing (adjust prices over time)
  - Recommendation systems (maximize long-term engagement)
  - Resource allocation (schedule jobs on servers)
  - Robotics (learn motor control)

ASCENT focus: understanding RL for LLM alignment
Real-world RL engineering: specialised course topic
```

---

## Comparing Alignment Methods

| Method   | Reward Signal         | Complexity | Stability |
| -------- | --------------------- | ---------- | --------- |
| **SFT**  | Correct examples      | Low        | High      |
| **DPO**  | Preference pairs      | Medium     | High      |
| **RLHF** | Learned reward model  | High       | Lower     |
| **GRPO** | Group relative policy | Medium     | Medium    |

Start with SFT + DPO. Use RLHF only when DPO is insufficient.

---

## Exercise Preview

**Exercise 6.3: RL for Pricing Strategy**

You will:

1. Define a reward function for HDB pricing accuracy
2. Train a pricing agent with `RLTrainer` using PPO
3. Build a reward model from preference data
4. Compare DPO vs RLHF aligned models

Scaffolding level: **Minimal (~20% code provided)**

---

## Common Pitfalls

| Mistake                                         | Fix                                                    |
| ----------------------------------------------- | ------------------------------------------------------ |
| Reward hacking (agent exploits reward function) | Design rewards carefully; add safety constraints       |
| PPO clip range too large                        | Start at 0.2; decrease if training is unstable         |
| No KL penalty in RLHF                           | Model drifts from base capabilities without KL         |
| Using RLHF when DPO suffices                    | DPO is simpler and often comparable                    |
| Discount factor too low                         | Use 0.99 for most tasks; lower only for short horizons |

---

## Summary

- RL learns policies from reward signals, not labelled examples
- Bellman equation: value = immediate reward + discounted future value
- PPO clips policy updates for stable training
- RLHF trains a reward model, then optimises LLMs with PPO
- DPO is often preferred over RLHF for its simplicity and stability

---

## Next Lesson

**Lesson 6.4: Advanced Alignment**

We will learn:

- Model merging with TIES and DARE methods
- Combining multiple LoRA adapters
- Advanced alignment techniques for production systems
