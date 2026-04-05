---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 5.7: Multi-Agent Orchestration

### Module 5: LLMs and Agents

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Implement Supervisor-Worker, Sequential, Parallel, and Handoff patterns
- Choose the right orchestration pattern for the task
- Coordinate multiple agents for complex analysis
- Handle failures and retries in multi-agent systems

---

## Recap: Lesson 5.6

- Six specialised ML agents cover the full lifecycle
- Each agent wraps Kailash ML engines as callable tools
- Structured handoffs pass context between pipeline stages
- Agent pipelines are reproducible and explainable

---

## Why Multi-Agent?

```
Single agent:
  One LLM call handles everything
  → Overwhelmed by complex tasks
  → Context window limits
  → No specialisation

Multi-agent:
  Each agent handles its domain
  → Specialised knowledge per agent
  → Parallel execution where possible
  → Graceful failure isolation
```

---

## Four Orchestration Patterns

```
1. Sequential:        A → B → C → D
   Each agent completes before the next starts

2. Parallel:          A ─┬→ B
                         ├→ C    → Aggregate
                         └→ D

3. Supervisor-Worker:  Supervisor
                       ├→ Worker A
                       ├→ Worker B
                       └→ Worker C

4. Handoff:           A → decides → B or C or D
   Agent routes to the right specialist
```

---

## Pattern 1: Sequential

```python
from kailash_kaizen import Sequential

pipeline = Sequential(
    agents=[
        profiler_agent,
        feature_agent,
        model_agent,
        evaluation_agent,
    ],
)

# Each agent's output becomes the next agent's input
result = pipeline.execute(inputs={
    "question": "Build a price prediction model for HDB data"
})

# Trace shows all steps
for step in result.trace:
    print(f"Agent: {step.agent_name}")
    print(f"  Output: {step.output[:100]}...")
```

---

## Pattern 2: Parallel

```python
from kailash_kaizen import Parallel

# Compare three approaches simultaneously
parallel = Parallel(
    agents={
        "xgboost": xgboost_agent,
        "lightgbm": lightgbm_agent,
        "ensemble": ensemble_agent,
    },
)

results = parallel.execute(inputs={
    "question": "Train a price prediction model on HDB data"
})

# All three ran concurrently
for name, result in results.items():
    print(f"{name}: RMSE={result.metrics['rmse']:,.0f}")
```

---

## Pattern 3: Supervisor-Worker

```python
from kailash_kaizen import SupervisorWorker

orchestrator = SupervisorWorker()
orchestrator.configure(
    supervisor_model="claude-sonnet",
    workers={
        "data_analyst": profiler_agent,
        "feature_engineer": feature_agent,
        "model_trainer": model_agent,
        "evaluator": evaluation_agent,
    },
)

result = orchestrator.execute(inputs={
    "task": "Build and evaluate an HDB price prediction model. "
            "Profile the data first, then engineer features, "
            "train multiple models, and evaluate the best one."
})
```

---

## How Supervisor-Worker Works

```
Supervisor (LLM):
  "I need to accomplish this task. Let me break it down."

  Step 1: "data_analyst, profile the HDB dataset"
           → Worker executes, returns findings

  Step 2: "feature_engineer, based on the profile,
           engineer features with these strategies..."
           → Worker executes, returns features

  Step 3: "model_trainer, train lightgbm and xgboost
           on the engineered features"
           → Worker executes, returns metrics

  Step 4: "evaluator, evaluate the best model with SHAP"
           → Worker executes, returns explanations

  Supervisor synthesises: "Here is the complete analysis..."
```

---

## Pattern 4: Handoff

```python
from kailash_kaizen import Handoff

router = Handoff()
router.configure(
    router_model="claude-sonnet",
    agents={
        "pricing": pricing_agent,
        "market_trends": trends_agent,
        "policy_expert": policy_agent,
        "investment": investment_agent,
    },
    routing_prompt="""
    Route the question to the most appropriate specialist:
    - pricing: questions about specific flat valuations
    - market_trends: questions about market direction
    - policy_expert: questions about regulations and policies
    - investment: questions about investment strategy
    """,
)

result = router.execute(inputs={
    "question": "How will the new cooling measures affect prices?"
})
# Routes to: policy_expert
```

---

## Choosing the Right Pattern

```
Is the task decomposable into independent parts?
├─ YES → Parallel (fastest)
│
└─ NO → Are steps dependent?
   ├─ YES, fixed order → Sequential (simplest)
   │
   └─ YES, dynamic order → Does it need routing?
      ├─ YES → Handoff (specialist routing)
      └─ NO → Supervisor-Worker (flexible coordination)
```

---

## Pattern Comparison

| Pattern           | Speed  | Flexibility | Complexity | Best For           |
| ----------------- | ------ | ----------- | ---------- | ------------------ |
| Sequential        | Slow   | Low         | Low        | Fixed pipelines    |
| Parallel          | Fast   | Low         | Medium     | Independent tasks  |
| Supervisor-Worker | Medium | High        | High       | Complex multi-step |
| Handoff           | Fast   | Medium      | Medium     | Specialist routing |

---

## Combining Patterns

```python
# Real systems combine patterns

# Level 1: Handoff routes to the right pipeline
router = Handoff(agents={
    "quick_answer": delegate,
    "deep_analysis": analysis_pipeline,
})

# Level 2: analysis_pipeline is Sequential
analysis_pipeline = Sequential(agents=[
    profiler_agent,
    # Level 3: model comparison is Parallel
    Parallel(agents={
        "model_a": xgboost_agent,
        "model_b": lightgbm_agent,
    }),
    evaluation_agent,
])
```

---

## Error Handling in Multi-Agent Systems

```python
orchestrator = SupervisorWorker()
orchestrator.configure(
    supervisor_model="claude-sonnet",
    workers={...},

    # Error handling
    max_retries=2,
    timeout_per_worker=60,      # seconds
    fallback_strategy="skip",    # skip failed workers, continue

    # Failure modes
    on_worker_failure="retry",   # or "skip" or "abort"
    on_timeout="skip",
)
```

---

## Monitoring Multi-Agent Execution

```python
result = orchestrator.execute(inputs={...})

# Execution summary
print(f"Total time: {result.total_duration:.1f}s")
print(f"Agents used: {len(result.agent_traces)}")
print(f"Status: {result.status}")

# Per-agent details
for agent_name, trace in result.agent_traces.items():
    print(f"\n{agent_name}:")
    print(f"  Duration: {trace.duration:.1f}s")
    print(f"  Status: {trace.status}")
    print(f"  Tool calls: {len(trace.tool_calls)}")
    print(f"  Tokens used: {trace.tokens_used}")
```

---

## Exercise Preview

**Exercise 5.7: Multi-Agent HDB Analysis System**

You will:

1. Build Sequential and Parallel pipelines for model comparison
2. Implement a Supervisor-Worker system for end-to-end ML
3. Create a Handoff router for different question types
4. Combine patterns into a nested orchestration system

Scaffolding level: **Light (~30% code provided)**

---

## Common Pitfalls

| Mistake                                  | Fix                                            |
| ---------------------------------------- | ---------------------------------------------- |
| Using Supervisor-Worker for simple tasks | Sequential is simpler and sufficient           |
| No timeout on worker agents              | Always set timeouts to prevent hangs           |
| Parallel agents sharing state            | Each parallel agent must be independent        |
| Handoff with vague routing descriptions  | Be specific about what each specialist handles |
| Not logging inter-agent communication    | Save all traces for debugging                  |

---

## Summary

- Sequential: fixed-order pipeline, simple and predictable
- Parallel: independent tasks run concurrently for speed
- Supervisor-Worker: flexible, dynamic task decomposition
- Handoff: routes to specialist agents based on question type
- Combine patterns for complex real-world systems

---

## Next Lesson

**Lesson 5.8: Production Deployment**

We will learn:

- Deploying agent systems with Kailash Nexus
- Authentication, middleware, and rate limiting
- Multi-channel deployment: API + CLI + MCP simultaneously
