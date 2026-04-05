---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 6.6: Governed Agents

### Module 6: Alignment and Governance

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Build agents that operate within governance constraints using `PactGovernedAgent`
- Attach `GovernanceContext` for runtime policy enforcement
- Handle escalation, blocking, and fallback within agent workflows
- Test governed agents for policy compliance

---

## Recap: Lesson 6.5

- PACT provides D/T/R accountability grammar for AI agents
- Operating envelopes bound what agents can do
- `GovernanceEngine` enforces duties, triggers, and rights
- Verification gradient matches oversight to action sensitivity

---

## From Governance Engine to Governed Agent

```
Lesson 6.5:  GovernanceEngine defines the RULES
Lesson 6.6:  PactGovernedAgent ENFORCES them at runtime

GovernanceEngine:
  "Agents must log predictions, cannot give financial advice,
   must escalate when confidence < 0.6"

PactGovernedAgent:
  Before every action: "Am I allowed to do this?"
  After every action:  "Did I fulfil my duties?"
  On trigger:          "Do I need to escalate?"
```

---

## PactGovernedAgent

```python
from kailash_kaizen import ReActAgent
from kailash_pact import PactGovernedAgent, GovernanceContext

# Wrap a regular agent with governance
base_agent = ReActAgent()
base_agent.configure(
    model="claude-sonnet",
    tools=[query_database, calculate, get_trend],
)

governed_agent = PactGovernedAgent(
    agent=base_agent,
    governance=engine,   # from Lesson 6.5
)

result = governed_agent.execute(sig, inputs={
    "question": "Should I invest in a Tampines flat at $540k?"
})
```

---

## GovernanceContext

Runtime context tracks agent state against governance policies.

```python
from kailash_pact import GovernanceContext

context = GovernanceContext(
    agent_id="advisor_01",
    session_id="sess_2026_04_06_001",
    clearance="public",
    budget_remaining=50.00,

    # Caller information
    user_role="retail_customer",
    request_source="api",
)

# Context is passed through the entire agent execution
result = governed_agent.execute(
    sig,
    inputs={"question": "..."},
    governance_context=context,
)
```

---

## Pre-Action Checks

```python
# PactGovernedAgent automatically checks BEFORE each tool call

# Agent wants to call: query_database(town="TAMPINES")
# Governance checks:
#   1. Does agent have "query_hdb_database" right?  → YES
#   2. Is budget sufficient?                         → YES (cost: $0.001)
#   3. Any matching triggers?                        → NO
#   4. Verification level?                           → 1 (automated)
# → PROCEED

# Agent wants to call: give_investment_advice(...)
# Governance checks:
#   1. Does agent have "provide_financial_advice" right?  → NO
# → BLOCKED
```

---

## Post-Action Duties

```python
# After each action, duties are checked

# Agent generated a prediction: $485,000 (confidence: 0.72)
# Post-action duty checks:
#   1. "Log all predictions" → logged to database ✓
#   2. "Include source attribution" → sources attached ✓
#   3. "Disclaim financial advice" → disclaimer added ✓

# Agent generated a prediction: $485,000 (confidence: 0.45)
# Post-action trigger checks:
#   1. "confidence < 0.6 → escalate" → TRIGGER FIRED
#   → Prediction held, human reviewer notified
```

---

## Escalation Flow

```
Agent generates response
    ↓
Trigger: confidence < 0.6
    ↓
┌─────────────────────────────────────┐
│  ESCALATION                          │
│                                      │
│  Agent: advisor_01                   │
│  Action: price prediction            │
│  Confidence: 0.45                    │
│  Proposed: $485,000                  │
│                                      │
│  Reason: Below confidence threshold  │
│                                      │
│  [Approve]  [Modify]  [Reject]       │
└─────────────────────────────────────┘
    ↓
Human approves/modifies/rejects
    ↓
Response delivered (or blocked)
```

---

## Handling Blocked Actions

```python
result = governed_agent.execute(sig, inputs={
    "question": "Give me specific financial advice on buying this flat"
})

if result.governance_status == "blocked":
    print(f"Action blocked: {result.block_reason}")
    print(f"Suggestion: {result.fallback_message}")
```

```
Action blocked: "provide_financial_advice" requires human-only clearance
Suggestion: "I can provide market data and price comparisons,
but for specific financial advice, please consult a licensed
financial advisor."
```

---

## Governed Tool Wrapper

```python
from kailash_pact import governed_tool

@governed_tool(
    engine=engine,
    right="query_hdb_database",
    verification_level=1,
    cost=0.001,
)
def query_database(town: str, flat_type: str = None) -> dict:
    """Query HDB transaction data."""
    query = df.filter(pl.col("town") == town)
    if flat_type:
        query = query.filter(pl.col("flat_type") == flat_type)
    return {"avg_price": query["price"].mean(), "count": len(query)}

# When the agent calls this tool, governance checks happen automatically
```

---

## Testing Governance Compliance

```python
from kailash_pact import GovernanceTestSuite

test_suite = GovernanceTestSuite(engine=engine)

# Test that blocked actions are actually blocked
test_suite.assert_blocked(
    agent=governed_agent,
    action="provide_financial_advice",
    context=GovernanceContext(clearance="public"),
)

# Test that duties are fulfilled
test_suite.assert_duty_fulfilled(
    agent=governed_agent,
    duty="Log all predictions",
    test_input={"question": "What is a fair price for..."},
)

# Test that triggers fire correctly
test_suite.assert_trigger_fires(
    trigger="confidence < 0.6",
    test_input={"question": "Very unusual property..."},
)

results = test_suite.run()
print(f"Passed: {results.passed}/{results.total}")
```

---

## Governance Dashboard

```python
# Runtime governance monitoring
dashboard = engine.get_dashboard(
    period="24h",
)

print(f"Total requests:  {dashboard.total_requests}")
print(f"Blocked:         {dashboard.blocked_count} ({dashboard.blocked_pct:.1%})")
print(f"Escalated:       {dashboard.escalated_count}")
print(f"Budget used:     ${dashboard.budget_used:.2f} / ${dashboard.budget_limit:.2f}")
print(f"Duties fulfilled: {dashboard.duties_met_pct:.1%}")

# Top block reasons
for reason, count in dashboard.top_block_reasons:
    print(f"  {reason}: {count}")
```

---

## Exercise Preview

**Exercise 6.6: Building a Governed HDB Advisor**

You will:

1. Wrap a ReAct agent with `PactGovernedAgent`
2. Configure governance context for different user roles
3. Test blocked actions, escalation triggers, and duty fulfilment
4. Build a governance compliance test suite

Scaffolding level: **Minimal (~20% code provided)**

---

## Common Pitfalls

| Mistake                               | Fix                                              |
| ------------------------------------- | ------------------------------------------------ |
| Governance checks only at entry point | Check before every tool call, not just the first |
| No fallback message when blocked      | Always provide a helpful alternative             |
| Testing governance in production      | Use GovernanceTestSuite before deployment        |
| Same clearance for all agents         | Assign minimum necessary clearance               |
| No governance dashboard               | Monitor blocked/escalated rates for tuning       |

---

## Summary

- `PactGovernedAgent` wraps any agent with governance enforcement
- `GovernanceContext` tracks agent state at runtime
- Pre-action checks verify rights and budget before tool calls
- Post-action duties ensure logging, attribution, and disclaimers
- Governance test suites verify compliance before deployment

---

## Next Lesson

**Lesson 6.7: Agent Governance at Scale**

We will learn:

- Clearance hierarchies for multi-agent organisations
- Budget cascading from supervisor to worker agents
- Governance patterns for enterprise-scale systems
