---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 6.5: AI Governance

### Module 6: Alignment and Governance

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain the PACT governance framework: Duties, Triggers, Rights
- Configure `GovernanceEngine` for policy enforcement
- Define operating envelopes for AI agent systems
- Design governance structures for enterprise AI deployment

---

## Recap: Lesson 6.4

- Model merging (TIES, DARE) combines specialist adapters
- Adapter stacking switches experts at runtime
- Iterative DPO progressively improves alignment
- Choose merging vs stacking vs multi-agent based on requirements

---

## Why Governance?

```
Without governance:
  Agent has access to database → deletes all records
  Agent gives financial advice → user loses money
  Agent accesses private data → privacy breach
  Agent runs expensive LLM calls → budget exceeded

With governance:
  Agent's actions are bounded by operating envelopes
  Every decision has accountability (who authorised this?)
  Budget limits prevent runaway costs
  Sensitive operations require escalation
```

---

## The PACT Framework

PACT provides an accountability grammar for AI agent organisations.

```
D/T/R — Duties, Triggers, Rights

Duties:   What MUST the agent do?
          "Always log predictions with confidence scores"

Triggers: What conditions activate governance?
          "When prediction confidence < 0.6, escalate to human"

Rights:   What is the agent ALLOWED to do?
          "May query database, may NOT modify records"
```

---

## Operating Envelopes

An operating envelope defines the **boundaries** within which an agent operates.

```
┌─────────────────────────────────────────────────┐
│                Operating Envelope                 │
│                                                   │
│  Allowed actions:                                │
│    ✅ Query HDB database                         │
│    ✅ Generate price predictions                 │
│    ✅ Send reports to authorised users            │
│                                                   │
│  Forbidden actions:                              │
│    ❌ Modify database records                    │
│    ❌ Share personal information                 │
│    ❌ Give definitive financial advice            │
│                                                   │
│  Budget: 1000 LLM calls per day                 │
│  Escalation: confidence < 0.6 → human review    │
└─────────────────────────────────────────────────┘
```

---

## GovernanceEngine

```python
from kailash_pact import GovernanceEngine

engine = GovernanceEngine()
engine.configure(
    name="hdb_advisor_governance",

    duties=[
        "Log all predictions with confidence scores",
        "Include data source attribution in every response",
        "Disclaim that this is not financial advice",
    ],

    triggers=[
        {"condition": "confidence < 0.6", "action": "escalate_to_human"},
        {"condition": "query_cost > 100", "action": "require_approval"},
        {"condition": "personal_data_detected", "action": "block_and_alert"},
    ],

    rights=[
        "query_hdb_database",
        "generate_predictions",
        "send_reports",
    ],
)
```

---

## D/T/R in Practice

```python
# Duties: automatically enforced
@engine.duty("Log all predictions")
def log_prediction(prediction, confidence, inputs):
    db.save(PredictionLog(
        prediction=prediction,
        confidence=confidence,
        inputs=inputs,
        timestamp=datetime.now(),
    ))

# Triggers: condition-based actions
@engine.trigger(condition="confidence < 0.6")
def escalate(prediction, confidence):
    notify_human_reviewer(
        message=f"Low confidence prediction ({confidence:.2f}): "
                f"${prediction:,} — requires human review",
    )
    return {"status": "escalated", "reviewed": False}

# Rights: access control
@engine.right("query_hdb_database")
def check_db_access(agent_id, query):
    return agent_id in authorised_agents
```

---

## Verification Gradient

Not all decisions need the same level of oversight.

```
Level 1 — Automated (no human):
  Price lookups, trend queries, basic statistics
  → Agent operates freely within envelope

Level 2 — Logged (human can audit):
  Price predictions, market comparisons
  → Agent operates, actions are logged for review

Level 3 — Confirmed (human approves):
  Investment recommendations, policy interpretations
  → Agent proposes, human confirms before delivery

Level 4 — Human-only:
  Financial advice, legal statements
  → Agent cannot perform; routes to human expert
```

---

## Configuring Verification Levels

```python
engine.configure(
    verification_gradient={
        "query_database": 1,           # automated
        "predict_price": 2,            # logged
        "recommend_investment": 3,     # confirmed
        "provide_financial_advice": 4, # human-only
    },
)

# Agent checks before acting
@engine.governed
def handle_request(request):
    level = engine.get_verification_level(request.action)

    if level == 4:
        return {"status": "blocked",
                "message": "This requires a human advisor"}
    elif level == 3:
        return {"status": "pending_confirmation",
                "proposed_response": generate_response(request)}
    else:
        return execute_and_log(request)
```

---

## Knowledge Clearance

Control what information agents can access based on sensitivity.

```python
engine.configure(
    knowledge_clearance={
        "public": ["town_statistics", "price_trends", "policy_docs"],
        "internal": ["model_metrics", "training_data_stats"],
        "confidential": ["individual_transactions", "personal_details"],
        "restricted": ["financial_records", "identity_data"],
    },

    agent_clearance={
        "public_advisor": "public",
        "internal_analyst": "internal",
        "admin_agent": "confidential",
    },
)
```

---

## Budget Controls

```python
from kailash_pact import GovernanceEngine, BudgetPolicy

engine.configure(
    budget=BudgetPolicy(
        daily_llm_calls=1000,
        daily_cost_limit=50.00,      # USD
        per_request_limit=5.00,

        # Cascading: supervisor can allocate to workers
        allow_budget_delegation=True,
        delegation_max_percent=0.25,  # each worker gets max 25%
    ),
)

# Check budget before expensive operations
@engine.governed
def expensive_analysis(query):
    cost_estimate = estimate_cost(query)
    if not engine.check_budget(cost_estimate):
        return {"status": "budget_exceeded",
                "remaining": engine.remaining_budget()}
```

---

## Governance Audit Trail

```python
# Every governed action is logged
audit = engine.get_audit_trail(
    start_date="2026-04-01",
    end_date="2026-04-06",
)

for entry in audit:
    print(f"{entry.timestamp} | {entry.agent_id} | "
          f"{entry.action} | {entry.level} | {entry.outcome}")
```

```
2026-04-06 10:30 | advisor_01 | predict_price | LOGGED   | $485,000
2026-04-06 10:31 | advisor_01 | recommend_buy | ESCALATED| pending
2026-04-06 10:32 | advisor_01 | query_personal| BLOCKED  | access_denied
```

---

## Exercise Preview

**Exercise 6.5: Governed HDB Advisory System**

You will:

1. Define D/T/R governance for an HDB advisory agent
2. Configure verification gradient and knowledge clearance
3. Implement budget controls and escalation triggers
4. Review the audit trail for compliance

Scaffolding level: **Minimal (~20% code provided)**

---

## Common Pitfalls

| Mistake                                     | Fix                                           |
| ------------------------------------------- | --------------------------------------------- |
| No governance on agent systems              | Every production agent needs an envelope      |
| Verification level too high for all actions | Use gradient; only escalate sensitive actions |
| No budget limits on LLM calls               | Set daily and per-request cost limits         |
| Governance without audit trail              | Always log governed decisions for compliance  |
| Static governance (never updated)           | Review and update D/T/R quarterly             |

---

## Summary

- PACT provides D/T/R accountability grammar for AI agents
- Operating envelopes bound what agents can and cannot do
- `GovernanceEngine` enforces duties, triggers, and rights
- Verification gradient matches oversight level to action sensitivity
- Budget controls and audit trails ensure responsible deployment

---

## Next Lesson

**Lesson 6.6: Governed Agents**

We will learn:

- `PactGovernedAgent` for governance-aware agent execution
- `GovernanceContext` for runtime policy enforcement
- Building agents that respect operating envelopes
