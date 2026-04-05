---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 6.7: Agent Governance at Scale

### Module 6: Alignment and Governance

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Design clearance hierarchies for multi-agent organisations
- Implement budget cascading from supervisor to worker agents
- Apply governance patterns for enterprise-scale agent systems
- Monitor and audit governance across agent organisations

---

## Recap: Lesson 6.6

- `PactGovernedAgent` wraps agents with governance enforcement
- `GovernanceContext` tracks runtime state (clearance, budget, session)
- Pre-action checks verify rights; post-action checks enforce duties
- Governance test suites verify compliance before deployment

---

## From Single Agent to Organisation

```
Lesson 6.6: One governed agent
  └→ Simple: one envelope, one budget, one clearance

Lesson 6.7: Organisation of governed agents
  └→ Complex: hierarchy of envelopes, cascading budgets,
     delegated clearance, cross-agent accountability

  Supervisor (clearance: confidential, budget: $100/day)
  ├→ Analyst (clearance: internal, budget: $30/day)
  ├→ Advisor (clearance: public, budget: $40/day)
  └→ Reporter (clearance: internal, budget: $30/day)
```

---

## Clearance Hierarchies

```
Restricted ─────────────────── highest access
    │
    ▼
Confidential ────────────────── financial data, PII
    │
    ▼
Internal ────────────────────── model metrics, training stats
    │
    ▼
Public ──────────────────────── market data, policy docs

Rule: An agent can only delegate LOWER or EQUAL clearance.
  Supervisor (confidential) can create:
    ✅ Worker with "internal" clearance
    ✅ Worker with "public" clearance
    ❌ Worker with "restricted" clearance
```

---

## Clearance Configuration

```python
from kailash_pact import GovernanceEngine, ClearanceHierarchy

hierarchy = ClearanceHierarchy(
    levels=["public", "internal", "confidential", "restricted"],

    # What data each level can access
    access_map={
        "public": ["town_statistics", "price_trends", "policy_docs"],
        "internal": ["model_metrics", "training_stats", "error_logs"],
        "confidential": ["individual_transactions", "user_profiles"],
        "restricted": ["financial_records", "identity_documents"],
    },
)

engine = GovernanceEngine()
engine.configure(
    clearance_hierarchy=hierarchy,
)
```

---

## Budget Cascading

```python
from kailash_pact import BudgetPolicy

# Supervisor budget
supervisor_budget = BudgetPolicy(
    daily_limit=100.00,
    per_request_limit=10.00,

    # Cascading rules
    delegation_enabled=True,
    max_delegation_percent=0.40,  # delegate max 40% to any worker
    reserve_percent=0.10,         # keep 10% for own use
)

# Worker budgets are carved from supervisor
# Supervisor: $100/day
#   → Analyst:  $30/day (30%)
#   → Advisor:  $40/day (40%)
#   → Reporter: $20/day (20%)
#   → Reserve:  $10/day (10%)
```

---

## Budget Cascading in Practice

```python
from kailash_pact import PactGovernedAgent, GovernanceContext

# Supervisor creates worker contexts with delegated budget
supervisor_context = GovernanceContext(
    agent_id="supervisor_01",
    clearance="confidential",
    budget_remaining=100.00,
)

# Delegate budget to worker
analyst_context = supervisor_context.delegate(
    agent_id="analyst_01",
    clearance="internal",         # cannot exceed supervisor's clearance
    budget_allocation=30.00,      # carved from supervisor's budget
)

# Supervisor's remaining budget
print(f"Supervisor remaining: ${supervisor_context.budget_remaining:.2f}")
# $70.00 (after delegating $30)
```

---

## Multi-Agent Governed Organisation

```python
from kailash_kaizen import SupervisorWorker
from kailash_pact import PactGovernedAgent, GovernanceEngine

# Create governed agents
supervisor = PactGovernedAgent(
    agent=supervisor_agent,
    governance=engine,
    clearance="confidential",
    budget=100.00,
)

workers = {
    "analyst": PactGovernedAgent(
        agent=analyst_agent, governance=engine,
        clearance="internal", budget=30.00,
    ),
    "advisor": PactGovernedAgent(
        agent=advisor_agent, governance=engine,
        clearance="public", budget=40.00,
    ),
}

# Orchestrate with governance
orchestrator = SupervisorWorker()
orchestrator.configure(
    supervisor=supervisor,
    workers=workers,
)
```

---

## Cross-Agent Accountability

```
Every action traces back through the hierarchy:

Reporter generates report
  → authorised by: Supervisor (session_123)
  → data from: Analyst query (governed, logged)
  → displayed to: User (public clearance)
  → budget: $2.50 from Reporter's $20 allocation
  → audit trail: complete chain of responsibility
```

---

## Governance Patterns

| Pattern          | When to Use                                      |
| ---------------- | ------------------------------------------------ |
| **Flat**         | Small team, all agents equal clearance           |
| **Hierarchical** | Enterprise, clear chain of command               |
| **Mesh**         | Peer agents with specialised clearances          |
| **Gateway**      | All external requests through one governed entry |

---

## Gateway Pattern

```
External requests
      ↓
┌─────────────────┐
│ Gateway Agent    │ ← All governance checks here
│ (public clearance│
│  rate limiting   │
│  input validation│
└────────┬────────┘
         │
    ┌────┼────┐
    ↓    ↓    ↓
  Agent  Agent  Agent  ← Internal agents with higher clearance
  (int)  (int)  (conf)
```

The gateway is the single point of governance enforcement.

---

## Organisation-Wide Monitoring

```python
# Monitor governance across all agents
from kailash_pact import OrganisationMonitor

monitor = OrganisationMonitor(engine=engine)
report = monitor.generate_report(period="24h")

print("=== Organisation Governance Report ===")
print(f"Total agents active: {report.active_agents}")
print(f"Total requests: {report.total_requests}")
print(f"Budget utilisation: {report.budget_utilisation:.1%}")

print("\nPer-agent breakdown:")
for agent in report.agents:
    print(f"  {agent.id}:")
    print(f"    Requests: {agent.requests}")
    print(f"    Blocked: {agent.blocked} ({agent.blocked_pct:.1%})")
    print(f"    Escalated: {agent.escalated}")
    print(f"    Budget used: ${agent.budget_used:.2f}/{agent.budget_limit:.2f}")
```

---

## Governance Drift Detection

```python
# Detect when agents approach governance boundaries
alerts = monitor.check_drift(
    budget_warning_threshold=0.8,     # 80% budget used
    escalation_rate_threshold=0.15,   # 15%+ escalations
    block_rate_threshold=0.05,        # 5%+ blocks
)

for alert in alerts:
    print(f"[{alert.severity}] {alert.agent_id}: {alert.message}")
```

```
[WARNING] advisor_01: Budget 85% utilised (projected to exceed by 14:00)
[INFO] analyst_01: Escalation rate 12% (below threshold)
[CRITICAL] advisor_02: Block rate 8% — possible misconfiguration
```

---

## Exercise Preview

**Exercise 6.7: Governed Multi-Agent Organisation**

You will:

1. Design a clearance hierarchy for an HDB advisory organisation
2. Implement budget cascading from supervisor to workers
3. Build a governed SupervisorWorker system
4. Monitor and audit governance across the organisation

Scaffolding level: **Minimal (~20% code provided)**

---

## Common Pitfalls

| Mistake                                | Fix                                                |
| -------------------------------------- | -------------------------------------------------- |
| All agents with same clearance         | Apply least-privilege: minimum necessary clearance |
| No budget reserve for supervisor       | Reserve 10% for supervisor's own operations        |
| Worker can exceed supervisor clearance | Framework prevents this; verify in tests           |
| No organisation-wide monitoring        | Always aggregate governance metrics                |
| Static budget allocation               | Monitor utilisation; rebalance weekly              |

---

## Summary

- Clearance hierarchies control data access across agent organisations
- Budget cascading delegates cost limits from supervisor to workers
- Gateway pattern centralises governance enforcement
- Organisation monitors track governance health across all agents
- Every action traces through the accountability chain

---

## Next Lesson

**Lesson 6.8: Capstone — Full Platform**

We will learn:

- Combining all 8 Kailash packages into one platform
- End-to-end: data to model to agent to governed deployment
- The final ASCENT capstone project
