---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 5.3: ReAct Agents with Tools

### Module 5: LLMs and Agents

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain the ReAct pattern: interleaving reasoning and action
- Give agents access to tools (database, calculator, search)
- Build a `ReActAgent` that gathers information before answering
- Design safe tool boundaries for production agents

---

## Recap: Lesson 5.2

- Chain-of-thought breaks complex problems into verifiable steps
- `ChainOfThoughtAgent` provides structured multi-step reasoning
- Reasoning templates guide analytical frameworks
- Self-consistency validates reasoning stability

---

## The Limitation of Pure Reasoning

```
CoT Agent:
  "Step 1: The average price in Tampines is approximately $490k"
  → But is this ACTUALLY $490k? The LLM is guessing from training data.

ReAct Agent:
  "Thought: I need the actual average price in Tampines.
   Action: query_database(town='TAMPINES', metric='avg_price')
   Observation: $512,340
   Thought: Now I have the real number. Continuing analysis..."
  → Grounded in real data.
```

ReAct = **Re**asoning + **Act**ing. Think, then use tools, then think again.

---

## The ReAct Loop

```
┌──────────┐
│ Thought  │  "I need to check current prices..."
└────┬─────┘
     ↓
┌──────────┐
│  Action  │  query_database(town="TAMPINES")
└────┬─────┘
     ↓
┌──────────┐
│Observation│  {"avg_price": 512340, "count": 1234}
└────┬─────┘
     ↓
┌──────────┐
│ Thought  │  "Average is $512k. Now I need floor area data..."
└────┬─────┘
     ↓
   ... repeat until answer is complete ...
     ↓
┌──────────┐
│  Answer  │  "Based on actual market data, this flat is..."
└──────────┘
```

---

## Defining Tools

```python
from kailash_kaizen import Tool

# Database query tool
@Tool(description="Query HDB transaction database")
def query_database(town: str, flat_type: str = None, metric: str = "avg_price"):
    """Query the HDB database for market statistics."""
    query = df.filter(pl.col("town") == town)
    if flat_type:
        query = query.filter(pl.col("flat_type") == flat_type)

    if metric == "avg_price":
        return {"avg_price": query["price"].mean(), "count": len(query)}
    elif metric == "price_range":
        return {"min": query["price"].min(), "max": query["price"].max()}

# Calculator tool
@Tool(description="Perform arithmetic calculations")
def calculate(expression: str):
    """Evaluate a mathematical expression safely."""
    allowed = set("0123456789+-*/.() ")
    if all(c in allowed for c in expression):
        return {"result": eval(expression)}
    return {"error": "Invalid expression"}
```

---

## Building a ReActAgent

```python
from kailash_kaizen import ReActAgent, Signature

agent = ReActAgent()
agent.configure(
    model="claude-sonnet",
    tools=[query_database, calculate],
    max_iterations=10,
    temperature=0.0,
)

sig = Signature(
    input_fields={
        "question": "The property question to answer",
    },
    output_fields={
        "answer": "The detailed answer based on real data",
        "data_sources": "List of tools/queries used",
    },
)

result = agent.execute(sig, inputs={
    "question": "Is a 4-room flat in Tampines at $540k a good deal?"
})
```

---

## Tracing Agent Actions

```python
# Every action is logged
for step in result.trace:
    print(f"[{step.type}] {step.content}")
```

```
[THOUGHT]      I need to check the average price for 4-room flats in Tampines.
[ACTION]       query_database(town="TAMPINES", flat_type="4 ROOM", metric="avg_price")
[OBSERVATION]  {"avg_price": 498500, "count": 856}
[THOUGHT]      Average is $498,500. Let me check the price range.
[ACTION]       query_database(town="TAMPINES", flat_type="4 ROOM", metric="price_range")
[OBSERVATION]  {"min": 380000, "max": 650000}
[THOUGHT]      $540k is above average but within range. Let me calculate percentile.
[ACTION]       calculate("(540000 - 498500) / 498500 * 100")
[OBSERVATION]  {"result": 8.32}
[ANSWER]       At $540k, this is 8.3% above the average of $498,500...
```

---

## Tool Design Principles

```
1. Clear descriptions
   → The agent decides WHICH tool to use based on descriptions

2. Typed parameters
   → Agent provides correct types; validation catches errors

3. Bounded scope
   → Each tool does ONE thing well

4. Safe execution
   → No side effects (read-only where possible)
   → Input validation prevents injection

5. Informative return values
   → Return enough context for the agent to reason
```

---

## Read-Only vs Write Tools

```python
# Read-only: safe for any agent
@Tool(description="Look up transaction data")
def lookup_transactions(town: str, flat_type: str):
    return df.filter(...).to_dicts()

# Write tool: needs careful access control
@Tool(description="Save a valuation report", requires_confirmation=True)
def save_report(report: dict):
    """Save report to database. Requires user confirmation."""
    db.save(ValuationReport(**report))
    return {"status": "saved", "id": report_id}
```

Write tools should require confirmation or operate within strict boundaries.

---

## Multiple Specialised Tools

```python
@Tool(description="Get town demographics and amenities")
def get_town_info(town: str):
    return demographics.filter(pl.col("town") == town).to_dicts()

@Tool(description="Get recent price trends for a town")
def get_price_trend(town: str, months: int = 6):
    recent = df.filter(pl.col("town") == town).sort("date").tail(months)
    return {"trend": recent["price"].to_list(), "direction": "rising"}

@Tool(description="Compare two properties")
def compare_properties(property_a: dict, property_b: dict):
    # Structured comparison logic
    return {"price_diff": ..., "area_diff": ..., "value_comparison": ...}

agent.configure(
    tools=[query_database, calculate, get_town_info,
           get_price_trend, compare_properties],
)
```

---

## ReAct vs CoT: When to Use Each

| Scenario                         | Use   | Why                         |
| -------------------------------- | ----- | --------------------------- |
| Analysis with known data         | CoT   | All info in the prompt      |
| Questions needing real-time data | ReAct | Must query external sources |
| Multi-source research            | ReAct | Combines multiple tools     |
| Pure reasoning/logic             | CoT   | No external data needed     |
| Calculations needed              | ReAct | LLMs are bad at arithmetic  |

---

## Error Recovery

```python
agent.configure(
    model="claude-sonnet",
    tools=[query_database, calculate],
    max_iterations=10,

    # Error handling
    retry_on_tool_error=True,
    max_retries=2,
    fallback_response="I was unable to gather enough data to answer.",
)
```

Agents can retry failed tool calls or gracefully degrade.

---

## Exercise Preview

**Exercise 5.3: Data-Grounded Property Advisor**

You will:

1. Define tools for database queries, calculations, and comparisons
2. Build a `ReActAgent` that researches before answering
3. Trace agent actions and evaluate tool usage quality
4. Handle edge cases (missing data, invalid queries)

Scaffolding level: **Light (~30% code provided)**

---

## Common Pitfalls

| Mistake                          | Fix                                     |
| -------------------------------- | --------------------------------------- |
| Too many tools (>10)             | Agent gets confused; keep tools focused |
| Vague tool descriptions          | Agent picks wrong tool; be specific     |
| No tool error handling           | Always handle failures gracefully       |
| Allowing unsafe write operations | Read-only by default; confirm writes    |
| Not tracing agent actions        | Always inspect the trace for debugging  |

---

## Summary

- ReAct interleaves reasoning and tool use for grounded answers
- Tools give agents access to databases, calculators, and APIs
- The Thought-Action-Observation loop continues until the answer is complete
- Tool design: clear descriptions, typed parameters, bounded scope
- Read-only tools are safe; write tools need access control

---

## Next Lesson

**Lesson 5.4: RAG Systems**

We will learn:

- Retrieval-Augmented Generation with `RAGResearchAgent`
- Vector embeddings and similarity search
- Building knowledge bases for domain-specific Q&A
