---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 5.8: Production Deployment

### Module 5: LLMs and Agents

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Deploy agent systems with Kailash Nexus for multi-channel access
- Configure authentication, middleware, and rate limiting
- Expose workflows as API + CLI + MCP simultaneously
- Complete the Module 5 capstone project

---

## Recap: Lesson 5.7

- Sequential, Parallel, Supervisor-Worker, and Handoff patterns
- Choose pattern based on task structure and independence
- Combine patterns for complex orchestration systems
- Error handling and timeouts prevent cascading failures

---

## The Deployment Challenge

```
You have built:
  - ML models (Module 3-4)
  - Agents with tools (Module 5)
  - Multi-agent orchestration

Now you need to make them accessible:
  - API for web/mobile apps
  - CLI for data scientists
  - MCP for AI agent integration

Three interfaces, one codebase → Kailash Nexus
```

---

## Nexus: Zero-Config Multi-Channel

```python
from kailash_nexus import Nexus

app = Nexus(name="hdb-advisor")

@app.register
def predict_price(town: str, flat_type: str,
                  floor_area: float, lease_years: int) -> dict:
    """Predict HDB resale price."""
    model = registry.load("hdb_price_predictor", stage="production")
    prediction = model.predict([[town, flat_type, floor_area, lease_years]])
    return {
        "predicted_price": int(prediction[0]),
        "town": town,
        "flat_type": flat_type,
    }

# This single function is now accessible via API, CLI, AND MCP
app.start()
```

---

## Three Channels from One Function

```
API:
  POST /predict_price
  {"town": "TAMPINES", "flat_type": "4 ROOM",
   "floor_area": 92.0, "lease_years": 72}

CLI:
  $ hdb-advisor predict-price --town TAMPINES --flat-type "4 ROOM" \
      --floor-area 92 --lease-years 72

MCP:
  tool: predict_price
  args: {"town": "TAMPINES", "flat_type": "4 ROOM", ...}

All three call the same function. Zero duplication.
```

---

## Registering Agent Workflows

```python
from kailash_nexus import Nexus
from kailash_kaizen import ReActAgent

app = Nexus(name="hdb-advisor")

@app.register
def analyse_listing(description: str, asking_price: int) -> dict:
    """Analyse an HDB listing with AI agent."""
    agent = ReActAgent()
    agent.configure(
        model="claude-sonnet",
        tools=[query_database, calculate, get_trend],
    )

    result = agent.execute(analysis_sig, inputs={
        "description": description,
        "asking_price": asking_price,
    })

    return {
        "assessment": result.assessment,
        "fair_value": result.fair_value,
        "reasoning": result.reasoning_steps,
    }
```

---

## Authentication

```python
from kailash_nexus import Nexus, auth

app = Nexus(name="hdb-advisor")

# API key authentication
app.configure(
    auth=auth.APIKey(
        header="X-API-Key",
        keys_env="HDB_API_KEYS",  # comma-separated in env var
    ),
)

# Or JWT authentication
app.configure(
    auth=auth.JWT(
        secret_env="JWT_SECRET",
        algorithm="HS256",
    ),
)
```

---

## Middleware

```python
from kailash_nexus import Nexus, middleware

app = Nexus(name="hdb-advisor")

# Rate limiting
app.use(middleware.RateLimit(
    requests_per_minute=60,
    requests_per_hour=1000,
))

# Request logging
app.use(middleware.Logger(
    level="INFO",
    log_request_body=False,  # do not log sensitive data
))

# CORS (for web frontends)
app.use(middleware.CORS(
    origins=["https://hdb-advisor.example.com"],
    methods=["GET", "POST"],
))
```

---

## Session Management

```python
from kailash_nexus import Nexus

app = Nexus(name="hdb-advisor")

@app.register
def chat(message: str, session_id: str = None) -> dict:
    """Conversational HDB advisor with session memory."""
    session = app.get_session(session_id)

    # Session maintains conversation history
    session.add_message("user", message)

    result = agent.execute(chat_sig, inputs={
        "message": message,
        "history": session.get_history(),
    })

    session.add_message("assistant", result.response)

    return {
        "response": result.response,
        "session_id": session.id,
    }
```

---

## Health Checks and Monitoring

```python
# Built-in endpoints (automatic)
# GET /health          → {"status": "healthy", "version": "1.0"}
# GET /docs            → OpenAPI documentation
# GET /metrics         → Prometheus-compatible metrics

# Custom health check
@app.health_check
def check_model_loaded():
    model = registry.load("hdb_price_predictor", stage="production")
    return model is not None
```

---

## Multi-Agent Deployment

```python
app = Nexus(name="hdb-platform")

# Register multiple agent-powered endpoints
@app.register
def quick_valuation(town: str, flat_type: str, floor_area: float) -> dict:
    """Quick price estimate (Delegate — fast)."""
    return delegate.execute(valuation_sig, inputs={...})

@app.register
def deep_analysis(description: str, asking_price: int) -> dict:
    """Deep analysis with reasoning (ReAct — thorough)."""
    return react_agent.execute(analysis_sig, inputs={...})

@app.register
def market_report(town: str) -> dict:
    """Comprehensive market report (Multi-agent — detailed)."""
    return supervisor.execute(report_sig, inputs={...})

app.start(host="0.0.0.0", port=8080)
```

---

## Deployment Configuration

```python
app = Nexus(name="hdb-advisor")

app.configure(
    # Server settings
    host="0.0.0.0",
    port=8080,
    workers=4,

    # API settings
    api_prefix="/api/v1",
    docs_enabled=True,

    # MCP settings
    mcp_enabled=True,
    mcp_transport="sse",

    # CLI settings
    cli_enabled=True,
    cli_name="hdb-advisor",
)

app.start()
```

---

## Module 5 Capstone

**Project: AI-Powered HDB Advisory Platform**

```
1. LLM Foundation (5.1-5.2)
   └→ Signature/Delegate for extraction, CoT for analysis

2. Tool-Augmented Agents (5.3-5.5)
   └→ ReAct agents with MCP tools and RAG knowledge

3. Agent Orchestration (5.6-5.7)
   └→ ML agent pipeline with multi-agent coordination

4. Production Deployment (5.8)
   └→ Nexus with auth, rate limiting, multi-channel
```

---

## Exercise Preview

**Exercise 5.8: Deploy the HDB Advisory Platform**

You will:

1. Register agent-powered functions with Nexus
2. Configure authentication and rate limiting
3. Deploy as API + CLI + MCP simultaneously
4. Test all three channels with the same query

Scaffolding level: **Light (~30% code provided)**

---

## Common Pitfalls

| Mistake                                   | Fix                                         |
| ----------------------------------------- | ------------------------------------------- |
| No authentication on agent endpoints      | Always add auth for production              |
| No rate limiting on LLM-powered endpoints | LLM calls are expensive; limit wisely       |
| Synchronous agents blocking the API       | Use async execution for long-running agents |
| API keys hardcoded in code                | Use environment variables                   |
| No health check for model availability    | Check model loads before serving            |

---

## Module 5 Summary

| Lesson | Key Skills                                       |
| ------ | ------------------------------------------------ |
| 5.1    | Signature, Delegate, LLM fundamentals            |
| 5.2    | ChainOfThoughtAgent, reasoning chains            |
| 5.3    | ReActAgent, tools, Thought-Action-Observation    |
| 5.4    | RAGResearchAgent, vector search, knowledge bases |
| 5.5    | MCP servers, tool registration, transports       |
| 5.6    | Six ML agents, agent-driven pipelines            |
| 5.7    | Sequential, Parallel, Supervisor-Worker, Handoff |
| 5.8    | Nexus deployment, auth, middleware               |

---

## What Comes Next

**Module 6: Alignment and Governance**

- Fine-tuning LLMs with LoRA and DPO
- Reinforcement learning from human feedback
- AI governance with PACT framework
- Governed agent systems for enterprise deployment

You can now build and deploy AI agents. Next, we ensure they are aligned and governed.
