---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 5.5: MCP Servers

### Module 5: LLMs and Agents

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain the Model Context Protocol and why it standardises tool access
- Register tools and resources on an MCP server
- Configure transports (stdio, SSE) for different deployment scenarios
- Connect Kaizen agents to MCP servers for standardised tool use

---

## Recap: Lesson 5.4

- RAG retrieves relevant documents before generating answers
- Chunking, embedding, and vector search form the retrieval pipeline
- `RAGResearchAgent` combines retrieval with LLM generation
- Source attribution and retrieval quality are critical

---

## The Tool Fragmentation Problem

```
Agent A uses: query_database(), calculate(), search_web()
Agent B uses: db_query(), math_eval(), web_search()
Agent C uses: sql_query(), compute(), internet_lookup()

Same capabilities, different interfaces.
Every agent needs custom tool integration code.

MCP solves this:
  Standard protocol → Any agent can use any MCP server
  Like HTTP standardised web access
```

---

## Model Context Protocol (MCP)

```
┌──────────────────┐         ┌──────────────────┐
│   MCP Client     │◄───────►│   MCP Server     │
│   (Agent)        │ Standard│   (Tool Provider) │
│                  │ Protocol│                    │
│ - Discovers tools│         │ - Registers tools  │
│ - Calls tools    │         │ - Executes tools   │
│ - Gets results   │         │ - Returns results  │
└──────────────────┘         └──────────────────┘

Any MCP client can use any MCP server.
No custom integration needed.
```

---

## Building an MCP Server

```python
from kailash_mcp import MCPServer, tool, resource

server = MCPServer(name="hdb-data-server")

@server.tool(description="Query HDB transaction data")
def query_transactions(town: str, flat_type: str = None) -> dict:
    query = df.filter(pl.col("town") == town)
    if flat_type:
        query = query.filter(pl.col("flat_type") == flat_type)
    return {
        "avg_price": query["price"].mean(),
        "count": len(query),
        "min": query["price"].min(),
        "max": query["price"].max(),
    }

@server.tool(description="Get price trend for a town")
def get_trend(town: str, months: int = 12) -> dict:
    # ... trend computation ...
    return {"trend": trend_data, "direction": "rising"}
```

---

## Registering Resources

```python
@server.resource(
    uri="hdb://towns",
    description="List of all HDB towns with basic stats",
)
def list_towns() -> list:
    return df["town"].unique().sort().to_list()

@server.resource(
    uri="hdb://towns/{town}/summary",
    description="Summary statistics for a specific town",
)
def town_summary(town: str) -> dict:
    subset = df.filter(pl.col("town") == town)
    return {
        "town": town,
        "total_transactions": len(subset),
        "avg_price": subset["price"].mean(),
        "flat_types": subset["flat_type"].unique().to_list(),
    }
```

Resources are read-only data endpoints. Tools perform actions.

---

## Transport: How Client and Server Communicate

```python
# Stdio transport (for local/CLI use)
server.configure(transport="stdio")
server.start()

# SSE transport (for network/web use)
server.configure(
    transport="sse",
    host="0.0.0.0",
    port=8090,
)
server.start()
```

| Transport | Use Case                                        |
| --------- | ----------------------------------------------- |
| **stdio** | Local process, CLI tools, desktop agents        |
| **SSE**   | Network access, web applications, remote agents |

---

## Connecting Agents to MCP

```python
from kailash_kaizen import ReActAgent
from kailash_mcp import MCPClient

# Connect to the MCP server
client = MCPClient()
client.connect("http://localhost:8090")  # SSE transport

# Discover available tools
tools = client.list_tools()
for tool in tools:
    print(f"  {tool.name}: {tool.description}")

# Agent automatically uses MCP tools
agent = ReActAgent()
agent.configure(
    model="claude-sonnet",
    mcp_clients=[client],   # agent discovers tools via MCP
)

result = agent.execute(sig, inputs={"question": "Best value 4-room flat?"})
```

---

## Tool Discovery

```python
# Agent discovers tools at runtime — no hardcoding
tools = client.list_tools()
```

```
Available tools:
  query_transactions: Query HDB transaction data
  get_trend: Get price trend for a town

Available resources:
  hdb://towns: List of all HDB towns with basic stats
  hdb://towns/{town}/summary: Summary statistics for a specific town
```

The agent reads descriptions and decides which tools to call.

---

## Multiple MCP Servers

```python
# Agent connects to multiple specialised servers
hdb_client = MCPClient()
hdb_client.connect("http://localhost:8090")   # HDB data server

weather_client = MCPClient()
weather_client.connect("http://localhost:8091")  # Weather data server

news_client = MCPClient()
news_client.connect("http://localhost:8092")  # News search server

agent = ReActAgent()
agent.configure(
    model="claude-sonnet",
    mcp_clients=[hdb_client, weather_client, news_client],
)

# Agent can now use tools from ALL servers
```

---

## Input Validation

```python
from pydantic import BaseModel, Field

class TransactionQuery(BaseModel):
    town: str = Field(description="HDB town name (uppercase)")
    flat_type: str = Field(default=None, description="e.g., '4 ROOM'")
    min_price: int = Field(default=0, ge=0)
    max_price: int = Field(default=10_000_000, le=10_000_000)

@server.tool(
    description="Query HDB transactions with filters",
    input_model=TransactionQuery,
)
def query_filtered(query: TransactionQuery) -> dict:
    # Input is validated before reaching this function
    result = df.filter(
        (pl.col("town") == query.town)
        & (pl.col("price") >= query.min_price)
        & (pl.col("price") <= query.max_price)
    )
    return {"count": len(result), "avg_price": result["price"].mean()}
```

---

## MCP Server Testing

```python
# Test tools directly without starting the server
from kailash_mcp import MCPTestClient

test_client = MCPTestClient(server)

# Call tools in tests
result = test_client.call_tool("query_transactions", {
    "town": "TAMPINES",
    "flat_type": "4 ROOM",
})
assert result["count"] > 0
assert result["avg_price"] > 0

# Test resource access
towns = test_client.read_resource("hdb://towns")
assert "TAMPINES" in towns
```

---

## Exercise Preview

**Exercise 5.5: HDB Data MCP Server**

You will:

1. Build an MCP server with tools for HDB data queries
2. Register resources for town listings and summaries
3. Connect a ReAct agent to the MCP server
4. Test the server with the MCP test client

Scaffolding level: **Light (~30% code provided)**

---

## Common Pitfalls

| Mistake                                | Fix                                              |
| -------------------------------------- | ------------------------------------------------ |
| Vague tool descriptions                | Agent picks wrong tool; be specific              |
| No input validation                    | Use Pydantic models for type safety              |
| Exposing write operations without auth | Read-only by default for safety                  |
| Too many tools on one server           | Group related tools; split into multiple servers |
| Not testing tools independently        | Use MCPTestClient before connecting agents       |

---

## Summary

- MCP standardises how agents discover and use tools
- Servers register tools (actions) and resources (read-only data)
- Transports: stdio for local, SSE for network access
- Agents discover tools at runtime via MCP protocol
- Multiple servers can provide specialised capabilities to one agent

---

## Next Lesson

**Lesson 5.6: ML Agent Pipeline**

We will learn:

- Six specialised ML agents for the full lifecycle
- Orchestrating data profiling, feature engineering, training, and evaluation with agents
- Building an agent-driven ML pipeline
