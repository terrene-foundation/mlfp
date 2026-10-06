# MLFP06 — Task 2: A Governed Agent's Tools and Operating Config

**Weight**: 20 marks · **Framework**: Kaizen (`ToolRegistry`) + kaizen-agents
(`GovernedSupervisor`) · **Outcomes assessed**: tool registration with JSON
schemas, per-agent budget and clearance config (6.5, 6.7)

## Scenario

A back-office agent answers staff questions with three small tools. Before it
can go near production, two things must exist and be right:

1. a **tool registry** — the only way the model can call your functions is if
   each is registered with a name, a description, a JSON-schema parameter
   card, and an executor (handing the agent bare Python functions registers
   nothing);
2. a **governed agent config** — a per-agent dollar budget and a data
   clearance, set at construction so the envelope is real from the first call.

No LLM is needed to verify either: the grader calls your registered executors
with its own inputs, reads your agent's envelope, and runs one governed
objective with a grader-supplied executor.

## Interfaces

```python
def build_tools() -> ToolRegistry: ...
def build_agent(registry: ToolRegistry) -> GovernedSupervisor: ...
```

### The three tools

Each executor is async, takes keyword arguments, and returns a **JSON
string**. Behaviour contracts:

| Tool                 | Arguments                    | Returns (JSON string)                                                                                                       |
| -------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `compute_statistics` | `numbers: list[float]`       | `{"count": int, "mean": float (4 dp), "min": float, "max": float}`                                                          |
| `normalise_text`     | `text: str`                  | `{"normalised": str}` — lowercase, leading/trailing whitespace stripped, runs of internal whitespace collapsed to one space |
| `convert_currency`   | `amount: float, rate: float` | `{"converted": float}` — `amount * rate`, rounded to 2 dp                                                                   |

Each tool's `parameters` card is a JSON schema object with a `"properties"`
map covering exactly the arguments above (plus `"type": "object"`).

### The agent

`build_agent` returns a `GovernedSupervisor` constructed with:

- `budget_usd=0.25`;
- `data_clearance="internal"` (the organisation's internal-data alias);
- `tools=[...]` the three tool names;
- `model=` the course default from `shared.mlfp06._ollama_bootstrap`
  (`DEFAULT_CHAT_MODEL`) — no hardcoded model names. Construction makes no
  network calls.

## Acceptance criteria (what the grader measures)

| #   | Check                                                                                |
| --- | ------------------------------------------------------------------------------------ |
| 1   | `build_tools()` returns a registry exposing `tool_names` (gate)                      |
| 2   | Exactly the three required tool names are registered                                 |
| 3   | `compute_statistics` correct on three grader-drawn inputs                            |
| 4   | `normalise_text` correct on three grader-drawn strings                               |
| 5   | `convert_currency` correct on three grader-drawn (amount, rate) pairs                |
| 6   | Every tool carries a JSON-schema parameters card naming its arguments                |
| 7   | Agent clearance is `RESTRICTED` (the `internal` alias maps to it)                    |
| 8   | Agent envelope carries `budget_usd` 0.25                                             |
| 9   | Agent's operational envelope lists exactly the three tools                           |
| 10  | One governed run with a grader executor succeeds; the audit chain grows and verifies |

Marks = 20 × (non-gate checks passed / 9). If the gate fails, the task
scores 0.

## Rules

- No LLM calls anywhere in this task.
- Executors must compute from their arguments — the grader draws fresh inputs
  every run, so canned outputs fail.
- Deterministic: same arguments, same JSON.
- Self-check: run `starter.py`; it registers your tools, calls each once, and
  prints your agent's envelope summary.
