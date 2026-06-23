# MLFP06 Assessment — Language Models & Agentic Workflows

End-of-module assessment for **MLFP06: Machine Learning with Language Models and
Agentic Workflows**. Four auto-graded tasks covering the module's core skills:
prompt engineering with structured output, retrieval-augmented generation,
tool-using agents, and production governance.

**Total: 100 marks · Duration: 3 hours · Open-book · No AI assistants.**

All tasks run against a **local Ollama** model (`llama3.2:3b`) plus the
`nomic-embed-text` embedding model — no API keys, no cloud models, no internet
required during the exam. Ensure the Ollama daemon is running (`ollama serve`)
and both models are pulled before you begin.

## Tasks

| Task   | Topic                                   | Framework                             | Marks |
| ------ | --------------------------------------- | ------------------------------------- | ----- |
| Task 1 | Prompt engineering & structured output  | Kaizen `Signature` + `BaseAgent`      | 20    |
| Task 2 | RAG pipeline with evaluation            | Kaizen Ollama embeddings + `Delegate` | 25    |
| Task 3 | Tool-using agent over a real dataset    | Kaizen `Delegate` + `ToolRegistry`    | 25    |
| Task 4 | Governance for a production agent fleet | PACT `GovernanceEngine`               | 30    |

Each task lives in its own folder (`task_1/` … `task_4/`) and contains:

- `problem.md` — the scenario, exact contract, and grading checklist
- `starter.py` — the file you complete and submit

## How to work

1. Read `task_N/problem.md` in full. It specifies the exact return contract,
   the datasets, the target, and the grading checklist.
2. Complete the `solve()` function in `task_N/starter.py`. The placeholder does
   not pass — you must implement the TODOs.
3. Run your file directly to sanity-check it (e.g.
   `.venv/bin/python task_1/starter.py`) and compare against the visible sanity
   checks in `problem.md`.
4. **Submit your completed `starter.py` files to the portal.**

## Environment

```bash
# One-time: confirm the daemon is up and the models are present
ollama serve            # if not already running
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# Run a task file (always use the project venv)
.venv/bin/python modules/mlfp06/assessment/task_1/starter.py
```

- **Polars only** — no pandas anywhere.
- **Temperature 0** for every LLM call (the tasks require it for stable output).
- Read model names / endpoint from the course Ollama bootstrap
  (`shared.mlfp06._ollama_bootstrap`); never hardcode keys.

## Grading

Each task is graded automatically against robust, deterministic outcomes —
schema compliance, retrieval recall@k, tool-selection + arguments, and
governance-policy verdicts — **not** exact LLM text (which is not bit-stable).
The graders are withheld; your visible sanity checks in each `problem.md` tell
you when your implementation is on track. A task passes only when **all** of its
checks pass.
