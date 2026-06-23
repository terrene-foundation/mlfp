---
type: DECISION
date: 2026-04-17
created_at: 2026-04-17T00:00:00Z
author: co-authored
session_id: m6-ollama-migration-2026-04-17
session_turn: 1
project: mlfp
topic: M6 LLM provider switched to local Ollama; silent OpenAI stub deleted
phase: implement
tags: [m6, ollama, llm, redline-14, zero-tolerance]
---

# DECISION: M6 routes every LLM call through local Ollama; silent stub fallback BLOCKED

## What was decided

M6 (LLMs and Agentic Workflows) ships pre-configured for **local Ollama**
as the default LLM provider for every lesson. No API keys, no commercial
provider account. Every M6 LLM call constructs its Kaizen Delegate via a
single sanctioned factory — `shared.mlfp06._ollama_bootstrap.make_delegate`
— that:

1. Uses Kaizen's Ollama adapter (`get_adapter("ollama", ...)`) so the
   provider routing is explicit, not driven by env var ambiguity.
2. Forces `budget_usd=None` because Kaizen mis-prices Ollama at $3/$15
   per million tokens (verified via the kaizen-specialist's source read of
   `Delegate._estimate_cost` line 144).
3. Reads `OLLAMA_BASE_URL` / `OLLAMA_CHAT_MODEL` / `OLLAMA_EMBED_MODEL`
   from `.env` for overrides.

The previous `run_delegate(prompt, max_cost=0.5)` helper that swallowed
any provider exception and returned `("unknown", 0.0, latency)` is
**deleted**. Daemon-down or model-not-pulled now raises
`OllamaUnreachableError` with the exact `ollama serve` / `ollama pull`
command needed to fix the environment.

This change is codified as **Redline 14** in `specs/redlines.md`.

## Why this approach

User mandate: _"i want the most complete rigorous pristine implementation.
no openai, use ollama."_

The silent stub fallback was caught by the M6 holistic audit
(`workspaces/curriculum/journal/.pending/1776437164195-0-DISCOVERY.md`,
open question #2) — a student running ex_1/01_zero_shot without an
OpenAI key got `accuracy=0% cost=$0.00`, visually indistinguishable from
a successful run. That violated `rules/zero-tolerance.md` Rule 2 (no
silent placeholders) and Rule 3 (no silent fallbacks). The Ollama
migration removes the fallback AND removes the API-key dependency that
made the fallback "needed" in the first place.

## Alternatives considered

**Keep OpenAI, add a loud "OFFLINE" banner**: would have closed the
"fake-zero accuracy" trap but would still require every student to hold
an API key and would still burn budget every notebook re-run. Rejected
in favour of removing the API-key dependency entirely.

**Hybrid (OpenAI primary, Ollama fallback)**: more complex provider
routing, two code paths to test, divergent UX between students with and
without keys. Rejected — pick one, do it well.

**Keep silent fallback for CI smoke tests**: the silent path was never
actually a CI feature — the CI guards check `ast.parse` and `Cell 1
exec`, neither of which exercises the LLM call. Removing the fallback
costs nothing in CI.

## Model selection

Per-lesson manifest (in `_ollama_bootstrap.LESSON_MODELS` and mirrored in
`scripts/generate_selfcontained_notebook.py::_M6_LESSON_MODELS`):

| Lesson | Models                                            |
| ------ | ------------------------------------------------- |
| 6.1    | `llama3.2:3b` — chat                              |
| 6.2    | `qwen2.5:0.5b` (served), `llama3.2:3b` (judge)    |
| 6.3    | `qwen2.5:0.5b` (served), `llama3.2:3b` (judge)    |
| 6.4    | `llama3.2:3b` + `nomic-embed-text` (768-dim)      |
| 6.5    | `llama3.2:3b` — tool-capable per Kaizen allowlist |
| 6.6    | `llama3.2:3b`                                     |
| 6.7    | `llama3.2:3b`                                     |
| 6.8    | `llama3.2:3b`                                     |

`llama3.2:3b` is on Kaizen's `OLLAMA_TOOL_CAPABLE_FAMILIES` allowlist
(adapter line 43), so ReAct and multi-agent lessons (6.5, 6.6) get
function calling out of the box.

## Generator contract

`scripts/generate_selfcontained_notebook.py` injects a Cell 1 between
the pip-install Cell 0 and the inlined-helpers Cell 2 for every M6
notebook. The cell is idempotent: on Colab it installs Ollama if missing
and pulls the lesson's models; on local it verifies the daemon is up and
the models are present, raising with the exact fix command if not.

## Consequences

- **Student-side**: one-time `ollama pull llama3.2:3b nomic-embed-text
qwen2.5:0.5b` (~3GB). After that, every M6 notebook runs offline
  forever.
- **Colab cold-start**: ~5 min for the first cell of the first notebook
  in a new session (install + pull). Subsequent notebooks in the same
  session reuse the running daemon.
- **Cost**: $0 across all M6 lessons.
- **Ratchet against future regressions**: Redline 14 + the audit grep
  in its checklist (`grep -rn 'Delegate(' shared/mlfp06/ modules/mlfp06/`
  returns zero non-bootstrap hits) makes drifting back to OpenAI a
  detectable redline failure at `/redteam` time.

## Follow-up

- The two pre-existing direct-`Delegate(...)` sites in solutions
  (`ex_3/04_grpo_and_judge.py` and `ex_6/05_memory_and_security.py`)
  were migrated to `make_delegate` in this commit. Solutions ex_5
  dataclass configs (`03_structured_agent`, `04_critic_agent`) had
  their `LLM_PROVIDER` defaults flipped from `"openai"` to `"ollama"`
  with `base_url` added.
- Smoke-validated end-to-end: SST-2 zero-shot classifier scored 3/3
  against real Ollama (`qwen3:latest` as a stand-in for the user's
  not-yet-pulled `llama3.2:3b`), 1403 tokens, 5s avg latency.
- Open questions Q1 (`obs.attention` vs `obs.interp`) and Q3 (`report()`
  format) from the M6 holistic DISCOVERY remain unresolved — they are
  independent of the provider migration and can be revisited later.

## For Discussion

1. The bootstrap exposes `make_delegate` and `make_embedder` as the only
   sanctioned constructors and raises if a caller tries to override
   `api_key` / `base_url` / `budget_usd` / `adapter`. Is this constraint
   the right balance, or should we allow callers to inject a custom
   adapter for advanced lessons (e.g. a future "compare Ollama vs
   vLLM" exercise)?
2. The Colab bootstrap waits 30s for the daemon to come up. Empirically
   it answers within 2-3s; the 30s ceiling is generous to absorb
   first-time GPU initialization. Should the cell print progress
   feedback while waiting, or is the silent wait acceptable for the
   <5% of cases where it takes more than 5s?
3. We picked `llama3.2:3b` as the chat model because it's tool-capable
   AND fits T4. The next obvious upgrade is `qwen2.5:7b` (better
   reasoning, larger). Should the Foundation course be opinionated
   about the default and recommend an override for advanced students,
   or should we expose a `OLLAMA_PROFILE=heavyweight` switch?
