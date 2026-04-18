---
type: DISCOVERY
date: 2026-04-17
created_at: 2026-04-17T00:00:00Z
author: agent
session_id: m6-ollama-migration-2026-04-17
session_turn: 2
project: mlfp
topic: Kaizen TextDelta event uses .text not .delta_text — silent empty-string trap
phase: implement
tags: [m6, kaizen, ollama, event-handling, smoke-validation]
---

# DISCOVERY: Kaizen 0.9.2 TextDelta event exposes streamed chunk on `.text`, NOT `.delta_text`

## What was discovered

Authoritative source-read by the kaizen-specialist (early in this session)
asserted that Kaizen's `StreamEvent` has a `.delta_text` attribute carrying
each streamed chunk. The first version of `_ollama_bootstrap.run_delegate_text`
trusted that guidance and accumulated `getattr(event, "delta_text", None)`.

End-to-end smoke against a real Ollama daemon (qwen3:latest) returned
`text=''` after 270 seconds with `usage={'prompt_tokens': 0,
'completion_tokens': 0, 'total_tokens': 0}` — the call ran, the model
generated tokens, but the helper extracted nothing.

Direct introspection of the events emitted by `Delegate.run()` showed:

```
TextDelta(event_type='text_delta', timestamp=..., text='Hello')
TextDelta(event_type='text_delta', timestamp=..., text='!')
TextDelta(event_type='text_delta', timestamp=..., text=' How')
...
TurnComplete(event_type='turn_complete', timestamp=...,
             text='Hello! How can I assist you today?',
             usage={'prompt_tokens': 70, 'completion_tokens': 99,
                    'total_tokens': 169})
```

The streamed chunk lives on `.text` (not `.delta_text`). The terminal
`TurnComplete` event also carries the full final text on `.text` plus
the usage dict.

## Evidence

- `kaizen_agents` version installed: `0.9.2`
- `Delegate.__init__` signature accepts `adapter`, `base_url`, `api_key`,
  `budget_usd` (verified via inspect)
- Source files: `kaizen_agents/delegate/adapters/ollama_adapter.py`,
  `kaizen_agents/delegate/adapters/registry.py`, `delegate.py`
- Reproduction: 10-event run of `Delegate(model="qwen3:latest", ...)` —
  9 TextDelta events with `.text='Hello'`, `'!'`, `' How'`, ..., 1
  TurnComplete with the joined text + usage.

## What it caused

- 270-second silent run with empty result on the first end-to-end smoke
  attempt.
- Would have masked every M6 LLM call as "no tokens, no text" even though
  the model fired correctly. The `accuracy=0%` outcome would have been
  identical to the OpenAI-stub failure we just deleted (Redline 14 issue).

## Fix

`shared/mlfp06/_ollama_bootstrap.py::run_delegate_text` was updated to:

1. Accumulate `event.text` only when `event.event_type == "text_delta"`
2. Capture `event.text` AND `event.usage` from the
   `event.event_type == "turn_complete"` event as a fallback (in case the
   Ollama path emits the final text only on TurnComplete with no
   intervening TextDelta events).
3. Return the accumulated stream if non-empty, otherwise the
   TurnComplete final text.

After the fix, the same SST-2 smoke ran in 5s avg latency with
`accuracy=100% n=3 tokens=1403`.

## Why this matters

The kaizen-specialist's source-read was correct about the existence of
`StreamEvent` shapes — Kaizen has multiple event flavours and
`delta_text` exists on some of them in newer adapter implementations.
But the Ollama adapter in 0.9.2 uses the older `TextDelta` shape with
`.text`. Trusting an upstream-source narrative without **executing** the
event loop against a real adapter would have shipped 78 broken notebooks.

## Lessons

- For helpers that consume framework events, source-reads are necessary
  but not sufficient — **execute** the event loop against the real
  provider before declaring the helper done.
- Belt-and-braces field extraction (try `text_delta.text`, fall back to
  `turn_complete.text`) is the right trade-off for a helper meant to
  survive Kaizen API drift.

## For Discussion

1. Should the bootstrap helper log a WARN when it returns the
   TurnComplete fallback (i.e. when the adapter emitted no TextDelta
   events)? That signal would surface a future Kaizen change before it
   silently degraded the streaming UX in lessons that show token-by-token
   output.
2. The kaizen-specialist agent answered authoritatively from a source
   read but cited the wrong attribute. Should we add a Kaizen contract
   test in the course's CI that runs a 1-token smoke against a tiny
   Ollama model and asserts `.text` extraction works, so future
   kaizen-agents version bumps surface an event-shape change at a code
   gate rather than at a student's notebook?
3. The fix is a 6-line change in one helper but it was the difference
   between "M6 ships" and "M6 ships broken". How much of the M6 audit
   surface depends on `run_delegate_text` being correct? Probably 100%
   of LLM-calling lessons. Worth promoting it to a directly-tested unit
   in `tests/unit/test_mlfp06_bootstrap.py`?
