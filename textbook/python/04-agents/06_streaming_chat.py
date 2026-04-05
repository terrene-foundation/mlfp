# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Agents / Streaming Chat Agent
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build a StreamingChatAgent for real-time responses
# LEVEL: Intermediate
# PARITY: Python-only — no Rust equivalent
# VALIDATES: StreamingChatAgent, StreamingChatConfig, ChatSignature,
#            StreamingStrategy, stream() method
#
# Run: uv run python textbook/python/04-agents/06_streaming_chat.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os

from kaizen_agents.agents.specialized.streaming_chat import (
    StreamingChatAgent,
    StreamingChatConfig,
    ChatSignature,
)
from kaizen.core.base_agent import BaseAgent
from kaizen.strategies.streaming import StreamingStrategy

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")

# ── 1. Streaming pattern ──────────────────────────────────────────────
# StreamingChatAgent provides real-time token-by-token output.
# Instead of waiting for the complete response, the consumer receives
# tokens as they are generated — creating a "typing" effect.
#
# Two interfaces:
#   .stream(message) -> async iterator of tokens (streaming mode)
#   .run(message=...) -> dict (non-streaming mode, standard BaseAgent)

# ── 2. ChatSignature — minimal chat I/O ───────────────────────────────
# The simplest possible signature: one input, one output.

assert "message" in ChatSignature._signature_inputs
assert "response" in ChatSignature._signature_outputs

message_field = ChatSignature._signature_inputs["message"]
assert message_field.required is True

# ── 3. StreamingChatConfig — streaming-specific settings ──────────────

config = StreamingChatConfig()

assert config.streaming is True, "Streaming enabled by default"
assert config.chunk_size == 1, "Token-by-token streaming by default"
assert config.temperature == 0.7, "Higher temperature for conversational style"
assert config.max_tokens == 500

# Chunk size controls throughput vs latency:
#   chunk_size=1  -> token-by-token (lowest latency, more overhead)
#   chunk_size=5  -> 5 tokens at a time (balanced)
#   chunk_size=20 -> batch streaming (higher throughput, more latency)

# ── 4. StreamingChatAgent with streaming enabled ──────────────────────

agent = StreamingChatAgent(
    llm_provider="mock",
    model=model,
    streaming=True,
    chunk_size=1,
)

assert isinstance(agent, StreamingChatAgent)
assert isinstance(agent, BaseAgent)
assert isinstance(
    agent.strategy, StreamingStrategy
), "Streaming mode uses StreamingStrategy"
assert agent.chat_config.streaming is True

# ── 5. StreamingChatAgent with streaming disabled ─────────────────────
# When streaming=False, the agent falls back to standard BaseAgent
# execution (AsyncSingleShotStrategy). The .run() method works
# normally; .stream() raises ValueError.

sync_agent = StreamingChatAgent(
    llm_provider="mock",
    model=model,
    streaming=False,
)

assert sync_agent.chat_config.streaming is False
assert not isinstance(
    sync_agent.strategy, StreamingStrategy
), "Non-streaming mode uses default strategy"

# stream() requires StreamingStrategy
try:
    import asyncio

    asyncio.run(sync_agent.stream("test").__anext__())
    assert False, "stream() should raise ValueError without StreamingStrategy"
except ValueError as e:
    assert "StreamingStrategy" in str(e)
except StopAsyncIteration:
    pass  # Also acceptable if strategy validates differently

# ── 6. Streaming consumption pattern ──────────────────────────────────
# The standard pattern for consuming streaming output:
#
#   import asyncio
#
#   agent = StreamingChatAgent()
#
#   async def chat():
#       async for token in agent.stream("What is Python?"):
#           print(token, end="", flush=True)
#       print()  # Newline after streaming completes
#
#   asyncio.run(chat())
#
# Key points:
# - stream() is async (must be awaited in async context)
# - Each yield is a string token or chunk
# - flush=True ensures immediate display
# - The loop ends when the response is complete
#
# NOTE: We do not call stream() with real messages here because
# it requires an LLM API key.

# ── 7. Non-streaming fallback ─────────────────────────────────────────
# .run() always works, regardless of streaming setting.
# For non-streaming, it returns the standard BaseAgent dict:
#
#   result = agent.run(message="What is AI?")
#   print(result["response"])

# ── 8. Custom chunk size ──────────────────────────────────────────────
# Larger chunk sizes reduce overhead at the cost of latency.

batch_agent = StreamingChatAgent(
    llm_provider="mock",
    model=model,
    streaming=True,
    chunk_size=10,
)

assert batch_agent.chat_config.chunk_size == 10
assert isinstance(batch_agent.strategy, StreamingStrategy)

# ── 9. Environment variable configuration ─────────────────────────────
# All settings can be configured via KAIZEN_* environment variables:
#   KAIZEN_STREAMING=true       -> streaming enabled
#   KAIZEN_CHUNK_SIZE=5         -> 5 tokens per chunk
#   KAIZEN_LLM_PROVIDER=openai  -> provider selection
#   KAIZEN_MODEL=gpt-4          -> model selection
#
# The config reads these at instantiation time.

# ── 10. Node metadata ─────────────────────────────────────────────────

assert StreamingChatAgent.metadata.name == "StreamingChatAgent"
assert "streaming" in StreamingChatAgent.metadata.tags
assert "chat" in StreamingChatAgent.metadata.tags
assert "real-time" in StreamingChatAgent.metadata.tags

print("PASS: 04-agents/06_streaming_chat")
