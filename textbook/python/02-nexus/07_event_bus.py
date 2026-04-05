# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Nexus / Event Bus
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Use the EventBus for event-driven patterns
# LEVEL: Advanced
# PARITY: Full — Rust has EventBus with same NexusEventType enum
# VALIDATES: EventBus, NexusEvent, NexusEventType, publish, subscribe,
#            subscribe_filtered, get_history
#
# Run: uv run python textbook/python/02-nexus/07_event_bus.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from nexus.events import EventBus, NexusEvent, NexusEventType

# ── 1. NexusEventType enum ─────────────────────────────────────────
# Events are typed using the NexusEventType enum. Each value is a
# dotted string matching kailash-rs event types.

assert NexusEventType.HANDLER_REGISTERED.value == "handler.registered"
assert NexusEventType.HANDLER_CALLED.value == "handler.called"
assert NexusEventType.HANDLER_COMPLETED.value == "handler.completed"
assert NexusEventType.HANDLER_ERROR.value == "handler.error"
assert NexusEventType.HEALTH_CHECK.value == "health.check"
assert NexusEventType.CUSTOM.value == "custom"

# ── 2. NexusEvent dataclass ───────────────────────────────────────
# NexusEvent is a dataclass with event_type, timestamp, data,
# handler_name, and request_id. It mirrors kailash-rs NexusEvent.

event = NexusEvent(
    event_type=NexusEventType.HANDLER_CALLED,
    handler_name="greet",
    request_id="req-001",
    data={"input_name": "Alice"},
)

assert event.event_type == NexusEventType.HANDLER_CALLED
assert event.handler_name == "greet"
assert event.request_id == "req-001"
assert event.data["input_name"] == "Alice"
assert isinstance(event.timestamp, datetime)

# ── 3. Event serialization ────────────────────────────────────────
# to_dict() serializes an event for JSON transport (SSE, logging).
# from_dict() reconstructs it from a dict.

event_dict = event.to_dict()

assert event_dict["event_type"] == "handler.called"
assert event_dict["handler_name"] == "greet"
assert event_dict["request_id"] == "req-001"
assert event_dict["data"]["input_name"] == "Alice"
assert isinstance(event_dict["timestamp"], str)  # ISO format

# Round-trip
restored = NexusEvent.from_dict(event_dict)
assert restored.event_type == event.event_type
assert restored.handler_name == event.handler_name
assert restored.data == event.data

# ── 4. Create an EventBus ─────────────────────────────────────────
# EventBus uses a janus.Queue to bridge sync publishers and async
# consumers. The capacity parameter sets the bounded buffer size.

bus = EventBus(capacity=128)

assert bus.subscriber_count == 0

# ── 5. Publish events (sync, thread-safe) ──────────────────────────
# publish() is non-blocking and thread-safe. It uses a janus.Queue
# to bridge sync producers and async consumers. Events enter the
# queue immediately and are dispatched to subscribers + history
# once bus.start() launches the async dispatch loop.

bus.publish(
    NexusEvent(
        event_type=NexusEventType.HANDLER_CALLED,
        handler_name="add",
        data={"a": 1, "b": 2},
    )
)

bus.publish(
    NexusEvent(
        event_type=NexusEventType.HANDLER_COMPLETED,
        handler_name="add",
        data={"result": 3},
    )
)

bus.publish(
    NexusEvent(
        event_type=NexusEventType.HANDLER_ERROR,
        handler_name="divide",
        data={"error": "division by zero"},
    )
)

# ── 6. Convenience method ─────────────────────────────────────────
# publish_handler_registered() creates and publishes a
# HANDLER_REGISTERED event in one call.

bus.publish_handler_registered("new_handler")

# ── 7. Event history with dispatch loop ───────────────────────────
# get_history() reads from the internal bounded deque. Events only
# appear in history after the dispatch loop (bus.start()) moves
# them from the janus queue. We can verify this by running the
# loop briefly with asyncio.


async def run_bus_briefly():
    """Start the bus, let it dispatch, then stop."""
    await bus.start()
    # Give the dispatch loop a moment to drain the janus queue
    await asyncio.sleep(0.1)
    await bus.stop()


asyncio.run(run_bus_briefly())

# After the dispatch loop ran, events are now in history
all_history = bus.get_history()
assert len(all_history) == 4

# Filter by event type
called_only = bus.get_history(event_type="handler.called")
assert all(e["type"] == "handler.called" for e in called_only)
assert len(called_only) == 1

error_only = bus.get_history(event_type="handler.error")
assert len(error_only) == 1
assert error_only[0]["data"]["error"] == "division by zero"

# Limit results (most recent N)
recent_two = bus.get_history(limit=2)
assert len(recent_two) == 2

# ── 8. Subscribe (async consumers) ────────────────────────────────
# subscribe() returns an asyncio.Queue that receives all events
# once the dispatch loop is running. Subscribing before start()
# is safe; events flow once the loop begins.

q_all = bus.subscribe()
assert isinstance(q_all, asyncio.Queue)
assert bus.subscriber_count == 1

# ── 9. Filtered subscriptions ─────────────────────────────────────
# subscribe_filtered() takes a predicate function and only delivers
# events for which the predicate returns True.

q_errors = bus.subscribe_filtered(
    lambda evt: evt.event_type == NexusEventType.HANDLER_ERROR
)
assert bus.subscriber_count == 2

q_handler_events = bus.subscribe_filtered(
    lambda evt: evt.event_type
    in (
        NexusEventType.HANDLER_CALLED,
        NexusEventType.HANDLER_COMPLETED,
    )
)
assert bus.subscriber_count == 3

# ── 10. History deque is bounded ───────────────────────────────────
# The internal history deque respects the capacity limit (maxlen).
# When full, oldest events are evicted automatically.
# We verify by appending directly to _history (simulating what the
# dispatch loop does) to demonstrate the bounded behavior.

small_bus = EventBus(capacity=3)

for i in range(5):
    # Append directly to history to demonstrate bounded deque behavior.
    # In production, the dispatch loop does this after reading from
    # the janus queue.
    small_bus._history.append(
        NexusEvent(
            event_type=NexusEventType.CUSTOM,
            data={"index": i},
        )
    )

# Only the 3 most recent events survive (deque maxlen=3)
history = small_bus.get_history()
assert len(history) == 3
assert history[0]["data"]["index"] == 2  # Oldest surviving
assert history[2]["data"]["index"] == 4  # Most recent

# ── 11. EventBus inside Nexus ─────────────────────────────────────
# Nexus creates an EventBus internally (capacity=256). The
# HandlerRegistry publishes HANDLER_REGISTERED events automatically
# when handlers are added via app.handler() or app.register_handler().
# In production with a running server, the dispatch loop moves these
# events to history and fans them out to subscribers.

from nexus import Nexus

app = Nexus(api_port=8000, enable_durability=False)

# The internal EventBus exists from construction
assert isinstance(app._event_bus, EventBus)


# Register a handler -- this publishes a HANDLER_REGISTERED event
# into the janus queue via the HandlerRegistry
@app.handler("multiply", description="Multiply two numbers")
async def multiply(x: int, y: int = 1) -> dict:
    return {"result": x * y}


# The handler is registered in the registry
handler_def = app._registry.get_handler("multiply")
assert handler_def is not None
assert handler_def.name == "multiply"
assert handler_def.description == "Multiply two numbers"

# ── 12. SSE URL ────────────────────────────────────────────────────
# EventBus.sse_url() returns the canonical SSE endpoint path.
# This is used by register_sse_endpoint() to wire up streaming.

assert EventBus.sse_url() == "/events/stream"

# ── 13. Key concepts ──────────────────────────────────────────────
# - NexusEventType: typed enum (handler.registered, .called, .completed, .error, custom)
# - NexusEvent: dataclass with to_dict()/from_dict() for serialization
# - EventBus(capacity=256): bounded, thread-safe event bus
# - publish(): sync, thread-safe, works before async loop starts
# - subscribe(): async queue for all events
# - subscribe_filtered(predicate): async queue for matching events only
# - get_history(): read bounded deque with optional filters
# - Nexus creates EventBus internally; handlers auto-publish events
# - NOTE: We don't call bus.start() or app.start() because they block

print("PASS: 02-nexus/07_event_bus")
