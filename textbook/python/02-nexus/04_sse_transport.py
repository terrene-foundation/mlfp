# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Nexus / SSE Transport
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Configure SSE (Server-Sent Events) transport for real-time
#            event streaming from the Nexus EventBus
# LEVEL: Intermediate
# PARITY: Python-only (Rust has CLI channel instead)
# VALIDATES: register_sse_endpoint, EventBus integration, SSE streaming
#
# Run: uv run python textbook/python/02-nexus/04_sse_transport.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from nexus import Nexus, register_sse_endpoint
from nexus.events import EventBus, NexusEvent, NexusEventType

# ── 1. SSE overview ────────────────────────────────────────────────
# Server-Sent Events (SSE) provide one-way real-time streaming from
# server to client over HTTP. Nexus uses SSE to stream EventBus
# events to browser dashboards, monitoring tools, and other clients.
#
# The SSE endpoint is registered at GET /events/stream and supports
# an optional ?event_type= query parameter for filtering.

# ── 2. The register_sse_endpoint function ──────────────────────────
# SSE is opt-in. Call register_sse_endpoint(app) to add the
# GET /events/stream route to the Nexus HTTP transport.

app = Nexus(
    api_port=8000,
    enable_durability=False,
)

# Register the SSE endpoint
register_sse_endpoint(app)

# The endpoint is now registered on the HTTP transport.
# It will be available at GET /events/stream when the server starts.

# ── 3. EventBus SSE URL ───────────────────────────────────────────
# The EventBus class has a static method that returns the canonical
# SSE endpoint path. This matches the kailash-rs EventBus::sse_url()
# interface for cross-SDK parity.

sse_path = EventBus.sse_url()
assert sse_path == "/events/stream"

# ── 4. EventBus basics ────────────────────────────────────────────
# The EventBus is the source of events for the SSE endpoint.
# Nexus creates an EventBus internally (app._event_bus) with a
# bounded capacity (default 256 events).

bus = app._event_bus
assert isinstance(bus, EventBus)

# ── 5. Publishing events ──────────────────────────────────────────
# Events can be published to the bus from any thread (thread-safe).
# publish() uses a janus.Queue to bridge sync and async contexts.
# Events enter the janus queue and are dispatched to subscribers
# (and stored in history) once the async dispatch loop starts via
# bus.start(). Without the loop, events queue up for later delivery.

event = NexusEvent(
    event_type=NexusEventType.HANDLER_CALLED,
    handler_name="greet",
    data={"user": "Alice"},
)

bus.publish(event)

# The event is queued in the janus sync queue for dispatch.
# In production, bus.start() runs the dispatch loop which moves
# events from the queue into history and fans them to subscribers.

# ── 6. Convenience publish methods ────────────────────────────────
# EventBus provides convenience methods for common event types.
# publish_handler_registered() is called automatically by the
# HandlerRegistry when new handlers are added.

bus.publish_handler_registered("my_handler")

# ── 7. NexusEvent serialization ──────────────────────────────────
# Events serialize to dicts for JSON transport (SSE, logging).
# to_dict() and from_dict() provide round-trip serialization.

event_dict = event.to_dict()
assert event_dict["event_type"] == "handler.called"
assert event_dict["handler_name"] == "greet"
assert event_dict["data"]["user"] == "Alice"

restored = NexusEvent.from_dict(event_dict)
assert restored.event_type == event.event_type
assert restored.handler_name == event.handler_name

# ── 8. Event history API ──────────────────────────────────────────
# get_history() reads from the internal bounded deque with optional
# filters by event_type, session_id, and limit. The history is
# populated by the dispatch loop (bus.start()) in production.
# Here we verify the API shape.

history = bus.get_history()
assert isinstance(history, list)

history_filtered = bus.get_history(event_type="handler.called", limit=10)
assert isinstance(history_filtered, list)

# ── 8. SSE streaming format ───────────────────────────────────────
# When a client connects to GET /events/stream, the SSE generator
# yields events in the standard SSE format:
#
#   data: {"event_type": "handler.called", "timestamp": "...", ...}\n\n
#
# If no events arrive within 15 seconds, a keepalive comment is sent:
#
#   : keepalive\n\n
#
# The response headers are:
#   Content-Type: text/event-stream
#   Cache-Control: no-cache
#   Connection: keep-alive
#   X-Accel-Buffering: no  (for nginx proxies)

# ── 9. Filtered SSE subscriptions ─────────────────────────────────
# Clients can filter by event type using the query parameter:
#   GET /events/stream?event_type=handler.called
#
# The SSE generator uses EventBus.subscribe_filtered() with a
# predicate that matches the event_type value.

# Verify subscribe and subscribe_filtered exist on EventBus
import asyncio

# subscribe() returns an asyncio.Queue for all events
q_all = bus.subscribe()
assert isinstance(q_all, asyncio.Queue)

# subscribe_filtered() takes a predicate function
q_filtered = bus.subscribe_filtered(
    lambda evt: evt.event_type == NexusEventType.HANDLER_CALLED
)
assert isinstance(q_filtered, asyncio.Queue)

# Track subscriber count
assert bus.subscriber_count == 2  # one unfiltered + one filtered

# ── 10. Key concepts ──────────────────────────────────────────────
# - register_sse_endpoint(app) adds GET /events/stream
# - EventBus.sse_url() returns the canonical path "/events/stream"
# - Events are published via bus.publish(NexusEvent(...))
# - SSE format: "data: {json}\n\n" with 15s keepalive
# - Optional ?event_type= query param for server-side filtering
# - EventBus stores bounded history (default 256 events)
# - NOTE: We don't call app.start() because it blocks

print("PASS: 02-nexus/04_sse_transport")
