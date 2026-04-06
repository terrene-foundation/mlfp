# Chapter 4: SSE Transport

## Overview

Server-Sent Events (SSE) provide one-way real-time streaming from server to client over HTTP. Nexus uses SSE to stream EventBus events to browser dashboards, monitoring tools, and other clients. This chapter covers the `register_sse_endpoint()` function, the `EventBus` class, event publishing and subscription, and how SSE filtering works. SSE is the bridge between Nexus's internal event system and external consumers.

## Prerequisites

- Completed Chapters 1-3 (Nexus basics, HTTP/MCP transports)
- Understanding of HTTP streaming concepts
- Familiarity with async generators in Python

## Concepts

### What Are Server-Sent Events?

SSE is a W3C standard for one-way streaming from server to client over HTTP. Unlike WebSockets (bidirectional), SSE is server-to-client only, making it simpler and sufficient for monitoring dashboards and event feeds. The format is plain text:

```
data: {"event_type": "handler.called", "timestamp": "...", ...}\n\n
```

### The EventBus

Nexus's `EventBus` is the source of all events. It uses a `janus.Queue` to bridge synchronous publishers (any thread) and asynchronous consumers (SSE endpoint, subscribers). The bus maintains a bounded history (default 256 events) and supports filtered subscriptions.

### Why Is SSE Opt-In?

SSE requires an active EventBus dispatch loop and adds an HTTP endpoint. Not every application needs real-time event streaming, so SSE is registered explicitly via `register_sse_endpoint(app)`.

## Key API

| Method / Property                            | Parameters                                                      | Returns         | Description                                       |
| -------------------------------------------- | --------------------------------------------------------------- | --------------- | ------------------------------------------------- |
| `register_sse_endpoint(app)`                 | `app: Nexus`                                                    | `None`          | Add `GET /events/stream` to the HTTP transport    |
| `EventBus.sse_url()`                         | --                                                              | `str`           | Returns the canonical SSE path `"/events/stream"` |
| `EventBus(capacity)`                         | `capacity: int = 256`                                           | `EventBus`      | Create a bounded event bus                        |
| `bus.publish(event)`                         | `event: NexusEvent`                                             | `None`          | Publish an event (sync, thread-safe)              |
| `bus.publish_handler_registered(name)`       | `name: str`                                                     | `None`          | Convenience: publish a HANDLER_REGISTERED event   |
| `bus.subscribe()`                            | --                                                              | `asyncio.Queue` | Subscribe to all events                           |
| `bus.subscribe_filtered(predicate)`          | `predicate: Callable`                                           | `asyncio.Queue` | Subscribe to matching events only                 |
| `bus.subscriber_count`                       | --                                                              | `int`           | Number of active subscribers                      |
| `bus.get_history(event_type, limit)`         | `event_type: str`, `limit: int`                                 | `list[dict]`    | Read from bounded history deque                   |
| `NexusEvent(event_type, handler_name, data)` | `event_type: NexusEventType`, `handler_name: str`, `data: dict` | `NexusEvent`    | Create an event                                   |
| `event.to_dict()`                            | --                                                              | `dict`          | Serialize for JSON transport                      |
| `NexusEvent.from_dict(d)`                    | `d: dict`                                                       | `NexusEvent`    | Deserialize from dict                             |

## Code Walkthrough

### Step 1: Register the SSE Endpoint

```python
from nexus import Nexus, register_sse_endpoint

app = Nexus(api_port=8000, enable_durability=False)
register_sse_endpoint(app)
```

This adds `GET /events/stream` to the HTTP transport. The endpoint streams events from the EventBus in SSE format.

### Step 2: EventBus SSE URL

```python
from nexus.events import EventBus

sse_path = EventBus.sse_url()
assert sse_path == "/events/stream"
```

`sse_url()` returns the canonical path, matching the Rust SDK's `EventBus::sse_url()`.

### Step 3: Access the Internal EventBus

```python
bus = app._event_bus
assert isinstance(bus, EventBus)
```

Nexus creates an EventBus internally with a default capacity of 256 events.

### Step 4: Publish Events

```python
from nexus.events import NexusEvent, NexusEventType

event = NexusEvent(
    event_type=NexusEventType.HANDLER_CALLED,
    handler_name="greet",
    data={"user": "Alice"},
)
bus.publish(event)

# Convenience method
bus.publish_handler_registered("my_handler")
```

`publish()` is thread-safe and non-blocking. Events enter the janus sync queue and are dispatched to subscribers once `bus.start()` launches the async dispatch loop.

### Step 5: Event Serialization

```python
event_dict = event.to_dict()
assert event_dict["event_type"] == "handler.called"
assert event_dict["handler_name"] == "greet"
assert event_dict["data"]["user"] == "Alice"

restored = NexusEvent.from_dict(event_dict)
assert restored.event_type == event.event_type
```

`to_dict()` produces JSON-serializable output for SSE transport. `from_dict()` reconstructs the event for round-trip serialization.

### Step 6: Subscribe to Events

```python
import asyncio

q_all = bus.subscribe()
assert isinstance(q_all, asyncio.Queue)

q_filtered = bus.subscribe_filtered(
    lambda evt: evt.event_type == NexusEventType.HANDLER_CALLED
)
assert isinstance(q_filtered, asyncio.Queue)

assert bus.subscriber_count == 2
```

`subscribe()` returns a queue that receives all events. `subscribe_filtered()` takes a predicate function and only delivers matching events.

### Step 7: Query Event History

```python
history = bus.get_history()
assert isinstance(history, list)

history_filtered = bus.get_history(event_type="handler.called", limit=10)
```

`get_history()` reads from the internal bounded deque with optional filters. The history is populated by the dispatch loop in production.

## Common Mistakes

| Mistake                                         | Problem                                                         | Fix                                                      |
| ----------------------------------------------- | --------------------------------------------------------------- | -------------------------------------------------------- |
| Expecting history before `bus.start()`          | Events are in the janus queue, not history, until dispatch runs | Run the dispatch loop or check the queue directly        |
| Forgetting to call `register_sse_endpoint(app)` | No SSE endpoint is created                                      | SSE is opt-in; register explicitly                       |
| Not handling SSE keepalives                     | Client may disconnect after 15s silence                         | The SSE generator sends keepalive comments automatically |
| Creating EventBus with very small capacity      | Events evicted before consumers read them                       | Use capacity >= 256 for production                       |

## Exercises

1. **Event Round-Trip**: Create five `NexusEvent` instances with different event types. Serialize each with `to_dict()`, then restore with `from_dict()`. Verify all fields survive the round trip.

2. **Filtered Subscriptions**: Create an EventBus and add three subscribers: one unfiltered, one for `HANDLER_CALLED` events only, and one for `HANDLER_ERROR` events only. Verify the subscriber count is 3.

3. **History Capacity**: Create an EventBus with `capacity=5`. Directly append 10 events to `_history`. Verify that only the 5 most recent events survive, demonstrating the bounded deque behavior.

## Key Takeaways

- `register_sse_endpoint(app)` adds `GET /events/stream` for real-time event streaming.
- `EventBus.sse_url()` returns the canonical SSE path `"/events/stream"`.
- Events are published via `bus.publish()` (sync, thread-safe).
- SSE format: `data: {json}\n\n` with 15-second keepalive comments.
- Optional `?event_type=` query parameter enables server-side filtering.
- EventBus stores bounded history (default 256 events) with LRU eviction.

## Next Chapter

[Chapter 5: Middleware](05_middleware.md) -- Add middleware to Nexus for CORS, security headers, CSRF protection, and response caching.
