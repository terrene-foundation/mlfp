# Chapter 7: Event Bus

## Overview

The EventBus is Nexus's internal event system for event-driven architectures. It provides thread-safe publishing, async subscriptions with optional filtering, bounded history with LRU eviction, and round-trip serialization. This chapter provides a deep dive into the `NexusEventType` enum, `NexusEvent` dataclass, the `EventBus` lifecycle (start/stop), and how Nexus integrates the EventBus internally.

## Prerequisites

- Completed Chapters 1-6 (especially Chapter 4: SSE Transport for EventBus basics)
- Understanding of async/await and asyncio.Queue
- Familiarity with the producer-consumer pattern

## Concepts

### Event Types

Events are typed via the `NexusEventType` enum. Each value is a dotted string matching the Rust SDK's event types:

| Event Type           | Value                  | When Published                  |
| -------------------- | ---------------------- | ------------------------------- |
| `HANDLER_REGISTERED` | `"handler.registered"` | A new handler is added          |
| `HANDLER_CALLED`     | `"handler.called"`     | A handler begins execution      |
| `HANDLER_COMPLETED`  | `"handler.completed"`  | A handler finishes successfully |
| `HANDLER_ERROR`      | `"handler.error"`      | A handler throws an error       |
| `HEALTH_CHECK`       | `"health.check"`       | A health check is performed     |
| `CUSTOM`             | `"custom"`             | Application-defined events      |

### Janus Queue Bridge

The EventBus uses a `janus.Queue` to bridge synchronous publishers (any thread) and asynchronous consumers (SSE endpoint, subscribers). `publish()` writes to the sync side; the dispatch loop reads from the async side and fans events to subscribers and history.

### Bounded History

The internal history is a `collections.deque` with `maxlen=capacity`. When full, the oldest events are automatically evicted. This prevents unbounded memory growth in long-running services.

## Key API

| Method / Property                      | Parameters                                         | Returns         | Description                       |
| -------------------------------------- | -------------------------------------------------- | --------------- | --------------------------------- |
| `EventBus(capacity)`                   | `capacity: int = 256`                              | `EventBus`      | Create a bounded event bus        |
| `bus.publish(event)`                   | `event: NexusEvent`                                | `None`          | Publish event (sync, thread-safe) |
| `bus.publish_handler_registered(name)` | `name: str`                                        | `None`          | Convenience publish               |
| `bus.subscribe()`                      | --                                                 | `asyncio.Queue` | Subscribe to all events           |
| `bus.subscribe_filtered(predicate)`    | `predicate: Callable[[NexusEvent], bool]`          | `asyncio.Queue` | Subscribe with filter             |
| `bus.subscriber_count`                 | --                                                 | `int`           | Active subscriber count           |
| `bus.get_history(event_type, limit)`   | `event_type: str`, `limit: int`                    | `list[dict]`    | Query bounded history             |
| `bus.start()`                          | --                                                 | `None` (async)  | Start the dispatch loop           |
| `bus.stop()`                           | --                                                 | `None` (async)  | Stop the dispatch loop            |
| `EventBus.sse_url()`                   | --                                                 | `str`           | Returns `"/events/stream"`        |
| `NexusEvent(...)`                      | `event_type`, `handler_name`, `request_id`, `data` | `NexusEvent`    | Create an event                   |
| `event.to_dict()`                      | --                                                 | `dict`          | Serialize for JSON                |
| `NexusEvent.from_dict(d)`              | `d: dict`                                          | `NexusEvent`    | Deserialize                       |

## Code Walkthrough

### Step 1: NexusEventType Enum

```python
from nexus.events import EventBus, NexusEvent, NexusEventType

assert NexusEventType.HANDLER_REGISTERED.value == "handler.registered"
assert NexusEventType.HANDLER_CALLED.value == "handler.called"
assert NexusEventType.HANDLER_COMPLETED.value == "handler.completed"
assert NexusEventType.HANDLER_ERROR.value == "handler.error"
assert NexusEventType.HEALTH_CHECK.value == "health.check"
assert NexusEventType.CUSTOM.value == "custom"
```

### Step 2: Create and Serialize Events

```python
from datetime import datetime

event = NexusEvent(
    event_type=NexusEventType.HANDLER_CALLED,
    handler_name="greet",
    request_id="req-001",
    data={"input_name": "Alice"},
)

assert isinstance(event.timestamp, datetime)

event_dict = event.to_dict()
assert event_dict["event_type"] == "handler.called"
assert isinstance(event_dict["timestamp"], str)  # ISO format

restored = NexusEvent.from_dict(event_dict)
assert restored.event_type == event.event_type
assert restored.data == event.data
```

### Step 3: Publish Events

```python
bus = EventBus(capacity=128)

bus.publish(NexusEvent(
    event_type=NexusEventType.HANDLER_CALLED,
    handler_name="add",
    data={"a": 1, "b": 2},
))

bus.publish(NexusEvent(
    event_type=NexusEventType.HANDLER_COMPLETED,
    handler_name="add",
    data={"result": 3},
))

bus.publish_handler_registered("new_handler")
```

### Step 4: Run the Dispatch Loop

```python
import asyncio

async def run_bus_briefly():
    await bus.start()
    await asyncio.sleep(0.1)
    await bus.stop()

asyncio.run(run_bus_briefly())
```

After the dispatch loop runs, events move from the janus queue into history and are fanned to subscribers.

### Step 5: Query History

```python
all_history = bus.get_history()
called_only = bus.get_history(event_type="handler.called")
recent_two = bus.get_history(limit=2)
```

### Step 6: Subscribe

```python
q_all = bus.subscribe()
assert bus.subscriber_count == 1

q_errors = bus.subscribe_filtered(
    lambda evt: evt.event_type == NexusEventType.HANDLER_ERROR
)
assert bus.subscriber_count == 2
```

### Step 7: Bounded History Demonstration

```python
small_bus = EventBus(capacity=3)

for i in range(5):
    small_bus._history.append(
        NexusEvent(event_type=NexusEventType.CUSTOM, data={"index": i})
    )

history = small_bus.get_history()
assert len(history) == 3
assert history[0]["data"]["index"] == 2  # Oldest surviving
assert history[2]["data"]["index"] == 4  # Most recent
```

### Step 8: EventBus Inside Nexus

```python
from nexus import Nexus

app = Nexus(api_port=8000, enable_durability=False)
assert isinstance(app._event_bus, EventBus)

@app.handler("multiply", description="Multiply two numbers")
async def multiply(x: int, y: int = 1) -> dict:
    return {"result": x * y}

handler_def = app._registry.get_handler("multiply")
assert handler_def.name == "multiply"
```

Nexus creates an EventBus internally (capacity=256). The HandlerRegistry publishes `HANDLER_REGISTERED` events automatically.

## Common Mistakes

| Mistake                                          | Problem                          | Fix                                         |
| ------------------------------------------------ | -------------------------------- | ------------------------------------------- |
| Querying history before dispatch loop runs       | History is empty                 | Call `bus.start()` to drain the janus queue |
| Creating EventBus with capacity=1                | All but the latest event evicted | Use capacity >= 128 for production          |
| Forgetting `await` on `bus.start()`/`bus.stop()` | Coroutine never runs             | Both are async methods                      |
| Publishing after `bus.stop()`                    | Events lost                      | Ensure the bus is running during publish    |

## Exercises

1. **Event Lifecycle**: Create an EventBus, publish events of each type (HANDLER_REGISTERED, HANDLER_CALLED, HANDLER_COMPLETED, HANDLER_ERROR, CUSTOM). Run the dispatch loop briefly, then query history with type filters.

2. **Subscriber Fan-Out**: Create three filtered subscribers -- one for errors, one for handler calls, one for completions. Publish mixed events and verify each subscriber receives only matching events (requires running the dispatch loop).

3. **Capacity Testing**: Create buses with capacity 1, 5, and 100. Add 50 events to each. Verify the history length matches the capacity.

## Key Takeaways

- `NexusEventType` is a typed enum with dotted string values matching the Rust SDK.
- `NexusEvent` provides `to_dict()`/`from_dict()` for round-trip serialization.
- `EventBus(capacity=256)` is bounded, thread-safe, and uses janus for sync/async bridging.
- `publish()` is sync and thread-safe; works before the async dispatch loop starts.
- `subscribe()` and `subscribe_filtered()` return async queues for event consumption.
- `get_history()` reads from the bounded deque with optional type and limit filters.
- Nexus creates an EventBus internally; handlers auto-publish registration events.

## Next Chapter

[Chapter 8: OpenAPI](08_openapi.md) -- Auto-generate OpenAPI documentation from registered handlers and workflows.
