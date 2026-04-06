# Chapter 9: Probes

## Overview

Kubernetes probes determine whether a container is alive, ready to serve traffic, and finished starting up. Nexus provides a `ProbeManager` that models the Kubernetes probe lifecycle with thread-safe, atomic state transitions. This chapter covers `ProbeState`, the state machine transitions, liveness/readiness/startup checks, custom readiness callbacks, and the `ProbeResponse` serialization format.

## Prerequisites

- Completed Chapters 1-8
- Understanding of Kubernetes probe concepts (liveness, readiness, startup)
- Familiarity with state machines

## Concepts

### Probe State Machine

ProbeState models the Kubernetes probe lifecycle with monotonic transitions:

```
STARTING  -->  READY  -->  DRAINING
    |            |            |
    +-----+------+-----+------+
          |            |
          v            v
        FAILED  <------+
```

- **STARTING**: Container is booting. Liveness passes, readiness fails.
- **READY**: Accepting traffic. All three probes pass.
- **DRAINING**: Graceful shutdown. Liveness passes, readiness fails.
- **FAILED**: Terminal state. Only `reset()` can recover.

### Three Probe Types

| Probe     | Endpoint   | Passes When            | Purpose                             |
| --------- | ---------- | ---------------------- | ----------------------------------- |
| Liveness  | `/healthz` | Not FAILED             | Should the container be restarted?  |
| Readiness | `/readyz`  | READY + callbacks pass | Should traffic be sent to this pod? |
| Startup   | `/startup` | Past STARTING          | Has the initial boot completed?     |

### Readiness Callbacks

Additional checks can be registered as callbacks (e.g., database connectivity, model loading). All callbacks must return `True` for readiness to pass. Failed callbacks are reported by name in the response details.

## Key API

| Method / Property                | Parameters                            | Returns         | Description                                 |
| -------------------------------- | ------------------------------------- | --------------- | ------------------------------------------- |
| `ProbeManager()`                 | --                                    | `ProbeManager`  | Create a probe manager (starts in STARTING) |
| `probes.state`                   | --                                    | `ProbeState`    | Current state                               |
| `probes.is_alive`                | --                                    | `bool`          | True unless FAILED                          |
| `probes.is_ready`                | --                                    | `bool`          | True only in READY                          |
| `probes.is_started`              | --                                    | `bool`          | True once past STARTING                     |
| `probes.mark_ready()`            | --                                    | `bool`          | Transition to READY                         |
| `probes.mark_draining()`         | --                                    | `bool`          | Transition to DRAINING                      |
| `probes.mark_failed(reason)`     | `reason: str`                         | `bool`          | Transition to FAILED                        |
| `probes.reset()`                 | --                                    | `None`          | Return to STARTING                          |
| `probes.check_liveness()`        | --                                    | `ProbeResponse` | Liveness check                              |
| `probes.check_readiness()`       | --                                    | `ProbeResponse` | Readiness check                             |
| `probes.check_startup()`         | --                                    | `ProbeResponse` | Startup check                               |
| `probes.add_readiness_check(cb)` | `cb: Callable[[], bool]`              | `None`          | Add custom readiness check                  |
| `probes.set_workflow_count(n)`   | `n: int`                              | `None`          | Track registered workflows                  |
| `ProbeResponse`                  | `.status`, `.http_status`, `.details` | dataclass       | Probe response                              |
| `response.to_dict()`             | --                                    | `dict`          | Serialize for JSON                          |

## Code Walkthrough

### Step 1: ProbeState Enum

```python
from nexus.probes import ProbeManager, ProbeResponse, ProbeState

assert ProbeState.STARTING.value == "starting"
assert ProbeState.READY.value == "ready"
assert ProbeState.DRAINING.value == "draining"
assert ProbeState.FAILED.value == "failed"
```

### Step 2: Initial State

```python
probes = ProbeManager()

assert probes.state == ProbeState.STARTING
assert probes.is_alive is True
assert probes.is_ready is False
assert probes.is_started is False
```

### Step 3: Liveness Check (STARTING)

```python
liveness = probes.check_liveness()
assert liveness.status == "ok"
assert liveness.http_status == 200
assert "uptime_seconds" in liveness.details
```

Even while starting, the container is alive (should not be restarted).

### Step 4: Startup Check (STARTING)

```python
startup = probes.check_startup()
assert startup.status == "starting"
assert startup.http_status == 503
```

The startup probe fails until the application transitions past STARTING.

### Step 5: Transition to READY

```python
probes.mark_ready()

assert probes.state == ProbeState.READY
assert probes.check_liveness().http_status == 200
assert probes.check_readiness().http_status == 200
assert probes.check_startup().http_status == 200
```

All three probes now return 200.

### Step 6: Custom Readiness Checks

```python
def check_database_connection() -> bool:
    return True

def check_model_loaded() -> bool:
    return True

probes.add_readiness_check(check_database_connection)
probes.add_readiness_check(check_model_loaded)

assert probes.check_readiness().http_status == 200
```

### Step 7: Failing Readiness Callback

```python
probes2 = ProbeManager()
probes2.mark_ready()

def always_fails() -> bool:
    return False

probes2.add_readiness_check(always_fails)

readiness = probes2.check_readiness()
assert readiness.http_status == 503
assert "always_fails" in readiness.details["failed_checks"]
```

### Step 8: DRAINING State

```python
probes.mark_draining()

assert probes.is_alive is True   # Don't restart
assert probes.is_ready is False  # Stop sending traffic
assert probes.check_liveness().http_status == 200
assert probes.check_readiness().http_status == 503
```

### Step 9: FAILED State

```python
probes3 = ProbeManager()
probes3.mark_ready()
probes3.mark_failed(reason="Out of memory")

assert probes3.is_alive is False
liveness = probes3.check_liveness()
assert liveness.http_status == 503
assert liveness.details["reason"] == "Out of memory"
```

### Step 10: Invalid Transitions

```python
probes4 = ProbeManager()

# Can't skip STARTING -> DRAINING
assert probes4.mark_draining() is False
assert probes4.state == ProbeState.STARTING

# FAILED is terminal
probes4.mark_failed(reason="crash")
assert probes4.mark_ready() is False

# reset() is the only recovery
probes4.reset()
assert probes4.state == ProbeState.STARTING
assert probes4.mark_ready() is True
```

## Common Mistakes

| Mistake                                   | Problem                         | Fix                                      |
| ----------------------------------------- | ------------------------------- | ---------------------------------------- |
| Skipping STARTING -> READY                | `mark_draining()` returns False | Must go through READY first              |
| Expecting recovery from FAILED            | `mark_ready()` returns False    | Use `reset()` first                      |
| Readiness callback that raises exceptions | Unexpected behavior             | Callbacks must return bool, not raise    |
| Forgetting to call `mark_ready()`         | Readiness always 503            | Transition to READY after initialization |

## Exercises

1. **Full Lifecycle**: Walk a ProbeManager through the complete lifecycle: STARTING -> READY -> DRAINING. Check all three probes at each state. Create a table of expected HTTP status codes.

2. **Readiness Matrix**: Create a ProbeManager in READY state. Add three readiness callbacks (two passing, one failing). Verify readiness fails and the response identifies the failing check. Fix the callback and verify readiness passes.

3. **Recovery Pattern**: Transition a ProbeManager to FAILED with a reason. Verify all probes fail. Call `reset()`, verify the state returns to STARTING, then transition back to READY.

## Key Takeaways

- `ProbeState`: STARTING -> READY -> DRAINING (FAILED is terminal, `reset()` recovers).
- `ProbeManager` is thread-safe with atomic state transitions.
- Liveness (`/healthz`): 200 unless FAILED.
- Readiness (`/readyz`): 200 only in READY state AND all callbacks pass.
- Startup (`/startup`): 200 once past STARTING.
- Custom readiness checks via `add_readiness_check(callback)`.
- `ProbeResponse.to_dict()` produces JSON-serializable output.
- Invalid transitions return `False` without changing state.

## Next Package

Continue to [Package 03: Kaizen](../03-kaizen/01_signatures.md) -- Build AI agents with signature-based programming.
