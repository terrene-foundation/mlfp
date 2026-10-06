# MLFP06 — Task 3: Serve a Governed Endpoint (JWT + Rate Limit + Role Routing)

**Weight**: 30 marks · **Framework**: Nexus (`Nexus`, `NexusAuthPlugin`) +
PACT (`GovernanceEngine`) · **Outcomes assessed**: production serving with
authentication, rate limiting, CORS, and governance on the handler (6.8, 6.7)

## Scenario

A public agency ships an internal policy assistant. Officers from several
ministries call one HTTP endpoint; each ministry's SSO issues HS256 JWTs
carrying a `roles` claim. The endpoint must:

- **refuse** requests with no token or a forged token (401, from the
  middleware — before any handler code runs);
- **throttle** bursts (429 from the rate limiter);
- answer **CORS** correctly: echo only the allow-listed origin;
- take the caller's tier from the **verified token claim** — never from the
  request body (a body saying `"role": "admin"` must not escalate anyone);
- run a **PACT governance check** for the tier before serving.

No LLM is involved: the governed "answer" is a deterministic string built
from the question. What is real is the middleware stack and the governance
decision — that is what is graded.

## Interfaces

```python
def solve() -> dict: ...
```

Returns `{"app": Nexus, "issuer": JWTValidator, "endpoint": str}`.

- `app`: a Nexus app with one async handler registered under the name
  `serve_qa` (so the endpoint is `/workflows/serve_qa/execute`), an
  auth plugin configured with HS256 JWT and a rate limit of
  **6 requests/minute, burst 2**, and a CORS allow-list containing exactly
  `https://intranet.example.sg`.
- `issuer`: a `JWTValidator` bound to the same HS256 secret the app verifies
  with (read the secret from `MLFP_JWT_SECRET` if set, else generate one per
  run — never hardcode it). The grader mints its own tokens through this
  issuer.
- `endpoint`: the path string above.

### The governance tiers

Build a PACT engine (the canonical course org compiles structurally with
`compile_governance(apply_specs=False)`) and attach two envelopes:

| Tier    | Role address  | Cap (USD) | Allowed actions   |
| ------- | ------------- | --------- | ----------------- |
| `qa`    | `D1-R1-T1-R1` | 0.015     | `generate_answer` |
| `admin` | `D1-R1-T2-R1` | 0.05      | `generate_answer` |

A request's cost is `$0.02` when the question is longer than 280 characters,
else `$0.01`. The handler calls
`engine.verify_action(role_address=..., action="generate_answer", context={"cost": cost})`
for the token's tier and serves only when the verdict allows it. A token
whose role is neither `qa` nor `admin` is refused.

### Handler response contract

On success (HTTP 200), the handler returns a dict with at least:
`{"answer": f"[{role}] {question}", "role": role, "verdict": "served",
"blocked": False, "user": <token subject>, "cost": cost}`.

On a governance refusal (unknown tier, over-budget question), still HTTP 200
with `{"blocked": True, "verdict": "blocked", "role": <role or "">, "user":
<token subject>, "error": <reason>}`.

## Acceptance criteria (what the grader measures)

The grader sends real HTTP requests through your app's middleware stack
(in-process, no ports) with tokens it mints itself:

| #   | Check                                                                            |
| --- | -------------------------------------------------------------------------------- |
| 1   | Contract: app, issuer and endpoint returned (gate)                               |
| 2   | `qa` token + short question → 200, served, role `qa`, correct answer string      |
| 3   | `qa` token + long question → 200, **blocked** (over the qa cap)                  |
| 4   | `admin` token + long question → 200, served                                      |
| 5   | `qa` token + body `"role": "admin"` + long question → still the `qa` decision    |
| 6   | Token with an unknown role (grader-chosen per run) → blocked                     |
| 7   | The handler output carries the grader's per-run token subject                    |
| 8   | No token → 401                                                                   |
| 9   | Forged token (grader's own secret) → 401                                         |
| 10  | A burst of requests hits 429                                                     |
| 11  | `Origin: https://intranet.example.sg` is echoed in `access-control-allow-origin` |
| 12  | `Origin: https://evil.example.com` is not echoed                                 |

Marks = 30 × (non-gate checks passed / 11). If the gate fails, the task
scores 0.

## Rules

- No LLM calls. No network listeners — the grader drives the app in-process.
- The role comes from `request.state.user` (set by the JWT middleware);
  the request body's `role` field is ignored.
- Keep the middleware order: rate limit first, then JWT (the plugin's
  documented order), so unauthenticated bursts are refused cheaply.
- Self-check: run `starter.py`; it drives your app in-process and prints the
  status codes and handler outputs.
