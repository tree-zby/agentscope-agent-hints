# Protocol compliance — AgentScope adapter

Target: **agent-hint-protocol v1.0**.
Implementation level: **L1 + L2** (L3 not claimed).

The adapter is a sender only. Sections marked *sender* below describe what
leaves the process; receiver-side checks (R-01 … R-10) are out of scope.

## 1. Where the hint is injected

| Harness | Injection point | Why |
| --- | --- | --- |
| AgentScope 2.0.12 | `MiddlewareBase.on_model_call` → per-request model proxy → provider `extra_headers` | Runs once per main-line call inside the agent retry loop; context compression calls `agent.model` directly and therefore stays hint-free |
| Direct model use | `HintedChatModel._call_api` → provider `extra_headers` | No agent, no compression; `auxiliary=True` available for background models |

The provider adapters (`OpenAIChatModel`, `AnthropicChatModel`, DeepSeek,
Volcengine, Moonshot, Ollama, xAI, …) merge unknown generate kwargs into the
SDK call; `extra_headers` is documented by both SDKs. `count_tokens` is local
and never emits a request.

## 2. Trigger table (protocol §7.5)

| state | AgentScope trigger | Required | Degradation when missing |
| --- | --- | --- | --- |
| `start` / `append` | first accepted main-line request / all later ones, decided by `on_model_call` + 2xx | yes | none |
| `pause` | `await middleware.pause(agent)` — host knows the user left and no stream is in flight | L2 | only sent when the host calls it |
| `resume` | automatically before the next main-line request while paused; host may call `middleware.resume(agent)` earlier | L2 | no early prefetch (no draft signal) |
| `stop` | `await middleware.stop(agent)` on destroy/archive | L2 | no stop on process exit |
| `compact` | `on_compress_context` detects a committed rewrite (context length or summary changed) | L2 | `epoch` still advances; no manage request without a `ManageClient` |

Boundaries respected: a finished reply is not `pause`/`stop`; a tool-result
continuation is `append`, never `resume`; the compression summary request
itself carries no hint.

## 3. Facts and algorithms (protocol §6)

| Fact | Source |
| --- | --- |
| session id | `agent.state.session_id` (stable, ≤256 chars; agent name as fallback) |
| call index | in-process counter; retries reuse the hint via a request fingerprint (model + sorted tool names + message views) |
| epoch | incremented only on a detected committed rewrite |
| turn opening | trailing user message that is not a tool result (falls back to `call_index == 1` when no messages) |
| interval | previous successful response end → current call start, monotonic clock |
| prefix signature | model id + sorted tool names |
| prefix size | CJK 1 char/token, other text 4 chars/token; unmeasurable text does not suppress |
| thread safety | per-session `RLock` around derivation |

Restart behaviour: a restored `HintSession` with `call_index > 1` and no
remembered signature suppresses `cache_control` for one call, then resumes
(C-07).

## 4. Compliance checklist (protocol §13.2)

**Transport**

- [x] W-01 hint only in `X-Agent-Hint` (exact case on send).
- [x] W-02 request body contains no hint fields; only headers are modified.
- [x] W-03 base64url, unpadded, UTF-8 JSON.
- [x] W-04 single header value, typical < 1 KB, bounded by schema size.
- [x] W-05 auxiliary requests carry no hint (compression bypasses the middleware; `auxiliary=True` for wrapper use).
- [x] W-06 no session identity → no hint (middleware falls through).
- [x] W-07 retry reuses the same hint; fingerprint dedupe keeps `call_index`.
- [x] W-08 only an opaque random session id; no prompts, keys or user data.

**Session**

- [x] S-01 session id stable per agent and unique per agent state.
- [x] S-02 `parent_session_id` only via explicit `HintConfig.parent_session_id`.
- [x] S-03 `call_index` starts at 1, includes the current call, retries do not increase it.
- [x] S-04 `epoch` starts at 0, +1 per committed rewrite, never decreases.
- [x] S-05 every hint has `state`.
- [x] S-06 `start` once per new session; restored/history sessions use `append`.
- [x] S-07 derivation is request-scoped and serialized per session.
- [x] S-08 a forked/sub-agent gets its own `AgentState` → new session id.

**Priority**

- [x] P-01 every ordinary request carries `priority_control` unless `priority="omit"`.
- [x] P-02 turn opening → `3/user_waiting`.
- [x] P-03 continuation → `1/critical_path`.
- [x] P-04 level/reason are validated as a pair; static overrides fold to `{1,3}`.
- [x] P-05 manage/auxiliary requests carry no priority.

**Cache**

- [x] C-01 a single `ephemeral` object.
- [x] C-02 `ttl_s` clamped to `[60,3600]`; 600 with no interval sample.
- [x] C-03 `msg_offset` = logical input message count (system included, tool results counted).
- [x] C-04 suppressed for the first call after a rewrite.
- [x] C-05 suppressed when model or tool signature changes.
- [x] C-06 suppressed below 128 estimated prefix tokens.
- [x] C-07 suppressed for one call after restart with unknown signature.
- [x] C-08 manage/auxiliary requests carry no cache.

**Lifecycle**

- [x] L-01 `start` on the first accepted main-line request of a new session.
- [x] L-02 `append` elsewhere.
- [x] L-03 `pause` only via an explicit idle signal; deduplicated per pause cycle.
- [x] L-04 `resume` before the next use (earlier via explicit API); deduplicated.
- [x] L-05 `stop` only on destroy/archive; not per turn.
- [x] L-06 `compact` after a committed rewrite, carrying the old epoch; the next request uses `epoch + 1`.
- [x] L-07 `state` and `edits[].type` are fixed by the verb.
- [x] L-08 missing triggers are documented here, not fabricated.

**Manage**

- [x] M-01 lifecycle verbs only in manage requests.
- [x] M-02 `manage_request === true` (boolean).
- [x] M-03 exactly one edit, correct type, `target: "session"`.
- [x] M-04 placeholder body `{model, messages: [], stream: false}`; no hint in body.
- [x] M-05 credentials come from the caller's `extra_headers`; they are not copied from the model client automatically.
- [x] M-06 only `pause`, `resume`, `stop`, `compact`.
- [x] M-07 per-session lock in `ManageClient`; unknown results isolate the session.
- [x] M-08 `SessionIsolatedError` blocks new main-line requests until `clear_isolation()`.
- [x] M-09 only `resume` retries transient rejections (bounded).
- [x] M-10 no dispatch after teardown is the host's responsibility; exit-time `stop` loss is documented.

**Receiver** — not implemented (sender-side plugin).

## 5. Degradation profile (protocol §12.4)

| Missing capability | Behaviour |
| --- | --- |
| No persistent event log | in-process counters; `session.snapshot()` / `restore()` available for the host |
| No user-input boundary signal | derived from message roles; trailing user message = turn opening |
| No interval observation | TTL = 600 |
| No draft signal | `resume` only before the next request (or explicit API) |
| No archive event | `stop` only on explicit destroy/archive |
| No manage endpoint | L1 only (`ManageClient` omitted); `compact` still advances epoch |
| Prefix size unknown | no 128-token suppression |

## 6. Known limits

* Streaming responses mark the request accepted on the first chunk and the
  response end when the stream closes.
* `on_compress_context` detects rewrites through context length and summary;
  an exotic compression that keeps both unchanged would not be detected.
* Restored sessions need `HintConfig(has_history=True)` (or
  `assume_existing_session=True`) to avoid a `start` after a process restart.

## 7. Evidence

`e2e/wire_capture.py` starts a local OpenAI-compatible stub, runs a real
`Agent` with the middleware, and records the bytes that leave the process.
`e2e/evidence/agentscope.json` contains the captured headers and bodies
(desensitized). Covered: `start` → tool continuation (`append`, priority 1) →
new user turn (priority 3), no hint on the compression-style auxiliary call,
and a `pause`/`resume`/`stop` manage sequence.
