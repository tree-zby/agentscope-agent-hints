# agentscope-agent-hints

Agent Hint protocol (`X-Agent-Hint`) support for [AgentScope](https://github.com/agentscope-ai/agentscope) 2.x.

The plugin tells a hint-aware inference gateway **which session a model call
belongs to, where it sits in the session lifecycle, how urgent it is and
whether its prefix is worth keeping in the KV cache** — without changing the
request body. Everything travels in one HTTP header:

```http
POST /v1/chat/completions
X-Agent-Hint: eyJzZXNzaW9uIjp7InNlc3Npb25faWQiOiJhZ2VudC0xIiw...
```

```jsonc
{
  "session": {
    "session_id": "agent-1",
    "call_index": 3,
    "epoch": 0,
    "state": "append"
  },
  "priority_control": { "priority": { "level": 1, "reason": "critical_path" } },
  "cache_control": { "type": "ephemeral", "ttl_s": 180, "msg_offset": 7 }
}
```

Specification: [agent-hint-protocol v1.0](https://github.com/evan7ang/dsh-agent-hints).
This repository is an independent sender-side implementation; it does not
patch AgentScope.

## Why a middleware

AgentScope compresses context by calling the model **directly**
(`agent.model.generate_structured_output(...)`), bypassing the middleware
chain. That is exactly the protocol's "auxiliary request" case: summary calls
must carry no hint. `AgentHintMiddleware` hooks `on_model_call`, which only
wraps main-line reasoning calls, so auxiliary traffic stays clean with no
content sniffing.

## Install

```bash
pip install "agentscope-agent-hints[agentscope]"
```

The protocol core has no third-party dependencies; the `agentscope` extra
installs the harness. Requires Python 3.11+ (AgentScope 2.x requires it).

## Quick start

```python
import asyncio

from agentscope.agent import Agent
from agentscope.credential import OpenAICredential
from agentscope.message import UserMsg
from agentscope.model import OpenAIChatModel

from agentscope_agent_hints import HintConfig
from agentscope_agent_hints.adapter import AgentHintMiddleware


async def main() -> None:
    model = OpenAIChatModel(
        credential=OpenAICredential(
            api_key="sk-...",
            base_url="https://gateway.example.com/v1",
        ),
        model="your-model",
        stream=True,
    )

    hints = AgentHintMiddleware(
        config=HintConfig(),  # defaults: dynamic priority and cache
    )
    agent = Agent(
        name="assistant",
        system_prompt="You are a helpful assistant.",
        model=model,
        middlewares=[hints],
    )

    await agent.reply(UserMsg(name="user", content="hello"))


asyncio.run(main())
```

`session_id` is taken from `agent.state.session_id`, so it survives agent
serialization/restoration and is stable across turns. One middleware instance
can serve several agents; sessions are keyed per agent.

## Lifecycle (L2)

Compaction is observed automatically: `on_compress_context` detects a
committed rewrite, advances `epoch` and sends a `compact` manage request that
carries the **old** epoch.

`pause` / `resume` / `stop` have no reliable AgentScope event. The host calls
them when it knows the session went idle, came back, or is being destroyed:

```python
from agentscope_agent_hints import ManageClient

manage = ManageClient(
    endpoint="https://gateway.example.com/v1/chat/completions",
    model="your-model",
    extra_headers={"authorization": "Bearer sk-..."},
)
hints = AgentHintMiddleware(manage=manage)

await hints.pause(agent)   # user switched away
await hints.resume(agent)  # user is back (also sent automatically before the next call)
await hints.stop(agent)    # agent destroyed / session archived
```

A finished reply is **not** a pause, and a tool continuation is **not** a
resume; the adapter never fabricates those events.

## Direct model use

If you call a model without an `Agent`, wrap it instead:

```python
from agentscope_agent_hints.adapter import hinted_model

model = hinted_model(inner_model, agent_state=state)
# background/summary model: hinted_model(inner_model, auxiliary=True)
```

Do not combine the wrapper with the middleware on the same model: the wrapper
detects an already-injected header and leaves it untouched, but the middleware
is the compliant choice inside an agent because it excludes compression calls.

## Configuration

```python
HintConfig(
    enabled=True,                # master switch
    priority="dynamic",          # "dynamic" | "omit"
    cache="dynamic",             # "dynamic" | "omit"
    priority_level=None,         # static override, folded to {1, 3}
    priority_reason=None,        # must pair with the level
    cache_min_prefix_tokens=128,
    default_ttl_s=600,           # no interval sample
    ttl_min_s=60,
    ttl_max_s=3600,
    parent_session_id=None,      # only for context continuation
    has_history=False,           # restored/imported session => append
    assume_existing_session=False,
)
```

## What is emitted

| Situation | Hint |
| --- | --- |
| First accepted main-line call of a new session | `state=start`, priority `3/user_waiting`, cache when qualified |
| Tool-result continuation | `state=append`, priority `1/critical_path` |
| New user turn | `state=append`, priority `3/user_waiting` |
| Retry of the same logical request | identical hint, same `call_index` |
| Context compression request | no hint |
| Model/tool signature changed | no `cache_control` for one call |
| First call after a rewrite | no `cache_control`, `epoch` already incremented |
| Restart with `call_index > 1` and unknown signature | no `cache_control` for one call |

## Compliance

* **L1 (core wire): yes** — encoded header, `start`/`append`, priority,
  cache qualification, auxiliary exclusion, retry reuse, body untouched.
* **L2 (lifecycle): yes** — manage requests with explicit edits, placeholder
  body, isolation on unknown results, resume-only retry. Triggers for
  `pause`/`resume`/`stop` are host-driven (no such AgentScope events).
* **L3 (early prefetch/archive): no** — no draft or archive signal is exposed.

See [docs/protocol-compliance.md](docs/protocol-compliance.md) for the full
checklist, trigger table and degradation profile. Wire-level evidence from a
capture stub is in [e2e/](e2e/).

## Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[agentscope,dev]"
pytest
ruff check .
python e2e/wire_capture.py   # needs the agentscope extra + a free local port
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [PRIVACY.md](PRIVACY.md).
