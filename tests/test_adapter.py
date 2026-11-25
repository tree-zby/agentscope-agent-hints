from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from agentscope.credential import OpenAICredential
from agentscope.message import UserMsg
from agentscope.model import ChatModelBase, OpenAIChatModel

from agentscope_agent_hints.adapter import (
    AgentHintMiddleware,
    HintedChatModel,
    message_views,
)
from agentscope_agent_hints.encoder import decode
from agentscope_agent_hints.manage import ManageClient
from agentscope_agent_hints.protocol import HEADER_NAME, validate
from agentscope_agent_hints.state import HintConfig, HintSession

COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "m-1",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop",
        },
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}


class FakeState:
    def __init__(self, session_id: str = "s-1") -> None:
        self.session_id = session_id
        self.context: list[Any] = []
        self.summary: Any = ""


class FakeAgent:
    def __init__(self, session_id: str = "s-1") -> None:
        self.name = "agent"
        self.state = FakeState(session_id)


class FakeModel:
    """Minimal stand-in for ``ChatModelBase`` used by the middleware tests."""

    def __init__(self) -> None:
        self.model = "m-1"
        self.calls: list[dict[str, Any]] = []

    async def _call_api(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return None

    async def __call__(
        self,
        messages: list[Any],
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Any:
        self.calls.append({"messages": messages, "tools": tools, **kwargs})
        return SimpleNamespace(content=[])


class StubChatModel(ChatModelBase):
    """Real ``ChatModelBase`` subclass with no network access."""

    def __init__(self) -> None:
        super().__init__(
            credential=OpenAICredential(api_key="sk-test"),
            model="m-1",
            parameters=self.Parameters(),
            stream=False,
        )
        self.calls: list[dict[str, Any]] = []

    async def _call_api(
        self,
        model_name: str,
        messages: list[Any],
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(
            {
                "model_name": model_name,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                **kwargs,
            },
        )
        return SimpleNamespace(content=[])

    @classmethod
    def _get_retryable_exceptions(cls) -> tuple[type[Exception], ...]:
        return ()

    def _normalize_exception(self, error: Exception) -> Exception:
        return error

    def _get_disable_thinking_kwargs(self) -> dict[str, Any]:
        return {}

    def _get_structured_output_fallback_exceptions(self) -> tuple[type[Exception], ...]:
        return ()


def user(text: str) -> Any:
    return UserMsg(name="user", content=text)


def chain(input_kwargs: dict[str, Any]):
    async def next_handler(**overrides: Any) -> Any:
        kwargs = {**input_kwargs, **overrides}
        model = kwargs["current_model"]
        return await model(
            messages=kwargs["messages"],
            tools=kwargs.get("tools"),
            tool_choice=kwargs.get("tool_choice"),
        )

    return next_handler


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def decode_call(call: dict[str, Any]) -> dict[str, Any]:
    return decode(call["extra_headers"][HEADER_NAME])


def test_middleware_emits_start_then_append() -> None:
    middleware = AgentHintMiddleware(HintConfig())
    agent = FakeAgent()
    model = FakeModel()
    input_kwargs = {
        "current_model": model,
        "messages": [user("hello")],
        "tools": [],
        "tool_choice": None,
    }

    first = run(middleware.on_model_call(agent, dict(input_kwargs), chain(input_kwargs)))
    assert first is not None

    hint = decode_call(model.calls[0])
    assert validate(hint) == []
    assert hint["session"]["state"] == "start"
    assert hint["session"]["call_index"] == 1
    assert hint["session"]["session_id"] == "s-1"
    assert hint["priority_control"]["priority"] == {
        "level": 3,
        "reason": "user_waiting",
    }

    input_kwargs["messages"] = [
        user("hello"),
        SimpleNamespace(role="assistant", content=[SimpleNamespace(type="tool_call")]),
        SimpleNamespace(role="user", content=[SimpleNamespace(type="tool_result")]),
    ]
    run(middleware.on_model_call(agent, dict(input_kwargs), chain(input_kwargs)))

    hint = decode_call(model.calls[1])
    assert hint["session"]["state"] == "append"
    assert hint["session"]["call_index"] == 2
    assert hint["priority_control"]["priority"] == {
        "level": 1,
        "reason": "critical_path",
    }


def test_middleware_keeps_one_session_per_agent() -> None:
    middleware = AgentHintMiddleware()
    first_agent = FakeAgent("s-a")
    second_agent = FakeAgent("s-b")

    for agent in (first_agent, second_agent):
        model = FakeModel()
        kwargs = {
            "current_model": model,
            "messages": [user("hi")],
            "tools": [],
            "tool_choice": None,
        }
        run(middleware.on_model_call(agent, dict(kwargs), chain(kwargs)))
        assert decode_call(model.calls[0])["session"]["session_id"] == agent.state.session_id
        assert decode_call(model.calls[0])["session"]["state"] == "start"


def test_middleware_retry_reuses_the_same_hint() -> None:
    middleware = AgentHintMiddleware()
    agent = FakeAgent()
    kwargs = {
        "messages": [user("hello")],
        "tools": [],
        "tool_choice": None,
    }

    class FailingModel(FakeModel):
        async def __call__(self, *args: Any, **kwargs: Any) -> Any:
            await super().__call__(*args, **kwargs)
            raise RuntimeError("transient provider failure")

    failing = FailingModel()
    with pytest.raises(RuntimeError):
        run(
            middleware.on_model_call(
                agent,
                {"current_model": failing, **kwargs},
                chain({"current_model": failing, **kwargs}),
            ),
        )

    retry = FakeModel()
    run(
        middleware.on_model_call(
            agent,
            {"current_model": retry, **kwargs},
            chain({"current_model": retry, **kwargs}),
        ),
    )

    assert decode_call(failing.calls[0]) == decode_call(retry.calls[0])
    assert decode_call(retry.calls[0])["session"]["call_index"] == 1
    assert decode_call(retry.calls[0])["session"]["state"] == "start"


def test_compression_emits_compact_with_the_superseded_epoch() -> None:
    sent: list[dict[str, Any]] = []

    def transport(url, body, headers, timeout):
        sent.append({"url": url, "body": body, "headers": dict(headers)})
        return 200, None

    manage = ManageClient("http://gateway/v1/chat/completions", "m-1", transport=transport)
    middleware = AgentHintMiddleware(HintConfig(), manage)
    agent = FakeAgent()

    async def scenario() -> None:
        model = FakeModel()
        kwargs = {
            "current_model": model,
            "messages": [user("hello")],
            "tools": [],
            "tool_choice": None,
        }
        await middleware.on_model_call(agent, dict(kwargs), chain(kwargs))

        async def compress() -> None:
            agent.state.context.append("summary")
            agent.state.summary = "compressed"

        await middleware.on_compress_context(agent, {}, compress)

    run(scenario())

    assert len(sent) == 1
    hint = decode(sent[0]["headers"][HEADER_NAME])
    assert validate(hint) == []
    assert hint["session"]["state"] == "compact"
    assert hint["session"]["epoch"] == 0
    assert hint["context_management"]["edits"] == [
        {"type": "evict", "target": "session"},
    ]

    session = middleware.session_for(agent)
    assert session.epoch == 1

    # The next main-line request carries the new epoch and skips the cache.
    model = FakeModel()
    kwargs = {
        "current_model": model,
        "messages": [user("hello"), user("after compact")],
        "tools": [],
        "tool_choice": None,
    }
    run(middleware.on_model_call(agent, dict(kwargs), chain(kwargs)))
    hint = decode_call(model.calls[0])
    assert hint["session"]["epoch"] == 1
    assert "cache_control" not in hint


def test_unchanged_compression_does_not_advance_epoch() -> None:
    middleware = AgentHintMiddleware()
    agent = FakeAgent()

    async def noop() -> None:
        return None

    run(middleware.on_compress_context(agent, {}, noop))
    assert middleware.session_for(agent).epoch == 0


def test_lifecycle_verbs_use_the_manage_client() -> None:
    verbs: list[str] = []

    def transport(url, body, headers, timeout):
        verbs.append(decode(headers[HEADER_NAME])["session"]["state"])
        return 200, None

    manage = ManageClient("http://gateway/v1/chat/completions", "m-1", transport=transport)
    middleware = AgentHintMiddleware(HintConfig(), manage)
    agent = FakeAgent()

    async def scenario() -> None:
        await middleware.pause(agent)
        await middleware.resume(agent)
        await middleware.stop(agent)

    run(scenario())
    assert verbs == ["pause", "resume", "stop"]


def test_wrapped_model_forwards_the_header_to_the_provider_sdk() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["hint"] = request.headers.get(HEADER_NAME)
        return httpx.Response(200, json=COMPLETION)

    credential = OpenAICredential(api_key="sk-test", base_url="http://test/v1")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    inner = OpenAIChatModel(
        credential=credential,
        model="m-1",
        stream=False,
        client_kwargs={"http_client": client},
    )
    wrapped = HintedChatModel(inner, session_id="s-sdk")

    async def scenario() -> None:
        await wrapped([UserMsg(name="user", content="hello")])
        await client.aclose()

    run(scenario())

    hint = decode(captured["hint"])
    assert validate(hint) == []
    assert hint["session"]["session_id"] == "s-sdk"
    assert hint["session"]["state"] == "start"


def test_auxiliary_model_never_sends_a_hint() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["present"] = HEADER_NAME in request.headers
        return httpx.Response(200, json=COMPLETION)

    credential = OpenAICredential(api_key="sk-test", base_url="http://test/v1")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    inner = OpenAIChatModel(
        credential=credential,
        model="m-1",
        stream=False,
        client_kwargs={"http_client": client},
    )
    wrapped = HintedChatModel(inner, session_id="s-aux", auxiliary=True)

    async def scenario() -> None:
        await wrapped([UserMsg(name="user", content="summarize")])
        await client.aclose()

    run(scenario())
    assert captured["present"] is False


def test_wrapper_does_not_double_count_when_middleware_already_injected() -> None:
    inner = StubChatModel()
    session = HintSession("s-1")
    wrapped = HintedChatModel(inner, session=session)

    provided = {
        HEADER_NAME: "already-there",
    }

    async def scenario() -> None:
        await wrapped._call_api(
            "m-1",
            [user("hello")],
            tools=None,
            tool_choice=None,
            extra_headers=dict(provided),
        )

    run(scenario())
    assert session.call_index == 0
    assert inner.calls[0]["extra_headers"] == provided


def test_priority_and_cache_can_be_disabled() -> None:
    middleware = AgentHintMiddleware(HintConfig(priority="omit", cache="omit"))
    agent = FakeAgent()
    model = FakeModel()
    kwargs = {
        "current_model": model,
        "messages": [user("hello")],
        "tools": [],
        "tool_choice": None,
    }
    run(middleware.on_model_call(agent, dict(kwargs), chain(kwargs)))

    hint = decode_call(model.calls[0])
    assert set(hint) == {"session"}


def test_message_views_mirror_the_openai_wire_shape() -> None:
    """msg_offset must equal the number of messages the provider sends."""

    def block(kind: str, **fields: Any) -> SimpleNamespace:
        return SimpleNamespace(type=kind, **fields)

    messages = [
        SimpleNamespace(role="system", content=[block("text", text="sys")]),
        SimpleNamespace(role="user", content=[block("text", text="context " * 120)]),
        # Empty assistant placeholder with a runtime hint: the formatter drops
        # the placeholder and sends the hint as a user message.
        SimpleNamespace(
            role="assistant",
            content=[block("hint", hint="current time: 12:00")],
        ),
    ]
    views = message_views(messages)
    assert [view.role for view in views] == ["system", "user", "user"]
    assert views[-1].text == "current time: 12:00"

    messages.append(
        SimpleNamespace(
            role="assistant",
            content=[
                block("tool_call", id="c1"),
                block("tool_result", id="c1", output="12:00"),
            ],
        ),
    )
    messages.append(SimpleNamespace(role="user", content=[block("text", text="again")]))
    views = message_views(messages)
    assert [view.role for view in views] == [
        "system",
        "user",
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert views[3].has_tool_call is True
    assert views[4].has_tool_result is True

    middleware = AgentHintMiddleware(HintConfig())
    agent = FakeAgent()
    model = FakeModel()
    kwargs = {
        "current_model": model,
        "messages": messages,
        "tools": [],
        "tool_choice": None,
    }
    run(middleware.on_model_call(agent, dict(kwargs), chain(kwargs)))
    hint = decode_call(model.calls[0])
    assert hint["cache_control"]["msg_offset"] == 6
