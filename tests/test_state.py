from __future__ import annotations

import pytest

from agentscope_agent_hints.state import (
    HintConfig,
    HintSession,
    MessageView,
    RequestView,
    SessionIsolatedError,
    SessionSnapshot,
    derive_turn_opening,
    estimate_prefix_tokens,
    openai_message_views,
    request_fingerprint,
    tool_names,
)

LONG = "context " * 120  # ~240 tokens, above the 128-token cache floor


class Clock:
    def __init__(self, value: float = 1_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def msg(
    role: str,
    text: str = "",
    *,
    tool_result: bool = False,
    tool_call: bool = False,
) -> MessageView:
    return MessageView(role=role, text=text, has_tool_result=tool_result, has_tool_call=tool_call)


def request(messages, *, model="m-1", tools=("search",), now=None, auxiliary=False, opening=None):
    return RequestView(
        messages=messages,
        model=model,
        tools=tools,
        now=now,
        auxiliary=auxiliary,
        turn_opening=opening,
    )


def new_session(clock: Clock, **config) -> HintSession:
    return HintSession("s-1", HintConfig(**config), clock=clock)


# --------------------------------------------------------------------- V1/V2/V3
def test_v1_new_session_start() -> None:
    clock = Clock()
    session = new_session(clock)
    hint = session.build(
        request([msg("system", LONG), msg("user", "hello")], now=clock()),
    )

    assert hint == {
        "session": {
            "session_id": "s-1",
            "call_index": 1,
            "epoch": 0,
            "state": "start",
        },
        "priority_control": {"priority": {"level": 3, "reason": "user_waiting"}},
        "cache_control": {"type": "ephemeral", "ttl_s": 600, "msg_offset": 2},
    }


def test_v2_tool_continuation_is_append_priority_1() -> None:
    clock = Clock()
    session = new_session(clock)
    session.build(request([msg("system", LONG), msg("user", "hello")], now=clock()))
    session.note_accepted()
    session.note_response_end()

    clock.advance(30)
    hint = session.build(
        request(
            [
                msg("system", LONG),
                msg("user", "hello"),
                msg("assistant", "", tool_call=True),
                msg("user", "", tool_result=True),
            ],
            now=clock(),
        ),
    )

    assert hint["session"]["state"] == "append"
    assert hint["session"]["call_index"] == 2
    assert hint["session"]["epoch"] == 0
    assert hint["priority_control"]["priority"] == {"level": 1, "reason": "critical_path"}
    assert hint["cache_control"]["ttl_s"] == 60
    assert hint["cache_control"]["msg_offset"] == 4


def test_v3_new_user_turn_raises_priority_again() -> None:
    clock = Clock()
    session = new_session(clock)
    session.build(request([msg("system", LONG), msg("user", "one")], now=clock()))
    session.note_accepted()
    session.note_response_end()

    clock.advance(5)
    session.build(
        request(
            [
                msg("system", LONG),
                msg("user", "one"),
                msg("assistant", "", tool_call=True),
                msg("user", "", tool_result=True),
            ],
            now=clock(),
        ),
    )
    session.note_accepted()
    session.note_response_end()

    clock.advance(5)
    hint = session.build(
        request(
            [
                msg("system", LONG),
                msg("user", "one"),
                msg("assistant", "done"),
                msg("user", "two"),
            ],
            now=clock(),
        ),
    )
    assert hint["session"]["call_index"] == 3
    assert hint["priority_control"]["priority"] == {"level": 3, "reason": "user_waiting"}


# ------------------------------------------------------------------------- V4
def test_v4_restored_session_is_append() -> None:
    clock = Clock()
    session = new_session(clock, has_history=True)
    hint = session.build(request([msg("system", LONG), msg("user", "resume me")], now=clock()))
    assert hint["session"]["state"] == "append"
    assert hint["session"]["call_index"] == 1


def test_v4_restart_unknown_signature_suppresses_one_cache_declaration() -> None:
    clock = Clock()
    session = new_session(clock)
    session.restore(
        SessionSnapshot(session_id="s-1", call_index=2, epoch=0, first_accepted=True),
    )
    hint = session.build(
        request(
            [msg("system", LONG), msg("user", "hello"), msg("assistant", "hi")],
            now=clock(),
        ),
    )
    assert "cache_control" not in hint
    assert hint["session"]["call_index"] == 3

    hint = session.build(
        request(
            [
                msg("system", LONG),
                msg("user", "hello"),
                msg("assistant", "hi"),
                msg("user", "again"),
            ],
            now=clock(),
        ),
    )
    assert hint["cache_control"]["msg_offset"] == 4


# ------------------------------------------------------------------------- V6
def test_v6_rewrite_advances_epoch_and_suppresses_cache_once() -> None:
    clock = Clock()
    session = new_session(clock)
    session.build(request([msg("system", LONG), msg("user", "hello")], now=clock()))
    session.note_accepted()
    session.note_response_end()

    old_epoch = session.note_rewrite()
    assert old_epoch == 0
    assert session.epoch == 1

    hint = session.build(
        request([msg("system", LONG), msg("user", "post-rewrite")], now=clock()),
    )
    assert hint["session"]["epoch"] == 1
    assert "cache_control" not in hint

    hint = session.build(
        request(
            [msg("system", LONG), msg("user", "post-rewrite"), msg("assistant", "ok")],
            now=clock(),
        ),
    )
    assert hint["cache_control"]["msg_offset"] == 3


# ------------------------------------------------------------------------- V8
def test_v8_auxiliary_request_has_no_hint() -> None:
    clock = Clock()
    session = new_session(clock)
    assert session.build(request([msg("system", "summarize")], auxiliary=True)) is None
    assert session.call_index == 0


def test_disabled_sender_has_no_hint() -> None:
    clock = Clock()
    session = new_session(clock, enabled=False)
    assert session.build(request([msg("user", "hi")])) is None


# ------------------------------------------------------------------------- V9
def test_v9_retry_reuses_the_same_hint_without_incrementing() -> None:
    clock = Clock()
    session = new_session(clock)
    first = session.build(request([msg("system", LONG), msg("user", "hello")], now=clock()))
    retry = session.build(request([msg("system", LONG), msg("user", "hello")], now=clock()))

    assert retry == first
    assert retry["session"]["call_index"] == 1
    assert retry["session"]["state"] == "start"

    session.note_accepted()
    next_call = session.build(request([msg("system", LONG), msg("user", "hello")], now=clock()))
    assert next_call["session"]["call_index"] == 2


# ------------------------------------------------------------------------ V10
def test_v10_model_or_tool_change_suppresses_cache_for_one_call() -> None:
    clock = Clock()
    session = new_session(clock)
    session.build(request([msg("system", LONG), msg("user", "a")], now=clock()))
    session.note_accepted()

    changed = session.build(
        request([msg("system", LONG), msg("user", "a"), msg("assistant", "b")], model="m-2"),
    )
    assert "cache_control" not in changed

    stable = session.build(
        request(
            [msg("system", LONG), msg("user", "a"), msg("assistant", "b"), msg("user", "c")],
            model="m-2",
        ),
    )
    assert stable["cache_control"]["msg_offset"] == 4


def test_start_is_emitted_exactly_once() -> None:
    clock = Clock()
    session = new_session(clock)
    states = []
    for index in range(4):
        messages = [msg("system", LONG)] + [msg("user", f"turn {i}") for i in range(index + 1)]
        hint = session.build(request(messages, now=clock()))
        states.append(hint["session"]["state"])
        session.note_accepted()
        session.note_response_end()
    assert states == ["start", "append", "append", "append"]


def test_parent_session_id_is_only_emitted_when_configured() -> None:
    clock = Clock()
    session = new_session(clock, parent_session_id="s-root")
    hint = session.build(request([msg("user", "hi")]))
    assert hint["session"]["parent_session_id"] == "s-root"


def test_priority_and_cache_can_be_omitted() -> None:
    clock = Clock()
    session = new_session(clock, priority="omit", cache="omit")
    hint = session.build(request([msg("system", LONG), msg("user", "hi")]))
    assert set(hint) == {"session"}


def test_static_priority_is_folded_to_two_levels() -> None:
    clock = Clock()
    session = new_session(clock, priority_level=7)
    hint = session.build(request([msg("system", LONG), msg("user", "hi")]))
    assert hint["priority_control"]["priority"] == {"level": 3, "reason": "user_waiting"}


def test_isolated_session_refuses_new_requests() -> None:
    clock = Clock()
    session = new_session(clock)
    session.isolate()
    with pytest.raises(SessionIsolatedError):
        session.build(request([msg("user", "hi")]))
    session.clear_isolation()
    assert session.build(request([msg("user", "hi")])) is not None


def test_snapshot_roundtrip() -> None:
    clock = Clock()
    session = new_session(clock)
    session.build(request([msg("system", LONG), msg("user", "hello")], now=clock()))
    session.note_accepted()
    session.note_response_end(1234.0)
    data = session.snapshot().to_dict()

    restored = new_session(clock)
    restored.restore(data)
    assert restored.call_index == 1
    assert restored.first_accepted is True
    assert restored.last_response_end == 1234.0


# ---------------------------------------------------------------- helpers
def test_msg_offset_equals_logical_message_count() -> None:
    clock = Clock()
    session = new_session(clock)
    messages = [msg("system", LONG)] + [msg("user", f"m{i}") for i in range(6)]
    hint = session.build(request(messages, now=clock()))
    assert hint["cache_control"]["msg_offset"] == len(messages)


def test_turn_opening_derivation() -> None:
    assert derive_turn_opening([msg("user", "hi")]) is True
    assert derive_turn_opening([msg("user", "hi"), msg("user", "", tool_result=True)]) is False
    assert derive_turn_opening([msg("user", "hi"), msg("assistant", "ok")]) is False
    assert derive_turn_opening([]) is False
    # AgentScope appends an empty assistant placeholder as the generation
    # target; it does not mean the turn already has a reply.
    assert (
        derive_turn_opening(
            [msg("system", "sys"), msg("user", "hi"), msg("assistant", "")],
        )
        is True
    )
    assert (
        derive_turn_opening(
            [
                msg("system", "sys"),
                msg("user", "hi"),
                msg("assistant", "", tool_call=True),
            ],
        )
        is False
    )
    assert (
        derive_turn_opening(
            [msg("user", "hi"), msg("user", "", tool_result=True), msg("assistant", "")],
        )
        is False
    )


def test_prefix_estimate_handles_cjk_and_empty() -> None:
    assert estimate_prefix_tokens([msg("user", "你好世界")]) == 4
    assert estimate_prefix_tokens([]) is None
    assert estimate_prefix_tokens([msg("user", "")]) is None


def test_openai_messages_and_tool_names() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "result"},
    ]
    views = openai_message_views(messages)
    assert views[1].text == "hi"
    assert views[2].has_tool_call is True
    assert views[3].has_tool_result is True

    assert tool_names([{"type": "function", "function": {"name": "search"}}]) == ("search",)
    assert tool_names(["alpha", "beta"]) == ("alpha", "beta")


def test_fingerprint_is_stable_and_order_independent_for_tools() -> None:
    a = request([msg("user", "hi")], tools=("a", "b"))
    b = request([msg("user", "hi")], tools=("b", "a"))
    assert request_fingerprint(a) == request_fingerprint(b)
