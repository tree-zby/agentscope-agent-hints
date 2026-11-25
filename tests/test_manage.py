from __future__ import annotations

import json

from agentscope_agent_hints.encoder import decode
from agentscope_agent_hints.manage import ManageClient
from agentscope_agent_hints.protocol import HEADER_NAME, validate
from agentscope_agent_hints.state import HintSession


class RecordingTransport:
    def __init__(self, responses=None) -> None:
        self.responses = list(responses or [])
        self.calls = []

    def __call__(self, url, body, headers, timeout):
        self.calls.append(
            {
                "url": url,
                "body": body,
                "headers": dict(headers),
                "timeout": timeout,
            },
        )
        if self.responses:
            return self.responses.pop(0)
        return 200, None

    def hint(self, index=0):
        return decode(self.calls[index]["headers"][HEADER_NAME])


def make_client(responses=None) -> tuple[ManageClient, RecordingTransport]:
    transport = RecordingTransport(responses)
    client = ManageClient(
        "http://gateway.local/v1/chat/completions",
        "m-1",
        transport=transport,
        sleep=lambda _: None,
    )
    return client, transport


def make_session() -> HintSession:
    session = HintSession("s-1")
    session.call_index = 4
    session.epoch = 1
    return session


def test_manage_body_is_a_placeholder_without_hint_fields() -> None:
    client, transport = make_client()
    client.pause(make_session())
    body = json.loads(transport.calls[0]["body"])
    assert body == {"model": "m-1", "messages": [], "stream": False}


def test_pause_hint_shape_and_validity() -> None:
    client, transport = make_client()
    session = make_session()
    result = client.pause(session)

    assert result.accepted is True
    hint = transport.hint()
    assert validate(hint) == []
    assert hint["session"]["state"] == "pause"
    assert hint["session"]["call_index"] == 4
    assert hint["session"]["epoch"] == 1
    assert hint["context_management"] == {
        "manage_request": True,
        "edits": [{"type": "offload", "target": "session"}],
    }
    assert "priority_control" not in hint
    assert "cache_control" not in hint


def test_pause_is_deduplicated_per_pause_cycle() -> None:
    client, transport = make_client()
    session = make_session()
    client.pause(session)
    second = client.pause(session)
    assert second.skipped is True
    assert len(transport.calls) == 1


def test_resume_uses_the_upcoming_call_index() -> None:
    client, transport = make_client()
    session = make_session()
    client.pause(session)
    client.resume(session)

    hint = transport.hint(1)
    assert hint["session"]["state"] == "resume"
    assert hint["session"]["call_index"] == 5
    assert hint["context_management"]["edits"][0]["type"] == "prefetch"


def test_resume_is_skipped_when_not_paused() -> None:
    client, transport = make_client()
    assert client.resume(make_session()).skipped is True
    assert transport.calls == []


def test_stop_is_sent_once() -> None:
    client, transport = make_client()
    session = make_session()
    client.stop(session)
    assert client.stop(session).skipped is True
    assert len(transport.calls) == 1
    assert transport.hint()["context_management"]["edits"][0]["type"] == "evict"


def test_compact_carries_the_superseded_epoch() -> None:
    client, transport = make_client()
    session = HintSession("s-1")
    session.call_index = 2

    result = client.compact(session)
    assert result.accepted is True
    hint = transport.hint()
    assert hint["session"]["state"] == "compact"
    assert hint["session"]["epoch"] == 0  # old epoch
    assert hint["context_management"]["edits"][0]["type"] == "evict"
    assert session.epoch == 1


def test_compact_is_deduplicated_by_epoch() -> None:
    client, transport = make_client()
    session = make_session()
    client.send(session, "compact", epoch=1)
    again = client.send(session, "compact", epoch=1)
    assert again.skipped is True
    assert len(transport.calls) == 1


def test_resume_retries_transient_rejections() -> None:
    client, transport = make_client()
    session = make_session()
    client.pause(session)
    transport.responses = [(429, "slow down"), (200, None)]

    result = client.resume(session)
    assert result.accepted is True
    assert result.attempts == 2


def test_pause_never_retries_and_isolates_on_unknown_result() -> None:
    client, transport = make_client([(500, "boom")])
    session = make_session()

    result = client.pause(session)
    assert result.accepted is False
    assert result.unknown is True
    assert result.attempts == 1
    assert session.isolated is True
    assert len(transport.calls) == 1


def test_transport_failure_is_unknown_and_isolates() -> None:
    client, _ = make_client([(None, "connection reset")])
    session = make_session()
    result = client.stop(session)
    assert result.unknown is True
    assert session.isolated is True


def test_4xx_is_a_definite_rejection_and_does_not_isolate() -> None:
    client, _ = make_client([(400, "bad request")])
    session = make_session()
    result = client.pause(session)
    assert result.unknown is False
    assert session.isolated is False
