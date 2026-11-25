from __future__ import annotations

import base64

import pytest

from agentscope_agent_hints.encoder import HintDecodeError, decode, encode

HINT = {
    "session": {
        "session_id": "s-7f3a2c",
        "call_index": 1,
        "epoch": 0,
        "state": "start",
    },
    "priority_control": {"priority": {"level": 3, "reason": "user_waiting"}},
}


def test_roundtrip_is_unpadded_base64url() -> None:
    value = encode(HINT)
    assert "=" not in value
    assert set(value) <= set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    )
    assert decode(value) == HINT


def test_decode_accepts_padding() -> None:
    value = encode({"session": {"session_id": "s", "call_index": 1, "epoch": 0, "state": "append"}})
    padded = value + "=" * (-len(value) % 4)
    assert decode(padded)["session"]["state"] == "append"


def test_non_ascii_survives() -> None:
    hint = {
        "session": {
            "session_id": "会话-1",
            "call_index": 1,
            "epoch": 0,
            "state": "start",
        },
    }
    assert decode(encode(hint)) == hint


def test_rejects_non_object_payload() -> None:
    value = base64.urlsafe_b64encode(b"[1,2,3]").rstrip(b"=").decode()
    with pytest.raises(HintDecodeError):
        decode(value)


def test_rejects_invalid_json() -> None:
    value = base64.urlsafe_b64encode(b"{not json").rstrip(b"=").decode()
    with pytest.raises(HintDecodeError):
        decode(value)
