from __future__ import annotations

from agentscope_agent_hints.protocol import validate

ORDINARY = {
    "session": {
        "session_id": "s-1",
        "call_index": 3,
        "epoch": 0,
        "state": "append",
    },
    "priority_control": {"priority": {"level": 1, "reason": "critical_path"}},
    "cache_control": {"type": "ephemeral", "ttl_s": 120, "msg_offset": 5},
}

MANAGE = {
    "session": {
        "session_id": "s-1",
        "call_index": 3,
        "epoch": 1,
        "state": "pause",
    },
    "context_management": {
        "manage_request": True,
        "edits": [{"type": "offload", "target": "session"}],
    },
}


def test_valid_ordinary_hint() -> None:
    assert validate(ORDINARY) == []


def test_valid_manage_hint() -> None:
    assert validate(MANAGE) == []


def test_rejects_legacy_top_level_keys() -> None:
    hint = dict(ORDINARY, agent_status={"state": "running"})
    assert any("unexpected top-level key" in error for error in validate(hint))


def test_rejects_inline_manage_request_string() -> None:
    hint = dict(ORDINARY)
    hint["context_management"] = {"manage_request": "false", "edits": []}
    errors = validate(hint)
    assert any("boolean true" in error for error in errors)
    assert any("MUST NOT appear on an ordinary request" in error for error in errors)


def test_rejects_lifecycle_state_on_ordinary_request() -> None:
    hint = {
        "session": {
            "session_id": "s-1",
            "call_index": 2,
            "epoch": 0,
            "state": "pause",
        },
    }
    assert any("ordinary request state" in error for error in validate(hint))


def test_rejects_priority_on_manage_request() -> None:
    hint = dict(MANAGE, priority_control={"priority": {"level": 3, "reason": "user_waiting"}})
    assert any("MUST NOT appear on a manage request" in error for error in validate(hint))


def test_rejects_mismatched_level_and_reason() -> None:
    hint = dict(ORDINARY, priority_control={"priority": {"level": 3, "reason": "critical_path"}})
    assert any("must pair with reason" in error for error in validate(hint))


def test_rejects_legacy_range_field() -> None:
    hint = dict(
        ORDINARY,
        cache_control={
            "type": "ephemeral",
            "ttl_s": 120,
            "msg_offset": 5,
            "range": {"end": 5, "unit": "message"},
        },
    )
    assert any("unexpected cache_control key" in error for error in validate(hint))


def test_rejects_ttl_out_of_bounds() -> None:
    hint = dict(ORDINARY, cache_control={"type": "ephemeral", "ttl_s": 30, "msg_offset": 5})
    assert any("ttl_s" in error for error in validate(hint))


def test_rejects_wrong_edit_for_state() -> None:
    hint = dict(MANAGE)
    hint["context_management"] = {
        "manage_request": True,
        "edits": [{"type": "evict", "target": "session"}],
    }
    assert any("edits[0].type" in error for error in validate(hint))


def test_rejects_empty_edits_on_manage_request() -> None:
    hint = dict(MANAGE)
    hint["context_management"] = {"manage_request": True, "edits": []}
    assert any("exactly one edit" in error for error in validate(hint))


def test_rejects_non_object() -> None:
    assert validate(["nope"]) == ["hint must be a JSON object"]
