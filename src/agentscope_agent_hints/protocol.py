# Agent Hint protocol v1 — wire schema and validation.
#
# This module is dependency-free on purpose: the same code runs inside an
# AgentScope agent, a CrewAI crew and a plain test process.
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

HEADER_NAME = "X-Agent-Hint"
"""HTTP header that carries the base64url-encoded hint."""

MAX_HEADER_BYTES = 4096
"""Sender budget from protocol section 3.3."""

SESSION_STATES = ("start", "append", "pause", "resume", "stop", "compact")
ORDINARY_STATES = ("start", "append")
MANAGE_STATES = ("pause", "resume", "stop", "compact")

PRIORITY_LEVELS = (1, 3)
PRIORITY_REASONS = ("user_waiting", "critical_path")
PRIORITY_PAIRS = {3: "user_waiting", 1: "critical_path"}

EDIT_FOR_STATE = {
    "pause": "offload",
    "resume": "prefetch",
    "stop": "evict",
    "compact": "evict",
}
VALID_EDIT_TYPES = ("offload", "prefetch", "evict")
EDIT_TARGET = "session"


class HintValidationError(ValueError):
    """Raised by :func:`validate_or_raise` when a hint breaks the contract."""


def is_manage_hint(hint: Mapping[str, Any]) -> bool:
    """Return True when this hint is a manage request (protocol section 9.2)."""
    context = hint.get("context_management")
    return isinstance(context, Mapping) and context.get("manage_request") is True


def validate(hint: Any) -> list[str]:
    """Return every contract violation found in ``hint``.

    The sender uses this for self-checks. A receiver MUST NOT reject a model
    request because of these violations, so this function never raises.
    """
    errors: list[str] = []

    if not isinstance(hint, Mapping):
        return ["hint must be a JSON object"]

    allowed_top = {"session", "context_management", "priority_control", "cache_control"}
    for key in hint:
        if key not in allowed_top:
            errors.append(f"unexpected top-level key: {key!r}")

    session = hint.get("session")
    if not isinstance(session, Mapping):
        errors.append("session is required and must be an object")
        return errors

    _validate_session(session, errors)

    state = session.get("state")
    context = hint.get("context_management")
    manage = is_manage_hint(hint)

    if context is not None:
        _validate_context_management(context, errors)

    if manage:
        if state not in MANAGE_STATES:
            errors.append(
                f"manage request state must be one of {MANAGE_STATES}, got {state!r}",
            )
        if state in MANAGE_STATES and isinstance(context, Mapping):
            edits = context.get("edits")
            if isinstance(edits, list) and len(edits) == 1 and isinstance(edits[0], Mapping):
                expected = EDIT_FOR_STATE[state]
                if edits[0].get("type") != expected:
                    errors.append(
                        f"edits[0].type must be {expected!r} for state {state!r}",
                    )
        for forbidden in ("priority_control", "cache_control"):
            if forbidden in hint:
                errors.append(f"{forbidden} MUST NOT appear on a manage request")
    else:
        if context is not None:
            errors.append("context_management MUST NOT appear on an ordinary request")
        if state not in ORDINARY_STATES:
            errors.append(
                f"ordinary request state must be one of {ORDINARY_STATES}, got {state!r}",
            )

    if "priority_control" in hint:
        _validate_priority_control(hint["priority_control"], errors)

    if "cache_control" in hint:
        _validate_cache_control(hint["cache_control"], errors)

    return errors


def validate_or_raise(hint: Any) -> None:
    errors = validate(hint)
    if errors:
        raise HintValidationError("; ".join(errors))


def _validate_session(session: Mapping[str, Any], errors: list[str]) -> None:
    allowed = {"session_id", "parent_session_id", "call_index", "epoch", "state"}
    for key in session:
        if key not in allowed:
            errors.append(f"unexpected session key: {key!r}")

    session_id = session.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        errors.append("session.session_id must be a non-empty string")
    elif len(session_id) > 256:
        errors.append("session.session_id must be at most 256 characters")

    parent = session.get("parent_session_id", None)
    if parent is not None and (not isinstance(parent, str) or not parent):
        errors.append("session.parent_session_id must be a non-empty string when present")

    call_index = session.get("call_index")
    if not isinstance(call_index, int) or isinstance(call_index, bool) or call_index < 1:
        errors.append("session.call_index must be an integer >= 1")

    epoch = session.get("epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        errors.append("session.epoch must be an integer >= 0")

    state = session.get("state")
    if state not in SESSION_STATES:
        errors.append(f"session.state must be one of {SESSION_STATES}, got {state!r}")


def _validate_context_management(context: Any, errors: list[str]) -> None:
    if not isinstance(context, Mapping):
        errors.append("context_management must be an object")
        return

    allowed = {"manage_request", "edits"}
    for key in context:
        if key not in allowed:
            errors.append(f"unexpected context_management key: {key!r}")

    if context.get("manage_request") is not True:
        errors.append("context_management.manage_request must be the boolean true")

    edits = context.get("edits")
    if not isinstance(edits, list):
        errors.append("context_management.edits must be an array")
        return
    if len(edits) != 1:
        errors.append("context_management.edits must contain exactly one edit")
        return

    edit = edits[0]
    if not isinstance(edit, Mapping):
        errors.append("context_management.edits[0] must be an object")
        return
    for key in edit:
        if key not in ("type", "target"):
            errors.append(f"unexpected edit key: {key!r}")
    if edit.get("type") not in VALID_EDIT_TYPES:
        errors.append(f"edit.type must be one of {VALID_EDIT_TYPES}")
    if edit.get("target") != EDIT_TARGET:
        errors.append("edit.target must be 'session'")


def _validate_priority_control(control: Any, errors: list[str]) -> None:
    if not isinstance(control, Mapping):
        errors.append("priority_control must be an object")
        return
    for key in control:
        if key != "priority":
            errors.append(f"unexpected priority_control key: {key!r}")

    priority = control.get("priority")
    if not isinstance(priority, Mapping):
        errors.append("priority_control.priority must be an object")
        return
    for key in priority:
        if key not in ("level", "reason"):
            errors.append(f"unexpected priority key: {key!r}")

    level = priority.get("level")
    reason = priority.get("reason")
    if level not in PRIORITY_LEVELS:
        errors.append(f"priority.level must be one of {PRIORITY_LEVELS}")
    if reason not in PRIORITY_REASONS:
        errors.append(f"priority.reason must be one of {PRIORITY_REASONS}")
    if level in PRIORITY_PAIRS and reason in PRIORITY_REASONS:
        if PRIORITY_PAIRS[level] != reason:
            errors.append(
                f"priority.level {level} must pair with reason "
                f"{PRIORITY_PAIRS[level]!r}, got {reason!r}",
            )


def _validate_cache_control(cache: Any, errors: list[str]) -> None:
    if not isinstance(cache, Mapping):
        errors.append("cache_control must be an object")
        return
    for key in cache:
        if key not in ("type", "ttl_s", "msg_offset"):
            errors.append(f"unexpected cache_control key: {key!r}")

    if cache.get("type") != "ephemeral":
        errors.append("cache_control.type must be 'ephemeral'")

    ttl = cache.get("ttl_s")
    if not isinstance(ttl, int) or isinstance(ttl, bool) or not 60 <= ttl <= 3600:
        errors.append("cache_control.ttl_s must be an integer in [60, 3600]")

    offset = cache.get("msg_offset")
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 1:
        errors.append("cache_control.msg_offset must be an integer >= 1")


def iter_states(hints: Iterable[Mapping[str, Any]]) -> list[Any]:
    """Small helper used by tests and the capture checker."""
    return [hint.get("session", {}).get("state") for hint in hints]
