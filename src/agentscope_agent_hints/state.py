# Deterministic hint derivation (protocol sections 5-6).
#
# The state machine is the *sender* side of the contract. It is deliberately
# pure with respect to the request: one request in, one hint out, with all
# mutable facts held on the session object.
from __future__ import annotations

import copy
import hashlib
import json
import secrets
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .protocol import EDIT_FOR_STATE, HEADER_NAME  # re-exported for callers


class SessionIsolatedError(RuntimeError):
    """Raised when a session has an unresolved manage mutation.

    Protocol section 8.4: after a manage request with an unknown result the
    sender MUST refuse new main-line requests for that session until the
    operator clears the condition.
    """


@dataclass(frozen=True)
class HintConfig:
    """Sender configuration.

    ``priority`` / ``cache`` accept ``"dynamic"`` (derive from facts) or
    ``"omit"`` (leave the block out). A static priority can be pinned with
    ``priority_level`` / ``priority_reason``; the level is folded to {1, 3}.
    """

    enabled: bool = True
    priority: str = "dynamic"
    cache: str = "dynamic"
    priority_level: int | None = None
    priority_reason: str | None = None
    cache_min_prefix_tokens: int = 128
    default_ttl_s: int = 600
    ttl_min_s: int = 60
    ttl_max_s: int = 3600
    parent_session_id: str | None = None
    has_history: bool = False
    assume_existing_session: bool = False

    def __post_init__(self) -> None:
        if self.priority not in ("dynamic", "omit"):
            raise ValueError("priority must be 'dynamic' or 'omit'")
        if self.cache not in ("dynamic", "omit"):
            raise ValueError("cache must be 'dynamic' or 'omit'")
        if self.priority_level is not None and self.priority_level not in (1, 3):
            if self.priority_level < 1:
                raise ValueError("priority_level must be >= 1")
            object.__setattr__(self, "priority_level", 3 if self.priority_level >= 3 else 1)
        if self.ttl_min_s < 60 or self.ttl_max_s > 3600 or self.ttl_min_s > self.ttl_max_s:
            raise ValueError("TTL bounds must satisfy 60 <= min <= max <= 3600")
        if self.default_ttl_s < self.ttl_min_s or self.default_ttl_s > self.ttl_max_s:
            raise ValueError("default_ttl_s must be inside the TTL bounds")

    @property
    def static_priority(self) -> tuple[int, str] | None:
        if self.priority_level is None:
            return None
        level = self.priority_level
        reason = self.priority_reason or ("user_waiting" if level == 3 else "critical_path")
        expected = "user_waiting" if level == 3 else "critical_path"
        if reason != expected:
            raise ValueError(
                f"priority_level {level} must pair with reason {expected!r}",
            )
        return level, reason


@dataclass(frozen=True)
class MessageView:
    """Provider-neutral view of one logical input message."""

    role: str
    text: str = ""
    has_tool_result: bool = False
    has_tool_call: bool = False


@dataclass(frozen=True)
class RequestView:
    """Everything the derivation needs to know about one model request."""

    messages: Sequence[MessageView]
    model: str = ""
    tools: Sequence[str] = ()
    auxiliary: bool = False
    turn_opening: bool | None = None
    fingerprint: str | None = None
    now: float | None = None


@dataclass
class SessionSnapshot:
    """Serializable facts, for persistence across processes."""

    session_id: str
    call_index: int = 0
    epoch: int = 0
    first_accepted: bool = False
    started: bool = False
    paused: bool = False
    stopped: bool = False
    isolated: bool = False
    last_signature: list[str] | None = None
    last_response_end: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "call_index": self.call_index,
            "epoch": self.epoch,
            "first_accepted": self.first_accepted,
            "started": self.started,
            "paused": self.paused,
            "stopped": self.stopped,
            "isolated": self.isolated,
            "last_signature": self.last_signature,
            "last_response_end": self.last_response_end,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SessionSnapshot:
        return cls(
            session_id=str(data["session_id"]),
            call_index=int(data.get("call_index", 0)),
            epoch=int(data.get("epoch", 0)),
            first_accepted=bool(data.get("first_accepted", False)),
            started=bool(data.get("started", False)),
            paused=bool(data.get("paused", False)),
            stopped=bool(data.get("stopped", False)),
            isolated=bool(data.get("isolated", False)),
            last_signature=list(data["last_signature"]) if data.get("last_signature") else None,
            last_response_end=data.get("last_response_end"),
        )


def new_session_id(prefix: str = "ah") -> str:
    """Generate an opaque, non-identifying session id (max 256 chars)."""
    return f"{prefix}-{secrets.token_hex(12)}"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def request_fingerprint(req: RequestView) -> str:
    """Stable identity of a logical request, used to reuse a hint on retry."""
    payload = {
        "model": req.model,
        "tools": sorted(req.tools),
        "messages": [
            [m.role, m.text, m.has_tool_result, m.has_tool_call] for m in req.messages
        ],
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def estimate_prefix_tokens(messages: Sequence[MessageView]) -> int | None:
    """Approximate the prefix size; ``None`` when there is no text to measure.

    CJK characters count as one token each, other text as four characters per
    token (protocol section 6.8).
    """
    saw_text = False
    total = 0.0
    for message in messages:
        text = message.text or ""
        if text:
            saw_text = True
        cjk = sum(1 for char in text if "\u3400" <= char <= "\u9fff")
        total += cjk + (len(text) - cjk) / 4.0
    if not saw_text:
        return None
    return int(total)


def derive_turn_opening(messages: Sequence[MessageView]) -> bool:
    """A turn opens when the latest user input has no reply after it.

    Walks backwards: empty assistant placeholders (AgentScope appends one as
    the generation target) are skipped, the first real assistant message or
    tool call means the turn is already in progress, and a plain user message
    means this is the first main-line call of the turn.
    """
    for message in reversed(messages):
        if message.role == "assistant":
            if not message.text and not message.has_tool_call:
                continue
            return False
        if message.has_tool_call or message.has_tool_result:
            return False
        if message.role == "user":
            return True
    return False


class HintSession:
    """Mutable per-session facts plus the pure hint builders."""

    def __init__(
        self,
        session_id: str | None = None,
        config: HintConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or HintConfig()
        self.session_id = session_id or new_session_id()
        if not self.session_id or len(self.session_id) > 256:
            raise ValueError("session_id must be a non-empty string <= 256 chars")
        self._clock = clock
        self.call_index = 0
        self.epoch = 0
        self.first_accepted = False
        self.started = False
        self.last_signature: tuple[str, ...] | None = None
        self.last_response_end: float | None = None
        self.after_rewrite = False
        self.isolated = False
        self._last_fingerprint: str | None = None
        self._last_hint: dict[str, Any] | None = None
        self._last_outcome: str | None = None
        self._manage_sent: set[tuple[str, int]] = set()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ facts
    def snapshot(self) -> SessionSnapshot:
        return SessionSnapshot(
            session_id=self.session_id,
            call_index=self.call_index,
            epoch=self.epoch,
            first_accepted=self.first_accepted,
            started=self.started,
            isolated=self.isolated,
            last_signature=list(self.last_signature) if self.last_signature else None,
            last_response_end=self.last_response_end,
        )

    def restore(self, snapshot: SessionSnapshot | Mapping[str, Any]) -> None:
        if not isinstance(snapshot, SessionSnapshot):
            snapshot = SessionSnapshot.from_dict(snapshot)
        if snapshot.session_id != self.session_id:
            raise ValueError("snapshot belongs to a different session")
        self.call_index = snapshot.call_index
        self.epoch = snapshot.epoch
        self.first_accepted = snapshot.first_accepted
        self.started = snapshot.started
        self.isolated = snapshot.isolated
        self.last_signature = (
            tuple(snapshot.last_signature) if snapshot.last_signature else None
        )
        self.last_response_end = snapshot.last_response_end

    def note_accepted(self) -> None:
        """Record that the endpoint accepted the in-flight ordinary request."""
        with self._lock:
            self.first_accepted = True
            self._last_outcome = "accepted"

    def note_response_end(self, now: float | None = None) -> None:
        with self._lock:
            self.last_response_end = self._clock() if now is None else now

    def note_rewrite(self) -> int:
        """Record a committed history rewrite; return the superseded epoch."""
        with self._lock:
            old_epoch = self.epoch
            self.epoch += 1
            self.after_rewrite = True
            return old_epoch

    def isolate(self) -> None:
        self.isolated = True

    def clear_isolation(self) -> None:
        self.isolated = False
        self._last_outcome = None

    def assert_ready(self) -> None:
        if self.isolated:
            raise SessionIsolatedError(
                f"session {self.session_id!r} is isolated after an unresolved "
                "manage request; call clear_isolation() after confirming the "
                "backend state, or continue with a new session_id",
            )

    # --------------------------------------------------------------- builders
    def build(self, request: RequestView) -> dict[str, Any] | None:
        """Return the hint for an ordinary request, or ``None`` to send none."""
        with self._lock:
            return self._build(request)

    def _build(self, request: RequestView) -> dict[str, Any] | None:
        cfg = self.config
        if not cfg.enabled or request.auxiliary:
            return None
        self.assert_ready()

        fingerprint = request.fingerprint or request_fingerprint(request)
        if (
            fingerprint == self._last_fingerprint
            and self._last_hint is not None
            and self._last_outcome != "accepted"
        ):
            # Same logical request again: a retry reuses the exact hint.
            return copy.deepcopy(self._last_hint)

        self.call_index += 1
        opening = (
            request.turn_opening
            if request.turn_opening is not None
            else derive_turn_opening(request.messages)
        )
        state = self._ordinary_state()
        session: dict[str, Any] = {
            "session_id": self.session_id,
            "call_index": self.call_index,
            "epoch": self.epoch,
            "state": state,
        }
        if cfg.parent_session_id:
            session["parent_session_id"] = cfg.parent_session_id

        hint: dict[str, Any] = {"session": session}

        if cfg.priority != "omit":
            hint["priority_control"] = {"priority": self._priority(opening)}

        cache = self._cache(request)
        if cache is not None:
            hint["cache_control"] = cache

        self._last_fingerprint = fingerprint
        self._last_hint = hint
        self._last_outcome = "pending"
        self.after_rewrite = False
        return copy.deepcopy(hint)

    def build_manage(
        self,
        verb: str,
        *,
        call_index: int | None = None,
        epoch: int | None = None,
    ) -> dict[str, Any]:
        """Build a manage hint (protocol sections 5.3 and 8.2)."""
        with self._lock:
            return self._build_manage(verb, call_index=call_index, epoch=epoch)

    def _build_manage(
        self,
        verb: str,
        *,
        call_index: int | None = None,
        epoch: int | None = None,
    ) -> dict[str, Any]:
        if verb not in EDIT_FOR_STATE:
            raise ValueError(f"manage verb must be one of {tuple(EDIT_FOR_STATE)}")
        if call_index is None:
            call_index = self._manage_call_index(verb)
        if epoch is None:
            epoch = self.epoch
        session: dict[str, Any] = {
            "session_id": self.session_id,
            "call_index": max(1, call_index),
            "epoch": epoch,
            "state": verb,
        }
        if self.config.parent_session_id:
            session["parent_session_id"] = self.config.parent_session_id
        return {
            "session": session,
            "context_management": {
                "manage_request": True,
                "edits": [{"type": EDIT_FOR_STATE[verb], "target": "session"}],
            },
        }

    # ---------------------------------------------------------------- helpers
    def _ordinary_state(self) -> str:
        cfg = self.config
        fresh = (
            self.call_index == 1
            and not cfg.has_history
            and not cfg.assume_existing_session
            and not self.first_accepted
            and not self.started
        )
        if fresh:
            self.started = True
            return "start"
        return "append"

    def _priority(self, opening: bool) -> dict[str, Any]:
        static = self.config.static_priority
        if static is not None:
            level, reason = static
        elif opening:
            level, reason = 3, "user_waiting"
        else:
            level, reason = 1, "critical_path"
        return {"level": level, "reason": reason}

    def _cache(self, request: RequestView) -> dict[str, Any] | None:
        cfg = self.config
        if cfg.cache == "omit":
            return None
        if self.after_rewrite:
            return None

        signature = tuple(sorted(request.tools))
        signature_key = (request.model, *signature)
        stable = (
            self.last_signature == signature_key
            if self.last_signature is not None
            else self.call_index <= 1
        )
        self.last_signature = signature_key
        if not stable:
            return None

        tokens = estimate_prefix_tokens(request.messages)
        if tokens is not None and tokens < cfg.cache_min_prefix_tokens:
            return None

        msg_offset = len(request.messages)
        if msg_offset < 1:
            return None

        ttl = self._ttl(request.now)
        return {"type": "ephemeral", "ttl_s": ttl, "msg_offset": msg_offset}

    def _ttl(self, now: float | None) -> int:
        cfg = self.config
        if now is None or self.last_response_end is None:
            return cfg.default_ttl_s
        gap = max(0.0, now - self.last_response_end)
        ttl = int(round(gap * 2))
        return max(cfg.ttl_min_s, min(cfg.ttl_max_s, ttl))

    def _manage_call_index(self, verb: str) -> int:
        if verb == "resume":
            return self.call_index + 1
        return max(1, self.call_index)


def openai_message_views(messages: Any) -> list[MessageView]:
    """Convert OpenAI-style chat messages into provider-neutral views."""
    if isinstance(messages, str):
        return [MessageView(role="user", text=messages)]
    if isinstance(messages, Mapping):
        messages = [messages]

    views: list[MessageView] = []
    for message in messages or []:
        if not isinstance(message, Mapping):
            views.append(MessageView(role="user", text=str(message)))
            continue
        role = str(message.get("role") or "user")
        content = message.get("content")
        text = _content_text(content)
        has_tool_result = role == "tool" or bool(message.get("tool_call_id"))
        has_tool_call = bool(message.get("tool_calls"))
        views.append(
            MessageView(
                role=role,
                text=text,
                has_tool_result=has_tool_result,
                has_tool_call=has_tool_call,
            ),
        )
    return views


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                value = item.get("text")
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(parts)
    return str(content)


def tool_names(tools: Any) -> tuple[str, ...]:
    """Extract tool names from OpenAI or AgentScope tool schemas."""
    names: list[str] = []
    for tool in tools or []:
        if isinstance(tool, str):
            names.append(tool)
            continue
        if not isinstance(tool, Mapping):
            name = getattr(tool, "name", None)
            names.append(str(name) if name else str(tool))
            continue
        function = tool.get("function")
        if isinstance(function, Mapping) and function.get("name"):
            names.append(str(function["name"]))
        elif tool.get("name"):
            names.append(str(tool["name"]))
        elif tool and "type" not in tool:
            # A mapping of tool name -> tool object (CrewAI passes these in
            # some code paths).
            names.extend(str(key) for key in tool)
    return tuple(sorted(names))


__all__ = [
    "HEADER_NAME",
    "HintConfig",
    "HintSession",
    "MessageView",
    "RequestView",
    "SessionIsolatedError",
    "SessionSnapshot",
    "derive_turn_opening",
    "estimate_prefix_tokens",
    "new_session_id",
    "openai_message_views",
    "request_fingerprint",
    "tool_names",
]
