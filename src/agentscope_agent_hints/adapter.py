# AgentScope 2.x adapter.
#
# Two integration surfaces, in order of preference:
#
#   * ``AgentHintMiddleware`` — an AgentScope middleware. ``on_model_call``
#     runs once per main-line model request, *inside* the agent retry loop and
#     *outside* context compression, so auxiliary summary calls never carry a
#     hint. It swaps ``current_model`` for a per-request proxy that forwards
#     ``extra_headers`` to the provider adapter.
#   * ``HintedChatModel`` — a transparent ``ChatModelBase`` wrapper for direct
#     model use. Do not combine it with the middleware: the wrapper detects an
#     already-injected header and leaves it untouched.
from __future__ import annotations

import hashlib
import inspect
import time
from collections.abc import AsyncGenerator, Iterable, Mapping
from typing import Any

from .encoder import encode
from .manage import ManageClient, ManageResult
from .protocol import HEADER_NAME
from .state import (
    HintConfig,
    HintSession,
    MessageView,
    RequestView,
    tool_names,
)

try:  # pragma: no cover - exercised only with the optional dependency
    from agentscope.middleware import MiddlewareBase
    from agentscope.model import ChatModelBase
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "AgentScope is not installed. Install the extra with "
        "`pip install 'agentscope-agent-hints[agentscope]'`.",
    ) from exc


def _block_type(block: Any) -> str:
    value = getattr(block, "type", None)
    if value is None and isinstance(block, Mapping):
        value = block.get("type")
    return str(value or "")


def _block_text(block: Any) -> str:
    value = getattr(block, "text", None)
    if value is None and isinstance(block, Mapping):
        value = block.get("text")
    return value if isinstance(value, str) else ""


def _block_hint_text(block: Any) -> str:
    value = getattr(block, "hint", None)
    if value is None and isinstance(block, Mapping):
        value = block.get("hint")
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(text for text in (_block_text(item) for item in value) if text)
    return ""


def _block_output_text(block: Any) -> str:
    value = getattr(block, "output", None)
    if value is None and isinstance(block, Mapping):
        value = block.get("output")
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(text for text in (_block_text(item) for item in value) if text)
    return ""


def message_views(messages: Iterable[Any]) -> list[MessageView]:
    """Translate AgentScope ``Msg`` objects into wire-aligned views.

    One view is produced per message the provider formatter will send, so
    ``len(views)`` is the protocol's ``msg_offset``. The OpenAI formatter
    splits a message around ``HintBlock`` (sent as a ``user`` message) and
    ``ToolResultBlock`` (sent as a ``tool`` message), and drops empty
    placeholders; this function mirrors that structure.
    """
    views: list[MessageView] = []
    for message in messages:
        role = getattr(message, "role", None)
        if role is None and isinstance(message, Mapping):
            role = message.get("role")
        content = getattr(message, "content", None)
        if content is None and isinstance(message, Mapping):
            content = message.get("content")
        role = str(role or "user")

        if not isinstance(content, (list, tuple)):
            if isinstance(content, str):
                text = content
            elif content is None:
                text = ""
            else:
                text = str(content)
            views.append(MessageView(role=role, text=text))
            continue

        segment: list[str] = []
        segment_has_call = False

        def flush(role: str = role) -> None:
            nonlocal segment, segment_has_call
            text = "\n".join(part for part in segment if part)
            if text or segment_has_call:
                views.append(
                    MessageView(
                        role=role,
                        text=text,
                        has_tool_call=segment_has_call,
                    ),
                )
            segment = []
            segment_has_call = False

        for block in content:
            kind = _block_type(block)
            if kind == "text":
                segment.append(_block_text(block))
            elif kind == "tool_call":
                segment_has_call = True
            elif kind == "hint":
                flush()
                hint_text = _block_hint_text(block)
                if hint_text:
                    views.append(MessageView(role="user", text=hint_text))
            elif kind == "tool_result":
                flush()
                views.append(
                    MessageView(
                        role="tool",
                        text=_block_output_text(block),
                        has_tool_result=True,
                    ),
                )
        flush()
    return views


def _forwards_extra_headers(model: Any) -> bool:
    try:
        signature = inspect.signature(model._call_api)
    except (AttributeError, TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


class _HintedCall:
    """Per-request proxy that attaches the hint to one model call.

    ``Agent`` invokes ``current_model(messages=..., tools=..., tool_choice=...)``
    and the model forwards unknown kwargs to its provider adapter, which is
    where ``extra_headers`` is supported.
    """

    def __init__(
        self,
        model: Any,
        hint: dict[str, Any],
        session: HintSession,
        *,
        inject: bool,
    ) -> None:
        self._model = model
        self._hint = hint
        self._session = session
        self._inject = inject

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    async def __call__(
        self,
        messages: list[Any],
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Any:
        if self._inject:
            headers = dict(kwargs.pop("extra_headers", None) or {})
            headers.setdefault(HEADER_NAME, encode(self._hint))
            kwargs["extra_headers"] = headers

        result = await self._model(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )
        if inspect.isasyncgen(result):
            return self._observe(result)
        self._session.note_accepted()
        self._session.note_response_end()
        return result

    async def _observe(self, stream: AsyncGenerator[Any, None]) -> AsyncGenerator[Any, None]:
        accepted = False
        try:
            async for chunk in stream:
                if not accepted:
                    self._session.note_accepted()
                    accepted = True
                yield chunk
        finally:
            if accepted:
                self._session.note_response_end()


class AgentHintMiddleware(MiddlewareBase):
    """AgentScope middleware that emits the Agent Hint protocol.

    One middleware instance may serve several agents: sessions are keyed by
    ``agent.state.session_id``. ``on_compress_context`` reports a committed
    rewrite as a ``compact`` manage request; ``pause`` / ``resume`` / ``stop``
    are explicit calls made by the host application when it knows the session
    went idle, came back or was destroyed.
    """

    def __init__(
        self,
        config: HintConfig | None = None,
        manage: ManageClient | None = None,
        *,
        clock: Any = time.monotonic,
    ) -> None:
        self.config = config or HintConfig()
        self.manage = manage
        self._clock = clock
        self._sessions: dict[str, HintSession] = {}

    # ---------------------------------------------------------------- sessions
    @staticmethod
    def _has_session_id(agent: Any) -> bool:
        state = getattr(agent, "state", None)
        return bool(getattr(state, "session_id", None) or getattr(agent, "name", None))

    def session_for(self, agent: Any) -> HintSession:
        state = getattr(agent, "state", None)
        session_id = getattr(state, "session_id", None) or getattr(agent, "name", None)
        if not session_id:
            raise ValueError("agent has no state.session_id; cannot identify the session")
        session = self._sessions.get(session_id)
        if session is None:
            session = HintSession(str(session_id), self.config, clock=self._clock)
            self._sessions[str(session_id)] = session
        return session

    # ------------------------------------------------------------ model calls
    async def on_model_call(
        self,
        agent: Any,
        input_kwargs: dict,
        next_handler: Any,
    ) -> Any:
        session = self.session_for(agent) if self._has_session_id(agent) else None
        if session is None:
            # No session identity: send no hint rather than invent one.
            return await next_handler()
        if self.manage is not None and self.manage.is_paused(session):
            # Resume before the next main-line request (protocol section 7.5).
            await self.manage.aresume(session)

        model = input_kwargs.get("current_model")
        request = RequestView(
            messages=message_views(input_kwargs.get("messages") or []),
            model=str(getattr(model, "model", "")),
            tools=tool_names(input_kwargs.get("tools")),
            now=self._clock(),
        )
        hint = session.build(request)
        if hint is None:
            return await next_handler()
        return await next_handler(
            current_model=_HintedCall(
                model,
                hint,
                session,
                inject=_forwards_extra_headers(model),
            ),
        )

    # ------------------------------------------------------------- compaction
    async def on_compress_context(
        self,
        agent: Any,
        input_kwargs: dict,
        next_handler: Any,
    ) -> None:
        before = self._context_key(agent)
        await next_handler()
        after = self._context_key(agent)
        if before == after:
            return

        if not self._has_session_id(agent):
            return
        session = self.session_for(agent)
        if self.manage is not None:
            await self.manage.acompact(session)
        else:
            session.note_rewrite()

    # -------------------------------------------------------------- lifecycle
    async def pause(self, agent: Any) -> ManageResult | None:
        return None if self.manage is None else await self.manage.apause(self.session_for(agent))

    async def resume(self, agent: Any) -> ManageResult | None:
        return None if self.manage is None else await self.manage.aresume(self.session_for(agent))

    async def stop(self, agent: Any) -> ManageResult | None:
        return None if self.manage is None else await self.manage.astop(self.session_for(agent))

    @staticmethod
    def _context_key(agent: Any) -> str:
        state = getattr(agent, "state", None)
        summary = getattr(state, "summary", "") or ""
        context = getattr(state, "context", None) or []
        payload = f"{len(context)}|{summary!r}"
        return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()


class HintedChatModel(ChatModelBase):
    """Transparent ``ChatModelBase`` wrapper for direct model use.

    Prefer :class:`AgentHintMiddleware` inside an ``Agent``: context
    compression calls the model directly, so a wrapped model would attach a
    hint to auxiliary summary requests. Set ``auxiliary=True`` for an
    instance that is only used for background work.
    """

    def __init__(
        self,
        inner: ChatModelBase,
        *,
        session: HintSession | None = None,
        session_id: str | None = None,
        config: HintConfig | None = None,
        manage: ManageClient | None = None,
        auxiliary: bool = False,
    ) -> None:
        super().__init__(
            credential=inner.credential,
            model=inner.model,
            parameters=inner.parameters,
            stream=inner.stream,
            max_retries=inner.max_retries,
            retry_delay=inner.retry_delay,
            context_size=inner.context_size,
            raise_on_refusal=getattr(inner, "raise_on_refusal", False),
            raise_on_content_filter=getattr(inner, "raise_on_content_filter", False),
        )
        self.inner = inner
        self.session = session or HintSession(session_id, config)
        self.manage = manage
        self.auxiliary = auxiliary
        self._inject = _forwards_extra_headers(inner)

    def __getattr__(self, name: str) -> Any:
        # ``formatter``, ``client``, ``extra_body`` and provider-specific
        # attributes stay on the wrapped instance.
        return getattr(self.inner, name)

    # Retry/error behaviour must match the wrapped adapter exactly.
    def _get_retryable_exceptions(self) -> tuple[type[Exception], ...]:
        return self.inner._get_retryable_exceptions()

    def _normalize_exception(self, error: Exception) -> Exception:
        return self.inner._normalize_exception(error)

    def _get_disable_thinking_kwargs(self) -> dict[str, Any]:
        return self.inner._get_disable_thinking_kwargs()

    def _get_structured_output_fallback_exceptions(self) -> tuple[type[Exception], ...]:
        return self.inner._get_structured_output_fallback_exceptions()

    async def _call_api(
        self,
        model_name: str,
        messages: list[Any],
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Any:
        headers = dict(kwargs.get("extra_headers", None) or {})
        if HEADER_NAME in headers:
            # A middleware already produced the hint for this request.
            return await self.inner._call_api(
                model_name,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )

        if self.manage is not None and self.manage.is_paused(self.session):
            await self.manage.aresume(self.session)

        hint = None
        if not self.auxiliary:
            hint = self.session.build(
                RequestView(
                    messages=message_views(messages),
                    model=model_name,
                    tools=tool_names(tools),
                    now=time.monotonic(),
                ),
            )

        if hint is not None and self._inject:
            headers[HEADER_NAME] = encode(hint)
            kwargs["extra_headers"] = headers

        result = await self.inner._call_api(
            model_name,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )
        if inspect.isasyncgen(result):
            return self._observe_stream(result, hint is not None)
        if hint is not None:
            self.session.note_accepted()
            self.session.note_response_end()
        return result

    async def _observe_stream(
        self,
        stream: AsyncGenerator[Any, None],
        count: bool,
    ) -> AsyncGenerator[Any, None]:
        accepted = False
        try:
            async for chunk in stream:
                if count and not accepted:
                    self.session.note_accepted()
                    accepted = True
                yield chunk
        finally:
            if count:
                self.session.note_response_end()

    def __repr__(self) -> str:
        return f"HintedChatModel(inner={self.inner!r}, session_id={self.session.session_id!r})"


def hinted_model(
    inner: ChatModelBase,
    *,
    agent_state: Any | None = None,
    session: HintSession | None = None,
    config: HintConfig | None = None,
    manage: ManageClient | None = None,
    auxiliary: bool = False,
) -> HintedChatModel:
    """Build a hinted model bound to an ``AgentState`` when one is supplied."""
    session_id = None
    if session is None and agent_state is not None:
        session_id = getattr(agent_state, "session_id", None)
    return HintedChatModel(
        inner,
        session=session,
        session_id=session_id,
        config=config,
        manage=manage,
        auxiliary=auxiliary,
    )


__all__ = [
    "AgentHintMiddleware",
    "HintedChatModel",
    "hinted_model",
    "message_views",
]
