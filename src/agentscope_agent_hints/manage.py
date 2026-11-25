# Manage requests (protocol sections 5.3, 7.3, 8 and 9.2).
from __future__ import annotations

import asyncio
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .encoder import encode
from .protocol import HEADER_NAME, MANAGE_STATES
from .state import HintSession

Transport = Callable[[str, bytes, Mapping[str, str], float], tuple[int | None, str | None]]


@dataclass(frozen=True)
class ManageResult:
    verb: str
    skipped: bool = False
    status: int | None = None
    accepted: bool = False
    error: str | None = None
    attempts: int = 0

    @property
    def unknown(self) -> bool:
        """True when the backend may or may not have applied the mutation."""
        if self.skipped or self.accepted:
            return False
        if self.status is None:
            return True
        return self.status >= 500 or self.status in (408, 429)


def urllib_transport(
    url: str,
    body: bytes,
    headers: Mapping[str, str],
    timeout: float,
) -> tuple[int | None, str | None]:
    """Default control-plane transport; keeps the package dependency-free."""
    request = urllib.request.Request(
        url,
        data=body,
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            return int(response.status), None
    except urllib.error.HTTPError as exc:
        return int(exc.code), str(exc.reason)
    except Exception as exc:  # noqa: BLE001 - transport failures are reported as unknown
        return None, str(exc)


class ManageClient:
    """Serial, deduplicated manage-request sender.

    One instance can serve several sessions; per-session operations are
    serialized with a lock. Only ``resume`` retries, and only on transient
    rejections (protocol section 8.4).
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        timeout: float = 5.0,
        transport: Transport | None = None,
        extra_headers: Mapping[str, str] | None = None,
        max_resume_retries: int = 2,
        retry_backoff_s: float = 0.25,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not endpoint:
            raise ValueError("a control endpoint is required")
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.transport = transport or urllib_transport
        self.extra_headers = dict(extra_headers or {})
        self.max_resume_retries = max(0, max_resume_retries)
        self.retry_backoff_s = max(0.0, retry_backoff_s)
        self._sleep = sleep
        self._lock = threading.Lock()
        self._paused: set[str] = set()
        self._stopped: set[str] = set()
        self._sent: set[tuple[str, str, int]] = set()

    # ------------------------------------------------------------------ verbs
    def pause(self, session: HintSession) -> ManageResult:
        return self.send(session, "pause")

    def resume(self, session: HintSession) -> ManageResult:
        return self.send(session, "resume")

    def stop(self, session: HintSession) -> ManageResult:
        return self.send(session, "stop")

    def compact(self, session: HintSession) -> ManageResult:
        """Record a committed rewrite and notify the backend with the old epoch."""
        old_epoch = session.note_rewrite()
        return self.send(session, "compact", epoch=old_epoch, call_index=session.call_index)

    # ------------------------------------------------------------------- send
    def send(
        self,
        session: HintSession,
        verb: str,
        *,
        call_index: int | None = None,
        epoch: int | None = None,
    ) -> ManageResult:
        if verb not in MANAGE_STATES:
            raise ValueError(f"verb must be one of {MANAGE_STATES}")

        with self._lock:
            skip = self._dedupe(session.session_id, verb, epoch)
            if skip is not None:
                return skip

        hint = session.build_manage(verb, call_index=call_index, epoch=epoch)
        payload = {
            "model": self.model,
            "messages": [],
            "stream": False,
        }
        import json

        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers = {
            "content-type": "application/json",
            "accept": "application/json",
            HEADER_NAME: encode(hint),
            **self.extra_headers,
        }

        attempts = 0
        max_attempts = 1 + (self.max_resume_retries if verb == "resume" else 0)
        while True:
            attempts += 1
            status, error = self.transport(self.endpoint, body, headers, self.timeout)
            accepted = status is not None and 200 <= status < 300
            if accepted:
                self._commit(session, verb, epoch)
                return ManageResult(
                    verb=verb,
                    status=status,
                    accepted=True,
                    attempts=attempts,
                )

            transient = status is None or status == 408 or status == 429 or status >= 500
            if verb == "resume" and transient and attempts < max_attempts:
                self._sleep(self.retry_backoff_s * attempts)
                continue

            result = ManageResult(
                verb=verb,
                status=status,
                accepted=False,
                error=error,
                attempts=attempts,
            )
            if result.unknown:
                # The mutation may have landed; refuse new main-line requests.
                session.isolate()
            return result

    async def asend(
        self,
        session: HintSession,
        verb: str,
        *,
        call_index: int | None = None,
        epoch: int | None = None,
    ) -> ManageResult:
        return await asyncio.to_thread(
            self.send,
            session,
            verb,
            call_index=call_index,
            epoch=epoch,
        )

    async def apause(self, session: HintSession) -> ManageResult:
        return await self.asend(session, "pause")

    async def aresume(self, session: HintSession) -> ManageResult:
        return await self.asend(session, "resume")

    async def astop(self, session: HintSession) -> ManageResult:
        return await self.asend(session, "stop")

    async def acompact(self, session: HintSession) -> ManageResult:
        old_epoch = session.note_rewrite()
        return await self.asend(
            session,
            "compact",
            epoch=old_epoch,
            call_index=session.call_index,
        )

    # ---------------------------------------------------------------- internal
    def _dedupe(
        self,
        session_id: str,
        verb: str,
        epoch: int | None,
    ) -> ManageResult | None:
        if verb == "stop":
            if session_id in self._stopped:
                return ManageResult(verb=verb, skipped=True, accepted=True)
        elif verb == "pause":
            if session_id in self._paused:
                return ManageResult(verb=verb, skipped=True, accepted=True)
        elif verb == "resume":
            if session_id not in self._paused:
                return ManageResult(verb=verb, skipped=True, accepted=True)
        elif verb == "compact":
            key = (session_id, verb, epoch if epoch is not None else -1)
            if key in self._sent:
                return ManageResult(verb=verb, skipped=True, accepted=True)
        return None

    def _commit(self, session: HintSession, verb: str, epoch: int | None) -> None:
        with self._lock:
            if verb == "pause":
                self._paused.add(session.session_id)
            elif verb == "resume":
                self._paused.discard(session.session_id)
            elif verb == "stop":
                self._stopped.add(session.session_id)
                self._paused.discard(session.session_id)
            elif verb == "compact":
                self._sent.add(
                    (session.session_id, verb, epoch if epoch is not None else -1),
                )

    def is_paused(self, session: HintSession) -> bool:
        return session.session_id in self._paused
