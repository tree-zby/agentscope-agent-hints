"""Wire-level capture for the AgentScope adapter (protocol section 13.3).

Starts a local OpenAI-compatible stub, runs a real ``Agent`` with
``AgentHintMiddleware`` and records the headers/bodies that leave the process.
Writes a desensitised JSON report next to this file.

    python e2e/wire_capture.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from agentscope.agent import Agent
from agentscope.credential import OpenAICredential
from agentscope.message import UserMsg
from agentscope.model import OpenAIChatModel
from agentscope.permission import PermissionContext, PermissionMode
from agentscope.state import AgentState
from agentscope.tool import FunctionTool, Toolkit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentscope_agent_hints import HintConfig, ManageClient  # noqa: E402
from agentscope_agent_hints.adapter import (  # noqa: E402
    AgentHintMiddleware,
    hinted_model,
)
from agentscope_agent_hints.encoder import decode  # noqa: E402
from agentscope_agent_hints.protocol import HEADER_NAME, validate  # noqa: E402

LONG = "The quick brown fox jumps over the lazy dog. " * 12


def get_time() -> str:
    """Return the current local time as a short string."""
    return "12:00"


class Stub:
    """OpenAI-compatible chat-completions stub with a scripted tool round."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # silence
                pass

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(length)
                body = json.loads(raw or b"{}")
                hint = self.headers.get(HEADER_NAME)
                outer.records.append(
                    {
                        "path": self.path,
                        "hint": decode(hint) if hint else None,
                        "body_keys": sorted(body),
                        "message_count": len(body.get("messages") or []),
                        "roles": [
                            (message or {}).get("role")
                            for message in body.get("messages") or []
                        ],
                        "shapes": _shapes(body.get("messages") or []),
                        "model": body.get("model"),
                        "manage": not body.get("messages"),
                    },
                )

                if not body.get("messages"):
                    payload = {"ok": True}
                elif any(
                    isinstance(message, dict) and message.get("role") == "tool"
                    for message in body.get("messages") or []
                ):
                    payload = _completion("Final answer: 12:00")
                else:
                    payload = _completion(
                        None,
                        tool_calls=[
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_time",
                                    "arguments": "{}",
                                },
                            },
                        ],
                    )
                encoded = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> Stub:
        self.thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.server.shutdown()
        self.server.server_close()

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/v1"


def _shapes(messages: list[Any]) -> list[dict[str, Any]]:
    """Compact structural summary of a message list (no content)."""
    shapes: list[dict[str, Any]] = []
    for message in messages:
        content = (message or {}).get("content")
        kinds: list[str] = []
        if isinstance(content, str):
            kinds.append("str")
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    kinds.append(str(part.get("type") or "dict"))
                else:
                    kinds.append(type(part).__name__)
        elif content is None:
            kinds.append("none")
        else:
            kinds.append(type(content).__name__)
        shapes.append(
            {
                "role": (message or {}).get("role"),
                "content": kinds,
                "tool_calls": len((message or {}).get("tool_calls") or []),
                "tool_call_id": bool((message or {}).get("tool_call_id")),
                "name": (message or {}).get("name"),
            },
        )
    return shapes


def _completion(content: str | None, tool_calls: list[dict] | None = None) -> dict:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
        message["content"] = None
    return {
        "id": "chatcmpl-stub",
        "object": "chat.completion",
        "created": 0,
        "model": "stub-model",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            },
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


async def main() -> None:
    with Stub() as stub:
        model = OpenAIChatModel(
            credential=OpenAICredential(api_key="sk-test", base_url=stub.base_url),
            model="stub-model",
            stream=False,
        )
        manage = ManageClient(
            endpoint=f"{stub.base_url}/chat/completions",
            model="stub-model",
        )
        middleware = AgentHintMiddleware(HintConfig(), manage)
        agent = Agent(
            name="capture-agent",
            system_prompt="You are a concise assistant.",
            model=model,
            toolkit=Toolkit(tools=[FunctionTool(get_time)]),
            state=AgentState(
                permission_context=PermissionContext(mode=PermissionMode.DONT_ASK),
            ),
            middlewares=[middleware],
        )

        await agent.reply(UserMsg(name="user", content=LONG))
        await agent.reply(UserMsg(name="user", content="And the time again?"))

        await middleware.pause(agent)
        await middleware.resume(agent)
        await middleware.stop(agent)

        # Background work must stay hint-free. A dedicated auxiliary model
        # mirrors how AgentScope uses `agent.model` for compression.
        auxiliary = hinted_model(
            OpenAIChatModel(
                credential=OpenAICredential(api_key="sk-test", base_url=stub.base_url),
                model="stub-model",
                stream=False,
            ),
            auxiliary=True,
        )
        await auxiliary([UserMsg(name="user", content="Summarise the context")])

    report = {
        "harness": "agentscope",
        "harness_version": _version("agentscope"),
        "protocol": "v1.0",
        "requests": stub.records,
        "checks": {
            "every_hint_valid": all(
                record["hint"] is None or not validate(record["hint"])
                for record in stub.records
            ),
            "body_has_no_hint_fields": all(
                not ({"session", "priority_control", "cache_control", "context_management"}
                     & set(record["body_keys"]))
                for record in stub.records
            ),
            "auxiliary_request_has_no_hint": stub.records[-1]["hint"] is None,
            "states": [
                (record["hint"] or {}).get("session", {}).get("state")
                for record in stub.records
            ],
        },
    }
    out = Path(__file__).resolve().parent / "evidence" / "agentscope.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["checks"], indent=2))
    print(f"wrote {out}")


def _version(name: str) -> str:
    from importlib.metadata import version

    return version(name)


if __name__ == "__main__":
    asyncio.run(main())
