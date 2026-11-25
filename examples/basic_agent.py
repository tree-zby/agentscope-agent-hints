"""Minimal AgentScope agent with the Agent Hint protocol attached.

Run with a hint-aware gateway:

    pip install "agentscope-agent-hints[agentscope]"
    python examples/basic_agent.py

Environment:
    AGENT_HINT_BASE_URL  gateway base URL (must accept ``X-Agent-Hint``)
    AGENT_HINT_API_KEY   gateway key
    AGENT_HINT_MODEL     model id
"""

from __future__ import annotations

import asyncio
import os

from agentscope.agent import Agent
from agentscope.credential import OpenAICredential
from agentscope.message import UserMsg
from agentscope.model import OpenAIChatModel

from agentscope_agent_hints import HintConfig, ManageClient
from agentscope_agent_hints.adapter import AgentHintMiddleware


async def main() -> None:
    base_url = os.environ.get("AGENT_HINT_BASE_URL", "http://127.0.0.1:8080/v1")
    api_key = os.environ.get("AGENT_HINT_API_KEY", "sk-local")
    model_name = os.environ.get("AGENT_HINT_MODEL", "your-model")

    model = OpenAIChatModel(
        credential=OpenAICredential(api_key=api_key, base_url=base_url),
        model=model_name,
        stream=False,
    )

    manage = ManageClient(
        endpoint=f"{base_url.rstrip('/')}/chat/completions",
        model=model_name,
        extra_headers={"authorization": f"Bearer {api_key}"},
    )
    hints = AgentHintMiddleware(HintConfig(), manage)

    agent = Agent(
        name="assistant",
        system_prompt="You are a concise assistant.",
        model=model,
        middlewares=[hints],
    )

    await agent.reply(UserMsg(name="user", content="Hello! What can you do?"))
    await agent.reply(UserMsg(name="user", content="Give me one example."))

    # The host decides when the session is idle/archived (protocol section 7.5).
    await hints.pause(agent)
    await hints.resume(agent)
    await hints.stop(agent)


if __name__ == "__main__":
    asyncio.run(main())
