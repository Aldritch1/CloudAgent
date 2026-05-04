import os
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from cloudagent.agent.chat_agent import ChatAgent


async def _async_generator(items):
    for item in items:
        yield MagicMock(content=item)


@pytest.mark.asyncio
async def test_chat_agent_stream_yields_tokens():
    agent = ChatAgent(model_name="gpt-test", api_key="test-key")

    async def mock_astream(*args, **kwargs):
        for item in ["Hello", " ", "world"]:
            yield MagicMock(content=item)

    mock_llm = MagicMock()
    mock_llm.astream = mock_astream
    agent._llm = mock_llm

    tokens = []
    async for token in agent.run_stream([{"role": "user", "content": "hi"}]):
        tokens.append(token)

    assert tokens == ["Hello", " ", "world"]
