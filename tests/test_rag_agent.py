from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cloudagent.agent.rag_agent import RAGAgent


@pytest.mark.asyncio
async def test_rag_agent_includes_context_in_prompt():
    mock_retriever = AsyncMock()
    mock_retriever.search.return_value = [
        MagicMock(content="支持7天无理由退款", source="vector", metadata={}),
    ]

    with patch("cloudagent.agent.rag_agent.ChatOpenAI") as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="我们支持7天无理由退款。"))
        mock_llm_cls.return_value = mock_llm

        agent = RAGAgent(model_name="gpt-test", api_key="test-key", retriever=mock_retriever)

        result = await agent.run({
            "messages": [
                {"role": "user", "content": "怎么退款？"},
            ],
        })

        assert result == "我们支持7天无理由退款。"
        mock_retriever.search.assert_called_once_with("怎么退款？", top_k=5)

        call_messages = mock_llm.ainvoke.call_args[0][0]
        assert "支持7天无理由退款" in call_messages[0].content
        assert call_messages[1].content == "怎么退款？"


@pytest.mark.asyncio
async def test_rag_agent_empty_context():
    mock_retriever = AsyncMock()
    mock_retriever.search.return_value = []

    with patch("cloudagent.agent.rag_agent.ChatOpenAI") as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="抱歉，我暂时无法回答。"))
        mock_llm_cls.return_value = mock_llm

        agent = RAGAgent(model_name="gpt-test", api_key="test-key", retriever=mock_retriever)

        result = await agent.run({
            "messages": [{"role": "user", "content": "未知问题"}],
        })

        assert result == "抱歉，我暂时无法回答。"
