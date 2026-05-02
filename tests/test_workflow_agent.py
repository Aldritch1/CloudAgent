import os
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from cloudagent.agent.workflow_agent import WorkflowAgent


@pytest.mark.asyncio
@patch("cloudagent.agent.workflow_agent.ChatOpenAI")
async def test_workflow_agent_calls_tool(mock_llm_class):
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[
        MagicMock(content='{"action": "order:query_order", "action_input": {"order_id": "12345"}}'),
        MagicMock(content="订单已发货"),
    ])
    mock_llm_class.return_value = mock_llm

    mock_mcp = MagicMock()
    mock_mcp.list_tools.return_value = [
        {"name": "order:query_order", "description": "查询订单", "parameters": {}}
    ]
    mock_mcp.call_tool = AsyncMock(return_value="订单已发货")

    agent = WorkflowAgent(model_name="gpt-test", api_key="test-key", mcp_client=mock_mcp)

    result = await agent.run({"messages": [{"role": "user", "content": "查订单"}]})
    mock_mcp.call_tool.assert_called_once_with("order:query_order", {"order_id": "12345"})
    assert "订单已发货" in result


@pytest.mark.asyncio
@patch("cloudagent.agent.workflow_agent.ChatOpenAI")
async def test_workflow_agent_no_tool_needed(mock_llm_class):
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
        content='{"final_answer": "请问有什么可以帮您？"}'
    ))
    mock_llm_class.return_value = mock_llm

    mock_mcp = MagicMock()
    mock_mcp.list_tools.return_value = []

    agent = WorkflowAgent(model_name="gpt-test", api_key="test-key", mcp_client=mock_mcp)

    result = await agent.run({"messages": [{"role": "user", "content": "你好"}]})
    assert "请问有什么可以帮您？" in result
