from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cloudagent.mcp.client import MCPClient
from mcp.types import TextContent


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
    session.call_tool = AsyncMock(return_value=MagicMock(content=[]))
    session.close = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_client_lists_tools(mock_session):
    client = MCPClient()
    client._sessions["order"] = mock_session
    client._tool_registry["order:query_order"] = {
        "server": "order", "name": "query_order",
        "description": "查询订单", "schema": {},
    }

    tools = client.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "order:query_order"


@pytest.mark.asyncio
async def test_client_calls_tool(mock_session):
    client = MCPClient()
    client._sessions["order"] = mock_session
    client._tool_registry["order:query_order"] = {
        "server": "order", "name": "query_order",
        "description": "查询订单", "schema": {},
    }
    mock_session.call_tool = AsyncMock(
        return_value=MagicMock(content=[TextContent(type="text", text="订单已发货")])
    )

    result = await client.call_tool("order:query_order", {"order_id": "12345"})
    assert "订单已发货" in result
