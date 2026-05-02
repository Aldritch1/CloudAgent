from unittest.mock import patch

import pytest

from cloudagent.mcp.servers.order import OrderMCPServer


@pytest.fixture
def order_server():
    return OrderMCPServer(dsn="postgresql://test")


@pytest.mark.asyncio
async def test_list_tools(order_server):
    tools = await order_server.list_tools()
    names = [t.name for t in tools.root.tools]
    assert "query_order" in names
    assert "cancel_order" in names
    assert "request_refund" in names


@pytest.mark.asyncio
async def test_query_order_tool(order_server):
    with patch.object(order_server, "query_order", return_value="订单已发货") as mock_query:
        result = await order_server.call_tool("query_order", {"order_id": "12345"})
        mock_query.assert_called_once_with(order_id="12345")
        assert "订单已发货" in result.root.content[0].text
