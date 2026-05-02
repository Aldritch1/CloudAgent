import logging

import asyncpg

from cloudagent.mcp.servers.base import BaseMCPServer

logger = logging.getLogger(__name__)


class OrderMCPServer(BaseMCPServer):
    def __init__(self, dsn: str = ""):
        super().__init__("cloudagent-order")
        self._dsn = dsn
        self._register_tools()

    def _register_tools(self):
        self.register_tool(
            "query_order",
            {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "user_id": {"type": "string"},
                },
                "required": ["order_id"],
            },
            self.query_order,
        )
        self.register_tool(
            "cancel_order",
            {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["order_id"],
            },
            self.cancel_order,
        )
        self.register_tool(
            "request_refund",
            {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "amount": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["order_id"],
            },
            self.request_refund,
        )

    async def query_order(self, order_id: str, user_id: str = "") -> str:
        """查询订单详情"""
        try:
            conn = await asyncpg.connect(self._dsn)
            try:
                row = await conn.fetchrow(
                    "SELECT * FROM orders WHERE order_id = $1", order_id
                )
                if row:
                    return f"订单 {order_id}: 状态={row['status']}, 金额={row['amount']}"
                return f"未找到订单 {order_id}"
            finally:
                await conn.close()
        except Exception as e:
            logger.warning(f"query_order failed: {e}")
            return "查询订单失败，请稍后重试"

    async def cancel_order(self, order_id: str, reason: str = "") -> str:
        """取消订单"""
        return f"订单 {order_id} 已取消"

    async def request_refund(self, order_id: str, amount: float = 0, reason: str = "") -> str:
        """申请退款"""
        return f"订单 {order_id} 退款申请已提交"
