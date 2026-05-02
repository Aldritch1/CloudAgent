import logging

import asyncpg

from cloudagent.mcp.servers.base import BaseMCPServer

logger = logging.getLogger(__name__)


class TicketMCPServer(BaseMCPServer):
    def __init__(self, dsn: str = ""):
        super().__init__("cloudagent-ticket")
        self._dsn = dsn
        self._register_tools()

    def _register_tools(self):
        self.register_tool(
            "create_ticket",
            {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "category": {"type": "string", "enum": ["refund", "complaint", "inquiry"]},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                },
                "required": ["user_id", "category", "title"],
            },
            self.create_ticket,
        )
        self.register_tool(
            "query_ticket",
            {
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string"},
                    "user_id": {"type": "string"},
                },
                "required": ["ticket_id"],
            },
            self.query_ticket,
        )

    async def create_ticket(self, user_id: str, category: str, title: str,
                           description: str = "", priority: str = "medium") -> str:
        """创建客服工单"""
        try:
            conn = await asyncpg.connect(self._dsn)
            try:
                row = await conn.fetchrow(
                    """
                    INSERT INTO tickets (user_id, category, title, description, priority, status, created_at)
                    VALUES ($1, $2, $3, $4, $5, 'open', NOW())
                    RETURNING ticket_id
                    """,
                    user_id, category, title, description, priority,
                )
                return f"工单 {row['ticket_id']} 已创建"
            finally:
                await conn.close()
        except Exception as e:
            logger.warning(f"create_ticket failed: {e}")
            return "工单创建失败"

    async def query_ticket(self, ticket_id: str, user_id: str = "") -> str:
        """查询工单状态"""
        try:
            conn = await asyncpg.connect(self._dsn)
            try:
                row = await conn.fetchrow(
                    "SELECT * FROM tickets WHERE ticket_id = $1", ticket_id
                )
                if row:
                    return f"工单 {ticket_id}: 状态={row['status']}, 类别={row['category']}"
                return f"未找到工单 {ticket_id}"
            finally:
                await conn.close()
        except Exception as e:
            logger.warning(f"query_ticket failed: {e}")
            return "查询工单失败"
