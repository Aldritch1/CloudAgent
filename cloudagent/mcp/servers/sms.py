import logging

import httpx

from cloudagent.mcp.servers.base import BaseMCPServer

logger = logging.getLogger(__name__)


class SMSMCPServer(BaseMCPServer):
    def __init__(self, api_url: str = ""):
        super().__init__("cloudagent-sms")
        self._api_url = api_url
        self._register_tools()

    def _register_tools(self):
        self.register_tool(
            "send_sms",
            {
                "type": "object",
                "properties": {
                    "phone": {"type": "string"},
                    "template": {"type": "string", "enum": ["verification", "notification", "refund_notice"]},
                    "params": {"type": "object"},
                },
                "required": ["phone", "template"],
            },
            self.send_sms,
        )

    async def send_sms(self, phone: str, template: str, params: dict = None) -> str:
        """发送短信通知"""
        try:
            if self._api_url:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f"{self._api_url}/send",
                        json={"phone": phone, "template": template, "params": params or {}},
                    )
                    resp.raise_for_status()
                    return f"短信已发送至 {phone}"
            return f"短信已发送至 {phone}（模拟模式）"
        except Exception as e:
            logger.warning(f"send_sms failed: {e}")
            return "短信发送失败"
