import logging

logger = logging.getLogger(__name__)


class HITLManager:
    """Human-in-the-Loop manager for sensitive operations."""

    SENSITIVE_ACTIONS = {"refund", "cancel", "delete"}
    CONFIRM_KEYWORDS = {"确认", "是的", "confirm", "yes", "ok"}
    REJECT_KEYWORDS = {"取消", "拒绝", "reject", "no", "cancel"}

    def is_sensitive(self, intent: str, params: dict) -> bool:
        action = params.get("action", intent)
        return action in self.SENSITIVE_ACTIONS

    def build_confirmation_message(self, action: str, params: dict) -> str:
        return f"您即将执行敏感操作：{action}，请回复'确认'继续或'取消'放弃。"

    def is_confirm(self, message: str) -> bool:
        return any(kw in message.lower() for kw in self.CONFIRM_KEYWORDS)

    def is_reject(self, message: str) -> bool:
        return any(kw in message.lower() for kw in self.REJECT_KEYWORDS)
