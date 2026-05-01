from typing import TypedDict


class AgentState(TypedDict, total=False):
    messages: list[dict]
    user_id: str
    session_id: str
    intent: str | None
    confidence: float
    target_agent: str | None
    context: dict
    retrieved_context: list[str]
    response: str | None
    clarification_question: str | None
    pending_action: dict | None
    action_required: str | None
