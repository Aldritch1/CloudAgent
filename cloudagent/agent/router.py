import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

INTENT_PROMPT = """You are an intent classifier for a customer service system.
Analyze the user's message and output ONLY a JSON object with this exact schema:
{{
  "intent": "chat|faq|workflow",
  "confidence": 0.0-1.0,
  "target_agent": "chat|faq|workflow|clarify",
  "clarification_question": "optional question string"
}}

Intent definitions:
- "chat": casual conversation, greetings, small talk, general chitchat
- "faq": knowledge questions about policies, shipping rules, pricing, product info (asking "how to" or "what is" without taking action)
- "workflow": business transactions requiring action like order queries, refunds, returns, cancellations, ticket creation (user wants to DO something, not just ask about it)

Examples:
- "我要退货" → workflow (user wants to perform a return)
- "怎么退货" → faq (user is asking about the process)
- "申请退款" → workflow (user wants to take action)
- "退款政策是什么" → faq (user is asking about policy)
- "查一下我的订单" → workflow (user wants to query an order)
- "订单多久发货" → faq (user is asking about shipping time)

Rules:
- confidence > 0.8: user intent is clearly one of the above, set target_agent = intent
- 0.5 < confidence <= 0.8: intent is somewhat unclear, set target_agent = "clarify" and provide a brief clarification_question
- confidence <= 0.5: unclear or unrelated, fallback to chat agent

User message: {message}

Output JSON only, no markdown, no explanation."""


class EntryAgent:
    def __init__(self, model_name: str, api_key: str, base_url: str = None):
        self._llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
            max_tokens=128,
        )

    _WORKFLOW_KEYWORDS = ["退货", "退款", "查订单", "取消订单", "工单", "投诉", "发货", "改地址"]
    _FAQ_KEYWORDS = ["怎么", "如何", "什么是", "多久", "多少钱", "政策", "规则"]

    def run(self, state: dict) -> dict:
        msgs = state.get("messages", [])
        user_message = ""
        for msg in reversed(msgs):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        # Keyword-based fallback routing
        has_workflow_kw = any(kw in user_message for kw in self._WORKFLOW_KEYWORDS)
        has_faq_kw = any(kw in user_message for kw in self._FAQ_KEYWORDS)

        prompt = INTENT_PROMPT.format(message=user_message)
        messages = [SystemMessage(content=prompt)]

        try:
            response = self._llm.invoke(messages)
            parsed = json.loads(response.content.strip())
            intent = parsed.get("intent", "chat")
            confidence = float(parsed.get("confidence", 0.0))
            target_agent = parsed.get("target_agent", "chat")
        except Exception as e:
            logger.error(f"Intent recognition failed: {e}")
            intent = "chat"
            confidence = 0.0
            target_agent = "chat"
            parsed = {}

        # Override with keyword rules when LLM is uncertain or keywords are strong signals
        if has_workflow_kw and not has_faq_kw:
            intent = "workflow"
            target_agent = "workflow"
            confidence = max(confidence, 0.92)
        elif has_faq_kw and not has_workflow_kw:
            intent = "faq"
            target_agent = "faq"
            confidence = max(confidence, 0.92)

        state["intent"] = intent
        state["confidence"] = confidence
        state["target_agent"] = target_agent

        # Routing logic: phase3 adds clarify for mid-confidence
        if 0.5 < state["confidence"] <= 0.8:
            state["target_agent"] = "clarify"
            state["clarification_question"] = parsed.get("clarification_question", "能再详细说明一下吗？")
        elif state["confidence"] <= 0.5:
            state["target_agent"] = "chat"

        return state
