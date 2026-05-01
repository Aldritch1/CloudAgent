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
  "target_agent": "chat|faq|workflow"
}}

Intent definitions:
- "chat": casual conversation, greetings, small talk, general chitchat
- "faq": knowledge questions about policies, refunds, shipping, pricing, product info
- "workflow": business transactions like order queries, refunds, ticket creation

Rules:
- confidence > 0.8: user intent is clearly one of the above
- confidence <= 0.5: unclear or unrelated, fallback to chat agent
- Set target_agent to the most appropriate agent for the intent.

User message: {message}

Output JSON only, no markdown, no explanation."""


class EntryAgent:
    def __init__(self, model_name: str, api_key: str):
        self._llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.0,
            max_tokens=128,
        )

    def run(self, state: dict) -> dict:
        user_message = ""
        for msg in reversed(state["messages"]):
            if msg["role"] == "user":
                user_message = msg["content"]
                break

        prompt = INTENT_PROMPT.format(message=user_message)
        messages = [SystemMessage(content=prompt)]

        try:
            response = self._llm.invoke(messages)
            parsed = json.loads(response.content.strip())
            state["intent"] = parsed.get("intent", "chat")
            state["confidence"] = float(parsed.get("confidence", 0.0))
            state["target_agent"] = parsed.get("target_agent", "chat")
        except Exception as e:
            logger.error(f"Intent recognition failed: {e}")
            state["intent"] = "chat"
            state["confidence"] = 0.0
            state["target_agent"] = "chat"

        # Routing logic: phase2 has chat, faq, and workflow intents
        if state["confidence"] <= 0.5:
            state["target_agent"] = "chat"

        return state
