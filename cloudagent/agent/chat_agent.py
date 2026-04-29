from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


SYSTEM_PROMPT = """You are a helpful customer service assistant.
Answer user questions politely and concisely in Chinese.
If you don't know something, say so honestly."""


class ChatAgentError(Exception):
    """Domain-specific exception for ChatAgent failures."""
    pass


class ChatAgent:
    def __init__(self, model_name: str, api_key: str):
        self._llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.7,
        )

    def _convert_messages(self, messages: list[dict]) -> list:
        converted = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in messages:
            if msg["role"] == "user":
                converted.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                converted.append(AIMessage(content=msg["content"]))
        return converted

    def run(self, messages: list[dict]) -> str:
        converted = self._convert_messages(messages)
        try:
            response = self._llm.invoke(converted)
        except Exception as exc:
            raise ChatAgentError(f"LLM invocation failed: {exc}") from exc
        return response.content
