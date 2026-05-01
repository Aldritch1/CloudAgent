from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


SYSTEM_PROMPT = """You are a helpful customer service assistant.
Answer user questions politely and concisely in Chinese.
If you don't know something, say so honestly."""


class ChatAgentError(Exception):
    """Domain-specific exception for ChatAgent failures."""
    pass


class ChatAgent:
    def __init__(self, model_name: str, api_key: str, breaker=None):
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.7,
        )
        if breaker is not None:
            from cloudagent.circuit_breaker import CircuitBreakerChatOpenAI
            self._llm = CircuitBreakerChatOpenAI(llm, breaker)
        else:
            self._llm = llm

    def _convert_messages(self, messages: list[dict]) -> list:
        converted = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in messages:
            if msg["role"] == "user":
                converted.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                converted.append(AIMessage(content=msg["content"]))
        return converted

    def run(self, messages: list[dict]) -> str:
        from cloudagent.metrics import record_llm_call
        converted = self._convert_messages(messages)
        try:
            response = self._llm.invoke(converted)
            record_llm_call("chat", "success")
        except Exception as exc:
            record_llm_call("chat", "failure")
            raise ChatAgentError(f"LLM invocation failed: {exc}") from exc
        return response.content
