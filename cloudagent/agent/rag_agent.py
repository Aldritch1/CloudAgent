import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """你是 CloudAgent 智能客服助手。请根据以下参考资料回答用户问题。
如果参考资料不足以回答，请坦诚告知用户。

参考资料：
{context}
"""


class RAGAgent:
    def __init__(self, model_name: str, api_key: str, retriever, breaker=None):
        llm = ChatOpenAI(model=model_name, api_key=api_key, temperature=0.3)
        if breaker is not None:
            from cloudagent.circuit_breaker import CircuitBreakerChatOpenAI
            self._llm = CircuitBreakerChatOpenAI(llm, breaker)
        else:
            self._llm = llm
        self._retriever = retriever

    @staticmethod
    def _extract_last_user(messages: list[dict]) -> str:
        for msg in reversed(messages):
            if msg["role"] == "user":
                return msg["content"]
        return ""

    @staticmethod
    def _convert_messages(messages: list[dict]):
        lc_messages = []
        for m in messages:
            if m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                lc_messages.append(AIMessage(content=m["content"]))
            elif m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
        return lc_messages

    async def run(self, state: dict) -> str:
        last_user_msg = self._extract_last_user(state["messages"])
        contexts = await self._retriever.search(last_user_msg, top_k=5)
        context_text = "\n\n".join([c.content for c in contexts])

        system_prompt = RAG_SYSTEM_PROMPT.format(context=context_text)
        lc_messages = [SystemMessage(content=system_prompt)]
        lc_messages.extend(self._convert_messages(state["messages"]))

        try:
            response = await self._llm.ainvoke(lc_messages)
            return response.content
        except Exception as e:
            logger.error(f"RAG agent failed: {e}")
            raise
