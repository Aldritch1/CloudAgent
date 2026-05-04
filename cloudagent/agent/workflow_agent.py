import json
import logging
from collections.abc import AsyncIterator

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

logger = logging.getLogger(__name__)

WORKFLOW_SYSTEM_PROMPT = """你是 CloudAgent 的业务办理助手。你可以使用以下工具帮助用户：

{tools}

请根据用户需求选择合适工具，输出 JSON 格式的工具调用：
{{
  "thought": "思考过程",
  "action": "tool_name",
  "action_input": {{"param": "value"}}
}}

或直接回复用户（无需工具时）：
{{
  "thought": "思考过程",
  "final_answer": "回复内容"
}}
"""


class WorkflowAgent:
    def __init__(self, model_name: str, api_key: str, mcp_client=None):
        self._llm = ChatOpenAI(model=model_name, api_key=api_key, temperature=0.3)
        self._mcp_client = mcp_client

    def _build_tool_descriptions(self) -> str:
        if self._mcp_client is None:
            return "（暂无可用工具）"
        tools = self._mcp_client.list_tools()
        lines = []
        for t in tools:
            lines.append(f"- {t['name']}: {t['description']}")
            lines.append(f"  参数: {json.dumps(t['parameters'], ensure_ascii=False)}")
        return "\n".join(lines)

    async def run(self, state: dict) -> str:
        tools_desc = self._build_tool_descriptions()
        system_prompt = WORKFLOW_SYSTEM_PROMPT.format(tools=tools_desc)

        messages = [SystemMessage(content=system_prompt)]
        for msg in state.get("messages", []):
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        try:
            response = await self._llm.ainvoke(messages)
            content = response.content.strip()

            if content.startswith("{"):
                parsed = json.loads(content)
                if "action" in parsed and self._mcp_client:
                    result = await self._mcp_client.call_tool(
                        parsed["action"], parsed.get("action_input", {})
                    )
                    messages.append(AIMessage(content=content))
                    messages.append(ToolMessage(content=result, tool_call_id=parsed["action"]))
                    follow_up = await self._llm.ainvoke(messages)
                    return follow_up.content
                elif "final_answer" in parsed:
                    return parsed["final_answer"]
            return content
        except Exception as e:
            logger.error(f"Workflow agent failed: {e}")
            return "业务办理暂时无法完成，请稍后重试。"

    async def run_stream(self, state: dict) -> AsyncIterator[dict]:
        """Yield dict events: {event: 'token', data: str} or {event: 'tool_call', data: str} or {event: 'tool_result', data: str}"""
        tools_desc = self._build_tool_descriptions()
        system_prompt = WORKFLOW_SYSTEM_PROMPT.format(tools=tools_desc)

        messages = [SystemMessage(content=system_prompt)]
        for msg in state.get("messages", []):
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        try:
            response = await self._llm.ainvoke(messages)
            content = response.content.strip()

            if content.startswith("{"):
                import json
                parsed = json.loads(content)
                if "action" in parsed and self._mcp_client:
                    yield {"event": "tool_call", "data": json.dumps({"tool": parsed["action"], "args": parsed.get("action_input", {})})}
                    result = await self._mcp_client.call_tool(
                        parsed["action"], parsed.get("action_input", {})
                    )
                    yield {"event": "tool_result", "data": result}
                    messages.append(AIMessage(content=content))
                    messages.append(ToolMessage(content=result, tool_call_id=parsed["action"]))
                    async for chunk in self._llm.astream(messages):
                        yield {"event": "token", "data": chunk.content}
                elif "final_answer" in parsed:
                    yield {"event": "token", "data": parsed["final_answer"]}
                else:
                    yield {"event": "token", "data": content}
            else:
                yield {"event": "token", "data": content}
        except Exception as e:
            logger.error(f"Workflow agent stream failed: {e}")
            yield {"event": "token", "data": "业务办理暂时无法完成，请稍后重试。"}
