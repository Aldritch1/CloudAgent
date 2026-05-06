import json
import logging
from collections.abc import AsyncIterator

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

logger = logging.getLogger(__name__)

WORKFLOW_SYSTEM_PROMPT = """你是 CloudAgent 的业务办理助手。

当前可用工具：
{tools}

重要规则：
1. 如果上方工具列表为空（显示"暂无可用工具"），你必须直接回复用户，禁止使用任何工具
2. 你只能使用上方列表中真实存在的工具，禁止虚构不存在的工具名
3. 如需工具调用，输出严格 JSON：
{{
  "thought": "思考过程",
  "action": "tool_name",
  "action_input": {{"param": "value"}}
}}
4. 如无需工具，直接回复用户：
{{
  "thought": "思考过程",
  "final_answer": "回复内容"
}}

退货流程规则：
- 用户提出退货时，只需询问订单号即可
- 拿到订单号后，确认收到并告知正在处理，不要索要其他信息
"""


class WorkflowAgent:
    def __init__(self, model_name: str, api_key: str, mcp_client=None, base_url: str = None):
        self._llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, temperature=0.3)
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
