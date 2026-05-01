# CloudAgent 阶段5设计文档 — MCP 工具生态

**版本：** v1.0  
**日期：** 2026-05-01  
**目标：** 通过 MCP（Model Context Protocol）将 Workflow Agent 从占位符升级为可调用真实业务工具的智能体，实现订单查询、短信通知、工单创建等客服核心能力

---

## 1. 总体架构

阶段5在阶段4的 LangGraph + 生产加固基础上，引入 **MCP（Model Context Protocol）** 作为 Agent 与外部业务系统的标准连接协议。CloudAgent 作为 MCP Host，内置 Client 与多个 MCP Server 通信，Workflow Agent 通过 tool-calling 模式执行业务操作。

```
┌─────────────────────────────────────────┐
│ API网关层：FastAPI + JWT + 限流 + 多租户    │
├─────────────────────────────────────────┤
│ Agent引擎层：LangGraph StateGraph         │
│ 职责：entry → route → chat / rag /        │
│       workflow(MCP tools) / clarify / HITL│
├─────────────────────────────────────────┤
│ Workflow Agent 层                         │
│ 职责：LLM 决策 → MCP tool call →          │
│       业务执行 → 结果汇总 → 用户回复       │
├─────────────────────────────────────────┤
│ MCP Client 层                             │
│ 职责：Server 发现、Tool 列表同步、         │
│       JSON-RPC 调用、结果反序列化          │
├─────────────────────────────────────────┤
│ MCP Server 层（内置）                      │
│ ├─ Order Server   订单查询/取消/退款       │
│ ├─ SMS Server     短信发送                 │
│ └─ Ticket Server  工单创建/查询            │
├─────────────────────────────────────────┤
│ 业务数据层（现有 PG / 外部 API）            │
└─────────────────────────────────────────┘
```

---

## 2. 基础设施

### 2.1 依赖扩展

`pyproject.toml` 新增：

```toml
dependencies = [
    # ... 现有依赖 ...
    "mcp>=1.0.0",
]
```

> `mcp` 为 Anthropic 官方 Python SDK，提供 `Server`、`ClientSession`、`stdio_client` 等核心组件。

### 2.2 配置扩展

`cloudagent/config.py` 新增：

```python
class Settings(BaseSettings):
    # ... 现有配置 ...
    mcp_servers: str = "order,sms,ticket"  # 逗号分隔的启用的 server 列表
    order_service_url: str = ""
    sms_service_url: str = ""
    ticket_service_url: str = ""
```

---

## 3. 文件结构

```
cloudagent/
├── mcp/
│   ├── __init__.py
│   ├── client.py            # NEW: MCPClient manager
│   └── servers/
│       ├── __init__.py
│       ├── base.py          # NEW: BaseMCPServer
│       ├── order.py         # NEW: OrderMCPServer
│       ├── sms.py           # NEW: SMSMCPServer
│       └── ticket.py        # NEW: TicketMCPServer
├── agent/
│   ├── router.py            # MODIFIED: workflow 路由至真实 agent
│   ├── chat_agent.py        # 不变
│   ├── rag_agent.py         # 不变
│   └── workflow_agent.py    # NEW: Tool-calling workflow agent
├── graph.py                 # MODIFIED: workflow_node 替换 placeholder
├── main.py                  # MODIFIED: 初始化 MCP client + servers
└── state.py                 # 不变 (AgentState 已含 tenant_id)

tests/
├── test_workflow_agent.py   # NEW: Tool-calling + 参数提取测试
├── test_mcp_client.py       # NEW: MCP client + server 集成测试
├── test_mcp_servers.py      # NEW: Order/SMS/Ticket server 单元测试
├── test_graph.py            # MODIFIED: workflow 节点流程测试
└── test_main.py             # MODIFIED: workflow 端到端测试
```

---

## 4. 模块设计

### 4.1 MCP Base Server (`mcp/servers/base.py`)

所有内置 MCP Server 的基类，封装 `mcp.Server` 的通用逻辑：

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

class BaseMCPServer:
    def __init__(self, name: str):
        self._server = Server(name)
        self._tools: dict[str, callable] = {}
        self._register_handlers()

    def _register_handlers(self):
        @self._server.list_tools()
        async def list_tools() -> list[Tool]:
            return [Tool(name=name, description=fn.__doc__, inputSchema=schema)
                    for name, (fn, schema) in self._tools.items()]

        @self._server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            if name not in self._tools:
                raise ValueError(f"Unknown tool: {name}")
            fn, _ = self._tools[name]
            result = await fn(**arguments)
            return [TextContent(type="text", text=str(result))]

    def register_tool(self, name: str, schema: dict, fn: callable):
        self._tools[name] = (fn, schema)

    async def run(self):
        """启动 server（stdio 或 in-memory transport）"""
        pass
```

### 4.2 Order Server (`mcp/servers/order.py`)

```python
class OrderMCPServer(BaseMCPServer):
    def __init__(self, dsn: str = ""):
        super().__init__("cloudagent-order")
        self._dsn = dsn
        self._register_tools()

    def _register_tools(self):
        self.register_tool(
            "query_order",
            {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "user_id": {"type": "string"},
                },
                "required": ["order_id"],
            },
            self.query_order,
        )
        self.register_tool(
            "cancel_order",
            {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["order_id"],
            },
            self.cancel_order,
        )
        self.register_tool(
            "request_refund",
            {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "amount": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["order_id"],
            },
            self.request_refund,
        )

    async def query_order(self, order_id: str, user_id: str = "") -> str:
        """查询订单详情"""
        # 实际实现：查询 PostgreSQL 或调用外部订单服务
        pass

    async def cancel_order(self, order_id: str, reason: str = "") -> str:
        """取消订单"""
        pass

    async def request_refund(self, order_id: str, amount: float = 0, reason: str = "") -> str:
        """申请退款"""
        pass
```

### 4.3 SMS Server (`mcp/servers/sms.py`)

```python
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
        pass
```

### 4.4 Ticket Server (`mcp/servers/ticket.py`)

```python
class TicketMCPServer(BaseMCPServer):
    def __init__(self, dsn: str = ""):
        super().__init__("cloudagent-ticket")
        self._dsn = dsn
        self._register_tools()

    def _register_tools(self):
        self.register_tool(
            "create_ticket",
            {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "category": {"type": "string", "enum": ["refund", "complaint", "inquiry"]},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                },
                "required": ["user_id", "category", "title"],
            },
            self.create_ticket,
        )
        self.register_tool(
            "query_ticket",
            {
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string"},
                    "user_id": {"type": "string"},
                },
                "required": ["ticket_id"],
            },
            self.query_ticket,
        )

    async def create_ticket(self, user_id: str, category: str, title: str,
                           description: str = "", priority: str = "medium") -> str:
        """创建客服工单"""
        pass

    async def query_ticket(self, ticket_id: str, user_id: str = "") -> str:
        """查询工单状态"""
        pass
```

### 4.5 MCP Client (`mcp/client.py`)

```python
import logging
from typing import Any

from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.types import TextContent

logger = logging.getLogger(__name__)


class MCPClient:
    """Manages connections to multiple MCP servers and exposes unified tool interface."""

    def __init__(self):
        self._sessions: dict[str, ClientSession] = {}
        self._tool_registry: dict[str, dict] = {}

    async def connect_server(self, name: str, command: str, args: list[str] = None):
        """Connect to an MCP server via stdio."""
        params = StdioServerParameters(command=command, args=args or [])
        transport = stdio_client(params)
        read, write = await transport.__aenter__()
        session = ClientSession(read, write)
        await session.initialize()
        self._sessions[name] = session

        tools = await session.list_tools()
        for tool in tools.tools:
            full_name = f"{name}:{tool.name}"
            self._tool_registry[full_name] = {
                "server": name,
                "name": tool.name,
                "description": tool.description,
                "schema": tool.inputSchema,
            }
        logger.info(f"Connected to MCP server '{name}' with {len(tools.tools)} tools")

    async def call_tool(self, full_name: str, arguments: dict) -> str:
        """Call a tool by its fully qualified name (server:tool)."""
        if full_name not in self._tool_registry:
            raise ValueError(f"Unknown tool: {full_name}")
        meta = self._tool_registry[full_name]
        session = self._sessions[meta["server"]]
        result = await session.call_tool(meta["name"], arguments)
        texts = [item.text for item in result.content if isinstance(item, TextContent)]
        return "\n".join(texts)

    def list_tools(self) -> list[dict]:
        """List all available tools across all servers."""
        return [
            {
                "name": name,
                "description": meta["description"],
                "parameters": meta["schema"],
            }
            for name, meta in self._tool_registry.items()
        ]

    async def close(self):
        for session in self._sessions.values():
            await session.close()
```

### 4.6 Workflow Agent (`agent/workflow_agent.py`)

Tool-calling agent，使用 LangChain 的 `bind_tools` + ReAct 模式：

```python
import json
import logging

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

            # Try to parse as JSON tool call
            if content.startswith("{"):
                parsed = json.loads(content)
                if "action" in parsed and self._mcp_client:
                    result = await self._mcp_client.call_tool(
                        parsed["action"], parsed.get("action_input", {})
                    )
                    # Append tool result and ask LLM to summarize
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
```

### 4.7 Graph 节点更新 (`graph.py`)

替换 `workflow_placeholder_node` 为真实的 `workflow_node`：

```python
class GraphNodes:
    def __init__(self, ... workflow_agent=None, ...):
        # ... existing agents ...
        self.workflow_agent = workflow_agent

    # ... existing nodes ...

    async def workflow_node(self, state: AgentState) -> AgentState:
        if self.workflow_agent is not None:
            response = await self.workflow_agent.run(state)
        else:
            response = "业务办理功能正在开发中，请稍后再试。"
        state["response"] = response
        return state
```

条件路由更新：

```python
def route_node(self, state: AgentState) -> str:
    target = state.get("target_agent")
    confidence = state.get("confidence", 0.0)

    if target == "clarify":
        return "clarify"
    if target == "workflow" and self.hitl.is_sensitive("workflow", {}):
        return "hitl_request"
    if target == "workflow":
        return "workflow"  # 替换原来的 "workflow_placeholder"
    if target in ("chat", "faq") and confidence > 0.5:
        return target
    return "chat"
```

Graph builder 更新：

```python
builder.add_node("workflow", nodes.workflow_node)
# 替换条件路由中的 workflow_placeholder → workflow
builder.add_conditional_edges(
    "entry",
    nodes.route_node,
    {
        "chat": "chat",
        "faq": "rag",
        "workflow": "workflow",  # 新增真实 workflow 节点
        "hitl_request": "hitl_request",
        "clarify": "clarify",
    },
)
builder.add_edge("workflow", "save_memory")
```

### 4.8 HITL 敏感操作检测增强

`hitl.py` 增加对 MCP tool 的敏感度检测：

```python
class HITLManager:
    SENSITIVE_ACTIONS = {"refund", "cancel", "delete", "request_refund", "cancel_order"}
    # ... 其余不变 ...

    def is_sensitive_tool(self, tool_name: str) -> bool:
        return tool_name in self.SENSITIVE_ACTIONS
```

在 `workflow_node` 中，若 LLM 选择的 tool 属于敏感操作，则触发 HITL 中断。

---

## 5. 数据流

```
用户："帮我查一下订单 12345 的状态"

  1. EntryAgent 识别 intent="workflow", confidence=0.92, target_agent="workflow"
  2. route_node: workflow → 非敏感 → "workflow"
  3. workflow_node:
     3a. WorkflowAgent.run(state)
     3b. LLM 输出 tool call: {"action": "order:query_order", "action_input": {"order_id": "12345"}}
     3c. MCPClient.call_tool("order:query_order", {...})
     3d. OrderMCPServer.query_order → 查询 PG / 外部 API
     3e. 结果返回："订单 12345 已发货，预计明天送达"
     3f. WorkflowAgent 汇总结果 → "您的订单 12345 已发货，预计明天送达"
  4. save_memory_node 持久化
  5. 返回 ChatResponse

用户："我要退款"

  1. EntryAgent 识别 intent="workflow", confidence=0.88
  2. route_node: workflow → HITL 检测敏感 → "hitl_request"
  3. hitl_request_node: action_required="confirm"
  4. 图 INTERRUPT，返回确认请求

用户："确认"

  5. 图恢复，workflow_node 执行
  6. LLM 输出 tool call: {"action": "order:request_refund", ...}
  7. MCPClient 调用 → 退款申请提交
  8. 返回结果
```

---

## 6. 错误处理

| 场景 | 处理策略 |
|------|----------|
| MCP Server 未启动 | WorkflowAgent 降级为直接回复，提示"业务系统暂时不可用" |
| Tool 调用参数错误 | 捕获 `ValidationError`，提示用户补充必要信息 |
| Tool 执行失败（如订单不存在） | 返回错误信息给 LLM，LLM 生成友好提示 |
| MCP Client 连接超时 | 记录 error，返回降级回复 |
| 敏感 tool 未确认 | HITL 中断，要求用户确认 |

---

## 7. 测试策略

| 测试类型 | 覆盖内容 | 工具/方法 |
|----------|----------|-----------|
| 单元测试 | OrderMCPServer: query/cancel/refund | mock asyncpg / httpx |
| 单元测试 | SMSMCPServer: send_sms | mock httpx |
| 单元测试 | TicketMCPServer: create/query | mock asyncpg |
| 单元测试 | BaseMCPServer: list_tools / call_tool | 直接实例化 |
| 单元测试 | MCPClient: 连接、list_tools、call_tool | mock ClientSession |
| 单元测试 | WorkflowAgent: tool 选择、参数提取、结果汇总 | mock LLM + mock MCPClient |
| 集成测试 | Graph workflow 节点：正常流程 | InMemorySaver + mock agents |
| 集成测试 | Graph HITL + workflow tool：敏感操作中断 | InMemorySaver + mock |
| API 测试 | `/chat` workflow 意图端到端 | TestClient + patch |

---

## 8. 阶段5明确边界（不做）

- MCP Server 的分布式部署（当前为内置同进程）— 阶段6
- SSE transport 支持（当前仅用 stdio/in-memory）— 阶段6
- 第三方 MCP Server 的自动发现（如 Slack、GitHub）— 阶段6
- 前端 Vue3 + SSE 流式输出 — 阶段6
- Workflow Agent 的复杂多步规划（当前为单轮 tool call）— 阶段6+

---

## 9. 成功标准

1. `pytest tests/ -v` 全部通过（含阶段1~5测试）。
2. "查订单" 意图触发 Workflow Agent，正确调用 `order:query_order` 并返回结果。
3. "我要退款" 触发 HITL 确认流程，确认后正确调用 `order:request_refund`。
4. MCP Server 故障时 Workflow Agent 优雅降级，不中断服务。
5. 所有 tool 调用记录到 Prometheus 指标（可选扩展）。
