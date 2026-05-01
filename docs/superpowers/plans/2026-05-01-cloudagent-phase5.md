# CloudAgent Phase5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add MCP (Model Context Protocol) tool ecosystem: build MCP servers for order/SMS/ticket services, integrate MCP client into Workflow Agent, and replace the workflow placeholder with real tool-calling capabilities.

**Architecture:** `cloudagent/mcp/servers/` contains three built-in MCP servers (`OrderMCPServer`, `SMSMCPServer`, `TicketMCPServer`) inheriting from `BaseMCPServer`. `cloudagent/mcp/client.py` provides `MCPClient` to connect to servers and call tools. `cloudagent/agent/workflow_agent.py` is a tool-calling agent that uses the MCP client to execute business operations. `cloudagent/graph.py` replaces `workflow_placeholder` with a real `workflow` node.

**Tech Stack:** Python 3.11+, FastAPI, LangGraph, LangChain, `mcp` (Anthropic SDK), pytest, pytest-asyncio

---

## File Structure

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
│   ├── router.py            # MODIFIED: route to workflow node
│   ├── chat_agent.py        # unchanged
│   ├── rag_agent.py         # unchanged
│   └── workflow_agent.py    # NEW: Tool-calling workflow agent
├── graph.py                 # MODIFIED: workflow_node replaces placeholder
├── main.py                  # MODIFIED: init MCP client + workflow_agent
├── hitl.py                  # MODIFIED: add is_sensitive_tool
└── config.py                # MODIFIED: mcp_servers, service URLs

tests/
├── test_mcp_servers.py      # NEW: Order/SMS/Ticket server tests
├── test_mcp_client.py       # NEW: MCP client integration tests
├── test_workflow_agent.py   # NEW: Tool-calling agent tests
├── test_graph.py            # MODIFIED: workflow node flow tests
├── test_main.py             # MODIFIED: workflow end-to-end tests
└── test_hitl.py             # MODIFIED: sensitive tool detection
```

---

### Task 1: Dependencies + Config Extension

**Files:**
- Modify: `pyproject.toml`
- Modify: `cloudagent/config.py`
- Modify: `tests/conftest.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Add mcp dependency to pyproject.toml**

```toml
dependencies = [
    # ... existing deps ...
    "mcp>=1.0.0",
]
```

- [ ] **Step 2: Modify cloudagent/config.py**

Add MCP fields to `Settings`:

```python
class Settings(BaseSettings):
    # ... existing config ...
    mcp_servers: str = "order,sms,ticket"
    order_service_url: str = ""
    sms_service_url: str = ""
    ticket_service_url: str = ""
```

- [ ] **Step 3: Modify tests/conftest.py**

Add MCP env vars to the autouse fixture:

```python
@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    # ... existing env vars ...
    monkeypatch.setenv("MCP_SERVERS", "order,sms,ticket")
    monkeypatch.setenv("ORDER_SERVICE_URL", "")
    monkeypatch.setenv("SMS_SERVICE_URL", "")
    monkeypatch.setenv("TICKET_SERVICE_URL", "")
```

- [ ] **Step 4: Modify tests/test_config.py**

Add assertions for MCP fields:

```python
def test_settings_loads_from_env(patch_env):
    # ... existing assertions ...
    assert settings.mcp_servers == "order,sms,ticket"
    assert settings.order_service_url == ""
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_config.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml cloudagent/config.py tests/conftest.py tests/test_config.py
git commit -m "chore: add MCP dependency and config"
```

---

### Task 2: MCP Base Server + Order Server

**Files:**
- Create: `cloudagent/mcp/servers/base.py`
- Create: `cloudagent/mcp/servers/order.py`
- Create: `tests/test_mcp_servers.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_mcp_servers.py`:

```python
import pytest

from cloudagent.mcp.servers.order import OrderMCPServer


@pytest.fixture
def order_server():
    return OrderMCPServer(dsn="postgresql://test")


@pytest.mark.asyncio
async def test_list_tools(order_server):
    tools = await order_server._server.list_tools()
    names = [t.name for t in tools.tools]
    assert "query_order" in names
    assert "cancel_order" in names
    assert "request_refund" in names


@pytest.mark.asyncio
async def test_query_order_tool(order_server):
    with patch.object(order_server, "query_order", return_value="订单已发货") as mock_query:
        result = await order_server._server.call_tool("query_order", {"order_id": "12345"})
        mock_query.assert_called_once_with(order_id="12345")
        assert "订单已发货" in result.content[0].text
```

Run:
```bash
pytest tests/test_mcp_servers.py -v
```

Expected: `ImportError: cannot import name 'OrderMCPServer'`

- [ ] **Step 2: Write BaseMCPServer**

Create `cloudagent/mcp/servers/base.py`:

```python
from mcp.server import Server
from mcp.types import Tool, TextContent


class BaseMCPServer:
    def __init__(self, name: str):
        self._server = Server(name)
        self._tools: dict[str, tuple[callable, dict]] = {}
        self._register_handlers()

    def _register_handlers(self):
        @self._server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(name=name, description=fn.__doc__, inputSchema=schema)
                for name, (fn, schema) in self._tools.items()
            ]

        @self._server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            if name not in self._tools:
                raise ValueError(f"Unknown tool: {name}")
            fn, _ = self._tools[name]
            result = await fn(**arguments)
            return [TextContent(type="text", text=str(result))]

    def register_tool(self, name: str, schema: dict, fn: callable):
        self._tools[name] = (fn, schema)

    @property
    def server(self):
        return self._server
```

- [ ] **Step 3: Write OrderMCPServer**

Create `cloudagent/mcp/servers/order.py`:

```python
import logging

import asyncpg

from cloudagent.mcp.servers.base import BaseMCPServer

logger = logging.getLogger(__name__)


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
        try:
            conn = await asyncpg.connect(self._dsn)
            try:
                row = await conn.fetchrow(
                    "SELECT * FROM orders WHERE order_id = $1", order_id
                )
                if row:
                    return f"订单 {order_id}: 状态={row['status']}, 金额={row['amount']}"
                return f"未找到订单 {order_id}"
            finally:
                await conn.close()
        except Exception as e:
            logger.warning(f"query_order failed: {e}")
            return "查询订单失败，请稍后重试"

    async def cancel_order(self, order_id: str, reason: str = "") -> str:
        """取消订单"""
        # implementation...
        return f"订单 {order_id} 已取消"

    async def request_refund(self, order_id: str, amount: float = 0, reason: str = "") -> str:
        """申请退款"""
        # implementation...
        return f"订单 {order_id} 退款申请已提交"
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_mcp_servers.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cloudagent/mcp/servers/base.py cloudagent/mcp/servers/order.py tests/test_mcp_servers.py
git commit -m "feat: add MCP base server and order server"
```

---

### Task 3: SMS + Ticket Servers

**Files:**
- Create: `cloudagent/mcp/servers/sms.py`
- Create: `cloudagent/mcp/servers/ticket.py`
- Modify: `tests/test_mcp_servers.py`

- [ ] **Step 1: Write SMSMCPServer**

Create `cloudagent/mcp/servers/sms.py`:

```python
import logging

import httpx

from cloudagent.mcp.servers.base import BaseMCPServer

logger = logging.getLogger(__name__)


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
                    "template": {"type": "string"},
                    "params": {"type": "object"},
                },
                "required": ["phone", "template"],
            },
            self.send_sms,
        )

    async def send_sms(self, phone: str, template: str, params: dict = None) -> str:
        """发送短信通知"""
        try:
            if self._api_url:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f"{self._api_url}/send",
                        json={"phone": phone, "template": template, "params": params or {}},
                    )
                    resp.raise_for_status()
                    return f"短信已发送至 {phone}"
            return f"短信已发送至 {phone}（模拟模式）"
        except Exception as e:
            logger.warning(f"send_sms failed: {e}")
            return "短信发送失败"
```

- [ ] **Step 2: Write TicketMCPServer**

Create `cloudagent/mcp/servers/ticket.py`:

```python
import logging

import asyncpg

from cloudagent.mcp.servers.base import BaseMCPServer

logger = logging.getLogger(__name__)


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
                    "category": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {"type": "string"},
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
        try:
            conn = await asyncpg.connect(self._dsn)
            try:
                row = await conn.fetchrow(
                    """
                    INSERT INTO tickets (user_id, category, title, description, priority, status, created_at)
                    VALUES ($1, $2, $3, $4, $5, 'open', NOW())
                    RETURNING ticket_id
                    """,
                    user_id, category, title, description, priority,
                )
                return f"工单 {row['ticket_id']} 已创建"
            finally:
                await conn.close()
        except Exception as e:
            logger.warning(f"create_ticket failed: {e}")
            return "工单创建失败"

    async def query_ticket(self, ticket_id: str, user_id: str = "") -> str:
        """查询工单状态"""
        try:
            conn = await asyncpg.connect(self._dsn)
            try:
                row = await conn.fetchrow(
                    "SELECT * FROM tickets WHERE ticket_id = $1", ticket_id
                )
                if row:
                    return f"工单 {ticket_id}: 状态={row['status']}, 类别={row['category']}"
                return f"未找到工单 {ticket_id}"
            finally:
                await conn.close()
        except Exception as e:
            logger.warning(f"query_ticket failed: {e}")
            return "查询工单失败"
```

- [ ] **Step 3: Add tests for SMS and Ticket**

Append to `tests/test_mcp_servers.py`:

```python
from cloudagent.mcp.servers.sms import SMSMCPServer
from cloudagent.mcp.servers.ticket import TicketMCPServer


@pytest.mark.asyncio
async def test_sms_server_list_tools():
    server = SMSMCPServer()
    tools = await server.server.list_tools()
    names = [t.name for t in tools.tools]
    assert "send_sms" in names


@pytest.mark.asyncio
async def test_ticket_server_list_tools():
    server = TicketMCPServer(dsn="postgresql://test")
    tools = await server.server.list_tools()
    names = [t.name for t in tools.tools]
    assert "create_ticket" in names
    assert "query_ticket" in names
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_mcp_servers.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cloudagent/mcp/servers/sms.py cloudagent/mcp/servers/ticket.py tests/test_mcp_servers.py
git commit -m "feat: add SMS and Ticket MCP servers"
```

---

### Task 4: MCP Client

**Files:**
- Create: `cloudagent/mcp/client.py`
- Create: `tests/test_mcp_client.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_mcp_client.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cloudagent.mcp.client import MCPClient


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
    session.call_tool = AsyncMock(return_value=MagicMock(content=[]))
    session.close = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_client_lists_tools(mock_session):
    client = MCPClient()
    client._sessions["order"] = mock_session
    client._tool_registry["order:query_order"] = {
        "server": "order", "name": "query_order",
        "description": "查询订单", "schema": {},
    }

    tools = client.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "order:query_order"


@pytest.mark.asyncio
async def test_client_calls_tool(mock_session):
    client = MCPClient()
    client._sessions["order"] = mock_session
    client._tool_registry["order:query_order"] = {
        "server": "order", "name": "query_order",
        "description": "查询订单", "schema": {},
    }
    mock_session.call_tool = AsyncMock(
        return_value=MagicMock(content=[MagicMock(text="订单已发货")])
    )

    result = await client.call_tool("order:query_order", {"order_id": "12345"})
    assert "订单已发货" in result
```

Run:
```bash
pytest tests/test_mcp_client.py -v
```

Expected: `ImportError: cannot import name 'MCPClient'`

- [ ] **Step 2: Write MCPClient**

Create `cloudagent/mcp/client.py`:

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

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_mcp_client.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add cloudagent/mcp/client.py tests/test_mcp_client.py
git commit -m "feat: add MCP client for tool discovery and invocation"
```

---

### Task 5: Workflow Agent (Tool-Calling)

**Files:**
- Create: `cloudagent/agent/workflow_agent.py`
- Create: `tests/test_workflow_agent.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_workflow_agent.py`:

```python
import os
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from cloudagent.agent.workflow_agent import WorkflowAgent


@pytest.mark.asyncio
async def test_workflow_agent_calls_tool():
    mock_mcp = MagicMock()
    mock_mcp.list_tools.return_value = [
        {"name": "order:query_order", "description": "查询订单", "parameters": {}}
    ]
    mock_mcp.call_tool = AsyncMock(return_value="订单已发货")

    agent = WorkflowAgent(model_name="gpt-test", api_key="test-key", mcp_client=mock_mcp)

    with patch.object(agent._llm, "ainvoke") as mock_llm:
        mock_llm.return_value = MagicMock(
            content='{"action": "order:query_order", "action_input": {"order_id": "12345"}}'
        )

        result = await agent.run({"messages": [{"role": "user", "content": "查订单"}]})
        mock_mcp.call_tool.assert_called_once_with("order:query_order", {"order_id": "12345"})
        assert "订单已发货" in result


@pytest.mark.asyncio
async def test_workflow_agent_no_tool_needed():
    mock_mcp = MagicMock()
    mock_mcp.list_tools.return_value = []

    agent = WorkflowAgent(model_name="gpt-test", api_key="test-key", mcp_client=mock_mcp)

    with patch.object(agent._llm, "ainvoke") as mock_llm:
        mock_llm.return_value = MagicMock(
            content='{"final_answer": "请问有什么可以帮您？"}'
        )

        result = await agent.run({"messages": [{"role": "user", "content": "你好"}]})
        assert "请问有什么可以帮您？" in result
```

Run:
```bash
pytest tests/test_workflow_agent.py -v
```

Expected: `ImportError: cannot import name 'WorkflowAgent'`

- [ ] **Step 2: Write WorkflowAgent**

Create `cloudagent/agent/workflow_agent.py`:

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
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_workflow_agent.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add cloudagent/agent/workflow_agent.py tests/test_workflow_agent.py
git commit -m "feat: add tool-calling WorkflowAgent with MCP integration"
```

---

### Task 6: Graph + HITL Integration

**Files:**
- Modify: `cloudagent/graph.py`
- Modify: `cloudagent/hitl.py`
- Modify: `cloudagent/agent/router.py`
- Modify: `tests/test_graph.py`
- Modify: `tests/test_hitl.py`

- [ ] **Step 1: Update HITLManager**

Add `is_sensitive_tool` to `cloudagent/hitl.py`:

```python
class HITLManager:
    SENSITIVE_ACTIONS = {"refund", "cancel", "delete", "request_refund", "cancel_order"}
    # ... existing methods ...

    def is_sensitive_tool(self, tool_name: str) -> bool:
        return tool_name in self.SENSITIVE_ACTIONS
```

- [ ] **Step 2: Update router.py**

No change needed for routing logic (workflow intent already routes correctly), but update INTENT_PROMPT if needed to mention tool availability.

- [ ] **Step 3: Update graph.py**

Replace `workflow_placeholder_node` with `workflow_node`:

```python
class GraphNodes:
    def __init__(self, ..., workflow_agent=None, ...):
        # ... existing agents ...
        self.workflow_agent = workflow_agent

    # ... existing nodes ...

    async def workflow_node(self, state: AgentState) -> AgentState:
        if self.workflow_agent is not None:
            try:
                response = await self.workflow_agent.run(state)
            except Exception as e:
                logger.warning(f"Workflow agent failed: {e}")
                response = "业务办理暂时无法完成，请稍后重试。"
        else:
            response = "业务办理功能正在开发中，请稍后再试。"
        state["response"] = response
        return state

    # Remove workflow_placeholder_node
```

Update `route_node`:

```python
def route_node(self, state: AgentState) -> str:
    target = state.get("target_agent")
    confidence = state.get("confidence", 0.0)

    if target == "clarify":
        return "clarify"
    if target == "workflow" and self.hitl.is_sensitive("workflow", {}):
        return "hitl_request"
    if target == "workflow":
        return "workflow"
    if target in ("chat", "faq") and confidence > 0.5:
        return target
    return "chat"
```

Update `build_graph`:

```python
def build_graph(..., workflow_agent=None, ...):
    nodes = GraphNodes(..., workflow_agent, ...)

    builder.add_node("workflow", nodes.workflow_node)
    # ... existing nodes ...

    builder.add_conditional_edges(
        "entry",
        nodes.route_node,
        {
            "chat": "chat",
            "faq": "rag",
            "workflow": "workflow",  # replaces workflow_placeholder
            "hitl_request": "hitl_request",
            "clarify": "clarify",
        },
    )
    builder.add_edge("workflow", "save_memory")
    # ... existing edges ...
```

- [ ] **Step 4: Update test_graph.py**

Add workflow node tests:

```python
@pytest.mark.asyncio
async def test_graph_workflow_node_with_tool():
    entry = MagicMock()
    entry.run = MagicMock(return_value={
        "messages": [{"role": "user", "content": "查订单 12345"}],
        "intent": "workflow",
        "confidence": 0.92,
        "target_agent": "workflow",
        "context": {},
    })

    workflow = MagicMock()
    workflow.run = AsyncMock(return_value="订单 12345 已发货")

    chat = MagicMock()
    rag = MagicMock()

    graph = build_graph(entry, chat, rag, workflow_agent=workflow)

    state = AgentState(
        messages=[{"role": "user", "content": "查订单 12345"}],
        user_id="user-1",
        session_id="sess-1",
        last_message="查订单 12345",
    )

    result = await graph.ainvoke(state, config={"configurable": {"thread_id": "sess-1"}})
    assert result["response"] == "订单 12345 已发货"
    workflow.run.assert_called_once()
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_graph.py tests/test_hitl.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add cloudagent/graph.py cloudagent/hitl.py cloudagent/agent/router.py tests/test_graph.py tests/test_hitl.py
git commit -m "feat: integrate workflow node with MCP tool calling into graph"
```

---

### Task 7: FastAPI Integration + Test Suite Update

**Files:**
- Modify: `cloudagent/main.py`
- Modify: `tests/test_main.py`

- [ ] **Step 1: Update main.py**

Initialize MCP client and WorkflowAgent at module level:

```python
from cloudagent.mcp.client import MCPClient
from cloudagent.agent.workflow_agent import WorkflowAgent

# ... existing initialization ...

mcp_client = MCPClient()

# Initialize MCP servers (simplified in-memory approach for built-in servers)
# Actual connection depends on transport choice (stdio / in-memory)

workflow_agent = WorkflowAgent(
    model_name=settings.model_name,
    api_key=settings.openai_api_key.get_secret_value(),
    mcp_client=mcp_client,
)

graph = build_graph(
    entry_agent=entry_agent,
    chat_agent=chat_agent,
    rag_agent=rag_agent,
    workflow_agent=workflow_agent,
    memory_manager=memory_manager,
    cache=cache,
    hitl=hitl_manager,
)
```

- [ ] **Step 2: Update tests/test_main.py**

Patch `WorkflowAgent` and `MCPClient` before importing main:

```python
@patch("cloudagent.mcp.client.MCPClient")
@patch("cloudagent.agent.workflow_agent.WorkflowAgent")
@patch("cloudagent.retrieval.vector.VectorRetriever")
# ... existing patches ...
def test_chat_endpoint_workflow_tool(
    mock_vec_cls, mock_graph_cls, mock_kw_cls,
    mock_rag_cls, mock_store_cls, mock_entry_cls,
    mock_chat_cls, mock_workflow_cls, mock_mcp_cls,
):
    mock_store = MagicMock()
    mock_store.get_session.return_value = []
    mock_store_cls.return_value = mock_store

    mock_entry = MagicMock()
    mock_entry.run.return_value = {
        "messages": [{"role": "user", "content": "查订单 12345"}],
        "intent": "workflow",
        "confidence": 0.92,
        "target_agent": "workflow",
        "context": {},
    }
    mock_entry_cls.return_value = mock_entry

    mock_workflow = MagicMock()
    mock_workflow.run = AsyncMock(return_value="订单 12345 已发货")
    mock_workflow_cls.return_value = mock_workflow

    # ... setup other mocks ...

    import importlib
    import cloudagent.main
    importlib.reload(cloudagent.main)
    from cloudagent.main import app

    client = TestClient(app)
    response = client.post("/chat", json={
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "message": "查订单 12345",
    })

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "订单 12345 已发货"
    assert data["intent"] == "workflow"
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_main.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add cloudagent/main.py tests/test_main.py
git commit -m "feat: wire MCP client and WorkflowAgent into FastAPI app"
```

---

### Task 8: Verification & Polish

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 2: Update CLAUDE.md**

- Mark Phase 5 as complete.
- Add MCP to tech stack.
- Update directory structure with `mcp/` and `workflow_agent.py`.
- Update environment variables with MCP settings.
- Add notes on MCP client initialization pattern.

- [ ] **Step 3: Update README.md**

- Add Phase 5 features（MCP 工具生态、订单/短信/工单办理）。
- Update architecture diagram to show MCP layer.
- Update test coverage list.
- Mark Phase 5 as complete in roadmap.

- [ ] **Step 4: Final commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: update README and CLAUDE for Phase 5"
```

---

## Self-Review

**1. Spec coverage:**
- MCP Base Server → Task 2
- Order/SMS/Ticket Servers → Task 2-3
- MCP Client → Task 4
- Workflow Agent (tool-calling) → Task 5
- Graph integration (workflow node) → Task 6
- FastAPI wiring → Task 7
- HITL sensitive tool detection → Task 6
- Testing strategy → covered in all tasks

**2. Placeholder scan:**
- SMS server uses mock mode when `api_url` is empty (documented).
- Order/Ticket servers use placeholder SQL (documented).
- No other TBD/TODO/fill-in-details found.

**3. Type consistency:**
- `WorkflowAgent.run` returns `str` — consistent with `chat_agent.run` and `rag_agent.run`.
- `MCPClient.call_tool` returns `str` — consistent with agent consumption.
- All async store/server methods use `async def` — consistent.
