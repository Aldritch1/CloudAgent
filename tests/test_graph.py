import os
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from cloudagent.graph import GraphNodes, build_graph
from cloudagent.state import AgentState


@pytest.fixture
def mock_agents():
    entry = MagicMock()
    entry.run = MagicMock(return_value={
        "messages": [{"role": "user", "content": "hello"}],
        "intent": "chat",
        "confidence": 0.92,
        "target_agent": "chat",
        "context": {},
    })

    chat = MagicMock()
    chat.run = MagicMock(return_value="Hi there!")

    rag = MagicMock()
    rag.run = AsyncMock(return_value="支持7天无理由退款。")

    return entry, chat, rag


@pytest.mark.asyncio
async def test_graph_chat_flow(mock_agents):
    entry, chat, rag = mock_agents
    graph = build_graph(entry, chat, rag)

    state = AgentState(
        messages=[{"role": "user", "content": "hello"}],
        user_id="user-1",
        session_id="sess-1",
        last_message="hello",
    )

    result = await graph.ainvoke(state, config={"configurable": {"thread_id": "sess-1"}})

    assert result["response"] == "Hi there!"
    assert result["intent"] == "chat"
    entry.run.assert_called_once()
    chat.run.assert_called_once()


@pytest.mark.asyncio
async def test_graph_faq_flow(mock_agents):
    entry = MagicMock()
    entry.run = MagicMock(return_value={
        "messages": [{"role": "user", "content": "怎么退款？"}],
        "intent": "faq",
        "confidence": 0.94,
        "target_agent": "faq",
        "context": {},
    })

    chat = MagicMock()
    rag = MagicMock()
    rag.run = AsyncMock(return_value="支持7天无理由退款。")

    graph = build_graph(entry, chat, rag)

    state = AgentState(
        messages=[{"role": "user", "content": "怎么退款？"}],
        user_id="user-1",
        session_id="sess-1",
        last_message="怎么退款？",
    )

    result = await graph.ainvoke(state, config={"configurable": {"thread_id": "sess-1"}})

    assert result["response"] == "支持7天无理由退款。"
    rag.run.assert_called_once()


@pytest.mark.asyncio
async def test_graph_clarify_flow(mock_agents):
    entry = MagicMock()
    entry.run = MagicMock(return_value={
        "messages": [{"role": "user", "content": "我想查东西"}],
        "intent": "workflow",
        "confidence": 0.65,
        "target_agent": "clarify",
        "clarification_question": "您想查询订单还是退款？",
        "context": {},
    })

    chat = MagicMock()
    rag = MagicMock()

    graph = build_graph(entry, chat, rag)

    state = AgentState(
        messages=[{"role": "user", "content": "我想查东西"}],
        user_id="user-1",
        session_id="sess-1",
        last_message="我想查东西",
    )

    result = await graph.ainvoke(state, config={"configurable": {"thread_id": "sess-1"}})

    assert result["response"] == "您想查询订单还是退款？"
    assert result["action_required"] == "clarify"


@pytest.mark.asyncio
async def test_graph_interrupt_on_sensitive_workflow():
    entry = MagicMock()
    entry.run = MagicMock(return_value={
        "messages": [{"role": "user", "content": "我要退款"}],
        "intent": "workflow",
        "confidence": 0.91,
        "target_agent": "workflow",
        "context": {},
    })

    from cloudagent.hitl import HITLManager

    class TestHITL(HITLManager):
        SENSITIVE_ACTIONS = {"workflow", "refund", "cancel", "delete"}

    chat = MagicMock()
    rag = MagicMock()

    graph = build_graph(entry, chat, rag, hitl=TestHITL())

    state = AgentState(
        messages=[{"role": "user", "content": "我要退款"}],
        user_id="user-1",
        session_id="sess-1",
        last_message="我要退款",
    )

    result = await graph.ainvoke(state, config={"configurable": {"thread_id": "sess-1"}})

    assert result["action_required"] == "confirm"
    assert "pending_action" in result

    # Resume with confirmation — pass None to continue from interrupt
    result2 = await graph.ainvoke(None, config={"configurable": {"thread_id": "sess-1"}})

    assert result2["response"] == "请回复'确认'或'取消'。"
    assert result2.get("action_required") == "confirm"


def test_graph_compiles():
    entry = MagicMock()
    chat = MagicMock()
    rag = MagicMock()
    graph = build_graph(entry, chat, rag)
    assert graph is not None


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
