from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@patch("cloudagent.retrieval.vector.VectorRetriever")
@patch("cloudagent.retrieval.graph.GraphRetriever")
@patch("cloudagent.retrieval.keyword.KeywordRetriever")
@patch("cloudagent.agent.rag_agent.RAGAgent")
@patch("cloudagent.memory.redis_store.SessionStore")
@patch("cloudagent.agent.router.EntryAgent")
@patch("cloudagent.agent.chat_agent.ChatAgent")
@patch("cloudagent.api.sse.get_current_user")
@patch("cloudagent.api.sse.tenant_dependency")
def test_sse_endpoint_returns_events(
    mock_tenant_dep,
    mock_get_user,
    mock_chat_cls,
    mock_entry_cls,
    mock_store_cls,
    mock_rag_cls,
    mock_kw_cls,
    mock_graph_cls,
    mock_vec_cls,
):
    mock_get_user.return_value = "test-user"
    mock_tenant_dep.return_value = "default"

    mock_entry = MagicMock()
    mock_entry.run.return_value = {
        "messages": [{"role": "user", "content": "hello"}],
        "intent": "chat",
        "confidence": 0.92,
        "target_agent": "chat",
        "context": {},
    }
    mock_entry_cls.return_value = mock_entry

    mock_chat = MagicMock()
    async def mock_stream(msgs):
        for token in ["Hi", " ", "there", "!"]:
            yield token
    mock_chat.run_stream = mock_stream
    mock_chat_cls.return_value = mock_chat

    import importlib
    import cloudagent.main
    importlib.reload(cloudagent.main)
    from cloudagent.main import app

    client = TestClient(app)
    response = client.post("/chat/stream", json={
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "message": "hello",
    })

    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    body = response.text
    assert "event: intent" in body
    assert "event: token" in body
    assert "event: done" in body
