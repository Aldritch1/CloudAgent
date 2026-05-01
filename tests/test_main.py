from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# Patch original modules before importing main (module-level init runs at import time)
@patch("cloudagent.retrieval.vector.VectorRetriever")
@patch("cloudagent.retrieval.graph.GraphRetriever")
@patch("cloudagent.retrieval.keyword.KeywordRetriever")
@patch("cloudagent.agent.rag_agent.RAGAgent")
@patch("cloudagent.memory.redis_store.SessionStore")
@patch("cloudagent.agent.router.EntryAgent")
@patch("cloudagent.agent.chat_agent.ChatAgent")
def test_health_endpoint(mock_chat_cls, mock_entry_cls, mock_store_cls,
                          mock_rag_cls, mock_kw_cls, mock_graph_cls, mock_vec_cls):
    mock_store_cls.return_value = MagicMock()
    mock_entry_cls.return_value = MagicMock()
    mock_chat_cls.return_value = MagicMock()
    mock_rag_cls.return_value = MagicMock()
    mock_kw_cls.return_value = MagicMock()
    mock_graph_cls.return_value = MagicMock()
    mock_vec_cls.return_value = MagicMock()

    from cloudagent.main import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@patch("cloudagent.retrieval.vector.VectorRetriever")
@patch("cloudagent.retrieval.graph.GraphRetriever")
@patch("cloudagent.retrieval.keyword.KeywordRetriever")
@patch("cloudagent.agent.rag_agent.RAGAgent")
@patch("cloudagent.memory.redis_store.SessionStore")
@patch("cloudagent.agent.router.EntryAgent")
@patch("cloudagent.agent.chat_agent.ChatAgent")
def test_chat_endpoint_success(mock_chat_cls, mock_entry_cls, mock_store_cls,
                                mock_rag_cls, mock_kw_cls, mock_graph_cls, mock_vec_cls):
    mock_store = MagicMock()
    mock_store.get_session.return_value = []
    mock_store_cls.return_value = mock_store

    mock_entry = MagicMock()
    mock_entry.run.return_value = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        "intent": "chat",
        "confidence": 0.92,
        "target_agent": "chat",
        "context": {},
    }
    mock_entry_cls.return_value = mock_entry

    mock_chat = MagicMock()
    mock_chat.run.return_value = "Hi there!"
    mock_chat_cls.return_value = mock_chat

    import importlib
    import cloudagent.main
    importlib.reload(cloudagent.main)
    from cloudagent.main import app

    client = TestClient(app)
    response = client.post("/chat", json={
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "message": "hello",
    })

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Hi there!"
    assert data["intent"] == "chat"
    assert data["confidence"] == 0.92
    mock_store.save_session.assert_called_once()
    saved_messages = mock_store.save_session.call_args[0][1]
    assert saved_messages[-2] == {"role": "user", "content": "hello"}
    assert saved_messages[-1] == {"role": "assistant", "content": "Hi there!"}


@patch("cloudagent.retrieval.vector.VectorRetriever")
@patch("cloudagent.retrieval.graph.GraphRetriever")
@patch("cloudagent.retrieval.keyword.KeywordRetriever")
@patch("cloudagent.agent.rag_agent.RAGAgent")
@patch("cloudagent.memory.redis_store.SessionStore")
@patch("cloudagent.agent.router.EntryAgent")
@patch("cloudagent.agent.chat_agent.ChatAgent")
def test_chat_endpoint_invalid_request(mock_chat_cls, mock_entry_cls, mock_store_cls,
                                        mock_rag_cls, mock_kw_cls, mock_graph_cls, mock_vec_cls):
    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store
    mock_entry_cls.return_value = MagicMock()
    mock_chat_cls.return_value = MagicMock()
    mock_rag_cls.return_value = MagicMock()
    mock_kw_cls.return_value = MagicMock()
    mock_graph_cls.return_value = MagicMock()
    mock_vec_cls.return_value = MagicMock()

    from cloudagent.main import app

    client = TestClient(app)
    response = client.post("/chat", json={"message": "missing session_id"})
    assert response.status_code == 422


@patch("cloudagent.retrieval.vector.VectorRetriever")
@patch("cloudagent.retrieval.graph.GraphRetriever")
@patch("cloudagent.retrieval.keyword.KeywordRetriever")
@patch("cloudagent.agent.rag_agent.RAGAgent")
@patch("cloudagent.memory.redis_store.SessionStore")
@patch("cloudagent.agent.router.EntryAgent")
@patch("cloudagent.agent.chat_agent.ChatAgent")
def test_chat_agent_failure(mock_chat_cls, mock_entry_cls, mock_store_cls,
                             mock_rag_cls, mock_kw_cls, mock_graph_cls, mock_vec_cls):
    mock_store = MagicMock()
    mock_store.get_session.return_value = []
    mock_store_cls.return_value = mock_store

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
    mock_chat.run.side_effect = Exception("LLM error")
    mock_chat_cls.return_value = mock_chat
    mock_rag_cls.return_value = MagicMock()
    mock_kw_cls.return_value = MagicMock()
    mock_graph_cls.return_value = MagicMock()
    mock_vec_cls.return_value = MagicMock()

    import importlib
    import cloudagent.main
    importlib.reload(cloudagent.main)
    from cloudagent.main import app

    client = TestClient(app)
    response = client.post("/chat", json={
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "message": "hello",
    })

    assert response.status_code == 500
    assert response.json() == {"detail": "服务暂时繁忙，请稍后重试"}


@patch("cloudagent.retrieval.vector.VectorRetriever")
@patch("cloudagent.retrieval.graph.GraphRetriever")
@patch("cloudagent.retrieval.keyword.KeywordRetriever")
@patch("cloudagent.agent.rag_agent.RAGAgent")
@patch("cloudagent.memory.redis_store.SessionStore")
@patch("cloudagent.agent.router.EntryAgent")
@patch("cloudagent.agent.chat_agent.ChatAgent")
def test_chat_endpoint_faq_routing(mock_chat_cls, mock_entry_cls, mock_store_cls,
                                    mock_rag_cls, mock_kw_cls, mock_graph_cls, mock_vec_cls):
    mock_store = MagicMock()
    mock_store.get_session.return_value = []
    mock_store_cls.return_value = mock_store

    mock_entry = MagicMock()
    mock_entry.run.return_value = {
        "messages": [{"role": "user", "content": "怎么退款？"}],
        "intent": "faq",
        "confidence": 0.94,
        "target_agent": "faq",
        "context": {},
    }
    mock_entry_cls.return_value = mock_entry

    mock_rag = MagicMock()
    mock_rag.run = AsyncMock(return_value="支持7天无理由退款。")
    mock_rag_cls.return_value = mock_rag

    mock_chat = MagicMock()
    mock_chat_cls.return_value = mock_chat

    import importlib
    import cloudagent.main
    importlib.reload(cloudagent.main)
    from cloudagent.main import app

    client = TestClient(app)
    response = client.post("/chat", json={
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "message": "怎么退款？",
    })

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "支持7天无理由退款。"
    assert data["intent"] == "faq"
    mock_rag.run.assert_called_once()


@patch("cloudagent.retrieval.vector.VectorRetriever")
@patch("cloudagent.retrieval.graph.GraphRetriever")
@patch("cloudagent.retrieval.keyword.KeywordRetriever")
@patch("cloudagent.agent.rag_agent.RAGAgent")
@patch("cloudagent.memory.redis_store.SessionStore")
@patch("cloudagent.agent.router.EntryAgent")
@patch("cloudagent.agent.chat_agent.ChatAgent")
def test_chat_endpoint_workflow_placeholder(mock_chat_cls, mock_entry_cls, mock_store_cls,
                                             mock_rag_cls, mock_kw_cls, mock_graph_cls, mock_vec_cls):
    mock_store = MagicMock()
    mock_store.get_session.return_value = []
    mock_store_cls.return_value = mock_store

    mock_entry = MagicMock()
    mock_entry.run.return_value = {
        "messages": [{"role": "user", "content": "查订单"}],
        "intent": "workflow",
        "confidence": 0.91,
        "target_agent": "workflow",
        "context": {},
    }
    mock_entry_cls.return_value = mock_entry

    mock_rag = MagicMock()
    mock_rag_cls.return_value = mock_rag
    mock_chat = MagicMock()
    mock_chat_cls.return_value = mock_chat

    import importlib
    import cloudagent.main
    importlib.reload(cloudagent.main)
    from cloudagent.main import app

    client = TestClient(app)
    response = client.post("/chat", json={
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "message": "查订单",
    })

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "业务办理功能正在开发中，请稍后再试。"
    assert data["intent"] == "workflow"
    mock_rag.run.assert_not_called()
    mock_chat.run.assert_not_called()


@patch("cloudagent.rate_limit.RateLimiter")
@patch("cloudagent.circuit_breaker.LLMCircuitBreaker")
@patch("cloudagent.metrics.MetricsMiddleware")
@patch("cloudagent.retrieval.vector.VectorRetriever")
@patch("cloudagent.retrieval.graph.GraphRetriever")
@patch("cloudagent.retrieval.keyword.KeywordRetriever")
@patch("cloudagent.agent.rag_agent.RAGAgent")
@patch("cloudagent.memory.redis_store.SessionStore")
@patch("cloudagent.agent.router.EntryAgent")
@patch("cloudagent.agent.chat_agent.ChatAgent")
def test_chat_endpoint_rate_limit_returns_429(
    mock_chat_cls, mock_entry_cls, mock_store_cls,
    mock_rag_cls, mock_kw_cls, mock_graph_cls, mock_vec_cls,
    mock_metrics_cls, mock_breaker_cls, mock_rate_cls,
):
    mock_store = MagicMock()
    mock_store.get_session.return_value = []
    mock_store_cls.return_value = mock_store

    mock_rate = MagicMock()
    mock_rate.check.return_value = False
    mock_rate_cls.return_value = mock_rate

    mock_entry_cls.return_value = MagicMock()
    mock_chat_cls.return_value = MagicMock()
    mock_rag_cls.return_value = MagicMock()
    mock_kw_cls.return_value = MagicMock()
    mock_graph_cls.return_value = MagicMock()
    mock_vec_cls.return_value = MagicMock()
    mock_metrics_cls.return_value = MagicMock()
    mock_breaker_cls.return_value = MagicMock()

    import importlib
    import cloudagent.main
    importlib.reload(cloudagent.main)
    from cloudagent.main import app

    client = TestClient(app)
    response = client.post("/chat", json={
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "message": "hello",
    })

    assert response.status_code == 429
    assert response.json()["detail"] == "请求过于频繁，请稍后再试"
    assert response.headers["X-RateLimit-Remaining"] == "0"


@patch("cloudagent.rate_limit.RateLimiter")
@patch("cloudagent.circuit_breaker.LLMCircuitBreaker")
@patch("cloudagent.metrics.MetricsMiddleware")
@patch("cloudagent.retrieval.vector.VectorRetriever")
@patch("cloudagent.retrieval.graph.GraphRetriever")
@patch("cloudagent.retrieval.keyword.KeywordRetriever")
@patch("cloudagent.agent.rag_agent.RAGAgent")
@patch("cloudagent.memory.redis_store.SessionStore")
@patch("cloudagent.agent.router.EntryAgent")
@patch("cloudagent.agent.chat_agent.ChatAgent")
def test_chat_endpoint_circuit_breaker_returns_503(
    mock_chat_cls, mock_entry_cls, mock_store_cls,
    mock_rag_cls, mock_kw_cls, mock_graph_cls, mock_vec_cls,
    mock_metrics_cls, mock_breaker_cls, mock_rate_cls,
):
    mock_store = MagicMock()
    mock_store.get_session.return_value = []
    mock_store_cls.return_value = mock_store

    mock_rate = MagicMock()
    mock_rate.check.return_value = True
    mock_rate_cls.return_value = mock_rate

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
    mock_chat.run.side_effect = Exception("circuit open")
    mock_chat_cls.return_value = mock_chat

    mock_rag_cls.return_value = MagicMock()
    mock_kw_cls.return_value = MagicMock()
    mock_graph_cls.return_value = MagicMock()
    mock_vec_cls.return_value = MagicMock()
    mock_metrics_cls.return_value = MagicMock()
    mock_breaker_cls.return_value = MagicMock()

    import importlib
    import cloudagent.main
    importlib.reload(cloudagent.main)
    from cloudagent.main import app

    # Simulate CircuitBreakerError propagating from graph
    from pybreaker import CircuitBreakerError
    from cloudagent.main import graph
    graph.ainvoke = AsyncMock(side_effect=CircuitBreakerError("Circuit open"))

    client = TestClient(app)
    response = client.post("/chat", json={
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "message": "hello",
    })

    assert response.status_code == 503
    assert response.json()["detail"] == "服务暂时不可用，请稍后重试"
