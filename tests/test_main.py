import os
os.environ["OPENAI_API_KEY"] = "test-key"

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# Patch original modules before importing main (module-level init runs at import time)
@patch("cloudagent.memory.redis_store.SessionStore")
@patch("cloudagent.agent.router.EntryAgent")
@patch("cloudagent.agent.chat_agent.ChatAgent")
def test_chat_endpoint_success(mock_chat_cls, mock_entry_cls, mock_store_cls):
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


@patch("cloudagent.memory.redis_store.SessionStore")
@patch("cloudagent.agent.router.EntryAgent")
@patch("cloudagent.agent.chat_agent.ChatAgent")
def test_chat_endpoint_invalid_request(mock_chat_cls, mock_entry_cls, mock_store_cls):
    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store
    mock_entry_cls.return_value = MagicMock()
    mock_chat_cls.return_value = MagicMock()

    from cloudagent.main import app

    client = TestClient(app)
    response = client.post("/chat", json={"message": "missing session_id"})
    assert response.status_code == 422


@patch("cloudagent.memory.redis_store.SessionStore")
@patch("cloudagent.agent.router.EntryAgent")
@patch("cloudagent.agent.chat_agent.ChatAgent")
def test_chat_agent_failure(mock_chat_cls, mock_entry_cls, mock_store_cls):
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
    assert "detail" in response.json()
