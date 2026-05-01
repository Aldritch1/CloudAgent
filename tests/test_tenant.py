import os
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import fakeredis
import pytest
from fastapi.testclient import TestClient

from cloudagent.memory.cold_store import ColdStore
from cloudagent.memory.redis_store import SessionStore
from cloudagent.memory.warm_store import WarmStore
import pytest

from cloudagent.tenant import tenant_dependency
from cloudagent.tenant_context import get_tenant_id, set_tenant_id


@pytest.fixture(autouse=True)
def reset_tenant():
    set_tenant_id("")
    yield


def test_tenant_defaults():
    assert get_tenant_id() == ""


def test_set_and_get_tenant():
    set_tenant_id("acme-corp")
    assert get_tenant_id() == "acme-corp"
    set_tenant_id("default")


@pytest.mark.asyncio
async def test_tenant_dependency_from_header():
    tenant_id = await tenant_dependency(x_tenant_id="tenant-a")
    assert tenant_id == "tenant-a"
    assert get_tenant_id() == "tenant-a"
    set_tenant_id("default")


@pytest.mark.asyncio
async def test_tenant_dependency_fallback_to_context():
    set_tenant_id("tenant-b")
    tenant_id = await tenant_dependency(x_tenant_id=None)
    assert tenant_id == "tenant-b"
    set_tenant_id("default")


def test_tenant_isolation_in_redis():
    redis_client = fakeredis.FakeRedis()
    store = SessionStore(redis_url="redis://fake", _redis_client=redis_client)

    set_tenant_id("tenant-a")
    store.save_session("sess-1", [{"role": "user", "content": "hello"}])

    set_tenant_id("tenant-b")
    assert store.get_session("sess-1") == []

    set_tenant_id("tenant-a")
    assert store.get_session("sess-1") == [{"role": "user", "content": "hello"}]

    set_tenant_id("default")


def test_redis_key_prefix_with_tenant():
    redis_client = fakeredis.FakeRedis()
    store = SessionStore(redis_url="redis://fake", _redis_client=redis_client)

    set_tenant_id("acme")
    store.save_session("sess-1", [{"role": "user", "content": "hi"}])

    keys = list(redis_client.scan_iter())
    assert any(b"acme:session:sess-1" in k for k in keys)

    set_tenant_id("default")


@pytest.mark.asyncio
async def test_tenant_isolation_in_warm_store():
    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = {"user_id": "u1", "preferences": "{}", "tenant_id": "t1"}

    with patch("cloudagent.memory.warm_store.asyncpg.connect", return_value=mock_conn):
        store = WarmStore("postgresql://test")
        set_tenant_id("tenant-a")
        await store.get_user_profile("u1")

    call_args = mock_conn.fetchrow.call_args[0]
    assert "tenant_id" in call_args[0].lower()
    assert "tenant-a" in call_args[1:]

    set_tenant_id("default")


@pytest.mark.asyncio
async def test_tenant_in_warm_store_upsert():
    mock_conn = AsyncMock()

    with patch("cloudagent.memory.warm_store.asyncpg.connect", return_value=mock_conn):
        store = WarmStore("postgresql://test")
        set_tenant_id("tenant-b")
        await store.save_user_profile("u1", {"lang": "zh"})

    sql = mock_conn.execute.call_args[0][0]
    assert "tenant_id" in sql.lower()
    assert "tenant-b" in mock_conn.execute.call_args[0]

    set_tenant_id("default")


@pytest.mark.asyncio
async def test_tenant_isolation_in_cold_store():
    mock_client = MagicMock()
    mock_client.has_collection.return_value = True

    with patch("cloudagent.memory.cold_store.MilvusClient", return_value=mock_client):
        with patch("cloudagent.memory.cold_store.OpenAIEmbeddings") as mock_emb_cls:
            mock_emb = MagicMock()
            mock_emb.aembed_query = AsyncMock(return_value=[0.1] * 1536)
            mock_emb_cls.return_value = mock_emb

            store = ColdStore("http://localhost:19530", "test-key")
            set_tenant_id("tenant-c")
            await store.save_memory("u1", "s1", "content")

    insert_data = mock_client.insert.call_args[1]["data"][0]
    assert insert_data["tenant_id"] == "tenant-c"

    set_tenant_id("default")


@pytest.mark.asyncio
async def test_tenant_filter_in_cold_store_search():
    mock_client = MagicMock()
    mock_client.has_collection.return_value = True
    mock_client.search.return_value = [[]]

    with patch("cloudagent.memory.cold_store.MilvusClient", return_value=mock_client):
        with patch("cloudagent.memory.cold_store.OpenAIEmbeddings") as mock_emb_cls:
            mock_emb = MagicMock()
            mock_emb.aembed_query = AsyncMock(return_value=[0.1] * 1536)
            mock_emb_cls.return_value = mock_emb

            store = ColdStore("http://localhost:19530", "test-key")
            set_tenant_id("tenant-d")
            await store.search_memories("u1", "query")

    filter_expr = mock_client.search.call_args[1]["filter"]
    assert "tenant-d" in filter_expr

    set_tenant_id("default")


@patch("cloudagent.retrieval.vector.VectorRetriever")
@patch("cloudagent.retrieval.graph.GraphRetriever")
@patch("cloudagent.retrieval.keyword.KeywordRetriever")
@patch("cloudagent.agent.rag_agent.RAGAgent")
@patch("cloudagent.memory.redis_store.SessionStore")
@patch("cloudagent.agent.router.EntryAgent")
@patch("cloudagent.agent.chat_agent.ChatAgent")
def test_chat_endpoint_with_tenant_header(
    mock_chat_cls, mock_entry_cls, mock_store_cls,
    mock_rag_cls, mock_kw_cls, mock_graph_cls, mock_vec_cls,
):
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
    mock_chat.run.return_value = "Hi there!"
    mock_chat_cls.return_value = mock_chat

    import importlib
    import cloudagent.main
    importlib.reload(cloudagent.main)
    from cloudagent.main import app

    client = TestClient(app)
    response = client.post(
        "/chat",
        json={
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "message": "hello",
        },
        headers={"X-Tenant-ID": "tenant-x"},
    )

    assert response.status_code == 200
    # Verify that graph state included tenant_id
    mock_entry.run.assert_called_once()
    state_arg = mock_entry.run.call_args[0][0]
    assert state_arg.get("tenant_id") == "tenant-x"
