from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cloudagent.memory.cold_store import ColdStore


@pytest.mark.asyncio
async def test_save_memory():
    mock_client = MagicMock()
    mock_client.has_collection.return_value = True

    with patch("cloudagent.memory.cold_store.MilvusClient", return_value=mock_client):
        with patch("cloudagent.memory.cold_store.OpenAIEmbeddings") as mock_emb_cls:
            mock_emb = MagicMock()
            mock_emb.aembed_query = AsyncMock(return_value=[0.1] * 1536)
            mock_emb_cls.return_value = mock_emb

            store = ColdStore("http://localhost:19530", "test-key")
            await store.save_memory("u1", "s1", "用户喜欢红色")

    mock_client.insert.assert_called_once()


@pytest.mark.asyncio
async def test_search_memories():
    mock_client = MagicMock()
    mock_client.has_collection.return_value = True
    mock_client.search.return_value = [[
        {"entity": {"content": "用户喜欢红色"}},
        {"entity": {"content": "用户经常购买电子产品"}},
    ]]

    with patch("cloudagent.memory.cold_store.MilvusClient", return_value=mock_client):
        with patch("cloudagent.memory.cold_store.OpenAIEmbeddings") as mock_emb_cls:
            mock_emb = MagicMock()
            mock_emb.aembed_query = AsyncMock(return_value=[0.1] * 1536)
            mock_emb_cls.return_value = mock_emb

            store = ColdStore("http://localhost:19530", "test-key")
            result = await store.search_memories("u1", "用户喜欢什么颜色？")

    assert "用户喜欢红色" in result
    assert len(result) == 2


@pytest.mark.asyncio
async def test_degrades_on_milvus_failure():
    with patch("cloudagent.memory.cold_store.MilvusClient", side_effect=Exception("Milvus down")):
        store = ColdStore("http://localhost:19530", "test-key")
        await store.save_memory("u1", "s1", "test")
        result = await store.search_memories("u1", "test")
        assert result == []
