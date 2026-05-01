from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cloudagent.retrieval.vector import VectorRetriever


@pytest.fixture
def mock_milvus():
    with patch("cloudagent.retrieval.vector.MilvusClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_cls.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_embeddings():
    with patch("cloudagent.retrieval.vector.OpenAIEmbeddings") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.aembed_query = AsyncMock(return_value=[0.1] * 1536)
        mock_cls.return_value = mock_instance
        yield mock_instance


@pytest.mark.asyncio
async def test_vector_search(mock_milvus, mock_embeddings):
    mock_milvus.search.return_value = [[
        {"entity": {"content": "退款政策", "category": "售后"}, "distance": 0.9},
    ]]

    retriever = VectorRetriever(uri="http://localhost:19530", api_key="test-key")
    results = await retriever.search("怎么退款", top_k=5)

    assert len(results) == 1
    assert results[0].content == "退款政策"
    assert results[0].source == "vector"
    assert results[0].score == 0.9
    assert results[0].metadata["category"] == "售后"
    mock_milvus.search.assert_called_once()


@pytest.mark.asyncio
async def test_vector_search_degrades_on_failure(mock_milvus, mock_embeddings):
    mock_milvus.search.side_effect = Exception("Milvus down")

    retriever = VectorRetriever(uri="http://localhost:19530", api_key="test-key")
    results = await retriever.search("怎么退款", top_k=5)

    assert results == []
