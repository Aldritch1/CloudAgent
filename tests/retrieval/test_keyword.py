from unittest.mock import AsyncMock, patch

import pytest

from cloudagent.retrieval.keyword import KeywordRetriever


@pytest.fixture
def mock_asyncpg():
    with patch("cloudagent.retrieval.keyword.asyncpg.connect") as mock_connect:
        mock_conn = AsyncMock()
        mock_connect.return_value = mock_conn
        yield mock_conn


@pytest.mark.asyncio
async def test_keyword_search(mock_asyncpg):
    mock_asyncpg.fetch.return_value = [
        {"title": "退款政策", "content": "支持7天无理由退款", "category": "售后"},
    ]

    retriever = KeywordRetriever(dsn="postgresql://u:p@localhost/db")
    results = await retriever.search("怎么退款", top_k=5)

    assert len(results) == 1
    assert results[0].content == "支持7天无理由退款"
    assert results[0].source == "keyword"
    assert results[0].metadata["title"] == "退款政策"
    assert results[0].metadata["category"] == "售后"
    mock_asyncpg.fetch.assert_called_once()


@pytest.mark.asyncio
async def test_keyword_search_degrades_on_failure(mock_asyncpg):
    mock_asyncpg.fetch.side_effect = Exception("PG down")

    retriever = KeywordRetriever(dsn="postgresql://u:p@localhost/db")
    results = await retriever.search("怎么退款", top_k=5)

    assert results == []
