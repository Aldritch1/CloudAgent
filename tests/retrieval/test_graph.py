from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cloudagent.retrieval.graph import GraphRetriever


class AsyncIter:
    def __init__(self, items):
        self.items = items

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.items:
            raise StopAsyncIteration
        return self.items.pop(0)


@pytest.fixture
def mock_neo4j():
    with patch("cloudagent.retrieval.graph.AsyncGraphDatabase.driver") as mock_driver:
        mock_session = AsyncMock()
        mock_driver.return_value.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_driver.return_value.session.return_value.__aexit__ = AsyncMock(return_value=False)
        yield mock_session


@pytest.mark.asyncio
async def test_graph_search(mock_neo4j):
    mock_neo4j.run.return_value = AsyncIter([
        {"content": "退款流程说明", "metadata": "售后"},
    ])

    retriever = GraphRetriever(uri="bolt://localhost:7687", user="neo4j", password="pass")
    results = await retriever.search("怎么退款", top_k=5)

    assert len(results) == 1
    assert results[0].content == "退款流程说明"
    assert results[0].source == "graph"
    assert results[0].metadata["category"] == "售后"


@pytest.mark.asyncio
async def test_graph_search_degrades_on_failure(mock_neo4j):
    mock_neo4j.run.side_effect = Exception("Neo4j down")

    retriever = GraphRetriever(uri="bolt://localhost:7687", user="neo4j", password="pass")
    results = await retriever.search("怎么退款", top_k=5)

    assert results == []
