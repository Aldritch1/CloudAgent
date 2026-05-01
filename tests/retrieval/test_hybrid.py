from unittest.mock import AsyncMock

import pytest

from cloudagent.retrieval.hybrid import HybridRetriever, rrf_fuse
from cloudagent.retrieval.base import RetrievalResult


def test_rrf_fuse_ranking():
    vector_results = [
        RetrievalResult("doc-b", "vector"),
        RetrievalResult("doc-a", "vector"),
    ]
    graph_results = [
        RetrievalResult("doc-b", "graph"),
        RetrievalResult("doc-c", "graph"),
    ]
    keyword_results = [
        RetrievalResult("doc-c", "keyword"),
        RetrievalResult("doc-a", "keyword"),
    ]

    fused = rrf_fuse([vector_results, graph_results, keyword_results], k=60, final_top_k=5)
    contents = [r.content for r in fused]

    # doc-b: rank 2 (vector) + rank 1 (graph) -> highest RRF score
    assert contents[0] == "doc-b"
    assert len(contents) == 3


@pytest.mark.asyncio
async def test_hybrid_search_concurrent():
    v = AsyncMock()
    v.search.return_value = [RetrievalResult("a", "vector")]
    g = AsyncMock()
    g.search.return_value = [RetrievalResult("b", "graph")]
    k = AsyncMock()
    k.search.return_value = [RetrievalResult("c", "keyword")]

    hybrid = HybridRetriever(v, g, k)
    results = await hybrid.search("query", top_k=2)

    assert len(results) == 2
    v.search.assert_called_once_with("query", top_k=10)
    g.search.assert_called_once_with("query", top_k=10)
    k.search.assert_called_once_with("query", top_k=10)
