import asyncio
import logging
from collections import defaultdict

from cloudagent.retrieval.base import RetrievalResult

logger = logging.getLogger(__name__)


def rrf_fuse(result_lists: list[list[RetrievalResult]], k: int = 60, final_top_k: int = 5) -> list[RetrievalResult]:
    scores: dict[str, float] = defaultdict(float)
    items: dict[str, RetrievalResult] = {}
    for results in result_lists:
        for rank, r in enumerate(results, start=1):
            scores[r.content] += 1.0 / (k + rank)
            if r.content not in items:
                items[r.content] = r
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [items[content] for content, _ in sorted_items[:final_top_k]]


class HybridRetriever:
    def __init__(self, vector, graph, keyword):
        self.vector = vector
        self.graph = graph
        self.keyword = keyword

    async def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        v_task = self.vector.search(query, top_k=10)
        g_task = self.graph.search(query, top_k=10)
        k_task = self.keyword.search(query, top_k=10)
        v_results, g_results, k_results = await asyncio.gather(v_task, g_task, k_task)
        return rrf_fuse([v_results, g_results, k_results], k=60, final_top_k=top_k)
