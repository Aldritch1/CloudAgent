import logging

import asyncpg

from cloudagent.retrieval.base import RetrievalResult

logger = logging.getLogger(__name__)


class KeywordRetriever:
    def __init__(self, dsn: str):
        self._dsn = dsn

    async def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        try:
            conn = await asyncpg.connect(self._dsn)
            try:
                rows = await conn.fetch(
                    """
                    SELECT title, content, category
                    FROM kb_documents
                    WHERE fts_vector @@ plainto_tsquery('chinese', $1)
                    ORDER BY ts_rank(fts_vector, plainto_tsquery('chinese', $1)) DESC
                    LIMIT $2
                    """,
                    query,
                    top_k,
                )
                return [
                    RetrievalResult(
                        content=row["content"],
                        source="keyword",
                        metadata={"title": row.get("title", ""), "category": row.get("category", "")},
                    )
                    for row in rows
                ]
            finally:
                await conn.close()
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")
            return []
