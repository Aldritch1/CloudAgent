import logging

from neo4j import AsyncGraphDatabase

from cloudagent.retrieval.base import RetrievalResult

logger = logging.getLogger(__name__)


class GraphRetriever:
    def __init__(self, uri: str, user: str, password: str):
        self._driver = None
        try:
            self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        except Exception as e:
            logger.warning(f"Neo4j connection failed: {e}")

    async def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if self._driver is None:
            return []
        try:
            async with self._driver.session() as session:
                result = await session.run(
                    """
                    MATCH (f:FAQ)
                    WHERE f.question CONTAINS $query OR f.answer CONTAINS $query
                    RETURN f.question AS content, f.category AS metadata
                    LIMIT $limit
                    """,
                    query=query,
                    limit=top_k,
                )
                records = []
                async for record in result:
                    records.append(
                        RetrievalResult(
                            content=record["content"],
                            source="graph",
                            metadata={"category": record.get("metadata", "")},
                        )
                    )
                return records
        except Exception as e:
            logger.warning(f"Graph search failed: {e}")
            return []
