import logging

from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient

from cloudagent.retrieval.base import RetrievalResult

logger = logging.getLogger(__name__)


class VectorRetriever:
    def __init__(self, uri: str, api_key: str, collection_name: str = "kb_documents"):
        self._collection = collection_name
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
        try:
            self._client = MilvusClient(uri=uri)
            if not self._client.has_collection(collection_name):
                self._client.create_collection(
                    collection_name=collection_name,
                    dimension=1536,
                )
        except Exception as e:
            logger.warning(f"Milvus connection failed: {e}")
            self._client = None

    async def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if self._client is None:
            return []
        try:
            embedding = await self._embeddings.aembed_query(query)
            results = self._client.search(
                collection_name=self._collection,
                data=[embedding],
                limit=top_k,
                output_fields=["content", "category"],
            )
            return [
                RetrievalResult(
                    content=hit["entity"]["content"],
                    source="vector",
                    score=hit.get("distance", 0.0),
                    metadata={"category": hit["entity"].get("category", "")},
                )
                for hit in results[0]
            ]
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []
