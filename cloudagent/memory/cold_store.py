import logging

from langchain_openai import OpenAIEmbeddings
from pymilvus import DataType, FieldSchema, MilvusClient

from cloudagent.tenant_context import get_tenant_id

logger = logging.getLogger(__name__)


class ColdStore:
    """Milvus cold storage for cross-session semantic memories."""

    COLLECTION_NAME = "user_memories"
    DIMENSION = 1536

    def __init__(self, uri: str, api_key: str):
        self._api_key = api_key
        try:
            self._client = MilvusClient(uri=uri)
            self._embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small", api_key=api_key
            )
            self._ensure_collection()
        except Exception as e:
            logger.warning(f"Cold store init failed: {e}")
            self._client = None

    def _ensure_collection(self) -> None:
        if self._client is None:
            return
        if self._client.has_collection(self.COLLECTION_NAME):
            return

        schema = self._client.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("tenant_id", DataType.VARCHAR, max_length=64)
        schema.add_field("user_id", DataType.VARCHAR, max_length=64)
        schema.add_field("session_id", DataType.VARCHAR, max_length=64)
        schema.add_field("content", DataType.VARCHAR, max_length=4096)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.DIMENSION)

        self._client.create_collection(
            collection_name=self.COLLECTION_NAME,
            schema=schema,
        )
        self._client.create_index(
            collection_name=self.COLLECTION_NAME,
            index_params={
                "field_name": "vector",
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            },
        )

    async def save_memory(self, user_id: str, session_id: str, content: str) -> None:
        if self._client is None:
            return
        try:
            vector = await self._embeddings.aembed_query(content)
            self._client.insert(
                collection_name=self.COLLECTION_NAME,
                data=[{
                    "tenant_id": get_tenant_id(),
                    "user_id": user_id,
                    "session_id": session_id,
                    "content": content,
                    "vector": vector,
                }],
            )
        except Exception as e:
            logger.warning(f"Cold store save_memory failed: {e}")

    async def search_memories(self, user_id: str, query: str, top_k: int = 5) -> list[str]:
        if self._client is None:
            return []
        try:
            vector = await self._embeddings.aembed_query(query)
            results = self._client.search(
                collection_name=self.COLLECTION_NAME,
                data=[vector],
                filter=f"tenant_id == '{get_tenant_id()}' && user_id == '{user_id}'",
                limit=top_k,
                output_fields=["content"],
            )
            return [hit["entity"]["content"] for hit in results[0]]
        except Exception as e:
            logger.warning(f"Cold store search_memories failed: {e}")
            return []
