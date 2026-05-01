import hashlib
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import redis

logger = logging.getLogger(__name__)


class QueryCache:
    """L1 exact-match (Redis) + L2 semantic (Milvus) query cache."""

    def __init__(self, redis_client=None, milvus_uri: str = "", api_key: str = ""):
        self._redis = redis_client
        self._milvus_uri = milvus_uri
        self._api_key = api_key

    def _l1_key(self, query: str) -> str:
        normalized = query.strip().lower()
        h = hashlib.sha256(normalized.encode()).hexdigest()
        return f"cache:l1:{h}"

    async def get(self, query: str) -> dict | None:
        # L1 exact match
        if self._redis is not None:
            try:
                raw = self._redis.get(self._l1_key(query))
                if raw:
                    return json.loads(raw)
            except Exception as e:
                logger.warning(f"L1 cache get failed: {e}")

        # L2 semantic match (placeholder for future full implementation)
        # Would query Milvus collection 'semantic_cache' with embedding similarity
        return None

    async def set(self, query: str, answer: str, intent: str, confidence: float) -> None:
        if self._redis is not None:
            try:
                value = json.dumps(
                    {"answer": answer, "intent": intent, "confidence": confidence},
                    ensure_ascii=False,
                )
                self._redis.setex(self._l1_key(query), 300, value)
            except Exception as e:
                logger.warning(f"L1 cache set failed: {e}")
