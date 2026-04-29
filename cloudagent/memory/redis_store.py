import json
import logging

import redis

logger = logging.getLogger(__name__)


class SessionStore:
    def __init__(self, redis_url: str, _redis_client=None):
        self._fallback: dict[str, list] = {}
        self._use_fallback = False

        if _redis_client is not None:
            self._redis = _redis_client
        else:
            try:
                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
            except redis.ConnectionError:
                logger.warning("Redis connection failed, falling back to in-memory store")
                self._use_fallback = True
                self._redis = None

    def get_session(self, session_id: str) -> list[dict]:
        if self._use_fallback:
            return self._fallback.get(session_id, [])

        try:
            raw = self._redis.get(f"session:{session_id}")
            if raw is None:
                return []
            return json.loads(raw)
        except redis.RedisError:
            logger.warning("Redis read error, falling back to in-memory store")
            self._use_fallback = True
            return self._fallback.get(session_id, [])

    def save_session(self, session_id: str, messages: list[dict]) -> None:
        if self._use_fallback:
            self._fallback[session_id] = messages
            return

        try:
            self._redis.setex(
                f"session:{session_id}",
                3600,
                json.dumps(messages, ensure_ascii=False),
            )
        except redis.RedisError:
            logger.warning("Redis write error, falling back to in-memory store")
            self._use_fallback = True
            self._fallback[session_id] = messages
