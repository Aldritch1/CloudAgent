import logging
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    """Per-user sliding window rate limiter backed by Redis."""

    def __init__(self, redis_client, requests_per_minute: int = 60):
        self._redis = redis_client
        self._rpm = requests_per_minute
        self._window = 60

    def check(self, user_id: str) -> bool:
        if self._redis is None:
            return True
        try:
            key = f"ratelimit:{user_id}"
            now = time.time()
            window_start = now - self._window

            pipe = self._redis.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            pipe.zadd(key, {str(now): now})
            pipe.expire(key, self._window)
            _, count, _, _ = pipe.execute()

            return count < self._rpm
        except Exception as e:
            logger.warning(f"Rate limit check failed: {e}")
            return True

    def get_remaining(self, user_id: str) -> int:
        if self._redis is None:
            return self._rpm
        try:
            key = f"ratelimit:{user_id}"
            now = time.time()
            window_start = now - self._window
            self._redis.zremrangebyscore(key, 0, window_start)
            count = self._redis.zcard(key)
            return max(0, self._rpm - count)
        except Exception as e:
            logger.warning(f"Rate limit stats failed: {e}")
            return self._rpm
