import json
from unittest.mock import MagicMock

import pytest

from cloudagent.cache import QueryCache


@pytest.mark.asyncio
async def test_l1_exact_hit():
    mock_redis = MagicMock()
    mock_redis.get.return_value = json.dumps(
        {"answer": "支持7天退款", "intent": "faq", "confidence": 0.95}
    )

    cache = QueryCache(redis_client=mock_redis)
    result = await cache.get("怎么退款？")

    assert result == {"answer": "支持7天退款", "intent": "faq", "confidence": 0.95}
    mock_redis.get.assert_called_once()


@pytest.mark.asyncio
async def test_l1_miss():
    mock_redis = MagicMock()
    mock_redis.get.return_value = None

    cache = QueryCache(redis_client=mock_redis)
    result = await cache.get("未知问题")

    assert result is None


@pytest.mark.asyncio
async def test_cache_set_and_ttl():
    mock_redis = MagicMock()

    cache = QueryCache(redis_client=mock_redis)
    await cache.set("怎么退款？", "支持7天退款", "faq", 0.95)

    mock_redis.setex.assert_called_once()
    call_args = mock_redis.setex.call_args[0]
    assert call_args[1] == 300  # TTL
    assert json.loads(call_args[2])["answer"] == "支持7天退款"
