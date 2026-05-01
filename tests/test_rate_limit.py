from unittest.mock import MagicMock, patch

import pytest
from fakeredis import FakeRedis

from cloudagent.rate_limit import RateLimiter


def test_within_limit_allowed():
    redis = FakeRedis()
    limiter = RateLimiter(redis, requests_per_minute=5)
    assert limiter.check("user-1") is True
    assert limiter.check("user-1") is True


def test_exceed_limit():
    redis = FakeRedis()
    limiter = RateLimiter(redis, requests_per_minute=2)
    assert limiter.check("user-1") is True
    assert limiter.check("user-1") is True
    assert limiter.check("user-1") is False


def test_different_users_independent():
    redis = FakeRedis()
    limiter = RateLimiter(redis, requests_per_minute=2)
    assert limiter.check("user-1") is True
    assert limiter.check("user-1") is True
    assert limiter.check("user-2") is True


def test_degrades_without_redis():
    limiter = RateLimiter(None, requests_per_minute=5)
    assert limiter.check("user-1") is True


def test_get_remaining():
    redis = FakeRedis()
    limiter = RateLimiter(redis, requests_per_minute=5)
    limiter.check("user-1")
    limiter.check("user-1")
    remaining = limiter.get_remaining("user-1")
    assert remaining == 3
