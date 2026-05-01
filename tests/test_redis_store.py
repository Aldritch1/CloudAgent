import json

import fakeredis

from cloudagent.memory.redis_store import SessionStore


def test_save_and_get_session():
    redis_client = fakeredis.FakeRedis()
    store = SessionStore(redis_url="redis://fake", _redis_client=redis_client)

    messages = [{"role": "user", "content": "hello"}]
    store.save_session("sess-1", messages)

    loaded = store.get_session("sess-1")
    assert loaded == messages


def test_get_nonexistent_session_returns_empty():
    redis_client = fakeredis.FakeRedis()
    store = SessionStore(redis_url="redis://fake", _redis_client=redis_client)

    loaded = store.get_session("sess-none")
    assert loaded == []


def test_redis_fallback_on_connection_error():
    # Pass a URL that will fail connection; store should fallback to memory
    store = SessionStore(redis_url="redis://invalid:9999/0")

    messages = [{"role": "user", "content": "hi"}]
    store.save_session("sess-fb", messages)

    loaded = store.get_session("sess-fb")
    assert loaded == messages


def test_ttl_set_on_save():
    redis_client = fakeredis.FakeRedis()
    store = SessionStore(redis_url="redis://fake", _redis_client=redis_client)

    store.save_session("sess-ttl", [{"role": "user", "content": "hello"}])
    ttl = redis_client.ttl("session:sess-ttl")
    assert ttl >= 3599
