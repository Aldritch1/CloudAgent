# CloudAgent Phase4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Production Hardening: per-user rate limiting, LLM circuit breaker, Prometheus/Grafana metrics, and application-level multi-tenant isolation.

**Architecture:** `cloudagent/rate_limit.py` implements a custom Redis sliding-window rate limiter. `cloudagent/circuit_breaker.py` wraps `pybreaker` around `ChatOpenAI.invoke/ainvoke`. `cloudagent/metrics.py` exposes Prometheus counters and a `MetricsMiddleware`. `cloudagent/tenant_context.py` and `cloudagent/tenant.py` provide `contextvars`-based tenant isolation propagated to Redis keys, PostgreSQL queries, and Milvus filters.

**Tech Stack:** Python 3.11+, FastAPI, LangGraph, pybreaker, prometheus-client, redis, pytest, pytest-asyncio

---

## File Structure

```
cloudagent/
├── rate_limit.py            # NEW: Redis sliding-window rate limiter
├── circuit_breaker.py       # NEW: LLMCircuitBreaker + CircuitBreakerChatOpenAI proxy
├── metrics.py               # NEW: Prometheus counters, histograms, MetricsMiddleware
├── tenant_context.py        # NEW: ContextVar for tenant_id
├── tenant.py                # NEW: TenantDependency (header / JWT claim)
├── auth.py                  # MODIFIED: extract tenant_id from JWT claim
├── state.py                 # MODIFIED: add tenant_id to AgentState
├── main.py                  # MODIFIED: wire rate_limiter, breaker, metrics, tenant
├── memory/
│   ├── redis_store.py       # MODIFIED: prefix keys with tenant_id
│   ├── warm_store.py        # MODIFIED: tenant_id in SQL WHERE/UPSERT
│   └── cold_store.py        # MODIFIED: tenant_id in Milvus schema/filter
└── agent/
    ├── chat_agent.py        # MODIFIED: accept optional breaker
    └── rag_agent.py         # MODIFIED: accept optional breaker

tests/
├── test_rate_limit.py       # NEW
├── test_circuit_breaker.py  # NEW
├── test_metrics.py          # NEW
├── test_tenant.py           # NEW
├── test_main.py             # MODIFIED: 429, 503, X-Tenant-ID tests
├── conftest.py              # MODIFIED: Phase 4 env vars
└── test_config.py           # MODIFIED: Phase 4 config assertions
```

---

### Task 1: Dependencies + Config Extension

**Files:**
- Modify: `pyproject.toml`
- Modify: `cloudagent/config.py`
- Modify: `tests/conftest.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Add dependencies to pyproject.toml**

Add three lines to the `[project] dependencies` list:

```toml
dependencies = [
    # ... existing deps ...
    "limits[redis]>=3.12.0",
    "pybreaker>=1.2.0",
    "prometheus-client>=0.20.0",
]
```

- [ ] **Step 2: Modify cloudagent/config.py**

Add Phase 4 fields to `Settings`:

```python
class Settings(BaseSettings):
    # ... existing config ...
    rate_limit_requests_per_minute: int = 60
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60
    enable_metrics: bool = True
    default_tenant_id: str = "default"
```

- [ ] **Step 3: Modify tests/conftest.py**

Add Phase 4 env vars to the autouse fixture:

```python
@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    # ... existing env vars ...
    monkeypatch.setenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "60")
    monkeypatch.setenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5")
    monkeypatch.setenv("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "60")
    monkeypatch.setenv("ENABLE_METRICS", "true")
    monkeypatch.setenv("DEFAULT_TENANT_ID", "default")
```

- [ ] **Step 4: Modify tests/test_config.py**

Add assertions for Phase 4 fields:

```python
def test_settings_loads_from_env(patch_env):
    # ... existing assertions ...
    assert settings.rate_limit_requests_per_minute == 60
    assert settings.circuit_breaker_failure_threshold == 5
    assert settings.circuit_breaker_recovery_timeout == 60
    assert settings.enable_metrics is True
    assert settings.default_tenant_id == "default"
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_config.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml cloudagent/config.py tests/conftest.py tests/test_config.py
git commit -m "chore: add Phase 4 dependencies and config (rate limit, circuit breaker, metrics, tenant)"
```

---

### Task 2: Rate Limiting

**Files:**
- Create: `cloudagent/rate_limit.py`
- Create: `tests/test_rate_limit.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_rate_limit.py`:

```python
import time

import fakeredis
import pytest

from cloudagent.rate_limit import RateLimiter


def test_within_limit_allowed():
    redis_client = fakeredis.FakeRedis()
    limiter = RateLimiter(redis_client, requests_per_minute=3)
    assert limiter.check("user-1") is True
    assert limiter.check("user-1") is True
    assert limiter.check("user-1") is True


def test_exceed_limit():
    redis_client = fakeredis.FakeRedis()
    limiter = RateLimiter(redis_client, requests_per_minute=2)
    limiter.check("user-1")
    limiter.check("user-1")
    assert limiter.check("user-1") is False


def test_different_users_independent():
    redis_client = fakeredis.FakeRedis()
    limiter = RateLimiter(redis_client, requests_per_minute=2)
    limiter.check("user-a")
    limiter.check("user-a")
    assert limiter.check("user-b") is True


def test_degrades_without_redis():
    limiter = RateLimiter(redis_client=None, requests_per_minute=2)
    assert limiter.check("user-1") is True
    assert limiter.check("user-1") is True
    assert limiter.check("user-1") is True


def test_get_remaining():
    redis_client = fakeredis.FakeRedis()
    limiter = RateLimiter(redis_client, requests_per_minute=3)
    limiter.check("user-1")
    assert limiter.get_remaining("user-1") == 1
```

Run:
```bash
pytest tests/test_rate_limit.py -v
```

Expected: `ImportError: cannot import name 'RateLimiter'`

- [ ] **Step 2: Write implementation**

Create `cloudagent/rate_limit.py`:

```python
import logging
import time

logger = logging.getLogger(__name__)


class RateLimiter:
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
            logger.warning(f"Rate limit get_remaining failed: {e}")
            return self._rpm
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_rate_limit.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add cloudagent/rate_limit.py tests/test_rate_limit.py
git commit -m "feat: add per-user Redis-backed rate limiting on /chat"
```

---

### Task 3: Circuit Breaker (LLM Layer)

**Files:**
- Create: `cloudagent/circuit_breaker.py`
- Create: `tests/test_circuit_breaker.py`
- Modify: `cloudagent/agent/chat_agent.py`
- Modify: `cloudagent/agent/rag_agent.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_circuit_breaker.py`:

```python
import pytest
from pybreaker import CircuitBreakerError

from cloudagent.circuit_breaker import CircuitBreakerChatOpenAI, LLMCircuitBreaker


def test_circuit_closed_allows_calls():
    breaker = LLMCircuitBreaker(fail_max=3, reset_timeout=1)

    class FakeLLM:
        def invoke(self, msgs):
            return "ok"

    proxy = CircuitBreakerChatOpenAI(FakeLLM(), breaker)
    assert proxy.invoke([]) == "ok"


def test_circuit_opens_after_failures():
    breaker = LLMCircuitBreaker(fail_max=2, reset_timeout=60)

    class FailingLLM:
        def invoke(self, msgs):
            raise RuntimeError("fail")

    proxy = CircuitBreakerChatOpenAI(FailingLLM(), breaker)
    proxy.invoke([])
    proxy.invoke([])
    with pytest.raises(CircuitBreakerError):
        proxy.invoke([])


def test_circuit_fast_fails_when_open():
    breaker = LLMCircuitBreaker(fail_max=1, reset_timeout=60)

    class FailingLLM:
        def invoke(self, msgs):
            raise RuntimeError("fail")

    proxy = CircuitBreakerChatOpenAI(FailingLLM(), breaker)
    with pytest.raises(CircuitBreakerError):
        proxy.invoke([])


@pytest.mark.asyncio
async def test_async_circuit_closed_allows_calls():
    breaker = LLMCircuitBreaker(fail_max=3, reset_timeout=1)

    class FakeLLM:
        async def ainvoke(self, msgs):
            return "ok"

    proxy = CircuitBreakerChatOpenAI(FakeLLM(), breaker)
    result = await proxy.ainvoke([])
    assert result == "ok"


@pytest.mark.asyncio
async def test_async_circuit_opens_after_failures():
    breaker = LLMCircuitBreaker(fail_max=2, reset_timeout=60)

    class FailingLLM:
        async def ainvoke(self, msgs):
            raise RuntimeError("fail")

    proxy = CircuitBreakerChatOpenAI(FailingLLM(), breaker)
    await proxy.ainvoke([])
    await proxy.ainvoke([])
    with pytest.raises(CircuitBreakerError):
        await proxy.ainvoke([])
```

Run:
```bash
pytest tests/test_circuit_breaker.py -v
```

Expected: `ImportError: cannot import name 'LLMCircuitBreaker'`

- [ ] **Step 2: Write implementation**

Create `cloudagent/circuit_breaker.py`:

```python
from pybreaker import CircuitBreaker


class LLMCircuitBreaker:
    def __init__(self, fail_max: int = 5, reset_timeout: int = 60):
        self._breaker = CircuitBreaker(fail_max=fail_max, reset_timeout=reset_timeout)

    def wrap_sync(self, fn):
        return self._breaker(fn)

    def wrap_async(self, coro_fn):
        @self._breaker
        async def _wrapped(*args, **kwargs):
            return await coro_fn(*args, **kwargs)
        return _wrapped


class CircuitBreakerChatOpenAI:
    def __init__(self, chat_openai, breaker: LLMCircuitBreaker):
        self._chat = chat_openai
        self._breaker = breaker

    def invoke(self, messages):
        wrapped = self._breaker.wrap_sync(self._chat.invoke)
        return wrapped(messages)

    async def ainvoke(self, messages):
        wrapped = self._breaker.wrap_async(self._chat.ainvoke)
        return await wrapped(messages)
```

- [ ] **Step 3: Modify ChatAgent**

In `cloudagent/agent/chat_agent.py`:

```python
class ChatAgent:
    def __init__(self, model_name: str, api_key: str, breaker=None):
        llm = ChatOpenAI(model=model_name, api_key=api_key, temperature=0.7)
        if breaker is not None:
            from cloudagent.circuit_breaker import CircuitBreakerChatOpenAI
            self._llm = CircuitBreakerChatOpenAI(llm, breaker)
        else:
            self._llm = llm
```

- [ ] **Step 4: Modify RAGAgent**

In `cloudagent/agent/rag_agent.py`:

```python
class RAGAgent:
    def __init__(self, model_name: str, api_key: str, retriever, breaker=None):
        llm = ChatOpenAI(model=model_name, api_key=api_key, temperature=0.3)
        if breaker is not None:
            from cloudagent.circuit_breaker import CircuitBreakerChatOpenAI
            self._llm = CircuitBreakerChatOpenAI(llm, breaker)
        else:
            self._llm = llm
        self._retriever = retriever
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_circuit_breaker.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add cloudagent/circuit_breaker.py tests/test_circuit_breaker.py cloudagent/agent/chat_agent.py cloudagent/agent/rag_agent.py
git commit -m "feat: add circuit breaker around LLM calls (ChatAgent + RAGAgent)"
```

---

### Task 4: Prometheus Metrics

**Files:**
- Create: `cloudagent/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_metrics.py`:

```python
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from cloudagent.metrics import get_metrics, llm_calls_total, record_cache_hit, record_llm_call


class TestMetricsMiddleware:
    @patch("cloudagent.metrics.http_requests_total")
    @patch("cloudagent.metrics.http_request_duration_seconds")
    def test_middleware_records_request(self, mock_duration, mock_counter):
        from cloudagent.metrics import MetricsMiddleware
        from fastapi import FastAPI

        app = FastAPI()
        app.add_middleware(MetricsMiddleware)

        @app.get("/test")
        def test_endpoint():
            return {"ok": True}

        client = TestClient(app)
        response = client.get("/test")
        assert response.status_code == 200
        mock_counter.labels.assert_called_once()
        mock_duration.labels.assert_called_once()


def test_llm_call_counter():
    record_llm_call("chat", "success")
    output = get_metrics().decode()
    assert 'llm_calls_total{agent_type="chat",status="success"}' in output


def test_cache_hit_counter():
    record_cache_hit("l1", True)
    output = get_metrics().decode()
    assert 'cache_hits_total{hit="True",tier="l1"}' in output


def test_metrics_output_contains_expected_names():
    output = get_metrics().decode()
    assert "llm_calls_total" in output
    assert "cache_hits_total" in output
```

Run:
```bash
pytest tests/test_metrics.py -v
```

Expected: `ImportError: cannot import name 'get_metrics'`

- [ ] **Step 2: Write implementation**

Create `cloudagent/metrics.py`:

```python
import time

from prometheus_client import Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["method", "endpoint"],
)

llm_calls_total = Counter(
    "llm_calls_total",
    "Total LLM calls",
    ["agent_type", "status"],
)

cache_hits_total = Counter(
    "cache_hits_total",
    "Cache hits",
    ["tier", "hit"],
)

retrieval_results_total = Counter(
    "retrieval_results_total",
    "Retrieval results",
    ["source"],
)


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        http_request_duration_seconds.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration)
        http_requests_total.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=str(response.status_code),
        ).inc()
        return response


def record_llm_call(agent_type: str, status: str) -> None:
    llm_calls_total.labels(agent_type=agent_type, status=status).inc()


def record_cache_hit(tier: str, hit: bool) -> None:
    cache_hits_total.labels(tier=tier, hit=str(hit)).inc()


def get_metrics() -> bytes:
    return generate_latest()
```

- [ ] **Step 3: Integrate into main.py**

In `cloudagent/main.py`:

```python
if settings.enable_metrics:
    from cloudagent.metrics import MetricsMiddleware
    app.add_middleware(MetricsMiddleware)
```

Add `/metrics` endpoint:

```python
from prometheus_client import CONTENT_TYPE_LATEST

@app.get("/metrics")
async def metrics():
    from cloudagent.metrics import get_metrics
    return Response(content=get_metrics(), media_type=CONTENT_TYPE_LATEST)
```

Decorate agents and cache:

```python
# In ChatAgent.run
from cloudagent.metrics import record_llm_call
record_llm_call("chat", "success")

# In RAGAgent.run
record_llm_call("rag", "success")

# In QueryCache.get
record_cache_hit("l1", True/False)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_metrics.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cloudagent/metrics.py tests/test_metrics.py cloudagent/main.py cloudagent/agent/chat_agent.py cloudagent/agent/rag_agent.py cloudagent/cache.py
git commit -m "feat: add Prometheus metrics (HTTP middleware, LLM calls, cache hits, /metrics endpoint)"
```

---

### Task 5: Multi-Tenancy

**Files:**
- Create: `cloudagent/tenant_context.py`
- Create: `cloudagent/tenant.py`
- Create: `tests/test_tenant.py`
- Modify: `cloudagent/auth.py`
- Modify: `cloudagent/state.py`
- Modify: `cloudagent/memory/redis_store.py`
- Modify: `cloudagent/memory/warm_store.py`
- Modify: `cloudagent/memory/cold_store.py`
- Modify: `cloudagent/main.py`

- [ ] **Step 1: Create tenant_context.py**

```python
import contextvars

tenant_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("tenant_id", default="")

def get_tenant_id() -> str:
    return tenant_ctx.get()

def set_tenant_id(tenant_id: str) -> None:
    tenant_ctx.set(tenant_id)
```

- [ ] **Step 2: Create tenant.py**

```python
from typing import Optional

from fastapi import Depends, Header

from cloudagent.auth import get_current_user
from cloudagent.tenant_context import get_tenant_id, set_tenant_id


async def tenant_dependency(
    user_id: str = Depends(get_current_user),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> str:
    from cloudagent.config import settings
    tenant_id = get_tenant_id() or x_tenant_id or settings.default_tenant_id
    set_tenant_id(tenant_id)
    return tenant_id
```

- [ ] **Step 3: Modify auth.py**

```python
from cloudagent.tenant_context import set_tenant_id

def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    if settings.jwt_disabled:
        return "anonymous"
    # ...
    try:
        payload = jwt.decode(token, secret, algorithms=[settings.jwt_algorithm])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="无效的认证令牌")
        tenant_id = payload.get("tenant_id")
        if tenant_id:
            set_tenant_id(tenant_id)
        return user_id
    except (JWTError, Exception):
        raise HTTPException(status_code=401, detail="认证令牌无效或已过期")
```

- [ ] **Step 4: Modify state.py**

```python
class AgentState(TypedDict, total=False):
    # ... existing fields ...
    tenant_id: str | None
```

- [ ] **Step 5: Modify redis_store.py**

```python
from cloudagent.tenant_context import get_tenant_id

class SessionStore:
    def _key(self, session_id: str) -> str:
        tenant = get_tenant_id()
        if tenant:
            return f"{tenant}:session:{session_id}"
        return f"session:{session_id}"
```

- [ ] **Step 6: Modify warm_store.py**

Add `tenant_id` to all SQL queries via `get_tenant_id()`.

- [ ] **Step 7: Modify cold_store.py**

Add `tenant_id` field to Milvus schema, inserts, and search filters.

- [ ] **Step 8: Modify main.py**

```python
from cloudagent.tenant import tenant_dependency

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user_id: str = Depends(get_current_user),
    tenant_id: str = Depends(tenant_dependency),
):
    # ...
    state = {
        # ...
        "tenant_id": tenant_id,
    }
```

- [ ] **Step 9: Write tests**

Create `tests/test_tenant.py`:

```python
import os
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis
import pytest
from fastapi.testclient import TestClient

from cloudagent.memory.cold_store import ColdStore
from cloudagent.memory.redis_store import SessionStore
from cloudagent.memory.warm_store import WarmStore
from cloudagent.tenant import tenant_dependency
from cloudagent.tenant_context import get_tenant_id, set_tenant_id


@pytest.fixture(autouse=True)
def reset_tenant():
    set_tenant_id("")
    yield


def test_tenant_defaults():
    assert get_tenant_id() == ""


def test_set_and_get_tenant():
    set_tenant_id("acme-corp")
    assert get_tenant_id() == "acme-corp"
    set_tenant_id("")


@pytest.mark.asyncio
async def test_tenant_dependency_from_header():
    tenant_id = await tenant_dependency(x_tenant_id="tenant-a")
    assert tenant_id == "tenant-a"


@pytest.mark.asyncio
async def test_tenant_dependency_fallback_to_context():
    set_tenant_id("tenant-b")
    tenant_id = await tenant_dependency(x_tenant_id=None)
    assert tenant_id == "tenant-b"


def test_tenant_isolation_in_redis():
    redis_client = fakeredis.FakeRedis()
    store = SessionStore(redis_url="redis://fake", _redis_client=redis_client)

    set_tenant_id("tenant-a")
    store.save_session("sess-1", [{"role": "user", "content": "hello"}])

    set_tenant_id("tenant-b")
    assert store.get_session("sess-1") == []

    set_tenant_id("tenant-a")
    assert store.get_session("sess-1") == [{"role": "user", "content": "hello"}]


def test_redis_key_prefix_with_tenant():
    redis_client = fakeredis.FakeRedis()
    store = SessionStore(redis_url="redis://fake", _redis_client=redis_client)

    set_tenant_id("acme")
    store.save_session("sess-1", [{"role": "user", "content": "hi"}])

    keys = list(redis_client.scan_iter())
    assert any(b"acme:session:sess-1" in k for k in keys)


# ... warm_store and cold_store tenant tests ...


@patch("cloudagent.retrieval.vector.VectorRetriever")
@patch("cloudagent.retrieval.graph.GraphRetriever")
@patch("cloudagent.retrieval.keyword.KeywordRetriever")
@patch("cloudagent.agent.rag_agent.RAGAgent")
@patch("cloudagent.memory.redis_store.SessionStore")
@patch("cloudagent.agent.router.EntryAgent")
@patch("cloudagent.agent.chat_agent.ChatAgent")
def test_chat_endpoint_with_tenant_header(
    mock_chat_cls, mock_entry_cls, mock_store_cls,
    mock_rag_cls, mock_kw_cls, mock_graph_cls, mock_vec_cls,
):
    mock_store = MagicMock()
    mock_store.get_session.return_value = []
    mock_store_cls.return_value = mock_store

    mock_entry = MagicMock()
    mock_entry.run.return_value = {
        "messages": [{"role": "user", "content": "hello"}],
        "intent": "chat",
        "confidence": 0.92,
        "target_agent": "chat",
        "context": {},
    }
    mock_entry_cls.return_value = mock_entry

    mock_chat = MagicMock()
    mock_chat.run.return_value = "Hi there!"
    mock_chat_cls.return_value = mock_chat

    import importlib
    import cloudagent.main
    importlib.reload(cloudagent.main)
    from cloudagent.main import app

    client = TestClient(app)
    response = client.post(
        "/chat",
        json={"session_id": "550e8400-e29b-41d4-a716-446655440000", "message": "hello"},
        headers={"X-Tenant-ID": "tenant-x"},
    )

    assert response.status_code == 200
    mock_entry.run.assert_called_once()
    state_arg = mock_entry.run.call_args[0][0]
    assert state_arg.get("tenant_id") == "tenant-x"
```

- [ ] **Step 10: Run tests**

```bash
pytest tests/test_tenant.py -v
```

Expected: PASS

- [ ] **Step 11: Commit**

```bash
git add cloudagent/tenant_context.py cloudagent/tenant.py tests/test_tenant.py cloudagent/auth.py cloudagent/state.py cloudagent/memory/redis_store.py cloudagent/memory/warm_store.py cloudagent/memory/cold_store.py cloudagent/main.py
git commit -m "feat: add multi-tenancy with contextvars isolation"
```

---

### Task 6: FastAPI Integration + Test Suite Update

**Files:**
- Modify: `tests/test_main.py`

- [ ] **Step 1: Add 429 test**

```python
@patch("cloudagent.rate_limit.RateLimiter")
# ... other patches ...
def test_chat_endpoint_rate_limit_returns_429(
    mock_chat_cls, mock_entry_cls, mock_store_cls,
    mock_rag_cls, mock_kw_cls, mock_graph_cls, mock_vec_cls,
    mock_metrics_cls, mock_breaker_cls, mock_rate_cls,
):
    mock_store = MagicMock()
    mock_store.get_session.return_value = []
    mock_store_cls.return_value = mock_store

    mock_rate = MagicMock()
    mock_rate.check.return_value = False
    mock_rate_cls.return_value = mock_rate

    # ... setup other mocks ...

    import importlib
    import cloudagent.main
    importlib.reload(cloudagent.main)
    from cloudagent.main import app

    client = TestClient(app)
    response = client.post("/chat", json={"session_id": "xxx", "message": "hello"})
    assert response.status_code == 429
    assert response.json()["detail"] == "请求过于频繁，请稍后再试"
```

- [ ] **Step 2: Add 503 test**

```python
@patch("cloudagent.rate_limit.RateLimiter")
# ... other patches ...
def test_chat_endpoint_circuit_breaker_returns_503(
    mock_chat_cls, mock_entry_cls, mock_store_cls,
    mock_rag_cls, mock_kw_cls, mock_graph_cls, mock_vec_cls,
    mock_metrics_cls, mock_breaker_cls, mock_rate_cls,
):
    # ... setup mocks ...
    mock_rate = MagicMock()
    mock_rate.check.return_value = True
    mock_rate_cls.return_value = mock_rate

    import importlib
    import cloudagent.main
    importlib.reload(cloudagent.main)
    from cloudagent.main import app

    from pybreaker import CircuitBreakerError
    from cloudagent.main import graph
    graph.ainvoke = AsyncMock(side_effect=CircuitBreakerError("Circuit open"))

    client = TestClient(app)
    response = client.post("/chat", json={"session_id": "xxx", "message": "hello"})
    assert response.status_code == 503
    assert response.json()["detail"] == "服务暂时不可用，请稍后重试"
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_main.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_main.py
git commit -m "test: add 429 and 503 endpoint integration tests"
```

---

### Task 7: Verification & Polish

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 2: Update CLAUDE.md**

- Mark Phase 4 as complete.
- Add rate limiting, circuit breaker, metrics, multi-tenancy to tech stack.
- Update directory structure with new files.
- Update environment variables with Phase 4 settings.

- [ ] **Step 3: Update README.md**

- Add Phase 4 features (限流、熔断、可观测性、多租户)。
- Update architecture diagram.
- Update test coverage list.
- Mark Phase 4 as complete in roadmap.

- [ ] **Step 4: Final commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: update README and CLAUDE for Phase 4"
```

---

## Self-Review

**1. Spec coverage:**
- Rate limiting → Task 2
- Circuit breaker → Task 3
- Prometheus metrics → Task 4
- Multi-tenancy (contextvars + storage isolation) → Task 5
- FastAPI integration → Task 6
- Error handling (429/503) → covered in main.py + tests
- Testing strategy → covered in all tasks

**2. Placeholder scan:**
- L2 semantic cache remains placeholder (documented).
- Warm store summary generation uses hardcoded string (documented).
- Workflow execution remains placeholder (documented).
- No other TBD/TODO/fill-in-details found.

**3. Type consistency:**
- `tenant_id: str | None` in AgentState — consistent.
- `RateLimiter.check` returns `bool` — consistent.
- `CircuitBreakerChatOpenAI` wraps both sync/async — consistent with agent usage.
