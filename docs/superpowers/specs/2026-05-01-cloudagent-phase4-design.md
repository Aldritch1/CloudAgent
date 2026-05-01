# CloudAgent 阶段4设计文档 — 生产加固

**版本：** v1.0
**日期：** 2026-05-01
**目标：** 实现限流、熔断、Prometheus 可观测性、多租户隔离，使 CloudAgent 具备生产环境部署能力

---

## 1. 总体架构

阶段4在阶段3的 LangGraph StateGraph + 分层记忆基础上，增加四层生产加固能力：

```
┌─────────────────────────────────────────┐
│ API网关层：FastAPI + JWT Bearer + 多租户    │
│ 职责：/chat 端点，请求校验，限流检查，Agent 驱动 │
├─────────────────────────────────────────┤
│ 可观测性层：Prometheus + MetricsMiddleware  │
│ 职责：HTTP 延迟直方图、LLM/cache/检索计数器   │
├─────────────────────────────────────────┤
│ 稳定性层：限流 + 熔断                        │
│ 职责：Redis 滑动窗口限流、pybreaker LLM 熔断  │
├─────────────────────────────────────────┤
│ Agent引擎层：LangGraph StateGraph         │
│ 职责：load_memory → entry → route → ...   │
├─────────────────────────────────────────┤
│ 多租户隔离层：contextvars + 存储过滤         │
│ 职责：Redis key 前缀、PG/Milvus tenant_id    │
├─────────────────────────────────────────┤
│ 数据层：Milvus + Neo4j + PG + Redis       │
└─────────────────────────────────────────┘
```

---

## 2. 基础设施

### 2.1 依赖扩展

`pyproject.toml` 新增：

```toml
dependencies = [
    # ... 现有依赖 ...
    "limits[redis]>=3.12.0",
    "pybreaker>=1.2.0",
    "prometheus-client>=0.20.0",
]
```

> 注：`limits[redis]` 的 `RedisStorage` API 与现有 redis client 不兼容，最终采用自定义 Redis sorted-set 滑动窗口实现。

### 2.2 配置扩展

`cloudagent/config.py` 新增：

```python
class Settings(BaseSettings):
    # ... 现有配置 ...
    rate_limit_requests_per_minute: int = 60
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60
    enable_metrics: bool = True
    default_tenant_id: str = "default"
```

---

## 3. 文件结构

```
cloudagent/
├── rate_limit.py            # NEW: 滑动窗口限流器
├── circuit_breaker.py       # NEW: LLMCircuitBreaker + CircuitBreakerChatOpenAI
├── metrics.py               # NEW: Prometheus 指标定义 + MetricsMiddleware
├── tenant_context.py        # NEW: ContextVar 租户上下文
├── tenant.py                # NEW: TenantDependency 依赖注入
├── auth.py                  # MODIFIED: JWT 解析 + 设置租户上下文
├── state.py                 # MODIFIED: AgentState 增加 tenant_id
├── main.py                  # MODIFIED: 接入限流/熔断/指标/多租户
├── memory/
│   ├── redis_store.py       # MODIFIED: key 前缀加入 tenant_id
│   ├── warm_store.py        # MODIFIED: SQL 加入 tenant_id
│   └── cold_store.py        # MODIFIED: Milvus schema/filter 加入 tenant_id
└── agent/
    ├── chat_agent.py        # MODIFIED: 可选 breaker 参数
    └── rag_agent.py         # MODIFIED: 可选 breaker 参数

tests/
├── test_rate_limit.py       # NEW
├── test_circuit_breaker.py  # NEW
├── test_metrics.py          # NEW
├── test_tenant.py           # NEW
├── test_main.py             # MODIFIED: 429/503/多租户集成测试
└── conftest.py              # MODIFIED: Phase 4 环境变量
```

---

## 4. 模块设计

### 4.1 限流器 (`rate_limit.py`)

自定义 Redis 滑动窗口（替代 `limits` 库）：

```python
class RateLimiter:
    def __init__(self, redis_client, requests_per_minute: int = 60):
        self._redis = redis_client
        self._rpm = requests_per_minute
        self._window = 60

    def check(self, user_id: str) -> bool:
        key = f"ratelimit:{user_id}"
        now = time.time()
        pipe = self._redis.pipeline()
        pipe.zremrangebyscore(key, 0, now - self._window)
        pipe.zcard(key)
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, self._window)
        _, count, _, _ = pipe.execute()
        return count < self._rpm
```

**关键原则：**
- Redis 不可用时不阻塞请求（`redis_client=None` → 始终返回 `True`）。
- Key 格式：`ratelimit:{user_id}`，sorted set 以时间戳为 score。

### 4.2 熔断器 (`circuit_breaker.py`)

`pybreaker.CircuitBreaker` 包装 + ChatOpenAI 代理：

```python
class LLMCircuitBreaker:
    def __init__(self, fail_max=5, reset_timeout=60):
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
        return self._breaker.wrap_sync(self._chat.invoke)(messages)

    async def ainvoke(self, messages):
        return await self._breaker.wrap_async(self._chat.ainvoke)(messages)
```

**关键原则：**
- 熔断状态 OPEN 时抛出 `CircuitBreakerError`，FastAPI 捕获后返回 HTTP 503。
- 同时支持 `invoke`（ChatAgent）和 `ainvoke`（RAGAgent）。

### 4.3 Prometheus 指标 (`metrics.py`)

模块级指标定义：

```python
http_requests_total = Counter(...)
http_request_duration_seconds = Histogram(...)
llm_calls_total = Counter(...)
cache_hits_total = Counter(...)
retrieval_results_total = Counter(...)
```

`MetricsMiddleware` 基于 `BaseHTTPMiddleware`：

```python
class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        http_request_duration_seconds.labels(...).observe(duration)
        http_requests_total.labels(...).inc()
        return response
```

`/metrics` 端点返回 `generate_latest()` bytes。

### 4.4 多租户上下文 (`tenant_context.py`, `tenant.py`)

```python
# tenant_context.py
tenant_ctx: ContextVar[str] = ContextVar("tenant_id", default="")

def get_tenant_id() -> str:
    return tenant_ctx.get()

def set_tenant_id(tenant_id: str) -> None:
    tenant_ctx.set(tenant_id)

# tenant.py
async def tenant_dependency(
    user_id: str = Depends(get_current_user),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> str:
    tenant_id = get_tenant_id() or x_tenant_id or settings.default_tenant_id
    set_tenant_id(tenant_id)
    return tenant_id
```

**关键原则：**
- JWT `tenant_id` claim 优先级高于 `X-Tenant-ID` header。
- `auth.py` 解析 JWT 时调用 `set_tenant_id()`。
- `tenant_dependency` 在 FastAPI 端点注入，保证每个请求都有明确的租户上下文。

### 4.5 存储层租户隔离

| 存储 | 隔离方式 |
|------|----------|
| Redis (`redis_store.py`) | `get_tenant_id()` 前缀：`{tenant}:session:{sid}`，空租户时不加前缀 |
| PostgreSQL (`warm_store.py`) | WHERE `tenant_id = $1 AND user_id = $2`；UPSERT ON CONFLICT `(tenant_id, user_id)` |
| Milvus (`cold_store.py`) | schema 增加 `tenant_id` VARCHAR 字段；filter: `tenant_id == '{t}' && user_id == '{u}'` |

### 4.6 AgentState 扩展 (`state.py`)

```python
class AgentState(TypedDict, total=False):
    # ... 现有字段 ...
    tenant_id: str | None
```

`main.py` 构建初始 state 时注入 `tenant_id`。

---

## 5. 数据流

```
POST /chat
Authorization: Bearer <jwt>
X-Tenant-ID: acme-corp
{ "session_id": "xxx", "message": "怎么退款？" }

  1. MetricsMiddleware 记录请求延迟
  2. tenant_dependency: JWT claim tenant_id="acme-corp" → set_tenant_id()
  3. rate_limiter.check(user_id): Redis 滑动窗口检查 → 通过
  4. get_current_user: 解析 JWT → user_id
  5. main.py 构建 AgentState（含 tenant_id="acme-corp"）
  6. cache.get(): L1 Redis 检查 → miss
  7. graph.ainvoke(state)
     7a. load_memory_node: SessionStore 读取 key = "acme-corp:session:xxx"
     7b. entry → route → rag_node
     7c. rag_agent.run(): LLM 调用受 CircuitBreakerChatOpenAI 保护
     7d. save_memory_node: 持久化到 tenant 隔离存储
  8. cache.set(): L1 Redis 写入 key 不含租户（缓存按查询内容独立）
  9. 返回 ChatResponse
```

---

## 6. 错误处理

| 场景 | HTTP 状态码 | 处理策略 |
|------|-------------|----------|
| 限流触发 | 429 | `X-RateLimit-Remaining: 0`，提示"请求过于频繁，请稍后再试" |
| 熔断器开启 | 503 | 提示"服务暂时不可用，请稍后重试" |
| LLM 调用失败（熔断闭合） | 500 | 提示"服务暂时繁忙，请稍后重试" |
| Redis 不可用（限流） | 无影响 | 降级为允许所有请求 |
| 租户未指定 | 200 | 使用 `default_tenant_id` |

---

## 7. 测试策略

| 测试类型 | 覆盖内容 | 工具/方法 |
|----------|----------|-----------|
| 单元测试 | RateLimiter：命中/未命中、独立用户、Redis 降级 | fakeredis |
| 单元测试 | CircuitBreaker：闭合/开启/半开、同步/异步 | mock pybreaker |
| 单元测试 | Metrics：计数器增量、中间件记录 | prometheus_client.REGISTRY |
| 单元测试 | TenantContext：header/JWT/默认值、ContextVar 隔离 | 直接函数调用 |
| 单元测试 | RedisStore：租户 key 前缀、空租户兼容 | fakeredis |
| 单元测试 | WarmStore：SQL 含 tenant_id | mock asyncpg |
| 单元测试 | ColdStore：Milvus filter 含 tenant_id | mock MilvusClient |
| 集成测试 | `/chat` 429 响应 | TestClient + mock rate_limiter.check=False |
| 集成测试 | `/chat` 503 响应 | TestClient + mock graph.ainvoke 抛 CircuitBreakerError |
| 集成测试 | `/chat` X-Tenant-ID header | TestClient + 验证 state 含 tenant_id |

---

## 8. 阶段4明确边界（不做）

- Redis 持久化 checkpointer（当前使用 InMemorySaver）— 阶段5+
- L2 语义缓存的完整 Milvus 实现（当前为 placeholder）— 阶段5+
- Grafana Dashboard 配置 — 运维侧
- 数据面多租户（数据库级 schema/pg_namespace 隔离）— 超出范围
- MCP 工具生态 — 阶段5
- SSE 流式输出、Vue3 前端 — 阶段6

---

## 9. 成功标准

1. `pytest tests/ -v` 全部通过（含阶段1~4测试）。
2. 超过 60 RPM 时 `/chat` 返回 HTTP 429。
3. LLM 连续失败 5 次后触发熔断，`/chat` 返回 HTTP 503。
4. Prometheus `/metrics` 端点可抓取 `http_requests_total`、`llm_calls_total`、`cache_hits_total`。
5. 携带 `X-Tenant-ID: tenant-a` 的请求与 `tenant-b` 的数据完全隔离。
6. 外部服务（Redis）故障时，限流自动降级，服务不中断。
