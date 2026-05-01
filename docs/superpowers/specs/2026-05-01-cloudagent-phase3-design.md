# CloudAgent 阶段3设计文档 — 记忆 + 安全 + 优化

**版本：** v1.0  
**日期：** 2026-05-01  
**目标：** 实现 JWT 认证、分层记忆（Redis/PG/Milvus）、L1/L2 查询缓存、澄清追问、HITL 人机协同，并将主流程重构为 LangGraph StateGraph

---

## 1. 总体架构

阶段3在阶段2的 Hybrid RAG 基础上，引入 **LangGraph StateGraph** 作为核心编排引擎，替代 `main.py` 中的直接函数调用。所有业务逻辑以节点形式接入图，支持原生中断（interrupt）实现 HITL。

```
┌─────────────────────────────────────────┐
│ API网关层：FastAPI + JWT Bearer 认证      │
│ 职责：/chat 端点，请求校验，Agent 状态机驱动 │
├─────────────────────────────────────────┤
│ Agent引擎层：LangGraph StateGraph         │
│ 职责：load_memory → entry → route →      │
│       chat / rag / clarify / hitl_request │
│       → [INTERRUPT] → hitl_resume        │
│       → save_memory → END                │
├─────────────────────────────────────────┤
│ Agent实现层                               │
│ 职责：EntryAgent（意图+路由+澄清）         │
│       ChatAgent（LLM直连）                │
│       RAGAgent（检索增强问答）             │
├─────────────────────────────────────────┤
│ 记忆层：cloudagent/memory/               │
│ 职责：Redis hot（会话消息）               │
│       PostgreSQL warm（用户画像+摘要）    │
│       Milvus cold（跨会话语义记忆）       │
├─────────────────────────────────────────┤
│ 缓存层：cloudagent/cache.py              │
│ 职责：L1 Redis 精确匹配（<50ms）          │
│       L2 Milvus 语义匹配（placeholder）   │
├─────────────────────────────────────────┤
│ HITL层：cloudagent/hitl.py               │
│ 职责：敏感操作检测、确认/拒绝关键词匹配     │
├─────────────────────────────────────────┤
│ 数据层：Milvus + Neo4j + PG + Redis      │
│ 职责：向量语义检索、图谱关系、全文检索、会话  │
└─────────────────────────────────────────┘
```

---

## 2. 基础设施

### 2.1 依赖扩展

`pyproject.toml` 新增：

```toml
dependencies = [
    # ... 现有依赖 ...
    "python-jose[cryptography]>=3.3.0",
    "passlib[bcrypt]>=1.7.4",
    "numpy>=1.26.0",
]
```

### 2.2 配置扩展

`cloudagent/config.py` 新增：

```python
class Settings(BaseSettings):
    # ... 现有配置 ...
    jwt_secret: SecretStr = SecretStr("")
    jwt_algorithm: str = "HS256"
    jwt_disabled: bool = False
```

- `jwt_secret`：JWT 签名密钥，至少32位字符。为空时认证自动降级为匿名。
- `jwt_disabled=true`：开发/测试环境一键关闭认证。

---

## 3. 文件结构

```
cloudagent/
├── auth.py                  # NEW: JWT Bearer 解析，get_current_user
├── state.py                 # NEW: AgentState TypedDict
├── graph.py                 # NEW: StateGraph 构建器 + GraphNodes
├── cache.py                 # NEW: QueryCache (L1 Redis + L2 Milvus)
├── hitl.py                  # NEW: HITLManager 敏感操作确认
├── main.py                  # MODIFIED: 图编排替代直接调用
├── models.py                # MODIFIED: ChatRequest.action / ChatResponse.action_required
├── config.py                # MODIFIED: JWT 配置字段
├── agent/
│   ├── router.py            # MODIFIED: 澄清追问逻辑
│   ├── chat_agent.py        # 不变
│   └── rag_agent.py         # 不变
├── memory/
│   ├── redis_store.py       # 不变 (热存储)
│   ├── warm_store.py        # NEW: PostgreSQL 用户画像+摘要
│   ├── cold_store.py        # NEW: Milvus 跨会话记忆
│   └── manager.py           # NEW: TieredMemoryManager 分层聚合
└── retrieval/               # 不变 (阶段2已交付)

tests/
├── test_auth.py             # NEW: JWT 认证测试
├── test_graph.py            # NEW: LangGraph 流程+中断测试
├── test_cache.py            # NEW: 缓存测试
├── test_hitl.py             # NEW: HITL 状态机测试
├── test_memory_manager.py   # NEW: 分层记忆管理器测试
├── test_warm_store.py       # NEW: PostgreSQL 温存储测试
├── test_cold_store.py       # NEW: Milvus 冷存储测试
├── test_main.py             # MODIFIED: 图编排+认证集成
├── test_router.py           # MODIFIED: 澄清逻辑覆盖
├── conftest.py              # MODIFIED: JWT_DISABLED=true
└── retrieval/               # 不变
```

---

## 4. 模块设计

### 4.1 JWT 认证 (`auth.py`)

```python
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    if settings.jwt_disabled: return "anonymous"
    secret = settings.jwt_secret.get_secret_value()
    if not secret: return "anonymous"
    if not token: raise HTTPException(401, ...)
    payload = jwt.decode(token, secret, algorithms=[settings.jwt_algorithm])
    return payload.get("sub")
```

**关键原则：**
- `auto_error=False`：允许无 Token 请求进入自定义降级逻辑。
- `jwt_disabled` 或 `jwt_secret` 为空时自动降级为 `"anonymous"`，开发/测试零配置可用。
- 解析失败统一返回 401，FastAPI 依赖注入到 `/chat` 端点。

### 4.2 状态模型 (`state.py`)

```python
class AgentState(TypedDict, total=False):
    messages: list[dict]
    user_id: str
    session_id: str
    intent: str | None
    confidence: float
    target_agent: str | None
    context: dict
    retrieved_context: list[str]
    response: str | None
    clarification_question: str | None
    pending_action: dict | None
    action_required: str | None   # "confirm" | "clarify" | None
```

`total=False` 允许节点只返回变更的字段，LangGraph 自动合并到现有状态。

### 4.3 LangGraph StateGraph (`graph.py`)

**节点列表：**

| 节点 | 类型 | 职责 |
|------|------|------|
| `load_memory` | async | 从 TieredMemoryManager 加载热/温/冷记忆，追加用户新消息 |
| `entry` | sync | 调用 EntryAgent 识别意图，填充 `intent`/`confidence`/`target_agent` |
| `route` | sync (conditional) | 根据 `target_agent` + `confidence` 分支到下游节点 |
| `chat` | sync | ChatAgent 生成回复 |
| `rag` | async | RAGAgent 检索 + 生成回复 |
| `workflow_placeholder` | sync | 占位："业务办理功能正在开发中" |
| `clarify` | sync | 返回澄清问题，`action_required="clarify"` |
| `hitl_request` | sync | 设置 `pending_action`，返回确认消息，`action_required="confirm"` |
| `hitl_resume` | sync | 解析用户确认/拒绝，执行或取消敏感操作 |
| `save_memory` | async | 持久化本轮对话到分层记忆 |

**条件路由逻辑：**

```python
def route_node(self, state: AgentState) -> str:
    target = state.get("target_agent")
    confidence = state.get("confidence", 0.0)

    if target == "clarify": return "clarify"
    if target == "workflow" and self.hitl.is_sensitive("workflow", {}):
        return "hitl_request"
    if target == "workflow": return "workflow_placeholder"
    if target in ("chat", "faq") and confidence > 0.5: return target
    return "chat"
```

**HITL 中断：**

```python
checkpointer = InMemorySaver()
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["hitl_resume"]
)
```

- 当敏感 workflow 触发 `hitl_request` → `hitl_resume` 边时，图在 `hitl_resume` 前暂停。
- FastAPI 端点检测到 `action_required == "confirm"` 后返回特殊响应。
- 客户端再次请求（带 `action: "confirm"` 或 `"reject"`），端点调用 `graph.ainvoke(None, config)` 恢复执行。
- `thread_id = session_id`，同一 session 的多次请求共享图状态。

### 4.4 分层记忆 (`memory/`)

#### 4.4.1 WarmStore (`warm_store.py`)

基于 `asyncpg` 的 PostgreSQL 存储：

| 方法 | 职责 |
|------|------|
| `get_user_profile(user_id)` | 查询 `user_profiles` 表 |
| `save_user_profile(user_id, profile)` | UPSERT 用户画像 |
| `get_session_history(user_id, limit)` | 查询最近会话摘要 |
| `save_summary(session_id, user_id, summary)` | UPSERT 会话摘要 |

**降级**：任何 `asyncpg` 异常返回 `None` / `[]`，记录 warning。

#### 4.4.2 ColdStore (`cold_store.py`)

基于 `MilvusClient` + `OpenAIEmbeddings` 的语义记忆：

- Collection：`user_memories`（dim=1536, metric=COSINE）
- 字段：`id`, `user_id`, `session_id`, `content`, `vector`
- `save_memory(user_id, session_id, content)`：嵌入后写入
- `search_memories(user_id, query, top_k)`：按用户过滤语义搜索

**降级**：初始化失败或查询异常返回 `[]`。

#### 4.4.3 TieredMemoryManager (`manager.py`)

```python
class TieredMemoryManager:
    def __init__(self, hot_store, warm_store, cold_store):
        ...

    async def get_context(session_id, user_id) -> dict:
        # 聚合三层：hot messages + warm profile + cold memories
        return {"messages": [...], "profile": {...}, "memories": [...]}

    async def save_turn(session_id, user_id, messages):
        # 写入 hot；每5轮生成摘要写入 warm + cold
```

**主应用中的特殊处理**：`main.py` 初始化 `TieredMemoryManager(hot_store=None)`，让 `main.py` 直接管理 Redis 会话读写，避免图的 `save_memory_node` 和主流程重复保存。

### 4.5 L1/L2 查询缓存 (`cache.py`)

```python
class QueryCache:
    def __init__(self, redis_client, milvus_uri, api_key):
        ...

    async def get(query: str) -> dict | None:
        # L1: Redis 精确匹配 (sha256(normalized_query), TTL 300s)
        # L2: Milvus 语义匹配 (placeholder)

    async def set(query, answer, intent, confidence):
        # 写入 L1 Redis
```

**跳过缓存场景**：workflow 意图、HITL 恢复请求、澄清响应。

### 4.6 澄清逻辑 (`agent/router.py`)

EntryAgent 更新后的路由规则：

```
confidence > 0.8     → target_agent = intent (chat / faq / workflow)
0.5 < confidence <= 0.8 → target_agent = "clarify"，填充 clarification_question
confidence <= 0.5    → target_agent = "chat"
```

Prompt 扩展：要求 LLM 在 JSON 输出中增加 `clarification_question` 字段。

### 4.7 HITL 人机协同 (`hitl.py`)

```python
class HITLManager:
    SENSITIVE_ACTIONS = {"refund", "cancel", "delete"}
    CONFIRM_KEYWORDS = {"确认", "是的", "confirm", "yes", "ok"}
    REJECT_KEYWORDS = {"取消", "拒绝", "reject", "no", "cancel"}

    def is_sensitive(self, intent, params) -> bool:
        return params.get("action", intent) in SENSITIVE_ACTIONS

    def is_confirm(self, message) -> bool:
        return any(kw in message.lower() for kw in CONFIRM_KEYWORDS)

    def is_reject(self, message) -> bool:
        return any(kw in message.lower() for kw in REJECT_KEYWORDS)
```

**流程**：
1. `hitl_request_node` 检测到敏感操作 → 设置 `pending_action` + `action_required="confirm"`
2. 图编译时 `interrupt_before=["hitl_resume"]` → 执行暂停
3. FastAPI 返回 `ChatResponse(action_required="confirm")`
4. 客户端再次发送消息（如"确认"）
5. `hitl_resume_node` 解析确认/拒绝 → 执行或取消 → 清除 `pending_action`

---

## 5. 数据流

```
POST /chat
Authorization: Bearer <jwt>
{ "session_id": "xxx", "message": "我要退款" }

  1. get_current_user 解析 JWT → user_id
  2. main.py 加载历史消息，构建 AgentState
  3. 检查 QueryCache（workflow 意图跳过缓存）
  4. graph.ainvoke(state, config={"thread_id": session_id})
     4a. load_memory_node: TieredMemoryManager.get_context → 聚合记忆
     4b. entry_node: EntryAgent → intent="workflow", confidence=0.91, target_agent="workflow"
     4c. route_node: workflow + is_sensitive → "hitl_request"
     4d. hitl_request_node: pending_action={action:"workflow"}, response="请确认...", action_required="confirm"
     4e. hitl_resume 前 INTERRUPT → 图暂停
  5. main.py 检测到 action_required="confirm" → 返回 ChatResponse

客户端再次 POST /chat
{ "session_id": "xxx", "message": "确认", "action": "confirm" }

  6. graph.ainvoke(None, config={"thread_id": session_id}) → 恢复执行
     6a. hitl_resume_node: 解析 "确认" → is_confirm=True
         response="业务办理已确认执行。", pending_action=None, action_required=None
     6b. save_memory_node: 持久化到分层记忆
  7. main.py 返回最终 ChatResponse
```

---

## 6. 错误处理

| 场景 | 处理策略 |
|------|----------|
| JWT 缺失/过期/无效 | HTTP 401，`{"detail": "认证令牌无效或已过期"}` |
| JWT 未配置 (`jwt_disabled`/`secret` 空) | 降级为 `"anonymous"`，记录 warning |
| Redis 连接失败 | SessionStore 降级内存 dict；QueryCache L1 失效，透传到 LLM |
| PostgreSQL 连接失败 | WarmStore 返回 None/[]；TieredMemoryManager 继续 |
| Milvus 连接失败 | ColdStore 返回 []；QueryCache L2 返回 None |
| 图节点异常 | 节点内部捕获并记录 warning，返回空/默认值，不中断图执行 |
| LLM 调用失败 | 异常向上传播到 FastAPI，返回 HTTP 500 |
| HITL 恢复时消息不匹配 | 返回 `"请回复'确认'或'取消'。"`，保持中断状态 |

---

## 7. 测试策略

| 测试类型 | 覆盖内容 | 工具/方法 |
|----------|----------|-----------|
| 单元测试 | JWT 解码：有效/过期/缺失/禁用 | pytest + mock jose.jwt.decode |
| 单元测试 | HITLManager：敏感检测、确认/拒绝匹配 | 直接实例化断言 |
| 单元测试 | QueryCache：L1 命中/未命中、TTL、降级 | fakeredis |
| 单元测试 | WarmStore：mock asyncpg | unittest.mock |
| 单元测试 | ColdStore：mock MilvusClient + OpenAIEmbeddings | unittest.mock |
| 单元测试 | TieredMemoryManager：聚合/保存/降级 | MagicMock 三层 store |
| 单元测试 | EntryAgent：澄清逻辑（0.5 < conf <= 0.8） | mock ChatOpenAI |
| 流程测试 | LangGraph：chat/faq/clarify/HITL 完整路径 | InMemorySaver + MagicMock agents |
| 流程测试 | HITL 中断+恢复：确认执行/拒绝取消 | graph.ainvoke + interrupt_before |
| API 测试 | `/chat`：认证头、workflow 触发确认、恢复 | TestClient + patch 所有模块级依赖 |

**关键测试模式**：
- `main.py` 模块级初始化 → 测试必须在 import 前 patch 原始模块类，需要 `importlib.reload(cloudagent.main)`
- `conftest.py` 设置 `JWT_DISABLED=true`，使大多数测试无需携带 Authorization 头
- LangGraph 中断测试：`graph.ainvoke(state)` 返回中断状态后，再次调用 `graph.ainvoke(None, config)` 验证恢复

---

## 8. 阶段3明确边界（不做）

- Workflow Agent 真实业务逻辑（退款、取消等后端调用）— 阶段4
- L2 语义缓存的完整 Milvus 实现（当前为 placeholder）— 阶段4
- 会话摘要的 LLM 生成（当前为硬编码字符串）— 阶段4
- Redis 持久化 checkpointer（当前使用 InMemorySaver）— 阶段4
- 限流、熔断、Prometheus 监控 — 阶段4
- MCP 工具生态 — 阶段5
- SSE 流式输出、Vue3 前端 — 阶段6

---

## 9. 成功标准

1. `pytest tests/ -v` 全部通过（含阶段1、阶段2、阶段3测试）。
2. `/chat` 端点无 Authorization 头时：若 `JWT_DISABLED=true` 则正常响应；若 `JWT_DISABLED=false` 则返回 401。
3. FAQ 问题走 RAGAgent，闲聊走 ChatAgent，workflow 走 HITL 确认流程。
4. 置信度 0.5~0.8 时返回澄清问题（`action_required="clarify"`），不直接路由。
5. HITL 敏感操作触发中断，客户端回复"确认"后正确恢复并执行。
6. 任意外部服务（Redis、PG、Milvus、Neo4j）故障时，服务不中断，仅该功能降级。
