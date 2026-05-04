# CloudAgent

CloudAgent 是一个基于 FastAPI + LangGraph + LangChain 构建的智能客服系统，采用多智能体（Multi-Agent）架构与混合 RAG（Hybrid Retrieval-Augmented Generation）技术，支持语义检索、知识图谱查询和关键词搜索的融合，实现精准的用户意图识别与答复生成。

---

## 功能特性

- **多智能体路由**：EntryAgent 自动识别用户意图（聊天 / FAQ / 业务办理），支持置信度 0.5~0.8 的追问澄清
- **混合 RAG 检索**：同时查询 Milvus（向量语义搜索）、Neo4j（知识图谱）、PostgreSQL（BM25 关键词搜索），通过 RRF 融合排序
- **分层记忆**：Redis 热存储（会话上下文）、PostgreSQL 温存储（用户画像 + 摘要）、Milvus 冷存储（跨会话语义记忆）
- **L1/L2 缓存**：Redis 精确匹配缓存（<50ms）+ Milvus 语义缓存，减少重复 LLM 调用
- **JWT 认证**：API Gateway 层解析 Bearer Token，支持开发环境一键关闭
- **HITL 人机协同**：敏感操作通过 LangGraph `interrupt` 暂停，等待用户确认后继续执行
- **限流熔断**：Redis 滑动窗口限流（60 RPM）+ LLM 层熔断器（5 次失败/60s 恢复），保障服务稳定性
- **可观测性**：Prometheus HTTP 请求延迟、LLM 调用、缓存命中率等指标，原生 `/metrics` 端点
- **多租户隔离**：基于 `contextvars` 的租户上下文，`X-Tenant-ID` 或 JWT `tenant_id` 声明，Redis/PG/Milvus 全链路隔离
- **MCP 工具生态**：内置 Order / SMS / Ticket MCP 服务，Workflow Agent 通过 tool-calling 执行业务操作
- **前端 + SSE 流式输出**：Vue3 聊天界面，实时 token-by-token 流式展示 LLM 输出，工具调用可视化卡片
- **优雅降级**：任何检索服务、LLM 或认证故障时，系统自动降级，保证服务可用性

---

## 架构

```
┌─────────────────────────────────────────┐
│  Frontend: Vue3 + SSE (Phase 6)         │
├─────────────────────────────────────────┤
│  API Gateway: FastAPI + Nginx           │
├─────────────────────────────────────────┤
│  Agent Engine: LangGraph + LangChain    │
│  ├─ StateGraph    (编排 + HITL 中断)      │
│  ├─ Entry Agent   (意图识别 + 路由 + 澄清) │
│  ├─ RAG Agent     (混合检索 + 生成)       │
│  ├─ Workflow Agent (MCP tool calling)   │
│  └─ Chat Agent    (LLM 直接对话)          │
├─────────────────────────────────────────┤
│  Data Layer                             │
│  ├─ Milvus      (向量语义搜索 + 冷记忆)    │
│  ├─ Neo4j       (知识图谱)                │
│  ├─ PostgreSQL  (结构化数据 + 温记忆)      │
│  └─ Redis       (会话热存储 + L1 缓存)     │
├─────────────────────────────────────────┤
│  MCP Servers (内置)                      │
│  ├─ Order Server   订单查询/取消/退款      │
│  ├─ SMS Server     短信发送               │
│  └─ Ticket Server  工单创建/查询           │
└─────────────────────────────────────────┘
```

---

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| Web 框架 | FastAPI | Pydantic v2 请求/响应校验 |
| 智能体框架 | LangGraph | 基于 StateGraph 的意图路由 |
| LLM | OpenAI GPT | 通过 `MODEL_NAME` 环境变量配置 |
| 向量数据库 | Milvus | 语义搜索，1536 维，COSINE 度量 |
| 图数据库 | Neo4j | FAQ 知识图谱关系查询 |
| 关系数据库 | PostgreSQL | 业务数据 + `tsvector`/`pg_trgm` 全文检索 |
| 会话存储 | Redis | TTL 3600s，连接失败降级内存存储 |
| 认证 | python-jose | JWT Bearer Token 解析，开发环境可禁用 |
| 缓存 | Redis + Milvus | L1 精确匹配（Redis TTL 300s）+ L2 语义相似（Milvus） |
| 限流 | 自定义 Redis 滑动窗口 | 每用户 `ratelimit:<user_id>`，默认 60 RPM |
| 熔断 | pybreaker | LLM 调用层熔断，5 次失败开启，60s 半开恢复 |
| 可观测性 | prometheus-client | HTTP 延迟直方图、LLM/缓存/检索计数器，`/metrics` 端点 |
| 多租户 | contextvars | 应用层隔离：Redis key 前缀、PG/Milvus `tenant_id` 过滤 |
| MCP | mcp (Anthropic SDK) | 内置 Order/SMS/Ticket 服务，stdio 传输 |
| 前端 | Vue3 + Vite + Element Plus | 聊天界面、SSE 流式输出、工具调用卡片 |
| 配置管理 | pydantic-settings | `.env` 文件支持，`SecretStr` 保护密钥 |
| 测试 | pytest | `pytest-asyncio` + `fakeredis` + `MagicMock` |

---

## 快速开始

### 1. 克隆仓库

```bash
git clone git@github.com:Aldritch1/CloudAgent.git
cd CloudAgent
```

### 2. 安装依赖

```bash
pip install -e ".[dev]"
```

### 3. 启动基础设施

```bash
docker-compose up -d
```

这将启动：
- **Milvus** (`localhost:19530`) — 向量检索
- **Neo4j** (`localhost:7474` / `7687`) — 知识图谱
- **PostgreSQL** (`localhost:5432`) — 关键词检索与业务数据

> Redis 可本地安装，或默认降级为内存存储（仅开发环境）。

### 4. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 OpenAI API Key：

```bash
OPENAI_API_KEY=sk-...
REDIS_URL=redis://localhost:6379/0
MODEL_NAME=gpt-3.5-turbo

# Phase 2: 混合 RAG
MILVUS_URI=http://localhost:19530
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
DATABASE_URL=postgresql://cloudagent:cloudagent@localhost:5432/cloudagent

# Phase 3: JWT 认证（开发环境可设为 true 跳过认证）
JWT_SECRET=your-jwt-secret-key-at-least-32-characters-long
JWT_ALGORITHM=HS256
JWT_DISABLED=false

# Phase 4: 生产加固
RATE_LIMIT_REQUESTS_PER_MINUTE=60
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60
ENABLE_METRICS=true
DEFAULT_TENANT_ID=default

# Phase 5: MCP 工具生态
MCP_SERVERS=order,sms,ticket
ORDER_SERVICE_URL=
SMS_SERVICE_URL=
TICKET_SERVICE_URL=

# Phase 6: 前端 + SSE
ENABLE_SSE=true
CORS_ORIGINS=http://localhost:5173
```

### 5. 运行服务

```bash
uvicorn cloudagent.main:app --reload
```

服务默认运行在 `http://localhost:8000`。

---

## API 接口

### 健康检查

```bash
curl http://localhost:8000/health
```

响应：
```json
{"status": "ok", "version": "0.1.0"}
```

### 对话接口

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "message": "怎么退款？"
  }'
```

响应：
```json
{
  "response": "支持7天无理由退款。",
  "intent": "faq",
  "confidence": 0.94
}
```

### SSE 流式对话接口

```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "message": "帮我查一下订单"
  }'
```

SSE 事件类型：
| Event | 说明 |
|-------|------|
| `intent` | 意图识别结果（intent, confidence, target_agent） |
| `token` | LLM 生成的单个 token（流式输出） |
| `tool_call` | Tool 调用信息（tool_name, arguments） |
| `tool_result` | Tool 执行结果 |
| `hitl` | HITL 中断，需要用户确认 |
| `done` | 流结束，包含完整 response |
| `error` | 错误信息 |

### 意图路由规则

| 置信度 | 处理方式 |
|--------|----------|
| `> 0.8` | 直接路由到目标智能体（FAQ / 业务办理 / 聊天） |
| `0.5 ~ 0.8` | 返回澄清问题，追问用户具体意图 |
| `<= 0.5` | 降级为通用聊天智能体 |

---

## 测试

```bash
pytest tests/ -v
```

当前测试覆盖：
- API 端点（健康检查、对话、异常处理、意图路由、JWT 认证、429 限流、503 熔断）
- EntryAgent 意图识别（chat / faq / workflow / clarify）
- ChatAgent 系统提示词与消息转换
- RAGAgent 检索上下文注入与 LLM 调用
- LangGraph StateGraph 流程（chat / faq / clarify / HITL interrupt）
- JWT 认证（有效/过期/缺失/禁用）
- Redis 会话存储、TTL、连接降级、多租户 key 前缀隔离
- 分层记忆：TieredMemoryManager、WarmStore、ColdStore（含租户隔离）
- L1/L2 查询缓存
- HITL 状态机
- 限流：滑动窗口、独立用户配额、Redis 降级
- 熔断：闭合/开启/半开状态、同步/异步调用
- Prometheus 指标：HTTP 中间件、LLM 调用、缓存命中
- 多租户：Header/JWT 声明提取、Redis/PG/Milvus 隔离
- MCP 工具生态：MCPClient、Order/SMS/Ticket Server、WorkflowAgent tool-calling
- SSE 流式输出：`/chat/stream` 端点、EventSourceResponse、意图/token/tool_call/done 事件
- 前端：Vue3 聊天界面、Pinia 状态管理、SSE 手动解析
- 检索层：VectorRetriever、GraphRetriever、KeywordRetriever、HybridRetriever（RRF 融合）

---

## 项目结构

```
cloudagent/
├── main.py                  # FastAPI 应用，LangGraph 编排
├── config.py                # Settings(BaseSettings) 单例
├── models.py                # ChatRequest, ChatResponse
├── state.py                 # AgentState TypedDict
├── graph.py                 # StateGraph 构建器（含 interrupt）
├── auth.py                  # JWT 认证依赖 + 租户上下文
├── cache.py                 # L1/L2 查询缓存
├── hitl.py                  # HITL 敏感操作确认
├── rate_limit.py            # 滑动窗口限流器
├── circuit_breaker.py       # LLM 层熔断器
├── metrics.py               # Prometheus 指标与中间件
├── tenant_context.py        # 租户上下文 ContextVar
├── tenant.py                # TenantDependency 依赖注入
├── api/
│   ├── __init__.py
│   └── sse.py               # SSE 流式输出端点
├── agent/
│   ├── router.py            # EntryAgent: 意图识别 + 路由 + 澄清
│   ├── chat_agent.py        # ChatAgent: 系统提示词 + LLM 调用
│   ├── rag_agent.py         # RAGAgent: 检索增强生成
│   └── workflow_agent.py    # WorkflowAgent: MCP tool-calling 业务办理
├── mcp/
│   ├── client.py            # MCPClient: 工具发现与调用
│   └── servers/
│       ├── base.py          # BaseMCPServer
│       ├── order.py         # OrderMCPServer
│       ├── sms.py           # SMSMCPServer
│       └── ticket.py        # TicketMCPServer
├── memory/
│   ├── redis_store.py       # SessionStore: 热存储（租户前缀隔离）
│   ├── warm_store.py        # WarmStore: PostgreSQL 用户画像（租户隔离）
│   ├── cold_store.py        # ColdStore: Milvus 语义记忆（租户隔离）
│   └── manager.py           # TieredMemoryManager: 分层聚合
└── retrieval/
    ├── base.py              # RetrievalResult / Retriever Protocol
    ├── vector.py            # VectorRetriever: Milvus 语义搜索
    ├── graph.py             # GraphRetriever: Neo4j 图谱搜索
    ├── keyword.py           # KeywordRetriever: PostgreSQL 全文检索
    └── hybrid.py            # HybridRetriever: RRF 多路融合

frontend/                    # Vue3 + Vite + Element Plus 前端
├── index.html
├── package.json
├── vite.config.ts
├── src/
│   ├── main.ts
│   ├── App.vue
│   ├── router/
│   ├── views/ChatView.vue
│   ├── components/
│   ├── api/chat.ts
│   ├── stores/chat.ts
│   └── types/chat.ts
└── public/

tests/
├── conftest.py              # 全局 autouse fixture（环境变量隔离）
├── test_main.py             # API 端点集成测试
├── test_router.py           # EntryAgent 路由逻辑（含澄清）
├── test_chat_agent.py       # ChatAgent 单元测试
├── test_rag_agent.py        # RAGAgent 单元测试
├── test_auth.py             # JWT 认证测试
├── test_graph.py            # LangGraph 流程 + 中断测试
├── test_hitl.py             # HITL 状态机测试
├── test_cache.py            # 缓存测试
├── test_memory_manager.py   # 分层记忆管理器测试
├── test_warm_store.py       # PostgreSQL 温存储测试
├── test_cold_store.py       # Milvus 冷存储测试
├── test_redis_store.py      # Redis 存储测试
├── test_models.py           # Pydantic 模型校验
├── test_config.py           # 配置加载测试
├── test_mcp_client.py       # MCPClient 工具发现与调用测试
├── test_mcp_servers.py      # Order/SMS/Ticket MCP 服务测试
├── test_workflow_agent.py   # WorkflowAgent tool-calling 测试
└── retrieval/               # 检索层单元测试
```

---

## 开发路线图

| 阶段 | 目标 | 关键交付 |
|------|------|----------|
| **1** ✅ | 核心 API 骨架 | FastAPI、EntryAgent、ChatAgent、Redis 会话 |
| **2** ✅ | 多智能体 + 混合 RAG | RAGAgent、Milvus + Neo4j + PG 检索、RRF 融合 |
| **3** ✅ | 记忆 + 安全 + 优化 | JWT 认证、分层记忆（Redis/PG/Milvus）、L1/L2 缓存、HITL |
| **4** ✅ | 生产加固 | 限流、熔断、Prometheus/Grafana、多租户 |
| **5** ✅ | MCP 工具生态 | 订单 / 短信 / 工单等 MCP 服务 |
| **6** ✅ | 前端 + SSE | Vue3 UI、SSE 流式输出、可视化 |

---

## 许可证

[MIT](LICENSE)
