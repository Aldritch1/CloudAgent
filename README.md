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
│  ├─ Workflow Agent (业务办理, 开发中)     │
│  └─ Chat Agent    (LLM 直接对话)          │
├─────────────────────────────────────────┤
│  Data Layer                             │
│  ├─ Milvus      (向量语义搜索 + 冷记忆)    │
│  ├─ Neo4j       (知识图谱)                │
│  ├─ PostgreSQL  (结构化数据 + 温记忆)      │
│  └─ Redis       (会话热存储 + L1 缓存)     │
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
- API 端点（健康检查、对话、异常处理、意图路由、JWT 认证）
- EntryAgent 意图识别（chat / faq / workflow / clarify）
- ChatAgent 系统提示词与消息转换
- RAGAgent 检索上下文注入与 LLM 调用
- LangGraph StateGraph 流程（chat / faq / clarify / HITL interrupt）
- JWT 认证（有效/过期/缺失/禁用）
- Redis 会话存储、TTL、连接降级
- 分层记忆：TieredMemoryManager、WarmStore、ColdStore
- L1/L2 查询缓存
- HITL 状态机
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
├── auth.py                  # JWT 认证依赖
├── cache.py                 # L1/L2 查询缓存
├── hitl.py                  # HITL 敏感操作确认
├── agent/
│   ├── router.py            # EntryAgent: 意图识别 + 路由 + 澄清
│   ├── chat_agent.py        # ChatAgent: 系统提示词 + LLM 调用
│   └── rag_agent.py         # RAGAgent: 检索增强生成
├── memory/
│   ├── redis_store.py       # SessionStore: 热存储
│   ├── warm_store.py        # WarmStore: PostgreSQL 用户画像
│   ├── cold_store.py        # ColdStore: Milvus 语义记忆
│   └── manager.py           # TieredMemoryManager: 分层聚合
└── retrieval/
    ├── base.py              # RetrievalResult / Retriever Protocol
    ├── vector.py            # VectorRetriever: Milvus 语义搜索
    ├── graph.py             # GraphRetriever: Neo4j 图谱搜索
    ├── keyword.py           # KeywordRetriever: PostgreSQL 全文检索
    └── hybrid.py            # HybridRetriever: RRF 多路融合

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
└── retrieval/               # 检索层单元测试
```

---

## 开发路线图

| 阶段 | 目标 | 关键交付 |
|------|------|----------|
| **1** ✅ | 核心 API 骨架 | FastAPI、EntryAgent、ChatAgent、Redis 会话 |
| **2** ✅ | 多智能体 + 混合 RAG | RAGAgent、Milvus + Neo4j + PG 检索、RRF 融合 |
| **3** ✅ | 记忆 + 安全 + 优化 | JWT 认证、分层记忆（Redis/PG/Milvus）、L1/L2 缓存、HITL |
| **4** | 生产加固 | 限流、熔断、Prometheus/Grafana、多租户 |
| **5** | MCP 工具生态 | 订单 / 短信 / 工单等 MCP 服务 |
| **6** | 前端 + SSE | Vue3 UI、SSE 流式输出、可视化 |

---

## 许可证

[MIT](LICENSE)
