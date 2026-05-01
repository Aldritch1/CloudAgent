# CloudAgent

CloudAgent 是一个基于 FastAPI + LangGraph + LangChain 构建的智能客服系统，采用多智能体（Multi-Agent）架构与混合 RAG（Hybrid Retrieval-Augmented Generation）技术，支持语义检索、知识图谱查询和关键词搜索的融合，实现精准的用户意图识别与答复生成。

---

## 功能特性

- **多智能体路由**：EntryAgent 自动识别用户意图（聊天 / FAQ / 业务办理），并路由到对应智能体
- **混合 RAG 检索**：同时查询 Milvus（向量语义搜索）、Neo4j（知识图谱）、PostgreSQL（BM25 关键词搜索），通过 RRF 融合排序
- **会话管理**：基于 Redis 的会话存储，支持 TTL 自动过期，连接失败时降级为内存存储
- **优雅降级**：任何检索服务或 LLM 故障时，系统自动降级，保证服务可用性
- **流式响应**：FastAPI 提供高性能异步 API，为后续 SSE 流式输出预留扩展点

---

## 架构

```
┌─────────────────────────────────────────┐
│  Frontend: Vue3 + SSE (Phase 6)         │
├─────────────────────────────────────────┤
│  API Gateway: FastAPI + Nginx           │
├─────────────────────────────────────────┤
│  Agent Engine: LangGraph + LangChain    │
│  ├─ Entry Agent   (意图识别 + 路由)      │
│  ├─ RAG Agent     (混合检索 + 生成)      │
│  ├─ Workflow Agent (业务办理, 开发中)    │
│  └─ Chat Agent    (LLM 直接对话)         │
├─────────────────────────────────────────┤
│  Data Layer                             │
│  ├─ Milvus      (向量语义搜索)            │
│  ├─ Neo4j       (知识图谱)               │
│  ├─ PostgreSQL  (结构化数据 + 全文检索)    │
│  └─ Redis       (会话、缓存、锁)          │
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
| `0.5 ~ 0.8` | 直接路由（Phase 3 增加澄清确认） |
| `<= 0.5` | 降级为通用聊天智能体 |

---

## 测试

```bash
pytest tests/ -v
```

当前测试覆盖：
- API 端点（健康检查、对话、异常处理、意图路由）
- EntryAgent 意图识别（chat / faq / workflow）
- ChatAgent 系统提示词与消息转换
- RAGAgent 检索上下文注入与 LLM 调用
- Redis 会话存储、TTL、连接降级
- 检索层：VectorRetriever、GraphRetriever、KeywordRetriever、HybridRetriever（RRF 融合）

---

## 项目结构

```
cloudagent/
├── main.py                  # FastAPI 应用，模块级依赖初始化
├── config.py                # Settings(BaseSettings) 单例
├── models.py                # ChatRequest, ChatResponse
├── agent/
│   ├── router.py            # EntryAgent: 意图识别 + 路由
│   ├── chat_agent.py        # ChatAgent: 系统提示词 + LLM 调用
│   └── rag_agent.py         # RAGAgent: 检索增强生成
├── memory/
│   └── redis_store.py       # SessionStore: 会话读写
└── retrieval/
    ├── base.py              # RetrievalResult / Retriever Protocol
    ├── vector.py            # VectorRetriever: Milvus 语义搜索
    ├── graph.py             # GraphRetriever: Neo4j 图谱搜索
    ├── keyword.py           # KeywordRetriever: PostgreSQL 全文检索
    └── hybrid.py            # HybridRetriever: RRF 多路融合

tests/
├── conftest.py              # 全局 autouse fixture（环境变量隔离）
├── test_main.py             # API 端点集成测试
├── test_router.py           # EntryAgent 路由逻辑
├── test_chat_agent.py       # ChatAgent 单元测试
├── test_rag_agent.py        # RAGAgent 单元测试
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
| **3** | 记忆 + 安全 + 优化 | JWT 认证、分层记忆（Redis/PG/Milvus）、L1/L2 缓存、HITL |
| **4** | 生产加固 | 限流、熔断、Prometheus/Grafana、多租户 |
| **5** | MCP 工具生态 | 订单 / 短信 / 工单等 MCP 服务 |
| **6** | 前端 + SSE | Vue3 UI、SSE 流式输出、可视化 |

---

## 许可证

[MIT](LICENSE)
