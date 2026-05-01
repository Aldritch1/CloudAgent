# CloudAgent 阶段2设计文档 — 多Agent + Hybrid RAG

**版本：** v1.0  
**日期：** 2026-05-01  
**目标：** 能区分"问答"和"闲聊"，实现 Hybrid RAG（Milvus + Neo4j + PostgreSQL 三路检索 + RRF 融合）

---

## 1. 总体架构

阶段2聚焦核心检索能力，不实现 Workflow Agent 和 MCP 工具（留到阶段2b）。

```
┌─────────────────────────────────────────┐
│ API网关层：FastAPI                       │
│ 职责：/chat 端点，请求校验，Agent 路由分发 │
├─────────────────────────────────────────┤
│ Agent引擎层：LangGraph + LangChain       │
│ 职责：入口Agent（意图识别+路由）           │
│       Chat Agent（LLM直连，闲聊兜底）      │
│       RAG Agent（检索增强问答）            │
├─────────────────────────────────────────┤
│ 检索层：cloudagent/retrieval/            │
│ 职责：VectorRetriever（Milvus语义检索）   │
│       GraphRetriever（Neo4j图谱检索）     │
│       KeywordRetriever（PG BM25关键词）   │
│       HybridRetriever（并发+RRF融合）     │
├─────────────────────────────────────────┤
│ 数据层：Milvus + Neo4j + PG + Redis      │
│ 职责：向量语义检索、图谱关系、全文检索、会话  │
└─────────────────────────────────────────┘
```

---

## 2. 基础设施

### 2.1 Docker Compose 服务

新增 `docker-compose.yml`（或扩展现有）：

```yaml
services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - etcd-data:/etcd
    command: etcd -advertise-client-urls http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - minio-data:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  milvus-standalone:
    image: milvusdb/milvus:v2.4.1
    ports:
      - "19530:19530"
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    depends_on:
      - etcd
      - minio

  neo4j:
    image: neo4j:5.19-community
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      NEO4J_AUTH: neo4j/password
      NEO4J_PLUGINS: '["apoc"]'  # 如需 APOC

  postgres:
    image: pgvector/pgvector:pg16
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: cloudagent
      POSTGRES_PASSWORD: cloudagent
      POSTGRES_DB: cloudagent

volumes:
  etcd-data:
  minio-data:
```

### 2.2 配置扩展

`cloudagent/config.py` 新增：

```python
class Settings(BaseSettings):
    # ... 现有配置 ...
    milvus_uri: str = "http://localhost:19530"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("password")
    database_url: str = "postgresql+asyncpg://cloudagent:cloudagent@localhost:5432/cloudagent"
```

---

## 3. 文件结构

```
cloudagent/
├── retrieval/               # 新增：检索层抽象
│   ├── __init__.py
│   ├── base.py              # RetrievalResult, Retriever Protocol
│   ├── vector.py            # VectorRetriever (Milvus)
│   ├── graph.py             # GraphRetriever (Neo4j)
│   ├── keyword.py           # KeywordRetriever (PG BM25)
│   └── hybrid.py            # HybridRetriever (并发 + RRF)
├── agent/
│   ├── __init__.py
│   ├── router.py            # 修改：扩展意图（chat / faq / workflow）
│   ├── chat_agent.py        # 不变
│   └── rag_agent.py         # 新增：RAG Agent
├── memory/
│   └── redis_store.py       # 不变
├── main.py                  # 修改：注册 RAG Agent，扩展路由分发
├── models.py                # 不变（ChatRequest / ChatResponse 复用）
└── config.py                # 修改：新增数据库连接配置

tests/
├── conftest.py              # 修改：如有需要新增 mock fixture
├── test_main.py             # 修改：覆盖 faq 路由路径
├── test_router.py           # 修改：覆盖 faq / workflow 意图
├── test_chat_agent.py       # 不变
├── test_rag_agent.py        # 新增
├── test_retrieval/          # 新增：检索层测试包
│   ├── test_vector.py
│   ├── test_graph.py
│   ├── test_keyword.py
│   └── test_hybrid.py
└── fixtures/
    └── seed_faq.json        # 新增：测试用 FAQ 种子数据
```

---

## 4. 检索层设计

### 4.1 统一接口

```python
from typing import Protocol
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    content: str
    source: str           # "vector" | "graph" | "keyword"
    score: float          # 原始分数（调试用）
    metadata: dict        # 来源附加信息

class Retriever(Protocol):
    async def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]: ...
```

### 4.2 VectorRetriever (`retrieval/vector.py`)

- 客户端：`pymilvus` 同步客户端（封装为 async wrapper）。
- Embedding：复用 OpenAI `text-embedding-3-small`（1536维），调用 `openai.Embedding.create`。
- Collection：`kb_documents`
  - 字段：`id` (VARCHAR), `content` (VARCHAR), `embedding` (FLOAT_VECTOR, dim=1536), `category` (VARCHAR)
  - 索引：IVF_FLAT, metric=COSINE
- 首次初始化时自动建 collection 和索引（如果不存在）。
- **降级**：连接失败返回空列表，记录 warning。

### 4.3 GraphRetriever (`retrieval/graph.py`)

- 客户端：`neo4j.AsyncDriver`。
- 查询策略：Phase 2 先用简单 Cypher 模糊匹配，后续再接入 APOC 全文索引。
  ```cypher
  MATCH (f:FAQ)
  WHERE f.question CONTAINS $token OR f.answer CONTAINS $token
  RETURN f.question AS content, f.category AS metadata
  LIMIT $top_k
  ```
- **降级**：连接失败返回空列表。

### 4.4 KeywordRetriever (`retrieval/keyword.py`)

- 客户端：`asyncpg`。
- 依赖 `kb_documents` 表的 `tsvector` + GIN 索引：
  ```sql
  SELECT title, content, category
  FROM kb_documents
  WHERE fts_vector @@ plainto_tsquery('chinese', $1)
  ORDER BY ts_rank(fts_vector, plainto_tsquery('chinese', $1)) DESC
  LIMIT $2
  ```
- **降级**：连接失败返回空列表。

### 4.5 HybridRetriever (`retrieval/hybrid.py`)

```python
class HybridRetriever:
    def __init__(self, vector: VectorRetriever, graph: GraphRetriever, keyword: KeywordRetriever):
        self.vector = vector
        self.graph = graph
        self.keyword = keyword

    async def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        v_task = self.vector.search(query, top_k=10)
        g_task = self.graph.search(query, top_k=10)
        k_task = self.keyword.search(query, top_k=10)
        v_results, g_results, k_results = await asyncio.gather(v_task, g_task, k_task)
        return rrf_fuse([v_results, g_results, k_results], k=60, final_top_k=top_k)
```

**RRF 融合公式：**
```python
from collections import defaultdict

def rrf_fuse(result_lists: list[list[RetrievalResult]], k: int = 60, final_top_k: int = 5) -> list[RetrievalResult]:
    scores = defaultdict(float)
    items = {}
    for results in result_lists:
        for rank, r in enumerate(results, start=1):
            scores[r.content] += 1.0 / (k + rank)
            if r.content not in items:
                items[r.content] = r
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [items[content] for content, _ in sorted_items[:final_top_k]]
```

**关键原则：**
- 三路检索并发执行（`asyncio.gather`），非串行。
- 任意一路失败只影响该路召回，不影响融合和其他路。
- RRF 只看 rank 不看原始 score，避免不同检索方式分数尺度不一致。

---

## 5. Agent 层设计

### 5.1 RAG Agent (`agent/rag_agent.py`)

```python
class RAGAgent:
    def __init__(self, model_name: str, api_key: str, retriever: HybridRetriever):
        self._llm = ChatOpenAI(model=model_name, api_key=api_key, temperature=0.3)
        self._retriever = retriever

    async def run(self, state: dict) -> str:
        last_user_msg = _extract_last_user(state["messages"])
        contexts = await self._retriever.search(last_user_msg, top_k=5)
        context_text = "\n\n".join([c.content for c in contexts])

        system_prompt = f"""你是 CloudAgent 智能客服助手。请根据以下参考资料回答用户问题。
如果参考资料不足以回答，请坦诚告知用户。
参考资料：
{context_text}
"""
        lc_messages = [SystemMessage(content=system_prompt)]
        lc_messages.extend(_convert_dict_messages(state["messages"]))
        response = await self._llm.ainvoke(lc_messages)
        return response.content
```

### 5.2 Entry Agent 意图扩展 (`agent/router.py`)

Phase 2 新增两种意图：
- `faq`：知识问答（退款政策、运费规则等）→ 路由到 RAG Agent
- `workflow`：业务办理（查订单、申请退款等）→ Phase 2 暂不做，识别后兜底提示"该功能即将上线"
- `chat`：闲聊/寒暄（Phase 1 已有）→ 路由到 Chat Agent

**路由逻辑（不变）：**
```
confidence > 0.8   → route to target_agent
0.5 < conf <= 0.8  → route directly（追问澄清在 Phase 3 实现）
confidence <= 0.5  → fallback to chat
```

### 5.3 `main.py` 路由分发

延续 Phase 1 的模块级初始化模式，所有新依赖在 import 时创建：

```python
# 检索层初始化（与 SessionStore 模式一致）
vector_retriever = VectorRetriever(uri=settings.milvus_uri, api_key=...)
graph_retriever = GraphRetriever(uri=settings.neo4j_uri, user=..., password=...)
keyword_retriever = KeywordRetriever(dsn=settings.database_url)
hybrid_retriever = HybridRetriever(vector_retriever, graph_retriever, keyword_retriever)

rag_agent = RAGAgent(
    model_name=settings.model_name,
    api_key=settings.openai_api_key.get_secret_value(),
    retriever=hybrid_retriever,
)

# /chat 端点内部
target = state["target_agent"]

if target == "faq":
    response_text = await rag_agent.run(state)
elif target == "workflow":
    response_text = "业务办理功能正在开发中，请稍后再试。"
else:
    response_text = await chat_agent.run(state["messages"])
```

**测试注意：** 由于模块级初始化，测试仍需在 import `main` 前 patch 原始模块（`cloudagent.retrieval.vector.VectorRetriever` 等），与 Phase 1 的 `SessionStore` / `ChatAgent` 补丁策略一致。

---

## 6. 数据流

```
POST /chat
{ "session_id": "xxx", "message": "怎么退款？" }

  1. main.py 解析请求，Pydantic 校验
  2. redis_store.get_session → 加载历史消息
  3. 追加用户消息到历史
  4. EntryAgent 识别意图
     → intent="faq", confidence=0.94, target_agent="faq"
  5. main.py 按 target_agent 路由到 RAGAgent
  6. RAGAgent:
     6a. 提取最后一条用户消息 "怎么退款？"
     6b. HybridRetriever 并发查 Milvus + Neo4j + PG
     6c. RRF 融合，取 top5 知识片段
     6d. 拼接 system prompt（含参考资料）+ 历史消息
     6e. LLM 生成回复
  7. 追加 assistant 回复到历史
  8. redis_store.save_session → 更新 Redis，刷新 TTL
  9. 返回 ChatResponse { response, intent, confidence }
```

---

## 7. 错误处理

| 场景 | 处理策略 |
|------|----------|
| Milvus 连接失败 | VectorRetriever 返回空列表，记录 warning |
| Neo4j 连接失败 | GraphRetriever 返回空列表，记录 warning |
| PostgreSQL 连接失败 | KeywordRetriever 返回空列表，记录 warning |
| 三路全部失败 | HybridRetriever 返回空列表，RAG Agent system prompt 中 context_text=""，LLM 按无参考知识回答 |
| RAG Agent LLM 调用失败 | 返回 HTTP 500，`{"detail": "服务暂时繁忙，请稍后重试"}` |
| 意图识别 LLM 失败 | confidence=0.0，fallback 到 chat，用户无感知 |

---

## 8. 测试策略

| 测试类型 | 覆盖内容 | 工具/方法 |
|----------|----------|-----------|
| 单元测试 | VectorRetriever：mock pymilvus，验证返回格式 | unittest.mock |
| 单元测试 | GraphRetriever：mock neo4j.AsyncDriver | unittest.mock |
| 单元测试 | KeywordRetriever：mock asyncpg | unittest.mock |
| 单元测试 | HybridRetriever：mock 三路输入，验证 RRF 排名正确 | pytest |
| 单元测试 | RAG Agent：mock retriever + mock LLM，验证 prompt 包含知识片段 | pytest |
| 单元测试 | Entry Agent：mock LLM 返回 faq/workflow/chat，验证路由目标 | pytest |
| API 测试 | `/chat` faq 路径：验证响应包含检索上下文 | TestClient |
| 集成测试 | Docker Compose 全链路：写入 seed FAQ → hybrid_search 返回非空 | pytest + docker compose |

**Seed Data：**
`tests/fixtures/seed_faq.json` 提供 mock FAQ 数据，用于单元测试断言和集成测试数据初始化。

---

## 9. 阶段2明确边界（不做）

- Workflow Agent（阶段2b）
- MCP 工具调用（阶段5）
- 前端文件上传与可视化（阶段6）
- JWT 认证与用户权限（阶段3）
- L1/L2 语义缓存（阶段3）
- 用户画像/长期记忆（阶段4）
- SSE 流式输出（阶段6）

---

## 10. 成功标准

1. `docker compose up -d` 后，服务能正常启动并连接三个新数据库。
2. 用户问 FAQ 类问题（如"怎么退款"），EntryAgent 能识别为 `faq` 意图，RAGAgent 返回基于知识库的回答。
3. 用户闲聊时，仍走 Chat Agent，体验与 Phase 1 一致。
4. 任意一个数据库故障时，服务不中断，只是该路检索失效。
5. 所有单元测试和 API 测试通过；集成测试（如有）通过。
