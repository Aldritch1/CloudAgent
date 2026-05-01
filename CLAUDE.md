# CloudAgent

CloudAgent is an intelligent customer service system built with FastAPI + LangGraph + LangChain. It uses a multi-Agent architecture with hybrid RAG (Milvus + Neo4j + PostgreSQL) and a tiered memory system (Redis hot / PostgreSQL warm / Milvus cold).

**Current Phase:** Phase 3 complete — Memory + Security + Optimization (JWT auth, tiered memory, L1/L2 cache, clarification logic, HITL via LangGraph interrupt).

---

## Architecture

Four-layer architecture (from design doc):

```
┌─────────────────────────────────────────┐
│  Frontend: Vue3 + SSE (Phase 6)         │
├─────────────────────────────────────────┤
│  API Gateway: FastAPI + Nginx           │
├─────────────────────────────────────────┤
│  Agent Engine: LangGraph + LangChain    │
│  ├─ StateGraph (orchestration + HITL)   │
│  ├─ Entry Agent (intent + routing)      │
│  ├─ RAG Agent (Milvus + Neo4j + PG)     │
│  ├─ Workflow Agent (PG transactions)    │
│  └─ Chat Agent (LLM direct)             │
├─────────────────────────────────────────┤
│  Data Layer                             │
│  ├─ Milvus (vector semantic search)     │
│  ├─ Neo4j (knowledge graph)             │
│  ├─ PostgreSQL (structured business)    │
│  └─ Redis (sessions, cache, locks)      │
└─────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Choice | Notes |
|-----------|--------|-------|
| Web Framework | FastAPI | Pydantic v2 models for request/response validation |
| Agent Framework | LangGraph | StateGraph per agent, intent-based routing |
| LLM | OpenAI (GPT-3.5-turbo / GPT-4) | Configurable via `MODEL_NAME` env var |
| Session Store | Redis | TTL 3600s, in-memory fallback on connection failure |
| Testing | pytest | `monkeypatch` for env isolation, `fakeredis` for Redis tests |
| Config | pydantic-settings | `.env` file support, `SecretStr` for API keys |
| Vector DB | Milvus | Semantic search with OpenAI embeddings (dim=1536, COSINE) |
| Graph DB | Neo4j | Knowledge graph for FAQ relationship queries |
| Structured DB | PostgreSQL | Business data + BM25 keyword search (`pg_trgm`, `tsvector`) |
| Auth | python-jose | JWT Bearer token parsing, `jwt_disabled` dev switch |
| Cache | Redis + Milvus | L1 exact match (Redis, TTL 300s) + L2 semantic (Milvus) |

---

## Directory Structure

```
cloudagent/
├── main.py                  # FastAPI app, module-level dependency init, graph orchestration
├── config.py                # Settings(BaseSettings) singleton
├── models.py                # ChatRequest, ChatResponse
├── state.py                 # AgentState TypedDict for LangGraph
├── graph.py                 # StateGraph builder with nodes + interrupt
├── auth.py                  # JWT dependency: get_current_user
├── cache.py                 # QueryCache: L1/L2 query caching
├── hitl.py                  # HITLManager: sensitive operation confirmation
├── agent/
│   ├── __init__.py
│   ├── router.py            # EntryAgent: intent recognition + routing + clarify
│   ├── chat_agent.py        # ChatAgent: system prompt + LLM invoke
│   └── rag_agent.py         # RAGAgent: retrieval + context-augmented generation
├── memory/
│   ├── __init__.py
│   ├── redis_store.py       # SessionStore: hot store (get_session / save_session)
│   ├── warm_store.py        # WarmStore: PostgreSQL profiles + summaries
│   ├── cold_store.py        # ColdStore: Milvus semantic memory embeddings
│   └── manager.py           # TieredMemoryManager: aggregates hot/warm/cold
└── retrieval/
    ├── __init__.py
    ├── base.py              # RetrievalResult dataclass, Retriever Protocol
    ├── vector.py            # VectorRetriever: Milvus semantic search
    ├── graph.py             # GraphRetriever: Neo4j FAQ search
    ├── keyword.py           # KeywordRetriever: PostgreSQL BM25/tsvector
    └── hybrid.py            # HybridRetriever: RRF fusion of all three

tests/
├── conftest.py              # Autouse fixture: patches env vars before import
├── test_main.py             # API endpoint tests (health, /chat, routing, auth)
├── test_router.py           # EntryAgent routing logic (chat/faq/workflow/clarify)
├── test_chat_agent.py       # ChatAgent system prompt + message conversion
├── test_rag_agent.py        # RAGAgent prompt construction + LLM invoke
├── test_auth.py             # JWT dependency tests
├── test_graph.py            # LangGraph StateGraph flow + interrupt tests
├── test_hitl.py             # HITL state machine tests
├── test_cache.py            # L1/L2 cache tests
├── test_memory_manager.py   # Tiered memory aggregation tests
├── test_warm_store.py       # PostgreSQL warm store tests
├── test_cold_store.py       # Milvus cold store tests
├── test_redis_store.py      # Redis storage + TTL + fallback
├── test_models.py           # Pydantic model validation
├── test_config.py           # Settings env loading
└── retrieval/
    ├── test_base.py
    ├── test_vector.py
    ├── test_graph.py
    ├── test_keyword.py
    └── test_hybrid.py
```

---

## Key Patterns

### Module-Level Dependency Initialization

Dependencies are initialized at module import time in `main.py`:

```python
session_store = SessionStore(str(settings.redis_url))
entry_agent = EntryAgent(model_name=settings.model_name, api_key=...)
chat_agent = ChatAgent(model_name=settings.model_name, api_key=...)
rag_agent = RAGAgent(model_name=settings.model_name, api_key=..., retriever=hybrid_retriever)
memory_manager = TieredMemoryManager(hot_store=None, warm_store=None, cold_store=None)
cache = QueryCache(redis_client=session_store._redis)
hitl = HITLManager()
graph = build_graph(entry_agent, chat_agent, rag_agent, memory_manager, cache, hitl)
```

This means **tests must patch the original modules before importing `main`**:

```python
@patch("cloudagent.retrieval.vector.VectorRetriever")
@patch("cloudagent.retrieval.graph.GraphRetriever")
@patch("cloudagent.retrieval.keyword.KeywordRetriever")
@patch("cloudagent.agent.rag_agent.RAGAgent")
@patch("cloudagent.memory.redis_store.SessionStore")
@patch("cloudagent.agent.router.EntryAgent")
@patch("cloudagent.agent.chat_agent.ChatAgent")
def test_something(mock_chat_cls, mock_entry_cls, mock_store_cls,
                    mock_rag_cls, mock_kw_cls, mock_graph_cls, mock_vec_cls):
    from cloudagent.main import app
    ...
```

If a previous test imported `main` with different mocks, use `importlib.reload(cloudagent.main)` before re-importing.

### Environment Variable Isolation

`tests/conftest.py` provides an `autouse` fixture that sets `OPENAI_API_KEY`, `REDIS_URL`, and `MODEL_NAME` before any test imports `config.py` or `main.py`.

### Error Handling Conventions

- **LLM failures in ChatAgent**: Log error, raise `HTTPException(status_code=500, detail="服务暂时繁忙，请稍后重试")`
- **LLM failures in EntryAgent**: Log error, set `confidence=0.0`, fallback to `chat` agent (user-transparent)
- **LLM failures in RAGAgent**: Log error, raise `HTTPException(status_code=500, detail="服务暂时繁忙，请稍后重试")`
- **Redis connection failures**: Degrade to in-memory `dict` storage, log warning, service continues
- **Retrieval service failures (Milvus/Neo4j/PG)**: Return empty list, log warning, Hybrid RRF continues with remaining sources

### Intent Routing Logic

```
confidence > 0.8   → route to target_agent (chat / faq / workflow)
0.5 < conf <= 0.8  → target_agent = "clarify", return clarification question
confidence <= 0.5  → fallback to chat
```

### LangGraph Interrupt (HITL)

Sensitive workflow operations trigger a graph interrupt before execution:
- `hitl_request_node` sets `action_required = "confirm"`
- Graph compiles with `interrupt_before=["hitl_resume_node"]`
- FastAPI `/chat` returns `ChatResponse(action_required="confirm")`
- Client re-submits; server calls `graph.invoke(None, config)` to resume
- Node failures in chat/rag propagate as exceptions caught by FastAPI (500)

---

## Testing

Run the full suite:

```bash
pytest tests/ -v
```

Key testing patterns:
- Patch original module classes before importing `main` (see above)
- Use `MagicMock` to mock LLM responses; verify call args for prompt assertions
- Use `fakeredis` for Redis tests without a running server
- TTL assertions should tolerate 1-second drift: `assert ttl >= 3599` instead of `== 3600`

---

## Development Roadmap

| Phase | Goal | Key Deliverables |
|-------|------|------------------|
| **1** ✅ | Core API skeleton | FastAPI, Entry Agent, Chat Agent, Redis sessions |
| **2** ✅ | Multi-Agent + Hybrid RAG | RAG Agent, Milvus + Neo4j + PG retrieval, RRF fusion |
| **3** ✅ | Memory + Security + Optimization | JWT auth, tiered memory (Redis/PG/Milvus), L1/L2 cache, HITL |
| **4** | Production hardening | Rate limiting, circuit breaker, Prometheus/Grafana, multi-tenant |
| **5** | MCP tool ecosystem | MCP servers for order/SMS/ticket services |
| **6** | Frontend + SSE | Vue3 UI, SSE streaming, visualization |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```bash
OPENAI_API_KEY=sk-...
REDIS_URL=redis://localhost:6379/0
MODEL_NAME=gpt-3.5-turbo

# Phase 2: Multi-Agent + Hybrid RAG
MILVUS_URI=http://localhost:19530
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
DATABASE_URL=postgresql://cloudagent:cloudagent@localhost:5432/cloudagent

# Phase 3: JWT Authentication
JWT_SECRET=your-jwt-secret-key-at-least-32-characters-long
JWT_ALGORITHM=HS256
JWT_DISABLED=false  # set true in dev/tests to bypass auth
```

---

## Notes for AI Assistants

- Always run `pytest tests/ -v` before declaring a task complete.
- When adding new agents, follow the existing pattern: class with `__init__(model_name, api_key)` and `run(state/messages) -> result`.
- When adding new retrievers, implement the `Retriever` Protocol: `async def search(self, query: str, top_k: int) -> list[RetrievalResult]`.
- When modifying `main.py`, remember module-level initialization — update tests to patch new dependencies before import.
- Keep error messages in Chinese for user-facing responses; English is fine for logs.
- Do not add comments explaining WHAT the code does — use clear identifiers instead. Only comment non-obvious WHYs.
