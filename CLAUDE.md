# CloudAgent

CloudAgent is an intelligent customer service system built with FastAPI + LangGraph + LangChain. It uses a multi-Agent architecture with hybrid RAG (Milvus + Neo4j + PostgreSQL) and a tiered memory system (Redis hot / PostgreSQL warm / Milvus cold).

**Current Phase:** Phase 1 complete — Core API skeleton (FastAPI + Entry Agent + Chat Agent + Redis session storage).

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

---

## Directory Structure

```
cloudagent/
├── main.py                  # FastAPI app, module-level dependency init
├── config.py                # Settings(BaseSettings) singleton
├── models.py                # ChatRequest, ChatResponse
├── agent/
│   ├── __init__.py
│   ├── router.py            # EntryAgent: intent recognition + routing
│   └── chat_agent.py        # ChatAgent: system prompt + LLM invoke
└── memory/
    ├── __init__.py
    └── redis_store.py       # SessionStore: get_session / save_session

tests/
├── conftest.py              # Autouse fixture: patches env vars before import
├── test_main.py             # API endpoint tests (health, /chat)
├── test_router.py           # EntryAgent routing logic
├── test_chat_agent.py       # ChatAgent system prompt + message conversion
├── test_redis_store.py      # Redis storage + TTL + fallback
├── test_models.py           # Pydantic model validation
└── test_config.py           # Settings env loading
```

---

## Key Patterns

### Module-Level Dependency Initialization

Dependencies are initialized at module import time in `main.py`:

```python
session_store = SessionStore(str(settings.redis_url))
entry_agent = EntryAgent(model_name=settings.model_name, api_key=...)
chat_agent = ChatAgent(model_name=settings.model_name, api_key=...)
```

This means **tests must patch the original modules before importing `main`**:

```python
@patch("cloudagent.memory.redis_store.SessionStore")
@patch("cloudagent.agent.router.EntryAgent")
@patch("cloudagent.agent.chat_agent.ChatAgent")
def test_something(mock_chat_cls, mock_entry_cls, mock_store_cls):
    from cloudagent.main import app
    ...
```

If a previous test imported `main` with different mocks, use `importlib.reload(cloudagent.main)` before re-importing.

### Environment Variable Isolation

`tests/conftest.py` provides an `autouse` fixture that sets `OPENAI_API_KEY`, `REDIS_URL`, and `MODEL_NAME` before any test imports `config.py` or `main.py`.

### Error Handling Conventions

- **LLM failures in ChatAgent**: Log error, raise `HTTPException(status_code=500, detail="服务暂时繁忙，请稍后重试")`
- **LLM failures in EntryAgent**: Log error, set `confidence=0.0`, fallback to `chat` agent (user-transparent)
- **Redis connection failures**: Degrade to in-memory `dict` storage, log warning, service continues

### Intent Routing Logic

```
confidence > 0.8   → route to target_agent
0.5 < conf <= 0.8  → route directly (clarification in Phase 3)
confidence <= 0.5  → fallback to chat
```

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
| **2** | Multi-Agent + Hybrid RAG | RAG Agent, Milvus + Neo4j + PG retrieval, RRF fusion |
| **3** | Memory + Security + Optimization | JWT auth, tiered memory (Redis/PG/Milvus), L1/L2 cache, HITL |
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
```

---

## Notes for AI Assistants

- Always run `pytest tests/ -v` before declaring a task complete.
- When adding new agents, follow the existing pattern: class with `__init__(model_name, api_key)` and `run(state/messages) -> result`.
- When modifying `main.py`, remember module-level initialization — update tests to patch new dependencies before import.
- Keep error messages in Chinese for user-facing responses; English is fine for logs.
- Do not add comments explaining WHAT the code does — use clear identifiers instead. Only comment non-obvious WHYs.
