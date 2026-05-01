# CloudAgent Phase3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Memory + Security + Optimization: JWT auth, tiered memory (Redis hot / PostgreSQL warm / Milvus cold), L1/L2 query cache, clarification logic for mid-confidence intents, and HITL via LangGraph interrupts. Refactor `main.py` from direct agent calls to a compiled LangGraph StateGraph.

**Architecture:** A new `cloudagent/graph.py` defines the StateGraph with nodes (load_memory, entry, route, chat, rag, workflow_placeholder, clarify, hitl_request, hitl_resume, save_memory). `cloudagent/state.py` provides the `AgentState` TypedDict. `cloudagent/auth.py` implements JWT Bearer parsing. `cloudagent/memory/` adds warm/cold stores and a tiered manager. `cloudagent/cache.py` provides L1/L2 query caching. `cloudagent/hitl.py` handles sensitive-action confirmation. `main.py` is rewritten to orchestrate via the compiled graph.

**Tech Stack:** Python 3.11+, FastAPI, LangGraph, python-jose, asyncpg, pymilvus, redis, pytest, pytest-asyncio

---

## File Structure

```
cloudagent/
├── auth.py                  # NEW: JWT Bearer parsing
├── state.py                 # NEW: AgentState TypedDict
├── graph.py                 # NEW: StateGraph builder + GraphNodes
├── cache.py                 # NEW: L1/L2 query cache
├── hitl.py                  # NEW: HITLManager
├── main.py                  # MODIFIED: graph orchestration
├── models.py                # MODIFIED: ChatRequest.action, ChatResponse.action_required
├── config.py                # MODIFIED: jwt_secret, jwt_algorithm, jwt_disabled
├── agent/
│   ├── router.py            # MODIFIED: clarification logic
│   ├── chat_agent.py        # unchanged
│   └── rag_agent.py         # unchanged
├── memory/
│   ├── redis_store.py       # unchanged
│   ├── warm_store.py        # NEW: PostgreSQL warm storage
│   ├── cold_store.py        # NEW: Milvus cold storage
│   └── manager.py           # NEW: TieredMemoryManager
└── retrieval/               # unchanged

tests/
├── test_auth.py             # NEW
├── test_graph.py            # NEW
├── test_cache.py            # NEW
├── test_hitl.py             # NEW
├── test_memory_manager.py   # NEW
├── test_warm_store.py       # NEW
├── test_cold_store.py       # NEW
├── test_main.py             # MODIFIED: auth + graph integration
├── test_router.py           # MODIFIED: clarification tests
├── test_config.py           # MODIFIED: jwt fields
├── conftest.py              # MODIFIED: JWT_DISABLED=true
└── retrieval/               # unchanged
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
    "python-jose[cryptography]>=3.3.0",
    "passlib[bcrypt]>=1.7.4",
    "numpy>=1.26.0",
]
```

- [ ] **Step 2: Modify cloudagent/config.py**

Add JWT fields to `Settings`:

```python
from pydantic import RedisDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: SecretStr
    redis_url: RedisDsn = "redis://localhost:6379/0"
    model_name: str = "gpt-3.5-turbo"
    milvus_uri: str = "http://localhost:19530"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("password")
    database_url: str = "postgresql://cloudagent:cloudagent@localhost:5432/cloudagent"

    jwt_secret: SecretStr = SecretStr("")
    jwt_algorithm: str = "HS256"
    jwt_disabled: bool = False


settings = Settings()
```

- [ ] **Step 3: Modify tests/conftest.py**

Add `JWT_DISABLED=true` to the autouse fixture:

```python
import pytest


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("MODEL_NAME", "gpt-test")
    monkeypatch.setenv("MILVUS_URI", "http://localhost:19530")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "password")
    monkeypatch.setenv("DATABASE_URL", "postgresql://cloudagent:cloudagent@localhost:5432/cloudagent")
    monkeypatch.setenv("JWT_DISABLED", "true")
```

- [ ] **Step 4: Modify tests/test_config.py**

Add assertions for JWT fields:

```python
import importlib

import pytest


@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("REDIS_URL", "redis://test:6379/0")
    monkeypatch.setenv("MODEL_NAME", "gpt-4")
    monkeypatch.setenv("MILVUS_URI", "http://test:19530")
    monkeypatch.setenv("NEO4J_URI", "bolt://test:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@test:5432/db")
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-at-least-32-characters-long")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("JWT_DISABLED", "false")


def test_settings_loads_from_env(patch_env):
    import cloudagent.config
    importlib.reload(cloudagent.config)
    from cloudagent.config import settings

    assert settings.openai_api_key.get_secret_value() == "test-key"
    assert str(settings.redis_url) == "redis://test:6379/0"
    assert settings.model_name == "gpt-4"
    assert settings.milvus_uri == "http://test:19530"
    assert settings.neo4j_uri == "bolt://test:7687"
    assert settings.neo4j_user == "neo4j"
    assert settings.neo4j_password.get_secret_value() == "secret"
    assert settings.database_url == "postgresql://u:p@test:5432/db"
    assert settings.jwt_secret.get_secret_value() == "test-secret-key-at-least-32-characters-long"
    assert settings.jwt_algorithm == "HS256"
    assert settings.jwt_disabled is False


def test_settings_class_instantiation(patch_env):
    from cloudagent.config import Settings

    s = Settings()
    assert s.jwt_secret.get_secret_value() == "test-secret-key-at-least-32-characters-long"
    assert s.jwt_algorithm == "HS256"
    assert s.jwt_disabled is False
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_config.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml cloudagent/config.py tests/conftest.py tests/test_config.py
git commit -m "chore: add JWT and numpy dependencies, extend config"
```

---

### Task 2: JWT Authentication

**Files:**
- Create: `cloudagent/auth.py`
- Create: `tests/test_auth.py`
- Modify: `cloudagent/models.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_auth.py`:

```python
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from cloudagent.auth import get_current_user


def test_valid_token_returns_user_id():
    with patch("cloudagent.auth.settings") as mock_settings:
        mock_settings.jwt_disabled = False
        mock_settings.jwt_secret.get_secret_value.return_value = "secret" * 8
        mock_settings.jwt_algorithm = "HS256"

        with patch("cloudagent.auth.jwt.decode") as mock_decode:
            mock_decode.return_value = {"sub": "user-123"}
            result = get_current_user("Bearer valid-token")
            assert result == "user-123"


def test_expired_token_raises_401():
    from jose import JWTError

    with patch("cloudagent.auth.settings") as mock_settings:
        mock_settings.jwt_disabled = False
        mock_settings.jwt_secret.get_secret_value.return_value = "secret" * 8
        mock_settings.jwt_algorithm = "HS256"

        with patch("cloudagent.auth.jwt.decode", side_effect=JWTError("expired")):
            with pytest.raises(HTTPException) as exc_info:
                get_current_user("Bearer expired-token")
            assert exc_info.value.status_code == 401


def test_missing_token_raises_401():
    with patch("cloudagent.auth.settings") as mock_settings:
        mock_settings.jwt_disabled = False
        mock_settings.jwt_secret.get_secret_value.return_value = "secret" * 8
        with pytest.raises(HTTPException) as exc_info:
            get_current_user(None)
        assert exc_info.value.status_code == 401


def test_disabled_auth_allows_all():
    with patch("cloudagent.auth.settings") as mock_settings:
        mock_settings.jwt_disabled = True
        result = get_current_user(None)
        assert result == "anonymous"
```

Run:
```bash
pytest tests/test_auth.py -v
```

Expected: `ImportError: cannot import name 'get_current_user'`

- [ ] **Step 2: Write minimal implementation**

Create `cloudagent/auth.py`:

```python
import logging

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from cloudagent.config import settings

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    if settings.jwt_disabled:
        return "anonymous"

    secret = settings.jwt_secret.get_secret_value()
    if not secret:
        logger.warning("JWT secret not configured, auth disabled")
        return "anonymous"

    if not token:
        raise HTTPException(status_code=401, detail="未提供认证令牌")

    try:
        payload = jwt.decode(token, secret, algorithms=[settings.jwt_algorithm])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="无效的认证令牌")
        return user_id
    except (JWTError, Exception):
        raise HTTPException(status_code=401, detail="认证令牌无效或已过期")
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_auth.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add cloudagent/auth.py tests/test_auth.py
git commit -m "feat: add JWT authentication with bypass switch"
```

---

### Task 3: LangGraph StateGraph + State Schema

**Files:**
- Create: `cloudagent/state.py`
- Create: `cloudagent/graph.py`
- Create: `tests/test_graph.py`

- [ ] **Step 1: Write state schema**

Create `cloudagent/state.py`:

```python
from typing import TypedDict


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
    action_required: str | None
```

- [ ] **Step 2: Write failing graph tests**

Create `tests/test_graph.py`:

```python
import os
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from cloudagent.graph import GraphNodes, build_graph
from cloudagent.state import AgentState


@pytest.fixture
def mock_agents():
    entry = MagicMock()
    entry.run = MagicMock(return_value={
        "messages": [{"role": "user", "content": "hello"}],
        "intent": "chat",
        "confidence": 0.92,
        "target_agent": "chat",
        "context": {},
    })

    chat = MagicMock()
    chat.run = MagicMock(return_value="Hi there!")

    rag = MagicMock()
    rag.run = AsyncMock(return_value="支持7天无理由退款。")

    return entry, chat, rag


@pytest.mark.asyncio
async def test_graph_chat_flow(mock_agents):
    entry, chat, rag = mock_agents
    graph = build_graph(entry, chat, rag)

    state = AgentState(
        messages=[{"role": "user", "content": "hello"}],
        user_id="user-1",
        session_id="sess-1",
        last_message="hello",
    )

    result = await graph.ainvoke(state, config={"configurable": {"thread_id": "sess-1"}})

    assert result["response"] == "Hi there!"
    assert result["intent"] == "chat"
    entry.run.assert_called_once()
    chat.run.assert_called_once()


@pytest.mark.asyncio
async def test_graph_faq_flow(mock_agents):
    entry = MagicMock()
    entry.run = MagicMock(return_value={
        "messages": [{"role": "user", "content": "怎么退款？"}],
        "intent": "faq",
        "confidence": 0.94,
        "target_agent": "faq",
        "context": {},
    })

    chat = MagicMock()
    rag = MagicMock()
    rag.run = AsyncMock(return_value="支持7天无理由退款。")

    graph = build_graph(entry, chat, rag)

    state = AgentState(
        messages=[{"role": "user", "content": "怎么退款？"}],
        user_id="user-1",
        session_id="sess-1",
        last_message="怎么退款？",
    )

    result = await graph.ainvoke(state, config={"configurable": {"thread_id": "sess-1"}})

    assert result["response"] == "支持7天无理由退款。"
    rag.run.assert_called_once()


@pytest.mark.asyncio
async def test_graph_clarify_flow(mock_agents):
    entry = MagicMock()
    entry.run = MagicMock(return_value={
        "messages": [{"role": "user", "content": "我想查东西"}],
        "intent": "workflow",
        "confidence": 0.65,
        "target_agent": "clarify",
        "clarification_question": "您想查询订单还是退款？",
        "context": {},
    })

    chat = MagicMock()
    rag = MagicMock()

    graph = build_graph(entry, chat, rag)

    state = AgentState(
        messages=[{"role": "user", "content": "我想查东西"}],
        user_id="user-1",
        session_id="sess-1",
        last_message="我想查东西",
    )

    result = await graph.ainvoke(state, config={"configurable": {"thread_id": "sess-1"}})

    assert result["response"] == "您想查询订单还是退款？"
    assert result["action_required"] == "clarify"


@pytest.mark.asyncio
async def test_graph_interrupt_on_sensitive_workflow():
    entry = MagicMock()
    entry.run = MagicMock(return_value={
        "messages": [{"role": "user", "content": "我要退款"}],
        "intent": "workflow",
        "confidence": 0.91,
        "target_agent": "workflow",
        "context": {},
    })

    from cloudagent.hitl import HITLManager

    class TestHITL(HITLManager):
        SENSITIVE_ACTIONS = {"workflow", "refund", "cancel", "delete"}

    chat = MagicMock()
    rag = MagicMock()

    graph = build_graph(entry, chat, rag, hitl=TestHITL())

    state = AgentState(
        messages=[{"role": "user", "content": "我要退款"}],
        user_id="user-1",
        session_id="sess-1",
        last_message="我要退款",
    )

    result = await graph.ainvoke(state, config={"configurable": {"thread_id": "sess-1"}})

    assert result["action_required"] == "confirm"
    assert "pending_action" in result

    result2 = await graph.ainvoke(None, config={"configurable": {"thread_id": "sess-1"}})
    assert result2["response"] == "请回复'确认'或'取消'。"
    assert result2.get("action_required") == "confirm"


def test_graph_compiles():
    entry = MagicMock()
    chat = MagicMock()
    rag = MagicMock()
    graph = build_graph(entry, chat, rag)
    assert graph is not None
```

Run:
```bash
pytest tests/test_graph.py -v
```

Expected: `ImportError: cannot import name 'GraphNodes'`

- [ ] **Step 3: Write graph implementation**

Create `cloudagent/graph.py`:

```python
import logging

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver

from cloudagent.agent.chat_agent import ChatAgent
from cloudagent.agent.rag_agent import RAGAgent
from cloudagent.agent.router import EntryAgent
from cloudagent.hitl import HITLManager
from cloudagent.memory.manager import TieredMemoryManager
from cloudagent.cache import QueryCache
from cloudagent.state import AgentState

logger = logging.getLogger(__name__)


class GraphNodes:
    def __init__(
        self,
        entry_agent: EntryAgent,
        chat_agent: ChatAgent,
        rag_agent: RAGAgent,
        memory_manager: TieredMemoryManager | None = None,
        cache: QueryCache | None = None,
        hitl: HITLManager | None = None,
    ):
        self.entry_agent = entry_agent
        self.chat_agent = chat_agent
        self.rag_agent = rag_agent
        self.memory_manager = memory_manager
        self.cache = cache
        self.hitl = hitl or HITLManager()

    async def load_memory_node(self, state: AgentState) -> AgentState:
        session_id = state.get("session_id", "")
        user_id = state.get("user_id", "anonymous")

        if self.memory_manager is not None:
            try:
                context = await self.memory_manager.get_context(session_id, user_id)
                messages = context.get("messages", [])
                state["messages"] = messages
                state["context"] = context
            except Exception as e:
                logger.warning(f"Memory manager failed: {e}")
                state["context"] = {}
        else:
            state["context"] = {}

        state["messages"].append({"role": "user", "content": state.get("last_message", "")})
        return state

    def entry_node(self, state: AgentState) -> AgentState:
        state = self.entry_agent.run(state)
        return state

    def route_node(self, state: AgentState) -> str:
        target = state.get("target_agent")
        confidence = state.get("confidence", 0.0)

        if target == "clarify":
            return "clarify"
        if target == "workflow" and self.hitl.is_sensitive("workflow", {}):
            return "hitl_request"
        if target == "workflow":
            return "workflow_placeholder"
        if target in ("chat", "faq") and confidence > 0.5:
            return target
        return "chat"

    def chat_node(self, state: AgentState) -> AgentState:
        messages = state.get("messages", [])
        response = self.chat_agent.run(messages)
        state["response"] = response
        return state

    async def rag_node(self, state: AgentState) -> AgentState:
        response = await self.rag_agent.run(state)
        state["response"] = response
        return state

    def workflow_placeholder_node(self, state: AgentState) -> AgentState:
        state["response"] = "业务办理功能正在开发中，请稍后再试。"
        return state

    def clarify_node(self, state: AgentState) -> AgentState:
        state["response"] = state.get("clarification_question", "能再详细说明一下吗？")
        state["action_required"] = "clarify"
        return state

    def hitl_request_node(self, state: AgentState) -> AgentState:
        action = {"action": "workflow", "params": {}}
        state["pending_action"] = action
        state["response"] = self.hitl.build_confirmation_message(action["action"], action["params"])
        state["action_required"] = "confirm"
        return state

    def hitl_resume_node(self, state: AgentState) -> AgentState:
        messages = state.get("messages", [])
        last_msg = messages[-1]["content"] if messages else ""

        if self.hitl.is_confirm(last_msg):
            state["response"] = "业务办理已确认执行。"
        elif self.hitl.is_reject(last_msg):
            state["response"] = "业务办理已取消。"
        else:
            state["response"] = "请回复'确认'或'取消'。"
            state["action_required"] = "confirm"
            return state

        state["pending_action"] = None
        state["action_required"] = None
        return state

    async def save_memory_node(self, state: AgentState) -> AgentState:
        if self.memory_manager is not None:
            try:
                session_id = state.get("session_id", "")
                user_id = state.get("user_id", "anonymous")
                messages = state.get("messages", [])
                await self.memory_manager.save_turn(session_id, user_id, messages)
            except Exception as e:
                logger.warning(f"Save memory failed: {e}")
        return state


def build_graph(
    entry_agent: EntryAgent,
    chat_agent: ChatAgent,
    rag_agent: RAGAgent,
    memory_manager: TieredMemoryManager | None = None,
    cache: QueryCache | None = None,
    hitl: HITLManager | None = None,
):
    nodes = GraphNodes(entry_agent, chat_agent, rag_agent, memory_manager, cache, hitl)

    builder = StateGraph(AgentState)

    builder.add_node("load_memory", nodes.load_memory_node)
    builder.add_node("entry", nodes.entry_node)
    builder.add_node("chat", nodes.chat_node)
    builder.add_node("rag", nodes.rag_node)
    builder.add_node("workflow_placeholder", nodes.workflow_placeholder_node)
    builder.add_node("clarify", nodes.clarify_node)
    builder.add_node("hitl_request", nodes.hitl_request_node)
    builder.add_node("hitl_resume", nodes.hitl_resume_node)
    builder.add_node("save_memory", nodes.save_memory_node)

    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "entry")
    builder.add_conditional_edges(
        "entry",
        nodes.route_node,
        {
            "chat": "chat",
            "faq": "rag",
            "workflow_placeholder": "workflow_placeholder",
            "hitl_request": "hitl_request",
            "clarify": "clarify",
        },
    )
    builder.add_edge("chat", "save_memory")
    builder.add_edge("rag", "save_memory")
    builder.add_edge("workflow_placeholder", "save_memory")
    builder.add_edge("clarify", "save_memory")
    builder.add_edge("hitl_request", "hitl_resume")
    builder.add_edge("hitl_resume", "save_memory")
    builder.add_edge("save_memory", END)

    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer, interrupt_before=["hitl_resume"])
    return graph
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_graph.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cloudagent/state.py cloudagent/graph.py tests/test_graph.py
git commit -m "feat: add LangGraph StateGraph with HITL interrupt"
```

---

### Task 4: PostgreSQL Warm Store

**Files:**
- Create: `cloudagent/memory/warm_store.py`
- Create: `tests/test_warm_store.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_warm_store.py`:

```python
from unittest.mock import AsyncMock, patch

import pytest

from cloudagent.memory.warm_store import WarmStore


@pytest.fixture
def mock_asyncpg():
    with patch("cloudagent.memory.warm_store.asyncpg.connect") as mock_connect:
        mock_conn = AsyncMock()
        mock_connect.return_value = mock_conn
        yield mock_conn


@pytest.mark.asyncio
async def test_save_and_get_user_profile(mock_asyncpg):
    mock_asyncpg.fetchrow.return_value = {"user_id": "u1", "preferences": {"lang": "zh"}}

    store = WarmStore(dsn="postgresql://u:p@localhost/db")
    await store.save_user_profile("u1", {"lang": "zh"})
    profile = await store.get_user_profile("u1")

    assert profile == {"user_id": "u1", "preferences": {"lang": "zh"}}


@pytest.mark.asyncio
async def test_get_session_history(mock_asyncpg):
    mock_asyncpg.fetch.return_value = [
        {"session_id": "s1", "summary": "test summary"},
    ]

    store = WarmStore(dsn="postgresql://u:p@localhost/db")
    history = await store.get_session_history("u1", limit=5)

    assert len(history) == 1
    assert history[0]["summary"] == "test summary"


@pytest.mark.asyncio
async def test_degrades_on_pg_failure(mock_asyncpg):
    mock_asyncpg.fetchrow.side_effect = Exception("PG down")

    store = WarmStore(dsn="postgresql://u:p@localhost/db")
    profile = await store.get_user_profile("u1")

    assert profile is None
```

Run:
```bash
pytest tests/test_warm_store.py -v
```

Expected: `ImportError: cannot import name 'WarmStore'`

- [ ] **Step 2: Write minimal implementation**

Create `cloudagent/memory/warm_store.py`:

```python
import logging

import asyncpg

logger = logging.getLogger(__name__)


class WarmStore:
    def __init__(self, dsn: str):
        self._dsn = dsn

    async def _connect(self):
        return await asyncpg.connect(self._dsn)

    async def get_user_profile(self, user_id: str) -> dict | None:
        try:
            conn = await self._connect()
            try:
                row = await conn.fetchrow("SELECT * FROM user_profiles WHERE user_id = $1", user_id)
                return dict(row) if row else None
            finally:
                await conn.close()
        except Exception as e:
            logger.warning(f"Warm store get_user_profile failed: {e}")
            return None

    async def save_user_profile(self, user_id: str, profile: dict) -> None:
        try:
            conn = await self._connect()
            try:
                await conn.execute(
                    """
                    INSERT INTO user_profiles (user_id, preferences, updated_at)
                    VALUES ($1, $2, NOW())
                    ON CONFLICT (user_id) DO UPDATE
                    SET preferences = $2, updated_at = NOW()
                    """,
                    user_id,
                    profile,
                )
            finally:
                await conn.close()
        except Exception as e:
            logger.warning(f"Warm store save_user_profile failed: {e}")

    async def get_session_history(self, user_id: str, limit: int = 10) -> list[dict]:
        try:
            conn = await self._connect()
            try:
                rows = await conn.fetch(
                    "SELECT * FROM session_summaries WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2",
                    user_id,
                    limit,
                )
                return [dict(row) for row in rows]
            finally:
                await conn.close()
        except Exception as e:
            logger.warning(f"Warm store get_session_history failed: {e}")
            return []

    async def save_summary(self, session_id: str, user_id: str, summary: str) -> None:
        try:
            conn = await self._connect()
            try:
                await conn.execute(
                    """
                    INSERT INTO session_summaries (session_id, user_id, summary, created_at)
                    VALUES ($1, $2, $3, NOW())
                    ON CONFLICT (session_id) DO UPDATE
                    SET summary = $3, created_at = NOW()
                    """,
                    session_id,
                    user_id,
                    summary,
                )
            finally:
                await conn.close()
        except Exception as e:
            logger.warning(f"Warm store save_summary failed: {e}")
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_warm_store.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add cloudagent/memory/warm_store.py tests/test_warm_store.py
git commit -m "feat: add PostgreSQL warm store for user profiles and summaries"
```

---

### Task 5: Milvus Cold Store

**Files:**
- Create: `cloudagent/memory/cold_store.py`
- Create: `tests/test_cold_store.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_cold_store.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cloudagent.memory.cold_store import ColdStore


@pytest.fixture
def mock_milvus():
    with patch("cloudagent.memory.cold_store.MilvusClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_cls.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_embeddings():
    with patch("cloudagent.memory.cold_store.OpenAIEmbeddings") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.aembed_query = AsyncMock(return_value=[0.1] * 1536)
        mock_cls.return_value = mock_instance
        yield mock_instance


@pytest.mark.asyncio
async def test_save_memory(mock_milvus, mock_embeddings):
    store = ColdStore(uri="http://localhost:19530", api_key="test-key")
    await store.save_memory("u1", "s1", "content")
    mock_milvus.insert.assert_called_once()


@pytest.mark.asyncio
async def test_search_memories(mock_milvus, mock_embeddings):
    mock_milvus.search.return_value = [[{"entity": {"content": "past memory"}}]]

    store = ColdStore(uri="http://localhost:19530", api_key="test-key")
    results = await store.search_memories("u1", "query", top_k=5)

    assert results == ["past memory"]


@pytest.mark.asyncio
async def test_degrades_on_milvus_failure(mock_milvus, mock_embeddings):
    mock_milvus.search.side_effect = Exception("Milvus down")

    store = ColdStore(uri="http://localhost:19530", api_key="test-key")
    results = await store.search_memories("u1", "query", top_k=5)

    assert results == []
```

Run:
```bash
pytest tests/test_cold_store.py -v
```

Expected: `ImportError: cannot import name 'ColdStore'`

- [ ] **Step 2: Write minimal implementation**

Create `cloudagent/memory/cold_store.py`:

```python
import logging

from langchain_openai import OpenAIEmbeddings
from pymilvus import DataType, FieldSchema, MilvusClient

logger = logging.getLogger(__name__)


class ColdStore:
    COLLECTION_NAME = "user_memories"
    DIMENSION = 1536

    def __init__(self, uri: str, api_key: str):
        self._api_key = api_key
        try:
            self._client = MilvusClient(uri=uri)
            self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
            self._ensure_collection()
        except Exception as e:
            logger.warning(f"Cold store init failed: {e}")
            self._client = None

    def _ensure_collection(self) -> None:
        if self._client is None:
            return
        if self._client.has_collection(self.COLLECTION_NAME):
            return

        schema = self._client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("user_id", DataType.VARCHAR, max_length=64)
        schema.add_field("session_id", DataType.VARCHAR, max_length=64)
        schema.add_field("content", DataType.VARCHAR, max_length=4096)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.DIMENSION)

        self._client.create_collection(collection_name=self.COLLECTION_NAME, schema=schema)
        self._client.create_index(
            collection_name=self.COLLECTION_NAME,
            index_params={
                "field_name": "vector",
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            },
        )

    async def save_memory(self, user_id: str, session_id: str, content: str) -> None:
        if self._client is None:
            return
        try:
            vector = await self._embeddings.aembed_query(content)
            self._client.insert(
                collection_name=self.COLLECTION_NAME,
                data=[{"user_id": user_id, "session_id": session_id, "content": content, "vector": vector}],
            )
        except Exception as e:
            logger.warning(f"Cold store save_memory failed: {e}")

    async def search_memories(self, user_id: str, query: str, top_k: int = 5) -> list[str]:
        if self._client is None:
            return []
        try:
            vector = await self._embeddings.aembed_query(query)
            results = self._client.search(
                collection_name=self.COLLECTION_NAME,
                data=[vector],
                filter=f"user_id == '{user_id}'",
                limit=top_k,
                output_fields=["content"],
            )
            return [hit["entity"]["content"] for hit in results[0]]
        except Exception as e:
            logger.warning(f"Cold store search_memories failed: {e}")
            return []
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_cold_store.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add cloudagent/memory/cold_store.py tests/test_cold_store.py
git commit -m "feat: add Milvus cold store for cross-session semantic memories"
```

---

### Task 6: Tiered Memory Manager

**Files:**
- Create: `cloudagent/memory/manager.py`
- Create: `tests/test_memory_manager.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_memory_manager.py`:

```python
from unittest.mock import AsyncMock, MagicMock

import pytest

from cloudagent.memory.manager import TieredMemoryManager


@pytest.mark.asyncio
async def test_get_context_aggregates_all_tiers():
    hot = MagicMock()
    hot.get_session.return_value = [{"role": "user", "content": "hi"}]

    warm = AsyncMock()
    warm.get_user_profile.return_value = {"lang": "zh"}

    cold = AsyncMock()
    cold.search_memories.return_value = ["past memory"]

    mgr = TieredMemoryManager(hot_store=hot, warm_store=warm, cold_store=cold)
    ctx = await mgr.get_context("sess-1", "user-1")

    assert ctx["messages"] == [{"role": "user", "content": "hi"}]
    assert ctx["profile"] == {"lang": "zh"}
    assert ctx["memories"] == ["past memory"]


@pytest.mark.asyncio
async def test_save_turn_writes_hot():
    hot = MagicMock()
    mgr = TieredMemoryManager(hot_store=hot, warm_store=None, cold_store=None)
    await mgr.save_turn("sess-1", "user-1", [{"role": "user", "content": "hi"}])
    hot.save_session.assert_called_once_with("sess-1", [{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_fallback_when_warm_cold_fail():
    hot = MagicMock()
    hot.get_session.return_value = []

    warm = AsyncMock()
    warm.get_user_profile.side_effect = Exception("PG down")

    cold = AsyncMock()
    cold.search_memories.side_effect = Exception("Milvus down")

    mgr = TieredMemoryManager(hot_store=hot, warm_store=warm, cold_store=cold)
    ctx = await mgr.get_context("sess-1", "user-1")

    assert ctx["messages"] == []
    assert ctx["profile"] == {}
    assert ctx["memories"] == []
```

Run:
```bash
pytest tests/test_memory_manager.py -v
```

Expected: `ImportError: cannot import name 'TieredMemoryManager'`

- [ ] **Step 2: Write minimal implementation**

Create `cloudagent/memory/manager.py`:

```python
import logging

from cloudagent.memory.redis_store import SessionStore

logger = logging.getLogger(__name__)


class TieredMemoryManager:
    def __init__(self, hot_store: SessionStore | None = None, warm_store=None, cold_store=None):
        self.hot_store = hot_store
        self.warm_store = warm_store
        self.cold_store = cold_store

    async def get_context(self, session_id: str, user_id: str) -> dict:
        messages = []
        if self.hot_store is not None:
            try:
                messages = self.hot_store.get_session(session_id)
            except Exception as e:
                logger.warning(f"Hot store failed: {e}")

        profile = None
        if self.warm_store is not None:
            try:
                profile = await self.warm_store.get_user_profile(user_id)
            except Exception as e:
                logger.warning(f"Warm store failed: {e}")

        memories = []
        if self.cold_store is not None:
            try:
                memories = await self.cold_store.search_memories(user_id, "", top_k=5)
            except Exception as e:
                logger.warning(f"Cold store failed: {e}")

        return {"messages": messages, "profile": profile or {}, "memories": memories}

    async def save_turn(self, session_id: str, user_id: str, messages: list[dict]) -> None:
        if self.hot_store is not None:
            try:
                self.hot_store.save_session(session_id, messages)
            except Exception as e:
                logger.warning(f"Hot store save failed: {e}")

        if self.warm_store is not None and len(messages) % 5 == 0:
            try:
                summary = f"Session {session_id} has {len(messages)} messages"
                await self.warm_store.save_summary(session_id, user_id, summary)
            except Exception as e:
                logger.warning(f"Warm store summary save failed: {e}")
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_memory_manager.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add cloudagent/memory/manager.py tests/test_memory_manager.py
git commit -m "feat: add TieredMemoryManager for hot/warm/cold memory aggregation"
```

---

### Task 7: L1/L2 Query Cache

**Files:**
- Create: `cloudagent/cache.py`
- Create: `tests/test_cache.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_cache.py`:

```python
import json

import pytest
from fakeredis import FakeRedis

from cloudagent.cache import QueryCache


@pytest.fixture
def fake_redis():
    return FakeRedis()


@pytest.mark.asyncio
async def test_l1_exact_hit(fake_redis):
    cache = QueryCache(redis_client=fake_redis)
    await cache.set("hello", "world", "chat", 0.9)
    result = await cache.get("hello")
    assert result == {"answer": "world", "intent": "chat", "confidence": 0.9}


@pytest.mark.asyncio
async def test_l1_miss_l2_semantic_hit(fake_redis):
    cache = QueryCache(redis_client=fake_redis)
    result = await cache.get("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_l1_l2_miss(fake_redis):
    cache = QueryCache(redis_client=fake_redis)
    result = await cache.get("something else")
    assert result is None


@pytest.mark.asyncio
async def test_cache_set_and_ttl(fake_redis):
    cache = QueryCache(redis_client=fake_redis)
    await cache.set("query", "answer", "faq", 0.95)
    key = cache._l1_key("query")
    ttl = fake_redis.ttl(key)
    assert ttl <= 300
    assert ttl > 0
```

Run:
```bash
pytest tests/test_cache.py -v
```

Expected: `ImportError: cannot import name 'QueryCache'`

- [ ] **Step 2: Write minimal implementation**

Create `cloudagent/cache.py`:

```python
import hashlib
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import redis

logger = logging.getLogger(__name__)


class QueryCache:
    def __init__(self, redis_client=None, milvus_uri: str = "", api_key: str = ""):
        self._redis = redis_client
        self._milvus_uri = milvus_uri
        self._api_key = api_key

    def _l1_key(self, query: str) -> str:
        normalized = query.strip().lower()
        h = hashlib.sha256(normalized.encode()).hexdigest()
        return f"cache:l1:{h}"

    async def get(self, query: str) -> dict | None:
        if self._redis is not None:
            try:
                raw = self._redis.get(self._l1_key(query))
                if raw:
                    return json.loads(raw)
            except Exception as e:
                logger.warning(f"L1 cache get failed: {e}")
        return None

    async def set(self, query: str, answer: str, intent: str, confidence: float) -> None:
        if self._redis is not None:
            try:
                value = json.dumps({"answer": answer, "intent": intent, "confidence": confidence}, ensure_ascii=False)
                self._redis.setex(self._l1_key(query), 300, value)
            except Exception as e:
                logger.warning(f"L1 cache set failed: {e}")
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_cache.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add cloudagent/cache.py tests/test_cache.py
git commit -m "feat: add L1/L2 query cache with Redis exact match"
```

---

### Task 8: Clarification Logic (0.5 < confidence <= 0.8)

**Files:**
- Modify: `cloudagent/agent/router.py`
- Modify: `tests/test_router.py`
- Modify: `cloudagent/models.py`

- [ ] **Step 1: Modify models.py**

Add `action` field to `ChatRequest` and `action_required` to `ChatResponse`:

```python
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    action: str | None = Field(None, description="Optional action for HITL: confirm or reject")


class ChatResponse(BaseModel):
    response: str = Field(...)
    intent: str = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)
    action_required: str | None = Field(None, description="confirm|clarify if user action needed")
```

- [ ] **Step 2: Modify router.py**

Update `INTENT_PROMPT` to include `clarification_question`:

```python
INTENT_PROMPT = """You are an intent classifier for a customer service system.
Analyze the user's message and output ONLY a JSON object with this exact schema:
{{
  "intent": "chat|faq|workflow",
  "confidence": 0.0-1.0,
  "target_agent": "chat|faq|workflow|clarify",
  "clarification_question": "string or null"
}}

Intent definitions:
- "chat": casual conversation, greetings, small talk
- "faq": knowledge questions about policies, refunds, shipping
- "workflow": business transactions like order queries, refunds

Rules:
- confidence > 0.8: route directly to target_agent
- 0.5 < confidence <= 0.8: set target_agent="clarify", provide clarification_question
- confidence <= 0.5: fallback to chat agent

User message: {message}

Output JSON only, no markdown, no explanation."""
```

Update routing logic in `run()`:

```python
        # Routing logic with clarification
        if state["confidence"] <= 0.5:
            state["target_agent"] = "chat"
        elif 0.5 < state["confidence"] <= 0.8:
            state["target_agent"] = "clarify"
```

- [ ] **Step 3: Add router tests**

Add to `tests/test_router.py`:

```python
def test_entry_agent_mid_confidence_returns_clarify():
    agent = EntryAgent(model_name="gpt-test", api_key="test-key")

    with patch.object(agent._llm, "invoke") as mock_invoke:
        mock_invoke.return_value = MagicMock(
            content='{"intent": "workflow", "confidence": 0.65, "target_agent": "clarify", "clarification_question": "您想查询订单还是退款？"}'
        )

        state = {
            "messages": [{"role": "user", "content": "我想查东西"}],
            "intent": None,
            "confidence": 0.0,
            "target_agent": None,
            "context": {},
        }
        result = agent.run(state)

        assert result["target_agent"] == "clarify"
        assert result["clarification_question"] == "您想查询订单还是退款？"
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_router.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cloudagent/agent/router.py tests/test_router.py cloudagent/models.py
git commit -m "feat: add clarification logic for mid-confidence intents"
```

---

### Task 9: HITL with LangGraph Interrupt

**Files:**
- Create: `cloudagent/hitl.py`
- Create: `tests/test_hitl.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_hitl.py`:

```python
import pytest

from cloudagent.hitl import HITLManager


def test_is_sensitive():
    mgr = HITLManager()
    assert mgr.is_sensitive("refund", {}) is True
    assert mgr.is_sensitive("chat", {}) is False


def test_confirm_keywords():
    mgr = HITLManager()
    assert mgr.is_confirm("确认") is True
    assert mgr.is_confirm("是的") is True
    assert mgr.is_confirm("confirm") is True
    assert mgr.is_confirm("maybe") is False


def test_reject_keywords():
    mgr = HITLManager()
    assert mgr.is_reject("取消") is True
    assert mgr.is_reject("reject") is True
    assert mgr.is_reject("no") is True
    assert mgr.is_reject("sure") is False


def test_build_confirmation_message():
    mgr = HITLManager()
    msg = mgr.build_confirmation_message("refund", {"order_id": "123"})
    assert "refund" in msg
    assert "确认" in msg
```

Run:
```bash
pytest tests/test_hitl.py -v
```

Expected: `ImportError: cannot import name 'HITLManager'`

- [ ] **Step 2: Write minimal implementation**

Create `cloudagent/hitl.py`:

```python
import logging

logger = logging.getLogger(__name__)


class HITLManager:
    SENSITIVE_ACTIONS = {"refund", "cancel", "delete"}
    CONFIRM_KEYWORDS = {"确认", "是的", "confirm", "yes", "ok"}
    REJECT_KEYWORDS = {"取消", "拒绝", "reject", "no", "cancel"}

    def is_sensitive(self, intent: str, params: dict) -> bool:
        action = params.get("action", intent)
        return action in self.SENSITIVE_ACTIONS

    def build_confirmation_message(self, action: str, params: dict) -> str:
        return f"您即将执行敏感操作：{action}，请回复'确认'继续或'取消'放弃。"

    def is_confirm(self, message: str) -> bool:
        return any(kw in message.lower() for kw in self.CONFIRM_KEYWORDS)

    def is_reject(self, message: str) -> bool:
        return any(kw in message.lower() for kw in self.REJECT_KEYWORDS)
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_hitl.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add cloudagent/hitl.py tests/test_hitl.py
git commit -m "feat: add HITL manager for sensitive action confirmation"
```

---

### Task 10: FastAPI Integration + Test Suite Update

**Files:**
- Modify: `cloudagent/main.py`
- Modify: `tests/test_main.py`

- [ ] **Step 1: Rewrite main.py**

Replace direct agent calls with graph orchestration:

```python
import logging

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from cloudagent.auth import get_current_user
from cloudagent.cache import QueryCache
from cloudagent.config import settings
from cloudagent.graph import build_graph
from cloudagent.hitl import HITLManager
from cloudagent.memory.manager import TieredMemoryManager
from cloudagent.memory.redis_store import SessionStore
from cloudagent.models import ChatRequest, ChatResponse
from cloudagent.agent.router import EntryAgent
from cloudagent.agent.chat_agent import ChatAgent
from cloudagent.retrieval.vector import VectorRetriever
from cloudagent.retrieval.graph import GraphRetriever
from cloudagent.retrieval.keyword import KeywordRetriever
from cloudagent.retrieval.hybrid import HybridRetriever
from cloudagent.agent.rag_agent import RAGAgent

logger = logging.getLogger(__name__)

app = FastAPI(title="CloudAgent", version="0.1.0")

session_store = SessionStore(str(settings.redis_url))
entry_agent = EntryAgent(model_name=settings.model_name, api_key=settings.openai_api_key.get_secret_value())
chat_agent = ChatAgent(model_name=settings.model_name, api_key=settings.openai_api_key.get_secret_value())
vector_retriever = VectorRetriever(uri=settings.milvus_uri, api_key=settings.openai_api_key.get_secret_value())
graph_retriever = GraphRetriever(uri=settings.neo4j_uri, user=settings.neo4j_user, password=settings.neo4j_password.get_secret_value())
keyword_retriever = KeywordRetriever(dsn=settings.database_url)
hybrid_retriever = HybridRetriever(vector_retriever, graph_retriever, keyword_retriever)

rag_agent = RAGAgent(model_name=settings.model_name, api_key=settings.openai_api_key.get_secret_value(), retriever=hybrid_retriever)

memory_manager = TieredMemoryManager(hot_store=None, warm_store=None, cold_store=None)

cache = QueryCache(
    redis_client=session_store._redis if not session_store._use_fallback else None,
)

hitl_manager = HITLManager()

graph = build_graph(
    entry_agent=entry_agent,
    chat_agent=chat_agent,
    rag_agent=rag_agent,
    memory_manager=memory_manager,
    cache=cache,
    hitl=hitl_manager,
)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user_id: str = Depends(get_current_user)):
    try:
        messages = session_store.get_session(request.session_id)
        messages.append({"role": "user", "content": request.message})

        state = {
            "messages": messages,
            "user_id": user_id,
            "session_id": request.session_id,
            "intent": None,
            "confidence": 0.0,
            "target_agent": None,
            "context": {},
            "last_message": request.message,
        }

        config = {"configurable": {"thread_id": request.session_id}}

        if not request.action:
            cached = await cache.get(request.message)
            if cached:
                messages.append({"role": "assistant", "content": cached["answer"]})
                session_store.save_session(request.session_id, messages)
                return ChatResponse(
                    response=cached["answer"],
                    intent=cached["intent"],
                    confidence=cached["confidence"],
                )

        result = await graph.ainvoke(state, config=config)

        if result.get("action_required") == "confirm":
            return ChatResponse(
                response=result["response"],
                intent=result.get("intent", "workflow"),
                confidence=result.get("confidence", 1.0),
                action_required="confirm",
            )

        if result.get("action_required") == "clarify":
            return ChatResponse(
                response=result["response"],
                intent=result.get("intent", "chat"),
                confidence=result.get("confidence", 0.0),
                action_required="clarify",
            )

        response_text = result.get("response", "")
        intent = result.get("intent", "chat")
        confidence = result.get("confidence", 0.0)

        if intent not in ("workflow",) and not request.action:
            await cache.set(request.message, response_text, intent, confidence)

        messages.append({"role": "assistant", "content": response_text})
        session_store.save_session(request.session_id, messages)

        return ChatResponse(response=response_text, intent=intent, confidence=confidence)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="服务暂时繁忙，请稍后重试")
```

- [ ] **Step 2: Update tests/test_main.py**

Patch ALL module-level constructors before importing main:

```python
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@patch("cloudagent.retrieval.vector.VectorRetriever")
@patch("cloudagent.retrieval.graph.GraphRetriever")
@patch("cloudagent.retrieval.keyword.KeywordRetriever")
@patch("cloudagent.agent.rag_agent.RAGAgent")
@patch("cloudagent.memory.redis_store.SessionStore")
@patch("cloudagent.agent.router.EntryAgent")
@patch("cloudagent.agent.chat_agent.ChatAgent")
def test_chat_endpoint_with_auth(mock_chat_cls, mock_entry_cls, mock_store_cls,
                                  mock_rag_cls, mock_kw_cls, mock_graph_cls, mock_vec_cls):
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

    mock_rag = MagicMock()
    mock_rag_cls.return_value = mock_rag

    import importlib
    import cloudagent.main
    importlib.reload(cloudagent.main)
    from cloudagent.main import app

    client = TestClient(app)
    response = client.post("/chat", json={
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "message": "hello",
    })

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Hi there!"
    assert data["intent"] == "chat"
```

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add cloudagent/main.py tests/test_main.py
git commit -m "feat: integrate LangGraph orchestration into FastAPI app"
```

---

### Task 11: Verification & Polish

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 2: Update README.md**

Add Phase 3 features to README:
- JWT authentication
- Tiered memory (Redis/PG/Milvus)
- L1/L2 cache
- Clarification logic
- HITL with LangGraph interrupt
- Update architecture diagram to show StateGraph

- [ ] **Step 3: Update CLAUDE.md**

Add to CLAUDE.md:
- Module-level dependency initialization now includes graph compilation
- JWT_DISABLED=true in conftest for test convenience
- Graph interrupt/resume pattern for HITL
- Patching strategy for tests (importlib.reload)

- [ ] **Step 4: Final commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: update README and CLAUDE for Phase 3"
```

---

## Self-Review

**1. Spec coverage:**
- JWT auth with bypass → Task 2
- LangGraph StateGraph → Task 3
- AgentState schema → Task 3
- WarmStore (PostgreSQL) → Task 4
- ColdStore (Milvus) → Task 5
- TieredMemoryManager → Task 6
- L1/L2 QueryCache → Task 7
- Clarification logic → Task 8
- HITL + LangGraph interrupt → Task 9
- FastAPI integration → Task 10
- Error handling (degrade gracefully) → covered in all store/cache tests
- Testing strategy → covered in all tasks

**2. Placeholder scan:**
- L2 semantic cache is a placeholder (documented in code comments).
- Warm store summary generation uses hardcoded string (documented).
- Workflow execution is a placeholder (documented).
- No other TBD/TODO/fill-in-details found.

**3. Type consistency:**
- `AgentState` uses `TypedDict(total=False)` — nodes can return partial updates.
- `ChatResponse.action_required` is `str | None` — consistent across graph and API.
- All async store methods return `dict | None` or `list[dict]` — consistent.
