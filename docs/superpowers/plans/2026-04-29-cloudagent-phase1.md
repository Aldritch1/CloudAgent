# CloudAgent Phase1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working conversational API with FastAPI, an entry Agent (intent recognition + routing), a Chat Agent, and Redis session storage.

**Architecture:** FastAPI exposes a single `POST /chat` endpoint. Requests flow through: load session from Redis → entry Agent (LangGraph StateGraph) classifies intent with confidence → routes to Chat Agent → saves session → returns JSON. Redis failures degrade to in-memory dict.

**Tech Stack:** Python 3.11+, FastAPI, LangGraph, LangChain, Redis, pytest, fakeredis, pydantic

---

## File Structure

```
cloudagent/
├── __init__.py
├── main.py              # FastAPI app, /chat endpoint
├── config.py            # Settings from env vars
├── models.py            # Pydantic ChatRequest, ChatResponse
├── agent/
│   ├── __init__.py
│   ├── router.py        # Entry Agent StateGraph (intent → route)
│   └── chat_agent.py    # Chat Agent (LLM with system prompt)
└── memory/
    ├── __init__.py
    └── redis_store.py   # Redis session store with in-memory fallback
tests/
├── __init__.py
├── test_models.py
├── test_redis_store.py
├── test_chat_agent.py
├── test_router.py
└── test_main.py
pyproject.toml
.env.example
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `cloudagent/__init__.py`
- Create: `cloudagent/agent/__init__.py`
- Create: `cloudagent/memory/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "cloudagent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.29.0",
    "pydantic>=2.6.0",
    "pydantic-settings>=2.2.0",
    "langchain>=0.1.0",
    "langchain-openai>=0.0.8",
    "langgraph>=0.0.26",
    "redis>=5.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "httpx>=0.27.0",
    "fakeredis>=2.21.0",
]
```

- [ ] **Step 2: Create .env.example**

```bash
OPENAI_API_KEY=sk-...
REDIS_URL=redis://localhost:6379/0
MODEL_NAME=gpt-3.5-turbo
```

- [ ] **Step 3: Create empty __init__.py files**

Create empty files at:
- `cloudagent/__init__.py`
- `cloudagent/agent/__init__.py`
- `cloudagent/memory/__init__.py`
- `tests/__init__.py`

- [ ] **Step 4: Install dependencies**

```bash
pip install -e ".[dev]"
```

Expected: installs all packages without error.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .env.example cloudagent/__init__.py cloudagent/agent/__init__.py cloudagent/memory/__init__.py tests/__init__.py
git commit -m "chore: project scaffolding for phase1"
```

---

### Task 2: Configuration Management

**Files:**
- Create: `cloudagent/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_config.py`:

```python
import os

from cloudagent.config import Settings


def test_settings_loads_from_env():
    os.environ["OPENAI_API_KEY"] = "test-key"
    os.environ["REDIS_URL"] = "redis://test:6379/0"
    os.environ["MODEL_NAME"] = "gpt-4"

    settings = Settings()
    assert settings.openai_api_key.get_secret_value() == "test-key"
    assert str(settings.redis_url) == "redis://test:6379/0"
    assert settings.model_name == "gpt-4"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```

Expected: `ImportError: cannot import name 'Settings' from 'cloudagent.config'`

- [ ] **Step 3: Write minimal implementation**

Create `cloudagent/config.py`:

```python
from pydantic import RedisDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: SecretStr
    redis_url: RedisDsn = "redis://localhost:6379/0"
    model_name: str = "gpt-3.5-turbo"


settings = Settings()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_config.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_config.py cloudagent/config.py
git commit -m "feat: add configuration management with pydantic-settings"
```

---

### Task 3: Pydantic Models

**Files:**
- Create: `cloudagent/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_models.py`:

```python
import pytest
from pydantic import ValidationError

from cloudagent.models import ChatRequest, ChatResponse


def test_chat_request_valid():
    req = ChatRequest(session_id="550e8400-e29b-41d4-a716-446655440000", message="hello")
    assert req.session_id == "550e8400-e29b-41d4-a716-446655440000"
    assert req.message == "hello"


def test_chat_request_missing_session_id():
    with pytest.raises(ValidationError):
        ChatRequest(message="hello")


def test_chat_response_valid():
    resp = ChatResponse(response="hi", intent="chat", confidence=0.92)
    assert resp.response == "hi"
    assert resp.confidence == 0.92
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_models.py -v
```

Expected: `ImportError: cannot import name 'ChatRequest' from 'cloudagent.models'`

- [ ] **Step 3: Write minimal implementation**

Create `cloudagent/models.py`:

```python
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Unique session identifier")
    message: str = Field(..., min_length=1, description="User message")


class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant response")
    intent: str = Field(..., description="Detected intent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Intent confidence")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_models.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_models.py cloudagent/models.py
git commit -m "feat: add chat request and response models"
```

---

### Task 4: Redis Session Store with Fallback

**Files:**
- Create: `cloudagent/memory/redis_store.py`
- Create: `tests/test_redis_store.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_redis_store.py`:

```python
import json

import fakeredis

from cloudagent.memory.redis_store import SessionStore


def test_save_and_get_session():
    redis_client = fakeredis.FakeRedis()
    store = SessionStore(redis_url="redis://fake", _redis_client=redis_client)

    messages = [{"role": "user", "content": "hello"}]
    store.save_session("sess-1", messages)

    loaded = store.get_session("sess-1")
    assert loaded == messages


def test_get_nonexistent_session_returns_empty():
    redis_client = fakeredis.FakeRedis()
    store = SessionStore(redis_url="redis://fake", _redis_client=redis_client)

    loaded = store.get_session("sess-none")
    assert loaded == []


def test_redis_fallback_on_connection_error():
    # Pass a URL that will fail connection; store should fallback to memory
    store = SessionStore(redis_url="redis://invalid:9999/0")

    messages = [{"role": "user", "content": "hi"}]
    store.save_session("sess-fb", messages)

    loaded = store.get_session("sess-fb")
    assert loaded == messages


def test_ttl_set_on_save():
    redis_client = fakeredis.FakeRedis()
    store = SessionStore(redis_url="redis://fake", _redis_client=redis_client)

    store.save_session("sess-ttl", [{"role": "user", "content": "hello"}])
    ttl = redis_client.ttl("session:sess-ttl")
    assert ttl == 3600
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_redis_store.py -v
```

Expected: `ImportError: cannot import name 'SessionStore'`

- [ ] **Step 3: Write minimal implementation**

Create `cloudagent/memory/redis_store.py`:

```python
import json
import logging

import redis

logger = logging.getLogger(__name__)


class SessionStore:
    def __init__(self, redis_url: str, _redis_client=None):
        self._fallback: dict[str, list] = {}
        self._use_fallback = False

        if _redis_client is not None:
            self._redis = _redis_client
        else:
            try:
                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
            except redis.ConnectionError:
                logger.warning("Redis connection failed, falling back to in-memory store")
                self._use_fallback = True
                self._redis = None

    def get_session(self, session_id: str) -> list[dict]:
        if self._use_fallback:
            return self._fallback.get(session_id, [])

        try:
            raw = self._redis.get(f"session:{session_id}")
            if raw is None:
                return []
            return json.loads(raw)
        except redis.RedisError:
            logger.warning("Redis read error, falling back to in-memory store")
            self._use_fallback = True
            return self._fallback.get(session_id, [])

    def save_session(self, session_id: str, messages: list[dict]) -> None:
        if self._use_fallback:
            self._fallback[session_id] = messages
            return

        try:
            self._redis.setex(
                f"session:{session_id}",
                3600,
                json.dumps(messages, ensure_ascii=False),
            )
        except redis.RedisError:
            logger.warning("Redis write error, falling back to in-memory store")
            self._use_fallback = True
            self._fallback[session_id] = messages
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_redis_store.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_redis_store.py cloudagent/memory/redis_store.py
git commit -m "feat: add Redis session store with in-memory fallback"
```

---

### Task 5: Chat Agent

**Files:**
- Create: `cloudagent/agent/chat_agent.py`
- Create: `tests/test_chat_agent.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_chat_agent.py`:

```python
from unittest.mock import MagicMock, patch

from cloudagent.agent.chat_agent import ChatAgent


@patch("cloudagent.agent.chat_agent.ChatOpenAI")
def test_chat_agent_returns_response(mock_llm_class):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Hello there!")
    mock_llm_class.return_value = mock_llm

    agent = ChatAgent(model_name="gpt-test", api_key="test-key")
    messages = [{"role": "user", "content": "hi"}]
    response = agent.run(messages)

    assert response == "Hello there!"
    mock_llm.invoke.assert_called_once()
    call_args = mock_llm.invoke.call_args[0][0]
    assert len(call_args) == 2  # system + user message
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_chat_agent.py -v
```

Expected: `ImportError: cannot import name 'ChatAgent'`

- [ ] **Step 3: Write minimal implementation**

Create `cloudagent/agent/chat_agent.py`:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


SYSTEM_PROMPT = """You are a helpful customer service assistant.
Answer user questions politely and concisely in Chinese.
If you don't know something, say so honestly."""


class ChatAgent:
    def __init__(self, model_name: str, api_key: str):
        self._llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.7,
        )

    def _convert_messages(self, messages: list[dict]) -> list:
        converted = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in messages:
            if msg["role"] == "user":
                converted.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                converted.append(AIMessage(content=msg["content"]))
        return converted

    def run(self, messages: list[dict]) -> str:
        converted = self._convert_messages(messages)
        response = self._llm.invoke(converted)
        return response.content
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_chat_agent.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_chat_agent.py cloudagent/agent/chat_agent.py
git commit -m "feat: add chat agent with system prompt"
```

---

### Task 6: Entry Agent (Router)

**Files:**
- Create: `cloudagent/agent/router.py`
- Create: `tests/test_router.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_router.py`:

```python
import json
from unittest.mock import MagicMock, patch

from cloudagent.agent.router import EntryAgent


@patch("cloudagent.agent.router.ChatOpenAI")
def test_high_confidence_routes_to_chat(mock_llm_class):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content=json.dumps({
            "intent": "chat",
            "confidence": 0.92,
            "target_agent": "chat",
        })
    )
    mock_llm_class.return_value = mock_llm

    agent = EntryAgent(model_name="gpt-test", api_key="test-key")
    state = {
        "messages": [{"role": "user", "content": "hello"}],
        "intent": None,
        "confidence": 0.0,
        "target_agent": None,
        "context": {},
    }
    result = agent.run(state)

    assert result["intent"] == "chat"
    assert result["confidence"] == 0.92
    assert result["target_agent"] == "chat"


@patch("cloudagent.agent.router.ChatOpenAI")
def test_low_confidence_defaults_to_chat(mock_llm_class):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content=json.dumps({
            "intent": "unknown",
            "confidence": 0.3,
            "target_agent": "unknown",
        })
    )
    mock_llm_class.return_value = mock_llm

    agent = EntryAgent(model_name="gpt-test", api_key="test-key")
    state = {
        "messages": [{"role": "user", "content": "xyz"}],
        "intent": None,
        "confidence": 0.0,
        "target_agent": None,
        "context": {},
    }
    result = agent.run(state)

    assert result["target_agent"] == "chat"


@patch("cloudagent.agent.router.ChatOpenAI")
def test_llm_failure_fallback(mock_llm_class):
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM error")
    mock_llm_class.return_value = mock_llm

    agent = EntryAgent(model_name="gpt-test", api_key="test-key")
    state = {
        "messages": [{"role": "user", "content": "hello"}],
        "intent": None,
        "confidence": 0.0,
        "target_agent": None,
        "context": {},
    }
    result = agent.run(state)

    assert result["confidence"] == 0.0
    assert result["target_agent"] == "chat"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_router.py -v
```

Expected: `ImportError: cannot import name 'EntryAgent'`

- [ ] **Step 3: Write minimal implementation**

Create `cloudagent/agent/router.py`:

```python
import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

INTENT_PROMPT = """You are an intent classifier for a customer service system.
Analyze the user's message and output ONLY a JSON object with this exact schema:
{
  "intent": "chat",
  "confidence": 0.0-1.0,
  "target_agent": "chat"
}

Intent definitions:
- "chat": casual conversation, greetings, small talk, general chitchat

Rules:
- confidence > 0.8: user is clearly making small talk or greeting
- confidence <= 0.5: unclear or unrelated, fallback to chat agent
- Always set target_agent to "chat" for phase1.

User message: {message}

Output JSON only, no markdown, no explanation."""


class EntryAgent:
    def __init__(self, model_name: str, api_key: str):
        self._llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.0,
            max_tokens=128,
        )

    def run(self, state: dict) -> dict:
        user_message = ""
        for msg in reversed(state["messages"]):
            if msg["role"] == "user":
                user_message = msg["content"]
                break

        prompt = INTENT_PROMPT.format(message=user_message)
        messages = [SystemMessage(content=prompt)]

        try:
            response = self._llm.invoke(messages)
            parsed = json.loads(response.content.strip())
            state["intent"] = parsed.get("intent", "chat")
            state["confidence"] = float(parsed.get("confidence", 0.0))
            state["target_agent"] = parsed.get("target_agent", "chat")
        except Exception as e:
            logger.error(f"Intent recognition failed: {e}")
            state["intent"] = "chat"
            state["confidence"] = 0.0
            state["target_agent"] = "chat"

        # Routing logic: phase1 only has chat agent
        if state["confidence"] <= 0.5:
            state["target_agent"] = "chat"

        return state
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_router.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_router.py cloudagent/agent/router.py
git commit -m "feat: add entry agent with intent recognition and routing"
```

---

### Task 7: FastAPI Main Application

**Files:**
- Create: `cloudagent/main.py`
- Create: `tests/test_main.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_main.py`:

```python
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# Patch settings before importing main
@patch("cloudagent.main.settings")
@patch("cloudagent.main.SessionStore")
@patch("cloudagent.main.EntryAgent")
@patch("cloudagent.main.ChatAgent")
def test_chat_endpoint_success(mock_chat_cls, mock_entry_cls, mock_store_cls, mock_settings):
    mock_settings.openai_api_key = MagicMock(get_secret_value=MagicMock(return_value="test-key"))
    mock_settings.model_name = "gpt-test"
    mock_settings.redis_url = "redis://test"

    mock_store = MagicMock()
    mock_store.get_session.return_value = []
    mock_store_cls.return_value = mock_store

    mock_entry = MagicMock()
    mock_entry.run.return_value = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        "intent": "chat",
        "confidence": 0.92,
        "target_agent": "chat",
        "context": {},
    }
    mock_entry_cls.return_value = mock_entry

    mock_chat = MagicMock()
    mock_chat.run.return_value = "Hi there!"
    mock_chat_cls.return_value = mock_chat

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
    assert data["confidence"] == 0.92
    mock_store.save_session.assert_called_once()


@patch("cloudagent.main.settings")
@patch("cloudagent.main.SessionStore")
@patch("cloudagent.main.EntryAgent")
@patch("cloudagent.main.ChatAgent")
def test_chat_endpoint_invalid_request(mock_chat_cls, mock_entry_cls, mock_store_cls, mock_settings):
    mock_settings.openai_api_key = MagicMock(get_secret_value=MagicMock(return_value="test-key"))
    mock_settings.model_name = "gpt-test"
    mock_settings.redis_url = "redis://test"

    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store
    mock_entry_cls.return_value = MagicMock()
    mock_chat_cls.return_value = MagicMock()

    from cloudagent.main import app

    client = TestClient(app)
    response = client.post("/chat", json={"message": "missing session_id"})
    assert response.status_code == 422


@patch("cloudagent.main.settings")
@patch("cloudagent.main.SessionStore")
@patch("cloudagent.main.EntryAgent")
@patch("cloudagent.main.ChatAgent")
def test_chat_agent_failure(mock_chat_cls, mock_entry_cls, mock_store_cls, mock_settings):
    mock_settings.openai_api_key = MagicMock(get_secret_value=MagicMock(return_value="test-key"))
    mock_settings.model_name = "gpt-test"
    mock_settings.redis_url = "redis://test"

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
    mock_chat.run.side_effect = Exception("LLM error")
    mock_chat_cls.return_value = mock_chat

    from cloudagent.main import app

    client = TestClient(app)
    response = client.post("/chat", json={
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "message": "hello",
    })

    assert response.status_code == 500
    assert "error" in response.json()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_main.py -v
```

Expected: `ImportError: cannot import name 'app' from 'cloudagent.main'`

- [ ] **Step 3: Write minimal implementation**

Create `cloudagent/main.py`:

```python
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from cloudagent.config import settings
from cloudagent.models import ChatRequest, ChatResponse
from cloudagent.memory.redis_store import SessionStore
from cloudagent.agent.router import EntryAgent
from cloudagent.agent.chat_agent import ChatAgent

logger = logging.getLogger(__name__)

app = FastAPI(title="CloudAgent", version="0.1.0")

# Initialize dependencies
session_store = SessionStore(str(settings.redis_url))
entry_agent = EntryAgent(
    model_name=settings.model_name,
    api_key=settings.openai_api_key.get_secret_value(),
)
chat_agent = ChatAgent(
    model_name=settings.model_name,
    api_key=settings.openai_api_key.get_secret_value(),
)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Load session history
        messages = session_store.get_session(request.session_id)

        # Append user message
        messages.append({"role": "user", "content": request.message})

        # Run entry agent (intent recognition + routing)
        state = {
            "messages": messages,
            "intent": None,
            "confidence": 0.0,
            "target_agent": None,
            "context": {},
        }
        state = entry_agent.run(state)

        # Phase1: only chat agent exists
        try:
            response_text = chat_agent.run(state["messages"])
        except Exception as e:
            logger.error(f"Chat agent failed: {e}")
            raise HTTPException(status_code=500, detail="服务暂时繁忙，请稍后重试")

        # Append assistant message
        messages.append({"role": "assistant", "content": response_text})
        state["messages"] = messages

        # Save session
        session_store.save_session(request.session_id, messages)

        return ChatResponse(
            response=response_text,
            intent=state["intent"],
            confidence=state["confidence"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "服务暂时繁忙，请稍后重试"},
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_main.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_main.py cloudagent/main.py
git commit -m "feat: add FastAPI app with /chat endpoint"
```

---

### Task 8: Integration Verification

**Files:**
- Modify: `cloudagent/main.py` (add health check endpoint)
- Modify: `tests/test_main.py` (add health check test)

- [ ] **Step 1: Add health check endpoint**

Edit `cloudagent/main.py`, add before the `chat` endpoint:

```python
@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
```

- [ ] **Step 2: Add health check test**

Add to `tests/test_main.py`:

```python
@patch("cloudagent.main.settings")
@patch("cloudagent.main.SessionStore")
@patch("cloudagent.main.EntryAgent")
@patch("cloudagent.main.ChatAgent")
def test_health_endpoint(mock_chat_cls, mock_entry_cls, mock_store_cls, mock_settings):
    mock_settings.openai_api_key = MagicMock(get_secret_value=MagicMock(return_value="test-key"))
    mock_settings.model_name = "gpt-test"
    mock_settings.redis_url = "redis://test"
    mock_store_cls.return_value = MagicMock()
    mock_entry_cls.return_value = MagicMock()
    mock_chat_cls.return_value = MagicMock()

    from cloudagent.main import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
```

- [ ] **Step 3: Run all tests**

```bash
pytest tests/ -v
```

Expected: All tests PASS.

- [ ] **Step 4: Run the app locally (manual smoke test)**

```bash
# Ensure OPENAI_API_KEY is set in environment or .env
uvicorn cloudagent.main:app --reload --port 8000
```

In another terminal:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test-1","message":"你好"}'
```

Expected: JSON response with `response`, `intent`, `confidence` fields.

```bash
curl http://localhost:8000/health
```

Expected: `{"status":"ok","version":"0.1.0"}`

- [ ] **Step 5: Final commit**

```bash
git add tests/test_main.py cloudagent/main.py
git commit -m "feat: add health check endpoint and integration smoke tests"
```

---

## Self-Review

**1. Spec coverage:**
- FastAPI + `/chat` endpoint → Task 7
- Pydantic models → Task 3
- Config management → Task 2
- Entry Agent with intent recognition and routing logic → Task 6
- Chat Agent with system prompt → Task 5
- Redis session store with TTL and fallback → Task 4
- Error handling (LLM failure, Redis failure, validation) → Tasks 4, 6, 7
- Testing strategy (unit + integration + API) → Every task includes tests
- Health endpoint for monitoring → Task 8

**2. Placeholder scan:** No TBD, TODO, or vague steps found. Every step has exact file paths, exact code, exact commands, expected output.

**3. Type consistency:**
- `ChatRequest`/`ChatResponse` used consistently
- `SessionStore.get_session` returns `list[dict]`, used in main.py
- `EntryAgent.run` takes/returns `dict`, used in main.py
- `ChatAgent.run` takes `list[dict]` returns `str`, used in main.py
- Config `settings.openai_api_key.get_secret_value()` used in main.py

No gaps or inconsistencies found.
