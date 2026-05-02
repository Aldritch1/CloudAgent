# CloudAgent Phase6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Vue3 frontend with SSE streaming support, enabling real-time token-by-token LLM output visualization and tool-call cards in the chat UI.

**Architecture:** `cloudagent/api/sse.py` provides `/chat/stream` endpoint using `EventSourceResponse`. ChatAgent/RAGAgent/WorkflowAgent gain `run_stream()` async generators. `frontend/` is a standalone Vite + Vue3 + Element Plus project.

**Tech Stack:** Python 3.11+, FastAPI, sse-starlette, Vue3, Vite, Element Plus, Pinia, TypeScript

---

## File Structure

```
cloudagent/
├── main.py                  # MODIFIED: Add SSE router + CORS middleware
├── config.py                # MODIFIED: Add enable_sse, cors_origins
├── api/
│   ├── __init__.py
│   └── sse.py               # NEW: SSE streaming endpoint
├── agent/
│   ├── chat_agent.py        # MODIFIED: Add run_stream()
│   ├── rag_agent.py         # MODIFIED: Add run_stream()
│   └── workflow_agent.py    # MODIFIED: Add run_stream()
└── graph.py                 # MODIFIED: Add streaming execution path

frontend/                    # NEW
├── index.html
├── package.json
├── vite.config.ts
├── tsconfig.json
├── src/
│   ├── main.ts
│   ├── App.vue
│   ├── router/index.ts
│   ├── views/ChatView.vue
│   ├── components/
│   │   ├── ChatMessage.vue
│   │   ├── ChatInput.vue
│   │   ├── IntentBadge.vue
│   │   └── Sidebar.vue
│   ├── api/chat.ts
│   ├── stores/chat.ts
│   └── types/chat.ts
└── public/

tests/
├── test_sse.py              # NEW
└── test_main.py             # MODIFIED: CORS/SSE tests
```

---

### Task 1: Dependencies + Config Extension

**Files:**
- Modify: `pyproject.toml`
- Modify: `cloudagent/config.py`
- Modify: `tests/conftest.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Add `sse-starlette` to pyproject.toml**

```toml
dependencies = [
    # ... existing deps ...
    "sse-starlette>=1.6.1",
]
```

- [ ] **Step 2: Modify cloudagent/config.py**

Add fields to `Settings`:
```python
enable_sse: bool = True
cors_origins: str = "*"
```

- [ ] **Step 3: Modify tests/conftest.py**

```python
monkeypatch.setenv("ENABLE_SSE", "true")
monkeypatch.setenv("CORS_ORIGINS", "http://localhost:5173")
```

- [ ] **Step 4: Modify tests/test_config.py**

Add assertions for new fields.

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_config.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml cloudagent/config.py tests/conftest.py tests/test_config.py
git commit -m "chore: add SSE and CORS dependencies and config"
```

---

### Task 2: SSE Endpoint

**Files:**
- Create: `cloudagent/api/__init__.py`
- Create: `cloudagent/api/sse.py`
- Create: `tests/test_sse.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_sse.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

@patch("cloudagent.api.sse.get_current_user", return_value="test-user")
@patch("cloudagent.api.sse.tenant_dependency", return_value="default")
def test_sse_endpoint_returns_events(mock_tenant, mock_user):
    from cloudagent.main import app
    client = TestClient(app)
    response = client.post("/chat/stream", json={
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "message": "hello",
    }, headers={"Accept": "text/event-stream"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
```

Run: `pytest tests/test_sse.py -v`
Expected: `404` or import error (endpoint doesn't exist yet).

- [ ] **Step 2: Implement SSE endpoint**

Create `cloudagent/api/sse.py`:

```python
import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from cloudagent.auth import get_current_user
from cloudagent.tenant import tenant_dependency
from cloudagent.models import ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter()

async def event_generator(request: ChatRequest, user_id: str, tenant_id: str) -> AsyncIterator[dict]:
    # Placeholder: yield a simple done event
    yield {"event": "done", "data": json.dumps({"response": "SSE streaming placeholder"})}

@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    user_id: str = Depends(get_current_user),
    tenant_id: str = Depends(tenant_dependency),
):
    return EventSourceResponse(event_generator(request, user_id, tenant_id))
```

- [ ] **Step 3: Wire into main.py**

Add to `cloudagent/main.py`:
```python
from cloudagent.api import sse
app.include_router(sse.router, prefix="/api")
```

Wait — the existing `/chat` is at root. For consistency, put SSE at `/chat/stream` directly (no `/api` prefix).

Update: `app.include_router(sse.router)` and set router path to `/chat/stream`.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_sse.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cloudagent/api/ tests/test_sse.py cloudagent/main.py
git commit -m "feat: add SSE streaming endpoint skeleton"
```

---

### Task 3: Agent Streaming (run_stream)

**Files:**
- Modify: `cloudagent/agent/chat_agent.py`
- Modify: `cloudagent/agent/rag_agent.py`
- Modify: `cloudagent/agent/workflow_agent.py`
- Create: `tests/test_chat_agent_stream.py`

- [ ] **Step 1: Add run_stream to ChatAgent**

```python
async def run_stream(self, messages: list) -> AsyncIterator[str]:
    """Yield tokens as they are generated."""
    from langchain_core.messages import SystemMessage, HumanMessage
    system_prompt = "..."
    msgs = [SystemMessage(content=system_prompt)]
    for msg in messages:
        if msg["role"] == "user":
            msgs.append(HumanMessage(content=msg["content"]))
    async for chunk in self._llm.astream(msgs):
        yield chunk.content
```

- [ ] **Step 2: Add run_stream to RAGAgent**

Similar pattern, but include retrieved context in the prompt and stream the response.

- [ ] **Step 3: Add run_stream to WorkflowAgent**

Stream LLM output; when tool call detected, yield tool_call event, execute tool, then stream follow-up.

- [ ] **Step 4: Write tests**

Mock `self._llm.astream` to yield chunks, verify tokens are yielded.

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_chat_agent.py tests/test_rag_agent.py tests/test_workflow_agent.py -v
```

Expected: PASS (existing + new stream tests).

- [ ] **Step 6: Commit**

```bash
git add cloudagent/agent/chat_agent.py cloudagent/agent/rag_agent.py cloudagent/agent/workflow_agent.py tests/
git commit -m "feat: add streaming run_stream to all agents"
```

---

### Task 4: Graph Streaming Integration

**Files:**
- Modify: `cloudagent/graph.py`
- Modify: `cloudagent/api/sse.py`
- Modify: `tests/test_graph.py`

- [ ] **Step 1: Add streaming node to GraphNodes**

```python
async def stream_node(self, state: AgentState) -> AsyncIterator[dict]:
    target = state.get("target_agent")
    if target == "chat":
        async for token in self.chat_agent.run_stream(state.get("messages", [])):
            yield {"event": "token", "data": token}
    elif target == "faq":
        async for token in self.rag_agent.run_stream(state):
            yield {"event": "token", "data": token}
    elif target == "workflow":
        async for event in self._stream_workflow(state):
            yield event
    yield {"event": "done", "data": json.dumps({"response": state.get("response", "")})}
```

- [ ] **Step 2: Update SSE endpoint to use graph streaming**

Replace placeholder `event_generator` with real graph streaming logic.

- [ ] **Step 3: Add graph stream tests**

Test that streaming through the graph yields expected events.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_graph.py tests/test_sse.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cloudagent/graph.py cloudagent/api/sse.py tests/
git commit -m "feat: integrate agent streaming into graph and SSE endpoint"
```

---

### Task 5: CORS Middleware

**Files:**
- Modify: `cloudagent/main.py`
- Modify: `tests/test_main.py`

- [ ] **Step 1: Add CORS middleware to main.py**

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

- [ ] **Step 2: Add CORS test**

```python
def test_cors_preflight(mock_chat_cls, mock_entry_cls, mock_store_cls, ...):
    from cloudagent.main import app
    client = TestClient(app)
    response = client.options("/chat", headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
    })
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_main.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add cloudagent/main.py tests/test_main.py
git commit -m "feat: add CORS middleware for frontend access"
```

---

### Task 6: Vue3 Frontend Scaffold

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tsconfig.json`
- Create: `frontend/index.html`
- Create: `frontend/src/main.ts`
- Create: `frontend/src/App.vue`
- Create: `frontend/src/router/index.ts`

- [ ] **Step 1: Initialize frontend project**

Create `frontend/package.json`:
```json
{
  "name": "cloudagent-frontend",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "vite",
    "build": "vue-tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "vue": "^3.4.0",
    "vue-router": "^4.3.0",
    "element-plus": "^2.7.0",
    "@element-plus/icons-vue": "^2.3.0",
    "pinia": "^2.1.0"
  },
  "devDependencies": {
    "vite": "^5.2.0",
    "@vitejs/plugin-vue": "^5.0.0",
    "typescript": "^5.4.0",
    "vue-tsc": "^2.0.0"
  }
}
```

- [ ] **Step 2: Create base config files**

`vite.config.ts`, `tsconfig.json`, `index.html`

- [ ] **Step 3: Create entry files**

`src/main.ts`, `src/App.vue`, `src/router/index.ts`

- [ ] **Step 4: Install and verify build**

```bash
cd frontend
npm install
npm run build
```

Expected: Build succeeds with no errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/
git commit -m "feat: add Vue3 frontend scaffold"
```

---

### Task 7: Frontend Chat UI

**Files:**
- Create: `frontend/src/types/chat.ts`
- Create: `frontend/src/api/chat.ts`
- Create: `frontend/src/stores/chat.ts`
- Create: `frontend/src/components/ChatMessage.vue`
- Create: `frontend/src/components/ChatInput.vue`
- Create: `frontend/src/components/IntentBadge.vue`
- Create: `frontend/src/components/Sidebar.vue`
- Create: `frontend/src/views/ChatView.vue`

- [ ] **Step 1: Create TypeScript types**

```typescript
export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  toolCalls?: ToolCall[];
}

export interface Intent {
  intent: string;
  confidence: number;
  target_agent: string;
}

export interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
  result?: string;
}
```

- [ ] **Step 2: Create API module**

Use `fetch` + `ReadableStream` to consume POST SSE:

```typescript
export async function streamChat(
  message: string,
  sessionId: string,
  onEvent: (event: { event: string; data: string }) => void
): Promise<void> {
  const response = await fetch('/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  const reader = response.body?.getReader();
  const decoder = new TextDecoder();
  // Parse SSE format and call onEvent for each event
}
```

- [ ] **Step 3: Create Pinia store**

Manage messages, streaming state, current session.

- [ ] **Step 4: Create UI components**

ChatMessage, ChatInput, IntentBadge, Sidebar

- [ ] **Step 5: Create ChatView**

Assemble all components.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/
git commit -m "feat: add frontend chat UI with SSE streaming"
```

---

### Task 8: Verification & Polish

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 2: Test frontend dev server**

```bash
cd frontend && npm run dev
```

Verify UI loads and can send messages (if backend is running).

- [ ] **Step 3: Update CLAUDE.md**

- Mark Phase 6 as complete.
- Add Vue3/Element Plus/Vite to tech stack.
- Update directory structure with `frontend/` and `api/sse.py`.
- Update environment variables with CORS/SSE settings.

- [ ] **Step 4: Update README.md**

- Add Phase 6 features（Vue3 前端、SSE 流式输出）。
- Update architecture diagram to show Frontend layer.
- Update test coverage list.
- Mark Phase 6 as complete in roadmap.

- [ ] **Step 5: Final commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: update README and CLAUDE for Phase 6"
```

---

## Self-Review

**1. Spec coverage:**
- SSE endpoint → Task 2
- Agent streaming → Task 3
- Graph streaming integration → Task 4
- CORS → Task 5
- Vue3 scaffold → Task 6
- Frontend chat UI → Task 7
- Documentation → Task 8

**2. Placeholder scan:**
- SSE event_generator starts as placeholder in Task 2, replaced in Task 4.
- Frontend build verification in Task 6.
- No other TBD/TODO found.

**3. Type consistency:**
- All `run_stream` methods return `AsyncIterator[str]` or `AsyncIterator[dict]`.
- SSE events use consistent `{event, data}` structure.
