# CloudAgent 阶段6设计文档 — 前端 + SSE 流式输出

**版本：** v1.0
**日期：** 2026-05-02
**目标：** 构建 Vue3 前端聊天界面，接入 SSE 流式输出，实现对话可视化与实时监控

---

## 1. 总体架构

阶段6在阶段5的 MCP 工具生态基础上，增加 **前端层** 和 **SSE 流式输出层**，使用户能够通过 Web UI 与 CloudAgent 交互，并实时看到 LLM 的流式生成结果。

```
┌─────────────────────────────────────────┐
│  Frontend: Vue3 + Vite + Element Plus   │
│  职责：聊天界面、历史会话、意图可视化       │
├─────────────────────────────────────────┤
│  SSE Gateway: FastAPI SSE Endpoint       │
│  职责：/chat/stream 端点，流式推送 token   │
├─────────────────────────────────────────┤
│  API网关层：FastAPI + JWT + 限流 + 多租户  │
├─────────────────────────────────────────┤
│  Agent引擎层：LangGraph StateGraph        │
│  职责：entry → route → chat / rag /       │
│       workflow(MCP tools) / clarify / HITL│
├─────────────────────────────────────────┤
│  MCP Client + Servers                    │
├─────────────────────────────────────────┤
│  Data Layer                              │
└─────────────────────────────────────────┘
```

---

## 2. 基础设施

### 2.1 依赖扩展

`pyproject.toml` 新增：

```toml
dependencies = [
    # ... 现有依赖 ...
    "sse-starlette>=1.6.1",
]
```

> `sse-starlette` 已随 `mcp` 包安装，显式声明以解除对 `mcp` 的隐式依赖。

前端依赖（`frontend/package.json`）：

```json
{
  "dependencies": {
    "vue": "^3.4.0",
    "vue-router": "^4.3.0",
    "element-plus": "^2.7.0",
    "@element-plus/icons-vue": "^2.3.0",
    "axios": "^1.6.0"
  },
  "devDependencies": {
    "vite": "^5.2.0",
    "@vitejs/plugin-vue": "^5.0.0",
    "typescript": "^5.4.0"
  }
}
```

### 2.2 配置扩展

`cloudagent/config.py` 新增：

```python
class Settings(BaseSettings):
    # ... 现有配置 ...
    enable_sse: bool = True
    cors_origins: str = "*"  # 生产环境应限制为前端域名
```

---

## 3. 文件结构

```
cloudagent/
├── main.py                  # MODIFIED: 添加 SSE endpoint + CORS
├── config.py                # MODIFIED: enable_sse, cors_origins
├── api/
│   ├── __init__.py
│   └── sse.py               # NEW: SSE streaming endpoint
└── ...

frontend/                    # NEW: Vue3 前端项目
├── index.html
├── package.json
├── vite.config.ts
├── tsconfig.json
├── src/
│   ├── main.ts
│   ├── App.vue
│   ├── router/
│   │   └── index.ts
│   ├── views/
│   │   └── ChatView.vue     # 主聊天界面
│   ├── components/
│   │   ├── ChatMessage.vue  # 单条消息组件
│   │   ├── ChatInput.vue    # 输入框组件
│   │   ├── IntentBadge.vue  # 意图标签
│   │   └── Sidebar.vue      # 会话侧边栏
│   ├── api/
│   │   └── chat.ts          # API 调用封装
│   ├── stores/
│   │   └── chat.ts          # Pinia store
│   └── types/
│       └── chat.ts          # TypeScript 类型定义
└── public/

tests/
├── test_sse.py              # NEW: SSE endpoint 测试
└── test_main.py             # MODIFIED: CORS/SSE 测试
```

---

## 4. 模块设计

### 4.1 SSE Endpoint (`api/sse.py`)

FastAPI 端点使用 `EventSourceResponse` 返回 SSE 流：

```python
from sse_starlette.sse import EventSourceResponse
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter()

async def chat_stream_generator(request: ChatRequest, user_id: str, tenant_id: str):
    """Yields SSE events: intent, token, tool_call, done, error"""
    # 1. Send intent event
    yield {"event": "intent", "data": json.dumps({"intent": "workflow", "confidence": 0.92})}

    # 2. For streaming LLM responses, yield tokens one by one
    # 3. For tool calls, yield tool_call event
    # 4. Finally yield done event

@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    user_id: str = Depends(get_current_user),
    tenant_id: str = Depends(tenant_dependency),
):
    return EventSourceResponse(chat_stream_generator(request, user_id, tenant_id))
```

**SSE Event 类型：**

| Event | 说明 |
|-------|------|
| `intent` | 意图识别结果（intent, confidence, target_agent） |
| `token` | LLM 生成的单个 token（流式输出） |
| `tool_call` | Tool 调用信息（tool_name, arguments） |
| `tool_result` | Tool 执行结果 |
| `hitl` | HITL 中断，需要用户确认 |
| `done` | 流结束，包含完整 response |
| `error` | 错误信息 |

### 4.2 ChatAgent 流式输出改造

当前 `ChatAgent.run()` 返回完整字符串。为支持 SSE，增加流式模式：

```python
class ChatAgent:
    async def run_stream(self, messages: list) -> AsyncIterator[str]:
        """Yield tokens as they are generated."""
        response = await self._llm.astream(messages)
        async for chunk in response:
            yield chunk.content
```

同理 `RAGAgent` 和 `WorkflowAgent` 也增加 `run_stream` 方法。

### 4.3 Graph 流式节点

`graph.py` 增加流式执行路径：

```python
async def stream_chat_node(self, state: AgentState) -> AsyncIterator[dict]:
    # 根据 target_agent 选择流式 agent
    if state.get("target_agent") == "chat":
        async for token in self.chat_agent.run_stream(state.get("messages", [])):
            yield {"event": "token", "data": token}
    # ... rag, workflow 同理
```

### 4.4 Vue3 前端

**ChatView.vue** — 主聊天界面：

```vue
<template>
  <div class="chat-container">
    <Sidebar :sessions="sessions" @select="loadSession" />
    <div class="chat-main">
      <div class="messages">
        <ChatMessage
          v-for="msg in messages"
          :key="msg.id"
          :message="msg"
        />
      </div>
      <ChatInput @send="sendMessage" :disabled="streaming" />
      <IntentBadge v-if="currentIntent" :intent="currentIntent" />
    </div>
  </div>
</template>
```

**API 封装 (`api/chat.ts`)：**

```typescript
export function streamChat(message: string, sessionId: string, onEvent: (event: SSEEvent) => void) {
  const eventSource = new EventSource(`/api/chat/stream?session_id=${sessionId}&message=${encodeURIComponent(message)}`);
  // 或使用 fetch + ReadableStream 处理 POST SSE
}
```

> 由于 SSE 规范不支持 POST body，实际使用 `fetch` + `ReadableStream` 解析 SSE 格式，或改用 GET 传递参数。

**Pinia Store (`stores/chat.ts`)：**

```typescript
export const useChatStore = defineStore('chat', {
  state: () => ({
    messages: [] as Message[],
    streaming: false,
    currentIntent: null as Intent | null,
  }),
  actions: {
    async sendMessage(content: string) {
      this.streaming = true;
      await streamChat(content, this.sessionId, (event) => {
        if (event.event === 'token') {
          this.appendToken(event.data);
        } else if (event.event === 'intent') {
          this.currentIntent = JSON.parse(event.data);
        } else if (event.event === 'done') {
          this.streaming = false;
        }
      });
    },
  },
});
```

### 4.5 CORS 配置

`main.py` 添加 CORS 中间件：

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

---

## 5. 数据流

```
用户在前端输入: "帮我查一下订单 12345"

  1. 前端通过 fetch + ReadableStream 调用 POST /chat/stream
  2. SSE endpoint:
     2a. yield {"event": "intent", "data": {"intent": "workflow", "confidence": 0.92}}
     2b. EntryAgent 识别 intent
     2c. route_node 决定 target = "workflow"
  3. workflow_stream_node:
     3a. WorkflowAgent 调用 LLM
     3b. yield {"event": "token", "data": "正在"} ... {"event": "token", "data": "查询"}
     3c. LLM 输出 tool call JSON
     3d. yield {"event": "tool_call", "data": {"tool": "order:query_order", "args": {...}}}
     3e. MCPClient 调用 tool
     3f. yield {"event": "tool_result", "data": "订单 12345 已发货"}
     3g. WorkflowAgent 汇总结果，流式输出 token
  4. yield {"event": "done", "data": {"response": "...", "intent": "workflow"}}
  5. 前端组装消息，显示意图标签和工具调用卡片
```

---

## 6. 错误处理

| 场景 | 处理策略 |
|------|----------|
| SSE 连接断开 | 前端自动重连（指数退避） |
| LLM 流式生成失败 | yield error event，前端显示错误提示 |
| 用户发送新消息时正在流式输出 | 中断当前流，开始新请求 |
| CORS 预检失败 | 检查 `cors_origins` 配置 |

---

## 7. 测试策略

| 测试类型 | 覆盖内容 | 工具/方法 |
|----------|----------|-----------|
| 单元测试 | SSE event 生成器 | pytest + AsyncMock |
| 单元测试 | ChatAgent.run_stream token 产量 | mock LLM astream |
| API 测试 | `/chat/stream` 返回 SSE 格式 | TestClient + 手动解析 SSE |
| API 测试 | CORS 预检响应 | TestClient OPTIONS 请求 |
| 前端测试 | ChatMessage 组件渲染 | Vitest + Vue Test Utils |
| 集成测试 | 端到端流式对话 | Playwright / Cypress |

---

## 8. 阶段6明确边界（不做）

- 用户认证界面（登录/注册）— 复用现有 JWT，前端只做 token 输入
- 管理后台（知识库管理、指标看板）— 阶段7
- 多语言国际化 — 阶段7
- 移动端适配优化 — 阶段7
- WebSocket 双向通信 — 当前用 SSE 单向足够

---

## 9. 成功标准

1. `pytest tests/ -v` 全部通过（含阶段1~6测试）。
2. 前端能发起对话，实时看到 LLM token 流式输出。
3. Workflow 意图触发时，前端显示 tool call 和 tool result 卡片。
4. HITL 中断时，前端显示确认/取消按钮。
5. CORS 配置正确，前端与后端跨域通信正常。
