# CloudAgent 阶段1设计文档 — 核心API骨架

**版本：** v1.0  
**日期：** 2026-04-29  
**目标：** 构建可对话的最小可用版本：FastAPI + 入口Agent + Chat Agent + Redis会话存储

---

## 1. 总体架构

阶段1只实现四层架构中的核心两层：

```
┌─────────────────────────────────────────┐
│ API网关层：FastAPI                      │
│ 职责：提供 /chat 端点，请求校验，响应封装│
├─────────────────────────────────────────┤
│ Agent引擎层：LangGraph + LangChain      │
│ 职责：入口Agent（意图识别+路由）          │
│       Chat Agent（LLM直连）              │
├─────────────────────────────────────────┤
│ 数据层：Redis（热数据，TTL 1小时）       │
│ 职责：当前会话上下文存储                  │
└─────────────────────────────────────────┘
```

- 前端层、Nginx网关、Milvus/Neo4j/PG 数据层在阶段1暂不实现。
- 入口Agent只识别一种意图 `chat`，但保留完整的置信度路由逻辑，为阶段2-3预留扩展位。

---

## 2. 核心组件

### 2.1 文件结构

```
cloudagent/
├── main.py                  # FastAPI 入口
├── config.py                # 配置管理
├── models.py                # Pydantic 模型
├── agent/
│   ├── __init__.py
│   ├── router.py            # 入口Agent（意图识别 + 路由）
│   └── chat_agent.py        # Chat Agent
└── memory/
    ├── __init__.py
    └── redis_store.py       # Redis 会话存储
```

### 2.2 各组件职责

| 组件 | 职责 |
|------|------|
| `main.py` | 启动 FastAPI，注册 `POST /chat` 端点，协调请求 → Agent → 存储 → 响应 |
| `models.py` | `ChatRequest(session_id, message)`、`ChatResponse(response, intent, confidence)` |
| `config.py` | 从环境变量加载 `OPENAI_API_KEY`、`REDIS_URL`、`MODEL_NAME` 等 |
| `agent/router.py` | LangGraph StateGraph。节点：意图识别（LLM调用）→ 路由决策。状态：`IntentState` |
| `agent/chat_agent.py` | 接收消息历史，拼接 system prompt，调用 LLM 生成回复 |
| `memory/redis_store.py` | `get_session(session_id)` / `save_session(session_id, messages)`。key 格式：`session:{id}`，TTL 3600s |

### 2.3 入口Agent状态机

```python
class IntentState(TypedDict):
    messages: list[AnyMessage]   # 完整对话历史
    intent: str | None           # 识别的意图
    confidence: float            # 置信度 0.0~1.0
    target_agent: str | None     # 路由目标
    context: dict                # 携带的上下文
```

**路由逻辑：**
- `confidence > 0.8`：路由到 `target_agent`（阶段1只有 `chat`）
- `0.5 < confidence <= 0.8`：本阶段简化处理，直接路由（追问澄清在阶段3实现）
- `confidence <= 0.5`：兜底到 `chat`

**意图识别Prompt：** 要求LLM输出JSON `{intent, confidence, target_agent}`，confidence基于用户对"闲聊/寒暄/打招呼"类问题的判断。

---

## 3. 数据流

```
POST /chat
{
  "session_id": "uuid",
  "message": "你好"
}

  1. main.py 解析请求，校验模型
  2. redis_store.get_session(session_id) → 加载历史消息（若无则空列表）
  3. 将用户消息追加到历史
  4. 入口Agent处理：
     - intent_node: 调用LLM识别意图 → 填充 intent, confidence, target_agent
     - route_node: 根据 confidence 决定下一个节点
  5. chat_agent节点: 拼接 system prompt + 历史 → LLM → 生成 assistant 回复
  6. 将 assistant 回复追加到历史
  7. redis_store.save_session(session_id, 历史) → 更新Redis，刷新TTL
  8. 返回 ChatResponse
```

---

## 4. 接口定义

### 4.1 POST /chat

**Request:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "你好，请问你们支持退款吗？"
}
```

**Response (200):**
```json
{
  "response": "你好！我们支持7天无理由退款，请问需要了解具体的退款流程吗？",
  "intent": "chat",
  "confidence": 0.92
}
```

**Response (422):** 请求参数校验失败
**Response (500):** LLM调用失败（返回结构化错误，前端友好提示）

---

## 5. 错误处理

| 场景 | 处理策略 |
|------|----------|
| 意图识别LLM调用失败/超时 | 记录error日志，confidence设为0.0，兜底到Chat Agent，用户无感知 |
| Chat Agent LLM调用失败 | 返回HTTP 500，`{"error": "服务暂时繁忙，请稍后重试"}` |
| Redis连接失败 | 降级为内存字典存储（`dict[session_id, messages]`），打印warning日志，服务不中断 |
| Redis读写失败 | 同上降级策略 |

---

## 6. 测试策略

| 测试类型 | 覆盖内容 |
|----------|----------|
| 单元测试 | 入口Agent路由逻辑：mock LLM返回高/中/低confidence，验证路由目标 |
| 单元测试 | Chat Agent：mock LLM，验证system prompt拼接正确 |
| 集成测试 | Redis存储：验证读写、TTL过期、序列化/反序列化 |
| API测试 | `/chat` 端点：正常流程、空session、超长message、LLM失败降级 |

---

## 7. 阶段1不做的内容（明确边界）

- 前端界面（阶段6）
- Nginx网关（阶段1直接暴露FastAPI，后续再加）
- Milvus/Neo4j/PostgreSQL（阶段2-4）
- 多Agent并行（阶段3）
- MCP工具调用（阶段5）
- 意图识别的追问澄清（confidence 0.5~0.8区间，阶段3完善）
- 用户画像/长期记忆（阶段4）
- SSE流式输出（阶段6前端对接时添加）

---

## 8. 成功标准

1. `curl /chat` 能进行多轮对话，Redis中能看到session历史
2. 入口Agent的意图识别能区分`chat`意图并给出合理confidence
3. Redis故障时服务不中断（降级内存存储）
4. 所有单元测试和API测试通过
