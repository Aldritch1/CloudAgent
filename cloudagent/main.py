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

# Dependencies initialized lazily to allow test-time patching
_session_store = None
_entry_agent = None
_chat_agent = None


def _get_session_store():
    global _session_store
    if _session_store is None:
        _session_store = SessionStore(str(settings.redis_url))
    return _session_store


def _get_entry_agent():
    global _entry_agent
    if _entry_agent is None:
        _entry_agent = EntryAgent(
            model_name=settings.model_name,
            api_key=settings.openai_api_key.get_secret_value(),
        )
    return _entry_agent


def _get_chat_agent():
    global _chat_agent
    if _chat_agent is None:
        _chat_agent = ChatAgent(
            model_name=settings.model_name,
            api_key=settings.openai_api_key.get_secret_value(),
        )
    return _chat_agent


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        session_store = _get_session_store()
        entry_agent = _get_entry_agent()
        chat_agent = _get_chat_agent()

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
    except HTTPException as exc:
        if exc.status_code == 500:
            return JSONResponse(
                status_code=500,
                content={"error": "服务暂时繁忙，请稍后重试"},
            )
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "服务暂时繁忙，请稍后重试"},
        )
