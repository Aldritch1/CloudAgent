import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from cloudagent.auth import get_current_user
from cloudagent.tenant import tenant_dependency
from cloudagent.models import ChatRequest
from cloudagent.state import AgentState
from cloudagent.graph import GraphNodes

logger = logging.getLogger(__name__)
router = APIRouter()


async def event_generator(request: ChatRequest, user_id: str, tenant_id: str) -> AsyncIterator[dict]:
    from cloudagent import main as main_module

    state = AgentState(
        messages=[{"role": "user", "content": request.message}],
        user_id=user_id,
        session_id=request.session_id,
        last_message=request.message,
        tenant_id=tenant_id,
    )

    nodes = GraphNodes(
        entry_agent=main_module.entry_agent,
        chat_agent=main_module.chat_agent,
        rag_agent=main_module.rag_agent,
        workflow_agent=getattr(main_module, "workflow_agent", None),
        memory_manager=getattr(main_module, "memory_manager", None),
        cache=getattr(main_module, "cache", None),
        hitl=getattr(main_module, "hitl_manager", None),
    )

    state = nodes.entry_node(state)

    yield {"event": "intent", "data": json.dumps({
        "intent": state.get("intent", "chat"),
        "confidence": state.get("confidence", 0.0),
        "target_agent": state.get("target_agent", "chat"),
    })}

    async for event in nodes.stream_node(state):
        yield event

    try:
        await nodes.save_memory_node(state)
    except Exception as e:
        logger.warning(f"Save memory failed: {e}")


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    user_id: str = Depends(get_current_user),
    tenant_id: str = Depends(tenant_dependency),
):
    return EventSourceResponse(event_generator(request, user_id, tenant_id))
