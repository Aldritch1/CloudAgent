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
    """Placeholder generator — yields a simple done event."""
    yield {"event": "done", "data": json.dumps({"response": "SSE streaming placeholder"})}

@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    user_id: str = Depends(get_current_user),
    tenant_id: str = Depends(tenant_dependency),
):
    return EventSourceResponse(event_generator(request, user_id, tenant_id))
