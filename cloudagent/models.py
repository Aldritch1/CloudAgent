from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Unique session identifier")
    message: str = Field(..., min_length=1, description="User message")


class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant response")
    intent: str = Field(..., description="Detected intent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Intent confidence")
