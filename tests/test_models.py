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
