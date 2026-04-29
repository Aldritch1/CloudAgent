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
