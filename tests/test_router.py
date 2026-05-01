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


@patch("cloudagent.agent.router.ChatOpenAI")
def test_entry_agent_recognizes_faq_intent(mock_llm_class):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content=json.dumps({
            "intent": "faq",
            "confidence": 0.92,
            "target_agent": "faq",
        })
    )
    mock_llm_class.return_value = mock_llm

    agent = EntryAgent(model_name="gpt-test", api_key="test-key")
    state = {
        "messages": [{"role": "user", "content": "怎么退款？"}],
        "intent": None,
        "confidence": 0.0,
        "target_agent": None,
        "context": {},
    }
    result = agent.run(state)

    assert result["intent"] == "faq"
    assert result["confidence"] == 0.92
    assert result["target_agent"] == "faq"


@patch("cloudagent.agent.router.ChatOpenAI")
def test_entry_agent_recognizes_workflow_intent(mock_llm_class):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content=json.dumps({
            "intent": "workflow",
            "confidence": 0.88,
            "target_agent": "workflow",
        })
    )
    mock_llm_class.return_value = mock_llm

    agent = EntryAgent(model_name="gpt-test", api_key="test-key")
    state = {
        "messages": [{"role": "user", "content": "帮我查一下订单"}],
        "intent": None,
        "confidence": 0.0,
        "target_agent": None,
        "context": {},
    }
    result = agent.run(state)

    assert result["intent"] == "workflow"
    assert result["confidence"] == 0.88
    assert result["target_agent"] == "workflow"


@patch("cloudagent.agent.router.ChatOpenAI")
def test_entry_agent_low_confidence_fallback_to_chat(mock_llm_class):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content=json.dumps({
            "intent": "faq",
            "confidence": 0.3,
            "target_agent": "faq",
        })
    )
    mock_llm_class.return_value = mock_llm

    agent = EntryAgent(model_name="gpt-test", api_key="test-key")
    state = {
        "messages": [{"role": "user", "content": "随便说点啥"}],
        "intent": None,
        "confidence": 0.0,
        "target_agent": None,
        "context": {},
    }
    result = agent.run(state)

    assert result["target_agent"] == "chat"
