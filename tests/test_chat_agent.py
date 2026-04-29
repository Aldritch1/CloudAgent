from unittest.mock import MagicMock, patch

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from cloudagent.agent.chat_agent import ChatAgent, ChatAgentError
import pytest


@patch("cloudagent.agent.chat_agent.ChatOpenAI")
def test_chat_agent_returns_response(mock_llm_class):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Hello there!")
    mock_llm_class.return_value = mock_llm

    agent = ChatAgent(model_name="gpt-test", api_key="test-key")
    messages = [{"role": "user", "content": "hi"}]
    response = agent.run(messages)

    assert response == "Hello there!"
    mock_llm.invoke.assert_called_once()
    call_args = mock_llm.invoke.call_args[0][0]
    assert len(call_args) == 2  # system + user message
    assert isinstance(call_args[0], SystemMessage)
    assert isinstance(call_args[1], HumanMessage)


@patch("cloudagent.agent.chat_agent.ChatOpenAI")
def test_chat_agent_converts_assistant_message(mock_llm_class):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Got it!")
    mock_llm_class.return_value = mock_llm

    agent = ChatAgent(model_name="gpt-test", api_key="test-key")
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello there!"},
    ]
    response = agent.run(messages)

    assert response == "Got it!"
    mock_llm.invoke.assert_called_once()
    call_args = mock_llm.invoke.call_args[0][0]
    assert len(call_args) == 3  # system + user + assistant message
    assert isinstance(call_args[0], SystemMessage)
    assert isinstance(call_args[1], HumanMessage)
    assert isinstance(call_args[2], AIMessage)


@patch("cloudagent.agent.chat_agent.ChatOpenAI")
def test_chat_agent_raises_domain_exception_on_llm_failure(mock_llm_class):
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("API down")
    mock_llm_class.return_value = mock_llm

    agent = ChatAgent(model_name="gpt-test", api_key="test-key")
    messages = [{"role": "user", "content": "hi"}]

    with pytest.raises(ChatAgentError, match="LLM invocation failed"):
        agent.run(messages)
