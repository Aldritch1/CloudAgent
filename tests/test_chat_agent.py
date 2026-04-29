from unittest.mock import MagicMock, patch

from cloudagent.agent.chat_agent import ChatAgent


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
