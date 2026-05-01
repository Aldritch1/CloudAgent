from unittest.mock import AsyncMock, MagicMock

import pytest
from pybreaker import CircuitBreakerError

from cloudagent.circuit_breaker import CircuitBreakerChatOpenAI, LLMCircuitBreaker


def test_circuit_closed_allows_calls():
    breaker = LLMCircuitBreaker(fail_max=5, reset_timeout=60)
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(return_value="ok")

    wrapped = CircuitBreakerChatOpenAI(mock_llm, breaker)
    result = wrapped.invoke("msg")

    assert result == "ok"
    mock_llm.invoke.assert_called_once()


def test_circuit_opens_after_failures():
    breaker = LLMCircuitBreaker(fail_max=2, reset_timeout=60)
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(side_effect=Exception("LLM down"))

    wrapped = CircuitBreakerChatOpenAI(mock_llm, breaker)

    # First failure
    with pytest.raises(Exception):
        wrapped.invoke("msg")
    # Second failure — circuit should OPEN now
    with pytest.raises(Exception):
        wrapped.invoke("msg")
    # Third call — fast fail with CircuitBreakerError
    with pytest.raises(CircuitBreakerError):
        wrapped.invoke("msg")


def test_circuit_fast_fails_when_open():
    breaker = LLMCircuitBreaker(fail_max=1, reset_timeout=60)
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(side_effect=Exception("fail"))

    wrapped = CircuitBreakerChatOpenAI(mock_llm, breaker)

    # Open the circuit
    with pytest.raises(Exception):
        wrapped.invoke("msg")

    # Fast fail
    with pytest.raises(CircuitBreakerError):
        wrapped.invoke("msg")


@pytest.mark.asyncio
async def test_async_circuit_closed_allows_calls():
    breaker = LLMCircuitBreaker(fail_max=5, reset_timeout=60)
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value="async ok")

    wrapped = CircuitBreakerChatOpenAI(mock_llm, breaker)
    result = await wrapped.ainvoke("msg")

    assert result == "async ok"


@pytest.mark.asyncio
async def test_async_circuit_opens_after_failures():
    breaker = LLMCircuitBreaker(fail_max=2, reset_timeout=60)
    mock_llm = MagicMock()
    mock_llm.ainvoke = MagicMock(side_effect=Exception("LLM down"))

    wrapped = CircuitBreakerChatOpenAI(mock_llm, breaker)

    with pytest.raises(Exception):
        await wrapped.ainvoke("msg")
    with pytest.raises(Exception):
        await wrapped.ainvoke("msg")
    with pytest.raises(CircuitBreakerError):
        await wrapped.ainvoke("msg")
