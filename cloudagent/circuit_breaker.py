from pybreaker import CircuitBreaker


class LLMCircuitBreaker:
    """Circuit breaker for LLM calls using pybreaker."""

    def __init__(self, fail_max: int = 5, reset_timeout: int = 60):
        self._breaker = CircuitBreaker(fail_max=fail_max, reset_timeout=reset_timeout)

    def wrap_sync(self, func):
        return self._breaker(func)

    def wrap_async(self, func):
        return self._breaker(func)


class CircuitBreakerChatOpenAI:
    """Proxy around ChatOpenAI that adds circuit breaker protection."""

    def __init__(self, chat_openai, breaker: LLMCircuitBreaker):
        self._chat = chat_openai
        self._breaker = breaker

    def invoke(self, messages):
        wrapped = self._breaker.wrap_sync(self._chat.invoke)
        return wrapped(messages)

    async def ainvoke(self, messages):
        wrapped = self._breaker.wrap_async(self._chat.ainvoke)
        return await wrapped(messages)
