from unittest.mock import AsyncMock, MagicMock

import pytest

from cloudagent.memory.manager import TieredMemoryManager


@pytest.mark.asyncio
async def test_get_context_aggregates_all_tiers():
    hot = MagicMock()
    hot.get_session.return_value = [{"role": "user", "content": "hi"}]

    warm = AsyncMock()
    warm.get_user_profile.return_value = {"lang": "zh"}

    cold = AsyncMock()
    cold.search_memories.return_value = ["用户喜欢红色"]

    manager = TieredMemoryManager(hot, warm, cold)
    ctx = await manager.get_context("s1", "u1")

    assert ctx["messages"] == [{"role": "user", "content": "hi"}]
    assert ctx["profile"] == {"lang": "zh"}
    assert ctx["memories"] == ["用户喜欢红色"]


@pytest.mark.asyncio
async def test_save_turn_writes_hot():
    hot = MagicMock()
    manager = TieredMemoryManager(hot, None, None)
    await manager.save_turn("s1", "u1", [{"role": "user", "content": "hi"}])
    hot.save_session.assert_called_once_with("s1", [{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_fallback_when_warm_cold_fail():
    hot = MagicMock()
    hot.get_session.return_value = []

    warm = AsyncMock()
    warm.get_user_profile.side_effect = Exception("PG down")

    cold = AsyncMock()
    cold.search_memories.side_effect = Exception("Milvus down")

    manager = TieredMemoryManager(hot, warm, cold)
    ctx = await manager.get_context("s1", "u1")

    assert ctx["messages"] == []
    assert ctx["profile"] == {}
    assert ctx["memories"] == []
