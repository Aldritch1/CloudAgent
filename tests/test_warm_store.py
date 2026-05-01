from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cloudagent.memory.warm_store import WarmStore


@pytest.mark.asyncio
async def test_save_and_get_user_profile():
    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = {"user_id": "u1", "preferences": '{"lang": "zh"}'}

    with patch("cloudagent.memory.warm_store.asyncpg.connect", return_value=mock_conn):
        store = WarmStore("postgresql://test")
        await store.save_user_profile("u1", {"lang": "zh"})
        result = await store.get_user_profile("u1")

    assert result == {"user_id": "u1", "preferences": '{"lang": "zh"}'}
    mock_conn.execute.assert_called_once()
    mock_conn.fetchrow.assert_called_once()
    mock_conn.close.assert_called()


@pytest.mark.asyncio
async def test_get_session_history():
    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = [
        {"session_id": "s1", "summary": "关于退款的对话"},
    ]

    with patch("cloudagent.memory.warm_store.asyncpg.connect", return_value=mock_conn):
        store = WarmStore("postgresql://test")
        result = await store.get_session_history("u1", limit=5)

    assert len(result) == 1
    assert result[0]["session_id"] == "s1"


@pytest.mark.asyncio
async def test_degrades_on_pg_failure():
    with patch("cloudagent.memory.warm_store.asyncpg.connect", side_effect=Exception("PG down")):
        store = WarmStore("postgresql://test")
        result = await store.get_user_profile("u1")
        assert result is None

        history = await store.get_session_history("u1")
        assert history == []
