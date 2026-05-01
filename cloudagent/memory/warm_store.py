import logging

import asyncpg

logger = logging.getLogger(__name__)


class WarmStore:
    """PostgreSQL warm storage for session summaries and user profiles."""

    def __init__(self, dsn: str):
        self._dsn = dsn

    async def _connect(self):
        return await asyncpg.connect(self._dsn)

    async def get_user_profile(self, user_id: str) -> dict | None:
        try:
            conn = await self._connect()
            try:
                row = await conn.fetchrow(
                    "SELECT * FROM user_profiles WHERE user_id = $1", user_id
                )
                return dict(row) if row else None
            finally:
                await conn.close()
        except Exception as e:
            logger.warning(f"Warm store get_user_profile failed: {e}")
            return None

    async def save_user_profile(self, user_id: str, profile: dict) -> None:
        try:
            conn = await self._connect()
            try:
                await conn.execute(
                    """
                    INSERT INTO user_profiles (user_id, preferences, updated_at)
                    VALUES ($1, $2, NOW())
                    ON CONFLICT (user_id) DO UPDATE
                    SET preferences = $2, updated_at = NOW()
                    """,
                    user_id,
                    profile,
                )
            finally:
                await conn.close()
        except Exception as e:
            logger.warning(f"Warm store save_user_profile failed: {e}")

    async def get_session_history(self, user_id: str, limit: int = 10) -> list[dict]:
        try:
            conn = await self._connect()
            try:
                rows = await conn.fetch(
                    "SELECT * FROM session_summaries WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2",
                    user_id,
                    limit,
                )
                return [dict(row) for row in rows]
            finally:
                await conn.close()
        except Exception as e:
            logger.warning(f"Warm store get_session_history failed: {e}")
            return []

    async def save_summary(self, session_id: str, user_id: str, summary: str) -> None:
        try:
            conn = await self._connect()
            try:
                await conn.execute(
                    """
                    INSERT INTO session_summaries (session_id, user_id, summary, created_at)
                    VALUES ($1, $2, $3, NOW())
                    ON CONFLICT (session_id) DO UPDATE
                    SET summary = $3, created_at = NOW()
                    """,
                    session_id,
                    user_id,
                    summary,
                )
            finally:
                await conn.close()
        except Exception as e:
            logger.warning(f"Warm store save_summary failed: {e}")
