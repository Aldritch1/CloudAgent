import logging

import asyncpg

from cloudagent.tenant_context import get_tenant_id

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
                    "SELECT * FROM user_profiles WHERE tenant_id = $1 AND user_id = $2",
                    get_tenant_id(),
                    user_id,
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
                    INSERT INTO user_profiles (tenant_id, user_id, preferences, updated_at)
                    VALUES ($1, $2, $3, NOW())
                    ON CONFLICT (tenant_id, user_id) DO UPDATE
                    SET preferences = $3, updated_at = NOW()
                    """,
                    get_tenant_id(),
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
                    "SELECT * FROM session_summaries WHERE tenant_id = $1 AND user_id = $2 ORDER BY created_at DESC LIMIT $3",
                    get_tenant_id(),
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
                    INSERT INTO session_summaries (tenant_id, session_id, user_id, summary, created_at)
                    VALUES ($1, $2, $3, $4, NOW())
                    ON CONFLICT (tenant_id, session_id) DO UPDATE
                    SET summary = $4, created_at = NOW()
                    """,
                    get_tenant_id(),
                    session_id,
                    user_id,
                    summary,
                )
            finally:
                await conn.close()
        except Exception as e:
            logger.warning(f"Warm store save_summary failed: {e}")
