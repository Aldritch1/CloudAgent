import logging

from cloudagent.memory.redis_store import SessionStore

logger = logging.getLogger(__name__)


class TieredMemoryManager:
    """Coordinates hot (Redis), warm (PostgreSQL), and cold (Milvus) memory tiers."""

    def __init__(
        self,
        hot_store: SessionStore | None = None,
        warm_store=None,
        cold_store=None,
    ):
        self.hot_store = hot_store
        self.warm_store = warm_store
        self.cold_store = cold_store

    async def get_context(self, session_id: str, user_id: str) -> dict:
        messages = []
        if self.hot_store is not None:
            try:
                messages = self.hot_store.get_session(session_id)
            except Exception as e:
                logger.warning(f"Hot store failed: {e}")

        profile = None
        if self.warm_store is not None:
            try:
                profile = await self.warm_store.get_user_profile(user_id)
            except Exception as e:
                logger.warning(f"Warm store failed: {e}")

        memories = []
        if self.cold_store is not None:
            try:
                memories = await self.cold_store.search_memories(user_id, "", top_k=5)
            except Exception as e:
                logger.warning(f"Cold store failed: {e}")

        return {
            "messages": messages,
            "profile": profile or {},
            "memories": memories,
        }

    async def save_turn(self, session_id: str, user_id: str, messages: list[dict]) -> None:
        if self.hot_store is not None:
            try:
                self.hot_store.save_session(session_id, messages)
            except Exception as e:
                logger.warning(f"Hot store save failed: {e}")

        # Warm store: save summary every 5 turns (placeholder for now)
        if self.warm_store is not None and len(messages) % 5 == 0:
            try:
                summary = f"Session {session_id} has {len(messages)} messages"
                await self.warm_store.save_summary(session_id, user_id, summary)
            except Exception as e:
                logger.warning(f"Warm store summary save failed: {e}")
