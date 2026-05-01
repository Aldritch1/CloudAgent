import logging

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from cloudagent.auth import get_current_user
from cloudagent.cache import QueryCache
from cloudagent.config import settings
from cloudagent.graph import build_graph
from cloudagent.hitl import HITLManager
from cloudagent.memory.manager import TieredMemoryManager
from cloudagent.memory.redis_store import SessionStore
from cloudagent.models import ChatRequest, ChatResponse
from cloudagent.rate_limit import RateLimiter
from cloudagent.agent.router import EntryAgent
from cloudagent.agent.chat_agent import ChatAgent
from cloudagent.retrieval.vector import VectorRetriever
from cloudagent.retrieval.graph import GraphRetriever
from cloudagent.retrieval.keyword import KeywordRetriever
from cloudagent.retrieval.hybrid import HybridRetriever
from cloudagent.agent.rag_agent import RAGAgent

logger = logging.getLogger(__name__)

app = FastAPI(title="CloudAgent", version="0.1.0")

# Initialize dependencies
session_store = SessionStore(str(settings.redis_url))
entry_agent = EntryAgent(
    model_name=settings.model_name,
    api_key=settings.openai_api_key.get_secret_value(),
)
chat_agent = ChatAgent(
    model_name=settings.model_name,
    api_key=settings.openai_api_key.get_secret_value(),
)
vector_retriever = VectorRetriever(
    uri=settings.milvus_uri,
    api_key=settings.openai_api_key.get_secret_value(),
)
graph_retriever = GraphRetriever(
    uri=settings.neo4j_uri,
    user=settings.neo4j_user,
    password=settings.neo4j_password.get_secret_value(),
)
keyword_retriever = KeywordRetriever(dsn=settings.database_url)
hybrid_retriever = HybridRetriever(vector_retriever, graph_retriever, keyword_retriever)

rag_agent = RAGAgent(
    model_name=settings.model_name,
    api_key=settings.openai_api_key.get_secret_value(),
    retriever=hybrid_retriever,
)

memory_manager = TieredMemoryManager(
    hot_store=None,  # main.py handles hot store saving after graph execution
    warm_store=None,
    cold_store=None,
)

cache = QueryCache(
    redis_client=session_store._redis if not session_store._use_fallback else None,
)

rate_limiter = RateLimiter(
    redis_client=session_store._redis if not session_store._use_fallback else None,
    requests_per_minute=settings.rate_limit_requests_per_minute,
)

hitl_manager = HITLManager()

graph = build_graph(
    entry_agent=entry_agent,
    chat_agent=chat_agent,
    rag_agent=rag_agent,
    memory_manager=memory_manager,
    cache=cache,
    hitl=hitl_manager,
)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user_id: str = Depends(get_current_user)):
    try:
        # Rate limiting
        if not rate_limiter.check(user_id):
            raise HTTPException(
                status_code=429,
                detail="请求过于频繁，请稍后再试",
                headers={
                    "X-RateLimit-Limit": str(settings.rate_limit_requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                },
            )

        # Load existing messages from hot store
        messages = session_store.get_session(request.session_id)
        messages.append({"role": "user", "content": request.message})

        # Build initial state
        state = {
            "messages": messages,
            "user_id": user_id,
            "session_id": request.session_id,
            "intent": None,
            "confidence": 0.0,
            "target_agent": None,
            "context": {},
            "last_message": request.message,
        }

        config = {"configurable": {"thread_id": request.session_id}}

        # Check cache first (skip for workflow intents and HITL resume)
        if not request.action:
            cached = await cache.get(request.message)
            if cached:
                messages.append({"role": "assistant", "content": cached["answer"]})
                session_store.save_session(request.session_id, messages)
                return ChatResponse(
                    response=cached["answer"],
                    intent=cached["intent"],
                    confidence=cached["confidence"],
                )

        # Invoke graph
        result = await graph.ainvoke(state, config=config)

        # If interrupted (HITL), return confirmation request
        if result.get("action_required") == "confirm":
            return ChatResponse(
                response=result["response"],
                intent=result.get("intent", "workflow"),
                confidence=result.get("confidence", 1.0),
                action_required="confirm",
            )

        # If clarification needed
        if result.get("action_required") == "clarify":
            return ChatResponse(
                response=result["response"],
                intent=result.get("intent", "chat"),
                confidence=result.get("confidence", 0.0),
                action_required="clarify",
            )

        response_text = result.get("response", "")

        # Save to cache
        intent = result.get("intent", "chat")
        confidence = result.get("confidence", 0.0)
        if intent not in ("workflow",) and not request.action:
            await cache.set(request.message, response_text, intent, confidence)

        # Persist session (graph save_memory_node already saved without assistant; add it here)
        messages.append({"role": "assistant", "content": response_text})
        session_store.save_session(request.session_id, messages)

        return ChatResponse(
            response=response_text,
            intent=intent,
            confidence=confidence,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="服务暂时繁忙，请稍后重试")


@app.get("/metrics")
async def metrics():
    return {"status": "placeholder"}
