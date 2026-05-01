import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from cloudagent.config import settings
from cloudagent.models import ChatRequest, ChatResponse
from cloudagent.memory.redis_store import SessionStore
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


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Load session history
        messages = session_store.get_session(request.session_id)

        # Append user message
        messages.append({"role": "user", "content": request.message})

        # Run entry agent (intent recognition + routing)
        state = {
            "messages": messages,
            "intent": None,
            "confidence": 0.0,
            "target_agent": None,
            "context": {},
        }
        state = entry_agent.run(state)

        # Phase2: route by target_agent
        target = state["target_agent"]
        try:
            if target == "faq":
                response_text = await rag_agent.run(state)
            elif target == "workflow":
                response_text = "业务办理功能正在开发中，请稍后再试。"
            else:
                response_text = chat_agent.run(state["messages"])
        except Exception as e:
            logger.error(f"Agent failed: {e}")
            raise HTTPException(status_code=500, detail="服务暂时繁忙，请稍后重试")

        # Append assistant message
        messages.append({"role": "assistant", "content": response_text})
        state["messages"] = messages

        # Save session
        session_store.save_session(request.session_id, messages)

        return ChatResponse(
            response=response_text,
            intent=state["intent"],
            confidence=state["confidence"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="服务暂时繁忙，请稍后重试")
