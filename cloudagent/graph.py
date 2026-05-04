import json
import logging
from collections.abc import AsyncIterator

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver

from cloudagent.agent.chat_agent import ChatAgent
from cloudagent.agent.rag_agent import RAGAgent
from cloudagent.agent.router import EntryAgent
from cloudagent.config import settings
from cloudagent.hitl import HITLManager
from cloudagent.memory.manager import TieredMemoryManager
from cloudagent.cache import QueryCache
from cloudagent.state import AgentState

logger = logging.getLogger(__name__)


class GraphNodes:
    def __init__(
        self,
        entry_agent: EntryAgent,
        chat_agent: ChatAgent,
        rag_agent: RAGAgent,
        memory_manager: TieredMemoryManager | None = None,
        cache: QueryCache | None = None,
        hitl: HITLManager | None = None,
        workflow_agent=None,
    ):
        self.entry_agent = entry_agent
        self.chat_agent = chat_agent
        self.rag_agent = rag_agent
        self.memory_manager = memory_manager
        self.cache = cache
        self.hitl = hitl or HITLManager()
        self.workflow_agent = workflow_agent

    async def load_memory_node(self, state: AgentState) -> AgentState:
        session_id = state.get("session_id", "")
        user_id = state.get("user_id", "anonymous")

        if self.memory_manager is not None:
            try:
                context = await self.memory_manager.get_context(session_id, user_id)
                messages = context.get("messages", [])
                state["messages"] = messages
                state["context"] = context
            except Exception as e:
                logger.warning(f"Memory manager failed: {e}")
                state["context"] = {}
        else:
            state["context"] = {}

        state["messages"].append({"role": "user", "content": state.get("last_message", "")})
        return state

    def entry_node(self, state: AgentState) -> AgentState:
        state = self.entry_agent.run(state)
        return state

    def route_node(self, state: AgentState) -> str:
        target = state.get("target_agent")
        confidence = state.get("confidence", 0.0)

        if target == "clarify":
            return "clarify"
        if target == "workflow" and self.hitl.is_sensitive("workflow", {}):
            return "hitl_request"
        if target == "workflow":
            return "workflow"
        if target in ("chat", "faq") and confidence > 0.5:
            return target
        return "chat"

    def chat_node(self, state: AgentState) -> AgentState:
        messages = state.get("messages", [])
        response = self.chat_agent.run(messages)
        state["response"] = response
        return state

    async def rag_node(self, state: AgentState) -> AgentState:
        response = await self.rag_agent.run(state)
        state["response"] = response
        return state

    async def workflow_node(self, state: AgentState) -> AgentState:
        if self.workflow_agent is not None:
            try:
                response = await self.workflow_agent.run(state)
            except Exception as e:
                logger.warning(f"Workflow agent failed: {e}")
                response = "业务办理暂时无法完成，请稍后重试。"
        else:
            response = "业务办理功能正在开发中，请稍后再试。"
        state["response"] = response
        return state

    def clarify_node(self, state: AgentState) -> AgentState:
        state["response"] = state.get("clarification_question", "能再详细说明一下吗？")
        state["action_required"] = "clarify"
        return state

    def hitl_request_node(self, state: AgentState) -> AgentState:
        action = {"action": "workflow", "params": {}}
        state["pending_action"] = action
        state["response"] = self.hitl.build_confirmation_message(action["action"], action["params"])
        state["action_required"] = "confirm"
        return state

    def hitl_resume_node(self, state: AgentState) -> AgentState:
        messages = state.get("messages", [])
        last_msg = messages[-1]["content"] if messages else ""

        if self.hitl.is_confirm(last_msg):
            state["response"] = "业务办理已确认执行。"
        elif self.hitl.is_reject(last_msg):
            state["response"] = "业务办理已取消。"
        else:
            state["response"] = "请回复'确认'或'取消'。"
            state["action_required"] = "confirm"
            return state

        state["pending_action"] = None
        state["action_required"] = None
        return state

    async def save_memory_node(self, state: AgentState) -> AgentState:
        if self.memory_manager is not None:
            try:
                session_id = state.get("session_id", "")
                user_id = state.get("user_id", "anonymous")
                messages = state.get("messages", [])
                await self.memory_manager.save_turn(session_id, user_id, messages)
            except Exception as e:
                logger.warning(f"Save memory failed: {e}")
        return state

    async def stream_node(self, state: AgentState) -> AsyncIterator[dict]:
        target = state.get("target_agent")

        if target == "clarify":
            yield {"event": "token", "data": state.get("clarification_question", "能再详细说明一下吗？")}
            yield {"event": "done", "data": json.dumps({"response": state.get("clarification_question", "能再详细说明一下吗？"), "intent": "clarify"})}
            return

        if target == "workflow" and self.hitl.is_sensitive("workflow", {}):
            yield {"event": "hitl", "data": json.dumps({"action_required": "confirm", "message": self.hitl.build_confirmation_message("workflow", {})})}
            yield {"event": "done", "data": json.dumps({"response": "请确认是否执行此操作。", "intent": "workflow", "action_required": "confirm"})}
            return

        if target == "chat":
            tokens = []
            async for token in self.chat_agent.run_stream(state.get("messages", [])):
                tokens.append(token)
                yield {"event": "token", "data": token}
            response = "".join(tokens)
            state["response"] = response

        elif target == "faq":
            tokens = []
            async for token in self.rag_agent.run_stream(state):
                tokens.append(token)
                yield {"event": "token", "data": token}
            response = "".join(tokens)
            state["response"] = response

        elif target == "workflow":
            if self.workflow_agent is not None:
                try:
                    async for event in self.workflow_agent.run_stream(state):
                        yield event
                    state["response"] = "Workflow completed"
                except Exception as e:
                    logger.warning(f"Workflow stream failed: {e}")
                    yield {"event": "token", "data": "业务办理暂时无法完成，请稍后重试。"}
                    state["response"] = "业务办理暂时无法完成，请稍后重试。"
            else:
                yield {"event": "token", "data": "业务办理功能正在开发中，请稍后再试。"}
                state["response"] = "业务办理功能正在开发中，请稍后再试。"

        else:
            tokens = []
            async for token in self.chat_agent.run_stream(state.get("messages", [])):
                tokens.append(token)
                yield {"event": "token", "data": token}
            response = "".join(tokens)
            state["response"] = response

        yield {"event": "done", "data": json.dumps({"response": state.get("response", ""), "intent": state.get("intent", "chat")})}


def build_graph(
    entry_agent: EntryAgent,
    chat_agent: ChatAgent,
    rag_agent: RAGAgent,
    memory_manager: TieredMemoryManager | None = None,
    cache: QueryCache | None = None,
    hitl: HITLManager | None = None,
    workflow_agent=None,
):
    nodes = GraphNodes(entry_agent, chat_agent, rag_agent, memory_manager, cache, hitl, workflow_agent)

    builder = StateGraph(AgentState)

    builder.add_node("load_memory", nodes.load_memory_node)
    builder.add_node("entry", nodes.entry_node)
    builder.add_node("chat", nodes.chat_node)
    builder.add_node("rag", nodes.rag_node)
    builder.add_node("workflow", nodes.workflow_node)
    builder.add_node("clarify", nodes.clarify_node)
    builder.add_node("hitl_request", nodes.hitl_request_node)
    builder.add_node("hitl_resume", nodes.hitl_resume_node)
    builder.add_node("save_memory", nodes.save_memory_node)

    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "entry")
    builder.add_conditional_edges(
        "entry",
        nodes.route_node,
        {
            "chat": "chat",
            "faq": "rag",
            "workflow": "workflow",
            "hitl_request": "hitl_request",
            "clarify": "clarify",
        },
    )
    builder.add_edge("chat", "save_memory")
    builder.add_edge("rag", "save_memory")
    builder.add_edge("workflow", "save_memory")
    builder.add_edge("clarify", "save_memory")
    builder.add_edge("hitl_request", "hitl_resume")
    builder.add_edge("hitl_resume", "save_memory")
    builder.add_edge("save_memory", END)

    # HITL interrupt: before hitl_resume, pause for user confirmation
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer, interrupt_before=["hitl_resume"])
    return graph
