import logging
from typing import Any

from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.types import TextContent

logger = logging.getLogger(__name__)


class MCPClient:
    """Manages connections to multiple MCP servers and exposes unified tool interface."""

    def __init__(self):
        self._sessions: dict[str, ClientSession] = {}
        self._tool_registry: dict[str, dict] = {}

    async def connect_server(self, name: str, command: str, args: list[str] = None):
        """Connect to an MCP server via stdio."""
        params = StdioServerParameters(command=command, args=args or [])
        transport = stdio_client(params)
        read, write = await transport.__aenter__()
        session = ClientSession(read, write)
        await session.initialize()
        self._sessions[name] = session

        tools = await session.list_tools()
        for tool in tools.tools:
            full_name = f"{name}:{tool.name}"
            self._tool_registry[full_name] = {
                "server": name,
                "name": tool.name,
                "description": tool.description,
                "schema": tool.inputSchema,
            }
        logger.info(f"Connected to MCP server '{name}' with {len(tools.tools)} tools")

    async def call_tool(self, full_name: str, arguments: dict) -> str:
        """Call a tool by its fully qualified name (server:tool)."""
        if full_name not in self._tool_registry:
            raise ValueError(f"Unknown tool: {full_name}")
        meta = self._tool_registry[full_name]
        session = self._sessions[meta["server"]]
        result = await session.call_tool(meta["name"], arguments)
        texts = [item.text for item in result.content if isinstance(item, TextContent)]
        return "\n".join(texts)

    def list_tools(self) -> list[dict]:
        """List all available tools across all servers."""
        return [
            {
                "name": name,
                "description": meta["description"],
                "parameters": meta["schema"],
            }
            for name, meta in self._tool_registry.items()
        ]

    async def close(self):
        for session in self._sessions.values():
            await session.close()
