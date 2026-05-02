from mcp.server import Server
from mcp.types import Tool, TextContent, ListToolsRequest, CallToolRequest


class BaseMCPServer:
    def __init__(self, name: str):
        self._server = Server(name)
        self._tools: dict[str, tuple[callable, dict]] = {}
        self._register_handlers()

    def _register_handlers(self):
        @self._server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(name=name, description=fn.__doc__, inputSchema=schema)
                for name, (fn, schema) in self._tools.items()
            ]

        @self._server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            if name not in self._tools:
                raise ValueError(f"Unknown tool: {name}")
            fn = self._get_tool_fn(name)
            result = await fn(**arguments)
            return [TextContent(type="text", text=str(result))]

    def register_tool(self, name: str, schema: dict, fn: callable):
        self._tools[name] = (fn, schema)

    def _get_tool_fn(self, name: str):
        _, schema = self._tools[name]
        fn = getattr(self, name, None)
        if fn is None:
            fn, _ = self._tools[name]
        return fn

    async def list_tools(self):
        handler = self._server.request_handlers[ListToolsRequest]
        return await handler(ListToolsRequest(method="tools/list"))

    async def call_tool(self, name: str, arguments: dict):
        handler = self._server.request_handlers[CallToolRequest]
        req = CallToolRequest(method="tools/call", params={"name": name, "arguments": arguments})
        return await handler(req)

    @property
    def server(self):
        return self._server
