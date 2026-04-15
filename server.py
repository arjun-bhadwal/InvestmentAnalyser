"""
Investment Analyser MCP Server — Entry Point

Imports all tool modules (triggering @mcp.tool() registration) and starts the server.
Tool implementations live in   tools/*.py
Shared config & MCP instance in app.py
Caching & helpers in           helpers.py
"""
import os

from app import mcp
import tools  # noqa: F401 — triggers tool registration

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=port,
        path="/mcp",
        stateless_http=True,
    )
