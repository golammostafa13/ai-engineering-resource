"""End-to-end MCP test: act as an MCP CLIENT and drive the server over stdio,
exactly the way an agent (Claude Code) does — spawn the server, handshake,
list tools, and CALL them through JSON-RPC. This exercises the real protocol
path (serialization + schema validation + FastMCP dispatch), not the Python
functions directly.

    ../.venv/bin/python ai_advisor_agent/test_mcp_client.py
"""
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parent


def _text(result):
    """Pull the JSON payload out of a tool result (structured or text content)."""
    if getattr(result, "structuredContent", None):
        return result.structuredContent
    for block in result.content:
        if getattr(block, "text", None):
            try:
                return json.loads(block.text)
            except json.JSONDecodeError:
                return block.text
    return None


async def main():
    params = StdioServerParameters(command=sys.executable, args=[str(HERE / "mcp_server.py")])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            print(f"✓ connected to MCP server: {info.serverInfo.name}")

            tools = (await session.list_tools()).tools
            print(f"✓ tools/list → {[t.name for t in tools]}\n")

            print("→ calling recommend_model('Fix this race condition in my worker pool')")
            r1 = await session.call_tool("recommend_model",
                                         {"prompt": "Fix this race condition in my worker pool"})
            print(json.dumps(_text(r1), indent=2, default=str), "\n")

            print("→ calling run_with_best_model('Reply with just the number: 2+2')")
            r2 = await session.call_tool("run_with_best_model",
                                         {"prompt": "Reply with just the number: 2+2"})
            print(json.dumps(_text(r2), indent=2, default=str), "\n")

            print("→ calling run_with_best_model(hard task, allow_paid=false) — expect skip")
            r3 = await session.call_tool("run_with_best_model",
                                         {"prompt": "Prove sqrt(2) is irrational, step by step",
                                          "only_provider": "claude-cli", "allow_paid": False})
            out = _text(r3)
            print(json.dumps(out, indent=2, default=str))
            ok = isinstance(out, dict) and out.get("executed") is False
            print(f"\n{'✓' if ok else '✗'} paid pick correctly {'skipped' if ok else 'NOT skipped'} without allow_paid")


if __name__ == "__main__":
    asyncio.run(main())
