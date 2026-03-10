"""AutoML MCP Server for Claude integration.

This module implements a Model Context Protocol (MCP) server that exposes the
AutoML pipeline as tools that Claude can invoke directly.  When the server is
made publicly available, every request to the underlying FastAPI backend is
authenticated using the ``AUTOML_API_KEY`` environment variable, ensuring that
only authorised callers can trigger ML jobs.

Usage
-----
Set the required environment variables (see .env.example), then run::

    python mcp_server.py

Claude Desktop configuration (claude_desktop_config.json)::

    {
      "mcpServers": {
        "automl": {
          "command": "python",
          "args": ["/path/to/automl/mcp_server.py"],
          "env": {
            "AUTOML_API_URL": "https://<your-public-host>:8000",
            "AUTOML_API_KEY": "<your-secret-api-key>"
          }
        }
      }
    }
"""

import asyncio
import json
import os
import sys

import httpx
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AUTOML_API_URL: str = os.getenv("AUTOML_API_URL", "http://127.0.0.1:8000")
AUTOML_API_KEY: str = os.getenv("AUTOML_API_KEY", "")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auth_headers() -> dict:
    """Return authentication headers for requests to the AutoML backend."""
    if AUTOML_API_KEY:
        return {"X-API-Key": AUTOML_API_KEY}
    return {}


async def _post(path: str, **kwargs) -> dict:
    """Send an authenticated POST request to the AutoML backend.

    Args:
        path: URL path relative to ``AUTOML_API_URL`` (e.g. ``"/eda"``).
        **kwargs: Extra keyword arguments forwarded to ``httpx.AsyncClient.post``.

    Returns:
        Parsed JSON response body.

    Raises:
        RuntimeError: When the backend returns a non-2xx status code.
    """
    url = f"{AUTOML_API_URL}{path}"
    headers = _auth_headers()
    if "headers" in kwargs:
        headers = {**headers, **kwargs.pop("headers")}

    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(url, headers=headers, **kwargs)

    if not response.is_success:
        raise RuntimeError(
            f"AutoML backend error {response.status_code}: {response.text}"
        )
    return response.json()


# ---------------------------------------------------------------------------
# MCP server definition
# ---------------------------------------------------------------------------

server = Server("automl")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Declare the tools that Claude can call."""
    return [
        types.Tool(
            name="upload_dataset",
            description=(
                "Upload a CSV or Excel dataset to the AutoML backend. "
                "Returns a session_id that must be used in subsequent calls."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the CSV or Excel file on the server.",
                    }
                },
                "required": ["file_path"],
            },
        ),
        types.Tool(
            name="run_eda",
            description=(
                "Generate an Exploratory Data Analysis (EDA) HTML report for a "
                "previously uploaded dataset. Returns the URL to the report."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID returned by upload_dataset.",
                    }
                },
                "required": ["session_id"],
            },
        ),
        types.Tool(
            name="train_models",
            description=(
                "Train machine learning models on a previously uploaded dataset. "
                "Automatically detects the problem type (regression or classification) "
                "and trains multiple algorithms. Returns performance metrics."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID returned by upload_dataset.",
                    },
                    "problem_statement": {
                        "type": "string",
                        "description": (
                            "Natural-language description of what you want to predict, "
                            "e.g. 'Predict house prices based on features'."
                        ),
                    },
                },
                "required": ["session_id", "problem_statement"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Dispatch a tool call from Claude to the AutoML backend.

    Args:
        name: Tool name as declared in ``list_tools``.
        arguments: Validated arguments matching the tool's input schema.

    Returns:
        A list containing a single ``TextContent`` item with the JSON result.

    Raises:
        ValueError: When an unknown tool name is requested.
    """
    if name == "upload_dataset":
        file_path = arguments["file_path"]

        # Resolve to an absolute path and verify the file stays within a safe
        # directory.  The allowed root defaults to the system's temp directory
        # but can be overridden via AUTOML_UPLOAD_DIR.
        allowed_root = os.path.realpath(
            os.getenv("AUTOML_UPLOAD_DIR", os.path.expanduser("~"))
        )
        real_path = os.path.realpath(file_path)
        if not real_path.startswith(allowed_root + os.sep) and real_path != allowed_root:
            raise ValueError(
                f"Access denied: '{file_path}' is outside the allowed upload "
                f"directory '{allowed_root}'. Set AUTOML_UPLOAD_DIR to override."
            )

        with open(real_path, "rb") as fh:
            file_bytes = fh.read()
        filename = os.path.basename(real_path)
        result = await _post(
            "/upload",
            files={"file": (filename, file_bytes)},
        )
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    if name == "run_eda":
        result = await _post(
            "/eda",
            json={"session_id": arguments["session_id"]},
        )
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    if name == "train_models":
        result = await _post(
            "/ml-models",
            json={
                "session_id": arguments["session_id"],
                "problem_statement": arguments["problem_statement"],
            },
        )
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
