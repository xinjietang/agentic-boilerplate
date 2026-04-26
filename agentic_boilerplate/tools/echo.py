"""Echo and file-read tools."""

from __future__ import annotations

import os

from agentic_boilerplate.tools.base import ToolResult, ToolSpec

_MAX_FILE_CHARS = 2000


def echo(text: str) -> ToolResult:
    """Return the input text unchanged."""
    return ToolResult.ok(text)


def read_file(path: str) -> ToolResult:
    """Read the first 2000 characters of a file."""
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return ToolResult.fail(f"File not found: {path}")
    if not os.path.isfile(abs_path):
        return ToolResult.fail(f"Not a file: {path}")
    try:
        with open(abs_path, encoding="utf-8", errors="replace") as fh:
            content = fh.read(_MAX_FILE_CHARS)
        truncated = os.path.getsize(abs_path) > _MAX_FILE_CHARS
        suffix = "\n... [truncated]" if truncated else ""
        return ToolResult.ok(content + suffix)
    except OSError as exc:
        return ToolResult.fail(str(exc))


ECHO_SPEC = ToolSpec(
    name="echo",
    description="Return the provided text unchanged",
    parameters={
        "text": {
            "type": "string",
            "description": "Text to echo back",
        }
    },
    requires_approval=False,
    tags=["utility", "safe"],
)

READ_FILE_SPEC = ToolSpec(
    name="read_file",
    description="Read the contents of a file (first 2000 characters)",
    parameters={
        "path": {
            "type": "string",
            "description": "Path to the file to read",
        }
    },
    requires_approval=True,
    tags=["filesystem"],
)


def register_echo_tools(registry) -> None:
    """Register echo and read_file tools in the given ToolRegistry."""
    registry.register(ECHO_SPEC, echo)
    registry.register(READ_FILE_SPEC, read_file)
