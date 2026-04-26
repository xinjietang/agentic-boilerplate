"""System prompt templates."""

from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful AI assistant. You answer questions clearly and concisely.
When you are unsure, say so. Do not make up information.
"""

TOOL_USE_SYSTEM_PROMPT = """\
You are a helpful AI assistant with access to tools.
When a tool is available that can help with the user's request, use it.
Always explain what you are doing and why.
After receiving tool results, summarise the outcome for the user.
"""


def get_system_prompt(tools: list, style: str = "default") -> str:
    """Return an appropriate system prompt, optionally listing available tools.

    Args:
        tools: List of ToolSpec objects whose names and descriptions will be
               appended to the prompt when *style* is "tool_use".
        style: "default" returns DEFAULT_SYSTEM_PROMPT; "tool_use" returns
               TOOL_USE_SYSTEM_PROMPT with a tool listing appended.

    Returns:
        The system prompt string.
    """
    if style == "tool_use" and tools:
        tool_lines = "\n".join(
            f"  - {t.name}: {t.description}" for t in tools
        )
        return TOOL_USE_SYSTEM_PROMPT + f"\nAvailable tools:\n{tool_lines}\n"
    if style == "tool_use":
        return TOOL_USE_SYSTEM_PROMPT
    return DEFAULT_SYSTEM_PROMPT
