"""Tool registry and router."""

from __future__ import annotations

from typing import Callable

from agentic_boilerplate.tools.base import ToolResult, ToolSpec


class ToolRegistry:
    """Central registry that maps tool names to specs and handlers."""

    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSpec, Callable]] = {}

    def register(self, spec: ToolSpec, handler: Callable) -> None:
        """Register a tool spec together with its handler callable."""
        self._tools[spec.name] = (spec, handler)

    def get(self, name: str) -> tuple[ToolSpec, Callable] | None:
        """Return (spec, handler) for the given tool name, or None."""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSpec]:
        """Return all registered tool specs."""
        return [spec for spec, _ in self._tools.values()]

    async def execute(self, name: str, params: dict) -> ToolResult:
        """Execute a registered tool by name with the supplied parameters."""
        import asyncio

        entry = self._tools.get(name)
        if entry is None:
            return ToolResult.fail(f"Unknown tool: '{name}'")

        spec, handler = entry
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**params)
            else:
                result = handler(**params)

            if isinstance(result, ToolResult):
                return result
            return ToolResult.ok(str(result))
        except Exception as exc:  # noqa: BLE001
            return ToolResult.fail(str(exc))

    def to_llm_schema(self) -> list[dict]:
        """Return tool definitions in a format suitable for LLM API calls."""
        schema = []
        for spec, _ in self._tools.values():
            properties = {}
            required = []
            for param_name, param_info in spec.parameters.items():
                properties[param_name] = {
                    "type": param_info.get("type", "string"),
                    "description": param_info.get("description", ""),
                }
                required.append(param_name)

            schema.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                }
            )
        return schema
