"""Base dataclasses for tool specifications and results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolSpec:
    """Describes a tool that the agent can call."""

    name: str
    description: str
    parameters: dict  # JSON-schema style: {"param_name": {"type": ..., "description": ...}}
    requires_approval: bool = False
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to a plain dict."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "requires_approval": self.requires_approval,
            "tags": self.tags,
        }


@dataclass
class ToolResult:
    """The outcome of executing a tool."""

    success: bool
    output: str
    error: str | None = None

    @classmethod
    def ok(cls, output: str) -> "ToolResult":
        return cls(success=True, output=output)

    @classmethod
    def fail(cls, error: str) -> "ToolResult":
        return cls(success=False, output="", error=error)
