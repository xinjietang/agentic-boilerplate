"""Base dataclasses for skill specifications and results."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_boilerplate.tools.base import ToolResult, ToolSpec


@dataclass
class SkillSpec:
    """Describes an agent skill that can be invoked with a natural-language input."""

    name: str
    description: str
    input_description: str  # describes what kind of natural-language input to pass
    requires_approval: bool = False
    tags: list[str] = field(default_factory=list)

    def to_tool_spec(self) -> ToolSpec:
        """Convert to a :class:`~agentic_boilerplate.tools.base.ToolSpec` for registration in a :class:`~agentic_boilerplate.tools.registry.ToolRegistry`."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters={
                "input": {
                    "type": "string",
                    "description": self.input_description,
                }
            },
            requires_approval=self.requires_approval,
            tags=["skill"] + self.tags,
        )

    def to_dict(self) -> dict:
        """Serialise to a plain dict."""
        return {
            "name": self.name,
            "description": self.description,
            "input_description": self.input_description,
            "requires_approval": self.requires_approval,
            "tags": self.tags,
        }


@dataclass
class SkillResult:
    """The outcome of invoking a skill."""

    success: bool
    output: str
    error: str | None = None

    @classmethod
    def ok(cls, output: str) -> "SkillResult":
        return cls(success=True, output=output)

    @classmethod
    def fail(cls, error: str) -> "SkillResult":
        return cls(success=False, output="", error=error)

    def to_tool_result(self) -> ToolResult:
        """Convert to a :class:`~agentic_boilerplate.tools.base.ToolResult`."""
        if self.success:
            return ToolResult.ok(self.output)
        return ToolResult.fail(self.error or "Skill failed")
