"""Skill registry — manages and dispatches agent skills."""

from __future__ import annotations

from typing import Callable

from agentic_boilerplate.skills.base import SkillResult, SkillSpec


class SkillRegistry:
    """Central registry that maps skill names to specs and handlers.

    Skills are exposed to the LLM by registering them in a
    :class:`~agentic_boilerplate.tools.registry.ToolRegistry` via
    :meth:`register_in_tool_registry`.  Each skill is surfaced as a
    single-parameter tool whose parameter is named ``input``.
    """

    def __init__(self) -> None:
        self._skills: dict[str, tuple[SkillSpec, Callable]] = {}

    def register(self, spec: SkillSpec, handler: Callable) -> None:
        """Register a skill spec together with its handler callable.

        The *handler* must accept a single keyword argument ``input: str``
        and return a :class:`~agentic_boilerplate.skills.base.SkillResult`
        (or any value that will be coerced to one).
        """
        self._skills[spec.name] = (spec, handler)

    def get(self, name: str) -> tuple[SkillSpec, Callable] | None:
        """Return ``(spec, handler)`` for the given skill name, or ``None``."""
        return self._skills.get(name)

    def list_skills(self) -> list[SkillSpec]:
        """Return all registered skill specs."""
        return [spec for spec, _ in self._skills.values()]

    async def execute(self, name: str, input: str) -> SkillResult:
        """Execute a registered skill by name with the supplied input string."""
        import asyncio

        entry = self._skills.get(name)
        if entry is None:
            return SkillResult.fail(f"Unknown skill: '{name}'")

        spec, handler = entry
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(input=input)
            else:
                result = handler(input=input)

            if isinstance(result, SkillResult):
                return result
            return SkillResult.ok(str(result))
        except Exception as exc:  # noqa: BLE001
            return SkillResult.fail(str(exc))

    def register_in_tool_registry(self, tool_registry) -> None:
        """Register all skills as tools in *tool_registry*.

        Each skill is exposed as a tool with a single ``input`` string
        parameter so the LLM can invoke it using the standard
        function-calling interface.
        """
        for name, (spec, handler) in self._skills.items():
            tool_spec = spec.to_tool_spec()

            # Capture handler in default arg to avoid late-binding closure issues.
            async def _adapted(input: str, _h: Callable = handler) -> object:
                import asyncio

                if asyncio.iscoroutinefunction(_h):
                    result = await _h(input=input)
                else:
                    result = _h(input=input)

                if isinstance(result, SkillResult):
                    return result.to_tool_result()

                from agentic_boilerplate.tools.base import ToolResult

                return ToolResult.ok(str(result))

            tool_registry.register(tool_spec, _adapted)
