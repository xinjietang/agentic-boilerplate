"""AgentSkill — wraps an AgentLoop as a callable skill handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentic_boilerplate.skills.base import SkillResult

if TYPE_CHECKING:
    from agentic_boilerplate.core.agent_loop import AgentLoop
    from agentic_boilerplate.core.events import AgentEvent


class AgentSkill:
    """Wraps an :class:`~agentic_boilerplate.core.agent_loop.AgentLoop` as a
    callable skill handler.

    When invoked, it runs one turn of the sub-agent's loop and returns the
    last assistant message as the skill output.

    Example::

        sub_loop = build_summariser_loop()
        skill = AgentSkill(sub_loop)

        spec = SkillSpec(
            name="summariser",
            description="Summarise text",
            input_description="The text to summarise",
        )
        skill_registry.register(spec, skill)
    """

    def __init__(self, loop: AgentLoop) -> None:
        self._loop = loop
        self._last_output: str = ""
        loop.event_bus.subscribe(self._on_event)

    def _on_event(self, event: AgentEvent) -> None:
        from agentic_boilerplate.core.events import EventType

        if event.type == EventType.ASSISTANT_MESSAGE:
            self._last_output = event.data.get("content", "")

    async def __call__(self, input: str) -> SkillResult:
        """Run one agent turn with *input* and return the assistant's response."""
        self._last_output = ""
        await self._loop.run_turn(input)
        if self._last_output:
            return SkillResult.ok(self._last_output)
        return SkillResult.fail("Skill produced no output")
