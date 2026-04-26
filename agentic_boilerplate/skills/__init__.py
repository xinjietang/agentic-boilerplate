"""Agent skills — higher-level, natural-language callable capabilities.

A *skill* wraps a callable (often a full ``AgentLoop``) behind a single
``input: str`` interface, making it easy to compose agents hierarchically.
Skills are exposed to the LLM as regular tools (via :class:`SkillRegistry`),
but emit dedicated ``SKILL_CALL`` / ``SKILL_RESULT`` events so they can be
observed and rendered separately from lower-level tool calls.

Typical usage::

    from agentic_boilerplate.skills import SkillSpec, SkillRegistry, AgentSkill

    spec = SkillSpec(
        name="summariser",
        description="Summarise a block of text",
        input_description="The text to summarise",
    )
    skill = AgentSkill(summariser_loop)

    skill_registry = SkillRegistry()
    skill_registry.register(spec, skill)
    skill_registry.register_in_tool_registry(tool_registry)
"""

from agentic_boilerplate.skills.agent_skill import AgentSkill
from agentic_boilerplate.skills.base import SkillResult, SkillSpec
from agentic_boilerplate.skills.registry import SkillRegistry

__all__ = ["AgentSkill", "SkillResult", "SkillSpec", "SkillRegistry"]
