"""Tests for the skills package: SkillSpec, SkillResult, SkillRegistry, AgentSkill."""

from __future__ import annotations

import pytest

from agentic_boilerplate.skills.base import SkillResult, SkillSpec
from agentic_boilerplate.skills.registry import SkillRegistry
from agentic_boilerplate.tools.base import ToolResult
from agentic_boilerplate.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# SkillResult helpers
# ---------------------------------------------------------------------------

def test_skill_result_ok():
    r = SkillResult.ok("done")
    assert r.success is True
    assert r.output == "done"
    assert r.error is None


def test_skill_result_fail():
    r = SkillResult.fail("boom")
    assert r.success is False
    assert r.output == ""
    assert r.error == "boom"


def test_skill_result_to_tool_result_ok():
    tr = SkillResult.ok("out").to_tool_result()
    assert isinstance(tr, ToolResult)
    assert tr.success is True
    assert tr.output == "out"


def test_skill_result_to_tool_result_fail():
    tr = SkillResult.fail("err").to_tool_result()
    assert isinstance(tr, ToolResult)
    assert tr.success is False
    assert tr.error == "err"


def test_skill_result_to_tool_result_fail_no_error_message():
    sr = SkillResult(success=False, output="", error=None)
    tr = sr.to_tool_result()
    assert tr.success is False
    assert tr.error == "Skill failed"


# ---------------------------------------------------------------------------
# SkillSpec
# ---------------------------------------------------------------------------

def test_skill_spec_to_tool_spec():
    spec = SkillSpec(
        name="my_skill",
        description="Does something",
        input_description="The thing to do",
        requires_approval=True,
        tags=["custom"],
    )
    ts = spec.to_tool_spec()
    assert ts.name == "my_skill"
    assert ts.description == "Does something"
    assert "input" in ts.parameters
    assert ts.parameters["input"]["type"] == "string"
    assert ts.parameters["input"]["description"] == "The thing to do"
    assert ts.requires_approval is True
    assert "skill" in ts.tags
    assert "custom" in ts.tags


def test_skill_spec_to_dict():
    spec = SkillSpec(name="s", description="d", input_description="i")
    d = spec.to_dict()
    assert d["name"] == "s"
    assert d["description"] == "d"
    assert d["input_description"] == "i"
    assert d["requires_approval"] is False
    assert d["tags"] == []


# ---------------------------------------------------------------------------
# SkillRegistry
# ---------------------------------------------------------------------------

def test_registry_register_and_list():
    reg = SkillRegistry()
    spec = SkillSpec(name="s1", description="d", input_description="i")
    reg.register(spec, lambda input: SkillResult.ok("ok"))
    assert len(reg.list_skills()) == 1
    assert reg.list_skills()[0].name == "s1"


def test_registry_get_known():
    reg = SkillRegistry()
    spec = SkillSpec(name="s1", description="d", input_description="i")
    handler = lambda input: SkillResult.ok("ok")
    reg.register(spec, handler)
    entry = reg.get("s1")
    assert entry is not None
    got_spec, got_handler = entry
    assert got_spec.name == "s1"
    assert got_handler is handler


def test_registry_get_unknown():
    reg = SkillRegistry()
    assert reg.get("nonexistent") is None


@pytest.mark.asyncio
async def test_registry_execute_known():
    reg = SkillRegistry()
    spec = SkillSpec(name="echo_skill", description="Echo", input_description="Text")
    reg.register(spec, lambda input: SkillResult.ok(f"echoed: {input}"))
    result = await reg.execute("echo_skill", "hello")
    assert result.success is True
    assert result.output == "echoed: hello"


@pytest.mark.asyncio
async def test_registry_execute_async_handler():
    reg = SkillRegistry()
    spec = SkillSpec(name="async_skill", description="Async", input_description="Text")

    async def async_handler(input: str) -> SkillResult:
        return SkillResult.ok(f"async: {input}")

    reg.register(spec, async_handler)
    result = await reg.execute("async_skill", "world")
    assert result.success is True
    assert result.output == "async: world"


@pytest.mark.asyncio
async def test_registry_execute_unknown():
    reg = SkillRegistry()
    result = await reg.execute("ghost", "anything")
    assert result.success is False
    assert "Unknown skill" in result.error


@pytest.mark.asyncio
async def test_registry_execute_handler_raises():
    reg = SkillRegistry()
    spec = SkillSpec(name="bad_skill", description="Bad", input_description="Text")

    def bad_handler(input: str) -> SkillResult:
        raise RuntimeError("oops")

    reg.register(spec, bad_handler)
    result = await reg.execute("bad_skill", "input")
    assert result.success is False
    assert "oops" in result.error


@pytest.mark.asyncio
async def test_registry_execute_non_skill_result_return():
    reg = SkillRegistry()
    spec = SkillSpec(name="str_skill", description="Str", input_description="Text")
    reg.register(spec, lambda input: "raw string")
    result = await reg.execute("str_skill", "x")
    assert result.success is True
    assert result.output == "raw string"


# ---------------------------------------------------------------------------
# register_in_tool_registry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_register_in_tool_registry():
    skill_reg = SkillRegistry()
    spec = SkillSpec(name="greet", description="Greet", input_description="Name to greet")
    skill_reg.register(spec, lambda input: SkillResult.ok(f"Hello, {input}!"))

    tool_reg = ToolRegistry()
    skill_reg.register_in_tool_registry(tool_reg)

    # The skill should now appear as a tool
    entry = tool_reg.get("greet")
    assert entry is not None
    tool_spec, _ = entry
    assert tool_spec.name == "greet"
    assert "input" in tool_spec.parameters
    assert "skill" in tool_spec.tags

    # Executing via ToolRegistry should work
    result = await tool_reg.execute("greet", {"input": "Alice"})
    assert result.success is True
    assert result.output == "Hello, Alice!"


@pytest.mark.asyncio
async def test_register_in_tool_registry_multiple_skills():
    skill_reg = SkillRegistry()

    spec_a = SkillSpec(name="skill_a", description="A", input_description="input a")
    spec_b = SkillSpec(name="skill_b", description="B", input_description="input b")
    skill_reg.register(spec_a, lambda input: SkillResult.ok("a"))
    skill_reg.register(spec_b, lambda input: SkillResult.ok("b"))

    tool_reg = ToolRegistry()
    skill_reg.register_in_tool_registry(tool_reg)

    assert tool_reg.get("skill_a") is not None
    assert tool_reg.get("skill_b") is not None

    result_a = await tool_reg.execute("skill_a", {"input": "x"})
    result_b = await tool_reg.execute("skill_b", {"input": "y"})
    assert result_a.output == "a"
    assert result_b.output == "b"


# ---------------------------------------------------------------------------
# AgentSkill
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_skill_wraps_loop():
    """AgentSkill should capture the sub-agent's assistant message as output."""
    from agentic_boilerplate.config.settings import AgentConfig
    from agentic_boilerplate.context.manager import ContextManager
    from agentic_boilerplate.core.agent_loop import AgentLoop
    from agentic_boilerplate.core.events import EventBus
    from agentic_boilerplate.policies.approval import ApprovalPolicy
    from agentic_boilerplate.skills.agent_skill import AgentSkill
    from agentic_boilerplate.tools.registry import ToolRegistry

    config = AgentConfig()
    context = ContextManager()
    context.add_message("system", config.system_prompt)
    tools = ToolRegistry()
    approval = ApprovalPolicy(yolo_mode=True)
    bus = EventBus()

    loop = AgentLoop(config, context, tools, approval, bus)
    skill = AgentSkill(loop)

    result = await skill(input="hello")
    assert result.success is True
    assert "hello" in result.output  # stub LLM echoes the message


@pytest.mark.asyncio
async def test_agent_skill_fail_on_no_output():
    """AgentSkill should return a failure result when the sub-agent emits no content."""
    from unittest.mock import AsyncMock, MagicMock

    from agentic_boilerplate.core.events import EventBus
    from agentic_boilerplate.skills.agent_skill import AgentSkill

    # Build a mock loop that does nothing (never emits ASSISTANT_MESSAGE)
    mock_loop = MagicMock()
    mock_loop.event_bus = EventBus()
    mock_loop.run_turn = AsyncMock()

    skill = AgentSkill(mock_loop)
    result = await skill(input="ping")
    assert result.success is False
    assert "no output" in result.error.lower()


# ---------------------------------------------------------------------------
# SKILL_CALL / SKILL_RESULT event integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_loop_emits_skill_events():
    """AgentLoop should emit SKILL_CALL/SKILL_RESULT when a skill is invoked."""
    from agentic_boilerplate.config.settings import AgentConfig
    from agentic_boilerplate.context.manager import ContextManager
    from agentic_boilerplate.core.agent_loop import AgentLoop
    from agentic_boilerplate.core.events import AgentEvent, EventBus, EventType
    from agentic_boilerplate.policies.approval import ApprovalPolicy
    from agentic_boilerplate.skills.registry import SkillRegistry
    from agentic_boilerplate.tools.registry import ToolRegistry

    config = AgentConfig()
    context = ContextManager()
    context.add_message("system", config.system_prompt)
    tool_reg = ToolRegistry()
    skill_reg = SkillRegistry()

    spec = SkillSpec(name="echo_skill", description="Echo", input_description="Text")
    skill_reg.register(spec, lambda input: SkillResult.ok(f"skill says: {input}"))
    skill_reg.register_in_tool_registry(tool_reg)

    approval = ApprovalPolicy(yolo_mode=True)
    bus = EventBus()
    loop = AgentLoop(config, context, tool_reg, approval, bus, skill_reg)

    emitted: list[AgentEvent] = []
    bus.subscribe(emitted.append)

    # Manually craft a turn that will call the skill by patching _call_llm
    original_call_llm = loop._call_llm

    async def patched_call_llm(messages):
        return {
            "content": "",
            "tool_calls": [
                {
                    "id": "call_skill_1",
                    "name": "echo_skill",
                    "parameters": {"input": "test input"},
                }
            ],
        }

    loop._call_llm = patched_call_llm
    await loop.run_turn("use the skill")

    event_types = [e.type for e in emitted]
    assert EventType.SKILL_CALL in event_types
    assert EventType.SKILL_RESULT in event_types
    # Regular TOOL_CALL/TOOL_RESULT should NOT be emitted for a skill invocation
    assert EventType.TOOL_CALL not in event_types
    assert EventType.TOOL_RESULT not in event_types

    skill_call_event = next(e for e in emitted if e.type == EventType.SKILL_CALL)
    assert skill_call_event.data["name"] == "echo_skill"
    assert skill_call_event.data["input"] == "test input"
