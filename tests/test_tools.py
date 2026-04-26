"""Tests for tool registry, calculator, echo, and ToolResult."""

from __future__ import annotations

import pytest

from agentic_boilerplate.tools.base import ToolResult, ToolSpec
from agentic_boilerplate.tools.calculator import CALCULATOR_SPEC, calculate, register_calculator
from agentic_boilerplate.tools.echo import ECHO_SPEC, READ_FILE_SPEC, echo, register_echo_tools
from agentic_boilerplate.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# ToolResult helpers
# ---------------------------------------------------------------------------

def test_tool_result_ok():
    r = ToolResult.ok("hello")
    assert r.success is True
    assert r.output == "hello"
    assert r.error is None


def test_tool_result_fail():
    r = ToolResult.fail("oops")
    assert r.success is False
    assert r.output == ""
    assert r.error == "oops"


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

def test_registry_register_and_list():
    reg = ToolRegistry()
    spec = ToolSpec(name="dummy", description="A dummy tool", parameters={})
    reg.register(spec, lambda: ToolResult.ok("ok"))
    tools = reg.list_tools()
    assert len(tools) == 1
    assert tools[0].name == "dummy"


def test_registry_get_known():
    reg = ToolRegistry()
    register_calculator(reg)
    entry = reg.get("calculator")
    assert entry is not None
    spec, handler = entry
    assert spec.name == "calculator"


def test_registry_get_unknown():
    reg = ToolRegistry()
    assert reg.get("nonexistent") is None


@pytest.mark.asyncio
async def test_registry_execute_unknown():
    reg = ToolRegistry()
    result = await reg.execute("ghost", {})
    assert result.success is False
    assert "Unknown tool" in result.error


@pytest.mark.asyncio
async def test_registry_execute_calculator():
    reg = ToolRegistry()
    register_calculator(reg)
    result = await reg.execute("calculator", {"expression": "2 + 3"})
    assert result.success is True
    assert result.output == "5"


# ---------------------------------------------------------------------------
# to_llm_schema
# ---------------------------------------------------------------------------

def test_to_llm_schema_structure():
    reg = ToolRegistry()
    register_calculator(reg)
    schema = reg.to_llm_schema()
    assert len(schema) == 1
    entry = schema[0]
    assert entry["type"] == "function"
    fn = entry["function"]
    assert fn["name"] == "calculator"
    assert "parameters" in fn
    assert "expression" in fn["parameters"]["properties"]


# ---------------------------------------------------------------------------
# Calculator tool
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expr,expected", [
    ("2 + 2", "4"),
    ("10 - 3", "7"),
    ("3 * 4", "12"),
    ("10 / 4", "2.5"),
    ("2 ** 10", "1024"),
    ("10 % 3", "1"),
    ("(2 + 3) * 4", "20"),
    ("-5 + 10", "5"),
])
def test_calculator_expressions(expr, expected):
    result = calculate(expression=expr)
    assert result.success is True
    assert result.output == expected


def test_calculator_division_by_zero():
    result = calculate(expression="1 / 0")
    assert result.success is False
    assert "zero" in result.error.lower()


def test_calculator_invalid_expression():
    result = calculate(expression="import os")
    assert result.success is False


def test_calculator_unsupported_builtin():
    result = calculate(expression="__import__('os')")
    assert result.success is False


# ---------------------------------------------------------------------------
# Echo tool
# ---------------------------------------------------------------------------

def test_echo_returns_input():
    result = echo(text="hello world")
    assert result.success is True
    assert result.output == "hello world"


def test_echo_empty_string():
    result = echo(text="")
    assert result.success is True
    assert result.output == ""


# ---------------------------------------------------------------------------
# Read file tool (registration check)
# ---------------------------------------------------------------------------

def test_read_file_spec_requires_approval():
    assert READ_FILE_SPEC.requires_approval is True


def test_echo_spec_no_approval():
    assert ECHO_SPEC.requires_approval is False
