"""Safe calculator tool."""

from __future__ import annotations

import ast
import operator
from typing import Any

from agentic_boilerplate.tools.base import ToolResult, ToolSpec

# Allowed operators for the safe evaluator
_OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_MAX_POWER = 1000  # guard against huge exponents


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an AST node using only whitelisted operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if op_type is ast.Pow and abs(right) > _MAX_POWER:
            raise ValueError(f"Exponent too large (max {_MAX_POWER})")
        return _OPERATORS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return _OPERATORS[op_type](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def calculate(expression: str) -> ToolResult:
    """Safely evaluate a mathematical expression and return the result."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        # Format: strip trailing .0 for whole numbers
        formatted = int(result) if result == int(result) else result
        return ToolResult.ok(str(formatted))
    except ZeroDivisionError:
        return ToolResult.fail("Division by zero")
    except (ValueError, TypeError, SyntaxError) as exc:
        return ToolResult.fail(f"Invalid expression: {exc}")


CALCULATOR_SPEC = ToolSpec(
    name="calculator",
    description="Evaluate a mathematical expression",
    parameters={
        "expression": {
            "type": "string",
            "description": "Math expression to evaluate (e.g. '2 + 2', '10 * (3 + 4)')",
        }
    },
    requires_approval=False,
    tags=["math", "safe"],
)


def register_calculator(registry) -> None:
    """Register the calculator tool in the given ToolRegistry."""
    registry.register(CALCULATOR_SPEC, calculate)
