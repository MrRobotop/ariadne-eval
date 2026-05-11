"""Stub tools used by the reference ReAct agent.

The calculator parses its input via Python's ``ast`` module and walks the
syntax tree with a whitelisted visitor. Python's built-in expression
evaluator is never invoked. The search tool is a dict lookup against a
small fixed knowledge base.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from ariadne_eval.core.trajectory import JsonValue

__all__ = ["TOOLS", "Tool", "calculator", "search"]


@dataclass(frozen=True)
class Tool:
    """A tool the reference agent can call."""

    name: str
    description: str
    fn: Callable[[str], JsonValue]


_ALLOWED_BIN_OPS: Final = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.FloorDiv,
)
_ALLOWED_UNARY_OPS: Final = (ast.UAdd, ast.USub)


def _safe_compute(expression: str) -> float:
    """Parse and walk an arithmetic expression with an AST whitelist.

    Accepts numeric literals and the basic arithmetic operators. Rejects
    everything else with ``ValueError``.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"could not parse expression: {expression!r}") from exc

    def _walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY_OPS):
            operand = _walk(node.operand)
            return +operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BIN_OPS):
            left = _walk(node.left)
            right = _walk(node.right)
            op = node.op
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
            if isinstance(op, ast.Div):
                return left / right
            if isinstance(op, ast.Mod):
                return left % right
            if isinstance(op, ast.Pow):
                return float(left**right)
            if isinstance(op, ast.FloorDiv):
                return float(left // right)
        raise ValueError(
            f"disallowed expression node: {type(node).__name__} in {expression!r}"
        )

    return _walk(tree)


def calculator(expression: str) -> float:
    """Evaluate a basic arithmetic expression safely."""
    return _safe_compute(expression)


_SEARCH_DB: Final[dict[str, str]] = {
    "banana": "Banana is a fruit. The word has 6 letters.",
    "ariadne": "Ariadne gave Theseus a thread to navigate the labyrinth.",
}


def search(query: str) -> str:
    """Return a fixed answer for a small set of demo queries."""
    return _SEARCH_DB.get(query.lower().strip(), "No results.")


TOOLS: Final[dict[str, Tool]] = {
    "calculator": Tool(
        name="calculator",
        description=(
            "calculator(expression: str) -> float — evaluate a basic arithmetic expression"
        ),
        fn=calculator,
    ),
    "search": Tool(
        name="search",
        description="search(query: str) -> str — search a small fixed knowledge base",
        fn=search,
    ),
}
