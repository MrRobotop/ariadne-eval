"""Sidecar ground-truth model used by metrics during evaluation.

A ``Case`` is the "what should have happened" complement to a
``Trajectory`` ("what did happen"). Cases are intentionally a separate
type: they live next to benchmarks, not next to traced runs, and the
``Trajectory`` schema stays untouched.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ariadne_eval.core.trajectory import JsonValue

__all__ = ["Case", "ExpectedTool"]


class ExpectedTool(BaseModel):
    """A single expected tool invocation in a Case.

    ``args=None`` means "match this expected tool against any actual call
    of the same name, regardless of arguments". When ``ToolAccuracy`` is
    constructed with ``match_args=True``, this becomes a per-tool wildcard.
    """

    model_config = {"frozen": True}

    name: str
    args: dict[str, JsonValue] | None = None


class Case(BaseModel):
    """Ground-truth reference for a single evaluation example."""

    model_config = {"frozen": True}

    case_id: str
    task: str
    expected_answer: str | None = None
    expected_tools: tuple[ExpectedTool, ...] = ()
    expected_max_steps: int | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
