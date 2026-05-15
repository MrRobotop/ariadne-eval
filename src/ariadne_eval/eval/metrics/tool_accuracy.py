"""Tool-call accuracy metric."""

from __future__ import annotations

import json
from typing import Literal, cast

from ariadne_eval.core.trajectory import (
    JsonValue,
    Step,
    ToolCallPayload,
    Trajectory,
)
from ariadne_eval.eval.case import Case, ExpectedTool
from ariadne_eval.eval.metrics.base import MetricResult

__all__ = ["ToolAccuracy"]


def _canon_args(args: dict[str, JsonValue]) -> str:
    """Canonicalize argument dict to a stable JSON string."""
    return json.dumps(args, sort_keys=True, default=str)


def _label(score: float) -> Literal["pass", "fail", "partial"]:
    """Map a numeric score to a human-readable label."""
    if score >= 0.99:
        return "pass"
    if score <= 0.01:
        return "fail"
    return "partial"


class ToolAccuracy:
    """Score how well the agent's tool calls match the expected ones."""

    name: str

    def __init__(
        self,
        mode: Literal["set", "ordered_prefix"] = "set",
        *,
        match_args: bool = False,
        name: str = "tool_accuracy",
    ) -> None:
        """Initialise ToolAccuracy with a scoring mode and optional arg matching."""
        self._mode = mode
        self._match_args = match_args
        self.name = name

    def score(self, trajectory: Trajectory, steps: list[Step], case: Case) -> MetricResult:
        """Score a trajectory against a ground-truth case."""
        actual = sorted(
            (s for s in steps if isinstance(s.payload, ToolCallPayload)),
            key=lambda s: s.started_at,
        )
        actual_payloads: list[ToolCallPayload] = [
            s.payload for s in actual if isinstance(s.payload, ToolCallPayload)
        ]

        if self._mode == "set":
            return self._score_set(trajectory, actual_payloads, case)
        return self._score_ordered_prefix(trajectory, actual_payloads, case)

    def _score_set(
        self,
        trajectory: Trajectory,
        actual: list[ToolCallPayload],
        case: Case,
    ) -> MetricResult:
        """Score using multiset F1 over tool names (or name+args when match_args=True)."""
        expected_keys = [self._expected_key(t) for t in case.expected_tools]
        actual_keys = [self._actual_key(p) for p in actual]

        # Per-tool wildcard handling for match_args=True with args=None:
        # remove a single matching actual call by name only.
        remaining_actual = list(actual_keys)
        matched: list[str] = []
        missing: list[str] = []
        for exp_key, exp_tool in zip(expected_keys, case.expected_tools, strict=True):
            if exp_key in remaining_actual:
                remaining_actual.remove(exp_key)
                matched.append(exp_tool.name)
            elif self._match_args and exp_tool.args is None:
                # name-only fallback
                wildcard_hit = next(
                    (k for k in remaining_actual if k.startswith(f"{exp_tool.name}::")),
                    None,
                )
                if wildcard_hit is not None:
                    remaining_actual.remove(wildcard_hit)
                    matched.append(exp_tool.name)
                else:
                    missing.append(exp_tool.name)
            else:
                missing.append(exp_tool.name)

        extra_names = [k.split("::", 1)[0] for k in remaining_actual]

        tp = len(matched)
        fp = len(extra_names)
        fn = len(missing)
        if tp == 0 and (fp > 0 or fn > 0):
            f1 = 0.0
            precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
            recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        elif tp == 0 and fp == 0 and fn == 0:
            f1 = 1.0
            precision = 1.0
            recall = 1.0
        else:
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        details: dict[str, JsonValue] = {
            "precision": precision,
            "recall": recall,
            "matched": cast("list[JsonValue]", matched),
            "missing": cast("list[JsonValue]", missing),
            "extra": cast("list[JsonValue]", extra_names),
        }
        return MetricResult(
            metric=self.name,
            case_id=case.case_id,
            trajectory_id=trajectory.id,
            score=f1,
            label=_label(f1),
            details=details,
        )

    def _score_ordered_prefix(
        self,
        trajectory: Trajectory,
        actual: list[ToolCallPayload],
        case: Case,
    ) -> MetricResult:
        """Score using longest matching prefix of the expected tool sequence."""
        expected = case.expected_tools
        prefix_len = 0
        first_divergence: int | None = None
        for i, exp in enumerate(expected):
            if i >= len(actual):
                first_divergence = i
                break
            if not self._matches(exp, actual[i]):
                first_divergence = i
                break
            prefix_len += 1

        score = 1.0 if not expected else prefix_len / len(expected)

        details: dict[str, JsonValue] = {
            "prefix_length": prefix_len,
            "expected_length": len(expected),
            "first_divergence_index": first_divergence,
        }
        return MetricResult(
            metric=self.name,
            case_id=case.case_id,
            trajectory_id=trajectory.id,
            score=score,
            label=_label(score),
            details=details,
        )

    def _expected_key(self, t: ExpectedTool) -> str:
        """Build lookup key for an expected tool."""
        if not self._match_args or t.args is None:
            return f"{t.name}::*"
        return f"{t.name}::{_canon_args(t.args)}"

    def _actual_key(self, p: ToolCallPayload) -> str:
        """Build lookup key for an actual tool call payload."""
        if not self._match_args:
            return f"{p.tool_name}::*"
        return f"{p.tool_name}::{_canon_args(p.arguments)}"

    def _matches(self, exp: ExpectedTool, act: ToolCallPayload) -> bool:
        """Return True if an actual payload satisfies an expected tool spec."""
        if exp.name != act.tool_name:
            return False
        if not self._match_args or exp.args is None:
            return True
        return _canon_args(exp.args) == _canon_args(act.arguments)
