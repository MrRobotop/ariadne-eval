"""Final-answer match metric."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Literal

from ariadne_eval.core.trajectory import Step, Trajectory
from ariadne_eval.eval.case import Case
from ariadne_eval.eval.errors import MissingReferenceError
from ariadne_eval.eval.metrics.base import MetricResult

__all__ = ["FinalAnswerMatch"]


_WHITESPACE = re.compile(r"\s+")


def _normalize(s: str) -> str:
    return _WHITESPACE.sub(" ", s.strip().lower())


def _render(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _label_from_score(score: float) -> Literal["pass", "fail", "partial"]:
    if score >= 0.99:
        return "pass"
    if score <= 0.01:
        return "fail"
    return "partial"


class FinalAnswerMatch:
    """Compare ``trajectory.final_answer`` against ``case.expected_answer``."""

    name: str

    def __init__(
        self,
        comparator: Literal["normalized_exact", "exact"]
        | Callable[[str, str], float] = "normalized_exact",
        *,
        name: str = "final_answer_match",
    ) -> None:
        """Initialise with an optional comparator and metric name."""
        self._comparator = comparator
        self.name = name

    def score(self, trajectory: Trajectory, steps: list[Step], case: Case) -> MetricResult:
        """Score a trajectory's final answer against the ground-truth case."""
        if case.expected_answer is None:
            raise MissingReferenceError("expected_answer", case_id=case.case_id)

        if trajectory.final_answer is None:
            return MetricResult(
                metric=self.name,
                case_id=case.case_id,
                trajectory_id=trajectory.id,
                score=0.0,
                label="fail",
                details={"reason": "no_final_answer"},
            )

        actual = _render(trajectory.final_answer)
        expected = case.expected_answer

        if self._comparator == "normalized_exact":
            score = 1.0 if _normalize(actual) == _normalize(expected) else 0.0
            label: Literal["pass", "fail", "partial"] = "pass" if score == 1.0 else "fail"
        elif self._comparator == "exact":
            score = 1.0 if actual == expected else 0.0
            label = "pass" if score == 1.0 else "fail"
        else:
            score = float(self._comparator(actual, expected))
            label = _label_from_score(score)

        return MetricResult(
            metric=self.name,
            case_id=case.case_id,
            trajectory_id=trajectory.id,
            score=score,
            label=label,
        )
