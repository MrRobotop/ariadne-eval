"""Step-efficiency metric."""

from __future__ import annotations

from typing import Literal

from ariadne_eval.core.trajectory import JsonValue, Step, Trajectory
from ariadne_eval.eval.case import Case
from ariadne_eval.eval.errors import MissingReferenceError
from ariadne_eval.eval.metrics.base import MetricResult

__all__ = ["StepEfficiency"]


class StepEfficiency:
    """Score = min(1, expected_max_steps / max(actual_steps, 1))."""

    name: str

    def __init__(self, *, name: str = "step_efficiency") -> None:
        """Initialise with a metric name, defaulting to 'step_efficiency'."""
        self.name = name

    def score(self, trajectory: Trajectory, steps: list[Step], case: Case) -> MetricResult:
        """Score a trajectory against a ground-truth case.

        Returns a MetricResult whose score is min(1.0, budget / max(actual, 1)).
        Label is 'pass' when actual <= budget, otherwise 'partial'.

        Raises:
            MissingReferenceError: if ``case.expected_max_steps`` is None.
        """
        if case.expected_max_steps is None:
            raise MissingReferenceError("expected_max_steps", case_id=case.case_id)

        actual = len(steps)
        budget = case.expected_max_steps
        score = min(1.0, budget / max(actual, 1))
        label: Literal["pass", "partial"] = "pass" if actual <= budget else "partial"
        details: dict[str, JsonValue] = {
            "actual_steps": actual,
            "expected_max_steps": budget,
        }
        return MetricResult(
            metric=self.name,
            case_id=case.case_id,
            trajectory_id=trajectory.id,
            score=score,
            label=label,
            details=details,
        )
