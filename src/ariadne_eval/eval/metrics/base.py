"""Base types for evaluation metrics."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ariadne_eval.core.trajectory import JsonValue, Step, Trajectory
from ariadne_eval.eval.case import Case

__all__ = ["Metric", "MetricResult"]


class MetricResult(BaseModel):
    """Per-(trajectory, case) output from a single Metric.

    ``score`` is always populated. For Phase-5 metrics it is in ``[0, 1]``,
    but the type does not enforce a range — future metrics may produce
    negative or unbounded scores.
    """

    model_config = {"frozen": True}

    metric: str
    case_id: str
    trajectory_id: str
    score: float
    label: Literal["pass", "fail", "partial"] | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


@runtime_checkable
class Metric(Protocol):
    """Pure-compute, sync per-trajectory scoring contract.

    Implementations are expected to be deterministic. Async metrics
    (judges) arrive in Phase 6 behind a separate ``AsyncMetric`` Protocol.
    """

    name: str

    def score(self, trajectory: Trajectory, steps: list[Step], case: Case) -> MetricResult:
        """Score a trajectory against a ground-truth case."""
        ...
