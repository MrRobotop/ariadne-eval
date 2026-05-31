"""PlanQuality: async metric that wraps any Judge to score the agent's plan."""

from __future__ import annotations

from typing import cast

from ariadne_eval.core.trajectory import JsonValue, Step, Trajectory
from ariadne_eval.eval.case import Case
from ariadne_eval.eval.judges.base import Judge
from ariadne_eval.eval.judges.trajectory_judge import _extract_plan
from ariadne_eval.eval.metrics.base import MetricResult

__all__ = ["PlanQuality"]


class PlanQuality:
    """Async-only metric: scores the agent's first articulated plan via a Judge."""

    name: str

    def __init__(self, judge: Judge, *, name: str = "plan_quality") -> None:
        """Initialise with any Judge implementation."""
        self._judge = judge
        self.name = name

    async def ascore(
        self,
        trajectory: Trajectory,
        steps: list[Step],
        case: Case | None,
    ) -> MetricResult:
        """Score the trajectory via the configured Judge."""
        case_id = case.case_id if case is not None else "<none>"
        if _extract_plan(steps) is None:
            return MetricResult(
                metric=self.name,
                case_id=case_id,
                trajectory_id=trajectory.id,
                score=0.0,
                label="fail",
                details={"reason": "no_llm_step"},
            )
        verdict = await self._judge.judge(trajectory, steps, case)
        details: dict[str, JsonValue] = {
            "rationale": verdict.rationale,
            "raw": cast("dict[str, JsonValue]", dict(verdict.raw)),
        }
        return MetricResult(
            metric=self.name,
            case_id=case_id,
            trajectory_id=trajectory.id,
            score=verdict.score,
            label=verdict.label,
            details=details,
        )
