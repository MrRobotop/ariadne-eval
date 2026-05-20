"""Worked example: write a custom Metric and run it through the Runner.

Run with:
    uv run python examples/03_custom_metric/main.py
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ariadne_eval import (
    Case,
    ExpectedTool,
    FinalAnswerMatch,
    Metric,
    MetricResult,
    Runner,
    StepEfficiency,
    ToolAccuracy,
    new_id,
)
from ariadne_eval.core.status import StepStatus, TrajectoryStatus
from ariadne_eval.core.trajectory import (
    Step,
    ToolCallPayload,
    Trajectory,
)


class FinalAnswerLength(Metric):
    """A toy custom metric: closer to expected length is better."""

    name = "final_answer_length"

    def score(self, trajectory: Trajectory, steps: list[Step], case: Case) -> MetricResult:
        """Score how close the final-answer length is to the target length."""
        actual = trajectory.final_answer
        actual_len = len(actual) if isinstance(actual, str) else 0
        target = case.metadata.get("target_length", 1)
        target_int = int(target) if isinstance(target, (int, float, str)) else 1
        score = max(0.0, 1.0 - abs(actual_len - target_int) / max(target_int, 1))
        return MetricResult(
            metric=self.name,
            case_id=case.case_id,
            trajectory_id=trajectory.id,
            score=score,
            details={"actual_length": actual_len, "target_length": target_int},
        )


def _make_traj(answer: str) -> Trajectory:
    """Build a minimal Trajectory with the given final answer."""
    started = datetime(2026, 5, 14, 12, 0, 0, tzinfo=UTC)
    return Trajectory(
        id=new_id(),
        task="demo",
        agent_name="example",
        agent_version="0.0.0",
        model_id="example/model",
        started_at=started,
        finished_at=started + timedelta(seconds=1),
        final_status=TrajectoryStatus.SUCCEEDED,
        final_answer=answer,
    )


def _make_step(traj_id: str, tool: str) -> Step:
    """Build a minimal succeeded tool-call Step."""
    started = datetime(2026, 5, 14, 12, 0, 0, tzinfo=UTC)
    return Step(
        id=new_id(),
        trajectory_id=traj_id,
        parent_step_id=None,
        name=f"tool:{tool}",
        started_at=started,
        finished_at=started + timedelta(milliseconds=5),
        status=StepStatus.SUCCEEDED,
        payload=ToolCallPayload(tool_name=tool, arguments={}, result=None, latency_ms=5.0),
    )


def main() -> None:
    """Run three synthetic trajectories through the Runner and print aggregates."""
    pairs = []
    for i, (answer, tools) in enumerate(
        [("4", ["calc"]), ("forty-two", ["calc", "search"]), ("4", ["calc"])]
    ):
        traj = _make_traj(answer)
        steps = [_make_step(traj.id, t) for t in tools]
        case = Case(
            case_id=f"c{i}",
            task="what is 2+2?",
            expected_answer="4",
            expected_tools=(ExpectedTool(name="calc"),),
            expected_max_steps=1,
            metadata={"target_length": 1},
        )
        pairs.append((traj, steps, case))

    runner = Runner(
        metrics=[
            FinalAnswerMatch(),
            ToolAccuracy(),
            StepEfficiency(),
            FinalAnswerLength(),
        ],
        seed=0,
        n_resamples=500,
    )
    report = runner.evaluate(pairs)

    print(f"n_cases = {report.n_cases}")
    for name, ci in report.aggregates.items():
        print(f"  {name:24s} mean={ci.mean:.3f}  95% CI=[{ci.lo:.3f}, {ci.hi:.3f}]  n={ci.n}")


if __name__ == "__main__":
    main()
