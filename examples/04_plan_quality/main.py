"""Plan-quality evaluation via async Runner + StubJudge (no LLM, no API key).

Run with:
    uv run python examples/04_plan_quality/main.py

To swap in a real LLM judge: replace ``StubJudge(...)`` with
``TrajectoryJudge(model="claude-sonnet")`` and ensure ``ANTHROPIC_API_KEY``
is set.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from ariadne_eval import (
    AsyncMetric,
    Case,
    ExpectedTool,
    FinalAnswerMatch,
    Metric,
    Runner,
    StepEfficiency,
    ToolAccuracy,
    new_id,
)
from ariadne_eval.core.status import StepStatus, TrajectoryStatus
from ariadne_eval.core.trajectory import (
    LLMCallPayload,
    Message,
    Step,
    ToolCallPayload,
    Trajectory,
)
from ariadne_eval.eval import PlanQuality
from ariadne_eval.eval.judges import JudgeVerdict, StubJudge


def _make_traj(answer: str, plan: str) -> tuple[Trajectory, list[Step]]:
    started = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)
    traj = Trajectory(
        id=new_id(),
        task="add 17 and 23",
        agent_name="example",
        agent_version="0.0.0",
        model_id="example/model",
        started_at=started,
        finished_at=started + timedelta(seconds=1),
        final_status=TrajectoryStatus.SUCCEEDED,
        final_answer=answer,
    )
    llm = Step(
        id=new_id(),
        trajectory_id=traj.id,
        parent_step_id=None,
        name="llm",
        started_at=started,
        finished_at=started + timedelta(milliseconds=5),
        status=StepStatus.SUCCEEDED,
        payload=LLMCallPayload(
            model_id="example/model",
            prompt_messages=[Message(role="user", content="hi")],
            completion=plan,
            input_tokens=10,
            output_tokens=10,
            latency_ms=5.0,
            cost_usd=0.0,
        ),
    )
    tool = Step(
        id=new_id(),
        trajectory_id=traj.id,
        parent_step_id=None,
        name="tool:calc",
        started_at=started + timedelta(milliseconds=10),
        finished_at=started + timedelta(milliseconds=15),
        status=StepStatus.SUCCEEDED,
        payload=ToolCallPayload(
            tool_name="calc",
            arguments={},
            result=None,
            latency_ms=5.0,
        ),
    )
    return traj, [llm, tool]


async def main() -> None:
    judge = StubJudge(
        JudgeVerdict(score=0.8, label="pass", rationale="clear plan, executable"),
    )
    metrics: list[Metric | AsyncMetric] = [
        FinalAnswerMatch(),
        ToolAccuracy(),
        StepEfficiency(),
        PlanQuality(judge),
    ]
    runner = Runner(metrics=metrics, seed=0, n_resamples=500, concurrency=4)

    pairs = []
    for i, (ans, plan) in enumerate(
        [
            ("40", "I'll use the calculator on 17+23."),
            ("forty", "Just guessing."),
            ("40", "Plan: calc(17+23)."),
        ]
    ):
        traj, steps = _make_traj(ans, plan)
        case = Case(
            case_id=f"c{i}",
            task="add 17 and 23",
            expected_answer="40",
            expected_tools=(ExpectedTool(name="calc"),),
            expected_max_steps=2,
        )
        pairs.append((traj, steps, case))

    report = await runner.aevaluate(pairs)

    print(f"n_cases = {report.n_cases}")
    for name, ci in report.aggregates.items():
        print(f"  {name:22s} mean={ci.mean:.3f}  95% CI=[{ci.lo:.3f}, {ci.hi:.3f}]  n={ci.n}")


if __name__ == "__main__":
    asyncio.run(main())
