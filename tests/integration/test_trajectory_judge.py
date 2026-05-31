"""End-to-end TrajectoryJudge through litellm via a hand-crafted VCR cassette.

The cassette is hand-edited (record_mode='none' in conftest); no real API
call is ever made. Re-record by rerunning with --record-mode=rewrite and
real keys after deliberate prompt changes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ariadne_eval.core.ids import new_id
from ariadne_eval.core.status import StepStatus
from ariadne_eval.core.trajectory import LLMCallPayload, Message, Step
from ariadne_eval.eval.case import Case
from ariadne_eval.eval.judges.trajectory_judge import TrajectoryJudge
from tests.unit.eval._factories import make_trajectory


def _llm_step(traj_id: str, completion: str) -> Step:
    t0 = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)
    return Step(
        id=new_id(),
        trajectory_id=traj_id,
        parent_step_id=None,
        name="llm_call",
        started_at=t0,
        finished_at=t0 + timedelta(milliseconds=10),
        status=StepStatus.SUCCEEDED,
        payload=LLMCallPayload(
            model_id="gpt-4o-mini",
            prompt_messages=[Message(role="user", content="hi")],
            completion=completion,
            input_tokens=10,
            output_tokens=10,
            latency_ms=10.0,
            cost_usd=0.0,
        ),
    )


@pytest.mark.integration
@pytest.mark.vcr
async def test_trajectory_judge_end_to_end() -> None:
    judge = TrajectoryJudge(model="gpt-4o-mini")
    traj = make_trajectory(task="add 17 and 23")
    steps = [_llm_step(traj.id, "I will use the calculator on 17 + 23.")]
    case = Case(case_id="c1", task="add 17 and 23")
    verdict = await judge.judge(traj, steps, case)
    assert verdict.label == "pass"
    assert verdict.score == 0.8
    assert "calculator" in verdict.rationale.lower()
