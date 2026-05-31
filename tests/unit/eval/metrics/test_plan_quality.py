"""Tests for PlanQuality (async metric, composes any Judge)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ariadne_eval.core.ids import new_id
from ariadne_eval.core.status import StepStatus
from ariadne_eval.core.trajectory import LLMCallPayload, Message, Step
from ariadne_eval.eval.case import Case
from ariadne_eval.eval.judges.base import JudgeVerdict
from ariadne_eval.eval.judges.stub import StubJudge
from ariadne_eval.eval.metrics.plan_quality import PlanQuality
from tests.unit.eval._factories import make_trajectory

pytestmark = pytest.mark.fast


def _llm_step(trajectory_id: str, completion: str) -> Step:
    t0 = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)
    return Step(
        id=new_id(),
        trajectory_id=trajectory_id,
        parent_step_id=None,
        name="llm_call",
        started_at=t0,
        finished_at=t0 + timedelta(milliseconds=10),
        status=StepStatus.SUCCEEDED,
        payload=LLMCallPayload(
            model_id="m",
            prompt_messages=[Message(role="user", content="hi")],
            completion=completion,
            input_tokens=10,
            output_tokens=10,
            latency_ms=10.0,
            cost_usd=0.0,
        ),
    )


async def test_plan_quality_pass() -> None:
    judge = StubJudge(
        JudgeVerdict(
            score=1.0,
            label="pass",
            rationale="clear plan",
            raw={"clarity": 5},
        )
    )
    metric = PlanQuality(judge)
    assert metric.name == "plan_quality"
    traj = make_trajectory(task="t")
    steps = [_llm_step(traj.id, "first plan")]
    case = Case(case_id="c", task="t")
    result = await metric.ascore(traj, steps, case)
    assert result.score == 1.0
    assert result.label == "pass"
    assert result.metric == "plan_quality"
    assert result.case_id == "c"
    assert result.trajectory_id == traj.id
    assert result.details["rationale"] == "clear plan"
    assert result.details["raw"] == {"clarity": 5}


async def test_plan_quality_fail() -> None:
    judge = StubJudge(JudgeVerdict(score=0.0, label="fail", rationale="bad"))
    metric = PlanQuality(judge)
    traj = make_trajectory()
    steps = [_llm_step(traj.id, "x")]
    case = Case(case_id="c", task="t")
    result = await metric.ascore(traj, steps, case)
    assert result.score == 0.0
    assert result.label == "fail"


async def test_plan_quality_no_llm_step_is_immediate_fail() -> None:
    """No LLMCallPayload step → score 0.0, label fail, judge is NOT called."""

    def boom(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("judge must not be called when no llm step")

    judge = StubJudge(boom, name="boom")
    metric = PlanQuality(judge)
    traj = make_trajectory()
    case = Case(case_id="c", task="t")
    result = await metric.ascore(traj, [], case)
    assert result.score == 0.0
    assert result.label == "fail"
    assert result.details["reason"] == "no_llm_step"


async def test_plan_quality_with_case_none_uses_synthetic_id() -> None:
    """The Runner always passes a Case; standalone use without one falls back."""
    judge = StubJudge(JudgeVerdict(score=0.5, label="partial", rationale="ok"))
    metric = PlanQuality(judge)
    traj = make_trajectory()
    steps = [_llm_step(traj.id, "p")]
    result = await metric.ascore(traj, steps, None)
    assert result.case_id == "<none>"
    assert result.score == 0.5
