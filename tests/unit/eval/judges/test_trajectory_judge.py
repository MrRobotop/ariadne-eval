"""Tests for TrajectoryJudge using an injected stub client (no LLM)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ariadne_eval.core.ids import new_id
from ariadne_eval.core.status import StepStatus
from ariadne_eval.core.trajectory import LLMCallPayload, Message, Step
from ariadne_eval.eval.case import Case
from ariadne_eval.eval.judges.base import JudgeParseError, JudgeVerdict
from ariadne_eval.eval.judges.prompts import (
    PLAN_QUALITY_SYSTEM,
    PLAN_QUALITY_USER_TEMPLATE,  # noqa: F401  (import asserts symbol is exposed)
)
from ariadne_eval.eval.judges.trajectory_judge import TrajectoryJudge, _extract_plan
from tests.unit.eval._factories import make_trajectory

pytestmark = pytest.mark.fast


def _llm_step(trajectory_id: str, completion: str, *, started_at: datetime) -> Step:
    return Step(
        id=new_id(),
        trajectory_id=trajectory_id,
        parent_step_id=None,
        name="llm_call",
        started_at=started_at,
        finished_at=started_at + timedelta(milliseconds=10),
        status=StepStatus.SUCCEEDED,
        payload=LLMCallPayload(
            model_id="test/model",
            prompt_messages=[Message(role="user", content="hi")],
            completion=completion,
            input_tokens=10,
            output_tokens=10,
            latency_ms=10.0,
            cost_usd=0.0,
        ),
    )


def test_extract_plan_picks_first_llm_completion() -> None:
    traj = make_trajectory()
    t0 = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)
    steps = [
        _llm_step(traj.id, "first plan text", started_at=t0),
        _llm_step(traj.id, "second", started_at=t0 + timedelta(seconds=1)),
    ]
    assert _extract_plan(steps) == "first plan text"


def test_extract_plan_returns_none_when_no_llm_step() -> None:
    assert _extract_plan([]) is None


async def test_trajectory_judge_with_stub_client_happy() -> None:
    captured: dict[str, object] = {}

    async def stub_client(*, model: str, messages: list[dict[str, str]], temperature: float) -> str:
        captured["model"] = model
        captured["messages"] = messages
        captured["temperature"] = temperature
        return "SCORE: 0.7\nLABEL: pass\nRATIONALE: good plan"

    judge = TrajectoryJudge(model="test/model", client=stub_client)
    traj = make_trajectory(task="ship phase 6")
    t0 = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)
    steps = [_llm_step(traj.id, "Step 1: do X. Step 2: do Y.", started_at=t0)]
    case = Case(case_id="c", task="ship phase 6")
    verdict = await judge.judge(traj, steps, case)
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.score == 0.7
    assert verdict.label == "pass"
    assert captured["model"] == "test/model"
    assert captured["temperature"] == 0.0
    msgs = captured["messages"]
    assert isinstance(msgs, list) and len(msgs) == 2
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == PLAN_QUALITY_SYSTEM
    assert msgs[1]["role"] == "user"
    assert "ship phase 6" in msgs[1]["content"]
    assert "Step 1: do X" in msgs[1]["content"]


async def test_trajectory_judge_propagates_parse_error() -> None:
    async def bad_stub(*, model, messages, temperature):  # type: ignore[no-untyped-def]
        return "garbage with no fields"

    judge = TrajectoryJudge(model="test/model", client=bad_stub)
    traj = make_trajectory()
    t0 = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)
    steps = [_llm_step(traj.id, "plan", started_at=t0)]
    case = Case(case_id="c", task="t")
    with pytest.raises(JudgeParseError):
        await judge.judge(traj, steps, case)


async def test_trajectory_judge_empty_plan_when_no_llm_step() -> None:
    """The judge still runs but the rendered prompt's {plan} is the empty placeholder."""
    captured: dict[str, object] = {}

    async def stub_client(*, model, messages, temperature):  # type: ignore[no-untyped-def]
        captured["messages"] = messages
        return "SCORE: 0.0\nLABEL: fail\nRATIONALE: no plan"

    judge = TrajectoryJudge(model="test/model", client=stub_client)
    traj = make_trajectory(task="t")
    case = Case(case_id="c", task="t")
    verdict = await judge.judge(traj, [], case)
    assert verdict.label == "fail"
    user_msg = captured["messages"][1]["content"]  # type: ignore[index]
    assert "no plan recorded" in user_msg
