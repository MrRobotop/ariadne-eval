"""Tests for the deterministic StubJudge."""

from __future__ import annotations

import pytest

from ariadne_eval.eval.case import Case
from ariadne_eval.eval.judges.base import JudgeVerdict
from ariadne_eval.eval.judges.stub import StubJudge
from tests.unit.eval._factories import make_trajectory

pytestmark = pytest.mark.fast


async def test_stub_with_fixed_verdict() -> None:
    v = JudgeVerdict(score=1.0, label="pass", rationale="fixed")
    judge = StubJudge(v)
    traj = make_trajectory()
    case = Case(case_id="c", task="t")
    result = await judge.judge(traj, [], case)
    assert result == v
    assert judge.name == "stub"


async def test_stub_with_callable_verdict() -> None:
    def verdict_fn(trajectory, steps, case):  # type: ignore[no-untyped-def]
        return JudgeVerdict(
            score=0.5,
            label="partial",
            rationale=f"case={case.case_id}",
        )

    judge = StubJudge(verdict_fn, name="callable_stub")
    traj = make_trajectory()
    case = Case(case_id="x42", task="t")
    result = await judge.judge(traj, [], case)
    assert result.rationale == "case=x42"
    assert judge.name == "callable_stub"


async def test_stub_accepts_none_case() -> None:
    v = JudgeVerdict(score=0.0, label="fail", rationale="no case")
    judge = StubJudge(v)
    traj = make_trajectory()
    result = await judge.judge(traj, [], None)
    assert result == v
