"""Tests for the Judge Protocol, JudgeVerdict, and JudgeParseError."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ariadne_eval.eval.judges.base import Judge, JudgeParseError, JudgeVerdict

pytestmark = pytest.mark.fast


def test_judge_verdict_minimal() -> None:
    v = JudgeVerdict(score=0.75, label="partial", rationale="ok")
    assert v.raw == {}


def test_judge_verdict_full() -> None:
    v = JudgeVerdict(
        score=1.0,
        label="pass",
        rationale="great",
        raw={"clarity": 5, "decomposition": 4},
    )
    assert v.raw["clarity"] == 5


def test_judge_verdict_is_frozen() -> None:
    v = JudgeVerdict(score=0.0, label="fail", rationale="x")
    with pytest.raises(ValidationError):
        v.score = 0.5  # type: ignore[misc]


def test_judge_verdict_label_literal_validated() -> None:
    with pytest.raises(ValidationError):
        JudgeVerdict(score=0.5, label="bogus", rationale="x")  # type: ignore[arg-type]


def test_judge_parse_error_is_value_error() -> None:
    assert issubclass(JudgeParseError, ValueError)


def test_judge_protocol_runtime_checkable() -> None:
    class _OkJudge:
        name = "ok"

        async def judge(self, trajectory, steps, case):  # type: ignore[no-untyped-def]
            return JudgeVerdict(score=0.5, label="partial", rationale="x")

    assert isinstance(_OkJudge(), Judge)

    class _NotJudge:
        name = "no_judge"

    assert not isinstance(_NotJudge(), Judge)
