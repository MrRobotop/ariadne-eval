"""Tests for FinalAnswerMatch metric."""

from __future__ import annotations

import pytest

from ariadne_eval.core.status import TrajectoryStatus
from ariadne_eval.eval.case import Case
from ariadne_eval.eval.errors import MissingReferenceError
from ariadne_eval.eval.metrics.final_answer import FinalAnswerMatch
from tests.unit.eval._factories import make_trajectory

pytestmark = pytest.mark.fast


def test_normalized_exact_pass() -> None:
    traj = make_trajectory(final_answer="  4  ")
    case = Case(case_id="c", task="2+2", expected_answer="4")
    r = FinalAnswerMatch().score(traj, [], case)
    assert r.score == 1.0
    assert r.label == "pass"
    assert r.metric == "final_answer_match"


def test_normalized_exact_collapses_internal_whitespace() -> None:
    traj = make_trajectory(final_answer="The   answer  is   four")
    case = Case(case_id="c", task="t", expected_answer="the answer is four")
    r = FinalAnswerMatch().score(traj, [], case)
    assert r.score == 1.0


def test_normalized_exact_fail() -> None:
    traj = make_trajectory(final_answer="five")
    case = Case(case_id="c", task="t", expected_answer="four")
    r = FinalAnswerMatch().score(traj, [], case)
    assert r.score == 0.0
    assert r.label == "fail"


def test_exact_mode_distinguishes_case() -> None:
    traj = make_trajectory(final_answer="Hello")
    case = Case(case_id="c", task="t", expected_answer="hello")
    r = FinalAnswerMatch(comparator="exact").score(traj, [], case)
    assert r.score == 0.0


def test_custom_comparator_partial() -> None:
    def half(a: str, b: str) -> float:
        return 0.5

    traj = make_trajectory(final_answer="x")
    case = Case(case_id="c", task="t", expected_answer="y")
    r = FinalAnswerMatch(comparator=half).score(traj, [], case)
    assert r.score == 0.5
    assert r.label == "partial"


def test_missing_reference_raises() -> None:
    traj = make_trajectory(final_answer="x")
    case = Case(case_id="c", task="t")
    with pytest.raises(MissingReferenceError):
        FinalAnswerMatch().score(traj, [], case)


def test_no_final_answer_is_fail() -> None:
    traj = make_trajectory(
        final_answer=None, final_status=TrajectoryStatus.FAILED
    )
    case = Case(case_id="c", task="t", expected_answer="x")
    r = FinalAnswerMatch().score(traj, [], case)
    assert r.score == 0.0
    assert r.label == "fail"
    assert r.details["reason"] == "no_final_answer"


def test_non_string_final_answer_is_json_serialized() -> None:
    traj = make_trajectory(final_answer={"value": 4})
    case = Case(
        case_id="c", task="t", expected_answer='{"value": 4}'
    )
    r = FinalAnswerMatch(comparator="exact").score(traj, [], case)
    assert r.score == 1.0
