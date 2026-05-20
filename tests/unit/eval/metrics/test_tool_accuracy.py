"""Tests for ToolAccuracy metric."""

from __future__ import annotations

import pytest

from ariadne_eval.eval.case import Case, ExpectedTool
from ariadne_eval.eval.metrics.tool_accuracy import ToolAccuracy
from tests.unit.eval._factories import make_tool_step, make_trajectory

pytestmark = pytest.mark.fast


def _scenario(
    *, expected_names: list[str], actual_names: list[str]
) -> tuple[object, list[object], object]:
    traj = make_trajectory()
    steps = [make_tool_step(trajectory_id=traj.id, tool_name=n) for n in actual_names]
    case = Case(
        case_id="c",
        task="t",
        expected_tools=tuple(ExpectedTool(name=n) for n in expected_names),
    )
    return traj, steps, case


def test_set_mode_perfect_match() -> None:
    traj, steps, case = _scenario(expected_names=["a", "b"], actual_names=["a", "b"])
    r = ToolAccuracy().score(traj, steps, case)  # type: ignore[arg-type]
    assert r.score == 1.0
    assert r.label == "pass"
    assert r.details["precision"] == 1.0
    assert r.details["recall"] == 1.0


def test_set_mode_partial_f1() -> None:
    # expected = {a, b}, actual = {a, c} => tp=1, fp=1, fn=1 => P=R=0.5, F1=0.5
    traj, steps, case = _scenario(expected_names=["a", "b"], actual_names=["a", "c"])
    r = ToolAccuracy().score(traj, steps, case)  # type: ignore[arg-type]
    assert r.score == 0.5
    assert r.label == "partial"
    assert r.details["matched"] == ["a"]
    assert r.details["missing"] == ["b"]
    assert r.details["extra"] == ["c"]


def test_set_mode_treats_duplicates_as_multiset() -> None:
    # expected = [a, a, b], actual = [a, b] => tp=2, fn=1, fp=0
    # P = 2/2 = 1.0, R = 2/3 ≈ 0.667, F1 = 0.8
    traj, steps, case = _scenario(expected_names=["a", "a", "b"], actual_names=["a", "b"])
    r = ToolAccuracy().score(traj, steps, case)  # type: ignore[arg-type]
    assert r.score == pytest_approx(0.8)


def test_ordered_prefix_full_match() -> None:
    traj, steps, case = _scenario(expected_names=["a", "b", "c"], actual_names=["a", "b", "c", "d"])
    r = ToolAccuracy(mode="ordered_prefix").score(traj, steps, case)  # type: ignore[arg-type]
    assert r.score == 1.0
    assert r.details["prefix_length"] == 3


def test_ordered_prefix_partial() -> None:
    traj, steps, case = _scenario(expected_names=["a", "b", "c"], actual_names=["a", "x", "c"])
    r = ToolAccuracy(mode="ordered_prefix").score(traj, steps, case)  # type: ignore[arg-type]
    assert r.score == pytest_approx(1 / 3)
    assert r.details["first_divergence_index"] == 1


def test_match_args_true_strict() -> None:
    traj = make_trajectory()
    steps = [make_tool_step(trajectory_id=traj.id, tool_name="search", arguments={"q": "x"})]
    case = Case(
        case_id="c",
        task="t",
        expected_tools=(ExpectedTool(name="search", args={"q": "x"}),),
    )
    r = ToolAccuracy(match_args=True).score(traj, steps, case)
    assert r.score == 1.0


def test_match_args_true_args_mismatch() -> None:
    traj = make_trajectory()
    steps = [make_tool_step(trajectory_id=traj.id, tool_name="search", arguments={"q": "x"})]
    case = Case(
        case_id="c",
        task="t",
        expected_tools=(ExpectedTool(name="search", args={"q": "y"}),),
    )
    r = ToolAccuracy(match_args=True).score(traj, steps, case)
    assert r.score == 0.0


def test_match_args_true_with_args_none_is_per_tool_wildcard() -> None:
    traj = make_trajectory()
    steps = [make_tool_step(trajectory_id=traj.id, tool_name="search", arguments={"q": "x"})]
    case = Case(
        case_id="c",
        task="t",
        expected_tools=(ExpectedTool(name="search", args=None),),
    )
    r = ToolAccuracy(match_args=True).score(traj, steps, case)
    assert r.score == 1.0


def test_empty_expected_set_mode_with_extras_fails() -> None:
    traj = make_trajectory()
    steps = [make_tool_step(trajectory_id=traj.id, tool_name="a")]
    case = Case(case_id="c", task="t", expected_tools=())
    r = ToolAccuracy().score(traj, steps, case)
    assert r.score == 0.0
    assert r.label == "fail"


def test_empty_expected_no_extras_passes() -> None:
    traj = make_trajectory()
    case = Case(case_id="c", task="t", expected_tools=())
    r = ToolAccuracy().score(traj, [], case)
    assert r.score == 1.0
    assert r.label == "pass"


def test_empty_expected_ordered_prefix_quirk_documented() -> None:
    # Documented quirk: empty prefix matches everything in ordered_prefix mode.
    traj = make_trajectory()
    steps = [make_tool_step(trajectory_id=traj.id, tool_name="a")]
    case = Case(case_id="c", task="t", expected_tools=())
    r = ToolAccuracy(mode="ordered_prefix").score(traj, steps, case)
    assert r.score == 1.0


# tiny local approx (avoid pytest.approx import noise)
def pytest_approx(value: float, tol: float = 1e-9) -> object:
    class _A:
        def __eq__(self, other: object) -> bool:
            return isinstance(other, float) and abs(other - value) <= tol

        def __repr__(self) -> str:
            return f"approx({value})"

    return _A()
