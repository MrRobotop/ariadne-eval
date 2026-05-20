"""Tests for StepEfficiency metric."""

from __future__ import annotations

import pytest

from ariadne_eval.eval.case import Case
from ariadne_eval.eval.errors import MissingReferenceError
from ariadne_eval.eval.metrics.efficiency import StepEfficiency
from tests.unit.eval._factories import make_tool_step, make_trajectory

pytestmark = pytest.mark.fast


def test_under_budget_pass() -> None:
    """Single step under a 3-step budget scores 1.0 / pass."""
    traj = make_trajectory()
    steps = [make_tool_step(trajectory_id=traj.id, tool_name="a")]
    case = Case(case_id="c", task="t", expected_max_steps=3)
    r = StepEfficiency().score(traj, steps, case)
    assert r.score == 1.0
    assert r.label == "pass"
    assert r.details == {"actual_steps": 1, "expected_max_steps": 3}


def test_at_budget_pass() -> None:
    """Exactly at budget scores 1.0 / pass."""
    traj = make_trajectory()
    steps = [
        make_tool_step(trajectory_id=traj.id, tool_name="a"),
        make_tool_step(trajectory_id=traj.id, tool_name="b"),
        make_tool_step(trajectory_id=traj.id, tool_name="c"),
    ]
    case = Case(case_id="c", task="t", expected_max_steps=3)
    r = StepEfficiency().score(traj, steps, case)
    assert r.score == 1.0
    assert r.label == "pass"


def test_over_budget_partial() -> None:
    """4 steps against a budget of 2 yields score 0.5 / partial."""
    traj = make_trajectory()
    steps = [make_tool_step(trajectory_id=traj.id, tool_name=n) for n in "abcd"]
    case = Case(case_id="c", task="t", expected_max_steps=2)
    r = StepEfficiency().score(traj, steps, case)
    assert r.score == 0.5
    assert r.label == "partial"


def test_zero_steps_with_budget() -> None:
    """0 actual steps clamps denominator to 1, yielding score 1.0 / pass."""
    traj = make_trajectory()
    case = Case(case_id="c", task="t", expected_max_steps=2)
    r = StepEfficiency().score(traj, [], case)
    # 0 actual steps => max(actual,1)=1 => score = min(1, 2/1) = 1.0, pass
    assert r.score == 1.0
    assert r.label == "pass"


def test_missing_reference_raises() -> None:
    """Missing expected_max_steps raises MissingReferenceError."""
    traj = make_trajectory()
    case = Case(case_id="c", task="t")
    with pytest.raises(MissingReferenceError):
        StepEfficiency().score(traj, [], case)
