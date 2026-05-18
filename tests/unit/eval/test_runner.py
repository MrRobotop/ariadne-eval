"""Tests for Runner and EvalReport."""

from __future__ import annotations

from pathlib import Path

import pytest

from ariadne_eval.eval.case import Case, ExpectedTool
from ariadne_eval.eval.errors import BootstrapInsufficientDataWarning, MissingReferenceError
from ariadne_eval.eval.metrics.efficiency import StepEfficiency
from ariadne_eval.eval.metrics.final_answer import FinalAnswerMatch
from ariadne_eval.eval.metrics.tool_accuracy import ToolAccuracy
from ariadne_eval.eval.runner import EvalReport, Runner
from tests.unit.eval._factories import make_tool_step, make_trajectory

pytestmark = pytest.mark.fast


def _three_pairs() -> list[tuple[object, list[object], Case]]:
    pairs: list[tuple[object, list[object], Case]] = []
    for i, (ans, tools, budget) in enumerate(
        [("4", ["calc"], 2), ("5", ["calc"], 2), ("4", ["calc", "search"], 2)]
    ):
        traj = make_trajectory(final_answer=ans)
        steps = [make_tool_step(trajectory_id=traj.id, tool_name=t) for t in tools]
        case = Case(
            case_id=f"c{i}",
            task="t",
            expected_answer="4",
            expected_tools=(ExpectedTool(name="calc"),),
            expected_max_steps=budget,
        )
        pairs.append((traj, steps, case))
    return pairs


def test_runner_evaluates_three_metrics_with_aggregates() -> None:
    runner = Runner(
        metrics=[FinalAnswerMatch(), ToolAccuracy(), StepEfficiency()],
        seed=0,
        n_resamples=200,
    )
    report = runner.evaluate(_three_pairs())  # type: ignore[arg-type]
    assert isinstance(report, EvalReport)
    assert report.n_cases == 3
    assert report.seed == 0
    # 3 cases x 3 metrics
    assert len(report.results) == 9
    assert set(report.aggregates) == {
        "final_answer_match",
        "tool_accuracy",
        "step_efficiency",
    }
    # final_answer_match: 2/3 pass => mean ≈ 0.667
    fa = report.aggregates["final_answer_match"]
    assert abs(fa.mean - 2 / 3) < 1e-9
    assert fa.n == 3


def test_runner_skip_on_missing_reference() -> None:
    traj = make_trajectory(final_answer="x")
    case_with = Case(case_id="c1", task="t", expected_answer="x")
    case_without = Case(case_id="c2", task="t")  # no expected_answer
    runner = Runner(metrics=[FinalAnswerMatch()], seed=0, n_resamples=100)
    with pytest.warns(BootstrapInsufficientDataWarning):
        report = runner.evaluate(
            [(traj, [], case_with), (traj, [], case_without)],
        )
    # Only c1 produced a result
    assert len(report.results) == 1
    assert report.results[0].case_id == "c1"
    assert report.aggregates["final_answer_match"].n == 1


def test_runner_error_on_missing_reference() -> None:
    traj = make_trajectory(final_answer="x")
    case_without = Case(case_id="c", task="t")
    runner = Runner(metrics=[FinalAnswerMatch()], on_missing_reference="error")
    with pytest.raises(MissingReferenceError):
        runner.evaluate([(traj, [], case_without)])


def test_eval_report_jsonl_round_trip(tmp_path: Path) -> None:
    runner = Runner(
        metrics=[FinalAnswerMatch(), StepEfficiency()],
        seed=3,
        n_resamples=200,
    )
    report = runner.evaluate(_three_pairs())  # type: ignore[arg-type]
    out = tmp_path / "report.jsonl"
    report.to_jsonl(out)
    loaded = EvalReport.from_jsonl(out)
    assert loaded == report
