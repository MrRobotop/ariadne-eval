"""Tests for Runner and EvalReport."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from ariadne_eval.eval.case import Case, ExpectedTool
from ariadne_eval.eval.errors import BootstrapInsufficientDataWarning, MissingReferenceError
from ariadne_eval.eval.metrics.efficiency import StepEfficiency
from ariadne_eval.eval.metrics.final_answer import FinalAnswerMatch
from ariadne_eval.eval.metrics.tool_accuracy import ToolAccuracy
from ariadne_eval.eval.runner import EvalReport, Runner
from ariadne_eval.eval.stats.bootstrap import BootstrapCI
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
    # Two input items were presented to the runner …
    assert report.n_cases == 2
    # … but only c1 produced a scored result (c2 was skipped due to missing reference)
    assert len(report.results) == 1
    assert report.results[0].case_id == "c1"
    # aggregate.n reflects only scored cases, not total input items
    assert report.aggregates["final_answer_match"].n == 1


def test_runner_error_on_missing_reference_non_first_metric() -> None:
    """Error path escapes cleanly even when an earlier metric already succeeded."""
    traj = make_trajectory(final_answer="42")
    # Case has expected_max_steps so StepEfficiency succeeds, but NO expected_answer
    # so FinalAnswerMatch raises MissingReferenceError.
    case = Case(case_id="c", task="t", expected_max_steps=5)
    runner = Runner(
        metrics=[StepEfficiency(), FinalAnswerMatch()],
        on_missing_reference="error",
    )
    with pytest.raises(MissingReferenceError):
        runner.evaluate([(traj, [], case)])


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


def test_eval_report_jsonl_handles_nan_aggregates(tmp_path: Path) -> None:
    """NaN floats in BootstrapCI survive a JSONL round-trip via null."""
    # Hand-craft a report whose aggregate has NaN fields (mirrors n=0 case)
    nan_ci = BootstrapCI(
        mean=math.nan,
        lo=math.nan,
        hi=math.nan,
        n=0,
        n_resamples=1000,
        confidence=0.95,
    )
    report = EvalReport(
        results=(),
        aggregates={"some_metric": nan_ci},
        n_cases=0,
        seed=0,
    )
    out = tmp_path / "nan_report.jsonl"
    report.to_jsonl(out)
    raw = out.read_text(encoding="utf-8")
    # MUST be valid RFC-8259 JSON — null, not bare NaN
    assert "NaN" not in raw
    assert "null" in raw
    # Round-trip preserves NaN via null
    loaded = EvalReport.from_jsonl(out)
    assert loaded.n_cases == 0
    ci = loaded.aggregates["some_metric"]
    assert math.isnan(ci.mean)
    assert math.isnan(ci.lo)
    assert math.isnan(ci.hi)
    assert ci.n == 0
    assert ci.n_resamples == 1000
