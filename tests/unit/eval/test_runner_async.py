"""Tests for Runner.aevaluate — async dispatch + ordering + concurrency."""

from __future__ import annotations

import asyncio

import pytest

from ariadne_eval.eval.case import Case
from ariadne_eval.eval.errors import MissingReferenceError
from ariadne_eval.eval.metrics.base import MetricResult
from ariadne_eval.eval.metrics.final_answer import FinalAnswerMatch
from ariadne_eval.eval.runner import EvalReport, Runner
from tests.unit.eval._factories import make_trajectory

pytestmark = pytest.mark.fast


class _AsyncStubMetric:
    """Async-only metric returning a fixed score, with a sleep to test ordering."""

    def __init__(self, name: str, score: float, *, sleep: float = 0.0) -> None:
        self.name = name
        self._score = score
        self._sleep = sleep

    async def ascore(self, trajectory, steps, case) -> MetricResult:  # type: ignore[no-untyped-def]
        if self._sleep:
            await asyncio.sleep(self._sleep)
        return MetricResult(
            metric=self.name,
            case_id=case.case_id,
            trajectory_id=trajectory.id,
            score=self._score,
            label="pass" if self._score >= 0.99 else "fail",
        )


class _ConcurrencyRecorder:
    """Async metric that records the max in-flight count it observes."""

    def __init__(self, name: str = "recorder") -> None:
        self.name = name
        self._in_flight = 0
        self.max_observed = 0
        self._lock = asyncio.Lock()

    async def ascore(self, trajectory, steps, case) -> MetricResult:  # type: ignore[no-untyped-def]
        async with self._lock:
            self._in_flight += 1
            self.max_observed = max(self.max_observed, self._in_flight)
        await asyncio.sleep(0.01)
        async with self._lock:
            self._in_flight -= 1
        return MetricResult(
            metric=self.name,
            case_id=case.case_id,
            trajectory_id=trajectory.id,
            score=1.0,
            label="pass",
        )


def _three_items():
    items = []
    for i, ans in enumerate(["4", "4", "5"]):
        traj = make_trajectory(final_answer=ans)
        case = Case(case_id=f"c{i}", task="t", expected_answer="4")
        items.append((traj, [], case))
    return items


async def test_aevaluate_mix_of_sync_and_async() -> None:
    """Sync FinalAnswerMatch + async stub — both kinds composable, order preserved."""
    runner = Runner(
        metrics=[FinalAnswerMatch(), _AsyncStubMetric("async_one", 0.5)],
        seed=0,
        n_resamples=100,
    )
    report = await runner.aevaluate(_three_items())
    assert isinstance(report, EvalReport)
    assert report.n_cases == 3
    # 3 cases x 2 metrics
    assert len(report.results) == 6
    # Deterministic per-(case, metric) order, regardless of async completion
    expected = [
        ("final_answer_match", "c0"),
        ("async_one", "c0"),
        ("final_answer_match", "c1"),
        ("async_one", "c1"),
        ("final_answer_match", "c2"),
        ("async_one", "c2"),
    ]
    actual = [(r.metric, r.case_id) for r in report.results]
    assert actual == expected
    # Aggregates present for both
    assert set(report.aggregates) == {"final_answer_match", "async_one"}


async def test_aevaluate_order_preserved_under_variable_latency() -> None:
    """Slow first item must not push its results after faster later items."""
    slow = _AsyncStubMetric("slow", 1.0, sleep=0.05)
    fast = _AsyncStubMetric("fast", 0.0, sleep=0.001)
    runner = Runner(metrics=[slow, fast], seed=0, n_resamples=100)
    report = await runner.aevaluate(_three_items())
    # Order is (item0,slow), (item0,fast), (item1,slow), (item1,fast), ...
    expected = [
        ("slow", "c0"),
        ("fast", "c0"),
        ("slow", "c1"),
        ("fast", "c1"),
        ("slow", "c2"),
        ("fast", "c2"),
    ]
    actual = [(r.metric, r.case_id) for r in report.results]
    assert actual == expected


async def test_aevaluate_concurrency_bound() -> None:
    """Semaphore caps in-flight async metric tasks at `concurrency`."""
    recorder = _ConcurrencyRecorder()
    runner = Runner(metrics=[recorder], seed=0, n_resamples=100, concurrency=3)
    # 10 items x 1 metric = 10 async tasks; max in-flight must be <= 3
    items = [(make_trajectory(), [], Case(case_id=f"c{i}", task="t")) for i in range(10)]
    await runner.aevaluate(items)
    assert recorder.max_observed <= 3
    assert recorder.max_observed >= 1  # at least one ran


async def test_aevaluate_skip_on_missing_async() -> None:
    """Async metric raising MissingReferenceError is skipped under default policy."""

    class _NeedsRef:
        name = "needs_ref"

        async def ascore(self, trajectory, steps, case):  # type: ignore[no-untyped-def]
            if case.expected_answer is None:
                raise MissingReferenceError("expected_answer", case_id=case.case_id)
            return MetricResult(
                metric=self.name,
                case_id=case.case_id,
                trajectory_id=trajectory.id,
                score=1.0,
                label="pass",
            )

    runner = Runner(metrics=[_NeedsRef()], seed=0, n_resamples=100)
    traj = make_trajectory()
    case_with = Case(case_id="c1", task="t", expected_answer="x")
    case_without = Case(case_id="c2", task="t")
    with pytest.warns():  # bootstrap may warn with n=1
        report = await runner.aevaluate([(traj, [], case_with), (traj, [], case_without)])
    assert len(report.results) == 1
    assert report.results[0].case_id == "c1"
    assert report.n_cases == 2
    assert report.aggregates["needs_ref"].n == 1


async def test_aevaluate_error_on_missing_async() -> None:
    """Async metric raising MissingReferenceError propagates under error mode."""

    class _NeedsRef:
        name = "needs_ref"

        async def ascore(self, trajectory, steps, case):  # type: ignore[no-untyped-def]
            raise MissingReferenceError("expected_answer", case_id=case.case_id)

    runner = Runner(
        metrics=[_NeedsRef()],
        seed=0,
        n_resamples=100,
        on_missing_reference="error",
    )
    traj = make_trajectory()
    case = Case(case_id="c", task="t")
    with pytest.raises(MissingReferenceError):
        await runner.aevaluate([(traj, [], case)])
