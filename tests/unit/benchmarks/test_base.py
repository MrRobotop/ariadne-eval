"""Tests for the Benchmark Protocol + dataclasses."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import pytest

from ariadne_eval.benchmarks.base import (
    Benchmark,
    BenchmarkRunResult,
    BenchmarkTask,
)
from ariadne_eval.core.trajectory import Step, Trajectory  # noqa: F401  (used in type hints)
from ariadne_eval.storage.base import Store  # noqa: F401

pytestmark = pytest.mark.fast


def test_benchmark_task_minimal() -> None:
    t = BenchmarkTask(task_id="t1", task_index=0, instruction="do X", payload={"raw": 1})
    assert t.task_id == "t1"
    assert t.task_index == 0
    assert t.instruction == "do X"
    assert t.payload == {"raw": 1}


def test_benchmark_task_is_frozen() -> None:
    t = BenchmarkTask(task_id="t1", task_index=0, instruction="i", payload=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.task_id = "t2"  # type: ignore[misc]


def test_benchmark_run_result_minimal() -> None:
    r = BenchmarkRunResult(trajectory_id="01J0", success=True, raw_score=1.0)
    assert r.trajectory_id == "01J0"
    assert r.success is True
    assert r.raw_score == 1.0
    assert r.error is None


def test_benchmark_run_result_with_error() -> None:
    r = BenchmarkRunResult(
        trajectory_id="01J0",
        success=False,
        raw_score=0.0,
        error="provider timeout after 4 retries",
    )
    assert r.error == "provider timeout after 4 retries"


def test_benchmark_protocol_runtime_checkable() -> None:
    class _OkBenchmark:
        name = "ok"

        def tasks(
            self, *, split: str = "test", limit: int | None = None
        ) -> Sequence[BenchmarkTask]:
            return []

        async def run_task(
            self,
            task: BenchmarkTask,
            model: str,
            provider: str,
            *,
            store: Any,
            seed: int = 42,
        ) -> BenchmarkRunResult:
            return BenchmarkRunResult(trajectory_id="x", success=True, raw_score=1.0)

    assert isinstance(_OkBenchmark(), Benchmark)

    class _NotBenchmark:
        name = "no"

    assert not isinstance(_NotBenchmark(), Benchmark)
