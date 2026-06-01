"""Benchmark Protocol + result dataclasses."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ariadne_eval.core.trajectory import Step, Trajectory  # noqa: F401  (re-exported via Protocol)
from ariadne_eval.storage.base import Store

__all__ = ["Benchmark", "BenchmarkRunResult", "BenchmarkTask"]


@dataclass(frozen=True)
class BenchmarkTask:
    """One task in a benchmark.

    ``payload`` is benchmark-specific and not interpreted by the
    runner; concrete adapters (e.g. ``TauBenchAdapter``) carry the
    benchmark's native task object through here so ``run_task`` can
    reconstitute env state per task.
    """

    task_id: str
    task_index: int
    instruction: str
    payload: Any


@dataclass(frozen=True)
class BenchmarkRunResult:
    """What the benchmark hands back per task.

    ``trajectory_id`` is a foreign key into the store; the full trace
    lives there. ``success`` is the benchmark's boolean verdict;
    ``raw_score`` is its native numeric score (tau-bench: ≈1.0 → pass).
    """

    trajectory_id: str
    success: bool
    raw_score: float
    error: str | None = None


@runtime_checkable
class Benchmark(Protocol):
    """Trajectory-agnostic benchmark contract."""

    name: str

    def tasks(
        self,
        *,
        split: str = "test",
        limit: int | None = None,
    ) -> Sequence[BenchmarkTask]:
        """Return the benchmark's task list. May read disk / network on first call."""
        ...

    async def run_task(
        self,
        task: BenchmarkTask,
        model: str,
        provider: str,
        *,
        store: Store,
        seed: int = 42,
    ) -> BenchmarkRunResult:
        """Run ``task`` against ``model`` and persist the trajectory to ``store``."""
        ...
