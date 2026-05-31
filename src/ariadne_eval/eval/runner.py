"""Runner that evaluates (Trajectory, Steps, Case) triples through metrics."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ariadne_eval.core.trajectory import Step, Trajectory
from ariadne_eval.eval.case import Case
from ariadne_eval.eval.errors import MissingReferenceError
from ariadne_eval.eval.metrics.base import AsyncMetric, Metric, MetricResult
from ariadne_eval.eval.stats.bootstrap import BootstrapCI, bootstrap_mean_ci

__all__ = ["EvalReport", "Runner"]

_BOOTSTRAP_FLOAT_FIELDS = ("mean", "lo", "hi")

_MISSING = object()  # marker for an async task whose metric raised MissingReferenceError


def _is_async_only(metric: object) -> bool:
    """A metric is async-only if it implements ``ascore`` without a sync ``score``."""
    return callable(getattr(metric, "ascore", None)) and not callable(
        getattr(metric, "score", None)
    )


def _nan_for_nulls(d: dict[str, object]) -> dict[str, object]:
    """Coerce ``null``-valued BootstrapCI float fields back to ``NaN``.

    Inverse of the ``allow_nan=False`` JSON serialization in ``to_jsonl``.
    """
    out = dict(d)
    for key in _BOOTSTRAP_FLOAT_FIELDS:
        if key in out and out[key] is None:
            out[key] = math.nan
    return out


class EvalReport(BaseModel):
    """Per-(case, metric) results plus bootstrap aggregates."""

    model_config = {"frozen": True}

    results: tuple[MetricResult, ...] = Field(default_factory=tuple)
    aggregates: dict[str, BootstrapCI] = Field(default_factory=dict)
    n_cases: int = 0
    seed: int = 0

    def to_jsonl(self, path: str | Path) -> None:
        """Write a JSONL file: one header line then one MetricResult per line.

        Non-finite floats (``NaN``, ``+Inf``, ``-Inf``) in ``BootstrapCI``
        aggregates serialize as ``null`` for RFC-8259 compliance. Round-trip
        via ``from_jsonl`` rehydrates them back to ``NaN``.
        """
        p = Path(path)
        with p.open("w", encoding="utf-8") as f:
            header = {
                "_kind": "header",
                "n_cases": self.n_cases,
                "seed": self.seed,
                # Pydantic's JSON serializer emits ``null`` for non-finite floats.
                "aggregates": {
                    k: json.loads(v.model_dump_json()) for k, v in self.aggregates.items()
                },
            }
            f.write(json.dumps(header, sort_keys=True, allow_nan=False))
            f.write("\n")
            for r in self.results:
                line = {"_kind": "result", **r.model_dump()}
                f.write(json.dumps(line, sort_keys=True, default=str, allow_nan=False))
                f.write("\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> EvalReport:
        """Read a JSONL file written by ``to_jsonl`` and reconstruct an EvalReport.

        ``null`` in ``BootstrapCI`` float fields is rehydrated to ``NaN``.
        """
        p = Path(path)
        results: list[MetricResult] = []
        n_cases = 0
        seed = 0
        aggregates: dict[str, BootstrapCI] = {}
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                kind = obj.pop("_kind", None)
                if kind == "header":
                    n_cases = int(obj["n_cases"])
                    seed = int(obj["seed"])
                    aggregates = {
                        k: BootstrapCI.model_validate(_nan_for_nulls(v))
                        for k, v in obj["aggregates"].items()
                    }
                elif kind == "result":
                    results.append(MetricResult.model_validate(obj))
                else:  # pragma: no cover - defensive
                    raise ValueError(f"Unknown JSONL line kind: {kind!r}")
        return cls(
            results=tuple(results),
            aggregates=aggregates,
            n_cases=n_cases,
            seed=seed,
        )


class Runner:
    """Evaluate a stream of (Trajectory, Steps, Case) triples."""

    def __init__(
        self,
        metrics: Sequence[Metric | AsyncMetric],
        *,
        seed: int = 0,
        n_resamples: int = 1000,
        confidence: float = 0.95,
        on_missing_reference: Literal["skip", "error"] = "skip",
        concurrency: int = 4,
    ) -> None:
        """Initialise with a list of metrics and evaluation options."""
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency!r}")
        self._metrics: list[Metric | AsyncMetric] = list(metrics)
        self._seed = seed
        self._n_resamples = n_resamples
        self._confidence = confidence
        self._on_missing = on_missing_reference
        self._concurrency = concurrency

    def evaluate(
        self,
        items: Iterable[tuple[Trajectory, list[Step], Case]],
    ) -> EvalReport:
        """Score every item through all metrics and return an EvalReport.

        Sync only — raises ``RuntimeError`` if any metric implements
        ``ascore`` without a sync ``score``. Use ``aevaluate`` for those.
        """
        for metric in self._metrics:
            if not callable(getattr(metric, "score", None)):
                raise RuntimeError(
                    f"Metric {metric.name!r} is async-only; "
                    "use Runner.aevaluate instead of Runner.evaluate."
                )
        per_metric: dict[str, list[float]] = {m.name: [] for m in self._metrics}
        results: list[MetricResult] = []
        n_cases = 0

        for traj, steps, case in items:
            n_cases += 1
            for metric in self._metrics:
                try:
                    res = metric.score(traj, steps, case)  # type: ignore[union-attr]
                except MissingReferenceError:
                    if self._on_missing == "error":
                        raise
                    continue
                results.append(res)
                per_metric[metric.name].append(res.score)

        aggregates = {
            name: bootstrap_mean_ci(
                values,
                n_resamples=self._n_resamples,
                confidence=self._confidence,
                seed=self._seed,
            )
            for name, values in per_metric.items()
        }
        return EvalReport(
            results=tuple(results),
            aggregates=aggregates,
            n_cases=n_cases,
            seed=self._seed,
        )

    async def aevaluate(
        self,
        items: Iterable[tuple[Trajectory, list[Step], Case]],
    ) -> EvalReport:
        """Score every item through all metrics, dispatching async metrics concurrently.

        Sync metrics are called inline (pure compute). Async-only metrics are
        scheduled under ``asyncio.Semaphore(concurrency)``. Results are
        collected in deterministic per-(item, metric) input order regardless
        of completion order. ``MissingReferenceError`` is honored per the
        ``on_missing_reference`` policy; other exceptions cancel in-flight
        tasks and propagate.
        """
        items_list = list(items)
        n_cases = len(items_list)
        n_metrics = len(self._metrics)
        # grid[i][j] holds MetricResult or _MISSING (skipped) or None (sync OK assigned)
        grid: list[list[object]] = [[None] * n_metrics for _ in range(n_cases)]
        sem = asyncio.Semaphore(self._concurrency)

        async def _run_async(
            metric: AsyncMetric, traj: Trajectory, steps: list[Step], case: Case
        ) -> object:
            async with sem:
                try:
                    return await metric.ascore(traj, steps, case)
                except MissingReferenceError:
                    return _MISSING

        async with asyncio.TaskGroup() as tg:
            async_tasks: list[tuple[int, int, asyncio.Task[object]]] = []
            for i, (traj, steps, case) in enumerate(items_list):
                for j, metric in enumerate(self._metrics):
                    if _is_async_only(metric):
                        task = tg.create_task(_run_async(metric, traj, steps, case))  # type: ignore[arg-type]
                        async_tasks.append((i, j, task))
                    else:
                        try:
                            grid[i][j] = metric.score(traj, steps, case)  # type: ignore[union-attr]
                        except MissingReferenceError:
                            if self._on_missing == "error":
                                raise
                            grid[i][j] = _MISSING
        for i, j, task in async_tasks:
            grid[i][j] = task.result()

        # Error-mode promotion: if any async cell came back _MISSING under error
        # mode, raise now (after all tasks finished — async semantics).
        if self._on_missing == "error":
            for i, row in enumerate(grid):
                for _j, cell in enumerate(row):
                    if cell is _MISSING:
                        case_id = items_list[i][2].case_id
                        raise MissingReferenceError("expected_answer", case_id=case_id)

        # Flatten in deterministic order; collect per-metric scores
        results: list[MetricResult] = []
        per_metric: dict[str, list[float]] = {m.name: [] for m in self._metrics}
        for _i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if cell is None or cell is _MISSING:
                    continue
                if not isinstance(cell, MetricResult):  # pragma: no cover
                    raise TypeError(f"Expected MetricResult, got {type(cell)!r}")
                results.append(cell)
                per_metric[self._metrics[j].name].append(cell.score)

        aggregates = {
            name: bootstrap_mean_ci(
                values,
                n_resamples=self._n_resamples,
                confidence=self._confidence,
                seed=self._seed,
            )
            for name, values in per_metric.items()
        }
        return EvalReport(
            results=tuple(results),
            aggregates=aggregates,
            n_cases=n_cases,
            seed=self._seed,
        )
