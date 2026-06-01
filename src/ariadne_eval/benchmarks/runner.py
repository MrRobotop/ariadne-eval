"""Benchmark orchestrator: runs (task x model) cells, writes the result bundle."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ariadne_eval._transient import (
    MAX_TRANSIENT_RETRIES,
    TRANSIENT_BACKOFF_BASE,
    is_transient,
)
from ariadne_eval._version import __version__
from ariadne_eval.benchmarks.base import (
    Benchmark,
    BenchmarkRunResult,
    BenchmarkTask,
)
from ariadne_eval.benchmarks.config import ModelSpec
from ariadne_eval.core.status import TrajectoryStatus
from ariadne_eval.core.trajectory import Step, Trajectory
from ariadne_eval.eval.case import Case
from ariadne_eval.eval.metrics.base import AsyncMetric, Metric
from ariadne_eval.eval.runner import EvalReport, Runner
from ariadne_eval.eval.stats.bootstrap import bootstrap_mean_ci
from ariadne_eval.storage.base import Store

if TYPE_CHECKING:
    from ariadne_eval.benchmarks.config import BenchmarkConfig

__all__ = ["BenchmarkReport", "BenchmarkRunner", "CellResult"]


# Calibration note that travels with every plan_quality aggregate.
# Comes from Phase 6.1's published kappa.
_PLAN_QUALITY_CALIBRATION_NOTE = "judge κ = 0.32 (fair); see docs/concepts/calibration.md"


@dataclass(frozen=True)
class CellResult:
    """One (task, model) cell's outcome."""

    task: BenchmarkTask
    model: ModelSpec
    run_result: BenchmarkRunResult
    trajectory: Trajectory
    steps: list[Step]


@dataclass(frozen=True)
class BenchmarkReport:
    """Aggregated benchmark output, ready for bundle write."""

    benchmark_name: str
    task_count: int
    seed: int
    run_date: str
    per_model: dict[ModelSpec, EvalReport]
    cell_results: list[CellResult]


class BenchmarkRunner:
    """Orchestrates a benchmark run and writes a result bundle."""

    def __init__(
        self,
        benchmark: Benchmark,
        models: Sequence[ModelSpec],
        *,
        store: Store,
        metrics: Sequence[Metric | AsyncMetric] = (),
        seed: int = 42,
        concurrency: int = 4,
        n_resamples: int = 1000,
        confidence: float = 0.95,
    ) -> None:
        """Initialise the runner with a benchmark, model list, and store."""
        self._benchmark = benchmark
        self._models = list(models)
        self._store = store
        self._metrics = list(metrics)
        self._seed = seed
        self._concurrency = concurrency
        self._n_resamples = n_resamples
        self._confidence = confidence

    async def run(
        self,
        tasks: Sequence[BenchmarkTask],
        *,
        resume_from_store: bool = False,
    ) -> BenchmarkReport:
        """Run every (task x model) cell under bounded concurrency."""
        _ = resume_from_store  # resume support is exercised via CLI / manual run

        # NOTE: ``concurrency`` bounds in-flight semaphore slots, but the
        # store_lock below serializes the entire run_task + get_trajectory
        # block end-to-end. With the current single-connection DuckDB store,
        # effective LLM-call parallelism is 1 regardless of this value.
        # Lifting this requires either a multi-connection DuckDB store
        # variant or a Benchmark Protocol redesign where run_task returns a
        # trajectory without persisting (then the runner persists outside
        # the lock).
        sem = asyncio.Semaphore(self._concurrency)
        # Serialize all store reads so that concurrent cells don't send
        # simultaneous asyncio.to_thread calls to a single DuckDB connection.
        # (DuckDB connections are not thread-safe for concurrent access.)
        store_lock = asyncio.Lock()

        async def _one(task: BenchmarkTask, model: ModelSpec) -> CellResult:
            async with sem:
                last_transient: Exception | None = None
                run_result: BenchmarkRunResult | None = None
                traj: Trajectory | None = None
                steps: list[Step] = []
                for attempt in range(MAX_TRANSIENT_RETRIES):
                    try:
                        # Serialize ALL store access (write + immediate read) so
                        # that concurrent cells do not interleave asyncio.to_thread
                        # calls on a single DuckDB connection (not thread-safe).
                        async with store_lock:
                            run_result = await self._benchmark.run_task(
                                task,
                                model.model,
                                model.provider,
                                store=self._store,
                                seed=self._seed,
                            )
                            # Load back from store to catch silent Pydantic
                            # round-trip drift.
                            traj, steps = await self._store.get_trajectory(run_result.trajectory_id)
                        break
                    except Exception as exc:
                        if not is_transient(exc):
                            raise
                        last_transient = exc
                        await asyncio.sleep(TRANSIENT_BACKOFF_BASE * (2**attempt))
                else:
                    # All retries exhausted
                    return CellResult(
                        task=task,
                        model=model,
                        run_result=BenchmarkRunResult(
                            trajectory_id="",
                            success=False,
                            raw_score=0.0,
                            error=f"transient: {last_transient}",
                        ),
                        trajectory=Trajectory(
                            id="00000000000000000000000000",
                            task=task.instruction,
                            agent_name="failed",
                            agent_version="0",
                            model_id=f"{model.provider}/{model.model}",
                            started_at=datetime.now(UTC),
                            final_status=TrajectoryStatus.FAILED,
                        ),
                        steps=[],
                    )

                if run_result is None or traj is None:  # pragma: no cover
                    raise RuntimeError("run_result/traj not set after retry loop")

                return CellResult(
                    task=task,
                    model=model,
                    run_result=run_result,
                    trajectory=traj,
                    steps=steps,
                )

        # Dispatch all cells eagerly under the semaphore
        cells = await asyncio.gather(
            *(_one(task, model) for task in tasks for model in self._models)
        )

        # Aggregate per model with the eval Runner
        per_model: dict[ModelSpec, EvalReport] = {}
        for model in self._models:
            model_cells = [c for c in cells if c.model == model]
            triples = [
                (
                    c.trajectory,
                    c.steps,
                    Case(case_id=c.task.task_id, task=c.task.instruction),
                )
                for c in model_cells
                if c.run_result.error is None
            ]
            runner = Runner(
                metrics=self._metrics,
                seed=self._seed,
                n_resamples=self._n_resamples,
                confidence=self._confidence,
                on_missing_reference="skip",
            )
            per_model[model] = await runner.aevaluate(triples)

        return BenchmarkReport(
            benchmark_name=self._benchmark.name,
            task_count=len(tasks),
            seed=self._seed,
            run_date=datetime.now(UTC).strftime("%Y-%m-%d"),
            per_model=per_model,
            cell_results=list(cells),
        )

    def write_bundle(
        self,
        report: BenchmarkReport,
        out_dir: Path,
        *,
        config: BenchmarkConfig,
    ) -> None:
        """Materialize the result bundle to ``out_dir``."""
        out_dir.mkdir(parents=True, exist_ok=True)

        # config.yaml (audit trail copy)
        cfg_yaml = yaml.safe_dump(
            json.loads(config.model_dump_json()),
            sort_keys=False,
            allow_unicode=True,
        )
        (out_dir / "config.yaml").write_text(cfg_yaml, encoding="utf-8")

        # trajectories.jsonl, sorted by (task_id, model_id) to remain stable
        # across re-runs. We pair each row with the task_id from the cell
        # since the trajectory dict alone doesn't carry it as a top-level field.
        rows: list[tuple[str, str, dict[str, object]]] = []
        for cell in report.cell_results:
            if cell.run_result.error is not None:
                continue
            row = {
                "trajectory": json.loads(cell.trajectory.model_dump_json()),
                "steps": [json.loads(s.model_dump_json()) for s in cell.steps],
            }
            rows.append((cell.task.task_id, str(row["trajectory"]["model_id"]), row))
        rows.sort(key=lambda triple: (triple[0], triple[1]))
        with (out_dir / "trajectories.jsonl").open("w", encoding="utf-8") as f:
            for _task_id, _model_id, row in rows:
                f.write(json.dumps(row, default=str))
                f.write("\n")

        # summary.json
        models_block: list[dict[str, object]] = []
        for model, eval_report in report.per_model.items():
            model_cells = [c for c in report.cell_results if c.model == model]
            ok = [c for c in model_cells if c.run_result.error is None]
            successes = [1.0 if c.run_result.success else 0.0 for c in ok]
            pr_ci = bootstrap_mean_ci(
                successes, seed=report.seed, n_resamples=1000, confidence=0.95
            )
            metrics_block: dict[str, dict[str, object]] = {}
            for name, ci in eval_report.aggregates.items():
                block: dict[str, object] = {
                    "mean": ci.mean,
                    "lo": ci.lo,
                    "hi": ci.hi,
                    "n": ci.n,
                }
                if name == "plan_quality":
                    block["calibration_note"] = _PLAN_QUALITY_CALIBRATION_NOTE
                metrics_block[name] = block

            median_steps = sorted(len(c.steps) for c in ok)[len(ok) // 2] if ok else 0

            models_block.append(
                {
                    "model": model.model,
                    "provider": model.provider,
                    "pass_rate": {
                        "mean": pr_ci.mean,
                        "lo": pr_ci.lo,
                        "hi": pr_ci.hi,
                        "n": pr_ci.n,
                        "method": "bootstrap-percentile",
                    },
                    "metrics": metrics_block,
                    "median_steps": median_steps,
                    "errored_cells": len(model_cells) - len(ok),
                }
            )

        summary = {
            "_kind": "benchmark_summary",
            "benchmark": report.benchmark_name,
            "task_count": report.task_count,
            "seed": report.seed,
            "run_date": report.run_date,
            "ariadne_version": __version__,
            "models": models_block,
        }
        (out_dir / "summary.json").write_text(
            json.dumps(
                _sanitize_nans(summary),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )


def _sanitize_nans(obj: Any) -> Any:
    """Recursively replace non-finite floats with None for RFC 8259-valid JSON.

    Matches the convention used by ``EvalReport.to_jsonl`` so committed
    summary.json files are parseable by ``jq``, browsers, and strict
    validators.
    """
    import math

    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nans(v) for v in obj]
    return obj
