"""End-to-end tests for BenchmarkRunner using StubBenchmark + StubJudge."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ariadne_eval.benchmarks.base import (
    Benchmark,  # noqa: F401
    BenchmarkRunResult,
    BenchmarkTask,
)
from ariadne_eval.benchmarks.config import (
    BenchmarkConfig,
    BootstrapSpec,
    JudgeSpec,
    ModelSpec,
    OutputSpec,
    TasksSpec,
    TauBenchSpec,
)
from ariadne_eval.benchmarks.runner import BenchmarkRunner
from ariadne_eval.core.ids import new_id
from ariadne_eval.core.status import StepStatus, TrajectoryStatus
from ariadne_eval.core.trajectory import (
    LLMCallPayload,
    Message,
    Step,
    Trajectory,
)
from ariadne_eval.eval.errors import BootstrapInsufficientDataWarning
from ariadne_eval.eval.judges.base import JudgeVerdict
from ariadne_eval.eval.judges.stub import StubJudge
from ariadne_eval.eval.metrics.efficiency import StepEfficiency
from ariadne_eval.eval.metrics.plan_quality import PlanQuality
from ariadne_eval.storage.duckdb_store import DuckDBStore

pytestmark = pytest.mark.fast


class StubBenchmark:
    """Deterministic in-memory benchmark for unit tests."""

    name = "stub-bench"

    def __init__(self, *, n_tasks: int = 3) -> None:
        self._n = n_tasks

    def tasks(self, *, split: str = "test", limit: int | None = None) -> Sequence[BenchmarkTask]:
        n = min(limit, self._n) if limit else self._n
        return [
            BenchmarkTask(
                task_id=f"stub-{i}",
                task_index=i,
                instruction=f"do stub task {i}",
                payload={"i": i},
            )
            for i in range(n)
        ]

    async def run_task(
        self,
        task: BenchmarkTask,
        model: str,
        provider: str,
        *,
        store,
        seed: int = 42,
    ) -> BenchmarkRunResult:
        started = datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC)
        traj_id = new_id()
        # Every other task "succeeds" for variance in pass-rate aggregation
        reward = 1.0 if task.task_index % 2 == 0 else 0.0
        traj = Trajectory(
            id=traj_id,
            task=task.instruction,
            agent_name=f"stub-agent/{model}",
            agent_version="0.0.0",
            model_id=f"{provider}/{model}",
            started_at=started,
            finished_at=started + timedelta(seconds=1),
            final_status=(
                TrajectoryStatus.SUCCEEDED if reward >= 0.999999 else TrajectoryStatus.FAILED
            ),
            final_answer="ok" if reward >= 0.999999 else "no",
            metadata={"tau_bench_reward": reward, "tau_bench_task_id": task.task_id},
        )
        step = Step(
            id=new_id(),
            trajectory_id=traj_id,
            parent_step_id=None,
            name="llm_call",
            started_at=started,
            finished_at=started + timedelta(milliseconds=10),
            status=StepStatus.SUCCEEDED,
            payload=LLMCallPayload(
                model_id=f"{provider}/{model}",
                prompt_messages=[Message(role="user", content=task.instruction)],
                completion="Step 1: stub. Step 2: stub.",
                input_tokens=0,
                output_tokens=0,
                latency_ms=10.0,
                cost_usd=0.0,
            ),
        )
        await store.save_trajectory(traj, [step])
        return BenchmarkRunResult(
            trajectory_id=traj_id,
            success=reward >= 0.999999,
            raw_score=reward,
        )


def _config(bundle_dir: Path, store_path: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        benchmark=TauBenchSpec(kind="tau-bench", env="retail"),
        models=[
            ModelSpec(model="m-a", provider="prov"),
            ModelSpec(model="m-b", provider="prov"),
        ],
        tasks=TasksSpec(limit=3, seed=42),
        concurrency=2,
        bootstrap=BootstrapSpec(n_resamples=200, confidence=0.95),
        metrics=["step_efficiency", "plan_quality"],
        judge=JudgeSpec(model="judge/m", temperature=0.0),
        output=OutputSpec(bundle_dir=bundle_dir, store_path=store_path),
    )


async def test_runner_writes_bundle_with_calibration_note(tmp_path: Path) -> None:
    bench = StubBenchmark(n_tasks=3)
    bundle_dir = tmp_path / "bundle"
    store_path = tmp_path / "bench.duckdb"
    cfg = _config(bundle_dir, store_path)
    store = DuckDBStore(path=store_path)
    judge = StubJudge(JudgeVerdict(score=0.6, label="partial", rationale="meh"))

    runner = BenchmarkRunner(
        benchmark=bench,
        models=cfg.models,
        store=store,
        metrics=[StepEfficiency(), PlanQuality(judge)],
        seed=cfg.tasks.seed,
        concurrency=cfg.concurrency,
        n_resamples=cfg.bootstrap.n_resamples,
        confidence=cfg.bootstrap.confidence,
    )
    # StepEfficiency yields no scores (cases have no expected_max_steps) →
    # bootstrap_mean_ci(n=0) emits BootstrapInsufficientDataWarning (x2 models).
    with pytest.warns(BootstrapInsufficientDataWarning):
        report = await runner.run(bench.tasks(limit=cfg.tasks.limit))
    runner.write_bundle(report, bundle_dir, config=cfg)
    await store.close()

    assert (bundle_dir / "config.yaml").exists()
    assert (bundle_dir / "trajectories.jsonl").exists()
    assert (bundle_dir / "summary.json").exists()

    summary = json.loads((bundle_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["_kind"] == "benchmark_summary"
    assert summary["task_count"] == 3
    assert len(summary["models"]) == 2
    for m in summary["models"]:
        # Pass rate of stub bench: tasks 0,2 succeed → 2/3 ≈ 0.667
        assert m["pass_rate"]["n"] == 3
        assert m["pass_rate"]["mean"] == pytest.approx(2 / 3, abs=1e-6)
        # PlanQuality's score from StubJudge is 0.6 on every trajectory
        assert m["metrics"]["plan_quality"]["mean"] == pytest.approx(0.6, abs=1e-6)
        # Calibration note travels with plan_quality
        assert "calibration_note" in m["metrics"]["plan_quality"]
        assert "κ" in m["metrics"]["plan_quality"]["calibration_note"]


async def test_runner_trajectories_jsonl_is_sorted(tmp_path: Path) -> None:
    bench = StubBenchmark(n_tasks=2)
    bundle_dir = tmp_path / "bundle"
    store_path = tmp_path / "bench.duckdb"
    cfg = _config(bundle_dir, store_path)
    store = DuckDBStore(path=store_path)
    judge = StubJudge(JudgeVerdict(score=0.5, label="partial", rationale="ok"))

    runner = BenchmarkRunner(
        benchmark=bench,
        models=cfg.models,
        store=store,
        metrics=[StepEfficiency(), PlanQuality(judge)],
        seed=42,
        concurrency=1,
        n_resamples=200,
        confidence=0.95,
    )
    # StepEfficiency yields no scores (cases have no expected_max_steps) →
    # bootstrap_mean_ci(n=0) emits BootstrapInsufficientDataWarning (x2 models).
    with pytest.warns(BootstrapInsufficientDataWarning):
        report = await runner.run(bench.tasks(limit=2))
    runner.write_bundle(report, bundle_dir, config=cfg)
    await store.close()

    lines = (bundle_dir / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
    # 2 tasks x 2 models = 4 lines
    assert len(lines) == 4
    # Each line is a JSON object with trajectory + steps
    keys = [json.loads(line) for line in lines]
    sort_tuples = [(k["trajectory"]["task"], k["trajectory"]["model_id"]) for k in keys]
    assert sort_tuples == sorted(sort_tuples)
