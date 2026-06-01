"""`ariadne bench` subcommand group."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from pydantic import ValidationError

from ariadne_eval.benchmarks.config import load_benchmark_config


@click.group("bench")
def bench() -> None:
    """Run and compare agent benchmarks."""


@bench.command("run")
@click.argument("config_path", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate the config without making any LLM calls.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Override config.tasks.limit (run fewer tasks for partial reruns).",
)
@click.option(
    "--models",
    multiple=True,
    default=(),
    help=(
        "Subset of config.models to run (repeatable). "
        "Example: --models anthropic/claude-haiku-4-5-20251001"
    ),
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Skip (task, model) cells whose trajectory already exists in the store.",
)
def bench_run(
    config_path: Path,
    dry_run: bool,
    limit: int | None,
    models: tuple[str, ...],
    resume: bool,
) -> None:
    """Run a benchmark from a YAML config."""
    try:
        cfg = load_benchmark_config(config_path)
    except (ValueError, ValidationError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if dry_run:
        click.echo(f"Config valid: {cfg.benchmark.kind} / {cfg.benchmark.env}")
        click.echo(f"  models   : {len(cfg.models)}")
        click.echo(f"  tasks    : {cfg.tasks.limit or 'all'} (seed={cfg.tasks.seed})")
        click.echo(f"  metrics  : {', '.join(cfg.metrics)}")
        click.echo(f"  bundle   : {cfg.output.bundle_dir}")
        return

    # Production path: lazy imports so --dry-run never loads tau-bench.
    from ariadne_eval.benchmarks.runner import BenchmarkRunner
    from ariadne_eval.benchmarks.tau_bench import TauBenchAdapter
    from ariadne_eval.eval.judges.trajectory_judge import TrajectoryJudge
    from ariadne_eval.eval.metrics.base import AsyncMetric, Metric
    from ariadne_eval.eval.metrics.efficiency import StepEfficiency
    from ariadne_eval.eval.metrics.plan_quality import PlanQuality
    from ariadne_eval.storage.duckdb_store import DuckDBStore

    benchmark = TauBenchAdapter(
        env_name=cfg.benchmark.env,
        user_model=cfg.benchmark.user_model,
        user_strategy=cfg.benchmark.user_strategy,
        agent_kind=cfg.benchmark.agent_kind,
    )
    selected_models = (
        [m for m in cfg.models if m.model in set(models)] if models else list(cfg.models)
    )
    if not selected_models:
        raise click.UsageError("--models filter excluded every config.models entry")

    judge = TrajectoryJudge(model=cfg.judge.model, temperature=cfg.judge.temperature)
    metric_objs: list[Metric | AsyncMetric] = []
    for name in cfg.metrics:
        if name == "step_efficiency":
            metric_objs.append(StepEfficiency())
        elif name == "plan_quality":
            metric_objs.append(PlanQuality(judge))

    store = DuckDBStore(path=cfg.output.store_path.expanduser())
    runner = BenchmarkRunner(
        benchmark=benchmark,
        models=selected_models,
        store=store,
        metrics=metric_objs,
        seed=cfg.tasks.seed,
        concurrency=cfg.concurrency,
        n_resamples=cfg.bootstrap.n_resamples,
        confidence=cfg.bootstrap.confidence,
    )

    async def _go() -> None:
        tasks = benchmark.tasks(split=cfg.benchmark.task_split, limit=limit or cfg.tasks.limit)
        report = await runner.run(tasks, resume_from_store=resume)
        runner.write_bundle(report, cfg.output.bundle_dir, config=cfg)
        await store.close()

    asyncio.run(_go())
    click.echo(f"bundle written: {cfg.output.bundle_dir}")
