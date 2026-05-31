"""Calibration harness: run a Judge over labeled trajectories and emit a kappa report.

Usage:
    uv run python scripts/build_calibration_set.py \\
        --store ~/.ariadne/store.duckdb \\
        --gold-labels gold_labels.jsonl \\
        --judge-model claude-sonnet \\
        --out calibration_report.jsonl \\
        --concurrency 4

For tests, ``ARIADNE_TEST_JUDGE_FACTORY`` env var may point at a
dotted-path callable that returns a Judge (used in place of TrajectoryJudge).
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
from pathlib import Path
from typing import Any

import click

from ariadne_eval.eval.judges.base import Judge, JudgeParseError
from ariadne_eval.eval.judges.trajectory_judge import TrajectoryJudge
from ariadne_eval.eval.stats.agreement import cohens_kappa
from ariadne_eval.storage.duckdb_store import DuckDBStore


def _resolve_test_factory() -> Any | None:
    """If ARIADNE_TEST_JUDGE_FACTORY is set, import and return the callable."""
    path = os.environ.get("ARIADNE_TEST_JUDGE_FACTORY")
    if not path:
        return None
    module_path, _, attr = path.rpartition(".")
    if not module_path:
        raise ValueError(f"ARIADNE_TEST_JUDGE_FACTORY is not a dotted path: {path!r}")
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


def _make_judge(model: str) -> Judge:
    factory = _resolve_test_factory()
    if factory is not None:
        return factory(model)  # type: ignore[no-any-return]
    return TrajectoryJudge(model=model)


async def run(
    *,
    store_path: Path,
    gold_labels: Path,
    judge_model: str,
    out_path: Path,
    concurrency: int,
) -> None:
    """Load each labeled trajectory, judge it, write per-line + summary report."""
    judge = _make_judge(judge_model)
    store = DuckDBStore(path=store_path)

    entries: list[dict[str, str]] = []
    for line in gold_labels.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))

    loaded: list[tuple[dict[str, str], object, object, str | None]] = []
    for entry in entries:
        traj_id = entry["trajectory_id"]
        try:
            traj, steps = await store.get_trajectory(traj_id)
        except Exception as exc:
            loaded.append((entry, None, None, f"load: {exc}"))
            continue
        loaded.append((entry, traj, steps, None))
    await store.close()

    sem = asyncio.Semaphore(concurrency)

    async def _judge_one(
        entry: dict[str, str],
        traj: object,
        steps: object,
        load_error: str | None,
    ) -> dict[str, object]:
        traj_id = entry["trajectory_id"]
        gold_label = entry["label"]
        if load_error is not None:
            return {
                "trajectory_id": traj_id,
                "gold_label": gold_label,
                "error": load_error,
            }
        async with sem:
            try:
                verdict = await judge.judge(traj, steps, None)  # type: ignore[arg-type]
            except JudgeParseError as exc:
                return {
                    "trajectory_id": traj_id,
                    "gold_label": gold_label,
                    "error": f"parse: {exc}",
                }
        return {
            "trajectory_id": traj_id,
            "gold_label": gold_label,
            "judge_label": verdict.label,
            "judge_score": verdict.score,
            "judge_rationale": verdict.rationale,
        }

    rows = await asyncio.gather(*(_judge_one(*item) for item in loaded))

    judged_pairs: list[tuple[str, str]] = [
        (str(r["gold_label"]), str(r["judge_label"])) for r in rows if "judge_label" in r
    ]
    gold = [g for g, _ in judged_pairs]
    judge_l = [j for _, j in judged_pairs]
    summary: dict[str, object]
    if judged_pairs:
        kappa_result = cohens_kappa(gold, judge_l)
        summary = {
            "_kind": "summary",
            "n": kappa_result.n,
            "kappa": kappa_result.kappa,
            "interpretation": kappa_result.interpretation,
            "label_set": list(kappa_result.label_set),
        }
    else:
        summary = {
            "_kind": "summary",
            "n": 0,
            "kappa": None,
            "interpretation": "poor",
            "label_set": [],
        }

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str))
            f.write("\n")
        f.write(json.dumps(summary, default=str))
        f.write("\n")


@click.command()
@click.option(
    "--store",
    "store_path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Path to the DuckDB store.",
)
@click.option(
    "--gold-labels",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="JSONL of {trajectory_id, label}.",
)
@click.option(
    "--judge-model",
    required=True,
    help="litellm model name for the judge.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL report.",
)
@click.option(
    "--concurrency",
    type=int,
    default=4,
    show_default=True,
    help="Max in-flight judge calls.",
)
def main(
    store_path: Path,
    gold_labels: Path,
    judge_model: str,
    out_path: Path,
    concurrency: int,
) -> None:
    """Run the calibration harness."""
    asyncio.run(
        run(
            store_path=store_path,
            gold_labels=gold_labels,
            judge_model=judge_model,
            out_path=out_path,
            concurrency=concurrency,
        )
    )


if __name__ == "__main__":
    main()
