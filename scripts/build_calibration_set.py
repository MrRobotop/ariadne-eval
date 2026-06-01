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
from typing import Any, Literal

import click

from ariadne_eval.core.trajectory import Step, Trajectory
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


def _build_confusion_block(
    judged_pairs: list[tuple[str, str]],
) -> dict[str, object]:
    """Build the ``_kind: "confusion"`` JSONL block from judged pairs.

    ``judged_pairs`` is a list of ``(gold_label, judge_label)``. The
    returned dict contains the sorted label set, the row=gold/col=judge
    confusion matrix as nested ints, and per-label precision / recall /
    support computed against the matrix. Precision and recall are 0.0
    when the relevant denominator is zero (label never predicted by the
    judge, or never present in gold, respectively).
    """
    if not judged_pairs:
        return {"_kind": "confusion", "labels": [], "matrix": [], "per_label": {}}

    labels = sorted({g for g, _ in judged_pairs} | {j for _, j in judged_pairs})
    n = len(labels)
    idx = {lbl: i for i, lbl in enumerate(labels)}

    matrix: list[list[int]] = [[0] * n for _ in range(n)]
    for gold, judge in judged_pairs:
        matrix[idx[gold]][idx[judge]] += 1

    per_label: dict[str, dict[str, float | int]] = {}
    for i, lbl in enumerate(labels):
        tp = matrix[i][i]
        row_sum = sum(matrix[i])  # gold count of this label
        col_sum = sum(matrix[r][i] for r in range(n))  # judge count of this label
        precision = tp / col_sum if col_sum else 0.0
        recall = tp / row_sum if row_sum else 0.0
        per_label[lbl] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "support": row_sum,
        }

    return {
        "_kind": "confusion",
        "labels": labels,
        "matrix": matrix,
        "per_label": per_label,
    }


def _build_meta_block(
    *,
    judge_model: str,
    temperature: float,
    run_date: str,
    n_gold: int,
) -> dict[str, object]:
    """Build the ``_kind: "meta"`` JSONL block.

    Carries the judge model + temperature for reproducibility,
    sha256 digests of the two prompt constants so future re-runs can
    detect silent prompt drift, the run date, the ariadne version, and
    the total gold-set size.
    """
    import hashlib

    from ariadne_eval import __version__
    from ariadne_eval.eval.judges.prompts import (
        PLAN_QUALITY_SYSTEM,
        PLAN_QUALITY_USER_TEMPLATE,
    )

    return {
        "_kind": "meta",
        "judge_model": judge_model,
        "temperature": temperature,
        "system_prompt_sha256": hashlib.sha256(PLAN_QUALITY_SYSTEM.encode("utf-8")).hexdigest(),
        "user_template_sha256": hashlib.sha256(
            PLAN_QUALITY_USER_TEMPLATE.encode("utf-8")
        ).hexdigest(),
        "run_date": run_date,
        "ariadne_version": __version__,
        "n_gold": n_gold,
    }


def _make_judge(model: str) -> Judge:
    factory = _resolve_test_factory()
    if factory is not None:
        return factory(model)  # type: ignore[no-any-return]
    return TrajectoryJudge(model=model)


async def run(
    *,
    source: Literal["synth", "store"] = "store",
    store_path: Path | None,
    gold_labels: Path,
    judge_model: str,
    out_path: Path,
    concurrency: int,
    temperature: float = 0.0,
    run_date: str | None = None,
) -> None:
    """Load each labeled trajectory, judge it, write per-line + summary report."""
    from datetime import UTC, datetime

    if run_date is None:
        run_date = datetime.now(UTC).strftime("%Y-%m-%d")

    judge = _make_judge(judge_model)

    loaded: list[tuple[dict[str, str], Trajectory | None, list[Step] | None, str | None]] = []

    if source == "synth":
        # --source synth: load directly from gold_plans.jsonl. No DuckDB needed.
        import sys

        # Make tests/data importable. scripts/ sits at repo-root/scripts;
        # tests/ sits at repo-root/tests. Resolve from __file__.
        _repo_root = Path(__file__).resolve().parents[1]
        if str(_repo_root) not in sys.path:
            sys.path.insert(0, str(_repo_root))
        from tests.data._load_gold_plans import iter_gold_plans

        with gold_labels.open(encoding="utf-8") as f:
            for gold_entry in iter_gold_plans(f):
                # Match the gold-labels JSONL key ("label") that the store path consumes;
                # _judge_one rewrites it to "gold_label" in the per-row output.
                entry_dict = {
                    "trajectory_id": gold_entry.trajectory.id,
                    "label": gold_entry.gold_label,
                }
                loaded.append((entry_dict, gold_entry.trajectory, gold_entry.steps, None))
    else:
        # --source store: production path against a user's DuckDB store.
        if store_path is None:
            raise ValueError("--source store requires --store / store_path to be set")
        store = DuckDBStore(path=store_path)
        entries: list[dict[str, str]] = []
        for line in gold_labels.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))
        for entry in entries:
            traj_id = entry["trajectory_id"]
            try:
                traj, steps = await store.get_trajectory(traj_id)
            except Exception as exc:
                loaded.append((entry, None, None, f"load: {exc}"))
                continue
            loaded.append((entry, traj, steps, None))
        await store.close()

    n_gold = len(loaded)
    sem = asyncio.Semaphore(concurrency)

    async def _judge_one(
        entry: dict[str, str],
        traj: Trajectory | None,
        steps: list[Step] | None,
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

    # NOTE: temperature is recorded in meta for the run's audit trail; it
    # is NOT forwarded to the judge yet. TrajectoryJudge uses its own
    # default (0.0). Thread it through _make_judge() if non-default
    # temperatures (e.g. for ensemble calibration) become a real use case.
    confusion_block = _build_confusion_block(judged_pairs)
    meta_block = _build_meta_block(
        judge_model=judge_model,
        temperature=temperature,
        run_date=run_date,
        n_gold=n_gold,
    )

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str))
            f.write("\n")
        f.write(json.dumps(summary, default=str))
        f.write("\n")
        f.write(json.dumps(confusion_block, default=str))
        f.write("\n")
        f.write(json.dumps(meta_block, default=str))
        f.write("\n")


@click.command()
@click.option(
    "--source",
    type=click.Choice(["synth", "store"], case_sensitive=False),
    default="store",
    show_default=True,
    help="Where to load trajectories from: 'synth' for tests/data/gold_plans.jsonl, "
    "'store' for a user's DuckDB store.",
)
@click.option(
    "--store",
    "store_path",
    type=click.Path(path_type=Path, exists=True),
    required=False,
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
    source: str,
    store_path: Path | None,
    gold_labels: Path,
    judge_model: str,
    out_path: Path,
    concurrency: int,
) -> None:
    """Run the calibration harness."""
    if source == "store" and store_path is None:
        raise click.UsageError("--source store requires --store to be set.")
    asyncio.run(
        run(
            source=source,  # type: ignore[arg-type]
            store_path=store_path,
            gold_labels=gold_labels,
            judge_model=judge_model,
            out_path=out_path,
            concurrency=concurrency,
        )
    )


if __name__ == "__main__":
    main()
