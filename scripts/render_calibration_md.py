"""Render a calibration JSONL report to the human-readable docs page.

Usage:
    uv run python scripts/render_calibration_md.py \\
        --report docs/calibration/v0.0.8-alpha-report.jsonl \\
        --out docs/concepts/calibration.md

The rendered page leads with a one-sentence headline, then a confusion
matrix, then per-label precision / recall / support, then methodology +
limitations + recalibration recipe.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import click


def _format_kappa(kappa: float) -> str:
    """Format kappa as a two-decimal string (e.g. ``0.50``)."""
    return f"{kappa:.2f}"


def _hash_prefix(digest: str, length: int = 12) -> str:
    """Return the first ``length`` hex chars of a sha256 digest."""
    return digest[:length] if digest else "(unknown)"


def _section_headline(summary: dict[str, object], meta: dict[str, object]) -> str:
    kappa = _format_kappa(float(cast(float, summary["kappa"])))
    interp = str(summary["interpretation"])
    n = int(cast(int, summary["n"]))
    model = str(meta["judge_model"])
    run_date = str(meta["run_date"])
    return (
        f"> **TrajectoryJudge agrees with the maintainer at "
        f"κ = {kappa} ({interp}), n = {n}, {model}, {run_date}.**"
    )


def _section_confusion(confusion: dict[str, object]) -> str:
    labels = cast(list[str], confusion["labels"])
    matrix = cast(list[list[int]], confusion["matrix"])
    header = "| gold ↓ / judge → | " + " | ".join(labels) + " | total |"
    divider = "|---|" + "|".join(["---"] * (len(labels) + 1)) + "|"
    rows = []
    for lbl, row in zip(labels, matrix, strict=True):
        cells = " | ".join(str(c) for c in row)
        rows.append(f"| **{lbl}** | {cells} | {sum(row)} |")
    return "\n".join(
        [
            "## Confusion matrix",
            "",
            "Rows = maintainer (gold) labels; columns = judge labels. Cell values are raw counts.",
            "",
            header,
            divider,
            *rows,
        ]
    )


def _section_per_label(confusion: dict[str, object]) -> str:
    labels = cast(list[str], confusion["labels"])
    per_label = cast(dict[str, dict[str, float | int]], confusion["per_label"])
    rows = []
    for lbl in labels:
        d = per_label[lbl]
        rows.append(
            f"| {lbl} | {float(cast(float, d['precision'])):.3f}"
            f" | {float(cast(float, d['recall'])):.3f}"
            f" | {int(cast(int, d['support']))} |"
        )
    return "\n".join(
        [
            "## Per-label precision, recall, and support",
            "",
            "| label | precision | recall | support |",
            "|---|---|---|---|",
            *rows,
        ]
    )


def _section_methodology(meta: dict[str, object], summary: dict[str, object]) -> str:
    n_gold = int(cast(int, meta["n_gold"]))
    per_bucket = n_gold // 3
    sys_hash = _hash_prefix(str(meta["system_prompt_sha256"]))
    usr_hash = _hash_prefix(str(meta["user_template_sha256"]))
    model = str(meta["judge_model"])
    temp = float(cast(float, meta["temperature"]))
    return "\n".join(
        [
            "## Methodology",
            "",
            f"- Gold set: {n_gold} synthetic trajectories spanning plan-quality buckets "
            f"(`pass`, `partial`, `fail`), balanced {per_bucket}/{per_bucket}/{per_bucket}.",
            "- One maintainer authored the labels; one judge model was evaluated.",
            f"- Judge configuration: model `{model}`, `temperature={temp}`. "
            f"Prompt-hash digests (first 12 hex): system=`{sys_hash}`, user-template=`{usr_hash}`. "
            f"Re-running with the same model and the same prompts (verified by hash) "
            f"reproduces the numbers above modulo provider determinism.",
        ]
    )


_LIMITATIONS = "\n".join(
    [
        "## Limitations",
        "",
        (
            "- The gold set is synthetic and was written by one person;"
            " biases are present but unknown."
        ),
        (
            "- A single judge model is evaluated. Cross-model agreement"
            " (e.g. `gpt-4o-mini`, open-weights) is deferred to a later phase."
        ),
        '- Kappa bands follow Landis-Koch (1977); "good enough" kappa depends on use case.',
    ]
)


def _section_recalibration() -> str:
    return "\n".join(
        [
            "## Recalibration",
            "",
            "```bash",
            "uv run python scripts/build_calibration_set.py \\",
            "    --source synth \\",
            "    --gold-labels tests/data/gold_plans.jsonl \\",
            "    --judge-model anthropic/claude-sonnet-4-6 \\",
            "    --out docs/calibration/<version>-report.jsonl \\",
            "    --concurrency 4",
            "",
            "uv run python scripts/render_calibration_md.py \\",
            "    --report docs/calibration/<version>-report.jsonl \\",
            "    --out docs/concepts/calibration.md",
            "```",
        ]
    )


def _section_raw_report(report_basename: str) -> str:
    return "\n".join(
        [
            "## Raw report",
            "",
            "The JSONL report this page was rendered from is committed at",
            f"[`docs/calibration/{report_basename}`](../calibration/{report_basename}).",
        ]
    )


def render(*, report_path: Path, out_path: Path, report_basename: str) -> None:
    """Read a calibration JSONL report and write the docs Markdown page."""
    summary: dict[str, object] | None = None
    confusion: dict[str, object] | None = None
    meta: dict[str, object] | None = None
    for line in report_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        kind = obj.get("_kind")
        if kind == "summary":
            summary = obj
        elif kind == "confusion":
            confusion = obj
        elif kind == "meta":
            meta = obj

    if summary is None or confusion is None or meta is None:
        raise ValueError("report is missing one of: _kind=summary, _kind=confusion, _kind=meta")

    parts = [
        "# Calibration",
        "",
        _section_headline(summary, meta),
        "",
        _section_confusion(confusion),
        "",
        _section_per_label(confusion),
        "",
        _section_methodology(meta, summary),
        "",
        _LIMITATIONS,
        "",
        _section_recalibration(),
        "",
        _section_raw_report(report_basename),
        "",
    ]
    out_path.write_text("\n".join(parts), encoding="utf-8")


@click.command()
@click.option(
    "--report",
    "report_path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Path to the calibration JSONL report.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    default=Path("docs/concepts/calibration.md"),
    show_default=True,
    help="Output Markdown path.",
)
def main(report_path: Path, out_path: Path) -> None:
    """Render a calibration JSONL report to a Markdown docs page."""
    render(
        report_path=report_path,
        out_path=out_path,
        report_basename=report_path.name,
    )


if __name__ == "__main__":
    main()
