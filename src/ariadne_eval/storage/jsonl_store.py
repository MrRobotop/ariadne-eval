"""JSON Lines export / import for trajectory portability.

JSONL is the archival / share-with-collaborators format. It is a *format*,
not a separate storage backend — both functions take any ``Store`` and
read/write a single ``.jsonl`` file. Each line is a JSON object with two
keys: ``"trajectory"`` and ``"steps"``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ariadne_eval.core.status import TrajectoryStatus
from ariadne_eval.core.trajectory import Step, Trajectory

if TYPE_CHECKING:
    from ariadne_eval.storage.base import Store

__all__ = ["export_jsonl", "import_jsonl"]


async def export_jsonl(
    store: "Store",
    path: Path,
    *,
    agent_name: str | None = None,
    model_id: str | None = None,
    final_status: TrajectoryStatus | None = None,
    started_after: datetime | None = None,
    started_before: datetime | None = None,
    batch_size: int = 100,
) -> int:
    """Stream matching trajectories from ``store`` to ``path`` as JSONL.

    Returns the number of trajectories written. The output file is opened in
    write mode (truncates). Each line is::

        {"trajectory": {...}, "steps": [{...}, ...]}
    """
    written = 0
    offset = 0
    with Path(path).open("w", encoding="utf-8") as fh:
        while True:
            page = await store.list_trajectories(
                agent_name=agent_name,
                model_id=model_id,
                final_status=final_status,
                started_after=started_after,
                started_before=started_before,
                limit=batch_size,
                offset=offset,
            )
            if not page:
                break
            for traj in page:
                _, steps = await store.get_trajectory(traj.id)
                fh.write(
                    json.dumps(
                        {
                            "trajectory": json.loads(traj.model_dump_json()),
                            "steps": [json.loads(s.model_dump_json()) for s in steps],
                        }
                    )
                    + "\n"
                )
                written += 1
            offset += batch_size
    return written


async def import_jsonl(path: Path, store: "Store") -> int:
    """Read a JSONL file and save each trajectory + steps into ``store``.

    Returns the number imported. Raises :class:`ValueError` (with the
    offending line number) if any line is not valid JSON, does not have the
    expected shape, or fails Pydantic validation.
    """
    from pydantic import ValidationError

    imported = 0
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {lineno}: invalid JSON ({exc})") from exc
            if not isinstance(obj, dict) or "trajectory" not in obj or "steps" not in obj:
                raise ValueError(
                    f"line {lineno}: missing 'trajectory' or 'steps' key"
                )
            try:
                traj = Trajectory.model_validate(obj["trajectory"])
                steps = [Step.model_validate(s) for s in obj["steps"]]
            except ValidationError as exc:
                raise ValueError(
                    f"line {lineno}: failed to validate trajectory/steps: {exc}"
                ) from exc
            await store.save_trajectory(traj, steps)
            imported += 1
    return imported
