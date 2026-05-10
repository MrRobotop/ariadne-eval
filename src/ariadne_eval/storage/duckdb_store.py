"""DuckDB-backed Store implementation.

DuckDB's Python API is synchronous. Each public coroutine wraps its blocking
work in ``asyncio.to_thread`` so callers can stay on the event loop. A single
``asyncio.Lock`` per store instance serializes writes — DuckDB is
single-writer, but reads are safe to run concurrently.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

import duckdb

from ariadne_eval.core.status import TrajectoryStatus
from ariadne_eval.core.trajectory import Step, Trajectory
from ariadne_eval.storage import migrations as _migrations
from ariadne_eval.storage.base import MetadataTooLargeError, TrajectoryNotFoundError

__all__ = ["DuckDBStore"]


_MAX_METADATA_BYTES = 1_048_576  # 1 MB


def _check_metadata_size(metadata: dict[str, Any]) -> None:
    encoded = json.dumps(metadata, default=str)
    size = len(encoded.encode("utf-8"))
    if size > _MAX_METADATA_BYTES:
        raise MetadataTooLargeError(actual=size, max=_MAX_METADATA_BYTES)


def _row_to_trajectory(row: tuple[Any, ...]) -> Trajectory:
    """Reconstruct a Trajectory from a SELECT * row in column order.

    Column order matches 001_initial.sql.
    """
    return Trajectory.model_validate(
        {
            "id": row[0],
            "schema_version": row[1],
            "task": row[2],
            "agent_name": row[3],
            "agent_version": row[4],
            "model_id": row[5],
            "started_at": row[6],
            "finished_at": row[7],
            "final_status": row[8],
            "final_answer": json.loads(row[9]) if row[9] is not None else None,
            "root_step_id": row[10],
            "metadata": json.loads(row[11]),
        }
    )


def _row_to_step(row: tuple[Any, ...]) -> Step:
    """Reconstruct a Step from a SELECT * row in column order."""
    payload = json.loads(row[8])
    payload["step_type"] = row[3]
    return Step.model_validate(
        {
            "id": row[0],
            "trajectory_id": row[1],
            "parent_step_id": row[2],
            "name": row[4],
            "started_at": row[5],
            "finished_at": row[6],
            "status": row[7],
            "payload": payload,
            "error": json.loads(row[9]) if row[9] is not None else None,
            "metadata": json.loads(row[10]),
        }
    )


def _resolve_path(arg_path: Path | None) -> Path:
    """Resolve the database path. Constructor arg > env var > default."""
    if arg_path is not None:
        return Path(arg_path)
    env_value = os.environ.get("ARIADNE_STORE_PATH")
    if env_value:
        return Path(env_value)
    # Recompute the default each call so HOME monkeypatching in tests is honoured.
    return Path.home() / ".ariadne" / "store.duckdb"


class DuckDBStore:
    """DuckDB-backed implementation of the ``Store`` Protocol."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = _resolve_path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: duckdb.DuckDBPyConnection | None = duckdb.connect(str(self._path))
        self._write_lock = asyncio.Lock()

        bundled = files("ariadne_eval.storage") / "migrations_sql"
        _migrations.apply_pending(self._conn, Path(str(bundled)))

    @property
    def path(self) -> Path:
        """The on-disk DuckDB file path."""
        return self._path

    async def close(self) -> None:
        """Close the underlying DuckDB connection. Idempotent."""
        if self._conn is not None:
            conn = self._conn
            self._conn = None
            await asyncio.to_thread(conn.close)

    async def save_trajectory(self, traj: Trajectory, steps: list[Step]) -> None:
        """Persist a trajectory and its steps. Upserts on the same id."""
        _check_metadata_size(traj.metadata)
        for step in steps:
            _check_metadata_size(step.metadata)

        traj_row: tuple[Any, ...] = (
            traj.id,
            traj.schema_version,
            traj.task,
            traj.agent_name,
            traj.agent_version,
            traj.model_id,
            traj.started_at,
            traj.finished_at,
            traj.final_status.value,
            json.dumps(traj.final_answer, default=str)
            if traj.final_answer is not None
            else None,
            traj.root_step_id,
            json.dumps(traj.metadata, default=str),
        )
        step_rows: list[tuple[Any, ...]] = [
            (
                s.id,
                s.trajectory_id,
                s.parent_step_id,
                s.payload.step_type,
                s.name,
                s.started_at,
                s.finished_at,
                s.status.value,
                s.payload.model_dump_json(),
                s.error.model_dump_json() if s.error is not None else None,
                json.dumps(s.metadata, default=str),
            )
            for s in steps
        ]

        async with self._write_lock:
            await asyncio.to_thread(self._save_sync, traj_row, step_rows, traj.id)

    def _save_sync(
        self,
        traj_row: tuple[Any, ...],
        step_rows: list[tuple[Any, ...]],
        traj_id: str,
    ) -> None:
        assert self._conn is not None
        conn = self._conn
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO trajectories
                (id, schema_version, task, agent_name, agent_version, model_id,
                 started_at, finished_at, final_status, final_answer,
                 root_step_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                list(traj_row),
            )
            conn.execute(
                "DELETE FROM steps WHERE trajectory_id = ?", [traj_id]
            )
            if step_rows:
                conn.executemany(
                    """
                    INSERT INTO steps
                    (id, trajectory_id, parent_step_id, step_type, name,
                     started_at, finished_at, status, payload, error, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [list(r) for r in step_rows],
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    async def get_trajectory(
        self, traj_id: str
    ) -> tuple[Trajectory, list[Step]]:
        """Load a trajectory + its steps. Raises TrajectoryNotFoundError on miss."""
        traj_row, step_rows = await asyncio.to_thread(self._get_sync, traj_id)
        if traj_row is None:
            raise TrajectoryNotFoundError(traj_id)
        traj = _row_to_trajectory(traj_row)
        steps = [_row_to_step(r) for r in step_rows]
        return traj, steps

    def _get_sync(
        self, traj_id: str
    ) -> tuple[tuple[Any, ...] | None, list[tuple[Any, ...]]]:
        assert self._conn is not None
        traj_row = self._conn.execute(
            "SELECT * FROM trajectories WHERE id = ?", [traj_id]
        ).fetchone()
        if traj_row is None:
            return None, []
        step_rows = self._conn.execute(
            "SELECT * FROM steps WHERE trajectory_id = ? ORDER BY started_at, id",
            [traj_id],
        ).fetchall()
        return traj_row, step_rows

    async def list_trajectories(
        self,
        *,
        agent_name: str | None = None,
        model_id: str | None = None,
        final_status: TrajectoryStatus | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Trajectory]:
        """List trajectory metadata (no steps), most-recent first."""
        rows = await asyncio.to_thread(
            self._list_sync,
            agent_name,
            model_id,
            final_status.value if final_status is not None else None,
            started_after,
            started_before,
            limit,
            offset,
        )
        return [_row_to_trajectory(r) for r in rows]

    def _list_sync(
        self,
        agent_name: str | None,
        model_id: str | None,
        final_status: str | None,
        started_after: datetime | None,
        started_before: datetime | None,
        limit: int,
        offset: int,
    ) -> list[tuple[Any, ...]]:
        assert self._conn is not None
        sql = """
            SELECT * FROM trajectories
            WHERE (? IS NULL OR agent_name   = ?)
              AND (? IS NULL OR model_id     = ?)
              AND (? IS NULL OR final_status = ?)
              AND (? IS NULL OR started_at  >= ?)
              AND (? IS NULL OR started_at  <= ?)
            ORDER BY started_at DESC
            LIMIT ? OFFSET ?
        """
        params: list[Any] = [
            agent_name, agent_name,
            model_id, model_id,
            final_status, final_status,
            started_after, started_after,
            started_before, started_before,
            limit, offset,
        ]
        return self._conn.execute(sql, params).fetchall()

    async def count(
        self,
        *,
        agent_name: str | None = None,
        model_id: str | None = None,
        final_status: TrajectoryStatus | None = None,
    ) -> int:
        """Count trajectories matching the filters."""
        return await asyncio.to_thread(
            self._count_sync,
            agent_name,
            model_id,
            final_status.value if final_status is not None else None,
        )

    def _count_sync(
        self,
        agent_name: str | None,
        model_id: str | None,
        final_status: str | None,
    ) -> int:
        assert self._conn is not None
        sql = """
            SELECT COUNT(*) FROM trajectories
            WHERE (? IS NULL OR agent_name   = ?)
              AND (? IS NULL OR model_id     = ?)
              AND (? IS NULL OR final_status = ?)
        """
        params: list[Any] = [
            agent_name, agent_name,
            model_id, model_id,
            final_status, final_status,
        ]
        row = self._conn.execute(sql, params).fetchone()
        return int(row[0]) if row else 0

    async def delete_trajectory(self, traj_id: str) -> None:
        """Idempotently remove a trajectory and its steps."""
        async with self._write_lock:
            await asyncio.to_thread(self._delete_sync, traj_id)

    def _delete_sync(self, traj_id: str) -> None:
        assert self._conn is not None
        conn = self._conn
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute("DELETE FROM steps WHERE trajectory_id = ?", [traj_id])
            conn.execute("DELETE FROM trajectories WHERE id = ?", [traj_id])
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
