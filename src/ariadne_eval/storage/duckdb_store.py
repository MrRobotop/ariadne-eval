"""DuckDB-backed Store implementation.

DuckDB's Python API is synchronous. Each public coroutine wraps its blocking
work in ``asyncio.to_thread`` so callers can stay on the event loop. A single
``asyncio.Lock`` per store instance serializes writes — DuckDB is
single-writer, but reads are safe to run concurrently.
"""

from __future__ import annotations

import asyncio
import os
from importlib.resources import files
from pathlib import Path

import duckdb

from ariadne_eval.storage import migrations as _migrations

__all__ = ["DuckDBStore"]


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
