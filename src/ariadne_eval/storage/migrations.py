"""Filesystem-backed schema migrations.

Convention: each migration is a ``.sql`` file in a directory, named
``NNN_<short_name>.sql`` where ``NNN`` is a zero-padded three-digit version.
Versions are unique and applied in numerical order.

A ``_meta`` table in the target database tracks which versions have been
applied. Re-running ``apply_pending`` is idempotent: only versions strictly
greater than ``MAX(version)`` in ``_meta`` are applied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ariadne_eval.storage.base import StoreError

if TYPE_CHECKING:
    import duckdb

__all__ = ["Migration", "apply_pending", "discover_migrations"]


_FILENAME_RE = re.compile(r"^(\d{3})_([a-z0-9_]+)\.sql$", re.IGNORECASE)


@dataclass(frozen=True)
class Migration:
    """A single migration: version, short name, and the SQL body."""

    version: int
    name: str
    sql: str


def discover_migrations(directory: Path) -> list[Migration]:
    """Return the migrations in ``directory`` sorted by version ascending.

    Raises ``ValueError`` if any ``.sql`` file's name does not match the
    convention. Non-``.sql`` files are silently ignored.
    """
    migrations: list[Migration] = []
    for entry in sorted(directory.iterdir()):
        if entry.suffix.lower() != ".sql":
            continue
        match = _FILENAME_RE.match(entry.name)
        if match is None:
            raise ValueError(f"migration filename does not match NNN_<name>.sql: {entry.name!r}")
        version = int(match.group(1))
        name = match.group(2)
        migrations.append(Migration(version=version, name=name, sql=entry.read_text()))
    migrations.sort(key=lambda m: m.version)
    return migrations


def _ensure_meta_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _meta (
            version INTEGER PRIMARY KEY,
            name VARCHAR NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL
        )
        """
    )


def _current_version(conn: duckdb.DuckDBPyConnection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(version), 0) FROM _meta").fetchone()
    if row is None:  # pragma: no cover - COALESCE guarantees a row
        return 0
    return int(row[0])


def apply_pending(conn: duckdb.DuckDBPyConnection, directory: Path) -> int:
    """Apply pending migrations and return the number applied.

    Every migration in ``directory`` strictly newer than the ``MAX(version)``
    recorded in ``_meta`` is applied. Each runs inside a transaction; on
    failure the transaction is rolled back and the exception is re-raised
    wrapped in :class:`StoreError`.
    """
    _ensure_meta_table(conn)
    current = _current_version(conn)

    pending = [m for m in discover_migrations(directory) if m.version > current]
    if not pending:
        return 0

    for mig in pending:
        try:
            conn.execute("BEGIN TRANSACTION")
            conn.execute(mig.sql)
            conn.execute(
                "INSERT INTO _meta (version, name, applied_at) VALUES (?, ?, ?)",
                [mig.version, mig.name, datetime.now(tz=UTC)],
            )
            conn.execute("COMMIT")
        except Exception as exc:
            conn.execute("ROLLBACK")
            raise StoreError(f"migration {mig.version:03d}_{mig.name} failed: {exc}") from exc

    return len(pending)
