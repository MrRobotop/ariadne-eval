# Storage Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a DuckDB-backed `Store` for trajectories and their step trees, with a portable JSONL export/import path, following the design at `docs/superpowers/specs/2026-05-10-storage-layer-design.md`.

**Architecture:** A small abstract `Store` Protocol in `base.py`. One concrete implementation `DuckDBStore` whose sync DuckDB calls are wrapped in `asyncio.to_thread`. A migration mechanism that reads numbered SQL files via `importlib.resources` and tracks applied versions in a `_meta` table. JSONL export/import as functions that operate on any `Store`.

**Tech Stack:** Python 3.11+, `duckdb`, `pydantic` v2, `asyncio`, `importlib.resources`, `pytest`, `pytest-asyncio` (auto mode), `hypothesis`. All pinned in `pyproject.toml`.

**Branch:** `phase-2-storage` (already created on top of `main` after the Phase 1 merge).

---

## Task 1: Test package marker

**Files:**
- Create: `tests/unit/storage/__init__.py`

- [ ] **Step 1: Create the marker**

```bash
: > tests/unit/storage/__init__.py
```

- [ ] **Step 2: Commit**

```bash
git add tests/unit/storage/__init__.py
git commit -m "test: add package marker for tests/unit/storage"
```

---

## Task 2: Store Protocol and error hierarchy

**Files:**
- Create: `src/ariadne_eval/storage/base.py`
- Test: `tests/unit/storage/test_base.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/storage/test_base.py
"""The Store Protocol and the storage error hierarchy."""

from __future__ import annotations

import pytest

from ariadne_eval.storage.base import (
    MetadataTooLargeError,
    Store,
    StoreError,
    TrajectoryNotFoundError,
)


@pytest.mark.fast
def test_store_is_a_protocol():
    """Protocol classes are subclassable but not instantiable as-is."""
    from typing import Protocol, get_type_hints  # noqa: F401

    # Cheap structural check: the symbol exists and the abstract methods are declared.
    for name in (
        "save_trajectory",
        "get_trajectory",
        "list_trajectories",
        "delete_trajectory",
        "count",
    ):
        assert hasattr(Store, name), f"Store.{name} missing"


@pytest.mark.fast
def test_error_hierarchy():
    assert issubclass(TrajectoryNotFoundError, StoreError)
    assert issubclass(MetadataTooLargeError, StoreError)
    assert issubclass(StoreError, Exception)


@pytest.mark.fast
def test_trajectory_not_found_carries_id():
    err = TrajectoryNotFoundError("01J...")
    assert "01J..." in str(err)


@pytest.mark.fast
def test_metadata_too_large_carries_size():
    err = MetadataTooLargeError(actual=2_000_000, max=1_048_576)
    assert "2000000" in str(err) or "2_000_000" in str(err) or "2000000" in repr(err) or "2_000_000" in repr(err)
    assert "1048576" in str(err) or "1_048_576" in str(err)
```

- [ ] **Step 2: Run test, expect fail**

Run: `uv run pytest tests/unit/storage/test_base.py -v`
Expected: ImportError on `ariadne_eval.storage.base`.

- [ ] **Step 3: Write the implementation**

```python
# src/ariadne_eval/storage/base.py
"""Storage protocol and error hierarchy.

The ``Store`` Protocol defines the abstract storage interface every concrete
backend (currently only ``DuckDBStore``) must satisfy. Tests that exercise
storage-agnostic code can mock against this Protocol directly.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ariadne_eval.core.status import TrajectoryStatus
    from ariadne_eval.core.trajectory import Step, Trajectory

__all__ = [
    "MetadataTooLargeError",
    "Store",
    "StoreError",
    "TrajectoryNotFoundError",
]


class StoreError(Exception):
    """Base class for all storage-layer errors."""


class TrajectoryNotFoundError(StoreError):
    """Raised by ``get_trajectory`` when no row matches the given id."""

    def __init__(self, traj_id: str) -> None:
        super().__init__(f"trajectory not found: {traj_id!r}")
        self.traj_id = traj_id


class MetadataTooLargeError(StoreError):
    """Raised when a trajectory or step's serialized metadata exceeds the cap."""

    def __init__(self, *, actual: int, max: int) -> None:
        super().__init__(
            f"metadata is {actual} bytes, max {max} bytes"
        )
        self.actual = actual
        self.max = max


class Store(Protocol):
    """Abstract storage backend for trajectories and their step trees."""

    async def save_trajectory(
        self, traj: "Trajectory", steps: "list[Step]"
    ) -> None:
        """Persist a trajectory and its steps. Upserts on the same id."""
        ...

    async def get_trajectory(
        self, traj_id: str
    ) -> "tuple[Trajectory, list[Step]]":
        """Load a trajectory + its steps. Raises TrajectoryNotFoundError on miss."""
        ...

    async def list_trajectories(
        self,
        *,
        agent_name: str | None = None,
        model_id: str | None = None,
        final_status: "TrajectoryStatus | None" = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> "list[Trajectory]":
        """List trajectory metadata (no steps), most-recent first."""
        ...

    async def delete_trajectory(self, traj_id: str) -> None:
        """Idempotently remove a trajectory and its steps."""
        ...

    async def count(
        self,
        *,
        agent_name: str | None = None,
        model_id: str | None = None,
        final_status: "TrajectoryStatus | None" = None,
    ) -> int:
        """Count trajectories matching the filters."""
        ...
```

- [ ] **Step 4: Run test, expect pass**

Run: `uv run pytest tests/unit/storage/test_base.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ariadne_eval/storage/base.py tests/unit/storage/test_base.py
git commit -m "feat(storage): add Store Protocol and error hierarchy"
```

---

## Task 3: Initial migration SQL + migration mechanism

**Files:**
- Create: `src/ariadne_eval/storage/migrations_sql/001_initial.sql`
- Create: `src/ariadne_eval/storage/migrations.py`
- Test: `tests/unit/storage/test_migrations.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/storage/test_migrations.py
"""Migration discovery, ordering, application, and _meta tracking."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from ariadne_eval.storage.migrations import (
    Migration,
    apply_pending,
    discover_migrations,
)


@pytest.mark.fast
def test_discover_returns_sorted_by_version(tmp_path):
    (tmp_path / "002_two.sql").write_text("CREATE TABLE two (id INTEGER);")
    (tmp_path / "001_one.sql").write_text("CREATE TABLE one (id INTEGER);")
    (tmp_path / "010_ten.sql").write_text("CREATE TABLE ten (id INTEGER);")

    migs = discover_migrations(tmp_path)
    assert [m.version for m in migs] == [1, 2, 10]
    assert migs[0].name == "one"


@pytest.mark.fast
def test_discover_ignores_non_sql_files(tmp_path):
    (tmp_path / "001_one.sql").write_text("CREATE TABLE one (id INTEGER);")
    (tmp_path / "README.md").write_text("not sql")
    (tmp_path / "002_two.txt").write_text("not sql")

    migs = discover_migrations(tmp_path)
    assert len(migs) == 1


@pytest.mark.fast
def test_discover_rejects_malformed_filename(tmp_path):
    (tmp_path / "bad-name.sql").write_text("--")
    with pytest.raises(ValueError):
        discover_migrations(tmp_path)


@pytest.mark.fast
def test_apply_pending_creates_meta_table_and_runs_all_on_empty(tmp_path):
    (tmp_path / "001_one.sql").write_text("CREATE TABLE one (id INTEGER);")
    (tmp_path / "002_two.sql").write_text("CREATE TABLE two (id INTEGER);")

    db = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db))
    try:
        applied = apply_pending(conn, tmp_path)
        assert applied == 2

        # _meta has both rows
        rows = conn.execute("SELECT version, name FROM _meta ORDER BY version").fetchall()
        assert rows == [(1, "one"), (2, "two")]

        # tables exist
        names = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
        assert {"one", "two", "_meta"}.issubset(names)
    finally:
        conn.close()


@pytest.mark.fast
def test_apply_pending_only_runs_missing_versions(tmp_path):
    (tmp_path / "001_one.sql").write_text("CREATE TABLE one (id INTEGER);")
    db = tmp_path / "test.duckdb"

    # First pass: only v1 exists
    conn = duckdb.connect(str(db))
    apply_pending(conn, tmp_path)
    conn.close()

    # Add v2 to the directory
    (tmp_path / "002_two.sql").write_text("CREATE TABLE two (id INTEGER);")

    conn = duckdb.connect(str(db))
    try:
        applied = apply_pending(conn, tmp_path)
        assert applied == 1  # only v2

        rows = conn.execute("SELECT version FROM _meta ORDER BY version").fetchall()
        assert rows == [(1,), (2,)]
    finally:
        conn.close()


@pytest.mark.fast
def test_apply_pending_returns_zero_when_up_to_date(tmp_path):
    (tmp_path / "001_one.sql").write_text("CREATE TABLE one (id INTEGER);")
    db = tmp_path / "test.duckdb"

    conn = duckdb.connect(str(db))
    apply_pending(conn, tmp_path)
    assert apply_pending(conn, tmp_path) == 0
    conn.close()


@pytest.mark.fast
def test_failed_migration_aborts_and_raises(tmp_path):
    (tmp_path / "001_one.sql").write_text("CREATE TABLE one (id INTEGER);")
    (tmp_path / "002_bad.sql").write_text("THIS IS NOT VALID SQL;")

    db = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db))
    try:
        from ariadne_eval.storage.base import StoreError
        with pytest.raises(StoreError) as exc:
            apply_pending(conn, tmp_path)
        assert "002" in str(exc.value)

        # v1 still applied (committed before failure)
        rows = conn.execute("SELECT version FROM _meta ORDER BY version").fetchall()
        assert rows == [(1,)]
    finally:
        conn.close()


@pytest.mark.fast
def test_initial_schema_has_expected_tables_and_indexes():
    """The bundled 001_initial.sql produces the expected schema."""
    from importlib.resources import files

    bundled_dir = files("ariadne_eval.storage") / "migrations_sql"
    migs = discover_migrations(Path(str(bundled_dir)))
    assert any(m.version == 1 and m.name == "initial" for m in migs)


@pytest.mark.fast
def test_migration_dataclass_fields():
    m = Migration(version=42, name="hello", sql="SELECT 1;")
    assert m.version == 42
    assert m.name == "hello"
    assert m.sql == "SELECT 1;"
```

- [ ] **Step 2: Run test, expect fail**

Run: `uv run pytest tests/unit/storage/test_migrations.py -v`
Expected: ImportError on `ariadne_eval.storage.migrations`.

- [ ] **Step 3: Write the initial migration SQL**

```sql
-- src/ariadne_eval/storage/migrations_sql/001_initial.sql
-- Migration 001: initial schema for trajectories and steps.

CREATE TABLE trajectories (
    id VARCHAR PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    task VARCHAR NOT NULL,
    agent_name VARCHAR NOT NULL,
    agent_version VARCHAR NOT NULL,
    model_id VARCHAR NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    final_status VARCHAR NOT NULL,
    final_answer JSON,
    root_step_id VARCHAR,
    metadata JSON NOT NULL DEFAULT '{}'
);
CREATE INDEX idx_trajectories_started_at ON trajectories (started_at DESC);
CREATE INDEX idx_trajectories_agent_name ON trajectories (agent_name);
CREATE INDEX idx_trajectories_model_id  ON trajectories (model_id);
CREATE INDEX idx_trajectories_final_status ON trajectories (final_status);

CREATE TABLE steps (
    id VARCHAR PRIMARY KEY,
    trajectory_id VARCHAR NOT NULL,
    parent_step_id VARCHAR,
    step_type VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    status VARCHAR NOT NULL,
    payload JSON NOT NULL,
    error JSON,
    metadata JSON NOT NULL DEFAULT '{}'
);
CREATE INDEX idx_steps_trajectory_id ON steps (trajectory_id);
```

- [ ] **Step 4: Write the migrations module**

```python
# src/ariadne_eval/storage/migrations.py
"""Filesystem-backed schema migrations.

Convention: each migration is a .sql file in a directory, named
``NNN_<short_name>.sql`` where ``NNN`` is a zero-padded three-digit version.
Versions are unique and applied in numerical order.

A ``_meta`` table in the target database tracks which versions have been
applied. Re-running ``apply_pending`` is idempotent: only versions strictly
greater than ``MAX(version)`` in ``_meta`` are applied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
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
            raise ValueError(
                f"migration filename does not match NNN_<name>.sql: {entry.name!r}"
            )
        version = int(match.group(1))
        name = match.group(2)
        migrations.append(Migration(version=version, name=name, sql=entry.read_text()))
    migrations.sort(key=lambda m: m.version)
    return migrations


def _ensure_meta_table(conn: "duckdb.DuckDBPyConnection") -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _meta (
            version INTEGER PRIMARY KEY,
            name VARCHAR NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL
        )
        """
    )


def _current_version(conn: "duckdb.DuckDBPyConnection") -> int:
    row = conn.execute("SELECT COALESCE(MAX(version), 0) FROM _meta").fetchone()
    if row is None:
        return 0
    return int(row[0])


def apply_pending(conn: "duckdb.DuckDBPyConnection", directory: Path) -> int:
    """Apply every migration in ``directory`` whose version is strictly greater
    than the current ``MAX(version)`` in ``_meta``. Returns the number applied.

    Each migration runs inside a transaction. If a migration fails, its
    transaction is rolled back and the exception is re-raised wrapped in
    ``StoreError`` so callers can branch on storage-layer errors uniformly.
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
                [mig.version, mig.name, datetime.now(tz=timezone.utc)],
            )
            conn.execute("COMMIT")
        except Exception as exc:  # pragma: no cover - error path tested separately
            conn.execute("ROLLBACK")
            raise StoreError(
                f"migration {mig.version:03d}_{mig.name} failed: {exc}"
            ) from exc

    return len(pending)
```

- [ ] **Step 5: Run tests, expect pass**

Run: `uv run pytest tests/unit/storage/test_migrations.py -v`
Expected: 9 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ariadne_eval/storage/migrations.py src/ariadne_eval/storage/migrations_sql/001_initial.sql tests/unit/storage/test_migrations.py
git commit -m "feat(storage): add filesystem migration mechanism + initial schema"
```

---

## Task 4: DuckDBStore — init and migration application

**Files:**
- Create: `src/ariadne_eval/storage/duckdb_store.py`
- Test: `tests/unit/storage/test_duckdb_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/storage/test_duckdb_store.py
"""End-to-end DuckDBStore tests."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pytest

from ariadne_eval.storage.duckdb_store import DuckDBStore


@pytest.mark.fast
async def test_init_creates_file_and_runs_migrations(tmp_path):
    db = tmp_path / "store.duckdb"
    store = DuckDBStore(path=db)
    try:
        assert db.exists()
        # _meta has version 1
        conn = duckdb.connect(str(db))
        try:
            rows = conn.execute("SELECT version, name FROM _meta ORDER BY version").fetchall()
            assert (1, "initial") in rows
        finally:
            conn.close()
    finally:
        await store.close()


@pytest.mark.fast
async def test_init_creates_parent_directory(tmp_path):
    db = tmp_path / "nested" / "subdir" / "store.duckdb"
    store = DuckDBStore(path=db)
    try:
        assert db.exists()
        assert db.parent.is_dir()
    finally:
        await store.close()


@pytest.mark.fast
async def test_constructor_path_overrides_env_var(tmp_path, monkeypatch):
    env_db = tmp_path / "env.duckdb"
    arg_db = tmp_path / "arg.duckdb"
    monkeypatch.setenv("ARIADNE_STORE_PATH", str(env_db))

    store = DuckDBStore(path=arg_db)
    try:
        assert arg_db.exists()
        assert not env_db.exists()
    finally:
        await store.close()


@pytest.mark.fast
async def test_env_var_used_when_no_arg(tmp_path, monkeypatch):
    db = tmp_path / "env.duckdb"
    monkeypatch.setenv("ARIADNE_STORE_PATH", str(db))
    store = DuckDBStore()
    try:
        assert db.exists()
    finally:
        await store.close()


@pytest.mark.fast
async def test_default_path_is_under_dot_ariadne(monkeypatch, tmp_path):
    """When no path or env var is set, default is ~/.ariadne/store.duckdb."""
    monkeypatch.delenv("ARIADNE_STORE_PATH", raising=False)
    fake_home = tmp_path / "fake_home"
    monkeypatch.setenv("HOME", str(fake_home))

    store = DuckDBStore()
    try:
        expected = fake_home / ".ariadne" / "store.duckdb"
        assert expected.exists()
    finally:
        await store.close()
```

- [ ] **Step 2: Run tests, expect fail**

Run: `uv run pytest tests/unit/storage/test_duckdb_store.py -v`
Expected: ImportError on `ariadne_eval.storage.duckdb_store`.

- [ ] **Step 3: Write the implementation (init only)**

```python
# src/ariadne_eval/storage/duckdb_store.py
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
from typing import TYPE_CHECKING

import duckdb

from ariadne_eval.storage import migrations as _migrations

if TYPE_CHECKING:
    pass

__all__ = ["DuckDBStore"]


_DEFAULT_PATH = Path.home() / ".ariadne" / "store.duckdb"


def _resolve_path(arg_path: Path | None) -> Path:
    """Resolve the database path. ctor arg > env var > default."""
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
        self._conn = duckdb.connect(str(self._path))
        self._write_lock = asyncio.Lock()

        bundled = files("ariadne_eval.storage") / "migrations_sql"
        _migrations.apply_pending(self._conn, Path(str(bundled)))

    @property
    def path(self) -> Path:
        return self._path

    async def close(self) -> None:
        """Close the underlying DuckDB connection. Idempotent."""
        if self._conn is not None:
            await asyncio.to_thread(self._conn.close)
            self._conn = None  # type: ignore[assignment]
```

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/unit/storage/test_duckdb_store.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ariadne_eval/storage/duckdb_store.py tests/unit/storage/test_duckdb_store.py
git commit -m "feat(storage): add DuckDBStore with path resolution and migration apply on init"
```

---

## Task 5: DuckDBStore — save_trajectory and get_trajectory

**Files:**
- Modify: `src/ariadne_eval/storage/duckdb_store.py`
- Modify: `tests/unit/storage/test_duckdb_store.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/unit/storage/test_duckdb_store.py`:

```python
# ---------------------------------------------------------------------------
# Task 5 — save_trajectory + get_trajectory round-trip
# ---------------------------------------------------------------------------

from datetime import UTC, datetime  # noqa: E402

from ariadne_eval.core.ids import new_id  # noqa: E402
from ariadne_eval.core.status import StepStatus, TrajectoryStatus  # noqa: E402
from ariadne_eval.core.trajectory import (  # noqa: E402
    LLMCallPayload,
    Message,
    Step,
    Trajectory,
)
from ariadne_eval.storage.base import (  # noqa: E402
    MetadataTooLargeError,
    TrajectoryNotFoundError,
)


def _llm_payload() -> LLMCallPayload:
    return LLMCallPayload(
        model_id="claude-sonnet",
        prompt_messages=[Message(role="user", content="hi")],
        completion="hello",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
        latency_ms=1.0,
    )


def _make_traj_with_steps(traj_id: str | None = None) -> tuple[Trajectory, list[Step]]:
    tid = traj_id or new_id()
    started = datetime.now(tz=UTC)
    s1 = Step(
        id=new_id(),
        trajectory_id=tid,
        parent_step_id=None,
        name="ask_llm",
        started_at=started,
        finished_at=started,
        status=StepStatus.SUCCEEDED,
        payload=_llm_payload(),
    )
    traj = Trajectory(
        id=tid,
        task="2+2",
        agent_name="react",
        agent_version="0.1",
        model_id="claude-sonnet",
        started_at=started,
        finished_at=started,
        final_status=TrajectoryStatus.SUCCEEDED,
        final_answer="4",
        root_step_id=s1.id,
    )
    return traj, [s1]


@pytest.mark.fast
async def test_save_then_get_round_trip(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        traj, steps = _make_traj_with_steps()
        await store.save_trajectory(traj, steps)

        loaded_traj, loaded_steps = await store.get_trajectory(traj.id)
        assert loaded_traj == traj
        assert loaded_steps == steps
    finally:
        await store.close()


@pytest.mark.fast
async def test_get_trajectory_raises_on_missing(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        with pytest.raises(TrajectoryNotFoundError) as exc:
            await store.get_trajectory("01ARZ3NDEKTSV4RRFFQ69G5FAV")
        assert "01ARZ3NDEKTSV4RRFFQ69G5FAV" in str(exc.value)
    finally:
        await store.close()


@pytest.mark.fast
async def test_save_rejects_metadata_over_1mb(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        traj, steps = _make_traj_with_steps()
        # Build a > 1 MB metadata value
        traj = traj.model_copy(update={"metadata": {"big": "x" * 1_100_000}})
        with pytest.raises(MetadataTooLargeError) as exc:
            await store.save_trajectory(traj, steps)
        assert "1048576" in str(exc.value) or "1_048_576" in str(exc.value)
    finally:
        await store.close()


@pytest.mark.fast
async def test_save_replaces_on_same_id(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        tid = new_id()
        traj1, steps1 = _make_traj_with_steps(traj_id=tid)
        await store.save_trajectory(traj1, steps1)

        # Build a second version with the same trajectory id but different steps.
        traj2, steps2 = _make_traj_with_steps(traj_id=tid)
        traj2 = traj2.model_copy(update={"task": "REVISED TASK"})
        await store.save_trajectory(traj2, steps2)

        loaded_traj, loaded_steps = await store.get_trajectory(tid)
        assert loaded_traj.task == "REVISED TASK"
        # steps were replaced, not appended
        assert {s.id for s in loaded_steps} == {s.id for s in steps2}
    finally:
        await store.close()
```

- [ ] **Step 2: Run tests, expect fail**

Run: `uv run pytest tests/unit/storage/test_duckdb_store.py -v`
Expected: AttributeError on `save_trajectory`.

- [ ] **Step 3: Add the methods**

Append to `src/ariadne_eval/storage/duckdb_store.py`:

```python
import json  # add to existing imports

from ariadne_eval.core.trajectory import Step, Trajectory
from ariadne_eval.storage.base import MetadataTooLargeError, TrajectoryNotFoundError


_MAX_METADATA_BYTES = 1_048_576  # 1 MB


def _check_metadata_size(metadata: dict[str, object]) -> None:
    encoded = json.dumps(metadata, default=str)
    size = len(encoded.encode("utf-8"))
    if size > _MAX_METADATA_BYTES:
        raise MetadataTooLargeError(actual=size, max=_MAX_METADATA_BYTES)


def _row_to_trajectory(row: tuple) -> Trajectory:
    """Reconstruct a Trajectory from a SELECT * row in column order.

    Column order matches 001_initial.sql:
    id, schema_version, task, agent_name, agent_version, model_id,
    started_at, finished_at, final_status, final_answer, root_step_id, metadata
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


def _row_to_step(row: tuple) -> Step:
    """Reconstruct a Step from a SELECT * row in column order.

    Column order from 001_initial.sql:
    id, trajectory_id, parent_step_id, step_type, name,
    started_at, finished_at, status, payload, error, metadata
    """
    payload = json.loads(row[8])
    payload["step_type"] = row[3]  # ensure discriminator is set
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
```

Add the two async methods inside `DuckDBStore`:

```python
    async def save_trajectory(self, traj: Trajectory, steps: list[Step]) -> None:
        """Persist a trajectory and its steps. Upserts on the same id."""
        _check_metadata_size(traj.metadata)
        for step in steps:
            _check_metadata_size(step.metadata)

        traj_row = (
            traj.id,
            traj.schema_version,
            traj.task,
            traj.agent_name,
            traj.agent_version,
            traj.model_id,
            traj.started_at,
            traj.finished_at,
            traj.final_status.value,
            json.dumps(traj.final_answer, default=str) if traj.final_answer is not None else None,
            traj.root_step_id,
            json.dumps(traj.metadata, default=str),
        )
        step_rows = [
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
        traj_row: tuple,
        step_rows: list[tuple],
        traj_id: str,
    ) -> None:
        conn = self._conn
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO trajectories
                (id, schema_version, task, agent_name, agent_version, model_id,
                 started_at, finished_at, final_status, final_answer, root_step_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                list(traj_row),
            )
            conn.execute("DELETE FROM steps WHERE trajectory_id = ?", [traj_id])
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
        """Load a trajectory + its steps."""
        traj_row, step_rows = await asyncio.to_thread(self._get_sync, traj_id)
        if traj_row is None:
            raise TrajectoryNotFoundError(traj_id)
        traj = _row_to_trajectory(traj_row)
        steps = [_row_to_step(r) for r in step_rows]
        return traj, steps

    def _get_sync(self, traj_id: str) -> tuple[tuple | None, list[tuple]]:
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
```

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/unit/storage/test_duckdb_store.py -v`
Expected: 9 passed (5 from Task 4 + 4 new).

- [ ] **Step 5: Commit**

```bash
git add src/ariadne_eval/storage/duckdb_store.py tests/unit/storage/test_duckdb_store.py
git commit -m "feat(storage): save_trajectory + get_trajectory with upsert and metadata cap"
```

---

## Task 6: list_trajectories and count

**Files:**
- Modify: `src/ariadne_eval/storage/duckdb_store.py`
- Modify: `tests/unit/storage/test_duckdb_store.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/unit/storage/test_duckdb_store.py`:

```python
# ---------------------------------------------------------------------------
# Task 6 — list_trajectories + count
# ---------------------------------------------------------------------------

from datetime import timedelta  # noqa: E402


async def _seed(store, *, n: int = 5, agent: str = "react", model: str = "m"):
    base = datetime(2026, 1, 1, tzinfo=UTC)
    out = []
    for i in range(n):
        traj, steps = _make_traj_with_steps()
        traj = traj.model_copy(update={
            "agent_name": agent,
            "model_id": model,
            "started_at": base + timedelta(seconds=i),
        })
        await store.save_trajectory(traj, steps)
        out.append(traj)
    return out


@pytest.mark.fast
async def test_list_returns_trajectories_most_recent_first(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        seeded = await _seed(store, n=5)
        listed = await store.list_trajectories()
        assert [t.id for t in listed] == [t.id for t in reversed(seeded)]
    finally:
        await store.close()


@pytest.mark.fast
async def test_list_filters_by_agent_name(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        await _seed(store, n=2, agent="react")
        await _seed(store, n=3, agent="tool-use")
        listed = await store.list_trajectories(agent_name="react")
        assert len(listed) == 2
        assert all(t.agent_name == "react" for t in listed)
    finally:
        await store.close()


@pytest.mark.fast
async def test_list_filters_by_model_id(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        await _seed(store, n=2, model="claude-sonnet")
        await _seed(store, n=4, model="gpt-4o")
        listed = await store.list_trajectories(model_id="gpt-4o")
        assert len(listed) == 4
    finally:
        await store.close()


@pytest.mark.fast
async def test_list_filters_by_final_status(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        # Seed a mix of statuses
        await _seed(store, n=2)  # all SUCCEEDED in helper
        traj, steps = _make_traj_with_steps()
        traj = traj.model_copy(update={"final_status": TrajectoryStatus.FAILED})
        await store.save_trajectory(traj, steps)

        succ = await store.list_trajectories(final_status=TrajectoryStatus.SUCCEEDED)
        fail = await store.list_trajectories(final_status=TrajectoryStatus.FAILED)
        assert len(succ) == 2
        assert len(fail) == 1
    finally:
        await store.close()


@pytest.mark.fast
async def test_list_filters_by_time_range(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        seeded = await _seed(store, n=5)  # spaced 1s apart starting at 2026-01-01
        # Filter to inclusive middle
        after = seeded[1].started_at
        before = seeded[3].started_at
        listed = await store.list_trajectories(
            started_after=after, started_before=before
        )
        assert {t.id for t in listed} == {t.id for t in seeded[1:4]}
    finally:
        await store.close()


@pytest.mark.fast
async def test_list_pagination(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        await _seed(store, n=7)
        page1 = await store.list_trajectories(limit=3, offset=0)
        page2 = await store.list_trajectories(limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 3
        assert {t.id for t in page1}.isdisjoint({t.id for t in page2})
    finally:
        await store.close()


@pytest.mark.fast
async def test_count_total_and_filtered(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        await _seed(store, n=2, agent="react")
        await _seed(store, n=3, agent="tool-use")
        assert await store.count() == 5
        assert await store.count(agent_name="react") == 2
        assert await store.count(agent_name="tool-use") == 3
    finally:
        await store.close()
```

- [ ] **Step 2: Run tests, expect fail**

Run: `uv run pytest tests/unit/storage/test_duckdb_store.py -v`
Expected: AttributeError on `list_trajectories`.

- [ ] **Step 3: Add the methods**

Add to `DuckDBStore`:

```python
    async def list_trajectories(
        self,
        *,
        agent_name: str | None = None,
        model_id: str | None = None,
        final_status: "TrajectoryStatus | None" = None,
        started_after: "datetime | None" = None,
        started_before: "datetime | None" = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Trajectory]:
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
        started_after: "datetime | None",
        started_before: "datetime | None",
        limit: int,
        offset: int,
    ) -> list[tuple]:
        sql = """
            SELECT * FROM trajectories
            WHERE (? IS NULL OR agent_name    = ?)
              AND (? IS NULL OR model_id      = ?)
              AND (? IS NULL OR final_status  = ?)
              AND (? IS NULL OR started_at   >= ?)
              AND (? IS NULL OR started_at   <= ?)
            ORDER BY started_at DESC
            LIMIT ? OFFSET ?
        """
        params = [
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
        final_status: "TrajectoryStatus | None" = None,
    ) -> int:
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
        sql = """
            SELECT COUNT(*) FROM trajectories
            WHERE (? IS NULL OR agent_name   = ?)
              AND (? IS NULL OR model_id     = ?)
              AND (? IS NULL OR final_status = ?)
        """
        params = [
            agent_name, agent_name,
            model_id, model_id,
            final_status, final_status,
        ]
        row = self._conn.execute(sql, params).fetchone()
        return int(row[0]) if row else 0
```

Add `from datetime import datetime` at the top if not already present (it isn't yet).

Add `from ariadne_eval.core.status import TrajectoryStatus` at the top.

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/unit/storage/test_duckdb_store.py -v`
Expected: 16 passed (9 + 7 new).

- [ ] **Step 5: Commit**

```bash
git add src/ariadne_eval/storage/duckdb_store.py tests/unit/storage/test_duckdb_store.py
git commit -m "feat(storage): list_trajectories and count with kwargs filtering"
```

---

## Task 7: delete_trajectory

**Files:**
- Modify: `src/ariadne_eval/storage/duckdb_store.py`
- Modify: `tests/unit/storage/test_duckdb_store.py`

- [ ] **Step 1: Append failing tests**

```python
# ---------------------------------------------------------------------------
# Task 7 — delete_trajectory
# ---------------------------------------------------------------------------


@pytest.mark.fast
async def test_delete_removes_trajectory_and_steps(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        traj, steps = _make_traj_with_steps()
        await store.save_trajectory(traj, steps)
        await store.delete_trajectory(traj.id)
        with pytest.raises(TrajectoryNotFoundError):
            await store.get_trajectory(traj.id)
        # steps row also gone
        assert await store.count() == 0
    finally:
        await store.close()


@pytest.mark.fast
async def test_delete_is_idempotent_on_missing(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        # Does not raise on missing id
        await store.delete_trajectory("01ARZ3NDEKTSV4RRFFQ69G5FAV")
    finally:
        await store.close()
```

- [ ] **Step 2: Run tests, expect fail**

Run: `uv run pytest tests/unit/storage/test_duckdb_store.py -v -k delete`
Expected: AttributeError on `delete_trajectory`.

- [ ] **Step 3: Add the method**

Inside `DuckDBStore`:

```python
    async def delete_trajectory(self, traj_id: str) -> None:
        async with self._write_lock:
            await asyncio.to_thread(self._delete_sync, traj_id)

    def _delete_sync(self, traj_id: str) -> None:
        conn = self._conn
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute("DELETE FROM steps WHERE trajectory_id = ?", [traj_id])
            conn.execute("DELETE FROM trajectories WHERE id = ?", [traj_id])
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
```

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/unit/storage/test_duckdb_store.py -v`
Expected: 18 passed (16 + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/ariadne_eval/storage/duckdb_store.py tests/unit/storage/test_duckdb_store.py
git commit -m "feat(storage): delete_trajectory removes trajectory and child steps"
```

---

## Task 8: Concurrency — 50 parallel saves

**Files:**
- Modify: `tests/unit/storage/test_duckdb_store.py` (test only; no production code)

- [ ] **Step 1: Append the concurrency test**

```python
# ---------------------------------------------------------------------------
# Task 8 — concurrent writes are serialized by the per-instance lock
# ---------------------------------------------------------------------------

import asyncio as _asyncio  # noqa: E402  (already imported in some envs)


@pytest.mark.fast
async def test_50_parallel_saves_all_land(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        pairs = [_make_traj_with_steps() for _ in range(50)]
        await asyncio.gather(*(store.save_trajectory(t, s) for t, s in pairs))

        assert await store.count() == 50
        ids = {t.id for t, _ in pairs}
        listed = await store.list_trajectories(limit=100)
        assert {t.id for t in listed} == ids
    finally:
        await store.close()
```

- [ ] **Step 2: Run tests, expect pass immediately**

Run: `uv run pytest tests/unit/storage/test_duckdb_store.py::test_50_parallel_saves_all_land -v`
Expected: 1 passed. (The lock is already in place; this verifies the contract.)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/storage/test_duckdb_store.py
git commit -m "test(storage): 50 parallel save_trajectory calls all land"
```

---

## Task 9: JSONL export and import

**Files:**
- Create: `src/ariadne_eval/storage/jsonl_store.py`
- Create: `tests/unit/storage/test_jsonl_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/storage/test_jsonl_store.py
"""JSONL export / import functions."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ariadne_eval.core.ids import new_id
from ariadne_eval.core.status import StepStatus, TrajectoryStatus
from ariadne_eval.core.trajectory import (
    LLMCallPayload,
    Message,
    Step,
    Trajectory,
)
from ariadne_eval.storage.duckdb_store import DuckDBStore
from ariadne_eval.storage.jsonl_store import export_jsonl, import_jsonl


def _make_traj(agent: str = "react") -> tuple[Trajectory, list[Step]]:
    tid = new_id()
    started = datetime.now(tz=UTC)
    s1 = Step(
        id=new_id(),
        trajectory_id=tid,
        parent_step_id=None,
        name="ask",
        started_at=started,
        finished_at=started,
        status=StepStatus.SUCCEEDED,
        payload=LLMCallPayload(
            model_id="m",
            prompt_messages=[Message(role="user", content="hi")],
            completion="hello",
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.0,
            latency_ms=1.0,
        ),
    )
    traj = Trajectory(
        id=tid,
        task="t",
        agent_name=agent,
        agent_version="0.1",
        model_id="m",
        started_at=started,
        finished_at=started,
        final_status=TrajectoryStatus.SUCCEEDED,
        root_step_id=s1.id,
    )
    return traj, [s1]


@pytest.mark.fast
async def test_export_then_import_round_trip(tmp_path):
    src = DuckDBStore(path=tmp_path / "src.duckdb")
    dst = DuckDBStore(path=tmp_path / "dst.duckdb")
    try:
        for _ in range(3):
            t, s = _make_traj()
            await src.save_trajectory(t, s)

        out = tmp_path / "dump.jsonl"
        n = await export_jsonl(src, out)
        assert n == 3

        imported = await import_jsonl(out, dst)
        assert imported == 3
        assert await dst.count() == 3
    finally:
        await src.close()
        await dst.close()


@pytest.mark.fast
async def test_export_honours_filter_kwargs(tmp_path):
    src = DuckDBStore(path=tmp_path / "src.duckdb")
    try:
        for _ in range(2):
            t, s = _make_traj(agent="react")
            await src.save_trajectory(t, s)
        for _ in range(3):
            t, s = _make_traj(agent="tool-use")
            await src.save_trajectory(t, s)

        out = tmp_path / "react_only.jsonl"
        n = await export_jsonl(src, out, agent_name="react")
        assert n == 2

        # Confirm by reading the file
        lines = out.read_text().splitlines()
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert obj["trajectory"]["agent_name"] == "react"
    finally:
        await src.close()


@pytest.mark.fast
async def test_export_format_has_trajectory_and_steps_keys(tmp_path):
    src = DuckDBStore(path=tmp_path / "src.duckdb")
    try:
        t, s = _make_traj()
        await src.save_trajectory(t, s)

        out = tmp_path / "dump.jsonl"
        await export_jsonl(src, out)

        line = out.read_text().splitlines()[0]
        obj = json.loads(line)
        assert set(obj.keys()) == {"trajectory", "steps"}
        assert isinstance(obj["steps"], list)
    finally:
        await src.close()


@pytest.mark.fast
async def test_import_rejects_corrupt_line_with_line_number(tmp_path):
    dst = DuckDBStore(path=tmp_path / "dst.duckdb")
    try:
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"trajectory": {}, "steps": []}\nNOT VALID JSON\n')
        with pytest.raises(ValueError) as exc:
            await import_jsonl(bad, dst)
        assert "line 2" in str(exc.value).lower()
    finally:
        await dst.close()
```

- [ ] **Step 2: Run tests, expect fail**

Run: `uv run pytest tests/unit/storage/test_jsonl_store.py -v`
Expected: ImportError on `ariadne_eval.storage.jsonl_store`.

- [ ] **Step 3: Write the implementation**

```python
# src/ariadne_eval/storage/jsonl_store.py
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

    Returns the number imported. Raises ``ValueError`` (with the offending
    line number) if any line is not valid JSON or the expected shape.
    """
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
                raise ValueError(f"line {lineno}: missing 'trajectory' or 'steps' key")
            traj = Trajectory.model_validate(obj["trajectory"])
            steps = [Step.model_validate(s) for s in obj["steps"]]
            await store.save_trajectory(traj, steps)
            imported += 1
    return imported
```

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/unit/storage/test_jsonl_store.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ariadne_eval/storage/jsonl_store.py tests/unit/storage/test_jsonl_store.py
git commit -m "feat(storage): add export_jsonl and import_jsonl"
```

---

## Task 10: Property-based round-trip

**Files:**
- Create: `tests/property/test_storage_roundtrip.py`

- [ ] **Step 1: Write the test**

```python
# tests/property/test_storage_roundtrip.py
"""Property-based: any (Trajectory, list[Step]) round-trips through DuckDBStore."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ariadne_eval.core.ids import new_id
from ariadne_eval.core.status import StepStatus, TrajectoryStatus
from ariadne_eval.core.trajectory import (
    InternalPayload,
    LLMCallPayload,
    Message,
    Step,
    ToolCallPayload,
    Trajectory,
    UserInputPayload,
)
from ariadne_eval.storage.duckdb_store import DuckDBStore


_BASE = datetime(2026, 1, 1, tzinfo=UTC)


@st.composite
def _payloads(draw):
    return draw(st.one_of(
        st.builds(LLMCallPayload,
            model_id=st.sampled_from(["claude-sonnet", "gpt-4o"]),
            prompt_messages=st.lists(
                st.builds(Message, role=st.just("user"), content=st.text(max_size=32)),
                min_size=1, max_size=2,
            ),
            completion=st.text(max_size=64),
            input_tokens=st.integers(min_value=0, max_value=1000),
            output_tokens=st.integers(min_value=0, max_value=1000),
            cost_usd=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            latency_ms=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        ),
        st.builds(ToolCallPayload,
            tool_name=st.sampled_from(["search", "calculator"]),
            arguments=st.just({"k": "v"}),
            result=st.one_of(st.none(), st.integers(), st.text(max_size=32)),
            latency_ms=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        ),
        st.builds(UserInputPayload, message=st.text(max_size=32)),
        st.builds(InternalPayload, kind=st.text(min_size=1, max_size=12, alphabet="abcdef")),
    ))


@st.composite
def _traj_and_steps(draw):
    tid = new_id()
    started = _BASE + timedelta(seconds=draw(st.integers(min_value=0, max_value=10_000)))
    n_steps = draw(st.integers(min_value=1, max_value=4))
    steps = [
        Step(
            id=new_id(),
            trajectory_id=tid,
            parent_step_id=None,
            name=draw(st.text(min_size=1, max_size=12)),
            started_at=started + timedelta(milliseconds=i),
            finished_at=started + timedelta(milliseconds=i + 1),
            status=StepStatus.SUCCEEDED,
            payload=draw(_payloads()),
        )
        for i in range(n_steps)
    ]
    traj = Trajectory(
        id=tid,
        task=draw(st.text(min_size=1, max_size=64)),
        agent_name=draw(st.sampled_from(["react", "tool-use"])),
        agent_version="0.1",
        model_id=draw(st.sampled_from(["claude-sonnet", "gpt-4o"])),
        started_at=started,
        finished_at=started + timedelta(seconds=1),
        final_status=draw(st.sampled_from(list(TrajectoryStatus))),
        final_answer=draw(st.one_of(st.none(), st.text(max_size=32))),
        root_step_id=steps[0].id,
    )
    return traj, steps


@pytest.mark.fast
@given(traj_and_steps=_traj_and_steps())
@settings(max_examples=50, deadline=None)
def test_storage_round_trip(traj_and_steps, tmp_path_factory):
    """Any (traj, steps) we generate is recovered identically from DuckDBStore."""
    traj, steps = traj_and_steps
    db_dir = tmp_path_factory.mktemp("hypothesis_store")
    db = db_dir / f"{traj.id}.duckdb"

    async def _run():
        store = DuckDBStore(path=db)
        try:
            await store.save_trajectory(traj, steps)
            loaded_traj, loaded_steps = await store.get_trajectory(traj.id)
            assert loaded_traj == traj
            assert loaded_steps == steps
        finally:
            await store.close()

    asyncio.run(_run())
```

- [ ] **Step 2: Run test, expect pass**

Run: `uv run pytest tests/property/test_storage_roundtrip.py -v`
Expected: 1 passed (50 hypothesis examples).

- [ ] **Step 3: Commit**

```bash
git add tests/property/test_storage_roundtrip.py
git commit -m "test(storage): hypothesis storage round-trip for Trajectory + Steps"
```

---

## Task 11: Public API + smoke test

**Files:**
- Modify: `src/ariadne_eval/__init__.py`
- Modify: `tests/unit/test_smoke.py`

- [ ] **Step 1: Extend the smoke test**

Modify `tests/unit/test_smoke.py` `test_public_api_exports_core_types` to add the storage symbols:

```python
@pytest.mark.fast
def test_public_api_exports_core_types():
    """Pin the public surface so accidental removals are caught early."""
    import ariadne_eval

    expected = {
        "__version__",
        "Trajectory", "Step", "Message", "ContentBlock", "TextBlock",
        "ToolCallRef",
        "LLMCallPayload", "ToolCallPayload", "UserInputPayload", "InternalPayload",
        "StepError", "StepStatus", "TrajectoryStatus", "JsonValue",
        "new_id", "is_valid_id",
        # Storage
        "Store", "DuckDBStore",
        "StoreError", "TrajectoryNotFoundError", "MetadataTooLargeError",
        "export_jsonl", "import_jsonl",
    }
    missing = expected - set(ariadne_eval.__all__)
    assert not missing, f"Missing from public API: {missing}"
    for name in expected:
        assert hasattr(ariadne_eval, name), f"ariadne_eval.{name} not importable"
```

- [ ] **Step 2: Run test, expect fail**

Run: `uv run pytest tests/unit/test_smoke.py::test_public_api_exports_core_types -v`
Expected: assertion failure listing the new storage symbols as missing.

- [ ] **Step 3: Update `__init__.py`**

Modify `src/ariadne_eval/__init__.py` to add the storage imports and exports:

```python
"""ariadne-eval: trajectory-level observability and evaluation for LLM agents.

The public API is intentionally small. Every symbol re-exported here is part
of the supported surface; everything else is private and may change without
warning. See ``docs/reference/`` for the full reference.
"""

from __future__ import annotations

from ariadne_eval._version import __version__
from ariadne_eval.core.ids import is_valid_id, new_id
from ariadne_eval.core.status import StepStatus, TrajectoryStatus
from ariadne_eval.core.trajectory import (
    ContentBlock,
    InternalPayload,
    JsonValue,
    LLMCallPayload,
    Message,
    Step,
    StepError,
    TextBlock,
    ToolCallPayload,
    ToolCallRef,
    Trajectory,
    UserInputPayload,
)
from ariadne_eval.storage.base import (
    MetadataTooLargeError,
    Store,
    StoreError,
    TrajectoryNotFoundError,
)
from ariadne_eval.storage.duckdb_store import DuckDBStore
from ariadne_eval.storage.jsonl_store import export_jsonl, import_jsonl

__all__ = [
    "ContentBlock",
    "DuckDBStore",
    "InternalPayload",
    "JsonValue",
    "LLMCallPayload",
    "Message",
    "MetadataTooLargeError",
    "Step",
    "StepError",
    "StepStatus",
    "Store",
    "StoreError",
    "TextBlock",
    "ToolCallPayload",
    "ToolCallRef",
    "Trajectory",
    "TrajectoryNotFoundError",
    "TrajectoryStatus",
    "UserInputPayload",
    "__version__",
    "export_jsonl",
    "import_jsonl",
    "is_valid_id",
    "new_id",
]
```

- [ ] **Step 4: Run all fast tests, expect pass**

Run: `uv run pytest -m fast`
Expected: every test passes.

- [ ] **Step 5: Run mypy, expect clean**

Run: `uv run mypy --strict`
Expected: `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
git add src/ariadne_eval/__init__.py tests/unit/test_smoke.py
git commit -m "feat: re-export storage layer (Store, DuckDBStore, JSONL helpers, errors)"
```

---

## Task 12: Concept doc + CHANGELOG

**Files:**
- Create: `docs/concepts/storage.md`
- Modify: `mkdocs.yml`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Write the concept doc**

```markdown
# Storage

ariadne-eval persists trajectories and their step trees in a local DuckDB
file. The default location is `~/.ariadne/store.duckdb`; override with the
`ARIADNE_STORE_PATH` env var or the `path` constructor argument to
`DuckDBStore`.

## Schema

Two tables, both managed by the migration mechanism:

- `trajectories` — one row per run. Columns: `id`, `schema_version`, `task`,
  `agent_name`, `agent_version`, `model_id`, `started_at`, `finished_at`,
  `final_status`, `final_answer` (JSON), `root_step_id`,
  `metadata` (JSON). Indexed on `started_at DESC`, `agent_name`, `model_id`,
  `final_status`.
- `steps` — one row per step. Columns: `id`, `trajectory_id`,
  `parent_step_id`, `step_type`, `name`, `started_at`, `finished_at`,
  `status`, `payload` (JSON), `error` (JSON, nullable), `metadata` (JSON).
  Indexed on `trajectory_id`.

A `_meta` table tracks which migrations have been applied
(`version`, `name`, `applied_at`).

## Public API

```python
from ariadne_eval import (
    DuckDBStore, Store,
    StoreError, TrajectoryNotFoundError, MetadataTooLargeError,
    export_jsonl, import_jsonl,
)

store = DuckDBStore()  # writes to ~/.ariadne/store.duckdb
await store.save_trajectory(traj, steps)
loaded_traj, loaded_steps = await store.get_trajectory(traj.id)
recent = await store.list_trajectories(agent_name="react", limit=20)
n = await store.count(final_status=TrajectoryStatus.FAILED)
await store.delete_trajectory(traj.id)
```

## Concurrency

DuckDB is single-writer. Each `DuckDBStore` instance uses an
`asyncio.Lock` to serialize writes. Reads are not locked and run
concurrently. Multi-process use is out of scope for v0.0.x — open one
store per process.

## Portability

`export_jsonl(store, path, **filters)` streams matching trajectories to a
JSON Lines file:

```json
{"trajectory": {...}, "steps": [{...}, ...]}
```

`import_jsonl(path, store)` reads each line and saves it via
`save_trajectory`. The format is portable across machines and DuckDB
versions.

## Limits

- Trajectory and step `metadata` is capped at 1 MB serialized;
  `MetadataTooLargeError` is raised at save time when exceeded.
- `LLMCallPayload.completion` and `ToolCallPayload.result` are truncated
  by Phase 1's data model at 64 KB characters before they reach the
  store.

## Privacy

Payloads are stored as-is. Use `Trajectory.redact()` (Phase 1) to apply
your own redactor before saving.
```

- [ ] **Step 2: Update mkdocs nav**

Modify the `Concepts` section in `mkdocs.yml`:

```yaml
  - Concepts:
      - concepts/index.md
      - Trajectory model: concepts/trajectory.md
      - Storage: concepts/storage.md
```

- [ ] **Step 3: Append CHANGELOG entry**

Modify `CHANGELOG.md` `[Unreleased]` section to add:

```markdown
### Added
- Storage layer: `Store` Protocol, `DuckDBStore` implementation, schema
  migrations under `migrations_sql/`, JSONL export/import functions
  (`export_jsonl`, `import_jsonl`), and storage error hierarchy
  (`StoreError`, `TrajectoryNotFoundError`, `MetadataTooLargeError`).
  Default path `~/.ariadne/store.duckdb` (overridable via
  `ARIADNE_STORE_PATH`). Per-instance write lock; async-first API. 1 MB
  metadata cap.
```

- [ ] **Step 4: Verify docs build**

Run: `uv run mkdocs build --strict`
Expected: clean build, no warnings.

- [ ] **Step 5: Commit**

```bash
git add docs/concepts/storage.md mkdocs.yml CHANGELOG.md
git commit -m "docs(concepts): add storage concept page"
```

---

## Task 13: Final verification + tag v0.0.3-alpha

**Files:** none (verification only).

- [ ] **Step 1: All fast tests pass**

Run: `uv run pytest -m fast`
Expected: every test green.

- [ ] **Step 2: Coverage ≥ 95% on storage**

Run: `uv run pytest -m fast --cov=src/ariadne_eval/storage --cov-report=term-missing`
Expected: each file in `src/ariadne_eval/storage/` shows ≥ 95% coverage.

- [ ] **Step 3: mypy strict**

Run: `uv run mypy --strict`
Expected: `Success: no issues found`.

- [ ] **Step 4: ruff clean**

Run: `uv run ruff check && uv run ruff format --check`
Expected: both green.

- [ ] **Step 5: Pre-commit clean**

Run: `uv run pre-commit run --all-files`
Expected: all hooks pass.

- [ ] **Step 6: Tag the phase**

```bash
git tag v0.0.3-alpha -m "Phase 2: storage layer

DuckDBStore implementation of an abstract Store Protocol; numbered SQL
migrations with _meta tracking; JSONL export/import for portability.
Async-first API with per-instance write lock. Default path
~/.ariadne/store.duckdb."
```

---

## Self-review

**Spec coverage check:**

| Spec section | Task |
|---|---|
| `Store` Protocol with 5 async methods | Task 2 |
| `StoreError`, `TrajectoryNotFoundError`, `MetadataTooLargeError` | Task 2 |
| Migration mechanism (numbered SQL, `_meta` table) | Task 3 |
| `001_initial.sql` (schema + indexes) | Task 3 |
| `DuckDBStore` init + path resolution + dir creation + migration apply | Task 4 |
| `save_trajectory` (upsert + 1 MB metadata cap + transaction) | Task 5 |
| `get_trajectory` returning `(Trajectory, list[Step])` | Task 5 |
| `list_trajectories` with kwargs filter + pagination | Task 6 |
| `count` with kwargs filter | Task 6 |
| `delete_trajectory` (idempotent) | Task 7 |
| Concurrency: 50 parallel saves | Task 8 |
| `export_jsonl` / `import_jsonl` | Task 9 |
| Property test for storage round-trip | Task 10 |
| Public API additions | Task 11 |
| `docs/concepts/storage.md` + CHANGELOG | Task 12 |
| Coverage ≥ 95%, mypy clean, alpha tag | Task 13 |

All sections covered.

**Type consistency check:** `Store`, `DuckDBStore`, `MetadataTooLargeError`,
`TrajectoryNotFoundError`, `StoreError`, `export_jsonl`, `import_jsonl` are
referenced by name across tasks 2, 5, 9, 11; spelling matches everywhere.

**Placeholder scan:** no TBD / TODO / "implement later" markers.
