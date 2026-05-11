# Phase 2 Design: Storage Layer

**Status:** Approved (2026-05-10) · **Phase:** 2 · **Target version:** 0.0.3

## Goal

A DuckDB-backed persistent store for trajectories and their step trees, with a
clean abstract `Store` Protocol, schema migrations, and a portable JSONL
export/import path. This is the substrate every later phase (tracing, the UI,
benchmark runner, drift detection) reads and writes.

## Scope

In scope: `Store` protocol, `DuckDBStore` implementation, migration mechanism,
JSONL export/import functions, error hierarchy, public API additions.

Out of scope: in-memory stores, alternative backends (Postgres, sqlite),
streaming/observable interfaces, soft-delete. Add when an actual consumer
needs them.

## Architecture

Five files under `src/ariadne_eval/storage/`:

| File | Responsibility |
|---|---|
| `base.py` | `Store` Protocol; `StoreError`, `TrajectoryNotFoundError`, `MetadataTooLargeError`. |
| `duckdb_store.py` | `DuckDBStore`: one `duckdb.Connection` per instance, asyncio.Lock for writes, `asyncio.to_thread` for sync ops. |
| `migrations.py` | Migration discovery (filesystem), ordering (numeric prefix), application (transactional), `_meta` bookkeeping. |
| `migrations_sql/001_initial.sql` | First migration: trajectories, steps, _meta tables and indexes. |
| `jsonl_store.py` | `export_jsonl(store, path, **filter)` and `import_jsonl(path, store)` functions. |

DuckDB's Python API is synchronous. Each public coroutine wraps its blocking
DB call in `asyncio.to_thread()`. A single `asyncio.Lock` per store instance
serializes writes (DuckDB is single-writer); reads run concurrently.

## Store protocol

```python
# storage/base.py
from typing import Protocol


class Store(Protocol):
    async def save_trajectory(
        self, traj: Trajectory, steps: list[Step]
    ) -> None: ...

    async def get_trajectory(
        self, traj_id: str
    ) -> tuple[Trajectory, list[Step]]: ...

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
    ) -> list[Trajectory]: ...

    async def delete_trajectory(self, traj_id: str) -> None: ...

    async def count(
        self,
        *,
        agent_name: str | None = None,
        model_id: str | None = None,
        final_status: TrajectoryStatus | None = None,
    ) -> int: ...


class StoreError(Exception):
    """Base class for storage-layer errors."""


class TrajectoryNotFoundError(StoreError):
    """Raised when ``get_trajectory`` is called with an id that does not exist."""


class MetadataTooLargeError(StoreError):
    """Raised when a trajectory or step's metadata serializes to more than 1 MB."""
```

## DuckDB schema (migration 001)

```sql
CREATE TABLE _meta (
    version INTEGER PRIMARY KEY,
    name VARCHAR NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL
);

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

DuckDB's native `JSON` column type stores serialized payload / metadata. We
do not need JSON-path queries in v0.0.3, but the type leaves room for
`metadata->>'$.user_email'` filters in a future minor release.

**`schema_version` is two different things.** `trajectories.schema_version`
is the *data-model* version owned by `Trajectory` (Phase 1) — bumped when the
Pydantic schema changes. `_meta.version` is the *database* version — bumped
when the DuckDB tables / indexes change. Both start at 1; they evolve
independently.

## Migration mechanism

`storage/migrations_sql/` contains numbered SQL files: `001_initial.sql`,
`002_<name>.sql`, …

On `DuckDBStore.__init__`:

1. Resolve the database file path. Precedence:
   constructor `path=` argument → `ARIADNE_STORE_PATH` env var →
   `~/.ariadne/store.duckdb`. Parent directory is created with
   `Path.mkdir(parents=True, exist_ok=True)`.
2. Open the DuckDB connection.
3. Create `_meta` table if absent (idempotent).
4. List `*.sql` files in `migrations_sql/`, parse the leading three digits as
   the version.
5. `SELECT MAX(version) FROM _meta` → apply each higher-numbered migration
   in numerical order, each wrapped in a `BEGIN ... COMMIT` transaction.
   Insert a `_meta` row on success.
6. Migration failures abort the transaction and re-raise as
   `StoreError(f"migration {n} failed: ...")`.

The migrations module exposes `apply_pending(conn) -> int` that returns the
number of migrations applied. Init can call this synchronously before any
async traffic begins, since it runs before the lock is ever needed.

## Save / load flow

### `save_trajectory(traj, steps)`

1. Validate `len(json.dumps(traj.metadata)) <= 1_048_576`; same check on each
   step's metadata. Otherwise `MetadataTooLargeError`.
2. Serialize trajectory + each step via `model_dump_json()`.
3. Acquire `self._write_lock`.
4. In a transaction:
   - `INSERT OR REPLACE INTO trajectories (...) VALUES (...)`.
   - `DELETE FROM steps WHERE trajectory_id = ?`.
   - `INSERT INTO steps (...) VALUES (...), (...), ...` (batched parameter list).
5. Commit; release lock.

The whole sequence is two database round-trips: trajectory upsert and a
combined steps DELETE+INSERT inside one transaction.

### `get_trajectory(id)`

1. `SELECT * FROM trajectories WHERE id = ?`.
2. If empty: raise `TrajectoryNotFoundError(id)`.
3. `SELECT * FROM steps WHERE trajectory_id = ? ORDER BY started_at`.
4. Reconstruct `Trajectory` and each `Step` via `model_validate`.
5. Return `(traj, steps)`.

### `list_trajectories(...)`

Single SELECT. Build the `WHERE` clause from non-None kwargs:

```sql
WHERE
   (? IS NULL OR agent_name    = ?)
   AND (? IS NULL OR model_id     = ?)
   AND (? IS NULL OR final_status = ?)
   AND (? IS NULL OR started_at   >= ?)
   AND (? IS NULL OR started_at   <= ?)
ORDER BY started_at DESC
LIMIT ? OFFSET ?
```

Returns trajectory metadata only — no steps.

### `delete_trajectory(id)`

Single transaction: `DELETE FROM steps WHERE trajectory_id = ?`,
`DELETE FROM trajectories WHERE id = ?`. Idempotent (no-op on missing id).

### `count(...)`

Same `WHERE` builder as `list_trajectories`, with `SELECT COUNT(*)`.

## JSONL export / import

```python
# storage/jsonl_store.py
async def export_jsonl(
    store: Store,
    path: Path,
    *,
    agent_name: str | None = None,
    model_id: str | None = None,
    final_status: TrajectoryStatus | None = None,
    started_after: datetime | None = None,
    started_before: datetime | None = None,
    batch_size: int = 100,
) -> int: ...


async def import_jsonl(path: Path, store: Store) -> int: ...
```

`export_jsonl` calls `list_trajectories` in `batch_size` chunks (paginated
via offset), fetches each trajectory's steps via `get_trajectory`, and
writes one JSON object per line:

```json
{"trajectory": {...}, "steps": [{...}, {...}]}
```

Returns the number of trajectories written.

`import_jsonl` reads line by line, validates each as `(Trajectory, list[Step])`
via Pydantic, and calls `store.save_trajectory` for each. Returns the number
imported.

Streaming (no full-file buffering) so it works on large dumps.

## Edge cases handled at the storage boundary

1. **Saving same id twice.** `INSERT OR REPLACE` replaces the row;
   `DELETE FROM steps` clears children. Atomic via the transaction.
2. **Metadata over 1 MB.** Rejected at the entry point with
   `MetadataTooLargeError("metadata is N bytes, max 1048576")`.
3. **Concurrent writes.** Per-instance `asyncio.Lock`. 50 parallel
   `save_trajectory` calls land deterministically.
4. **`get_trajectory` on missing id.** `TrajectoryNotFoundError(id)`.
5. **Missing parent directory** for the DuckDB file. Auto-created.
6. **Schema mismatch on read.** Pydantic's `model_validate` raises; we let it
   propagate. Future migrations preserve old-row readability.

## Testing strategy (TDD)

| File | Coverage |
|---|---|
| `tests/unit/storage/test_duckdb_store.py` | Each Store method round-trips; filters work (each kwarg + combinations); upsert replaces; not-found raises; metadata-too-large raises; `asyncio.gather(50× save)` succeeds; bulk insert under 50 ms for a 100-step trajectory. |
| `tests/unit/storage/test_migrations.py` | Empty file → latest schema; partial state (only v1 applied) → v2 applied on next init; `_meta` rows are populated correctly; failed migration aborts transaction. |
| `tests/unit/storage/test_jsonl_store.py` | export → import round-trip preserves full equality; filter kwargs honoured on export; corrupt line raises with line number. |
| `tests/property/test_storage_roundtrip.py` | Hypothesis: any (Trajectory, list[Step]) we generate round-trips through DuckDBStore with full equality. |

All tests use `tmp_path` fixtures for ephemeral DuckDB files. No tests touch
`~/.ariadne/` directly.

**Coverage target:** ≥95 % on `src/ariadne_eval/storage/`.

**Type strictness:** `mypy --strict` clean. Async signatures and DuckDB row
unpacking are explicitly typed.

## Public API additions

```python
__all__ += [
    "Store",
    "DuckDBStore",
    "StoreError",
    "TrajectoryNotFoundError",
    "MetadataTooLargeError",
    "export_jsonl",
    "import_jsonl",
]
```

## Documentation

- `docs/concepts/storage.md` — narrative covering: where data lives, schema
  shape (with the SQL above), migration model, JSONL portability, performance
  expectations, and the privacy posture (payloads stored as-is unless redacted
  by the user — Phase 1's hook).
- API reference auto-generated by `mkdocstrings`.
- `CHANGELOG.md [Unreleased]` entry.

## Out of scope

- In-memory test stores. `tmp_path` DuckDB files are fast enough.
- Soft delete / tombstones.
- Streaming change feeds.
- Multi-process write coordination (DuckDB single-writer is enforced by file
  lock; multi-process is YAGNI for v0.0.3).
- `BlobStore`-style large-payload offload. The 64K truncation in Phase 1 plus
  1 MB metadata cap keeps row sizes bounded.
