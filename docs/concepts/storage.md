# Storage

ariadne-eval persists trajectories and their step trees in a local DuckDB
file. The default location is `~/.ariadne/store.duckdb`; override with the
`ARIADNE_STORE_PATH` env var or the `path` argument to `DuckDBStore`.

## Schema

Two tables, managed by the migration mechanism in `storage/migrations.py`:

- `trajectories` — one row per run. Columns: `id`, `schema_version`, `task`,
  `agent_name`, `agent_version`, `model_id`, `started_at`, `finished_at`,
  `final_status`, `final_answer` (JSON), `root_step_id`, `metadata` (JSON).
  Indexed on `started_at`, `agent_name`, `model_id`, `final_status`.
- `steps` — one row per step. Columns: `id`, `trajectory_id`, `parent_step_id`,
  `step_type`, `name`, `started_at`, `finished_at`, `status`, `payload` (JSON),
  `error` (JSON, nullable), `metadata` (JSON). Indexed on `trajectory_id`.

A `_meta` table tracks applied migrations (`version`, `name`, `applied_at`).

## Public API

```python
from ariadne_eval import (
    DuckDBStore, Store,
    StoreError, TrajectoryNotFoundError, MetadataTooLargeError,
    export_jsonl, import_jsonl,
)
from ariadne_eval import TrajectoryStatus

store = DuckDBStore()  # writes to ~/.ariadne/store.duckdb
await store.save_trajectory(traj, steps)
loaded_traj, loaded_steps = await store.get_trajectory(traj.id)
recent = await store.list_trajectories(agent_name="react", limit=20)
n_failed = await store.count(final_status=TrajectoryStatus.FAILED)
await store.delete_trajectory(traj.id)
await store.close()
```

## Concurrency

DuckDB is single-writer. Each `DuckDBStore` instance uses an `asyncio.Lock`
to serialize writes. Reads are not locked and run concurrently. Multi-process
use is out of scope for v0.0.x — open one store per process.

## Portability

`export_jsonl(store, path, **filters)` streams matching trajectories to a
JSON Lines file:

```json
{"trajectory": {...}, "steps": [{...}, ...]}
```

`import_jsonl(path, store)` reads each line and saves it via `save_trajectory`.
The format is portable across machines and DuckDB versions.

## Limits

- Trajectory and step `metadata` is capped at 1 MB serialized;
  `MetadataTooLargeError` is raised at save time when exceeded.
- `LLMCallPayload.completion` and `ToolCallPayload.result` are truncated by
  the Phase 1 data model at 64 K characters before they reach the store.

## Privacy

Payloads are stored as-is. Use `Trajectory.redact()` (Phase 1) to apply
your own redactor before saving.
