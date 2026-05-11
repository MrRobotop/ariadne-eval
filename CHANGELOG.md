# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Tracing instrumentation: `start_trajectory` async context manager,
  `@trace_step` decorator (sync + async via `inspect.iscoroutinefunction`),
  `record_llm_call` / `record_tool_call` recorders, `Sampler` Protocol
  with `AlwaysSampler` / `RateSampler` / `TaskFilterSampler`,
  `enable_litellm_autotrace` for LiteLLM integration. Per-trajectory
  sampling produces full no-ops for unsampled trajectories. Fail mode
  policy (`ARIADNE_FAIL_MODE`: strict / warn / silent) governs unattached
  recordings via a real runtime raise (not `assert`). ContextVar-based
  parent attachment works correctly under `asyncio.gather` and
  `TaskGroup`. Persistence model: build in memory, save once on context
  exit (async queue + background drainer deferred to a later phase).
- `examples/01_quickstart/` runnable ReAct-style traced example.
- `benchmarks/overhead.py` measures `@trace_step` overhead against both a
  no-op micro-benchmark and a realistic I/O-bound loop (~3% overhead on
  the realistic benchmark; ~5 μs absolute add per step).

- Storage layer: `Store` Protocol, `DuckDBStore` implementation, filesystem
  migrations under `storage/migrations_sql/`, JSONL export/import functions
  (`export_jsonl`, `import_jsonl`), and storage error hierarchy
  (`StoreError`, `TrajectoryNotFoundError`, `MetadataTooLargeError`).
  Default path `~/.ariadne/store.duckdb`, overridable via
  `ARIADNE_STORE_PATH` env var or the constructor `path` argument.
  Per-instance asyncio write lock; async-first API with sync DuckDB calls
  wrapped in `asyncio.to_thread`. 1 MB metadata cap. Hypothesis storage
  round-trip property test (50 examples).
- `pytz` added as a runtime dependency for DuckDB `TIMESTAMPTZ` round-trips.

- Core trajectory data model: `Trajectory`, `Step`, `Message`, four payload
  variants (`LLMCallPayload`, `ToolCallPayload`, `UserInputPayload`,
  `InternalPayload`), `StepError`, `StepStatus`, `TrajectoryStatus`,
  `JsonValue`, `new_id`, `is_valid_id`. Validators: tz-aware datetimes,
  ULID format, no self-parenting, failed-step requires error. Truncation on
  `completion` and `result` above 64K chars. Opt-in `Trajectory.redact()`
  hook. Hypothesis round-trip property tests (200 examples each).

## [0.0.1] - 2026-05-10

### Added
- Bootstrap: project scaffold, Apache-2.0 license, ruff/mypy/pytest configuration,
  pre-commit hooks, README skeleton, mkdocs site skeleton, no-op CLI entrypoint
  (`ariadne --version`), and a smoke test asserting the package imports and the
  CLI is registered.

[Unreleased]: https://github.com/MrRobotop/ariadne-eval/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/MrRobotop/ariadne-eval/releases/tag/v0.0.1
