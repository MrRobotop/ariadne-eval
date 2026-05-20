# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ariadne_eval.eval` namespace: `Case`, `ExpectedTool`, `Metric`,
  `MetricResult`, `FinalAnswerMatch`, `ToolAccuracy`, `StepEfficiency`,
  `Runner`, `EvalReport`, `bootstrap_mean_ci`, `BootstrapCI`,
  `MissingReferenceError`, `BootstrapInsufficientDataWarning`. All
  re-exported from the top-level `ariadne_eval`.
- `docs/concepts/metrics.md` and `docs/reference/eval.md`.
- `examples/03_custom_metric/` walkthrough.

### Known issues

- `EvalReport.to_jsonl` serializes header floats with Python's `json`
  defaults, so an aggregate produced from `n=0` (every case skipped for a
  metric) emits `mean`/`lo`/`hi` as the bare token `NaN` — accepted by
  Python's `json.loads` but rejected by RFC 8259 consumers (`jq`, browser
  `JSON.parse`, `serde_json`). Round-trip via `EvalReport.from_jsonl`
  still works. A coordinated `null↔NaN` serialization rule will land in
  Phase 5.1 before any 0.1.0 promotion.
- Reference ReAct agent (`ariadne_eval.examples.react_agent.ReactAgent`)
  with text-parsed ReAct loop, two stub tools (`calculator` via
  AST-whitelisted arithmetic, `search` via dict lookup), and
  `StepLimitExhausted` / `ReactParseError` errors. Used by
  `examples/01_quickstart/` and the new end-to-end integration test.
- End-to-end integration test via a hand-crafted VCR cassette
  (`tests/integration/test_react_end_to_end.py`) with
  `record_mode="none"` so CI never makes real HTTP calls. Auth headers
  redacted via the `vcr_config` fixture. `LITELLM_LOCAL_MODEL_COST_MAP`
  set so litellm does not phone home for its price list.
- mkdocs `pymdownx.snippets` extension wired up so docs can include
  files verbatim via `--8<--` syntax — the quickstart docs page now
  pulls `examples/01_quickstart/main.py` directly, so the example and
  the docs cannot drift.

### Changed
- LiteLLM autotrace adapter now registers on `litellm.callbacks` (the
  unified sync+async registry) in addition to the legacy
  `success_callback` / `failure_callback`. Falls back gracefully when
  `callbacks` is absent on older litellm versions.
- LiteLLM autotrace `_on_success` / `_on_failure` accept either float
  (Unix timestamp) or `datetime.datetime` for `start_time` / `end_time`.
  A new `_latency_ms` helper converts the difference uniformly to a
  float in milliseconds.

### Added (Phase 3, before merge)
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
