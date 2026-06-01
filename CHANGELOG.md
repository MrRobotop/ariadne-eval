# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Phase 7 ships the tau-bench benchmark stack as library code: a
  tau-agnostic `Benchmark` Protocol, `BenchmarkTask`,
  `BenchmarkRunResult`, `BenchmarkConfig` (YAML), `BenchmarkRunner`,
  `BenchmarkReport`. New optional extra `[tau-bench]` pins Sierra's
  τ-bench at commit `59a200c6d575d595120f1cb70fea53cef0632f6b`. New
  CLI subcommand `ariadne bench run` with `--dry-run` / `--limit` /
  `--models` / `--resume`.
- `TauBenchAdapter` (gated behind the `[tau-bench]` extra) wraps
  Sierra's τ-bench retail/airline domains. Converts
  τ-bench's `EnvRunResult` / `SolveResult` shape to the ariadne
  `Trajectory` schema via `_convert_tau_traj` so downstream metrics
  (`StepEfficiency`, `PlanQuality`) and the judge run unchanged over
  benchmark-sourced trajectories.
- `_transient` retry primitives extracted from Phase 6.1's
  calibration script into `src/ariadne_eval/_transient.py`; both the
  calibration script and the new benchmark runner import from it.
- `BenchmarkRunner` records provider 4xx errors per-cell instead of
  killing the run, retries transient provider errors with bounded
  exponential backoff, and writes a result bundle with sorted-keys +
  `allow_nan=False` JSON (RFC-8259 valid). The κ = 0.32 (fair) judge
  calibration note travels in `summary.json` next to every
  `plan_quality` aggregate.
- `configs/benchmarks/tau_retail_baseline.yaml` is the canonical run
  config (2 agent models × 50 retail tasks, Sonnet judge, Groq
  user-simulator).
- `docs/concepts/benchmarks.md` documents the benchmark stack, the
  bundle layout, and the API-tier constraint that defers the headline
  numbers to Phase 7.1.
- Version bumped to `0.0.9-alpha`.

### Deferred

- The canonical τ-retail headline bundle
  (`docs/benchmarks/v0.0.9-alpha-tau-retail-50/`) is deferred to
  Phase 7.1 — the maintainer's current API tiers don't support
  τ-bench's user simulator at 50-task scale (Anthropic Tier 1 caps a
  single input request at 50,000 tokens; Groq free tier caps daily
  tokens-per-model at 100,000). The library is feature-complete and
  fully tested; only the bundle awaits API access.

### Fixed

- `EvalReport.to_jsonl` now serializes non-finite `BootstrapCI` floats as
  `null` (RFC-8259 valid) and `from_jsonl` rehydrates them back to `NaN`.
  Absorbs the Phase-5.1 follow-up.

### Added

- Phase 6.1: judge symbols (`Judge`, `JudgeParseError`, `JudgeVerdict`,
  `PlanQuality`, `StubJudge`, `TrajectoryJudge`) are now top-level
  public (`from ariadne_eval import TrajectoryJudge`).
- 51-fixture synthetic plan-quality gold set
  (`tests/data/gold_plans.jsonl`, balanced 17/17/17 across `pass`/
  `partial`/`fail`).
- Calibration evidence: `TrajectoryJudge` achieves κ = 0.32 (fair)
  against the maintainer on the gold set. Report committed at
  `docs/calibration/v0.0.8-alpha-report.jsonl`; human-readable page at
  `docs/concepts/calibration.md`.
- `scripts/build_calibration_set.py`: `--source synth|store` flag;
  `click.UsageError` UX on missing `--store`; bounded
  exponential-backoff retry on transient provider errors; `_kind:
  "confusion"` and `_kind: "meta"` trailing JSONL lines (with
  prompt-hash digests for drift detection).
- `scripts/render_calibration_md.py`: renders the JSONL report into a
  human-readable docs page; golden-file tested.
- Version bumped to `0.0.8-alpha` (`__version__`, `pyproject.toml`,
  smoke tests).
- `AsyncMetric` Protocol and `Runner.aevaluate` with bounded concurrency
  (`asyncio.Semaphore`, default 4). Sync `Runner.evaluate` now raises a
  clear `RuntimeError` directing users to `aevaluate` for async-only
  metrics.
- `ariadne_eval.eval.judges` namespace: `Judge` Protocol, `JudgeVerdict`,
  `JudgeParseError`, `TrajectoryJudge` (litellm-backed, injectable
  client), `StubJudge` for tests, and `parse_plan_quality_verdict`
  + prompt constants. NOT re-exported from top-level `ariadne_eval` —
  pending calibration data in Phase 6.1 per Hard Rule #5.
- `PlanQuality` async metric (importable from `ariadne_eval.eval` and
  `ariadne_eval.eval.metrics.plan_quality`).
- Top-level public: `AsyncMetric`, `cohens_kappa`, `KappaResult`,
  `KappaInsufficientDataWarning`.
- `cohens_kappa` with Landis-Koch interpretation bands in
  `ariadne_eval.eval.stats.agreement`.
- `scripts/build_calibration_set.py` CLI: takes a DuckDB store and a
  gold-labels JSONL, runs the judge, writes a per-trajectory report
  with a kappa summary line.
- `docs/concepts/judges.md`, `docs/reference/judges.md`.
- `examples/04_plan_quality/` async runner walkthrough using `StubJudge`.
- One end-to-end VCR cassette integration test for `TrajectoryJudge`.
- `ariadne_eval.eval` namespace: `Case`, `ExpectedTool`, `Metric`,
  `MetricResult`, `FinalAnswerMatch`, `ToolAccuracy`, `StepEfficiency`,
  `Runner`, `EvalReport`, `bootstrap_mean_ci`, `BootstrapCI`,
  `MissingReferenceError`, `BootstrapInsufficientDataWarning`. All
  re-exported from the top-level `ariadne_eval`.
- `docs/concepts/metrics.md` and `docs/reference/eval.md`.
- `examples/03_custom_metric/` walkthrough.
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
