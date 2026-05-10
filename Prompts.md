# Prompts.md

Sequential prompts to build `ariadne-eval` from empty directory to PyPI-published, GitHub-starred open-source library. Each prompt is self-contained and can be pasted directly into Claude Code. **Read `prompts-readme.md` first** for setup, ordering rules, and recovery from failures.

Each phase follows the Superpowers cycle: **brainstorm → plan → execute → verify**. Don't skip the brainstorm on substantive phases — that's where design errors get caught cheaply.

---

## Phase 0 — Repository Bootstrap

**Goal:** Project scaffold, tooling, license, README skeleton, smoke-tested.

**Skills:** `writing-plans`, `verification-before-completion`, `superpowers` or any other skill you think is good for this project

**Prompt:**

```
We're building `ariadne-eval` — an open-source Python library for trajectory-level observability and evaluation of LLM agents. The CLAUDE.md in the repo root has the full context; read it before starting.

Goal for this session: bootstrap the repository to be PyPI-ready from day one.

Use the writing-plans skill to produce a task list that includes:

1. Initialize git repo and Python project with `uv init --lib`. The project name is `ariadne-eval`, the import name is `ariadne_eval`, the version is `0.0.1`. Python >=3.11.

2. Configure pyproject.toml:
   - Project metadata: name, version, description, authors, license = Apache-2.0, readme = README.md, classifiers (Development Status :: 3 - Alpha, License :: OSI Approved :: Apache Software License, Programming Language :: Python :: 3.11/3.12/3.13, Topic :: Software Development :: Libraries).
   - URLs: Homepage, Documentation, Repository, Issues, Changelog (placeholder GitHub URLs the user will fill in).
   - Optional dependencies (`[project.optional-dependencies]`): `langgraph`, `crewai`, `openai-assistants`, `dev`, `docs`, `all`.
   - Console script: `ariadne = "ariadne_eval.cli.main:cli"`.

3. Create directory layout from CLAUDE.md (with __init__.py and a `_version.py`).

4. Add LICENSE (Apache 2.0 full text), CONTRIBUTING.md (basic), CODE_OF_CONDUCT.md (Contributor Covenant 2.1), CHANGELOG.md (Keep a Changelog format with v0.0.1 = "Bootstrap").

5. Add dev dependencies to dev extra:
   pytest, pytest-asyncio, pytest-cov, pytest-recording, hypothesis, ruff, mypy, pre-commit, types-requests, mkdocs-material, mkdocstrings[python].

6. Add core runtime dependencies:
   pydantic>=2.0, duckdb, click, rich, anyio, python-ulid, litellm, scipy, numpy, statsmodels.

7. Configure ruff:
   - line-length = 100, target = py311
   - select = ["E","F","I","N","B","UP","S","RUF","SIM","ANN","D"]
   - per-file-ignores: tests get ANN, S, D off.
   - pydocstyle convention = google.

8. Configure mypy:
   - strict = true on src/, ignore_missing_imports for optional integrations only.

9. Configure pytest:
   - markers: fast, integration, slow.
   - default selection: `-m "fast and not integration"`.
   - asyncio_mode = "auto".

10. Configure pre-commit (.pre-commit-config.yaml):
    - ruff format, ruff check
    - mypy on src/
    - end-of-file-fixer, trailing-whitespace, check-yaml.

11. Create .env.example (ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY) and .gitignore (.env, .venv/, dist/, *.egg-info, .mypy_cache, .pytest_cache, .ruff_cache, .duckdb files in tests/, results/, htmlcov/, site/).

12. Create README.md skeleton with:
    - One-line description.
    - Status badge placeholders (CI, PyPI, license, python versions).
    - "Why ariadne-eval?" paragraph (3 sentences, from CLAUDE.md).
    - Install: `pip install ariadne-eval`.
    - Placeholder Quickstart code block (will be filled in later phases).
    - Placeholder for screenshots and benchmark results.
    - Links to docs, contributing, license.

13. Create docs/ skeleton with mkdocs.yml (Material theme, mkdocstrings configured for src/ariadne_eval) and `docs/index.md` mirroring the README intro.

14. Add smoke test `tests/unit/test_smoke.py`:
    - imports `ariadne_eval`
    - asserts `ariadne_eval.__version__ == "0.0.1"`
    - asserts the CLI entrypoint is registered (subprocess `ariadne --version` succeeds).

15. Set `__version__` in `src/ariadne_eval/_version.py` and re-export from `__init__.py`.

16. Stub `src/ariadne_eval/cli/main.py` with a click group that has a `--version` option and one no-op `hello` subcommand.

17. Verify: `uv sync --all-extras`, `pre-commit run --all-files`, `pytest -m fast`, `mypy src/` all pass.

After the plan is approved, execute it. Do not skip the verification step.
```

**Acceptance:**
- `uv sync --all-extras` succeeds.
- `pre-commit run --all-files` is clean.
- `pytest -m fast` reports 1 passed.
- `mypy --strict src/` is clean.
- `ariadne --version` prints `0.0.1`.
- First commit on `main`: `chore: bootstrap project scaffold`.

---

## Phase 1 — Core Trajectory Data Model

**Goal:** Pydantic models for the trajectory graph: `Trajectory`, `Step`, `LLMCall`, `ToolCall`, `StepStatus`. ULID-based IDs. JSON-serializable, JSON-deserializable, round-trip tested.

**Skills:** `brainstorming`, `writing-plans`, `test-driven-development`.

**Prompt:**

```
Create a feature branch with `using-git-worktrees`, then start with `/superpowers:brainstorm`.

Brainstorm the trajectory data model. The shape of this matters more than any other design decision in the project — every downstream component reads it.

Discussion points:
- A `Trajectory` is a single end-to-end agent run on one task. It has: id (ULID), task description, root_step_id, started_at, finished_at, final_answer, final_status, agent_name, agent_version, model_id, metadata (free-form dict).
- A `Step` is one node in the trajectory graph. It has: id (ULID), trajectory_id, parent_step_id (None for root), step_type ("llm_call" | "tool_call" | "user_input" | "internal"), name, started_at, finished_at, status (pending/running/succeeded/failed/skipped), payload (typed by step_type), error (optional), metadata.
- An `LLMCall` payload: model_id, prompt_messages, completion, input_tokens, output_tokens, cost_usd, temperature, latency_ms, tool_calls_emitted (list of references to child ToolCall step IDs).
- A `ToolCall` payload: tool_name, arguments (dict), result (any), result_truncated (bool), latency_ms.
- The graph is a tree, not a DAG, in v0.1. (Re-entry is rare and can be modeled with explicit "branch" steps.) Document this decision.
- IDs are ULIDs (sortable by time). Use python-ulid.
- Time is tz-aware UTC datetime, serialized as ISO 8601 with offset.
- Payload polymorphism: use Pydantic discriminated unions on `step_type`.

Edge cases to discuss:
- Long completions / large tool results: payload truncation policy (default 64KB per field, configurable). Truncation is recorded explicitly via a flag.
- PII / secrets: provide a `redact()` hook on the model, but do not auto-apply (user opt-in).
- Streaming LLM responses: capture the final assembled text plus the timestamp of first-token (TTFT) on LLMCall.
- Tool call that errors: status="failed", error populated with type and message; result is None.

After brainstorm, use `writing-plans` for TDD.

Required tests:
- Round-trip JSON serialization for each model with and without optional fields.
- Discriminated union correctly resolves to the right payload type on deserialization.
- ULIDs are valid and time-ordered.
- Property test (hypothesis): generate random trajectories, serialize, deserialize, equality holds.
- Truncation: a payload over the limit is truncated and `result_truncated` is True.
- Datetime tz-awareness: naive datetimes raise validation error.

Files:
- src/ariadne_eval/core/trajectory.py
- src/ariadne_eval/core/status.py
- src/ariadne_eval/core/ids.py
- tests/unit/core/test_trajectory.py
- tests/unit/core/test_ids.py
- tests/property/test_trajectory_roundtrip.py

Verify: `pytest -m fast tests/unit/core tests/property` green, `mypy --strict src/ariadne_eval/core` clean, coverage >95% on touched files.
```

**Acceptance:**
- All tests green, property tests run 200+ examples without failure.
- Round-trip equality demonstrated.
- `docs/concepts/trajectory.md` written with a diagram (ASCII art is fine for now).

**Common pitfalls:**
- Don't make the parent reference a Pydantic field that points to another `Step` object — that creates cycles in serialization. Use parent_step_id (a string) and reconstruct trees on read.
- Don't use `datetime.utcnow()` — it returns a naive datetime. Use `datetime.now(tz=timezone.utc)`.

---

## Phase 2 — Storage Layer

**Goal:** A DuckDB-backed store with a clean abstract interface, schema migrations, and a portable JSONL exporter.

**Skills:** `brainstorming`, `writing-plans`, `test-driven-development`.

**Prompt:**

```
Brainstorm:
- Define an abstract `Store` protocol with: `save_trajectory`, `get_trajectory`, `list_trajectories(filters, limit, offset)`, `delete_trajectory`, `count`. All async.
- Concrete `DuckDBStore`: writes to `~/.ariadne/store.duckdb` by default, configurable via env var or constructor.
- Schema:
  - `trajectories` table: id, task, agent_name, agent_version, model_id, started_at, finished_at, final_status, final_answer, metadata (JSON), schema_version.
  - `steps` table: id, trajectory_id, parent_step_id, step_type, name, started_at, finished_at, status, payload (JSON), error (JSON nullable), metadata (JSON).
  - Indexes: trajectory_id on steps, started_at desc on trajectories.
- Migrations: a `migrations/` directory with numbered SQL files. On store init, check `_meta` table for current version, apply pending migrations.
- JSONL exporter: dump trajectories+steps in a portable line-delimited format. Importer reads it back. This is the archival/portability path.
- Concurrency: DuckDB is single-writer. Use a write lock (asyncio.Lock) per-store-instance. Reads are concurrent.
- Bulk insert: use parametrized batch inserts; saving a trajectory with N steps should be O(2) database calls (one for trajectory, one batched for steps).

Edge cases:
- Saving a trajectory that already exists with the same id: upsert, replace.
- Filtering by time range, agent_name, final_status, model_id — all should be index-usable.
- Large metadata blobs (>1MB): rejected with a clear error.

Tests:
- All Store protocol methods round-trip via the DuckDB implementation.
- Migrations: starting from an empty file produces the latest schema; starting from v1 schema migrates correctly to v2 (mock a v1 by writing the v1 migration only first, then add v2).
- JSONL export → import preserves semantic equivalence (trajectory + steps).
- Concurrent saves are correctly serialized (use asyncio.gather with 50 saves; assert all present).
- Filter queries return correct results.
- Property test: any trajectory we generate can be saved and loaded with full equality.

Files:
- src/ariadne_eval/storage/base.py
- src/ariadne_eval/storage/duckdb_store.py
- src/ariadne_eval/storage/jsonl_store.py
- src/ariadne_eval/storage/migrations.py
- src/ariadne_eval/storage/migrations_sql/001_initial.sql
- tests/unit/storage/test_duckdb_store.py
- tests/unit/storage/test_jsonl_store.py
- tests/unit/storage/test_migrations.py
- tests/property/test_storage_roundtrip.py

Verify: all tests pass, mypy strict clean, the duckdb file is created at the expected path on init.
```

**Acceptance:**
- Round-trip persistence verified.
- Migrations work both forward and from-empty.
- Concurrency test passes (50 parallel saves).
- Bulk insert benchmarked: a trajectory with 100 steps saves in <50ms.

---

## Phase 3 — Tracing Instrumentation

**Goal:** The `@trace` decorator and context manager API. The headline user-facing feature. Must work for sync and async functions, propagate via contextvars, never leak state.

**Skills:** `brainstorming`, `writing-plans`, `test-driven-development`.

**Prompt:**

```
This phase defines the user-facing API. Brainstorm carefully — every choice here is a public API commitment.

Brainstorm:
- Top-level functions: `start_trajectory(task, *, agent_name, agent_version, model_id, store=None)` returns a context manager that yields a TrajectoryHandle. `@trace_step(name, step_type="internal")` decorator. `record_llm_call(...)`, `record_tool_call(...)` for direct recording from inside a manually-traced block.
- Context propagation: `contextvars.ContextVar` holding the current TrajectoryHandle and current Step. Decorators read from these to attach to the right parent.
- Async-correctness: the contextvar approach works correctly with asyncio.gather and TaskGroup. Verify with tests.
- Sync support: same decorator works on sync functions. Use `inspect.iscoroutinefunction` to switch.
- Failure modes: what happens if no trajectory is active when a step is recorded? Configurable via `ARIADNE_FAIL_MODE`: "strict" (raise, default in dev), "warn" (log, default in prod), "silent" (do nothing). The fail mode is read at module init from env var.
- Sampling: a `Sampler` protocol with `should_sample(trajectory_metadata) -> bool`. Default `AlwaysSampler`. Provide `RateSampler(0.1)` and `TaskFilterSampler(predicate)`.
- Performance: tracing must not block the agent's hot path. Step records are written to an asyncio queue, drained by a background task that batches writes to the store. Default queue size 1000, drop-oldest on overflow with a warning.
- LiteLLM auto-trace: a small helper `enable_litellm_autotrace()` that registers a LiteLLM callback to record every LLM call as a step under the current trajectory. This is the single most useful integration.

Public API surface (must appear in `ariadne_eval/__init__.py`):
- `start_trajectory`, `current_trajectory`, `current_step`
- `trace_step`
- `record_llm_call`, `record_tool_call`
- `Sampler`, `AlwaysSampler`, `RateSampler`
- `enable_litellm_autotrace`
- `Trajectory`, `Step`, `LLMCall`, `ToolCall` (re-export)

Tests:
- `start_trajectory` opens and closes a trajectory; on exit, finished_at is set.
- Nested `@trace_step` correctly attaches steps under the current parent.
- Context propagation through `asyncio.gather` produces the right tree (run two parallel branches; both attach to the right parent).
- Sync and async functions both work.
- Manual `record_llm_call` attaches to the current step or trajectory root if no step active.
- Strict fail mode raises on unattached records; warn mode logs once.
- Sampler: `RateSampler(0.0)` skips entirely; `RateSampler(1.0)` always samples.
- Performance: with `AlwaysSampler`, decorating a function that does 1000 trivial steps adds <2% latency vs untraced (use the benchmarks/overhead.py script as the test, mark it `@pytest.mark.slow`).
- Property: any tree of `@trace_step`-decorated calls produces a Trajectory whose tree shape matches the call tree.

Files:
- src/ariadne_eval/tracing/decorator.py
- src/ariadne_eval/tracing/context.py
- src/ariadne_eval/tracing/sampler.py
- src/ariadne_eval/adapters/litellm.py
- src/ariadne_eval/__init__.py (update public API)
- tests/unit/tracing/...
- tests/property/test_tracing_tree.py
- benchmarks/overhead.py

Verify: tests green, mypy clean, the overhead benchmark prints a number <2% of baseline.
```

**Acceptance:**
- All tests pass including async-context propagation.
- Overhead benchmark shows <2% latency added.
- Public API documented in `docs/reference/tracing.md`.
- A working code example in `examples/01_quickstart/main.py` that traces a tiny ReAct-style loop.

---

## Phase 4 — Reference Agent and End-to-End Wiring

**Goal:** A simple reference ReAct agent that uses tracing end-to-end, demonstrating tracing → storage → retrieval.

**Skills:** `brainstorming`, `writing-plans`, `test-driven-development`.

**Prompt:**

```
Brainstorm:
- A minimal ReAct agent in `src/ariadne_eval/examples/react_agent.py` that:
  - Takes a task and a list of tools.
  - In a loop: ask the LLM for next action, parse the tool call, execute, append observation, repeat until the LLM emits "FINAL ANSWER:" or max_steps reached.
  - Uses `start_trajectory` and `enable_litellm_autotrace`.
  - Each tool execution is wrapped in `record_tool_call`.
- Tools for the demo: a `calculator(expression: str) -> float` and a `search(query: str) -> str` (mocked). Real internet search is out of scope for the reference agent.
- An end-to-end test that:
  - Sets up an in-memory DuckDB store.
  - Runs the ReAct agent on "What is 17 * 23, and then divide by the number of letters in 'banana'?"
  - Asserts the trajectory was saved with the expected structure.
  - Asserts at least one LLM call step and at least one tool call step.

Use VCR (pytest-recording) for the LLM call so the test is deterministic and runs in CI without keys. Record once with a real key, commit the cassette.

Files:
- src/ariadne_eval/examples/react_agent.py
- src/ariadne_eval/examples/tools.py
- tests/integration/test_react_end_to_end.py
- tests/integration/cassettes/test_react_end_to_end.yaml (committed)
- examples/01_quickstart/main.py
- examples/01_quickstart/README.md
- docs/quickstart.md (update with the working example)

Verify:
- `pytest -m integration tests/integration/test_react_end_to_end.py` passes using the cassette.
- The example in `examples/01_quickstart/main.py` runs end-to-end with a real API key.
- `docs/quickstart.md` matches the example file (use mkdocs include if possible).
```

**Acceptance:**
- Reference agent runs and produces a sensible trajectory.
- E2E test green via VCR cassette.
- Quickstart docs page accurate.

**Critical pitfall:** Make sure the cassette redacts API keys before committing. Use pytest-recording's `filter_headers` configuration.

---

## Phase 5 — Programmatic Trajectory Metrics

**Goal:** Implement five trajectory-level metrics, all with confidence-interval-aware reporting.

**Skills:** `brainstorming`, `writing-plans`, `test-driven-development`.

**Prompt:**

```
Brainstorm each metric and its contract:

1. `final_answer_accuracy(trajectory, gold_answer, comparator) -> MetricResult`. The comparator is pluggable: exact_string, normalized_string, semantic_match (LLM-based, optional), or a user callable. Output: score in {0.0, 1.0}, plus details.

2. `tool_call_accuracy(trajectory, gold_tool_calls) -> MetricResult`. Computes per-tool-call F1 against an ordered list of gold calls. Match criteria: tool_name exact + arguments structurally equivalent (with optional argument-aware comparison via per-tool schemas). Output: precision, recall, F1, plus details.

3. `plan_quality(trajectory, *, judge=None) -> MetricResult`. Identify the first LLM call's planning output (heuristic: first message that mentions a plan or has bullet-point structure). Score the plan via judge. Phase 6 implements the judge; in this phase, leave plan_quality as a callable that takes a judge and just calls it.

4. `recovery_rate(trajectory) -> MetricResult`. Count tool_call steps with status="failed". A "recovery" is when a failure is followed by a step that addresses the same intent (e.g., retries with corrected arguments) and that subsequent step succeeds. Heuristic detection of "addressing the same intent": same tool_name within next 3 steps. Output: recoveries / failures (or 1.0 if no failures).

5. `efficiency(trajectory, gold_step_count=None) -> MetricResult`. If gold_step_count is provided, return gold/actual (capped at 1.0). Otherwise return raw step count and let aggregation report distribution.

All metrics return a `MetricResult` Pydantic model: name, score, details (dict), confidence_interval (tuple, default None — set by aggregator).

Aggregator:
- `aggregate_metric(results: list[MetricResult]) -> AggregatedMetric` returns mean and bootstrapped 95% CI.
- For binary metrics (accuracy), bootstrap is on the mean of the 0/1 scores.

Tests:
- Each metric has unit tests on hand-crafted trajectories.
- Property test: identical trajectories produce identical scores.
- Property test: aggregator's bootstrap CI covers the true mean ~95% of the time on synthetic data (calibration check).

Files:
- src/ariadne_eval/eval/metrics/base.py
- src/ariadne_eval/eval/metrics/final_answer.py
- src/ariadne_eval/eval/metrics/tool_accuracy.py
- src/ariadne_eval/eval/metrics/plan_quality.py
- src/ariadne_eval/eval/metrics/recovery.py
- src/ariadne_eval/eval/metrics/efficiency.py
- src/ariadne_eval/eval/stats/bootstrap.py
- tests/unit/eval/metrics/...
- tests/property/test_metrics_properties.py
- METHODOLOGY.md (start it; document each metric's definition and known limitations)

Verify: tests green, mypy clean, the calibration test for bootstrap passes (this is a probabilistic test; allow up to 2 retries with different seeds).
```

**Acceptance:**
- Five metrics implemented and tested.
- Bootstrap aggregator calibrated.
- METHODOLOGY.md has a "Metrics" section.

---

## Phase 6 — Calibrated Trajectory Judge

**Goal:** An LLM judge for trajectory quality, with documented kappa vs hand-labeled gold set. This is the credibility-establishing phase.

**Skills:** `brainstorming`, `writing-plans`, `test-driven-development`.

**Prompt:**

```
This phase is the trust-establishing differentiator. We do NOT ship a black-box judge.

Brainstorm:
- Two judge granularities:
  - Trajectory-level: input is the full trajectory + task + (optional) reference answer; output is binary success and structured rubric scores (1–5) on coherence, efficiency, and correctness.
  - Step-level: input is a single step in context; output is a step quality verdict and a free-form rationale.
- Three prompt variants per granularity to compare: zero-shot, with rubric, with few-shot examples.
- Two judge models to compare: claude-sonnet (current), gpt-4o.
- Calibration set: 50 hand-labeled trajectories sampled from the reference agent on tau-bench tasks. Labels are binary task-success and 1–5 rubric scores from the maintainer.

Workflow:
- `scripts/build_calibration_set.py`: samples N trajectories from a configured store (or generates them from the reference agent), presents each to the user via a rich CLI (one trajectory at a time, formatted for readability), captures labels, writes to `data/calibration/labels.jsonl`. The labeling tool must let the user pause and resume.
- `scripts/run_judge_calibration.py`: runs each (model, prompt-variant) judge config against the calibration set, computes accuracy / kappa / FPR / FNR vs human, writes a calibration report to `results/judge_calibration_<timestamp>.html` and a machine-readable JSON.
- The maintainer commits the labels file (it's the work). Reproduction is documented.

Implementation:
- `src/ariadne_eval/eval/judges/base.py`: abstract `Judge` with async `judge(trajectory, **context) -> JudgeVerdict`.
- `src/ariadne_eval/eval/judges/trajectory_judge.py`, `stepwise_judge.py`.
- `src/ariadne_eval/eval/judges/prompts.py`: versioned prompts (v1, v2, ...). Versions are explicit; changing a prompt requires bumping the version.
- `src/ariadne_eval/eval/judges/ensemble.py`: majority-vote / averaged-rubric ensemble.
- `src/ariadne_eval/eval/stats/agreement.py`: cohen_kappa, krippendorff_alpha, percent_agreement.

Tests:
- Unit: prompt formatting matches expectations on fixture trajectories.
- Unit: ensemble correctly aggregates synthetic verdicts.
- Unit: kappa matches sklearn.metrics.cohen_kappa_score on synthetic data.
- Integration: run a single judge on one fixture trajectory using a recorded cassette; verdict shape is correct.

Documentation deliverable:
- METHODOLOGY.md updated with a full "Judge Calibration" section: methodology, results table (model × prompt-variant × kappa), the chosen production config, known failure modes (e.g., "judges over-credit verbose plans").
- A figure: kappa heatmap per (judge, prompt). SVG, committed to docs/figures/.
- README updated with the chosen judge config and its kappa.

If the best kappa is < 0.6, that is a finding, not a failure. Document it. Discuss why and what would help.

Files:
- src/ariadne_eval/eval/judges/...
- src/ariadne_eval/eval/stats/agreement.py
- scripts/build_calibration_set.py
- scripts/run_judge_calibration.py
- data/calibration/labels.jsonl (created during the labeling session, then committed)
- docs/figures/judge_kappa_heatmap.svg
- METHODOLOGY.md (extended)
- README.md (calibration result added)

Verify: unit tests green, calibration script runnable end-to-end, calibration report exists.
```

**Acceptance:**
- 50 hand-labeled trajectories committed.
- Calibration report with kappa values produced.
- Best judge config documented.
- Heatmap figure in docs.

---

## Phase 7 — Benchmark Runner (tau-bench Integration)

**Goal:** Run reference and user agents on tau-bench. Produce headline result tables with confidence intervals.

**Skills:** `brainstorming`, `writing-plans`, `test-driven-development`.

**Prompt:**

```
Brainstorm:
- tau-bench is Sierra's open-source benchmark for tool-using agents in retail and airline domains. It's pip-installable.
- Our role: be a *runner* on top of tau-bench, not a replacement. We trace each task, save the trajectory, and compute our metrics on it.
- An `AbstractBenchmark` protocol: `tasks() -> Iterable[BenchmarkTask]`, `verify(trajectory, task) -> bool`. Concrete `TauBenchAdapter` wraps tau-bench's task loader and verifier.
- A benchmark run config (YAML) specifies: benchmark, agent factory (import path), models to compare, sampling (N tasks, seed), concurrency, output directory.
- The runner: dispatch tasks across the configured concurrency, save each trajectory, run all trajectory-level metrics, run the chosen judge config, aggregate with bootstrapped CIs, write a result bundle.
- Result bundle layout: `results/<run_id>/{config.yaml, summary.json, trajectories.jsonl, metrics.parquet, report.html}`.

CLI:
- `ariadne bench run <config.yaml>` — execute the benchmark.
- `ariadne bench compare <run_id_a> <run_id_b>` — paired comparison with McNemar's test.
- `ariadne bench list` — list past runs with summary metrics.

Headline-result deliverable:
- A run config `configs/benchmarks/tau_retail_baseline.yaml` that runs three agents (reference ReAct on claude-sonnet, claude-haiku, gpt-4o-mini) on the tau-retail subset, 50 tasks, seed=42.
- The README's headline table is generated from this run.

Tests:
- Unit: TauBenchAdapter correctly wraps task iteration and verification (use a mock benchmark in tests; don't depend on tau-bench network).
- Integration: a 2-task smoke run completes with cassettes.
- Property: any task that succeeds via the verifier is recorded with `final_status=succeeded`.

Files:
- src/ariadne_eval/benchmarks/base.py
- src/ariadne_eval/benchmarks/tau_bench.py
- src/ariadne_eval/eval/runner.py
- src/ariadne_eval/cli/bench.py
- configs/benchmarks/tau_retail_baseline.yaml
- tests/unit/benchmarks/...
- tests/integration/test_bench_smoke.py

Verify: smoke test passes; manual run on 5 tasks produces a result bundle viewable as expected.
```

**Acceptance:**
- `ariadne bench run` works end-to-end.
- Headline result table generated and committed to docs.
- Paired comparison CLI produces sensible output.

**Caveat:** Document tau-bench's licensing and any rate limits in METHODOLOGY.md.

---

## Phase 8 — CLI Polish

**Goal:** A coherent, well-documented CLI: `ariadne ui`, `ariadne eval`, `ariadne bench`, `ariadne export`, `ariadne import`.

**Skills:** `writing-plans`, `verification-before-completion`.

**Prompt:**

```
Brainstorm command surface (no large changes — just polish):
- `ariadne ui [--port 8501] [--store PATH]`: launch the Streamlit replay UI.
- `ariadne eval <metric_set> --store PATH [--filter ...]`: run metrics on existing trajectories.
- `ariadne bench run <config.yaml>`: already from Phase 7.
- `ariadne bench compare <run_a> <run_b>`: paired comparison.
- `ariadne export jsonl --output FILE [--filter ...]`: export trajectories to portable JSONL.
- `ariadne import jsonl --input FILE`: import.
- `ariadne version`: detailed version info (lib, python, duckdb).
- `ariadne --help` is excellent. Every subcommand has examples in its help text.

Use `rich` for output: progress bars for long ops, tables for results, colored status indicators.

Tests:
- CLI smoke tests via `click.testing.CliRunner` for each subcommand's `--help` and basic invocation.
- An end-to-end CLI test: bench run (mocked) → ui spawns → export → import → equality.

Files:
- src/ariadne_eval/cli/main.py (group + plumbing)
- src/ariadne_eval/cli/eval.py
- src/ariadne_eval/cli/ui.py
- src/ariadne_eval/cli/bench.py (extend)
- src/ariadne_eval/cli/export.py
- tests/unit/cli/...

Verify: CLI tests green; manual `ariadne --help` is informative.
```

**Acceptance:**
- All CLI commands functional and documented.
- `ariadne --help` reads well.

---

## Phase 9 — Streamlit Replay UI

**Goal:** A multipage Streamlit app that turns trajectories into something you'd actually want to look at.

**Skills:** `brainstorming`, `writing-plans`. Use the `frontend-design` skill for component design.

**Prompt:**

```
Use the frontend-design skill (`/mnt/skills/public/frontend-design/SKILL.md`) before designing pages.

Brainstorm pages:
1. Runs — table of recent trajectories with filters (agent, model, status, time range), paginated, click-through to drill-down.
2. Trajectory — drill-down on a single run. Top section: task, agent, model, total cost, latency, final status. Tree/graph visualization of steps. Click a step → side panel with full payload (prompts, completions, tool args, results), pretty-printed and copy-able.
3. Compare — pick two trajectories or two runs, see metric deltas, McNemar p-values, list of trajectories where outcomes differ.
4. Metrics — across runs, plot metrics over time / by agent version / by model. Include the Pareto plot (accuracy vs cost vs latency).
5. Drift — timeseries dashboards: success rate over time, mean latency, mean cost, with drift markers (Phase 10 will compute drift; here just show the timeseries).
6. Calibration — render the judge calibration report from Phase 6 inline: kappa heatmap, confusion matrices.

Graph view:
- Use a small custom component or pyvis to render the step tree. Each node colored by status. Click to drill in.
- Lay out as a tree (left to right or top to bottom). For wide trees, allow collapse/expand.

Performance:
- Cache store reads with `@st.cache_data(ttl=10)`.
- For trajectories with >500 steps, lazy-load the step list.

Polish:
- Dark mode by default. Brand color (pick one — recommend a deep teal #0E5F66 and stick with it).
- Empty states are designed: "No trajectories yet — try the quickstart."
- Every page has a help expander with usage tips.

Deliverable:
- All six pages functional on the data from the Phase 7 benchmark run.
- Six screenshots committed to `docs/screenshots/`, each ≤500KB.
- An animated GIF (≤2MB) of clicking through Runs → Trajectory → step drill-down, embedded in the README.

Files:
- src/ariadne_eval/ui/app.py
- src/ariadne_eval/ui/pages/...
- src/ariadne_eval/ui/components/graph.py
- src/ariadne_eval/ui/components/diff.py
- docs/screenshots/*.png
- docs/screencast.gif
- README.md (updated with screenshots and GIF)

Verify: `ariadne ui` launches; each page renders without error on real data; screenshots are committed.
```

**Acceptance:**
- All pages functional.
- GIF and screenshots polished.
- README opens with the GIF embedded.

---

## Phase 10 — Drift Detection

**Goal:** Detect quality regressions in production trajectories. Operationalize "is my agent getting worse?"

**Skills:** `brainstorming`, `writing-plans`, `test-driven-development`.

**Prompt:**

```
Brainstorm:
- Drift signals to monitor:
  - Success rate (binary timeseries).
  - Mean trajectory cost.
  - Mean trajectory latency.
  - Tool failure rate.
  - Mean steps to completion.
- Detection methods:
  - CUSUM (cumulative sum) for monotonic drift on continuous metrics.
  - ADWIN-style adaptive windowing for change-point detection.
  - For binary metrics, use a sequential probability ratio test.
- Output: a `DriftReport` per metric with: detected (bool), change_point_timestamp (optional), magnitude, p-value, recent vs baseline window summary.
- The detector runs on demand (CLI / UI page) and as a scheduled GitHub Action that posts a comment on a tracking issue if drift is detected on the public benchmark dataset.
- Document the false-positive rate of each method on synthetic stationary data — this is critical methodology.

Implementation:
- `src/ariadne_eval/eval/stats/drift.py` with `cusum_detect`, `adwin_detect`, `sprt_detect`.
- `src/ariadne_eval/eval/drift_runner.py` to compute reports across all metrics.
- A CLI command: `ariadne drift report --since "30d"`.

Tests:
- Synthetic stationary series: false positive rate stays under specified alpha (5%) over 1000 trials.
- Synthetic shifted series: detection rate at least 80% for shifts of magnitude ≥1 sigma.
- Property: detector output is deterministic for a fixed seed.

GitHub Action:
- A workflow that runs nightly: pulls the latest store from a release artifact (or skips if missing), runs drift report, posts to a sticky issue if drift detected.
- Document this as a pattern users can adopt for their own deployments.

Files:
- src/ariadne_eval/eval/stats/drift.py
- src/ariadne_eval/eval/drift_runner.py
- src/ariadne_eval/cli/main.py (add `drift` subcommand)
- .github/workflows/benchmark-tracking.yml
- src/ariadne_eval/ui/pages/05_drift.py (add detection markers from Phase 9 placeholder)
- tests/property/test_drift_calibration.py
- METHODOLOGY.md updated with drift methodology and FPR results

Verify: tests pass, FPR calibration test passes, CLI command produces a report on Phase 7 data.
```

**Acceptance:**
- Drift detector with documented FPR.
- CLI and UI integration.
- METHODOLOGY.md has a "Drift Detection" section with calibration results.

---

## Phase 11 — Documentation, Packaging, and PyPI Release

**Goal:** Ship a 0.1.0 release. Real users can install it and use it.

**Skills:** `brainstorming`, `verification-before-completion`.

**Prompt:**

```
Brainstorm the public-facing artifacts:

Documentation:
- mkdocs site with: index, quickstart, concepts (trajectory, tracing model, judges, drift), tutorials (5 worked examples), reference (auto-generated via mkdocstrings), contributing.
- Each page has runnable code where applicable.
- API reference covers all public symbols.
- A "Compared to alternatives" page that honestly compares to LangSmith, Langfuse, Helicone, Arize Phoenix. State each tool's strengths, where ariadne-eval differs, who should use what. (This page is gold for SEO and credibility.)

README polish:
- Open with: GIF, one-line, three-bullet "what makes this different", install command, 5-line code quickstart, headline benchmark table with CIs, link to docs.
- Badges: PyPI version, Python versions, license, CI status, docs status, GitHub stars (auto-updating).
- Below the fold: features, examples, comparison, contributing, citation.

PyPI release:
- `.github/workflows/publish-pypi.yml` triggered by tag push. Uses Trusted Publisher (OIDC, no API key in repo).
- Test the publish flow on TestPyPI first.
- Tag v0.1.0, publish.
- Verify: in a fresh venv, `pip install ariadne-eval` works and the quickstart runs.

GitHub repo polish:
- Issue templates (bug report, feature request, question).
- PR template with checklist.
- "Good first issue" labels on 5+ tractable starter issues.
- Discussions enabled.
- Topics set: llm, llm-evaluation, observability, agents, opentelemetry, ai-safety.
- A pinned issue: "Roadmap to 1.0".

Docs deployment:
- `.github/workflows/docs.yml` deploys to GitHub Pages on push to main.

Citation:
- `CITATION.cff` so the project is citable.
- A Zenodo-style DOI (optional, document how to get one).

Blog post:
- `docs/blog/01-why-trajectory-eval.md`: a 1500-word post titled something like "Why your agent eval is lying to you" — first-person, honest about the gap between "final answer accuracy" and "did the agent actually behave well." This is the artifact you'll link from your portfolio.

Verify the 90-second test:
- Open the GitHub repo as a stranger.
- Within 90 seconds, can you (a) understand what it does, (b) see proof it works, (c) install it?

Files:
- docs/* (all)
- mkdocs.yml (final)
- .github/workflows/publish-pypi.yml
- .github/workflows/docs.yml
- .github/ISSUE_TEMPLATE/*
- .github/PULL_REQUEST_TEMPLATE.md
- CITATION.cff
- README.md (final)
- docs/blog/01-why-trajectory-eval.md

Verify: live docs site, live PyPI release, working install in a fresh venv.
```

**Acceptance:**
- v0.1.0 released on PyPI.
- Docs deployed to GitHub Pages.
- README passes the 90-second test.
- Blog post written.

---

## Phase 12 (optional but high-value) — Reference Production Deployment

**Goal:** Show the library running in a realistic deployment, with a live demo.

**Skills:** `brainstorming`, `writing-plans`.

**Prompt:**

```
Brainstorm a reference deployment that demonstrates production usage:

Pick one:
A. A small ReAct agent deployed to fly.io / Railway, with ariadne-eval tracing every interaction. The Streamlit UI is publicly accessible (read-only) showing anonymized trajectories. Updated daily via a scheduled run on a fixed task set, so visitors always see fresh data.

B. A Discord bot or Slack bot that does something useful (e.g., answers questions about a small KB) and uses ariadne-eval tracing. Logs are public, so contributors can see the agent's behavior.

C. A GitHub Action that any user can drop into their own LLM-app repo to enable trajectory tracking on PRs.

(A) is the strongest portfolio signal; (C) is the most useful long-term to other users.

Whichever you pick:
- The deployment itself is in a separate repo (`ariadne-demo`) linked from the main README.
- Anonymization: any task content from real users is hashed/dropped. Only synthetic example tasks appear publicly.
- Cost cap: hard daily spend limit enforced at the agent level.
- Documentation: a `docs/tutorials/06_production_deployment.md` page walks through how to do the same thing.

This phase signals: "I built a library, but I also know how to run it."
```

**Acceptance:**
- Live demo URL in the README.
- Tutorial documenting the deployment.

---

## Tips for using this file

- **Don't paste the whole file into Claude Code at once.** One phase per session.
- **Always start a phase by re-reading CLAUDE.md.** Sessions don't share memory.
- **If a phase fails halfway**, debug it now with `systematic-debugging`. Bad foundations compound.
- **Commit at the end of every phase** with a Conventional Commit message. The git history is part of the portfolio.
- **Update the public docs in lockstep with code.** Don't let docs lag.
- **Tag a release at the end of each phase from Phase 7 onward** as `v0.0.7-alpha`, `v0.0.8-alpha`, etc. This makes the project history visible on GitHub releases.
