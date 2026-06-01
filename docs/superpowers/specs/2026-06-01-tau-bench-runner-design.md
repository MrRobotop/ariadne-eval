# Phase 7 Design: tau-bench Benchmark Runner

**Status:** Approved (2026-06-01) · **Phase:** 7 · **Target version:** 0.0.9-alpha

## Goal

Ship the headline benchmark machinery promised by the README. Wrap
Sierra's open-source τ-bench (the canonical tool-using-agent benchmark)
behind a tau-agnostic `Benchmark` Protocol, trace each task end-to-end
through ariadne-eval's existing tracing + storage stack, score the
captured trajectories with the existing eval Runner + `PlanQuality`
judge, and write a reproducible result bundle (config + trajectories +
summary) for one canonical run: **two agent models × 50 retail tasks**.

The deliverable that unblocks the README's headline table is the
committed result bundle at
`docs/benchmarks/v0.0.9-alpha-tau-retail-50/`.

## Budget envelope

This phase's manual run must fit a **$5 Anthropic cap** (Groq is
unlimited under the maintainer's account). The model lineup and
tau-bench user-strategy choice below are constrained by this cap.

Estimated Anthropic spend on the canonical run:
- Haiku 4.5 agent on 50 tasks: ~$2.00
- Sonnet 4.6 judge on 100 trajectories (`PlanQuality`): ~$1.00
- **Anthropic total: ~$3.00** (leaves ~$2 retry headroom)

Groq spend on the canonical run (free under maintainer's account):
- Llama 3.3 70B agent on 50 tasks
- Llama 3.3 70B simulated user across 100 cells (the biggest
  line-item if it were on Anthropic — running it on Groq is the
  load-bearing budget decision).

## Scope

In, in build order:

1. **`_transient` extraction.** Move the transient-error retry helpers
   (`_TRANSIENT_EXC_NAMES`, `_is_transient`,
   `_MAX_TRANSIENT_RETRIES`, `_TRANSIENT_BACKOFF_BASE`) out of
   `scripts/build_calibration_set.py` and into a new
   `src/ariadne_eval/_transient.py`. The calibration script imports
   from it; the new benchmark runner also imports from it. Single
   source of truth.
2. **`Benchmark` Protocol + `BenchmarkTask` + `BenchmarkRunResult`.**
   `src/ariadne_eval/benchmarks/base.py`. Trajectory-agnostic
   contract: tasks, run_task, success/score. No tau-bench types
   leak through; `BenchmarkTask.payload` is `Any` so a future
   SWE-bench-Lite adapter can plug in.
3. **`TauBenchAdapter`.** `src/ariadne_eval/benchmarks/tau_bench.py`.
   Wraps `tau_bench.envs.get_env` + the chosen agent class
   (`ToolCallingAgent` default). Lazy-imports `tau_bench` so the
   absence of the `[tau-bench]` extra is an actionable error, not a
   startup crash for users who don't need it.
4. **`_convert_tau_traj` helper.** Same file. Converts tau-bench's
   flat message-list `EnvRunResult.traj` into the ariadne
   `Trajectory` + parent/child `Step` tree. The convention matches
   what `enable_litellm_autotrace` produces natively, so downstream
   metrics see indistinguishable trajectories.
5. **`BenchmarkConfig` + YAML loader.** `src/ariadne_eval/benchmarks/config.py`.
   Pydantic-validated. Bad YAML fails loudly.
6. **`BenchmarkRunner`.** `src/ariadne_eval/benchmarks/runner.py`.
   Orchestrates `(task × model)` cells under bounded concurrency,
   persists trajectories to the store, runs the existing eval
   `Runner` per-trajectory, aggregates per-model results with
   bootstrap CIs, writes the result bundle.
7. **`ariadne bench run` CLI.** `src/ariadne_eval/cli/bench.py`. Click
   subcommand. Flags: `--dry-run`, `--limit`, `--models`, `--resume`.
8. **Headline config.** `configs/benchmarks/tau_retail_baseline.yaml`
   declares the 2-model × 50-task retail run (Haiku 4.5 + Llama 3.3 70B).
9. **One-shot manual run** by the maintainer with real
   `ANTHROPIC_API_KEY` + `GROQ_API_KEY`. Produces the committed
   bundle.
10. **Docs + CHANGELOG + version bump + tag.**

Out (deferred to later 7.x or future phases):

- `ariadne bench compare` (paired McNemar's test) — 7.1.
- `ariadne bench list` (run discovery from a results dir) — 7.1.
- HTML report generator — Phase 9 (UI) ships its own reader.
- tau-airline domain run — 7.1.
- Per-tool failure-rate breakdown — Phase 10 (drift) shares
  primitives.
- The "why per-step judges disagree more" blog post — Phase 11
  (docs/blog) territory.
- SWE-Bench-Lite adapter — separate phase; the Protocol is designed
  to accept it.

## Why these scope cuts

Phase 7 is shipping the **first real benchmark table**. The minimum
viable surface is "load tasks, run agent, capture trajectory, score it,
write a bundle." Comparison / list / HTML report are user-experience
layers that depend on having at least one bundle to compare against.
Shipping them in the same phase would slow down the headline result by
weeks; shipping them in 7.1 against a real bundle is cleaner.

## Architecture

```
                ┌───── tau-bench (git, MIT) ─────┐
                │   tau_bench.envs.get_env       │
                │   tau_bench.agents.*           │
                └────────────┬───────────────────┘
                             │ (lazy import, only inside [tau-bench] extra)
                             ▼
src/ariadne_eval/benchmarks/
├── __init__.py              Public re-exports: Benchmark, BenchmarkTask, BenchmarkRunResult,
│                            BenchmarkConfig, BenchmarkRunner. NOT tau_bench (extra-gated).
├── base.py                  Benchmark Protocol; BenchmarkTask + BenchmarkRunResult dataclasses
├── tau_bench.py             TauBenchAdapter; _convert_tau_traj helper
├── config.py                BenchmarkConfig Pydantic model + YAML loader
└── runner.py                BenchmarkRunner

src/ariadne_eval/
└── _transient.py            transient-error retry primitives (extracted from Phase 6.1 script)

src/ariadne_eval/cli/
└── bench.py                 ariadne bench run

configs/benchmarks/
└── tau_retail_baseline.yaml the headline run config

docs/benchmarks/v0.0.9-alpha-tau-retail-50/
├── config.yaml              copy of input config (audit trail)
├── trajectories.jsonl       100 lines (50 tasks × 2 models)
└── summary.json             headline pass rates + bootstrap CIs + per-model metric aggregates
```

**Hard Rule #3 compliance:** zero `tau_bench` references in
`ariadne_eval/__init__.py`. `benchmarks.tau_bench` lazy-imports
`tau_bench` and raises an actionable `ImportError` ("Install
ariadne-eval with the [tau-bench] extra: `pip install
'ariadne-eval[tau-bench]'`") when the extra isn't installed. Anyone
who doesn't want tau-bench pays no startup cost and gets no
`tau_bench` symbol in their namespace.

## Public API

```python
# src/ariadne_eval/benchmarks/base.py

@dataclass(frozen=True)
class BenchmarkTask:
    """One task in a benchmark."""
    task_id: str
    task_index: int           # 0-based ordinal in the benchmark's task list
    instruction: str          # the user-facing task description
    payload: Any              # benchmark-specific; not interpreted by the runner


@dataclass(frozen=True)
class BenchmarkRunResult:
    """What the benchmark hands back per task."""
    trajectory_id: str        # FK into the store; the full trace lives there
    success: bool             # derived from the benchmark's verifier
    raw_score: float          # the benchmark's native score (tau-bench: ≈1.0 → pass)
    error: str | None = None  # populated iff the task hit a hard failure


@runtime_checkable
class Benchmark(Protocol):
    """Trajectory-agnostic benchmark contract."""
    name: str

    def tasks(self, *, split: str = "test", limit: int | None = None) -> Sequence[BenchmarkTask]: ...

    async def run_task(
        self,
        task: BenchmarkTask,
        model: str,
        provider: str,
        *,
        store: Store,
        seed: int = 42,
    ) -> BenchmarkRunResult: ...
```

```python
# src/ariadne_eval/benchmarks/tau_bench.py

class TauBenchAdapter:
    """Concrete Benchmark over tau-bench's retail or airline domains."""

    def __init__(
        self,
        env_name: Literal["retail", "airline"],
        *,
        user_model: str = "groq/llama-3.3-70b-versatile",
        user_strategy: str = "llm",
        agent_kind: Literal["tool-calling", "react", "few-shot-tool-calling"] = "tool-calling",
    ) -> None: ...

    @property
    def name(self) -> str:
        return f"tau-{self._env_name}"

    def tasks(self, *, split: str = "test", limit: int | None = None) -> Sequence[BenchmarkTask]: ...

    async def run_task(
        self,
        task: BenchmarkTask,
        model: str,
        provider: str,
        *,
        store: Store,
        seed: int = 42,
    ) -> BenchmarkRunResult: ...
```

```python
# src/ariadne_eval/benchmarks/config.py

class ModelSpec(BaseModel):
    model_config = {"frozen": True}
    model: str
    provider: str


class JudgeSpec(BaseModel):
    model_config = {"frozen": True}
    model: str
    temperature: float = 0.0


class BootstrapSpec(BaseModel):
    model_config = {"frozen": True}
    n_resamples: int = 1000
    confidence: float = 0.95


class TasksSpec(BaseModel):
    model_config = {"frozen": True}
    limit: int | None = None
    seed: int = 42


class TauBenchSpec(BaseModel):
    model_config = {"frozen": True}
    kind: Literal["tau-bench"]
    env: Literal["retail", "airline"]
    task_split: str = "test"
    user_model: str = "groq/llama-3.3-70b-versatile"
    user_strategy: str = "llm"
    agent_kind: Literal["tool-calling", "react", "few-shot-tool-calling"] = "tool-calling"


class OutputSpec(BaseModel):
    model_config = {"frozen": True}
    bundle_dir: Path
    store_path: Path


class BenchmarkConfig(BaseModel):
    """Root config; loaded from YAML."""
    model_config = {"frozen": True}
    benchmark: TauBenchSpec      # extensible to other benchmarks via discriminated union later
    models: list[ModelSpec]
    tasks: TasksSpec
    concurrency: int = 4
    bootstrap: BootstrapSpec
    metrics: list[Literal["step_efficiency", "plan_quality"]]
    judge: JudgeSpec
    output: OutputSpec


def load_benchmark_config(path: Path) -> BenchmarkConfig: ...
```

```python
# src/ariadne_eval/benchmarks/runner.py

class BenchmarkRunner:
    def __init__(
        self,
        benchmark: Benchmark,
        models: Sequence[ModelSpec],
        *,
        store: Store,
        metrics: Sequence[Metric | AsyncMetric] = (),
        seed: int = 42,
        concurrency: int = 4,
        n_resamples: int = 1000,
        confidence: float = 0.95,
    ) -> None: ...

    async def run(
        self,
        tasks: Sequence[BenchmarkTask],
        *,
        resume_from_store: bool = False,
    ) -> "BenchmarkReport": ...

    def write_bundle(self, report: "BenchmarkReport", out_dir: Path) -> None: ...
```

## Data flow

For each `(task × model)` cell:

1. Acquire semaphore.
2. If `resume_from_store=True` and the store already has a trajectory
   with `(case_id=task.task_id, model_id=f"{provider}/{model}")`, skip
   the LLM call and use the stored trajectory.
3. Otherwise call `await benchmark.run_task(task, model, provider,
   store=store, seed=seed)`. The adapter is responsible for opening
   the trajectory handle, calling the agent, and persisting on exit.
4. Reload `(trajectory, steps)` from the store. This is intentional —
   it catches silent Pydantic round-trip drift early.
5. Build a `Case(case_id=task.task_id, task=task.instruction)`. No
   `expected_answer` field: this is a benchmark, not a fixture set.
6. Pass `(trajectory, steps, case)` to the eval `Runner.aevaluate`.
   `StepEfficiency` scores; `PlanQuality` scores (with the κ=0.32
   calibration caveat surfaced in summary.json).

Per-model aggregation:

- **Pass rate** = `bootstrap_mean_ci([1.0 if r.success else 0.0 for r in cell_results], seed=seed)`.
- **Per-metric aggregates** = the eval `Runner`'s existing
  `BootstrapCI` outputs over the metric scores.
- **Aux fields**: median step count, total input/output tokens, wall
  time.

## Trajectory conversion

`_convert_tau_traj(env_result, *, model_id, agent_name, agent_version) -> tuple[Trajectory, list[Step]]`
maps tau-bench's flat message list to our `Trajectory` + Step tree:

| Source | Maps to |
|---|---|
| `assistant` message with `tool_calls` | One `Step` per tool_call, payload = `LLMCallPayload` for the assistant reasoning + child `Step`s for each tool_call with `ToolCallPayload`. Parent/child relationship preserved. |
| `assistant` message without `tool_calls` (final answer) | One `Step` with `LLMCallPayload`; trajectory's `final_answer` = the assistant message content. |
| `tool` message (tool result) | Attached to the matching `Step`'s `ToolCallPayload.result`. |
| `user` message (the simulated user) | One `Step` with `UserInputPayload`. Surfaces tau-bench's two-LLM nature in the trace. |

This is the convention `enable_litellm_autotrace` produces from a real
ReAct loop. Converted tau-bench trajectories are indistinguishable from
natively-traced ones to downstream metrics. That's what lets us reuse
`Runner.aevaluate` and `PlanQuality` without modification.

The trajectory's metadata carries `tau_bench_reward` (the raw 0..1
float) and `tau_bench_task_id`. Anyone querying the store for failed
tau-bench tasks does `metadata->>'tau_bench_reward' < '1.0'` in
DuckDB.

## Run config YAML (canonical)

`configs/benchmarks/tau_retail_baseline.yaml`:

```yaml
# Phase 7 headline benchmark: tau-retail × 2 models × 50 tasks
benchmark:
  kind: tau-bench
  env: retail
  task_split: test
  user_model: groq/llama-3.3-70b-versatile   # tau-bench's simulated user; Groq to stay in budget
  user_strategy: llm
  agent_kind: tool-calling

models:
  - model: anthropic/claude-haiku-4-5-20251001
    provider: anthropic
  - model: groq/llama-3.3-70b-versatile
    provider: groq

tasks:
  limit: 50
  seed: 42

concurrency: 4

bootstrap:
  n_resamples: 1000
  confidence: 0.95

metrics:
  - step_efficiency
  - plan_quality

judge:
  model: anthropic/claude-sonnet-4-6
  temperature: 0.0

output:
  bundle_dir: docs/benchmarks/v0.0.9-alpha-tau-retail-50
  store_path: ~/.ariadne/bench-store.duckdb
```

The lineup is **two agent models** (down from three): Sonnet is dropped
to keep the run inside the $5 Anthropic envelope. The narrative
becomes "production-grade Anthropic mini vs production-grade open-
weights" — a sharper portfolio comparison than the within-Anthropic
ladder anyway. The simulated user runs on Groq (`groq/llama-3.3-70b-
versatile`) because it's the biggest line-item across all cells and
Groq is unlimited.

## CLI surface

```bash
ariadne bench run configs/benchmarks/tau_retail_baseline.yaml \
    [--dry-run]                 # validate config only, no LLM calls
    [--limit N]                 # override config tasks.limit
    [--models MODEL ...]        # subset config.models for partial reruns
    [--resume]                  # skip cells already in store; persist new ones
```

`--resume` is non-negotiable for a 100-cell run: any single failure
(rate-limit cascade, provider blip) must not force re-running the whole
thing. Even at the revised ~$3 budget, restarting from cell 0 after a
failure two-thirds through wastes both money and time.

## Result bundle

```
docs/benchmarks/v0.0.9-alpha-tau-retail-50/
├── config.yaml              copy of input config (audit trail)
├── trajectories.jsonl       100 lines (50 × 2), sorted by (task_id, model)
└── summary.json             headline numbers
```

**`summary.json` shape** (illustrative; numbers come from the run):

```json
{
  "_kind": "benchmark_summary",
  "benchmark": "tau-retail",
  "task_count": 50,
  "seed": 42,
  "run_date": "2026-06-XX",
  "ariadne_version": "0.0.9-alpha",
  "tau_bench_commit": "59a200c6d575d595120f1cb70fea53cef0632f6b",
  "models": [
    {
      "model": "anthropic/claude-sonnet-4-6",
      "provider": "anthropic",
      "pass_rate": {
        "mean": 0.640, "lo": 0.520, "hi": 0.760, "n": 50,
        "method": "bootstrap-percentile"
      },
      "metrics": {
        "step_efficiency": {"mean": 0.81, "lo": 0.74, "hi": 0.87, "n": 50},
        "plan_quality": {
          "mean": 0.47, "lo": 0.38, "hi": 0.56, "n": 50,
          "calibration_note": "judge κ = 0.32 (fair); see docs/concepts/calibration.md"
        }
      },
      "median_steps": 12,
      "total_input_tokens": 187432,
      "total_output_tokens": 24190,
      "wall_time_seconds": 1182.4
    }
  ]
}
```

The `calibration_note` field on every `plan_quality` aggregate makes
the κ=0.32 caveat travel with the number. A reader of `summary.json`
who hasn't read `calibration.md` still gets pointed there before they
over-trust the score.

**Sorted-keys + 2-space indent** on `summary.json`. One JSON object
per line on `trajectories.jsonl`, sorted by `(task_id, model)` tuple.
Diff-friendly across re-runs.

## Transient-error retry

Reuse Phase 6.1's `_is_transient` / `_TRANSIENT_EXC_NAMES` policy.
Extract from `scripts/build_calibration_set.py` into a new
`src/ariadne_eval/_transient.py`:

```python
# src/ariadne_eval/_transient.py

_MAX_TRANSIENT_RETRIES = 4
_TRANSIENT_BACKOFF_BASE = 2.0  # seconds; doubles each attempt: 2, 4, 8, 16
_TRANSIENT_EXC_NAMES = (
    "InternalServerError",
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "ServiceUnavailableError",
)


def is_transient(exc: Exception) -> bool:
    """Identify provider-side transient errors by class name (provider-portable)."""
    return type(exc).__name__ in _TRANSIENT_EXC_NAMES
```

Both `scripts/build_calibration_set.py` and the new
`BenchmarkRunner.run_task` wrapper import from this module. The
benchmark runner wraps each `tau-bench agent.solve` invocation with the
same retry loop pattern (try, on transient sleep + exponential backoff,
re-try up to N times, give up with a recorded error).

## Determinism

- All LLM calls (judge, agent, tau-bench's simulated user) use
  `temperature=0.0`.
- Anthropic at temp=0 is empirically bit-exact deterministic on the
  Phase 6.1 prompts; assume it is here too. Groq's determinism is
  less well-characterized — flag this in the result bundle's
  `summary.json` as a caveat once the headline run completes.
- Bootstrap CIs use `seed=42` (config-driven).
- Trajectory IDs come from `ariadne_eval.core.ids.new_id()` and are
  ULID-time-ordered — re-runs produce different IDs but the
  `tau_bench_task_id` metadata field is stable. The `resume` path
  matches on `tau_bench_task_id`, not on trajectory ID.

## Testing strategy

| Test | Type | File |
|---|---|---|
| `BenchmarkTask` / `BenchmarkRunResult` round-trip | fast unit | `tests/unit/benchmarks/test_base.py` |
| `Benchmark` Protocol is runtime-checkable | fast unit | same |
| `_convert_tau_traj` on hand-crafted EnvRunResult fixtures: zero tool-calls, multi-step, error-then-recover | fast unit | `tests/unit/benchmarks/test_tau_bench_convert.py` |
| `TauBenchAdapter.tasks()` raises actionable `ImportError` without `[tau-bench]` extra | fast unit | `tests/unit/benchmarks/test_tau_bench.py` |
| `TauBenchAdapter` with stubbed `tau_bench.envs.get_env` (monkeypatched) — full plumbing minus the real package | fast unit | same |
| `BenchmarkConfig` Pydantic validation: bad YAML, missing fields, invalid models | fast unit | `tests/unit/benchmarks/test_config.py` |
| `BenchmarkRunner` end-to-end with a `StubBenchmark` (3 tasks, 2 models) + `StubJudge` | fast unit | `tests/unit/benchmarks/test_runner.py` |
| `summary.json` shape is sorted-keys + 2-space-indent, contains `calibration_note` on plan_quality | fast unit | same |
| `ariadne bench run --dry-run` validates config without LLM calls | fast unit (`click.testing.CliRunner`) | `tests/unit/cli/test_bench.py` |
| `is_transient` recognizes the 5 expected error class names | fast unit | `tests/unit/test_transient.py` |
| Calibration script still works after `_transient` extraction (regression) | fast unit (existing test) | `tests/unit/scripts/test_build_calibration_set.py` |

Coverage gate: ≥ 90% on `src/ariadne_eval/benchmarks/` and on
`src/ariadne_eval/_transient.py`.

No integration test for the real tau-bench install. The real run is
the maintainer's one-shot Task 9 (this phase's equivalent of Phase
6.1's Task 7).

## Risks & non-obvious decisions

- **Why no `ariadne bench compare` / `list` in this phase.** A
  compare CLI is useful only against ≥ 2 bundles. This phase ships
  one bundle. Shipping the comparison machinery the same phase the
  data lands is YAGNI. 7.1 is the right home.
- **Why the existing `Runner.aevaluate` instead of a benchmark-specific
  runner.** Both run async metrics over `(trajectory, steps, case)`
  trios with bootstrap CIs. Reusing `Runner` keeps the
  trajectory→score code path identical between calibration and
  benchmark contexts. The BenchmarkRunner is a thin orchestration
  layer on top — it does NOT duplicate the metric loop.
- **Why store round-trip before scoring.** Catches silent Pydantic
  schema drift early. Costs one DuckDB read per cell (negligible
  next to LLM latency).
- **Why pin tau-bench by commit SHA, not by tag.** tau-bench has no
  release tags as of writing; pinning by SHA is the only
  reproducibility lever available. Re-pin when Sierra tags 0.2.0+.
- **Why surface the κ=0.32 calibration caveat in summary.json.** The
  number IS the deliverable; readers will skim it without reading
  `calibration.md`. A self-describing summary that points at its own
  caveats is the methodology-honest move.
- **Why drop `metrics.parquet` and `report.html` from the bundle.**
  Parquet is a real dep; JSONL + DuckDB already round-trip cleanly.
  HTML report is a Phase 9 concern (the Streamlit UI will read the
  bundle).
- **Why no integration test that runs real tau-bench.** Cost (real
  LLM dollars even at the revised lineup) and flakiness (multiple
  providers in the loop). The unit tests cover plumbing via stubs;
  the actual run IS the integration test.
- **Why `--resume` is in scope.** A 100-cell run hitting a single
  rate-limit cascade two-thirds through cannot reasonably mean
  re-paying for the first 100 cells. Resume-from-store is a
  must-have, not a nice-to-have.

## Definition of Done (Phase 7)

Standard project DoD plus:

- [ ] `src/ariadne_eval/_transient.py` exists; both
      `scripts/build_calibration_set.py` and the benchmark runner
      import from it (no duplicated constants).
- [ ] `src/ariadne_eval/benchmarks/{base.py, tau_bench.py, config.py,
      runner.py, __init__.py}` exist with the documented surface.
- [ ] `src/ariadne_eval/cli/bench.py` exists; `ariadne bench run
      --dry-run` validates a config without LLM calls.
- [ ] `configs/benchmarks/tau_retail_baseline.yaml` exists and
      Pydantic-validates.
- [ ] `pyproject.toml` declares the `[tau-bench]` extra pinned to
      commit `59a200c6d575d595120f1cb70fea53cef0632f6b` (current main
      as of 2026-06-01).
- [ ] Headline run executed once by the maintainer with
      `ANTHROPIC_API_KEY` + `GROQ_API_KEY`. Bundle committed at
      `docs/benchmarks/v0.0.9-alpha-tau-retail-50/{config.yaml,
      trajectories.jsonl, summary.json}`.
- [ ] `summary.json` carries the `calibration_note` field on every
      `plan_quality` aggregate.
- [ ] README updated with the headline table (2 rows: model × pass
      rate × CI). Linked from the "What's shipped" phase row.
- [ ] `docs/concepts/benchmarks.md` new page explaining the run
      methodology, pinned tau-bench commit, link to `summary.json`.
- [ ] mkdocs nav adds `Benchmarks` under Concepts. `mkdocs build
      --strict` clean.
- [ ] CHANGELOG `[Unreleased]` entry.
- [ ] Version bumped to `0.0.9-alpha` (`_version.py` +
      `pyproject.toml` + smoke tests). Bumped as a deliberate Task in
      the plan, not as an afterthought (Phase 6.1 lesson).
- [ ] All gates green: 90 %+ coverage on touched modules; mypy
      `--strict` clean; ruff clean; mkdocs strict clean.
- [ ] Tagged `v0.0.9-alpha` on `main` after `--no-ff` merge via
      GitHub PR.
- [ ] Memory updated: Phase 7 shipped, next phase = Phase 8 (CLI
      polish per `Prompts.md`).
