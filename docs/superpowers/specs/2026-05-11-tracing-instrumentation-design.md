# Phase 3 Design: Tracing Instrumentation

**Status:** Approved (2026-05-11) · **Phase:** 3 · **Target version:** 0.0.4

## Goal

The headline user-facing API: a `start_trajectory` async context manager and
a `@trace_step` decorator that turn any agent code into a recorded
`Trajectory`. Sync and async functions both work. Context propagation via
`contextvars` so `asyncio.gather` and `TaskGroup` produce the right tree.
LiteLLM auto-trace for the most common LLM-call pattern. <2 % runtime
overhead on a representative workload.

## Scope

In scope: `start_trajectory`, `@trace_step`, `record_llm_call`,
`record_tool_call`, the `Sampler` protocol with three concrete samplers,
the `ARIADNE_FAIL_MODE` policy, and `enable_litellm_autotrace`.

Out of scope: an asyncio queue + background drainer (deferred to a later
phase — the simpler save-on-exit model meets the overhead target); the
OpenTelemetry export bridge; streaming partial-chunk capture; sync context
manager (use `asyncio.run(...)` if you need to call `start_trajectory` from
sync code).

## Architecture

Five files under `src/ariadne_eval/tracing/` plus a LiteLLM adapter:

| File | Responsibility |
|---|---|
| `tracing/context.py` | `TrajectoryHandle`, `start_trajectory`, `current_trajectory`, `current_step`. Owns the two `ContextVar`s. |
| `tracing/decorator.py` | `@trace_step` (sync + async), `record_llm_call`, `record_tool_call`. |
| `tracing/sampler.py` | `Sampler` Protocol; `AlwaysSampler`, `RateSampler`, `TaskFilterSampler`. |
| `tracing/_fail_mode.py` | Reads `ARIADNE_FAIL_MODE`; `_handle_unattached(...)` policy applier. |
| `adapters/litellm.py` | `enable_litellm_autotrace` callback registration + the callback function. |

Persistence model: **in-memory build, save once on exit.** Every recording
appends to a list on the live `TrajectoryHandle`. At `async with` exit the
handle calls `store.save_trajectory(traj, steps)` exactly once. This meets
the <2 % overhead target with no concurrent I/O during step recording and
uses the Phase 2 API unchanged. Crash-resilience is a deferred concern.

## Context propagation

Two `ContextVar`s, set on entry and reset on exit:

```python
_current_trajectory: ContextVar[TrajectoryHandle | None] = ContextVar(
    "current_trajectory", default=None
)
_current_step: ContextVar[Step | None] = ContextVar(
    "current_step", default=None
)


def current_trajectory() -> TrajectoryHandle | None:
    return _current_trajectory.get()


def current_step() -> Step | None:
    return _current_step.get()
```

ContextVars are the right primitive: each `asyncio.gather` / `TaskGroup`
child gets a snapshot of the calling context, so children correctly attach
to the parent step. This is tested explicitly (parallel-branch test).

## `start_trajectory`

```python
@asynccontextmanager
async def start_trajectory(
    task: str,
    *,
    agent_name: str,
    agent_version: str,
    model_id: str,
    store: Store | None = None,
    sampler: Sampler | None = None,
    metadata: dict[str, JsonValue] | None = None,
) -> AsyncIterator[TrajectoryHandle]:
```

Flow on entry:

1. Build the underlying `Trajectory` model with `started_at=now(tz=UTC)`,
   `final_status=RUNNING`, `id=new_id()`.
2. Ask the sampler (default `AlwaysSampler()`). If it returns `False`,
   yield a `_NoOpHandle`. All subsequent recordings under that handle are
   silent no-ops, and nothing is saved on exit.
3. Build a real `TrajectoryHandle`. Set the `_current_trajectory`
   ContextVar.
4. Yield the handle.

Flow on exit:

- Clean exit: `final_status = SUCCEEDED` (or whatever
  `set_final_status(...)` set it to), `finished_at = now()`. If `store` is
  not `None`, `await store.save_trajectory(traj, steps)`.
- Exception propagates: `final_status = FAILED`. The exception is recorded
  in the trajectory metadata under the `_trajectory_error` key as
  `{"type": ..., "message": ...}`. Save (if store), then re-raise.
- Reset the `ContextVar`.

## `TrajectoryHandle`

Thin object the context manager yields:

```python
@dataclass
class TrajectoryHandle:
    id: str
    task: str
    agent_name: str
    agent_version: str
    model_id: str
    started_at: datetime
    _steps: list[Step] = field(default_factory=list)
    _metadata: dict[str, JsonValue] = field(default_factory=dict)
    _final_answer: JsonValue = None
    _final_status_override: TrajectoryStatus | None = None

    def add_metadata(self, key: str, value: JsonValue) -> None: ...
    def set_final_answer(self, answer: JsonValue) -> None: ...
    def set_final_status(self, status: TrajectoryStatus) -> None: ...

    def snapshot(self, *, finished_at: datetime, default_status: TrajectoryStatus) -> Trajectory:
        """Build the Trajectory model for persistence."""
```

User code mostly uses `traj.id` for logging plus `traj.set_final_answer(...)`
at the end of the run.

## `@trace_step`

```python
def trace_step(
    name: str,
    *,
    step_type: Literal["internal"] = "internal",
) -> Callable[[F], F]:
```

In v0.0.4, `step_type` only accepts `"internal"`. The parameter is reserved
for future expansion; passing anything else raises `ValueError` at decorator
construction.

Implementation:

- `inspect.iscoroutinefunction(fn)` chooses the sync or async wrapper.
- On entry: build a `Step` with `status=RUNNING`, `parent_step_id =
  current_step().id or None`, `started_at = now()`. Construct an
  `InternalPayload(kind=name, data=None)`. Set the `_current_step`
  ContextVar to this step.
- Run the wrapped function.
- On clean return: mutate the step to `status=SUCCEEDED`,
  `finished_at = now()`. Append to the trajectory's `_steps`. Restore the
  prior `_current_step`.
- On exception: `status=FAILED`, `error = StepError(type=type(exc).__name__,
  message=str(exc))`. Append, restore ContextVar, then re-raise.
- If `current_trajectory() is None`: apply `ARIADNE_FAIL_MODE`. `strict`
  raises before calling the function; `warn` logs once and calls the
  function untraced; `silent` calls the function untraced.

## `record_llm_call` / `record_tool_call`

```python
async def record_llm_call(
    *,
    model_id: str,
    prompt_messages: list[Message],
    completion: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    temperature: float | None = None,
    latency_ms: float,
    ttft_ms: float | None = None,
    tool_calls_emitted: list[str] | None = None,
    name: str = "llm_call",
) -> str:
    """Record an LLM call. Returns the step id."""


async def record_tool_call(
    *,
    tool_name: str,
    arguments: dict[str, JsonValue],
    result: JsonValue,
    latency_ms: float,
    name: str | None = None,
    error: StepError | None = None,
) -> str:
    """Record a tool call. Returns the step id."""
```

Both:

1. Build the typed payload (`LLMCallPayload` / `ToolCallPayload`).
2. Build the `Step` with `parent_step_id = current_step().id or None`,
   `status = SUCCEEDED` (or `FAILED` for tool calls with `error` set),
   `started_at = finished_at = now()` (the call already happened — these
   recorders are point-in-time).
3. Append to the trajectory's `_steps`. Return the step id.
4. If `current_trajectory() is None`: apply `ARIADNE_FAIL_MODE`.

The async signature exists so future implementations can do non-trivial
work without an API break. v0.0.4 does only in-memory bookkeeping.

## Sampler

```python
class Sampler(Protocol):
    def should_sample(
        self,
        *,
        task: str,
        agent_name: str,
        agent_version: str,
        model_id: str,
        metadata: dict[str, JsonValue],
    ) -> bool: ...


class AlwaysSampler:  # default
    def should_sample(self, **kw: object) -> bool:
        return True


@dataclass
class RateSampler:
    rate: float
    seed: int | None = None
    def should_sample(self, **kw: object) -> bool: ...


@dataclass
class TaskFilterSampler:
    predicate: Callable[[str], bool]
    def should_sample(self, *, task: str, **kw: object) -> bool:
        return self.predicate(task)
```

The sampler is consulted once per call to `start_trajectory`. If it returns
`False`, the entire trajectory is a no-op — every `@trace_step` and
`record_*` call inside it does nothing. This is what makes sampling cheap
in production: unsampled trajectories pay near-zero overhead.

`RateSampler` accepts an optional `seed` for deterministic test runs. With
no seed, it uses `random.random()`.

## Fail mode

```python
class FailMode(StrEnum):
    STRICT = "strict"
    WARN = "warn"
    SILENT = "silent"


def get_fail_mode() -> FailMode:
    """Read ARIADNE_FAIL_MODE env var at process start; cached."""
```

When a recording is attempted with no active trajectory, the configured
`FailMode` controls the behaviour:

- `STRICT` (default): `raise RuntimeError("no active trajectory")`. Loud
  failure in development.
- `WARN`: emit a warning *once per process* via `warnings.warn(...)` with a
  custom category (`UnattachedTracingWarning`) so users can filter or
  promote. The operation becomes a no-op.
- `SILENT`: no-op. Use in production when you do not want orphaned LLM
  calls to halt the agent.

Per the saved memory: this is a real runtime check that raises (or warns or
silently no-ops), not an `assert`. The behaviour is part of the public
contract.

## LiteLLM auto-trace

```python
def enable_litellm_autotrace() -> None:
    """Register a callback that records every litellm completion as an llm_call step."""
```

Implementation: append our callback to `litellm.success_callback` and
`litellm.failure_callback`. The callback receives `(kwargs, response,
start_time, end_time)`:

- Pull `model`, `messages`, `usage`, `temperature` from kwargs.
- Pull `completion` text and any tool-use blocks from the response.
- Compute `latency_ms = (end_time - start_time) * 1000`.
- Use `litellm.completion_cost(response)` for `cost_usd` (defaulting to 0
  on error so the trace still works).
- Schedule `record_llm_call(...)` on the active event loop via
  `asyncio.get_running_loop().create_task(...)`. If no loop is running or
  no trajectory is active, the fail mode policy applies.

The adapter is lazy-imported. `import ariadne_eval` does not import
LiteLLM. `enable_litellm_autotrace()` does.

## Public API additions

In `ariadne_eval/__init__.py`:

```python
"start_trajectory", "current_trajectory", "current_step",
"trace_step",
"record_llm_call", "record_tool_call",
"Sampler", "AlwaysSampler", "RateSampler", "TaskFilterSampler",
"enable_litellm_autotrace",
"TrajectoryHandle",
"FailMode",
"UnattachedTracingWarning",
```

## Testing strategy

| File | Coverage |
|---|---|
| `tests/unit/tracing/test_context.py` | start_trajectory entry / exit; ContextVar reset on exit including exception; `traj.snapshot()` shape; no-op handle when sampler returns False. |
| `tests/unit/tracing/test_decorator.py` | sync + async decoration; nested parent attachment; clean and failed exits; ContextVar reset; `step_type` validation. |
| `tests/unit/tracing/test_asyncio_gather.py` | Two parallel branches under `gather`; two parallel branches under `TaskGroup`; both attach to the right parent. |
| `tests/unit/tracing/test_sampler.py` | `AlwaysSampler`, `RateSampler(0.0)` skips, `RateSampler(1.0)` keeps, seeded determinism, `TaskFilterSampler`. |
| `tests/unit/tracing/test_fail_mode.py` | `strict` raises; `warn` logs once via caplog; `silent` no-op. Env-var override tested via monkeypatch + reload. |
| `tests/unit/tracing/test_recorders.py` | `record_llm_call` and `record_tool_call` attach to current step or trajectory root; produce correct payloads; return step id. |
| `tests/unit/tracing/test_storage_integration.py` | End-to-end: `start_trajectory(store=duckdb_store)` + several traced calls. Loaded trajectory matches the in-memory one. |
| `tests/property/test_tracing_tree.py` | Hypothesis: any tree of `@trace_step`-decorated calls produces a Trajectory whose tree shape matches the call tree. |
| `tests/unit/adapters/test_litellm.py` | Stub litellm; verify the callback registers and that a fake call recording produces an `llm_call` step. |
| `benchmarks/overhead.py` (`@pytest.mark.slow`) | 1000 trivial steps with vs without `AlwaysSampler`; assert <2 % overhead. |

**Coverage target:** ≥95 % on `src/ariadne_eval/tracing/` and the litellm
adapter. The litellm callback's error-path branches are allowed to be
pragma-excluded.

## Edge cases handled

1. **Sampler returns False.** `_NoOpHandle` returned; nothing recorded, no
   save. `current_trajectory()` still returns the no-op handle so `is None`
   checks see "active".
2. **Exception in wrapped function.** Step marked `FAILED` with
   `StepError`. Exception re-raised so callers can handle it normally.
3. **Exception escapes `start_trajectory`.** Trajectory marked `FAILED`;
   error captured in metadata; save + re-raise.
4. **Recording without an active trajectory.** Fail mode policy applied.
5. **Concurrent fan-out.** ContextVar snapshot per task → children attach
   to the right parent.
6. **`store=None`.** Trajectory is built in memory and discarded at exit.
   Useful for tests and ephemeral runs.

## Documentation

- `docs/concepts/tracing.md` — narrative.
- `docs/reference/tracing.md` — auto-generated API reference.
- `examples/01_quickstart/main.py` — a runnable ReAct-style example.
- `CHANGELOG.md [Unreleased]` entry.

## Deferred

- Asyncio queue + background drainer + drop-oldest overflow. The save-on-
  exit model meets <2 % overhead. The queue model adds significant
  complexity (lifecycle, multi-trajectory coordination) and only helps for
  crash-resilience and very-long-running trajectories. Revisit in Phase 12.
- OpenTelemetry export bridge.
- Streaming partial-chunk capture (we capture only the final assembled
  completion + TTFT).
- Sync `with` flavour of `start_trajectory`. Wrap in `asyncio.run` if you
  need it.
