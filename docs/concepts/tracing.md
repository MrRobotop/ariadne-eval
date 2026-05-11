# Tracing

`ariadne-eval` traces an agent run with two primitives: a context manager
that scopes the trajectory and a decorator that records steps inside it.

## The minimum

```python
import ariadne_eval as ae

async with ae.start_trajectory(
    "compute 2+2",
    agent_name="react",
    agent_version="0.1",
    model_id="claude-sonnet",
    store=ae.DuckDBStore(),
) as traj:
    answer = await run_agent()
    traj.set_final_answer(answer)
```

That writes one trajectory (metadata only — no steps yet) to DuckDB at
context exit.

## Adding steps

```python
@ae.trace_step("plan")
async def plan(task: str) -> str: ...

@ae.trace_step("execute")
async def execute(plan_text: str) -> float: ...
```

Each call to a decorated function becomes a Step under the active
trajectory. Steps nest correctly through `asyncio.gather` and
`asyncio.TaskGroup` — parent attachment propagates via Python's
`contextvars`.

## Typed step payloads

`@trace_step` always produces an `InternalPayload`. For LLM and tool
calls, use the explicit recorders inside your wrapped functions:

```python
await ae.record_llm_call(
    model_id="claude-sonnet",
    prompt_messages=[ae.Message(role="user", content="hi")],
    completion="hello",
    input_tokens=10, output_tokens=2, cost_usd=0.0001, latency_ms=120.0,
)

await ae.record_tool_call(
    tool_name="search",
    arguments={"q": "ariadne"},
    result=["hit1", "hit2"],
    latency_ms=15.0,
)
```

## Sampling

For production loads, pass a sampler:

```python
async with ae.start_trajectory(
    ..., sampler=ae.RateSampler(rate=0.1),
) as traj: ...
```

Unsampled trajectories are full no-ops — every decorator and recorder
short-circuits with near-zero overhead.

## Fail mode

Set `ARIADNE_FAIL_MODE`:

| value | behaviour when a recorder runs outside a trajectory |
|---|---|
| `strict` (default) | raise `RuntimeError("no active trajectory")` |
| `warn` | emit `UnattachedTracingWarning` once per process; no-op |
| `silent` | no-op silently |

Production deployments typically set `warn` or `silent`.

## LiteLLM auto-trace

```python
import ariadne_eval as ae
ae.enable_litellm_autotrace()
```

After this, every `litellm.completion(...)` (or `acompletion`) call inside
an `async with ae.start_trajectory(...)` is automatically recorded as an
`llm_call` step.

## What gets persisted

On clean context exit: one `Trajectory` row plus all its steps via a
single `save_trajectory` call. On exception: same, with
`final_status=FAILED` and the exception captured in metadata.
Crash-resilience (writing on every step) is a future-phase enhancement.

## Performance

`@trace_step` adds ~5 μs of overhead per call on a modern Mac (Pydantic
model construction + ContextVar set/reset + list append). Against any
real LLM call (typically 100–5000 ms), this is invisible. See
`benchmarks/overhead.py` for the measurement.
