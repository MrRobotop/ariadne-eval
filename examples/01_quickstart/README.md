# 01 — Quickstart

A 60-line example that traces a tiny ReAct-style loop end-to-end.

## Run

```bash
uv run python examples/01_quickstart/main.py
```

You should see something like:

```
trajectory id: 01J...
final answer: 65.16666666666667
```

The trajectory is persisted to `~/.ariadne/quickstart.duckdb`. Once the
replay UI ships (v0.0.9), you can run `ariadne ui` to view it.

## What it shows

- `start_trajectory(...)` opens an async context.
- `@trace_step("name")` makes any function appear as a Step in the trace.
- `record_llm_call(...)` and `record_tool_call(...)` capture typed payloads.
- The whole trajectory is persisted to DuckDB on context exit.
