# 01 — Quickstart

A real ReAct agent traced end-to-end. The agent uses an LLM (default
`gpt-4o-mini` via litellm) and two stub tools (`calculator` + `search`).

## Prerequisites

Set `OPENAI_API_KEY` in your shell or in `.env`:

```bash
export OPENAI_API_KEY=sk-...
```

To use a different model (e.g. Anthropic Claude), set the matching env
var and edit `model_id="..."` in `main.py`. LiteLLM handles routing.

## Run

```bash
uv run python examples/01_quickstart/main.py
```

Expected output (the LLM's exact wording may vary):

```
final answer: 65.16666666666667
trajectory persisted to: /Users/.../.ariadne/quickstart.duckdb
```

## What it shows

- `start_trajectory(...)` opens an async tracing context.
- `enable_litellm_autotrace()` auto-records every `litellm.acompletion`
  call as an `llm_call` Step.
- `@trace_step("tool_calculator")` wraps each tool invocation as an
  `internal` Step (the structural step in the trace tree).
- `record_tool_call(...)` adds the typed `ToolCallPayload` as a child.
- The full trajectory is saved to DuckDB at context exit.

Once the replay UI ships (v0.0.9), point `ariadne ui` at the same DuckDB
file to drill into the trace.

## Re-running

Each run produces a new trajectory; the DuckDB file grows over time.
Delete `~/.ariadne/quickstart.duckdb` to start fresh.
