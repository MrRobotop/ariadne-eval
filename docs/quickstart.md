# Quickstart

A real ReAct agent, traced end-to-end. Takes ~5 seconds and a few cents
of OpenAI credit.

## Install

```bash
pip install ariadne-eval
```

## Set your API key

```bash
export OPENAI_API_KEY=sk-...
```

## Run

```bash
uv run python examples/01_quickstart/main.py
```

The example, verbatim from the repo:

```python
--8<-- "examples/01_quickstart/main.py"
```

You should see something like:

```
final answer: 65.16666666666667
trajectory persisted to: ~/.ariadne/quickstart.duckdb
```

## What just happened

The agent:

1. Asked gpt-4o-mini for the next action (LLM call #1, auto-traced).
2. Got `Action: calculator, Action Input: 17*23` back.
3. Ran the calculator tool inside `@trace_step("tool_calculator")` →
   `record_tool_call(...)`.
4. Sent the observation back to the LLM (call #2).
5. Looped one more time to produce the final answer.

The whole tree is saved as a single `Trajectory` in DuckDB with five
or so `Step` rows. Once the replay UI ships (v0.0.9), point
`ariadne ui` at the file to drill in.

## Next

- [Tracing concepts](concepts/tracing.md) — how `@trace_step`, recorders,
  and sampling fit together.
- [Storage](concepts/storage.md) — schema, JSONL portability, the limits.
- The repo's `examples/01_quickstart/README.md` — full prerequisites and
  troubleshooting.
