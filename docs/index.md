# ariadne-eval

> Trajectory-level observability and evaluation for LLM agents. Open source, self-hosted, framework-agnostic.

Most LLM observability tools were designed for chat: they treat an agent run
as a flat sequence of API calls and lose the structure — plans, tool calls,
recovery from errors, decision branches — that *is* the agent. The few tools
that handle agents are SaaS-only, vendor-locked, and rarely ship with rigorous
evaluation built in. `ariadne-eval` is open source, self-hosted, framework-
agnostic, and treats trajectory-level evaluation (not just final-answer
accuracy) as a first-class concern.

## Status

This site mirrors the project's current state: **v0.0.1, scaffold only**. The
sections below will be populated as later phases ship.

- [Quickstart](quickstart.md) — end-to-end "trace one agent run" walkthrough (lands with v0.0.4).
- [Concepts](concepts/index.md) — trajectory model, tracing, judges, drift detection.
- [Tutorials](tutorials/index.md) — five worked examples covering common workflows.
- [Reference](reference/index.md) — auto-generated API reference (mkdocstrings).

## Why this exists

When an agent gets lost twelve steps into a task, the question is rarely
"was the final answer right?" — it's "where did it go wrong, and is that
something I can detect automatically the next time?" `ariadne-eval` is built
around that question.
