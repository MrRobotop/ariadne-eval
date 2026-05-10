# Concepts

Conceptual documentation lands as each subsystem ships:

- **Trajectory model** (v0.0.2) — how runs, steps, LLM calls, and tool calls relate.
- **Tracing** (v0.0.3) — context propagation, samplers, the `@trace_step` decorator.
- **Storage** (v0.0.2) — DuckDB schema, JSONL portability, migrations.
- **Metrics** (v0.0.5) — final-answer accuracy, tool accuracy, plan quality, recovery rate, efficiency.
- **Judges** (v0.0.6) — LLM-as-judge with kappa-calibrated agreement.
- **Drift detection** (v0.0.10) — CUSUM, ADWIN, sequential probability ratio test.
