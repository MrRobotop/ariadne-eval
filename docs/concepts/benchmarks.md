# Benchmarks

Phase 7 (v0.0.9-alpha) ships the *machinery* for trajectory-level
benchmarks: a tau-agnostic `Benchmark` Protocol, a
`TauBenchAdapter` that wraps Sierra's open-source τ-bench, a
`BenchmarkRunner` that orchestrates `(task × model)` cells under
bounded concurrency with bootstrap CIs, and an `ariadne bench run`
CLI driven by a YAML config.

> **Headline numbers from the canonical τ-retail run are deferred to
> Phase 7.1.** The library is feature-complete, fully tested, and the
> canonical config is committed at
> [`configs/benchmarks/tau_retail_baseline.yaml`](https://github.com/MrRobotop/ariadne-eval/blob/main/configs/benchmarks/tau_retail_baseline.yaml).
> The bundle (`docs/benchmarks/v0.0.9-alpha-tau-retail-50/`) lands
> when the maintainer's API tiers support tau-bench's user
> simulator at 50-task scale (see "Why the headline run is deferred"
> below).

## What runs today

You can already use the benchmark stack against `StubBenchmark` (the
in-memory test double the unit suite ships) or against your own
`Benchmark` Protocol implementation. The `--dry-run` flag validates
configs without making any LLM calls:

```bash
uv run ariadne bench run configs/benchmarks/tau_retail_baseline.yaml --dry-run
```

If you have an Anthropic Tier 2+ account (or a paid Groq tier), the
canonical config works end-to-end:

```bash
pip install 'ariadne-eval[tau-bench]'
export ANTHROPIC_API_KEY=sk-ant-...
export GROQ_API_KEY=...
uv run ariadne bench run configs/benchmarks/tau_retail_baseline.yaml
```

The bundle layout is documented at the bottom of this page.

## Public surface

| Symbol | Importable from |
|---|---|
| `Benchmark` Protocol | `ariadne_eval.benchmarks` |
| `BenchmarkTask`, `BenchmarkRunResult` | `ariadne_eval.benchmarks` |
| `BenchmarkConfig` + `load_benchmark_config` (YAML) | `ariadne_eval.benchmarks` |
| `BenchmarkRunner`, `BenchmarkReport` | `ariadne_eval.benchmarks` |
| `TauBenchAdapter` | `ariadne_eval.benchmarks.tau_bench` (gated behind `[tau-bench]` extra) |
| CLI: `ariadne bench run` | console script |

τ-bench itself is pinned in the optional `[tau-bench]` extra to commit
`59a200c6d575d595120f1cb70fea53cef0632f6b` so re-installs are
deterministic.

## Bundle layout

When the canonical run completes, it writes a result bundle to the
path configured in `output.bundle_dir`:

```
docs/benchmarks/v0.0.9-alpha-tau-retail-50/
├── config.yaml              copy of input config (audit trail)
├── trajectories.jsonl       one JSON object per (task, model) cell,
│                            sorted by (task_id, model_id)
└── summary.json             per-model pass-rate + bootstrap CIs +
                             aggregates for every metric, with the
                             κ = 0.32 (fair) calibration note on
                             every plan_quality block
```

The `calibration_note` field on every `plan_quality` aggregate is
non-negotiable: it travels with the number so any reader of
`summary.json` is pointed at the [Calibration](./calibration.md) page
before trusting an LLM-as-judge score. This is the methodology-honest
move that Hard Rule #5 in `CLAUDE.md` codifies.

## Why the headline run is deferred

τ-bench's retail domain works through a user simulator: a second LLM
that role-plays the customer while the agent uses the tools. The
simulator's first message in retail carries the full wiki + tools
schema as context, which is roughly 50k input tokens per call. Across
a 50-task run with two agent models, the simulator alone burns
roughly 5M tokens.

The maintainer's current API tiers don't fit that:

- Anthropic Tier 1 caps a single input request at 50,000 tokens. The
  retail simulator's first call already sits at that ceiling and
  occasionally exceeds it, so Haiku-as-user-simulator fails fast on
  most tasks.
- Groq's free tier caps tokens per day per model at 100,000. A single
  retail task consumes most of that on the simulator alone; the
  second task fails with `RateLimitError: tokens per day exceeded`.

Phase 7.1 (or whenever the tiers expand) will:

1. Re-run `configs/benchmarks/tau_retail_baseline.yaml` end-to-end
   against the same pinned τ-bench commit.
2. Commit the bundle to `docs/benchmarks/v0.0.9-alpha-tau-retail-50/`
   (the directory name carries the *library* version that produced
   the run, not the date).
3. Update this page in-place with the rendered headline table, the
   confusion of pass-rate-vs-`plan_quality` per model, and a one-line
   summary in the README.

No code changes are required between today and that follow-up — the
library is feature-complete. The wait is purely on API access.

## Reproducing once tiers allow

```bash
pip install 'ariadne-eval[tau-bench]'
export ANTHROPIC_API_KEY=sk-ant-...
export GROQ_API_KEY=...

# canonical run
uv run ariadne bench run configs/benchmarks/tau_retail_baseline.yaml

# optional: --resume after a transient failure mid-run
uv run ariadne bench run configs/benchmarks/tau_retail_baseline.yaml --resume

# optional: subset of models or tasks for a partial re-run
uv run ariadne bench run configs/benchmarks/tau_retail_baseline.yaml \
    --models anthropic/claude-haiku-4-5-20251001 --limit 10
```
