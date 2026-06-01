# ariadne-eval

> Trajectory-level observability and evaluation for LLM agents. Open source, self-hosted, framework-agnostic.

[![tests](https://github.com/MrRobotop/ariadne-eval/actions/workflows/tests.yml/badge.svg)](https://github.com/MrRobotop/ariadne-eval/actions/workflows/tests.yml)
[![docs](https://github.com/MrRobotop/ariadne-eval/actions/workflows/docs.yml/badge.svg)](https://mrrobotop.github.io/ariadne-eval/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/MrRobotop/ariadne-eval/blob/main/pyproject.toml)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/typed-mypy--strict-blue)](https://mypy.readthedocs.io/en/stable/command_line.html#cmdoption-mypy-strict)
[![tests](https://img.shields.io/badge/tests-285_passing-brightgreen)](#test-surface-at-v007-alpha)
[![license](https://img.shields.io/badge/license-Apache_2.0-blue)](./LICENSE)
[![status](https://img.shields.io/badge/status-alpha%20(v0.0.7--alpha)-orange)](./CHANGELOG.md)

> **Status: alpha (v0.0.7-alpha).** The tracing layer, storage, reference ReAct agent,
> deterministic eval metrics, async runner, and LLM-as-judge primitives are shipped
> and tested. Judge symbols are deliberately namespace-private until the kappa table
> ships in Phase 6.1 (per [Hard Rule #5](./CLAUDE.md)). Headline tau-bench results,
> replay UI, drift detection, and the PyPI release are scheduled for v0.0.8 through
> v0.1.0. See [CHANGELOG.md](./CHANGELOG.md) for the precise per-tag delta.

---

## Why ariadne-eval?

Most LLM observability tools were designed for chat: they treat an agent run as a
flat sequence of API calls and lose the structure — plans, tool calls, recovery
from errors, decision branches — that *is* the agent. The few tools that do handle
agents are SaaS-only, vendor-locked, and rarely ship rigorous evaluation. `ariadne-eval`
is open source, self-hosted, framework-agnostic, and treats trajectory-level
evaluation (not just final-answer accuracy) as a first-class concern.

The metaphor: Ariadne gave Theseus a thread to find his way back through the
labyrinth. When your agent gets lost twelve steps into a task, `ariadne-eval` is
the thread that lets you trace what happened, score the trajectory, and catch
regressions before users do.

---

## Install

The package is not yet on PyPI (Phase 11 ships v0.1.0 with a Trusted Publisher
release). Until then, install from GitHub:

```bash
pip install git+https://github.com/MrRobotop/ariadne-eval.git
```

Or, if you use [uv](https://docs.astral.sh/uv/):

```bash
uv pip install git+https://github.com/MrRobotop/ariadne-eval.git
```

Optional extras (lazy-imported, no overhead if unused):

```bash
pip install "ariadne-eval[langgraph] @ git+https://github.com/MrRobotop/ariadne-eval.git"
pip install "ariadne-eval[crewai] @ git+https://github.com/MrRobotop/ariadne-eval.git"
pip install "ariadne-eval[openai-assistants] @ git+https://github.com/MrRobotop/ariadne-eval.git"
```

---

## Quickstart

Trace a real ReAct agent end-to-end in ten lines. Needs `OPENAI_API_KEY` in your
environment.

```python
import asyncio
from pathlib import Path

from ariadne_eval import DuckDBStore
from ariadne_eval.examples.react_agent import ReactAgent


async def main() -> None:
    store = DuckDBStore(path=Path("~/.ariadne/quickstart.duckdb").expanduser())
    try:
        agent = ReactAgent(model_id="gpt-4o-mini")
        answer = await agent.arun(
            "What is 17 * 23, and then divide by the number of letters in 'banana'?",
            store=store,
        )
        print(f"final answer: {answer}")
    finally:
        await store.close()


asyncio.run(main())
```

The trajectory — every LLM call, every tool invocation, the parent/child step
tree, costs, latencies, the final answer — is persisted to a DuckDB file. Read it
back with `await store.get_trajectory(trajectory_id)` or query the file directly
with any DuckDB client.

Worked examples in `examples/`:

- [`01_quickstart`](./examples/01_quickstart/) — the snippet above, runnable.
- [`03_custom_metric`](./examples/03_custom_metric/) — write a custom metric, plug into the eval `Runner`, get bootstrap CIs.
- [`04_plan_quality`](./examples/04_plan_quality/) — async runner + judge-backed `PlanQuality` metric (uses `StubJudge`, no API key needed).

LangGraph / CrewAI integration walkthroughs land alongside Phase 7.

---

## What's shipped

| Phase | Milestone | Tag | Status |
|---|---|---|---|
| 0 | Bootstrap, license, CI scaffold | v0.0.1 | shipped |
| 1 | Trajectory data model (Pydantic, ULIDs, tz-aware) | v0.0.2-alpha | shipped |
| 2 | DuckDB storage layer + JSONL exporter + migrations | v0.0.3-alpha | shipped |
| 3 | `@trace_step` decorator, context API, LiteLLM autotrace | v0.0.4-alpha | shipped |
| 4 | Reference ReAct agent + end-to-end VCR cassette test | v0.0.5-alpha | shipped |
| 5 | Programmatic metrics + bootstrap CIs (`FinalAnswerMatch`, `ToolAccuracy`, `StepEfficiency`) | v0.0.6-alpha | shipped |
| 6 | Async `Runner.aevaluate`, `Judge` Protocol, `TrajectoryJudge`, `PlanQuality`, `cohens_kappa` | v0.0.7-alpha | shipped (judges namespace-private pending 6.1) |
| 6.1 | 51-trajectory hand-crafted gold set, kappa table, judge top-level promotion | v0.0.8-alpha | **in progress** |
| 7 | tau-bench Protocol + adapter + runner + `ariadne bench run` CLI | v0.0.9-alpha | shipped (library only; bundle pending 7.1) |
| 8 | CLI polish (`ariadne ui`, `ariadne eval`, `ariadne bench`, `ariadne export`) | v0.0.10-alpha | planned |
| 9 | Streamlit replay UI (trajectory tree, diff, compare, calibration page) | v0.0.11-alpha | planned |
| 10 | CUSUM / ADWIN drift detection with calibrated false-positive rates | v0.0.12-alpha | planned |
| 11 | Documentation, PyPI Trusted Publisher release, v0.1.0 | v0.1.0 | planned |

Each phase has a published design spec in `docs/superpowers/specs/` and an
implementation plan in `docs/superpowers/plans/` — the specs are part of the
portfolio, not just internal notes.

---

## Headline benchmark

Phase 7 (v0.0.9-alpha) ships the benchmark stack: tau-agnostic `Benchmark` Protocol, `TauBenchAdapter` (gated behind the `[tau-bench]` extra), `BenchmarkRunner`, `ariadne bench run` CLI, and the canonical config at [`configs/benchmarks/tau_retail_baseline.yaml`](./configs/benchmarks/tau_retail_baseline.yaml).

The headline τ-retail run is deferred to Phase 7.1 — the simulator-LLM token costs exceed the maintainer's current API tier ceilings. The library code is feature-complete and fully tested; only the bundle is pending. See [Benchmarks](./docs/concepts/benchmarks.md) for the methodology and reproducing instructions.

---

## Test surface, at v0.0.7-alpha

- **283 fast tests** + **2 integration tests** (hand-crafted VCR cassettes, no network) — 285 total.
- **`mypy --strict`** clean across 45 source files.
- **Coverage:** 96% overall on `src/ariadne_eval/eval/*` + `scripts/build_calibration_set.py`; every eval file ≥ 91%.
- **Tracing overhead:** <2% on the representative workload in `benchmarks/overhead.py` (enforced by CI).

---

## Documentation

- [Full documentation](https://mrrobotop.github.io/ariadne-eval/) — concepts, reference, tutorials.
- [Methodology](./METHODOLOGY.md) — metric definitions, known limitations, judge calibration (lands in v0.0.8-alpha).
- [Contributing guide](./CONTRIBUTING.md).
- [Changelog](./CHANGELOG.md).
- [CLAUDE.md](./CLAUDE.md) — project charter, hard rules, audience.

---

## License

[Apache License 2.0](./LICENSE). Includes explicit patent grants — friendly to
enterprise adoption.

---

## Citation

A `CITATION.cff` file ships with v0.1.0. Until then, cite as:

> Patil, R. (2026). *ariadne-eval: trajectory-level observability and evaluation
> for LLM agents* (Version 0.0.7-alpha) [Computer software].
> https://github.com/MrRobotop/ariadne-eval
